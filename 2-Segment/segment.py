from contextlib import contextmanager
from torchvision import transforms, utils
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from PIL import Image
from sklearn import metrics
from sklearn.metrics import roc_curve, auc, confusion_matrix
import copy
import numpy as np
import glob

import os
import time
import shutil
import argparse


from Dataloader import prepare_data, data_prefetcher, MyDataset, fast_collate
import apex
from apex import amp
from apex.fp16_utils import *
from apex.parallel import DistributedDataParallel
from apex.multi_tensor_apply import multi_tensor_applier

N = 0

transform_valid = transforms.Compose([
    transforms.CenterCrop(384),
    transforms.Resize(299),
    transforms.ToTensor(),
    transforms.Normalize([0.66670404, 0.3369676, 0.57027892], [
                         0.18942341, 0.2342381, 0.17067234])
])


def gather_tensor(tensor: torch.Tensor):
    rt = tensor.clone()
    var_list = [torch.zeros_like(rt) for _ in range(
        torch.distributed.get_world_size())]
    torch.distributed.all_gather(var_list, rt, async_op=False)
    return [i.cpu().numpy() for i in var_list]


class Test_dataset(Dataset):
    def __init__(self, ptid, slide, mag, transform):
        self.imgs = glob.glob(
            f'.../{ptid}/{slide}/{mag}/*')
        self.transform = transform

    def __getitem__(self, index):
        path = self.imgs[index]
        img = Image.open(path)
        img = self.transform(img)
        return img, index

    def __len__(self):
        return len(self.imgs)


def eval_model(args, model, loss_fn, dataloader, device):
    if args.local_rank == 0:
        print(len(all_labels))
        all_labels, all_values = np.array(all_labels), np.array(all_values)
        Loss = training_loss / len(train_loader)
        AUC, Acc = get_cm(all_labels, all_values)
        print("Val Auc: {:.4f}, Acc: {:.4f}, Loss: {:.4f}".format(
            AUC, Acc, Loss))
    return all_labels, all_values


def get_parser():
    parser = argparse.ArgumentParser(
        description='PyTorch Implementation of multiple Instance learning')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--init_method', type=str)
    return parser.parse_args()


@contextmanager
def dist_zero_first(local_rank: int):
    if local_rank not in [-1, 0]:
        torch.distributed.barrier()
    yield
    if local_rank == 0:
        torch.distributed.barrier()


if __name__ == '__main__':
    args = get_parser()
    torch.backends.cudnn.benchmark = True
    torch.cuda.set_device(args.local_rank)
    torch.distributed.init_process_group(
        'nccl',
        init_method=args.init_method
    )
    N = torch.distributed.get_world_size()

    device = torch.device(f"cuda:{args.local_rank}")

    model = torchvision.models.inception_v3(pretrained=True, aux_logits=False)
    model.fc = nn.Linear(2048, 2)

    model = apex.parallel.convert_syncbn_model(model).to(device)
    model = amp.initialize(model, opt_level="O1", keep_batchnorm_fp32=None)
    model = DistributedDataParallel(model, delay_allreduce=False)
    model.load_state_dict(torch.load('./40X_result/model_30_0.9974.pkl'))
    model.eval()

    import json
    from config import SDH_SMU_labels, metadata

    PATCH_DATA = '...'
    ptids = os.listdir(f'{PATCH_DATA}/')

    for ptid in ptids:
        for slide in os.listdir(f'{PATCH_DATA}/{ptid}'):
            if ptid not in metadata or not slide in metadata[ptid]['slides']:
                continue

            tumor_path = f'{PATCH_DATA}/{ptid}/{slide}/40_tumor'

            with dist_zero_first(args.local_rank):
                if args.local_rank == 0:
                    print(ptid, slide)

                    if os.path.exists(tumor_path):
                        print('Tumor path exists:', tumor_path, ', delete it.')
                        shutil.rmtree(tumor_path)
                    os.mkdir(tumor_path)
                    os.chmod(tumor_path, 0o755)

            dataset = Test_dataset(ptid=ptid,
                                   slide=slide,
                                   mag='40',
                                   transform=transform_valid,
                                   )

            dataloader = DataLoader(dataset,
                                    batch_size=32,
                                    num_workers=4,
                                    shuffle=False,
                                    sampler=torch.utils.data.distributed.DistributedSampler(dataset, shuffle=False,
                                                                                            num_replicas=torch.distributed.get_world_size(),
                                                                                            rank=args.local_rank)
                                    )

            all_values = []
            patches_path = []

            for idx, (inputs, paths_index) in enumerate(dataloader):
                with torch.no_grad():
                    inputs = inputs.to(device)
                    output = F.softmax(model(inputs), dim=1)
                output = output[:, 1]
                all_values.extend(output)
                patches_path.extend(paths_index)
            patches_path = [dataset.imgs[i] for i in patches_path]
            count = 0
            for i in range(len(all_values)):
                if all_values[i] >= 0.99:
                    try:
                        count += 1
                        os.symlink(
                            os.path.join(
                                '../40/', os.path.basename(patches_path[i])),
                            os.path.join(
                                tumor_path, os.path.basename(patches_path[i]))
                        )
                    except:
                        continue
            print(ptid, slide, count)
