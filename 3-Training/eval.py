from __future__ import print_function

import argparse

import apex
import numpy as np
import torch
from apex import amp
from apex.fp16_utils import *
from apex.parallel import DistributedDataParallel
from sklearn.metrics import roc_curve, auc, confusion_matrix

from Dataloader import data_prefetcher
from model import Attention_Gated


# from warmup_scheduler import GradualWarmupScheduler


def reduce_tensor(tensor: torch.Tensor) -> torch.Tensor:
    rt = tensor.clone()
    torch.distributed.all_reduce(rt, op=torch.distributed.ReduceOp.SUM)
    return rt


def gather_tensor(tensor: torch.Tensor):
    rt = tensor.clone()
    var_list = [torch.zeros_like(rt) for _ in range(
        torch.distributed.get_world_size())]
    torch.distributed.all_gather(var_list, rt, async_op=False)
    return [i.item() for i in var_list]


def eval_model(args, dataloader, model):
    model.eval()
    all_labels = []
    all_values = []
    train_loss = 0

    prefetcher = data_prefetcher(dataloader)
    patches, label = prefetcher.next()
    index = 0
    while patches is not None:
        index += 1
        label = label[0].float()

        with torch.no_grad():
            Y_prob = model.forward(patches)
            Y_prob = torch.clamp(Y_prob, min=1e-5, max=1. - 1e-5)

            J = -1. * (
                label * torch.log(Y_prob) +
                (1. - label) * torch.log(1. - Y_prob)
            )

        reduced_loss = reduce_tensor(J.data)

        train_loss += reduced_loss.item()
        all_labels.extend(gather_tensor(label))
        all_values.extend(gather_tensor(Y_prob[0][0]))

        if args.local_rank == 0:
            print("[{}/{}] - Loss:{:.4f}".format(index,
                  len(dataloader), reduced_loss.item()))

        patches, label = prefetcher.next()

    if args.local_rank == 0:
        print(len(all_labels))
        all_labels = np.array(all_labels)
        Loss = train_loss / len(all_labels)
        AUC, Acc = get_cm(all_labels, all_values)
        print('Loss', Loss)

    return all_labels, all_values


def get_cm(AllLabels, AllValues):
    fpr, tpr, threshold = roc_curve(AllLabels, AllValues, pos_label=1)
    Auc = auc(fpr, tpr)
    m = t = 0

    for i in range(len(threshold)):
        if tpr[i] - fpr[i] > m:
            m = abs(-fpr[i] + tpr[i])
            t = threshold[i]
    AllPred = [int(i >= t) for i in AllValues]
    Acc = sum([AllLabels[i] == AllPred[i]
              for i in range(len(AllPred))]) / len(AllPred)

    Pos_num = sum(AllLabels)
    Neg_num = len(AllLabels) - Pos_num
    cm = confusion_matrix(AllLabels, AllPred)
    print("[AUC/{:.4f}] [Threshold/{:.4f}] [Acc/{:.4f}]".format(Auc, t, Acc))
    print("{:.2f}% {:.2f}%".format(
        cm[0][0] / Neg_num * 100, cm[0][1] / Neg_num * 100))
    print("{:.2f}% {:.2f}%".format(
        cm[1][0] / Pos_num * 100, cm[1][1] / Pos_num * 100))

    return Auc, Acc


def get_parser():
    parser = argparse.ArgumentParser(
        description='PyTorch Implementation of multiple Instance learning')
    parser.add_argument('--mag', default='5', type=str)
    parser.add_argument('--model', default=18, type=str)
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--init_method', type=str)
    return parser.parse_args()


if __name__ == '__main__':
    args = get_parser()
    #     args.local_rank = int(os.environ["LOCAL_RANK"])

    torch.backends.cudnn.benchmark = True
    torch.cuda.set_device(args.local_rank)
    torch.distributed.init_process_group(
        'nccl',
        init_method=args.init_method
    )
    # Dataset
    from Dataloader import MVIDataset, test_transform, fast_collate, TestDistSlideSampler
    from config import metadata, SDH_SMU_labels, gxmu_labels, train_labels, SYSUCC_labels
    from torch.utils.data import DataLoader as DL
    from itertools import chain

    print(f"local rank {args.local_rank}")

    device = torch.device(f"cuda:{args.local_rank}")

    model = apex.parallel.convert_syncbn_model(
        Attention_Gated(args.model, True)
    ).to(device)

    model = amp.initialize(model, opt_level="O1", keep_batchnorm_fp32=None)
    model = DistributedDataParallel(model, delay_allreduce=True)

#     slide = sampler.slide

    for path in ['./checkpoints/2021-12-16_X10/26.pt']:
        model.load_state_dict(torch.load(path))

        # DATASET
        eval_datasets = MVIDataset(
            '...',
            SYSUCC_labels,
            Mag=args.mag,
            transforms=test_transform,
            limit=16)

        memory_format = torch.contiguous_format
        def collate_fn(b): return fast_collate(b, memory_format)
        sampler = TestDistSlideSampler(eval_datasets, limit=512)
        eval_loader = DL(eval_datasets,
                         batch_sampler=sampler,
                         num_workers=4,
                         pin_memory=True,
                         collate_fn=collate_fn)
        slide = sampler.slide
        ###
        all_labels, all_values = eval_model(args, eval_loader, model)
        temp_slide = list(zip(*slide[len(slide) - len(all_labels):]))
        if args.local_rank == 0:
            import pandas as pd

            pd.DataFrame({
                'ptid': temp_slide[0],
                'slide': temp_slide[1],
                'label': all_labels,
                'value': all_values,
            }).to_csv(f'SYSUCC-X10.csv')
