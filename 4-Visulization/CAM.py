from __future__ import print_function

import numpy as np

import argparse
import torch
import torch.nn as nn
import torch.utils.data as data_utils
import torch.optim as optim
from warmup_scheduler import GradualWarmupScheduler

from torch.autograd import Variable
from torchvision import models, transforms
import torch.nn.functional as F

from Dataloader import prepare_dataset, data_prefetcher
from model import Attention, Attention_Gated, Attention_CAM
from tensorboardX import SummaryWriter
from sklearn.metrics import roc_curve, auc, confusion_matrix

import os
import cv2
import time
import glob
import argparse
from PIL import Image

import apex
from apex import amp
from apex.fp16_utils import *
from apex.parallel import DistributedDataParallel
from apex.multi_tensor_apply import multi_tensor_applier


def get_parser():
    parser = argparse.ArgumentParser(
        description='PyTorch Implementation of multiple Instance learning')
    parser.add_argument('--mag', default='5', type=str)
    parser.add_argument('--model', default=18, type=str)
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--init_method', type=str)
    return parser.parse_args()


test_transform = transforms.Compose([
    transforms.CenterCrop(384),
    transforms.Resize(299),
    transforms.ToTensor(),
    transforms.Normalize([0.7063, 0.3755, 0.6296], [0.1787, 0.2260, 0.1472])
])


img_transform = transforms.Compose([
    transforms.CenterCrop(384),
    transforms.Resize(299)
])


def get_last_conv_name(net):
    layer_name = None
    for name, m in net.named_modules():
        if isinstance(m, nn.Conv2d):
            layer_name = name
    return layer_name


def get_model(path, device):
    model = apex.parallel.convert_syncbn_model(
        Attention_Gated('inceptionv4', True)
    ).to(device)
    model = amp.initialize(model, opt_level="O0", keep_batchnorm_fp32=None)
    model = DistributedDataParallel(model, delay_allreduce=True)
    model.load_state_dict(torch.load(path))
    return model


if __name__ == '__main__':
    args = get_parser()

    torch.backends.cudnn.benchmark = True
    torch.cuda.set_device(args.local_rank)
    torch.distributed.init_process_group(
        'nccl',
        init_method=args.init_method
    )

    device = torch.device(f"cuda:{args.local_rank}")
    model = get_model('...', device)
    from Dataloader import MVIDataset, fast_collate, TestDistSlideSampler
    from torch.utils.data import DataLoader as DL
    from torch.utils.data.distributed import DistributedSampler

    eval_datasets = MVIDataset('...',
                               [],
                               Mag='10_tumor',
                               transforms=transforms.Compose([
                                   transforms.CenterCrop(384),
                                   transforms.Resize(299),
                               ]),
                               limit=16)

    memory_format = torch.contiguous_format
    def collate_fn(b): return fast_collate(b, memory_format)
    sampler = TestDistSlideSampler(eval_datasets, limit=128)
    eval_loader = DL(eval_datasets,
                     batch_sampler=sampler,
                     num_workers=16,
                     pin_memory=True,
                     collate_fn=collate_fn)

    layer_name = get_last_conv_name(model.module.feature_extractor_part1)
    print('Last conv layer name:', layer_name)
    model.eval()

    prefetcher = data_prefetcher(eval_loader)
    patches, label = prefetcher.next()
    index = 0
    while patches is not None:
        ptid, slide = sampler.slide[index]
        idxs = sampler.get_slide(ptid, slide)
        path = [eval_datasets.patch[i] for i in idxs]
        print(ptid, slide, len(idxs))

        model.zero_grad()

        handler = []
        feature = None
        gradient = None

        def get_feature_hook(module, input, output):
            global feature
            feature = output

        def get_grads_hook(module, input, output):
            global gradient
            gradient = output[0]

        for (name, module) in \
                model.module.feature_extractor_part1.named_modules():
            if name == layer_name:
                handler.append(module.register_forward_hook(get_feature_hook))
                handler.append(module.register_backward_hook(get_grads_hook))

        Y_prob = model.forward(patches)
        Y_prob.backward()
        print(feature.shape)
        for i in range(len(idxs)):
            f = feature[i].cpu().data.numpy()  # 256 * 8 * 8
            g = gradient[i].cpu().data.numpy()  # 256 * 8 * 8
            weight = np.mean(g, axis=(1, 2))  # 256,

            cam = f * weight[:, np.newaxis, np.newaxis]  # 256 * 8 * 8
            cam = np.sum(cam, axis=0)  # 256,
            cam -= np.min(cam)
            cam /= (1e-5 + np.max(cam))
            cam = cv2.resize(cam, (299, 299))

            img = Image.open(path[i])
            img = img_transform(img)
            img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

            cam = cv2.applyColorMap(np.uint8(255*cam), cv2.COLORMAP_JET)

            heatmap = cam*0.6 + img * 0.4

            file_name = '{}_{}_{}.jpeg'.format(
                ptid, slide, os.path.basename(path[i]).split('.')[0])
            cv2.imwrite(f'./CAM/{file_name}', heatmap)

        for h in handler:
            h.remove()
        patches, label = prefetcher.next()
        index += 1
