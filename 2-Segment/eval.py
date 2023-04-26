from torchvision import transforms, utils
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from sklearn.metrics import classification_report
from PIL import Image
from sklearn import metrics
from sklearn.metrics import roc_curve, auc, confusion_matrix
import copy
import numpy as np

import os
import time
import argparse
from Dataloader import prepare_data, data_prefetcher
import apex
from apex import amp
from apex.fp16_utils import *
from apex.parallel import DistributedDataParallel
from apex.multi_tensor_apply import multi_tensor_applier

N = 0


def reduce_tensor(tensor: torch.Tensor) -> torch.Tensor:
    rt = tensor.clone()
    torch.distributed.all_reduce(rt, op=torch.distributed.ReduceOp.SUM)
    return rt


def gather_tensor(tensor: torch.Tensor):
    rt = tensor.clone()
    var_list = [torch.zeros_like(rt) for _ in range(
        torch.distributed.get_world_size())]
    torch.distributed.all_gather(var_list, rt, async_op=False)
    return [i.cpu().numpy() for i in var_list]


def get_cm(AllLabels, AllValues):
    fpr, tpr, threshold = roc_curve(AllLabels, AllValues, pos_label=1)
    Auc = auc(fpr, tpr)
    m = t = 0

    for i in range(len(threshold)):
        if tpr[i] - fpr[i] > m:
            m = abs(-fpr[i]+tpr[i])
            t = threshold[i]
    AllPred = [int(i >= t) for i in AllValues]
    Acc = sum([AllLabels[i] == AllPred[i]
              for i in range(len(AllPred))]) / len(AllPred)

    Pos_num = sum(AllLabels)
    Neg_num = len(AllLabels) - Pos_num
    cm = confusion_matrix(AllLabels, AllPred)
    print("[AUC/{:.4f}] [Threshold/{:.4f}] [Acc/{:.4f}]".format(Auc, t,  Acc))
    print("{:.2f}% {:.2f}%".format(
        cm[0][0] / Neg_num * 100, cm[0][1]/Neg_num * 100))
    print("{:.2f}% {:.2f}%".format(
        cm[1][0] / Pos_num * 100, cm[1][1]/Pos_num * 100))

    return Auc, Acc


def eval_model(args, model, loss_fn, dataloader, device):
    model.eval()
    all_labels = []
    all_values = []
    training_loss = 0

    prefetcher = data_prefetcher(dataloader)
    inputs, targets = prefetcher.next()
    index = 0
    while inputs is not None:
        index += 1

        with torch.no_grad():
            output = F.softmax(model(inputs), dim=1)
            loss = loss_fn(output, targets)

        reduced_loss = reduce_tensor(loss.data) / N
        training_loss += reduced_loss.item()

        targets, output = gather_tensor(targets), gather_tensor(output[:, 1])

        for j in range(torch.distributed.get_world_size()):
            all_labels.extend(targets[j])
            all_values.extend(output[j])

        if args.local_rank == 0 and (index+1) % 1 == 0:
            print('\t[{}/{}] Loss: {:.4f}'.format(index +
                  1, len(dataloader), reduced_loss.item()))
        inputs, targets = prefetcher.next()

    if args.local_rank == 0:
        print(len(all_labels))
        all_labels, all_values = np.array(all_labels), np.array(all_values)
    return all_labels, all_values


def get_parser():
    parser = argparse.ArgumentParser(
        description='PyTorch Implementation of multiple Instance learning')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--init_method', type=str)
    return parser.parse_args()


if __name__ == '__main__':
    args = get_parser()
    torch.backends.cudnn.benchmark = True
    torch.cuda.set_device(args.local_rank)
    torch.distributed.init_process_group(
        'nccl',
        init_method=args.init_method
    )
    N = torch.distributed.get_world_size()

    train_loader, val_loader = prepare_data(args)
    device = torch.device(f"cuda:{args.local_rank}")

    model = torchvision.models.inception_v3(pretrained=False, aux_logits=False)
    model.fc = nn.Linear(2048, 2)

    model = apex.parallel.convert_syncbn_model(model).to(device)
    model = amp.initialize(model, opt_level="O1", keep_batchnorm_fp32=None)
    model = DistributedDataParallel(model, delay_allreduce=True)
    model.load_state_dict(torch.load('./40X_result/model_23_0.9974.pkl'))

    all_labels, all_values = eval_model(
        args, model, torch.nn.CrossEntropyLoss(), val_loader, device)
    if args.local_rank == 0:
        import pandas as pd
        pd.DataFrame({
            'Label': all_labels,
            'Value': all_values
        }).to_csv(f'./result-X20.csv')
