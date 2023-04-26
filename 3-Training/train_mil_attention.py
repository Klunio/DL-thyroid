from __future__ import print_function

import argparse
import os

import apex
import numpy as np
import torch
import torch.optim as optim
from apex import amp
from apex.fp16_utils import *
from apex.parallel import DistributedDataParallel
from sklearn.metrics import roc_curve, auc, confusion_matrix
from tensorboardX import SummaryWriter
from warmup_scheduler import GradualWarmupScheduler

from Dataloader import prepare_dataset, data_prefetcher
from model import Attention_Gated


def run(args, train_loader, test_loader, model, epochs, schduler, optimizer, device, Writer):
    best_auc = .0

    eval_model(args, test_loader, model, device, optimizer, 0, Writer)
    for epoch in range(1, epochs):
        if args.local_rank == 0:
            print('Epoch [{}/{}]'.format(epoch, epochs))
            print('### Train ###')

        # 1. train
        train_loader.batch_sampler.set_epoch(epoch)
        train_model(args, train_loader, model,
                    device, optimizer, epoch, Writer)
        schduler.step()
        current_lr = schduler.get_lr()[0]

        if args.local_rank == 0:
            Writer.add_scalar('Learning Rate', current_lr, epoch)

        # 2. Test
        test_loader.batch_sampler.set_epoch(epoch)
        eval_model(args, test_loader, model, device, optimizer, epoch, Writer)

        if args.local_rank == 0 and epoch >= 15:
            torch.save(model.state_dict(), os.path.join(
                '../checkpoints', args.comment, '{}.pt'.format(epoch))
            )


def reduce_tensor(tensor: torch.Tensor) -> torch.Tensor:
    rt = tensor.clone()
    torch.distributed.all_reduce(rt, op=torch.distributed.ReduceOp.SUM)
    return rt


def gather_tensor(tensor: torch.Tensor):
    rt = tensor.clone()
    var_list = [torch.zeros_like(rt) for _ in range(
        torch.distributed.get_world_size())]
    torch.distributed.all_gather(var_list, rt, async_op=False)
    return [i.cpu().numpy().tolist() for i in var_list]


def set_fn(v):
    def f(m):
        if isinstance(m, apex.parallel.SyncBatchNorm):
            m.momentum = v

    return f


def train_model(args, dataloader, model, device, optimizer, epoch, Writer):
    phase = 'train'
    model.train()

    all_labels = []
    all_values = []
    train_loss = 0

    prefetcher = data_prefetcher(dataloader)
    patches, label = prefetcher.next()
    index = 0
    while patches is not None:
        index += 1
        label = label[0].float()
        Y_prob = model.forward(patches)
        Y_prob = torch.clamp(Y_prob, min=1e-5, max=1. - 1e-5)

        J = -1. * (
            label * torch.log(Y_prob) +
            (1. - label) * torch.log(1. - Y_prob)
        )

        optimizer.zero_grad()
        with amp.scale_loss(J, optimizer) as scale_loss:
            scale_loss.backward()
        optimizer.step()

        reduced_loss = reduce_tensor(J.data)
        train_loss += reduced_loss.item()

        all_labels.extend(gather_tensor(label))
        all_values.extend(gather_tensor(Y_prob[0][0]))

        if args.local_rank == 0 and (index + 1) % 1 == 0:
            print('[{}/{}] Loss:{:.4f}'.format(index,
                  len(dataloader), reduced_loss.item() / 8))
        patches, label = prefetcher.next()

    if args.local_rank == 0:
        print(len(all_labels))
        all_labels = np.array(all_labels)
        Loss = train_loss / len(all_labels)
        AUC, Acc = get_cm(all_labels, all_values)

        Writer.add_scalar('{}/Acc'.format(phase.capitalize()), Acc, epoch)
        Writer.add_scalar('{}/Loss'.format(phase.capitalize()), Loss, epoch)
        Writer.add_scalar('{}/Auc'.format(phase.capitalize()), AUC, epoch)
        print('Epoch-{} Loss per bag:{:.4f} - Acc:{:.4f}'.format(epoch, Loss, Acc))

    return


def eval_model(args, dataloader, model, device, optimizer, epoch, Writer):
    phase = 'test'
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

        if args.local_rank == 0 and (index + 1) % 1 == 0:
            print('[{}/{}] Loss:{:.4f}'.format(index,
                  len(dataloader), reduced_loss.item() / 8))
        patches, label = prefetcher.next()

    if args.local_rank == 0:
        print(len(all_labels))
        all_labels = np.array(all_labels)
        all_values = np.array(all_values)

        Loss = train_loss / len(all_labels)
        AUC, Acc = get_cm(all_labels, all_values)

        Writer.add_scalar('{}/Acc'.format(phase.capitalize()), Acc, epoch)
        Writer.add_scalar('{}/Loss'.format(phase.capitalize()), Loss, epoch)
        Writer.add_scalar('{}/Auc'.format(phase.capitalize()), AUC, epoch)
        print('Epoch-{} Loss per bag:{:.4f} - Acc:{:.4f}'.format(epoch, Loss, Acc))

    return


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
    parser.add_argument('--epochs', default=150,
                        type=int, help='number of epochs')
    parser.add_argument('--mag', default='5', type=str)
    parser.add_argument('--lr', default=0.05, type=float,
                        help='initial learning rate (default: 0.05)')
    parser.add_argument('--momentum', default=0.95, type=float)
    parser.add_argument('--weight-decay', default=5e-4, type=float)
    parser.add_argument('--lrdrop', default=150, type=int,
                        help='multiply LR by 0.1 every (default: 150 epochs)')
    parser.add_argument('--padding', default=128, type=int)
    parser.add_argument('--comment', default='MIL-default', type=str)
    parser.add_argument('--model', default=18, type=str)
    parser.add_argument('--pretrain', action='store_true')
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
    name = args.comment

    writer = None
    if args.local_rank == 0:
        writer = SummaryWriter(f'../runs/{name}')
        writer.add_text('args', " \n".join(
            ['%s %s' % (arg, getattr(args, arg)) for arg in vars(args)]))

    train_loader, test_loader = prepare_dataset(
        args.padding, args.mag, args.comment)

    device = torch.device(f"cuda:{args.local_rank}")

    model = apex.parallel.convert_syncbn_model(
        Attention_Gated(args.model, args.pretrain)
    ).to(device)

    optimizer = optim.SGD(model.parameters(), lr=args.lr,
                          momentum=args.momentum, weight_decay=args.weight_decay)

    model, optimizer = amp.initialize(model, optimizer,
                                      opt_level="O0",
                                      keep_batchnorm_fp32=None)

    model = DistributedDataParallel(model, delay_allreduce=True)

    from torch.optim.lr_scheduler import CosineAnnealingLR

    scheduler = CosineAnnealingLR(
        optimizer, T_max=args.epochs - 5, eta_min=1e-6)
    scheduler = GradualWarmupScheduler(
        optimizer, multiplier=1, total_epoch=5, after_scheduler=scheduler)

    # 4. Run
    run(args, train_loader, test_loader, model,
        args.epochs, scheduler, optimizer, device, writer)
