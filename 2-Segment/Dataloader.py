from torch.utils.data import Dataset, DataLoader as DL, Sampler
from torch.utils.data.distributed import DistributedSampler
import torch
from PIL import Image
import numpy as np
import glob
import os
import json
from torchvision import models, transforms

transform_train = transforms.Compose([
    transforms.RandomCrop(384),
    transforms.Resize([299, 299]),
    transforms.ColorJitter(brightness=0.2, contrast=0.2,
                           saturation=0.1, hue=0.1),
])

transform_valid = transforms.Compose([
    transforms.CenterCrop(384),
    transforms.Resize(299),
])


class MyDataset(Dataset):
    def __init__(self, tumor_path, tissue_path, lymphocytes_path, transform=None):
        self.imgs = []

        for path in open(tumor_path, 'r').readlines():
            path = path.strip('\n')
            if os.path.exists(path):
                self.imgs.append((path, 1))
        for path in open(tissue_path, 'r').readlines():
            path = path.strip('\n')
            if os.path.exists(path):
                self.imgs.append((path, 0))
        for path in open(lymphocytes_path, 'r').readlines():
            path = path.strip('\n')
            if os.path.exists(path):
                self.imgs.append((path, 0))

        self.transform = transform

    def __getitem__(self, index):
        path, label = self.imgs[index]
        img = Image.open(path.strip('\n'))
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.imgs)


def fast_collate(batch, memory_format):
    imgs = [img[0] for img in batch]
    targets = torch.tensor([target[1] for target in batch], dtype=torch.int64)
    w, h = imgs[0].size[0], imgs[0].size[1]
    tensor = torch.zeros((len(imgs), 3, h, w), dtype=torch.uint8).contiguous(
        memory_format=memory_format)
    for i, img in enumerate(imgs):
        numpy_array = np.asarray(img, dtype=np.uint8)
        numpy_array = np.rollaxis(numpy_array, 2)
        tensor[i] += torch.from_numpy(numpy_array.copy())
    return tensor, targets


def prepare_data(args):
    train_data = MyDataset(tumor_path='./labels/train_tumor_40X.txt',
                           tissue_path='./labels/train_normal_40X.txt',
                           lymphocytes_path='./labels/train_lymphocytes_40X.txt',
                           transform=transform_train)
    val_data = MyDataset(tumor_path='./labels/val_tumor_40X.txt',
                         tissue_path='./labels/val_normal_40X.txt',
                         lymphocytes_path='./labels/val_lymphocytes_40X.txt',
                         transform=transform_valid)

    print('Train size:', len(train_data), 'Val size:', len(val_data))
    memory_format = torch.contiguous_format
    def collate_fn(b): return fast_collate(b, memory_format)

    train_loader = DL(train_data,
                      batch_size=128,
                      num_workers=32,
                      pin_memory=True,
                      collate_fn=collate_fn,
                      sampler=torch.utils.data.distributed.DistributedSampler(
                          train_data, shuffle=True, num_replicas=torch.distributed.get_world_size(), rank=args.local_rank)
                      )
    val_loader = DL(val_data,
                    batch_size=128,
                    num_workers=32,
                    pin_memory=True,
                    collate_fn=collate_fn,
                    sampler=torch.utils.data.distributed.DistributedSampler(
                        val_data, shuffle=False, num_replicas=torch.distributed.get_world_size(), rank=args.local_rank)
                    )
    return train_loader, val_loader


class data_prefetcher():
    def __init__(self, loader, dataset='train'):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        self.mean = torch.tensor(
            [0.66670404 * 255, 0.3369676 * 255, 0.57027892 * 255]).cuda().view(1, 3, 1, 1)
        self.std = torch.tensor(
            [0.18942341 * 255, 0.2342381 * 255, 0.17067234 * 255]).cuda().view(1, 3, 1, 1)
        self.preload()

    def preload(self):
        try:
            self.next_input, self.next_target = next(self.loader)
        except StopIteration:
            self.next_input = None
            self.next_target = None
            return
        with torch.cuda.stream(self.stream):
            self.next_input = self.next_input.cuda(non_blocking=True)
            self.next_target = self.next_target.cuda(non_blocking=True)
            self.next_input = self.next_input.float()
            self.next_input = self.next_input.sub_(self.mean).div_(self.std)

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        input = self.next_input
        target = self.next_target
        if input is not None:
            input.record_stream(torch.cuda.current_stream())
        if target is not None:
            target.record_stream(torch.cuda.current_stream())
        self.preload()
        return input, target
