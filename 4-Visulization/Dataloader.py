from itertools import chain
import glob
import json
import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader as DL
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms
from config import metadata, train_labels, val_labels, SDH_SMU_labels

train_transform = transforms.Compose([
    transforms.RandomRotation(15),
    transforms.RandomCrop(384),
    transforms.Resize(299),
    transforms.ColorJitter(brightness=0.3, contrast=0.3,
                           saturation=0.2, hue=0.1),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip()

])

test_transform = transforms.Compose([
    transforms.CenterCrop(384),
    transforms.Resize(299),
])


class MVIDataset(Dataset):
    def __init__(self, Data_path: str, ptids, Mag='5', transforms=None, limit=96):

        self.ptids = [p for p in ptids if os.path.exists(
            os.path.join(Data_path, p))]
        self.slide = [
            (ptid, slide)
            for ptid in self.ptids
            for slide in os.listdir(os.path.join(Data_path, ptid))
            if slide in metadata[ptid]['slides'] and
            metadata[ptid]['slides'][slide]['is_cancer'] and
            limit <= len(glob.glob(os.path.join(
                Data_path, ptid, slide, Mag, '*')))

        ]

        index = 0
        self.patch = []
        self.label = []
        self.indices = {}

        for i, (ptid, slide) in enumerate(self.slide):
            patches = glob.glob(os.path.join(Data_path, ptid, slide, Mag, '*'))
            self.patch.extend(patches)
            label = metadata[ptid]['label']
            self.label.extend([label] * len(patches))
            self.indices[(ptid, slide)] = np.arange(
                index, index + len(patches))
            index += len(patches)

        self.slide = np.array(self.slide)
        self.data_transforms = transforms

    def __len__(self):
        return len(self.patch)

    def __getitem__(self, index):
        img = Image.open(self.patch[index])
        label = self.label[index]
        if self.data_transforms is not None:
            img = self.data_transforms(img)
        return img, label


class DistSlideSampler(DistributedSampler):
    def __init__(self, dataset, padding, seed):
        super(DistSlideSampler, self).__init__(dataset)
        self.slide = dataset.slide
        self.indices = dataset.indices
        self.padding = padding
        self.seed = hash(seed)
        self.g = torch.Generator()

    def __iter__(self):
        self.g.manual_seed(self.seed + self.epoch)
        indices = torch.randperm(
            len(self.slide) - len(self.slide) % self.num_replicas,
            generator=self.g
        ).tolist()

        for i in indices[self.rank::self.num_replicas]:
            ptid, slide = self.slide[i]
            yield self.get_slide(ptid, slide)
        self.epoch += 1

    def __len__(self):
        return len(self.slide) // self.num_replicas

    def get_slide(self, ptid, slide):

        indice = self.indices[(ptid, slide)]
        patch_num = len(indice)

        np.random.seed(self.seed % (2 ** 32) + self.epoch)
        if patch_num <= self.padding:
            multip = self.padding // patch_num
            need = self.padding - multip * patch_num
            indice = np.concatenate(
                [indice] * multip +
                [np.random.choice(indice, need, replace=False)]
            )
        else:
            indice = np.random.choice(indice, self.padding, replace=False)

        shuffle = torch.randperm(self.padding, generator=self.g).tolist()

        return indice[shuffle]


class TestDistSlideSampler(DistributedSampler):
    def __init__(self, dataset, limit=512):
        super(TestDistSlideSampler, self).__init__(dataset)
        self.slide = dataset.slide
        self.indices = dataset.indices
        self.limit = limit

    def __len__(self):
        return len(self.slide) // self.num_replicas

    def __iter__(self):
        slide = self.slide[len(self.slide) % self.num_replicas:]

        for ptid, slide in slide[self.rank::self.num_replicas]:
            yield self.get_slide(ptid, slide)

    def get_slide(self, ptid, slide):
        indice = self.indices[(ptid, slide)]
        patch_num = len(indice)

        shuffle = torch.randperm(
            self.limit if patch_num > self.limit else patch_num,
            generator=torch.Generator().manual_seed(132)
        ).tolist()

        return indice[shuffle]


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


def prepare_dataset(padding=128, mag='5', seed=None):
    limit = 16
    train_set = train_labels
    val_set = val_labels

    train_datasets = MVIDataset('...',
                                train_set,
                                limit=limit,
                                Mag=mag,
                                transforms=train_transform)
    test_datasets = MVIDataset('...',
                               val_set,
                               limit=limit,
                               Mag=mag,
                               transforms=test_transform)

    print('Train slide number:', len(train_datasets.slide))
    print('Train patches number:', len(train_datasets))

    print('Test  slide number:', len(test_datasets.slide))
    print('Test  patches number:', len(test_datasets))

    memory_format = torch.contiguous_format
    def collate_fn(b): return fast_collate(b, memory_format)

    if not seed:
        import datetime
        seed = datetime.datetime.timestamp(datetime.datetime.now())
    print('Seed', seed)
    train_loader = DL(train_datasets,
                      batch_sampler=DistSlideSampler(
                          train_datasets, padding=padding, seed=seed),
                      num_workers=8,
                      pin_memory=True,
                      collate_fn=collate_fn
                      )

    test_loader = DL(test_datasets,
                     batch_sampler=TestDistSlideSampler(
                         test_datasets, limit=256),
                     num_workers=8,
                     pin_memory=True,
                     collate_fn=collate_fn
                     )

    return train_loader, test_loader


class data_prefetcher():
    def __init__(self, loader, dataset='train'):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        self.mean = torch.tensor(
            [0.7063 * 255, 0.3755 * 255, 0.6296 * 255]).cuda().view(1, 3, 1, 1)
        self.std = torch.tensor(
            [0.1787 * 255, 0.2260 * 255, 0.1472 * 255]).cuda().view(1, 3, 1, 1)

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

