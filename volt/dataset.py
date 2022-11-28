import os
from dataclasses import dataclass

import torch
import torchvision
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import transforms


import pgn
from volt import util
from volt.modules import classifier

@dataclass
class DataConfig:
    data_root: str = '~/data'
    batch_size: int = 128
    num_workers: int = 4

class SliceableDataset(Dataset):
    def __init__(self, dataset=None, slice=None):
        super(SliceableDataset, self).__init__()
        self.__dataset = dataset
        self.__slice = slice

    def get_item(self, index):
        raise NotImplementedError('You must override get_item')

    def __getitem__(self, index):
        if isinstance(index, slice):
            return SliceableDataset(self, index)
        elif self.__dataset is None:
            return self.get_item(index)
        else:
            start, stop, step = self.__slice.indices(len(self.__dataset))
            if index < 0:
                index += len(self)
            assert 0 <= index < len(self), 'Index out of range'
            return self.__dataset[start + index * step]

    def __len__(self):
        assert self.__dataset is not None, 'You must override __len__'
        start, stop, step = self.__slice.indices(len(self.__dataset))
        return (stop - start) // step

    def __getattr__(self, attr):
        if self.__dataset is None:
            raise AttributeError(f'{self.__class__.__name__} object has no attribute {attr}')
        return getattr(self.__dataset, attr)


def get_mnist(cfg):
    train_dataset = datasets.MNIST(os.path.join(cfg.main.data_path, 'mnist'), train=True, download=True)
    val_dataset = datasets.MNIST(os.path.join(cfg.main.data_path, 'mnist'), train=False)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    untransform = transforms.Compose([
        transforms.Normalize((-0.1307 / 0.3081,), (1 / 0.3081,)),
        transforms.ToPILImage()
    ])

    train_dataset = classifier.ClassDataset(train_dataset, transform, untransform, labels=range(10),
                                            label_names=train_dataset.classes)
    val_dataset = classifier.ClassDataset(val_dataset, transform, untransform, labels=range(10),
                                          label_names=val_dataset.classes)

    return train_dataset, val_dataset, None

# 32x32 resolution in accordance with the ramanujan et al paper
def get_cifar10(cfg):
    data_root = os.path.join(cfg.main.data_path, "cifar10")
    normalize = transforms.Normalize(
        mean=[0.491, 0.482, 0.447], std=[0.247, 0.243, 0.262]
    )
    train_dataset = torchvision.datasets.CIFAR10(
        root=data_root,
        train=True,
        download=True,
    )

    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]
    )

    val_dataset = torchvision.datasets.CIFAR10(
        root=data_root,
        train=False,
        download=True,
    )

    val_transform = transforms.Compose([transforms.ToTensor(), normalize])

    untransform = transforms.Compose([
        util.UnNormalize(mean=normalize.mean, std=normalize.std),
        transforms.ToPILImage()
    ])

    train_dataset = classifier.ClassDataset(train_dataset, train_transform, untransform, labels=range(10),
                                            label_names=train_dataset.classes)
    val_dataset = classifier.ClassDataset(val_dataset, val_transform, untransform, labels=range(10),
                                          label_names=val_dataset.classes)
    return train_dataset, val_dataset, None

def pgn_get_dataset(cfg, dmodule):
    data_root = os.path.join(cfg.main.data_path)
    datamodule = dmodule(data_root, 0, 0, 0, 1.0, 0.8, 0.2, 0.2)
    datamodule.setup('fit')

    train_dataset = datamodule.train_set
    val_dataset = datamodule.val_set

    transform = torch.nn.Identity()
    mean = (0.48145466, 0.4578275, 0.40821073)
    std = (0.26862954, 0.26130258, 0.27577711)
    untransform = transforms.Compose([
        util.UnNormalize(mean=mean, std=std),
        transforms.ToPILImage()
    ])

    train_dataset = classifier.ClassDataset(train_dataset, transform, untransform, labels=range(10), label_names=train_dataset.classes)
    val_dataset = classifier.ClassDataset(val_dataset, transform, untransform, labels=range(10), label_names=val_dataset.classes)

    return train_dataset, val_dataset, None

# 224x224 resolution in accordance with most imagenet pretrained models
def get_cifar10pgn(cfg):
    return pgn_get_dataset(cfg, pgn.datamodules.cifar10_datamodule.CIFAR10DataModule)

def get_flowers(cfg):
    return pgn_get_dataset(cfg, pgn.datamodules.flowers_datamodule.Flowers102DataModule)