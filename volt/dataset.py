import os
import random
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
import torchvision
from PIL import ImageFilter
from torch import nn
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import transforms

import pgn
from pgn import datamodules
from volt import util


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


class ClassDataset(SliceableDataset):
    def __init__(self, dataset, transform, inverse_transform, classes=None, labels=None, label_names=None):
        super().__init__()
        self.dataset = dataset
        self.transform = transform
        self.inverse_transform = inverse_transform

        if classes is None:
            assert labels is not None, 'Must provide either classes or labels and label_names'
            assert label_names is not None, 'Must provide either classes or labels and label_names'

            self.labels = labels
            self.label_names = label_names

            self.classes = {i: name for i, name in zip(self.labels, self.label_names)}
        else:
            assert self.classes is not None, 'Must provide either classes or labels and label_names'
            self.classes = classes

            items = sorted(self.classes.items(), key=lambda x: x[0])
            self.labels = [i for i, _ in items]
            self.label_names = [name for _, name in items]

    def get_item(self, index):
        image, label = self.dataset[index]
        image_transformed = self.transform(image)
        return image_transformed, label

    def __len__(self):
        return len(self.dataset)

    def untransform(self, x):
        return self.inverse_transform(x)


class ClassDatasetSubset(ClassDataset):
    r"""
    Subset of a dataset at specified indices.

    Args:
        dataset (Dataset): The whole Dataset
        indices (sequence): Indices in the whole set selected for subset
    """

    def __init__(self, indices: Sequence[int], *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.indices = indices

    def get_item(self, idx):
        if isinstance(idx, list):
            return super().get_item([self.indices[i] for i in idx])
        return super().get_item(self.indices[idx])

    def __len__(self):
        return len(self.indices)


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

    train_dataset = ClassDataset(train_dataset, transform, untransform, labels=range(10),
                                 label_names=train_dataset.classes)
    val_dataset = ClassDataset(val_dataset, transform, untransform, labels=range(10),
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

    train_dataset = ClassDataset(train_dataset, train_transform, untransform, labels=range(10),
                                 label_names=train_dataset.classes)
    val_dataset = ClassDataset(val_dataset, val_transform, untransform, labels=range(10),
                               label_names=val_dataset.classes)
    return train_dataset, val_dataset, None


def pgn_get_dataset(cfg, dmodule):
    data_root = os.path.join(cfg.main.data_path)
    datamodule = dmodule(data_root, 0, 0, 0, 1.0, 0.8, 0.2, 0.2)
    datamodule.setup('fit')

    train_dataset = datamodule.train_set
    val_dataset = datamodule.val_set
    train_dataset_unaugmented = datamodule.train_set_unaugmented

    transform = torch.nn.Identity()
    mean = (0.48145466, 0.4578275, 0.40821073)
    std = (0.26862954, 0.26130258, 0.27577711)
    untransform = transforms.Compose([
        util.UnNormalize(mean=mean, std=std),
        transforms.ToPILImage()
    ])

    classes = list(map(lambda x: x[0], sorted(list(train_dataset.class_to_idx.items()), key=lambda x: x[1])))
    assert all([train_dataset.class_to_idx[c] == i for i, c in enumerate(classes)])
    train_dataset = ClassDataset(train_dataset, transform, untransform, labels=range(len(classes)),
                                 label_names=classes)
    classes = list(map(lambda x: x[0], sorted(list(val_dataset.class_to_idx.items()), key=lambda x: x[1])))
    assert all([val_dataset.class_to_idx[c] == i for i, c in enumerate(classes)])
    val_dataset = ClassDataset(val_dataset, transform, untransform, labels=range(len(classes)),
                               label_names=classes)

    classes = list(
        map(lambda x: x[0], sorted(list(train_dataset_unaugmented.class_to_idx.items()), key=lambda x: x[1])))
    assert all([train_dataset_unaugmented.class_to_idx[c] == i for i, c in enumerate(classes)])
    train_dataset_unaugmented = ClassDataset(train_dataset_unaugmented, transform, untransform,
                                             labels=range(len(classes)), label_names=classes)

    return train_dataset, val_dataset, None, train_dataset_unaugmented

def get_dataset(cfg):
    name = cfg.main.dataset
    if name == 'cifar10':
        return get_cifar10(cfg)
    elif name == 'mnist':
        return get_mnist(cfg)
    elif name == 'cifar10pgn':
        return pgn_get_dataset(cfg, pgn.datamodules.cifar10_datamodule.CIFAR10DataModule)
    elif name == 'cifar100':
        return pgn_get_dataset(cfg, pgn.datamodules.cifar100_datamodule.CIFAR100DataModule)
    elif name == 'flowers':
        return pgn_get_dataset(cfg, pgn.datamodules.flowers102_datamodule.Flowers102DataModule)
    elif name == 'sun397':
        return pgn_get_dataset(cfg, pgn.datamodules.sun397_datamodule.SUN397DataModule)
    elif name == 'clevr_count':
        return pgn_get_dataset(cfg, pgn.datamodules.clevr_count_datamodule.CLEVRCountDataModule)
    elif name == 'dtd':
        return pgn_get_dataset(cfg, pgn.datamodules.dtd_datamodule.DTDDataModule)
    elif name == 'eurosat':
        return pgn_get_dataset(cfg, pgn.datamodules.eurosat_datamodule.EuroSATDataModule)
    elif name == 'food101':
        return pgn_get_dataset(cfg, pgn.datamodules.food101_datamodule.Food101DataModule)
    elif name == 'oxfordpets':
        return pgn_get_dataset(cfg, pgn.datamodules.oxfordpets_datamodule.OxfordPetsDataModule)
    elif name == 'resisc45':
        return pgn_get_dataset(cfg, pgn.datamodules.resisc45_datamodule.RESISC45DataModule)
    elif name == 'svhn':
        return pgn_get_dataset(cfg, pgn.datamodules.svhn_datamodule.SVHNDataModule)
    elif name == 'ucf101':
        return pgn_get_dataset(cfg, pgn.datamodules.ucf101_datamodule.UCF101DataModule)



# 224x224 resolution in accordance with most imagenet pretrained models
def get_cifar10pgn(cfg):
    return pgn_get_dataset(cfg, pgn.datamodules.cifar10_datamodule.CIFAR10DataModule)


def get_flowers(cfg):
    return pgn_get_dataset(cfg, pgn.datamodules.flowers102_datamodule.Flowers102DataModule)


def get_sun(cfg):
    return pgn_get_dataset(cfg, pgn.datamodules.sun397_datamodule.SUN397DataModule)


class MultiCropDataset(SliceableDataset):
    def __init__(
            self,
            dataset,
            size_crops,
            nmb_crops,
            min_scale_crops,
            max_scale_crops,
            return_index=True,
    ):
        super().__init__()
        assert len(size_crops) == len(nmb_crops)
        assert len(min_scale_crops) == len(nmb_crops)
        assert len(max_scale_crops) == len(nmb_crops)
        self.return_index = return_index

        self.dataset = dataset

        color_transform = [get_color_distortion(), PILRandomGaussianBlur()]
        mean = [0.485, 0.456, 0.406]
        std = [0.228, 0.224, 0.225]
        trans = []
        for i in range(len(size_crops)):
            randomresizedcrop = transforms.RandomResizedCrop(
                size_crops[i],
                scale=(min_scale_crops[i], max_scale_crops[i]),
            )
            trans.extend([transforms.Compose([
                randomresizedcrop,
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.Compose(color_transform),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)])
                         ] * nmb_crops[i])
        self.trans = trans

        self.untransform = transforms.Compose([
            util.UnNormalize(mean=mean, std=std),
            transforms.ToPILImage()
        ])
        self.labels = self.dataset.labels
        self.label_names = self.dataset.label_names

        self.classes = self.dataset.classes

    def __len__(self):
        return len(self.dataset)

    def get_item(self, index):
        image, label = self.dataset[index]
        multi_crops = list(map(lambda trans: trans(image), self.trans))
        if self.return_index:
            return index, multi_crops, label
        return multi_crops, label

    def untransform(self, x):
        return self.untransform(x)


class PILRandomGaussianBlur(object):
    """
    Apply Gaussian Blur to the PIL image. Take the radius and probability of
    application as the parameter.
    This transform was used in SimCLR - https://arxiv.org/abs/2002.05709
    """

    def __init__(self, p=0.5, radius_min=0.1, radius_max=2.):
        self.prob = p
        self.radius_min = radius_min
        self.radius_max = radius_max

    def __call__(self, img):
        do_it = np.random.rand() <= self.prob
        if not do_it:
            return img

        return img.filter(
            ImageFilter.GaussianBlur(
                radius=random.uniform(self.radius_min, self.radius_max)
            )
        )


def get_color_distortion(s=1.0):
    # s is the strength of color distortion.
    color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
    rnd_color_jitter = transforms.RandomApply([color_jitter], p=0.8)
    rnd_gray = transforms.RandomGrayscale(p=0.2)
    color_distort = transforms.Compose([rnd_color_jitter, rnd_gray])
    return color_distort


def get_multicrop_dataset(cfg):
    dataset_name = cfg.main.dataset

    test_transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize(size=cfg.swav.size_crops[0]),
        torchvision.transforms.CenterCrop(size=cfg.swav.size_crops[0]),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
            mean=(0.48145466, 0.4578275, 0.40821073),
            std=(0.26862954, 0.26130258, 0.27577711)
        ),
    ])

    if dataset_name == 'multicrop_cifar10':
        train_dataset = torchvision.datasets.CIFAR10(
            root=cfg.main.data_path, train=True, download=True
        )
        val_dataset = torchvision.datasets.CIFAR10(
            root=cfg.main.data_path, train=False, download=True
        )

        train_dataset_unaugmented = torchvision.datasets.CIFAR10(
            root=cfg.main.data_path, train=True, download=True, transform=test_transform)
        val_dataset_unaugmented = torchvision.datasets.CIFAR10(
            root=cfg.main.data_path, train=False, download=True, transform=test_transform)

        train_dataset = ClassDataset(train_dataset, nn.Identity(), nn.Identity(), labels=range(10),
                                     label_names=train_dataset.classes)
        val_dataset = ClassDataset(val_dataset, nn.Identity(), nn.Identity(), labels=range(10),
                                   label_names=val_dataset.classes)
        train_dataset_unaugmented = ClassDataset(train_dataset_unaugmented, nn.Identity(), nn.Identity(),
                                                 labels=range(10), label_names=train_dataset_unaugmented.classes)
        val_dataset_unaugmented = ClassDataset(val_dataset_unaugmented, nn.Identity(), nn.Identity(),
                                                  labels=range(10), label_names=val_dataset_unaugmented.classes)
    else:
        raise NotImplementedError()

    train_dataset_mc = MultiCropDataset(train_dataset, cfg.swav.size_crops, cfg.swav.nmb_crops,
                                        cfg.swav.min_scale_crops, cfg.swav.max_scale_crops)
    val_dataset_mc = MultiCropDataset(val_dataset, cfg.swav.size_crops, cfg.swav.nmb_crops, cfg.swav.min_scale_crops,
                                      cfg.swav.max_scale_crops)

    # TODO
    return train_dataset_mc, val_dataset_mc, None, train_dataset_unaugmented, val_dataset_unaugmented
