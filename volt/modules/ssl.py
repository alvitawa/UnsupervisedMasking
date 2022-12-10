from dataclasses import dataclass
from typing import Iterable

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import transforms

from volt import config
from volt.modules.classifier import ClassifierModule
from volt.modules.deep_learning import DeepLearningModule


class MultiCropDataset(Dataset):
    def __init__(
        self,
        dataset,
        size_crops,
        nmb_crops,
        min_scale_crops,
        max_scale_crops,
        return_index=False,
    ):
        assert len(size_crops) == len(nmb_crops)
        assert len(min_scale_crops) == len(nmb_crops)
        assert len(max_scale_crops) == len(nmb_crops)
        self.return_index = return_index

        self.dataset = dataset

        # color_transform = [get_color_distortion(), PILRandomGaussianBlur()]
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
                # transforms.RandomHorizontalFlip(p=0.5),
                # transforms.Compose(color_transform),
                # transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)])
            ] * nmb_crops[i])
        self.trans = trans

    def __getitem__(self, index):
        image, label = self.dataset[index]
        multi_crops = list(map(lambda trans: trans(image), self.trans))
        if self.return_index:
            return index, multi_crops, label
        return multi_crops, label

    def __len__(self):
        return len(self.dataset)

# class MultiPrototypes(nn.Module):
#     def __init__(self, output_dim, nmb_prototypes):
#         super(MultiPrototypes, self).__init__()
#         self.nmb_heads = len(nmb_prototypes)
#         for i, k in enumerate(nmb_prototypes):
#             self.add_module("prototypes" + str(i), nn.Linear(output_dim, k, bias=False))
#
#     def forward(self, x):
#         out = []
#         for i in range(self.nmb_heads):
#             out.append(getattr(self, "prototypes" + str(i))(x))
#         return out

@dataclass
class SWAVConfig:
    nmb_crops: list = (2, 6)
    size_crops: list = (224, 96)
    min_scale_crops: list = (0.14, 0.05)
    max_scale_crops: list = (1, 0.14)
    nmb_prototypes: int = 3000
    feat_dim: int = 128
    queue_length: int = 3840
    epoch_queue_starts: int = 15
    crops_for_assign: list = (0, 1)
    temperature: float = 0.1
    freeze_prototypes_niters: int = 5005
    epsilon: float = 0.05
    sinkhorn_iterations: int = 3



config.register('swav', SWAVConfig)


class SWAVModule(DeepLearningModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.multicrop_dataset = {namespace: MultiCropDataset(self.dataset[namespace], self.cfg.swav.size_crops,
                                                              self.cfg.swav.nmb_crops, self.cfg.swav.min_scale_crops,
                                                              self.cfg.swav.max_scale_crops, return_index=True)
                                  for namespace in self.dataset.keys()}

        # prototype layer
        self.prototypes = None
        # if isinstance(self.cfg.swav.nmb_prototypes, Iterable):
        #     self.prototypes = MultiPrototypes(self.cfg.swav.feat_dim, self.cfg.swav.nmb_prototypes)
        # el
        if self.cfg.swav.nmb_prototypes > 0:
            self.prototypes = nn.Linear(self.cfg.swav.feat_dim, self.cfg.swav.nmb_prototypes, bias=False)

        self.queue = None
        # the queue needs to be divisible by the batch size
        self.cfg.swav.queue_length -= self.cfg.swav.queue_length % (self.cfg.dl.batch_size * 1)

    def dataloader(self, namespace, shuffle):
        return DataLoader(self.multicrop_dataset[namespace], batch_size=self.cfg.dl.batch_size, shuffle=shuffle,
                          num_workers=self.cfg.dl.num_workers)

    def on_train_epoch_start(self) -> None:
        if self.cfg.swav.queue_length > 0 and self.current_epoch >= self.cfg.swav.epoch_queue_starts and self.queue is None:
            self.queue = torch.zeros(
                len(self.cfg.swav.crops_for_assign),
                self.cfg.swav.queue_length // 1,
                self.cfg.swav.feat_dim,
            ).to(self.device)

    def training_step(self, batch, batch_idx, namespace='train'):
        index, multi_crops, _label = batch

        if namespace == 'train':
            # normalize the prototypes
            with torch.no_grad():
                w = self.prototypes.weight.data.clone()
                w = nn.functional.normalize(w, dim=1, p=2)
                self.prototypes.weight.copy_(w)

        # The crops have different shapes, so they need to be batched separately
        idx_crops = torch.cumsum(torch.unique_consecutive(
            torch.tensor([inp.shape[-1] for inp in multi_crops]),
            return_counts=True,
        )[1], 0)
        start_idx = 0
        outs = []
        for end_idx in idx_crops:
            _out = self.model(torch.cat(multi_crops[start_idx: end_idx]).to(self.device))
            outs.append(_out)
            # if start_idx == 0:
            #     output = _out
            # else:
            #     output = torch.cat((output, _out))
            start_idx = end_idx
        output = torch.cat(outs)
        embedding = output

        # embedding = self.model(multi_crops)
        output = self.prototypes(embedding)
        embedding = embedding.detach()
        bs = multi_crops[0].size(0)

        # ============ swav loss ... ============
        loss = 0
        for i, crop_id in enumerate(self.cfg.swav.crops_for_assign):
            with torch.no_grad():
                out = output[bs * crop_id: bs * (crop_id + 1)].detach()

                # time to use the queue
                if self.queue is not None:
                    if use_the_queue or not torch.all(self.queue[i, -1, :] == 0):
                        use_the_queue = True
                        out = torch.cat((torch.mm(
                            self.queue[i],
                            self.prototypes.weight.t()
                        ), out))
                    # fill the queue
                    self.queue[i, bs:] = self.queue[i, :-bs].clone()
                    self.queue[i, :bs] = embedding[crop_id * bs: (crop_id + 1) * bs]

                # get assignments
                q = self.sinkhorn(out)[-bs:]

            # cluster assignment prediction
            subloss = 0
            for v in np.delete(np.arange(np.sum(self.cfg.swav.nmb_crops)), crop_id):
                x = output[bs * v: bs * (v + 1)] / self.cfg.swav.temperature
                subloss -= torch.mean(torch.sum(q * torch.nn.functional.log_softmax(x, dim=1), dim=1))
            loss = loss + subloss / (np.sum(self.cfg.swav.nmb_crops) - 1)

        loss = loss / len(self.cfg.swav.crops_for_assign)

        self.log(f'{namespace}/loss', loss)
        return {'loss': loss, 'target': _label}

    def on_after_backward(self):
        if self.global_step < self.cfg.swav.freeze_prototypes_niters:
            for p in self.prototypes.parameters():
                p.grad = None



    @torch.no_grad()
    def sinkhorn(self, out):
        Q = torch.exp(out / self.cfg.swav.epsilon).t() # Q is K-by-B for consistency with notations from our paper
        B = Q.shape[1] * 1 # number of samples to assign
        K = Q.shape[0] # how many prototypes

        # make the matrix sums to 1
        sum_Q = torch.sum(Q)
        # dist.all_reduce(sum_Q)
        Q /= sum_Q

        for it in range(self.cfg.swav.sinkhorn_iterations):
            # normalize each row: total weight per prototype must be 1/K
            sum_of_rows = torch.sum(Q, dim=1, keepdim=True)
            # dist.all_reduce(sum_of_rows)
            Q /= sum_of_rows
            Q /= K

            # normalize each column: total weight per sample must be 1/B
            Q /= torch.sum(Q, dim=0, keepdim=True)
            Q /= B

        Q *= B # the colomns must sum to 1 so that Q is an assignment
        return Q.t()
