from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from subnetworks import MaskedLinear, submasking, SubmaskedModel
from volt import config
from volt.modules.classifier import ClassifierModule


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
    size_crops: list = (224, 96)  # Should be from big to small
    min_scale_crops: list = (0.14, 0.05)
    max_scale_crops: list = (1, 0.14)
    nmb_prototypes: int = 3000
    feat_dim: int = 128
    queue_length: int = 3840
    epoch_queue_starts: int = 15
    crops_for_assign: list = (0, 1)
    temperature: float = 0.1
    freeze_prototypes_niters: int = 5005 #313
    epsilon: float = 0.05
    sinkhorn_iterations: int = 3
    fc: str = 'linear'
    prototypes: str = 'linear'



config.register('swav', SWAVConfig)

# knn monitor as in InstDisc https://arxiv.org/abs/1805.01978
# implementation follows http://github.com/zhirongw/lemniscate.pytorch and https://github.com/leftthomas/SimCLR

class SWAVModule(ClassifierModule):
    def __init__(self, fc_in_features, *args, **kwargs):
        super().__init__(fc_in_features, *args, **kwargs)

        self.args = args
        self.kwargs = kwargs

        # assert self.cfg.cls.loss == ''
        # assert self.cfg.cls.fc == ''

        # prototype layer
        self.prototypes = None
        # if isinstance(self.cfg.swav.nmb_prototypes, Iterable):
        #     self.prototypes = MultiPrototypes(self.cfg.swav.feat_dim, self.cfg.swav.nmb_prototypes)
        # el
        if self.cfg.swav.nmb_prototypes > 0:
            if self.cfg.swav.prototypes == 'linear':
                self.prototypes = nn.Linear(self.cfg.swav.feat_dim, self.cfg.swav.nmb_prototypes, bias=False)
            elif self.cfg.swav.prototypes == 'masked':
                # lin = nn.Linear(self.cfg.swav.feat_dim, self.cfg.swav.nmb_prototypes, bias=False).to(self.device_)
                # sc_init = submasking.normal_scores_init(1, 0)
                # self.prototypes = SubmaskedModel(lin, shell_mode='replace', scores_init=sc_init)
                self.prototypes = MaskedLinear(self.cfg.swav.feat_dim, self.cfg.swav.nmb_prototypes, 1)
                with torch.no_grad():
                    w = self.prototypes.W.weight.data.clone()
                    w = nn.functional.normalize(w, dim=1, p=2)
                    self.prototypes.W.weight.copy_(w)

        self.queue = None
        # the queue needs to be divisible by the batch size
        self.cfg.swav.queue_length -= self.cfg.swav.queue_length % (self.cfg.dl.batch_size * 1)

        self.use_the_queue = False

        if self.cfg.swav.fc == 'linear':
            self.fc = nn.Linear(fc_in_features, self.cfg.swav.feat_dim)
        elif self.cfg.swav.fc == 'masked':
            lin = nn.Linear(fc_in_features, self.cfg.swav.feat_dim).to(self.device_)
            sc_init = submasking.normal_scores_init(1, 0)
            self.fc = SubmaskedModel(lin, shell_mode='replace', scores_init=sc_init)


    # def dataloader(self, namespace, shuffle):
    #     return DataLoader(self.multicrop_dataset[namespace], batch_size=self.cfg.dl.batch_size, shuffle=shuffle,
    #                       num_workers=self.cfg.dl.num_workers)

    def probe(self):
        self.supercast()
        super().probe()

        # change the class as well
        # self.__class__ = ClassifierModule

    def supercast(self):
        self.training_step = super().training_step
        self.configure_optimizers = super().configure_optimizers
        self.on_train_epoch_start = super().on_train_epoch_start
        self.on_after_backward = super().on_after_backward
        super().__init__(self.fc.in_features, *self.args, **self.kwargs)


    def configure_optimizers(self, _head_params=None):
        # From self.fc and self.prototypes
        head_params = list(self.fc.parameters()) + list(self.prototypes.parameters())
        return super().configure_optimizers(head_params)

    def on_train_epoch_start(self) -> None:
        if self.cfg.swav.queue_length > 0 and self.current_epoch >= self.cfg.swav.epoch_queue_starts and self.queue is None:
            self.queue = torch.zeros(
                len(self.cfg.swav.crops_for_assign),
                self.cfg.swav.queue_length // 1,
                self.cfg.swav.feat_dim,
            ).to(self.device)

    def training_step(self, batch, batch_idx, namespace='train'):
        index, multi_crops, _label = batch

        if namespace == 'train' and self.cfg.swav.prototypes == 'linear':
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
        backbone_outs = []
        for end_idx in idx_crops:
            backbone_out = self.model(torch.cat(multi_crops[start_idx: end_idx]).to(self.device))
            backbone_outs.append(backbone_out.detach())
            _out = self.fc(backbone_out)
            outs.append(_out)
            # if start_idx == 0:
            #     output = _out
            # else:
            #     output = torch.cat((output, _out))
            start_idx = end_idx
        output = torch.cat(outs)
        embedding = nn.functional.normalize(output, dim=1, p=2)

        backbone_out = torch.cat(backbone_outs)
        backbone_embedding = nn.functional.normalize(backbone_out, dim=1, p=2)

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
                    if self.use_the_queue or not torch.all(self.queue[i, -1, :] == 0):
                        self.use_the_queue = True
                        # out = torch.cat((torch.mm(
                        #     self.queue[i],
                        #     self.prototypes.weight.t()
                        # ), out))
                        out = torch.cat((self.prototypes(self.queue[i]), out))
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

        # import pdb; pdb.set_trace()
        self.log(f'{namespace}/loss', loss)
        r = {'loss': loss, 'target': _label}
        if namespace in ['val', 'test']:
            r['embedding'] = backbone_embedding.view(sum(self.cfg.swav.nmb_crops), bs, self.fc_in_features).mean(dim=0)
        return r

    # def analyze(self, outputs, sample_inputs, sample_outputs, namespace):
    #     super().analyze(outputs, sample_inputs, sample_outputs, namespace)
    #     if namespace == 'val':
    #         feature_bank = outputs['embedding'].T.contiguous()
    #         feature_labels = outputs['target']
    #         classes = len(self.dataset['val'].label_names)
    #         if self.cfg.main.in_mvp:
    #             knn_k = 4
    #         else:
    #             knn_k = 200
    #             assert len(feature_bank) // classes / 2 < knn_k, 'knn_k is too large'
    #         pred_labels = knn_predict(outputs['embedding'], feature_bank, feature_labels, classes, knn_k, 0.1)
    #         accuracy = (pred_labels[:, 0] == feature_labels).float().sum().item() / outputs['embedding'].shape[0]
    #         self.logger_.experiment[f'training/{namespace}/knn_accuracy'].log(accuracy)



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
