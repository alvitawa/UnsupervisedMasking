import math
from dataclasses import dataclass
from typing import List

import torch
from pytorch_lightning import LightningModule
from torch.utils.data import DataLoader

from subnetworks.submasking import SubmaskedModel
from volt import log
from volt.util import ddict

from volt import config

from torch.utils.data._utils.collate import default_collate


@dataclass
class DeepLearningConfig:
    shuffle: bool = True
    batch_size: int = 128
    forward_batch_size: int = 8192
    epochs: int = 100
    num_workers: int = 4
    batch_accum: int = 1
    optimizer: str = 'adam'
    lr: float = 0.0001
    weight_decay: float = 0
    momentum: float = 0
    scheduler: str = 'none'
    warmup_length: int = 0
    analyze_train_every: int = 1
    analyze_val_every: int = 1
    analyze_sample_size: int = 2


config.register('deep_learning', DeepLearningConfig)


class DeepLearningModule(LightningModule):
    def __init__(self, cfg, model, device, logger, train_dataset=None, val_dataset=None, test_dataset=None):
        super().__init__()
        self.scheduler_ = None
        self.optimizer_ = None
        self.dataset = ddict(train=train_dataset, val=val_dataset,
                             test=test_dataset if test_dataset is not None else val_dataset)
        self.cfg = cfg
        self.model = model
        self.device_ = device
        self.logger_ = logger

        self.analyze_every = ddict(train=cfg.dl.analyze_train_every, val=cfg.dl.analyze_val_every, test=1)

    def dataloader(self, namespace, shuffle):
        return DataLoader(self.dataset[namespace], batch_size=self.cfg.dl.batch_size, shuffle=shuffle,
                          num_workers=self.cfg.dl.num_workers)

    def forward_dataloader(self, namespace):
        return self.dataloader(namespace, False)

    def train_dataloader(self):
        return self.dataloader('train', self.cfg.dl.shuffle)

    def val_dataloader(self):
        return self.forward_dataloader('val')

    def test_dataloader(self):
        return self.forward_dataloader('test')

    def get_ctx(self):
        return ddict(epoch=self.current_epoch)#, lr=self.optimizer_.param_groups[0]['lr'])

    def forward(self, x):
        if isinstance(self.model, SubmaskedModel):
            ctx = self.get_ctx()
            out = self.model(x, ctx=ctx)
        else:
            out = self.model(x)
        return out

    def analyze(self, outputs, sample_inputs, sample_outputs, namespace):
        if hasattr(self.model, 'analyze'):
            model_analysis = self.model.analyze(self.get_ctx())
            for k, v in model_analysis.items():
                self.logger_.experiment[f'training/{namespace}/model_analysis/{k}'].log(v)

    def training_step(self, batch, batch_idx, namespace='train'):
        raise NotImplementedError()

    def training_epoch_end(self, outputs, namespace='train'):
        # concatenate outputs, which is collating but by concatenation rather than stacking
        outputs = {k: torch.cat([x[k] if len(x[k].shape) > 0 else x[k].unsqueeze(0) for x in outputs], dim=0) for k in
                   outputs[0].keys()}

        loss = outputs['loss'].mean()
        self.log(f'{namespace}/loss_epoch', loss)

        with torch.no_grad():

            if self.current_epoch % self.analyze_every[namespace] == 0:

                if self.cfg.dl.analyze_sample_size > 0:
                    subset = torch.randperm(len(self.dataset[namespace]))[:self.cfg.dl.analyze_sample_size]
                    sample_inputs = default_collate([self.dataset[namespace][i] for i in subset])
                    # Move to device
                    if isinstance(sample_inputs, dict):
                        sample_inputs = {k: v.to(self.device_) for k, v in sample_inputs.items()}
                    elif isinstance(sample_inputs, (list, tuple)):
                        sample_inputs = [v.to(self.device_) for v in sample_inputs]
                    else:
                        sample_inputs = sample_inputs.to(self.device_)

                    sample_outputs = self.training_step(sample_inputs, -1, namespace=f'{namespace}_sample')
                else:
                    sample_inputs = None
                    sample_outputs = None

        self.analyze(outputs, sample_inputs, sample_outputs, namespace)

    def validation_step(self, batch, batch_idx, namespace='val'):
        result = self.training_step(batch, batch_idx, namespace=namespace)
        # self.logger.experiment.wait()
        return result

    def validation_epoch_end(self, outputs):
        self.training_epoch_end(outputs, namespace='val')

    def test_step(self, batch, batch_idx):
        return self.training_step(batch, batch_idx, namespace='test')

    def test_epoch_end(self, outputs):
        self.training_epoch_end(outputs, namespace='test')

    def configure_optimizers(self):
        if self.cfg.dl.optimizer == 'adam':
            optimizer = torch.optim.Adam(self.parameters(), lr=self.cfg.dl.lr, weight_decay=self.cfg.dl.weight_decay)
        elif self.cfg.dl.optimizer == 'lbfgs':
            optimizer = torch.optim.LBFGS(self.parameters(),  # lr=self.cfg.dl.lr, #, history_size=1000,
                                          # max_iter=100,
                                          line_search_fn="strong_wolfe")
        elif self.cfg.dl.optimizer == 'sgd':
            optimizer = torch.optim.SGD(self.parameters(), lr=self.cfg.dl.lr, weight_decay=self.cfg.dl.weight_decay,
                                        momentum=self.cfg.dl.momentum)
        else:
            raise ValueError(f'Unknown optimizer {self.cfg.dl.optimizer}')

        if self.cfg.dl.scheduler == 'none':
            return optimizer
        elif self.cfg.dl.scheduler == 'cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                                   self.cfg.dl.epochs - self.cfg.dl.warmup_length)
        elif self.cfg.dl.scheduler == 'cosine_warmup':
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=
                lambda epoch:
                    (epoch + 1) / self.cfg.dl.warmup_length
                    if epoch < self.cfg.dl.warmup_length
                    else 0.5 * (1. + math.cos(math.pi * (epoch - self.cfg.dl.warmup_length)
                                              / (self.cfg.dl.epochs - self.cfg.dl.warmup_length))))
        else:
            raise ValueError(f'Unknown scheduler {self.cfg.dl.scheduler}')


        self.scheduler_ = scheduler
        self.optimizer_ = optimizer
        return [optimizer], [scheduler]


def load_model_checkpoint(best_model_path, model):
    checkpoint_path = best_model_path
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['state_dict'])
    log.logger.info(f'Loaded model checkpoint from {best_model_path}')
