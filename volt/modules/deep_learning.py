import copy
import math
import os
import random
from dataclasses import dataclass

import numpy
import numpy as np
import torch
from pytorch_lightning import LightningModule
from torch.utils.data import DataLoader

from subnetworks.submasking import SubmaskedModel
from volt import log, util
from volt.util import ddict, move_to_device

from volt import config

from torch.utils.data._utils.collate import default_collate


@dataclass
class DeepLearningConfig:
    shuffle: bool = True
    batch_size: int = 128
    forward_batch_size: int = 8192
    epochs: int = 100
    num_workers: int = 8
    num_workers_auto: bool = False
    batch_accum: int = 1
    optimizer: str = 'adam'
    lr: float = 0.0001
    weight_decay: float = 0
    momentum: float = 0
    scheduler: str = 'none'
    scheduler_step: int = 50
    scheduler_gamma: float = 0.1
    eta_min: float = 0
    cooldown: int = 50
    start_lr: float = 0.3
    final_lr: float = 0.0048
    base_lr: float = 4.8
    warmup_epochs: int = 10
    analyze_train_every: int = 1
    analyze_val_every: int = 1
    analyze_sample_size: int = 2
    multiprocessing_context: str = 'default'

config.register('deep_learning', DeepLearningConfig)

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    numpy.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)


# Moves all tensors in a nested data structure (lists, dicts, tuples, ... etc) to self.device


class DeepLearningModule(LightningModule):
    def __init__(self, cfg, model, device, logger, train_dataset=None, val_dataset=None, test_dataset=None,
                 train_dataset_unaugmented=None, val_dataset_unaugmented=None, test_dataset_unaugmented=None):
        super().__init__()
        self.scheduler_ = None
        self.optimizer_ = None
        self.dataset = ddict(train=train_dataset, val=val_dataset,
                             test=test_dataset if test_dataset is not None else val_dataset,
                             train_unaugmented=train_dataset_unaugmented, val_unaugmented=val_dataset_unaugmented,
                             test_unaugmented=test_dataset_unaugmented if test_dataset_unaugmented is not None else val_dataset_unaugmented)

        self.cfg = cfg
        self.model = model
        self.device_ = device
        self.logger_ = logger

        self.analyze_every = ddict(train=cfg.dl.analyze_train_every, val=cfg.dl.analyze_val_every, test=1)

        self.learning_rate = cfg.dl.lr

        if cfg.dl.num_workers_auto:
            cfg.dl.num_workers = os.cpu_count()

    def dataloader(self, namespace, shuffle):
        g = torch.Generator()
        g.manual_seed(torch.initial_seed())
        return DataLoader(self.dataset[namespace], batch_size=self.cfg.dl.batch_size, shuffle=shuffle,
                          num_workers=self.cfg.dl.num_workers,
                          multiprocessing_context=self.cfg.dl.multiprocessing_context if self.cfg.dl.multiprocessing_context != 'default' else None)  # , worker_init_fn=seed_worker, generator=g)

    def forward_dataloader(self, namespace):
        return self.dataloader(namespace, False)

    def train_dataloader(self):
        return self.dataloader('train', self.cfg.dl.shuffle)

    def val_dataloader(self):
        return self.forward_dataloader('val')

    def test_dataloader(self):
        return self.forward_dataloader('test')

    def get_ctx(self):
        return ddict(epoch=self.current_epoch)  # , lr=self.optimizer_.param_groups[0]['lr'])

    def forward(self, x):
        if isinstance(self.model, SubmaskedModel):
            ctx = self.get_ctx()
            out = self.model(x, ctx=ctx)
        else:
            out = self.model(x)
        return out

    def process_samples(self, samples, name, namespace, path='root'):
        if isinstance(samples, torch.Tensor):
            if len(samples.shape) == 4 and samples.shape[1] in [1, 3]:
                images = [self.dataset[namespace].untransform(x) for x in samples]
                value = util.get_image_grid(images, tensor=False)
            # elif len(samples.shape) == 3 and samples.shape[0] == 3:
            #     value = self.dataset[namespace].untransform(samples)
            # elif len(samples.shape) == 3 and samples.shape[0] == 1:
            #     value = self.dataset[namespace].untransform(samples)
            # elif len(samples.shape) == 2:
            #     value = util.tensor_to_pil(samples.unsqueeze(0))
            # elif len(samples.shape) == 1 and samples.shape[0] == 1:
            #     value = samples.detach().cpu().numpy()[0]
            # elif len(samples.shape) == 0:
            #     value = samples.item()
            elif samples.numel() == 1:
                value = samples.item()
            elif samples.numel() < 100:
                value = str(samples)
            else:
                return
            self.logger_.experiment[f'training/{namespace}/{name}/{path}'].log(value)
        elif isinstance(samples, dict):
            for key in samples:
                self.process_samples(samples[key], name, namespace, f'{path}/key_{key}')
        elif isinstance(samples, (list, tuple)):
            for i, value in enumerate(samples):
                self.process_samples(value, name, namespace, f'{path}/index_{i}')
        elif isinstance(samples, (str, int, float, bool, np.generic)):
            self.logger_.experiment[f'training/{namespace}/{name}/{path}'].log(samples)
        else:
            return

    def analyze(self, outputs, sample_inputs, sample_outputs, namespace):
        if hasattr(self.model, 'analyze'):
            model_analysis = self.model.analyze(self.get_ctx())
            if hasattr(self.model, 'depth_analysis') and namespace == 'test':
                # Do it on a model copy to prevent corruption of the original model
                model_copy = copy.deepcopy(self.model)
                for k, v in model_copy.depth_analysis(self.get_ctx()).items():
                    self.logger_.experiment[f'training/{namespace}/depth_analysis/{k}'].log(v)
            for k, v in model_analysis.items():
                self.logger_.experiment[f'training/{namespace}/model_analysis/{k}'].log(v)
        if sample_inputs is not None:
            self.process_samples(sample_inputs, 'sample_inputs', namespace)
        if sample_outputs is not None:
            self.process_samples(sample_outputs, 'sample_outputs', namespace)

    def training_step(self, batch, batch_idx, namespace='train'):
        raise NotImplementedError()

    def training_epoch_end(self, outputs, namespace='train'):
        # concatenate outputs, which is collating but by concatenation rather than stacking
        outputs = util.concat_dict_list(outputs)

        loss = outputs['loss'].mean()
        self.log(f'{namespace}/loss_epoch', loss)

        with torch.no_grad():

            if self.current_epoch % self.analyze_every[namespace] == 0:

                if self.cfg.dl.analyze_sample_size > 0:
                    subset = torch.randperm(len(self.dataset[namespace]))[:self.cfg.dl.analyze_sample_size]
                    sample_inputs = default_collate([self.dataset[namespace][i] for i in subset])
                    # Move to device
                    sample_inputs = move_to_device(sample_inputs, self.device_)

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

    def configure_optimizers(self, groups=None):
        if self.cfg.dl.optimizer == 'adam':
            optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate,
                                         weight_decay=self.cfg.dl.weight_decay)
        elif self.cfg.dl.optimizer == 'lbfgs':
            optimizer = torch.optim.LBFGS(self.parameters(),  # lr=self.learning_rate, #, history_size=1000,
                                          # max_iter=100,
                                          line_search_fn="strong_wolfe")
        elif self.cfg.dl.optimizer == 'sgd':
            optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate, weight_decay=self.cfg.dl.weight_decay,
                                        momentum=self.cfg.dl.momentum)
        else:
            raise ValueError(f'Unknown optimizer {self.cfg.dl.optimizer}')

        if self.cfg.dl.scheduler == 'none':
            return optimizer
        elif self.cfg.dl.scheduler == 'cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                                   self.cfg.dl.epochs)
        elif self.cfg.dl.scheduler == 'cosine_warmup':
            steps = 1  # len(self.train_dataloader())
            warmup_lr_schedule = np.linspace(self.cfg.dl.start_lr, self.cfg.dl.base_lr,
                                             steps * self.cfg.dl.warmup_epochs)
            iters = np.arange(steps * (self.cfg.dl.epochs + 1 - self.cfg.dl.warmup_epochs))
            cosine_lr_schedule = np.array(
                [self.cfg.dl.final_lr + 0.5 * (self.cfg.dl.base_lr - self.cfg.dl.final_lr) * \
                 (1 + math.cos(math.pi * t / (steps * (self.cfg.dl.epochs + 1 - self.cfg.dl.warmup_epochs))))
                 for t in iters])
            lr_schedule = np.concatenate((warmup_lr_schedule, cosine_lr_schedule))
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: lr_schedule[epoch])
            # scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=
            # lambda epoch:
            # (epoch + 1) / self.cfg.dl.warmup_length
            # if epoch < self.cfg.dl.warmup_length
            # else 0.5 * (1. + math.cos(math.pi * (epoch - self.cfg.dl.warmup_length)
            #                           / (self.cfg.dl.epochs - self.cfg.dl.warmup_length))))
            # import pdb; pdb.set_trace()
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
