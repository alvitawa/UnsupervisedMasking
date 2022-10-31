import copy
import getpass
import json
import os
import sys
from dataclasses import dataclass

import torch

import hydra
from torchvision import datasets
from torchvision.transforms import transforms

import subnetworks.simple_mnist_example
from volt.modules import classifier
from volt.modules.deep_learning import load_model_checkpoint
from volt.log import Task

from pytorch_lightning.loggers import NeptuneLogger
import pytorch_lightning as pl

import neptune.new as neptune

from omegaconf import DictConfig, OmegaConf

from volt.util import ddict
from volt.xtrace import XTrace
import gc

from volt import config


@dataclass
class MainConfig:
    task: str = 'train'
    model: str = 'clip'
    weights: str = 'ViT-B/32'
    force_cpu: bool = False
    outputs: str = 'out'
    info: str = ''
    monitor: str = 'val/loss'
    monitor_mode: str = 'min'
    save_top_k: int = 4
    run: str = ''
    mvp: bool = True
    train: bool = True
    test: bool = True
    dataset: str = 'mnist'
    data_path: str = '~/data'
    save_as: str = ''


config.register('main', MainConfig)


def summary(cfg):
    pass

def log_config(cfg, logger, base='global/params/'):
    for k, v in cfg.items():
        if isinstance(v, dict):
            log_config(v, logger, base=f'{base}/{k}/')
        else:
            logger.experiment[f'{base}/{k}'] = str(v)


def train(cfg, mvp=False):
    device = "cuda" if torch.cuda.is_available() and not cfg.main.force_cpu else "cpu"

    if mvp:
        cfg = copy.deepcopy(cfg)
        cfg.dl.batch_size = 5
        cfg.dl.epochs = 1

    if mvp:
        # https://docs.neptune.ai/api/neptune/#init_run
        logging_mode = 'debug'
    else:
        logging_mode = 'async'

    # Api key and proj name in env. NEPTUNE_API_TOKEN and NEPTUNE_PROJECT
    run = neptune.init_run(tags=[], with_id=cfg.main.run if cfg.main.run != '' else None, mode=logging_mode)
    logger = NeptuneLogger(run=run, log_model_checkpoints=False)
    run_id = logger.experiment.get_url().split('/')[-1]

    log_config(OmegaConf.to_container(cfg), logger)

    logger.experiment[f'global/info'] = cfg.main.info
    logger.experiment[f'global/command'] = ' '.join(sys.argv)
    logger.experiment[f'global/username'] = getpass.getuser()

    # Currently if training was cancelled, and you restart it, the best_checkpoint_path is not set.
    checkpoint_path = f'data/models/checkpoints/{run_id}'
    checkpoint_callback = pl.callbacks.ModelCheckpoint(dirpath=checkpoint_path, monitor=cfg.main.monitor,
                                                       mode=cfg.main.monitor_mode, save_top_k=cfg.main.save_top_k,
                                                       save_last=True)
    trainer = pl.Trainer(logger=logger, max_epochs=cfg.dl.epochs, accelerator=device,
                         log_every_n_steps=1, check_val_every_n_epoch=1, accumulate_grad_batches=cfg.dl.batch_accum,
                         callbacks=[checkpoint_callback], num_sanity_val_steps=0)  # , callbacks=[checkpoint_callback])

    if cfg.main.dataset == 'mnist':
        # TODO: per-dataset configs as well
        train_dataset = datasets.MNIST(os.path.join(cfg.main.data_path, 'mnist'), train=True, download=True)
        val_dataset = datasets.MNIST(os.path.join(cfg.main.data_path, 'mnist'), train=False)
        test_dataset = None

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
    else:
        raise Exception(f"Unknown dataset: {cfg.main.dataset}")

    if mvp:
        train_dataset = train_dataset[:14] if train_dataset is not None else None
        val_dataset = val_dataset[:14] if val_dataset is not None else None
        test_dataset = test_dataset[:14] if test_dataset is not None else None

    logger.experiment[f'global/data/train/size'] = len(train_dataset) if train_dataset is not None else 0
    logger.experiment[f'global/data/val/size'] = len(val_dataset) if val_dataset is not None else 0
    logger.experiment[f'global/data/test/size'] = len(test_dataset) if test_dataset is not None else 0

    task = Task(f"Executing {cfg.main.task}")
    task.start()

    if cfg.main.model == 'supermask_convnet':
        # TODO: per-model configs
        subnetworks.simple_mnist_example.args = ddict(sparsity=cfg.models.supermask_convnet.sparsity)
        model = subnetworks.simple_mnist_example.Net()
        module = classifier.ClassifierModule(cfg, model, device, logger, train_dataset, val_dataset, test_dataset)
    else:
        raise Exception(f"Unknown model: {cfg.main.model}")

    module.to(device)

    if cfg.main.train:
        if cfg.main.run != '':
            last_ckpt_path = checkpoint_path + '/' + 'last.ckpt'
            load_model_checkpoint(last_ckpt_path, module)
        else:
            trainer.validate(module)
        trainer.fit(module)

        best_model_path = checkpoint_callback.best_model_path
    else:
        best_model_path = logger.experiment['training/model/best_model_path'].fetch()

    logger.experiment[f'training/model/best_model_path'] = best_model_path
    load_model_checkpoint(best_model_path, module)

    trainer.test(module)

    torch.cuda.empty_cache()
    gc.collect()

    task.done()
    return cfg, module


@hydra.main(version_base=None, config_path="conf", config_name="default")
def main(cfg: DictConfig):
    if cfg.main.task == 'summary':
        return summary(cfg)

    task = Task(f"Running mvp").start()
    if cfg.main.mvp:
        train(cfg, mvp=True)
    task.done()

    task = Task(f"Running main").start()
    result = train(cfg)
    task.done()

    return result


if __name__ == "__main__":
    cfg_, result_ = XTrace(main, whitelist=['prunetuning']).__call__()
