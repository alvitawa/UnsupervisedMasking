import copy
import getpass
import json
import os
import re
import sys
from collections import OrderedDict
from dataclasses import dataclass

import timm
import torch

import hydra
import torchvision
from pytorch_lightning.callbacks import ModelSummary
from pytorch_lightning.utilities import model_summary
from torch import nn
from torchvision import datasets
from torchvision.transforms import transforms
from transformers import AutoModel, AutoProcessor

import subnetworks.simple_mnist_example
from subnetworks import submasking
from volt.modules import classifier, ssl
from volt.modules.deep_learning import load_model_checkpoint
from volt.log import Task

from pytorch_lightning.loggers import NeptuneLogger
import pytorch_lightning as pl

import neptune.new as neptune

from omegaconf import DictConfig, OmegaConf

from volt.util import ddict, LambdaModule
from volt.xtrace import XTrace
import gc

from volt import config, util, dataset, log

import pgn.datamodules


@dataclass
class MainConfig:
    task: str = 'train'
    model: str = 'clip'
    force_cpu: bool = False
    outputs: str = 'out'
    info: str = ''
    monitor: str = 'val/loss'
    monitor_mode: str = 'min'
    save_top_k: int = 4
    run: str = ''
    mvp: bool = True
    main: bool = True
    train: bool = True
    test: bool = True
    dataset: str = 'mnist'
    data_path: str = 'data'
    checkpoint_path: str = 'data/models/checkpoints'
    save_as: str = ''


@dataclass
class ModelClipConfig:
    code: str = "openai/clip-vit-base-patch32"
    submask: bool = False
    submask_scale: bool = False
    submask_backbone: bool = True
    submask_head: bool = True
    freeze_backbone: bool = False
    scores_init: str = 'default_scores_init'
    module: str = 'prompt'


@dataclass
class ModelReproConfig:
    name: str = 'NetBaseline'
    submask: bool = False
    scale: bool = False


@dataclass
class ModelTimmConfig:
    name: str = 'resnet18'
    submask_scale: bool = True
    submask_head: bool = False
    submask_backbone: bool = True
    freeze_backbone: bool = False
    pretrained: bool = True
    init_scores_mean: float = 0.01
    init_scores_std: float = 0.0
    prune_criterion: str = 'threshold'
    prune_k: float = 0
    prune_progressive: bool = False
    prune_unit: str = 'epoch'


@dataclass
class ModelPretrainedConfig:
    source: str = 'hub'
    repo: str = 'facebookresearch/swav:main'
    name: str = 'resnet50'
    submask_scale: bool = True
    head: str = 'train'
    backbone: str = 'mask'
    pretrained: bool = True
    init_scores_mean: float = 0.01
    init_scores_std: float = 0.0
    prune_criterion: str = 'threshold'
    prune_k: float = 0
    prune_progressive: bool = False
    prune_k_start: float = 0.9
    prune_unit: str = 'epoch'
    module: str = 'classifier'


config.register('main', MainConfig)
config.register('clip', ModelClipConfig, 'model')
config.register('repro', ModelReproConfig, 'model')
config.register('timm', ModelTimmConfig, 'model')
config.register('pret', ModelPretrainedConfig, 'model')


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
    run = neptune.init_run(tags=[], with_id=cfg.main.run if cfg.main.run != '' else None, mode=logging_mode,
                           source_files=['main.py', 'volt/**/*.py', 'subnetworks/**/*.py', 'conf/**/*.yaml'])
    logger = NeptuneLogger(run=run, log_model_checkpoints=False)
    run_id = logger.experiment.get_url().split('/')[-1]

    log_config(OmegaConf.to_container(cfg), logger)

    logger.experiment[f'global/info'] = cfg.main.info
    logger.experiment[f'global/command'] = ' '.join(sys.argv)
    logger.experiment[f'global/username'] = getpass.getuser()

    # Currently if training was cancelled, and you restart it, the best_checkpoint_path is not set.
    checkpoint_path = f'{cfg.main.checkpoint_path}/{run_id}'
    checkpoint_callback = pl.callbacks.ModelCheckpoint(dirpath=checkpoint_path, monitor=cfg.main.monitor,
                                                       mode=cfg.main.monitor_mode, save_top_k=cfg.main.save_top_k,
                                                       save_last=True)
    lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval='step', log_momentum=True)
    modelsum_callback = ModelSummary(max_depth=-1)
    trainer = pl.Trainer(logger=logger, max_epochs=cfg.dl.epochs, accelerator=device,
                         log_every_n_steps=1, check_val_every_n_epoch=1, accumulate_grad_batches=cfg.dl.batch_accum,
                         callbacks=[checkpoint_callback, lr_monitor, modelsum_callback], num_sanity_val_steps=0)

    # TODO: per-dataset configs as well
    if cfg.main.dataset == 'mnist':
        train_dataset, val_dataset, test_dataset = dataset.get_mnist(cfg)
        logger.experiment[f'global/data/train/labels'] = train_dataset.label_names
        logger.experiment[f'global/data/val/labels'] = val_dataset.label_names
    elif cfg.main.dataset == 'cifar10':
        train_dataset, val_dataset, test_dataset = dataset.get_cifar10(cfg)
        logger.experiment[f'global/data/train/labels'] = train_dataset.label_names
        logger.experiment[f'global/data/val/labels'] = val_dataset.label_names
    elif cfg.main.dataset == 'cifar10pgn':
        train_dataset, val_dataset, test_dataset = dataset.get_cifar10pgn(cfg)
        logger.experiment[f'global/data/train/labels'] = train_dataset.label_names
        logger.experiment[f'global/data/val/labels'] = val_dataset.label_names
    elif cfg.main.dataset == 'flowers':
        train_dataset, val_dataset, test_dataset = dataset.get_flowers(cfg)
        logger.experiment[f'global/data/train/labels'] = train_dataset.label_names
        logger.experiment[f'global/data/val/labels'] = val_dataset.label_names
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
        mode = cfg.models.supermask_convnet.mode if 'mode' in cfg.models.supermask_convnet else 'topk'
        subnetworks.simple_mnist_example.args = ddict(sparsity=cfg.models.supermask_convnet.sparsity,
                                                      mode=mode)
        model = subnetworks.simple_mnist_example.Net()
        module = classifier.ClassifierModule(cfg, model, device, logger, train_dataset, val_dataset, test_dataset)
    elif cfg.main.model == 'convnet':
        model = subnetworks.simple_mnist_example.NetBaseline()
        module = classifier.ClassifierModule(cfg, model, device, logger, train_dataset, val_dataset, test_dataset)
    elif cfg.main.model == 'supermask_convnet_3c':
        mode = cfg.models.supermask_convnet.mode if 'mode' in cfg.models.supermask_convnet else 'topk'
        subnetworks.simple_mnist_example.args = ddict(sparsity=cfg.models.supermask_convnet.sparsity,
                                                      mode=mode)
        model = subnetworks.simple_mnist_example.Net3Channel()
        module = classifier.ClassifierModule(cfg, model, device, logger, train_dataset, val_dataset, test_dataset)
    elif cfg.main.model == 'convnet_3c':
        model = subnetworks.simple_mnist_example.Net3ChannelBaseline()
        module = classifier.ClassifierModule(cfg, model, device, logger, train_dataset, val_dataset, test_dataset)
    elif cfg.main.model == 'Resnet18_supermask':
        mode = cfg.models.supermask_convnet.mode if 'mode' in cfg.models.supermask_convnet else 'topk'
        subnetworks.simple_mnist_example.args = ddict(sparsity=cfg.models.supermask_convnet.sparsity,
                                                      mode=mode)
        model = subnetworks.simple_mnist_example.ResNet18(prune_rate=cfg.models.supermask_convnet.sparsity,
                                                          freeze_weights=True)
        module = classifier.ClassifierModule(cfg, model, device, logger, train_dataset, val_dataset, test_dataset)
    elif cfg.main.model == 'Resnet18':
        model = subnetworks.simple_mnist_example.ResNet18(prune_rate=1, freeze_weights=False)
        module = classifier.ClassifierModule(cfg, model, device, logger, train_dataset, val_dataset, test_dataset)
    elif cfg.main.model == 'clip':
        code = cfg.model.clip.code
        clip = AutoModel.from_pretrained(code).to(device)
        if cfg.model.clip.freeze_backbone:
            for param in clip.parameters():
                param.requires_grad = False
            for param in clip.visual_projection.parameters():
                param.requires_grad = True
        model = torch.nn.Sequential(OrderedDict([
            ('backbone', LambdaModule(lambda x, vision_model: vision_model(x, return_dict=False)[1],
                                      vision_model=clip.vision_model)),
            ('projection', clip.visual_projection)]))
        embedding_size = clip.visual_projection.out_features
        if cfg.model.clip.submask:
            test_input = torch.randn(2, 3, 224, 224).to(device)

            def par_sel(name, param):
                if not param.requires_grad:
                    return 'freeze'

                r = None

                if cfg.model.clip.submask_backbone:
                    if 'k_proj' in name:
                        r = ['mask', 'k_proj']
                    elif 'v_proj' in name:
                        r = ['mask', 'v_proj']
                    elif 'q_proj' in name:
                        r = ['mask', 'q_proj']
                    elif 'out_proj' in name:
                        r = ['mask', 'out_proj']
                    elif 'mlp.fc1' in name:
                        r = ['mask', 'mlp.fc1']
                    elif 'mlp.fc2' in name:
                        r = ['mask', 'mlp.fc2']

                if cfg.model.clip.submask_head:
                    if 'projection' in name:
                        r = ['mask', 'projection']

                if r is None:
                    return 'freeze'

                if 'weight' in name:
                    r[1] += '.weight'
                elif 'bias' in name:
                    r[1] += '.bias'

                return r

            model = submasking.SubmaskedModel(model, test_input=test_input, parameter_selection=par_sel,
                                              scale=cfg.model.clip.submask_scale,
                                              scores_init=eval(f"submasking.{cfg.model.clip.scores_init}"))
        logit_scale = clip.logit_scale
        if cfg.model.clip.module == 'prompt':
            prompts = list(f"This is a photo of a {label}" for label in train_dataset.label_names)
            processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")
            prompt_tokens = processor(text=prompts, return_tensors="pt", padding=True).input_ids
            prompt_embeds = clip.get_text_features(prompt_tokens.to(device)).detach()
            module = classifier.PromptClassifierModule(prompt_embeds, logit_scale, cfg, model, device, logger,
                                                       train_dataset, val_dataset, test_dataset)
        else:
            module = classifier.ContrastiveClassifierModule(embedding_size, logit_scale, cfg, model, device, logger,
                                                            train_dataset, val_dataset, test_dataset)
    elif cfg.main.model == 'repro':
        assert cfg.model.repro.name != 'Resnet18'
        assert cfg.main.dataset == 'cifar10' or cfg.main.dataset == 'mnist'
        model = eval(f"subnetworks.simple_mnist_example.{cfg.model.repro.name}()").to(device)
        for name, param in model.named_parameters():
            # nn.init.kaiming_uniform_(self.scores, a=math.sqrt(5))
            nn.init.kaiming_normal_(param, mode="fan_in", nonlinearity="relu")
            print(name, param.shape)
        print('DEVICE', device)
        if cfg.model.repro.submask:
            test_input = torch.randn(2, 3, 32, 32).to(device)
            if cfg.model.repro.name == 'Net3ChannelBaseline':
                parameter_selection = lambda name, param: (
                    'mask' if param.requires_grad else 'freeze', 'conv' if 'conv' in name else 'fc')
            elif cfg.model.repro.name == 'ResNet18Baseline':
                parameter_selection = lambda name, param: (
                    'mask' if param.requires_grad else 'freeze', 'conv' if 'conv' in name else 'shortcut/fc')
            else:
                raise ValueError(f"Unknown model {cfg.model.repro.name}")
            model = submasking.SubmaskedModel(model, scale=cfg.model.repro.scale, test_input=test_input,
                                              parameter_selection=parameter_selection).to(device)
        module = classifier.ClassifierModule(cfg, model, device, logger, train_dataset, val_dataset, test_dataset)

    elif cfg.main.model == 'pret':
        if cfg.model.pret.source == 'hub':
            model = torch.hub.load(cfg.model.pret.repo, cfg.model.pret.name)
            if cfg.model.pret.repo == 'facebookresearch/swav:main':
                if cfg.model.pret.module == 'classifier':
                    model.fc = torch.nn.Linear(model.fc.in_features, len(train_dataset.label_names))
                assert model.fc.weight.requires_grad
            elif cfg.model.pret.repo == 'facebookresearch/dino:main':
                model.fc = torch.nn.Linear(2048, len(train_dataset.label_names))
            else:
                raise ValueError(f"Unknown repo {cfg.model.pret.repo}")
        elif cfg.model.pret.source == 'timm':
            raise NotImplementedError()
        else:
            raise ValueError(f"Unknown source {cfg.model.pret.source}")

        # check with regex if it is one of the normal resnet models
        if re.match(r"(dino_)?resnet\d+", cfg.model.pret.name):
            def par_sel(name, param):
                if not param.requires_grad:
                    return 'freeze'
                if 'conv' in name:
                    return cfg.model.pret.backbone, 'conv'
                if 'fc' in name:
                    return cfg.model.pret.head, name
                return 'freeze'
        else:
            weight_summary = ""
            for name, param in model.named_parameters():
                row = f"{name}: {param.shape}, {param.numel()} elements, requires_grad={param.requires_grad}\n"
                weight_summary += row
                logger.experiment[f'global/weight_summary'].log(row)
            print(weight_summary)
            raise ValueError(f"Unknown model {cfg.model.pret.name}")

        assert cfg.model.pret.prune_unit == 'epoch'

        if cfg.model.pret.prune_progressive and cfg.model.pret.prune_criterion == 'topk':
            k = lambda ctx: cfg.model.pret.prune_k_start - (
                        ctx.epoch / cfg.dl.epochs * (cfg.model.pret.prune_k_start - cfg.model.pret.prune_k))
        elif cfg.model.pret.prune_progressive and cfg.model.pret.prune_criterion == 'threshold':
            k = lambda ctx: ctx.epoch / cfg.dl.epochs * cfg.model.pret.prune_k
        elif not cfg.model.pret.prune_progressive:
            k = lambda epoch: cfg.model.pret.prune_k
        else:
            raise NotImplementedError()

        model.to(device)
        test_input = torch.randn(2, 3, 224, 224).to(device)
        model = submasking.SubmaskedModel(model, scale=cfg.model.pret.submask_scale, test_input=test_input,
                                          parameter_selection=par_sel, k=k,
                                          prune_criterion=cfg.model.pret.prune_criterion,
                                          scores_init=submasking.normal_scores_init(cfg.model.pret.init_scores_mean,
                                                                                    cfg.model.pret.init_scores_std)).to(
            device)
        if cfg.model.pret.module == 'classifier':
            module = classifier.ClassifierModule(cfg, model, device, logger, train_dataset, val_dataset, test_dataset)
        elif cfg.model.pret.module == 'swav':
            module = ssl.SWAVModule(cfg, model, device, logger, train_dataset, val_dataset, test_dataset)
        else:
            raise ValueError(f"Unknown module {cfg.model.pret.module}")
    elif cfg.main.model == 'timm':
        model = timm.create_model(cfg.model.timm.name, pretrained=cfg.model.timm.pretrained,
                                  num_classes=len(train_dataset.label_names)).to(device)
        if cfg.model.timm.name in ['resnet18', 'resnet26', 'resnet34', 'resnet50']:
            def par_sel(name, param):
                if not param.requires_grad:
                    return 'freeze'
                if name == 'fc.weight':
                    return 'mask' if cfg.model.timm.submask_head else 'train', 'fc.weight'
                elif name == 'fc.bias':
                    return 'mask' if cfg.model.timm.submask_head else 'train', 'fc.bias'

                elif cfg.model.timm.freeze_backbone:
                    return 'freeze'
                elif not cfg.model.timm.submask_backbone:
                    return 'train'

                elif 'conv1' in name:
                    return 'mask', 'conv1'
                elif 'conv2' in name:
                    return 'mask', 'conv2'
                elif 'conv3' in name:
                    return 'mask', 'conv3'
                elif 'downsample.0' in name:
                    return 'mask', 'downsample.0' + name.split('downsample.0')[-1]
                elif 'downsample.1' in name:
                    return 'mask', 'downsample.1' + name.split('downsample.1')[-1]
                else:
                    return 'freeze'
        elif cfg.model.timm.name in ['skresnet18', 'ssl_resnet18']:
            def par_sel(name, param):
                if not param.requires_grad:
                    return 'freeze'
                if '.bn' in name:
                    return 'freeze'

                if name == 'fc.weight':
                    return 'mask' if cfg.model.timm.submask_head else 'train', 'fc.weight'
                if name == 'fc.bias':
                    return 'mask' if cfg.model.timm.submask_head else 'train', 'fc.bias'

                if cfg.model.timm.freeze_backbone:
                    return 'freeze'
                if not cfg.model.timm.submask_backbone:
                    return 'train'

                expression = r"^layer\d.\d.(.*)"
                match = re.match(expression, name)
                if match:
                    return 'mask', match.group(1)

                if name == 'conv1.weight':
                    return 'mask', 'conv1.weight'

                return 'freeze'
        else:
            weight_summary = ""
            for name, param in model.named_parameters():
                row = f"{name}: {param.shape}, {param.numel()} elements, requires_grad={param.requires_grad}\n"
                weight_summary += row
                logger.experiment[f'global/weight_summary'].log(row)
            print(weight_summary)
            raise ValueError(f"Unknown model {cfg.model.timm.name}")

        assert cfg.model.timm.prune_unit == 'epoch'

        if cfg.model.timm.prune_progressive and cfg.model.timm.prune_criterion == 'topk':
            k = lambda ctx: 0.9 - (ctx.epoch / cfg.dl.epochs * (1 - cfg.model.timm.prune_k))
        elif cfg.model.timm.prune_progressive and cfg.model.timm.prune_criterion == 'threshold':
            k = lambda ctx: ctx.epoch / cfg.dl.epochs * cfg.model.timm.prune_k
        elif not cfg.model.timm.prune_progressive:
            k = lambda epoch: cfg.model.timm.prune_k
        else:
            raise NotImplementedError()

        test_input = torch.randn(2, 3, 224, 224).to(device)
        model = submasking.SubmaskedModel(model, scale=cfg.model.timm.submask_scale, test_input=test_input,
                                          parameter_selection=par_sel, k=k,
                                          prune_criterion=cfg.model.timm.prune_criterion,
                                          scores_init=submasking.normal_scores_init(cfg.model.timm.init_scores_mean,
                                                                                    cfg.model.timm.init_scores_std)).to(
            device)
        module = classifier.ClassifierModule(cfg, model, device, logger, train_dataset, val_dataset, test_dataset)
    else:
        raise Exception(f"Unknown model: {cfg.main.model}")

    module.to(device)

    logger.experiment[f'global/summary'] = model_summary.ModelSummary(module, max_depth=-1).__str__()

    weight_summary = ""
    for name, param in module.named_parameters():
        row = f"{name}: {param.shape}, {param.numel()} elements, requires_grad={param.requires_grad}\n"
        weight_summary += row
        logger.experiment[f'global/weight_summary'].log(row)
    print(weight_summary)

    if cfg.main.train:
        if cfg.main.run != '':
            last_ckpt_path = checkpoint_path + '/' + 'last.ckpt'
            load_model_checkpoint(last_ckpt_path, module)
        else:
            trainer.validate(module)
            pass
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

    if cfg.main.mvp:
        task = Task(f"Running mvp").start()
        train(cfg, mvp=True)
        task.done()

    # Print big message indicating that the training will begin
    print(
        """
        ____________________________________________________________________
        |                                                                  |
        |                          BEGIN TRAINING                          |
        |                                                                  |
        --------------------------------------------------------------------
        """)

    if cfg.main.main:
        task = Task(f"Running main").start()
        train(cfg)
        task.done()


if __name__ == "__main__":
    XTrace(main, whitelist=['prunetuning']).__call__()
