import argparse
import copy
import dataclasses
import getpass
import json
import os
import random
import re
import sys
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pathlib
import timm
import torch

import hydra
import torchvision
from neptune.new.types import File
from pytorch_lightning.callbacks import ModelSummary
from pytorch_lightning.utilities import model_summary
from torch import nn
from torchvision import datasets, models
from torchvision.transforms import transforms
from transformers import AutoModel, AutoProcessor

from subnetworks import submasking

# check if not empty
if next(pathlib.Path('moco/').iterdir(), None) is not None:
    print('MOCO found, adding to path')
    sys.path.append('moco/')
    import moco, main_moco, main_lincls
    from moco import builder

from volt.modules import classifier, ssl
from volt.modules.deep_learning import load_model_checkpoint, DeepLearningModule
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
    save_every_n_epochs: int = -1
    save_top_k: int = 2
    run: str = ''
    mvp: bool = True
    main: bool = True
    train: bool = True
    test: bool = True
    validate: bool = True
    dataset: str = 'mnist'
    dataset_subset: str = ''  # 50p -> 50%, 4000 -> 4k labels
    dataset_subset_labels: str = ''
    dataset_subset_clusters_file: str = ''
    dataset_subset_cluster: int = 0
    data_path: str = 'data'
    checkpoint_path: str = 'data/models/checkpoints'
    load_checkpoint: str = ''
    strict_load_checkpoint: bool = True
    resume_checkpoint: str = ''
    save_as: str = ''
    in_mvp: bool = False
    verbose: bool = True
    logging_mode: str = 'async'
    fix_seed: bool = False
    find_lr: bool = False
    probe: bool = False


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
    model_checkpoint: str = ''
    source: str = 'hub'
    repo: str = 'facebookresearch/swav:main'
    name: str = 'resnet50'
    submask_scale: bool = True
    head: str = 'train'
    backbone: str = 'mask'
    train_layers: list = dataclasses.field(default_factory=lambda: ['conv'])
    pretrained_head: bool = False
    pretrained_backbone: bool = True
    head_type: str = 'linear'
    head_init: str = 'kaiming_uniform'
    init_scores_mean: float = 0.01
    init_scores_std: float = 0.0
    init_scores_magnitude: bool = False
    init_scores_shuffle: bool = False
    kaiming_scores_init: bool = False
    prune_criterion: str = 'threshold'
    prune_k: float = 0
    prune_progressive: bool = False
    prune_k_start: float = 0.9
    prune_unit: str = 'epoch'
    module: str = 'classifier'
    shell_mode: str = 'copy'


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

    if cfg.main.fix_seed:
        torch.manual_seed(0)
        random.seed(0)
        np.random.seed(0)

    if mvp:
        cfg = copy.deepcopy(cfg)
        cfg.dl.batch_size = 5
        cfg.dl.epochs = 1
        cfg.main.run = ''

    if mvp:
        # https://docs.neptune.ai/api/neptune/#init_run
        logging_mode = 'debug'
    else:
        logging_mode = cfg.main.logging_mode

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
    if cfg.main.save_every_n_epochs > 0:
        periodic_checkpoint_callback = pl.callbacks.ModelCheckpoint(dirpath=checkpoint_path, save_top_k=-1,
                                                                    save_last=False,
                                                                    every_n_epochs=cfg.main.save_every_n_epochs)
    else:
        periodic_checkpoint_callback = None

    lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval='step', log_momentum=True)
    modelsum_callback = ModelSummary(max_depth=-1)

    callbacks = [checkpoint_callback, lr_monitor, modelsum_callback]
    if periodic_checkpoint_callback is not None:
        callbacks.append(periodic_checkpoint_callback)

    trainer = pl.Trainer(logger=logger, max_epochs=cfg.dl.epochs, accelerator=device,
                         log_every_n_steps=1, check_val_every_n_epoch=1, accumulate_grad_batches=cfg.dl.batch_accum,
                         callbacks=callbacks,
                         num_sanity_val_steps=0)

    train_dataset, val_dataset, test_dataset, train_dataset_unaugmented, val_dataset_unaugmented = None, None, None, None, None
    # TODO: per-dataset configs as well
    if cfg.main.dataset == 'mnist':
        train_dataset, val_dataset, test_dataset = dataset.get_mnist(cfg)
        logger.experiment[f'global/data/train/labels'] = train_dataset.label_names
        logger.experiment[f'global/data/val/labels'] = val_dataset.label_names
    elif cfg.main.dataset == 'cifar10':
        train_dataset, val_dataset, test_dataset = dataset.get_cifar10(cfg)
        logger.experiment[f'global/data/train/labels'] = train_dataset.label_names
        logger.experiment[f'global/data/val/labels'] = val_dataset.label_names
    # elif cfg.main.dataset == 'cifar10pgn':
    #     train_dataset, val_dataset, test_dataset, train_dataset_unaugmented = dataset.get_cifar10pgn(cfg)
    #     logger.experiment[f'global/data/train/labels'] = train_dataset.label_names
    #     logger.experiment[f'global/data/val/labels'] = val_dataset.label_names
    # elif cfg.main.dataset == 'flowers':
    #     train_dataset, val_dataset, test_dataset, train_dataset_unaugmented = dataset.get_flowers(cfg)
    #     logger.experiment[f'global/data/train/labels'] = train_dataset.label_names
    #     logger.experiment[f'global/data/val/labels'] = val_dataset.label_names
    # elif cfg.main.dataset == 'sun397':
    #     train_dataset, val_dataset, test_dataset, train_dataset_unaugmented = dataset.get_sun(cfg)
    #     logger.experiment[f'global/data/train/labels'] = train_dataset.label_names
    #     logger.experiment[f'global/data/val/labels'] = val_dataset.label_names
    # elif cfg.main.dataset == 'cifar100':
    #     train_dataset, val_dataset, test_dataset = dataset.pgn_get_dataset(cfg,
    #                                                                        pgn.datamodules.cifar100_datamodule.CIFAR100DataModule)
    #     logger.experiment[f'global/data/train/labels'] = train_dataset.label_names
    #     logger.experiment[f'global/data/val/labels'] = val_dataset.label_names
    elif cfg.main.dataset.startswith('multicrop_'):
        train_dataset, val_dataset, test_dataset, train_dataset_unaugmented, val_dataset_unaugmented = dataset.get_multicrop_dataset(
            cfg)
    else:
        train_dataset, val_dataset, test_dataset, train_dataset_unaugmented = dataset.get_dataset(cfg)
        logger.experiment[f'global/data/train/labels'] = train_dataset.label_names
        logger.experiment[f'global/data/val/labels'] = val_dataset.label_names

    if mvp:
        train_dataset = train_dataset[:14] if train_dataset is not None else None
        val_dataset = val_dataset[:14] if val_dataset is not None else None
        test_dataset = test_dataset[:14] if test_dataset is not None else None
        train_dataset_unaugmented = train_dataset_unaugmented[:14] if train_dataset_unaugmented is not None else None
        val_dataset_unaugmented = val_dataset_unaugmented[:14] if val_dataset_unaugmented is not None else None
    elif cfg.main.dataset_subset != '':
        # Always use the same subset of data for all runs
        generator = torch.Generator()
        generator.manual_seed(42)

        percent_mode = cfg.main.dataset_subset[-1] == 'p'
        n = int(int(cfg.main.dataset_subset) if not percent_mode else len(train_dataset) * float(
            cfg.main.dataset_subset[:-1]) / 100)
        randperm = torch.randperm(len(train_dataset), generator=generator)[:n]

        subset = torch.utils.data.Subset(train_dataset, randperm)
        # Hack to make the subset behave like a classdataset
        subset.label_names = train_dataset.label_names
        subset.untransform = train_dataset.untransform
        subset.classes = train_dataset.classes
        subset.labels = train_dataset.labels
        subset.inverse_transform = train_dataset.inverse_transform
        assert len(subset) == n
        train_dataset = subset

        subset_unaugmented = torch.utils.data.Subset(train_dataset_unaugmented, randperm)
        # Hack to make the subset behave like a classdataset
        subset_unaugmented.label_names = train_dataset_unaugmented.label_names
        subset_unaugmented.untransform = train_dataset_unaugmented.untransform
        subset_unaugmented.classes = train_dataset_unaugmented.classes
        subset_unaugmented.labels = train_dataset_unaugmented.labels
        subset_unaugmented.inverse_transform = train_dataset_unaugmented.inverse_transform
        assert len(subset_unaugmented) == n
        train_dataset_unaugmented = subset_unaugmented
    elif cfg.main.dataset_subset_labels != '':
        subset_labels = cfg.main.dataset_subset_labels.split(':')
        label_indexes = list(train_dataset.label_names.index(label) for label in subset_labels)

        # The index returned by the dataset always refers to the absolute index in the dataset
        # even after subsetting.
        train_subset_indexes = list(index for index, multi_crops, label in train_dataset if label in label_indexes)
        val_subset_indexes = list(index for index, multi_crops, label in val_dataset if label in label_indexes)

        subset = torch.utils.data.Subset(train_dataset, train_subset_indexes)
        # Hack to make the subset behave like a classdataset
        subset.label_names = train_dataset.label_names
        subset.untransform = train_dataset.untransform
        subset.classes = train_dataset.classes
        subset.labels = train_dataset.labels
        subset.inverse_transform = train_dataset.inverse_transform
        train_dataset = subset

        subset_unaugmented = torch.utils.data.Subset(train_dataset_unaugmented, train_subset_indexes)
        # Hack to make the subset behave like a classdataset
        subset_unaugmented.label_names = train_dataset_unaugmented.label_names
        subset_unaugmented.untransform = train_dataset_unaugmented.untransform
        subset_unaugmented.classes = train_dataset_unaugmented.classes
        subset_unaugmented.labels = train_dataset_unaugmented.labels
        subset_unaugmented.inverse_transform = train_dataset_unaugmented.inverse_transform
        train_dataset_unaugmented = subset_unaugmented

        subset = torch.utils.data.Subset(val_dataset, val_subset_indexes)
        # Hack to make the subset behave like a classdataset
        subset.label_names = val_dataset.label_names
        subset.untransform = val_dataset.untransform
        subset.classes = val_dataset.classes
        subset.labels = val_dataset.labels
        subset.inverse_transform = val_dataset.inverse_transform
        val_dataset = subset

        subset_unaugmented = torch.utils.data.Subset(val_dataset_unaugmented, val_subset_indexes)
        # Hack to make the subset behave like a classdataset
        subset_unaugmented.label_names = val_dataset_unaugmented.label_names
        subset_unaugmented.untransform = val_dataset_unaugmented.untransform
        subset_unaugmented.classes = val_dataset_unaugmented.classes
        subset_unaugmented.labels = val_dataset_unaugmented.labels
        subset_unaugmented.inverse_transform = val_dataset_unaugmented.inverse_transform
        val_dataset_unaugmented = subset_unaugmented
    elif cfg.main.dataset_subset_clusters_file != '':
        cluster_assignments = torch.load(cfg.main.dataset_subset_clusters_file)
        train_subset_indexes = list(index for index in range(len(train_dataset)) if cluster_assignments[index] == cfg.main.dataset_subset_cluster)

        subset = torch.utils.data.Subset(train_dataset, train_subset_indexes)
        # Hack to make the subset behave like a classdataset
        subset.label_names = train_dataset.label_names
        subset.untransform = train_dataset.untransform
        subset.classes = train_dataset.classes
        subset.labels = train_dataset.labels
        subset.inverse_transform = train_dataset.inverse_transform
        train_dataset = subset

        subset_unaugmented = torch.utils.data.Subset(train_dataset_unaugmented, train_subset_indexes)
        # Hack to make the subset behave like a classdataset
        subset_unaugmented.label_names = train_dataset_unaugmented.label_names
        subset_unaugmented.untransform = train_dataset_unaugmented.untransform
        subset_unaugmented.classes = train_dataset_unaugmented.classes
        subset_unaugmented.labels = train_dataset_unaugmented.labels
        subset_unaugmented.inverse_transform = train_dataset_unaugmented.inverse_transform
        train_dataset_unaugmented = subset_unaugmented


    logger.experiment[f'global/data/train/size'] = len(train_dataset) if train_dataset is not None else 0
    logger.experiment[f'global/data/val/size'] = len(val_dataset) if val_dataset is not None else 0
    logger.experiment[f'global/data/test/size'] = len(test_dataset) if test_dataset is not None else 0

    logger.experiment[f'global/data/train/size_unaugmented'] = len(
        train_dataset_unaugmented) if train_dataset_unaugmented is not None else 0
    logger.experiment[f'global/data/val/size_unaugmented'] = len(
        val_dataset_unaugmented) if val_dataset_unaugmented is not None else 0

    task = Task(f"Executing {cfg.main.task}")
    task.start()

    if cfg.main.model == 'clip':
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
                # if cfg.model.pret.module == 'classifier':
                #     model.fc = torch.nn.Linear(model.fc.in_features, len(train_dataset.label_names))
                # assert model.fc.weight.requires_grad
                fc_name = 'fc'
                fc_in_features = model.fc.in_features
            elif cfg.model.pret.repo == 'facebookresearch/dino:main':
                # model.fc = torch.nn.Linear(2048, len(train_dataset.label_names))
                assert not cfg.model.pret.pretrained_head
                fc_name = 'fc'
                fc_in_features = 2048
            else:
                raise ValueError(f"Unknown repo {cfg.model.pret.repo}")
        elif cfg.model.pret.source == 'timm':
            model = timm.create_model(cfg.model.pret.name, pretrained=cfg.model.pret.pretrained_backbone)
            fc_name = 'fc'
            fc_in_features = model.fc.in_features
            # if cfg.model.pret.module == 'classifier':
            #     model = timm.create_model(cfg.model.pret.name, pretrained=True,
            #                               num_classes=len(train_dataset.label_names))
            # else:
            #     model = timm.create_model(cfg.model.pret.name, pretrained=True)
        elif cfg.model.pret.source == 'torch':
            model_class = getattr(torchvision.models, cfg.model.pret.name)
            if cfg.model.pret.name == 'resnet18':
                weights = torchvision.models.ResNet18_Weights.IMAGENET1K_V1
            elif cfg.model.pret.name == 'resnet50':
                weights = torchvision.models.ResNet50_Weights.IMAGENET1K_V2
            elif cfg.model.pret.name == 'wide_resnet50_2':
                weights = torchvision.models.Wide_ResNet50_2_Weights.IMAGENET1K_V2
            else:
                raise ValueError(f"Unknown model {cfg.model.pret.name}")
            model = model_class(weights=weights, progress=True)
            fc_name = 'fc'
            fc_in_features = model.fc.in_features
        elif cfg.model.pret.source == 'clip':
            code = cfg.model.pret.name
            clip = AutoModel.from_pretrained(code).to(device)
            if cfg.model.clip.freeze_backbone:
                for param in clip.parameters():
                    param.requires_grad = False
                for param in clip.visual_projection.parameters():
                    param.requires_grad = True

            embedding_size = clip.visual_projection.out_features

            print('CLIP', code, embedding_size)
            prompts = list(f"This is a photo of a {label}" for label in train_dataset.label_names)
            processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")
            prompt_tokens = processor(text=prompts, return_tensors="pt", padding=True).input_ids
            prompt_embeds = clip.get_text_features(prompt_tokens.to(device)).detach()

            class Contrastive(nn.Module):
                def __init__(self, clip, prompt_embeds):
                    super().__init__()
                    self.prompt_embeds = prompt_embeds
                    self.prompt_embeds.requires_grad = False
                    self.logit_scale = clip.logit_scale.to(device)
                    self.logit_scale.requires_grad = False
                    self.projection = clip.visual_projection.to(device)

                def forward(self, backbone_output):
                    output = self.projection(backbone_output)
                    contrast = torch.nn.functional.cosine_similarity(output[:, None], self.prompt_embeds[None, :],
                                                                     dim=2)
                    contrast *= self.logit_scale.exp()
                    return contrast

            fc = Contrastive(clip, prompt_embeds)
            # fc = torch.nn.Linear(embedding_size, len(train_dataset.label_names), bias=False)
            # fc.weight.data = prompt_embeds
            print('Computed prompt embeddings')

            model = LambdaModule(lambda x, vision_model: vision_model(x, return_dict=False)[1],
                                 vision_model=clip.vision_model)
            fc_name = None
            fc_in_features = clip.visual_projection.in_features
        elif cfg.model.pret.source == 'mocov2':
            model = models.__dict__['resnet50']()
            model.fc.weight.data.normal_(mean=0.0, std=0.01)
            model.fc.bias.data.zero_()

            checkpoint = torch.load("data/models/" + cfg.model.pret.name + ".pth.tar"
                                    , map_location='cpu')
            state_dict = checkpoint['state_dict']
            for k in list(state_dict.keys()):
                # retain only encoder_q up to before the embedding layer
                if k.startswith("module.encoder_q") and not k.startswith(
                    "module.encoder_q.fc"
                ):
                    # remove prefix
                    state_dict[k[len("module.encoder_q.") :]] = state_dict[k]
                # delete renamed or unused k
                del state_dict[k]

            msg = model.load_state_dict(state_dict, strict=False)
            assert set(msg.missing_keys) == {"fc.weight", "fc.bias"}

            fc_name = 'fc'
            fc_in_features = model.fc.in_features
        else:
            raise ValueError(f"Unknown source {cfg.model.pret.source}")

        # if not cfg.model.pret.pretrained_head:
        #     if cfg.model.pret.head_type == 'linear':
        #         assert cfg.model.pret.head_init == 'kaiming_uniform'
        #         head = torch.nn.Linear(fc_in_features, len(train_dataset.label_names))
        #     elif cfg.model.pret.head_type == 'dual':
        #         # sum of 2 heads, one with all ones and other with all -1, is full-featured when trained with masking
        #         raise NotImplementedError()
        #     elif cfg.model.pret.head_type == 'identity':
        #         head = torch.nn.Identity()
        #     else:
        #         raise ValueError(f"Unknown head type {cfg.model.pret.head_type}")
        #
        #     model.__setattr__(fc_name, head)
        assert not cfg.model.pret.pretrained_head
        assert cfg.model.pret.head_type == 'identity'
        head = torch.nn.Identity()
        if fc_name is not None:
            fc = model.__getattr__(fc_name)
            model.__setattr__(fc_name, head)

        if re.match(r"(dino_)?resnet\d+", cfg.model.pret.name) or cfg.model.pret.source == 'mocov2':
            def par_sel(name, param):
                if not param.requires_grad:
                    return 'freeze'
                for l in cfg.model.pret.train_layers:
                    if l in name:
                        assert not any(
                            l2 in name for l2 in cfg.model.pret.train_layers if l != l2), 'Ambiguity in layer names'
                        return cfg.model.pret.backbone, name
                if 'fc' in name:
                    return cfg.model.pret.head, name
                return 'freeze'
        elif cfg.model.pret.source == 'clip':
            def par_sel(name, param):
                if not param.requires_grad:
                    return 'freeze'

                r = None
                for l in cfg.model.pret.train_layers:
                    if l in name:
                        assert not any(
                            l2 in name for l2 in cfg.model.pret.train_layers if l != l2), 'Ambiguity in layer names'
                        r = [cfg.model.pret.backbone, name]
                        break

                if r is None:
                    return 'freeze'

                if 'weight' in name:
                    r[1] += '.weight'
                elif 'bias' in name:
                    r[1] += '.bias'

                return r
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
                    ctx['epoch'] / cfg.dl.epochs * (cfg.model.pret.prune_k_start - cfg.model.pret.prune_k))
        elif cfg.model.pret.prune_progressive and cfg.model.pret.prune_criterion == 'threshold':
            k = lambda ctx: ctx.epoch / cfg.dl.epochs * cfg.model.pret.prune_k
        elif not cfg.model.pret.prune_progressive:
            k = lambda epoch: cfg.model.pret.prune_k
        else:
            raise NotImplementedError()

        if not cfg.model.pret.init_scores_magnitude:
            if cfg.model.pret.kaiming_scores_init:
                scores_init = submasking.default_scores_init
            else:
                scores_init = submasking.normal_scores_init(cfg.model.pret.init_scores_mean,
                                                        cfg.model.pret.init_scores_std)
        else:
            scores_init = submasking.magnitude_scores_init(cfg.model.pret.init_scores_mean,
                                                           cfg.model.pret.init_scores_std,
                                                           cfg.model.pret.init_scores_shuffle)

        model.to(device)
        test_input = torch.randn(2, 3, 224, 224).to(device)
        model = submasking.SubmaskedModel(model, scale=cfg.model.pret.submask_scale, test_input=test_input,
                                          parameter_selection=par_sel, k=k,
                                          prune_criterion=cfg.model.pret.prune_criterion,
                                          scores_init=scores_init,
                                          shell_mode=cfg.model.pret.shell_mode).to(device)

        # if cfg.model.pret.model_checkpoint != '':
        #     chkpt_path = Path(cfg.main.checkpoint_path, cfg.model.pret.model_checkpoint)
        #     state_dict = torch.load(chkpt_path)
        #
        #
        #
        #     import pdb; pdb.set_trace()

        if cfg.model.pret.module == 'classifier':
            module = classifier.ClassifierModule(fc_in_features, fc, cfg, model, device, logger, train_dataset,
                                                 val_dataset,
                                                 test_dataset, train_dataset_unaugmented)
        elif cfg.model.pret.module == 'swav':
            module = ssl.SWAVModule(fc_in_features, fc, cfg, model, device, logger, train_dataset, val_dataset,
                                    test_dataset, train_dataset_unaugmented, val_dataset_unaugmented)
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
        module = classifier.ClassifierModule(0, cfg, model, device, logger, train_dataset, val_dataset, test_dataset)
    else:
        raise Exception(f"Unknown model: {cfg.main.model}")

    module.to(device)

    logger.experiment[f'global/summary'] = model_summary.ModelSummary(module, max_depth=-1).__str__()

    weight_summary = ""
    for name, param in module.named_parameters():
        row = f"{name}: {param.shape}, {param.numel()} elements, requires_grad={param.requires_grad}\n"
        weight_summary += row
        logger.experiment[f'global/weight_summary'].log(row)
    if cfg.main.verbose:
        print(weight_summary)

    if cfg.main.find_lr and not mvp:
        # Run learning rate finder
        lr_finder = trainer.tuner.lr_find(module)

        # # Results can be found in
        # results = lr_finder.results

        # Plot with
        fig = lr_finder.plot(suggest=True)
        logger.experiment[f'global/lr_finder'].log(fig)
        # fig.show()

        # Pick point based on plot, or get suggestion
        new_lr = lr_finder.suggestion()

        # import pdb; pdb.set_trace()

        # update hparams of themodule
        module.lr = new_lr
        module.learning_rate = new_lr

        # import pdb; pdb.set_trace()

        logger.experiment["global/lr_finder_interactive"].upload(File.as_html(fig))

        # trainer.tune(module)

    if cfg.main.load_checkpoint != '':
        load_model_checkpoint(cfg.main.checkpoint_path + '/' + cfg.main.load_checkpoint, module, strict=cfg.main.strict_load_checkpoint)
    elif cfg.main.run != '':
        last_ckpt_path = checkpoint_path + '/' + 'last.ckpt'
        load_model_checkpoint(last_ckpt_path, module, strict=cfg.main.strict_load_checkpoint)

    if cfg.main.probe:
        module.probe()

    if cfg.main.validate:
        trainer.validate(module)


    if cfg.main.train:
        ckpt_path = cfg.main.checkpoint_path + '/' + cfg.main.resume_checkpoint if cfg.main.resume_checkpoint != '' else None
        # if ckpt_path is not None:
        if not mvp:
            trainer.fit(module, ckpt_path=ckpt_path)
        else:
            trainer.fit(module)

        best_model_path = checkpoint_callback.best_model_path
        logger.experiment[f'training/model/best_model_path'] = best_model_path

    if cfg.main.test:
        # if not cfg.main.train:
        #     best_model_path = logger.experiment['training/model/best_model_path'].fetch()
        # if cfg.main.train:
        #     load_model_checkpoint(best_model_path, module)
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
        cfg.main.in_mvp = True
        train(cfg, mvp=True)
        cfg.main.in_mvp = False
        task.done()

    if cfg.main.main:
        # Print big message indicating that the training will begin
        print(
            """
    ____________________________________________________________________
    |                                                                  |
    |                          BEGIN TRAINING                          |
    |                                                                  |
    --------------------------------------------------------------------
            """)
        task = Task(f"Running main").start()
        train(cfg)
        task.done()


if __name__ == "__main__":
    XTrace(main, whitelist=['prunetuning']).__call__()
