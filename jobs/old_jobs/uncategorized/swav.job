#!/bin/bash

#SBATCH --job-name=TRAIN
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=3
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --mem=32000M
#SBATCH --partition=gpu
#SBATCH --output=out/slurm_%A.out

source load.sh

## Fixed lr for batch size
#python3 main.py --multirun main.model=pret dl.epochs=100 main.dataset=multicrop_cifar10 dl.optimizer=sgd dl.lr=0.15 dl.momentum=0.9 dl.batch_size=64 dl.weight_decay=0.000001 dl.scheduler=cosine model.pret.init_scores_mean=1.0 model.pret.source=timm model.pret.name=resnet18 model.pret.shell_mode=replace model.pret.module=swav model.pret.backbone=train model.pret.head_type=identity swav.queue_length=0 dl.scheduler=cosine_warmup dl.num_workers=12
## Fixed lr for batch size and added queue

python3 main.py --multirun main.model=pret dl.epochs=100 main.dataset=multicrop_cifar10 dl.optimizer=sgd dl.lr=0.15 dl.momentum=0.9 dl.batch_size=64 dl.weight_decay=0.000001 model.pret.init_scores_mean=1.0 model.pret.shell_mode=replace model.pret.module=swav model.pret.backbone=train model.pret.head_type=identity swav.queue_length=3840 swav.epoch_queue_starts=15 swav.nmb_prototypes=50,100,3000 dl.scheduler=cosine dl.final_lr=0.00015 dl.warmup_epochs=0 dl.num_workers=12 dl.batch_accum=1 model.pret.source=timm model.pret.name=resnet18 swav.min_scale_crops=[0.5,0.3] swav.max_scale_crops=[1.0,0.75] swav.size_crops=[224,18] hydra/launcher=joblib hydra.launcher.n_jobs=3 dl.multiprocessing_context=fork
