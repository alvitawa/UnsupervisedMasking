#!/bin/bash

#SBATCH --job-name=TRAIN
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=3
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --mem=32000M
#SBATCH --partition=gpu_titanrtx
#SBATCH --output=out/slurm_%A.out

source load.sh

srun python3 main.py --multirun main.model=timm dl.epochs=100 main.dataset=cifar10pgn dl.optimizer=sgd dl.lr=0.1 dl.momentum=0.9 dl.batch_size=128 dl.weight_decay=0.0005 dl.scheduler=cosine model.timm.init_scores_mean=1.0 model.timm.submask_head=1 model.timm.name=resnet18,resnet26,resnet34,resnet50