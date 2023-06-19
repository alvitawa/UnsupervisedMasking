# Self-Masking Networks for Unsupervised Adaptation

This code provides a PyTorch implementation for self-supervised fine-tuning through self-masking, as described in the paper Self-Masking Networks for Unsupervised Adaptation.

![image](https://github.com/alvitawa/UnsupervisedMasking/assets/10909323/8196e960-da4e-45ec-8221-1c5caa71dbef)

With the advent of billion-parameter foundation models, efficient fine-tuning has become
increasingly important for the adaptation of models to downstream tasks. However, especially
in computer vision, it can be hard to achieve good performance when access to quality labeled
data is lacking. In this work, we propose a method adapting pretrained generalist models in
a self-supervised manner by learning binary masks. These self-supervised masking networks
(SMNs) are up to 32x more efficient to store and significantly improve performance on label-
efficient downstream tasks. We validate the usefulness of learning binary masks as a fine-tuning
method on 8 datasets and 3 model architectures, and we demonstrate the effectiveness of SMNs
in 3 label-efficient settings.

# Usage

If you want to use masking in your own training setting, you can download and install the `subnetworks` package in this repository and install it with `pip install -e subnetworks`.

Below is an example on how to create a masked version of a ResNet50 model, which you can then train as you would any other model. 

```python
import timm
from subnetworks import submasking

# Grab pretrained model
model = timm.create_model('resnet50', pretrained=True)

# Print a summary of all the weights, this is usefull to know how to set up the parameter selection function below
weight_summary = ""
for name, param in model.named_parameters():
    row = f"{name}: {param.shape}, {param.numel()} elements, requires_grad={param.requires_grad}\n"
    weight_summary += row
    
# Select which parameters to train, mask or freeze based on the name of the parameter.
def par_sel(name, param):
    if not param.requires_grad:
        return 'freeze'
    for l in ['conv','downsample']:
        if l in name:
            return 'mask', name
    if 'fc' in name:
        return 'mask', name # Replace 'mask' here with 'train' if you don't want to mask the fc layer
    return 'freeze'

# Initialize with a normal distribution of mean one and std zero, i.e. initialize every score to a 1.0
scores_init = submasking.normal_scores_init(1.0, 0.0)

# Create a masked version of the model, using the default settings of a threshold of 0 
model = submasking.SubmaskedModel(model, parameter_selection=par_sel, scores_init=scores_init, shell_mode='replace')


# ... train the model
```

You can use this code to mask arbitrary architectures by only changing the `par_sel` function. In theory it should work for all, however, we only tested on ResNets from timm and Vision Transformers from CLIP.

# Reproduction

Clone repo into `UnsupervisedMasking/`

Go into the cloned repo `cd UnsupervisedMasking`

Create a virtual enviornment with

`python3 -m venv venv`

Activate the venv

`source venv/bin/activate`

Install dependencies with

`pip install -r requirements.txt`

Then, make a https://neptune.ai/ workspace and create a file `keys.sh` with

```
export NEPTUNE_PROJECT=...
export NEPTUNE_API_TOKEN=...
```

Finally, run `source load.sh`

Now you are ready to run the experiments.

## Experiments
### Supervised masking (ResNet)
```
python3 main.py --config-name submask --multirun main.dataset=cifar100 model.pret.source=hub model.pret.name=resnet50 main.dataset_subset=10p
```

See `volt/dataset.py->get_dataset()` for the available datasets.
Remove `main.dataset_subset=10p` to run on the full dataset rather than 10%.

Find `training/val/accuracy` in the Neptune UI.

### Supervised masking (CLIP-ViT)
```
python3 main.py --config-name submask_clip_p --multirun main.dataset=cifar100
```


### Full-Fine-Tuning baseline
```
python3 main.py --config-name baseline --multirun main.dataset=cifar100 model.pret.source=hub model.pret.name=resnet50 dl.lr=0.15 dl.weight_decay=0.000001 main.dataset_subset=10p
```

### Linear Probing
```
python3 main.py --config-name=baseline model.pret.source=hub model.pret.name=resnet50  main.train=0 main.probe=1 main.dataset_subset=10p
```

Probing happens after training, so you can remove `main.train=0` to train the model first.

### Self-supervised masking
```
python3 main.py --config-name=submask_lin model.pret.source=hub model.pret.name=resnet50 model.pret.module=swav main.dataset=multicrop_cifar100 dl.scheduler=cosine cls.head_lr=0.15 cls.head_wd=0.000001 swav.queue_length=512 swav.epoch_queue_starts=30 swav.nmb_prototypes=500 dl.scheduler=cosine
```

See `volt/dataset.py->get_multicrop_dataset()` for the available datasets.
Find `training/test/knn_accuracy_mem_train` in the Neptune UI for the k-NN accuracy (val set = test set in all our datasets).
For linear probe accuracy, append `main.probe=1` to the command, and find `training/val/accuracy` in the Neptune UI.

### Model Cascade
1. Train the dispatcher with self-supervised masking (see above).
2. Use `notebook/clustering_v2.ipynb` to create clusters (edit `run_id` in the first cell to match the run id of the model you trained in step 1). Output will be saved in files `output_clusters.pt` (cluster assignments on training set), `output_gm.pt` (gaussian mixture parameters), `output_pca.pt` (PCA parameters).
3. Now, for each of the 5 clusters, fine tune the model in step 1 with
```
python3 main.py --config-name=submask_lin model.pret.source=hub model.pret.name=resnet50 model.pret.module=swav main.dataset=multicrop_cifar100 dl.scheduler=cosine cls.head_lr=0.15 cls.head_wd=0.000001 swav.queue_length=512 swav.epoch_queue_starts={queue_start} swav.nmb_prototypes=500 dl.scheduler=cosine main.dataset_subset_clusters_file=notebook/output_clusters.pt main.dataset_subset_cluster={cluster_id} dl.epochs={epochs} main.load_checkpoint={run_id_step1}/last.ckpt main.strict_load_checkpoint=0
```
 Make sure to fill in the values for {queue_start}, {epochs}, {cluster_id} and {run_id_step1} (the run id of the model you trained in step 1). Use the formula $E = 50000/D ∗ 150$ to determine the number of epochs to train for, where $D$ is the number of datapoints in the cluster. Start the queue at $E/5$. Cluster id is 0-indexed. Run the command 5 times for the different cluster ids.

 
4. Now, embed the full dataset with each cluster by running

```
python3 main.py --config-name=submask_lin model.pret.source=hub model.pret.name=resnet50 model.pret.module=swav main.dataset=multicrop_cifar100 swav.nmb_prototypes=500 dl.scheduler=cosine main.train=0 main.load_checkpoint={run_id_step3}/last.ckpt cls.force_analysis_cpu=1
```
, replacing `{run_id_step3}` with the run id from step 3. Its essentially the same command from step3 but we add `main.train=0` to prevent training (embedding happens automatically) and remove the `main.dataset_subset_clusters_file` parameter as well as some redundant training hyperparameters.


5. Next, fill in `conf_ensembling/c100_fixed.yaml` with the run ids from step4 (not step3). Make sure to fill them in the right order. The first run id should be the run from step1, and subsequent run ids should be from step4, starting with cluster 0's run and on to cluster 4's run.
6. Finaly, run `python3 ensemble.py --config-name=c100_fixed --multirun method=pca_unconditional` for unconditional accuracy and `python3 ensemble.py --config-name=c100_fixed --multirun method=pca_conditional` for conditional accuracy. Accuracies will be printed to stdout.


# Hardware
At least 24GB of GPU memory are necessary to run the most demanding experiments. On an A100 gpu, most experiments take less than 24h.
