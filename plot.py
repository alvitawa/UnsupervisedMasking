import os

import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig

from volt.log import Task
from volt.xtrace import XTrace

import neptune.new as neptune

PLOT_DATA_PATH = "report/plots/plot_data/"

def variant_sparsity(cfg):
    runs = cfg.variant_sparsity.runs
    df = None
    for run in runs:
        run = neptune.init_run(with_id=run)
        acc = run["training/val/accuracy"].fetch_values()
        sparsity = run["training/val/model_analysis/prune_k"].fetch_values()
        model_name = run["global/params/model/timm/name"].fetch()

        assert np.all(acc['step'] == sparsity['step'])

        if df is None:
            df = pd.DataFrame({"sparsity": sparsity['value']})
        else:
            assert np.allclose(sparsity['value'], df["sparsity"], atol=1e-5)
            assert model_name not in df.columns
        df[model_name] = acc['value']
    df.to_csv(PLOT_DATA_PATH + "variant_sparsity.csv", sep="\t", index=False)

    df = None
    for free in cfg.variant_sparsity.frees:
        run = neptune.init_run(with_id=free)
        acc = run["training/val/accuracy"].fetch_values()['value'].iloc[-1]
        sparsity = run["training/val/model_analysis/mean_mask"].fetch_values()['value'].iloc[-1]
        model_name = run["global/params/model/timm/name"].fetch()
        if df is None:
            df = pd.DataFrame({"acc": [acc], "sparsity": [sparsity], "model": [model_name]})
        else:
            df = df.append({"acc": acc, "sparsity": sparsity, "model": model_name}, ignore_index=True)
    df.to_csv(PLOT_DATA_PATH + "variant_sparsity_free.csv", sep="\t", index=False)

    os.system("cd report/plots && arara variant_sparsity.tex && cd ../..")

def resnet_model_size(cfg):
    runs = cfg.resnet_model_size.runs
    df = None
    for run in runs:
        run = neptune.init_run(with_id=run)
        model_name = run["global/params/model/timm/name"].fetch()
        acc = run["training/val/accuracy"].fetch_values()['value'].iloc[-1]
        sparsity = run["training/val/model_analysis/mean_mask"].fetch_values()['value'].iloc[-1]
        if df is None:
            df = pd.DataFrame({"acc": [acc], "sparsity": [sparsity], "model": [model_name]})
        else:
            df = df.append({"acc": acc, "sparsity": sparsity, "model": model_name}, ignore_index=True)
    df.to_csv(PLOT_DATA_PATH + "resnet_model_size.csv", sep="\t", index=False)
    os.system("cd report/plots && arara resnet_model_size.tex && cd ../..")

    for csr in cfg.resnet_model_size.constrained_sparsity_runs:
        model_name = csr.model
        accs = []
        sparsities = []
        for run in csr.runs:
            run = neptune.init_run(with_id=run)
            acc = run["training/val/accuracy"].fetch_values()['value'].iloc[-1]
            sparsity = run["training/val/model_analysis/prune_k"].fetch_values()['value'].iloc[-1]
            accs.append(acc)
            sparsities.append(sparsity)
        df = pd.DataFrame({"sparsity": sparsities, "acc": accs})
        df.sort_values(by="sparsity", inplace=True)
        df.to_csv(PLOT_DATA_PATH + f"constrained_sparsity_{model_name}.csv", sep="\t", index=False)



@hydra.main(version_base=None, config_path="conf", config_name="plotting")
def main(cfg: DictConfig):

    # Connect to the project "classification" in the workspace "ml-team"
    project = neptune.init_project()

    task = Task(f"Generating plot data and plots").start()

    do = cfg.plot

    print(do)
    if "variant_sparsity" in do:
        variant_sparsity(cfg)
    if "resnet_model_size" in do:
        resnet_model_size(cfg)

    task.done()

    task = Task(f"Re-compiling PDF").start()

    os.system("cd report && arara title-page-ai.tex")

    task.done()


if __name__ == "__main__":
    XTrace(main, whitelist=['prunetuning']).__call__()
