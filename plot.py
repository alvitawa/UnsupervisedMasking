import os
import re
import time

import hydra
import numpy as np
import pandas as pd
from neptune.new.exceptions import MissingFieldException
from omegaconf import DictConfig

from volt.log import Task
from volt.xtrace import XTrace

import neptune.new as neptune
from joblib import Memory

location = 'data/cachedir'
memory = Memory(location, verbose=1)

PLOT_DATA_PATH = "report/plots/"


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


def get_params(run_id):
    run = neptune.init_run(with_id=run_id)
    params = run["global/params"].fetch()
    run.stop()
    time.sleep(0.4)
    return params


def get_params_cached(run_id):
    return memory.cache(get_params)(run_id)


def get_metrics(run_id):
    run = neptune.init_run(with_id=run_id)
    try:
        acc = run["training/val/accuracy"].fetch_values()['value'].iloc[-1]
    except MissingFieldException:
        acc = np.nan
    try:
        sparsity = run["training/val/model_analysis/mean_mask"].fetch_values()['value'].iloc[-1]
    except MissingFieldException:
        sparsity = np.nan
    try:
        acc_knn = run["training/test/knn_accuracy_mem_train"].fetch_values()['value'].iloc[-1]
    except MissingFieldException:
        acc_knn = np.nan
    try:
        epoch = run["training/epoch"].fetch_values()['value'].iloc[-1]
    except MissingFieldException:
        epoch = np.nan
    run.stop()
    time.sleep(0.4)
    return acc, sparsity, acc_knn, epoch


def get_metrics_cached(run_id):
    return memory.cache(get_metrics)(run_id)


exemptions = [
    r"^main.dataset$",
    r"^hydra",
    r"^dl.multiprocessing_context$",
    r"^dl.num_workers$",
    r"^dl.pin_memory$",
    r"^main.verbose$",
    r"^main.save_top_k$",
    r"^save_every_n_epochs$",
]


def iterdict(d, p=""):
    for k, v in d.items():
        if isinstance(v, dict):
            yield from iterdict(v, p + '.' + k)
        else:
            yield (p + '.' + k).strip('.'), v


def get_recursive(d, path: str):
    spl = path.split('.', 1)
    if len(spl) == 1:
        return d[spl[0]]
    next_path, rest = spl
    return get_recursive(d[next_path], rest)


def check_subseteq(anchor_params, candidate_params):
    for p, v in iterdict(anchor_params):
        if any(re.match(ex, p) for ex in exemptions):
            continue
        try:
            if get_recursive(candidate_params, p) != v:
                print(f"Param {p} is not equal: {v} != {get_recursive(candidate_params, p)}")
                break
        except KeyError:
            print(f"Param {p} is not in candidate")
            break
    else:
        return True
    return False


def get_similar(runs_df, run_ids):
    anchor_params_all = list(get_params_cached(run_id) for run_id in run_ids)
    results = []
    for candidate_run_id in runs_df['sys/id']:
        candidate_params = get_params_cached(candidate_run_id)
        # iterate over all k/v pairs in nested dict
        if any(check_subseteq(anchor_params, candidate_params) for anchor_params in anchor_params_all):
            results.append((candidate_run_id, candidate_params))
    return results


def part2_tables(cfg):
    project = neptune.init_project()
    runs = project.fetch_runs_table(columns=["sys/id", "global/command"], state="idle")
    df = runs.to_pandas().dropna()

    tables = {}
    for table in cfg.part2_tables:
        name = table.name
        points_ids = {}
        points_acc = {}  # {(row, col): [metric]}
        points_sparsity = {}
        points_acc_knn = {}
        for model_name, run_ids in table.anchors.items():
            anchor_metrics = get_metrics_cached(run_ids[0])
            sim = get_similar(df, run_ids)
            for run_id, params in sim:
                acc, sparsity, acc_knn, epoch = get_metrics_cached(run_id)
                if epoch != anchor_metrics[3]:
                    continue
                row = model_name
                col = get_recursive(params, 'main.dataset').replace('cifar10pgn', 'cifar10')
                if (row, col) not in points_acc:
                    points_ids[(row, col)] = []
                    points_acc[(row, col)] = []
                    points_sparsity[(row, col)] = []
                    points_acc_knn[(row, col)] = []
                points_ids[(row, col)].append(run_id)
                points_acc[(row, col)].append(acc)
                points_sparsity[(row, col)].append(sparsity)
                points_acc_knn[(row, col)].append(acc_knn)

        if 'manuals' in table:
            for model_name, manual_results in table.manuals.items():
                for dataset, manual_acc in manual_results.items():
                    row = model_name
                    col = dataset
                    if (row, col) not in points_acc:
                        points_ids[(row, col)] = []
                        points_acc[(row, col)] = []
                        points_sparsity[(row, col)] = []
                        points_acc_knn[(row, col)] = []
                    points_ids[(row, col)].append("manual")
                    points_acc[(row, col)].append(manual_acc)
                    points_sparsity[(row, col)].append(0)
                    points_acc_knn[(row, col)].append(0)

        columns = sorted(set(col for _, col in points_acc.keys()))
        rows = list(model_name for model_name, _ in table.anchors.items())

        df_ids = pd.DataFrame(columns=columns, index=rows, data=np.full((len(rows), len(columns)), ""))
        for (row, col), ids in points_ids.items():
            df_ids.loc[row, col] = ",".join(ids)
        df_ids.to_csv(PLOT_DATA_PATH + f"part2_accuracy/horizontal/part2_{name}_ids.csv", sep="\t")

        # empty df of right shape (filled with empty strings)
        df_acc = pd.DataFrame(columns=columns, index=rows, data=np.full((len(rows), len(columns)), ""))
        for (row, col), accs in points_acc.items():
            df_acc.loc[row, col] = f"{np.mean(accs):.2f} ± {np.std(accs):.2f}_{len(accs)}"
        df_acc.to_csv(PLOT_DATA_PATH + f"part2_accuracy/horizontal/part2_{name}_accuracy.csv", sep="\t")

        df_sparsity = pd.DataFrame(columns=columns, index=rows, data=np.full((len(rows), len(columns)), ""))
        for (row, col), sparsities in points_sparsity.items():
            sparsities = [1 - s for s in sparsities]
            df_sparsity.loc[row, col] = f"{np.mean(sparsities):.2f} ± {np.std(sparsities):.2f}_{len(sparsities)}"
        df_sparsity.to_csv(PLOT_DATA_PATH + f"part2_accuracy/horizontal/part2_{name}_sparsity.csv", sep="\t")

        df_acc_knn = pd.DataFrame(columns=columns, index=rows, data=np.full((len(rows), len(columns)), ""))
        for (row, col), accs in points_acc_knn.items():
            df_acc_knn.loc[row, col] = f"{np.mean(accs):.2f} ± {np.std(accs):.2f}_{len(accs)}"
        df_acc_knn.to_csv(PLOT_DATA_PATH + f"part2_accuracy/horizontal/part2_{name}_accuracy_knn.csv", sep="\t")

        tables[name] = {'ids': df_ids, 'acc': df_acc, 'sparsity': df_sparsity, 'acc_knn': df_acc_knn}

    model_names = sorted(set(k for t in cfg.part2_tables for k in t.anchors.keys()))

    # Now grouped by model name
    for model_name in model_names:
        for metric in ['ids', 'acc', 'sparsity', 'acc_knn']:
            points = {}  # {(row, col): [metric]}
            for table_name, table in tables.items():
                df = table[metric]
                for row_original in df.index:
                    if row_original != model_name:
                        continue
                    for col_original in df.columns:
                        row, col = col_original, table_name
                        if (row, col) not in points:
                            points[(row, col)] = []
                        points[(row, col)].append(df.loc[row_original, col_original])

            columns = set(col for _, col in points.keys())
            columns = list(t.name for t in cfg.part2_tables if t.name in columns)
            rows = sorted(set(row for row, _ in points.keys()))

            df = pd.DataFrame(columns=columns, index=rows, data=np.full((len(rows), len(columns)), ""))
            for (row, col), values in points.items():
                # values should only really contain one item, but this way
                # we can see if that doesn't happen
                df.loc[row, col] = ", ".join(values)
            if metric == 'acc_knn':
                # rename Linear probe col
                df = df.rename(columns={'Linear Probe': 'Zero Shot'})
            elif metric == 'sparsity':
                # Drop lp and fullft
                df = df.drop(columns=['Linear Probe', 'Full Fine-Tuning'], errors='ignore')
            df.rename(columns={'Full Fine-Tuning': 'Full FT'}, inplace=True, errors='ignore')
            df.to_csv(PLOT_DATA_PATH + f"part2_accuracy/vertical/part2_{model_name}_{metric}.csv", sep="\t",
                      index_label=model_name)

    # Now a single table per metric, where each row is a model+table_name(=method) and each column is a dataset
    for metric in ['ids', 'acc', 'sparsity', 'acc_knn']:  # metric
        points = {}  # {(row, col): [metric]}
        for table_name, table in tables.items():  # method
            df = table[metric]
            for row_original in df.index:  # model
                for col_original in df.columns:  # dataset
                    shorthand = table_name.replace("Full Fine-Tuning", "FFT").replace("Linear Probe", "LP").replace(
                        "Submasked", "FS")
                    row, col = f"{row_original}+{shorthand}", col_original
                    if (row, col) not in points:
                        points[(row, col)] = []
                    points[(row, col)].append(df.loc[row_original, col_original])
        columns = sorted(set(col for _, col in points.keys()))  # dataset

        rows_methods = ['LP', 'FS', 'FFT']
        rows_models = ['rn18-timm', 'rn50-swav']
        rows = list(f"{model_name}+{method}" for model_name in rows_models for method in rows_methods)

        df = pd.DataFrame(columns=columns, index=rows, data=np.full((len(rows), len(columns)), ""))
        for (row, col), values in points.items():
            if metric != 'ids':
                stripped = values[0].split(" ± ")[0]
            else:
                stripped = values[0]
            assert len(values) == 1, f"More than one value for {row}, {col}: {values}"
            df.loc[row, col] = stripped
        df.to_csv(PLOT_DATA_PATH + f"part2_accuracy/full/part2_{metric}.csv", sep="\t", index_label='Model+Method')

@hydra.main(version_base=None, config_path="conf_plotting", config_name="plotting")
def main(cfg: DictConfig):
    # Connect to the project "classification" in the workspace "ml-team"
    project = neptune.init_project()

    task = Task(f"Generating plot data and plots").start()

    do = cfg.plot

    print(do)
    for plot in cfg.plot:
        eval(plot)(cfg)

    task.done()


if __name__ == "__main__":
    XTrace(main, whitelist=['prunetuning']).__call__()
