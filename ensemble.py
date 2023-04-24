import io
import pickle

import hydra
import numpy as np
import torch
from omegaconf import DictConfig
from pathlib import Path

import scipy.stats as stats

from torch import nn

from volt import linear_probe
from volt.log import Task
from volt.modules.classifier import knn_predict
from volt.util import ddict
from volt.xtrace import XTrace


class CPUUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else:
            return super().find_class(module, name)


def load_data(run_id):
    path = Path('data/models/checkpoints', run_id, 'embeddings.pkl')

    feature_bank, feature_labels, embedding, targets, label_names = CPUUnpickler(open(path, 'rb')).load()

    path_unnorm = Path('data/models/checkpoints', run_id, 'embeddings_unnorm.pkl')
    feature_bank_unnorm, embedding_unnorm = CPUUnpickler(open(path_unnorm, 'rb')).load()

    return ddict(train_embeddings_norm=feature_bank, train_labels=feature_labels, val_embeddings_norm=embedding,
                 val_labels=targets, label_names=label_names, train_embeddings=feature_bank_unnorm,
                 val_embeddings=embedding_unnorm)


def recov(train_embeddings, train_cluster_assignments, n_clusters):
    # Initialize a list to store the covariance matrices
    covariance_matrices = []

    A = train_embeddings[0]  # nn.functional.normalize(train_embeddings[0], dim=1, p=2)
    U, S, V = torch.pca_lowrank(A, q=20, center=True, niter=2)
    # Train
    embeddings = U

    cluster_means = []
    # Loop through all clusters
    for cluster_id in range(n_clusters):
        # Extract the points belonging to the current cluster
        cluster_points = embeddings[train_cluster_assignments == cluster_id]

        # Compute the mean of the current cluster
        cluster_mean = torch.mean(cluster_points, dim=0)
        cluster_means.append(cluster_mean)

        # Compute the covariance matrix for the current cluster
        cluster_covariance = torch.cov((cluster_points - cluster_mean).T)

        # Add the computed covariance matrix to the list
        covariance_matrices.append(cluster_covariance)

    # Number of datapoints
    n_datapoints = embeddings.shape[0]

    # Initialize a matrix to store the probabilities
    # Rows represent datapoints, and columns represent clusters
    probabilities = torch.zeros((n_datapoints, n_clusters))

    # Loop through all clusters
    for cluster_id in range(n_clusters):
        # Get the mean and covariance matrix for the current cluster
        cluster_mean = cluster_means[cluster_id]
        cluster_covariance = covariance_matrices[cluster_id]

        distr = torch.distributions.multivariate_normal.MultivariateNormal(cluster_mean,
                                                                           covariance_matrix=cluster_covariance)
        probability = distr.log_prob(embeddings)

        # Compute the probability using the multivariate Gaussian pdf
        # probability = stats.multivariate_normal.pdf(train_embeddings[0].numpy(), mean=cluster_mean.numpy(), cov=cluster_covariance.numpy())

        # Store the probability in the matrix
        probabilities[:, cluster_id] = probability

    assert torch.argmax(probabilities, dim=1).eq(train_cluster_assignments).all()


@hydra.main(version_base=None, config_path="conf_ensembling", config_name="ensembling")
def main(cfg: DictConfig):
    method = cfg.method
    assert cfg.dispatcher == 0
    datas = [load_data(run_id) for run_id in cfg.run_ids]
    assert all([data['label_names'] == datas[0]['label_names'] for data in datas])
    assert all([torch.all(data['train_labels'] == datas[0]['train_labels']) for data in datas])
    assert all([torch.all(data['val_labels'] == datas[0]['val_labels']) for data in datas])
    label_names = datas[0]['label_names']
    train_labels = datas[0]['train_labels']
    val_labels = datas[0]['val_labels']

    train_embeddings = [data['train_embeddings'].T for data in datas]
    val_embeddings = [data['val_embeddings'] for data in datas]
    train_embeddings_norm = [data['train_embeddings_norm'].T for data in datas]
    val_embeddings_norm = [data['val_embeddings_norm'] for data in datas]

    train_cluster_assignments = torch.tensor(torch.load('notebook/output_clustersB.pt'))
    gm = torch.load('notebook/output_gmB.pt')
    A, U, S, V, E = torch.load('notebook/output_pcaB.pt')

    def assign(embeddings):
        embs = ((embeddings - A.mean(axis=0)) @ V @ torch.diag(1 / S)).cpu().numpy()
        predictions = gm.predict(embs)
        return torch.tensor(predictions)

    assert torch.all(assign(train_embeddings_norm[0]) == train_cluster_assignments)

    val_cluster_assignments = assign(val_embeddings_norm[0])

    # replace mode
    if method == 'replace':
        train_ensemble_embeddings = train_embeddings[0].clone()
        val_ensemble_embeddings = val_embeddings[0].clone()
        for i, cluster in enumerate(cfg.cluster_ids):
            train_ensemble_embeddings[train_cluster_assignments == cluster] = train_embeddings[i+1][train_cluster_assignments == cluster]
            val_ensemble_embeddings[val_cluster_assignments == cluster] = val_embeddings[i+1][val_cluster_assignments == cluster]
    elif method == 'add_conditional':
        train_ensemble_embeddings = train_embeddings[0].clone()
        val_ensemble_embeddings = val_embeddings[0].clone()
        for i, cluster in enumerate(cfg.cluster_ids):
            train_ensemble_embeddings[train_cluster_assignments == cluster] += train_embeddings[i + 1][
                train_cluster_assignments == cluster]
            val_ensemble_embeddings[val_cluster_assignments == cluster] += val_embeddings[i + 1][
                val_cluster_assignments == cluster]
    elif method == 'add_unconditional':
        train_ensemble_embeddings = train_embeddings[0].clone()
        val_ensemble_embeddings = val_embeddings[0].clone()
        for embed in train_embeddings[1:]:
            train_ensemble_embeddings += embed
        for embed in val_embeddings[1:]:
            val_ensemble_embeddings += embed
    elif method == 'unconditional':
        train_ensemble_embeddings = torch.cat(train_embeddings, dim=1)
        val_ensemble_embeddings = torch.cat(val_embeddings, dim=1)
    elif method == 'conditional':
        train_ensemble_embeddings = torch.cat(train_embeddings, dim=1)
        val_ensemble_embeddings = torch.cat(val_embeddings, dim=1)
        spl = train_embeddings[1].shape[1]
        # zero out the embeddings of each cluster on the data points that do not belong to that cluster
        for i, cluster in enumerate(cfg.cluster_ids):
            fro = (i + 1) * spl
            to = (i + 2) * spl
            train_ensemble_embeddings[train_cluster_assignments != cluster, fro:to] = 0
            val_ensemble_embeddings[val_cluster_assignments != cluster, fro:to] = 0
    elif method == 'baseline':
        train_ensemble_embeddings = train_embeddings[0]
        val_ensemble_embeddings = val_embeddings[0]
    else:
        raise NotImplementedError

    val_ensemble_embeddings_norm = nn.functional.normalize(val_ensemble_embeddings, dim=1, p=2)
    train_ensemble_embeddings_norm = nn.functional.normalize(train_ensemble_embeddings, dim=1, p=2)
    pred_labels = knn_predict(val_ensemble_embeddings_norm, train_ensemble_embeddings_norm.T, train_labels,
                              len(label_names), 200,
                              0.1)
    hits = (pred_labels[:, 0] == val_labels).float()
    knn_accuracy = hits.sum().item() / len(val_labels)
    print(f'KNN Accuracy: {knn_accuracy}')

    # KNN precision, recall
    dim_size = max(val_labels) + 1

    # These may be off by a bit due to buggy scatter_reduce 'mean'
    precision = torch.zeros(dim_size).scatter_reduce(0, pred_labels[:, 0], hits,
                                                     reduce='mean')
    recall = torch.zeros(dim_size).scatter_reduce(0, val_labels, hits, reduce='mean')
    # for i, l in enumerate(label_names):
    #     print(f'{l}: precision={precision[i]:.4f}, recall={recall[i]:.4f}')

    X = train_ensemble_embeddings.numpy().astype('float64')
    y = train_labels.numpy()

    print(X.shape, y.shape)

    task = Task('Probing').start()
    (r, b), score = linear_probe.probe_(cfg, X, y, bias=True)
    task.done()

    y_pred = val_ensemble_embeddings.numpy() @ r + b
    accuracy = (y_pred.argmax(axis=-1) == val_labels.numpy()).astype('float64').sum().item() / len(val_labels)
    print(f'Linear probe accuracy: {accuracy}')

    print(f'KNN Accuracy: {knn_accuracy}')
    print(f'Method: {method}')

    # import pdb;
    # pdb.set_trace()

def knn_assign(train_cluster_assignments, train_embeddings, val_embeddings):
    n_clusters = train_cluster_assignments.max() + 1
    feat_dim = train_embeddings[0].shape[1]
    train_means = torch.zeros((n_clusters, feat_dim)).scatter_reduce(0, train_cluster_assignments.unsqueeze(1).repeat(1,
                                                                                                                      feat_dim),
                                                                     torch.tensor(train_embeddings[0]),
                                                                     reduce='mean')


    train_cluster_distances = torch.cdist(train_embeddings[0], train_means)
    print(f"MATCH {torch.argmin(train_cluster_distances, dim=1).eq(train_cluster_assignments).float().mean()}")

    val_cluster_distances = torch.cdist(val_embeddings[0], train_means)
    val_cluster_assignments = torch.argmin(val_cluster_distances, dim=1)

if __name__ == "__main__":
    XTrace(main, whitelist=['prunetuning']).__call__()
