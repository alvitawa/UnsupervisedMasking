import io
import pickle

import hydra
import torch
from omegaconf import DictConfig
from pathlib import Path

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
                 val_labels=targets, label_names=label_names, train_embeddings=feature_bank_unnorm, val_embeddings=embedding_unnorm)


def combine_embeddings(embeddings, method):
    if method == 'average':
        ensemble_embeddings = torch.mean(torch.stack([emb for emb in embeddings]), dim=0)

    elif method == 'max_pooling':
        ensemble_embeddings = torch.max(torch.stack([emb for emb in embeddings]), dim=0).values

    elif method == 'min_pooling':
        ensemble_embeddings = torch.min(torch.stack([emb for emb in embeddings]), dim=0).values

    elif method == 'concatenation':
        ensemble_embeddings = torch.cat([emb for emb in embeddings], dim=-1)

    elif method == 'sum':
        ensemble_embeddings = torch.sum(torch.stack([emb for emb in embeddings]), dim=0)

    elif method == 'geom_mean':
        ensemble_embeddings = torch.prod(torch.stack([emb.clamp(min=1e-9) for emb in embeddings]),
                                         dim=0).pow(1 / len(embeddings))
    elif method == 'harmonic_mean':
        ensemble_embeddings = len(embeddings) / torch.sum(
            torch.stack([1 / (emb.clamp(min=1e-9)) for emb in embeddings]), dim=0)
    elif method == 'ranked_averaging':
        sorted_indices = torch.argsort(torch.stack([emb for emb in embeddings]), dim=0)
        ranked_embeddings = torch.zeros_like(sorted_indices, dtype=torch.float)
        for rank, idx in enumerate(sorted_indices):
            ranked_embeddings[idx] = rank + 1
        ensemble_embeddings = torch.mean(ranked_embeddings, dim=0)
    elif method == 'matrix_completion':
        from sklearn.impute import SimpleImputer
        imputer = SimpleImputer(strategy='mean')
        data_matrix = torch.stack([emb for emb in embeddings], dim=0).numpy()
        imputed_data = imputer.fit_transform(data_matrix)
        ensemble_embeddings = torch.tensor(imputed_data.mean(axis=0))
    else:
        raise ValueError(f'Unknown method: {method}')
    return ensemble_embeddings


@hydra.main(version_base=None, config_path="conf_ensembling", config_name="ensembling")
def main(cfg: DictConfig):
    method = cfg.method
    print(f'Ensembling method: {method}')
    datas = [load_data(run_id) for run_id in cfg.run_ids]
    assert all([data['label_names'] == datas[0]['label_names'] for data in datas])
    assert all([torch.all(data['train_labels'] == datas[0]['train_labels']) for data in datas])
    assert all([torch.all(data['val_labels'] == datas[0]['val_labels']) for data in datas])
    label_names = datas[0]['label_names']
    train_labels = datas[0]['train_labels']
    val_labels = datas[0]['val_labels']

    train_embeddings = [data['train_embeddings'].T for data in datas]
    val_embeddings = [data['val_embeddings'] for data in datas]
    train_ensemble_embeddings = combine_embeddings(train_embeddings, method)
    val_ensemble_embeddings = combine_embeddings(val_embeddings, method)



    filter_ = "bicycle:bus:motorcycle:pickup_truck:train:lawn_mower:rocket:streetcar:tank:tractor:cloud:forest:mountain:plain:sea:bridge:castle:house:road:skyscraper".split(":")
    filter_ = [label_names.index(f) for f in filter_]

    label_names = [label_names[i] for i in filter_]

    train_mask = torch.tensor([train_labels[i] in filter_ for i in range(len(train_labels))])
    val_mask = torch.tensor([val_labels[i] in filter_ for i in range(len(val_labels))])

    train_labels = train_labels[train_mask]
    for i in range(len(train_labels)):
        train_labels[i] = filter_.index(train_labels[i])
    val_labels = val_labels[val_mask]
    for i in range(len(val_labels)):
        val_labels[i] = filter_.index(val_labels[i])
    train_ensemble_embeddings = train_ensemble_embeddings[train_mask]
    val_ensemble_embeddings = val_ensemble_embeddings[val_mask]



    # allowed_classes = {'bicycle', 'bus', 'motorcycle', 'pickup_truck', 'train', 'lawn_mower', 'rocket', 'streetcar',
    #                    'tank', 'tractor', 'cloud', 'forest', 'mountain', 'plain', 'sea', 'bridge', 'castle', 'house',
    #                    'road', 'skyscraper'}
    #
    # def zero_out_second_half(ensemble_embeddings, labels, allowed_classes):
    #     half_size = ensemble_embeddings.shape[-1] // 2
    #     for idx, label in enumerate(labels):
    #         class_label = label_names[label]
    #         if class_label not in allowed_classes:
    #             ensemble_embeddings[idx, half_size:] = 0
    #     return ensemble_embeddings

    # train_ensemble_embeddings = zero_out_second_half(train_ensemble_embeddings, train_labels, allowed_classes)
    # val_ensemble_embeddings = zero_out_second_half(val_ensemble_embeddings, val_labels, allowed_classes)

    pred_labels = knn_predict(val_ensemble_embeddings, train_ensemble_embeddings.T, train_labels, len(label_names), 200,
                              0.1)
    hits = (pred_labels[:, 0] == val_labels).float()
    accuracy = hits.sum().item() / len(val_labels)
    print(f'KNN Accuracy: {accuracy}')

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

    import pdb; pdb.set_trace()


if __name__ == "__main__":
    XTrace(main, whitelist=['prunetuning']).__call__()
