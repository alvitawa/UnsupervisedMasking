from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from volt.modules.deep_learning import DeepLearningModule
from volt import config, util


@dataclass
class ClassifierConfig:
    loss: str = 'cross_entropy'
    fc: str = ''


config.register('classifier', ClassifierConfig)


class ClassifierModule(DeepLearningModule):
    def __init__(self, fc_in_features, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # assert all(isinstance(dataset, ClassDataset) for dataset in self.dataset.values())

        if self.cfg.cls.loss == 'cross_entropy':
            self.loss = torch.nn.CrossEntropyLoss()

        self.with_fc = True
        if self.cfg.cls.fc == 'linear':
            self.fc = torch.nn.Linear(fc_in_features, len(self.dataset['train'].label_names))
        else:
            self.with_fc = False


    def analyze(self, outputs, sample_inputs, sample_outputs, namespace):
        super().analyze(outputs, sample_inputs, sample_outputs, namespace)

        if 'output' in outputs and 'target' in outputs:
            hits = (outputs['output'] == outputs['target']).float()

            accuracy = hits.mean()

            labels = self.dataset[namespace].labels
            dim_size = max(labels) + 1

            # These may be off by a bit due to buggy scatter_reduce 'mean'
            precision = torch.zeros(dim_size, device=self.device).scatter_reduce(0, outputs['output'], hits, reduce='mean')
            recall = torch.zeros(dim_size, device=self.device).scatter_reduce(0, outputs['target'], hits, reduce='mean')

            self.logger_.experiment[f'training/{namespace}/accuracy'].log(accuracy)
            for i, name in self.dataset[namespace].classes.items():
                self.logger_.experiment[f'training/{namespace}/precision/{name}'].log(precision[i])
                self.logger_.experiment[f'training/{namespace}/recall/{name}'].log(recall[i])

        if namespace == 'val' and 'embedding' in outputs and 'target' in outputs:
            embedding = nn.functional.normalize(outputs['embedding'], dim=1, p=2).to(self.device)
            feature_bank = embedding.T.contiguous().to(self.device)
            feature_labels = outputs['target'].to(self.device)
            classes = len(self.dataset['val'].label_names)
            if self.cfg.main.in_mvp:
                knn_k = 4
            else:
                knn_k = 200
                assert len(feature_bank) // classes / 2 < knn_k, 'knn_k is too large'

            pred_labels = knn_predict(embedding, feature_bank, feature_labels, classes, knn_k, 0.1)
            accuracy = (pred_labels[:, 0] == feature_labels).float().sum().item() / embedding.shape[0]
            self.logger_.experiment[f'training/{namespace}/knn_accuracy'].log(accuracy)


    def training_step(self, batch, batch_idx, namespace='train'):
        data, target = batch
        output = self.forward(data)
        if self.with_fc:
            embeddings = output
            output = self.fc(output)
        loss = self.loss(output, target)

        self.log(f'{namespace}/loss', loss)
        r = {'loss': loss, 'output': torch.argmax(output, dim=1), 'target': target}
        if self.with_fc:
            r['embedding'] = embeddings
        return r


class ContrastiveClassifierModule(ClassifierModule):
    def __init__(self, embedding_size, logit_scale, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.logit_scale = logit_scale
        logit_scale.requires_grad = False

        # Note embeddings and target caches don't align with the dataset, they are 'shuffled' on each epoch
        self.embeddings_cache = torch.zeros((len(self.dataset['train']), embedding_size), device='cpu')
        self.targets_cache = torch.zeros((len(self.dataset['train']),), device='cpu', dtype=torch.long)
        # The index of the means cache does align with the index of the label
        self.means_cache = torch.zeros((len(self.dataset['train'].labels), embedding_size), device='cpu')

        assert self.dataset['train'].labels == self.dataset['val'].labels, \
            'Training and validation datasets must have the same labels'

        if self.cfg.cls.loss == 'bce':
            self.loss = torch.nn.BCEWithLogitsLoss()
        self.cosine_similarity = torch.nn.CosineSimilarity(dim=2)

    def analyze(self, outputs, sample_inputs, sample_outputs, namespace):
        if namespace == 'train':
            return
        super().analyze(outputs, sample_inputs, sample_outputs, namespace)

    def training_step(self, batch, batch_idx, namespace='train'):
        data, target = batch
        output = self.forward(data)
        # loss = self.loss(output, target)

        contrast = self.cosine_similarity(output.unsqueeze(1), output.unsqueeze(0))

        targets = (target[None, :] == target[:, None])
        if self.cfg.cls.loss == 'bce':
            targets = targets.float()

            loss = self.loss(contrast, targets)
        elif self.cfg.cls.loss == 'cross_entropy':
            contrast *= self.logit_scale.exp()
            eye = torch.eye(len(contrast), device=self.device).bool()
            contrast[targets ^ eye] = -1e9

            horizontal_loss = torch.nn.functional.cross_entropy(contrast, torch.arange(len(contrast), device=contrast.device))
            vertical_loss = torch.nn.functional.cross_entropy(contrast.T, torch.arange(len(contrast), device=contrast.device))
            loss = (horizontal_loss + vertical_loss) / 2
        else:
            raise NotImplementedError

        self.log(f'{namespace}/loss', loss)

        if namespace == 'val' or namespace == 'test' or namespace == 'val_sample' or namespace == 'test_sample':
            predictions = torch.nn.functional.cosine_similarity(output[:, None],
                                                                self.means_cache[None, :].to(self.device), dim=2)
            predictions = torch.argmax(predictions, dim=1)
        elif namespace == 'train':
            predictions = None
            index = batch_idx * self.cfg.dl.batch_size
            self.embeddings_cache[index:index + data.shape[0]] = output.detach().cpu()
            self.targets_cache[index:index + data.shape[0]] = target.detach().cpu()
        else:
            predictions = None


        return_value = {'loss': loss, 'target': target}
        if predictions is not None:
            return_value['output'] = predictions
        return return_value

    def on_validation_start(self):
        self.means_cache.zero_()
        self.means_cache.scatter_reduce_(0, self.targets_cache[:, None].expand(self.embeddings_cache.shape), self.embeddings_cache, reduce='sum')

        # Scatter_reduce 'mean' is buggy so I normalize manually here
        bins = torch.bincount(self.targets_cache, minlength=len(self.dataset['train'].labels))
        bins[bins == 0] = 1
        self.means_cache /= bins[:, None]

        plot = util.plot_matrix(self.means_cache.detach().cpu().numpy())
        self.logger_.experiment[f'training/means_cache'].log(plot)

    def on_test_start(self):
        self.on_validation_start()

class PromptClassifierModule(ClassifierModule):
    def __init__(self, prompt_embeddings, logit_scale, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.prompt_embeddings = prompt_embeddings
        self.prompt_embeddings.requires_grad = False
        self.logit_scale = logit_scale
        self.logit_scale.requires_grad = False

        assert self.cfg.cls.loss == 'cross_entropy', 'Prompt classifier only supports cross entropy loss'

    def training_step(self, batch, batch_idx, namespace='train'):
        data, target = batch
        output = self.forward(data)

        contrast = torch.nn.functional.cosine_similarity(output[:, None], self.prompt_embeddings[None, :], dim=2)
        contrast *= self.logit_scale.exp()

        loss = self.loss(contrast, target)

        self.log(f'{namespace}/loss', loss)
        return {'loss': loss, 'output': torch.argmax(contrast, dim=1), 'target': target}


def knn_predict(feature, feature_bank, feature_labels, classes, knn_k, knn_t):
    # compute cos similarity between each feature vector and feature bank ---> [B, N]
    sim_matrix = torch.mm(feature, feature_bank)
    # [B, K]
    sim_weight, sim_indices = sim_matrix.topk(k=knn_k, dim=-1)
    # [B, K]
    sim_labels = torch.gather(feature_labels.expand(feature.size(0), -1), dim=-1, index=sim_indices)
    sim_weight = (sim_weight / knn_t).exp()

    # counts for each class
    one_hot_label = torch.zeros(feature.size(0) * knn_k, classes, device=sim_labels.device)
    # [B*K, C]
    one_hot_label = one_hot_label.scatter(dim=-1, index=sim_labels.view(-1, 1), value=1.0)
    # weighted score ---> [B, C]
    pred_scores = torch.sum(one_hot_label.view(feature.size(0), -1, classes) * sim_weight.unsqueeze(dim=-1), dim=1)

    pred_labels = pred_scores.argsort(dim=-1, descending=True)
    return pred_labels
