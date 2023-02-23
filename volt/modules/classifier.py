from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm

from subnetworks import submasking
from volt.log import Task
from volt.modules.deep_learning import DeepLearningModule
from volt import config, util, linear_probe, optimizers


@dataclass
class ClassifierConfig:
    loss: str = 'cross_entropy'
    fc: str = ''
    fc_score_init: float = 1.0
    mem_train_every_epoch: bool = False
    knn_eval: bool = True
    head_lr: float = 1.0
    head_wd: float = 1.0
    head_scale_lr: bool = True
    head_scale_wd: bool = True
    mlp_depth: int = 2
    mlp_hidden_dim: int = -1
    size_scale_wd: bool = False
    multihead_heads: int = 4


config.register('classifier', ClassifierConfig)

class Mlp(nn.Module):
    def __init__(self, in_features, out_features=None, act_layer=nn.ReLU, hidden_layers=None,
                 normalize=True, dropout=0.0):
        super().__init__()
        if hidden_layers is None:
            hidden_layers = []
        self.args = (in_features, out_features, act_layer)
        out_features = out_features or in_features
        layers = []
        for i in hidden_layers:
            layers.append(nn.Linear(in_features, i))
            layers.append(act_layer())
            in_features = i
        final = nn.Linear(in_features, out_features)
        layers.append(final)
        self.composed = nn.Sequential(*layers)
        self.normalize = nn.functional.normalize if normalize else nn.Identity()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        return self.normalize(self.composed(self.dropout(x)))

class MultiHeadMlp(nn.Module):
    def __init__(self, in_features, out_features, n_heads, temperature=1.0, hidden_layers=None, act_layer=nn.ReLU):
        super().__init__()
        self.heads = nn.ModuleList([Mlp(in_features, out_features, act_layer, hidden_layers) for _ in range(n_heads)])
        self.dispatch = nn.Linear(in_features, n_heads)
        self.temperature = temperature

    def forward(self, x):
        head_outputs = []
        for head in self.heads:
            head_outputs.append(head(x))
        head_weights = self.get_head_weights(x)
        k = torch.sum(head_weights[:, :, None] * torch.stack(head_outputs, dim=1), dim=1)
        return nn.functional.normalize(k)

    def get_head_weights(self, x):
        dispatch_output = self.dispatch(x)
        head_weights = torch.softmax(dispatch_output*self.temperature, dim=-1)
        return head_weights



class ClassifierModule(DeepLearningModule):
    def __init__(self, fc_in_features, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # assert all(isinstance(dataset, ClassDataset) for dataset in self.dataset.values())

        if self.cfg.cls.loss == 'cross_entropy':
            self.loss = torch.nn.CrossEntropyLoss()

        self.with_fc = True
        if self.cfg.cls.fc == 'linear':
            self.fc = torch.nn.Linear(fc_in_features, len(self.dataset['train'].label_names))
        elif self.cfg.cls.fc == 'mlp_masked':
            depth = self.cfg.cls.mlp_depth
            hidden_dim = self.cfg.cls.mlp_hidden_dim if self.cfg.cls.mlp_hidden_dim > 0 else fc_in_features
            layers = []
            for i in range(depth - 1):
                layers.append(submasking.MaskedLinear(fc_in_features, hidden_dim, self.cfg.cls.fc_score_init))
                layers.append(torch.nn.ReLU())
            layers.append(submasking.MaskedLinear(hidden_dim, len(self.dataset['train'].label_names),
                                                  self.cfg.cls.fc_score_init))
            self.fc = torch.nn.Sequential(*layers)
        elif self.cfg.cls.fc.startswith('multihead'):
            self.fc = MultiHeadMlp(fc_in_features, len(self.dataset['train'].label_names), self.cfg.cls.multihead_heads).to(self.device_)
            if self.cfg.cls.fc == 'multihead_masked':
                self.fc = submasking.SubmaskedModel(self.fc, scores_init=submasking.normal_scores_init(self.cfg.cls.fc_score_init, 0), shell_mode='replace').to(self.device_)
        elif self.cfg.cls.fc == 'masked':
            self.fc = submasking.MaskedLinear(fc_in_features, len(self.dataset['train'].label_names),
                                              self.cfg.cls.fc_score_init)
        elif self.cfg.cls.fc == 'probed':
            if self.cfg.main.in_mvp:
                # During mvp some labels may be missing from the training set causing the matrix shape
                # to be wrong so just make a new one with the right shape as the values dont matter anyways
                # during mvp.
                w = torch.randn(fc_in_features, len(self.dataset['train'].label_names))
                b = torch.randn(len(self.dataset['train'].label_names))
            else:
                task = Task('Probing').start()
                r, score = linear_probe.probe(self.cfg, self, fc_in_features, self.forward_dataloader('train_unaugmented'),
                                              self.device_, bias=True)
                task.done()

                w, b = r

            self.fc = torch.nn.Linear(fc_in_features, len(self.dataset['train'].label_names), bias=True)
            self.fc.weight.data = torch.tensor(w.T).float()
            self.fc.weight.requires_grad = False
            self.fc.bias.data = torch.tensor(b).float()
            self.fc.bias.requires_grad = False

        elif self.cfg.cls.fc in ['positive', 'negative', 'dual']:
            positive = self.cfg.cls.fc in ['positive', 'dual']
            negative = self.cfg.cls.fc in ['negative', 'dual']
            self.fc = submasking.MaskedConstantLayer(fc_in_features, len(self.dataset['train'].label_names),
                                                     self.cfg.cls.fc_score_init, positive, negative)
        else:
            self.with_fc = False

    def knn_analysis(self, embedding, targets, feature_bank, feature_labels, namespace, bank=''):

        classes = len(self.dataset['val'].label_names)
        if self.cfg.main.in_mvp:
            knn_k = 4
        else:
            knn_k = 200
            assert len(feature_bank) // classes / 2 < knn_k, 'knn_k is too large'

        pred_labels = knn_predict(embedding, feature_bank, feature_labels, classes, knn_k, 0.1)
        accuracy = (pred_labels[:, 0] == targets).float().sum().item() / embedding.shape[0]
        self.logger_.experiment[f'training/{namespace}/knn_accuracy{bank}'].log(accuracy)

    def analyze(self, outputs, sample_inputs, sample_outputs, namespace):
        super().analyze(outputs, sample_inputs, sample_outputs, namespace)

        if 'output' in outputs and 'target' in outputs:
            hits = (outputs['output'] == outputs['target']).float()

            accuracy = hits.mean()

            labels = self.dataset[namespace].labels
            dim_size = max(labels) + 1

            # These may be off by a bit due to buggy scatter_reduce 'mean'
            precision = torch.zeros(dim_size, device=self.device).scatter_reduce(0, outputs['output'], hits,
                                                                                 reduce='mean')
            recall = torch.zeros(dim_size, device=self.device).scatter_reduce(0, outputs['target'], hits, reduce='mean')

            self.logger_.experiment[f'training/{namespace}/accuracy'].log(accuracy)
            for i, name in self.dataset[namespace].classes.items():
                self.logger_.experiment[f'training/{namespace}/precision/{name}'].log(precision[i])
                self.logger_.experiment[f'training/{namespace}/recall/{name}'].log(recall[i])

        if 'embedding' in outputs and 'target' in outputs and self.cfg.cls.knn_eval and namespace in ['val', 'test']:
            random_subset = torch.randperm(outputs['embedding'].shape[0])[:10000]
            embedding = nn.functional.normalize(outputs['embedding'][random_subset], dim=1, p=2).to(self.device)
            feature_bank = embedding.T.contiguous().to(self.device)
            feature_labels = outputs['target'][random_subset].to(self.device)
            self.knn_analysis(embedding, feature_labels, feature_bank, feature_labels, namespace)

            if self.dataset['train_unaugmented'] is not None and (
                    namespace in 'test' or self.cfg.cls.mem_train_every_epoch):
                if self.dataset['test_unaugmented'] is not None:
                    test_loader = self.forward_dataloader(namespace + '_unaugmented')
                    test_outputs = []
                    with torch.no_grad():
                        for i, batch in enumerate(tqdm(test_loader, desc='Test set embeddings')):
                            batch = util.move_to_device(batch, self.device)
                            data, target = batch
                            out = self.forward(data)
                            test_outputs.append({'embedding': out, 'target': target})
                    test_outputs = util.concat_dict_list(test_outputs)
                    embedding = nn.functional.normalize(test_outputs['embedding'], dim=1, p=2).to(self.device)
                    targets = test_outputs['target'].to(self.device)
                else:
                    embedding = nn.functional.normalize(outputs['embedding'], dim=1, p=2).to(self.device)
                    targets = outputs['target'].to(self.device)

                # Compute the embeddings for the whole training set
                train_loader = self.forward_dataloader('train_unaugmented')
                train_outputs = []
                with torch.no_grad():
                    for i, batch in enumerate(tqdm(train_loader, desc='Training set embeddings')):
                        batch = util.move_to_device(batch, self.device)
                        data, target = batch
                        out = self.forward(data)
                        train_outputs.append({'embedding': out, 'target': target})

                train_outputs = util.concat_dict_list(train_outputs)

                feature_bank = nn.functional.normalize(train_outputs['embedding'], dim=1, p=2).T.contiguous().to(
                    self.device)
                feature_labels = train_outputs['target'].to(self.device)
                self.knn_analysis(embedding, targets, feature_bank, feature_labels, namespace, bank='_mem_train')

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

    def configure_optimizers(self, head_params=None):
        if head_params is None:
            head_params = self.fc.parameters()
        if self.cfg.dl.optimizer == 'sgd' or self.cfg.dl.optimizer == 'sgdc':
            weight_decay = self.cfg.dl.weight_decay
            if self.cfg.cls.size_scale_wd:
                weight_decay *= 50000 / len(self.dataset['train'])
            self.logger_.experiment['global/computed_weight_decay'].log(weight_decay)

            head_lr = self.cfg.cls.head_lr * (self.learning_rate if self.cfg.cls.head_scale_lr else 1)
            head_wd = self.cfg.cls.head_wd * (weight_decay if self.cfg.cls.head_scale_wd else 1)
            op = (torch.optim.SGD if self.cfg.dl.optimizer == 'sgd' else optimizers.ConstantWD_SGD)
            optimizer = op(
                [{'params': self.model.parameters()},
                 {'params': head_params, 'lr': head_lr, 'weight_decay': head_wd}],
                lr=self.learning_rate,
                weight_decay=weight_decay,
                momentum=self.cfg.dl.momentum)
        else:
            raise ValueError(f'Unknown optimizer {self.cfg.dl.optimizer}')

        if self.cfg.dl.scheduler == 'none':
            return optimizer
        elif self.cfg.dl.scheduler == 'cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, self.cfg.dl.epochs, eta_min=self.cfg.dl.eta_min)
        elif self.cfg.dl.scheduler == 'cosine_warmup':
            linear_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda x: x / self.cfg.dl.warmup_epochs)
            cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, self.cfg.dl.epochs - self.cfg.dl.warmup_epochs)
            sequential_scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, [linear_scheduler, cosine_scheduler], milestones=[self.cfg.dl.warmup_epochs])
            scheduler = sequential_scheduler
        elif self.cfg.dl.scheduler == 'cosine_cooldown':
            cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, self.cfg.dl.epochs - self.cfg.dl.cooldown, eta_min=self.cfg.dl.eta_min)
            constant_scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer, self.cfg.dl.eta_min, total_iters=self.cfg.dl.cooldown)
            sequential_scheduler = torch.optim.lr_scheduler.SequentialLR(
                optimizer, [cosine_scheduler, constant_scheduler], milestones=[self.cfg.dl.epochs - self.cfg.dl.cooldown]
            )
            scheduler = sequential_scheduler

        elif self.cfg.dl.scheduler == 'step':
            scheduler = StepLR(optimizer, step_size=self.cfg.dl.scheduler_step, gamma=self.cfg.dl.scheduler_gamma)
        else:
            raise ValueError(f'Unknown scheduler {self.cfg.dl.scheduler}')

        self.scheduler_ = scheduler
        self.optimizer_ = optimizer
        return [optimizer], [scheduler]


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

            horizontal_loss = torch.nn.functional.cross_entropy(contrast,
                                                                torch.arange(len(contrast), device=contrast.device))
            vertical_loss = torch.nn.functional.cross_entropy(contrast.T,
                                                              torch.arange(len(contrast), device=contrast.device))
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
        self.means_cache.scatter_reduce_(0, self.targets_cache[:, None].expand(self.embeddings_cache.shape),
                                         self.embeddings_cache, reduce='sum')

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
