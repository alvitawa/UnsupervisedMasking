from dataclasses import dataclass

import torch

import torchvision
from PIL import Image

from volt.dataset import SliceableDataset
from volt.modules.deep_learning import DeepLearningModule
from volt import config, util


@dataclass
class ClassifierConfig:
    loss: str = 'cross_entropy'


config.register('classifier', ClassifierConfig)


class ClassDataset(SliceableDataset):
    def __init__(self, dataset, transform, inverse_transform, classes=None, labels=None, label_names=None):
        super().__init__()
        self.dataset = dataset
        self.transform = transform
        self.inverse_transform = inverse_transform

        if classes is None:
            assert labels is not None, 'Must provide either classes or labels and label_names'
            assert label_names is not None, 'Must provide either classes or labels and label_names'

            self.labels = labels
            self.label_names = label_names

            self.classes = {i: name for i, name in zip(self.labels, self.label_names)}
        else:
            assert self.classes is not None, 'Must provide either classes or labels and label_names'
            self.classes = classes

            items = sorted(self.classes.items(), key=lambda x: x[0])
            self.labels = [i for i, _ in items]
            self.label_names = [name for _, name in items]

    def get_item(self, index):
        image, label = self.dataset[index]
        image_transformed = self.transform(image)
        return image_transformed, label

    def __len__(self):
        return len(self.dataset)

    def untransform(self, x):
        return self.inverse_transform(x)


class ClassifierModule(DeepLearningModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # assert all(isinstance(dataset, ClassDataset) for dataset in self.dataset.values())

        if self.cfg.cls.loss == 'cross_entropy':
            self.loss = torch.nn.CrossEntropyLoss()

    def process_samples(self, samples, name, namespace):
        if isinstance(samples, torch.Tensor):
            samples = [samples]

        for key in samples if isinstance(samples, dict) else range(len(samples)):
            if isinstance(samples[key], torch.Tensor) and samples[key].dim() == 0:
                continue
            for i, value in enumerate(samples[key]):
                if isinstance(value, torch.Tensor):
                    if len(value.shape) == 3 and value.shape[0] == 3:
                        value = self.dataset[namespace].untransform(value)
                    elif len(value.shape) == 3 and value.shape[0] == 1:
                        value = self.dataset[namespace].untransform(value)
                    elif len(value.shape) == 2:
                        value = util.tensor_to_pil(value.unsqueeze(0))
                    elif len(value.shape) == 1 and value.shape[0] == 1:
                        value = value.detach().cpu().numpy()[0]
                    elif len(value.shape) == 0:
                        value = value.item()
                    else:
                        continue

                self.logger_.experiment[f'training/{namespace}/{name}/sample_{i}/key_{key}'].log(value)

    def analyze(self, outputs, sample_inputs, sample_outputs, namespace):
        super().analyze(outputs, sample_inputs, sample_outputs, namespace)

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

        if sample_inputs is not None and sample_outputs is not None and len(sample_inputs) > 0:
            self.process_samples(sample_inputs, 'sample_inputs', namespace)
            self.process_samples(sample_outputs, 'sample_outputs', namespace)

    def training_step(self, batch, batch_idx, namespace='train'):
        data, target = batch
        output = self.forward(data)
        loss = self.loss(output, target)

        self.log(f'{namespace}/loss', loss)
        return {'loss': loss, 'output': torch.argmax(output, dim=1), 'target': target}


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