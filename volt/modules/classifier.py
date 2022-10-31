from dataclasses import dataclass

import torch

import torch_scatter
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
        hits = (outputs['output'] == outputs['target']).float()

        accuracy = hits.mean()

        labels = self.dataset[namespace].labels
        dim_size = max(labels) + 1

        precision = torch_scatter.scatter_mean(hits, outputs['target'], dim=0, dim_size=dim_size)
        recall = torch_scatter.scatter_mean(hits, outputs['output'], dim=0, dim_size=dim_size)

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
