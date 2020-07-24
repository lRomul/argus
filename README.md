# Argus 

[![PyPI version](https://badge.fury.io/py/pytorch-argus.svg)](https://badge.fury.io/py/pytorch-argus)
[![Documentation Status](https://readthedocs.org/projects/pytorch-argus/badge/?version=latest)](https://pytorch-argus.readthedocs.io/en/latest/?badge=latest)
[![CodeFactor](https://www.codefactor.io/repository/github/lromul/argus/badge)](https://www.codefactor.io/repository/github/lromul/argus)
[![Downloads](https://pepy.tech/badge/pytorch-argus)](https://pepy.tech/project/pytorch-argus)


Argus is easy-to-use flexible library for training neural networks in PyTorch.

## Documentation

https://pytorch-argus.readthedocs.io

## Installation

From pip:

```bash
pip install pytorch-argus
```

From source:

```bash
git clone https://github.com/lRomul/argus
cd argus
python setup.py install
```

## Examples

Simple image classification example:

```python
import torch
from torch import nn
import torch.nn.functional as F
from mnist_utils import get_data_loaders

from argus import Model
from argus.callbacks import MonitorCheckpoint, EarlyStopping, ReduceLROnPlateau


class Net(nn.Module):
    def __init__(self, n_classes, p_dropout=0.5):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d(p=p_dropout)
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, n_classes)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return x


class MnistModel(Model):
    nn_module = Net
    optimizer = torch.optim.SGD
    loss = torch.nn.CrossEntropyLoss


if __name__ == "__main__":
    train_loader, val_loader = get_data_loaders()

    params = {
        'nn_module': {'n_classes': 10, 'p_dropout': 0.1},
        'optimizer': {'lr': 0.01},
        'device': 'cpu'
    }

    model = MnistModel(params)

    callbacks = [
        MonitorCheckpoint(dir_path='mnist', monitor='val_accuracy', max_saves=3),
        EarlyStopping(monitor='val_accuracy', patience=9),
        ReduceLROnPlateau(monitor='val_accuracy', factor=0.5, patience=3)
    ]

    model.fit(train_loader,
              val_loader=val_loader,
              num_epochs=50,
              metrics=['accuracy'],
              callbacks=callbacks,
              metrics_on_train=True)
```

Use Argus with `make_model` from [pytorch-cnn-finetune](https://github.com/creafz/pytorch-cnn-finetune).

```python
from cnn_finetune import make_model
from argus import Model

class CnnFinetune(Model):
    nn_module = make_model


params = {
    'nn_module': {
        'model_name': 'resnet18',
        'num_classes': 10,
        'pretrained': False,
        'input_size': (256, 256)
    },
    'optimizer': ('Adam', {'lr': 0.01}),
    'loss': 'CrossEntropyLoss',
    'device': 'cuda'
}

model = CnnFinetune(params)
```

More examples you can find [here](https://pytorch-argus.readthedocs.io/en/latest/examples.html).
