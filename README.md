```
___/\\\\\\\\\_______/\\\\\\\\\_________/\\\\\\\\\\\\__/\\\________/\\\_____/\\\\\\\\\\\___        
__/\\\\\\\\\\\\\___/\\\///////\\\_____/\\\//////////__\/\\\_______\/\\\___/\\\/////////\\\_       
__/\\\/////////\\\_\/\\\_____\/\\\____/\\\_____________\/\\\_______\/\\\__\//\\\______\///__      
 _\/\\\_______\/\\\_\/\\\\\\\\\\\/____\/\\\____/\\\\\\\_\/\\\_______\/\\\___\////\\\_________     
  _\/\\\\\\\\\\\\\\\_\/\\\//////\\\____\/\\\___\/////\\\_\/\\\_______\/\\\______\////\\\______    
   _\/\\\/////////\\\_\/\\\____\//\\\___\/\\\_______\/\\\_\/\\\_______\/\\\_________\////\\\___   
    _\/\\\_______\/\\\_\/\\\_____\//\\\__\/\\\_______\/\\\_\//\\\______/\\\___/\\\______\//\\\__  
     _\/\\\_______\/\\\_\/\\\______\//\\\_\//\\\\\\\\\\\\/___\///\\\\\\\\\/___\///\\\\\\\\\\\/__ 
      _\///________\///__\///________\///___\////////////_______\/////////_______\///////////___
```

[![PyPI version](https://badge.fury.io/py/pytorch-argus.svg)](https://badge.fury.io/py/pytorch-argus)
[![Documentation Status](https://readthedocs.org/projects/pytorch-argus/badge/?version=latest)](https://pytorch-argus.readthedocs.io/en/latest/?badge=latest)
![Test](https://github.com/lRomul/argus/workflows/Test/badge.svg)
[![CodeFactor](https://www.codefactor.io/repository/github/lromul/argus/badge)](https://www.codefactor.io/repository/github/lromul/argus)
[![codecov](https://codecov.io/gh/lRomul/argus/branch/master/graph/badge.svg)](https://codecov.io/gh/lRomul/argus)
[![Downloads](https://pepy.tech/badge/pytorch-argus)](https://pepy.tech/project/pytorch-argus)

Argus is a lightweight flexible library for training neural networks in PyTorch.

## Documentation

https://pytorch-argus.readthedocs.io

## Installation

Requirements: 
* torch>=1.1.0

From pip:

```bash
pip install pytorch-argus
```

From source:

```bash
pip install -U git+https://github.com/lRomul/argus.git
```

## Example

Simple image classification example with `create_model` from [pytorch-image-models](https://github.com/rwightman/pytorch-image-models):

```python
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor, Normalize

import timm

from argus import Model
from argus.callbacks import MonitorCheckpoint, EarlyStopping, ReduceLROnPlateau


def get_data_loaders(batch_size):
    data_transform = Compose([ToTensor(), Normalize((0.1307,), (0.3081,))])
    train_mnist_dataset = MNIST(download=True, root="mnist_data",
                                transform=data_transform, train=True)
    val_mnist_dataset = MNIST(download=False, root="mnist_data",
                              transform=data_transform, train=False)
    train_loader = DataLoader(train_mnist_dataset,
                              batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_mnist_dataset,
                            batch_size=batch_size * 2, shuffle=False)
    return train_loader, val_loader


class TimmModel(Model):
    nn_module = timm.create_model


if __name__ == "__main__":
    train_loader, val_loader = get_data_loaders(batch_size=256)

    params = {
        'nn_module': {
            'model_name': 'tf_efficientnet_b0_ns',
            'pretrained': False,
            'num_classes': 10,
            'in_chans': 1,
            'drop_rate': 0.2,
            'drop_path_rate': 0.2
        },
        'optimizer': ('Adam', {'lr': 0.01}),
        'loss': 'CrossEntropyLoss',
        'device': 'cuda'
    }

    model = TimmModel(params)

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

More examples you can find [here](https://pytorch-argus.readthedocs.io/en/latest/examples.html).
