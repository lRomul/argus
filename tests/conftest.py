import pytest

from torch import nn
import torch.nn.functional as F

from argus import Model
from argus.loss import pytorch_losses
from argus.optimizer import pytorch_optimizers


@pytest.fixture(scope='session', params=pytorch_losses.values())
def loss_class(request):
    return request.param


@pytest.fixture(scope='session', params=pytorch_optimizers.values())
def optimizer_class(request):
    return request.param


class LinearNet(nn.Module):
    def __init__(self, in_features=1, out_features=1):
        super().__init__()
        self.fc = nn.Linear(in_features, out_features)

    def forward(self, x):
        x = self.fc(x)
        return x


class VisionNet(nn.Module):
    def __init__(self, n_channels, n_classes, p_dropout=0.5):
        super().__init__()
        self.conv = nn.Conv2d(n_channels, 8, kernel_size=3)
        self.p_dropout = float(p_dropout)
        self.act = nn.ReLU()
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(8, n_classes)

    def forward(self, x):
        x = self.conv(x)
        x = self.act(x)
        x = self.gap(x)
        x = x.flatten(1)
        if self.p_dropout:
            x = F.dropout(x, p=self.p_dropout, training=self.training)
        x = self.fc(x)
        return x


class ArgusTestModel(Model):
    nn_module = {
        'LinearNet': LinearNet,
        'VisionNet': VisionNet
    }


@pytest.fixture(scope='session')
def linear_net_class():
    return LinearNet


@pytest.fixture(scope='session')
def vision_net_class():
    return VisionNet


@pytest.fixture(scope='session')
def argus_model_class():
    return ArgusTestModel


@pytest.fixture(scope='function')
def linear_argus_model_instance(argus_model_class):
    params = {
        'nn_module': ('LinearNet', {
            'in_features': 16,
            'out_features': 1
        }),
        'optimizer': ('Adam', {'lr': 0.001}),
        'loss': 'MSELoss',
        'device': 'cpu'
    }
    return argus_model_class(params)


@pytest.fixture(scope='function')
def vision_argus_model_instance(argus_model_class):
    params = {
        'nn_module': ('VisionNet', {
            'n_channels': 3,
            'n_classes': 1,
            'p_dropout': 0.2
        }),
        'optimizer': ('Adam', {'lr': 0.001}),
        'loss': 'CrossEntropyLoss',
        'device': 'cpu'
    }
    return argus_model_class(params)
