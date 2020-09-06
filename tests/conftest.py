import pytest
import logging

import torch
from torch import nn
import torch.nn.functional as F

from argus import Model
from argus.engine import Engine
from argus.utils import Identity
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
    prediction_transform = {
        'Sigmoid': nn.Sigmoid,
        'Identity': Identity
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


@pytest.fixture(scope='session')
def poly_degree():
    return 4


@pytest.fixture(scope='session')
def poly_coefficients(poly_degree):
    weights = torch.randn(poly_degree, 1) * 5
    bias = torch.randn(1) * 5
    return weights, bias


@pytest.fixture(scope='session')
def poly_function(poly_coefficients):
    weights, bias = poly_coefficients

    def poly(x):
        return x.mm(weights) + bias.item()

    return poly


@pytest.fixture(scope='session')
def make_features_function(poly_degree):
    def make_features(x):
        x = x.unsqueeze(1)
        return torch.cat([x ** i for i in range(1, poly_degree + 1)], 1)

    return make_features


@pytest.fixture(scope='session')
def get_batch_function(make_features_function, poly_function):
    def get_batch(batch_size=32):
        random = torch.randn(batch_size)
        x = make_features_function(random)
        y = poly_function(x)
        return x, y

    return get_batch


@pytest.fixture(scope='function')
def poly_batch(get_batch_function):
    return get_batch_function(batch_size=32)


@pytest.fixture(scope='function')
def linear_argus_model_instance(argus_model_class, poly_degree):
    params = {
        'nn_module': ('LinearNet', {
            'in_features': poly_degree,
            'out_features': 1
        }),
        'optimizer': ('SGD', {'lr': 0.01}),
        'loss': 'SmoothL1Loss',
        'prediction_transform': 'Identity',
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
        'prediction_transform': 'Identity',
        'device': 'cpu'
    }
    return argus_model_class(params)


@pytest.fixture(scope='function',
                params=[[4, 8, 15, 16, 23, 42],
                        list(range(42)),
                        torch.randint(1000, size=(42,)).tolist(),
                        (1e6 * torch.rand(dtype=torch.float32,
                                          size=(42,))).tolist()])
def one_dim_num_sequence(request):
    return request.param


@pytest.fixture(scope='function')
def engine(linear_argus_model_instance):
    return Engine(lambda batch, state: batch,
                  model=linear_argus_model_instance,
                  logger=linear_argus_model_instance.logger)


@pytest.fixture(scope='function')
def state(engine):
    return engine.state

