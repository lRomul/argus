import pytest

import torch
from torch import nn
import torch.nn.functional as F

from argus import Model
from argus.utils import Identity
from argus.model.build import (
    cast_nn_module,
    cast_optimizer,
    cast_loss,
    cast_device,
    cast_prediction_transform
)


class LinearNet(nn.Module):
    def __init__(self, in_features=1, out_features=1):
        super().__init__()
        self.fc = nn.Linear(in_features, out_features)

    def forward(self, x):
        x = self.fc(x)
        return x


class SimpleVisionNet(nn.Module):
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


class TestBuild:
    def test_simple_build(self):
        class TestModel1(Model):
            nn_module = LinearNet
            optimizer = torch.optim.SGD
            loss = torch.nn.MSELoss
            device = torch.device('cpu')

        params = {
            'nn_module': {'in_features': 10, 'out_features': 2},
            'optimizer': {'lr': 0.01}
        }

        model = TestModel1(params)
        assert isinstance(model.nn_module, LinearNet)
        assert model.nn_module.fc.in_features == 10
        assert model.nn_module.fc.out_features == 2
        assert model.device == torch.device('cpu')
        assert isinstance(model.optimizer, torch.optim.SGD)
        assert model.get_lr() == 0.01
        assert isinstance(model.loss, torch.nn.MSELoss)
        assert isinstance(model.prediction_transform, Identity)

    def test_default_and_string_build(self):
        class TestModel2(Model):
            nn_module = LinearNet
            optimizer = 'Adam'
            loss = 'NLLLoss'

        model = TestModel2(dict())
        assert isinstance(model.nn_module, LinearNet)
        assert model.nn_module.fc.in_features == 1
        assert model.nn_module.fc.out_features == 1
        assert model.device == torch.device('cpu')
        assert isinstance(model.optimizer, torch.optim.Adam)
        assert model.get_lr() == 0.001
        assert isinstance(model.loss, torch.nn.NLLLoss)

    def test_default_from_params_build(self):
        class TestModel3(Model):
            nn_module = LinearNet

        params = {
            'nn_module': {'in_features': 16, 'out_features': 4},
            'optimizer': ('AdamW', {'lr': 0.1}),
            'loss': 'BCEWithLogitsLoss',
            'device': 'cpu'
        }

        model = TestModel3(params)
        assert isinstance(model.nn_module, LinearNet)
        assert model.nn_module.fc.in_features == 16
        assert model.nn_module.fc.out_features == 4
        assert model.device == torch.device('cpu')
        assert isinstance(model.optimizer, torch.optim.AdamW)
        assert model.get_lr() == 0.1
        assert isinstance(model.loss, torch.nn.BCEWithLogitsLoss)

    def test_dict_flexibility_build(self):
        class TestModel4(Model):
            nn_module = {
                'linear': LinearNet,
                'vision': SimpleVisionNet
            }
            optimizer = {
                'sgd': 'SGD',
                'adam': torch.optim.Adam
            }
            loss = {
                'CrossEntropyLoss': nn.CrossEntropyLoss,
                'nll': 'NLLLoss'
            }

        params = {
            'nn_module': ('vision', {
                'n_channels': 3,
                'n_classes': 1,
                'p_dropout': 0.2
            }),
            'optimizer': ('adam', {'lr': 0.01}),
            'loss': 'CrossEntropyLoss',
            'device': 'cpu'
        }

        model = TestModel4(params)
        assert isinstance(model.nn_module, SimpleVisionNet)
        assert model.nn_module.conv.in_channels == 3
        assert model.nn_module.fc.out_features == 1
        assert model.device == torch.device('cpu')
        assert isinstance(model.optimizer, torch.optim.Adam)
        assert model.get_lr() == 0.01
        assert isinstance(model.loss, torch.nn.CrossEntropyLoss)

    def test_factory_function_build(self):
        def nn_module_factory(module_name, **kwargs):
            if module_name == 'linear':
                return LinearNet(**kwargs)
            elif module_name == 'vision':
                return SimpleVisionNet(**kwargs)
            return None

        class TestModel5(Model):
            nn_module = nn_module_factory
            optimizer = torch.optim.Adam
            loss = torch.nn.MSELoss

        params = {
            'nn_module': {
                'module_name': 'vision',
                'n_channels': 3,
                'n_classes': 1
            }
        }

        model = TestModel5(params)
        assert isinstance(model.nn_module, SimpleVisionNet)

        model = TestModel5({'nn_module': {'module_name': 'linear'}})
        assert isinstance(model.nn_module, LinearNet)


class TestCastFunction:
    def test_cast_nn_module(self):
        assert cast_nn_module(LinearNet) is LinearNet
        with pytest.raises(TypeError):
            cast_nn_module('qwerty')

    def test_cast_optimizer(self, optimizer_class):
        assert cast_optimizer(optimizer_class) is optimizer_class
        assert cast_optimizer(optimizer_class.__name__) is optimizer_class
        with pytest.raises(TypeError):
            cast_optimizer('qwerty')
        with pytest.raises(TypeError):
            cast_optimizer(None)

    def test_cast_loss(self, loss_class):
        assert cast_loss(loss_class) is loss_class
        assert cast_loss(loss_class.__name__) is loss_class
        with pytest.raises(TypeError):
            cast_loss('qwerty')
        with pytest.raises(TypeError):
            cast_loss(None)

    def test_cast_prediction_transform(self):
        assert cast_prediction_transform(Identity) is Identity
        with pytest.raises(TypeError):
            cast_prediction_transform('qwerty')
        with pytest.raises(TypeError):
            cast_prediction_transform(None)

    def test_cast_device(self):
        assert cast_device('cpu') == torch.device('cpu')
        assert cast_device('cuda') == torch.device('cuda')
        assert cast_device(torch.device('cpu')) == torch.device('cpu')
        devices = [torch.device('cuda:0'), torch.device('cuda:1')]
        assert cast_device(['cuda:0', 'cuda:1']) == devices
        assert cast_device(devices) == devices
        assert cast_device(['cuda:0']) == torch.device('cuda:0')
        with pytest.raises(ValueError):
            cast_device([])
