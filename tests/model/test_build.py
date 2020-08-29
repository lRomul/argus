import pytest

import torch
from torch import nn

from argus import Model
from argus.utils import Identity
from argus.model.build import (
    cast_nn_module,
    cast_optimizer,
    cast_loss,
    cast_device,
    cast_prediction_transform
)


class TestBuild:
    def test_simple_build(self, linear_net_class):
        class BuildModel1(Model):
            nn_module = linear_net_class
            optimizer = torch.optim.SGD
            loss = torch.nn.MSELoss
            device = torch.device('cpu')

        params = {
            'nn_module': {'in_features': 10, 'out_features': 2},
            'optimizer': {'lr': 0.01}
        }

        model = BuildModel1(params)
        assert isinstance(model.nn_module, linear_net_class)
        assert model.nn_module.fc.in_features == 10
        assert model.nn_module.fc.out_features == 2
        assert model.device == torch.device('cpu')
        assert isinstance(model.optimizer, torch.optim.SGD)
        assert model.get_lr() == 0.01
        assert isinstance(model.loss, torch.nn.MSELoss)
        assert isinstance(model.prediction_transform, Identity)

    def test_default_and_string_build(self, linear_net_class):
        class BuildModel2(Model):
            nn_module = linear_net_class
            optimizer = 'Adam'
            loss = 'NLLLoss'

        model = BuildModel2(dict())
        assert isinstance(model.nn_module, linear_net_class)
        assert model.nn_module.fc.in_features == 1
        assert model.nn_module.fc.out_features == 1
        assert model.device == torch.device('cpu')
        assert isinstance(model.optimizer, torch.optim.Adam)
        assert model.get_lr() == 0.001
        assert isinstance(model.loss, torch.nn.NLLLoss)

    def test_default_from_params_build(self, linear_net_class):
        class TestModel3(Model):
            nn_module = linear_net_class

        params = {
            'nn_module': {'in_features': 16, 'out_features': 4},
            'optimizer': ('AdamW', {'lr': 0.1}),
            'loss': 'BCEWithLogitsLoss',
            'device': 'cpu'
        }

        model = TestModel3(params)
        assert isinstance(model.nn_module, linear_net_class)
        assert model.nn_module.fc.in_features == 16
        assert model.nn_module.fc.out_features == 4
        assert model.device == torch.device('cpu')
        assert isinstance(model.optimizer, torch.optim.AdamW)
        assert model.get_lr() == 0.1
        assert isinstance(model.loss, torch.nn.BCEWithLogitsLoss)

    def test_dict_flexibility_build(self, linear_net_class, vision_net_class):
        class BuildModel4(Model):
            nn_module = {
                'linear': linear_net_class,
                'vision': vision_net_class
            }
            optimizer = {
                'sgd': 'SGD',
                'adam': torch.optim.Adam
            }
            loss = {
                'CrossEntropyLoss': nn.CrossEntropyLoss,
                'nll': 'NLLLoss'
            }
            prediction_transform = nn.Sigmoid

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

        model = BuildModel4(params)
        assert isinstance(model.nn_module, vision_net_class)
        assert model.nn_module.conv.in_channels == 3
        assert model.nn_module.fc.out_features == 1
        assert model.device == torch.device('cpu')
        assert isinstance(model.optimizer, torch.optim.Adam)
        assert model.get_lr() == 0.01
        assert isinstance(model.loss, torch.nn.CrossEntropyLoss)
        assert isinstance(model.prediction_transform, nn.Sigmoid)

    def test_factory_function_build(self, linear_net_class, vision_net_class):
        def nn_module_factory(module_name, **kwargs):
            if module_name == 'linear':
                return linear_net_class(**kwargs)
            elif module_name == 'vision':
                return vision_net_class(**kwargs)
            return None

        class BuildModel5(Model):
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

        model = BuildModel5(params)
        assert isinstance(model.nn_module, vision_net_class)

        model = BuildModel5({'nn_module': {'module_name': 'linear'}})
        assert isinstance(model.nn_module, linear_net_class)

    def test_none_nn_module_build(self):
        class BuildModel6(Model):
            nn_module = None

        with pytest.raises(ValueError):
            BuildModel6(dict())


class TestCastFunction:
    def test_cast_nn_module(self, linear_net_class):
        assert cast_nn_module(linear_net_class) is linear_net_class
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


class TestBuildModelMethod:
    def test_get_nn_module(self, linear_argus_model_instance):
        model = linear_argus_model_instance
        nn_module = model.nn_module
        assert isinstance(model.nn_module, nn.Module)
        model.nn_module = nn.parallel.DataParallel(model.nn_module)
        assert isinstance(model.get_nn_module(), nn.Module)
        assert isinstance(model.nn_module, nn.parallel.DataParallel)
        assert not isinstance(model.get_nn_module(), nn.parallel.DataParallel)
        assert model.get_nn_module() is nn_module
