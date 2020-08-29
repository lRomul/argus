import pytest

import torch

from argus import Model
from argus.model.build import (
    choose_attribute_from_dict,
    cast_optimizer,
    cast_nn_module
)


def test_custom_build_methods(vision_net_class):
    class CustomBuildMethodModel(Model):
        nn_module = vision_net_class

        def build_nn_module(self, nn_module_meta, nn_module_params):
            if nn_module_meta is None:
                raise ValueError("nn_module is required attribute for argus.Model")

            nn_module, nn_module_params = choose_attribute_from_dict(nn_module_meta,
                                                                     nn_module_params)
            nn_module = cast_nn_module(nn_module)
            nn_module = nn_module(**nn_module_params)

            # Replace last fully connected layer
            num_classes = self.params['num_classes']
            in_features = nn_module.fc.in_features
            nn_module.fc = torch.nn.Linear(in_features=in_features,
                                           out_features=num_classes)
            return nn_module

        def build_optimizer(self, optimizer_meta, optim_params):
            optimizer, optim_params = choose_attribute_from_dict(optimizer_meta,
                                                                 optim_params)
            optimizer = cast_optimizer(optimizer)

            # Set small LR for pretrained layers
            pretrain_modules = [
                self.nn_module.conv
            ]
            pretrain_params = []
            for pretrain_module in pretrain_modules:
                pretrain_params += pretrain_module.parameters()
            pretrain_ids = list(map(id, pretrain_params))
            other_params = filter(lambda p: id(p) not in pretrain_ids,
                                  self.nn_module.parameters())
            grad_params = [
                {"params": pretrain_params, "lr": optim_params['lr'] * 0.01},
                {"params": other_params, "lr": optim_params['lr']}
            ]
            del optim_params['lr']

            optimizer = optimizer(params=grad_params, **optim_params)
            return optimizer

    params = {
        'nn_module': {
            'n_channels': 3,
            'n_classes': 1,
            'p_dropout': 0.2
        },
        'optimizer': ('Adam', {'lr': 0.001}),
        'loss': 'CrossEntropyLoss',
        'device': 'cpu',
        'num_classes': 10
    }

    model = CustomBuildMethodModel(params)

    assert model.get_lr() == [1e-05, 0.001]
    assert model.nn_module.fc.out_features == 10

    model.set_lr([0.0, 0.1])
    assert model.get_lr() == [0.0, 0.1]
    model.set_lr(0.3)
    assert model.get_lr() == [0.3, 0.3]

    with pytest.raises(ValueError):
        model.set_lr([0.0, 0.1, 0.3])

    with pytest.raises(ValueError):
        model.set_lr(None)
