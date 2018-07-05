import torch
from torch import nn
from torch import optim
import collections
import types

from argus.utils import default, ALL_ATTRS
from argus.loss import pytorch_losses
from argus.optimizer import pytorch_optimizers


def cast_optimizer(optimizer):
    if isinstance(optimizer, types.FunctionType):
        return optimizer
    elif isinstance(optimizer, type) and hasattr(optimizer, 'step'):
        return optimizer
    elif isinstance(optimizer, str) and optimizer in pytorch_optimizers:
        optimizer = getattr(optim, optimizer)
        return optimizer
    raise TypeError


def cast_nn_module(nn_module):
    if isinstance(nn_module, types.FunctionType):
        return nn_module
    elif isinstance(nn_module, type):
        if issubclass(nn_module, nn.Module):
            return nn_module
    raise TypeError


def cast_loss(loss):
    if isinstance(loss, types.FunctionType):
        return loss
    elif isinstance(loss, type) and callable(loss):
        return loss
    elif isinstance(loss, str) and loss in pytorch_losses:
        loss = getattr(nn.modules.loss, loss)
        return loss
    raise TypeError


def cast_predict_transform(transform):
    if isinstance(transform, types.FunctionType):
        return transform
    elif isinstance(transform, type) and callable(transform):
        return transform
    raise TypeError


def cast_device(device):
    return torch.device(device)


ATTRIBUTE_CASTS = {
    'nn_module': cast_nn_module,
    'optimizer': cast_optimizer,
    'loss': cast_loss,
    'device': cast_device,
    'predict_transform': cast_predict_transform
}

DEFAULT_ATTRIBUTE_VALUES = {
    'nn_module': default,
    'optimizer': pytorch_optimizers,
    'loss': pytorch_losses,
    'device': torch.device('cpu'),
    'predict_transform': default
}


class ModelMeta(type):
    def __new__(mcs, name, bases, attrs, *args, **kwargs):
        cast_attrs = {"_meta": dict()}
        for key, value in attrs.items():
            if key in ATTRIBUTE_CASTS:
                if isinstance(value, collections.Mapping):
                    value = {k: ATTRIBUTE_CASTS[key](v) for k, v in value.items()}
                else:
                    value = ATTRIBUTE_CASTS[key](value)
                cast_attrs['_meta'][key] = value
            else:
                cast_attrs[key] = value

        for attr_name in ALL_ATTRS:
            if attr_name not in cast_attrs['_meta']:
                cast_attrs['_meta'][attr_name] = DEFAULT_ATTRIBUTE_VALUES[attr_name]
            cast_attrs[attr_name] = default

        new_class = super().__new__(mcs, name, bases, cast_attrs)
        return new_class


class BuildModel(metaclass=ModelMeta):
    def __init__(self, params):
        self.params = params
        self._build_nn_module(params)
        self._build_optimizer(params)
        self._build_loss(params)
        self._build_device(params)
        self._build_predict_transform(params)
        self.set_device(self.device)

    def _build_nn_module(self, params):
        nn_module_meta = self._meta['nn_module']
        if nn_module_meta is default:
            raise ValueError

        if isinstance(nn_module_meta, collections.Mapping):
            nn_module_info = params['nn_module']
            if isinstance(nn_module_info, (list, tuple)):
                nn_name, nn_params = nn_module_info
            elif isinstance(nn_module_info, str):
                nn_name, nn_params = nn_module_info, dict()
            else:
                raise ValueError
            nn_module = nn_module_meta[nn_name](**nn_params)
        else:
            nn_params = params.get('nn_module', dict())
            nn_module = nn_module_meta(**nn_params)

        self.nn_module = nn_module

    def _build_optimizer(self, params):
        assert 'params' not in params
        optimizer_meta = self._meta['optimizer']
        if self.nn_module is not default:
            if isinstance(optimizer_meta, collections.Mapping):
                if 'optimizer' not in params:
                    return
                optim_info = params['optimizer']
                if isinstance(optim_info, (list, tuple)) and len(optim_info) == 2:
                    optim_name, optim_params = optim_info
                elif isinstance(optim_info, str):
                    optim_name, optim_params = optim_info, dict()
                else:
                    raise ValueError
                optim_params['params'] = self.nn_module.parameters()
                optimizer = optimizer_meta[optim_name](**optim_params)
            else:
                optim_params = params.get('optimizer', dict())
                optim_params['params'] = self.nn_module.parameters()
                optimizer = optimizer_meta(**optim_params)

            self.optimizer = optimizer
        else:
            raise ValueError("Can't assign optimizer without nn_module")

    def _build_loss(self, params):
        loss_meta = self._meta['loss']
        if isinstance(loss_meta, collections.Mapping):
            if 'loss' not in params:
                return
            loss_info = params['loss']
            if isinstance(loss_info, (list, tuple)) and len(loss_info) == 2:
                loss_name, loss_params = loss_info
            elif isinstance(loss_info, str):
                loss_name, loss_params = loss_info, dict()
            else:
                raise ValueError
            loss = loss_meta[loss_name](**loss_params)
        else:
            loss_params = params.get('loss', dict())
            loss = loss_meta(**loss_params)

        self.loss = loss

    def set_device(self, device):
        self.device = torch.device(device)
        self.nn_module = self.nn_module.to(self.device)
        if self.loss is not default:
            self.loss = self.loss.to(self.device)

    def _build_device(self, params):
        if 'device' in params:
            self.device = torch.device(params['device'])
        else:
            self.device = self._meta['device']

    def _build_predict_transform(self, params):
        transform_meta = self._meta['predict_transform']
        if transform_meta is default:
            print("LUL")
            self.predict_transform = lambda x: x
            return

        if isinstance(transform_meta, collections.Mapping):
            if 'predict_transform' not in params:
                return
            trns_info = params['predict_transform']
            if isinstance(trns_info, (list, tuple)) and len(trns_info) == 2:
                trns_name, trns_params = trns_info
            elif isinstance(trns_info, str):
                trns_name, trns_params = trns_info, dict()
            else:
                raise ValueError
            predict_transform = transform_meta[trns_name](**trns_params)
        else:
            trns_params = params.get('predict_transform', dict())
            predict_transform = transform_meta(**trns_params)

        self.predict_transform = predict_transform
