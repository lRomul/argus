import collections
import warnings
import logging
import typing

import torch
from torch import nn
from torch.optim.optimizer import Optimizer
from torch.nn.parallel import DataParallel, DistributedDataParallel

from argus.utils import device_to_str
from argus.loss import pytorch_losses
from argus.optimizer import pytorch_optimizers


ATTRS_BUILD_ORDER = ('nn_module', 'optimizer', 'loss', 'device', 'prediction_transform')
TRAIN_ATTRS = {'nn_module', 'optimizer', 'loss', 'device', 'prediction_transform'}
PREDICT_ATTRS = {'nn_module', 'device', 'prediction_transform'}
ALL_ATTRS = TRAIN_ATTRS | PREDICT_ATTRS
MODEL_REGISTRY = {}


def cast_optimizer(optimizer):
    if callable(optimizer):
        return optimizer
    elif isinstance(optimizer, type) and hasattr(optimizer, 'step'):
        return optimizer
    elif isinstance(optimizer, str) and optimizer in pytorch_optimizers:
        optimizer = getattr(torch.optim, optimizer)
        return optimizer
    raise TypeError(f"Incorrect type for optimizer {type(optimizer)}")


def cast_nn_module(nn_module):
    if callable(nn_module):
        return nn_module
    raise TypeError(f"Incorrect type for nn_module {type(nn_module)}")


def cast_loss(loss):
    if callable(loss):
        return loss
    elif isinstance(loss, str) and loss in pytorch_losses:
        loss = getattr(nn.modules.loss, loss)
        return loss
    raise TypeError(f"Incorrect type for loss {type(loss)}")


def cast_prediction_transform(transform):
    if callable(transform):
        return transform
    raise TypeError(f"Incorrect type for prediction_transform: {type(transform)}")


def cast_device(device):
    if isinstance(device, torch.device):
        return device
    elif isinstance(device, (list, tuple)):
        if len(device) == 1:
            return torch.device(device[0])
        elif len(device) == 0:
            raise ValueError("Empty list of devices")
        else:
            return [torch.device(d) for d in device]
    else:
        return torch.device(device)


class Identity:
    def __call__(self, x):
        return x

    def __repr__(self):
        return "Identity()"


DEFAULT_ATTRIBUTE_VALUES = {
    'nn_module': None,
    'optimizer': pytorch_optimizers,
    'loss': pytorch_losses,
    'device': torch.device('cpu'),
    'prediction_transform': Identity
}


class ModelMeta(type):
    def __new__(mcs, name, bases, attrs, *args, **kwargs):
        meta_attrs = {"_meta": dict()}
        for key, value in attrs.items():
            if key in ALL_ATTRS:
                meta_attrs['_meta'][key] = value
            else:
                meta_attrs[key] = value

        for attr_name in ALL_ATTRS:
            if attr_name not in meta_attrs['_meta']:
                meta_attrs['_meta'][attr_name] = DEFAULT_ATTRIBUTE_VALUES[attr_name]
            meta_attrs[attr_name] = None

        new_class = super().__new__(mcs, name, bases, meta_attrs)
        if name in MODEL_REGISTRY:
            current_class = f"<class '{attrs['__module__']}.{attrs['__qualname__']}'>"
            warnings.warn(f"{current_class} redefined '{name}' "
                          f"that was already registered by {MODEL_REGISTRY[name]}")
        MODEL_REGISTRY[name] = new_class
        return new_class


def choose_attribute_from_dict(attribute_meta, attribute_params):
    if isinstance(attribute_meta, collections.Mapping):
        if isinstance(attribute_params, (list, tuple)) and len(attribute_params) == 2:
            name, params = attribute_params
            if name not in attribute_meta:
                raise ValueError(f"Attribute '{name}' there is not in "
                                 f"attribute params {attribute_meta}.")
            if not isinstance(params, collections.Mapping):
                raise TypeError(f"Attribute params should be a dictionary, "
                                f"not {type(params)}.")
        elif isinstance(attribute_params, str):
            name, params = attribute_params, dict()
        else:
            raise TypeError(f"Incorrect attribute params {attribute_params}")
        attribute = attribute_meta[name]
    else:
        attribute = attribute_meta
        params = attribute_params

    return attribute, params


class BuildModel(metaclass=ModelMeta):
    nn_module: nn.Module
    optimizer: Optimizer
    loss: nn.Module
    device: torch.device
    prediction_transform: typing.Callable

    def __init__(self, params: dict, build_order: list = ATTRS_BUILD_ORDER):
        self.params = params.copy()
        self.logger = self.build_logger()

        for attr_name in build_order:
            # Use _meta that was constructed in ModelMeta
            attribute_meta = self._meta[attr_name]
            attribute_params = self.params.get(attr_name, dict())
            attr_build_func = getattr(self, f"build_{attr_name}")
            attribute = attr_build_func(attribute_meta, attribute_params)
            setattr(self, attr_name, attribute)

        self.set_device(self.device)

    def build_nn_module(self, nn_module_meta, nn_module_params):
        if nn_module_meta is None:
            raise ValueError("nn_module is required attribute for argus.Model")

        nn_module, nn_module_params = choose_attribute_from_dict(nn_module_meta,
                                                                 nn_module_params)
        nn_module = cast_nn_module(nn_module)
        nn_module = nn_module(**nn_module_params)
        return nn_module

    def build_optimizer(self, optimizer_meta, optim_params):
        optimizer, optim_params = choose_attribute_from_dict(optimizer_meta,
                                                             optim_params)
        optimizer = cast_optimizer(optimizer)
        grad_params = (param for param in self.nn_module.parameters()
                       if param.requires_grad)
        optimizer = optimizer(params=grad_params, **optim_params)
        return optimizer

    def build_loss(self, loss_meta, loss_params):
        loss, loss_params = choose_attribute_from_dict(loss_meta,
                                                       loss_params)
        loss = cast_loss(loss)
        loss = loss(**loss_params)
        return loss

    def build_prediction_transform(self, transform_meta, transform_params):
        transform, transform_params = choose_attribute_from_dict(transform_meta,
                                                                 transform_params)
        transform = cast_prediction_transform(transform)
        prediction_transform = transform(**transform_params)
        return prediction_transform

    def build_device(self, device_meta, device_param):
        if device_param:
            device = device_param
        else:
            device = device_meta
        return cast_device(device)

    def build_logger(self):
        logging.basicConfig(
            format='%(asctime)s %(levelname)s %(message)s',
            level=logging.getLevelName(logging.INFO),
            handlers=[logging.StreamHandler()],
        )
        logger = logging.getLogger(__name__)
        return logger

    def get_nn_module(self):
        if isinstance(self.nn_module, (DataParallel, DistributedDataParallel)):
            return self.nn_module.module
        else:
            return self.nn_module

    def set_device(self, device):
        device = cast_device(device)
        str_device = device_to_str(device)
        nn_module = self.get_nn_module()

        if isinstance(device, (list, tuple)):
            device_ids = []
            for dev in device:
                if dev.type != 'cuda':
                    raise ValueError("Non cuda device in list of devices")
                if dev.index is None:
                    raise ValueError("Cuda device without index in list of devices")
                device_ids.append(dev.index)
            if len(device_ids) != len(set(device_ids)):
                raise ValueError("Cuda device indices must be unique")
            nn_module = DataParallel(nn_module, device_ids=device_ids)
            device = device[0]

        self.params['device'] = str_device
        self.device = device
        self.nn_module = nn_module.to(self.device)
        if self.loss is not None:
            self.loss = self.loss.to(self.device)

    def _check_attributes(self, attrs):
        for attr_name in attrs:
            attr_value = getattr(self, attr_name, None)
            if attr_value is None:
                return False
        return True

    def train_ready(self):
        return self._check_attributes(TRAIN_ATTRS)

    def predict_ready(self):
        return self._check_attributes(PREDICT_ATTRS)

    def __repr__(self):
        return str(self.__dict__)
