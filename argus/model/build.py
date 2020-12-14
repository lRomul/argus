import sys
import copy
import logging
import warnings
import collections
from typing import Callable, Union, Any, Type, Dict, Tuple, Iterable

import torch
from torch import nn
from torch.optim.optimizer import Optimizer
from torch.nn.parallel import DataParallel, DistributedDataParallel

import argus
from argus import types
from argus.loss import pytorch_losses
from argus.optimizer import pytorch_optimizers
from argus.utils import device_to_str, Identity, check_pickleble, get_device_indices


ATTRS_BUILD_ORDER = ('nn_module', 'optimizer', 'loss', 'device', 'prediction_transform')
TRAIN_ATTRS = {'nn_module', 'optimizer', 'loss', 'device', 'prediction_transform'}
PREDICT_ATTRS = {'nn_module', 'device', 'prediction_transform'}
ALL_ATTRS = TRAIN_ATTRS | PREDICT_ATTRS
MODEL_REGISTRY: Dict[str, Type['argus.model.Model']] = dict()

DEFAULT_ATTRIBUTE_VALUES = {
    'nn_module': None,
    'optimizer': pytorch_optimizers,
    'loss': pytorch_losses,
    'device': torch.device('cpu'),
    'prediction_transform': Identity
}


def cast_optimizer(
        optimizer: Union[Optimizer, Callable, str]
) -> Union[Optimizer, Callable]:
    if callable(optimizer):
        return optimizer
    elif isinstance(optimizer, str) and optimizer in pytorch_optimizers:
        return getattr(torch.optim, optimizer)
    raise TypeError(f"Incorrect type for optimizer {type(optimizer)}")


def cast_nn_module(nn_module: Union[nn.Module, Callable]) -> Union[nn.Module, Callable]:
    if callable(nn_module):
        return nn_module
    raise TypeError(f"Incorrect type for nn_module {type(nn_module)}")


def cast_loss(loss: Union[nn.Module, Callable, str]) -> Union[nn.Module, Callable]:
    if callable(loss):
        return loss
    elif isinstance(loss, str) and loss in pytorch_losses:
        return getattr(nn.modules.loss, loss)
    raise TypeError(f"Incorrect type for loss {type(loss)}")


def cast_prediction_transform(transform: Callable) -> Callable:
    if callable(transform):
        return transform
    raise TypeError(f"Incorrect type for prediction_transform: {type(transform)}")


def cast_device(device: types.InputDevices) -> types.Devices:
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


def choose_attribute_from_dict(
        attribute_meta: types.AttrMeta,
        attribute_params: types.Param
) -> Tuple[Any, collections.abc.Mapping]:
    if isinstance(attribute_meta, collections.abc.Mapping):
        if isinstance(attribute_params, (list, tuple)) and len(attribute_params) == 2:
            name, params = attribute_params
            if name not in attribute_meta:
                raise ValueError(f"Attribute '{name}' there is not in "
                                 f"attribute meta '{attribute_meta}'.")
            if not isinstance(params, collections.abc.Mapping):
                raise TypeError(f"Attribute params '{params}' should be a "
                                f"dictionary, not '{type(params)}'.")
        elif isinstance(attribute_params, str):
            name, params = attribute_params, dict()
        else:
            raise TypeError(f"Incorrect attribute params '{attribute_params}' "
                            f"for attribute meta '{attribute_meta}'. Attribute "
                            f"params should be str or (str, dict).")
        attribute = attribute_meta[name]
    else:
        attribute = attribute_meta
        if not isinstance(attribute_params, collections.abc.Mapping):
            raise TypeError(f"Attribute params '{attribute_params}' should be a "
                            f"dictionary, not '{type(attribute_params)}'.")
        params = attribute_params

    return attribute, params


class BuildModel(metaclass=ModelMeta):
    nn_module: nn.Module
    optimizer: Optimizer
    loss: nn.Module
    device: torch.device
    prediction_transform: Callable
    _meta: Dict[str, types.AttrMeta]

    def __init__(self, params: dict, build_order: Iterable = ATTRS_BUILD_ORDER):
        params = copy.deepcopy(params)
        check_pickleble(params)
        self.params = params
        self.logger = self.build_logger()

        for attr_name in build_order:
            # Use _meta that was constructed in ModelMeta
            attribute_meta = self._meta[attr_name]
            attribute_params = self.params.get(attr_name, dict())
            attribute = None
            if attribute_params is not None:
                attr_build_func = getattr(self, f"build_{attr_name}")
                attribute = attr_build_func(attribute_meta, attribute_params)
            setattr(self, attr_name, attribute)

    def build_nn_module(self,
                        nn_module_meta: types.AttrMeta,
                        nn_module_params: types.Param):
        if nn_module_meta is None:
            raise ValueError("nn_module is required attribute for argus.Model")

        nn_module, params = choose_attribute_from_dict(nn_module_meta,
                                                       nn_module_params)
        nn_module = cast_nn_module(nn_module)
        nn_module = nn_module(**params)
        return nn_module

    def build_optimizer(self,
                        optimizer_meta: types.AttrMeta,
                        optim_params: types.Param):
        optimizer, params = choose_attribute_from_dict(optimizer_meta,
                                                       optim_params)
        optimizer = cast_optimizer(optimizer)
        grad_params = (param for param in self.nn_module.parameters()
                       if param.requires_grad)
        optimizer = optimizer(params=grad_params, **params)
        return optimizer

    def build_loss(self,
                   loss_meta: types.AttrMeta,
                   loss_params: types.Param):
        loss, params = choose_attribute_from_dict(loss_meta,
                                                  loss_params)
        loss = cast_loss(loss)
        loss = loss(**params)
        return loss

    def build_prediction_transform(self,
                                   transform_meta: types.AttrMeta,
                                   transform_params: types.Param):
        transform, params = choose_attribute_from_dict(transform_meta,
                                                       transform_params)
        transform = cast_prediction_transform(transform)
        prediction_transform = transform(**params)
        return prediction_transform

    def build_device(self,
                     device_meta: types.InputDevices,
                     device_param: types.InputDevices) -> torch.device:
        if device_param:
            device = device_param
        else:
            device = device_meta
        self.set_device(cast_device(device))
        return self.device

    def build_logger(self):
        formatter = logging.Formatter('[%(asctime)s][%(levelname)s]: %(message)s')
        stdout = logging.StreamHandler(stream=sys.stdout)
        stdout.setLevel(logging.INFO)
        stdout.setFormatter(formatter)

        logger = logging.getLogger(f"{__name__}_{id(self)}")
        logger.setLevel(logging.INFO)
        if logger.hasHandlers():
            logger.handlers.clear()
        logger.addHandler(stdout)
        return logger

    def get_nn_module(self) -> nn.Module:
        """Get nn_module without :class:`torch.nn.DataParallel` or
        :class:`torch.nn.parallel.DistributedDataParallel`.

        Returns:
            :class:`torch.nn.Module`: nn_module without DP and DDP.

        """
        if isinstance(self.nn_module, (DataParallel, DistributedDataParallel)):
            return self.nn_module.module
        else:
            return self.nn_module

    def set_device(self, device: types.InputDevices):
        """Move nn_module and loss to the specified device.

        If a list of devices is passed, :class:`torch.nn.DataParallel` will be
        used. Batch tensors will be scattered on dim 0. The first device in the
        list is the location of the output. By default, device "cuda" is the
        GPU training on :func:`torch.cuda.current_device`.

        Example:

            .. code-block:: python

                model.set_device("cuda")
                model.set_device(torch.device("cuda"))

                model.set_device("cuda:0")
                model.set_device(["cuda:2", "cuda:3"])  # Use DataParallel

                model.set_device([torch.device("cuda:2"),
                                  torch.device("cuda", index=3)])

        Args:
            device (str, torch.device or list of devices): A device or list of
                devices.

        """
        torch_device = cast_device(device)
        nn_module = self.get_nn_module()

        if isinstance(torch_device, (list, tuple)):
            device_ids = get_device_indices(torch_device)
            nn_module = DataParallel(nn_module, device_ids=device_ids)
            output_device = torch_device[0]
        else:
            output_device = torch_device

        self.nn_module = nn_module.to(output_device)
        if self.loss is not None:
            self.loss = self.loss.to(output_device)
        self.params['device'] = device_to_str(torch_device)
        self.device = output_device

    def get_device(self) -> types.Devices:
        """Get device or list of devices in case of multi-GPU mode.

        Returns:
            torch.device or list of torch.device: A device or list of
            devices.

        """
        return cast_device(self.params['device'])

    def _check_attributes(self, attrs: Iterable) -> bool:
        for attr_name in attrs:
            attr_value = getattr(self, attr_name, None)
            if attr_value is None:
                return False
        return True

    def train_ready(self) -> bool:
        return self._check_attributes(TRAIN_ATTRS)

    def predict_ready(self) -> bool:
        return self._check_attributes(PREDICT_ATTRS)

    def _check_train_ready(self):
        if not self.train_ready():
            raise AttributeError(
                f"Not all required training attributes are there: {TRAIN_ATTRS}"
            )

    def _check_predict_ready(self):
        if not self.predict_ready():
            raise AttributeError(
                f"Not all required prediction attributes are there: {PREDICT_ATTRS}"
            )

    def __repr__(self) -> str:
        return str(self.__dict__)
