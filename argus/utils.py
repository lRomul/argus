import os
import torch
import collections

default = object()

TRAIN_ATTRS = {'nn_module', 'optimizer', 'loss', 'device'}
PREDICT_ATTRS = {'nn_module', 'predict_transform', 'device'}
ALL_ATTRS = TRAIN_ATTRS | PREDICT_ATTRS


def to_device(input_, device):
    if torch.is_tensor(input_):
        if device:
            input_ = input_.to(device=device)
        return input_
    elif isinstance(input_, str):
        return input_
    elif isinstance(input_, collections.Mapping):
        return {k: to_device(sample, device=device) for k, sample in input_.items()}
    elif isinstance(input_, collections.Sequence):
        return [to_device(sample, device=device) for sample in input_]
    else:
        raise TypeError(f"Input must contain tensor, dict or list, found {type(input_)}")


def mkdir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)


def inheritors(cls):
    subclasses = set()
    cls_list = [cls]
    while cls_list:
        parent = cls_list.pop()
        for child in parent.__subclasses__():
            if child not in subclasses:
                subclasses.add(child)
                cls_list.append(child)
    return subclasses
