import os
import torch
import collections
import logging
import sys

default = object()


def to_device(input_, device):
    if torch.is_tensor(input_):
        return input_.to(device=device)
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


def get_logger(log_file_path=None):
    logger = logging.getLogger()
    formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(module)s: %(message)s')
    stream_handler = logging.StreamHandler(stream=sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    if log_file_path is not None:
        file_handler = logging.FileHandler(log_file_path)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
