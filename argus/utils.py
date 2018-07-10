import torch
import collections
import logging


default = object()


def setup_logging(file_path=None):
    handlers = [logging.StreamHandler()]
    if file_path is not None:
        handlers.append(logging.FileHandler(file_path))
    logging.basicConfig(
        format='%(asctime)s %(levelname)s %(message)s',
        level=logging.getLevelName(logging.INFO),
        handlers=handlers,
    )


def to_device(input, device):
    if torch.is_tensor(input):
        return input.to(device=device)
    elif isinstance(input, str):
        return input
    elif isinstance(input, collections.Mapping):
        return {k: to_device(sample, device=device) for k, sample in input.items()}
    elif isinstance(input, collections.Sequence):
        return [to_device(sample, device=device) for sample in input]
    else:
        raise TypeError(f"Input must contain tensor, dict or list, found {type(input)}")


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


class AverageMeter(object):
    """Computes and stores the average by Welford's algorithm"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.average = 0
        self.count = 0

    def update(self, value, n=1):
        self.count += n
        self.average += (value - self.average) / self.count
