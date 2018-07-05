from torch.optim.optimizer import Optimizer
from argus.utils import inheritors


def get_pytorch_optimizers():
    optimizers = inheritors(Optimizer)
    optimizers_dict = {opt.__name__: opt for opt in optimizers}
    return optimizers_dict


pytorch_optimizers = get_pytorch_optimizers()
