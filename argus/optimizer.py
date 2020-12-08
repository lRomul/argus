from typing import Dict, Type
from torch.optim.optimizer import Optimizer
from argus.utils import inheritors


def _is_pytorch_optimizer(optimizer: Type) -> bool:
    if not optimizer.__module__.startswith('torch.optim'):
        return False
    elif optimizer.__module__.startswith('torch.optim._multi_tensor'):
        return False
    return True


def get_pytorch_optimizers() -> Dict[str, Type[Optimizer]]:
    optimizers = inheritors(Optimizer)
    optimizers_dict = {opt.__name__: opt for opt in optimizers
                       if _is_pytorch_optimizer(opt)}
    return optimizers_dict


pytorch_optimizers = get_pytorch_optimizers()
