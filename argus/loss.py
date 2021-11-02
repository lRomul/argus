from typing import Dict, Type
from torch.nn.modules.loss import _Loss
from argus.utils import inheritors

__all__ = ["get_pytorch_losses", "pytorch_losses"]


def _is_pytorch_loss(loss: Type) -> bool:
    if not loss.__module__.startswith('torch.nn.modules.loss'):
        return False
    elif loss.__name__.startswith('_'):  # filter _WeightedLoss
        return False
    return True


def get_pytorch_losses() -> Dict[str, Type[_Loss]]:
    losses = inheritors(_Loss)
    losses_dict = {loss.__name__: loss for loss in losses
                   if _is_pytorch_loss(loss)}
    return losses_dict


pytorch_losses = get_pytorch_losses()
