from torch.nn.modules.loss import _Loss
from argus.utils import inheritors


def _check_loss(loss):
    if not loss.__module__.startswith('torch.nn.modules.loss'):
        return False
    elif loss.__name__.startswith('_'):  # filter _WeightedLoss
        return False
    return True


def get_pytorch_losses():
    losses = inheritors(_Loss)
    losses_dict = {loss.__name__: loss for loss in losses
                   if _check_loss(loss)}
    return losses_dict


pytorch_losses = get_pytorch_losses()
