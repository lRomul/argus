from torch.nn.modules.loss import _Loss
from argus.utils import inheritors


def get_pytorch_losses():
    losses = inheritors(_Loss)
    losses_dict = {l.__name__: l for l in losses
                   if not l.__name__.startswith('_')}  # filter _WeightedLoss
    return losses_dict


pytorch_losses = get_pytorch_losses()
