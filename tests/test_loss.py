import torch

from argus.loss import get_pytorch_losses, _is_pytorch_loss


def test_get_pytorch_losses():
    pytorch_losses = get_pytorch_losses()
    assert isinstance(pytorch_losses, dict)
    for key, value in pytorch_losses.items():
        assert isinstance(key, str)
        assert _is_pytorch_loss(value)


def test_is_pytorch_loss():
    assert _is_pytorch_loss(torch.nn.BCELoss)
    assert not _is_pytorch_loss(torch.optim.SGD)
    assert not _is_pytorch_loss(torch.nn.modules.loss._WeightedLoss)
