import pytest
from distutils.version import LooseVersion

import torch

from argus.optimizer import get_pytorch_optimizers, _is_pytorch_optimizer


def test_get_pytorch_optimizers():
    pytorch_optimizers = get_pytorch_optimizers()
    assert isinstance(pytorch_optimizers, dict)
    for key, value in pytorch_optimizers.items():
        assert isinstance(key, str)
        assert _is_pytorch_optimizer(value)


def test_is_pytorch_optimizer():
    assert _is_pytorch_optimizer(torch.optim.SGD)
    assert not _is_pytorch_optimizer(torch.nn.BCELoss)


@pytest.mark.skipif(LooseVersion(torch.__version__) < LooseVersion("1.7.0"),
                    reason="Requires torch==1.7.0 or higher")
def test_is_multi_tensor_optimizer():
    from torch.optim import _multi_tensor
    assert not _is_pytorch_optimizer(_multi_tensor.SGD)
    assert not _is_pytorch_optimizer(_multi_tensor.Adam)
