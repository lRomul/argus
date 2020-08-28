import pytest

from argus.loss import pytorch_losses
from argus.optimizer import pytorch_optimizers


@pytest.fixture(scope='session', params=pytorch_losses.values())
def loss_class(request):
    return request.param


@pytest.fixture(scope='session', params=pytorch_optimizers.values())
def optimizer_class(request):
    return request.param
