import torch
import pytest

from argus.callbacks import Callback
from argus.metrics import CategoricalAccuracy


@pytest.mark.parametrize("batch_size, n_classes, n_iterations",
                         [(8, 4, 32), (16, 2, 15), (27, 13, 38)])
def test_categorical_accuracy(batch_size, n_classes, n_iterations):
    metric = CategoricalAccuracy()
    assert isinstance(metric, Callback)
    prediction_lst = []
    target_lst = []
    for _ in range(n_iterations):
        prediction = torch.rand(batch_size, n_classes)
        target = torch.randint(n_classes, size=(batch_size,))
        metric.update({'prediction': prediction, 'target': target})
        prediction_lst.append(prediction)
        target_lst.append(target)

    prediction = torch.cat(prediction_lst, dim=0)
    target = torch.cat(target_lst, dim=0)

    indices = torch.max(prediction, dim=1)[1]
    correct = torch.eq(indices, target).view(-1)
    accuracy = torch.sum(correct).item() / len(target)

    assert pytest.approx(metric.compute()) == accuracy

    metric.reset()
    with pytest.raises(RuntimeError):
        metric.compute()
