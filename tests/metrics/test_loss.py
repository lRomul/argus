import pytest

from argus.metrics.loss import Loss


def test_loss_metric(one_dim_num_sequence):
    metric = Loss()
    assert metric.name == 'loss'
    assert metric.better == 'min'
    step_outputs = [{'loss': s} for s in one_dim_num_sequence]
    for output in step_outputs:
        metric.update(output)

    average = sum(one_dim_num_sequence) / len(one_dim_num_sequence)
    assert pytest.approx(metric.compute()) == average

    metric.reset()
    with pytest.raises(RuntimeError):
        metric.compute()
