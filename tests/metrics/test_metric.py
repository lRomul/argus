import math
import pytest

import argus
from argus.metrics.metric import Metric, init_better
from argus.model.model import _attach_metrics
from argus.engine import Engine


class CustomMetric(Metric):
    name = 'custom_metric'
    better = 'max'

    def __init__(self):
        self.data = []
        super().__init__()

    def reset(self):
        self.data = []

    def update(self, step_output: dict):
        self.data.append(step_output)

    def compute(self):
        return len(self.data)


def test_init_better():
    def check_min(better, better_comp, best_value):
        assert better == 'min'
        assert better_comp(1, 2)
        assert best_value == math.inf

    def check_max(better, better_comp, best_value):
        assert better == 'max'
        assert better_comp(2, 1)
        assert best_value == -math.inf

    check_max(*init_better('max', ''))
    check_min(*init_better('min', ''))
    check_min(*init_better('auto', 'val_loss'))
    check_min(*init_better('auto', 'train_loss'))
    check_max(*init_better('auto', 'val_custom_metric'))
    check_max(*init_better('auto', 'train_custom_metric'))

    with pytest.raises(ValueError):
        init_better('qwerty', '')

    with pytest.raises(ImportError):
        init_better('auto', 'train_qwerty')


class TestMetric:
    def test_redefine_metric_warn(self, recwarn):
        class RedefineModel1(Metric):
            name = 'redefine_model'

        class RedefineModel2(Metric):
            name = 'redefine_model'

        assert len(recwarn) == 1
        warn = recwarn.pop()
        assert "redefined 'redefine_model' that was already" in str(warn.message)

    def test_custom_metric(self, engine):
        metric = CustomMetric()
        data_loader = [4, 8, 15, 16, 23, 42]
        _attach_metrics(engine, [metric])
        with pytest.raises(TypeError):
            _attach_metrics(engine, [None])
        state = engine.run(data_loader)
        assert metric.data == data_loader
        assert metric.compute() == len(data_loader)
        assert state.metrics == {"custom_metric": len(data_loader)}
        metric.reset()
        assert metric.data == []
        assert metric.compute() == 0

        engine = Engine(lambda batch, state: batch, phase='train')
        _attach_metrics(engine, [metric])
        state = engine.run(data_loader)
        assert metric.compute() == len(data_loader)
        assert state.metrics == {"train_custom_metric": len(data_loader)}

        @argus.callbacks.on_iteration_start
        def stop_on_first_iteration(state):
            state.stopped = True

        stop_on_first_iteration.attach(engine)
        engine.run(data_loader)
        assert metric.compute() == 1

    def test_custom_callback_by_name(self):
        data_loader = [4, 8, 15, 16, 23, 42]
        engine = Engine(lambda batch, state: batch, phase='val')
        _attach_metrics(engine, ["custom_metric"])
        state = engine.run(data_loader)
        assert state.metrics == {"val_custom_metric": len(data_loader)}

        with pytest.raises(ValueError):
            _attach_metrics(engine, ["qwerty"])

    def test_just_for_coverage(self):
        metric = Metric()
        assert metric.reset() is None
        assert metric.update(dict()) is None
        assert metric.compute() is None
