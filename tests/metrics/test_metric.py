import math
import pytest

from argus.metrics.metric import Metric, init_better
from argus.utils import AverageMeter


class CustomMetric(Metric):
    name = 'custom_metric'
    better = 'max'

    def __init__(self):
        self.avg_meter = AverageMeter()
        super().__init__()

    def reset(self):
        self.avg_meter.reset()

    def update(self, step_output: dict):
        self.avg_meter.update(step_output['loss'])

    def compute(self):
        return self.avg_meter.average


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
