from argus.metrics.metric import Metric
from argus.utils import AverageMeter


class Loss(Metric):
    name = 'loss'

    def __init__(self):
        self.avg_meter = AverageMeter()
        super().__init__()

    def reset(self):
        self.avg_meter.reset()

    def update(self, step_output: dict):
        self.avg_meter.update(step_output['loss'])

    def compute(self):
        if self.avg_meter.count == 0:
            raise Exception('Must be at least one example for computation')
        return self.avg_meter.average
