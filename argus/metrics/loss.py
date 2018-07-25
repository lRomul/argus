from argus.metrics.metric import Metric
from argus.utils import AverageMeter


class Loss(Metric):
    def __init__(self, name, loss):
        super().__init__(name)
        self._loss = loss
        self.reset()

    def reset(self):
        self._sum = 0
        self.count = 0

    def update(self, step_output):
        pred, trg = step_output
        average_loss = self._loss(pred, trg)
        self._sum += average_loss.item() * trg.shape[0]
        self.count += trg.shape[0]

    def compute(self):
        if self.count == 0:
            raise ZeroDivisionError
        return self._sum / self.count


class TrainLoss(Metric):
    def __init__(self, name):
        self.avg_meter = AverageMeter()
        super().__init__(name)

    def reset(self):
        self.avg_meter.reset()

    def update(self, step_output):
        loss = step_output
        self.avg_meter.update(loss)

    def compute(self):
        if self.avg_meter.count == 0:
            raise ZeroDivisionError
        return self.avg_meter.average
