from argus.metrics.metric import Metric
from argus.utils import AverageMeter


class Loss(Metric):
    name = 'loss'

    def __init__(self, loss):
        super().__init__(self.name)
        self._loss = loss
        self.reset()

    def reset(self):
        self._sum = 0
        self.count = 0

    def update(self, step_output: dict):
        pred = step_output['prediction']
        trg = step_output['target']
        average_loss = self._loss(pred, trg)
        self._sum += average_loss.item() * trg.shape[0]
        self.count += trg.shape[0]

    def compute(self):
        if self.count == 0:
            raise ZeroDivisionError
        return self._sum / self.count


class TrainLoss(Metric):
    name = 'train_loss'

    def __init__(self, ):
        self.avg_meter = AverageMeter()
        super().__init__(self.name)

    def reset(self):
        self.avg_meter.reset()

    def update(self, step_output: dict):
        loss = step_output['loss']
        self.avg_meter.update(loss)

    def compute(self):
        if self.avg_meter.count == 0:
            raise ZeroDivisionError
        return self.avg_meter.average
