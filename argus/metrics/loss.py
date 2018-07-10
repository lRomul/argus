
from argus.metrics.metric import Metric


class Loss(Metric):
    def __init__(self, loss, output_transform=lambda x: x):
        super(Loss, self).__init__(output_transform)
        self._loss = loss
        self.reset()

    def reset(self):
        self._sum = 0
        self._num_examples = 0

    def update(self, output):
        pred, trg = output
        average_loss = self._loss(pred, trg)
        self._sum += average_loss.item() * trg.shape[0]
        self._num_examples += trg.shape[0]

    def compute(self):
        if self._num_examples == 0:
            raise ZeroDivisionError
        return self._sum / self._num_examples
