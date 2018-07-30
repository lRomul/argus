import torch

from argus.metrics.metric import Metric


class CategoricalAccuracy(Metric):
    name = 'accuracy'
    better = 'max'

    def reset(self):
        self.correct = 0
        self.count = 0

    def update(self, step_output: dict):
        pred = step_output['prediction']
        trg = step_output['target']
        indices = torch.max(pred, dim=1)[1]
        correct = torch.eq(indices, trg).view(-1)
        self.correct += torch.sum(correct).item()
        self.count += correct.shape[0]

    def compute(self):
        if self.count == 0:
            raise Exception('Must be at least one example for computation')
        return self.correct / self.count
