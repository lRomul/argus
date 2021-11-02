import torch

from argus.metrics.metric import Metric

__all__ = ["CategoricalAccuracy"]


class CategoricalAccuracy(Metric):
    """Calculates the accuracy for multiclass classification."""

    name = 'accuracy'
    better = 'max'

    def __init__(self):
        self.correct = 0
        self.count = 0

    def reset(self):
        self.correct = 0
        self.count = 0

    def update(self, step_output: dict):
        indices = torch.max(step_output['prediction'], dim=1)[1]
        correct = torch.eq(indices, step_output['target']).view(-1)
        self.correct += torch.sum(correct).item()
        self.count += correct.shape[0]

    def compute(self) -> float:
        if self.count == 0:
            raise RuntimeError('Must be at least one example for computation')
        return self.correct / self.count
