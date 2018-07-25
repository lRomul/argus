from abc import ABCMeta, abstractmethod
from argus.callbacks import Callback


class Metric(Callback):
    __metaclass__ = ABCMeta

    def __init__(self, name):
        self.name = name
        self.reset()

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def update(self, step_output):
        pass

    @abstractmethod
    def compute(self):
        pass

    def start(self, engine):
        self.reset()

    def iteration_complete(self, engine):
        self.update(engine.state.step_output)

    def epoch_complete(self, engine):
        engine.state.metrics[self.name] = self.compute()
