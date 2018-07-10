from abc import ABCMeta, abstractmethod

from argus.engine import Events


class Metric(object):
    __metaclass__ = ABCMeta

    def __init__(self, output_transform=lambda x: x):
        self._output_transform = output_transform
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
        step_output = self._output_transform(engine.state.step_output)
        self.update(step_output)

    def epoch_complete(self, engine, name):
        engine.state.metrics[name] = self.compute()

    def attach(self, engine, name):
        engine.add_event_handler(Events.EPOCH_START, self.start)
        engine.add_event_handler(Events.ITERATION_COMPLETE, self.iteration_complete)
        engine.add_event_handler(Events.EPOCH_COMPLETE, self.epoch_complete, name)
