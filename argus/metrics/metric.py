from argus.callbacks import Callback
from argus.engine import State


class Metric(Callback):
    name = 'metric_name'

    def __init__(self):
        self.reset()

    def reset(self):
        pass

    def update(self, step_output: dict):
        pass

    def compute(self):
        pass

    def epoch_start(self, state: State):
        self.reset()

    def iteration_complete(self, state: State):
        self.update(state.step_output)

    def epoch_complete(self, state: State, name_prefix=''):
        state.metrics[name_prefix + self.name] = self.compute()
