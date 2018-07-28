from argus.callbacks import Callback
from argus.engine import State

class Metric(Callback):
    def __init__(self, name):
        self.name = name
        self.reset()

    def reset(self):
        pass

    def update(self, step_output):
        pass

    def compute(self):
        pass

    def start(self, state: State):
        self.reset()

    def iteration_complete(self, state: State):
        self.update(state.step_output)

    def epoch_complete(self, state: State):
        state.metrics[self.name] = self.compute()
