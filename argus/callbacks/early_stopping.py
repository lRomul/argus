import math

from argus.engine import State
from argus.callbacks.callback import Callback
from argus.metrics.metric import METRIC_REGISTRY


class EarlyStopping(Callback):
    def __init__(self,
                 monitor='val_loss',
                 patience=1,
                 better='auto'):
        self.monitor = monitor
        self.patience = patience
        self.better = better

        if self.better == 'auto':
            if monitor.startswith('val_'):
                metric_name = self.monitor[len('val_'):]
            else:
                metric_name = self.monitor[len('train_'):]
            if metric_name not in METRIC_REGISTRY:
                raise ImportError(f"Metric '{metric_name}' not found in scope")
            self.better = METRIC_REGISTRY[metric_name].better
        assert self.better in ['min', 'max', 'auto'], \
            f"Unknown better option '{self.better}'"

        if self.better == 'min':
            self.better_comp = lambda a, b: a < b
            self.best_value = math.inf
        elif self.better == 'max':
            self.better_comp = lambda a, b: a > b
            self.best_value = -math.inf

        self.wait = 0

    def start(self, state: State):
        self.wait = 0
        self.best_value = math.inf if self.better == 'min' else -math.inf

    def epoch_complete(self, state: State):
        assert self.monitor in state.metrics,\
            f"Monitor '{self.monitor}' metric not found in state"
        current_value = state.metrics[self.monitor]
        if self.better_comp(current_value, self.best_value):
            self.best_value = current_value
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                state.stopped = True
                state.logger.info(
                    f"Epoch {state.epoch}: Early stopping triggered, "
                    f"'{self.monitor}' didn't improve score {self.wait} epochs"
                )
