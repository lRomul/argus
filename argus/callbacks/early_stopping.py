"""A callback for argus model train stop after a metric has stopped improving.
"""
import math

from argus.engine import State
from argus.callbacks.callback import Callback
from argus.metrics.metric import init_better


class EarlyStopping(Callback):
    """Stop the model training after its metric has stopped improving.

    It is possible to monitor loss values during training as well as any
    metric available in the model State.

    Args:
        monitor (str, optional): Metric name to monitor. It should be
            prepended with *val_* for the metric value on validation data
            and *train_* for the metric value on the date from the train
            loader. A val_loader should be provided during the model fit to
            make it possible to monitor metrics start with *val_*.
            Defaults to *val_loss*.
        patience (int, optional): Number of training epochs without the
            metric improvement to stop training. Defaults to 1.
        better (str, optional): The metric improvement criterion. Should be
            'min', 'max' or 'auto'. 'auto' means the criterion should be
            taken from the metric itself, which is appropriate behavior in
            most cases. Defaults to 'auto'.

    """

    def __init__(self,
                 monitor: str = 'val_loss',
                 patience: int = 1,
                 better: str = 'auto'):
        self.monitor = monitor
        self.patience = patience
        self.better, self.better_comp, self.best_value = init_better(
            better, monitor)
        self.wait: int = 0

    def start(self, state: State):
        self.wait = 0
        self.best_value = math.inf if self.better == 'min' else -math.inf

    def epoch_complete(self, state: State):
        if self.monitor not in state.metrics:
            raise ValueError(f"Monitor '{self.monitor}' metric not found in state")
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
