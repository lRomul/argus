import math

from torch.optim import lr_scheduler as _scheduler

from argus.engine import State
from argus.callbacks.callback import Callback
from argus.metrics.metric import init_better


class LRScheduler(Callback):
    def __init__(self, scheduler_factory, monitor=None):
        self.scheduler_factory = scheduler_factory
        self._monitor = monitor
        self._scheduler = None

    def start(self, state: State):
        self._scheduler = self.scheduler_factory(state.model.optimizer)

    def epoch_complete(self, state: State):
        self._scheduler.step(epoch=state.epoch)


class LambdaLR(LRScheduler):
    def __init__(self, lr_lambda):
        super().__init__(lambda opt: _scheduler.LambdaLR(opt,
                                                         lr_lambda))


class StepLR(LRScheduler):
    def __init__(self, step_size, gamma=0.1):
        super().__init__(lambda opt: _scheduler.StepLR(opt,
                                                       step_size,
                                                       gamma=gamma))


class MultiStepLR(LRScheduler):
    def __init__(self, milestones, gamma=0.1):
        super().__init__(lambda opt: _scheduler.MultiStepLR(opt,
                                                            milestones,
                                                            gamma=gamma))


class ExponentialLR(LRScheduler):
    def __init__(self, gamma):
        super().__init__(lambda opt: _scheduler.ExponentialLR(opt,
                                                              gamma))


class CosineAnnealingLR(LRScheduler):
    def __init__(self, T_max, eta_min=0):
        super().__init__(lambda opt: _scheduler.CosineAnnealingLR(opt,
                                                                  T_max,
                                                                  eta_min=eta_min))


class ReduceLROnPlateau(LRScheduler):
    def __init__(self, monitor='val_loss', better='auto', factor=0.1,
                 patience=10, verbose=False, threshold=1e-4,
                 threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-8):

        self.monitor = monitor
        self.patience = patience
        self.better, self.better_comp, self.best_value = init_better(better, monitor)

        super().__init__(lambda opt: _scheduler.ReduceLROnPlateau(opt,
                                                                  mode=self.better,
                                                                  factor=factor,
                                                                  patience=patience,
                                                                  verbose=verbose,
                                                                  threshold=threshold,
                                                                  threshold_mode=threshold_mode,
                                                                  cooldown=cooldown,
                                                                  min_lr=min_lr,
                                                                  eps=eps))

    def start(self, state: State):
        self._scheduler = self.scheduler_factory(state.model.optimizer)
        self.best_value = math.inf if self.better == 'min' else -math.inf

    def epoch_complete(self, state: State):
        self._scheduler.step(metrics=state.metrics[self.monitor], epoch=state.epoch)
