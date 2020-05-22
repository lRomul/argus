"""Wrappers to use learning rate schedulers from PyTorch with argus models.

It enables the PyTorch lr_schedulers to be used as normal argus Callbacks.
"""
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
        if self._scheduler is None:
            self._scheduler = self.scheduler_factory(state.model.optimizer)

    def epoch_complete(self, state: State):
        self._scheduler.step()


class LambdaLR(LRScheduler):
    """LambdaLR scheduler.

    Multiply learning rate by a factor computed with a given function. The
    function should take int value number of epochs as the only argument.

    Args:
        lr_lambda (function or list of functions): Lambda function for the
            learning rate factor computation.

    """

    def __init__(self, lr_lambda):
        super().__init__(lambda opt: _scheduler.LambdaLR(opt,
                                                         lr_lambda))


class StepLR(LRScheduler):
    """StepLR scheduler.

    Multiply learning rate by a given factor with a given period.

    Args:
        step_size (int): Period of learning rate update in epochs.
        gamma (float, optional): Multiplicative factor. Defaults to 0.1.

    """

    def __init__(self, step_size, gamma=0.1):
        super().__init__(lambda opt: _scheduler.StepLR(opt,
                                                       step_size,
                                                       gamma=gamma))


class MultiStepLR(LRScheduler):
    """MultiStepLR scheduler.

    Multiply learning rate by a given factor on each epoch from a given list.

    Args:
        milestones (list of ints): List of epochs number to perform lr step.
        gamma (float, optional): Multiplicative factor. Defaults to 0.1.

    """

    def __init__(self, milestones, gamma=0.1):
        super().__init__(lambda opt: _scheduler.MultiStepLR(opt,
                                                            milestones,
                                                            gamma=gamma))


class ExponentialLR(LRScheduler):
    """MultiStepLR scheduler.

    Multiply learning rate by a given factor on each epoch.

    Args:
        gamma (float, optional): Multiplicative factor. Defaults to 0.1.

    """

    def __init__(self, gamma):
        super().__init__(lambda opt: _scheduler.ExponentialLR(opt,
                                                              gamma))


class CosineAnnealingLR(LRScheduler):
    """CosineAnnealingLR scheduler.

    Set the learning rate of each parameter group using a cosine annealing
    schedule.

    Args:
        T_max (int): Max number of epochs.
        eta_min (float, optional): Min learning rate. Defaults to 0.

    """

    def __init__(self, T_max, eta_min=0):
        super().__init__(lambda opt: _scheduler.CosineAnnealingLR(opt,
                                                                  T_max,
                                                                  eta_min=eta_min))


class ReduceLROnPlateau(LRScheduler):
    """ReduceLROnPlateau scheduler.

    Args:
        monitor (str, optional): Metric name to monitor. It should be
            prepended with *val_* for the metric value on validation data
            and *train_* for the metric value on the date from the train
            loader. A val_loader should be provided during the model fit to
            make it possible to monitor metrics start with *val_*.
            Defaults to *val_loss*.
        better (str, optional): The metric improvement criterion. Should be
            'min', 'max' or 'auto'. 'auto' means the criterion should be
            taken from the metric itself, which is appropriate behavior in
            most cases. Defaults to 'auto'.
        factor (float, optional): Multiplicative factor. Defaults to 0.1.
        patience (int, optional): Number of training epochs without the
            metric improvement to update the learning rate. Defaults to 10.
        verbose (bool, optional): Print info on each update to stdout.
            Defaults to False.
        threshold (float, optional): Threshold for considering the changes
            significant. Defaults to 1e-4.
        threshold_mode (str, optional): Should be 'rel', 'abs'.
            Defaults to 'rel'.
        cooldown (int, optional): Number of epochs to wait before resuming
            normal operation after lr has been updated. Defaults to 0.
        min_lr (float or list of floats, optional): Min learning rate.
            Defaults to 0.
        eps (float, optional): Min significant learning rate update.
            Defaults to 1e-8.

    """

    def __init__(self, monitor='val_loss', better='auto', factor=0.1,
                 patience=10, verbose=False, threshold=1e-4,
                 threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-8):
        self.monitor = monitor
        self.patience = patience
        self.better, self.better_comp, self.best_value = init_better(
            better, monitor)

        super().__init__(
            lambda opt: _scheduler.ReduceLROnPlateau(opt,
                                                     mode=self.better,
                                                     factor=factor,
                                                     patience=patience,
                                                     verbose=verbose,
                                                     threshold=threshold,
                                                     threshold_mode=threshold_mode,
                                                     cooldown=cooldown,
                                                     min_lr=min_lr,
                                                     eps=eps)
        )

    def start(self, state: State):
        self._scheduler = self.scheduler_factory(state.model.optimizer)
        self.best_value = math.inf if self.better == 'min' else -math.inf

    def epoch_complete(self, state: State):
        self._scheduler.step(metrics=state.metrics[self.monitor], epoch=state.epoch)


class CyclicLR(LRScheduler):
    """CyclicLR scheduler.

    Args:
        base_lr (float or list of floats): Initial learning rate.
        max_lr (float or list of floats): Max learning rate.
        step_size_up (int, optional): Increase phase duration in epochs.
            Defaults to 2000.
        step_size_down (int, optional): Decrease phase duration in epochs.
            Defaults to None.
        mode (str, optional): Should be 'triangular', 'triangular2' or
            'exp_range'. Defaults to 'triangular'.
        gamma (float, optional): Constant for the 'exp_range' policy.
            Defaults to 1.
        scale_fn (function, optional): Custom scaling policy function.
            Defaults to None.
        scale_mode (str, optional): Should be 'cycle' or 'iterations'.
            Defaults to 'cycle'.
        cycle_momentum (bool, optional): [description]. Defaults to True.
        base_momentum (float or list of floats, optional): [description].
            Defaults to 0.8.
        max_momentum (float or list of floats, optional): [description].
            Defaults to 0.9.

    """

    def __init__(self,
                 base_lr,
                 max_lr,
                 step_size_up=2000,
                 step_size_down=None,
                 mode='triangular',
                 gamma=1.,
                 scale_fn=None,
                 scale_mode='cycle',
                 cycle_momentum=True,
                 base_momentum=0.8,
                 max_momentum=0.9):
        super().__init__(
            lambda opt: _scheduler.CyclicLR(opt,
                                            base_lr,
                                            max_lr,
                                            step_size_up=step_size_up,
                                            step_size_down=step_size_down,
                                            mode=mode,
                                            gamma=gamma,
                                            scale_fn=scale_fn,
                                            scale_mode=scale_mode,
                                            cycle_momentum=cycle_momentum,
                                            base_momentum=base_momentum,
                                            max_momentum=max_momentum)
        )


class CosineAnnealingWarmRestarts(LRScheduler):
    """CosineAnnealingLR scheduler.

    Set the learning rate of each parameter group using a cosine annealing
    schedule with a warm restart.

    Args:
        T_0 (int): Number of epochs for the first restart.
        T_mult (int): T increase factor after a restart.
        eta_min (float, optional): Min learning rate. Defaults to 0.
    """

    def __init__(self,
                 T_0,
                 T_mult=1,
                 eta_min=0):
        super().__init__(
            lambda opt: _scheduler.CosineAnnealingWarmRestarts(opt,
                                                               T_0,
                                                               T_mult=T_mult,
                                                               eta_min=eta_min)
        )
