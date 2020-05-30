"""Wrappers to use learning rate schedulers from PyTorch with argus models.

It enables the PyTorch lr_schedulers to be used as normal argus Callbacks.
"""
import math

import torch
from torch.optim import lr_scheduler as _scheduler

from argus.engine import State
from argus.callbacks.callback import Callback
from argus.metrics.metric import init_better


class LRScheduler(Callback):
    def __init__(self, scheduler_factory, step_on_iteration=False):
        self.scheduler_factory = scheduler_factory
        self.step_on_iteration = step_on_iteration
        self._scheduler = None

    def start(self, state: State):
        if self._scheduler is None:
            self._scheduler = self.scheduler_factory(state.model.optimizer)

    def epoch_complete(self, state: State):
        if not self.step_on_iteration:
            self._scheduler.step()

    def iteration_complete(self, state: State):
        if self.step_on_iteration:
            self._scheduler.step()


class LambdaLR(LRScheduler):
    """LambdaLR scheduler.

    Multiply learning rate by a factor computed with a given function. The
    function should take int value number of epochs as the only argument.

    Args:
        lr_lambda (function or list of functions): Lambda function for the
            learning rate factor computation.
        step_on_iteration (bool): Step on each training iteration rather than each epoch.
            Defaults to False.

    """

    def __init__(self, lr_lambda, step_on_iteration=False):
        super().__init__(
            lambda opt: _scheduler.LambdaLR(opt,
                                            lr_lambda),
            step_on_iteration=step_on_iteration
        )


class StepLR(LRScheduler):
    """StepLR scheduler.

    Multiply learning rate by a given factor with a given period.

    Args:
        step_size (int): Period of learning rate update in epochs.
        gamma (float, optional): Multiplicative factor. Defaults to 0.1.
        step_on_iteration (bool): Step on each training iteration rather than each epoch.
            Defaults to False.

    """

    def __init__(self, step_size, gamma=0.1, step_on_iteration=False):
        super().__init__(
            lambda opt: _scheduler.StepLR(opt,
                                          step_size,
                                          gamma=gamma),
            step_on_iteration=step_on_iteration
        )


class MultiStepLR(LRScheduler):
    """MultiStepLR scheduler.

    Multiply learning rate by a given factor on each epoch from a given list.

    Args:
        milestones (list of ints): List of epochs number to perform lr step.
        gamma (float, optional): Multiplicative factor. Defaults to 0.1.
        step_on_iteration (bool): Step on each training iteration rather than each epoch.
            Defaults to False.

    """

    def __init__(self, milestones, gamma=0.1, step_on_iteration=False):
        super().__init__(
            lambda opt: _scheduler.MultiStepLR(opt,
                                               milestones,
                                               gamma=gamma),
            step_on_iteration=step_on_iteration
        )


class ExponentialLR(LRScheduler):
    """MultiStepLR scheduler.

    Multiply learning rate by a given factor on each epoch.

    Args:
        gamma (float, optional): Multiplicative factor. Defaults to 0.1.
        step_on_iteration (bool): Step on each training iteration rather than each epoch.
            Defaults to False.

    """

    def __init__(self, gamma, step_on_iteration=False):
        super().__init__(
            lambda opt: _scheduler.ExponentialLR(opt,
                                                 gamma),
            step_on_iteration=step_on_iteration
        )


class CosineAnnealingLR(LRScheduler):
    """CosineAnnealingLR scheduler.

    Set the learning rate of each parameter group using a cosine annealing
    schedule.

    Args:
        T_max (int): Max number of epochs or iterations.
        eta_min (float, optional): Min learning rate. Defaults to 0.
        step_on_iteration (bool): Step on each training iteration rather than each epoch.
            Defaults to False.

    """

    def __init__(self, T_max, eta_min=0, step_on_iteration=False):
        super().__init__(
            lambda opt: _scheduler.CosineAnnealingLR(opt,
                                                     T_max,
                                                     eta_min=eta_min),
            step_on_iteration=step_on_iteration
        )


class ReduceLROnPlateau(LRScheduler):
    """ReduceLROnPlateau scheduler.

    Reduce learning rate when a metric has stopped improving.

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
                                                     eps=eps),
            step_on_iteration=False
        )

    def start(self, state: State):
        self._scheduler = self.scheduler_factory(state.model.optimizer)
        self.best_value = math.inf if self.better == 'min' else -math.inf

    def epoch_complete(self, state: State):
        self._scheduler.step(metrics=state.metrics[self.monitor])


class CyclicLR(LRScheduler):
    """CyclicLR scheduler.

    Sets the learning rate of each parameter group according to cyclical
    learning rate policy.

    Args:
        base_lr (float or list of floats): Initial learning rate.
        max_lr (float or list of floats): Max learning rate.
        step_size_up (int, optional): Increase phase duration in epochs or iterations.
            Defaults to 2000.
        step_size_down (int, optional): Decrease phase duration in epochs or iterations.
            Defaults to None.
        mode (str, optional): Should be 'triangular', 'triangular2' or
            'exp_range'. Defaults to 'triangular'.
        gamma (float, optional): Constant for the 'exp_range' policy.
            Defaults to 1.
        scale_fn (function, optional): Custom scaling policy function.
            Defaults to None.
        scale_mode (str, optional): Should be 'cycle' or 'iterations'.
            Defaults to 'cycle'.
        cycle_momentum (bool, optional): Momentum is cycled inversely
            to learning rate between 'base_momentum' and 'max_momentum'.
            Defaults to True.
        base_momentum (float or list of floats, optional): Lower momentum
            boundaries in the cycle for each parameter group.
            Defaults to 0.8.
        max_momentum (float or list of floats, optional): Upper momentum
            boundaries in the cycle for each parameter group.
            Defaults to 0.9.
        step_on_iteration (bool): Step on each training iteration rather than each epoch.
            Defaults to True.

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
                 max_momentum=0.9,
                 step_on_iteration=True):
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
                                            max_momentum=max_momentum),
            step_on_iteration=step_on_iteration
        )


class CosineAnnealingWarmRestarts(LRScheduler):
    """CosineAnnealingLR scheduler.

    Set the learning rate of each parameter group using a cosine annealing
    schedule with a warm restart.

    Args:
        T_0 (int): Number of epochs or iterations for the first restart.
        T_mult (int): T increase factor after a restart.
        eta_min (float, optional): Min learning rate. Defaults to 0.
        step_on_iteration (bool): Step on each training iteration rather than each epoch.
            Defaults to False.

    """

    def __init__(self,
                 T_0,
                 T_mult=1,
                 eta_min=0,
                 step_on_iteration=False):
        super().__init__(
            lambda opt: _scheduler.CosineAnnealingWarmRestarts(opt,
                                                               T_0,
                                                               T_mult=T_mult,
                                                               eta_min=eta_min),
            step_on_iteration=step_on_iteration
        )


class MultiplicativeLR(LRScheduler):
    """MultiplicativeLR scheduler.

    Multiply the learning rate of each parameter group by the factor given
    in the specified function.

    Args:
        lr_lambda (function or list): A function which computes a multiplicative
            factor given an integer parameter epoch, or a list of such
            functions, one for each group in optimizer.param_groups.
        step_on_iteration (bool): Step on each training iteration rather than each epoch.
            Defaults to False.
    """

    def __init__(self, lr_lambda, step_on_iteration=False):
        from distutils.version import LooseVersion
        if LooseVersion(torch.__version__) >= LooseVersion("1.4.0"):
            super().__init__(
                lambda opt: _scheduler.MultiplicativeLR(opt,
                                                        lr_lambda),
                step_on_iteration=step_on_iteration
            )
        else:
            raise ImportError("Update torch>=1.4.0 to use 'MultiplicativeLR'")


class OneCycleLR(LRScheduler):
    """OneCycleLR scheduler.

    Sets the learning rate of each parameter group according to the
    1cycle learning rate policy. The 1cycle policy anneals the learning
    rate from an initial learning rate to some maximum learning rate and then
    from that maximum learning rate to some minimum learning rate much lower
    than the initial learning rate.

    Args:
        max_lr (float or list): Upper learning rate boundaries in the cycle
            for each parameter group.
        total_steps (int): The total number of steps in the cycle. Note that
            if a value is not provided here, then it must be inferred by providing
            a value for epochs and steps_per_epoch.
            Defaults to None.
        epochs (int): The number of epochs to train for. This is used along
            with steps_per_epoch in order to infer the total number of steps in the cycle
            if a value for total_steps is not provided.
            Defaults to None.
        steps_per_epoch (int): The number of steps per epoch to train for. This is
            used along with epochs in order to infer the total number of steps in the
            cycle if a value for total_steps is not provided.
            Defaults to None.
        pct_start (float): The percentage of the cycle (in number of steps) spent
            increasing the learning rate.
            Defaults to 0.3.
        anneal_strategy (str): {'cos', 'linear'}
            Specifies the annealing strategy: "cos" for cosine annealing, "linear" for
            linear annealing.
            Defaults to 'cos'.
        cycle_momentum (bool): If ``True``, momentum is cycled inversely
            to learning rate between 'base_momentum' and 'max_momentum'.
            Defaults to True.
        base_momentum (float or list): Lower momentum boundaries in the cycle
            for each parameter group. Note that momentum is cycled inversely
            to learning rate; at the peak of a cycle, momentum is
            'base_momentum' and learning rate is 'max_lr'.
            Defaults to 0.85.
        max_momentum (float or list): Upper momentum boundaries in the cycle
            for each parameter group. Functionally,
            it defines the cycle amplitude (max_momentum - base_momentum).
            Note that momentum is cycled inversely
            to learning rate; at the start of a cycle, momentum is 'max_momentum'
            and learning rate is 'base_lr'
            Defaults to 0.95.
        div_factor (float): Determines the initial learning rate via
            initial_lr = max_lr/div_factor
            Defaults to 25.
        final_div_factor (float): Determines the minimum learning rate via
            min_lr = initial_lr/final_div_factor
            Defaults to 1e4.

    """

    def __init__(self,
                 max_lr,
                 total_steps=None,
                 epochs=None,
                 steps_per_epoch=None,
                 pct_start=0.3,
                 anneal_strategy='cos',
                 cycle_momentum=True,
                 base_momentum=0.85,
                 max_momentum=0.95,
                 div_factor=25.,
                 final_div_factor=1e4):
        from distutils.version import LooseVersion
        if LooseVersion(torch.__version__) >= LooseVersion("1.3.0"):
            super().__init__(
                lambda opt: _scheduler.OneCycleLR(opt,
                                                  max_lr,
                                                  total_steps=total_steps,
                                                  epochs=epochs,
                                                  steps_per_epoch=steps_per_epoch,
                                                  pct_start=pct_start,
                                                  anneal_strategy=anneal_strategy,
                                                  cycle_momentum=cycle_momentum,
                                                  base_momentum=base_momentum,
                                                  max_momentum=max_momentum,
                                                  div_factor=div_factor,
                                                  final_div_factor=final_div_factor),
                step_on_iteration=True
            )
        else:
            raise ImportError("Update torch>=1.3.0 to use 'OneCycleLR'")
