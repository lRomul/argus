import pytest
from collections import Counter

from torch.optim import lr_scheduler
from torch.optim.optimizer import Optimizer

from argus.callbacks.lr_schedulers import (
    LRScheduler,
    LambdaLR,
    StepLR,
    MultiStepLR,
    ExponentialLR,
    CosineAnnealingLR,
    ReduceLROnPlateau,
    CyclicLR,
    CosineAnnealingWarmRestarts,
    MultiplicativeLR,
    OneCycleLR,
)


class MockTorchScheduler:
    def __init__(self, optimizer):
        self.optimizer = optimizer
        self.step_count = 0

    def step(self):
        self.step_count += 1


@pytest.fixture(scope='function', params=[True, False])
def step_on_iteration(request):
    return request.param


class TestLrSchedulers:
    def test_lr_scheduler(self, test_engine):
        scheduler = LRScheduler(MockTorchScheduler, step_on_iteration=False)
        scheduler.attach(test_engine)

        data_loader = [4, 8, 15, 16, 23, 42]
        test_engine.run(data_loader, start_epoch=0, end_epoch=3)

        assert scheduler.scheduler.step_count == 3
        scheduler.epoch_complete(test_engine.state)
        assert scheduler.scheduler.step_count == 4
        assert isinstance(scheduler.scheduler.optimizer, Optimizer)

    def test_lr_scheduler_step_on_iteration(self, test_engine):
        scheduler = LRScheduler(MockTorchScheduler, step_on_iteration=True)
        scheduler.attach(test_engine)

        data_loader = [4, 8, 15, 16, 23, 42]
        test_engine.run(data_loader, start_epoch=0, end_epoch=3)

        assert scheduler.scheduler.step_count == 3 * len(data_loader)
        scheduler.iteration_complete(test_engine.state)
        assert scheduler.scheduler.step_count == 3 * len(data_loader) + 1
        assert isinstance(scheduler.scheduler.optimizer, Optimizer)

    def test_lambda_lr(self, test_engine, step_on_iteration):
        lambda_lr = LambdaLR(lr_lambda=lambda epoch: 0.95 ** epoch,
                             step_on_iteration=step_on_iteration)
        lambda_lr.attach(test_engine)
        lambda_lr.start(test_engine.state)
        assert isinstance(lambda_lr.scheduler, lr_scheduler.LambdaLR)
        assert lambda_lr.scheduler.lr_lambdas[0](1) == 0.95 ** 1
        assert lambda_lr.step_on_iteration == step_on_iteration

    def test_step_lr(self, test_engine):
        step_lr = StepLR(step_size=10, gamma=0.1)
        step_lr.attach(test_engine)
        step_lr.start(test_engine.state)
        assert isinstance(step_lr.scheduler, lr_scheduler.StepLR)
        assert step_lr.scheduler.step_size == 10
        assert step_lr.scheduler.gamma == 0.1

    def test_multi_step_lr(self, test_engine, step_on_iteration):
        multi_step_lr = MultiStepLR(milestones=[30, 80], gamma=0.1,
                                    step_on_iteration=step_on_iteration)
        multi_step_lr.attach(test_engine)
        multi_step_lr.start(test_engine.state)
        assert isinstance(multi_step_lr.scheduler, lr_scheduler.MultiStepLR)
        assert multi_step_lr.scheduler.milestones == Counter([30, 80])
        assert multi_step_lr.scheduler.gamma == 0.1
        assert multi_step_lr.step_on_iteration == step_on_iteration

    def test_exponential_lr(self, test_engine, step_on_iteration):
        exponential_lr = ExponentialLR(gamma=0.1,
                                       step_on_iteration=step_on_iteration)
        exponential_lr.attach(test_engine)
        exponential_lr.start(test_engine.state)
        assert isinstance(exponential_lr.scheduler, lr_scheduler.ExponentialLR)
        assert exponential_lr.scheduler.gamma == 0.1
        assert exponential_lr.step_on_iteration == step_on_iteration

    def test_cosine_annealing_lr(self, test_engine):
        cosine_annealing_lr = CosineAnnealingLR(T_max=10, eta_min=0)
        cosine_annealing_lr.attach(test_engine)
        cosine_annealing_lr.start(test_engine.state)
        assert isinstance(cosine_annealing_lr.scheduler, lr_scheduler.CosineAnnealingLR)
        assert cosine_annealing_lr.scheduler.T_max == 10
        assert cosine_annealing_lr.scheduler.eta_min == 0

    def test_multiplicative_lr(self, test_engine, step_on_iteration):
        multiplicative_lr = MultiplicativeLR(lambda epoch: 0.95,
                                             step_on_iteration=step_on_iteration)
        multiplicative_lr.attach(test_engine)
        multiplicative_lr.start(test_engine.state)
        assert isinstance(multiplicative_lr.scheduler, lr_scheduler.MultiplicativeLR)
        assert multiplicative_lr.scheduler.lr_lambdas[0](1) == 0.95
        assert multiplicative_lr.step_on_iteration == step_on_iteration

    def test_one_cycle_lr(self, test_engine):
        one_cycle_lr = OneCycleLR(max_lr=0.01, steps_per_epoch=1000, epochs=10)
        one_cycle_lr.attach(test_engine)
        one_cycle_lr.start(test_engine.state)
        assert isinstance(one_cycle_lr.scheduler, lr_scheduler.OneCycleLR)
        assert one_cycle_lr.scheduler.total_steps == 10000
        assert one_cycle_lr.step_on_iteration

    def test_cosine_annealing_warm_restarts(self, test_engine, step_on_iteration):
        warm_restarts = CosineAnnealingWarmRestarts(T_0=1, T_mult=1, eta_min=0,
                                                    step_on_iteration=step_on_iteration)
        warm_restarts.attach(test_engine)
        warm_restarts.start(test_engine.state)
        assert isinstance(warm_restarts.scheduler,
                          lr_scheduler.CosineAnnealingWarmRestarts)
        assert warm_restarts.scheduler.T_0 == 1
        assert warm_restarts.scheduler.T_mult == 1
        assert warm_restarts.scheduler.eta_min == 0
        assert warm_restarts.step_on_iteration == step_on_iteration

    def test_cyclic_lr(self, test_engine, step_on_iteration):
        cyclic_lr = CyclicLR(base_lr=0.001, max_lr=0.01, gamma=1.,
                             mode='triangular', scale_mode='cycle',
                             cycle_momentum=True,
                             step_on_iteration=step_on_iteration)
        cyclic_lr.attach(test_engine)
        cyclic_lr.start(test_engine.state)
        assert isinstance(cyclic_lr.scheduler, lr_scheduler.CyclicLR)
        assert cyclic_lr.scheduler.base_lrs == [0.001]
        assert cyclic_lr.scheduler.max_lrs == [0.01]
        assert cyclic_lr.scheduler.gamma == 1.
        assert cyclic_lr.scheduler.mode == 'triangular'
        assert cyclic_lr.scheduler.scale_mode == 'cycle'
        assert cyclic_lr.scheduler.cycle_momentum
        assert cyclic_lr.step_on_iteration == step_on_iteration

    def test_reduce_lr_on_plateau(self, test_engine):
        lr_on_plateau = ReduceLROnPlateau(monitor='val_loss', better='auto',
                                          factor=0.1, patience=3,
                                          threshold=1e-6, threshold_mode='rel',
                                          cooldown=0, eps=1e-8)
        lr_on_plateau.attach(test_engine)
        lr_on_plateau.start(test_engine.state)
        assert isinstance(lr_on_plateau.scheduler, lr_scheduler.ReduceLROnPlateau)
        assert lr_on_plateau.better == 'min'
        assert lr_on_plateau.scheduler.factor == 0.1
        assert lr_on_plateau.scheduler.patience == 3
        assert lr_on_plateau.scheduler.threshold == 1e-6
        assert lr_on_plateau.scheduler.threshold_mode == 'rel'
        assert lr_on_plateau.scheduler.cooldown == 0
        assert lr_on_plateau.scheduler.eps == 1e-8
        assert not lr_on_plateau.step_on_iteration

        for value in list(range(13)):
            test_engine.state.metrics = {'val_loss': value}
            lr_on_plateau.epoch_complete(test_engine.state)
            print(test_engine.state.model.get_lr())

        assert test_engine.state.model.get_lr() == 1e-05
