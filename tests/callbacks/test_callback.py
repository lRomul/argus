import pytest

import argus
from argus.engine import Engine, Events, EventEnum
from argus.model.model import _attach_callbacks
from argus.callbacks import Callback


class CustomEvents(EventEnum):
    STEP_START = 'step_start'
    STEP_COMPLETE = 'step_complete'


class CustomTestCallback(Callback):
    def __init__(self):
        self.start_count = 0
        self.complete_count = 0
        self.epoch_start_count = 0
        self.epoch_complete_count = 0
        self.iteration_start_count = 0
        self.iteration_complete_count = 0
        self.catch_exception_count = 0
        self.step_start_count = 0
        self.step_complete_count = 0

    def start(self, state):
        self.start_count += 1

    def complete(self, state):
        self.complete_count += 1

    def epoch_start(self, state):
        self.epoch_start_count += 1

    def epoch_complete(self, state):
        self.epoch_complete_count += 1

    def iteration_start(self, state):
        self.iteration_start_count += 1

    def iteration_complete(self, state):
        self.iteration_complete_count += 1

    def catch_exception(self, state):
        self.catch_exception_count += 1

    def step_start(self, state):
        self.step_start_count += 1

    def step_complete(self, state):
        self.step_complete_count += 1


@pytest.fixture(scope='function')
def custom_test_callback():
    return CustomTestCallback()


class StepStorage:
    def __init__(self):
        self.batch_lst = []
        self.state = None

    def reset(self):
        self.batch_lst = []
        self.state = None

    def step_method(self, batch, state):
        state.engine.raise_event(CustomEvents.STEP_START)
        self.batch_lst.append(batch)
        self.state = state
        state.engine.raise_event(CustomEvents.STEP_COMPLETE)


@pytest.fixture(scope='function')
def step_storage():
    return StepStorage()


class TestCallbacks:
    @pytest.mark.parametrize("n_epochs", [0, 1, 2, 3, 16])
    def test_attach_callback(self, n_epochs, custom_test_callback, step_storage):
        callback = custom_test_callback
        engine = Engine(step_storage.step_method)
        callback.attach(engine)
        data_loader = [4, 8, 15, 16, 23, 42]
        engine.run(data_loader, start_epoch=0, end_epoch=n_epochs)
        assert callback.start_count == 1
        assert callback.complete_count == 1
        assert callback.epoch_start_count == n_epochs
        assert callback.epoch_complete_count == n_epochs
        assert callback.iteration_start_count == n_epochs * len(data_loader)
        assert callback.iteration_complete_count == n_epochs * len(data_loader)
        assert callback.catch_exception_count == 0
        assert callback.step_start_count == n_epochs * len(data_loader)
        assert callback.step_complete_count == n_epochs * len(data_loader)
        assert step_storage.batch_lst == n_epochs * data_loader
        assert step_storage.state is engine.state if n_epochs \
            else step_storage.state is None

    def test_attach_non_callable_handler(self, custom_test_callback):
        engine = Engine(lambda x: x)
        custom_test_callback.start = 'no_a_method_or_function'
        with pytest.raises(TypeError):
            custom_test_callback.attach(engine)

    def test_attach_not_a_callback(self):
        engine = Engine(lambda x: x)
        with pytest.raises(TypeError):
            _attach_callbacks(engine, [None])
        with pytest.raises(TypeError):
            _attach_callbacks(engine, [engine])


class TestDecoratorCallbacks:
    def test_on_event(self, step_storage):
        @argus.callbacks.on_event(Events.START)
        def some_function(state):
            state.special_secret = 42

        engine = Engine(step_storage.step_method)
        some_function.attach(engine)
        data_loader = [4, 8, 15, 16, 23, 42]
        state = engine.run(data_loader)
        assert state.special_secret == 42

    def test_on_decorators(self, step_storage):
        @argus.callbacks.on_start
        def on_start_function(state):
            state.call_count = 1
            state.on_start_flag = True

        @argus.callbacks.on_complete
        def on_complete_function(state):
            state.call_count += 1
            state.on_complete_flag = True

        @argus.callbacks.on_epoch_start
        def on_epoch_start_function(state):
            state.call_count += 1
            state.on_epoch_start_flag = True

        @argus.callbacks.on_epoch_complete
        def on_epoch_complete_function(state):
            state.call_count += 1
            state.on_epoch_complete_flag = True

        @argus.callbacks.on_iteration_start
        def on_iteration_start_function(state):
            state.call_count += 1
            state.on_iteration_start_flag = True

        @argus.callbacks.on_iteration_complete
        def on_iteration_complete_function(state):
            state.call_count += 1
            state.on_iteration_complete_flag = True

        @argus.callbacks.on_catch_exception
        def on_catch_exception_function(state):
            state.call_count += 1
            state.on_catch_exception_flag = True

        engine = Engine(step_storage.step_method)
        _attach_callbacks(engine, [
            on_start_function,
            on_complete_function,
            on_epoch_start_function,
            on_epoch_complete_function,
            on_iteration_start_function,
            on_iteration_complete_function,
            on_catch_exception_function
        ])
        data_loader = [4, 8, 15, 16, 23, 42]
        state = engine.run(data_loader, start_epoch=0, end_epoch=3)
        assert state.call_count == len(data_loader) * 3 * 2 + 3 * 2 + 2
        assert state.on_start_flag
        assert state.on_complete_flag
        assert state.on_epoch_start_flag
        assert state.on_epoch_complete_flag
        assert state.on_iteration_start_flag
        assert state.on_iteration_complete_flag
        assert not hasattr(state, 'on_catch_exception_flag')

        class CustomException(Exception):
            pass

        @argus.callbacks.on_start
        def on_start_raise_exception(state):
            raise CustomException

        on_start_raise_exception.attach(engine)
        with pytest.raises(CustomException):
            engine.run(data_loader, start_epoch=0, end_epoch=3)
        assert engine.state.on_catch_exception_flag
