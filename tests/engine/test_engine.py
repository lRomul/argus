import pytest

import argus
from argus.engine import State, Engine, Events, EventEnum
from argus.engine.engine import _init_step_method
from argus.model.model import _attach_callbacks
from argus.callbacks import Callback


def test_state_update(linear_argus_model_instance):
    state = State(linear_argus_model_instance.test_step,
                  qwerty=42)
    assert state.qwerty == 42
    state.update(asdf=12)
    assert state.asdf == 12


def test_init_step_method(test_engine):
    state = test_engine.state
    train_step = state.model.train_step
    step_method, model, phase = _init_step_method(train_step)
    assert phase == 'train'
    assert model is state.model
    assert step_method is train_step

    val_step = state.model.val_step
    step_method, model, phase = _init_step_method(val_step)
    assert phase == 'val'
    assert model is state.model
    assert step_method is val_step

    test_step = state.model.test_step
    step_method, model, phase = _init_step_method(test_step)
    assert phase == 'test'
    assert model is state.model
    assert step_method is test_step

    _, _, phase = _init_step_method(state.model.get_lr)
    assert phase == 'get_lr'

    with pytest.raises(TypeError):
        _init_step_method(lambda x: x)
    with pytest.raises(TypeError):
        _init_step_method(state.update)


class TestEngineMethods:
    def test_add_event_handler(self, test_engine):
        def some_function():
            pass
        assert len(test_engine.event_handlers[Events.START]) == 0
        test_engine.add_event_handler(Events.START, some_function)
        assert len(test_engine.event_handlers[Events.START]) == 1
        assert test_engine.event_handlers[Events.START][0][0] is some_function

        with pytest.raises(TypeError):
            test_engine.add_event_handler(42, some_function)

    @pytest.mark.parametrize("event", [e for e in Events])
    def test_raise_event(self, event, test_engine):
        class CallArgsStorage:
            def __init__(self):
                self.state = None
                self.args = None
                self.kwargs = None

            def __call__(self, state, *args, **kwargs):
                self.state = state
                self.args = args
                self.kwargs = kwargs

        call_args_storage = CallArgsStorage()
        assert len(test_engine.event_handlers[event]) == 0
        test_engine.add_event_handler(event, call_args_storage,
                                      4, 8, 15, 16, 23, 42, qwerty="qwerty")
        test_engine.raise_event(event)

        assert call_args_storage.args == (4, 8, 15, 16, 23, 42)
        assert call_args_storage.kwargs == {"qwerty": "qwerty"}

        with pytest.raises(TypeError):
            test_engine.raise_event(None)

    def test_run(self, test_engine):
        storage_model = test_engine.state.model
        storage_model.reset()

        data_loader = [4, 8, 15, 16, 23, 42]
        state = test_engine.run(data_loader, start_epoch=0, end_epoch=3)

        assert storage_model.batch_lst == data_loader * 3
        assert state.epoch == 3
        assert state.iteration == len(data_loader)

        def stop_function(state):
            state.stopped = True

        storage_model.reset()
        test_engine.add_event_handler(Events.EPOCH_COMPLETE, stop_function)
        state = test_engine.run(data_loader, start_epoch=0, end_epoch=3)
        assert storage_model.batch_lst == data_loader
        assert state.epoch == 1
        assert state.iteration == len(data_loader)

        storage_model.reset()
        test_engine.add_event_handler(Events.ITERATION_COMPLETE, stop_function)
        state = test_engine.run(data_loader, start_epoch=0, end_epoch=3)
        assert storage_model.batch_lst == [data_loader[0]]
        assert state.iteration == 0

        class CustomException(Exception):
            pass

        def exception_function(state):
            raise CustomException

        storage_model.reset()
        test_engine.add_event_handler(Events.START, exception_function)
        with pytest.raises(CustomException):
            test_engine.run(data_loader, start_epoch=0, end_epoch=3)
        assert storage_model.batch_lst == []
        assert test_engine.state.iteration == 0
        assert test_engine.state.epoch == 0

    def test_custom_events(self, linear_net_class):
        class CustomEvents(EventEnum):
            STEP_START = 'step_start'
            STEP_COMPLETE = 'step_complete'

        class CustomEventsModel(argus.Model):
            nn_module = linear_net_class

            def count_step(self, batch, state):
                state.step_output = batch
                state.engine.raise_event(CustomEvents.STEP_START)
                state.step_output += 1
                state.engine.raise_event(CustomEvents.STEP_COMPLETE)

        model = CustomEventsModel({
            'nn_module': {
                'in_features': 10,
                'out_features': 1,
            },
            'optimizer': None,
            'loss': None
        })

        class CustomCallback(Callback):
            def __init__(self):
                self.start_storage = []
                self.end_storage = []

            def step_start(self, state):
                self.start_storage.append(state.step_output)

            def step_complete(self, state):
                self.end_storage.append(state.step_output)

        data_loader = [4, 8, 15, 16, 23, 42]
        callback = CustomCallback()
        engine = Engine(model.count_step)
        _attach_callbacks(engine, [callback])
        engine.run(data_loader)
        assert callback.start_storage == data_loader
        assert callback.end_storage == [d + 1 for d in data_loader]
