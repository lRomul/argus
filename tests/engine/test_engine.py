import pytest

from argus.engine import State, Engine, Events, EventEnum
from argus.model.model import _attach_callbacks
from argus.callbacks import Callback


def test_state_update(linear_argus_model_instance):
    state = State(qwerty=42,
                  model=linear_argus_model_instance)
    assert state.qwerty == 42
    state.update(asdf=12)
    assert state.asdf == 12


class TestEngineMethods:
    def test_add_event_handler(self, linear_argus_model_instance):
        def some_function():
            pass
        engine = Engine(
            some_function,
            model=linear_argus_model_instance
        )
        assert len(engine.event_handlers[Events.START]) == 0
        engine.add_event_handler(Events.START, some_function)
        assert len(engine.event_handlers[Events.START]) == 1
        assert engine.event_handlers[Events.START][0][0] is some_function

        with pytest.raises(TypeError):
            engine.add_event_handler(42, some_function)

    @pytest.mark.parametrize("event", [e for e in Events])
    def test_raise_event(self, event, engine):
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
        assert len(engine.event_handlers[event]) == 0
        engine.add_event_handler(event, call_args_storage,
                                 4, 8, 15, 16, 23, 42, qwerty="qwerty")
        engine.raise_event(event)

        assert call_args_storage.args == (4, 8, 15, 16, 23, 42)
        assert call_args_storage.kwargs == {"qwerty": "qwerty"}

        with pytest.raises(TypeError):
            engine.raise_event(None)

    def test_run(self, linear_argus_model_instance):
        class StepStorage:
            def __init__(self):
                self.batch_lst = []
                self.state = None

            def reset(self):
                self.batch_lst = []
                self.state = None

            def step_method(self, batch, state):
                self.batch_lst.append(batch)
                self.state = state

        step_storage = StepStorage()

        data_loader = [4, 8, 15, 16, 23, 42]
        engine = Engine(
            step_storage.step_method,
            model=linear_argus_model_instance
        )
        state = engine.run(data_loader, start_epoch=0, end_epoch=3)

        assert step_storage.batch_lst == data_loader * 3
        assert state.epoch == 3
        assert state.iteration == len(data_loader)

        def stop_function(state):
            state.stopped = True

        step_storage.reset()
        engine.add_event_handler(Events.EPOCH_COMPLETE, stop_function)
        state = engine.run(data_loader, start_epoch=0, end_epoch=3)
        assert step_storage.batch_lst == data_loader
        assert state.epoch == 1
        assert state.iteration == len(data_loader)

        step_storage.reset()
        engine.add_event_handler(Events.ITERATION_COMPLETE, stop_function)
        state = engine.run(data_loader, start_epoch=0, end_epoch=3)
        assert step_storage.batch_lst == [data_loader[0]]
        assert state.iteration == 0

        class CustomException(Exception):
            pass

        def exception_function(state):
            raise CustomException

        step_storage.reset()
        engine.add_event_handler(Events.START, exception_function)
        with pytest.raises(CustomException):
            engine.run(data_loader, start_epoch=0, end_epoch=3)
        assert step_storage.batch_lst == []
        assert engine.state.iteration == 0
        assert engine.state.epoch == 0

    def test_custom_events(self, linear_argus_model_instance):
        class CustomEvents(EventEnum):
            STEP_START = 'step_start'
            STEP_COMPLETE = 'step_complete'

        def step_function(batch, state):
            state.step_output = batch
            state.engine.raise_event(CustomEvents.STEP_START)
            state.step_output += 1
            state.engine.raise_event(CustomEvents.STEP_COMPLETE)

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
        engine = Engine(
            step_function,
            model=linear_argus_model_instance
        )
        _attach_callbacks(engine, [callback])
        engine.run(data_loader)
        assert callback.start_storage == data_loader
        assert callback.end_storage == [d + 1 for d in data_loader]
