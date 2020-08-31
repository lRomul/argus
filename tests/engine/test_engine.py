import pytest
import logging

from argus.engine import State, Engine, Events


def test_state_update():
    state = State(qwerty=42)
    assert state.qwerty == 42
    state.update(asdf=12)
    assert state.asdf == 12


class TestEngineMethods:
    def test_add_event_handler(self):
        def some_function(): pass
        engine = Engine(some_function)
        assert len(engine.event_handlers[Events.START]) == 0
        engine.add_event_handler(Events.START, some_function)
        assert len(engine.event_handlers[Events.START]) == 1
        assert engine.event_handlers[Events.START][0][0] is some_function

        with pytest.raises(TypeError):
            engine.add_event_handler(42, some_function)

    @pytest.mark.parametrize("event", [e for e in Events])
    def test_raise_event(self, event):
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
        engine = Engine(lambda x: x)
        assert len(engine.event_handlers[event]) == 0
        engine.add_event_handler(event, call_args_storage,
                                 4, 8, 15, 16, 23, 42, qwerty="qwerty")
        engine.raise_event(event)

        assert call_args_storage.args == (4, 8, 15, 16, 23, 42)
        assert call_args_storage.kwargs == {"qwerty": "qwerty"}

        with pytest.raises(TypeError):
            engine.raise_event(None)

    def test_run(self):
        class StepStorage:
            def __init__(self):
                self.batch_lst = []
                self.state = None

            def reset(self):
                self.batch_lst = []
                self.state = None

            def step_function(self, batch, state):
                self.batch_lst.append(batch)
                self.state = state

        step_storage = StepStorage()
        data_loader = [4, 8, 15, 16, 23, 42]
        engine = Engine(step_storage.step_function,
                        logger=logging.getLogger(__name__))
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
        assert state.iteration == 1

        class MyException(Exception):
            pass

        def exception_function(state):
            raise MyException

        step_storage.reset()
        engine.add_event_handler(Events.START, exception_function)
        with pytest.raises(MyException):
            engine.run(data_loader, start_epoch=0, end_epoch=3)
        assert step_storage.batch_lst == []
        assert engine.state.iteration == 0
        assert engine.state.epoch == 0
