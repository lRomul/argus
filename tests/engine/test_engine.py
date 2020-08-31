import pytest

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
