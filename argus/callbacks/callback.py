from argus.engine import Events
from typing import Callable


class Callback:
    def attach(self, engine, handler_kwargs_dict=None):
        if handler_kwargs_dict is None:
            handler_kwargs_dict = dict()

        for key, event in Events.__members__.items():
            if hasattr(self, event.value):
                handler = getattr(self, event.value)
                if isinstance(handler, Callable):
                    handler_kwargs = handler_kwargs_dict.get(event.value, dict())
                    engine.add_event_handler(event, handler, **handler_kwargs)
                else:
                    raise TypeError


class FunctionCallback(Callback):
    def __init__(self, event: Events, handler):
        self.event = event
        self.handler = handler

    def attach(self, engine, *args, **kwargs):
        engine.add_event_handler(self.event, self.handler, *args, **kwargs)


def on_event(event):
    def wrap(func):
        callback = FunctionCallback(event, func)
        return callback
    return wrap


def on_start(func):
    callback = FunctionCallback(Events.START, func)
    return callback


def on_complete(func):
    callback = FunctionCallback(Events.COMPLETE, func)
    return callback


def on_epoch_start(func):
    callback = FunctionCallback(Events.EPOCH_START, func)
    return callback


def on_epoch_complete(func):
    callback = FunctionCallback(Events.EPOCH_COMPLETE, func)
    return callback


def on_iteration_start(func):
    callback = FunctionCallback(Events.ITERATION_START, func)
    return callback


def on_iteration_complete(func):
    callback = FunctionCallback(Events.ITERATION_COMPLETE, func)
    return callback
