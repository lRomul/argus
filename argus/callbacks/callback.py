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


def on_event(event):
    def wrap(func):
        callback = Callback()
        setattr(callback, event.value, func)
        return callback
    return wrap


def on_start(func):
    callback = Callback()
    callback.start = func
    return callback


def on_complete(func):
    callback = Callback()
    callback.complete = func
    return callback


def on_epoch_start(func):
    callback = Callback()
    callback.epoch_start = func
    return callback


def on_epoch_complete(func):
    callback = Callback()
    callback.epoch_complete = func
    return callback


def on_iteration_start(func):
    callback = Callback()
    callback.iteration_start = func
    return callback


def on_iteration_complete(func):
    callback = Callback()
    callback.iteration_complete = func
    return callback
