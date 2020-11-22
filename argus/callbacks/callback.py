"""Base class for Callbacks.
"""

from typing import Callable

from argus.utils import inheritors
from argus.engine import Events, EventEnum


class Callback:
    """Base callback class. All callbacks classes should inherit from this
    class.

    A callback may execute actions on the start and the end of the whole
    training process, each epoch or iteration, as well as any other custom
    events.

    The actions should be specified within corresponding methods that take the
    :class:`argus.engine.State` as input:

    * ``start`` : triggered when the training is started.
    * ``complete`` : triggered when the training is completed.
    * ``epoch_start`` : triggered when the epoch is started.
    * ``epoch_complete`` : triggered when the epoch is ended.
    * ``iteration_start`` : triggered when an iteration is started.
    * ``iteration_complete`` : triggered when the iteration is ended.
    * ``catch_exception`` : triggered on catching of an exception.

    Example:

        A simple custom callback which stops training after the specified time:

        .. code-block:: python

            from time import time

            from argus.engine import State
            from argus.callbacks.callback import Callback

            class TimerCallback(Callback):
                def __init__(self, time_limit: int):
                    self.time_limit = time_limit
                    self.start_time = 0

                def start(self, state: State):
                    self.start_time = time()

                def iteration_complete(self, state: State):
                    if time() - self.start_time > self.time_limit:
                        state.stopped = True
                        state.logger.info(f"Run out of time {self.time_limit} sec, "
                                          f"{(state.epoch + 1) * (state.iteration + 1)} "
                                          f"iterations performed!")

        Example of creating custom events you can find
        `here <https://github.com/lRomul/argus/blob/master/examples/custom_events.py>`_.

    Raises:
        TypeError: Attribute is not callable.

    """

    def attach(self, engine, handler_kwargs_dict=None):
        if handler_kwargs_dict is None:
            handler_kwargs_dict = dict()

        for event_enum in inheritors(EventEnum):
            for key, event in event_enum.__members__.items():
                if hasattr(self, event.value):
                    handler = getattr(self, event.value)
                    if isinstance(handler, Callable):
                        handler_kwargs = handler_kwargs_dict.get(event, dict())
                        engine.add_event_handler(event, handler, **handler_kwargs)
                    else:
                        raise TypeError(f"Attribute {event.value} is not callable.")


class FunctionCallback(Callback):
    def __init__(self, event: EventEnum, handler):
        self.event = event
        self.handler = handler

    def attach(self, engine, *args, **kwargs):
        engine.add_event_handler(self.event, self.handler, *args, **kwargs)


def on_event(event):
    def wrap(func):
        return FunctionCallback(event, func)
    return wrap


def on_start(func):
    return FunctionCallback(Events.START, func)


def on_complete(func):
    return FunctionCallback(Events.COMPLETE, func)


def on_epoch_start(func):
    return FunctionCallback(Events.EPOCH_START, func)


def on_epoch_complete(func):
    return FunctionCallback(Events.EPOCH_COMPLETE, func)


def on_iteration_start(func):
    return FunctionCallback(Events.ITERATION_START, func)


def on_iteration_complete(func):
    return FunctionCallback(Events.ITERATION_COMPLETE, func)


def on_catch_exception(func):
    return FunctionCallback(Events.CATCH_EXCEPTION, func)
