"""Base class for Callbacks.
"""

from types import FunctionType, MethodType
from typing import Optional, Callable, List

from argus.utils import inheritors
from argus.engine import Engine, Events, EventEnum


class Callback:
    """Base callback class. All callbacks classes should inherit from this
    class.

    A callback may execute actions on the start and the end of the whole
    training process, each epoch or iteration, as well as any other custom
    events.

    The actions should be specified within corresponding methods that take the
    :class:`argus.engine.State` as input:

    * ``start``: triggered when the training is started.
    * ``complete``: triggered when the training is completed.
    * ``epoch_start``: triggered when the epoch is started.
    * ``epoch_complete``: triggered when the epoch is ended.
    * ``iteration_start``: triggered when an iteration is started.
    * ``iteration_complete``: triggered when the iteration is ended.
    * ``catch_exception``: triggered on catching of an exception.

    Example:

        A simple custom callback which stops training after the specified time:

        .. code-block:: python

            from time import time
            from argus.engine import State
            from argus.callbacks.callback import Callback


            class TimerCallback(Callback):
                \"""Stop training after the specified time.

                Args:
                    time_limit (int): Time to run training in seconds.

                \"""

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

        You can find an example of creating custom events
        `here <https://github.com/lRomul/argus/blob/master/examples/custom_events.py>`_.

    Raises:
        TypeError: Attribute is not callable.

    """

    def attach(self, engine: Engine):
        """Attach callback to the :class:`argus.engine.Engine`.

        Args:
            engine (Engine): The engine to which the callback will be attached.

        """
        for event_enum in inheritors(EventEnum):
            for key, event in event_enum.__members__.items():
                if hasattr(self, event.value):
                    handler = getattr(self, event.value)
                    if isinstance(handler, (FunctionType, MethodType)):
                        engine.add_event_handler(event, handler)
                    else:
                        raise TypeError(f"Attribute {event.value} is not callable.")


class FunctionCallback(Callback):
    """Callback class for executing single function.

    Args:
        event (EventEnum): An event that will be associated with the
            handler.
        handler (Callable): A callable handler that will be executed on
            the event. The handler should take
            :class:`argus.engine.State` as the first argument.

    """

    def __init__(self, event: EventEnum, handler: Callable):
        self.event = event
        self.handler = handler

    def attach(self, engine: Engine, *args, **kwargs):
        """Attach callback to the :class:`argus.engine.Engine`.

        Args:
            engine (Engine): The engine to which the callback will be attached.
            *args: optional args arguments to be passed to the handler.
            **kwargs: optional kwargs arguments to be passed to the handler.

        """
        engine.add_event_handler(self.event, self.handler, *args, **kwargs)


def on_event(event: EventEnum) -> Callable:
    """Decorator for creating callback from function. The function will be
    executed when the event is triggered. The function should take
    :class:`argus.engine.State` as the first argument.

    Args:
        event (EventEnum): An event that will be associated with the
            function.

    Example:

        .. code-block:: python

            import argus
            from argus.engine import Events, State

            @argus.callbacks.on_event(Events.START)
            def start_callback(state: State):
                state.logger.info("Start training!")

            model.fit(train_loader,
                      val_loader=val_loader,
                      callbacks=[start_callback])

    """
    def wrap(func: Callable) -> FunctionCallback:
        return FunctionCallback(event, func)
    return wrap


def on_start(func: Callable) -> FunctionCallback:
    """Decorator for creating callback from function. The function will be
    executed when the `Events.START` is triggered. The function should take
    :class:`argus.engine.State` as the first argument.

    Example:

        .. code-block:: python

            import argus
            from argus.engine import State

            @argus.callbacks.on_start
            def start_callback(state: State):
                state.logger.info("Start training!")

            model.fit(train_loader,
                      val_loader=val_loader,
                      callbacks=[start_callback])

    """

    return FunctionCallback(Events.START, func)


def on_complete(func: Callable) -> FunctionCallback:
    """Decorator for creating callback from function. The function will be
    executed when the ``Events.COMPLETE`` is triggered. The function should
    take :class:`argus.engine.State` as the first argument.
    """
    return FunctionCallback(Events.COMPLETE, func)


def on_epoch_start(func: Callable) -> FunctionCallback:
    """Decorator for creating callback from function. The function will be
    executed when the ``Events.EPOCH_START`` is triggered. The function should
    take :class:`argus.engine.State` as the first argument.
    """
    return FunctionCallback(Events.EPOCH_START, func)


def on_epoch_complete(func: Callable) -> FunctionCallback:
    """Decorator for creating callback from function. The function will be
    executed when the ``Events.EPOCH_COMPLETE`` is triggered. The function
    should take :class:`argus.engine.State` as the first argument.
    """
    return FunctionCallback(Events.EPOCH_COMPLETE, func)


def on_iteration_start(func: Callable) -> FunctionCallback:
    """Decorator for creating callback from function. The function will be
    executed when the ``Events.ITERATION_START`` is triggered. The function
    should take :class:`argus.engine.State` as the first argument.
    """
    return FunctionCallback(Events.ITERATION_START, func)


def on_iteration_complete(func: Callable) -> FunctionCallback:
    """Decorator for creating callback from function. The function will be
    executed when the ``Events.ITERATION_COMPLETE`` is triggered. The function
    should take :class:`argus.engine.State` as the first argument.
    """
    return FunctionCallback(Events.ITERATION_COMPLETE, func)


def on_catch_exception(func: Callable) -> FunctionCallback:
    """Decorator for creating callback from function. The function will be
    executed when the ``Events.CATCH_EXCEPTION`` is triggered. The function
    should take :class:`argus.engine.State` as the first argument.
    """
    return FunctionCallback(Events.CATCH_EXCEPTION, func)


def attach_callbacks(engine: Engine, callbacks: Optional[List[Callback]]):
    """Attaches callbacks to the :class:`argus.engine.Engine`.

        Args:
            engine (Engine): The engine to which callbacks will be attached.
            callbacks (list of :class:`argus.callbacks.Callback`, optional):
                List of callbacks.

    """
    if callbacks is None:
        return
    for callback in callbacks:
        if isinstance(callback, Callback):
            callback.attach(engine)
        else:
            raise TypeError(f"Expected callback type {Callback}, "
                            f"got {type(callback)}")
