from enum import Enum
from typing import Callable


class Events(Enum):
    START = "start"
    COMPLETE = "complete"
    EPOCH_START = "epoch_start"
    EPOCH_COMPLETE = "epoch_complete"
    ITERATION_START = "iteration_start"
    ITERATION_COMPLETE = "iteration_complete"
    CATCH_EXCEPTION = "catch_exception"


class State(object):
    def __init__(self, **kwargs):
        self.iteration = None
        self.epoch = None
        self.model = None
        self.data_loader = None
        self.logger = None
        self.exception = None
        self.engine = None

        self.batch = None
        self.step_output = None

        self.metrics = dict()
        self.stopped = True

        self.update(**kwargs)

    def update(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


class Engine(object):
    def __init__(self, step_function: Callable, **kwargs):
        self.event_handlers = {event: [] for event in Events.__members__.values()}
        self.step_function = step_function
        self.state = State(
            step_function=step_function,
            engine=self,
            **kwargs
        )

    def add_event_handler(self, event: Events, handler: Callable, *args, **kwargs):
        self.event_handlers[event].append((handler, args, kwargs))

    def raise_event(self, event: Events):
        if not isinstance(event, Events):
            raise TypeError(f"Event should be 'argus.engine.Events' enum")

        if event in self.event_handlers:
            for handler, args, kwargs in self.event_handlers[event]:
                handler(self.state, *args, **kwargs)

    def run(self, data_loader, start_epoch=0, end_epoch=1):
        self.state.update(data_loader=data_loader,
                          epoch=start_epoch,
                          iteration=0,
                          stopped=False)

        try:
            self.raise_event(Events.START)
            while self.state.epoch < end_epoch and not self.state.stopped:
                self.state.iteration = 0
                self.state.metrics = dict()
                self.raise_event(Events.EPOCH_START)

                for batch in data_loader:
                    self.state.batch = batch
                    self.state.iteration += 1
                    self.raise_event(Events.ITERATION_START)
                    self.state.step_output = self.step_function(batch, self.state)
                    self.raise_event(Events.ITERATION_COMPLETE)
                    self.state.step_output = None
                    if self.state.stopped:
                        break

                self.raise_event(Events.EPOCH_COMPLETE)
                self.state.epoch += 1

            self.raise_event(Events.COMPLETE)

        except BaseException as exception:
            if self.state.logger is not None:
                self.state.logger.exception(exception)

            self.state.exception = exception
            self.raise_event(Events.CATCH_EXCEPTION)
            raise exception
        finally:
            self.state.stopped = True

        return self.state
