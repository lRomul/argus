import logging
from enum import Enum
from typing import Callable


class Events(Enum):
    START = "start"
    COMPLETE = "complete"
    EPOCH_START = "epoch_start"
    EPOCH_COMPLETE = "epoch_complete"
    ITERATION_START = "iteration_start"
    ITERATION_COMPLETE = "iteration_complete"


class State(object):
    def __init__(self, **kwargs):
        self.iteration = None
        self.epoch = None
        self.step_output = None
        self.metrics = dict()

        for key, value in kwargs.items():
            setattr(self, key, value)


class Engine(object):

    def __init__(self, step_function: Callable):
        self.event_handlers = {event: [] for event in Events.__members__.values()}
        self.step_function = step_function
        self.state = State()
        self.stopped = True
        self.logger = logging.getLogger(__name__)

    def add_event_handler(self, event: Events, handler: Callable, *args, **kwargs):
        self.event_handlers[event].append((handler, args, kwargs))

    def raise_event(self, event: Events):
        assert isinstance(event, Events)

        if event in self.event_handlers:
            for func, args, kwargs in self.event_handlers[event]:
                func(self, *args, **kwargs)

    def run(self, data_loader, max_epochs=1):
        self.state = State(iteration=0,
                           epoch=0,
                           data_loader=data_loader,
                           max_epochs=max_epochs)
        self.stopped = False

        try:
            self.raise_event(Events.START)
            while self.state.epoch < max_epochs and not self.stopped:
                self.state.iteration = 0
                self.state.epoch += 1
                self.raise_event(Events.EPOCH_START)

                for batch in data_loader:
                    self.state.batch = batch
                    self.state.iteration += 1
                    self.raise_event(Events.ITERATION_START)
                    self.state.step_output = self.step_function(batch)
                    self.raise_event(Events.ITERATION_COMPLETE)
                    if self.stopped:
                        break

                self.raise_event(Events.EPOCH_COMPLETE)

            self.raise_event(Events.COMPLETE)

        except Exception as e:
            self.logger.exception(e)
        finally:
            self.stopped = True

        return self.state
