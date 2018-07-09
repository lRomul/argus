from enum import Enum
from argus.utils import get_logger
from typing import Callable


class Events(Enum):
    START = "start"
    COMPLETE = "complete"
    EPOCH_START = "epoch_start"
    EPOCH_COMPLETE = "epoch_complete"
    ITERATION_START = "iteration_start"
    ITERATION_COMPLETE = "iteration_complete"


class State(object):
    def __init__(self, data_loader, iteration, epoch, max_epochs, **kwargs):
        self.data_loader = data_loader
        self.iteration = iteration
        self.epoch = epoch
        self.max_epochs = max_epochs
        for k, v in kwargs.items():
            setattr(self, k, v)


class Engine(object):

    def __init__(self, step_function: Callable, logger=None):
        self.event_handlers = {event: [] for event in Events.__members__.values()}
        self.step_function = step_function
        self.state = None
        self.stopped = True

        if logger is None:
            self.logger = get_logger()
        else:
            self.logger = logger

    def add_event_handler(self, event: Events, handler: Callable, *args, **kwargs):
        self.event_handlers[event].append((handler, args, kwargs))

    def raise_event(self, event: Events, *event_args):
        if event in self.event_handlers:
            for func, args, kwargs in self.event_handlers[event]:
                func(self, *(event_args + args), **kwargs)

    def run(self, data_loader, max_epochs=1):
        self.state = State(data_loader=data_loader,
                           iteration=0,
                           epoch=0,
                           max_epochs=max_epochs,
                           metrics={})

        try:
            self.raise_event(Events.START)
            while self.state.epoch < max_epochs and not self.stopped:
                self.state.iteration = 0
                self.state.epoch += 1
                self.raise_event(Events.EPOCH_START)

                for batch in self.state.data_loader:
                    self.state.batch = batch
                    self.state.iteration += 1
                    self.raise_event(Events.ITERATION_STARTED)
                    self.state.output = self.step_function(self, batch)
                    self.raise_event(Events.ITERATION_COMPLETED)
                    if self.stopped:
                        break

                self.raise_event(Events.EPOCH_COMPLETE)

            self.raise_event(Events.COMPLETE)

        except Exception as e:
            self.logger.exception(e)

        return self.state
