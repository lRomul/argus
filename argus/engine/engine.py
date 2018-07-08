from enum import Enum
from argus.utils import get_logger


class Events(Enum):
    START = "start"
    COMPLETE = "complete"
    EPOCH_START = "epoch_start"
    EPOCH_COMPLETE = "epoch_complete"
    ITERATION_START = "iteration_start"
    ITERATION_COMPLETE = "iteration_complete"


class State(object):
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


class Engine(object):

    def __init__(self, step_function, logger=None):
        self.event_handlers = {event: [] for event in Events.__members__.values()}
        self.step_function = step_function
        self.state = None
        self.stopped = True

        if logger is None:
            self.logger = get_logger()
        else:
            self.logger = logger

    def add_event_handler(self, event, handler, *args, **kwargs):
        pass

    def raise_event(self, event):
        pass

    def run(self, data_loader, max_epochs=1):
        pass
