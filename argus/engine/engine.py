import logging
from enum import Enum
from collections import defaultdict
from typing import Callable, Optional, Iterable, Dict, Any


class EventEnum(Enum):
    pass


class Events(EventEnum):
    START = "start"
    COMPLETE = "complete"
    EPOCH_START = "epoch_start"
    EPOCH_COMPLETE = "epoch_complete"
    ITERATION_START = "iteration_start"
    ITERATION_COMPLETE = "iteration_complete"
    CATCH_EXCEPTION = "catch_exception"


class State:
    def __init__(self, **kwargs):
        self.iteration: Optional[int] = None
        self.epoch: Optional[int] = None
        self.model: Optional['argus.model.Model'] = None
        self.data_loader: Optional[Iterable] = None
        self.logger: Optional[logging.Logger] = None
        self.exception: Optional[BaseException] = None
        self.engine: Optional[Engine] = None
        self.phase: str = ""

        self.batch: Any = None
        self.step_output: Any = None

        self.metrics: Dict[str, float] = dict()
        self.stopped: bool = True

        self.update(**kwargs)

    def update(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


class Engine:
    def __init__(self, step_function: Callable, **kwargs):
        self.event_handlers = defaultdict(list)
        self.step_function = step_function
        self.state = State(
            step_function=step_function,
            engine=self,
            **kwargs
        )

    def add_event_handler(self, event: EventEnum, handler: Callable, *args, **kwargs):
        if not isinstance(event, EventEnum):
            raise TypeError(f"Event should be 'argus.engine.EventEnum' enum")

        self.event_handlers[event].append((handler, args, kwargs))

    def raise_event(self, event: EventEnum):
        if not isinstance(event, EventEnum):
            raise TypeError(f"Event should be 'argus.engine.EventEnum' enum")

        if event in self.event_handlers:
            for handler, args, kwargs in self.event_handlers[event]:
                handler(self.state, *args, **kwargs)

    def run(self, data_loader: Iterable, start_epoch=0, end_epoch=1) -> State:
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
