from enum import Enum

from argus.build_model import BuildModel


class Events(Enum):
    EPOCH_STARTED = "epoch_started"
    EPOCH_COMPLETED = "epoch_completed"
    STARTED = "started"
    COMPLETED = "completed"
    ITERATION_STARTED = "iteration_started"
    ITERATION_COMPLETED = "iteration_completed"
    EXCEPTION_RAISED = "exception_raised"


class State(object):
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


class Model(BuildModel):
    def __init__(self, params):
        super().__init__(params)

    def fit(self, train_loader, val_loader, max_epochs=None, handlers=None):
        pass

    def set_lr(self, lr):
        pass

    def save_model(self, file_path):
        pass

    def validate(self, val_loader):
        pass

    def predict(self, input_):
        pass
