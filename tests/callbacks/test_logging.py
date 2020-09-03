import pytest

from argus.engine import Engine
from argus.callbacks.logging import (
    _format_lr_to_str,
    metrics_logging,
    LoggingToFile,
    LoggingToCSV
)


@pytest.fixture(scope='function')
def state(linear_argus_model_instance):
    engine = Engine(lambda batch, state: batch,
                    model=linear_argus_model_instance)
    return engine.state


def read_file(path):
    with open(path, 'r') as file:
        return file.readlines()


@pytest.mark.parametrize("lr, precision, str_lr", [
    (0.001, 5, "0.001"),
    (0.0000039, 6, "3.9e-06"),
    (0.99999, 1, "1"),
    ([0.000234, 0.00235], 2, "[0.00023, 0.0024]")
])
def test_format_lr_to_str(lr, precision, str_lr):
    result = _format_lr_to_str(lr, precision=precision)
    assert result == str_lr


def test_metrics_logging(state):
    class MockLogger:
        def __init__(self):
            self.message = ''

        def info(self, message):
            self.message = message

    state.logger = MockLogger()

    state.metrics = {"train_loss": 0.1}
    state.logger = MockLogger()
    metrics_logging.handler(state=state, train=True, print_epoch=False)
    assert state.logger.message == 'Train, LR: 0.01, train_loss: 0.1'

    state.metrics = {"val_loss": 0.01}
    metrics_logging.handler(state=state, train=False, print_epoch=False)
    assert state.logger.message == 'Validation, val_loss: 0.01'

    state.metrics = {"val_loss": 0.01, "val_accuracy": 0.42}
    state.epoch = 12
    metrics_logging.handler(state=state, train=False, print_epoch=True)
    assert state.logger.message == 'Validation - Epoch: 12, val_loss: 0.01, val_accuracy: 0.42'

    state.metrics = {"train_loss": 0.01, "train_accuracy": 0.42}
    state.epoch = 42
    metrics_logging.handler(state=state, train=True, print_epoch=True)
    assert state.logger.message == 'Train - Epoch: 42, LR: 0.01, train_loss: 0.01, train_accuracy: 0.42'


def test_logging_to_file(tmpdir, state):
    logger = state.model.logger
    state.logger = logger
    path = str(tmpdir.mkdir('logs').join("log.txt"))

    with open(path, 'w') as file:
        file.write('qwerty')

    logging_to_file = LoggingToFile(path, append=False, create_dir=True,
                                    formatter='[%(levelname)s]: %(message)s')

    logging_to_file.start(state)
    assert [logging_to_file.file_handler is h for h in logger.handlers]
    metrics_logging.handler(state=state, train=True, print_epoch=False)
    log_messages = ['[INFO]: Train, LR: 0.01\n']
    assert read_file(path) == log_messages
    state.epoch = 12
    state.metrics = {'val_loss': 0.123}
    metrics_logging.handler(state=state, train=False, print_epoch=True)
    log_messages += ['[INFO]: Validation - Epoch: 12, val_loss: 0.123\n']
    assert read_file(path) == log_messages
    logging_to_file.complete(state)
    assert not any(logging_to_file.file_handler is h for h in logger.handlers)

    logging_to_file = LoggingToFile(path, append=True, create_dir=True,
                                    formatter='[%(levelname)s]: %(message)s')
    logging_to_file.start(state)
    state.epoch = 42
    state.metrics = {'val_loss': 0.246}
    metrics_logging.handler(state=state, train=False, print_epoch=True)
    log_messages += ['[INFO]: Validation - Epoch: 42, val_loss: 0.246\n']
    assert read_file(path) == log_messages

    logging_to_file.start(state)
    assert any(logging_to_file.file_handler is h for h in logger.handlers)
    logging_to_file.catch_exception(state)
    assert not any(logging_to_file.file_handler is h for h in logger.handlers)


def test_logging_to_csv():
    pass
