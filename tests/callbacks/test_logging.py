import pytest

from argus.engine import Engine
from argus.callbacks.logging import (
    _format_lr_to_str,
    metrics_logging,
    LoggingToFile,
    LoggingToCSV
)


@pytest.mark.parametrize("lr, precision, str_lr", [
    (0.001, 5, "0.001"),
    (0.0000039, 6, "3.9e-06"),
    (0.99999, 1, "1"),
    ([0.000234, 0.00235], 2, "[0.00023, 0.0024]")
])
def test_format_lr_to_str(lr, precision, str_lr):
    result = _format_lr_to_str(lr, precision=precision)
    assert result == str_lr


def test_metrics_logging(linear_argus_model_instance):
    class MockLogger:
        def __init__(self):
            self.message = ''

        def info(self, message):
            self.message = message

    engine = Engine(lambda batch, state: batch,
                    model=linear_argus_model_instance,
                    logger=MockLogger())

    engine.state.metrics = {"train_loss": 0.1}
    engine.state.logger = MockLogger()
    metrics_logging.handler(state=engine.state, train=True, print_epoch=False)
    assert engine.state.logger.message == 'Train, LR: 0.01, train_loss: 0.1'

    engine.state.metrics = {"val_loss": 0.01}
    metrics_logging.handler(state=engine.state, train=False, print_epoch=False)
    assert engine.state.logger.message == 'Validation, val_loss: 0.01'

    engine.state.metrics = {"val_loss": 0.01, "val_accuracy": 0.42}
    engine.state.epoch = 12
    metrics_logging.handler(state=engine.state, train=False, print_epoch=True)
    assert engine.state.logger.message == 'Validation - Epoch: 12, val_loss: 0.01, val_accuracy: 0.42'

    engine.state.metrics = {"train_loss": 0.01, "train_accuracy": 0.42}
    engine.state.epoch = 42
    metrics_logging.handler(state=engine.state, train=True, print_epoch=True)
    assert engine.state.logger.message == 'Train - Epoch: 42, LR: 0.01, train_loss: 0.01, train_accuracy: 0.42'


def test_logging_to_file():
    pass


def test_logging_to_csv():
    pass
