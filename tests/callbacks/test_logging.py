import pytest

from argus.callbacks.logging import _format_lr_to_str


@pytest.mark.parametrize("lr, precision, str_lr", [
    (0.001, 5, "0.001"),
    (0.0000039, 6, "3.9e-06"),
    (0.99999, 1, "1"),
    ([0.000234, 0.00234], 2, "[0.00023, 0.0023]")
])
def test_format_lr_to_str(lr, precision, str_lr):
    result = _format_lr_to_str(lr, precision=precision)
    assert result == str_lr


def test_metrics_logging():
    pass


def test_logging_to_file():
    pass


def test_logging_to_csv():
    pass
