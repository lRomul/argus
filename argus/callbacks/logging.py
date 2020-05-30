"""Callbacks for logging argus model training process.
"""
import os
import csv
import logging
from datetime import datetime

from argus.engine import State
from argus.callbacks.callback import Callback, on_epoch_complete


def _format_lr_to_str(lr, precision=5):
    if isinstance(lr, (list, tuple)):
        lr = [f'{l:.{precision}g}' for l in lr]
        lr = "[" + ", ".join(lr) + "]"
    else:
        lr = f'{lr:.{precision}g}'
    return lr


@on_epoch_complete
def metrics_logging(state: State, train=False, print_epoch=True):
    if train:
        epoch_name = 'Train'
        prefix = 'train_'
    else:
        epoch_name = 'Validation'
        prefix = 'val_'

    if print_epoch:
        train_epoch = state.epoch
        message = [f"{epoch_name} - Epoch: {train_epoch}"]
    else:
        message = [epoch_name]

    if train:
        lr = state.model.get_lr()
        lr = _format_lr_to_str(lr)
        message.append(f'LR: {lr}')

    for metric_name, metric_value in state.metrics.items():
        if not metric_name.startswith(prefix):
            continue
        message.append(f"{metric_name}: {metric_value:.7g}")
    state.logger.info(", ".join(message))


class LoggingToFile(Callback):
    """Write the argus model training progress into a file.

    It adds a standard Python logger to log all losses and metrics values
    during training. The logger is used to output other messages, like info
    from callbacks and errors.

    Args:
        file_path (str): Path to the logging file.
        create_dir (bool, optional): Create the directory for the logging
            file if it does not exist. Defaults to True.
        formatter (str, optional): Standard Python logging formatter to
            format the log messages. Defaults to
            '%(asctime)s %(levelname)s %(message)s'.
        append (bool, optional): Append the log file if it already exists
            or rewrite it. Defaults to True.

    """

    def __init__(self, file_path,
                 create_dir=True,
                 formatter='%(asctime)s %(levelname)s %(message)s',
                 append=True):
        self.file_path = file_path
        self.create_dir = create_dir
        self.formatter = logging.Formatter(formatter)
        self.append = append
        self.file_handler = None

    def start(self, state: State):
        if self.create_dir:
            dir_path = os.path.dirname(self.file_path)
            if dir_path:
                if not os.path.exists(dir_path):
                    os.mkdir(dir_path)
        if not self.append and os.path.exists(self.file_path):
            os.remove(self.file_path)
        self.file_handler = logging.FileHandler(self.file_path)
        self.file_handler.setFormatter(self.formatter)
        state.logger.addHandler(self.file_handler)

    def complete(self, state: State):
        state.logger.removeHandler(self.file_handler)

    def catch_exception(self, state: State):
        self.complete(state)


class LoggingToCSV(Callback):
    """Write the argus model training progress into a CSV file.

    It logs all losses and metrics values during training into a .csv file
    for for further analysis or visualization.

    Args:
        file_path (str): Path to the .csv logging file.
        separator (str, optional): Values separator character to use.
            Defaults to ','.
        write_header (bool, optional): Write the column headers.
            Defaults to True.
        append (bool, optional):Append the log file if it already exists
            or rewrite it. Defaults to False.

    """

    def __init__(self, file_path,
                 separator=',',
                 write_header=True,
                 append=False):
        self.file_path = file_path
        self.separator = separator
        self.write_header = write_header
        self.append = append
        self.csv_file = None

    def start(self, state: State):
        if self.append:
            file_mode = 'a'
        else:
            file_mode = 'w'

        self.csv_file = open(self.file_path, file_mode, newline='')

    def epoch_complete(self, state: State):
        lr = state.model.get_lr()
        fields = {
            'time': str(datetime.now()),
            'epoch': state.epoch,
            'lr': _format_lr_to_str(lr),
            **state.metrics
        }
        writer = csv.DictWriter(self.csv_file,
                                fieldnames=fields,
                                delimiter=self.separator)

        if self.write_header:
            writer.writeheader()
            self.write_header = False

        writer.writerow(fields)
        self.csv_file.flush()

    def complete(self, state: State):
        if self.csv_file is not None:
            self.csv_file.close()

    def catch_exception(self, state: State):
        self.complete(state)
