"""Callbacks for logging argus model training process.
"""
import os
import csv
import logging
from datetime import datetime
from typing import Optional, Union, List, IO

from argus import types
from argus.engine import State
from argus.callbacks.callback import Callback, on_epoch_complete


def _format_lr_to_str(lr: Union[float, List[float]],
                      precision: int = 5) -> str:
    if isinstance(lr, (list, tuple)):
        str_lrs = [f'{l:.{precision}g}' for l in lr]
        str_lr = "[" + ", ".join(str_lrs) + "]"
    else:
        str_lr = f'{lr:.{precision}g}'
    return str_lr


@on_epoch_complete
def default_logging(state: State):
    message = f"{state.phase} - epoch: {state.epoch}"

    if state.phase == 'train':
        lr = state.model.get_lr()
        message += f', lr: {_format_lr_to_str(lr)}'

    prefix = f"{state.phase}_" if state.phase else ''
    for metric_name, metric_value in state.metrics.items():
        if metric_name.startswith(prefix):
            message += f", {metric_name}: {metric_value:.7g}"
    state.logger.info(message)


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
            or rewrite it. Defaults to False.

    """

    def __init__(self,
                 file_path: types.Path,
                 create_dir: bool = True,
                 formatter: str = '[%(asctime)s][%(levelname)s]: %(message)s',
                 append: bool = False):
        self.file_path = file_path
        self.create_dir = create_dir
        self.formatter = logging.Formatter(formatter)
        self.append = append
        self.file_handler: Optional[logging.FileHandler] = None

    def start(self, state: State):
        if self.create_dir:
            dir_path = os.path.dirname(self.file_path)
            if dir_path:
                if not os.path.exists(dir_path):
                    os.makedirs(dir_path, exist_ok=True)
        if not self.append and os.path.exists(self.file_path):
            os.remove(self.file_path)
        self.file_handler = logging.FileHandler(self.file_path)
        self.file_handler.setFormatter(self.formatter)
        state.logger.addHandler(self.file_handler)

    def complete(self, state: State):
        if self.file_handler is not None:
            state.logger.removeHandler(self.file_handler)

    def catch_exception(self, state: State):
        self.complete(state)


class LoggingToCSV(Callback):
    """Write the argus model training progress into a CSV file.

    It logs all losses and metrics values during training into a .csv file
    for for further analysis or visualization.

    Args:
        file_path (str): Path to the .csv logging file.
        create_dir (bool, optional): Create the directory for the logging
            file if it does not exist. Defaults to True.
        separator (str, optional): Values separator character to use.
            Defaults to ','.
        write_header (bool, optional): Write the column headers.
            Defaults to True.
        append (bool, optional):Append the log file if it already exists
            or rewrite it. Defaults to False.

    """

    def __init__(self,
                 file_path: types.Path,
                 create_dir: bool = True,
                 separator: str = ',',
                 write_header: bool = True,
                 append: bool = False):
        self.file_path = file_path
        self.separator = separator
        self.write_header = write_header
        self.append = append
        self.csv_file: Optional[IO] = None
        self.create_dir = create_dir

    def start(self, state: State):
        file_mode = 'a' if self.append else 'w'
        if self.create_dir:
            dir_path = os.path.dirname(self.file_path)
            if dir_path:
                if not os.path.exists(dir_path):
                    os.makedirs(dir_path, exist_ok=True)
        self.csv_file = open(self.file_path, file_mode, newline='')

    def epoch_complete(self, state: State):
        if self.csv_file is None:
            return
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
