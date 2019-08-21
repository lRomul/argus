import os
import csv
import logging

from argus.engine import State
from argus.callbacks.callback import Callback, on_epoch_complete


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
        message.append(f'LR: {state.model.get_lr():.5g}')

    for metric_name, metric_value in state.metrics.items():
        if not metric_name.startswith(prefix):
            continue
        message.append(f"{metric_name}: {metric_value:.7g}")
    state.logger.info(", ".join(message))


class LoggingToFile(Callback):
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
        fields = {
            'epoch': state.epoch,
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
