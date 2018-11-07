import os
import math
import shutil
import warnings

from argus.engine import State
from argus.callbacks.callback import Callback
from argus.metrics.metric import init_better


class Checkpoint(Callback):
    def __init__(self,
                 dir_path='',
                 file_format='model-{epoch:03d}-{train_loss:.6f}.pth',
                 max_saves=None,
                 period=1,
                 copy_last=False,
                 save_after_exception=False):
        assert max_saves is None or max_saves > 0

        self.dir_path = dir_path
        self.file_format = file_format
        self.max_saves = max_saves
        self.saved_files_paths = []
        if self.dir_path:
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)
            else:
                warnings.warn(f"Directory '{dir_path}' already exists")
        self.period = period
        self.copy_last = copy_last
        self.save_after_exception = save_after_exception
        self.epochs_since_last_save = 0

    def _format_file_path(self, state: State):
        format_state = {'epoch': state.epoch, **state.metrics}
        file_name = self.file_format.format(**format_state)
        file_path = os.path.join(self.dir_path, file_name)
        return file_path

    def start(self, state: State):
        self.epochs_since_last_save = 0
        self.saved_files_paths = []

    def save_checkpoint(self, state: State):
        self.epochs_since_last_save += 1
        if self.epochs_since_last_save >= self.period:
            self.epochs_since_last_save = 0

            file_path = self._format_file_path(state)
            state.model.save(file_path)
            if self.copy_last:
                last_model_path = os.path.join(self.dir_path, 'model-last.pth')
                shutil.copy(file_path, last_model_path)
            self.saved_files_paths.append(file_path)

            if self.max_saves is not None:
                if len(self.saved_files_paths) > self.max_saves:
                    old_file_path = self.saved_files_paths.pop(0)
                    if os.path.exists(old_file_path):
                        os.remove(old_file_path)
                        state.logger.info(f"Model removed '{old_file_path}'")

    def epoch_complete(self, state: State):
        self.save_checkpoint(state)

    def catch_exception(self, state: State):
        if self.save_after_exception:
            exception_model_path = os.path.join(self.dir_path,
                                                'model-after-exception.pth')
            state.model.save(exception_model_path)


class MonitorCheckpoint(Checkpoint):
    def __init__(self,
                 dir_path='',
                 file_format='model-{epoch:03d}-{monitor:.6f}.pth',
                 max_saves=None,
                 period=1,
                 copy_last=False,
                 save_after_exception=False,
                 monitor='val_loss',
                 better='auto'):
        assert monitor.startswith('val_') or monitor.startswith('train_')
        super().__init__(dir_path=dir_path,
                         file_format=file_format,
                         max_saves=max_saves,
                         period=period,
                         copy_last=copy_last,
                         save_after_exception=save_after_exception)
        self.monitor = monitor
        self.better, self.better_comp, self.best_value = init_better(better, monitor)

    def _format_file_path(self, state: State):
        format_state = {'epoch': state.epoch,
                        'monitor': state.metrics[self.monitor],
                        **state.metrics}
        file_name = self.file_format.format(**format_state)
        file_path = os.path.join(self.dir_path, file_name)
        return file_path

    def start(self, state: State):
        self.best_value = math.inf if self.better == 'min' else -math.inf

    def epoch_complete(self, state: State):
        assert self.monitor in state.metrics,\
            f"Monitor '{self.monitor}' metric not found in state"
        current_value = state.metrics[self.monitor]
        if self.better_comp(current_value, self.best_value):
            self.best_value = current_value
            self.save_checkpoint(state)
