import os
import math
import shutil
import warnings

from argus.callbacks.callback import Callback


class Checkpoint(Callback):
    def __init__(self,
                 dir_path='',
                 file_format='model-{epoch:03d}-{train_loss:.3f}.pth',
                 max_saves=None,
                 period=1):
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
        self.epochs_since_last_save = 0

    def get_file_path(self, engine):
        format_state = {'epoch': engine.state.epoch, **engine.state.metrics}
        file_name = self.file_format.format(**format_state)
        file_path = os.path.join(self.dir_path, file_name)
        return file_path

    def save_checkpoint(self, engine):
        self.epochs_since_last_save += 1
        if self.epochs_since_last_save >= self.period:
            self.epochs_since_last_save = 0

            file_path = self.get_file_path(engine)
            engine.model.save(file_path)
            last_model_path = os.path.join(os.path.dirname(file_path), 'model-last.pth')
            shutil.copy(file_path, last_model_path)
            self.saved_files_paths.append(file_path)

            if self.max_saves is not None:
                if len(self.saved_files_paths) > self.max_saves:
                    old_file_path = self.saved_files_paths.pop(0)
                    if os.path.exists(old_file_path):
                        os.remove(old_file_path)
                        engine.logger.info(f"Model removed '{old_file_path}'")

    def epoch_complete(self, engine):
        self.save_checkpoint(engine)


class MonitorCheckpoint(Checkpoint):
    def __init__(self,
                 dir_path='',
                 file_format='model-{epoch:03d}-{monitor:.3f}.pth',
                 max_saves=None,
                 period=1,
                 monitor='val_loss',
                 mode='min'):
        super().__init__(dir_path=dir_path,
                         file_format=file_format,
                         max_saves=max_saves,
                         period=period)
        self.monitor = monitor
        self.mode = mode

        if self.mode == 'min':
            self.better = lambda a, b: a < b
            self.best_value = math.inf
        elif self.mode == 'max':
            self.better = lambda a, b: b > a
            self.best_value = -math.inf

    def get_file_path(self, engine):
        format_state = {'epoch': engine.state.epoch,
                        'monitor': engine.state.metrics[self.monitor],
                        **engine.state.metrics}
        file_name = self.file_format.format(**format_state)
        file_path = os.path.join(self.dir_path, file_name)
        return file_path

    def epoch_complete(self, engine):
        assert self.monitor in engine.state.metrics,\
            f"Monitor '{self.monitor}' metric not found in state"
        current_value = engine.state.metrics[self.monitor]
        if self.better(current_value, self.best_value):
            self.best_value = current_value
            self.save_checkpoint(engine)
