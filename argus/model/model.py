import os
import numbers

import torch

from argus.model.build import BuildModel, MODEL_REGISTRY, cast_device
from argus.engine import Engine, Events
from argus.utils import deep_to, deep_detach, setup_logging, device_to_str
from argus.callbacks import Callback, on_epoch_complete
from argus.callbacks.logging import metrics_logging
from argus.metrics.metric import Metric, METRIC_REGISTRY
from argus.metrics.loss import Loss


def _attach_callbacks(engine, callbacks):
    if callbacks is not None:
        for callback in callbacks:
            if isinstance(callback, Callback):
                callback.attach(engine)
            else:
                raise TypeError


def _attach_metrics(engine, metrics, name_prefix=''):
    for metric in metrics:
        if isinstance(metric, str):
            if metric in METRIC_REGISTRY:
                metric = METRIC_REGISTRY[metric]()
            else:
                raise ValueError
        if isinstance(metric, Metric):
            metric.attach(engine, {
                Events.EPOCH_COMPLETE: {'name_prefix': name_prefix}
            })
        else:
            raise TypeError


class Model(BuildModel):
    def __init__(self, params):
        super().__init__(params)

    def prepare_batch(self, batch, device):
        input, target = batch
        input = deep_to(input, device, non_blocking=True)
        target = deep_to(target, device, non_blocking=True)
        return input, target

    def train_step(self, batch)-> dict:
        if not self.nn_module.training:
            self.nn_module.train()
        self.optimizer.zero_grad()
        input, target = self.prepare_batch(batch, self.device)
        prediction = self.nn_module(input)
        loss = self.loss(prediction, target)
        loss.backward()
        self.optimizer.step()

        prediction = deep_detach(prediction)
        target = deep_detach(target)
        return {
            'prediction': self.prediction_transform(prediction),
            'target': target,
            'loss': loss.item()
        }

    def val_step(self, batch) -> dict:
        if self.nn_module.training:
            self.nn_module.eval()
        with torch.no_grad():
            input, target = self.prepare_batch(batch, self.device)
            prediction = self.nn_module(input)
            loss = self.loss(prediction, target)
            return {
                'prediction': self.prediction_transform(prediction),
                'target': target,
                'loss': loss.item()
            }

    def fit(self,
            train_loader,
            val_loader=None,
            max_epochs=1,
            metrics=None,
            metrics_on_train=False,
            callbacks=None,
            val_callbacks=None):
        metrics = [] if metrics is None else metrics
        assert self.train_ready()
        setup_logging()

        train_engine = Engine(self.train_step, model=self, logger=self.logger)
        train_metrics = [Loss()] + metrics if metrics_on_train else [Loss()]
        _attach_metrics(train_engine, train_metrics, name_prefix='train_')
        metrics_logging.attach(train_engine, train=True)

        if val_loader is not None:
            self.validate(val_loader, metrics, val_callbacks)
            val_engine = Engine(self.val_step, model=self, logger=self.logger)
            _attach_metrics(val_engine, [Loss()] + metrics, name_prefix='val_')
            _attach_callbacks(val_engine, val_callbacks)

            @on_epoch_complete
            def validation_epoch(train_state, val_engine, val_loader):
                epoch = train_state.epoch
                val_state = val_engine.run(val_loader, epoch, epoch)
                train_state.metrics.update(val_state.metrics)

            validation_epoch.attach(train_engine, val_engine, val_loader)
            metrics_logging.attach(train_engine, train=False)

        _attach_callbacks(train_engine, callbacks)
        train_engine.run(train_loader, 1, max_epochs)

    def set_lr(self, lr):
        if self.train_ready():
            param_groups = self.optimizer.param_groups
            if isinstance(lr, (list, tuple)):
                lrs = list(lr)
                if len(lrs) != len(param_groups):
                    raise ValueError(f"Expected lrs length {len(param_groups)}, "
                                     f"got {len(lrs)}")
            elif isinstance(lr, numbers.Number):
                lrs = [lr] * len(param_groups)
            else:
                raise ValueError(f"Expected lr type 'list', 'tuple or number, "
                                 f"got {type(lr)}")
            for lr, param_group in zip(lrs, param_groups):
                param_group['lr'] = lr
        else:
            raise AttributeError

    def get_lr(self):
        lrs = []
        for param_group in self.optimizer.param_groups:
            lrs.append(param_group['lr'])
        if len(lrs) == 1:
            return lrs[0]
        return lrs

    def save(self, file_path):
        nn_module = self.get_nn_module()
        state = {
            'model_name': self.__class__.__name__,
            'params': self.params,
            'nn_state_dict': deep_to(nn_module.state_dict(), 'cpu')
        }
        torch.save(state, file_path)
        self.logger.info(f"Model saved to '{file_path}'")

    def validate(self, val_loader, metrics=None, callbacks=None):
        metrics = [] if metrics is None else metrics
        assert self.train_ready()
        val_engine = Engine(self.val_step, model=self, logger=self.logger)
        _attach_metrics(val_engine, [Loss()] + metrics, name_prefix='val_')
        _attach_callbacks(val_engine, callbacks)
        metrics_logging.attach(val_engine, train=False, print_epoch=False)
        return val_engine.run(val_loader).metrics

    def predict(self, input):
        assert self.predict_ready()
        with torch.no_grad():
            if self.nn_module.training:
                self.nn_module.eval()
            input = deep_to(input, self.device)
            prediction = self.nn_module(input)
            prediction = self.prediction_transform(prediction)
            return prediction


def load_model(file_path, device=None):
    if os.path.isfile(file_path):
        state = torch.load(file_path)

        if state['model_name'] in MODEL_REGISTRY:
            params = state['params']
            if device is not None:
                device = cast_device(device)
                device = device_to_str(device)
                params['device'] = device

            model_class = MODEL_REGISTRY[state['model_name']]
            model = model_class(params)
            nn_state_dict = deep_to(state['nn_state_dict'], model.device)

            model.get_nn_module().load_state_dict(nn_state_dict)
            model.nn_module.eval()
            return model
        else:
            raise ImportError(f"Model '{state['model_name']}' not found in scope")
    else:
        raise FileNotFoundError(f"No state found at {file_path}")
