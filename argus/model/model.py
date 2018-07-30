import torch
import os
import logging

from argus.model.build import BuildModel, MODEL_REGISTRY
from argus.engine import Engine, Events
from argus.utils import to_device, setup_logging
from argus.callbacks import Callback, on_epoch_complete
from argus.callbacks.logging import metrics_logging
from argus.metrics.metric import Metric, METRIC_REGISTRY
from argus.metrics.loss import Loss, TrainLoss


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
        self.logger = logging.getLogger(__name__)

    def prepare_batch(self, batch, device):
        inp, trg = batch
        return to_device(inp, device), to_device(trg, device)

    def train_step(self, batch)-> dict:
        self.nn_module.train()
        self.optimizer.zero_grad()
        inp, trg = self.prepare_batch(batch, self.device)
        pred = self.nn_module(inp)
        loss = self.loss(pred, trg)
        loss.backward()
        self.optimizer.step()
        return {
            'prediction': pred.detach(),
            'target': trg.detach(),
            'loss': loss.item()
        }

    def val_step(self, batch) -> dict:
        self.nn_module.eval()
        with torch.no_grad():
            inp, trg = self.prepare_batch(batch, self.device)
            pred = self.nn_module(inp)
            return {
                'prediction': pred,
                'target': trg
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
        train_loss = TrainLoss()
        train_loss.attach(train_engine)
        if metrics_on_train:
            _attach_metrics(train_engine, metrics, name_prefix='train_')
        metrics_logging.attach(train_engine, train=True)

        if val_loader is not None:
            self.validate(val_loader, metrics, val_callbacks)
            val_engine = Engine(self.val_step, model=self, logger=self.logger)
            val_loss = Loss(self.loss)
            _attach_metrics(val_engine, [val_loss] + metrics, name_prefix='val_')
            _attach_callbacks(val_engine, val_callbacks)

            @on_epoch_complete
            def validation_epoch(train_state, val_engine, val_loader):
                val_state = val_engine.run(val_loader)
                train_state.metrics.update(val_state.metrics)

            validation_epoch.attach(train_engine, val_engine, val_loader)
            metrics_logging.attach(train_engine, train=False)

        _attach_callbacks(train_engine, callbacks)
        train_engine.run(train_loader, max_epochs)

    def set_lr(self, lr):
        if self.train_ready():
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
        else:
            raise AttributeError

    def save(self, file_path):
        state = {
            'model_name': self.__class__.__name__,
            'params': self.params,
            'nn_state_dict': to_device(self.nn_module.state_dict(), 'cpu')
        }
        torch.save(state, file_path)
        self.logger.info(f"Model saved to {file_path}")

    def validate(self, val_loader, metrics=None, callbacks=None):
        metrics = [] if metrics is None else metrics
        assert self.train_ready()
        val_engine = Engine(self.val_step, model=self, logger=self.logger)
        val_loss = Loss(self.loss)
        _attach_metrics(val_engine, [val_loss] + metrics, name_prefix='val_')
        _attach_callbacks(val_engine, callbacks)
        metrics_logging.attach(val_engine, train=False, print_epoch=False)
        return val_engine.run(val_loader).metrics

    def predict(self, input):
        assert self.predict_ready()
        raise NotImplemented


def load_model(file_path, device=None):
    if os.path.isfile(file_path):
        state = torch.load(file_path)

        if state['model_name'] in MODEL_REGISTRY:
            params = state['params']
            if device is not None:
                device = torch.device(device).type
                params['device'] = device

            model_class = MODEL_REGISTRY[state['model_name']]
            model = model_class(params)
            nn_state_dict = to_device(state['nn_state_dict'], model.device)
            model.nn_module.load_state_dict(nn_state_dict)
            return model
        else:
            raise ImportError(f"Model '{state['model_name']}' not found in scope")
    else:
        raise FileNotFoundError(f"No state found at {file_path}")
