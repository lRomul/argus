import torch
import os
import logging

import argus
from argus.model.build import BuildModel, MODEL_REGISTRY
from argus.engine import Engine, Events
# from argus.engine import validation
from argus.utils import to_device, setup_logging
from argus.metrics import Metric
from argus.metrics.loss import Loss, TrainLoss
from argus.callbacks import Callback


def _attach_callbacks(engine, callbacks):
    if callbacks is not None:
        for callback in callbacks:
            if isinstance(callback, Callback):
                callback.attach(engine)
            else:
                raise TypeError


def _attach_metrics(engine, metrics):
    if metrics is not None:
        for metric in metrics:
            assert metric.name not in ['train_loss', 'val_loss']
            if isinstance(metric, Metric):
                metric.attach(engine)
            else:
                raise TypeError


@argus.callbacks.on_epoch_complete
def _train_loss_logging(engine):
    train_loss = engine.state.metrics.get('train_loss', None)
    message = f"Train - Epoch: {engine.state.epoch}, train_loss: {train_loss}"
    engine.logger.info(message)


@argus.callbacks.on_epoch_complete
def _val_metrics_logging(engine, print_epoch=True):
    if print_epoch:
        train_epoch = engine.state.epoch
        message = [f"Validation - Epoch: {train_epoch}"]
    else:
        message = ["Validation"]
    for metric_name, metric_value in engine.state.metrics.items():
        if metric_name == 'train_loss':
            continue
        message.append(f"{metric_name}: {metric_value}")
    engine.logger.info(", ".join(message))


class Model(BuildModel):
    def __init__(self, params):
        super().__init__(params)
        self.logger = logging.getLogger(__name__)

    def _prepare_batch(self, batch, device):
        inp, trg = batch
        return to_device(inp, device), to_device(trg, device)

    def _train_step(self, batch):
        self.nn_module.train()
        self.optimizer.zero_grad()
        inp, trg = self._prepare_batch(batch, self.device)
        pred = self.nn_module(inp)
        loss = self.loss(pred, trg)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def _val_step(self, batch):
        self.nn_module.eval()
        with torch.no_grad():
            inp, trg = self._prepare_batch(batch, self.device)
            pred = self.nn_module(inp)
            return pred, trg

    def fit(self,
            train_loader,
            val_loader=None,
            max_epochs=1,
            metrics=None,
            callbacks=None,
            val_callbacks=None):

        assert self.train_ready()
        setup_logging()
        train_engine = Engine(self._train_step)

        train_loss = TrainLoss('train_loss')
        train_loss.attach(train_engine)
        _train_loss_logging.attach(train_engine)

        if val_loader is not None:
            self.validate(val_loader, metrics, val_callbacks)

            val_engine = Engine(self._val_step)

            val_loss = Loss('val_loss', self.loss)
            val_loss.attach(val_engine)
            _attach_metrics(val_engine, metrics)

            @argus.callbacks.on_epoch_complete
            def validation_epoch(train_engine):
                val_state = val_engine.run(val_loader)
                train_engine.state.metrics.update(val_state.metrics)

            validation_epoch.attach(train_engine)
            _val_metrics_logging.attach(train_engine)
            _attach_callbacks(val_engine, val_callbacks)

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
        val_engine = Engine(self._val_step)

        val_loss = Loss('val_loss', self.loss)
        val_loss.attach(val_engine)
        _attach_metrics(val_engine, metrics)
        _attach_callbacks(val_engine, callbacks)
        _val_metrics_logging.attach(val_engine, {
            'epoch_complete': {'print_epoch': False}
        })
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
