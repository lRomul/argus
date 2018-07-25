import torch
import os
import logging

from argus.model.build import BuildModel, MODEL_REGISTRY
from argus.engine import Engine, Events
from argus.engine import validation, train_loss_logging
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
            train_callbacks=None,
            val_callbacks=None):

        assert self.train_ready()

        setup_logging()

        train_engine = Engine(self._train_step)

        train_loss = TrainLoss('train_loss')
        train_loss.attach(train_engine)
        train_engine.add_event_handler(Events.EPOCH_COMPLETE,
                                       train_loss_logging)

        if val_loader is not None:
            val_engine = Engine(self._val_step)

            val_loss = Loss('val_loss', self.loss)
            val_loss.attach(val_engine)

            if metrics is not None:
                for metric in metrics:
                    if isinstance(metric, Metric):
                        metric.attach(val_engine)
                    else:
                        raise TypeError

            validation(train_engine, val_engine, val_loader)
            train_engine.add_event_handler(Events.EPOCH_COMPLETE,
                                           validation,
                                           val_engine,
                                           val_loader)
            _attach_callbacks(val_engine, val_callbacks)

        _attach_callbacks(train_engine, train_callbacks)
        train_engine.run(train_loader, max_epochs)

    def set_lr(self, lr):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def save(self, file_path):
        state = {
            'model_name': self.__class__.__name__,
            'params': self.params,
            'nn_state_dict': to_device(self.nn_module.state_dict(), 'cpu')
        }
        torch.save(state, file_path)
        self.logger.info(f"Model saved to {file_path}")

    def validate(self, val_loader):
        raise NotImplemented

    def predict(self, input):
        assert self.predict_ready()
        raise NotImplemented


def load_model(file_path, device=None):
    if os.path.isfile(file_path):
        state = torch.load(file_path)

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
        raise FileNotFoundError(f"No state found at {file_path}")
