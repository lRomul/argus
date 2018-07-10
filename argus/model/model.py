import torch
import logging

from argus.model.build import BuildModel
from argus.engine import Engine, Events, validation_logging
from argus.utils import to_device, setup_logging
from argus.metrics.loss import Loss


class Model(BuildModel):
    def __init__(self, params):
        super().__init__(params)
        self.logger = logging.getLogger(__name__)

    @staticmethod
    def _prepare_batch(batch, device):
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

    def fit(self, train_loader, val_loader=None, max_epochs=1, metrics=None):
        setup_logging()
        if metrics is None:
            metrics = dict()

        train_engine = Engine(self._train_step)

        if val_loader is not None:
            val_engine = Engine(self._val_step)

            if 'val_loss' not in metrics:
                    metrics['val_loss'] = Loss(self.loss)

            for name, metric in metrics.items():
                metric.attach(val_engine, name)

            train_engine.add_event_handler(Events.EPOCH_COMPLETE,
                                           validation_logging,
                                           val_engine,
                                           val_loader)

        train_engine.run(train_loader, max_epochs)

    def set_lr(self, lr):
        pass

    def save(self, file_path):
        pass

    def validate(self, val_loader):
        pass

    def predict(self, input):
        pass
