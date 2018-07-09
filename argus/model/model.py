import torch

from argus.model.build import BuildModel
from argus.engine import Engine
from argus.utils import to_device


class Model(BuildModel):
    def __init__(self, params):
        super().__init__(params)

    @staticmethod
    def _prepare_batch(batch, device):
        inp, trg = batch
        return to_device(inp, device), to_device(trg, device)

    def _train_step(self, batch):
        self.model.train()
        self.optimizer.zero_grad()
        inp, trg = self._prepare_batch(batch, self.device)
        pred = self.nn_module(inp)
        loss = self.loss(pred, trg)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def _val_step(self, batch):
        self.model.eval()
        with torch.no_grad():
            inp, trg = self._prepare_batch(batch, self.device)
            pred = self.nn_module(inp)
            return pred, trg

    def fit(self, train_loader, val_loader=None, max_epochs=None, handlers=None):
        pass

    def set_lr(self, lr):
        pass

    def save(self, file_path):
        pass

    def validate(self, val_loader):
        pass

    def predict(self, input_):
        pass
