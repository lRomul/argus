import torch

from argus import Model
from argus.engine import EventEnum, State
from argus.callbacks import Callback
from argus.utils import deep_to, deep_detach, AverageMeter

from mnist import get_data_loaders, Net


class CustomEvents(EventEnum):
    STEP_START = 'step_start'
    STEP_COMPLETE = 'step_complete'


class CustomEventModel(Model):
    nn_module = Net
    optimizer = torch.optim.SGD
    loss = torch.nn.CrossEntropyLoss

    def train_step(self, batch, state: State) -> dict:
        state.input_batch = batch[0]
        state.engine.raise_event(CustomEvents.STEP_START)
        state.batch = None

        self.train()
        self.optimizer.zero_grad()
        input, target = deep_to(batch, device=self.device, non_blocking=True)
        prediction = self.nn_module(input)
        loss = self.loss(prediction, target)
        loss.backward()
        self.optimizer.step()

        prediction = deep_detach(prediction)
        target = deep_detach(target)
        prediction = self.prediction_transform(prediction)

        state.prediction = prediction
        state.engine.raise_event(CustomEvents.STEP_COMPLETE)
        state.prediction = None

        return {
            'prediction': prediction,
            'target': target,
            'loss': loss.item()
        }


class StepCallback(Callback):
    def __init__(self):
        self.image_mean = AverageMeter()
        self.prediction_mean = AverageMeter()

    def epoch_start(self, state):
        self.image_mean.reset()
        self.prediction_mean.reset()

    def step_start(self, state):
        mean = state.input_batch.mean().item()
        self.image_mean.update(mean)

    def step_complete(self, state):
        mean = state.prediction.mean().item()
        self.prediction_mean.update(mean)

    def epoch_complete(self, state):
        state.logger.info(f"Input image mean value: {self.image_mean.average}")
        state.logger.info(f"Prediction mean value: {self.prediction_mean.average}")


if __name__ == "__main__":
    _, data_loader = get_data_loaders(128, 128)

    params = {
        'nn_module': {'n_classes': 10, 'p_dropout': 0.1},
        'optimizer': {'lr': 0.01},
        'device': 'cuda'
    }
    model = CustomEventModel(params)

    step_callback = StepCallback()
    model.fit(data_loader,
              num_epochs=10,
              callbacks=[step_callback])
