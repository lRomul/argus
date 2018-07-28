import argus
from argus.engine import State


@argus.callbacks.on_epoch_complete
def train_loss_logging(state: State):
    train_loss = state.metrics.get('train_loss', None)
    message = f"Train - Epoch: {state.epoch}, train_loss: {train_loss}"
    state.logger.info(message)


@argus.callbacks.on_epoch_complete
def val_metrics_logging(state: State, print_epoch=True):
    if print_epoch:
        train_epoch = state.epoch
        message = [f"Validation - Epoch: {train_epoch}"]
    else:
        message = ["Validation"]
    for metric_name, metric_value in state.metrics.items():
        if metric_name == 'train_loss':
            continue
        message.append(f"{metric_name}: {metric_value}")
    state.logger.info(", ".join(message))
