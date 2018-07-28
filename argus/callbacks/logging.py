import argus
from argus.engine import State


@argus.callbacks.on_epoch_complete
def metrics_logging(state: State, train=False, print_epoch=True):
    if train:
        epoch_name = 'Train'
        prefix = 'train_'
    else:
        epoch_name = 'Validation'
        prefix = 'val_'

    if print_epoch:
        train_epoch = state.epoch
        message = [f"{epoch_name} - Epoch: {train_epoch}"]
    else:
        message = [epoch_name]
    for metric_name, metric_value in state.metrics.items():
        if not metric_name.startswith(prefix):
            continue
        message.append(f"{metric_name}: {metric_value:.8f}")
    state.logger.info(", ".join(message))
