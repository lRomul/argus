from argus.engine.engine import Engine, Events


def validation(train_engine, val_engine, val_loader):
    val_state = val_engine.run(val_loader)

    if train_engine.state is None:
        train_epoch = "before train"
    else:
        train_epoch = train_engine.state.epoch
        train_engine.state.metrics.update(val_state.metrics)
    message = [f"Validation - Epoch: {train_epoch}"]
    for metric_name, metric_value in val_state.metrics.items():
        message.append(f"{metric_name}: {metric_value}")
    val_engine.logger.info(", ".join(message))


def train_loss_logging(train_engine):
    train_loss = train_engine.state.metrics.get('train_loss', None)
    message = f"Train - Epoch: {train_engine.state.epoch}, train_loss: {train_loss}"
    train_engine.logger.info(message)
