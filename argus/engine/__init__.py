
from argus.engine.engine import Engine, Events


def validation_logging(train_engine, val_engine, val_loader):
    val_engine.run(val_loader)
    metrics = val_engine.state.metrics
    avg_loss = metrics['val_loss']
    val_engine.logger.info(f"Validation - Epoch: {train_engine.state.epoch}  Avg loss: {avg_loss}")
