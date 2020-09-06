import pytest
from pathlib import Path

import torch
from torch import nn

from argus import load_model
from argus.callbacks import Checkpoint, MonitorCheckpoint


def checkpoint_step_epoch(checkpoint, engine, epoch, val_loss):
    engine.state.epoch = epoch
    engine.state.metrics = {'val_loss': val_loss}
    nn.init.xavier_uniform_(engine.state.model.nn_module.fc.weight)
    checkpoint.epoch_complete(engine.state)


def check_weights(model, loaded_model):
    nn_state_dict = model.nn_module.state_dict()
    for layer_name, weight in loaded_model.nn_module.state_dict().items():
        assert layer_name in nn_state_dict
        assert torch.all(nn_state_dict[layer_name] == weight)
    return True


def check_checkpoint(path, engine, epoch, val_loss,
                     file_format='model-{epoch:03d}-{val_loss:.6f}.pth'):
    expected_path = path / file_format.format(epoch=epoch, val_loss=val_loss)
    assert expected_path.exists()
    loaded_model = load_model(expected_path)
    assert loaded_model.params == engine.state.model.params
    assert check_weights(engine.state.model, loaded_model)
    return True


class TestCheckpoints:
    def test_checkpoint(self, tmpdir, engine):
        model = engine.state.model
        path = Path(tmpdir.join("path/to/checkpoints/"))
        checkpoint = Checkpoint(dir_path=path, max_saves=None, period=1,
                                file_format='model-{epoch:03d}-{val_loss:.6f}.pth',
                                save_after_exception=True)
        checkpoint.attach(engine)
        checkpoint.start(engine.state)

        checkpoint_step_epoch(checkpoint, engine, 12, 0.42)
        assert check_checkpoint(path, engine, 12, 0.42)

        engine.state.epoch = 24
        engine.state.metrics = {'val_loss': 0.12}
        checkpoint_step_epoch(checkpoint, engine, 24, 0.12)
        assert check_checkpoint(path, engine, 24, 0.12)

        nn.init.xavier_uniform_(model.nn_module.fc.weight)
        checkpoint.catch_exception(engine.state)
        assert check_checkpoint(path, engine, None, None,
                                file_format='model-after-exception.pth')

        assert len(list(path.glob('*.pth'))) == 3

    def test_checkpoint_exceptions(self, tmpdir, engine, recwarn):
        path = Path(tmpdir.join("path/to/exception_checkpoints/"))
        with pytest.raises(ValueError):
            Checkpoint(dir_path=path, max_saves=-3)

        path.mkdir(parents=True)
        Checkpoint(dir_path=path)
        assert len(recwarn) == 1
        warn = recwarn.pop()
        assert f"Directory '{path}' already exists" == str(warn.message)

        with pytest.raises(ValueError):
            MonitorCheckpoint(dir_path=path, monitor='qwerty')

        checkpoint = MonitorCheckpoint(dir_path=path, monitor='train_loss')
        checkpoint.attach(engine)
        with pytest.raises(ValueError):
            checkpoint.epoch_complete(engine.state)

    @pytest.mark.parametrize('max_saves', [None, 1, 3, 12])
    def test_max_saves(self, tmpdir, engine, max_saves):
        path = Path(tmpdir.join("path/to/max_saves_checkpoints/"))
        checkpoint = Checkpoint(dir_path=path, max_saves=max_saves, period=1,
                                file_format='model-{epoch:03d}-{val_loss:.6f}.pth')
        checkpoint.attach(engine)
        checkpoint.start(engine.state)

        num_epochs = 29
        for epoch in range(1, num_epochs + 1):
            checkpoint_step_epoch(checkpoint, engine, epoch, 1 / epoch)
            check_checkpoint(path, engine, epoch, 1 / epoch)

        assert len(list(path.glob('*.pth'))) == max_saves \
            if max_saves is not None else num_epochs

    def test_monitor_checkpoint(self, tmpdir, engine):
        path = Path(tmpdir.join("path/to/monitor_checkpoints/"))
        checkpoint = MonitorCheckpoint(dir_path=path, max_saves=3, monitor='val_loss')
        checkpoint.attach(engine)
        checkpoint.start(engine.state)

        decreasing_seq = list(range(30))[::-1]
        for i in range(1, len(decreasing_seq), 2):
            decreasing_seq[i] = 100

        for epoch, val_loss in enumerate(decreasing_seq, 1):
            checkpoint_step_epoch(checkpoint, engine, epoch, val_loss)
            expected_path = path / f'model-{epoch:03d}-{val_loss:.6f}.pth'
            if val_loss != 100:
                assert check_checkpoint(path, engine, epoch, val_loss)
            else:
                assert not expected_path.exists()

        assert len(list(path.glob('*.pth'))) == 3
