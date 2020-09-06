import pytest
from pathlib import Path

from argus import load_model
from argus.callbacks import Checkpoint, MonitorCheckpoint


class TestCheckpoints:
    def test_checkpoint(self, tmpdir, engine, check_weights):
        model = engine.state.model
        path = Path(tmpdir.join("path/to/checkpoints/"))
        checkpoint = Checkpoint(dir_path=path, max_saves=None, period=1,
                                file_format='model-{epoch:03d}-{val_loss:.6f}.pth',
                                save_after_exception=True)
        checkpoint.attach(engine)
        checkpoint.start(engine.state)

        engine.state.epoch = 12
        engine.state.metrics = {'val_loss': 0.42}
        checkpoint.epoch_complete(engine.state)
        expected_path = path / 'model-012-0.420000.pth'
        assert expected_path.exists()
        loaded_model = load_model(expected_path)
        assert loaded_model.params == model.params
        assert check_weights(model, loaded_model)

        engine.state.epoch = 24
        engine.state.metrics = {'val_loss': 0.12}
        checkpoint.epoch_complete(engine.state)
        expected_path = path / 'model-024-0.120000.pth'
        assert expected_path.exists()
        loaded_model = load_model(expected_path)
        assert check_weights(model, loaded_model)

        checkpoint.catch_exception(engine.state)
        expected_path = path / 'model-after-exception.pth'
        assert expected_path.exists()
        loaded_model = load_model(expected_path)
        assert check_weights(model, loaded_model)

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
