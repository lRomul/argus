from pathlib import Path

from torch.utils.data import TensorDataset, DataLoader

from argus import load_model
from argus.callbacks import (
    MonitorCheckpoint,
    EarlyStopping,
    ReduceLROnPlateau,
    LoggingToFile,
    LoggingToCSV
)


def test_pipeline(tmpdir, get_batch_function,
                  linear_argus_model_instance):
    model = linear_argus_model_instance
    experiment_dir = Path(tmpdir.join("path/to/pipeline_experiment/"))
    train_dataset = TensorDataset(*get_batch_function(batch_size=4096))
    val_dataset = TensorDataset(*get_batch_function(batch_size=512))
    train_loader = DataLoader(train_dataset, shuffle=True,
                              drop_last=True, batch_size=32)
    val_loader = DataLoader(val_dataset, shuffle=False, batch_size=64)

    monitor_checkpoint = MonitorCheckpoint(dir_path=experiment_dir,
                                           monitor='val_loss', max_saves=1)
    callbacks = [
        monitor_checkpoint,
        EarlyStopping(monitor='val_loss', patience=9),
        ReduceLROnPlateau(monitor='val_loss', factor=0.64, patience=3),
        LoggingToFile(experiment_dir / 'log.txt'),
        LoggingToCSV(experiment_dir / 'log.csv')
    ]

    model.fit(train_loader,
              val_loader=val_loader,
              num_epochs=100,
              callbacks=callbacks)

    val_loss = model.validate(val_loader)['val_loss']
    assert val_loss < 0.1

    model_paths = sorted(experiment_dir.glob('*.pth'))
    assert len(model_paths) == 1

    loaded_model = load_model(model_paths[0])
    loaded_val_loss = loaded_model.validate(val_loader)['val_loss']
    assert loaded_val_loss == monitor_checkpoint.best_value

    assert (experiment_dir / 'log.txt').exists()
    assert (experiment_dir / 'log.csv').exists()
