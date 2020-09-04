import pytest
import logging

from argus.callbacks import EarlyStopping


class TestEarlyStopping:
    @pytest.mark.parametrize("patience, better",
                             [(1, 'auto'), (3, 'min'), (9, 'auto')])
    def test_increasing_decreasing_seq(self, engine, patience, better):
        early_stopping = EarlyStopping(monitor='val_loss',
                                       patience=patience,
                                       better='auto')
        early_stopping.attach(engine)

        engine.state.logger = logging.getLogger("test_early_stopping")
        engine.state.stopped = False

        increasing_seq = list(range(patience))
        decreasing_seq = increasing_seq[::-1]

        early_stopping.start(engine.state)
        for value in decreasing_seq:
            engine.state.metrics = {'val_loss': value}
            early_stopping.epoch_complete(engine.state)
            print(engine.state.stopped)
            assert early_stopping.wait == 0
            assert early_stopping.best_value == value
            assert not engine.state.stopped

        best_value = early_stopping.best_value

        for num, value in enumerate(increasing_seq, 1):
            engine.state.metrics = {'val_loss': value}
            early_stopping.epoch_complete(engine.state)
            print(engine.state.stopped)
            assert early_stopping.wait == num
            assert early_stopping.best_value == best_value
            if num == len(increasing_seq):
                assert engine.state.stopped
            else:
                assert not engine.state.stopped

    def test_metric_not_found(self, engine):
        early_stopping = EarlyStopping(monitor='val_loss',
                                       patience=1,
                                       better='min')
        early_stopping.attach(engine)

        engine.state.metrics = {'val_qwerty': 0.1}
        with pytest.raises(ValueError):
            early_stopping.epoch_complete(engine.state)

    def test_decreasing_with_spikes(self, engine):
        early_stopping = EarlyStopping(monitor='val_loss',
                                       patience=2,
                                       better='auto')
        early_stopping.attach(engine)
        engine.state.stopped = False

        decreasing_seq = list(range(30))[::-1]
        for i in range(0, len(decreasing_seq), 2):
            decreasing_seq[i] = 100
        assert not engine.state.stopped
