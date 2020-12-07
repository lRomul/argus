import pytest
import logging

from argus.callbacks import EarlyStopping


class TestEarlyStopping:
    @pytest.mark.parametrize("patience, better",
                             [(1, 'auto'), (3, 'min'), (9, 'auto')])
    def test_increasing_decreasing_seq(self, test_engine, patience, better):
        early_stopping = EarlyStopping(monitor='val_loss',
                                       patience=patience,
                                       better='auto')
        early_stopping.attach(test_engine)

        test_engine.state.logger = logging.getLogger("test_early_stopping")
        test_engine.state.stopped = False

        increasing_seq = list(range(patience))
        decreasing_seq = increasing_seq[::-1]

        early_stopping.start(test_engine.state)
        for value in decreasing_seq:
            test_engine.state.metrics = {'val_loss': value}
            early_stopping.epoch_complete(test_engine.state)
            print(test_engine.state.stopped)
            assert early_stopping.wait == 0
            assert early_stopping.best_value == value
            assert not test_engine.state.stopped

        best_value = early_stopping.best_value

        for num, value in enumerate(increasing_seq, 1):
            test_engine.state.metrics = {'val_loss': value}
            early_stopping.epoch_complete(test_engine.state)
            print(test_engine.state.stopped)
            assert early_stopping.wait == num
            assert early_stopping.best_value == best_value
            if num == len(increasing_seq):
                assert test_engine.state.stopped
            else:
                assert not test_engine.state.stopped

    def test_metric_not_found(self, test_engine):
        early_stopping = EarlyStopping(monitor='val_loss',
                                       patience=1,
                                       better='min')
        early_stopping.attach(test_engine)

        test_engine.state.metrics = {'val_qwerty': 0.1}
        with pytest.raises(ValueError):
            early_stopping.epoch_complete(test_engine.state)

    def test_decreasing_with_spikes(self, test_engine):
        early_stopping = EarlyStopping(monitor='val_loss',
                                       patience=2,
                                       better='auto')
        early_stopping.attach(test_engine)
        test_engine.state.stopped = False

        decreasing_seq = list(range(30))[::-1]
        for i in range(0, len(decreasing_seq), 2):
            decreasing_seq[i] = 100
        assert not test_engine.state.stopped
