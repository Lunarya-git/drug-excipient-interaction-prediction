"""Tests for early stopping."""

from __future__ import annotations

import pytest
from src.training.early_stopping import EarlyStopping


class TestEarlyStopping:
    def test_stops_when_no_improvement(self):
        stopper = EarlyStopping(monitor="val_loss", patience=3, min_epochs=0)
        for epoch in range(1, 10):
            should_stop = stopper.step(epoch, 1.0)  # constant loss
            if should_stop:
                assert epoch >= 3
                return
        pytest.fail("EarlyStopping never triggered")

    def test_does_not_stop_during_min_epochs(self):
        stopper = EarlyStopping(monitor="val_loss", patience=1, min_epochs=5)
        for epoch in range(1, 5):
            assert not stopper.step(epoch, 1.0 + epoch)  # worsening loss

    def test_resets_on_improvement(self):
        stopper = EarlyStopping(monitor="val_loss", patience=3, min_epochs=0)
        # Improve then stagnate
        stopper.step(1, 1.0)
        stopper.step(2, 0.9)  # improvement
        stopper.step(3, 0.9)  # no improvement
        stopper.step(4, 0.9)  # no improvement
        # After patience=3 from epoch 2, should stop at epoch 5 (wait=3)
        result = stopper.step(5, 0.9)
        assert result is True

    def test_val_ap_maximise(self):
        stopper = EarlyStopping(monitor="val_ap", patience=2, min_epochs=0)
        stopper.step(1, 0.5)
        stopper.step(2, 0.6)  # improvement
        stopper.step(3, 0.6)  # no improvement
        result = stopper.step(4, 0.6)  # should stop (wait == patience == 2)
        assert result is True

    def test_best_epoch_tracked(self):
        stopper = EarlyStopping(monitor="val_loss", patience=10, min_epochs=0)
        stopper.step(1, 1.0)
        stopper.step(2, 0.8)
        stopper.step(3, 0.9)
        assert stopper.best_epoch == 2

    def test_invalid_monitor_raises(self):
        with pytest.raises(ValueError):
            EarlyStopping(monitor="invalid_metric")
