"""Early stopping with an inner validation split.

The inner validation fold is carved out of the TRAINING fold so the CV
held-out fold is never used for stopping decisions or checkpoint selection.
"""

from __future__ import annotations


class EarlyStopping:
    """Tracks a monitored metric and signals when to stop training.

    Args:
        monitor: ``'val_loss'`` (minimise) or ``'val_ap'`` (maximise).
        patience: Epochs without improvement before stopping.
        min_delta: Minimum absolute improvement to count as progress.
        min_epochs: Minimum number of epochs before stopping is allowed.
    """

    def __init__(
        self,
        monitor: str = "val_loss",
        patience: int = 20,
        min_delta: float = 0.0,
        min_epochs: int = 10,
    ) -> None:
        if monitor not in ("val_loss", "val_ap"):
            raise ValueError(f"monitor must be 'val_loss' or 'val_ap', got '{monitor}'")
        self.monitor = monitor
        self.patience = int(patience)
        self.min_delta = float(min_delta)
        self.min_epochs = int(min_epochs)
        self._minimize = monitor == "val_loss"
        self._best: float = float("inf") if self._minimize else float("-inf")
        self._wait: int = 0
        self._best_epoch: int = 0

    # ------------------------------------------------------------------
    @property
    def best_epoch(self) -> int:
        return self._best_epoch

    @property
    def best_value(self) -> float:
        return self._best

    def step(self, epoch: int, value: float) -> bool:
        """Call once per epoch.

        Returns:
            ``True`` if training should stop.
        """
        improved = (
            (self._best - value) > self.min_delta
            if self._minimize
            else (value - self._best) > self.min_delta
        )
        if improved:
            self._best = value
            self._best_epoch = epoch
            self._wait = 0
        else:
            self._wait += 1

        if epoch < self.min_epochs:
            return False
        return self._wait >= self.patience

    def reset(self) -> None:
        self._best = float("inf") if self._minimize else float("-inf")
        self._wait = 0
        self._best_epoch = 0
