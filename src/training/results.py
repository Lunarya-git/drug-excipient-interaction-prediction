"""Training metrics collected per-step and per-epoch.

All metrics are plain Python floats or numpy scalars so they can be serialised
to JSON without further conversion.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class EpochMetrics:
    """Metrics produced at the end of one training epoch."""

    epoch: int
    train_loss: float
    inner_val_loss: float
    inner_val_ap: float
    # Diagnostic only – never used for stopping or checkpoint selection.
    heldout_loss: float | None = None
    heldout_ap: float | None = None
    lr: float | None = None


@dataclass
class FoldHistory:
    """Per-epoch metrics collected over one CV fold."""

    fold: int
    seed: int
    epochs: list[EpochMetrics] = field(default_factory=list)

    def append(self, m: EpochMetrics) -> None:
        self.epochs.append(m)

    def last(self) -> EpochMetrics | None:
        return self.epochs[-1] if self.epochs else None

    def to_dict(self) -> dict[str, Any]:
        return {
            "fold": self.fold,
            "seed": self.seed,
            "epochs": [
                {
                    "epoch": m.epoch,
                    "train_loss": m.train_loss,
                    "inner_val_loss": m.inner_val_loss,
                    "inner_val_ap": m.inner_val_ap,
                    "heldout_loss": m.heldout_loss,
                    "heldout_ap": m.heldout_ap,
                    "lr": m.lr,
                }
                for m in self.epochs
            ],
        }


@dataclass
class FoldResult:
    """Final result of one (fold, seed) run including out-of-fold predictions."""

    experiment_id: str
    fold: int
    seed: int
    best_epoch: int
    history: FoldHistory
    # Parallel arrays, length = number of OOF samples in this fold.
    oof_sample_ids: list[int]
    oof_labels: list[int]
    oof_scores: list[float]   # sigmoid probabilities of incompatibility (class 1)

    def to_dict(self) -> dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "fold": self.fold,
            "seed": self.seed,
            "best_epoch": self.best_epoch,
            "history": self.history.to_dict(),
            "oof_sample_ids": self.oof_sample_ids,
            "oof_labels": self.oof_labels,
            "oof_scores": self.oof_scores,
        }
