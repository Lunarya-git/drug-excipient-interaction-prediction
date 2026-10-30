"""Classification metrics for drug-excipient incompatibility prediction.

All metrics are computed from pooled out-of-fold (OOF) predictions.
The positive class is INCOMPATIBLE (label = 1).

Key design decisions
--------------------
* Average precision (AP/AUPRC) is the primary metric: the dataset is
  imbalanced and the clinical cost of a missed incompatibility is high.
* AUROC is reported as a secondary metric for comparability with the
  DE-INTERACT baseline.
* The operating threshold is chosen by maximising F-beta on the OOF
  predictions (development data only, never the test set).
* Calibration is measured by ECE (expected calibration error).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class ThresholdMetrics:
    """Threshold-dependent classification metrics at a specific cut-off."""

    threshold: float
    precision: float
    recall: float
    f1: float
    f_beta: float  # the beta used for threshold selection
    beta: float
    tp: int
    fp: int
    fn: int
    tn: int
    accuracy: float
    balanced_accuracy: float
    mcc: float


@dataclass
class CalibrationResult:
    """Expected calibration error (ECE) with reliability diagram data."""

    ece: float
    n_bins: int
    bin_boundaries: list[float]
    bin_accs: list[float]
    bin_confs: list[float]
    bin_counts: list[int]


@dataclass
class ExperimentMetrics:
    """Full suite of OOF metrics for one experiment."""

    experiment_id: str
    n_samples: int
    n_positive: int
    n_negative: int
    prevalence: float
    average_precision: float
    auroc: float
    threshold: ThresholdMetrics
    calibration: CalibrationResult
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "experiment_id": self.experiment_id,
            "n_samples": self.n_samples,
            "n_positive": self.n_positive,
            "n_negative": self.n_negative,
            "prevalence": self.prevalence,
            "average_precision": self.average_precision,
            "auroc": self.auroc,
            "threshold": vars(self.threshold),
            "calibration": {
                "ece": self.calibration.ece,
                "n_bins": self.calibration.n_bins,
                "bin_boundaries": self.calibration.bin_boundaries,
                "bin_accs": self.calibration.bin_accs,
                "bin_confs": self.calibration.bin_confs,
                "bin_counts": self.calibration.bin_counts,
            },
        }
        d.update(self.extra)
        return d


# ---------------------------------------------------------------------------
# Core metric functions
# ---------------------------------------------------------------------------

def compute_ap(labels: np.ndarray, scores: np.ndarray) -> float:
    from sklearn.metrics import average_precision_score
    if len(np.unique(labels)) < 2:
        return float("nan")
    return float(average_precision_score(labels, scores))


def compute_auroc(labels: np.ndarray, scores: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score
    if len(np.unique(labels)) < 2:
        return float("nan")
    return float(roc_auc_score(labels, scores))


def compute_threshold_metrics(
    labels: np.ndarray,
    scores: np.ndarray,
    threshold: float,
    beta: float = 1.0,
) -> ThresholdMetrics:
    preds = (scores >= threshold).astype(int)
    tp = int(((preds == 1) & (labels == 1)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())
    tn = int(((preds == 0) & (labels == 0)).sum())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    beta2 = beta ** 2
    f_beta = (1 + beta2) * precision * recall / max(beta2 * precision + recall, 1e-12)
    accuracy = (tp + tn) / max(len(labels), 1)
    sens = tp / max(tp + fn, 1)
    spec = tn / max(tn + fp, 1)
    balanced_acc = (sens + spec) / 2
    denom_mcc = math.sqrt(max((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn), 1))
    mcc = (tp * tn - fp * fn) / denom_mcc
    return ThresholdMetrics(
        threshold=float(threshold),
        precision=precision,
        recall=recall,
        f1=f1,
        f_beta=f_beta,
        beta=float(beta),
        tp=tp, fp=fp, fn=fn, tn=tn,
        accuracy=accuracy,
        balanced_accuracy=balanced_acc,
        mcc=float(mcc),
    )


def select_threshold(
    labels: np.ndarray,
    scores: np.ndarray,
    metric: str = "f2",
    beta: float = 2.0,
    start: float = 0.01,
    stop: float = 0.99,
    step: float = 0.01,
) -> float:
    """Select the operating threshold maximising ``metric`` on the grid.

    Supports: ``'f1'``, ``'f2'``, ``'fbeta'`` (uses ``beta``),
    ``'balanced_accuracy'``.
    """
    grid = np.arange(start, stop + step / 2, step)
    best_t = float(start)
    best_val = float("-inf")
    for t in grid:
        m = compute_threshold_metrics(labels, scores, t, beta=beta)
        if metric in ("f2", "fbeta"):
            v = m.f_beta
        elif metric == "f1":
            v = m.f1
        elif metric == "balanced_accuracy":
            v = m.balanced_accuracy
        else:
            raise ValueError(f"Unknown threshold selection metric '{metric}'")
        if v > best_val:
            best_val = v
            best_t = float(t)
    return best_t


def compute_calibration(
    labels: np.ndarray,
    scores: np.ndarray,
    n_bins: int = 10,
) -> CalibrationResult:
    """Compute ECE (equal-width bins) and reliability diagram data."""
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_accs, bin_confs, bin_counts = [], [], []
    ece = 0.0
    n = len(labels)
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (scores >= lo) & (scores < hi)
        if mask.sum() == 0:
            bin_accs.append(float("nan"))
            bin_confs.append(float((lo + hi) / 2))
            bin_counts.append(0)
            continue
        acc = float(labels[mask].mean())
        conf = float(scores[mask].mean())
        cnt = int(mask.sum())
        ece += (cnt / n) * abs(acc - conf)
        bin_accs.append(acc)
        bin_confs.append(conf)
        bin_counts.append(cnt)
    return CalibrationResult(
        ece=float(ece),
        n_bins=n_bins,
        bin_boundaries=[float(b) for b in bins],
        bin_accs=bin_accs,
        bin_confs=bin_confs,
        bin_counts=bin_counts,
    )


import math


def compute_experiment_metrics(
    experiment_id: str,
    labels: np.ndarray,
    scores: np.ndarray,
    cfg: dict[str, Any],
) -> ExperimentMetrics:
    """Compute the full suite of metrics for one experiment.

    Args:
        experiment_id: Label used in reports.
        labels: True binary labels (0/1); positive class = 1 (INCOMPATIBLE).
        scores: Predicted probabilities for class 1.
        cfg: Merged configuration (reads ``evaluation.*`` settings).

    Returns:
        :class:`ExperimentMetrics` dataclass.
    """
    cfg_eval = cfg.get("evaluation", {})
    n_pos = int((labels == 1).sum())
    n_neg = int((labels == 0).sum())

    ap = compute_ap(labels, scores)
    auroc = compute_auroc(labels, scores)

    t_cfg = cfg_eval.get("threshold_selection", {})
    t_metric = str(t_cfg.get("metric", "f2"))
    beta = float(t_cfg.get("beta", 2.0))
    grid = t_cfg.get("grid", {})
    threshold = select_threshold(
        labels, scores,
        metric=t_metric,
        beta=beta,
        start=float(grid.get("start", 0.01)),
        stop=float(grid.get("stop", 0.99)),
        step=float(grid.get("step", 0.01)),
    )
    tm = compute_threshold_metrics(labels, scores, threshold, beta=beta)
    cal = compute_calibration(labels, scores, n_bins=int(cfg_eval.get("calibration_bins", 10)))

    return ExperimentMetrics(
        experiment_id=experiment_id,
        n_samples=len(labels),
        n_positive=n_pos,
        n_negative=n_neg,
        prevalence=n_pos / max(len(labels), 1),
        average_precision=ap,
        auroc=auroc,
        threshold=tm,
        calibration=cal,
    )
