"""Publication-quality plots for the DE-INTERACT research project.

All functions accept pre-computed data and write to a caller-specified path.
They do NOT load data themselves, keeping the plotting layer pure.

Supported plot types
--------------------
* PR curves (per-experiment and overlaid comparison)
* ROC curves (per-experiment and overlaid comparison)
* Learning curves (train loss, inner-val loss, heldout AP)
* Calibration (reliability diagram)
* Bar chart comparison of experiments by metric
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Sequence

import numpy as np

logger = logging.getLogger(__name__)

# Default visual style ---------------------------------------------------------
_PALETTE = [
    "#4C72B0", "#DD8452", "#55A868", "#C44E52",
    "#8172B3", "#937860", "#DA8BC3", "#8C8C8C",
    "#CCB974", "#64B5CD",
]

_LINESTYLES = ["-", "--", "-.", ":"]


def _get_mpl():
    """Lazy import of matplotlib to avoid import-time cost."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _save(fig: Any, path: Path, formats: Sequence[str], dpi: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        out = path.with_suffix(f".{fmt}")
        fig.savefig(out, dpi=dpi, bbox_inches="tight")
        logger.debug("Plot saved: %s", out)


# ---------------------------------------------------------------------------
# PR and ROC curves
# ---------------------------------------------------------------------------

def plot_pr_curve(
    labels: np.ndarray,
    scores: np.ndarray,
    experiment_id: str,
    out_path: Path,
    formats: Sequence[str] = ("png",),
    dpi: int = 300,
    threshold: float | None = None,
) -> None:
    """Precision-recall curve for a single experiment."""
    from sklearn.metrics import precision_recall_curve, average_precision_score

    plt = _get_mpl()
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    ap = average_precision_score(labels, scores)
    baseline = labels.mean()

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(recall, precision, color=_PALETTE[0], lw=2, label=f"AP = {ap:.3f}")
    ax.axhline(baseline, ls="--", color="grey", lw=1, label=f"Chance (prevalence = {baseline:.3f})")
    if threshold is not None:
        scores_thresh = (scores >= threshold).astype(int)
        tp = int(((scores_thresh == 1) & (labels == 1)).sum())
        fp = int(((scores_thresh == 1) & (labels == 0)).sum())
        fn = int(((scores_thresh == 0) & (labels == 1)).sum())
        pr = tp / max(tp + fp, 1)
        re = tp / max(tp + fn, 1)
        ax.scatter([re], [pr], s=80, zorder=5, color=_PALETTE[3],
                   label=f"Threshold = {threshold:.2f}")
    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title(f"PR Curve — {experiment_id}", fontsize=13)
    ax.legend(fontsize=10)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _save(fig, out_path, formats, dpi)
    plt.close(fig)


def plot_roc_curve(
    labels: np.ndarray,
    scores: np.ndarray,
    experiment_id: str,
    out_path: Path,
    formats: Sequence[str] = ("png",),
    dpi: int = 300,
) -> None:
    """ROC curve for a single experiment."""
    from sklearn.metrics import roc_curve, roc_auc_score

    plt = _get_mpl()
    fpr, tpr, _ = roc_curve(labels, scores)
    auroc = roc_auc_score(labels, scores)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color=_PALETTE[0], lw=2, label=f"AUROC = {auroc:.3f}")
    ax.plot([0, 1], [0, 1], ls="--", color="grey", lw=1, label="Chance")
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title(f"ROC Curve — {experiment_id}", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _save(fig, out_path, formats, dpi)
    plt.close(fig)


def plot_pr_comparison(
    experiments: list[dict[str, Any]],
    out_path: Path,
    title: str = "PR Curves — Representation Comparison",
    formats: Sequence[str] = ("png", "svg"),
    dpi: int = 300,
) -> None:
    """Overlay PR curves for multiple experiments.

    Each element of ``experiments`` must have keys:
      ``labels`` (np.ndarray), ``scores`` (np.ndarray), ``label`` (str).
    """
    from sklearn.metrics import precision_recall_curve, average_precision_score

    plt = _get_mpl()
    fig, ax = plt.subplots(figsize=(8, 6))

    for i, exp in enumerate(experiments):
        labels = np.asarray(exp["labels"])
        scores = np.asarray(exp["scores"])
        name = str(exp["label"])
        precision, recall, _ = precision_recall_curve(labels, scores)
        ap = average_precision_score(labels, scores)
        color = _PALETTE[i % len(_PALETTE)]
        ls = _LINESTYLES[(i // len(_PALETTE)) % len(_LINESTYLES)]
        ax.plot(recall, precision, color=color, ls=ls, lw=1.8, label=f"{name} (AP={ap:.3f})")

    if experiments:
        baseline = np.asarray(experiments[0]["labels"]).mean()
        ax.axhline(baseline, ls=":", color="grey", lw=1, label=f"Chance ({baseline:.3f})")

    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.legend(fontsize=8, bbox_to_anchor=(1.02, 1), loc="upper left")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _save(fig, out_path, formats, dpi)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Learning curves
# ---------------------------------------------------------------------------

def plot_learning_curves(
    history: dict[str, Any],
    out_path: Path,
    formats: Sequence[str] = ("png",),
    dpi: int = 300,
) -> None:
    """Train loss, inner-val loss, and heldout AP on separate y-axes.

    ``history`` is the dict produced by :meth:`FoldHistory.to_dict`.
    """
    plt = _get_mpl()
    epochs_data = history.get("epochs", [])
    if not epochs_data:
        logger.warning("No epoch data to plot for fold %s seed %s", history.get("fold"), history.get("seed"))
        return

    epochs = [e["epoch"] for e in epochs_data]
    train_loss = [e["train_loss"] for e in epochs_data]
    iv_loss = [e["inner_val_loss"] for e in epochs_data]
    heldout_ap = [e.get("heldout_ap") for e in epochs_data]
    has_heldout = any(v is not None and not (isinstance(v, float) and v != v) for v in heldout_ap)

    fig, axes = plt.subplots(1, 2 if has_heldout else 1, figsize=(12 if has_heldout else 6, 4))
    if not has_heldout:
        axes = [axes]

    ax0 = axes[0]
    ax0.plot(epochs, train_loss, label="Train loss", color=_PALETTE[0], lw=1.5)
    ax0.plot(epochs, iv_loss, label="Inner-val loss", color=_PALETTE[1], lw=1.5, ls="--")
    ax0.set_xlabel("Epoch")
    ax0.set_ylabel("Loss (BCE)")
    ax0.set_title(f"Loss curves — fold {history.get('fold')} seed {history.get('seed')}")
    ax0.legend(fontsize=9)
    ax0.grid(True, alpha=0.3)

    if has_heldout:
        ax1 = axes[1]
        clean_ap = [v if v is not None else float("nan") for v in heldout_ap]
        ax1.plot(epochs, clean_ap, label="Heldout AP (diagnostic)", color=_PALETTE[2], lw=1.5)
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Average Precision")
        ax1.set_title("Heldout AP (diagnostic — not used for stopping)")
        ax1.legend(fontsize=9)
        ax1.grid(True, alpha=0.3)

    fig.tight_layout()
    _save(fig, out_path, formats, dpi)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------

def plot_calibration(
    cal: dict[str, Any],
    experiment_id: str,
    out_path: Path,
    formats: Sequence[str] = ("png",),
    dpi: int = 300,
) -> None:
    """Reliability diagram from a :class:`CalibrationResult` dict."""
    plt = _get_mpl()
    confs = [c for c in cal["bin_confs"] if c == c]  # drop NaN
    accs = [a for a in cal["bin_accs"] if a == a]
    counts = [cnt for cnt, c in zip(cal["bin_counts"], cal["bin_confs"]) if c == c]

    fig, ax = plt.subplots(figsize=(6, 5))
    width = 1.0 / max(cal["n_bins"], 1)
    ax.bar(
        [c - width / 2 for c in confs], accs,
        width=width * 0.9, alpha=0.7, color=_PALETTE[0], label="Fraction positive"
    )
    ax.plot([0, 1], [0, 1], ls="--", color="grey", lw=1.5, label="Perfect calibration")
    ax.set_xlabel("Mean predicted probability", fontsize=12)
    ax.set_ylabel("Fraction of positives", fontsize=12)
    ax.set_title(f"Reliability diagram — {experiment_id}\nECE = {cal['ece']:.4f}", fontsize=13)
    ax.legend(fontsize=10)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _save(fig, out_path, formats, dpi)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Bar chart comparison
# ---------------------------------------------------------------------------

def plot_metric_comparison(
    rows: list[dict[str, Any]],
    metric: str,
    out_path: Path,
    title: str | None = None,
    formats: Sequence[str] = ("png", "svg"),
    dpi: int = 300,
    x_label: str = "Experiment",
) -> None:
    """Horizontal bar chart of ``metric`` for all experiments.

    ``rows`` is the list produced by :func:`src.evaluation.reporting.build_summary_table`.
    """
    plt = _get_mpl()
    names = [r["experiment_id"] for r in rows]
    values = [float(r.get(metric, 0.0)) for r in rows]

    fig, ax = plt.subplots(figsize=(max(8, len(names) * 0.5 + 2), 5))
    colors = [_PALETTE[i % len(_PALETTE)] for i in range(len(names))]
    bars = ax.bar(names, values, color=colors, width=0.6, alpha=0.85)
    ax.bar_label(bars, fmt="%.3f", fontsize=8, padding=2)
    ax.set_ylabel(metric.upper(), fontsize=12)
    ax.set_xlabel(x_label, fontsize=11)
    ax.set_title(title or f"{metric.upper()} comparison", fontsize=13)
    ax.set_ylim(0, min(1.0, max(values, default=1.0) * 1.15))
    ax.tick_params(axis="x", rotation=45)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    _save(fig, out_path, formats, dpi)
    plt.close(fig)
