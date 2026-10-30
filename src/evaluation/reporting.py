"""Reporting: load OOF predictions, compute metrics, write summary tables.

Outputs
-------
* Per-experiment JSON metrics.
* Markdown and CSV summary tables for all experiments in a stage.
"""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from src.evaluation.metrics import ExperimentMetrics, compute_experiment_metrics

logger = logging.getLogger(__name__)


def load_oof_predictions(experiment_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load pooled OOF predictions written by the experiment runner.

    Returns:
        (labels, scores) as numpy arrays.
    """
    path = experiment_dir / "oof_predictions.json"
    if not path.is_file():
        raise FileNotFoundError(f"OOF predictions not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    labels = np.array(data["labels"], dtype=int)
    scores = np.array(data["scores"], dtype=float)
    return labels, scores


def evaluate_experiment(
    experiment_id: str,
    experiment_dir: Path,
    cfg: Mapping[str, Any],
    metrics_dir: Path | None = None,
) -> ExperimentMetrics:
    """Compute metrics for one experiment and optionally write a JSON file.

    Args:
        experiment_id: Human-readable name such as ``'PC-CONCAT'``.
        experiment_dir: Directory containing ``oof_predictions.json``.
        cfg: Merged configuration.
        metrics_dir: If given, writes ``<experiment_id>_metrics.json`` here.

    Returns:
        :class:`ExperimentMetrics` dataclass.
    """
    labels, scores = load_oof_predictions(experiment_dir)
    metrics = compute_experiment_metrics(experiment_id, labels, scores, cfg)

    if metrics_dir is not None:
        metrics_dir.mkdir(parents=True, exist_ok=True)
        out = metrics_dir / f"{experiment_id}_metrics.json"
        out.write_text(json.dumps(metrics.to_dict(), indent=2), encoding="utf-8")
        logger.info("Metrics written: %s", out)

    return metrics


def build_summary_table(
    all_metrics: list[ExperimentMetrics],
    primary_metric: str = "average_precision",
) -> list[dict[str, Any]]:
    """Build a list of dicts (rows) sorted by ``primary_metric`` descending."""
    rows = []
    for m in all_metrics:
        rows.append(
            {
                "experiment_id": m.experiment_id,
                "ap": round(m.average_precision, 4),
                "auroc": round(m.auroc, 4),
                "precision": round(m.threshold.precision, 4),
                "recall": round(m.threshold.recall, 4),
                "f1": round(m.threshold.f1, 4),
                f"f{m.threshold.beta:.0f}": round(m.threshold.f_beta, 4),
                "threshold": round(m.threshold.threshold, 4),
                "mcc": round(m.threshold.mcc, 4),
                "ece": round(m.calibration.ece, 4),
                "n_samples": m.n_samples,
                "prevalence": round(m.prevalence, 4),
            }
        )
    rows.sort(
        key=lambda r: r.get(
            "ap" if primary_metric == "average_precision" else primary_metric, float("-inf")
        ),
        reverse=True,
    )
    return rows


def write_csv_table(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    logger.info("CSV table written: %s", path)


def write_markdown_table(
    rows: list[dict[str, Any]],
    path: Path,
    title: str = "Results",
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"# {title}", ""]
    if not rows:
        lines.append("*No results yet.*")
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return
    headers = list(rows[0].keys())
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in rows:
        lines.append("| " + " | ".join(str(row[h]) for h in headers) + " |")
    lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info("Markdown table written: %s", path)
