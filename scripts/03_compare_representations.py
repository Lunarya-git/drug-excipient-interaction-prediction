#!/usr/bin/env python
"""Script 03 — Stage 1: Representation screening.

Runs one experiment per representation listed in
``experiments.stages.representation_screening.representations``
(configs/experiments.yaml), all with the fixed CONCAT architecture.

For each representation:
  * Loads the cached features.
  * Runs all (seed × fold) CV runs.
  * Computes OOF metrics (AP, AUROC, F-beta, ECE, …).
  * Generates per-experiment plots (PR curve, ROC curve).

After all representations have run, writes a cross-representation
comparison bar chart and summary table.

Usage
-----
  python scripts/03_compare_representations.py [--config configs/] [--dry-run]
  python scripts/03_compare_representations.py --override training.seeds=[42,0,1]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils.config import load_config, resolve_path, get_stage_config
from src.utils.logging_utils import setup_logging
from src.features.store import FeatureStore
from src.features.assemble import load_representation
from src.data.cleaned import load_cleaned_dataset
from src.data.schema import COL_LABEL
from src.training.runner import run_experiment
from src.evaluation.reporting import (
    evaluate_experiment,
    build_summary_table,
    write_csv_table,
    write_markdown_table,
)
from src.visualization.plots import (
    plot_pr_curve,
    plot_roc_curve,
    plot_pr_comparison,
    plot_metric_comparison,
)

logger = logging.getLogger(__name__)

STAGE_KEY = "representation_screening"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="DE-INTERACT: Stage 1 — representation screening")
    p.add_argument("--config", default=None, metavar="DIR")
    p.add_argument("--override", nargs="*", default=[], metavar="KEY=VALUE")
    p.add_argument("--representations", nargs="*", default=None, metavar="KW",
                   help="Override the representation list from experiments.yaml")
    p.add_argument("--dry-run", action="store_true",
                   help="Build configuration and print plan; do not train")
    p.add_argument("--log-level", default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(level=args.log_level)

    cfg = load_config(config_dir=args.config, overrides=args.override)
    stage = get_stage_config(cfg, STAGE_KEY)
    representations = args.representations or stage["representations"]
    architecture = stage["architectures"][0]   # CONCAT (fixed for Stage 1)
    training_overrides = stage["overrides"]

    logger.info(
        "Stage 1: %d representations × 1 architecture (%s). Dry-run=%s.",
        len(representations), architecture, args.dry_run,
    )

    # ------------------------------------------------------------------ Load data
    try:
        cleaned_df, cleaned_sha = load_cleaned_dataset(cfg)
    except FileNotFoundError as exc:
        logger.error("%s — run scripts/01_prepare_data.py first.", exc)
        sys.exit(1)
    labels = cleaned_df[COL_LABEL].to_numpy(dtype=int)

    store = FeatureStore.load(cfg, expected_dataset_sha256=cleaned_sha)

    # ------------------------------------------------------------------ Output dirs
    out_root = resolve_path(cfg, cfg["experiments"]["output_root"])
    stage_dir = out_root / cfg["experiments"]["subdirs"].get("experiments", "experiments") / STAGE_KEY
    metrics_dir = out_root / cfg["experiments"]["subdirs"].get("metrics", "metrics") / STAGE_KEY
    plots_dir = out_root / cfg["experiments"]["subdirs"].get("plots", "plots") / STAGE_KEY
    reports_dir = out_root / cfg["experiments"]["subdirs"].get("reports", "reports") / STAGE_KEY

    cfg_eval = cfg.get("evaluation", {})
    plot_formats = cfg_eval.get("plots", {}).get("formats", ["png"])
    agg_formats = cfg_eval.get("plots", {}).get("aggregate_formats", ["png", "svg"])
    dpi = int(cfg_eval.get("plots", {}).get("dpi", 300))

    all_metrics = []
    import numpy as np

    for rep in representations:
        exp_id = f"{rep}-{architecture}"
        logger.info("--- Experiment: %s ---", exp_id)

        try:
            data = load_representation(store, cfg, rep)
        except Exception as exc:
            logger.error("Failed to load representation '%s': %s — run script 02 first.", rep, exc)
            if not args.dry_run:
                sys.exit(1)
            continue

        exp_dir = stage_dir / exp_id
        run_experiment(
            experiment_id=exp_id,
            representation_keyword=rep,
            model_key=architecture,
            data=data,
            labels=labels,
            cfg=cfg,
            output_dir=exp_dir,
            overrides=training_overrides,
            dry_run=args.dry_run,
        )

        if args.dry_run:
            continue

        # ------------------------------------------------------------------ Metrics
        metrics = evaluate_experiment(exp_id, exp_dir, cfg, metrics_dir=metrics_dir)
        all_metrics.append(metrics)
        logger.info(
            "%s | AP=%.4f AUROC=%.4f F2=%.4f threshold=%.3f ECE=%.4f",
            exp_id,
            metrics.average_precision,
            metrics.auroc,
            metrics.threshold.f_beta,
            metrics.threshold.threshold,
            metrics.calibration.ece,
        )

        # Per-experiment plots
        oof_path = exp_dir / "oof_predictions.json"
        oof = json.loads(oof_path.read_text(encoding="utf-8"))
        oof_labels = np.array(oof["labels"])
        oof_scores = np.array(oof["scores"])

        plot_pr_curve(oof_labels, oof_scores, exp_id,
                      plots_dir / exp_id / "pr_curve",
                      formats=plot_formats, dpi=dpi,
                      threshold=metrics.threshold.threshold)
        plot_roc_curve(oof_labels, oof_scores, exp_id,
                       plots_dir / exp_id / "roc_curve",
                       formats=plot_formats, dpi=dpi)

    if args.dry_run:
        logger.info("[DRY RUN] No experiments were run.")
        return

    if not all_metrics:
        logger.warning("No experiments completed. Nothing to summarise.")
        return

    # ------------------------------------------------------------------ Aggregate
    rows = build_summary_table(all_metrics)
    write_csv_table(rows, metrics_dir / "stage1_summary.csv")
    write_markdown_table(rows, reports_dir / "stage1_summary.md",
                         title="Stage 1 — Representation Screening Results")
    logger.info("Summary table written.")

    plot_metric_comparison(
        rows, "ap",
        plots_dir / "stage1_ap_comparison",
        title="Stage 1: Average Precision by Representation",
        formats=agg_formats, dpi=dpi,
    )
    plot_metric_comparison(
        rows, "auroc",
        plots_dir / "stage1_auroc_comparison",
        title="Stage 1: AUROC by Representation",
        formats=agg_formats, dpi=dpi,
    )

    pr_data = []
    for m in all_metrics:
        oof_path = stage_dir / m.experiment_id / "oof_predictions.json"
        oof = json.loads(oof_path.read_text(encoding="utf-8"))
        pr_data.append({
            "labels": np.array(oof["labels"]),
            "scores": np.array(oof["scores"]),
            "label": m.experiment_id,
        })
    plot_pr_comparison(
        pr_data,
        plots_dir / "stage1_pr_comparison",
        title="Stage 1: PR Curves — All Representations",
        formats=agg_formats, dpi=dpi,
    )

    logger.info("Stage 1 complete. Best representation by AP: %s (AP=%.4f)",
                rows[0]["experiment_id"], rows[0]["ap"])


if __name__ == "__main__":
    main()
