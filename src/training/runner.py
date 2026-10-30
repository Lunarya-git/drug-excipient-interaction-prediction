"""Run all (seed, fold) combinations for one experiment and aggregate results.

The runner iterates over every seed x CV fold, calls :func:`train_fold`,
collects :class:`FoldResult` objects, pools out-of-fold (OOF) predictions,
and writes per-experiment JSON artefacts.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from src.data.splitting import load_splits
from src.features.assemble import RepresentationData
from src.training.results import FoldResult
from src.training.trainer import train_fold
from src.utils.config import deep_merge, resolve_path

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------

def run_experiment(
    experiment_id: str,
    representation_keyword: str,
    model_key: str,
    data: RepresentationData,
    labels: np.ndarray,
    cfg: Mapping[str, Any],
    output_dir: Path,
    overrides: Mapping[str, Any] | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Run all (seed × fold) combinations and write JSON artefacts.

    Args:
        experiment_id: Identifier such as ``'PC-CONCAT'``.
        representation_keyword: Feature keyword, e.g. ``'PC'``.
        model_key: Architecture key in ``cfg['models']``, e.g. ``'CONCAT'``.
        data: Loaded and cached feature matrices.
        labels: Integer labels (0/1) for all samples in the cleaned dataset.
        cfg: Merged global configuration.
        output_dir: Directory for this experiment's outputs.
        overrides: Stage-level training overrides (merged on top of cfg).
        dry_run: If True, print plans and return without running training.

    Returns:
        Dict with pooled OOF predictions and aggregate metadata.
    """
    if overrides:
        cfg = deep_merge(cfg, {"training": overrides})

    cfg_training = cfg["training"]
    seeds: list[int] = [int(s) for s in cfg_training.get("seeds", [42])]

    splits_dir = resolve_path(cfg, cfg["data"]["splits_dir"])
    split_result = load_splits(splits_dir)

    n_folds = split_result.n_folds
    logger.info(
        "Experiment %s | repr=%s model=%s | seeds=%s folds=%d",
        experiment_id, representation_keyword, model_key, seeds, n_folds,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    results_dir = output_dir / "fold_results"
    results_dir.mkdir(exist_ok=True)
    ckpt_dir = output_dir / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)

    all_results: list[FoldResult] = []

    for seed in seeds:
        for fold_idx, (train_ids, val_ids) in enumerate(split_result.folds):
            run_id = f"{experiment_id}_seed{seed}_fold{fold_idx}"
            ckpt_path = ckpt_dir / f"{run_id}.pt"

            result = train_fold(
                experiment_id=run_id,
                fold=fold_idx,
                seed=seed,
                train_ids=train_ids,
                val_ids=val_ids,
                labels=labels,
                data=data,
                cfg=cfg,
                model_key=model_key,
                checkpoint_path=ckpt_path if not dry_run else None,
                dry_run=dry_run,
            )
            all_results.append(result)

            # Save per-fold JSON
            fold_json = results_dir / f"{run_id}.json"
            fold_json.write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")

    # ------------------------------------------------------------------ pool OOF
    pooled = _pool_oof(all_results)
    pooled_path = output_dir / "oof_predictions.json"
    pooled_path.write_text(json.dumps(pooled, indent=2), encoding="utf-8")
    logger.info("Pooled OOF predictions: %s", pooled_path)

    meta = {
        "experiment_id": experiment_id,
        "representation": representation_keyword,
        "model": model_key,
        "seeds": seeds,
        "n_folds": n_folds,
        "n_oof_samples": len(pooled["sample_ids"]),
        "dry_run": dry_run,
    }
    meta_path = output_dir / "experiment_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return {"pooled_oof": pooled, "meta": meta}


def _pool_oof(results: list[FoldResult]) -> dict[str, Any]:
    """Aggregate OOF predictions across seeds and folds.

    When multiple seeds are used, OOF scores for the same sample are averaged.
    """
    from collections import defaultdict

    # sample_id -> (scores_list, label)
    by_sample: dict[int, tuple[list[float], int]] = defaultdict(lambda: ([], -1))
    for result in results:
        for sid, lab, score in zip(
            result.oof_sample_ids, result.oof_labels, result.oof_scores
        ):
            scores_list, _ = by_sample[sid]
            scores_list.append(score)
            by_sample[sid] = (scores_list, lab)

    sample_ids = sorted(by_sample)
    labels = [by_sample[sid][1] for sid in sample_ids]
    scores = [float(np.mean(by_sample[sid][0])) for sid in sample_ids]
    return {
        "sample_ids": sample_ids,
        "labels": labels,
        "scores": scores,
    }
