"""Stage 01: clean the raw dataset and create the saved development/test split and CV folds."""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from src.data.cleaning import clean_dataset, format_preprocessing_report
from src.data.leakage import run_leakage_checks
from src.data.loading import load_raw_dataframe, standardize_columns
from src.data.schema import (
    AUDIT_FILENAME,
    CLEANED_FILENAME,
    CLEANING_SUMMARY_FILENAME,
    INTERNAL_LABEL_NAMES,
    LABEL_MAPPING_FILENAME,
    REPORT_FILENAME,
    SPLIT_ASSIGNMENTS_FILENAME,
    SPLIT_COUNTS_FILENAME,
    SPLIT_INDICES_FILENAME,
    SPLIT_METADATA_FILENAME,
)
from src.data.splitting import build_class_count_table, create_splits, save_splits
from src.utils.config import resolve_path
from src.utils.io_utils import archive_existing, ensure_dir, save_json, sha256_file, write_text
from src.utils.paths import OutputPaths

SPLIT_REPORT_FILENAME = "split_report.txt"


def _targets(cfg: Mapping[str, Any]) -> dict[str, Path]:
    data = cfg["data"]
    processed = resolve_path(cfg, data["processed_dir"])
    splits = resolve_path(cfg, data["splits_dir"])
    prep = OutputPaths.from_config(cfg).preprocessing
    return {
        "processed_dir": processed,
        "splits_dir": splits,
        "prep_dir": prep,
        "cleaned": processed / CLEANED_FILENAME,
        "label_mapping": processed / LABEL_MAPPING_FILENAME,
        "label_mapping_copy": prep / LABEL_MAPPING_FILENAME,
        "audit": prep / AUDIT_FILENAME,
        "report": prep / REPORT_FILENAME,
        "summary": prep / CLEANING_SUMMARY_FILENAME,
        "split_report": prep / SPLIT_REPORT_FILENAME,
        "split_assignments": splits / SPLIT_ASSIGNMENTS_FILENAME,
        "split_indices": splits / SPLIT_INDICES_FILENAME,
        "split_counts": splits / SPLIT_COUNTS_FILENAME,
        "split_metadata": splits / SPLIT_METADATA_FILENAME,
    }


def _format_split_report(
    df: pd.DataFrame, counts: pd.DataFrame, leakage: Mapping[str, Any], meta: Mapping[str, Any]
) -> str:
    lines = ["=" * 78, "DE-INTERACT - SPLIT REPORT", "=" * 78]
    lines.append(f"Seed: {meta['split_seed']} | test fraction: {meta['test_fraction']} | folds: {meta['n_folds']}")
    lines.append("Protocol: " + meta["protocol"])
    lines.append("The test set is untouched in this project phase (no model selection / thresholding on it).")
    lines.append("")
    lines.append("Class counts (internal labels: 0 = COMPATIBLE, 1 = INCOMPATIBLE):")
    lines.append(counts.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    lines.append("")
    lines.append(f"Leakage / isolation checks passed: {leakage['passed']}")
    lines.append(f"  duplicate pair keys                 : {leakage['pair_key_duplicates']}")
    lines.append(f"  development/test index overlap      : {leakage['development_test_index_overlap']}")
    lines.append(f"  development/test pair overlap       : {leakage['development_test_pair_overlap']}")
    for fold in leakage["folds"]:
        lines.append(
            f"  fold {fold['fold']}: train={fold['n_train']} val={fold['n_val']} "
            f"index_overlap={fold['train_val_index_overlap']} pair_overlap={fold['train_val_pair_overlap']}"
        )
    reuse = leakage["molecule_reuse_test_vs_development"]
    lines.append("")
    lines.append("Informational molecule reuse (warm-start protocol), test pairs vs development set:")
    lines.append(f"  pairs whose drug was seen in development       : {reuse['pct_pairs_with_seen_drug']:.1f}%")
    lines.append(f"  pairs whose excipient was seen in development  : {reuse['pct_pairs_with_seen_excipient']:.1f}%")
    lines.append(f"  pairs with both molecules seen in development  : {reuse['pct_pairs_with_both_seen']:.1f}%")
    lines.append(leakage["note"])
    lines.append("=" * 78)
    return "\n".join(lines) + "\n"


def run_data_preparation(cfg: Mapping[str, Any], force: bool = False, logger: logging.Logger | None = None) -> dict[str, Any]:
    """Clean the raw CSV, write all audit artefacts, then create and save the splits.

    Args:
        cfg: Merged configuration.
        force: If False and outputs already exist, raise instead of overwriting.
            If True, existing outputs are moved to a time-stamped ``archive_*`` folder first.
        logger: Optional logger.

    Returns:
        A dict with the cleaning summary, the leakage report and the main output paths.
    """
    log = logger or logging.getLogger(__name__)
    t = _targets(cfg)
    outputs = [t[k] for k in (
        "cleaned", "label_mapping", "label_mapping_copy", "audit", "report", "summary",
        "split_report", "split_assignments", "split_indices", "split_counts", "split_metadata",
    )]
    existing = [p for p in outputs if p.exists()]
    if existing and not force:
        raise FileExistsError(
            "Data preparation outputs already exist and will not be overwritten silently:\n  "
            + "\n  ".join(str(p) for p in existing)
            + "\nRe-run with --force to archive them and regenerate (this invalidates cached "
            "features and saved splits, so downstream stages must be re-run)."
        )
    if existing and force:
        for key in ("processed_dir", "splits_dir", "prep_dir"):
            files = [p for p in existing if p.parent == t[key]]
            archived = archive_existing(files, t[key] / "_archive")
            if archived:
                log.warning("Archived previous outputs to %s", archived)

    for key in ("processed_dir", "splits_dir", "prep_dir"):
        ensure_dir(t[key])

    # ---- load & clean ------------------------------------------------------------------
    raw_path = resolve_path(cfg, cfg["data"]["raw_path"])
    log.info("Reading raw dataset %s", raw_path)
    df_raw = load_raw_dataframe(cfg)
    raw_hash = sha256_file(raw_path)
    df_std = standardize_columns(df_raw, cfg)
    result = clean_dataset(df_std, cfg)

    cleaned_path = t["cleaned"]
    result.cleaned.to_csv(cleaned_path, index=False)
    cleaned_hash = sha256_file(cleaned_path)
    result.audit.to_csv(t["audit"], index=False)

    mapping_payload = {
        "original_to_internal": {k: int(v) for k, v in cfg["data"]["label_mapping"].items()},
        "internal_to_name": {int(k): v for k, v in INTERNAL_LABEL_NAMES.items()},
        "positive_class": "INCOMPATIBLE (1)",
        "label_match": dict(cfg["data"].get("label_match", {})),
        "source_label_counts": result.label_inspection,
        "raw_csv_label_column": cfg["data"]["columns"]["label"],
    }
    save_json(mapping_payload, t["label_mapping"])
    save_json(mapping_payload, t["label_mapping_copy"])
    save_json(result.summary, t["summary"])

    report = format_preprocessing_report(
        result.summary,
        cfg,
        raw_path=str(raw_path),
        raw_sha256=raw_hash,
        cleaned_sha256=cleaned_hash,
        generated_at=datetime.now().isoformat(timespec="seconds"),
    )
    write_text(report, t["report"])
    print(report)  # console copy requested by the project protocol
    log.info("Cleaned dataset: %s (%d rows)", cleaned_path, len(result.cleaned))
    log.info("Removed-row audit: %s (%d rows)", t["audit"], len(result.audit))
    log.info("Preprocessing report: %s", t["report"])

    # ---- split ---------------------------------------------------------------------------
    sp = cfg["splits"]
    splits = create_splits(
        result.cleaned,
        test_fraction=float(sp["test_fraction"]),
        n_folds=int(sp["n_folds"]),
        seed=int(sp["split_seed"]),
    )
    leakage = run_leakage_checks(result.cleaned, splits, raise_on_error=True)
    save_splits(result.cleaned, splits, t["splits_dir"], cleaned_hash, leakage)
    counts = build_class_count_table(result.cleaned, splits)
    split_report = _format_split_report(result.cleaned, counts, leakage, splits.metadata)
    write_text(split_report, t["split_report"])
    print(split_report)
    log.info("Splits saved to %s", t["splits_dir"])

    return {
        "summary": result.summary,
        "leakage": leakage,
        "cleaned_path": str(cleaned_path),
        "cleaned_sha256": cleaned_hash,
        "splits_dir": str(t["splits_dir"]),
    }
