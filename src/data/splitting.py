"""Stratified development/test split and stratified K-fold CV on the development set."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

from src.data.schema import (
    COL_DRUG_ID,
    COL_DRUG_SMILES,
    COL_EXC_ID,
    COL_EXC_SMILES,
    COL_LABEL,
    COL_PAIR_KEY,
    COL_SAMPLE_ID,
    SPLIT_ASSIGNMENTS_FILENAME,
    SPLIT_COUNTS_FILENAME,
    SPLIT_INDICES_FILENAME,
    SPLIT_METADATA_FILENAME,
)
from src.utils.io_utils import load_json, save_json


@dataclass
class SplitResult:
    """Dev/test split and CV folds, expressed as ``sample_id`` arrays.

    ``folds[k] = (train_ids, val_ids)``; all ids are drawn from ``dev_ids``.
    """

    dev_ids: np.ndarray
    test_ids: np.ndarray
    folds: list[tuple[np.ndarray, np.ndarray]]
    metadata: dict[str, Any]

    @property
    def n_folds(self) -> int:
        return len(self.folds)


def create_splits(
    df: pd.DataFrame,
    test_fraction: float,
    n_folds: int,
    seed: int,
) -> SplitResult:
    """Create the stratified dev/test split and the stratified CV folds.

    Args:
        df: Cleaned dataset with ``sample_id`` equal to the row position.
        test_fraction: Fraction held out as the untouched test set (e.g. 0.10).
        n_folds: Number of stratified CV folds on the development set.
        seed: Random seed (both the hold-out split and the folds).
    """
    ids = df[COL_SAMPLE_ID].to_numpy()
    if not np.array_equal(ids, np.arange(len(df))):
        raise ValueError("sample_id must equal the row position 0..N-1 in the cleaned dataset")
    y = df[COL_LABEL].to_numpy()
    dev_ids, test_ids = train_test_split(ids, test_size=test_fraction, stratify=y, random_state=seed)
    dev_ids = np.sort(dev_ids)
    test_ids = np.sort(test_ids)

    y_dev = y[dev_ids]
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    folds: list[tuple[np.ndarray, np.ndarray]] = []
    for train_pos, val_pos in skf.split(dev_ids, y_dev):
        folds.append((np.sort(dev_ids[train_pos]), np.sort(dev_ids[val_pos])))

    metadata = {
        "split_seed": int(seed),
        "test_fraction": float(test_fraction),
        "n_folds": int(n_folds),
        "stratified_by": "label",
        "n_samples": int(len(df)),
        "n_development": int(len(dev_ids)),
        "n_test": int(len(test_ids)),
        "protocol": "stratified hold-out split, then stratified shuffled K-fold on development set",
    }
    return SplitResult(dev_ids=dev_ids, test_ids=test_ids, folds=folds, metadata=metadata)


def class_counts(y: np.ndarray) -> dict[str, int]:
    """Counts of the internal labels (0 = compatible, 1 = incompatible)."""
    return {"compatible_0": int((y == 0).sum()), "incompatible_1": int((y == 1).sum()), "total": int(len(y))}


def build_class_count_table(df: pd.DataFrame, splits: SplitResult) -> pd.DataFrame:
    """One row per split / fold role with class counts and incompatible prevalence."""
    y = df[COL_LABEL].to_numpy()
    rows: list[dict[str, Any]] = []

    def add(name: str, ids: np.ndarray) -> None:
        c = class_counts(y[ids])
        rows.append(
            {
                "subset": name,
                "n": c["total"],
                "compatible_0": c["compatible_0"],
                "incompatible_1": c["incompatible_1"],
                "incompatible_prevalence": c["incompatible_1"] / c["total"] if c["total"] else float("nan"),
            }
        )

    add("all", df[COL_SAMPLE_ID].to_numpy())
    add("development", splits.dev_ids)
    add("test", splits.test_ids)
    for k, (tr, va) in enumerate(splits.folds):
        add(f"fold{k}_train", tr)
        add(f"fold{k}_val", va)
    return pd.DataFrame(rows)


def save_splits(
    df: pd.DataFrame,
    splits: SplitResult,
    splits_dir: Path,
    dataset_sha256: str,
    leakage_report: Mapping[str, Any],
) -> None:
    """Persist split indices, identifiers, class counts and metadata."""
    splits_dir.mkdir(parents=True, exist_ok=True)
    fold_of = np.full(len(df), -1, dtype=int)
    for k, (_, va) in enumerate(splits.folds):
        fold_of[va] = k
    role = np.full(len(df), "development", dtype=object)
    role[splits.test_ids] = "test"
    assignments = df[[COL_SAMPLE_ID, COL_DRUG_ID, COL_EXC_ID, COL_DRUG_SMILES, COL_EXC_SMILES, COL_PAIR_KEY, COL_LABEL]].copy()
    assignments["split"] = role
    assignments["cv_fold"] = fold_of  # fold in which the sample is validated; -1 for test samples
    assignments.to_csv(splits_dir / SPLIT_ASSIGNMENTS_FILENAME, index=False)

    indices = {
        "development": splits.dev_ids.tolist(),
        "test": splits.test_ids.tolist(),
        "folds": [{"fold": k, "train": tr.tolist(), "val": va.tolist()} for k, (tr, va) in enumerate(splits.folds)],
    }
    save_json(indices, splits_dir / SPLIT_INDICES_FILENAME, indent=None)  # type: ignore[arg-type]
    build_class_count_table(df, splits).to_csv(splits_dir / SPLIT_COUNTS_FILENAME, index=False)
    meta = dict(splits.metadata)
    meta["cleaned_dataset_sha256"] = dataset_sha256
    meta["leakage_checks"] = dict(leakage_report)
    save_json(meta, splits_dir / SPLIT_METADATA_FILENAME)


def load_splits(splits_dir: Path, expected_dataset_sha256: str | None = None) -> SplitResult:
    """Load saved splits.

    Raises:
        FileNotFoundError: if script 01 has not been run.
        ValueError: if the splits were created for a different cleaned dataset.
    """
    idx_path = splits_dir / SPLIT_INDICES_FILENAME
    meta_path = splits_dir / SPLIT_METADATA_FILENAME
    if not idx_path.is_file() or not meta_path.is_file():
        raise FileNotFoundError(f"Saved splits not found in {splits_dir}. Run scripts/01_prepare_data.py first.")
    raw = load_json(idx_path)
    meta = load_json(meta_path)
    if expected_dataset_sha256 and meta.get("cleaned_dataset_sha256") != expected_dataset_sha256:
        raise ValueError(
            "The saved splits were created for a different cleaned dataset "
            f"({meta.get('cleaned_dataset_sha256')} != {expected_dataset_sha256}). "
            "Re-run scripts/01_prepare_data.py --force."
        )
    folds = [
        (np.asarray(f["train"], dtype=np.int64), np.asarray(f["val"], dtype=np.int64)) for f in raw["folds"]
    ]
    return SplitResult(
        dev_ids=np.asarray(raw["development"], dtype=np.int64),
        test_ids=np.asarray(raw["test"], dtype=np.int64),
        folds=folds,
        metadata=meta,
    )
