"""Pair-overlap and data-leakage checks for the saved splits."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.data.schema import COL_DRUG_CANON, COL_EXC_CANON, COL_PAIR_KEY, COL_SAMPLE_ID
from src.data.splitting import SplitResult


class LeakageError(RuntimeError):
    """Raised when a split violates isolation (overlap or incomplete coverage)."""


def pair_overlap(df: pd.DataFrame, ids_a: np.ndarray, ids_b: np.ndarray) -> int:
    """Number of distinct pair keys present in both id sets."""
    keys_a = set(df.iloc[ids_a][COL_PAIR_KEY])
    keys_b = set(df.iloc[ids_b][COL_PAIR_KEY])
    return len(keys_a & keys_b)


def index_overlap(ids_a: np.ndarray, ids_b: np.ndarray) -> int:
    """Number of sample ids present in both arrays."""
    return len(set(map(int, ids_a)) & set(map(int, ids_b)))


def molecule_overlap(df: pd.DataFrame, ids_eval: np.ndarray, ids_ref: np.ndarray) -> dict[str, float]:
    """Informational: how many evaluation pairs reuse a molecule seen in the reference set.

    A random pair-level split is a *warm-start* protocol: most test molecules are
    also present in training. This is reported so results are interpreted
    correctly; cold-start evaluation is a later project phase.
    """
    ev = df.iloc[ids_eval]
    ref = df.iloc[ids_ref]
    drugs_ref = set(ref[COL_DRUG_CANON])
    exc_ref = set(ref[COL_EXC_CANON])
    n = max(len(ev), 1)
    drug_seen = ev[COL_DRUG_CANON].isin(drugs_ref)
    exc_seen = ev[COL_EXC_CANON].isin(exc_ref)
    return {
        "pct_pairs_with_seen_drug": 100.0 * float(drug_seen.sum()) / n,
        "pct_pairs_with_seen_excipient": 100.0 * float(exc_seen.sum()) / n,
        "pct_pairs_with_both_seen": 100.0 * float((drug_seen & exc_seen).sum()) / n,
    }


def run_leakage_checks(df: pd.DataFrame, splits: SplitResult, raise_on_error: bool = True) -> dict[str, Any]:
    """Verify split isolation and coverage.

    Checks (all must hold):

    * pair keys are unique in the dataset;
    * development and test sample ids are disjoint and cover every sample exactly once;
    * development and test pair keys do not overlap;
    * in every fold, train and validation ids are disjoint, their union equals the
      development set and their pair keys do not overlap;
    * validation ids of different folds are disjoint and together cover the development set.

    Returns:
        A report dict with ``passed`` and a list of ``errors``; also includes
        informational molecule-reuse statistics.

    Raises:
        LeakageError: if any check fails and ``raise_on_error`` is True.
    """
    errors: list[str] = []
    all_ids = df[COL_SAMPLE_ID].to_numpy()
    dev = np.asarray(splits.dev_ids)
    test = np.asarray(splits.test_ids)

    n_dup_keys = int(df[COL_PAIR_KEY].duplicated().sum())
    if n_dup_keys:
        errors.append(f"{n_dup_keys} duplicated pair keys in the cleaned dataset")

    if index_overlap(dev, test):
        errors.append(f"development/test sample overlap: {index_overlap(dev, test)}")
    if len(dev) + len(test) != len(all_ids) or set(map(int, dev)) | set(map(int, test)) != set(map(int, all_ids)):
        errors.append("development + test ids do not cover the dataset exactly once")
    dev_test_pairs = pair_overlap(df, dev, test)
    if dev_test_pairs:
        errors.append(f"development/test pair overlap: {dev_test_pairs}")

    fold_reports: list[dict[str, Any]] = []
    seen_val: set[int] = set()
    dev_set = set(map(int, dev))
    for k, (tr, va) in enumerate(splits.folds):
        n_idx = index_overlap(tr, va)
        n_pair = pair_overlap(df, tr, va)
        union_ok = set(map(int, tr)) | set(map(int, va)) == dev_set and len(tr) + len(va) == len(dev)
        val_set = set(map(int, va))
        reused = len(val_set & seen_val)
        seen_val |= val_set
        if n_idx:
            errors.append(f"fold {k}: train/val sample overlap {n_idx}")
        if n_pair:
            errors.append(f"fold {k}: train/val pair overlap {n_pair}")
        if not union_ok:
            errors.append(f"fold {k}: train + val does not equal the development set")
        if reused:
            errors.append(f"fold {k}: validation samples already used as validation in an earlier fold ({reused})")
        if index_overlap(tr, test) or index_overlap(va, test):
            errors.append(f"fold {k}: test samples present in train/val")
        fold_reports.append(
            {
                "fold": k,
                "n_train": int(len(tr)),
                "n_val": int(len(va)),
                "train_val_index_overlap": n_idx,
                "train_val_pair_overlap": n_pair,
                "molecule_reuse_val_vs_train": molecule_overlap(df, va, tr),
            }
        )
    if splits.folds and seen_val != dev_set:
        errors.append("validation folds do not cover the development set")

    report = {
        "passed": not errors,
        "errors": errors,
        "pair_key_duplicates": n_dup_keys,
        "development_test_index_overlap": index_overlap(dev, test),
        "development_test_pair_overlap": dev_test_pairs,
        "molecule_reuse_test_vs_development": molecule_overlap(df, test, dev),
        "folds": fold_reports,
        "note": (
            "Pair-level splits are warm-start: molecules may appear in both train and validation/test. "
            "Cold-start (unseen molecule) evaluation is not part of this phase."
        ),
    }
    if errors and raise_on_error:
        raise LeakageError("Data leakage / split isolation check FAILED:\n  - " + "\n  - ".join(errors))
    return report


def assert_fold_isolation(df: pd.DataFrame, train_ids: np.ndarray, val_ids: np.ndarray, test_ids: np.ndarray) -> None:
    """Cheap per-fold guard executed immediately before training a fold."""
    problems: list[str] = []
    if index_overlap(train_ids, val_ids):
        problems.append("train/val sample overlap")
    if pair_overlap(df, train_ids, val_ids):
        problems.append("train/val pair overlap")
    if index_overlap(train_ids, test_ids) or index_overlap(val_ids, test_ids):
        problems.append("test samples present in the development folds")
    if problems:
        raise LeakageError("Fold isolation violated: " + ", ".join(problems))
