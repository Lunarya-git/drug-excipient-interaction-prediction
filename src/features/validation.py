"""Validation of cached features: dimensions, values, missing data and consistency."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Mapping

import numpy as np
import pandas as pd

from src.data.cleaned import load_cleaned_dataset
from src.data.schema import COL_DRUG_CANON, COL_EXC_CANON
from src.features.generators import FeatureGenerationError, expected_dimension, generate_family
from src.features.registry import get_family_specs
from src.features.store import FeatureStore


def validate_array(code: str, arr: np.ndarray, fam: Mapping[str, Any], n_molecules: int) -> dict[str, Any]:
    """Check shape, dtype, finite values and (for bit features) binary values.

    Raises:
        FeatureGenerationError: on any violated hard requirement.
    """
    dim = expected_dimension(fam)
    errors: list[str] = []
    if arr.ndim != 2:
        errors.append(f"expected a 2-D array, got ndim={arr.ndim}")
    elif arr.shape != (n_molecules, dim):
        errors.append(f"shape {arr.shape} != expected ({n_molecules}, {dim})")
    if errors:
        raise FeatureGenerationError(f"[{code}] " + "; ".join(errors))

    n_nonfinite = int((~np.isfinite(arr.astype(np.float64))).sum())
    if n_nonfinite:
        errors.append(f"{n_nonfinite} NaN/inf values")
    is_bits = fam["type"] != "rdkit_descriptors"
    if is_bits:
        uniq = np.unique(arr)
        if not set(map(float, uniq)).issubset({0.0, 1.0}):
            errors.append(f"bit features contain non-binary values {uniq[:10]}")
    if errors:
        raise FeatureGenerationError(f"[{code}] " + "; ".join(errors))

    const_cols = int((arr.max(axis=0) == arr.min(axis=0)).sum())
    row_sums = arr.astype(np.float64).sum(axis=1) if is_bits else None
    stats: dict[str, Any] = {
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
        "expected_dim": dim,
        "n_nonfinite": n_nonfinite,
        "constant_columns": const_cols,
        "all_zero_rows": int((row_sums == 0).sum()) if row_sums is not None else None,
        "mean_bits_set": float(row_sums.mean()) if row_sums is not None else None,
        "density": float(arr.mean()) if is_bits else None,
    }
    if not is_bits:
        stats["column_min"] = [float(v) for v in arr.min(axis=0)]
        stats["column_max"] = [float(v) for v in arr.max(axis=0)]
    return stats


def validate_feature_store(cfg: Mapping[str, Any], families: list[str] | None = None) -> dict[str, Any]:
    """Validate the whole feature cache against the cleaned dataset.

    Checks: manifest/dataset hash agreement, molecule/pair-index consistency,
    per-family array validity and a reproducibility re-generation of a sample of
    molecules (``features.consistency_check_samples``).

    Returns:
        Report dict with ``passed`` and per-family statistics. Raises on hard failures.
    """
    feat = cfg["features"]
    df, dataset_hash = load_cleaned_dataset(cfg)
    store = FeatureStore.load(cfg, expected_dataset_sha256=dataset_hash)
    specs = get_family_specs(cfg)
    codes = families or [c for c in store.families_available() if c in specs]
    errors: list[str] = []

    # Molecule / pair-index consistency ------------------------------------------------
    pi = store.pair_index
    if len(pi) != len(df):
        errors.append(f"pair_index has {len(pi)} rows but the cleaned dataset has {len(df)}")
    else:
        smiles = store.molecules["canonical_smiles"].to_numpy()
        drug_ok = (smiles[pi["drug_mol_idx"].to_numpy()] == df[COL_DRUG_CANON].to_numpy()).all()
        exc_ok = (smiles[pi["excipient_mol_idx"].to_numpy()] == df[COL_EXC_CANON].to_numpy()).all()
        if not (drug_ok and exc_ok):
            errors.append("pair_index does not map to the canonical SMILES of the cleaned dataset")
    if store.molecules["canonical_smiles"].duplicated().any():
        errors.append("duplicate molecules in molecules.csv")

    family_reports: dict[str, Any] = {}
    n_check = int(feat.get("consistency_check_samples", 0))
    rng = np.random.default_rng(0)
    for code in codes:
        spec = specs[code]
        arr = store.array(code)
        entry: dict[str, Any] = {"name": spec.name}
        try:
            entry.update(validate_array(code, arr, spec.config, store.n_molecules))
            if n_check > 0:
                pick = np.sort(rng.choice(store.n_molecules, size=min(n_check, store.n_molecules), replace=False))
                sample_smiles = store.molecules["canonical_smiles"].to_numpy()[pick].tolist()
                regenerated = generate_family(spec.config, sample_smiles, feat)
                same = bool(np.allclose(regenerated.astype(np.float64), arr[pick].astype(np.float64), atol=1e-5))
                entry["consistency_check_samples"] = int(len(pick))
                entry["consistency_check_passed"] = same
                if not same:
                    errors.append(f"[{code}] regenerated features differ from the cache")
            entry["passed"] = True
        except FeatureGenerationError as exc:
            entry["passed"] = False
            entry["error"] = str(exc)
            errors.append(str(exc))
        family_reports[code] = entry

    return {
        "passed": not errors,
        "errors": errors,
        "n_molecules": store.n_molecules,
        "n_pairs": int(len(pi)),
        "cleaned_dataset_sha256": dataset_hash,
        "validated_at": datetime.now().isoformat(timespec="seconds"),
        "families": family_reports,
    }


def format_validation_report(report: Mapping[str, Any]) -> str:
    """Render the validation report as readable text."""
    lines = ["=" * 78, "DE-INTERACT - FEATURE VALIDATION REPORT", "=" * 78]
    lines.append(f"Validated at : {report['validated_at']}")
    lines.append(f"Molecules    : {report['n_molecules']} unique canonical SMILES; pairs: {report['n_pairs']}")
    lines.append(f"Overall      : {'PASSED' if report['passed'] else 'FAILED'}")
    lines.append("")
    rows = []
    for code, rep in report["families"].items():
        rows.append(
            {
                "family": code,
                "name": rep.get("name"),
                "shape": rep.get("shape"),
                "dtype": rep.get("dtype"),
                "nonfinite": rep.get("n_nonfinite"),
                "const_cols": rep.get("constant_columns"),
                "all_zero_rows": rep.get("all_zero_rows"),
                "density": None if rep.get("density") is None else round(rep["density"], 4),
                "reproducible": rep.get("consistency_check_passed"),
                "passed": rep.get("passed"),
            }
        )
    if rows:
        lines.append(pd.DataFrame(rows).to_string(index=False))
    if report["errors"]:
        lines.append("")
        lines.append("ERRORS:")
        lines.extend(f"  - {e}" for e in report["errors"])
    lines.append("=" * 78)
    return "\n".join(lines) + "\n"
