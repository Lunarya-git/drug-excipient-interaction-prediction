"""Dataset cleaning: missing values, invalid SMILES, conflicting labels, duplicates.

Only the four operations requested for this project are performed:

1. rows with missing required values are removed;
2. rows whose drug or excipient SMILES RDKit cannot parse are removed;
3. pairs that appear with conflicting labels are removed (EVERY occurrence);
4. exact duplicate pairs with the same label are collapsed to one row.

No other cleaning or chemical standardisation is performed. RDKit canonical
SMILES are used only to build the pair key (configurable) and as the molecule
string from which features are later computed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np
import pandas as pd

from src.data.loading import inspect_labels, is_blank, map_labels
from src.data.schema import (
    CLEANED_COLUMNS,
    COL_DRUG_CANON,
    COL_DRUG_ID,
    COL_DRUG_SMILES,
    COL_EXC_CANON,
    COL_EXC_ID,
    COL_EXC_SMILES,
    COL_LABEL,
    COL_LABEL_RAW,
    COL_PAIR_KEY,
    COL_RAW_ROW,
    COL_SAMPLE_ID,
    INTERNAL_LABEL_NAMES,
    LABEL_COMPATIBLE,
    LABEL_INCOMPATIBLE,
)

REASON_MISSING = "missing_required_value"
REASON_INVALID_DRUG = "invalid_drug_smiles"
REASON_INVALID_EXC = "invalid_excipient_smiles"
REASON_INVALID_BOTH = "invalid_drug_and_excipient_smiles"
REASON_CONFLICT = "conflicting_label"
REASON_DUPLICATE = "exact_duplicate_pair"
REMOVAL_REASONS = (
    REASON_MISSING,
    REASON_INVALID_DRUG,
    REASON_INVALID_EXC,
    REASON_INVALID_BOTH,
    REASON_CONFLICT,
    REASON_DUPLICATE,
)

_LOGICAL_TO_INTERNAL = {
    "drug_id": COL_DRUG_ID,
    "excipient_id": COL_EXC_ID,
    "drug_smiles": COL_DRUG_SMILES,
    "excipient_smiles": COL_EXC_SMILES,
    "label": COL_LABEL_RAW,
}
_PAIR_SEPARATOR = "||"


def canonical_smiles(smiles: str) -> str | None:
    """Return the RDKit canonical SMILES, or None if RDKit cannot parse ``smiles``."""
    from rdkit import Chem, RDLogger

    RDLogger.DisableLog("rdApp.*")
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol)


def make_pair_key(drug: str, excipient: str) -> str:
    """Build the string key identifying an ordered drug-excipient pair."""
    return f"{drug}{_PAIR_SEPARATOR}{excipient}"


@dataclass
class CleaningResult:
    """Outputs of :func:`clean_dataset`."""

    cleaned: pd.DataFrame
    audit: pd.DataFrame
    summary: dict[str, Any]
    label_inspection: dict[str, int] = field(default_factory=dict)


def summarize_dataset(
    df: pd.DataFrame,
    label_col: str = COL_LABEL,
    canonical: bool = False,
) -> dict[str, Any]:
    """Compute the descriptive statistics reported before/after cleaning.

    Percentages are relative to the rows that carry a (mapped) label.
    """
    n_rows = int(len(df))
    labels = df[label_col].dropna().astype(int)
    n_labeled = int(len(labels))
    n_inc = int((labels == LABEL_INCOMPATIBLE).sum())
    n_comp = int((labels == LABEL_COMPATIBLE).sum())
    nonblank_drug = df[COL_DRUG_SMILES][~df[COL_DRUG_SMILES].map(is_blank)]
    nonblank_exc = df[COL_EXC_SMILES][~df[COL_EXC_SMILES].map(is_blank)]
    summary: dict[str, Any] = {
        "n_rows": n_rows,
        "n_labeled_rows": n_labeled,
        "n_compatible": n_comp,
        "n_incompatible": n_inc,
        "pct_compatible": (100.0 * n_comp / n_labeled) if n_labeled else float("nan"),
        "pct_incompatible": (100.0 * n_inc / n_labeled) if n_labeled else float("nan"),
        "unique_drugs_by_smiles": int(nonblank_drug.nunique()),
        "unique_excipients_by_smiles": int(nonblank_exc.nunique()),
        "unique_drugs_by_id": int(df[COL_DRUG_ID][~df[COL_DRUG_ID].map(is_blank)].nunique()),
        "unique_excipients_by_id": int(df[COL_EXC_ID][~df[COL_EXC_ID].map(is_blank)].nunique()),
        "unique_pairs_by_smiles": int(
            pd.Series(
                [
                    make_pair_key(d, e)
                    for d, e in zip(df[COL_DRUG_SMILES], df[COL_EXC_SMILES])
                    if not (is_blank(d) or is_blank(e))
                ]
            ).nunique()
        ),
    }
    if canonical:
        summary["unique_drugs_by_canonical_smiles"] = int(df[COL_DRUG_CANON].nunique())
        summary["unique_excipients_by_canonical_smiles"] = int(df[COL_EXC_CANON].nunique())
        summary["unique_pairs_by_pair_key"] = int(df[COL_PAIR_KEY].nunique())
    return summary


def _pair_key_series(work: pd.DataFrame, mode: str) -> pd.Series:
    if mode == "canonical_smiles":
        d, e = work[COL_DRUG_CANON], work[COL_EXC_CANON]
    elif mode == "raw_smiles":
        d, e = work[COL_DRUG_SMILES], work[COL_EXC_SMILES]
    elif mode == "ids":
        d, e = work[COL_DRUG_ID], work[COL_EXC_ID]
    else:  # pragma: no cover - guarded by config validation
        raise ValueError(f"Unknown pair_key mode '{mode}'")
    return pd.Series([make_pair_key(a, b) for a, b in zip(d, e)], index=work.index, dtype=object)


def clean_dataset(df_std: pd.DataFrame, cfg: Mapping[str, Any]) -> CleaningResult:
    """Clean a standardised raw dataframe (see :func:`src.data.loading.standardize_columns`).

    Removal is applied as a sequential funnel; each removed row is assigned exactly
    one reason so that the per-reason counts add up to the total removed:

    ``missing_required_value`` -> invalid SMILES -> ``conflicting_label``
    -> ``exact_duplicate_pair``.

    Args:
        df_std: Dataframe with columns raw_row, drug_id, excipient_id, drug_smiles,
            excipient_smiles, label_raw.
        cfg: Merged configuration.

    Returns:
        :class:`CleaningResult` with the cleaned dataframe (internal 0/1 labels,
        ``sample_id`` 0..N-1), the audit dataframe of removed rows and a summary dict.

    Raises:
        LabelMappingError: if a non-blank label is not covered by the label mapping.
    """
    data = cfg["data"]
    pair_mode = data["pair_key"]
    work = df_std.copy().reset_index(drop=True)
    # Only whitespace is stripped from SMILES strings (not a chemical operation).
    for col in (COL_DRUG_SMILES, COL_EXC_SMILES):
        work[col] = work[col].astype(str).str.strip()

    label_inspection = inspect_labels(work[COL_LABEL_RAW])
    work[COL_LABEL] = map_labels(work[COL_LABEL_RAW], cfg)  # raises on unknown labels
    original_summary = summarize_dataset(work)

    n = len(work)
    reason = np.full(n, None, dtype=object)
    detail = np.full(n, "", dtype=object)

    # 1. Missing required values --------------------------------------------------
    required_logical = list(data.get("required_values", []))
    if "label" not in required_logical:  # a row without a label can never be used
        required_logical.append("label")
    if pair_mode == "ids":
        for logical in ("drug_id", "excipient_id"):
            if logical not in required_logical:
                required_logical.append(logical)
    required_cols = [(lg, _LOGICAL_TO_INTERNAL[lg]) for lg in required_logical]
    for idx in range(n):
        blanks = [lg for lg, col in required_cols if is_blank(work.at[idx, col])]
        if blanks:
            reason[idx] = REASON_MISSING
            detail[idx] = "missing: " + ",".join(blanks)

    # 2. Invalid SMILES -----------------------------------------------------------
    cache: dict[str, str | None] = {}

    def _canon(smiles: str) -> str | None:
        if smiles not in cache:
            cache[smiles] = canonical_smiles(smiles)
        return cache[smiles]

    drug_canon: list[str | None] = [None] * n
    exc_canon: list[str | None] = [None] * n
    for idx in range(n):
        if reason[idx] is not None:
            continue
        dc = _canon(work.at[idx, COL_DRUG_SMILES])
        ec = _canon(work.at[idx, COL_EXC_SMILES])
        drug_canon[idx], exc_canon[idx] = dc, ec
        if dc is None and ec is None:
            reason[idx] = REASON_INVALID_BOTH
            detail[idx] = "RDKit cannot parse drug or excipient SMILES"
        elif dc is None:
            reason[idx] = REASON_INVALID_DRUG
            detail[idx] = f"RDKit cannot parse drug SMILES: {work.at[idx, COL_DRUG_SMILES]}"
        elif ec is None:
            reason[idx] = REASON_INVALID_EXC
            detail[idx] = f"RDKit cannot parse excipient SMILES: {work.at[idx, COL_EXC_SMILES]}"
    work[COL_DRUG_CANON] = drug_canon
    work[COL_EXC_CANON] = exc_canon

    # 3. Conflicting labels (remove EVERY occurrence of the pair) -------------------
    alive = pd.Series(pd.isna(reason), index=work.index)
    work[COL_PAIR_KEY] = ""
    work.loc[alive, COL_PAIR_KEY] = _pair_key_series(work.loc[alive], pair_mode)
    alive_df = work.loc[alive]
    label_sets = alive_df.groupby(COL_PAIR_KEY)[COL_LABEL].nunique()
    conflict_keys = set(label_sets[label_sets > 1].index)
    conflict_detail: dict[str, str] = {}
    if conflict_keys:
        sub = alive_df[alive_df[COL_PAIR_KEY].isin(conflict_keys)]
        for key, grp in sub.groupby(COL_PAIR_KEY):
            texts = sorted(set(grp[COL_LABEL_RAW].astype(str)))
            conflict_detail[key] = f"conflicting labels {'|'.join(texts)} ({len(grp)} rows for this pair)"
        for idx in sub.index:
            reason[idx] = REASON_CONFLICT
            detail[idx] = conflict_detail[work.at[idx, COL_PAIR_KEY]]

    # 4. Exact duplicate pairs (same label) -> keep the first occurrence -------------
    alive = pd.Series(pd.isna(reason), index=work.index)
    alive_df = work.loc[alive]
    first_row: dict[str, int] = {}
    for idx, key in zip(alive_df.index, alive_df[COL_PAIR_KEY]):
        if key in first_row:
            reason[idx] = REASON_DUPLICATE
            detail[idx] = f"duplicate of raw_row {int(work.at[first_row[key], COL_RAW_ROW])}"
        else:
            first_row[key] = idx
    n_duplicate_groups = int(
        alive_df[COL_PAIR_KEY].value_counts().gt(1).sum()
    )

    work["_reason"] = reason
    work["_detail"] = detail
    removed_mask = work["_reason"].notna()
    audit = work.loc[removed_mask].copy()
    audit = audit.rename(columns={"_reason": "removal_reason", "_detail": "removal_detail"})
    audit_cols = [
        COL_RAW_ROW,
        COL_DRUG_ID,
        COL_EXC_ID,
        COL_DRUG_SMILES,
        COL_EXC_SMILES,
        COL_LABEL_RAW,
        COL_LABEL,
        COL_DRUG_CANON,
        COL_EXC_CANON,
        COL_PAIR_KEY,
        "removal_reason",
        "removal_detail",
    ]
    audit = audit[audit_cols].reset_index(drop=True)

    cleaned = work.loc[~removed_mask].drop(columns=["_reason", "_detail"]).reset_index(drop=True)
    cleaned[COL_SAMPLE_ID] = np.arange(len(cleaned), dtype=np.int64)
    cleaned[COL_LABEL] = cleaned[COL_LABEL].astype(int)
    cleaned = cleaned[CLEANED_COLUMNS]

    removed_by_reason = {r: int((audit["removal_reason"] == r).sum()) for r in REMOVAL_REASONS}
    cleaned_summary = summarize_dataset(cleaned, canonical=True)
    total_removed = int(len(audit))
    if original_summary["n_rows"] != cleaned_summary["n_rows"] + total_removed:  # pragma: no cover
        raise RuntimeError("Internal error: removed + kept rows do not add up to the original row count")

    summary = {
        "original": original_summary,
        "cleaned": cleaned_summary,
        "removed_by_reason": removed_by_reason,
        "total_removed": total_removed,
        "n_conflicting_pairs": len(conflict_keys),
        "n_duplicate_pair_groups_collapsed": n_duplicate_groups,
        "label_inspection": label_inspection,
        "pair_key_mode": pair_mode,
    }
    return CleaningResult(cleaned=cleaned, audit=audit, summary=summary, label_inspection=label_inspection)


def _fmt_pct(value: float) -> str:
    return "n/a" if value != value else f"{value:6.2f}%"


def format_preprocessing_report(
    summary: Mapping[str, Any],
    cfg: Mapping[str, Any],
    raw_path: str,
    raw_sha256: str,
    cleaned_sha256: str,
    generated_at: str,
) -> str:
    """Render the human-readable ``preprocessing_report.txt`` contents."""
    data = cfg["data"]
    orig, clean = summary["original"], summary["cleaned"]
    removed = summary["removed_by_reason"]
    lines: list[str] = []
    add = lines.append
    add("=" * 78)
    add("DE-INTERACT - PREPROCESSING REPORT")
    add("=" * 78)
    add(f"Generated at       : {generated_at}")
    add(f"Raw dataset        : {raw_path}")
    add(f"Raw SHA-256        : {raw_sha256}")
    add(f"Cleaned SHA-256    : {cleaned_sha256}")
    add(f"Pair key mode      : {summary['pair_key_mode']} (ordered drug -> excipient pair)")
    add("")
    add("-" * 78)
    add("1. SOURCE LABELS (inspected before mapping)")
    add("-" * 78)
    for value, count in summary["label_inspection"].items():
        add(f"  '{value}': {count} rows")
    add("")
    add("Label mapping (original -> internal). Internal: 1 = INCOMPATIBLE (positive), 0 = COMPATIBLE")
    for source, internal in data["label_mapping"].items():
        add(f"  '{source}' -> {int(internal)} ({INTERNAL_LABEL_NAMES[int(internal)]})")
    add("")
    add("-" * 78)
    add("2. ORIGINAL vs CLEANED")
    add("-" * 78)
    width = 44
    add(f"{'':{width}}{'ORIGINAL':>14}{'CLEANED':>14}")

    def row(name: str, a: Any, b: Any) -> None:
        add(f"{name:{width}}{str(a):>14}{str(b):>14}")

    row("Number of rows", orig["n_rows"], clean["n_rows"])
    row("Compatible (0) rows", orig["n_compatible"], clean["n_compatible"])
    row("Incompatible (1) rows", orig["n_incompatible"], clean["n_incompatible"])
    row("Compatible %", _fmt_pct(orig["pct_compatible"]).strip(), _fmt_pct(clean["pct_compatible"]).strip())
    row("Incompatible %", _fmt_pct(orig["pct_incompatible"]).strip(), _fmt_pct(clean["pct_incompatible"]).strip())
    row("Unique drugs (raw SMILES)", orig["unique_drugs_by_smiles"], clean["unique_drugs_by_smiles"])
    row("Unique excipients (raw SMILES)", orig["unique_excipients_by_smiles"], clean["unique_excipients_by_smiles"])
    row("Unique drugs (identifier column)", orig["unique_drugs_by_id"], clean["unique_drugs_by_id"])
    row("Unique excipients (identifier column)", orig["unique_excipients_by_id"], clean["unique_excipients_by_id"])
    row("Unique drugs (canonical SMILES)", "-", clean["unique_drugs_by_canonical_smiles"])
    row("Unique excipients (canonical SMILES)", "-", clean["unique_excipients_by_canonical_smiles"])
    row("Unique pairs (raw SMILES / pair key)", orig["unique_pairs_by_smiles"], clean["unique_pairs_by_pair_key"])
    add("(Percentages are relative to rows carrying a label; original rows with blank labels are")
    add(" excluded from the compatible/incompatible counts.)")
    add("")
    add("-" * 78)
    add("3. ROWS REMOVED, BY REASON (sequential funnel; each row counted once)")
    add("-" * 78)
    for reason in REMOVAL_REASONS:
        add(f"  {reason:<40}{removed[reason]:>8}")
    add(f"  {'TOTAL REMOVED':<40}{summary['total_removed']:>8}")
    add(f"  {'ROWS KEPT':<40}{clean['n_rows']:>8}")
    add("")
    add(f"  Pairs with conflicting labels (all their rows removed): {summary['n_conflicting_pairs']}")
    add(f"  Duplicate pair groups collapsed to one retained row     : {summary['n_duplicate_pair_groups_collapsed']}")
    add("")
    add("-" * 78)
    add("4. OPERATIONS PERFORMED (and NOT performed)")
    add("-" * 78)
    add("  performed    : missing required values; RDKit SMILES validity; conflicting labels")
    add("                 (every occurrence removed); exact duplicate pairs (one retained)")
    add("  NOT performed: identifier-duplicate audits, salt stripping / tautomer / charge")
    add("                 standardisation, or any other chemical or statistical cleaning")
    add("  Canonical SMILES are used only to build the pair key and as the molecule string for")
    add("  feature generation.")
    add("=" * 78)
    return "\n".join(lines) + "\n"
