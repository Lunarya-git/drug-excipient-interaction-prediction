"""Raw CSV loading, column standardisation and label mapping."""

from __future__ import annotations

from typing import Any, Mapping

import pandas as pd

from src.data.schema import (
    COL_DRUG_ID,
    COL_DRUG_SMILES,
    COL_EXC_ID,
    COL_EXC_SMILES,
    COL_LABEL_RAW,
    COL_RAW_ROW,
)
from src.utils.config import ConfigError, resolve_path


class LabelMappingError(ValueError):
    """Raised when source labels cannot be mapped with the configured mapping."""


def load_raw_dataframe(cfg: Mapping[str, Any]) -> pd.DataFrame:
    """Read the raw CSV as strings (nothing is parsed, coerced or dropped).

    Raises:
        FileNotFoundError: if the CSV does not exist.
        ConfigError: if configured column names are not present in the CSV header.
    """
    data = cfg["data"]
    path = resolve_path(cfg, data["raw_path"])
    if not path.is_file():
        raise FileNotFoundError(
            f"Raw dataset not found: {path}. Place the CSV there or edit data.raw_path in configs/data.yaml."
        )
    csv_cfg = data.get("csv", {})
    df = pd.read_csv(
        path,
        sep=csv_cfg.get("sep", ","),
        encoding=csv_cfg.get("encoding", "utf-8"),
        dtype=str,
        keep_default_na=False,
        na_filter=False,
    )
    df.columns = [str(c).strip() for c in df.columns]
    configured = [c for c in data["columns"].values() if c]
    absent = [c for c in configured if c not in df.columns]
    if absent:
        raise ConfigError(
            f"Configured CSV columns {absent} are not in {path.name}. Actual header: {list(df.columns)}. "
            "Fix data.columns in configs/data.yaml."
        )
    return df


def standardize_columns(df_raw: pd.DataFrame, cfg: Mapping[str, Any]) -> pd.DataFrame:
    """Rename configured CSV columns to the logical internal names.

    The result has the columns ``raw_row`` (0-based data row number in the raw
    CSV), ``drug_id``, ``excipient_id``, ``drug_smiles``, ``excipient_smiles``
    and ``label_raw`` (original label text). If an identifier column is not
    configured, the corresponding SMILES string is used as the identifier.
    """
    cols = cfg["data"]["columns"]
    out = pd.DataFrame({COL_RAW_ROW: range(len(df_raw))})
    out[COL_DRUG_SMILES] = df_raw[cols["drug_smiles"]].astype(str).to_numpy()
    out[COL_EXC_SMILES] = df_raw[cols["excipient_smiles"]].astype(str).to_numpy()
    out[COL_LABEL_RAW] = df_raw[cols["label"]].astype(str).to_numpy()
    out[COL_DRUG_ID] = (
        df_raw[cols["drug_id"]].astype(str).to_numpy() if cols.get("drug_id") else out[COL_DRUG_SMILES]
    )
    out[COL_EXC_ID] = (
        df_raw[cols["excipient_id"]].astype(str).to_numpy()
        if cols.get("excipient_id")
        else out[COL_EXC_SMILES]
    )
    return out[[COL_RAW_ROW, COL_DRUG_ID, COL_EXC_ID, COL_DRUG_SMILES, COL_EXC_SMILES, COL_LABEL_RAW]]


def is_blank(value: Any) -> bool:
    """True for None/NaN or whitespace-only strings."""
    if value is None:
        return True
    if isinstance(value, float) and pd.isna(value):
        return True
    return str(value).strip() == ""


def _normalise_label(value: str, strip: bool, case_sensitive: bool) -> str:
    text = str(value)
    if strip:
        text = text.strip()
    if not case_sensitive:
        text = text.casefold()
    return text


def inspect_labels(labels: pd.Series) -> dict[str, int]:
    """Count the distinct source label strings (blank cells are reported as '<blank>')."""
    counts = labels.map(lambda v: "<blank>" if is_blank(v) else str(v)).value_counts(dropna=False)
    return {str(k): int(v) for k, v in counts.items()}


def build_label_lookup(label_cfg: Mapping[str, int], strip: bool = True, case_sensitive: bool = False) -> dict[str, int]:
    """Normalise the configured label mapping into a lookup table.

    Raises:
        LabelMappingError: if two source spellings collide onto different internal labels.
    """
    lookup: dict[str, int] = {}
    for source, internal in label_cfg.items():
        key = _normalise_label(source, strip, case_sensitive)
        if key in lookup and lookup[key] != int(internal):
            raise LabelMappingError(f"Label '{source}' collides with another mapped label after normalisation")
        lookup[key] = int(internal)
    return lookup


def map_labels(labels: pd.Series, cfg: Mapping[str, Any]) -> pd.Series:
    """Map source labels to internal integers (1 = INCOMPATIBLE, 0 = COMPATIBLE).

    Blank labels become ``<NA>`` (handled later as missing values). Any non-blank
    label that is not covered by ``data.label_mapping`` raises
    :class:`LabelMappingError` listing the unknown values - labels are never guessed.
    """
    data = cfg["data"]
    match = data.get("label_match", {})
    strip = bool(match.get("strip", True))
    case_sensitive = bool(match.get("case_sensitive", False))
    lookup = build_label_lookup(data["label_mapping"], strip, case_sensitive)

    mapped: list[Any] = []
    unknown: dict[str, int] = {}
    for value in labels:
        if is_blank(value):
            mapped.append(pd.NA)
            continue
        key = _normalise_label(value, strip, case_sensitive)
        if key in lookup:
            mapped.append(lookup[key])
        else:
            unknown[str(value)] = unknown.get(str(value), 0) + 1
            mapped.append(pd.NA)
    if unknown:
        raise LabelMappingError(
            f"Source labels not covered by data.label_mapping: {unknown}. Known mapping: "
            f"{dict(data['label_mapping'])}. Edit configs/data.yaml (internal: 1=INCOMPATIBLE, 0=COMPATIBLE)."
        )
    return pd.Series(mapped, index=labels.index, dtype="Int64")
