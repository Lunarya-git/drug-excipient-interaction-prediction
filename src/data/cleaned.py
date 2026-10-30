"""Loading of the cleaned dataset produced by script 01."""

from __future__ import annotations

from typing import Any, Mapping

import pandas as pd

from src.data.schema import CLEANED_COLUMNS, CLEANED_FILENAME, COL_LABEL, COL_RAW_ROW, COL_SAMPLE_ID
from src.utils.config import resolve_path
from src.utils.io_utils import sha256_file


def load_cleaned_dataset(cfg: Mapping[str, Any]) -> tuple[pd.DataFrame, str]:
    """Load ``cleaned_dataset.csv`` and return it with its SHA-256 hash.

    Raises:
        FileNotFoundError: if script 01 has not been run.
        ValueError: if the file does not have the expected columns / sample ids.
    """
    path = resolve_path(cfg, cfg["data"]["processed_dir"]) / CLEANED_FILENAME
    if not path.is_file():
        raise FileNotFoundError(f"Cleaned dataset not found: {path}. Run scripts/01_prepare_data.py first.")
    df = pd.read_csv(path, dtype=str, keep_default_na=False, na_filter=False)
    missing = [c for c in CLEANED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing columns {missing}")
    for col in (COL_SAMPLE_ID, COL_RAW_ROW, COL_LABEL):
        df[col] = df[col].astype(int)
    if not (df[COL_SAMPLE_ID].to_numpy() == range(len(df))).all():
        raise ValueError("sample_id must equal the row position 0..N-1 in cleaned_dataset.csv")
    return df[CLEANED_COLUMNS], sha256_file(path)
