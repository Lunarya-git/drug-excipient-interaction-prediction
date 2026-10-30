"""Tests for the data cleaning pipeline."""

from __future__ import annotations

import pandas as pd
import pytest

from src.data.schema import (
    COL_DRUG_CANON,
    COL_DRUG_SMILES,
    COL_EXC_CANON,
    COL_EXC_SMILES,
    COL_LABEL,
    COL_LABEL_RAW,
    COL_RAW_ROW,
    COL_DRUG_ID,
    COL_EXC_ID,
)


MINIMAL_CFG = {
    "_meta": {"repo_root": ".", "config_dir": "configs"},
    "data": {
        "pair_key": "canonical_smiles",
        "required_values": ["drug_smiles", "excipient_smiles", "label"],
        "label_mapping": {"Imcompatible": 1, "Compatible": 0},
        "label_match": {"strip": True, "case_sensitive": False},
        "columns": {
            "drug_id": "drug_cid",
            "excipient_id": "excipient_cid",
            "drug_smiles": "drug_smiles",
            "excipient_smiles": "excipient_smiles",
            "label": "output_value",
        },
    },
}


def _make_std_df(rows: list[dict]) -> pd.DataFrame:
    """Build a standardised dataframe from raw dicts."""
    df = pd.DataFrame(rows)
    # Ensure all internal columns exist.
    for col in [COL_RAW_ROW, COL_DRUG_ID, COL_EXC_ID, COL_DRUG_SMILES, COL_EXC_SMILES, COL_LABEL_RAW]:
        if col not in df.columns:
            df[col] = ""
    df[COL_RAW_ROW] = range(len(df))
    return df


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestCanonicalSmiles:
    def test_valid_smiles_returns_canonical(self):
        from src.data.cleaning import canonical_smiles
        result = canonical_smiles("c1ccccc1")
        assert result is not None
        assert isinstance(result, str)

    def test_invalid_smiles_returns_none(self):
        from src.data.cleaning import canonical_smiles
        result = canonical_smiles("NOTASMILES!!!")
        assert result is None

    def test_ethanol_canonical(self):
        from src.data.cleaning import canonical_smiles
        # Different representations of ethanol should produce the same canonical form.
        c1 = canonical_smiles("CCO")
        c2 = canonical_smiles("OCC")
        assert c1 is not None
        assert c1 == c2


class TestCleanDataset:
    def _run(self, rows):
        from src.data.cleaning import clean_dataset
        df = _make_std_df(rows)
        return clean_dataset(df, MINIMAL_CFG)

    def test_clean_simple_pair_kept(self):
        result = self._run([
            {COL_DRUG_SMILES: "CCO", COL_EXC_SMILES: "c1ccccc1", COL_LABEL_RAW: "Compatible"},
        ])
        assert len(result.cleaned) == 1
        assert len(result.audit) == 0

    def test_missing_smiles_removed(self):
        result = self._run([
            {COL_DRUG_SMILES: "", COL_EXC_SMILES: "c1ccccc1", COL_LABEL_RAW: "Compatible"},
        ])
        assert len(result.cleaned) == 0
        assert len(result.audit) == 1

    def test_invalid_drug_smiles_removed(self):
        result = self._run([
            {COL_DRUG_SMILES: "INVALID", COL_EXC_SMILES: "c1ccccc1", COL_LABEL_RAW: "Compatible"},
        ])
        assert len(result.cleaned) == 0
        assert result.audit.iloc[0]["removal_reason"] == "invalid_drug_smiles"

    def test_conflicting_labels_all_removed(self):
        result = self._run([
            {COL_DRUG_SMILES: "CCO", COL_EXC_SMILES: "c1ccccc1", COL_LABEL_RAW: "Compatible"},
            {COL_DRUG_SMILES: "CCO", COL_EXC_SMILES: "c1ccccc1", COL_LABEL_RAW: "Imcompatible"},
        ])
        assert len(result.cleaned) == 0
        assert result.summary["n_conflicting_pairs"] == 1

    def test_duplicate_pair_collapsed(self):
        result = self._run([
            {COL_DRUG_SMILES: "CCO", COL_EXC_SMILES: "c1ccccc1", COL_LABEL_RAW: "Compatible"},
            {COL_DRUG_SMILES: "CCO", COL_EXC_SMILES: "c1ccccc1", COL_LABEL_RAW: "Compatible"},
        ])
        assert len(result.cleaned) == 1
        assert len(result.audit) == 1
        assert result.audit.iloc[0]["removal_reason"] == "exact_duplicate_pair"

    def test_label_mapping_correct(self):
        result = self._run([
            {COL_DRUG_SMILES: "CCO", COL_EXC_SMILES: "c1ccccc1", COL_LABEL_RAW: "Imcompatible"},
        ])
        assert len(result.cleaned) == 1
        assert int(result.cleaned.iloc[0][COL_LABEL]) == 1

    def test_summary_row_counts_add_up(self):
        result = self._run([
            {COL_DRUG_SMILES: "CCO", COL_EXC_SMILES: "c1ccccc1", COL_LABEL_RAW: "Compatible"},
            {COL_DRUG_SMILES: "INVALID", COL_EXC_SMILES: "c1ccccc1", COL_LABEL_RAW: "Compatible"},
        ])
        orig_n = result.summary["original"]["n_rows"]
        kept = result.summary["cleaned"]["n_rows"]
        removed = result.summary["total_removed"]
        assert orig_n == kept + removed
