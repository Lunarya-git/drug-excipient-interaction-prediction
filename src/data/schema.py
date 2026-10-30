"""Internal column names and label conventions shared by the whole code base."""

from __future__ import annotations

COL_SAMPLE_ID = "sample_id"
COL_RAW_ROW = "raw_row"
COL_DRUG_ID = "drug_id"
COL_EXC_ID = "excipient_id"
COL_DRUG_SMILES = "drug_smiles"
COL_EXC_SMILES = "excipient_smiles"
COL_DRUG_CANON = "drug_canonical"
COL_EXC_CANON = "excipient_canonical"
COL_PAIR_KEY = "pair_key"
COL_LABEL_RAW = "label_raw"
COL_LABEL = "label"

#: Internal label convention (fixed). Incompatible is the positive class.
LABEL_INCOMPATIBLE = 1
LABEL_COMPATIBLE = 0
INTERNAL_LABEL_NAMES = {LABEL_INCOMPATIBLE: "INCOMPATIBLE", LABEL_COMPATIBLE: "COMPATIBLE"}

#: Columns of the cleaned dataset file (in this order).
CLEANED_COLUMNS = [
    COL_SAMPLE_ID,
    COL_RAW_ROW,
    COL_DRUG_ID,
    COL_EXC_ID,
    COL_DRUG_SMILES,
    COL_EXC_SMILES,
    COL_DRUG_CANON,
    COL_EXC_CANON,
    COL_PAIR_KEY,
    COL_LABEL_RAW,
    COL_LABEL,
]

#: Names of the files written by script 01 into ``data.processed_dir``.
CLEANED_FILENAME = "cleaned_dataset.csv"
LABEL_MAPPING_FILENAME = "label_mapping.json"
CLEANING_SUMMARY_FILENAME = "cleaning_summary.json"
AUDIT_FILENAME = "removed_rows_audit.csv"
REPORT_FILENAME = "preprocessing_report.txt"

#: Names of the files written into ``data.splits_dir``.
SPLIT_ASSIGNMENTS_FILENAME = "split_assignments.csv"
SPLIT_INDICES_FILENAME = "split_indices.json"
SPLIT_COUNTS_FILENAME = "split_class_counts.csv"
SPLIT_METADATA_FILENAME = "split_metadata.json"
