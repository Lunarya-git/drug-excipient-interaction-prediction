#!/usr/bin/env python
"""Script 01 — Data preparation.

Loads the raw CSV, applies the full cleaning pipeline, runs leakage checks,
creates the stratified dev/test split and CV folds, and writes all artefacts.

Outputs (under outputs/preprocessing/ and data/processed/, data/splits/):
  cleaned_dataset.csv       cleaned data (internal schema)
  removed_rows_audit.csv    removed rows with reasons
  preprocessing_report.txt  human-readable cleaning summary
  data/splits/              split_assignments.csv, split_indices.json, ...

Usage
-----
  python scripts/01_prepare_data.py [--config DIR] [--force] [--dry-run]
  python scripts/01_prepare_data.py --override splits.n_folds=10
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils.config import load_config
from src.utils.logging_utils import setup_logging
from src.data.prepare import run_data_preparation

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="DE-INTERACT: data preparation (script 01)")
    p.add_argument("--config", default=None, metavar="DIR",
                   help="Config directory (default: <repo>/configs)")
    p.add_argument("--override", nargs="*", default=[], metavar="KEY=VALUE",
                   help="Dotted key=value overrides applied after loading configs")
    p.add_argument("--force", action="store_true",
                   help="Archive existing outputs and regenerate them")
    p.add_argument("--dry-run", action="store_true",
                   help="Validate configuration; load and clean the data; "
                        "print the report; but do NOT write any files")
    p.add_argument("--log-level", default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(level=args.log_level)
    cfg = load_config(config_dir=args.config, overrides=args.override)
    logger.info("Configuration loaded and validated.")

    if args.dry_run:
        # Run cleaning and show the report, but skip file writes.
        from src.data.loading import load_raw_dataframe, standardize_columns
        from src.data.cleaning import clean_dataset, format_preprocessing_report
        from src.data.splitting import create_splits
        from src.utils.config import resolve_path
        from src.utils.io_utils import sha256_file
        from datetime import datetime, timezone

        raw_path = resolve_path(cfg, cfg["data"]["raw_path"])
        if not raw_path.is_file():
            logger.error("Raw data file not found: %s", raw_path)
            sys.exit(1)
        raw_df = load_raw_dataframe(cfg)
        raw_sha = sha256_file(raw_path)
        df_std = standardize_columns(raw_df, cfg)
        result = clean_dataset(df_std, cfg)
        sp = cfg["splits"]
        splits = create_splits(
            result.cleaned,
            test_fraction=float(sp["test_fraction"]),
            n_folds=int(sp["n_folds"]),
            seed=int(sp["split_seed"]),
        )
        report = format_preprocessing_report(
            result.summary, cfg,
            raw_path=str(raw_path),
            raw_sha256=raw_sha,
            cleaned_sha256="DRY_RUN",
            generated_at=datetime.now(timezone.utc).isoformat(),
        )
        print(report)
        logger.info(
            "[DRY RUN] Cleaned: %d rows kept, %d removed. Dev=%d, Test=%d, Folds=%d.",
            result.summary["cleaned"]["n_rows"],
            result.summary["total_removed"],
            len(splits.dev_ids),
            len(splits.test_ids),
            splits.n_folds,
        )
        logger.info("[DRY RUN] No files were written.")
        return

    try:
        output = run_data_preparation(cfg, force=args.force, logger=logger)
    except FileExistsError as exc:
        logger.error("%s", exc)
        sys.exit(1)

    logger.info(
        "Script 01 complete. Cleaned dataset: %s | Splits: %s",
        output["cleaned_path"],
        output["splits_dir"],
    )


if __name__ == "__main__":
    main()
