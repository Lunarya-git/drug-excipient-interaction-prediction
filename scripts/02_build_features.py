#!/usr/bin/env python
"""Script 02 — Build and cache molecular features.

Generates the four feature families (PC, MG, MK, RD) from the canonical
SMILES of all unique molecules in the cleaned dataset, caching them as .npy
files with JSON sidecar metadata.

Run this once after script 01. Subsequent scripts load from cache.

Usage
-----
  python scripts/02_build_features.py [--config DIR] [--force]
  python scripts/02_build_features.py --families PC MG
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
from src.features.store import generate_features

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="DE-INTERACT: build molecular features (script 02)")
    p.add_argument("--config", default=None, metavar="DIR")
    p.add_argument("--override", nargs="*", default=[], metavar="KEY=VALUE")
    p.add_argument("--families", nargs="*", default=None, metavar="CODE",
                   help="Family codes to generate (default: all enabled families in features.yaml)")
    p.add_argument("--force", action="store_true",
                   help="Archive and recompute existing caches")
    p.add_argument("--dry-run", action="store_true",
                   help="Print which families would be generated without computing them")
    p.add_argument("--log-level", default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(level=args.log_level)
    cfg = load_config(config_dir=args.config, overrides=args.override)
    logger.info("Configuration loaded.")

    if args.dry_run:
        from src.features.registry import get_family_specs
        specs = get_family_specs(cfg, enabled_only=True)
        requested = set(args.families) if args.families else set(specs)
        for code, spec in specs.items():
            if code in requested:
                logger.info("[DRY RUN] Would generate: %s (%s)", code, spec.name)
        logger.info("[DRY RUN] No files were written.")
        return

    try:
        result = generate_features(
            cfg,
            families=args.families,
            force=args.force,
            logger=logger,
        )
    except FileExistsError as exc:
        logger.error("%s", exc)
        sys.exit(1)

    logger.info("Script 02 complete. Cache directory: %s", result["cache_dir"])
    for code, status in result.get("status", {}).items():
        logger.info("  %s: %s", code, status)


if __name__ == "__main__":
    main()
