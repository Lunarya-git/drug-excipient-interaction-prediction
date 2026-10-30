"""Feature cache: molecule table, per-family arrays, metadata and (re)generation."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from tqdm import tqdm as _tqdm

from src.data.cleaned import load_cleaned_dataset
from src.data.schema import COL_DRUG_CANON, COL_EXC_CANON, COL_SAMPLE_ID
from src.features.generators import (
    FeatureGenerationError,
    expected_dimension,
    generate_family,
    library_versions,
)
from src.features.registry import FamilySpec, get_family_specs
from src.utils.config import ConfigError, config_hash, resolve_path
from src.utils.io_utils import archive_existing, ensure_dir, load_json, save_json, sha256_file, sha256_strings

MOLECULES_FILENAME = "molecules.csv"
PAIR_INDEX_FILENAME = "pair_index.csv"
MANIFEST_FILENAME = "feature_manifest.json"

#: Keys of a family configuration that influence the generated numbers.
_VALUE_DEFINING_KEYS = ("type", "backend", "n_bits", "radius", "use_chirality", "descriptors")


def feature_cache_dir(cfg: Mapping[str, Any]) -> Path:
    """Directory holding the cached features."""
    return resolve_path(cfg, cfg["features"]["cache_dir"])


def family_value_config(fam: Mapping[str, Any], feat: Mapping[str, Any]) -> dict[str, Any]:
    """The subset of the family configuration that defines the generated values."""
    subset = {k: fam[k] for k in _VALUE_DEFINING_KEYS if k in fam}
    if fam["type"] == "rdkit_descriptors":
        subset["nonfinite_policy"] = feat.get("nonfinite_policy", "error")
    return subset


def build_molecule_tables(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build the unique-molecule table and the per-pair molecule index.

    Molecules are unique canonical SMILES (drugs and excipients share one index
    space), ordered by first appearance (drug before excipient within a row).
    """
    index: dict[str, int] = {}
    order: list[str] = []
    drug_flags: set[str] = set(df[COL_DRUG_CANON])
    exc_flags: set[str] = set(df[COL_EXC_CANON])
    for d, e in zip(df[COL_DRUG_CANON], df[COL_EXC_CANON]):
        for s in (d, e):
            if s not in index:
                index[s] = len(order)
                order.append(s)
    molecules = pd.DataFrame(
        {
            "mol_idx": np.arange(len(order)),
            "canonical_smiles": order,
            "is_drug": [s in drug_flags for s in order],
            "is_excipient": [s in exc_flags for s in order],
        }
    )
    pair_index = pd.DataFrame(
        {
            COL_SAMPLE_ID: df[COL_SAMPLE_ID].to_numpy(),
            "drug_mol_idx": [index[s] for s in df[COL_DRUG_CANON]],
            "excipient_mol_idx": [index[s] for s in df[COL_EXC_CANON]],
        }
    )
    return molecules, pair_index


@dataclass
class FeatureStore:
    """Read access to cached features."""

    cache_dir: Path
    molecules: pd.DataFrame
    pair_index: pd.DataFrame
    manifest: dict[str, Any]

    @property
    def n_molecules(self) -> int:
        return int(len(self.molecules))

    def families_available(self) -> list[str]:
        return sorted(self.manifest.get("families", {}))

    def meta(self, code: str) -> dict[str, Any]:
        path = self.cache_dir / f"{code}.meta.json"
        if not path.is_file():
            raise FileNotFoundError(
                f"No cached features for family '{code}' in {self.cache_dir}. Run scripts/02_generate_features.py."
            )
        return load_json(path)

    def array(self, code: str) -> np.ndarray:
        """Return the cached (n_molecules, dim) array of a family (raw, unscaled)."""
        path = self.cache_dir / f"{code}.npy"
        if not path.is_file():
            raise FileNotFoundError(
                f"No cached features for family '{code}' ({path}). Run scripts/02_generate_features.py."
            )
        arr = np.load(path)
        if arr.shape[0] != self.n_molecules:
            raise FeatureGenerationError(
                f"Cached array {path.name} has {arr.shape[0]} rows but there are {self.n_molecules} molecules"
            )
        return arr

    @classmethod
    def load(cls, cfg: Mapping[str, Any], expected_dataset_sha256: str | None = None) -> "FeatureStore":
        """Load the store; optionally verify it matches the current cleaned dataset."""
        cache_dir = feature_cache_dir(cfg)
        mpath = cache_dir / MANIFEST_FILENAME
        if not mpath.is_file():
            raise FileNotFoundError(
                f"Feature cache manifest not found ({mpath}). Run scripts/02_generate_features.py first."
            )
        manifest = load_json(mpath)
        if expected_dataset_sha256 and manifest.get("cleaned_dataset_sha256") != expected_dataset_sha256:
            raise ValueError(
                "The feature cache was built from a different cleaned dataset. "
                "Re-run scripts/02_generate_features.py --force."
            )
        molecules = pd.read_csv(cache_dir / MOLECULES_FILENAME, dtype={"canonical_smiles": str}, keep_default_na=False)
        pair_index = pd.read_csv(cache_dir / PAIR_INDEX_FILENAME)
        return cls(cache_dir=cache_dir, molecules=molecules, pair_index=pair_index, manifest=manifest)


def _family_is_cached(cache_dir: Path, spec: FamilySpec, cfg: Mapping[str, Any], mol_hash: str) -> tuple[bool, str]:
    """Return (valid, reason) for the cache of one family."""
    meta_path = cache_dir / f"{spec.code}.meta.json"
    arr_path = cache_dir / f"{spec.code}.npy"
    if not meta_path.is_file() or not arr_path.is_file():
        return False, "not cached"
    meta = load_json(meta_path)
    cfg_hash = config_hash(family_value_config(spec.config, cfg["features"]))
    if meta.get("config_hash") != cfg_hash:
        return False, "family configuration changed since the cache was created"
    if meta.get("molecule_hash") != mol_hash:
        return False, "molecule set changed since the cache was created"
    if meta.get("array_sha256") != sha256_file(arr_path):
        return False, "cached array file does not match its recorded checksum"
    return True, "valid"


def generate_features(
    cfg: Mapping[str, Any],
    families: Sequence[str] | None = None,
    force: bool = False,
    logger: logging.Logger | None = None,
) -> dict[str, Any]:
    """Generate and cache the feature families (skipping valid caches).

    Args:
        cfg: Merged configuration.
        families: Family codes to generate; default: all enabled families.
        force: Regenerate caches that are stale or invalid. Previous files are
            moved to a time-stamped archive folder, never deleted silently.
        logger: Optional logger.

    Returns:
        Dict with per-family status (``generated`` / ``cache_hit``).

    Raises:
        FileExistsError: if a cache exists but is stale/invalid and ``force`` is False.
    """
    log = logger or logging.getLogger(__name__)
    feat = cfg["features"]
    df, dataset_hash = load_cleaned_dataset(cfg)
    cache_dir = ensure_dir(feature_cache_dir(cfg))
    specs = get_family_specs(cfg, enabled_only=True)
    if families:
        unknown = [c for c in families if c not in get_family_specs(cfg)]
        if unknown:
            raise ConfigError(f"Unknown feature family code(s) {unknown}")
        specs = {c: s for c, s in get_family_specs(cfg).items() if c in families}
    if not specs:
        raise ConfigError("No feature families selected / enabled.")

    molecules, pair_index = build_molecule_tables(df)
    mol_hash = sha256_strings(molecules["canonical_smiles"].tolist())
    smiles = molecules["canonical_smiles"].tolist()

    # Molecule tables -----------------------------------------------------------------
    mol_path, pair_path, man_path = (cache_dir / MOLECULES_FILENAME, cache_dir / PAIR_INDEX_FILENAME, cache_dir / MANIFEST_FILENAME)
    manifest: dict[str, Any] = load_json(man_path) if man_path.is_file() else {}
    dataset_changed = bool(manifest) and manifest.get("cleaned_dataset_sha256") != dataset_hash
    if dataset_changed:
        if not force:
            raise FileExistsError(
                "The cleaned dataset changed since the feature cache was created. Re-run with --force "
                "to archive the old cache and regenerate all features."
            )
        stale = [p for p in cache_dir.iterdir() if p.name != "_archive"]
        archived = archive_existing(stale, cache_dir / "_archive")
        log.warning("Archived stale feature cache to %s", archived)
        manifest = {}
    molecules.to_csv(mol_path, index=False)
    pair_index.to_csv(pair_path, index=False)
    manifest.update(
        {
            "cleaned_dataset_sha256": dataset_hash,
            "molecule_hash": mol_hash,
            "n_molecules": int(len(molecules)),
            "n_pairs": int(len(pair_index)),
            "updated_at": datetime.now().isoformat(timespec="seconds"),
        }
    )
    manifest.setdefault("families", {})

    status: dict[str, str] = {}
    for code, spec in specs.items():
        valid, why = _family_is_cached(cache_dir, spec, cfg, mol_hash)
        if valid:
            log.info("[%s] cache hit - skipping generation (%s)", code, spec.name)
            status[code] = "cache_hit"
            manifest["families"][code] = load_json(cache_dir / f"{code}.meta.json")
            continue
        exists = (cache_dir / f"{code}.npy").exists()
        if exists and not force:
            raise FileExistsError(
                f"[{code}] cached features exist but are not valid ({why}). Re-run with --force to archive and "
                "regenerate them."
            )
        if exists:
            archived = archive_existing(
                [cache_dir / f"{code}.npy", cache_dir / f"{code}.meta.json"], cache_dir / "_archive"
            )
            log.warning("[%s] archived previous cache to %s (%s)", code, archived, why)

        log.info("[%s] generating %s for %d unique molecules ...", code, spec.name, len(smiles))
        t0 = time.perf_counter()
        with _tqdm(
            total=len(smiles),
            desc=f"  [{code}] {spec.name}",
            unit="mol",
            dynamic_ncols=True,
            leave=False,
            colour="cyan",
        ) as mol_bar:
            arr = generate_family(spec.config, smiles, feat)
            mol_bar.update(len(smiles) - mol_bar.n)  # mark complete
        seconds = time.perf_counter() - t0
        dim = expected_dimension(spec.config)
        if arr.ndim != 2 or arr.shape != (len(smiles), dim):
            raise FeatureGenerationError(
                f"[{code}] generated array has shape {arr.shape}, expected ({len(smiles)}, {dim})"
            )
        arr_path = cache_dir / f"{code}.npy"
        np.save(arr_path, arr)
        meta = {
            "code": code,
            "name": spec.name,
            "type": spec.type,
            "dim": int(arr.shape[1]),
            "n_molecules": int(arr.shape[0]),
            "dtype": str(arr.dtype),
            "scale": spec.scale,
            "config": family_value_config(spec.config, feat),
            "config_hash": config_hash(family_value_config(spec.config, feat)),
            "molecule_hash": mol_hash,
            "array_sha256": sha256_file(arr_path),
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "generation_seconds": round(seconds, 2),
            "library_versions": library_versions(spec.config),
            "descriptor_names": list(spec.config.get("descriptors", [])) or None,
        }
        save_json(meta, cache_dir / f"{code}.meta.json")
        manifest["families"][code] = meta
        status[code] = "generated"
        log.info("[%s] done in %.1fs -> %s %s", code, seconds, arr_path.name, arr.shape)
    save_json(manifest, man_path)
    return {"status": status, "cache_dir": str(cache_dir), "n_molecules": int(len(molecules))}
