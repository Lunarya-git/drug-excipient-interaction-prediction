"""Molecular feature generators.

Every generator maps a list of (canonical) SMILES strings to a 2-D ``numpy``
array with one row per molecule. New generator types are added by decorating a
function with :func:`register_generator` and referencing the type name in
``configs/features.yaml``.
"""

from __future__ import annotations

import importlib
import logging
from typing import Any, Callable, Mapping, Sequence

import numpy as np

logger = logging.getLogger(__name__)

GeneratorFn = Callable[[Sequence[str], Mapping[str, Any], Mapping[str, Any]], np.ndarray]
_GENERATORS: dict[str, GeneratorFn] = {}


class FeatureGenerationError(RuntimeError):
    """Raised when features cannot be generated or fail validation."""


def register_generator(type_name: str) -> Callable[[GeneratorFn], GeneratorFn]:
    """Decorator registering a generator under ``type_name``."""

    def decorator(fn: GeneratorFn) -> GeneratorFn:
        _GENERATORS[type_name] = fn
        return fn

    return decorator


def available_generator_types() -> list[str]:
    """Names of the registered generator types."""
    return sorted(_GENERATORS)


def _mols_from_smiles(smiles: Sequence[str]) -> list[Any]:
    from rdkit import Chem, RDLogger

    RDLogger.DisableLog("rdApp.*")
    mols = []
    bad = []
    for s in smiles:
        mol = Chem.MolFromSmiles(s)
        if mol is None:
            bad.append(s)
        mols.append(mol)
    if bad:
        raise FeatureGenerationError(f"RDKit could not parse {len(bad)} molecule(s), e.g. {bad[:5]}")
    return mols


# --------------------------------------------------------------------------- PubChem
@register_generator("pubchem_fingerprint")
def generate_pubchem(smiles: Sequence[str], fam: Mapping[str, Any], feat: Mapping[str, Any]) -> np.ndarray:
    """881-bit PubChem (CACTVS) fingerprint via scikit-fingerprints.

    The backend is scikit-fingerprints' ``PubChemFingerprint`` (a Python
    implementation based on the PubChem key specification and the CDK
    ``PubchemFingerprinter``). It is not the PaDEL executable.
    """
    backend = str(fam.get("backend", "skfp")).lower()
    if backend != "skfp":
        raise FeatureGenerationError(
            f"Unsupported PubChem backend '{backend}'. This project uses scikit-fingerprints ('skfp'). "
            "No other fingerprint is ever substituted silently."
        )
    try:
        from skfp.fingerprints import PubChemFingerprint
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise FeatureGenerationError(
            "scikit-fingerprints is required for the PubChem fingerprint but could not be imported "
            f"({exc}). Install it with: pip install scikit-fingerprints"
        ) from exc

    n_jobs = int(feat.get("n_jobs", 1))
    transformer = PubChemFingerprint(count=False, sparse=False, n_jobs=n_jobs)
    try:
        arr = np.asarray(transformer.transform(list(smiles)))
    except Exception as exc:
        # Locate the offending molecule(s) to give an actionable error message.
        failures = []
        single = PubChemFingerprint(count=False, sparse=False, n_jobs=1)
        for s in smiles:
            try:
                single.transform([s])
            except Exception:  # noqa: BLE001
                failures.append(s)
        raise FeatureGenerationError(
            f"scikit-fingerprints failed on {len(failures)} molecule(s): {failures[:10]} ({exc})"
        ) from exc
    return arr.astype(np.uint8)


# --------------------------------------------------------------------------- Morgan
@register_generator("morgan")
def generate_morgan(smiles: Sequence[str], fam: Mapping[str, Any], feat: Mapping[str, Any]) -> np.ndarray:
    """Morgan / ECFP bit vectors (RDKit ``rdFingerprintGenerator``)."""
    from rdkit.Chem import rdFingerprintGenerator

    n_bits = int(fam.get("n_bits", 2048))
    gen = rdFingerprintGenerator.GetMorganGenerator(
        radius=int(fam.get("radius", 2)),
        fpSize=n_bits,
        includeChirality=bool(fam.get("use_chirality", False)),
    )
    out = np.zeros((len(smiles), n_bits), dtype=np.uint8)
    for i, mol in enumerate(_mols_from_smiles(smiles)):
        out[i] = gen.GetFingerprintAsNumPy(mol)
    return out


# --------------------------------------------------------------------------- MACCS
@register_generator("maccs")
def generate_maccs(smiles: Sequence[str], fam: Mapping[str, Any], feat: Mapping[str, Any]) -> np.ndarray:
    """RDKit MACCS keys (167 positions; position 0 is always 0)."""
    from rdkit import DataStructs
    from rdkit.Chem import MACCSkeys

    n_bits = int(fam.get("n_bits", 167))
    out = np.zeros((len(smiles), n_bits), dtype=np.uint8)
    for i, mol in enumerate(_mols_from_smiles(smiles)):
        fp = MACCSkeys.GenMACCSKeys(mol)
        buf = np.zeros((fp.GetNumBits(),), dtype=np.uint8)
        DataStructs.ConvertToNumpyArray(fp, buf)
        if buf.shape[0] != n_bits:
            raise FeatureGenerationError(f"RDKit MACCS returned {buf.shape[0]} bits, expected {n_bits}")
        out[i] = buf
    return out


# --------------------------------------------------------------------------- Descriptors
def _descriptor_functions(names: Sequence[str]) -> list[tuple[str, Callable[[Any], float]]]:
    from rdkit import Chem
    from rdkit.Chem import Descriptors

    table = dict(Descriptors.descList)
    funcs: list[tuple[str, Callable[[Any], float]]] = []
    unknown = []
    for name in names:
        if name == "FormalCharge":
            funcs.append((name, lambda m: float(Chem.GetFormalCharge(m))))
        elif name in table:
            funcs.append((name, table[name]))
        else:
            unknown.append(name)
    if unknown:
        raise FeatureGenerationError(
            f"Unknown RDKit descriptor name(s) {unknown}. Valid names are those in "
            "rdkit.Chem.Descriptors.descList, plus 'FormalCharge'."
        )
    return funcs


@register_generator("rdkit_descriptors")
def generate_rdkit_descriptors(smiles: Sequence[str], fam: Mapping[str, Any], feat: Mapping[str, Any]) -> np.ndarray:
    """Small, documented set of RDKit physicochemical descriptors (float32, unscaled)."""
    names = list(fam["descriptors"])
    funcs = _descriptor_functions(names)
    out = np.zeros((len(smiles), len(funcs)), dtype=np.float64)
    for i, mol in enumerate(_mols_from_smiles(smiles)):
        for j, (_, fn) in enumerate(funcs):
            try:
                out[i, j] = float(fn(mol))
            except Exception:  # noqa: BLE001 - descriptor failure becomes NaN, handled by policy below
                out[i, j] = np.nan
    bad = ~np.isfinite(out)
    if bad.any():
        policy = str(feat.get("nonfinite_policy", "error")).lower()
        rows = np.unique(np.where(bad)[0])
        cols = sorted({names[j] for j in np.unique(np.where(bad)[1])})
        if policy == "zero_fill":
            logger.warning(
                "Replacing %d non-finite descriptor values (descriptors %s) with 0 (nonfinite_policy=zero_fill)",
                int(bad.sum()),
                cols,
            )
            out[bad] = 0.0
        else:
            examples = [smiles[r] for r in rows[:5]]
            raise FeatureGenerationError(
                f"{int(bad.sum())} non-finite descriptor values in descriptors {cols}, e.g. molecules {examples}. "
                "Remove the descriptor from features.families.RD.descriptors or set "
                "features.nonfinite_policy: zero_fill."
            )
    return out.astype(np.float32)


def expected_dimension(fam: Mapping[str, Any]) -> int:
    """Expected number of features of a family according to its configuration."""
    if fam["type"] == "rdkit_descriptors":
        return len(fam["descriptors"])
    if "n_bits" in fam:
        return int(fam["n_bits"])
    raise FeatureGenerationError(f"Cannot determine the expected dimension of family type '{fam['type']}'")


def generate_family(
    fam: Mapping[str, Any], smiles: Sequence[str], feat: Mapping[str, Any]
) -> np.ndarray:
    """Generate the feature matrix of one family for ``smiles``."""
    ftype = str(fam["type"])
    if ftype not in _GENERATORS:
        raise FeatureGenerationError(f"No generator registered for type '{ftype}'. Available: {available_generator_types()}")
    return _GENERATORS[ftype](smiles, fam, feat)


def library_versions(fam: Mapping[str, Any]) -> dict[str, str | None]:
    """Versions of the libraries used by a family (for cache metadata)."""
    modules = ["rdkit"]
    if fam["type"] == "pubchem_fingerprint":
        modules.append("skfp")
    out: dict[str, str | None] = {}
    for mod in modules:
        try:
            module = importlib.import_module(mod)
            out[mod] = getattr(module, "__version__", None)
            if mod == "skfp" and out[mod] is None:
                from importlib.metadata import version

                out[mod] = version("scikit-fingerprints")
        except Exception:  # noqa: BLE001
            out[mod] = None
    return out
