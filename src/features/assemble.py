"""Assemble representation matrices for drug-excipient pairs and fit fold-wise scalers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

from src.features.generators import FeatureGenerationError
from src.features.registry import FamilySpec, get_family_specs, parse_representation
from src.features.store import FeatureStore
from src.utils.config import ConfigError


@dataclass(frozen=True)
class BlockSpec:
    """One feature family inside a (possibly combined) molecule vector."""

    code: str
    start: int
    end: int
    scale: bool
    clip_value: float | None
    #: Attention tokens: (name, start, end) slices of the molecule vector.
    token_slices: tuple[tuple[str, int, int], ...]

    @property
    def dim(self) -> int:
        return self.end - self.start


@dataclass(frozen=True)
class RepresentationSpec:
    """Layout of a representation keyword: ordered blocks of the per-molecule vector."""

    keyword: str
    codes: tuple[str, ...]
    blocks: tuple[BlockSpec, ...]

    @property
    def dim(self) -> int:
        """Feature dimension of ONE molecule (drug or excipient)."""
        return self.blocks[-1].end

    @property
    def pair_dim(self) -> int:
        """Dimension of the concatenated [drug, excipient] input."""
        return 2 * self.dim

    @property
    def token_slices(self) -> list[tuple[str, int, int]]:
        return [t for b in self.blocks for t in b.token_slices]

    @property
    def scaled_blocks(self) -> list[BlockSpec]:
        return [b for b in self.blocks if b.scale]


def _token_slices_for_block(code: str, start: int, dim: int, groups: Mapping[str, Any] | None) -> tuple[tuple[str, int, int], ...]:
    if not groups:
        return ((f"{code}", start, start + dim),)
    if "sections" in groups:
        out = []
        for i, (a, b) in enumerate(groups["sections"]):
            a, b = int(a), int(b)
            if not 0 <= a < b <= dim:
                raise ConfigError(f"features.families.{code}.token_groups section [{a}, {b}) is outside 0..{dim}")
            out.append((f"{code}.s{i}", start + a, start + b))
        return tuple(out)
    n_chunks = int(groups.get("equal_chunks", 1))
    if n_chunks < 1:
        raise ConfigError(f"features.families.{code}.token_groups.equal_chunks must be >= 1")
    n_chunks = min(n_chunks, dim)
    bounds = np.linspace(0, dim, n_chunks + 1).round().astype(int)
    return tuple(
        (f"{code}.c{i}", start + int(bounds[i]), start + int(bounds[i + 1])) for i in range(n_chunks) if bounds[i + 1] > bounds[i]
    )


def build_representation_spec(
    keyword: str, specs: Mapping[str, FamilySpec], dims: Mapping[str, int]
) -> RepresentationSpec:
    """Compute the block layout of ``keyword`` given family dimensions."""
    codes = parse_representation(keyword, list(specs))
    blocks: list[BlockSpec] = []
    offset = 0
    for code in codes:
        spec = specs[code]
        if not spec.enabled:
            raise ConfigError(f"Representation '{keyword}' uses family '{code}' which is disabled in features.yaml")
        dim = int(dims[code])
        blocks.append(
            BlockSpec(
                code=code,
                start=offset,
                end=offset + dim,
                scale=spec.scale,
                clip_value=(float(spec.config["clip_value"]) if spec.config.get("clip_value") is not None else None),
                token_slices=_token_slices_for_block(code, offset, dim, spec.config.get("token_groups")),
            )
        )
        offset += dim
    return RepresentationSpec(keyword=keyword, codes=tuple(codes), blocks=tuple(blocks))


def family_dims_from_config(cfg: Mapping[str, Any]) -> dict[str, int]:
    """Expected per-family dimensions from the configuration (no cache needed)."""
    from src.features.generators import expected_dimension

    return {c: expected_dimension(s.config) for c, s in get_family_specs(cfg).items()}


@dataclass
class RepresentationData:
    """Raw (unscaled) per-molecule matrix plus pair -> molecule index arrays."""

    spec: RepresentationSpec
    molecule_matrix: np.ndarray  # (n_molecules, D) float32
    drug_idx: np.ndarray  # (N,) molecule index of each pair's drug
    excipient_idx: np.ndarray  # (N,) molecule index of each pair's excipient

    def raw_pairs(self, sample_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Raw drug and excipient feature matrices for the given sample ids."""
        ids = np.asarray(sample_ids, dtype=np.int64)
        return self.molecule_matrix[self.drug_idx[ids]], self.molecule_matrix[self.excipient_idx[ids]]


def load_representation(store: FeatureStore, cfg: Mapping[str, Any], keyword: str) -> RepresentationData:
    """Load the cached families of ``keyword`` and concatenate them per molecule.

    Raises:
        FeatureGenerationError: if cached dimensions disagree with the configuration.
    """
    specs = get_family_specs(cfg)
    codes = parse_representation(keyword, list(specs))
    arrays = []
    dims: dict[str, int] = {}
    for code in codes:
        arr = store.array(code)
        meta = store.meta(code)
        if arr.shape[1] != int(meta["dim"]):
            raise FeatureGenerationError(f"[{code}] cached array dim {arr.shape[1]} != metadata dim {meta['dim']}")
        dims[code] = int(arr.shape[1])
        arrays.append(arr.astype(np.float32, copy=False))
    spec = build_representation_spec(keyword, specs, dims)
    matrix = np.concatenate(arrays, axis=1).astype(np.float32, copy=False)
    pi = store.pair_index
    return RepresentationData(
        spec=spec,
        molecule_matrix=matrix,
        drug_idx=pi["drug_mol_idx"].to_numpy(dtype=np.int64),
        excipient_idx=pi["excipient_mol_idx"].to_numpy(dtype=np.int64),
    )


@dataclass
class FoldScaler:
    """Standard scaler for the scaled blocks (descriptors), fitted on TRAINING rows only.

    Separate statistics are kept for the drug and the excipient role. Binary
    fingerprint blocks are never modified.
    """

    spec: RepresentationSpec
    stats: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]] = field(default_factory=dict)
    fitted: bool = False

    def fit(self, data: RepresentationData, train_ids: np.ndarray) -> "FoldScaler":
        xd, xe = data.raw_pairs(train_ids)
        self.stats = {}
        for block in self.spec.scaled_blocks:
            sl = slice(block.start, block.end)
            per_role = {}
            for role, x in (("drug", xd), ("excipient", xe)):
                mean = x[:, sl].astype(np.float64).mean(axis=0)
                std = x[:, sl].astype(np.float64).std(axis=0)
                std = np.where(std < 1e-8, 1.0, std)
                per_role[role] = (mean.astype(np.float32), std.astype(np.float32))
            self.stats[block.code] = per_role
        self.fitted = True
        return self

    def transform(self, xd: np.ndarray, xe: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Scale the scaled blocks (returns copies; inputs are not modified)."""
        if not self.fitted:
            raise RuntimeError("FoldScaler must be fitted on training data before transform()")
        if not self.spec.scaled_blocks:
            return xd, xe
        xd, xe = xd.copy(), xe.copy()
        for block in self.spec.scaled_blocks:
            sl = slice(block.start, block.end)
            for role, x in (("drug", xd), ("excipient", xe)):
                mean, std = self.stats[block.code][role]
                z = (x[:, sl] - mean) / std
                if block.clip_value is not None:
                    z = np.clip(z, -block.clip_value, block.clip_value)
                x[:, sl] = z
        return xd, xe

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable parameters (for saving next to each fold)."""
        return {
            code: {role: {"mean": m.tolist(), "std": s.tolist()} for role, (m, s) in per_role.items()}
            for code, per_role in self.stats.items()
        }


def make_fold_arrays(
    data: RepresentationData,
    scaler: FoldScaler,
    sample_ids: Sequence[int] | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Scaled drug and excipient matrices (float32) for ``sample_ids``."""
    xd, xe = data.raw_pairs(np.asarray(sample_ids))
    xd, xe = scaler.transform(xd, xe)
    return np.ascontiguousarray(xd, dtype=np.float32), np.ascontiguousarray(xe, dtype=np.float32)
