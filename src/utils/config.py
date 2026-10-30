"""Central YAML configuration loading, merging, overriding and validation."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

REPO_ROOT: Path = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_DIR: Path = REPO_ROOT / "configs"
CONFIG_FILES: tuple[str, ...] = ("data.yaml", "features.yaml", "training.yaml", "experiments.yaml")

_VALID_PAIR_KEYS = ("canonical_smiles", "raw_smiles", "ids")
_REQUIRED_TOP_LEVEL = (
    "data",
    "splits",
    "features",
    "training",
    "hardware",
    "evaluation",
    "models",
    "experiments",
)
_LOGICAL_COLUMNS = ("drug_id", "excipient_id", "drug_smiles", "excipient_smiles", "label")


class ConfigError(ValueError):
    """Raised when the configuration is missing, inconsistent or invalid."""


def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """Recursively merge ``override`` into a deep copy of ``base``.

    Dictionaries are merged key by key; any other value (including lists) in
    ``override`` replaces the value in ``base``.
    """
    merged: dict[str, Any] = copy.deepcopy(dict(base))
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _set_dotted(cfg: dict[str, Any], dotted: str, value: Any) -> None:
    parts = dotted.split(".")
    node = cfg
    for part in parts[:-1]:
        if part not in node or not isinstance(node[part], dict):
            node[part] = {}
        node = node[part]
    node[parts[-1]] = value


def apply_overrides(cfg: dict[str, Any], overrides: Sequence[str]) -> dict[str, Any]:
    """Apply ``dotted.key=value`` overrides (value parsed as YAML) to ``cfg`` in place."""
    for item in overrides:
        if "=" not in item:
            raise ConfigError(f"Invalid override '{item}'. Expected the form dotted.key=value")
        key, raw = item.split("=", 1)
        _set_dotted(cfg, key.strip(), yaml.safe_load(raw))
    return cfg


def load_config(
    config_dir: str | Path | None = None, overrides: Sequence[str] = ()
) -> dict[str, Any]:
    """Load and merge all YAML config files.

    Args:
        config_dir: Directory containing the YAML files. Defaults to ``<repo>/configs``.
        overrides: Optional ``dotted.key=value`` strings applied after loading.

    Returns:
        A single merged configuration dictionary. The special key ``_meta``
        holds the resolved ``repo_root`` and ``config_dir``.
    """
    cdir = Path(config_dir).resolve() if config_dir else DEFAULT_CONFIG_DIR
    if not cdir.is_dir():
        raise ConfigError(f"Config directory not found: {cdir}")
    cfg: dict[str, Any] = {}
    for fname in CONFIG_FILES:
        fpath = cdir / fname
        if not fpath.is_file():
            raise ConfigError(f"Missing configuration file: {fpath}")
        with fpath.open("r", encoding="utf-8") as handle:
            content = yaml.safe_load(handle) or {}
        if not isinstance(content, dict):
            raise ConfigError(f"{fpath} must contain a YAML mapping at the top level")
        duplicate = set(content) & set(cfg)
        if duplicate:
            raise ConfigError(f"Top-level keys {sorted(duplicate)} in {fname} are defined twice")
        cfg.update(content)
    cfg["_meta"] = {"config_dir": str(cdir), "repo_root": str(cdir.parent)}
    apply_overrides(cfg, overrides)
    validate_config(cfg)
    return cfg


def repo_root(cfg: Mapping[str, Any]) -> Path:
    """Return the repository root recorded in ``cfg``."""
    return Path(cfg["_meta"]["repo_root"])


def resolve_path(cfg: Mapping[str, Any], path: str | Path) -> Path:
    """Resolve ``path`` against the repository root (absolute paths are kept)."""
    p = Path(path)
    return p if p.is_absolute() else repo_root(cfg) / p


def config_hash(obj: Any) -> str:
    """Return a short, stable SHA-256 hash of a JSON-serialisable object."""
    payload = json.dumps(obj, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def validate_config(cfg: Mapping[str, Any]) -> None:
    """Validate structural consistency of the merged configuration.

    Raises:
        ConfigError: describing the first problem found.
    """
    missing = [k for k in _REQUIRED_TOP_LEVEL if k not in cfg]
    if missing:
        raise ConfigError(f"Missing top-level configuration sections: {missing}")

    data = cfg["data"]
    columns = data.get("columns", {})
    for logical in _LOGICAL_COLUMNS:
        if logical not in columns:
            raise ConfigError(f"data.columns is missing the logical column '{logical}'")
    for logical in ("drug_smiles", "excipient_smiles", "label"):
        if not columns.get(logical):
            raise ConfigError(f"data.columns.{logical} must be set to a CSV column name")
    for logical in data.get("required_values", []):
        if logical not in columns:
            raise ConfigError(f"data.required_values contains unknown logical column '{logical}'")

    mapping = data.get("label_mapping", {})
    if not mapping:
        raise ConfigError("data.label_mapping must define at least the two source labels")
    internal_values = set(mapping.values())
    if internal_values != {0, 1}:
        raise ConfigError(
            "data.label_mapping must map source labels onto exactly the internal values "
            f"{{0 (COMPATIBLE), 1 (INCOMPATIBLE)}}; got {sorted(internal_values)}"
        )
    if data.get("pair_key") not in _VALID_PAIR_KEYS:
        raise ConfigError(f"data.pair_key must be one of {_VALID_PAIR_KEYS}")
    if data["pair_key"] == "ids" and not (columns.get("drug_id") and columns.get("excipient_id")):
        raise ConfigError("data.pair_key=ids requires data.columns.drug_id and excipient_id")

    splits = cfg["splits"]
    if not 0.0 < float(splits["test_fraction"]) < 0.5:
        raise ConfigError("splits.test_fraction must be in (0, 0.5)")
    if int(splits["n_folds"]) < 2:
        raise ConfigError("splits.n_folds must be >= 2")

    families = cfg["features"].get("families", {})
    if not families:
        raise ConfigError("features.families must define at least one feature family")
    for code, fam in families.items():
        if len(str(code)) != 2 or not str(code).isalnum() or str(code) != str(code).upper():
            raise ConfigError(
                f"Feature family code '{code}' must be exactly 2 upper-case alphanumeric characters"
            )
        if "type" not in fam:
            raise ConfigError(f"features.families.{code} needs a 'type'")

    for key, model in cfg["models"].items():
        if "type" not in model:
            raise ConfigError(f"models.{key} needs a 'type'")
        if "-" in str(key):
            raise ConfigError(f"Architecture keyword '{key}' must not contain '-'")

    stages = cfg["experiments"].get("stages", {})
    if not stages:
        raise ConfigError("experiments.stages must define at least one stage")
    for stage_key, stage in stages.items():
        archs = _stage_architectures(stage)
        unknown = [a for a in archs if a not in cfg["models"]]
        if unknown:
            raise ConfigError(
                f"Stage '{stage_key}' references architectures {unknown} that are not defined "
                f"under models: in training.yaml (known: {sorted(cfg['models'])})"
            )


def _stage_architectures(stage: Mapping[str, Any]) -> list[str]:
    if "architectures" in stage and stage["architectures"] is not None:
        return list(stage["architectures"])
    if stage.get("architecture"):
        return [stage["architecture"]]
    return []


def get_stage_config(cfg: Mapping[str, Any], stage_key: str) -> dict[str, Any]:
    """Return a normalised stage definition.

    The result always contains ``representations`` (list[str]) and
    ``architectures`` (list[str]) plus ``overrides`` (dict).

    Raises:
        ConfigError: if the stage is unknown or its candidate lists are empty.
    """
    stages = cfg["experiments"]["stages"]
    if stage_key not in stages:
        raise ConfigError(f"Unknown stage '{stage_key}'. Defined stages: {sorted(stages)}")
    stage = stages[stage_key]
    reps = [str(r) for r in (stage.get("representations") or [])]
    archs = _stage_architectures(stage)
    if not reps:
        raise ConfigError(
            f"experiments.stages.{stage_key}.representations in configs/experiments.yaml is empty. "
            "Fill it with the representation keywords you want to run (this list is never "
            "chosen automatically)."
        )
    if not archs:
        raise ConfigError(
            f"experiments.stages.{stage_key} defines no architecture(s). Set 'architecture' "
            "or 'architectures' to keys of models: in configs/training.yaml."
        )
    return {
        "key": stage_key,
        "description": stage.get("description", ""),
        "representations": reps,
        "architectures": archs,
        "overrides": dict(stage.get("overrides") or {}),
    }
