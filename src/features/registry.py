"""Feature-family registry and representation-keyword parsing.

A *family* is a single molecular representation identified by a 2-character
code (e.g. ``PC``). A *representation keyword* is a concatenation of family
codes (``PC``, ``PCMG``, ``PCMGMKRD`` ...); its blocks are concatenated in the
order written.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from src.utils.config import ConfigError

CODE_LENGTH = 2


@dataclass(frozen=True)
class FamilySpec:
    """Static description of one feature family from ``features.yaml``."""

    code: str
    name: str
    type: str
    description: str
    scale: bool
    enabled: bool
    config: Mapping[str, Any]


def get_family_specs(cfg: Mapping[str, Any], enabled_only: bool = False) -> dict[str, FamilySpec]:
    """Return the feature families defined in the configuration (in config order)."""
    specs: dict[str, FamilySpec] = {}
    for code, fam in cfg["features"]["families"].items():
        spec = FamilySpec(
            code=str(code),
            name=str(fam.get("name", code)),
            type=str(fam["type"]),
            description=" ".join(str(fam.get("description", "")).split()),
            scale=bool(fam.get("scale", False)),
            enabled=bool(fam.get("enabled", True)),
            config=fam,
        )
        if enabled_only and not spec.enabled:
            continue
        specs[spec.code] = spec
    return specs


def parse_representation(keyword: str, available_codes: list[str] | tuple[str, ...]) -> list[str]:
    """Split a representation keyword into family codes.

    Raises:
        ConfigError: for an empty keyword, unknown or repeated family codes.
    """
    keyword = str(keyword).strip()
    if not keyword or len(keyword) % CODE_LENGTH:
        raise ConfigError(
            f"Representation keyword '{keyword}' must be a concatenation of {CODE_LENGTH}-letter "
            f"family codes from {list(available_codes)}"
        )
    codes = [keyword[i : i + CODE_LENGTH] for i in range(0, len(keyword), CODE_LENGTH)]
    unknown = [c for c in codes if c not in available_codes]
    if unknown:
        raise ConfigError(
            f"Representation '{keyword}' uses unknown feature family code(s) {unknown}. "
            f"Defined families: {list(available_codes)}"
        )
    if len(set(codes)) != len(codes):
        raise ConfigError(f"Representation '{keyword}' repeats a feature family")
    return codes


def representation_description(keyword: str, specs: Mapping[str, FamilySpec]) -> str:
    """Human-readable description, e.g. 'PubChem + Morgan'."""
    codes = parse_representation(keyword, list(specs))
    return " + ".join(specs[c].name for c in codes)


def experiment_id(representation: str, architecture: str) -> str:
    """Compose the experiment identifier, e.g. ``PC-CONCAT``."""
    return f"{representation}-{architecture}"
