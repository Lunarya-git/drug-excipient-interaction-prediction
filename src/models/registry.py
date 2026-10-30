"""Architecture registry: maps ``type`` names used in configs/training.yaml to model classes."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Mapping

if TYPE_CHECKING:  # pragma: no cover
    from src.features.assemble import RepresentationSpec
    from src.models.blocks import PairModel

_MODEL_TYPES: dict[str, type] = {}


def register_model(type_name: str) -> Callable[[type], type]:
    """Class decorator registering a :class:`PairModel` subclass that implements ``from_config``."""

    def decorator(cls: type) -> type:
        _MODEL_TYPES[type_name] = cls
        return cls

    return decorator


def available_model_types() -> list[str]:
    """Registered ``type`` names."""
    _import_all()
    return sorted(_MODEL_TYPES)


def _import_all() -> None:
    # Importing the modules triggers their @register_model decorators.
    from src.models import attention, concat, interact  # noqa: F401


def build_model(arch_cfg: Mapping[str, Any], spec: "RepresentationSpec") -> "PairModel":
    """Instantiate an architecture from its ``models:`` configuration entry.

    Args:
        arch_cfg: Entry of ``models`` (``type`` and ``params``).
        spec: Representation layout; determines the input dimension (and token slices).
    """
    _import_all()
    mtype = str(arch_cfg["type"])
    if mtype not in _MODEL_TYPES:
        raise ValueError(f"Unknown architecture type '{mtype}'. Registered: {sorted(_MODEL_TYPES)}")
    return _MODEL_TYPES[mtype].from_config(arch_cfg.get("params", {}) or {}, spec)  # type: ignore[attr-defined]
