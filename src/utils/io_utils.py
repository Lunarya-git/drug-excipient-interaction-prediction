"""Small file-system and serialisation helpers."""

from __future__ import annotations

import hashlib
import json
import math
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml


class _SafeEncoder(json.JSONEncoder):
    """JSON encoder handling numpy scalars/arrays, Paths and NaN/inf (-> null)."""

    def default(self, o: Any) -> Any:  # noqa: D401
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return _clean_float(float(o))
        if isinstance(o, np.bool_):
            return bool(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, Path):
            return str(o)
        if isinstance(o, (set, tuple)):
            return list(o)
        return super().default(o)


def _clean_float(value: float) -> float | None:
    return None if (math.isnan(value) or math.isinf(value)) else value


def _sanitize(obj: Any) -> Any:
    """Recursively replace NaN/inf floats with None (json.dump would emit NaN)."""
    if isinstance(obj, float):
        return _clean_float(obj)
    if isinstance(obj, np.floating):
        return _clean_float(float(obj))
    if isinstance(obj, dict):
        return {str(k): _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize(v) for v in obj]
    return obj


def ensure_dir(path: str | Path) -> Path:
    """Create ``path`` (and parents) if needed and return it as a Path."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_json(obj: Any, path: str | Path, indent: int = 2) -> Path:
    """Write ``obj`` as JSON (NaN/inf become null)."""
    p = Path(path)
    ensure_dir(p.parent)
    with p.open("w", encoding="utf-8") as handle:
        json.dump(_sanitize(obj), handle, indent=indent, cls=_SafeEncoder)
    return p


def load_json(path: str | Path) -> Any:
    """Read a JSON file."""
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_yaml(obj: Any, path: str | Path) -> Path:
    """Write ``obj`` as YAML (plain Python types only)."""
    p = Path(path)
    ensure_dir(p.parent)
    with p.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(json.loads(json.dumps(_sanitize(obj), cls=_SafeEncoder)), handle, sort_keys=False)
    return p


def write_text(text: str, path: str | Path) -> Path:
    """Write ``text`` to ``path`` (UTF-8)."""
    p = Path(path)
    ensure_dir(p.parent)
    p.write_text(text, encoding="utf-8")
    return p


def sha256_file(path: str | Path, chunk_size: int = 1 << 20) -> str:
    """Return the SHA-256 hex digest of a file."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_strings(values: list[str]) -> str:
    """Return the SHA-256 hex digest of an ordered list of strings."""
    digest = hashlib.sha256()
    for value in values:
        digest.update(value.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def timestamp(fmt: str = "%Y%m%d-%H%M%S") -> str:
    """Current local time formatted for run identifiers."""
    return datetime.now().strftime(fmt)


def archive_existing(paths: list[Path], archive_root: Path) -> Path | None:
    """Move existing files/directories into a time-stamped archive directory.

    Used by ``--force`` flows so that previous artefacts are never silently
    destroyed. Returns the archive directory (or None if nothing existed).
    """
    existing = [p for p in paths if p.exists()]
    if not existing:
        return None
    target = ensure_dir(archive_root / f"archive_{timestamp()}")
    for p in existing:
        shutil.move(str(p), str(target / p.name))
    return target


def git_commit_hash(cwd: str | Path | None = None) -> str | None:
    """Return the current git commit hash (with ``-dirty`` suffix) or None."""
    try:
        rev = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=cwd, timeout=10, check=True
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"], capture_output=True, text=True, cwd=cwd, timeout=10, check=True
        ).stdout.strip()
        return rev + ("-dirty" if dirty else "")
    except (OSError, subprocess.SubprocessError):
        return None
