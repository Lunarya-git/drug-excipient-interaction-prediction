"""conftest.py — shared fixtures for the test suite."""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def cfg(repo_root):
    from src.utils.config import load_config
    return load_config(config_dir=repo_root / "configs")
