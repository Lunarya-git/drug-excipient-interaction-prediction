"""Tests for the configuration system."""

from __future__ import annotations

import pytest


CONFIGS_DIR = None  # will be resolved relative to this file


def _cfg_dir():
    from pathlib import Path
    return Path(__file__).resolve().parents[1] / "configs"


class TestLoadConfig:
    def test_load_succeeds(self):
        from src.utils.config import load_config
        cfg = load_config(config_dir=_cfg_dir())
        assert "data" in cfg
        assert "features" in cfg
        assert "training" in cfg
        assert "experiments" in cfg

    def test_label_mapping_has_two_classes(self):
        from src.utils.config import load_config
        cfg = load_config(config_dir=_cfg_dir())
        mapping = cfg["data"]["label_mapping"]
        assert set(mapping.values()) == {0, 1}

    def test_override_applied(self):
        from src.utils.config import load_config
        cfg = load_config(config_dir=_cfg_dir(), overrides=["splits.n_folds=3"])
        assert cfg["splits"]["n_folds"] == 3

    def test_invalid_override_key_raises(self):
        from src.utils.config import load_config, ConfigError
        # Invalid override format (no equals sign)
        with pytest.raises(ConfigError):
            load_config(config_dir=_cfg_dir(), overrides=["splits.n_folds"])

    def test_family_codes_two_uppercase(self):
        from src.utils.config import load_config
        cfg = load_config(config_dir=_cfg_dir())
        for code in cfg["features"]["families"]:
            assert len(code) == 2, f"Family code '{code}' is not 2 characters"
            assert code.isupper(), f"Family code '{code}' is not upper-case"


class TestDeepMerge:
    def test_merge_overrides_scalar(self):
        from src.utils.config import deep_merge
        result = deep_merge({"a": 1, "b": {"c": 2}}, {"a": 99})
        assert result["a"] == 99
        assert result["b"]["c"] == 2

    def test_merge_nested_dict(self):
        from src.utils.config import deep_merge
        result = deep_merge({"a": {"b": 1, "c": 2}}, {"a": {"b": 99}})
        assert result["a"]["b"] == 99
        assert result["a"]["c"] == 2

    def test_base_not_mutated(self):
        from src.utils.config import deep_merge
        base = {"x": 1}
        deep_merge(base, {"x": 2})
        assert base["x"] == 1
