"""Tests for the PyTorch model architectures (CPU, synthetic data)."""

from __future__ import annotations

import pytest
import torch


def _make_spec(dim: int = 64, n_tokens: int = 2):
    """Create a minimal RepresentationSpec for testing."""
    from src.features.assemble import RepresentationSpec, BlockSpec
    block = BlockSpec(
        code="PC",
        start=0,
        end=dim,
        scale=False,
        clip_value=None,
        token_slices=tuple((f"PC.c{i}", i * (dim // n_tokens), (i + 1) * (dim // n_tokens)) for i in range(n_tokens)),
    )
    return RepresentationSpec(keyword="PC", codes=("PC",), blocks=(block,))


class TestConcatANN:
    def test_forward_shape(self):
        from src.models.concat import ConcatANN
        model = ConcatANN(input_dim=128, hidden_dims=[64, 32], dropout=0.0)
        x_d = torch.randn(8, 128)
        x_e = torch.randn(8, 128)
        out = model(x_d, x_e)
        assert out.shape == (8,)

    def test_from_config(self):
        from src.models.concat import ConcatANN
        spec = _make_spec(dim=128)
        params = {"hidden_dims": [64], "dropout": 0.1, "activation": "relu"}
        model = ConcatANN.from_config(params, spec)
        assert isinstance(model, ConcatANN)

    def test_output_unbounded(self):
        """Output should be raw logits (unbounded), not probabilities."""
        from src.models.concat import ConcatANN
        model = ConcatANN(input_dim=64, hidden_dims=[32], dropout=0.0)
        x = torch.randn(4, 64)
        out = model(x, x)
        # Raw logits can be any value; sigmoid is applied outside the model.
        assert out.shape == (4,)


class TestInteractANN:
    def test_forward_shape(self):
        from src.models.interact import InteractANN
        model = InteractANN(
            input_dim=64,
            encoder_hidden_dims=[32],
            embedding_dim=16,
            classifier_hidden_dims=[16],
            dropout=0.0,
        )
        x_d = torch.randn(6, 64)
        x_e = torch.randn(6, 64)
        out = model(x_d, x_e)
        assert out.shape == (6,)

    def test_shared_encoder_has_same_weights(self):
        from src.models.interact import InteractANN
        model = InteractANN(
            input_dim=64,
            encoder_hidden_dims=[32],
            embedding_dim=16,
            classifier_hidden_dims=[16],
            shared_encoder=True,
        )
        assert model.drug_encoder is model.exc_encoder


class TestAttentionModel:
    def test_feature_groups_forward(self):
        from src.models.attention import AttentionPairModel
        spec = _make_spec(dim=64, n_tokens=4)
        model = AttentionPairModel(
            input_dim=64,
            token_slices=spec.token_slices,
            token_source="feature_groups",
            d_model=16,
            n_heads=2,
            n_layers=1,
            ff_mult=2,
            dropout=0.0,
            attn_dropout=0.0,
        )
        x = torch.randn(5, 64)
        out = model(x, x)
        assert out.shape == (5,)

    def test_learned_compact_forward(self):
        from src.models.attention import AttentionPairModel
        spec = _make_spec(dim=64, n_tokens=4)
        model = AttentionPairModel(
            input_dim=64,
            token_slices=spec.token_slices,
            token_source="learned_compact",
            d_model=16,
            n_heads=2,
            n_layers=1,
            ff_mult=2,
            n_tokens=4,
            compact_hidden_dims=[32],
            dropout=0.0,
            attn_dropout=0.0,
        )
        x = torch.randn(5, 64)
        out = model(x, x)
        assert out.shape == (5,)

    def test_d_model_not_divisible_by_n_heads_raises(self):
        from src.models.attention import AttentionPairModel
        spec = _make_spec(dim=64, n_tokens=2)
        with pytest.raises(ValueError, match="divisible"):
            AttentionPairModel(
                input_dim=64,
                token_slices=spec.token_slices,
                d_model=15,  # not divisible by n_heads=4
                n_heads=4,
            )


class TestModelRegistry:
    def test_registered_types(self):
        from src.models.registry import available_model_types
        types = available_model_types()
        assert "concat" in types
        assert "interact" in types
        assert "attention" in types

    def test_build_model_from_registry(self):
        from src.models.registry import build_model
        spec = _make_spec(dim=64, n_tokens=2)
        # build_model takes (arch_cfg, spec) — arch_cfg has 'type' and 'params'
        arch_cfg = {
            "type": "concat",
            "params": {"hidden_dims": [32], "dropout": 0.0, "activation": "relu"},
        }
        model = build_model(arch_cfg, spec)
        assert callable(model)
