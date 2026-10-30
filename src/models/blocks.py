"""Shared building blocks and the base class of all pair models."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import torch
from torch import nn

ACTIVATIONS: dict[str, type[nn.Module]] = {
    "relu": nn.ReLU,
    "gelu": nn.GELU,
    "silu": nn.SiLU,
    "leaky_relu": nn.LeakyReLU,
    "tanh": nn.Tanh,
}


def make_activation(name: str) -> nn.Module:
    """Instantiate an activation by (case-insensitive) name."""
    key = str(name).lower()
    if key not in ACTIVATIONS:
        raise ValueError(f"Unknown activation '{name}'. Available: {sorted(ACTIVATIONS)}")
    return ACTIVATIONS[key]()


def make_mlp(
    in_dim: int,
    hidden_dims: Sequence[int],
    activation: str = "relu",
    dropout: float = 0.0,
    out_dim: int | None = None,
    final_activation: bool = False,
) -> nn.Sequential:
    """Build ``Linear -> activation -> Dropout`` blocks, optionally ending in a Linear to ``out_dim``.

    Args:
        in_dim: Input feature dimension.
        hidden_dims: Widths of the hidden layers (may be empty).
        activation: Activation name.
        dropout: Dropout probability after each hidden activation.
        out_dim: If given, a final Linear layer to this width is appended (no activation
            unless ``final_activation``).
        final_activation: Apply activation (and dropout) after the final Linear layer.
    """
    layers: list[nn.Module] = []
    prev = int(in_dim)
    for width in hidden_dims:
        layers.append(nn.Linear(prev, int(width)))
        layers.append(make_activation(activation))
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        prev = int(width)
    if out_dim is not None:
        layers.append(nn.Linear(prev, int(out_dim)))
        if final_activation:
            layers.append(make_activation(activation))
    return nn.Sequential(*layers)


def check_params(name: str, params: Mapping[str, Any], allowed: Sequence[str]) -> None:
    """Reject unknown hyper-parameter names (protects against config typos)."""
    unknown = sorted(set(params) - set(allowed))
    if unknown:
        raise ValueError(f"Unknown parameter(s) {unknown} for architecture type '{name}'. Allowed: {sorted(allowed)}")


class PairModel(nn.Module):
    """Base class: ``forward(x_drug, x_excipient) -> logits`` of shape ``(batch,)``.

    The logit is the score of the positive class (INCOMPATIBLE); apply a sigmoid
    for the probability. Subclasses implement ``from_config``.
    """

    def forward(self, x_drug: torch.Tensor, x_exc: torch.Tensor) -> torch.Tensor:  # pragma: no cover - abstract
        raise NotImplementedError

    def count_parameters(self, trainable_only: bool = True) -> int:
        """Number of (trainable) parameters."""
        return int(sum(p.numel() for p in self.parameters() if (p.requires_grad or not trainable_only)))
