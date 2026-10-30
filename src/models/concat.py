"""CONCAT architecture: the controlled DE-INTERACT-inspired baseline ANN."""

from __future__ import annotations

from typing import Any, Mapping

import torch

from src.features.assemble import RepresentationSpec
from src.models.blocks import PairModel, check_params, make_mlp
from src.models.registry import register_model


@register_model("concat")
class ConcatANN(PairModel):
    """Concatenate drug and excipient features and classify with a fully connected network.

    This is the controlled baseline *inspired by* DE-INTERACT (Patel et al., 2023):
    concatenated 881-bit PubChem fingerprints -> hidden ReLU layers -> output.
    It differs from the published model (single logit with weighted BCE instead of a
    2-unit softmax, AdamW instead of RMSprop, dropout/weight decay, class-weighted loss
    instead of oversampling) and is NOT claimed to be an exact reproduction.

    The input layer adapts to the representation: ``in = 2 * dim_per_molecule``;
    hidden widths, depth, activation and dropout come from the configuration only.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int],
        dropout: float = 0.3,
        activation: str = "relu",
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.net = make_mlp(2 * self.input_dim, hidden_dims, activation=activation, dropout=dropout, out_dim=1)

    def forward(self, x_drug: torch.Tensor, x_exc: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([x_drug, x_exc], dim=-1)).squeeze(-1)

    @classmethod
    def from_config(cls, params: Mapping[str, Any], spec: RepresentationSpec) -> "ConcatANN":
        check_params("concat", params, ["hidden_dims", "dropout", "activation"])
        return cls(
            input_dim=spec.dim,
            hidden_dims=[int(h) for h in params.get("hidden_dims", [1024, 1024, 1024])],
            dropout=float(params.get("dropout", 0.3)),
            activation=str(params.get("activation", "relu")),
        )
