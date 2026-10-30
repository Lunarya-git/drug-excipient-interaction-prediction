"""INTERACT architecture: separate encoders + explicit interaction fusion."""

from __future__ import annotations

from typing import Any, Mapping

import torch
from torch import nn

from src.features.assemble import RepresentationSpec
from src.models.blocks import PairModel, check_params, make_mlp
from src.models.registry import register_model


@register_model("interact")
class InteractANN(PairModel):
    """Encode drug and excipient separately and fuse them with explicit interaction terms.

    ``fused = [h_d, h_e, h_d * h_e, |h_d - h_e|]`` is fed to an MLP classifier.
    With ``shared_encoder=False`` (default) the drug and excipient encoders have
    independent weights; with ``True`` one Siamese encoder is used for both.
    """

    def __init__(
        self,
        input_dim: int,
        encoder_hidden_dims: list[int],
        embedding_dim: int,
        classifier_hidden_dims: list[int],
        dropout: float = 0.3,
        activation: str = "relu",
        shared_encoder: bool = False,
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.embedding_dim = int(embedding_dim)

        def make_encoder() -> nn.Sequential:
            return make_mlp(
                self.input_dim,
                encoder_hidden_dims,
                activation=activation,
                dropout=dropout,
                out_dim=self.embedding_dim,
                final_activation=True,
            )

        self.drug_encoder = make_encoder()
        self.exc_encoder = self.drug_encoder if shared_encoder else make_encoder()
        self.classifier = make_mlp(
            4 * self.embedding_dim, classifier_hidden_dims, activation=activation, dropout=dropout, out_dim=1
        )

    def forward(self, x_drug: torch.Tensor, x_exc: torch.Tensor) -> torch.Tensor:
        h_d = self.drug_encoder(x_drug)
        h_e = self.exc_encoder(x_exc)
        fused = torch.cat([h_d, h_e, h_d * h_e, torch.abs(h_d - h_e)], dim=-1)
        return self.classifier(fused).squeeze(-1)

    @classmethod
    def from_config(cls, params: Mapping[str, Any], spec: RepresentationSpec) -> "InteractANN":
        check_params(
            "interact",
            params,
            [
                "encoder_hidden_dims",
                "embedding_dim",
                "classifier_hidden_dims",
                "dropout",
                "activation",
                "shared_encoder",
            ],
        )
        return cls(
            input_dim=spec.dim,
            encoder_hidden_dims=[int(h) for h in params.get("encoder_hidden_dims", [1024])],
            embedding_dim=int(params.get("embedding_dim", 512)),
            classifier_hidden_dims=[int(h) for h in params.get("classifier_hidden_dims", [512])],
            dropout=float(params.get("dropout", 0.3)),
            activation=str(params.get("activation", "relu")),
            shared_encoder=bool(params.get("shared_encoder", False)),
        )
