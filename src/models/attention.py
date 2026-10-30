"""ATTN architecture: lightweight Transformer over compact molecule tokens."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import torch
from torch import nn

from src.features.assemble import RepresentationSpec
from src.models.blocks import PairModel, check_params, make_activation, make_mlp
from src.models.registry import register_model


class GroupTokenizer(nn.Module):
    """One token per feature group: ``token_g = Linear_g(x[:, start:end])``.

    Attention therefore runs over a handful of compact tokens (e.g. the 7 PubChem
    key sections), never over raw fingerprint bits.
    """

    def __init__(self, slices: Sequence[tuple[str, int, int]], d_model: int) -> None:
        super().__init__()
        self.bounds = [(int(a), int(b)) for _, a, b in slices]
        self.projections = nn.ModuleList([nn.Linear(b - a, d_model) for a, b in self.bounds])
        self.norm = nn.LayerNorm(d_model)
        self.token_embedding = nn.Parameter(torch.zeros(len(self.bounds), d_model))
        nn.init.normal_(self.token_embedding, std=0.02)

    @property
    def n_tokens(self) -> int:
        return len(self.bounds)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = torch.stack([proj(x[:, a:b]) for proj, (a, b) in zip(self.projections, self.bounds)], dim=1)
        return self.norm(tokens + self.token_embedding)


class CompactTokenizer(nn.Module):
    """Learned compact tokens: an MLP maps the molecule vector to ``n_tokens * d_model`` values."""

    def __init__(
        self, input_dim: int, n_tokens: int, d_model: int, hidden_dims: Sequence[int], dropout: float, activation: str
    ) -> None:
        super().__init__()
        self._n_tokens = int(n_tokens)
        self.d_model = int(d_model)
        self.mlp = make_mlp(input_dim, hidden_dims, activation=activation, dropout=dropout, out_dim=n_tokens * d_model)
        self.norm = nn.LayerNorm(d_model)
        self.token_embedding = nn.Parameter(torch.zeros(n_tokens, d_model))
        nn.init.normal_(self.token_embedding, std=0.02)

    @property
    def n_tokens(self) -> int:
        return self._n_tokens

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.mlp(x).view(x.shape[0], self._n_tokens, self.d_model)
        return self.norm(tokens + self.token_embedding)


class TransformerBlock(nn.Module):
    """Pre-norm self-attention block with separately configurable attention dropout."""

    def __init__(self, d_model: int, n_heads: int, ff_mult: int, dropout: float, attn_dropout: float) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=attn_dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, ff_mult * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_mult * d_model, d_model),
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        attn_out, _ = self.attn(h, h, h, need_weights=False)
        x = x + self.drop(attn_out)
        return x + self.drop(self.ff(self.norm2(x)))


@register_model("attention")
class AttentionPairModel(PairModel):
    """Joint attention over drug tokens, excipient tokens and a CLS token.

    Sequence: ``[CLS, drug tokens + role_drug, excipient tokens + role_excipient]``.
    The sequence is at most a few dozen tokens long, so the model stays lightweight.

    Token sources (``token_source``):

    * ``feature_groups``: one token per feature group of the representation
      (PubChem sections, Morgan/MACCS chunks, one descriptor token); see
      ``token_groups`` in configs/features.yaml.
    * ``learned_compact``: an MLP compresses each molecule into ``n_tokens`` tokens.
    """

    def __init__(
        self,
        input_dim: int,
        token_slices: Sequence[tuple[str, int, int]],
        token_source: str = "feature_groups",
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        ff_mult: int = 2,
        dropout: float = 0.1,
        attn_dropout: float = 0.1,
        pooling: str = "cls",
        classifier_hidden_dims: Sequence[int] = (128,),
        shared_encoder: bool = False,
        n_tokens: int = 8,
        compact_hidden_dims: Sequence[int] = (256,),
        activation: str = "gelu",
    ) -> None:
        super().__init__()
        if d_model % n_heads:
            raise ValueError(f"d_model ({d_model}) must be divisible by n_heads ({n_heads})")
        if pooling not in ("cls", "mean"):
            raise ValueError("pooling must be 'cls' or 'mean'")
        if token_source not in ("feature_groups", "learned_compact"):
            raise ValueError("token_source must be 'feature_groups' or 'learned_compact'")
        self.input_dim = int(input_dim)
        self.token_source = token_source
        self.pooling = pooling
        self.d_model = int(d_model)

        def make_tokenizer() -> nn.Module:
            if token_source == "feature_groups":
                if not token_slices:
                    raise ValueError("feature_groups tokenisation needs token slices from the representation spec")
                if max(b for _, _, b in token_slices) > self.input_dim:
                    raise ValueError("a token slice exceeds the molecule feature dimension")
                return GroupTokenizer(token_slices, d_model)
            return CompactTokenizer(self.input_dim, n_tokens, d_model, compact_hidden_dims, dropout, activation)

        self.drug_tokenizer = make_tokenizer()
        self.exc_tokenizer = self.drug_tokenizer if shared_encoder else make_tokenizer()
        self.role_embedding = nn.Parameter(torch.zeros(2, d_model))
        nn.init.normal_(self.role_embedding, std=0.02)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.cls_token, std=0.02)
        self.blocks = nn.ModuleList(
            [TransformerBlock(d_model, n_heads, ff_mult, dropout, attn_dropout) for _ in range(n_layers)]
        )
        self.final_norm = nn.LayerNorm(d_model)
        self.classifier = make_mlp(d_model, list(classifier_hidden_dims), activation=activation, dropout=dropout, out_dim=1)

    @property
    def sequence_length(self) -> int:
        return 1 + self.drug_tokenizer.n_tokens + self.exc_tokenizer.n_tokens  # type: ignore[operator]

    def forward(self, x_drug: torch.Tensor, x_exc: torch.Tensor) -> torch.Tensor:
        batch = x_drug.shape[0]
        td = self.drug_tokenizer(x_drug) + self.role_embedding[0]
        te = self.exc_tokenizer(x_exc) + self.role_embedding[1]
        seq = torch.cat([self.cls_token.expand(batch, -1, -1), td, te], dim=1)
        for block in self.blocks:
            seq = block(seq)
        seq = self.final_norm(seq)
        pooled = seq[:, 0] if self.pooling == "cls" else seq.mean(dim=1)
        return self.classifier(pooled).squeeze(-1)

    @classmethod
    def from_config(cls, params: Mapping[str, Any], spec: RepresentationSpec) -> "AttentionPairModel":
        check_params(
            "attention",
            params,
            [
                "token_source",
                "d_model",
                "n_heads",
                "n_layers",
                "ff_mult",
                "dropout",
                "attn_dropout",
                "pooling",
                "classifier_hidden_dims",
                "shared_encoder",
                "n_tokens",
                "compact_hidden_dims",
                "activation",
            ],
        )
        return cls(
            input_dim=spec.dim,
            token_slices=spec.token_slices,
            token_source=str(params.get("token_source", "feature_groups")),
            d_model=int(params.get("d_model", 64)),
            n_heads=int(params.get("n_heads", 4)),
            n_layers=int(params.get("n_layers", 2)),
            ff_mult=int(params.get("ff_mult", 2)),
            dropout=float(params.get("dropout", 0.1)),
            attn_dropout=float(params.get("attn_dropout", 0.1)),
            pooling=str(params.get("pooling", "cls")),
            classifier_hidden_dims=[int(h) for h in params.get("classifier_hidden_dims", [128])],
            shared_encoder=bool(params.get("shared_encoder", False)),
            n_tokens=int(params.get("n_tokens", 8)),
            compact_hidden_dims=[int(h) for h in params.get("compact_hidden_dims", [256])],
            activation=str(params.get("activation", "gelu")),
        )
