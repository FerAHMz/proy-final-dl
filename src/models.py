"""Arquitecturas comparadas: regresión logística, Wide & Deep y DeepFM.

Las tres reciben el mismo tensor de entrada: un índice entero por campo (B, F), ya desplazado
con el offset del campo para que todos los campos compartan una sola tabla de embeddings.

- LogisticRegression: un peso por categoría (equivale a una regresión logística sobre one-hot).
- WideDeep (Cheng et al., 2016): parte lineal + MLP sobre embeddings concatenados.
- DeepFM (Guo et al., 2017): parte lineal + máquina de factorización de 2.º orden + MLP,
  con los embeddings COMPARTIDOS entre la FM y la red profunda. La FM modela de forma
  explícita todas las interacciones por pares <v_i, v_j>, que W&D solo puede aprender
  implícitamente o con cruces hechos a mano.
"""
from __future__ import annotations

import torch
from torch import nn


class FeaturesLinear(nn.Module):
    """Término lineal: sum_i w_i x_i + b, con x one-hot por campo."""

    def __init__(self, num_features: int):
        super().__init__()
        self.w = nn.Embedding(num_features, 1)
        self.bias = nn.Parameter(torch.zeros(1))
        nn.init.zeros_(self.w.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, F) -> (B,)
        return self.w(x).sum(dim=(1, 2)) + self.bias


class FeaturesEmbedding(nn.Module):
    def __init__(self, num_features: int, embed_dim: int):
        super().__init__()
        self.emb = nn.Embedding(num_features, embed_dim)
        nn.init.normal_(self.emb.weight, std=0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, F) -> (B, F, D)
        return self.emb(x)


def mlp(in_dim: int, hidden: list[int], dropout: float) -> nn.Sequential:
    layers: list[nn.Module] = []
    for h in hidden:
        layers += [nn.Linear(in_dim, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
        in_dim = h
    layers.append(nn.Linear(in_dim, 1))
    return nn.Sequential(*layers)


def fm_interaction(e: torch.Tensor) -> torch.Tensor:
    """Suma de todas las interacciones por pares en O(F·D): 0.5 * ((Σv)^2 - Σv^2)."""
    square_of_sum = e.sum(dim=1) ** 2
    sum_of_square = (e ** 2).sum(dim=1)
    return 0.5 * (square_of_sum - sum_of_square).sum(dim=1)


class LogisticRegression(nn.Module):
    def __init__(self, num_features: int, num_fields: int, **_):
        super().__init__()
        self.linear = FeaturesLinear(num_features)

    def forward(self, x):
        return self.linear(x)  # logits


class WideDeep(nn.Module):
    def __init__(self, num_features: int, num_fields: int, embed_dim: int = 16,
                 mlp_dims: list[int] | None = None, dropout: float = 0.2, **_):
        super().__init__()
        self.linear = FeaturesLinear(num_features)
        self.embedding = FeaturesEmbedding(num_features, embed_dim)
        self.dnn = mlp(num_fields * embed_dim, mlp_dims or [256, 128], dropout)
        self.embed_out_dim = num_fields * embed_dim

    def forward(self, x):
        e = self.embedding(x)
        return self.linear(x) + self.dnn(e.view(-1, self.embed_out_dim)).squeeze(1)


class DeepFM(nn.Module):
    def __init__(self, num_features: int, num_fields: int, embed_dim: int = 16,
                 mlp_dims: list[int] | None = None, dropout: float = 0.2, **_):
        super().__init__()
        self.linear = FeaturesLinear(num_features)
        self.embedding = FeaturesEmbedding(num_features, embed_dim)
        self.dnn = mlp(num_fields * embed_dim, mlp_dims or [256, 128], dropout)
        self.embed_out_dim = num_fields * embed_dim

    def forward(self, x):
        e = self.embedding(x)
        return (self.linear(x) + fm_interaction(e)
                + self.dnn(e.view(-1, self.embed_out_dim)).squeeze(1))

    @torch.no_grad()
    def field_interaction_strength(self, x: torch.Tensor) -> torch.Tensor:
        """Magnitud media |<v_i, v_j>| por par de campos sobre un lote (explicabilidad)."""
        e = self.embedding(x)                     # (B, F, D)
        inter = torch.einsum("bfd,bgd->bfg", e, e)  # (B, F, F)
        return inter.abs().mean(dim=0)


MODELS = {"lr": LogisticRegression, "widedeep": WideDeep, "deepfm": DeepFM}


def build_model(name: str, num_features: int, num_fields: int, **cfg) -> nn.Module:
    kwargs = dict(embed_dim=cfg.get("embed_dim", 16), mlp_dims=cfg.get("mlp"),
                  dropout=cfg.get("dropout", 0.2))
    return MODELS[name](num_features, num_fields, **kwargs)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
