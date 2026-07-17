

from __future__ import annotations

import torch
from torch import Tensor, nn

from src.models.heads import ClassificationHead


class GatedAttentionPooling(nn.Module):
    def __init__(self, in_dim: int, attn_dim: int = 128) -> None:
        super().__init__()
        self.V = nn.Linear(in_dim, attn_dim)  # ramo "contenuto"
        self.U = nn.Linear(in_dim, attn_dim)  # ramo "gate"
        self.w = nn.Linear(attn_dim, 1)       # proiezione a scalare

    def forward(self, bag: Tensor) -> tuple[Tensor, Tensor]:
        # bag: (N, in_dim)
        gated = torch.tanh(self.V(bag)) * torch.sigmoid(self.U(bag))  # (N, attn_dim)
        scores = self.w(gated)                                        # (N, 1)
        weights = torch.softmax(scores, dim=0)                        # (N, 1), sommano a 1
        pooled = (weights * bag).sum(dim=0)                           # (in_dim,)
        return pooled, weights

class MILModel(nn.Module):
    def __init__(
        self,
        in_dim: int,
        task: str,
        num_classes: int | None = None,
        attn_dim: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.pool = GatedAttentionPooling(in_dim, attn_dim)
        self.head = ClassificationHead(in_dim, task, num_classes, dropout)

    def forward(self, embeddings: Tensor) -> tuple[Tensor, Tensor]:
        if embeddings.dim() != 2:
            raise ValueError(f"attesa matrice (N, in_dim), ricevuto shape {tuple(embeddings.shape)}.")
        if embeddings.shape[0] == 0:
            raise ValueError("bag vuoto (0 chunk): il documento non ha prodotto embedding.")

        pooled, weights = self.pool(embeddings)      # (in_dim,), (N, 1)
        logits = self.head(pooled.unsqueeze(0))      # (1, out_features)
        return logits, weights