"""
mil.py — Multiple Instance Learning con gated attention pooling (Ilse et al. 2018).

Legge un bag di embedding gia calcolati dalla cache (matrice (N_chunk, hidden)) e:
  1. assegna un peso di attenzione a ciascun chunk, INDIPENDENTEMENTE dagli altri
  2. produce il vettore-documento come media pesata dei chunk (embedding-based MIL)
  3. classifica il vettore-documento con la testa condivisa di heads.py

Cosa NON fa (ed e il punto del confronto con la GNN): i chunk NON si scambiano
informazione tra loro. Ogni peso dipende solo dal proprio chunk -> il bag e un
INSIEME NON ORDINATO. La GNN, a differenza, fara message passing tra chunk PRIMA
di aggregare. Stessa cache, stessa testa: l'unica differenza sara l'aggregazione.

I pesi di attenzione sono restituiti insieme ai logit: sono la chiave
interpretativa (quali chunk hanno pesato di piu nella predizione).
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from src.models.heads import ClassificationHead


class GatedAttentionPooling(nn.Module):
    """Gated attention pooling di Ilse et al. (2018).

    Per ogni istanza h_k calcola uno score da due rami — tanh(V h) e sigmoid(U h)
    — moltiplicati elemento per elemento (il "gate"): il ramo sigmoid modula
    quanto del ramo tanh far passare, dando un'attenzione piu espressiva del
    semplice tanh. I pesi finali sommano a 1 (softmax sulle istanze).
    """

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
    """Modello MIL completo: pooling + testa condivisa.

    forward processa UN bag alla volta (N chunk di un singolo documento). I bag
    hanno dimensione variabile tra documenti, quindi il loop di training lavora a
    batch_size=1 sul bag (eventuale batching con padding+maschera come ottimizzazione
    futura). Ritorna (logits, attention_weights).
    """

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