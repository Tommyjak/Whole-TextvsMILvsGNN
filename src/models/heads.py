"""
heads.py — Testa di classificazione e loss CONDIVISE dai tre modelli.

Questo file e il perno dell'equita sperimentale: MIL, GNN e whole-text producono
ciascuno un vettore-documento con la propria aggregazione, ma da li in poi usano
la STESSA testa e la STESSA loss definite qui. Cosi l'unica cosa che distingue i
tre modelli e come si arriva al vettore-documento, non come lo si classifica.

Tre tipi di task, un'unica astrazione:
  - "binary"      (DAIC)                 -> 1 logit,  BCEWithLogitsLoss
  - "multiclass"  (IMCS 10, MTSamples 9) -> C logit,  CrossEntropyLoss
  - "multilabel"  (ECtHR)                -> C logit,  BCEWithLogitsLoss (per-classe)

Il sigmoid/softmax NON e nel forward: si lavora sui LOGIT (piu stabile
numericamente), e la loss giusta li interpreta. logits_to_probs() fa la
conversione quando servono le probabilita (metriche, inferenza).
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
from src.data.canonical import Document

VALID_TASKS = ("binary", "multiclass", "multilabel")


def out_features_for(task: str, num_classes: int | None) -> int:
    """Quanti logit produce la testa per ciascun tipo di task."""
    if task == "binary":
        return 1
    if task in ("multiclass", "multilabel"):
        if num_classes is None or num_classes < 2:
            raise ValueError(f"task '{task}' richiede num_classes >= 2, ricevuto {num_classes}.")
        return num_classes
    raise ValueError(f"task deve essere uno di {VALID_TASKS}, ricevuto {task!r}.")


class ClassificationHead(nn.Module):
    """MLP condiviso: (B, in_dim) -> (B, out_features).

    Struttura identica per tutti e tre i modelli: Linear -> ReLU -> Dropout ->
    Linear. Deliberatamente semplice: il lavoro rappresentativo lo fa l'encoder
    frozen a monte; questa testa deve solo mappare il vettore-documento sui logit.
    """

    def __init__(
        self,
        in_dim: int,
        task: str,
        num_classes: int | None = None,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.task = task
        self.num_classes = num_classes
        self.out_features = out_features_for(task, num_classes)
        self.net = nn.Sequential(
            nn.Linear(in_dim, in_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(in_dim // 2, self.out_features),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


def build_loss(
    task: str,
    pos_weight: Tensor | None = None,
    class_weight: Tensor | None = None,
) -> nn.Module:
    """Restituisce la loss giusta per il task.

    - binary/multilabel -> BCEWithLogitsLoss; pos_weight bilancia le classi
      (scalare per il binario, shape (C,) per il multilabel).
    - multiclass -> CrossEntropyLoss; class_weight ha shape (C,).
    """
    if task in ("binary", "multilabel"):
        return nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    if task == "multiclass":
        return nn.CrossEntropyLoss(weight=class_weight)
    raise ValueError(f"task deve essere uno di {VALID_TASKS}, ricevuto {task!r}.")


def to_target(
    labels: list,
    task: str,
    num_classes: int | None = None,
    device: torch.device | str | None = None,
) -> Tensor:
    """Converte le label dei Document nel formato-target che la loss si aspetta.

    - binary     : list[int 0/1]        -> (B, 1) float
    - multiclass : list[int]            -> (B,)   long  (indice di classe)
    - multilabel : list[list[int]]      -> (B, C) float multi-hot
    """
    if task == "binary":
        return torch.tensor(labels, dtype=torch.float32, device=device).unsqueeze(1)
    if task == "multiclass":
        return torch.tensor(labels, dtype=torch.long, device=device)
    if task == "multilabel":
        if num_classes is None:
            raise ValueError("multilabel richiede num_classes per il multi-hot.")
        target = torch.zeros(len(labels), num_classes, dtype=torch.float32, device=device)
        for i, active in enumerate(labels):
            if active:  # lista non vuota di indici attivi
                target[i, active] = 1.0
        return target
    raise ValueError(f"task deve essere uno di {VALID_TASKS}, ricevuto {task!r}.")


def logits_to_probs(logits: Tensor, task: str) -> Tensor:
    """Converte i logit in probabilita, secondo il task (per metriche/inferenza)."""
    if task in ("binary", "multilabel"):
        return torch.sigmoid(logits)
    return torch.softmax(logits, dim=-1)

def compute_class_weights(
    train_docs: list[Document],
    task: str,
    num_classes: int | None = None,
) -> Tensor | None:
    """Calcola i pesi di classe dalla distribuzione delle SOLE label di training.

    Mai da dev/test: userebbe informazione fuori dal training set (leakage).

    Ritorna il tensore adatto alla loss del task:
      - binary     -> pos_weight scalare = n_neg / n_pos  (per BCEWithLogitsLoss)
      - multiclass -> class_weight (C,)  = n_tot / (C * n_c)  (per CrossEntropyLoss)
      - multilabel -> pos_weight (C,)    = n_neg_c / n_pos_c per ciascuna classe
    Ritorna None se il calcolo non e applicabile (es. una classe assente).
    """
    labels = [d.label for d in train_docs]

    if task == "binary":
        n_pos = sum(1 for y in labels if y == 1)
        n_neg = sum(1 for y in labels if y == 0)
        if n_pos == 0 or n_neg == 0:
            return None
        return torch.tensor([n_neg / n_pos], dtype=torch.float32)

    if task == "multiclass":
        if num_classes is None:
            raise ValueError("multiclass richiede num_classes per i pesi.")
        counts = torch.zeros(num_classes, dtype=torch.float32)
        for y in labels:
            counts[y] += 1
        if (counts == 0).any():
            # una classe assente dal training: pesi non definiti, meglio None
            return None
        n_tot = counts.sum()
        # peso inversamente proporzionale alla frequenza, normalizzato
        return n_tot / (num_classes * counts)

    if task == "multilabel":
        if num_classes is None:
            raise ValueError("multilabel richiede num_classes per i pesi.")
        pos = torch.zeros(num_classes, dtype=torch.float32)
        for active in labels:            # active = list[int] di classi attive
            for c in active:
                pos[c] += 1
        n_tot = len(labels)
        neg = n_tot - pos
        pos = pos.clamp(min=1.0)         # evita divisione per zero
        return neg / pos

    raise ValueError(f"task sconosciuto: {task!r}")