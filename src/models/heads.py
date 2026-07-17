from __future__ import annotations

import torch
from torch import Tensor, nn
from src.data.canonical import Document

VALID_TASKS = ("binary", "multiclass", "multilabel")

def out_features_for(task: str, num_classes: int | None) -> int:
    if task == "binary":
        return 1
    if task in ("multiclass", "multilabel"):
        if num_classes is None or num_classes < 2:
            raise ValueError(f"task '{task}' richiede num_classes >= 2, ricevuto {num_classes}.")
        return num_classes
    raise ValueError(f"task deve essere uno di {VALID_TASKS}, ricevuto {task!r}.")

class ClassificationHead(nn.Module):
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
    if task in ("binary", "multilabel"):
        return torch.sigmoid(logits)
    return torch.softmax(logits, dim=-1)

def compute_class_weights(
    train_docs: list[Document],
    task: str,
    num_classes: int | None = None,
) -> Tensor | None:

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