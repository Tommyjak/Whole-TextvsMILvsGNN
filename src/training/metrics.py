"""
metrics.py — Metriche di valutazione CONDIVISE dai tre modelli.

Riceve logit + label e calcola le metriche, IDENTICHE per MIL, GNN e whole-text.
Come heads.py, e un perno dell'equita: se i tre modelli fossero valutati con
metriche diverse, il confronto sarebbe falsato a monte.

Un'unica astrazione per i tre task:
  - "binary"     (DAIC)                 -> soglia 0.5 su sigmoid(logit)
  - "multiclass" (IMCS 10, MTSamples 9) -> argmax su softmax(logit)
  - "multilabel" (ECtHR)                -> soglia 0.5 per-classe su sigmoid(logit)

Metriche primarie (robuste allo sbilanciamento, come da design):
  F1 (macro), AUROC, AUPRC (average precision), MCC, balanced accuracy,
  piu accuracy come riferimento. torchmetrics gestisce internamente la logica
  per task via il parametro `task`.

Uso tipico (dentro il LightningModule):
    metrics = build_metrics(task="binary", num_classes=None)   # una volta
    metrics.update(logits, targets)                             # a ogni batch
    results = metrics.compute()                                 # a fine epoca
    metrics.reset()                                             # prima della prossima
"""

from __future__ import annotations

import torch
from torch import Tensor
from torchmetrics import MetricCollection
from torchmetrics.classification import (
    Accuracy,
    AUROC,
    AveragePrecision,
    F1Score,
    MatthewsCorrCoef,
)

# balanced accuracy = recall medio per classe -> Recall con average="macro"
from torchmetrics.classification import Recall


def build_metrics(
    task: str,
    num_classes: int | None = None,
    prefix: str = "",
) -> MetricCollection:
    """Costruisce la collezione di metriche adatta al task.

    prefix: prepende una stringa ai nomi (es. 'val/' o 'test/') per separare gli
    split nei log. torchmetrics accumula lo stato tra update e lo riduce in compute.
    """
    if task == "binary":
        tm_task = "binary"
        kwargs = {"task": tm_task}
        collection = {
            "acc": Accuracy(**kwargs),
            "f1": F1Score(**kwargs),                      # F1 sulla classe positiva
            "auroc": AUROC(**kwargs),
            "auprc": AveragePrecision(**kwargs),
            "mcc": MatthewsCorrCoef(**kwargs),
            "bal_acc": Recall(**kwargs),                  # su binario = recall pos.
        }

    elif task == "multiclass":
        if num_classes is None or num_classes < 2:
            raise ValueError(f"multiclass richiede num_classes >= 2, ricevuto {num_classes}.")
        kwargs = {"task": "multiclass", "num_classes": num_classes}
        collection = {
            "acc": Accuracy(**kwargs),                            # micro accuracy
            "f1_macro": F1Score(average="macro", **kwargs),       # tutte le classi pesano uguale
            "auroc": AUROC(average="macro", **kwargs),
            "auprc": AveragePrecision(average="macro", **kwargs),
            "mcc": MatthewsCorrCoef(**kwargs),
            "bal_acc": Recall(average="macro", **kwargs),         # recall medio per classe
        }

    elif task == "multilabel":
        if num_classes is None or num_classes < 2:
            raise ValueError(f"multilabel richiede num_classes >= 2, ricevuto {num_classes}.")
        kwargs = {"task": "multilabel", "num_labels": num_classes}
        collection = {
            "f1_macro": F1Score(average="macro", **kwargs),
            "f1_micro": F1Score(average="micro", **kwargs),
            "auroc": AUROC(average="macro", **kwargs),
            "auprc": AveragePrecision(average="macro", **kwargs),
        }

    else:
        raise ValueError(f"task sconosciuto: {task!r}")

    return MetricCollection(collection, prefix=prefix)


def prepare_for_metrics(
    logits: Tensor,
    targets: Tensor,
    task: str,
) -> tuple[Tensor, Tensor]:
    """Adatta logit e target ai formati che torchmetrics si aspetta.

    torchmetrics vuole:
      - binary     : preds = prob (o logit) shape (B,);  target = int (B,)
      - multiclass : preds = logit (B, C);               target = int (B,)
      - multilabel : preds = prob (o logit) (B, C);      target = int (B, C)
    Le metriche threshold-free (AUROC/AUPRC) usano la confidenza continua, quindi
    passiamo i LOGIT/probabilita, non le predizioni gia soglia.
    """
    if task == "binary":
        # logits (B,1) -> (B,); target (B,1) float -> (B,) int
        preds = logits.squeeze(-1)
        target = targets.squeeze(-1).long()
        return preds, target

    if task == "multiclass":
        # logits (B, C) restano; target gia (B,) long
        return logits, targets.long()

    if task == "multilabel":
        # logits (B, C); target (B, C) float -> int
        return logits, targets.long()

    raise ValueError(f"task sconosciuto: {task!r}")