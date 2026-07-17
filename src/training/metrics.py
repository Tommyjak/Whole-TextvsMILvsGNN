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