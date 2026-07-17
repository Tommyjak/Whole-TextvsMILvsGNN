from __future__ import annotations

import lightning.pytorch as pl
import torch
from torch import Tensor

from src.models.gnn import build_edges
from src.models.heads import build_loss, to_target
from src.training.metrics import build_metrics, prepare_for_metrics

class BaseLitModule(pl.LightningModule):
    def __init__(
        self,
        model: torch.nn.Module,
        task: str,
        num_classes: int | None = None,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        pos_weight: Tensor | None = None,
        class_weight: Tensor | None = None,
    ) -> None:
        super().__init__()
        self.model = model
        self.task = task
        self.num_classes = num_classes
        self.lr = lr
        self.weight_decay = weight_decay

        # loss condivisa, scelta in base al task
        self.loss_fn = build_loss(task, pos_weight=pos_weight, class_weight=class_weight)

        # tre collezioni di metriche separate
        self.train_metrics = build_metrics(task, num_classes, prefix="train/")
        self.val_metrics = build_metrics(task, num_classes, prefix="val/")
        self.test_metrics = build_metrics(task, num_classes, prefix="test/")

        # salva gli iperparametri scalari nel checkpoint (ignora il modello, che non è serializzabile)
        self.save_hyperparameters(ignore=["model", "pos_weight", "class_weight"])

    # da implementare nelle sottoclassi
    def _forward_batch(self, batch: dict) -> Tensor:
        raise NotImplementedError

    # ciclo condiviso, uguale per train/val/test 
    def _step(self, batch: dict, stage: str, metrics) -> Tensor:
        logits = self._forward_batch(batch)                       # (B, C)
        target = to_target(batch["labels"], self.task, self.num_classes, device=self.device)
        loss = self.loss_fn(logits, target)

        # aggiorna le metriche con logit (confidenza continua) + target
        preds, tgt = prepare_for_metrics(logits, target, self.task)
        metrics.update(preds, tgt)

        self.log(f"{stage}/loss", loss, prog_bar=(stage != "train"),
                 on_step=(stage == "train"), on_epoch=True, batch_size=len(batch["labels"]))
        return loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, "train", self.train_metrics)

    def validation_step(self, batch, batch_idx):
        self._step(batch, "val", self.val_metrics)

    def test_step(self, batch, batch_idx):
        self._step(batch, "test", self.test_metrics)

    # lifecycle delle metriche: compute + log + reset a fine epoca
    def on_train_epoch_end(self):
        self.log_dict(self.train_metrics.compute()); self.train_metrics.reset()

    def on_validation_epoch_end(self):
        self.log_dict(self.val_metrics.compute(), prog_bar=True); self.val_metrics.reset()

    def on_test_epoch_end(self):
        self.log_dict(self.test_metrics.compute()); self.test_metrics.reset()

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

class MILLitModule(BaseLitModule):
    def _forward_batch(self, batch: dict) -> Tensor:
        logits = []
        for emb in batch["embeddings"]:
            emb = emb.to(self.device)
            lg, _ = self.model(emb)          # (1, C); ignoriamo i pesi in training
            logits.append(lg)
        return torch.cat(logits, dim=0)      # (B, C)


class GNNLitModule(BaseLitModule):
    def __init__(self, *args, edge_mode: str = "sequential", knn_k: int = 5, **kwargs):
        super().__init__(*args, **kwargs)
        self.edge_mode = edge_mode
        self.knn_k = knn_k
        self.save_hyperparameters("edge_mode", "knn_k")

    def _forward_batch(self, batch: dict) -> Tensor:
        logits = []
        for emb in batch["embeddings"]:
            emb = emb.to(self.device)
            edge_index = build_edges(emb.shape[0], emb, mode=self.edge_mode,
                                     knn_k=self.knn_k).to(self.device)
            lg, _ = self.model(emb, edge_index)   # (1, C)
            logits.append(lg)
        return torch.cat(logits, dim=0)           # (B, C)


class WholeTextLitModule(BaseLitModule):
    def _forward_batch(self, batch: dict) -> Tensor:
        return self.model(batch["texts"])   # (B, C); il modello gestisce il device