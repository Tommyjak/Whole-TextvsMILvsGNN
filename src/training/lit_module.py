"""
lit_module.py — Il LightningModule che orchestra il training dei tre modelli.

Struttura a gerarchia:
  BaseLitModule   -> TUTTA la logica condivisa: loss (da heads.py), metriche (da
                     metrics.py), ottimizzatore, il ciclo di ogni step (forward ->
                     loss -> metriche -> log), il lifecycle delle metriche a fine
                     epoca. Non sa come si chiama il forward di un modello: delega.
  MILLitModule    -> sa solo come passare dal batch-cache al forward del MIL.
  GNNLitModule    -> come sopra, ma costruisce gli archi prima del forward.
  WholeTextLit... -> sa solo come passare dal batch-testo al forward del whole-text.

Perche la gerarchia e non un modulo unico con if: la diversita dei tre modelli
(firme di forward diverse, cache vs testo, per-documento vs per-batch) e confinata
in un metodo minuscolo per sottoclasse (_forward_batch). Tutto il resto — che deve
essere IDENTICO per l'equita del confronto — vive una volta sola nella base.
"""

from __future__ import annotations

import lightning.pytorch as pl
import torch
from torch import Tensor

from src.models.gnn import build_edges
from src.models.heads import build_loss, to_target
from src.training.metrics import build_metrics, prepare_for_metrics


# ============================== base condivisa ==============================

class BaseLitModule(pl.LightningModule):
    """Logica di training condivisa da tutti e tre i modelli.

    Le sottoclassi implementano SOLO `_forward_batch(batch) -> logits (B, C)`.
    Tutto il resto — loss, metriche, ottimizzatore, logging — e definito qui e
    identico per MIL, GNN e whole-text.
    """

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

        # loss condivisa (da heads.py), scelta in base al task
        self.loss_fn = build_loss(task, pos_weight=pos_weight, class_weight=class_weight)

        # tre collezioni di metriche separate: registrandole come attributi (sono
        # nn.Module), Lightning le sposta sul device giusto in automatico.
        self.train_metrics = build_metrics(task, num_classes, prefix="train/")
        self.val_metrics = build_metrics(task, num_classes, prefix="val/")
        self.test_metrics = build_metrics(task, num_classes, prefix="test/")

        # salva gli iperparametri scalari nel checkpoint (ignora il modello, che
        # non e serializzabile come hparam)
        self.save_hyperparameters(ignore=["model", "pos_weight", "class_weight"])

    # --- da implementare nelle sottoclassi ---
    def _forward_batch(self, batch: dict) -> Tensor:
        """Dal batch ai logit (B, out_features). Specifico per tipo di modello."""
        raise NotImplementedError

    # --- ciclo condiviso, uguale per train/val/test ---
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

    # --- lifecycle delle metriche: compute + log + reset a fine epoca ---
    def on_train_epoch_end(self):
        self.log_dict(self.train_metrics.compute()); self.train_metrics.reset()

    def on_validation_epoch_end(self):
        self.log_dict(self.val_metrics.compute(), prog_bar=True); self.val_metrics.reset()

    def on_test_epoch_end(self):
        self.log_dict(self.test_metrics.compute()); self.test_metrics.reset()

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)


# ============================== sottoclassi ==============================

class MILLitModule(BaseLitModule):
    """MIL: batch dalla cache. I bag hanno N variabile -> un documento alla volta,
    poi si concatenano i logit in (B, C)."""

    def _forward_batch(self, batch: dict) -> Tensor:
        logits = []
        for emb in batch["embeddings"]:
            emb = emb.to(self.device)
            lg, _ = self.model(emb)          # (1, C); ignoriamo i pesi in training
            logits.append(lg)
        return torch.cat(logits, dim=0)      # (B, C)


class GNNLitModule(BaseLitModule):
    """GNN: come MIL, ma costruisce gli archi prima del forward. edge_mode e knn_k
    sono iperparametri dell'esperimento (l'ablation sulla struttura del grafo)."""

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
    """Whole-text: batch di stringhe. Il modello tokenizza e processa l'intero
    batch in un forward (l'encoder e fine-tunato)."""

    def _forward_batch(self, batch: dict) -> Tensor:
        return self.model(batch["texts"])   # (B, C); il modello gestisce il device