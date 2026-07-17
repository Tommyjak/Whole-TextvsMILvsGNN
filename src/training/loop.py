from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
import time

import lightning.pytorch as pl
import torch
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, Callback
from torch.utils.data import DataLoader

from src.data.canonical import Document
from src.training.splits import resolve_splits, filter_by_split
from src.training.datasets import (
    CachedBagDataset, RawTextDataset, collate_bags, collate_texts,
)
from src.models.mil import MILModel
from src.models.gnn import GNNModel
from src.models.whole_text import WholeTextModel
from src.models.heads import logits_to_probs
from src.training.lit_module import MILLitModule, GNNLitModule, WholeTextLitModule
from src.models.heads import compute_class_weights

@dataclass
class ExperimentConfig:
    # esperimento
    dataset: str
    model: str                      
    task: str
    num_classes: int | None = None

    # dati
    datasets_root: str = "datasets"
    cache_root: str = "cache"
    encoder: str = "bert-base-uncased"
    chunk_size: int = 128
    overlap: int = 16
    cv: bool = False
    n_folds: int = 1

    # training
    seed: int = 42
    max_epochs: int = 50
    patience: int = 5
    batch_size: int = 8
    lr: float = 1e-3
    balance: bool = True
    weight_decay: float = 1e-4
    max_length: int = 512

    # GNN-specifici
    edge_mode: str = "sequential"
    knn_k: int = 5

    # output
    results_root: str = "results"
    monitor: str = "val/f1"
    monitor_mode: str = "max"

    def __post_init__(self):
        # Se l'utente non ha sovrascitto manualmente il monitor, lo correggiamo per multiclass/multilabel
        if self.monitor == "val/f1" and self.task in ("multiclass", "multilabel"):
            self.monitor = "val/f1_macro"
            
    def run_name(self) -> str:
        base = f"{self.dataset}_{self.model}_g{self.chunk_size}_seed{self.seed}"
        if self.model == "gnn":
            base += f"_{self.edge_mode}"
        if self.cv:
            base += "_cv"          # distingue i run CV da quelli con split ufficiale
        return base

def _load_documents(cfg: ExperimentConfig) -> list[Document]:
    root = Path(cfg.datasets_root)
    if cfg.dataset == "daic":
        from src.data.loaders.daic import load_daic
        return load_daic(root / "daic-woz")
    if cfg.dataset == "imcs21":
        from src.data.loaders.imcs21 import load_imcs21
        return load_imcs21(root / "imcs21")
    if cfg.dataset == "mtsamples":
        from src.data.loaders.mtsamples import load_mtsamples
        return load_mtsamples(root / "mtsamples" / "mtsamples.csv")
    if cfg.dataset == "ecthr":
        from src.data.loaders.ecthr import load_ecthr
        return load_ecthr(root / "ecthr")
    raise ValueError(f"dataset sconosciuto: {cfg.dataset}")

def _cache_dir(cfg: ExperimentConfig) -> Path:
    from src.encoders.cache import cache_dir_for
    return cache_dir_for(cfg.cache_root, cfg.dataset, cfg.encoder, cfg.chunk_size, cfg.overlap)


def _build_loaders_and_module(cfg: ExperimentConfig, documents, split_map):
    # pesi di classe dai SOLI documenti di training (se balance attivo)
    pos_weight, class_weight = None, None
    if cfg.balance:
        train_docs = filter_by_split(documents, split_map, "train")
        w = compute_class_weights(train_docs, cfg.task, cfg.num_classes)
        if cfg.task in ("binary", "multilabel"):
            pos_weight = w
        else:  # multiclass
            class_weight = w
        if w is not None:
            print(f"[loop] bilanciamento attivo ({cfg.task}): "
                  f"{'pos_weight' if pos_weight is not None else 'class_weight'} calcolato.")
            
    if cfg.model in ("mil", "gnn"):
        # percorso cache
        cache_dir = _cache_dir(cfg)
        make = lambda split: CachedBagDataset(cache_dir, split_map, split)
        loaders = {
            s: DataLoader(make(s), batch_size=cfg.batch_size,
                          shuffle=(s == "train"), collate_fn=collate_bags)
            for s in ("train", "dev", "test")
        }
        if cfg.model == "mil":
            model = MILModel(in_dim=768, task=cfg.task, num_classes=cfg.num_classes)
            lit = MILLitModule(model, task=cfg.task, num_classes=cfg.num_classes,
                               lr=cfg.lr, weight_decay=cfg.weight_decay,
                               pos_weight=pos_weight, class_weight=class_weight)
        else:
            model = GNNModel(in_dim=768, task=cfg.task, num_classes=cfg.num_classes)
            lit = GNNLitModule(model, task=cfg.task, num_classes=cfg.num_classes,
                               lr=cfg.lr, weight_decay=cfg.weight_decay,
                               pos_weight=pos_weight, class_weight=class_weight,
                               edge_mode=cfg.edge_mode, knn_k=cfg.knn_k)

    elif cfg.model == "whole_text":
        # percorso testo
        make = lambda split: RawTextDataset(documents, split_map, split)
        loaders = {
            s: DataLoader(make(s), batch_size=cfg.batch_size,
                          shuffle=(s == "train"), collate_fn=collate_texts)
            for s in ("train", "dev", "test")
        }
        model = WholeTextModel(cfg.encoder, task=cfg.task, num_classes=cfg.num_classes,
                               max_length=cfg.max_length)
        lit = WholeTextLitModule(model, task=cfg.task, num_classes=cfg.num_classes,
                                 lr=cfg.lr, weight_decay=cfg.weight_decay,
                                 pos_weight=pos_weight, class_weight=class_weight)
    else:
        raise ValueError(f"modello sconosciuto: {cfg.model}")

    return loaders["train"], loaders["dev"], loaders["test"], lit

class PredictionCollector(Callback):
    def __init__(self, task: str):
        self.task = task
        self.records: list[dict] = []

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        # rieseguo il forward per catturare i logit (il test_step non li ritorna)
        with torch.no_grad():
            logits = pl_module._forward_batch(batch)          # (B, C)
            probs = logits_to_probs(logits, self.task)        # (B, C)
        for i, doc_id in enumerate(batch["doc_ids"]):
            self.records.append({
                "doc_id": doc_id,
                "label": batch["labels"][i],
                "probs": [round(float(x), 5) for x in probs[i].tolist()]
                          if probs[i].dim() > 0 else [round(float(probs[i]), 5)],
            })

def train_one(cfg: ExperimentConfig) -> dict:
    pl.seed_everything(cfg.seed, workers=True)   # riproducibilita completa

    documents = _load_documents(cfg)
    if cfg.cv:
        from src.training.splits import resolve_splits_montecarlo
        split_map = resolve_splits_montecarlo(documents, fold_seed=cfg.seed)
    else:
        split_map = resolve_splits(documents, seed=cfg.seed)

    train_loader, val_loader, test_loader, lit = _build_loaders_and_module(
        cfg, documents, split_map
    )

    run_dir = Path(cfg.results_root) / cfg.run_name()
    run_dir.mkdir(parents=True, exist_ok=True)

    # early stopping sulla metrica scelta + salvataggio del best checkpoint
    ckpt = ModelCheckpoint(dirpath=run_dir, filename="best",
                           monitor=cfg.monitor, mode=cfg.monitor_mode, save_top_k=1)
    early = EarlyStopping(monitor=cfg.monitor, mode=cfg.monitor_mode, patience=cfg.patience)

    pred_collector = PredictionCollector(cfg.task)

    trainer = pl.Trainer(
        max_epochs=cfg.max_epochs,
        accelerator="auto",
        callbacks=[ckpt, early, pred_collector],
        default_root_dir=str(run_dir),
        enable_progress_bar=True,
        log_every_n_steps=10,
    )

    # tempo di training
    t_fit_start = time.perf_counter()
    trainer.fit(lit, train_loader, val_loader)
    fit_seconds = time.perf_counter() - t_fit_start

    # tempo di inferenza (test sul best checkpoint)
    t_test_start = time.perf_counter()
    test_results = trainer.test(lit, test_loader, ckpt_path="best", verbose=False)
    test_seconds = time.perf_counter() - t_test_start
    metrics = test_results[0] if test_results else {}

    # numero di documenti di test (per la latenza per-documento)
    n_test = len(pred_collector.records)

    result = {
        "run_name": cfg.run_name(),
        "config": asdict(cfg),
        "test_metrics": metrics,
        "best_checkpoint": str(ckpt.best_model_path),
        "best_score": float(ckpt.best_model_score) if ckpt.best_model_score is not None else None,
        # nuovi campi
        "timing": {
            "fit_seconds": round(fit_seconds, 2),
            "test_seconds": round(test_seconds, 2),
            "n_test_docs": n_test,
            "inference_ms_per_doc": round(1000 * test_seconds / n_test, 3) if n_test else None,
            "trainable_params": sum(p.numel() for p in lit.parameters() if p.requires_grad),
        },
        "predictions": pred_collector.records,
    }

    import json
    (run_dir / "result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"[loop] {cfg.run_name()} -> test {metrics} | fit {fit_seconds:.1f}s")
    return result