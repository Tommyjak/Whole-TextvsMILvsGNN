"""
datasets.py — I due percorsi di dati verso i modelli.

Due Dataset separati, stessa interfaccia esterna, ognuno con un solo compito:

  CachedBagDataset  -> per MIL e GNN. Legge la cache degli embedding (.pt).
                       Restituisce (embeddings (N,H), label, doc_id).
  RawTextDataset    -> per whole-text. Legge il testo grezzo dei Document.
                       Restituisce (text, label, doc_id).

Entrambi ricevono lo split_map (da splits.py) e tengono SOLO i documenti dello
split richiesto. La dualita cache-vs-testo vive qui e solo qui: a valle (metriche,
LightningModule, loop) nessuno sa piu quale dei due sta usando.

collate a batch_size variabile:
  - i bag/grafi hanno N_chunk diverso per documento -> NON si impilano in un
    tensore rettangolare. Il collate della cache restituisce una LISTA di matrici
    (il modello le processa una alla volta, com'e nel forward di MIL/GNN).
  - il testo si raggruppa come lista di stringhe: il tokenizer del whole-text
    fara il padding a batch dentro il suo forward.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import Tensor
from torch.utils.data import Dataset

from src.data.canonical import Document
from src.encoders.cache import list_cached, load_cached


# ============================ percorso cache (MIL/GNN) ============================

class CachedBagDataset(Dataset):
    """Dataset che legge gli embedding cachati per un dato split.

    cache_dir: cartella di una specifica (dataset, encoder, granularita), es.
        cache/daic/bert-base-uncased/g64_o8
    split_map: {doc_id: 'train'/'dev'/'test'} da resolve_splits.
    split: quale partizione tenere.

    Identico per MIL e GNN: la costruzione del grafo (archi) NON avviene qui, e a
    valle nel percorso GNN, perche dipende da 'mode' (iperparametro), non dai dati.
    """

    def __init__(self, cache_dir: str | Path, split_map: dict[str, str], split: str):
        self.cache_dir = Path(cache_dir)
        self.split = split

        all_files = list_cached(self.cache_dir)
        if not all_files:
            raise FileNotFoundError(
                f"Nessun .pt in {self.cache_dir}. Hai lanciato il precompute per "
                f"questa granularita?"
            )

        # tieni solo i file il cui doc_id appartiene allo split richiesto.
        # il doc_id e il nome del file senza estensione (cosi l'ha salvato cache.py).
        self.files = [f for f in all_files if split_map.get(f.stem) == split]
        if not self.files:
            raise ValueError(
                f"Nessun documento nello split '{split}' per {self.cache_dir}. "
                f"(file in cache: {len(all_files)}, ma nessuno matcha lo split.)"
            )

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> dict:
        data = load_cached(self.files[idx])
        return {
            "embeddings": data["embeddings"],   # (N_chunk, H) float32
            "label": data["label"],             # int o list[int]
            "doc_id": data["doc_id"],
        }


def collate_bags(batch: list[dict]) -> dict:
    """Collate per CachedBagDataset: NON impila gli embedding (N variabile), li
    tiene come lista. Il modello MIL/GNN processa un bag alla volta."""
    return {
        "embeddings": [b["embeddings"] for b in batch],  # lista di (N_i, H)
        "labels": [b["label"] for b in batch],           # lista di label
        "doc_ids": [b["doc_id"] for b in batch],
    }


# ============================ percorso testo (whole-text) ============================

class RawTextDataset(Dataset):
    """Dataset che restituisce il testo grezzo dei Document per un dato split.

    documents: la lista di Document dal loader.
    split_map: {doc_id: split}.  split: quale partizione tenere.
    La tokenizzazione NON avviene qui: la fa il forward del whole-text (l'encoder
    e fine-tunato, quindi il gradiente deve passare per la tokenizzazione)."""

    def __init__(self, documents: list[Document], split_map: dict[str, str], split: str):
        self.split = split
        self.docs = [d for d in documents if split_map.get(d.doc_id) == split]
        if not self.docs:
            raise ValueError(f"Nessun documento nello split '{split}'.")

    def __len__(self) -> int:
        return len(self.docs)

    def __getitem__(self, idx: int) -> dict:
        doc = self.docs[idx]
        return {"text": doc.text, "label": doc.label, "doc_id": doc.doc_id}


def collate_texts(batch: list[dict]) -> dict:
    """Collate per RawTextDataset: raggruppa i testi come lista di stringhe.
    Il tokenizer del whole-text fara il padding a batch nel forward."""
    return {
        "texts": [b["text"] for b in batch],     # lista di stringhe
        "labels": [b["label"] for b in batch],
        "doc_ids": [b["doc_id"] for b in batch],
    }