from __future__ import annotations

from pathlib import Path

import torch
from torch import Tensor
from torch.utils.data import Dataset

from src.data.canonical import Document
from src.encoders.cache import list_cached, load_cached

class CachedBagDataset(Dataset):
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
    return {
        "embeddings": [b["embeddings"] for b in batch],  # lista di (N_i, H)
        "labels": [b["label"] for b in batch],           # lista di label
        "doc_ids": [b["doc_id"] for b in batch],
    }

class RawTextDataset(Dataset):
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
    return {
        "texts": [b["text"] for b in batch],     # lista di stringhe
        "labels": [b["label"] for b in batch],
        "doc_ids": [b["doc_id"] for b in batch],
    }