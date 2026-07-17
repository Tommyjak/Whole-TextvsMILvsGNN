from __future__ import annotations

from pathlib import Path

import torch
from tqdm import tqdm

from src.data.canonical import Document
from src.data.chunking import chunk_text
from src.encoders.backbone import FrozenEncoder


def encoder_slug(model_name: str) -> str:
    return model_name.replace("/", "__")


def cache_dir_for(
    cache_root: Path,
    dataset: str,
    model_name: str,
    chunk_size: int,
    overlap: int,
) -> Path:
    # Costruisce il path della cartella di cache per una configurazione.
    return (
        Path(cache_root)
        / dataset
        / encoder_slug(model_name)
        / f"g{chunk_size}_o{overlap}"
    )


def precompute_embeddings(
    documents: list[Document],
    encoder: FrozenEncoder,
    model_name: str,
    dataset: str,
    cache_root: str | Path = "cache",
    chunk_size: int = 128,
    overlap: int = 16,
    batch_size: int = 32,
    overwrite: bool = False,
) -> Path:
    
    out_dir = cache_dir_for(cache_root, dataset, model_name, chunk_size, overlap)
    out_dir.mkdir(parents=True, exist_ok=True)

    skipped = 0
    for doc in tqdm(documents, desc=f"embedding {dataset} g{chunk_size}"):
        out_path = out_dir / f"{doc.doc_id}.pt"

        if out_path.exists() and not overwrite:
            skipped += 1
            continue

        chunks = chunk_text(
            doc.text, encoder.tokenizer, chunk_size=chunk_size, overlap=overlap
        )
        embeddings = encoder.encode(chunks, batch_size=batch_size)  # (N_chunk, H)

        torch.save(
            {
                "doc_id": doc.doc_id,
                "embeddings": embeddings,     # (N_chunk, hidden_size), float32, CPU
                "label": doc.label,           # int o list[int]
                "n_chunks": embeddings.shape[0],
            },
            out_path,
        )

    if skipped:
        print(f"[cache] {dataset} g{chunk_size}: {skipped} documenti già presenti, saltati.")
    return out_dir


def load_cached(path: str | Path) -> dict:
    return torch.load(path, map_location="cpu")


def list_cached(cache_dir: str | Path) -> list[Path]:
    return sorted(Path(cache_dir).glob("*.pt"))