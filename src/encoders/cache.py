"""
cache.py — Precompute e persistenza degli embedding frozen.

È il ponte tra Document (testo) e modelli (vettori). Per ogni documento:
  1. spezza il testo in chunk  (chunk_text, con il tokenizer DELL'encoder)
  2. encoda i chunk            (FrozenEncoder.encode)
  3. salva la matrice (N_chunk, hidden) su disco come .pt

Perché la cache è il cuore dell'equità sperimentale:
  MIL e GNN NON ricalcolano gli embedding: leggono lo STESSO file .pt. È
  materialmente impossibile che vedano chunk o vettori diversi. Il precompute
  gira UNA volta per (dataset, encoder, chunk_size, overlap); poi ogni training
  parte in secondi caricando i tensori invece di ri-encodare.

Schema delle chiavi di cache (deciso nel design):
  cache/{dataset}/{encoder_slug}/g{chunk_size}_o{overlap}/{doc_id}.pt
  - dataset      -> daic, imcs21, mtsamples, ecthr
  - encoder_slug -> nome HF reso sicuro per il filesystem (/ -> __)
  - g{cs}_o{ov}  -> granularità+overlap: l'ablation 64/128/256 vive in cartelle
                    separate, quindi cambiare granularità non sovrascrive nulla.
Ogni .pt contiene un dict: {"doc_id", "embeddings" (N,H), "label", "n_chunks"}.
La label viaggia con l'embedding così il training non deve ri-consultare il loader.
"""

from __future__ import annotations

from pathlib import Path

import torch
from tqdm import tqdm

from src.data.canonical import Document
from src.data.chunking import chunk_text
from src.encoders.backbone import FrozenEncoder


def encoder_slug(model_name: str) -> str:
    """Rende un nome-modello HuggingFace sicuro come nome di cartella.
    'hfl/chinese-roberta-wwm-ext' -> 'hfl__chinese-roberta-wwm-ext'."""
    return model_name.replace("/", "__")


def cache_dir_for(
    cache_root: Path,
    dataset: str,
    model_name: str,
    chunk_size: int,
    overlap: int,
) -> Path:
    """Costruisce il path della cartella di cache per una configurazione."""
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
    """Calcola e salva su disco gli embedding dei chunk per tutti i documenti.

    Idempotente: se il .pt di un documento esiste già e overwrite=False, lo
    salta. Così un precompute interrotto (ECtHR è grande) riprende senza
    rifare il lavoro. Ritorna la cartella di cache popolata.
    """
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
    """Carica un singolo .pt di cache. `path` è il file del documento."""
    return torch.load(path, map_location="cpu")


def list_cached(cache_dir: str | Path) -> list[Path]:
    """Elenca i .pt in una cartella di cache, in ordine deterministico.
    È così che il Dataset di training scopre quali documenti sono disponibili."""
    return sorted(Path(cache_dir).glob("*.pt"))