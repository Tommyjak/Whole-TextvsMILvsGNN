from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch

from src.data.canonical import Document
from src.encoders.backbone import FrozenEncoder
from src.encoders.cache import precompute_embeddings, cache_dir_for, list_cached, load_cached

def _load_daic(datasets_root: Path) -> list[Document]:
    from src.data.loaders.daic import load_daic
    return load_daic(datasets_root / "daic-woz")

def _load_imcs21(datasets_root: Path) -> list[Document]:
    from src.data.loaders.imcs21 import load_imcs21
    return load_imcs21(datasets_root / "imcs21")

def _load_mtsamples(datasets_root: Path) -> list[Document]:
    from src.data.loaders.mtsamples import load_mtsamples
    return load_mtsamples(datasets_root / "mtsamples" / "mtsamples.csv")

def _load_ecthr(datasets_root: Path) -> list[Document]:
    from src.data.loaders.ecthr import load_ecthr
    return load_ecthr(datasets_root / "ecthr")

REGISTRY: dict[str, dict] = {
    "daic": {"loader": _load_daic, "encoder": "mental/mental-bert-base-uncased"},
    "imcs21": {"loader": _load_imcs21, "encoder": "bert-base-chinese"},
    "mtsamples": {"loader": _load_mtsamples, "encoder": "emilyalsentzer/Bio_ClinicalBERT"},
    "ecthr": {"loader": _load_ecthr, "encoder": "bert-base-uncased"},
}

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Precompute frozen embeddings into cache.")
    p.add_argument("--dataset", required=True, choices=list(REGISTRY.keys()))
    p.add_argument("--encoder", default=None,
                   help="Nome HF dell'encoder. Se assente, usa il default del dataset.")
    p.add_argument("--granularities", type=int, nargs="+", default=[64, 128, 256],
                   help="Dimensioni di chunk (token) da precalcolare.")
    p.add_argument("--overlap", type=int, default=None,
                   help="Overlap fisso in token. Se assente, usa chunk_size // 8 per ciascuna granularita.")
    p.add_argument("--pooling", default="mean", choices=["mean", "cls"])
    p.add_argument("--max-length", type=int, default=512,
                   help="Max token per chunk in ingresso all'encoder.")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--datasets-root", default="datasets")
    p.add_argument("--cache-root", default="cache")
    p.add_argument("--limit", type=int, default=None,
                   help="Processa solo i primi N documenti (per smoke test).")
    p.add_argument("--overwrite", action="store_true",
                   help="Ricalcola anche gli embedding gia presenti in cache.")
    p.add_argument("--device", default=None, help="cuda / cpu (default: auto).")
    return p.parse_args()

def main() -> None:
    args = parse_args()
    entry = REGISTRY[args.dataset]
    model_name = args.encoder or entry["encoder"]
    datasets_root = Path(args.datasets_root)

    print("=" * 70)
    print(f"Dataset : {args.dataset}")
    print(f"Encoder : {model_name}")
    print(f"Chunks  : {args.granularities}  | pooling={args.pooling} | max_len={args.max_length}")
    print("=" * 70)

    # 1. carica i Document
    documents = entry["loader"](datasets_root)
    if args.limit is not None:
        documents = documents[: args.limit]
        print(f"[limit] ridotto a {len(documents)} documenti per lo smoke test.")

    if not documents:
        print("Nessun documento caricato: controlla i path in datasets/.")
        sys.exit(1)

    # 2. istanzia l'encoder frozen UNA volta (riusato per tutte le granularita)
    print(f"\nCaricamento encoder '{model_name}' (prima volta -> download da HF)...")
    encoder = FrozenEncoder(
        model_name,
        pooling=args.pooling,
        device=args.device,
        max_length=args.max_length,
    )
    print(f"Encoder pronto: hidden_size={encoder.hidden_size}, device={encoder.device}")

    # 3. precompute per ogni granularita
    for chunk_size in args.granularities:
        overlap = args.overlap if args.overlap is not None else chunk_size // 8
        print(f"\n--- granularita {chunk_size} (overlap {overlap}) ---")
        out_dir = precompute_embeddings(
            documents=documents,
            encoder=encoder,
            model_name=model_name,
            dataset=args.dataset,
            cache_root=args.cache_root,
            chunk_size=chunk_size,
            overlap=overlap,
            batch_size=args.batch_size,
            overwrite=args.overwrite,
        )

        # 4. verifica: rileggi un file di cache e mostra le shape (prova end-to-end)
        cached = list_cached(out_dir)
        print(f"[verifica] {len(cached)} file .pt in {out_dir}")
        if cached:
            sample = load_cached(cached[0])
            emb = sample["embeddings"]
            print(f"[verifica] esempio '{sample['doc_id']}': "
                  f"embeddings shape={tuple(emb.shape)}, "
                  f"n_chunks={sample['n_chunks']}, label={sample['label']}")
            # sanity: la seconda dimensione deve essere hidden_size
            assert emb.shape[1] == encoder.hidden_size, "hidden_size non coerente!"

    print("\nFatto. Cache popolata.")

if __name__ == "__main__":
    main()