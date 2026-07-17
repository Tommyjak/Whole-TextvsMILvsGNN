from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.data.canonical import Document

NUM_CLASSES = 10  # articoli CEDU in ECtHR Task A (LexGLUE)

# i tre file parquet e il loro split ufficiale
_SPLIT_FILES = {
    "train": "train-00000-of-00001.parquet",
    "dev": "validation-00000-of-00001.parquet",
    "test": "test-00000-of-00001.parquet",
}

def _build_text(paragraphs, para_markers: bool = False) -> str:
    parts = [str(p).strip() for p in paragraphs if str(p).strip()]
    return " ".join(parts).strip()

def load_ecthr(
    root: str | Path,
    splits: tuple[str, ...] = ("train", "dev", "test"),
) -> list[Document]:
    
    root = Path(root)
    documents: list[Document] = []
    skipped_empty = 0

    for split_name in splits:
        path = root / _SPLIT_FILES[split_name]
        if not path.exists():
            print(f"[ecthr] attenzione: {path.name} non trovato, split '{split_name}' saltato.")
            continue

        df = pd.read_parquet(path)
        for idx, row in df.iterrows():
            text = _build_text(row["text"])
            if not text:
                skipped_empty += 1
                continue

            # labels: array di indici -> list[int] (multi-label). Vuoto = nessuna
            # violazione (valido): resta lista vuota.
            labels = [int(l) for l in row["labels"]]

            documents.append(
                Document(
                    doc_id=f"ecthr_{split_name}_{idx}",
                    text=text,
                    label=labels,                      # list[int] -> multi-label
                    meta={
                        "split": split_name,
                        "n_violations": len(labels),
                        "n_paragraphs": len(row["text"]),
                        "n_chars": len(text),
                    },
                )
            )

    n_by_split = {}
    for d in documents:
        n_by_split[d.meta["split"]] = n_by_split.get(d.meta["split"], 0) + 1
    print(f"[ecthr] caricati {len(documents)} documenti {n_by_split} "
          f"({skipped_empty} vuoti). Task: multi-label, {NUM_CLASSES} classi.")
    return documents