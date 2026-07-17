"""
loaders/ecthr.py — Loader per ECtHR (LexGLUE, Task A).

Verificato sui parquet reali. Ogni caso ha due campi:
    text   : array di paragrafi fattuali (segmentazione GIA data dagli autori)
    labels : array di indici degli articoli CEDU violati -> MULTI-LABEL (10 classi)

Scelte:
  - I paragrafi vengono CONCATENATI in un'unica stringa `text`, come per gli altri
    dataset: la segmentazione la fa il chunker generico a valle (chunking uniforme).
    Non usiamo i paragrafi come segmenti pre-fatti, per coerenza del confronto.
  - Multi-label: label = list[int] (indici attivi). Lista VUOTA e valida e frequente
    (~900 casi senza violazione) -> il Document la gestisce (multi-hot tutto-zero).
  - Split: i tre parquet (train/validation/test) SONO lo split ufficiale cronologico
    di LexGLUE. Ogni file -> il suo meta['split'].

Lingua: inglese (legale) -> encoder inglese. Regime: documenti LUNGHI (media ~9800
char, coda a >200k) -> il caso dove il troncamento del whole-text morde davvero.
"""

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
    """Concatena i paragrafi fattuali in un'unica stringa.

    para_markers=False di default: i paragrafi ECtHR sono gia numerati nel testo
    ('11. At the beginning...'), quindi non servono marcatori aggiuntivi. Li
    uniamo con spazio, come farebbe un documento continuo."""
    parts = [str(p).strip() for p in paragraphs if str(p).strip()]
    return " ".join(parts).strip()


def load_ecthr(
    root: str | Path,
    splits: tuple[str, ...] = ("train", "dev", "test"),
) -> list[Document]:
    """Carica ECtHR come lista di Document (multi-label).

    root: cartella con i tre parquet (train/validation/test)."""
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