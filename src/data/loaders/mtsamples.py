from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.data.canonical import Document

# categorie che sono TIPI DI DOCUMENTO, non discipline cliniche -> escluse
DEFAULT_NON_SPECIALTY: set[str] = {
    "SOAP / Chart / Progress Notes",
    "Office Notes",
    "Letters",
    "Discharge Summary",
    "IME-QME-Work Comp etc.",
    "Consult - History and Phy.",
}


def _build_specialty_vocab(specialties: list[str], path: Path) -> dict[str, int]:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    mapping = {name: i for i, name in enumerate(sorted(set(specialties)))}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(mapping, ensure_ascii=False, indent=2), encoding="utf-8")
    return mapping


def load_mtsamples(
    csv_path: str | Path,
    min_per_class: int = 40,
    cap_per_class: int = 300,
    non_specialty: set[str] | None = None,
    seed: int = 42,
    vocab_path: str | Path | None = None,
) -> list[Document]:
    
    csv_path = Path(csv_path)
    non_specialty = DEFAULT_NON_SPECIALTY if non_specialty is None else non_specialty
    vocab_path = Path(vocab_path) if vocab_path else csv_path.parent / "specialty_vocab.json"

    df = pd.read_csv(csv_path)

    # 1. normalizza la specialita (ha spazi iniziali nel CSV) e droppa testo vuoto
    df["medical_specialty"] = df["medical_specialty"].astype(str).str.strip()
    df = df.dropna(subset=["transcription"])
    df["transcription"] = df["transcription"].astype(str).str.strip()
    df = df[df["transcription"] != ""]
    n_after_text = len(df)

    # 2. dedup su transcription (evita leakage dello stesso referto tra split)
    df = df.drop_duplicates(subset=["transcription"], keep="first")
    n_after_dedup = len(df)

    # 3. rimuovi le non-specialita
    df = df[~df["medical_specialty"].isin(non_specialty)]

    # 4. soglia minima per classe
    counts = df["medical_specialty"].value_counts()
    keep = counts[counts >= min_per_class].index
    df = df[df["medical_specialty"].isin(keep)]

    # 5. cap per classe (campionamento deterministico)
    capped_parts = []
    for _, group in df.groupby("medical_specialty"):
        n = min(len(group), cap_per_class)
        capped_parts.append(group.sample(n=n, random_state=seed))
    df = pd.concat(capped_parts).reset_index(drop=True)

    # vocabolario specialita -> indice
    vocab = _build_specialty_vocab(df["medical_specialty"].tolist(), vocab_path)

    documents: list[Document] = []
    for i, row in df.iterrows():
        specialty = row["medical_specialty"]
        text = row["transcription"]
        documents.append(
            Document(
                doc_id=f"mts_{int(row['Unnamed: 0'])}",  # id originale del CSV
                text=text,
                label=vocab[specialty],   # int -> single-label multi-classe
                meta={
                    "split": "all",       # split assegnato a valle
                    "specialty": specialty,
                    "n_chars": len(text),
                },
            )
        )

    print(
        f"[mtsamples] {n_after_text} con testo -> {n_after_dedup} dopo dedup -> "
        f"{len(documents)} finali su {len(vocab)} classi "
        f"(min>={min_per_class}, cap={cap_per_class})."
    )
    return documents