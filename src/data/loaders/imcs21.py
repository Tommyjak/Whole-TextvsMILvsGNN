"""
loaders/imcs21.py — Loader per IMCS-21 (consulti medici cinesi).

Verificato sui file reali (train/dev/test.json). Ogni record e un dict indicizzato
da example_id con chiavi:
    diagnosis      : str  -> una delle 10 malattie pediatriche (label multi-classe)
    self_report    : str  -> auto-anamnesi del paziente (x0)
    dialogue       : list -> turni; ogni turno ha speaker ('医生'/'患者') e 'sentence'
    explicit_info  : dict -> sintomi espliciti (NON usato per il testo)
    implicit_info  : dict -> sintomi impliciti  (NON usato per il testo)
    report         : list -> 2 referti di riferimento (annotazione, NON usata come input)

Split ufficiale: train 2472 / dev 833 / test 811, tutti CON label diagnosis.
(test_input.json e la versione SENZA label per la submission: non lo usiamo.)

Testo prodotto: self_report + turni (speaker 医生/患者 -> DOCTOR:/PATIENT:), in ordine.
Il testo resta in CINESE -> encoder BERT cinese; il chunker usa quel tokenizer.
Label: stringa malattia -> indice 0..9 via DiseaseVocab persistito su disco.
"""

from __future__ import annotations

import json
from pathlib import Path

from src.data.canonical import Document

# le 10 malattie sono note e fisse: definiamo un ordine CANONICO esplicito, cosi
# gli indici non dipendono da quali file carichi ne dal loro ordine.
CANONICAL_DISEASES: list[str] = [
    "上呼吸道感染",      # infezione delle alte vie respiratorie
    "小儿support",       # placeholder sostituito sotto (vedi nota)
]
# NB: definiti sotto per esteso per evitare errori di copia — vedi DISEASES.

DISEASES: list[str] = [
    "上呼吸道感染",   # 0  upper respiratory infection
    "小儿发热",       # 1  fever
    "小儿便秘",       # 2  constipation
    "小儿咳嗽",       # 3  cough
    "小儿支气管肺炎",  # 4  bronchopneumonia
    "小儿支气管炎",   # 5  bronchitis
    "小儿消化不良",   # 6  dyspepsia
    "小儿腹泻",       # 7  diarrhea
    "小儿感冒",       # 8  cold
    "新生儿黄疸",     # 9  neonatal jaundice
]
DISEASE2ID: dict[str, int] = {name: i for i, name in enumerate(DISEASES)}

_SPEAKER_MAP = {"医生": "DOCTOR:", "患者": "PATIENT:"}


def _build_text(record: dict, speaker_markers: bool = True) -> str:
    parts: list[str] = []

    self_report = str(record.get("self_report", "")).strip()
    if self_report:
        parts.append(f"SELF_REPORT: {self_report}" if speaker_markers else self_report)

    for turn in record.get("dialogue", []):
        sentence = str(turn.get("sentence", "")).strip()
        if not sentence:
            continue
        if speaker_markers:
            marker = _SPEAKER_MAP.get(turn.get("speaker", ""), f"{turn.get('speaker','')}:")
            parts.append(f"{marker} {sentence}")
        else:
            parts.append(sentence)

    return " ".join(parts).strip()


def _load_json(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_imcs21(
    root: str | Path,
    splits: tuple[str, ...] = ("train", "dev", "test"),
    speaker_markers: bool = True,
) -> list[Document]:
    """Carica IMCS-21 come lista di Document.

    root: cartella con train.json / dev.json / test.json.
    Usa test.json (CON label), non test_input.json.
    """
    root = Path(root)
    split_files = {"train": "train.json", "dev": "dev.json", "test": "test.json"}

    documents: list[Document] = []
    skipped_empty = 0
    unknown_disease = 0

    for split_name in splits:
        path = root / split_files[split_name]
        if not path.exists():
            print(f"[imcs21] attenzione: {path.name} non trovato, split '{split_name}' saltato.")
            continue

        data = _load_json(path)
        for example_id, record in data.items():
            disease = str(record.get("diagnosis", "")).strip()
            if disease not in DISEASE2ID:
                unknown_disease += 1
                print(f"[imcs21] {split_name}:{example_id}: malattia '{disease}' "
                      f"non tra le 10 canoniche, saltata.")
                continue

            text = _build_text(record, speaker_markers=speaker_markers)
            if not text:
                skipped_empty += 1
                continue

            documents.append(
                Document(
                    doc_id=f"imcs_{example_id}",
                    text=text,
                    label=DISEASE2ID[disease],   # int 0..9 -> single-label multi-classe
                    meta={
                        "split": split_name,
                        "disease_zh": disease,
                        "n_turns": len(record.get("dialogue", [])),
                        "n_chars": len(text),
                    },
                )
            )

    n_by_split = {}
    for d in documents:
        n_by_split[d.meta["split"]] = n_by_split.get(d.meta["split"], 0) + 1
    print(f"[imcs21] caricati {len(documents)} documenti {n_by_split} "
          f"({skipped_empty} vuoti, {unknown_disease} malattie ignote).")
    return documents


def id_to_disease(idx: int) -> str:
    """Utility inversa per interpretare le predizioni."""
    return DISEASES[idx]