"""
loaders/daic.py — Loader per DAIC-WOZ.

Converte le sessioni DAIC-WOZ in Document canonici. Come deciso, si usano SOLO i
file *_TRANSCRIPT.csv come fonte del testo (le label vengono dagli split CSV, che
sono metadati, non feature). Il pairing Q&A è ABBANDONATO: qui si ricostruisce la
trascrizione come un'unica stringa, in ordine temporale.

Formato della fonte:
  - datasets/daic-woz/**/XXX_TRANSCRIPT.csv : tab-separated, colonne
    start_time, stop_time, speaker (Ellie/Participant), value (testo).
  - split CSV con le label PHQ8_Binary:
        *train_split*.csv, *dev_split*.csv, *full_test_split*.csv
    ATTENZIONE: il file di test usa nomi di colonna leggermente diversi da
    train/dev (PHQ_Binary vs PHQ8_Binary, maiuscole incoerenti). Gestito sotto.

Sessioni escluse (convenzione AVEC2017 + trascrizione Ellie mancante):
  342, 394, 398, 460  -> problemi tecnici
  451, 458, 480       -> trascrizione di Ellie mancante
Le teniamo escluse per comparabilità con la letteratura e con la tua pipeline
MIL precedente, anche se ora — non facendo più il pairing — potrebbero tecnicamente
essere processate col solo testo del partecipante.
"""

from __future__ import annotations

import csv
import re
from pathlib import Path

import pandas as pd

from src.data.canonical import Document

EXCLUDED: set[int] = {342, 394, 398, 460, 451, 458, 480}


# ------------------------- helper robusti sui CSV -------------------------

def _find_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    """Trova la prima colonna che matcha (case-insensitive) uno dei candidati.
    Serve perché i file DAIC hanno nomi di colonna incoerenti tra split."""
    lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower:
            return lower[cand.lower()]
    return None


def _read_split_file(path: Path, split_name: str) -> dict[int, dict]:
    """Legge un split CSV e ritorna {session_id: {label, score, split}}."""
    df = pd.read_csv(path)
    id_col = _find_col(df, ["Participant_ID", "participant_ID"])
    lbl_col = _find_col(df, ["PHQ8_Binary", "PHQ_Binary"])
    score_col = _find_col(df, ["PHQ8_Score", "PHQ_Score"])

    if id_col is None or lbl_col is None:
        raise ValueError(
            f"{path.name}: colonne ID/label non trovate "
            f"(ho colonne: {list(df.columns)})."
        )

    out: dict[int, dict] = {}
    for _, row in df.iterrows():
        sid = int(row[id_col])
        out[sid] = {
            "label": int(row[lbl_col]),
            "score": int(row[score_col]) if score_col and not pd.isna(row[score_col]) else None,
            "split": split_name,
        }
    return out


def _build_label_map(root: Path) -> dict[int, dict]:
    """Unisce train/dev/test in un'unica mappa {session_id: {label, score, split}}.

    Per il test, i file candidati vengono provati IN ORDINE di preferenza e si usa
    il primo che contiene davvero una colonna di label:
      - full_test_split.csv          -> ha PHQ_Binary (label rivelate post-challenge)
      - test_split_Depression_*.csv  -> versione "cieca" della challenge, SENZA label
    Cosi non si aggancia per errore il file senza label.
    """
    patterns = {
        "train": ["*train_split*.csv"],
        "dev": ["*dev_split*.csv"],
        "test": ["full_test_split.csv", "*test_split*.csv"],  # priorità al file con label
    }
    label_map: dict[int, dict] = {}
    for split_name, pats in patterns.items():
        # raccogli i candidati nell'ordine di preferenza dei pattern
        candidates: list[Path] = []
        for pat in pats:
            for m in sorted(root.glob(pat)):
                if m not in candidates:
                    candidates.append(m)

        if not candidates:
            print(f"[daic] attenzione: nessun file per lo split '{split_name}'.")
            continue

        # usa il PRIMO candidato che contiene davvero una colonna di label
        chosen = None
        for cand in candidates:
            header = pd.read_csv(cand, nrows=0)
            if _find_col(header, ["PHQ8_Binary", "PHQ_Binary"]) is not None:
                chosen = cand
                break

        if chosen is None:
            print(f"[daic] attenzione: per lo split '{split_name}' nessun file "
                  f"con colonna label tra {[c.name for c in candidates]}, saltato.")
            continue

        label_map.update(_read_split_file(chosen, split_name))

    if not label_map:
        raise FileNotFoundError(f"Nessuno split CSV con label trovato in {root}.")
    return label_map


def _index_transcripts(root: Path) -> dict[int, Path]:
    """Mappa {session_id: path del suo *_TRANSCRIPT.csv}, cercando ovunque
    sotto root (funziona sia con transcripts/ sia con cartelle per sessione)."""
    index: dict[int, Path] = {}
    for path in root.glob("**/*TRANSCRIPT*.csv"):
        m = re.match(r"(\d+)", path.name)
        if m:
            index[int(m.group(1))] = path
    return index


# ------------------------- costruzione del testo -------------------------

def _load_transcript_text(
    path: Path,
    include_ellie: bool = True,
    speaker_markers: bool = True,
) -> str:
    """Ricostruisce la trascrizione come singola stringa, in ordine temporale.

    Turni consecutivi dello stesso speaker vengono accorpati, così il testo legge
    come una conversazione alternata e i marcatori non si ripetono a ogni riga.
    """
    df = pd.read_csv(path, sep="\t", quoting=csv.QUOTE_NONE, on_bad_lines="skip")
    df.columns = [c.strip().lower() for c in df.columns]
    if "speaker" not in df.columns or "value" not in df.columns:
        raise ValueError(f"{path.name}: attese colonne 'speaker' e 'value', trovate {list(df.columns)}.")

    parts: list[str] = []
    prev_speaker: str | None = None
    buffer: list[str] = []

    def flush() -> None:
        if not buffer:
            return
        text = " ".join(buffer).strip()
        if not text:
            return
        is_ellie = (prev_speaker or "").lower() == "ellie"
        if is_ellie and not include_ellie:
            return
        if speaker_markers:
            marker = "ELLIE:" if is_ellie else "PARTICIPANT:"
            parts.append(f"{marker} {text}")
        else:
            parts.append(text)

    for _, row in df.iterrows():
        speaker = row["speaker"]
        value = row["value"]
        if pd.isna(value) or not str(value).strip():
            continue
        if speaker != prev_speaker and prev_speaker is not None:
            flush()
            buffer = []
        buffer.append(str(value).strip())
        prev_speaker = speaker
    flush()  # ultimo gruppo

    return " ".join(parts).strip()


# ------------------------------- API pubblica -------------------------------

def load_daic(
    root: str | Path,
    include_ellie: bool = True,
    speaker_markers: bool = True,
    splits: tuple[str, ...] = ("train", "dev", "test"),
) -> list[Document]:
    """Carica DAIC-WOZ come lista di Document (tutti gli split richiesti).

    Ogni Document ha meta = {split, phq8_score, n_chars, source_file}. Lo split
    viaggia in meta perché il precompute embedda TUTTO una volta, e il filtro
    train/dev/test avviene a valle, al training.
    """
    root = Path(root)
    label_map = _build_label_map(root)
    transcripts = _index_transcripts(root)

    documents: list[Document] = []
    missing, excluded = 0, 0

    for sid, info in sorted(label_map.items()):
        if info["split"] not in splits:
            continue
        if sid in EXCLUDED:
            excluded += 1
            continue
        if sid not in transcripts:
            missing += 1
            print(f"[daic] sessione {sid}: trascrizione mancante, saltata.")
            continue

        text = _load_transcript_text(transcripts[sid], include_ellie, speaker_markers)
        if not text:
            missing += 1
            print(f"[daic] sessione {sid}: testo vuoto dopo il parsing, saltata.")
            continue

        documents.append(
            Document(
                doc_id=str(sid),
                text=text,
                label=info["label"],          # int 0/1 -> single-label
                meta={
                    "split": info["split"],
                    "phq8_score": info["score"],
                    "n_chars": len(text),
                    "source_file": transcripts[sid].name,
                },
            )
        )

    print(
        f"[daic] caricati {len(documents)} documenti "
        f"({excluded} esclusi, {missing} mancanti/vuoti)."
    )
    return documents