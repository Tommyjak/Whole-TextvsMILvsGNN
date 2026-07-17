from __future__ import annotations

import csv
import re
from pathlib import Path

import pandas as pd

from src.data.canonical import Document

EXCLUDED: set[int] = {342, 394, 398, 460, 451, 458, 480}

def _find_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower:
            return lower[cand.lower()]
    return None

def _read_split_file(path: Path, split_name: str) -> dict[int, dict]:
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
    index: dict[int, Path] = {}
    for path in root.glob("**/*TRANSCRIPT*.csv"):
        m = re.match(r"(\d+)", path.name)
        if m:
            index[int(m.group(1))] = path
    return index

def _load_transcript_text(
    path: Path,
    include_ellie: bool = True,
    speaker_markers: bool = True,
) -> str:
    
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

def load_daic(
    root: str | Path,
    include_ellie: bool = True,
    speaker_markers: bool = True,
    splits: tuple[str, ...] = ("train", "dev", "test"),
) -> list[Document]:
    
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