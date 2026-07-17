"""
splits.py — Partizione uniforme train/dev/test dei Document.

Risolve un'asimmetria tra dataset:
  - DAIC, IMCS  -> hanno uno split UFFICIALE, gia in meta['split']. Va RISPETTATO
                   (comparabilita con la letteratura).
  - MTSamples   -> nessuno split ufficiale (tutti meta['split']=='all'). Va
                   GENERATO, stratificato per label e deterministico per seed.

L'output e sempre lo stesso formato — un dict {doc_id: 'train'/'dev'/'test'} —
consultato IDENTICO da entrambi i Dataset (cache e testo) e da tutti e tre i
modelli. E' questo che garantisce che MIL, GNN e whole-text siano valutati sulla
STESSA partizione: senza, il confronto sarebbe falsato.

Stratificazione: mantiene le proporzioni di classe in ogni split (cruciale su
dataset piccoli/sbilanciati come MTSamples). Per il multilabel (ECtHR, futuro) la
stratificazione perfetta non esiste; si ripiega su uno split casuale per-seed.
"""

from __future__ import annotations

import random
from collections import defaultdict

from src.data.canonical import Document

VALID_SPLITS = ("train", "dev", "test")


def _has_official_split(documents: list[Document]) -> bool:
    """True se i documenti portano uno split ufficiale utilizzabile in meta.
    'all' non conta come ufficiale: e il segnaposto di chi non ne ha uno."""
    seen = {d.meta.get("split") for d in documents}
    return seen.issubset(set(VALID_SPLITS)) and seen != {None}


def _stratify_key(doc: Document) -> object:
    """Chiave di stratificazione: la label per single-label; None per multilabel
    (dove la stratificazione esatta non e definibile)."""
    return None if doc.is_multilabel else doc.label


def resolve_splits(
    documents: list[Document],
    seed: int = 42,
    ratios: tuple[float, float, float] = (0.7, 0.15, 0.15),
    force_generate: bool = False,
) -> dict[str, str]:
    """Ritorna {doc_id: split}. Usa lo split ufficiale se presente, altrimenti
    ne genera uno stratificato e deterministico.

    force_generate=True ignora lo split ufficiale e ne genera comunque uno nuovo
    (utile per ablation di robustezza allo split)."""
    if _has_official_split(documents) and not force_generate:
        mapping = {d.doc_id: d.meta["split"] for d in documents}
        _report(mapping, source="ufficiale")
        return mapping

    mapping = _generate_stratified(documents, seed=seed, ratios=ratios)
    _report(mapping, source=f"generato (seed={seed})")
    return mapping


def _generate_stratified(
    documents: list[Document],
    seed: int,
    ratios: tuple[float, float, float],
) -> dict[str, str]:
    """Genera uno split stratificato per label, deterministico per seed.

    Raggruppa i documenti per classe, mescola ogni gruppo con lo stesso seed, e
    assegna train/dev/test rispettando i ratios DENTRO ogni gruppo. Cosi le
    proporzioni di classe sono preservate in tutti e tre gli split."""
    if abs(sum(ratios) - 1.0) > 1e-6:
        raise ValueError(f"i ratios devono sommare a 1.0, ricevuto {ratios} (somma {sum(ratios)}).")

    rng = random.Random(seed)
    by_class: dict[object, list[str]] = defaultdict(list)
    for doc in documents:
        by_class[_stratify_key(doc)].append(doc.doc_id)

    r_train, r_dev, _ = ratios
    mapping: dict[str, str] = {}
    for _, ids in sorted(by_class.items(), key=lambda kv: str(kv[0])):
        ids = ids.copy()
        rng.shuffle(ids)
        n = len(ids)
        n_train = int(n * r_train)
        n_dev = int(n * r_dev)
        for i, doc_id in enumerate(ids):
            if i < n_train:
                mapping[doc_id] = "train"
            elif i < n_train + n_dev:
                mapping[doc_id] = "dev"
            else:
                mapping[doc_id] = "test"
    return mapping


def _report(mapping: dict[str, str], source: str) -> None:
    counts = {s: 0 for s in VALID_SPLITS}
    for split in mapping.values():
        counts[split] = counts.get(split, 0) + 1
    print(f"[splits] {source}: train={counts['train']} dev={counts['dev']} test={counts['test']}")


def filter_by_split(
    documents: list[Document],
    split_map: dict[str, str],
    split: str,
) -> list[Document]:
    """Sottoinsieme dei documenti che appartengono a `split`."""
    if split not in VALID_SPLITS:
        raise ValueError(f"split deve essere uno di {VALID_SPLITS}, ricevuto {split!r}.")
    return [d for d in documents if split_map.get(d.doc_id) == split]

def resolve_splits_montecarlo(
    documents: list[Document],
    fold_seed: int,
    ratios: tuple[float, float, float] = (0.7, 0.15, 0.15),
) -> dict[str, str]:
    """Monte Carlo CV: genera SEMPRE uno split stratificato nuovo dal fold_seed,
    ignorando lo split ufficiale. Ogni fold (fold_seed diverso) da' una partizione
    train/dev/test diversa -> ripetendo su piu' fold e mediando si ottiene una
    stima piu' robusta (adatta ai dataset piccoli).

    A differenza di resolve_splits, qui lo split UFFICIALE non viene mai usato:
    e' il punto della Monte Carlo CV. Resta comunque a 3 vie (train/dev/test), cosi
    l'early stopping usa il dev e il test resta held-out — nessun leakage."""
    mapping = _generate_stratified(documents, seed=fold_seed, ratios=ratios)
    _report(mapping, source=f"montecarlo-cv (fold_seed={fold_seed})")
    return mapping