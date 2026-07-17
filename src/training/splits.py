from __future__ import annotations

import random
from collections import defaultdict

from src.data.canonical import Document

VALID_SPLITS = ("train", "dev", "test")


def _has_official_split(documents: list[Document]) -> bool:
    seen = {d.meta.get("split") for d in documents}
    return seen.issubset(set(VALID_SPLITS)) and seen != {None}


def _stratify_key(doc: Document) -> object:
    return None if doc.is_multilabel else doc.label


def resolve_splits(
    documents: list[Document],
    seed: int = 42,
    ratios: tuple[float, float, float] = (0.7, 0.15, 0.15),
    force_generate: bool = False,
) -> dict[str, str]:
    
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
    if split not in VALID_SPLITS:
        raise ValueError(f"split deve essere uno di {VALID_SPLITS}, ricevuto {split!r}.")
    return [d for d in documents if split_map.get(d.doc_id) == split]

def resolve_splits_montecarlo(
    documents: list[Document],
    fold_seed: int,
    ratios: tuple[float, float, float] = (0.7, 0.15, 0.15),
) -> dict[str, str]:
    mapping = _generate_stratified(documents, seed=fold_seed, ratios=ratios)
    _report(mapping, source=f"montecarlo-cv (fold_seed={fold_seed})")
    return mapping