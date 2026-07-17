from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Document:
    doc_id: str
    text: str
    label: int | list[int]
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.doc_id, str) or not self.doc_id:
            raise ValueError(
                f"doc_id deve essere una stringa non vuota, ricevuto: {self.doc_id!r}"
            )
        if not isinstance(self.text, str) or not self.text.strip():
            raise ValueError(
                f"[{self.doc_id}] text vuoto o non stringa: "
                f"il loader deve filtrare i documenti senza testo (es. righe "
                f"vuote di MTSamples)."
            )
        if isinstance(self.label, bool) or not isinstance(self.label, (int, list)):
            raise ValueError(
                f"[{self.doc_id}] label deve essere int (single-label) o "
                f"list[int] (multi-label), ricevuto: {type(self.label).__name__}"
            )
        if isinstance(self.label, list) and not all(isinstance(x, int) for x in self.label):
            raise ValueError(
                f"[{self.doc_id}] label multi-label deve contenere solo int "
                f"(indici di classe), ricevuto: {self.label!r}"
            )

    @property
    def is_multilabel(self) -> bool:
        return isinstance(self.label, list)

    @property
    def char_len(self) -> int:
        return len(self.text)

    def __repr__(self) -> str:
        # troncamento dei log in output su terminale a 60 caratteri
        preview = self.text[:60].replace("\n", " ")
        suffix = "…" if len(self.text) > 60 else ""
        return (
            f"Document(doc_id={self.doc_id!r}, label={self.label}, "
            f"chars={self.char_len}, text={preview!r}{suffix})"
        )