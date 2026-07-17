"""
canonical.py — Il contratto dati del progetto.

Ogni loader (uno per dataset, in src/data/loaders/) legge la propria fonte
grezza e la converte in una lista di `Document`. Da qui in poi NESSUN modulo
sa più da quale dataset provengano i dati: chunker, encoder, modelli e harness
lavorano solo su `Document`. È questa uniformità che rende il confronto
MIL / GNN / whole-text pulito e comparabile.

Convenzioni:

- `text` è il documento COMPLETO come singola stringa. Il whole-text lo consuma
  intero (troncato a max_length); il chunker (src/data/chunking.py) lo spezza in
  finestre di token. La segmentazione NON avviene qui: la granularità è una
  variabile di ablation, quindi bakerla nel loader costringerebbe a rigenerare
  i Document a ogni granularità.

- `label` segue una convenzione unica:
    * single-label (DAIC binario, IMCS 10 classi, MTSamples n classi) -> int,
      l'indice di classe (0, 1, 2, ...).
    * multi-label (ECtHR) -> list[int], gli indici delle classi attive
      (lista vuota = nessuna classe attiva, es. caso senza violazioni).
  Il numero di classi e il tipo di task vivono nel CONFIG a livello di dataset,
  non qui: il singolo documento non deve saperlo.

- `meta` è una via di fuga per informazioni dataset-specifiche (split di
  appartenenza, id originale, lunghezza in token, eventuali confini di
  segmentazione naturale...). Nessun modello deve dipendere da `meta`:
  serve solo per analisi, stratificazione e debug.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Document:
    doc_id: str
    text: str
    label: int | list[int]
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Validazione difensiva: cattura subito i bug dei loader, invece di
        # farli propagare silenziosamente fino al training.
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
        """True se la label è una lista di indici (task multi-label)."""
        return isinstance(self.label, list)

    @property
    def char_len(self) -> int:
        """Lunghezza in caratteri: proxy rapido per stratificare i documenti
        (la lunghezza in token, che serve per la soglia 512, la calcola il
        tokenizer a valle)."""
        return len(self.text)

    def __repr__(self) -> str:
        # repr auto-generato dumperebbe migliaia di caratteri di testo:
        # lo tronchiamo per rendere leggibili i log e il debug.
        preview = self.text[:60].replace("\n", " ")
        suffix = "…" if len(self.text) > 60 else ""
        return (
            f"Document(doc_id={self.doc_id!r}, label={self.label}, "
            f"chars={self.char_len}, text={preview!r}{suffix})"
        )