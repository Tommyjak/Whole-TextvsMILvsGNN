"""
chunking.py — Segmentazione generica del testo lungo.

Prende il `text` completo di un Document e lo spezza in finestre di token di
dimensione fissa, con overlap. È il passo "a valle" dei loader: la granularità
(chunk_size) è una VARIABILE DI ABLATION (64 / 128 / 256), quindi vive qui e
non nei loader — cambiare granularità non deve costringere a rileggere i dataset.

Chi lo usa:
  - MIL e GNN condividono gli stessi chunk: il chunker viene chiamato UNA volta
    nel precompute degli embedding (encoders/cache.py), e i due modelli leggono
    gli stessi embedding cachati. L'identità dei chunk è così garantita.
  - il whole-text NON usa il chunker: consuma `text` intero, troncato a 512.

Perché serve un tokenizer:
  La finestra è misurata in TOKEN, non in caratteri o parole, perché è il token
  l'unità che conta per l'encoder (limite 512; granularità 64/128/256). Il
  tokenizer passato DEVE essere quello dell'encoder che poi encoderà i chunk
  (BERT cinese per IMCS, clinico EN per MTSamples, MentalBERT per DAIC):
  contare i token con un tokenizer diverso disallineerebbe i confini dei chunk.

Definizione di chunk_size:
  conta i token di CONTENUTO (add_special_tokens=False). L'encoder, quando
  ri-tokenizza ogni chunk, aggiunge da sé [CLS]/[SEP]: un chunk da 64 diventa
  ~66 token in ingresso, ampiamente sotto i 512.
"""

from __future__ import annotations

from transformers import PreTrainedTokenizerBase

from src.data.canonical import Document


def chunk_text(
    text: str,
    tokenizer: PreTrainedTokenizerBase,
    chunk_size: int = 128,
    overlap: int = 16,
) -> list[str]:
    """Spezza `text` in finestre di `chunk_size` token con `overlap` token di
    sovrapposizione tra finestre adiacenti.

    Ritorna una lista ORDINATA di stringhe: l'ordine è la sequenza del testo,
    e la GNN lo usa per gli archi sequenziali i->i+1. Se il testo sta in una
    sola finestra, ritorna [text]. Non fa padding (è compito dell'encoder).
    """
    if chunk_size <= 0:
        raise ValueError(f"chunk_size deve essere > 0, ricevuto {chunk_size}")
    if overlap < 0:
        raise ValueError(f"overlap deve essere >= 0, ricevuto {overlap}")
    if overlap >= chunk_size:
        # stride = chunk_size - overlap: se overlap >= chunk_size lo stride è
        # <= 0 e la finestra non avanzerebbe mai (loop infinito).
        raise ValueError(
            f"overlap ({overlap}) deve essere < chunk_size ({chunk_size})."
        )

    # Tokenizza UNA volta l'intero documento, senza special token.
    token_ids = tokenizer.encode(text, add_special_tokens=False, truncation=False, verbose = False)

    # Documento che sta in una sola finestra -> un solo chunk (il testo com'è).
    if len(token_ids) <= chunk_size:
        return [text]

    stride = chunk_size - overlap
    chunks: list[str] = []
    for start in range(0, len(token_ids), stride):
        window = token_ids[start : start + chunk_size]
        chunk = tokenizer.decode(
            window,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        ).strip()
        if chunk:  # scarta finestre che decodificano a vuoto
            chunks.append(chunk)
        # Fermati appena una finestra raggiunge la fine del documento: evita
        # code minuscole ridondanti e non lascia testo scoperto.
        if start + chunk_size >= len(token_ids):
            break

    return chunks


def chunk_document(
    doc: Document,
    tokenizer: PreTrainedTokenizerBase,
    chunk_size: int = 128,
    overlap: int = 16,
) -> list[str]:
    """Comodità: applica chunk_text al testo di un Document."""
    return chunk_text(doc.text, tokenizer, chunk_size=chunk_size, overlap=overlap)