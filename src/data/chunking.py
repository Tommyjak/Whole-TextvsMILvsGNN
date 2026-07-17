from __future__ import annotations

from transformers import PreTrainedTokenizerBase

from src.data.canonical import Document


def chunk_text(
    text: str,
    tokenizer: PreTrainedTokenizerBase,
    chunk_size: int = 128,
    overlap: int = 16,
) -> list[str]:

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
        if chunk:
            chunks.append(chunk)
        # evita code minuscole ridondanti e non lascia testo scoperto in fine al documento
        if start + chunk_size >= len(token_ids):
            break

    return chunks


def chunk_document(
    doc: Document,
    tokenizer: PreTrainedTokenizerBase,
    chunk_size: int = 128,
    overlap: int = 16,
) -> list[str]:
    return chunk_text(doc.text, tokenizer, chunk_size=chunk_size, overlap=overlap)