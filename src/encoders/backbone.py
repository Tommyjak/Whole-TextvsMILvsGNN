"""
backbone.py — Caricamento e congelamento dell'encoder condiviso.

Responsabilità unica: dato il nome di un modello HuggingFace, restituire un
encoder pronto a trasformare una lista di stringhe (i chunk) in una matrice di
embedding. È il "come" della produzione di embedding; il "cosa" e il "dove
salvare" sono di cache.py.

Perché frozen (ricapitolando la scelta di design):
  - MIL e GNN devono leggere gli STESSI embedding -> l'encoder è una costante
    identica per entrambi. Congelarlo (requires_grad=False) lo rende una
    funzione fissa: lo stesso chunk dà sempre lo stesso vettore.
  - eval() disattiva il dropout -> gli embedding sono DETERMINISTICI, quindi la
    cache su disco è valida e riproducibile. Senza eval(), lo stesso chunk
    darebbe vettori leggermente diversi a ogni chiamata.
  - @torch.no_grad() -> niente grafo dei gradienti in memoria: encoding molto
    più leggero, ed è ciò che rende praticabile il precompute su tutti i dataset.

Pooling:
  con encoder CONGELATO il token [CLS] è un sentence embedding debole (diventa
  buono solo col fine-tuning). Quindi il default è il MASKED MEAN POOLING dei
  token. L'opzione "cls" resta disponibile per l'ablation "mean-pool vs CLS".
  Il whole-text, che INVECE fa fine-tuning, non usa questa classe: caricherà il
  suo encoder allenabile in whole_text.py (via load_hf_encoder qui sotto) e userà
  il [CLS], perché lì il fine-tuning lo rende sensato.

Encoder per dataset (il nome preciso arriva dal config, non è hardcoded qui):
  - IMCS-21 (cinese)     -> es. 'bert-base-chinese' / 'hfl/chinese-roberta-wwm-ext'
  - MTSamples (EN clin.) -> es. 'emilyalsentzer/Bio_ClinicalBERT'
  - DAIC-WOZ (EN mental) -> es. 'mental/mental-bert-base-uncased'
"""

from __future__ import annotations

import torch
from torch import Tensor
from transformers import AutoModel, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase


def resolve_device(device: str | torch.device | None = None) -> torch.device:
    """Sceglie la GPU se disponibile, altrimenti la CPU, salvo device esplicito."""
    if device is not None:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_hf_encoder(
    model_name: str,
    device: str | torch.device | None = None,
) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    """Carica modello + tokenizer grezzi, SENZA congelare.

    Usato sia da FrozenEncoder (che poi congela) sia dal whole-text (che invece
    lascia allenabile per il fine-tuning). Centralizzare qui il caricamento
    evita di duplicare la logica in due posti.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.to(resolve_device(device))
    return model, tokenizer


def masked_mean_pool(last_hidden_state: Tensor, attention_mask: Tensor) -> Tensor:
    """Media dei token embedding pesata sulla attention mask.

    Il padding NON deve entrare nella media: i token di padding hanno mask=0 e
    vengono azzerati prima di sommare. Il denominatore è il numero di token reali
    (clampato a >=1 per non dividere per zero su input degeneri).

    last_hidden_state: (B, L, H)   attention_mask: (B, L)   ->   (B, H)
    """
    mask = attention_mask.unsqueeze(-1).to(last_hidden_state.dtype)  # (B, L, 1)
    summed = (last_hidden_state * mask).sum(dim=1)                   # (B, H)
    counts = mask.sum(dim=1).clamp(min=1.0)                         # (B, 1)
    return summed / counts


class FrozenEncoder:
    """Encoder congelato che trasforma stringhe in embedding.

    Espone `.tokenizer` (lo stesso che il chunker deve usare per contare i token)
    e `.hidden_size` (la dimensione dell'embedding, che serve ai modelli a valle).
    """

    def __init__(
        self,
        model_name: str,
        pooling: str = "mean",
        device: str | torch.device | None = None,
        max_length: int = 512,
    ) -> None:
        if pooling not in ("mean", "cls"):
            raise ValueError(f"pooling deve essere 'mean' o 'cls', ricevuto {pooling!r}")

        self.device = resolve_device(device)
        self.model, self.tokenizer = load_hf_encoder(model_name, self.device)
        self.pooling = pooling
        self.max_length = max_length

        # --- il congelamento vero e proprio ---
        self.model.eval()                          # dropout off -> deterministico
        for p in self.model.parameters():
            p.requires_grad_(False)                # i pesi non si aggiornano mai

        self.hidden_size: int = self.model.config.hidden_size

    @torch.no_grad()
    def encode(self, texts: list[str], batch_size: int = 32) -> Tensor:
        """Codifica una lista di stringhe in una matrice (N, hidden_size).

        Processa a batch per non saturare la memoria su documenti con molti chunk
        (es. ECtHR a granularità 64 -> centinaia di chunk). Ritorna un tensore su
        CPU in float32, pronto per essere salvato in cache.
        """
        if len(texts) == 0:
            # documento senza chunk: ritorna una matrice vuota ben formata,
            # così i modelli a valle gestiscono il caso senza crash.
            return torch.empty((0, self.hidden_size), dtype=torch.float32)

        out_chunks: list[Tensor] = []
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            enc = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            ).to(self.device)

            hidden = self.model(**enc).last_hidden_state  # (B, L, H)

            if self.pooling == "cls":
                pooled = hidden[:, 0]                      # (B, H)
            else:
                pooled = masked_mean_pool(hidden, enc["attention_mask"])

            out_chunks.append(pooled.to("cpu", dtype=torch.float32))

        return torch.cat(out_chunks, dim=0)  # (N, hidden_size)