"""
whole_text.py — Modello whole-text: NESSUNA frammentazione.

Il modello a se. NON legge la cache, NON usa il chunker: prende il testo INTERO
del Document, lo tronca a max_length, e lo processa in un unico forward. E' l'unico
dei tre che fa FINE-TUNING dell'encoder (il [CLS] e un buon vettore-documento solo
se addestrato) -> confounder dichiarato: encoder allenabile vs frozen di MIL/GNN.

max_length e' l'iperparametro centrale dell'esperimento:
  - 512  con BERT standard  -> esperimento principale (stesso backbone dei chunk)
  - 4096 con Longformer     -> cella di controllo a contesto lungo (su ECtHR)
Il troncamento (truncation=True) e' VOLUTO: e' il "costo del troncamento" che lo
studio misura, l'opposto di cio che il chunker fa in MIL/GNN.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from src.encoders.backbone import load_hf_encoder
from src.models.heads import ClassificationHead


class WholeTextModel(nn.Module):
    """Encoder (fine-tunato) sul testo intero -> [CLS] -> testa condivisa.

    forward prende una lista di stringhe (i documenti interi) e ritorna i logit.
    A differenza di MIL/GNN, l'encoder e' PARTE del modello e riceve gradiente.
    """

    def __init__(
        self,
        model_name: str,
        task: str,
        num_classes: int | None = None,
        max_length: int = 512,
        dropout: float = 0.1,
        device: str | torch.device | None = None,
        is_longformer: bool | None = None,
    ) -> None:
        super().__init__()
        self.encoder, self.tokenizer = load_hf_encoder(model_name, device)
        # NB: nessun freeze qui -> l'encoder si allena (fine-tuning)
        self.max_length = max_length
        self.hidden_size = self.encoder.config.hidden_size
        self.is_longformer = (
            "longformer" in model_name.lower() if is_longformer is None else is_longformer
        )
        self.head = ClassificationHead(self.hidden_size, task, num_classes, dropout)

    @property
    def device(self) -> torch.device:
        return next(self.encoder.parameters()).device

    def forward(self, texts: list[str]) -> Tensor:
        enc = self.tokenizer(
            texts,
            padding=True,
            truncation=True,                 # VOLUTO: tronca il documento a max_length
            max_length=self.max_length,
            return_tensors="pt",
        ).to(self.device)

        kwargs = {}
        if self.is_longformer:
            # global attention sul [CLS]: senza, il Longformer non "vede" tutto il doc
            gmask = torch.zeros_like(enc["input_ids"])
            gmask[:, 0] = 1
            kwargs["global_attention_mask"] = gmask

        out = self.encoder(**enc, **kwargs)
        doc_vec = out.last_hidden_state[:, 0]     # (B, H) — il [CLS]
        return self.head(doc_vec)                 # (B, out_features)