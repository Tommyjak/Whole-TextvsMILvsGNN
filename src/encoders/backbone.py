from __future__ import annotations

import torch
from torch import Tensor
from transformers import AutoModel, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase


def resolve_device(device: str | torch.device | None = None) -> torch.device:
    if device is not None:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_hf_encoder(
    model_name: str,
    device: str | torch.device | None = None,
) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.to(resolve_device(device))
    return model, tokenizer


def masked_mean_pool(last_hidden_state: Tensor, attention_mask: Tensor) -> Tensor:
    mask = attention_mask.unsqueeze(-1).to(last_hidden_state.dtype)  # (B, L, 1)
    summed = (last_hidden_state * mask).sum(dim=1)                   # (B, H)
    counts = mask.sum(dim=1).clamp(min=1.0)                         # (B, 1)
    return summed / counts


class FrozenEncoder:
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
        if len(texts) == 0:
            # documento senza chunk: ritorna una matrice vuota, così i modelli a valle gestiscono il caso senza crash.
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