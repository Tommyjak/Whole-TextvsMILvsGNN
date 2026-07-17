"""
gnn.py — Graph Neural Network sui chunk (GAT + readout).

Stessa cache, stessa testa condivisa del MIL. L'UNICA differenza rispetto al MIL
e l'aggregazione: qui i chunk NON sono un insieme non ordinato, sono i nodi di un
grafo, e si scambiano informazione lungo gli archi (message passing) PRIMA di
essere aggregati. Se la GNN batte il MIL, il merito e degli ARCHI (la struttura).

Nodi     : i chunk del documento, feature = embedding frozen dalla cache.
Archi    : costruiti da build_edges() — ABLATION centrale:
             - "sequential" : i->i+1 (l'ordine del testo)
             - "knn"        : k vicini piu simili nello spazio embedding (cosine)
             - "both"       : unione dei due
Encoder  : GATConv (attenzione sugli archi) x L layer.
Readout  : global attention pooling -> vettore-documento (analogo del bag MIL).
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, AttentionalAggregation

from src.models.heads import ClassificationHead


def build_edges(
    num_nodes: int,
    embeddings: Tensor,
    mode: str = "sequential",
    knn_k: int = 5,
) -> Tensor:
    """Costruisce edge_index (2, E) per un documento di `num_nodes` chunk.

    Ritorna archi NON diretti (ogni collegamento in entrambi i versi), come e
    convenzione per GAT. Con 0 o 1 nodo non ci sono archi (tensore vuoto).
    """
    if num_nodes <= 1:
        return torch.empty((2, 0), dtype=torch.long)

    edges: set[tuple[int, int]] = set()

    # archi sequenziali: catena i <-> i+1 (l'ordine del testo)
    if mode in ("sequential", "both"):
        for i in range(num_nodes - 1):
            edges.add((i, i + 1))
            edges.add((i + 1, i))

    # archi semantici: k vicini piu simili per cosine similarity
    if mode in ("knn", "both"):
        x = F.normalize(embeddings, p=2, dim=1)          # normalizza -> dot = cosine
        sim = x @ x.t()                                  # (N, N) matrice di similarita
        sim.fill_diagonal_(-1.0)                         # escludi l'auto-similarita
        k = min(knn_k, num_nodes - 1)
        topk = sim.topk(k, dim=1).indices                # (N, k) indici dei vicini
        for i in range(num_nodes):
            for j in topk[i].tolist():
                edges.add((i, j))
                edges.add((j, i))

    if not edges:
        return torch.empty((2, 0), dtype=torch.long)

    edge_index = torch.tensor(sorted(edges), dtype=torch.long).t().contiguous()
    return edge_index  # (2, E)


class GNNModel(nn.Module):
    """GAT multi-layer + global attention pooling + testa condivisa.

    forward processa UN documento alla volta (i suoi nodi e archi). Come il MIL,
    ritorna (logits, node_weights): i pesi del readout sono l'analogo interpretativo
    dei pesi di attenzione del MIL — quali nodi hanno pesato nel vettore-documento.
    """

    def __init__(
        self,
        in_dim: int,
        task: str,
        num_classes: int | None = None,
        hidden_dim: int = 128,
        num_layers: int = 2,
        heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.convs = nn.ModuleList()
        # primo layer: in_dim -> hidden_dim (con multi-head, concatenate)
        self.convs.append(GATConv(in_dim, hidden_dim, heads=heads, dropout=dropout))
        # layer intermedi: hidden*heads -> hidden
        for _ in range(num_layers - 2):
            self.convs.append(GATConv(hidden_dim * heads, hidden_dim, heads=heads, dropout=dropout))
        # ultimo layer: comprime a hidden_dim con una sola testa (media)
        if num_layers >= 2:
            self.convs.append(GATConv(hidden_dim * heads, hidden_dim, heads=1, concat=False, dropout=dropout))
            readout_dim = hidden_dim
        else:
            readout_dim = hidden_dim * heads

        # global attention pooling: impara un peso per nodo e aggrega (readout)
        gate = nn.Linear(readout_dim, 1)
        self.readout = AttentionalAggregation(gate_nn=gate)

        self.head = ClassificationHead(readout_dim, task, num_classes, dropout)

    def forward(self, embeddings: Tensor, edge_index: Tensor) -> tuple[Tensor, Tensor]:
        if embeddings.dim() != 2:
            raise ValueError(f"attesa matrice (N, in_dim), ricevuto {tuple(embeddings.shape)}.")
        if embeddings.shape[0] == 0:
            raise ValueError("grafo vuoto (0 nodi): il documento non ha prodotto embedding.")

        x = embeddings
        for conv in self.convs:
            x = F.elu(conv(x, edge_index))               # (N, hidden)

        # readout: batch tutto-zeri = un solo grafo
        batch = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)
        pooled = self.readout(x, batch)                  # (1, hidden)

        logits = self.head(pooled)                        # (1, out_features)
        node_weights = self.readout.gate_nn(x)            # (N, 1) pesi interpretativi
        return logits, node_weights