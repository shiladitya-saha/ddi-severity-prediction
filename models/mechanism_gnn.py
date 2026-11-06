"""
mechanism_gnn.py
Owner: M4 — see models/README.md for day/deadline and output contract.

Model 3: Mechanism-Aware Ordinal GNN (Day 1 skeleton, not yet trained).

Pipeline:
    drug fingerprints -> GATv2 layers that read mechanism edge features
    -> symmetric drug-pair representation + the pair's mechanism features
    -> CoralHead -> K-1 ordinal severity logits

Inputs to forward():
    x           [num_drugs, num_node_features]  fingerprints from M2
    edge_index  [2, num_edges]  message-passing graph, both directions included
    edge_attr   [num_edges, num_mechanism_features]
    pair_index  [2, num_pairs]  drug pairs to score
    pair_attr   [num_pairs, num_mechanism_features]

`edge_index` and `pair_index` are separate on purpose: message passing must use
training edges only, otherwise validation and test interactions leak into the
drug embeddings. Severity labels are never a model input.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv

from models.ordinal_loss import CoralHead

# Column order for edge_attr and pair_attr. Provisional until M2 fixes the tag
# strings written to features/mechanism_edges.csv.
MECHANISM_TAGS = (
    "cyp450_inhibition",
    "protein_binding_competition",
    "cns_depression",
    "qt_prolongation",
)


class MechanismOrdinalGNN(nn.Module):
    def __init__(
        self,
        num_node_features,
        num_mechanism_features=len(MECHANISM_TAGS),
        hidden_dim=128,
        num_layers=2,
        heads=4,
        num_classes=3,
        dropout=0.2,
    ):
        super().__init__()
        if hidden_dim % heads != 0:
            raise ValueError("hidden_dim must be divisible by heads")
        self.dropout = dropout
        self.input_proj = nn.Linear(num_node_features, hidden_dim)
        self.convs = nn.ModuleList(
            GATv2Conv(
                hidden_dim,
                hidden_dim // heads,
                heads=heads,
                edge_dim=num_mechanism_features,
                dropout=dropout,
            )
            for _ in range(num_layers)
        )
        self.pair_mlp = nn.Sequential(
            nn.Linear(3 * hidden_dim + num_mechanism_features, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.head = CoralHead(hidden_dim, num_classes)

    def encode(self, x, edge_index, edge_attr):
        """Return drug embeddings and each layer's (edge_index, attention) pair."""
        h = F.relu(self.input_proj(x))
        attention = []
        for conv in self.convs:
            out, layer_attention = conv(
                h, edge_index, edge_attr, return_attention_weights=True
            )
            # Residual connection keeps drugs with few or no edges informative,
            # which matters for the zero-shot new-drug experiments.
            h = h + F.dropout(F.relu(out), p=self.dropout, training=self.training)
            attention.append(layer_attention)
        return h, attention

    def forward(
        self, x, edge_index, edge_attr, pair_index, pair_attr, return_attention=False
    ):
        """Return ordinal logits [num_pairs, num_classes - 1].

        With return_attention=True, also return one (edge_index, alpha) tuple per
        layer. alpha is [num_edges + num_self_loops, heads], and the returned
        edge_index includes the self-loops GATv2Conv adds.
        """
        h, attention = self.encode(x, edge_index, edge_attr)
        h_a, h_b = h[pair_index[0]], h[pair_index[1]]
        # Severity of (a, b) equals severity of (b, a), so each term is symmetric.
        pair = torch.cat([h_a + h_b, (h_a - h_b).abs(), h_a * h_b, pair_attr], dim=1)
        logits = self.head(self.pair_mlp(pair))
        return (logits, attention) if return_attention else logits
