"""
mechanism_gnn.py
Owner: M4 — see models/README.md for day/deadline and output contract.

Model 3: Mechanism-Aware Ordinal GNN (Day 2: architecture complete, not yet trained).

Pipeline:
    drug fingerprints -> GATv2 layers that read mechanism edge features
    -> symmetric drug-pair representation + the pair's mechanism features
    -> CoralHead -> K-1 ordinal severity logits

Inputs to forward():
    x           [num_drugs, num_node_features]  fingerprints from M2
    edge_index  [2, num_edges]  message-passing graph, both directions included
    edge_attr   [num_edges, num_mechanism_features]
    pair_index  [2, num_pairs]  drug pairs to score
    pair_attr   [num_pairs, num_mechanism_features], or None to score pairs with
                no known mechanism (treated as all zeros)

`edge_index` and `pair_index` are separate on purpose: message passing must use
training edges only, otherwise validation and test interactions leak into the
drug embeddings. Severity labels are never a model input.

Building edge_attr and pair_attr from M2's features/mechanism_edges.csv:
    table = load_mechanism_table("features/mechanism_edges.csv", drug_ids)
    edge_attr = mechanism_matrix(edge_index, table)

Explainability (M5): layers do not add self-loops, so each layer's attention has
exactly one row per input edge, in edge_index order, and lines up with
mechanism_matrix(edge_index, table). `model.encode(x, edge_index, edge_attr)` runs
the attention layers without needing any drug pairs.
"""

import csv

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv

from models.ordinal_loss import CoralHead

# Column order for edge_attr and pair_attr. These strings match what
# features/mechanism_extraction.py writes to mechanism_edges.csv.
MECHANISM_TAGS = (
    "cyp450_inhibition",
    "protein_binding_competition",
    "cns_depression",
    "qt_prolongation",
)
# M2 writes this for an edge that matched no mechanism. It adds no feature, so
# the edge keeps an all-zero row.
UNKNOWN_TAG = "unknown"


def load_mechanism_table(csv_path, drug_ids):
    """Read mechanism_edges.csv into {(low_node, high_node): [confidence per tag]}.

    Pairs are stored undirected, and repeated rows keep the highest confidence.
    Unrecognised tags, drug IDs and confidences raise an error instead of being
    skipped, because skipping would silently zero out features and weaken the
    model without any visible failure.
    """
    node_of = {str(drug_id): node for node, drug_id in enumerate(drug_ids)}
    column_of = {tag: column for column, tag in enumerate(MECHANISM_TAGS)}
    table = {}
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {"drug_a", "drug_b", "mechanism_tag", "confidence"}
        if not required.issubset(reader.fieldnames or ()):
            raise ValueError(f"{csv_path}: expected columns {sorted(required)}")
        for row in reader:
            where = f"{csv_path} line {reader.line_num}"
            tag = row["mechanism_tag"]
            if tag == UNKNOWN_TAG:
                continue
            if tag not in column_of:
                raise ValueError(f"{where}: unrecognised mechanism tag {tag!r}")
            for drug in (row["drug_a"], row["drug_b"]):
                if drug not in node_of:
                    raise ValueError(f"{where}: drug {drug!r} is not in the graph")
            confidence = float(row["confidence"])
            if not 0.0 <= confidence <= 1.0:
                raise ValueError(f"{where}: confidence {confidence} is outside [0, 1]")
            a, b = node_of[row["drug_a"]], node_of[row["drug_b"]]
            features = table.setdefault((min(a, b), max(a, b)), [0.0] * len(MECHANISM_TAGS))
            features[column_of[tag]] = max(features[column_of[tag]], confidence)
    return table


def mechanism_matrix(pair_index, table):
    """Return float features [num_pairs, num_tags], one row per column of pair_index.

    Pairs missing from the table get all zeros.
    """
    no_mechanism = [0.0] * len(MECHANISM_TAGS)
    low = pair_index.min(dim=0).values.tolist()
    high = pair_index.max(dim=0).values.tolist()
    rows = [table.get(pair, no_mechanism) for pair in zip(low, high)]
    return torch.tensor(rows, dtype=torch.float, device=pair_index.device).reshape(
        -1, len(MECHANISM_TAGS)
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
        self.config = {
            "num_node_features": num_node_features,
            "num_mechanism_features": num_mechanism_features,
            "hidden_dim": hidden_dim,
            "num_layers": num_layers,
            "heads": heads,
            "num_classes": num_classes,
            "dropout": dropout,
        }
        self.num_mechanism_features = num_mechanism_features
        self.dropout = dropout
        self.input_proj = nn.Linear(num_node_features, hidden_dim)
        self.convs = nn.ModuleList(
            GATv2Conv(
                hidden_dim,
                hidden_dim // heads,
                heads=heads,
                edge_dim=num_mechanism_features,
                dropout=dropout,
                # The residual in encode() already carries each drug's own
                # features, and without self-loops attention aligns 1:1 with edges.
                add_self_loops=False,
            )
            for _ in range(num_layers)
        )
        self.pair_mlp = nn.Sequential(
            nn.Linear(3 * hidden_dim + num_mechanism_features, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.head = CoralHead(hidden_dim, num_classes)

    def _check_attr(self, name, attr, expected_rows):
        expected = (expected_rows, self.num_mechanism_features)
        if attr.dim() != 2 or tuple(attr.shape) != expected:
            raise ValueError(f"{name} must have shape {expected}, got {tuple(attr.shape)}")

    @staticmethod
    def _check_index(name, index, num_drugs):
        if index.dim() != 2 or index.size(0) != 2:
            raise ValueError(f"{name} must have shape [2, N], got {tuple(index.shape)}")
        # Negative indices would otherwise silently wrap around to other drugs.
        if index.numel() and (index.min() < 0 or index.max() >= num_drugs):
            raise ValueError(f"{name} contains a drug index outside [0, {num_drugs})")

    def encode(self, x, edge_index, edge_attr):
        """Return drug embeddings and each layer's (edge_index, attention) pair."""
        self._check_index("edge_index", edge_index, x.size(0))
        self._check_attr("edge_attr", edge_attr, edge_index.size(1))
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
        self, x, edge_index, edge_attr, pair_index, pair_attr=None, return_attention=False
    ):
        """Return ordinal logits [num_pairs, num_classes - 1].

        With return_attention=True, also return one (edge_index, alpha) tuple per
        layer. alpha is [num_edges, heads], in the same order as edge_index.
        """
        self._check_index("pair_index", pair_index, x.size(0))
        if pair_attr is None:
            pair_attr = x.new_zeros(pair_index.size(1), self.num_mechanism_features)
        self._check_attr("pair_attr", pair_attr, pair_index.size(1))
        h, attention = self.encode(x, edge_index, edge_attr)
        h_a, h_b = h[pair_index[0]], h[pair_index[1]]
        # Severity of (a, b) equals severity of (b, a), so each term is symmetric.
        pair = torch.cat([h_a + h_b, (h_a - h_b).abs(), h_a * h_b, pair_attr], dim=1)
        logits = self.head(self.pair_mlp(pair))
        return (logits, attention) if return_attention else logits
