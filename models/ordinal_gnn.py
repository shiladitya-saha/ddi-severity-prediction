"""M3's Model 2: ordinal GNN without mechanism edge features.

The model reuses :class:`DrugPairEncoder` from the binary baseline and M4's
CORAL output/loss interface. For three ordered severity classes it returns two
cumulative logits with targets ``severity > 0`` and ``severity > 1``. Train
with ``CoralLoss`` and convert predictions with ``coral_predict``.
"""

from __future__ import annotations

import torch
from torch import nn
from torch_geometric.data import Data

from models.baseline_gnn import DrugPairEncoder
from models.ordinal_loss import CoralHead, coral_logits_to_probs, coral_predict


class OrdinalGNN(nn.Module):
    """GCN drug encoder and symmetric pairwise CORAL severity head."""

    def __init__(
        self,
        num_node_features: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        num_classes: int = 3,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        if num_classes < 2:
            raise ValueError("num_classes must be at least two")
        self.encoder = DrugPairEncoder(num_node_features, hidden_dim, num_layers, dropout)
        self.num_node_features = self.encoder.num_node_features
        self.hidden_dim = self.encoder.hidden_dim
        self.num_layers = self.encoder.num_layers
        self.num_classes = int(num_classes)
        self.dropout = self.encoder.dropout
        self.pair_projection = nn.Sequential(
            nn.Linear(3 * self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
        )
        self.ordinal_head = CoralHead(self.hidden_dim, num_classes=self.num_classes)

    @property
    def config(self) -> dict[str, int | float]:
        return {
            "num_node_features": self.num_node_features,
            "hidden_dim": self.hidden_dim,
            "num_layers": self.num_layers,
            "num_classes": self.num_classes,
            "dropout": self.dropout,
        }

    def forward(
        self,
        x: torch.Tensor | Data,
        edge_index: torch.Tensor | None = None,
        pair_index: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return ``[num_pairs, num_classes - 1]`` CORAL logits.

        Accepts either ``model(x, edge_index, pair_index)`` or
        ``model(data, pair_index)`` for a PyG ``Data`` object with ``x`` and
        ``edge_index`` attributes.
        """
        if not isinstance(x, torch.Tensor):
            data = x
            if pair_index is None:
                pair_index = edge_index
            x = data.x
            edge_index = data.edge_index
        if edge_index is None or pair_index is None:
            raise ValueError("edge_index and pair_index are required")
        pair_features = self.encoder(x, edge_index, pair_index)
        pair_hidden = self.pair_projection(pair_features)
        return self.ordinal_head(pair_hidden)

    def probabilities(self, logits: torch.Tensor) -> torch.Tensor:
        """Convert model logits to class probabilities using M4's CORAL helper."""
        if logits.ndim != 2 or logits.size(1) != self.num_classes - 1:
            raise ValueError(
                f"expected logits [N, {self.num_classes - 1}], got {tuple(logits.shape)}"
            )
        return coral_logits_to_probs(logits)

    def predict(self, logits: torch.Tensor) -> torch.Tensor:
        """Return integer severity predictions for the three-class default."""
        if logits.ndim != 2 or logits.size(1) != self.num_classes - 1:
            raise ValueError(
                f"expected logits [N, {self.num_classes - 1}], got {tuple(logits.shape)}"
            )
        return coral_predict(logits)
