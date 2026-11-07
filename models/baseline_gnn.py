"""Binary GNN baseline for predicting whether a drug pair interacts.

The model uses node features and a message-passing graph, then scores requested
drug pairs with an order-invariant pair representation. The graph passed to
``forward`` is the message-passing context graph; callers are responsible for
removing validation/test target edges from it.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import GCNConv


class DrugPairEncoder(nn.Module):
    """Encode graph nodes and produce a symmetric representation per pair."""

    def __init__(
        self,
        num_node_features: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        if num_node_features <= 0:
            raise ValueError("num_node_features must be positive")
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        if num_layers < 1:
            raise ValueError("num_layers must be at least one")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")

        self.num_node_features = int(num_node_features)
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)
        self.dropout = float(dropout)
        self.convs = nn.ModuleList(
            GCNConv(
                self.num_node_features if layer == 0 else self.hidden_dim,
                self.hidden_dim,
                cached=False,
            )
            for layer in range(self.num_layers)
        )

    @property
    def config(self) -> dict[str, int | float]:
        return {
            "num_node_features": self.num_node_features,
            "hidden_dim": self.hidden_dim,
            "num_layers": self.num_layers,
            "dropout": self.dropout,
        }

    @staticmethod
    def validate_indices(name: str, indices: torch.Tensor, upper: int) -> None:
        if indices.dtype != torch.long:
            raise TypeError(f"{name} must have dtype torch.long")
        if indices.numel() and (indices.min().item() < 0 or indices.max().item() >= upper):
            raise ValueError(f"{name} contains an index outside [0, {upper})")

    def encode_nodes(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2:
            raise ValueError(f"x must have shape [num_nodes, num_features], got {tuple(x.shape)}")
        if x.size(1) != self.num_node_features:
            raise ValueError(
                f"x has {x.size(1)} features; model expects {self.num_node_features}"
            )
        if not x.is_floating_point():
            raise TypeError("x must be a floating-point tensor")
        if edge_index.ndim != 2 or edge_index.size(0) != 2:
            raise ValueError("edge_index must have shape [2, num_edges]")
        self.validate_indices("edge_index", edge_index, x.size(0))

        h = x
        for layer, conv in enumerate(self.convs):
            h = conv(h, edge_index)
            if layer < len(self.convs) - 1:
                h = F.relu(h)
                h = F.dropout(h, p=self.dropout, training=self.training)
        return h

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor, pair_index: torch.Tensor
    ) -> torch.Tensor:
        if pair_index.ndim != 2 or pair_index.size(0) != 2:
            raise ValueError("pair_index must have shape [2, num_pairs]")
        self.validate_indices("pair_index", pair_index, x.size(0))
        h = self.encode_nodes(x, edge_index)
        h_a = h[pair_index[0]]
        h_b = h[pair_index[1]]
        return torch.cat((h_a + h_b, torch.abs(h_a - h_b), h_a * h_b), dim=-1)


class BinaryBaselineGNN(nn.Module):
    """A small graph encoder and symmetric binary drug-pair classifier.

    Args:
        num_node_features: Width of each drug's input feature vector.
        hidden_dim: Width of the learned node embeddings.
        num_layers: Number of graph convolution layers (at least one).
        dropout: Dropout probability applied between graph layers and in the
            pair classifier.

    Input tensors use the standard PyG convention: ``x`` is ``[N, F]``,
    ``edge_index`` is ``[2, E]`` and ``pair_index`` is ``[2, P]``. The result
    is one raw binary logit per pair, shaped ``[P]``; pass it to
    ``BCEWithLogitsLoss`` (do not apply sigmoid first).
    """

    def __init__(
        self,
        num_node_features: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        if num_node_features <= 0:
            raise ValueError("num_node_features must be positive")
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        if num_layers < 1:
            raise ValueError("num_layers must be at least one")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")

        self.encoder = DrugPairEncoder(num_node_features, hidden_dim, num_layers, dropout)
        self.num_node_features = self.encoder.num_node_features
        self.hidden_dim = self.encoder.hidden_dim
        self.num_layers = self.encoder.num_layers
        self.dropout = self.encoder.dropout
        self.pair_classifier = nn.Sequential(
            nn.Linear(3 * self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, 1),
        )

    @property
    def config(self) -> dict[str, int | float]:
        """Return constructor settings suitable for checkpoint metadata."""
        return {
            "num_node_features": self.num_node_features,
            "hidden_dim": self.hidden_dim,
            "num_layers": self.num_layers,
            "dropout": self.dropout,
        }

    @staticmethod
    def encode(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Encode all nodes in a PyG message-passing graph."""
        return self.encoder.encode_nodes(x, edge_index)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor | None = None,
        pair_index: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return one interaction logit per pair.

        Both explicit tensors (``model(x, edge_index, pair_index)``) and a PyG
        ``Data`` object (``model(data, pair_index)``) are supported. In the
        latter form, ``data.x`` and ``data.edge_index`` are used.
        """
        if not isinstance(x, torch.Tensor):
            data = x
            if pair_index is None:
                pair_index = edge_index
            x = getattr(data, "x", None)
            edge_index = getattr(data, "edge_index", None)
            if x is None or edge_index is None:
                raise ValueError("PyG Data input must provide both x and edge_index")
        if edge_index is None:
            raise ValueError("edge_index is required")
        if pair_index is None:
            raise ValueError("pair_index is required")
        pair_features = self.encoder(x, edge_index, pair_index)
        return self.pair_classifier(pair_features).squeeze(-1)


# A concise alias for callers that use the project terminology.
BinaryBaseline = BinaryBaselineGNN
