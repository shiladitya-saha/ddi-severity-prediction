"""M3 Model 2: leakage-safe ordinal GNN for three-class DDI severity.

The module consumes M1's ``graph.pt`` and ``nodes.csv``, M2's feature map, and
the canonical split artifact produced by M5. The message-passing graph is built
from training pairs only. Validation, test, and reserved severe-benchmark
pairs are scored as pair queries and never added to message passing.

Run from the repository root after the project artifacts and M5 split exist::

    python -m models.ordinal_gnn --epochs 20 --seed 42

This writes a resumable checkpoint, ``results/model2_results.json``, and an
M5-compatible ``results/preds_ordinal_gnn.npz``. This model does not consume
edge-description or mechanism pair attributes.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from eval.benchmark_severe_subset import benchmark_severe
from eval.metrics import evaluate
from eval.split import SplitSet, load_splits, verify_splits
from models.baseline_gnn import DrugPairEncoder
from models.ordinal_loss import CoralHead, CoralLoss, coral_logits_to_probs, coral_predict
from models.train import (
    _graph_field,
    load_binary_inputs,
    resolve_device,
    set_seed,
)


@dataclass
class OrdinalEdgeSplit:
    """Canonical M5 pair array and its train/validation/test partitions."""

    pairs: torch.Tensor
    labels: torch.Tensor
    train_indices: torch.Tensor
    validation_indices: torch.Tensor
    test_indices: torch.Tensor
    severe_indices: torch.Tensor
    message_edge_index: torch.Tensor

    @property
    def train_pairs(self) -> torch.Tensor:
        return self.pairs[:, self.train_indices]

    @property
    def train_labels(self) -> torch.Tensor:
        return self.labels[self.train_indices]

    @property
    def validation_pairs(self) -> torch.Tensor:
        return self.pairs[:, self.validation_indices]

    @property
    def validation_labels(self) -> torch.Tensor:
        return self.labels[self.validation_indices]

    @property
    def test_pairs(self) -> torch.Tensor:
        return self.pairs[:, self.test_indices]

    @property
    def test_labels(self) -> torch.Tensor:
        return self.labels[self.test_indices]

    @property
    def severe_pairs(self) -> torch.Tensor:
        return self.pairs[:, self.severe_indices]

    @property
    def severe_labels(self) -> torch.Tensor:
        return self.labels[self.severe_indices]


class OrdinalGNN(nn.Module):
    """Symmetric GCN pair encoder with M4's rank-consistent CORAL head."""

    def __init__(
        self,
        num_node_features: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        num_classes: int = 3,
    ) -> None:
        super().__init__()
        if num_classes != 3:
            raise ValueError("M3 severity prediction requires exactly three classes")
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")

        self.encoder = DrugPairEncoder(
            num_node_features=num_node_features,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
        )
        self.pair_projection = nn.Sequential(
            nn.Linear(3 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.head = CoralHead(hidden_dim, num_classes=num_classes)
        self._config = {
            "num_node_features": int(num_node_features),
            "hidden_dim": int(hidden_dim),
            "num_layers": int(num_layers),
            "dropout": float(dropout),
            "num_classes": int(num_classes),
        }

    @property
    def config(self) -> dict[str, int | float]:
        return dict(self._config)

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor, pair_index: torch.Tensor
    ) -> torch.Tensor:
        """Return cumulative severity logits ``[num_pairs, 2]``."""
        if x.ndim != 2 or x.size(1) != self.config["num_node_features"]:
            raise ValueError(
                "x must have shape [num_nodes, "
                f"{self.config['num_node_features']}], got {tuple(x.shape)}"
            )
        if pair_index.ndim != 2 or pair_index.size(0) != 2:
            raise ValueError("pair_index must have shape [2, num_pairs]")
        if pair_index.dtype != torch.long:
            raise TypeError("pair_index must have dtype torch.long")
        if pair_index.numel() and (
            pair_index.min().item() < 0 or pair_index.max().item() >= x.size(0)
        ):
            raise ValueError("pair_index contains an out-of-range node index")
        if pair_index.size(1) and torch.any(pair_index[0] == pair_index[1]):
            raise ValueError("self-pairs are not valid DDI examples")

        pair_features = self.encoder(x, edge_index, pair_index)
        return self.head(self.pair_projection(pair_features))

    @staticmethod
    def probabilities(logits: torch.Tensor) -> torch.Tensor:
        """Convert two CORAL logits to three class probabilities."""
        if logits.ndim != 2 or logits.size(1) != 2:
            raise ValueError("three-class CORAL logits must have shape [N, 2]")
        probabilities = coral_logits_to_probs(logits)
        if not torch.isfinite(probabilities).all():
            raise FloatingPointError("CORAL conversion produced non-finite probabilities")
        if torch.any(probabilities < -1e-6):
            raise ValueError("CORAL probabilities must be non-negative")
        return probabilities.clamp_min(0.0)

    @staticmethod
    def predict(logits: torch.Tensor) -> torch.Tensor:
        """Decode cumulative logits as labels ``0=minor, 1=moderate, 2=major``."""
        if logits.ndim != 2 or logits.size(1) != 2:
            raise ValueError("three-class CORAL logits must have shape [N, 2]")
        return coral_predict(logits)


def _canonical_pair(a: int, b: int) -> tuple[int, int]:
    if a == b:
        raise ValueError("self-pairs are not valid DDI examples")
    return (a, b) if a < b else (b, a)


def _severity_labels(labels: Any, name: str) -> torch.Tensor:
    raw = torch.as_tensor(labels, device="cpu")
    if raw.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    integer_types = {
        torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8,
    }
    if raw.dtype not in integer_types:
        if not raw.is_floating_point() or not torch.isfinite(raw).all():
            raise TypeError(f"{name} must contain integer severity labels")
        if not torch.equal(raw, raw.round()):
            raise ValueError(f"{name} must contain integer severity labels")
    result = raw.to(dtype=torch.long)
    if result.numel() and (result.min().item() < 0 or result.max().item() > 2):
        raise ValueError(f"{name} must use 0=minor, 1=moderate, 2=major")
    return result


def build_ordinal_edge_split(
    canonical_pairs: torch.Tensor,
    labels: torch.Tensor,
    splits: SplitSet,
    num_nodes: int,
) -> OrdinalEdgeSplit:
    """Validate M5 indices and construct training-only bidirectional context."""
    raw_pairs = torch.as_tensor(canonical_pairs, device="cpu")
    integer_types = {torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8}
    if raw_pairs.dtype not in integer_types:
        raise TypeError("canonical_pairs must contain integer node indices")
    pairs = raw_pairs.to(dtype=torch.long)
    labels = _severity_labels(labels, "severity labels")
    if pairs.ndim != 2 or pairs.size(0) != 2:
        raise ValueError("canonical_pairs must have shape [2, num_edges]")
    if pairs.size(1) != labels.numel():
        raise ValueError("canonical pair count and label count must match")
    if pairs.numel() and (pairs.min().item() < 0 or pairs.max().item() >= num_nodes):
        raise ValueError("canonical_pairs contains an out-of-range node index")
    if pairs.size(1) and torch.any(pairs[0] >= pairs[1]):
        raise ValueError("canonical_pairs must be unique unordered pairs (low < high)")
    if len({tuple(pair) for pair in pairs.t().tolist()}) != pairs.size(1):
        raise ValueError("canonical_pairs contains duplicate pairs")

    expected = np.asarray(splits.meta.get("canonical_pairs", []), dtype=np.int64)
    if expected.shape != (pairs.size(1), 2):
        raise ValueError(
            "M5 split artifact must include canonical_pairs matching the deduplicated edge array"
        )
    if not np.array_equal(expected, pairs.t().numpy()):
        raise ValueError("M5 canonical pair order does not match the loaded graph labels")

    problems = verify_splits(
        splits,
        labels=labels.numpy(),
        edge_index=pairs.t().numpy(),
        strict=False,
    )
    if problems:
        raise ValueError("M5 split failed validation: " + "; ".join(problems))

    def indices(name: str, required: bool = True) -> torch.Tensor:
        if name not in splits:
            if required:
                raise ValueError(f"M5 split is missing the {name!r} partition")
            return torch.empty(0, dtype=torch.long)
        idx = torch.as_tensor(splits[name], dtype=torch.long, device="cpu").reshape(-1)
        if idx.numel() and (idx.min().item() < 0 or idx.max().item() >= pairs.size(1)):
            raise ValueError(f"M5 {name} indices are outside the canonical edge array")
        return idx

    train_idx = indices("train")
    val_idx = indices("val")
    test_idx = indices("test")
    severe_idx = indices("severe_benchmark", required=False)
    message_pairs = pairs[:, train_idx]
    message_edges = torch.cat((message_pairs, message_pairs.flip(0)), dim=1).contiguous()

    context = {_canonical_pair(a, b) for a, b in message_edges.t().tolist()}
    held_out = set()
    for held in (val_idx, test_idx, severe_idx):
        held_out.update(tuple(pair) for pair in pairs[:, held].t().tolist())
    if context & held_out:
        raise ValueError("held-out validation/test/severe edges entered message passing")

    return OrdinalEdgeSplit(
        pairs=pairs,
        labels=labels,
        train_indices=train_idx,
        validation_indices=val_idx,
        test_indices=test_idx,
        severe_indices=severe_idx,
        message_edge_index=message_edges,
    )


def load_ordinal_inputs(
    graph_path: str | Path = "data/processed/graph.pt",
    features_path: str | Path = "features/full_features.pt",
    split_path: str | Path = "eval/splits/ddi_v1.json",
    *,
    nodes_path: str | Path | None = None,
) -> tuple[list[str], torch.Tensor, OrdinalEdgeSplit, SplitSet]:
    """Load graph/features and align severity labels to the M5 canonical split."""
    drug_ids, x, _ = load_binary_inputs(graph_path, features_path, nodes_path=nodes_path)
    graph = torch.load(graph_path, map_location="cpu", weights_only=False)
    edge_index = _graph_field(graph, "edge_index")
    severity = _graph_field(graph, "edge_severity")
    if severity is None:
        severity = _graph_field(graph, "severity_label")
    if edge_index is None or severity is None:
        raise ValueError("M1 graph must contain edge_index and edge_severity labels")
    edge_index = torch.as_tensor(edge_index, dtype=torch.long, device="cpu")
    raw_severity = torch.as_tensor(severity, device="cpu")
    if edge_index.ndim != 2 or edge_index.size(0) != 2:
        raise ValueError("M1 graph edge_index must have shape [2, num_edges]")
    if raw_severity.ndim != 1:
        raise ValueError("M1 graph edge_severity must be one-dimensional")
    severity = _severity_labels(raw_severity, "M1 graph severity labels")
    if edge_index.size(1) != severity.numel():
        raise ValueError("M1 graph must have one severity label per edge")

    labels_by_pair: dict[tuple[int, int], int] = {}
    for (a, b), label in zip(edge_index.t().tolist(), severity.tolist()):
        key = _canonical_pair(a, b)
        previous = labels_by_pair.setdefault(key, int(label))
        if previous != int(label):
            raise ValueError(f"reversed/duplicate edges disagree on severity for {key}")

    splits = load_splits(split_path)
    meta_pairs = np.asarray(splits.meta.get("canonical_pairs", []), dtype=np.int64)
    if meta_pairs.ndim != 2 or meta_pairs.shape[1:] != (2,):
        raise ValueError("M5 split metadata has no valid canonical_pairs array")
    labels = []
    missing = []
    for raw_pair in meta_pairs.tolist():
        pair = _canonical_pair(int(raw_pair[0]), int(raw_pair[1]))
        if pair not in labels_by_pair:
            missing.append(pair)
        else:
            labels.append(labels_by_pair[pair])
    if missing:
        raise ValueError(f"M5 split contains {len(missing)} graph pairs absent from graph.pt")
    pairs = torch.as_tensor(meta_pairs.T.copy(), dtype=torch.long)
    edge_split = build_ordinal_edge_split(
        pairs, torch.tensor(labels, dtype=torch.long), splits, num_nodes=len(drug_ids)
    )
    return drug_ids, x, edge_split, splits


def _run_batches(
    model: OrdinalGNN,
    x: torch.Tensor,
    edge_index: torch.Tensor,
    pairs: torch.Tensor,
    labels: torch.Tensor,
    criterion: nn.Module,
    *,
    device: torch.device,
    batch_size: int,
    optimizer: torch.optim.Optimizer | None = None,
    seed: int = 42,
) -> tuple[float, torch.Tensor]:
    if pairs.ndim != 2 or pairs.size(0) != 2 or pairs.size(1) != labels.numel():
        raise ValueError("pairs must be [2, N] and match the number of labels")
    if pairs.size(1) == 0:
        raise ValueError("cannot process an empty ordinal split")
    training = optimizer is not None
    model.train(training)
    dataset = TensorDataset(pairs.t().contiguous(), labels.reshape(-1).long())
    generator = torch.Generator(device="cpu").manual_seed(seed)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=training, generator=generator)
    total_loss = 0.0
    logits_all = []
    for batch_pairs, batch_labels in loader:
        batch_pairs = batch_pairs.t().contiguous().to(device)
        batch_labels = batch_labels.to(device)
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            logits = model(x, edge_index, batch_pairs)
            if logits.shape != (batch_labels.numel(), 2):
                raise ValueError("ordinal model must produce two logits per example")
            if not torch.isfinite(logits).all():
                raise FloatingPointError("model produced non-finite ordinal logits")
            loss = criterion(logits, batch_labels)
            if not torch.isfinite(loss):
                raise FloatingPointError("model produced a non-finite CORAL loss")
            if optimizer is not None:
                loss.backward()
                optimizer.step()
        total_loss += float(loss.detach().item()) * batch_labels.numel()
        logits_all.append(logits.detach().cpu())
    return total_loss / len(dataset), torch.cat(logits_all, dim=0)


def save_checkpoint(
    path: str | Path,
    model: OrdinalGNN,
    optimizer: torch.optim.Optimizer,
    *,
    epoch: int,
    seed: int,
    validation_loss: float,
    metadata: Mapping[str, Any] | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_config": model.config,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": int(epoch),
            "seed": int(seed),
            "validation_loss": float(validation_loss),
            "metadata": dict(metadata or {}),
        },
        path,
    )


def load_checkpoint(
    path: str | Path,
    model: OrdinalGNN,
    optimizer: torch.optim.Optimizer | None = None,
    *,
    device: str | torch.device = "cpu",
) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if checkpoint.get("model_config") != model.config:
        raise ValueError("checkpoint model configuration does not match the model")
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None:
        if "optimizer_state_dict" not in checkpoint:
            raise ValueError("checkpoint has no optimizer state to resume")
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint


def train_ordinal_model(
    model: OrdinalGNN,
    x: torch.Tensor,
    message_edge_index: torch.Tensor,
    train_pairs: torch.Tensor,
    train_labels: torch.Tensor,
    validation_pairs: torch.Tensor,
    validation_labels: torch.Tensor,
    *,
    class_weights: torch.Tensor | None = None,
    epochs: int = 20,
    batch_size: int = 256,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    seed: int = 42,
    device: str | torch.device | None = None,
    checkpoint_path: str | Path | None = None,
    resume_from: str | Path | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Train with M4's CORAL loss and retain the lowest-validation-loss state."""
    if epochs < 1 or batch_size < 1:
        raise ValueError("epochs and batch_size must be positive")
    if learning_rate <= 0 or weight_decay < 0:
        raise ValueError("learning_rate must be positive and weight_decay non-negative")
    train_labels = _severity_labels(train_labels, "training labels")
    validation_labels = _severity_labels(validation_labels, "validation labels")
    if train_pairs.size(1) != train_labels.numel():
        raise ValueError("train pair/label counts differ")
    if validation_pairs.size(1) != validation_labels.numel():
        raise ValueError("validation pair/label counts differ")
    set_seed(seed)
    resolved = resolve_device(device)
    model = model.to(resolved)
    x = x.to(resolved)
    message_edge_index = message_edge_index.to(resolved)
    criterion = CoralLoss(
        num_classes=3,
        class_weights=None if class_weights is None else class_weights.to(resolved),
    ).to(resolved)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )

    start_epoch = 1
    best_validation_loss = float("inf")
    best_epoch = 0
    history: list[dict[str, Any]] = []
    if resume_from is not None:
        resumed = load_checkpoint(resume_from, model, optimizer, device=resolved)
        start_epoch = int(resumed["epoch"]) + 1
        best_epoch = int(resumed["epoch"])
        best_validation_loss = float(resumed["validation_loss"])
        history = list(resumed.get("metadata", {}).get("history", []))

    for epoch in range(start_epoch, epochs + 1):
        train_loss, _ = _run_batches(
            model,
            x,
            message_edge_index,
            train_pairs,
            train_labels,
            criterion,
            device=resolved,
            batch_size=batch_size,
            optimizer=optimizer,
            seed=seed + epoch,
        )
        val_loss, val_logits = _run_batches(
            model,
            x,
            message_edge_index,
            validation_pairs,
            validation_labels,
            criterion,
            device=resolved,
            batch_size=batch_size,
            seed=seed,
        )
        val_pred = coral_predict(val_logits).numpy()
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "validation_loss": val_loss,
                "validation_accuracy": float(
                    (val_pred == validation_labels.cpu().numpy()).mean()
                ),
            }
        )
        if val_loss < best_validation_loss:
            best_validation_loss = val_loss
            best_epoch = epoch
            if checkpoint_path is not None:
                save_checkpoint(
                    checkpoint_path,
                    model,
                    optimizer,
                    epoch=epoch,
                    seed=seed,
                    validation_loss=val_loss,
                    metadata={**dict(metadata or {}), "history": history},
                )

    if best_epoch == 0:
        raise ValueError("training produced no validation checkpoint")
    if checkpoint_path is not None:
        load_checkpoint(checkpoint_path, model, device=resolved)
    return {
        "history": history,
        "best_epoch": best_epoch,
        "best_validation_loss": best_validation_loss,
        "device": str(resolved),
        "seed": seed,
        "model_config": model.config,
        "optimizer": "Adam",
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "checkpoint": None if checkpoint_path is None else str(checkpoint_path),
    }


@torch.no_grad()
def predict_ordinal(
    model: OrdinalGNN,
    x: torch.Tensor,
    message_edge_index: torch.Tensor,
    pair_index: torch.Tensor,
    *,
    batch_size: int = 1024,
    device: str | torch.device | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return logits [N,2], probabilities [N,3], and labels [N]."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    resolved = resolve_device(device)
    model = model.to(resolved)
    model.eval()
    x = x.to(resolved)
    message_edge_index = message_edge_index.to(resolved)
    pairs = torch.as_tensor(pair_index, dtype=torch.long, device="cpu")
    if pairs.ndim != 2 or pairs.size(0) != 2:
        raise ValueError("pair_index must have shape [2, num_pairs]")
    logits = []
    for start in range(0, pairs.size(1), batch_size):
        batch = pairs[:, start : start + batch_size].to(resolved)
        batch_logits = model(x, message_edge_index, batch)
        logits.append(batch_logits.cpu())
    raw = torch.cat(logits, dim=0) if logits else torch.empty((0, 2))
    probabilities = OrdinalGNN.probabilities(raw)
    predictions = OrdinalGNN.predict(raw)
    return raw, probabilities, predictions


def run_ordinal_experiment(
    graph_path: str | Path = "data/processed/graph.pt",
    features_path: str | Path = "features/full_features.pt",
    split_path: str | Path = "eval/splits/ddi_v1.json",
    *,
    class_weights_path: str | Path = "data/processed/class_weights.json",
    epochs: int = 20,
    batch_size: int = 256,
    seed: int = 42,
    device: str | None = None,
    checkpoint_path: str | Path = "models/checkpoints/model2_ordinal_gnn.pt",
    results_path: str | Path = "results/model2_results.json",
    predictions_path: str | Path = "results/preds_ordinal_gnn.npz",
    dataset_kind: str = "project_artifacts",
) -> dict[str, Any]:
    """Run the real-artifact pipeline and export M5-compatible predictions."""
    drug_ids, x, split, splits = load_ordinal_inputs(
        graph_path, features_path, split_path
    )
    weights = None
    weights_path = Path(class_weights_path)
    if weights_path.is_file():
        raw_weights = json.loads(weights_path.read_text(encoding="utf-8"))
        weight_values = raw_weights.get("inverse_frequency")
        if weight_values is not None:
            weights = torch.tensor(weight_values, dtype=torch.float32)
            if weights.shape != (3,) or not torch.isfinite(weights).all() or torch.any(weights <= 0):
                raise ValueError("inverse_frequency class weights must be three finite positive values")

    set_seed(seed)
    model = OrdinalGNN(num_node_features=x.size(1))
    outcome = train_ordinal_model(
        model,
        x,
        split.message_edge_index,
        split.train_pairs,
        split.train_labels,
        split.validation_pairs,
        split.validation_labels,
        class_weights=weights,
        epochs=epochs,
        batch_size=batch_size,
        seed=seed,
        device=device,
        checkpoint_path=checkpoint_path,
        metadata={"split_path": str(split_path), "graph_path": str(graph_path)},
    )
    load_checkpoint(checkpoint_path, model, device=outcome["device"])

    held_indices = np.concatenate(
        [split.validation_indices.numpy(), split.test_indices.numpy(), split.severe_indices.numpy()]
    )
    held_pairs = split.pairs[:, torch.as_tensor(held_indices, dtype=torch.long)]
    held_logits, held_proba, held_pred = predict_ordinal(
        model,
        x,
        split.message_edge_index,
        held_pairs,
        device=outcome["device"],
    )
    labels_np = split.labels.numpy()
    pred_np = held_pred.numpy()
    proba_np = held_proba.numpy()
    result_metrics: dict[str, Any] = {}
    cursor = 0
    for name, indices in (
        ("val", split.validation_indices),
        ("test", split.test_indices),
        ("severe_benchmark", split.severe_indices),
    ):
        count = indices.numel()
        if count == 0:
            continue
        segment = slice(cursor, cursor + count)
        result = evaluate(
            labels_np[indices.numpy()],
            pred_np[segment],
            y_proba=proba_np[segment],
            model_name="ordinal_gnn",
            split=name,
        )
        if name == "severe_benchmark":
            full_predictions = np.zeros_like(labels_np)
            full_probabilities = np.zeros((labels_np.size, 3), dtype=float)
            full_predictions[indices.numpy()] = pred_np[segment]
            full_probabilities[indices.numpy()] = proba_np[segment]
            result = benchmark_severe(
                labels_np,
                full_predictions,
                splits,
                y_proba=full_probabilities,
                model_name="ordinal_gnn",
            )
        result_metrics[name] = result.to_dict()
        cursor += count

    predictions_file = Path(predictions_path)
    predictions_file.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        predictions_file,
        edge_idx=held_indices.astype(np.int64),
        y_pred=pred_np.astype(np.int64),
        y_proba=proba_np.astype(np.float32),
        y_true=labels_np[held_indices].astype(np.int64),
    )
    result_record = {
        "experiment": "model2_ordinal_gnn",
        "dataset_kind": dataset_kind,
        "data_source": str(graph_path),
        "feature_source": str(features_path),
        "split_source": str(split_path),
        "num_drugs": len(drug_ids),
        "num_node_features": int(x.size(1)),
        "num_canonical_pairs": int(split.pairs.size(1)),
        "num_message_passing_edges": int(split.message_edge_index.size(1)),
        "held_out_edges_in_message_graph": 0,
        "label_encoding": {"minor": 0, "moderate": 1, "major": 2},
        "metrics": result_metrics,
        "predictions": str(predictions_file),
        "checkpoint": str(checkpoint_path),
        **outcome,
    }
    results_file = Path(results_path)
    results_file.parent.mkdir(parents=True, exist_ok=True)
    results_file.write_text(json.dumps(result_record, indent=2) + "\n", encoding="utf-8")
    return result_record


def run_synthetic_smoke_test(
    *, epochs: int = 4, seed: int = 17, device: str = "cpu"
) -> dict[str, Any]:
    """Small deterministic fixture; results are synthetic and not project metrics."""
    from eval.split import make_splits

    set_seed(seed)
    n = 18
    x = torch.randn(n, 12, generator=torch.Generator().manual_seed(seed))
    pairs = torch.tensor(
        [(a, b) for a in range(n) for b in range(a + 1, n) if (a * 7 + b * 3) % 5 != 0],
        dtype=torch.long,
    ).t().contiguous()
    labels = torch.tensor(
        [(int(a) + 2 * int(b)) % 3 for a, b in pairs.t().tolist()], dtype=torch.long
    )
    splits = make_splits(
        pairs.t().numpy(), labels.numpy(), seed=seed, severe_benchmark_frac=0.15
    )
    edge = build_ordinal_edge_split(pairs, labels, splits, n)
    model = OrdinalGNN(12, hidden_dim=16, num_layers=2, dropout=0.0)
    with __import__("tempfile").TemporaryDirectory() as directory:
        checkpoint = Path(directory) / "synthetic.pt"
        outcome = train_ordinal_model(
            model,
            x,
            edge.message_edge_index,
            edge.train_pairs,
            edge.train_labels,
            edge.validation_pairs,
            edge.validation_labels,
            epochs=epochs,
            batch_size=32,
            seed=seed,
            device=device,
            checkpoint_path=checkpoint,
        )
        load_checkpoint(checkpoint, model, device=device)
        test_logits, test_proba, test_pred = predict_ordinal(
            model, x, edge.message_edge_index, edge.test_pairs, device=device
        )
    return {
        "dataset_kind": "synthetic_fixture",
        "seed": seed,
        "num_nodes": n,
        "num_pairs": int(pairs.size(1)),
        "test_logits_shape": list(test_logits.shape),
        "test_probability_shape": list(test_proba.shape),
        "test_prediction_shape": list(test_pred.shape),
        "test_probabilities_finite": bool(torch.isfinite(test_proba).all()),
        "test_loss": float(
            CoralLoss()(test_logits, edge.test_labels).item()
        ),
        **outcome,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph", default="data/processed/graph.pt")
    parser.add_argument("--features", default="features/full_features.pt")
    parser.add_argument("--splits", default="eval/splits/ddi_v1.json")
    parser.add_argument("--class-weights", default="data/processed/class_weights.json")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    parser.add_argument("--checkpoint", default="models/checkpoints/model2_ordinal_gnn.pt")
    parser.add_argument("--results", default="results/model2_results.json")
    parser.add_argument("--predictions", default="results/preds_ordinal_gnn.npz")
    parser.add_argument("--synthetic", action="store_true")
    args = parser.parse_args(argv)
    if args.synthetic:
        result = run_synthetic_smoke_test(epochs=args.epochs, seed=args.seed, device=args.device or "cpu")
    else:
        result = run_ordinal_experiment(
            args.graph,
            args.features,
            args.splits,
            class_weights_path=args.class_weights,
            epochs=args.epochs,
            batch_size=args.batch_size,
            seed=args.seed,
            device=args.device,
            checkpoint_path=args.checkpoint,
            results_path=args.results,
            predictions_path=args.predictions,
        )
    print(json.dumps({key: result[key] for key in ("dataset_kind", "best_epoch", "best_validation_loss", "checkpoint") if key in result}, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
