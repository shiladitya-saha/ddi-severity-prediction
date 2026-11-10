"""Training utilities and CLI for Model 1, the binary interaction baseline.

Example after M1 provides ``data/processed/graph.pt`` and M2 provides
``features/full_features.pt``::

    python -m models.train --splits eval/splits/ddi_v1.json --epochs 20 --seed 17

The CLI uses M5's canonical positive-edge folds and samples unique unordered
non-edges within those same folds. Only training positives enter the
message-passing graph. Unobserved true interactions may still be sampled as
negatives. `run_dry_run()` remains available for explicit standalone fixture
checks; Day 3 result files use `run_m5_detection_experiment()`.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from models.baseline_gnn import BinaryBaselineGNN
from eval.metrics import evaluate_detection
from eval.split import SplitSet, load_splits, save_splits


@dataclass
class BinaryEdgeSplit:
    """Pair-level binary split and the training-only message-passing graph."""

    message_edge_index: torch.Tensor
    train_pairs: torch.Tensor
    train_labels: torch.Tensor
    validation_pairs: torch.Tensor
    validation_labels: torch.Tensor
    test_pairs: torch.Tensor
    test_labels: torch.Tensor
    context_positive_pairs: torch.Tensor


@dataclass
class M5DetectionSplit:
    """Binary examples assigned to M5's positive-edge partitions.

    Negative candidates are sampled once, then allocated within the same
    train/validation/test/severe partitions as the canonical M5 positive
    pairs. ``detection_splits`` indexes this combined candidate-pair array.
    """

    message_edge_index: torch.Tensor
    pairs: torch.Tensor
    labels: torch.Tensor
    train_indices: torch.Tensor
    validation_indices: torch.Tensor
    test_indices: torch.Tensor
    severe_indices: torch.Tensor

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


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, CPU Torch and CUDA Torch RNGs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def resolve_device(device: str | torch.device | None = None) -> torch.device:
    """Resolve an explicit device or select CUDA when available."""
    if device is not None:
        resolved = torch.device(device)
        if resolved.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        return resolved
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def canonicalize_pairs(pairs: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """Return unique ``[2, P]`` unordered pairs; reject self/out-of-range pairs."""
    pairs = torch.as_tensor(pairs, dtype=torch.long, device="cpu")
    if pairs.ndim != 2 or pairs.size(0) != 2:
        raise ValueError("pairs must have shape [2, num_pairs]")
    if pairs.numel() and (pairs.min().item() < 0 or pairs.max().item() >= num_nodes):
        raise ValueError(f"pairs contain node indices outside [0, {num_nodes})")
    if pairs.size(1) and torch.any(pairs[0] == pairs[1]):
        raise ValueError("self-pairs are not valid DDI examples")
    ordered = torch.sort(pairs, dim=0).values.t().contiguous()
    ordered = torch.unique(ordered, dim=0, sorted=True)
    return ordered.t().contiguous()


def _sample_negative_pairs(
    num_nodes: int,
    positive_pairs: torch.Tensor,
    count: int,
    generator: torch.Generator,
) -> torch.Tensor:
    """Uniformly rejection-sample unique unordered pairs absent from positives."""
    if count == 0:
        return torch.empty((2, 0), dtype=torch.long)
    possible = num_nodes * (num_nodes - 1) // 2
    positive = {tuple(pair) for pair in positive_pairs.t().tolist()}
    if count > possible - len(positive):
        raise ValueError("requested more negative pairs than the graph has non-edges")

    negatives: set[tuple[int, int]] = set()
    attempts = 0
    max_attempts = max(10_000, count * 100)
    while len(negatives) < count and attempts < max_attempts:
        candidates = torch.randint(num_nodes, (2, max(256, min(count, 8192))), generator=generator)
        for a, b in candidates.t().tolist():
            attempts += 1
            if a == b:
                continue
            pair = (a, b) if a < b else (b, a)
            if pair not in positive:
                negatives.add(pair)
                if len(negatives) == count:
                    break
        if attempts >= max_attempts:
            break
    if len(negatives) != count:
        raise RuntimeError(
            "negative-pair rejection sampling stalled; the graph may be too dense"
        )
    return torch.tensor(sorted(negatives), dtype=torch.long).t().contiguous()


def _partition_indices(count: int, validation_fraction: float, test_fraction: float,
                       generator: torch.Generator) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between 0 and 1")
    if not 0.0 < test_fraction < 1.0:
        raise ValueError("test_fraction must be between 0 and 1")
    if validation_fraction + test_fraction >= 1.0:
        raise ValueError("validation_fraction + test_fraction must be less than 1")
    if count < 3:
        raise ValueError("at least three examples of each class are required")

    order = torch.randperm(count, generator=generator)
    n_validation = max(1, int(round(count * validation_fraction)))
    n_test = max(1, int(round(count * test_fraction)))
    if n_validation + n_test >= count:
        n_validation = max(1, min(n_validation, count - 2))
        n_test = max(1, count - n_validation - 1)
    n_train = count - n_validation - n_test
    if n_train < 1:
        raise ValueError("split leaves no training examples")
    return order[:n_train], order[n_train:n_train + n_validation], order[n_train + n_validation:]


def _combine_examples(
    positives: torch.Tensor,
    negatives: torch.Tensor,
    positive_indices: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    negative_indices: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    generator: torch.Generator,
) -> tuple[tuple[torch.Tensor, torch.Tensor], ...]:
    positive_splits = [positives[:, ix] for ix in positive_indices]
    negative_splits = [negatives[:, ix] for ix in negative_indices]
    result = []
    for pos, neg in zip(positive_splits, negative_splits):
        pairs = torch.cat((pos, neg), dim=1)
        labels = torch.cat((torch.ones(pos.size(1)), torch.zeros(neg.size(1))))
        order = torch.randperm(pairs.size(1), generator=generator)
        result.append((pairs[:, order].contiguous(), labels[order].float()))
    return tuple(result)


def build_binary_edge_split(
    positive_pairs: torch.Tensor,
    num_nodes: int,
    *,
    validation_fraction: float = 0.15,
    test_fraction: float = 0.15,
    negative_ratio: float = 1.0,
    seed: int = 17,
) -> BinaryEdgeSplit:
    """Split undirected interactions and negatives without held-out edge leakage.

    Negatives are sampled uniformly from unordered non-edges. Message passing
    uses only training positive pairs, represented in both directions. All
    reversed/duplicate positives are collapsed before splitting.
    """
    if num_nodes < 2:
        raise ValueError("at least two nodes are required")
    if negative_ratio <= 0:
        raise ValueError("negative_ratio must be positive")
    positives = canonicalize_pairs(positive_pairs, num_nodes)
    if positives.size(1) < 3:
        raise ValueError("at least three unique positive pairs are required")

    generator = torch.Generator(device="cpu").manual_seed(seed)
    negative_count = max(3, int(round(positives.size(1) * negative_ratio)))
    negatives = _sample_negative_pairs(num_nodes, positives, negative_count, generator)
    positive_indices = _partition_indices(
        positives.size(1), validation_fraction, test_fraction, generator
    )


    negative_indices = _partition_indices(
        negatives.size(1), validation_fraction, test_fraction, generator
    )
    positive_splits = [positives[:, ix] for ix in positive_indices]
    negative_splits = [negatives[:, ix] for ix in negative_indices]

    datasets = _combine_examples(
        positives, negatives, positive_indices, negative_indices, generator
    )
    # Make the message-passing topology from training positives only; the
    # validation and test positive target pairs are absent in either direction.
    context_positive = positive_splits[0]
    message_edges = torch.cat((context_positive, context_positive.flip(0)), dim=1)
    return BinaryEdgeSplit(
        message_edge_index=message_edges.contiguous(),
        train_pairs=datasets[0][0], train_labels=datasets[0][1],
        validation_pairs=datasets[1][0], validation_labels=datasets[1][1],
        test_pairs=datasets[2][0], test_labels=datasets[2][1],
        context_positive_pairs=context_positive,
    )


def build_m5_detection_split(
    positive_pairs: torch.Tensor,
    num_nodes: int,
    m5_splits: SplitSet,
    *,
    negative_ratio: float = 1.0,
    seed: int = 17,
) -> tuple[M5DetectionSplit, SplitSet]:
    """Build a detection dataset whose positive folds follow M5 exactly.

    Each canonical M5 positive partition gets its own unique sampled negatives
    at ``negative_ratio`` negatives per positive. Reserved severe positives
    and their assigned negatives remain outside training. The returned
    ``SplitSet`` indexes the combined binary candidate array and can be passed
    to M5's ``scripts/run_eval.py`` together with the exported predictions.
    """
    if num_nodes < 2:
        raise ValueError("at least two nodes are required")
    if negative_ratio <= 0:
        raise ValueError("negative_ratio must be positive")
    positives = canonicalize_pairs(positive_pairs, num_nodes)
    raw_canonical = np.asarray(m5_splits.meta.get("canonical_pairs", []), dtype=np.int64)
    if raw_canonical.shape != (positives.size(1), 2):
        raise ValueError("M5 split must contain canonical_pairs matching the positive graph")
    split_pairs = torch.as_tensor(raw_canonical.T.copy(), dtype=torch.long)
    if {tuple(p) for p in positives.t().tolist()} != {tuple(p) for p in split_pairs.t().tolist()}:
        raise ValueError("M5 split pairs do not match the loaded positive graph")
    if split_pairs.size(1) != len({tuple(p) for p in split_pairs.t().tolist()}):
        raise ValueError("M5 canonical_pairs contains duplicate positives")

    partition_names = ("train", "val", "test", "severe_benchmark")
    positive_by_split: dict[str, torch.Tensor] = {}
    negative_counts: dict[str, int] = {}
    occupied: set[int] = set()
    for name in partition_names:
        if name not in m5_splits:
            if name == "severe_benchmark":
                positive_by_split[name] = torch.empty((2, 0), dtype=torch.long)
                negative_counts[name] = 0
                continue
            raise ValueError(f"M5 split is missing the {name!r} partition")
        indices = torch.as_tensor(m5_splits[name], dtype=torch.long).reshape(-1)
        if indices.numel() and (
            indices.min().item() < 0 or indices.max().item() >= split_pairs.size(1)
        ):
            raise ValueError(f"M5 {name} indices are out of range")
        values = set(indices.tolist())
        if occupied & values:
            raise ValueError("M5 partitions overlap")
        occupied.update(values)
        positives_for_split = split_pairs[:, indices]
        positive_by_split[name] = positives_for_split
        negative_counts[name] = (
            max(1, int(round(indices.numel() * negative_ratio))) if indices.numel() else 0
        )
    if occupied != set(range(split_pairs.size(1))):
        raise ValueError("M5 partitions do not assign every canonical positive pair")

    total_negatives = sum(negative_counts.values())
    generator = torch.Generator(device="cpu").manual_seed(seed)
    negatives = _sample_negative_pairs(num_nodes, positives, total_negatives, generator)
    negative_by_split: dict[str, torch.Tensor] = {}
    cursor = 0
    for name in partition_names:
        count = negative_counts[name]
        negative_by_split[name] = negatives[:, cursor : cursor + count]
        cursor += count

    pair_parts: list[torch.Tensor] = []
    label_parts: list[torch.Tensor] = []
    index_parts: dict[str, torch.Tensor] = {}
    offset = 0
    for name in partition_names:
        pos = positive_by_split[name]
        neg = negative_by_split[name]
        part_pairs = torch.cat((pos, neg), dim=1)
        part_labels = torch.cat(
            (torch.ones(pos.size(1)), torch.zeros(neg.size(1)))
        ).float()
        if part_pairs.size(1):
            order = torch.randperm(part_pairs.size(1), generator=generator)
            part_pairs = part_pairs[:, order]
            part_labels = part_labels[order]
        pair_parts.append(part_pairs)
        label_parts.append(part_labels)
        index_parts[name] = torch.arange(offset, offset + part_pairs.size(1))
        offset += part_pairs.size(1)

    all_pairs = torch.cat(pair_parts, dim=1).contiguous()
    all_labels = torch.cat(label_parts).contiguous()
    train_positive = positive_by_split["train"]
    message_edges = torch.cat((train_positive, train_positive.flip(0)), dim=1).contiguous()
    held_positive = torch.cat(
        (positive_by_split["val"], positive_by_split["test"], positive_by_split["severe_benchmark"]),
        dim=1,
    )
    context_set = {tuple(p) for p in message_edges.t().tolist()}
    held_set = {tuple(p) for p in held_positive.t().tolist()}
    if context_set & held_set:
        raise ValueError("held-out positive pairs entered binary message passing")

    result = M5DetectionSplit(
        message_edge_index=message_edges,
        pairs=all_pairs,
        labels=all_labels,
        train_indices=index_parts["train"],
        validation_indices=index_parts["val"],
        test_indices=index_parts["test"],
        severe_indices=index_parts["severe_benchmark"],
    )
    detection_splits = SplitSet(
        indices=index_parts,
        seed=seed,
        ratios=m5_splits.ratios,
        severe_benchmark_frac=m5_splits.severe_benchmark_frac,
        strategy="m5_positive_folds_with_partitioned_sampled_negatives",
        meta={
            "task": "binary_interaction_detection",
            "positive_split_source": m5_splits.meta.get("source_file", "M5 canonical split"),
            "positive_split_seed": m5_splits.seed,
            "negative_ratio": negative_ratio,
            "negative_sampling": "uniform unordered non-edges; unobserved interactions may be false negatives",
            "num_candidate_pairs": int(all_pairs.size(1)),
        },
    )
    return result, detection_splits
def align_node_features(
    drug_ids: list[str] | tuple[str, ...],
    features_by_id: Mapping[str, torch.Tensor],
) -> torch.Tensor:
    """Stack an ID-keyed feature mapping in graph node order, without sorting IDs."""
    missing = [drug_id for drug_id in drug_ids if drug_id not in features_by_id]
    if missing:
        sample = ", ".join(map(str, missing[:5]))
        raise ValueError(f"features missing for {len(missing)} graph drugs (for example: {sample})")
    rows = [torch.as_tensor(features_by_id[drug_id], dtype=torch.float32) for drug_id in drug_ids]
    if not rows:
        raise ValueError("graph has no drug IDs")
    if any(row.ndim != 1 for row in rows):
        raise ValueError("each drug feature must be a one-dimensional vector")
    widths = {row.numel() for row in rows}
    if len(widths) != 1:
        raise ValueError("drug feature vectors have inconsistent widths")
    features = torch.stack(rows)
    if not torch.isfinite(features).all():
        raise ValueError("node features contain NaN or Inf values")
    return features


def _graph_field(graph: Any, name: str) -> Any:
    if isinstance(graph, Mapping):
        return graph.get(name)
    return getattr(graph, name, None)


def _load_node_ids(nodes_path: Path) -> list[str]:
    """Read M1's processed nodes table in its serialized node-index order."""
    if not nodes_path.is_file():
        raise FileNotFoundError(
            f"M1 graph has no drug_id field; its companion node table is missing: {nodes_path}"
        )
    with nodes_path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or "drug_id" not in reader.fieldnames:
            raise ValueError(f"M1 nodes table must contain a drug_id column: {nodes_path}")
        rows = list(reader)
    ids = [str(row["drug_id"]).strip() for row in rows]
    if not ids or any(not drug_id for drug_id in ids):
        raise ValueError(f"M1 nodes table contains no IDs or blank drug IDs: {nodes_path}")
    if len(set(ids)) != len(ids):
        raise ValueError(f"M1 nodes table contains duplicate drug IDs: {nodes_path}")
    if "node_idx" in (reader.fieldnames or []):
        try:
            node_indices = [int(row["node_idx"]) for row in rows]
        except (TypeError, ValueError) as error:
            raise ValueError(f"M1 nodes table has invalid node_idx values: {nodes_path}") from error
        if node_indices != list(range(len(ids))):
            raise ValueError("M1 node_idx must match the row order used by graph.pt")
    return ids


def _validated_graph_edges(graph: Any, edge_index: Any, num_nodes: int) -> torch.Tensor:
    """Validate M1's edge tensor before converting it to canonical pair indices."""
    raw_edges = torch.as_tensor(edge_index, device="cpu")
    if raw_edges.ndim != 2 or raw_edges.size(0) != 2:
        raise ValueError("graph edge_index must have shape [2, num_edges]")
    if raw_edges.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
        raise TypeError("graph edge_index must contain integer node indices")
    edges = raw_edges.to(dtype=torch.long)
    if edges.numel() and (edges.min().item() < 0 or edges.max().item() >= num_nodes):
        raise ValueError(f"graph edge_index contains an index outside [0, {num_nodes})")
    if edges.size(1) and torch.any(edges[0] == edges[1]):
        raise ValueError("graph edge_index contains self-interactions")

    # M1 serializes each labeled pair in both directions and repeats its label.
    severity = _graph_field(graph, "edge_severity")
    if severity is not None:
        severity = torch.as_tensor(severity, device="cpu")
        if severity.ndim != 1 or severity.numel() != edges.size(1):
            raise ValueError("graph edge_severity must have one value per edge_index column")
        if severity.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
            if not severity.is_floating_point() or not torch.equal(severity, severity.round()):
                raise TypeError("graph edge_severity must contain integer labels")
        if severity.numel() and (severity.min().item() < 0 or severity.max().item() > 2):
            raise ValueError("graph edge_severity labels must be in {0, 1, 2}")
        labels_by_pair: dict[tuple[int, int], int] = {}
        for pair, label in zip(edges.t().tolist(), severity.to(torch.long).tolist()):
            key = tuple(sorted(pair))
            previous = labels_by_pair.setdefault(key, label)
            if previous != label:
                raise ValueError(f"duplicate/reversed graph edges disagree on severity for {key}")
    return canonicalize_pairs(edges, num_nodes)


def load_binary_inputs(
    graph_path: str | Path,
    features_path: str | Path,
    *,
    nodes_path: str | Path | None = None,
) -> tuple[list[str], torch.Tensor, torch.Tensor]:
    """Load M1's graph and M2's ``{drug_id: feature}`` bundle.

    M1's current Day-2 ``graph.pt`` stores ``edge_index``, ``edge_severity``
    and ``num_nodes`` but omits IDs. In that case the companion ``nodes.csv``
    is required; its row order (and optional ``node_idx``) defines graph IDs.
    """
    graph_path, features_path = Path(graph_path), Path(features_path)
    if not graph_path.exists():
        raise FileNotFoundError(
            f"processed graph not found: {graph_path}. M1 must produce a graph with drug_id and edge_index."
        )
    if not features_path.exists():
        raise FileNotFoundError(
            f"feature bundle not found: {features_path}. M2 must produce full_features.pt keyed by drug_id."
        )
    graph = torch.load(graph_path, map_location="cpu", weights_only=False)
    features_by_id = torch.load(features_path, map_location="cpu", weights_only=False)
    drug_ids = _graph_field(graph, "drug_id")
    edge_index = _graph_field(graph, "edge_index")
    if edge_index is None:
        raise ValueError("processed graph must expose edge_index")
    if drug_ids is None:
        companion_nodes = Path(nodes_path) if nodes_path is not None else graph_path.parent / "nodes.csv"
        drug_ids = _load_node_ids(companion_nodes)
    else:
        drug_ids = [str(drug_id).strip() for drug_id in drug_ids]
        if not drug_ids or any(not drug_id for drug_id in drug_ids):
            raise ValueError("processed graph contains no drug IDs or blank drug IDs")
        if len(set(drug_ids)) != len(drug_ids):
            raise ValueError("processed graph contains duplicate drug IDs")
    declared_num_nodes = _graph_field(graph, "num_nodes")
    if declared_num_nodes is not None and int(declared_num_nodes) != len(drug_ids):
        raise ValueError(
            f"graph num_nodes={int(declared_num_nodes)} does not match {len(drug_ids)} drug IDs"
        )
    if not isinstance(features_by_id, Mapping):
        raise TypeError("feature bundle must be a mapping {drug_id: feature_tensor}")
    # Graph IDs are strings in the shared contract; normalize feature mapping
    # keys once while rejecting ambiguous collisions.
    normalized_features: dict[str, torch.Tensor] = {}
    for key, value in features_by_id.items():
        normalized_key = str(key)
        if normalized_key in normalized_features:
            raise ValueError(f"feature mapping has duplicate normalized drug ID {normalized_key!r}")
        normalized_features[normalized_key] = value
    x = align_node_features(drug_ids, normalized_features)
    positives = _validated_graph_edges(graph, edge_index, len(drug_ids))
    if positives.size(1) == 0:
        raise ValueError("processed graph has no positive interaction edges")
    return drug_ids, x, positives


def binary_metrics(labels: torch.Tensor, logits: torch.Tensor) -> dict[str, float | None]:
    """Calculate thresholded metrics and ROC-AUC/AP when both classes exist."""
    y_true = labels.detach().cpu().numpy().astype(np.int64)
    scores = logits.detach().cpu().numpy()
    predictions = (scores >= 0.0).astype(np.int64)
    both_classes = len(np.unique(y_true)) == 2
    return {
        "accuracy": float((predictions == y_true).mean()),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, scores)) if both_classes else None,
        "average_precision": float(average_precision_score(y_true, scores)) if both_classes else None,
    }


def _run_batches(
    model: BinaryBaselineGNN,
    x: torch.Tensor,
    edge_index: torch.Tensor,
    pairs: torch.Tensor,
    labels: torch.Tensor,
    criterion: nn.Module,
    *,
    device: torch.device,
    batch_size: int,
    optimizer: torch.optim.Optimizer | None = None,
    seed: int = 17,
) -> tuple[float, torch.Tensor]:
    if pairs.size(1) != labels.numel():
        raise ValueError("number of pairs and labels must match")
    training = optimizer is not None
    model.train(training)
    labels = labels.reshape(-1).float()
    dataset = TensorDataset(pairs.t().contiguous(), labels)
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=training, generator=generator)
    total_loss = 0.0
    logits_all = []
    for batch_pairs, batch_labels in loader:
        batch_pairs = batch_pairs.t().contiguous().to(device)
        batch_labels = batch_labels.to(device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            logits = model(x, edge_index, batch_pairs)
            if not torch.isfinite(logits).all():
                raise FloatingPointError("model produced non-finite logits")
            loss = criterion(logits, batch_labels)
            if not torch.isfinite(loss):
                raise FloatingPointError("model produced a non-finite loss")
            if training:
                loss.backward()
                optimizer.step()
        total_loss += loss.detach().item() * batch_labels.numel()
        logits_all.append(logits.detach().cpu())
    if not logits_all:
        raise ValueError("cannot run an epoch with no examples")
    return total_loss / len(dataset), torch.cat(logits_all)


def save_checkpoint(
    path: str | Path,
    model: BinaryBaselineGNN,
    optimizer: torch.optim.Optimizer,
    *,
    epoch: int,
    seed: int,
    validation_loss: float,
) -> None:
    """Save a resumable checkpoint under the caller-selected path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_config": model.config,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": int(epoch),
        "seed": int(seed),
        "validation_loss": float(validation_loss),
    }, path)


def load_checkpoint(
    path: str | Path,
    model: BinaryBaselineGNN,
    optimizer: torch.optim.Optimizer | None = None,
    *,
    device: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Restore model (and optionally optimizer) state from a checkpoint."""
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if checkpoint.get("model_config") != model.config:
        raise ValueError("checkpoint model configuration does not match the supplied model")
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint


def train_binary_model(
    model: BinaryBaselineGNN,
    x: torch.Tensor,
    message_edge_index: torch.Tensor,
    train_pairs: torch.Tensor,
    train_labels: torch.Tensor,
    validation_pairs: torch.Tensor,
    validation_labels: torch.Tensor,
    *,
    epochs: int = 20,
    batch_size: int = 256,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    seed: int = 17,
    device: str | torch.device | None = None,
    checkpoint_path: str | Path | None = None,
) -> dict[str, Any]:
    """Train the binary model and retain the checkpoint with best val loss."""
    if epochs < 1 or batch_size < 1:
        raise ValueError("epochs and batch_size must be positive")
    if learning_rate <= 0.0 or weight_decay < 0.0:
        raise ValueError("learning_rate must be positive and weight_decay non-negative")
    set_seed(seed)
    device = resolve_device(device)
    model = model.to(device)
    x = x.to(device)
    message_edge_index = message_edge_index.to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    history: list[dict[str, Any]] = []
    best_validation_loss = float("inf")
    best_epoch = 0

    for epoch in range(1, epochs + 1):
        train_loss, _ = _run_batches(
            model, x, message_edge_index, train_pairs, train_labels, criterion,
            device=device, batch_size=batch_size, optimizer=optimizer, seed=seed + epoch,
        )
        validation_loss, validation_logits = _run_batches(
            model, x, message_edge_index, validation_pairs, validation_labels, criterion,
            device=device, batch_size=batch_size, seed=seed,
        )
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "validation_loss": validation_loss,
            "validation_metrics": binary_metrics(validation_labels, validation_logits),
        }
        history.append(row)
        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            best_epoch = epoch
            if checkpoint_path is not None:
                save_checkpoint(
                    checkpoint_path, model, optimizer, epoch=epoch, seed=seed,
                    validation_loss=validation_loss,
                )
    if checkpoint_path is not None:
        load_checkpoint(checkpoint_path, model, optimizer, device=device)
    return {
        "history": history,
        "best_epoch": best_epoch,
        "best_validation_loss": best_validation_loss,
        "device": str(device),
        "seed": seed,
        "model_config": model.config,
        "optimizer": "Adam",
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
    }


def run_dry_run(
    graph_path: str | Path = "data/processed/graph.pt",
    features_path: str | Path = "features/full_features.pt",
    *,
    epochs: int = 20,
    batch_size: int = 256,
    seed: int = 17,
    negative_ratio: float = 1.0,
    checkpoint_path: str | Path = "models/checkpoints/model1_binary_baseline.pt",
    results_path: str | Path = "results/model1_dry_run.json",
    device: str | None = None,
    dataset_kind: str = "project_artifacts",
) -> dict[str, Any]:
    """Load real M1/M2 artifacts, train, and write a compact result record."""
    drug_ids, x, positives = load_binary_inputs(graph_path, features_path)
    split = build_binary_edge_split(
        positives, len(drug_ids), negative_ratio=negative_ratio, seed=seed
    )
    set_seed(seed)
    model = BinaryBaselineGNN(num_node_features=x.size(1))
    outcome = train_binary_model(
        model, x, split.message_edge_index, split.train_pairs, split.train_labels,
        split.validation_pairs, split.validation_labels, epochs=epochs,
        batch_size=batch_size, seed=seed, device=device, checkpoint_path=checkpoint_path,
    )
    model.to(outcome["device"])
    load_checkpoint(checkpoint_path, model, device=outcome["device"])
    test_loss, test_logits = _run_batches(
        model, x.to(outcome["device"]), split.message_edge_index.to(outcome["device"]),
        split.test_pairs, split.test_labels, nn.BCEWithLogitsLoss(),
        device=torch.device(outcome["device"]), batch_size=batch_size, seed=seed,
    )
    result = {
        "experiment": "model1_binary_baseline_dry_run",
        "dataset_kind": dataset_kind,
        "data_source": str(graph_path),
        "feature_source": str(features_path),
        "num_drugs": len(drug_ids),
        "num_unique_positive_pairs": positives.size(1),
        "num_message_passing_edges": split.message_edge_index.size(1),
        "split_sizes": {
            "train": int(split.train_labels.numel()),
            "validation": int(split.validation_labels.numel()),
            "test": int(split.test_labels.numel()),
        },
        "negative_sampling": "uniform random unordered non-edges; unobserved positives may be false negatives",
        "negative_ratio": negative_ratio,
        "held_out_positive_edges_in_message_graph": 0,
        **outcome,
        "test_loss": test_loss,
        "test_metrics": binary_metrics(split.test_labels, test_logits),
        "checkpoint": str(checkpoint_path),
    }
    result_path = Path(results_path)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def run_m5_detection_experiment(
    graph_path: str | Path = "data/processed/graph.pt",
    features_path: str | Path = "features/full_features.pt",
    split_path: str | Path = "eval/splits/ddi_v1.json",
    *,
    epochs: int = 20,
    batch_size: int = 256,
    seed: int = 17,
    negative_ratio: float = 1.0,
    checkpoint_path: str | Path = "models/checkpoints/model1_binary_baseline.pt",
    results_path: str | Path = "results/model1_results.json",
    predictions_path: str | Path = "results/preds_detection_baseline.npz",
    detection_splits_path: str | Path = "results/model1_detection_splits.json",
    device: str | None = None,
    dataset_kind: str = "project_artifacts",
) -> dict[str, Any]:
    """Train/evaluate Model 1 using positives assigned by M5's canonical split.

    Negatives are sampled once from unordered non-edges, then assigned to the
    same partition as the corresponding M5 positive pool. The generated
    detection split is separate from severity labels but preserves M5's fold
    boundaries for all observed positive interactions.
    """
    drug_ids, x, positive_pairs = load_binary_inputs(graph_path, features_path)
    canonical_splits = load_splits(split_path)
    split, detection_splits = build_m5_detection_split(
        positive_pairs,
        len(drug_ids),
        canonical_splits,
        negative_ratio=negative_ratio,
        seed=seed,
    )
    set_seed(seed)
    model = BinaryBaselineGNN(num_node_features=x.size(1))
    outcome = train_binary_model(
        model,
        x,
        split.message_edge_index,
        split.train_pairs,
        split.train_labels,
        split.validation_pairs,
        split.validation_labels,
        epochs=epochs,
        batch_size=batch_size,
        seed=seed,
        device=device,
        checkpoint_path=checkpoint_path,
    )
    model.to(outcome["device"])
    load_checkpoint(checkpoint_path, model, device=outcome["device"])

    val_loss, val_logits = _run_batches(
        model,
        x.to(outcome["device"]),
        split.message_edge_index.to(outcome["device"]),
        split.validation_pairs,
        split.validation_labels,
        nn.BCEWithLogitsLoss(),
        device=torch.device(outcome["device"]),
        batch_size=batch_size,
        seed=seed,
    )
    test_loss, test_logits = _run_batches(
        model,
        x.to(outcome["device"]),
        split.message_edge_index.to(outcome["device"]),
        split.test_pairs,
        split.test_labels,
        nn.BCEWithLogitsLoss(),
        device=torch.device(outcome["device"]),
        batch_size=batch_size,
        seed=seed,
    )
    val_scores = torch.sigmoid(val_logits).numpy()
    test_scores = torch.sigmoid(test_logits).numpy()
    val_result = evaluate_detection(
        split.validation_labels.numpy(),
        val_scores,
        model_name="detection_baseline",
        split="val",
    )
    test_result = evaluate_detection(
        split.test_labels.numpy(),
        test_scores,
        model_name="detection_baseline",
        split="test",
    )

    save_splits(detection_splits, detection_splits_path)
    held_indices = torch.cat(
        (split.validation_indices, split.test_indices)
    ).numpy().astype(np.int64)
    held_labels = torch.cat(
        (split.validation_labels, split.test_labels)
    ).numpy().astype(np.int64)
    held_scores = np.concatenate((val_scores, test_scores)).astype(np.float32)
    prediction_file = Path(predictions_path)
    prediction_file.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        prediction_file,
        edge_idx=held_indices,
        y_true_bin=held_labels,
        y_score=held_scores,
        task=np.asarray("detection"),
    )
    result = {
        "experiment": "model1_binary_interaction_detection",
        "task": "detection",
        "dataset_kind": dataset_kind,
        "data_source": str(graph_path),
        "feature_source": str(features_path),
        "severity_split_source": str(split_path),
        "detection_split_source": str(detection_splits_path),
        "num_drugs": len(drug_ids),
        "num_node_features": int(x.size(1)),
        "num_positive_pairs": int(positive_pairs.size(1)),
        "negative_sampling": "uniform random unordered non-edges; unobserved positives may be false negatives",
        "negative_ratio": negative_ratio,
        "split_sizes": {
            "train": int(split.train_labels.numel()),
            "validation": int(split.validation_labels.numel()),
            "test": int(split.test_labels.numel()),
            "severe_benchmark": int(split.severe_indices.numel()),
        },
        "held_out_positive_edges_in_message_graph": 0,
        "validation_loss": val_loss,
        "test_loss": test_loss,
        "metrics": {"val": val_result.to_dict(), "test": test_result.to_dict()},
        "predictions": str(prediction_file),
        "checkpoint": str(checkpoint_path),
        **outcome,
    }
    result_file = Path(results_path)
    result_file.parent.mkdir(parents=True, exist_ok=True)
    result_file.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def run_synthetic_dry_run(
    *,
    epochs: int = 20,
    batch_size: int = 32,
    seed: int = 17,
    results_path: str | Path = "results/model1_synthetic_dry_run.json",
    device: str | None = None,
) -> dict[str, Any]:
    """Run the same pipeline on an explicitly synthetic two-group fixture."""
    import tempfile

    generator = torch.Generator(device="cpu").manual_seed(seed)
    num_per_group = 8
    drug_ids = [f"SYNTHETIC-{i:03d}" for i in range(2 * num_per_group)]
    features = torch.zeros((len(drug_ids), 2), dtype=torch.float32)
    features[:num_per_group, 0] = 1.0
    features[num_per_group:, 1] = 1.0
    positives = torch.tensor(
        [(left, right) for left in range(num_per_group)
         for right in range(num_per_group, 2 * num_per_group)],
        dtype=torch.long,
    ).t().contiguous()
    # Small deterministic noise prevents the fixture from being a constant-only input.
    features += 0.01 * torch.randn(features.shape, generator=generator)
    with tempfile.TemporaryDirectory(prefix="ddi_m3_synthetic_") as tmp:
        root = Path(tmp)
        graph_path, features_path = root / "graph.pt", root / "features.pt"
        torch.save({"drug_id": drug_ids, "edge_index": positives}, graph_path)
        torch.save({drug_id: features[i] for i, drug_id in enumerate(drug_ids)}, features_path)
        result = run_dry_run(
            graph_path, features_path, epochs=epochs, batch_size=batch_size, seed=seed,
            negative_ratio=0.8, checkpoint_path=root / "model.pt",
            results_path=results_path, device=device, dataset_kind="synthetic_fixture",
        )
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph", default="data/processed/graph.pt")
    parser.add_argument("--features", default="features/full_features.pt")
    parser.add_argument("--splits", default="eval/splits/ddi_v1.json")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--device", default=None)
    parser.add_argument("--negative-ratio", type=float, default=1.0)
    parser.add_argument(
        "--synthetic", action="store_true",
        help="run a synthetic-only pipeline check; metrics do not represent project data",
    )
    parser.add_argument("--checkpoint", default="models/checkpoints/model1_binary_baseline.pt")
    parser.add_argument("--results", default=None)
    parser.add_argument("--predictions", default="results/preds_detection_baseline.npz")
    parser.add_argument("--detection-splits", default="results/model1_detection_splits.json")
    args = parser.parse_args(argv)
    try:
        if args.synthetic:
            result = run_synthetic_dry_run(
                epochs=args.epochs, batch_size=args.batch_size, seed=args.seed,
                device=args.device,
                results_path=args.results or "results/model1_synthetic_dry_run.json",
            )
        else:
            result = run_m5_detection_experiment(
                args.graph,
                args.features,
                args.splits,
                epochs=args.epochs,
                batch_size=args.batch_size,
                seed=args.seed,
                negative_ratio=args.negative_ratio,
                device=args.device,
                checkpoint_path=args.checkpoint,
                results_path=args.results or "results/model1_results.json",
                predictions_path=args.predictions,
                detection_splits_path=args.detection_splits,
            )
    except (FileNotFoundError, ValueError, TypeError, RuntimeError) as error:
        parser.error(str(error))
    print(json.dumps({
        "best_epoch": result["best_epoch"],
        "best_validation_loss": result["best_validation_loss"],
        "test_loss": result.get("test_loss"),
        "test_metrics": result.get("test_metrics", result.get("metrics", {}).get("test")),
        "checkpoint": result["checkpoint"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
