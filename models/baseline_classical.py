"""No-graph classical baselines for M3/M4's severity comparison.

Uses the same M5 canonical split, feature file, and scoring code as Models 1-3,
but with NO message passing at all: each drug pair is represented by its raw
feature vectors combined the same way models.baseline_gnn.DrugPairEncoder
combines learned embeddings (sum, |difference|, product), so the comparison
isolates what the graph/attention machinery adds over the molecular features
alone.

    python -m models.baseline_classical --model rf --tag baseline_rf
    python -m models.baseline_classical --model mlp --tag baseline_mlp

Writes results/<tag>_results.json (M5 EvalResult, same schema as Models 1-3)
and results/<tag>_details.json.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from eval.benchmark_severe_subset import benchmark_severe
from eval.metrics import evaluate
from models.ordinal_gnn import load_ordinal_inputs


def pair_features(x: torch.Tensor, pairs: torch.Tensor) -> np.ndarray:
    """Symmetric (sum, |diff|, product) pair representation, no graph context."""
    a, b = x[pairs[0]], x[pairs[1]]
    combined = torch.cat([a + b, (a - b).abs(), a * b], dim=1)
    return combined.numpy()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph", default="data/processed/graph.pt")
    parser.add_argument("--features", default="features/full_features.pt")
    parser.add_argument("--splits", default="eval/splits/ddi_v1.json")
    parser.add_argument("--model", choices=["rf", "mlp", "gbm"], default="rf")
    parser.add_argument("--tag", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--results-dir", default="results")
    args = parser.parse_args()
    tag = args.tag or f"baseline_{args.model}"

    drug_ids, x, split, splits = load_ordinal_inputs(args.graph, args.features, args.splits)

    X_train = pair_features(x, split.train_pairs)
    y_train = split.train_labels.numpy()
    X_val = pair_features(x, split.validation_pairs)
    y_val = split.validation_labels.numpy()
    X_test = pair_features(x, split.test_pairs)
    y_test = split.test_labels.numpy()
    X_severe = pair_features(x, split.severe_pairs)
    y_severe = split.severe_labels.numpy()

    if args.model == "rf":
        from sklearn.ensemble import RandomForestClassifier

        clf = RandomForestClassifier(
            n_estimators=300, max_depth=None, min_samples_leaf=2,
            class_weight="balanced", n_jobs=-1, random_state=args.seed,
        )
    elif args.model == "gbm":
        from sklearn.ensemble import HistGradientBoostingClassifier

        clf = HistGradientBoostingClassifier(
            max_iter=300, class_weight="balanced", random_state=args.seed,
        )
    else:
        from sklearn.neural_network import MLPClassifier

        clf = MLPClassifier(
            hidden_layer_sizes=(256, 128), max_iter=200, early_stopping=True,
            n_iter_no_change=8, random_state=args.seed,
        )

    print(f"Training {args.model} on {X_train.shape[0]} pairs, {X_train.shape[1]} features...", flush=True)
    clf.fit(X_train, y_train)

    def predict_proba3(X: np.ndarray) -> np.ndarray:
        proba = clf.predict_proba(X)
        full = np.zeros((X.shape[0], 3), dtype=float)
        for col, cls in enumerate(clf.classes_):
            full[:, int(cls)] = proba[:, col]
        return full

    results = {}
    held_pred, held_proba = [], []
    for name, X, y in (("val", X_val, y_val), ("test", X_test, y_test), ("severe_benchmark", X_severe, y_severe)):
        proba = predict_proba3(X)
        pred = proba.argmax(axis=1)
        held_pred.append(pred)
        held_proba.append(proba)
        if name == "severe_benchmark":
            labels_np = split.labels.numpy()
            full_pred = np.zeros_like(labels_np)
            full_proba = np.zeros((labels_np.size, 3), dtype=float)
            full_pred[split.severe_indices.numpy()] = pred
            full_proba[split.severe_indices.numpy()] = proba
            result = benchmark_severe(labels_np, full_pred, splits, y_proba=full_proba, model_name=tag)
        else:
            result = evaluate(y, pred, y_proba=proba, model_name=tag, split=name)
        results[name] = result
        print(name, "macro_f1" if hasattr(result, "macro_f1") else "major_recall",
              getattr(result, "macro_f1", getattr(result, "major_recall", None)), flush=True)

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    results["test"].save(results_dir / f"{tag}_results.json")

    held_idx = np.concatenate([
        split.validation_indices.numpy(), split.test_indices.numpy(), split.severe_indices.numpy(),
    ])
    np.savez_compressed(
        results_dir / f"preds_{tag}.npz",
        edge_idx=held_idx.astype(np.int64),
        y_pred=np.concatenate(held_pred).astype(np.int64),
        y_proba=np.concatenate(held_proba).astype(np.float32),
        y_true=split.labels.numpy()[held_idx].astype(np.int64),
    )
    print(f"wrote {results_dir / f'preds_{tag}.npz'}", flush=True)
    details = {
        "model": f"No-graph classical baseline ({args.model})",
        "tag": tag,
        "config": vars(args),
        "feature_representation": "sum, |difference|, product of raw per-drug feature vectors (no message passing)",
        "num_node_features": int(x.size(1)),
        "num_pair_features": int(X_train.shape[1]),
        "results": {k: v.to_dict() for k, v in results.items()},
    }
    (results_dir / f"{tag}_details.json").write_text(json.dumps(details, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {results_dir / f'{tag}_results.json'}", flush=True)


if __name__ == "__main__":
    main()
