"""A DeepDDI-style baseline (Ryu, Kim & Lee, PNAS 2018), re-implemented and
trained on OUR OWN data, split, and scoring code -- not a citation of their
reported numbers, which were measured on a different dataset and task.

Each drug is represented by its Structural Similarity Profile (SSP): Tanimoto
similarity of its ECFP4 fingerprint against every other drug's fingerprint
(a vector of length num_drugs). A pair's feature is the concatenation of both
drugs' SSPs, fed through a deep feedforward network -- no graph, no message
passing, same idea as the original paper adapted to our 3-class ordinal
severity labels instead of their 86 interaction types.

    python -m models.baseline_deepddi

Writes results/baseline_deepddi_results.json / _details.json, same schema as
Models 1-3 and the other no-graph baselines.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from eval.benchmark_severe_subset import benchmark_severe
from eval.metrics import evaluate
from models.ordinal_gnn import load_ordinal_inputs

FINGERPRINT_DIM = 1024  # first FINGERPRINT_DIM columns of full_features.pt are ECFP4


def tanimoto_similarity_matrix(fp: np.ndarray) -> np.ndarray:
    """fp: [N, bits] binary -> [N, N] pairwise Tanimoto similarity."""
    fp = fp.astype(np.float32)
    intersection = fp @ fp.T
    row_sums = fp.sum(axis=1, keepdims=True)
    union = row_sums + row_sums.T - intersection
    sim = np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)
    return sim


class DeepDDINet(nn.Module):
    def __init__(self, input_dim: int, num_classes: int = 3, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 2048), nn.BatchNorm1d(2048), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(2048, 1024), nn.BatchNorm1d(1024), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(1024, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        return self.net(x)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph", default="data/processed/graph.pt")
    parser.add_argument("--features", default="features/full_features.pt")
    parser.add_argument("--splits", default="eval/splits/ddi_v1.json")
    parser.add_argument("--tag", default="baseline_deepddi")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--results-dir", default="results")
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    drug_ids, x, split, splits = load_ordinal_inputs(args.graph, args.features, args.splits)
    fp = x[:, :FINGERPRINT_DIM].numpy()
    print(f"computing {fp.shape[0]}x{fp.shape[0]} Tanimoto similarity profiles...", flush=True)
    ssp = torch.as_tensor(tanimoto_similarity_matrix(fp), dtype=torch.float32)

    def pair_ssp(pairs: torch.Tensor) -> torch.Tensor:
        a, b = ssp[pairs[0]], ssp[pairs[1]]
        return torch.cat([a, b], dim=1)

    X_train, y_train = pair_ssp(split.train_pairs), split.train_labels
    X_val, y_val = pair_ssp(split.validation_pairs), split.validation_labels
    X_test, y_test = pair_ssp(split.test_pairs), split.test_labels
    X_severe, y_severe = pair_ssp(split.severe_pairs), split.severe_labels

    device = torch.device(args.device)
    model = DeepDDINet(input_dim=X_train.size(1)).to(device)
    counts = torch.bincount(y_train, minlength=3).float()
    class_weights = (counts.sum() / counts.clamp(min=1)).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)

    train_ds = torch.utils.data.TensorDataset(X_train, y_train)
    loader = torch.utils.data.DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)

    best_val_loss, best_state, bad_epochs, best_epoch = float("inf"), None, 0, 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = criterion(model(xb.to(device)), yb.to(device))
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            val_logits = model(X_val.to(device))
            val_loss = nn.functional.cross_entropy(val_logits, y_val.to(device), weight=class_weights).item()
        if val_loss < best_val_loss - 1e-5:
            best_val_loss, best_state, bad_epochs, best_epoch = val_loss, {k: v.clone() for k, v in model.state_dict().items()}, 0, epoch
        else:
            bad_epochs += 1
        print(f"epoch {epoch} val_loss {val_loss:.4f} best {best_val_loss:.4f} ({best_epoch})", flush=True)
        if bad_epochs >= args.patience:
            print("early stopping", flush=True)
            break

    model.load_state_dict(best_state)
    model.eval()

    def predict_proba3(X: torch.Tensor) -> np.ndarray:
        with torch.no_grad():
            return torch.softmax(model(X.to(device)), dim=1).cpu().numpy()

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
            result = benchmark_severe(labels_np, full_pred, splits, y_proba=full_proba, model_name=args.tag)
        else:
            result = evaluate(y.numpy(), pred, y_proba=proba, model_name=args.tag, split=name)
        results[name] = result
        print(name, getattr(result, "macro_f1", getattr(result, "major_recall", None)), flush=True)

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    results["test"].save(results_dir / f"{args.tag}_results.json")

    held_idx = np.concatenate([
        split.validation_indices.numpy(), split.test_indices.numpy(), split.severe_indices.numpy(),
    ])
    np.savez_compressed(
        results_dir / f"preds_{args.tag}.npz",
        edge_idx=held_idx.astype(np.int64),
        y_pred=np.concatenate(held_pred).astype(np.int64),
        y_proba=np.concatenate(held_proba).astype(np.float32),
        y_true=split.labels.numpy()[held_idx].astype(np.int64),
    )
    print(f"wrote {results_dir / f'preds_{args.tag}.npz'}", flush=True)
    details = {
        "model": "DeepDDI-style re-implementation (Ryu, Kim & Lee 2018), trained on our own split",
        "tag": args.tag,
        "config": vars(args),
        "feature_representation": "concatenated Tanimoto structural-similarity profiles (SSP), ECFP4-based",
        "best_epoch": best_epoch,
        "results": {k: v.to_dict() for k, v in results.items()},
        "citation_note": "Architecture and feature idea follow the published method; "
                          "numbers are NOT the paper's reported numbers -- they are measured "
                          "fresh on our dataset/split so the comparison to Models 1-3 is fair.",
    }
    (results_dir / f"{args.tag}_details.json").write_text(json.dumps(details, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {results_dir / f'{args.tag}_results.json'}", flush=True)


if __name__ == "__main__":
    main()
