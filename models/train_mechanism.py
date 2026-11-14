"""
train_mechanism.py
Owner: M4. Trains and evaluates Model 3, the Mechanism-Aware Ordinal GNN.

    python -m models.train_mechanism

Inputs
  M1  data/processed/nodes.csv, data/processed/edges_labeled.csv
  M2  full_features.pt ({drug_id: vector}) and mechanism_edges.csv; until M2's
      pipeline runs, build them with `python -m models.build_m4_inputs`
  M5  eval.split.make_splits (70/15/15 stratified + held-out MAJOR subset) and
      eval.metrics.evaluate, so results are comparable with Models 1 and 2

Leakage controls (both on by default)
  * The scored pair's own mechanism tags are NOT given to the model. M1's labels
    and M2's tags are both keyword rules over the same DrugBank interaction text,
    so those tags alone predict severity at 74% vs 54% for the majority class.
    --pair-mechanism turns them on, for the leak ablation only.
  * A training pair's own edge is removed from the message graph while that pair
    is being scored, so training matches evaluation, where validation and test
    pairs are never in the message graph.

Outputs (named after --tag, default "model3")
  results/<tag>_results.json   M5 EvalResult on the test split
  results/<tag>_details.json   config, data provenance, split sizes, training
                               history, validation and severe-subset results
  models/checkpoints/<tag>.pt  best-validation weights (git-ignored)
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score

from models.mechanism_gnn import MechanismOrdinalGNN, load_mechanism_table, mechanism_matrix
from models.ordinal_loss import CoralLoss, coral_logits_to_probs, coral_predict

DEFAULT_INPUTS = Path("models/checkpoints/m4_inputs")


@dataclass
class Inputs:
    drug_ids: list
    x: torch.Tensor  # [N, F + 1]; the last column flags drugs with no fingerprint
    pairs: torch.Tensor  # [2, E] undirected, low node first, in M1's edge order
    labels: torch.Tensor  # [E] severity 0/1/2
    table: dict  # mechanism table from load_mechanism_table
    drugs_without_features: int


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_inputs(proc_dir="data/processed", features_path=DEFAULT_INPUTS / "full_features.pt",
                mechanism_path=DEFAULT_INPUTS / "mechanism_edges.csv"):
    proc_dir = Path(proc_dir)
    nodes = pd.read_csv(proc_dir / "nodes.csv", dtype={"drug_id": str})
    if "node_idx" in nodes and not (nodes["node_idx"].to_numpy() == np.arange(len(nodes))).all():
        raise ValueError("nodes.csv node_idx must be 0..N-1 in row order")
    drug_ids = nodes["drug_id"].tolist()
    edges = pd.read_csv(proc_dir / "edges_labeled.csv", usecols=["src", "dst", "severity"])
    pairs = torch.tensor(edges[["src", "dst"]].to_numpy().T, dtype=torch.long)
    pairs = torch.stack([pairs.min(0).values, pairs.max(0).values])
    if torch.unique(pairs, dim=1).size(1) != pairs.size(1):
        raise ValueError("edges_labeled.csv repeats a drug pair")
    labels = torch.tensor(edges["severity"].to_numpy(), dtype=torch.long)

    features = torch.load(features_path, map_location="cpu", weights_only=False)
    widths = {int(torch.as_tensor(v).numel()) for v in features.values()}
    if len(widths) != 1:
        raise ValueError(f"feature vectors have inconsistent widths {sorted(widths)}")
    width = widths.pop()
    # Drugs without a usable structure (mostly biologics) get zeros plus a flag
    # column, so the model can tell "no fingerprint" from "no substructures".
    x = torch.zeros(len(drug_ids), width + 1)
    missing = 0
    for i, drug_id in enumerate(drug_ids):
        if drug_id in features:
            x[i, :width] = torch.as_tensor(features[drug_id], dtype=torch.float)
        else:
            x[i, width] = 1.0
            missing += 1
    table = load_mechanism_table(mechanism_path, drug_ids)
    return Inputs(drug_ids, x, pairs, labels, table, missing)


def message_graph(pairs, attr, use_edge_attr):
    """Undirected pairs [2, M] -> directed edge_index [2, 2M] and edge_attr [2M, T]."""
    edge_index = torch.cat([pairs, pairs.flip(0)], dim=1)
    edge_attr = torch.cat([attr, attr]) if use_edge_attr else torch.zeros(2 * attr.size(0), attr.size(1))
    return edge_index, edge_attr


@torch.no_grad()
def predict(model, x, context_pairs, table, target_pairs, *, use_pair_attr, use_edge_attr,
            chunk=65536):
    """Score target_pairs with context_pairs as the message graph. Returns logits."""
    model.eval()
    edge_index, edge_attr = message_graph(
        context_pairs, mechanism_matrix(context_pairs, table), use_edge_attr
    )
    out = []
    for start in range(0, target_pairs.size(1), chunk):
        targets = target_pairs[:, start:start + chunk]
        pair_attr = mechanism_matrix(targets, table) if use_pair_attr else None
        out.append(model(x, edge_index, edge_attr, targets, pair_attr))
    return torch.cat(out) if out else torch.empty(0, model.config["num_classes"] - 1)


def train_model(model, x, train_pairs, train_labels, table, val_pairs, val_labels, *,
                epochs=30, batch_size=8192, lr=1e-3, weight_decay=1e-4, class_weights=None,
                patience=5, use_pair_attr=False, use_edge_attr=True, seed=42, val_context=None,
                log=print):
    """Train with early stopping on validation macro F1; returns (history, best_epoch).

    val_context: message graph used to score validation pairs (default: all
    training pairs).
    """
    generator = torch.Generator().manual_seed(seed)
    train_attr = mechanism_matrix(train_pairs, table)
    val_context = train_pairs if val_context is None else val_context
    loss_fn = CoralLoss(class_weights=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    best_f1, best_epoch, best_state, stale = -1.0, 0, None, 0
    history = []
    for epoch in range(1, epochs + 1):
        start = time.time()
        model.train()
        total_loss = 0.0
        for batch in torch.randperm(train_pairs.size(1), generator=generator).split(batch_size):
            keep = torch.ones(train_pairs.size(1), dtype=torch.bool)
            keep[batch] = False  # the scored pairs' own edges leave the message graph
            edge_index, edge_attr = message_graph(train_pairs[:, keep], train_attr[keep], use_edge_attr)
            pair_attr = train_attr[batch] if use_pair_attr else None
            logits = model(x, edge_index, edge_attr, train_pairs[:, batch], pair_attr)
            loss = loss_fn(logits, train_labels[batch])
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss at epoch {epoch}")
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch.numel()

        val_logits = predict(model, x, val_context, table, val_pairs,
                             use_pair_attr=use_pair_attr, use_edge_attr=use_edge_attr)
        val_pred = coral_predict(val_logits)
        val_f1 = f1_score(val_labels.numpy(), val_pred.numpy(), average="macro", zero_division=0)
        val_mae = (val_pred - val_labels).abs().float().mean().item()
        row = {"epoch": epoch, "train_loss": total_loss / train_pairs.size(1),
               "val_macro_f1": val_f1, "val_mae": val_mae, "seconds": time.time() - start}
        history.append(row)
        log(f"epoch {epoch:3d}  loss {row['train_loss']:.4f}  val macroF1 {val_f1:.4f}  "
            f"val MAE {val_mae:.4f}  ({row['seconds']:.0f}s)")
        if val_f1 > best_f1:
            best_f1, best_epoch, best_state, stale = val_f1, epoch, copy.deepcopy(model.state_dict()), 0
        else:
            stale += 1
            if stale >= patience:
                log(f"early stop: no validation gain for {patience} epochs")
                break
    model.load_state_dict(best_state)
    return history, best_epoch


def inverse_frequency_weights(labels, num_classes=3):
    """N / (K * n_c), M1's 'inverse_frequency' formula, computed on the given labels."""
    counts = torch.bincount(labels, minlength=num_classes).float()
    if (counts == 0).any():
        raise ValueError(f"a severity class is missing from training labels: {counts.tolist()}")
    return counts.sum() / (num_classes * counts)


def evaluate_logits(logits, labels, model_name, split, with_proba=True):
    from eval.metrics import evaluate

    pred = coral_predict(logits).numpy()
    proba = coral_logits_to_probs(logits).numpy() if with_proba else None
    return evaluate(labels.numpy(), pred, y_proba=proba, model_name=model_name, split=split)


def run(args):
    from eval.split import make_splits, verify_splits

    set_seed(args.seed)
    data = load_inputs(args.proc_dir, args.features, args.mechanism_edges)
    edge_rows = data.pairs.t().numpy()
    splits = make_splits(edge_rows, data.labels.numpy(), seed=args.seed,
                         severe_benchmark_frac=args.severe_frac)
    verify_splits(splits, data.labels.numpy(), edge_rows)
    part = {name: torch.as_tensor(splits[name], dtype=torch.long) for name in splits.names}
    print(splits.summary(data.labels.numpy()))

    train_pairs, train_labels = data.pairs[:, part["train"]], data.labels[part["train"]]
    weights = inverse_frequency_weights(train_labels)
    model = MechanismOrdinalGNN(
        num_node_features=data.x.size(1), hidden_dim=args.hidden, num_layers=args.layers,
        heads=args.heads, dropout=args.dropout,
    )
    use_edge_attr = not args.no_edge_mechanism
    history, best_epoch = train_model(
        model, data.x, train_pairs, train_labels, data.table,
        data.pairs[:, part["val"]], data.labels[part["val"]],
        epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
        weight_decay=args.weight_decay, class_weights=weights, patience=args.patience,
        use_pair_attr=args.pair_mechanism, use_edge_attr=use_edge_attr, seed=args.seed,
    )

    results = {}
    for name in ("val", "test", "severe_benchmark"):
        logits = predict(model, data.x, train_pairs, data.table, data.pairs[:, part[name]],
                         use_pair_attr=args.pair_mechanism, use_edge_attr=use_edge_attr)
        # The severe subset is MAJOR only, so probability-based metrics are undefined.
        results[name] = evaluate_logits(logits, data.labels[part[name]], args.tag, name,
                                        with_proba=name != "severe_benchmark")
        pred = coral_predict(logits)
        results[name + "_predicted_class_share"] = {
            c: float((pred == i).float().mean()) for i, c in enumerate(("minor", "moderate", "major"))
        }

    results_dir, checkpoint_dir = Path(args.results_dir), Path(args.checkpoint_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    results["test"].save(results_dir / f"{args.tag}_results.json")
    checkpoint = checkpoint_dir / f"{args.tag}.pt"
    torch.save({"model_config": model.config, "model_state_dict": model.state_dict(),
                "args": vars(args), "best_epoch": best_epoch}, checkpoint)

    summary_path = Path(args.features).with_name("summary.json")
    details = {
        "model": "Mechanism-Aware Ordinal GNN (GATv2 + CORAL)",
        "tag": args.tag,
        "config": vars(args),
        "model_config": model.config,
        "leakage_controls": {
            "scored_pair_mechanism_input": bool(args.pair_mechanism),
            "edge_mechanism_in_message_graph": use_edge_attr,
            "scored_training_pairs_removed_from_message_graph": True,
            "val_test_pairs_in_message_graph": False,
        },
        "data": {
            "source": "Mendeley Data md5czfsfnd v1, DDI_data.csv (DrugBank v5.1-derived), CC BY 4.0, doi:10.17632/md5czfsfnd.1",
            "labels": "M1 heuristic severity rules (data/label_severity.py)",
            "drugs": len(data.drug_ids),
            "edges": int(data.pairs.size(1)),
            "drugs_without_features": data.drugs_without_features,
            "feature_width": int(data.x.size(1)),
            "input_summary": json.loads(summary_path.read_text()) if summary_path.exists() else None,
        },
        "split": {"seed": args.seed, "strategy": splits.strategy, "sizes": splits.sizes(),
                  "class_counts": splits.class_counts(data.labels.numpy())},
        "class_weights_train_inverse_frequency": weights.tolist(),
        "best_epoch": best_epoch,
        "history": history,
        "results": {k: (v.to_dict() if hasattr(v, "to_dict") else v) for k, v in results.items()},
        "checkpoint": str(checkpoint),
    }
    (results_dir / f"{args.tag}_details.json").write_text(json.dumps(details, indent=2) + "\n")
    test, severe = results["test"], results["severe_benchmark"]
    print(f"TEST macroF1 {test.macro_f1:.4f}  MAE {test.mae:.4f}  | severe subset MAJOR recall "
          f"{results['severe_benchmark_predicted_class_share']['major']:.4f}")
    return details


def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--proc-dir", default="data/processed")
    p.add_argument("--features", default=str(DEFAULT_INPUTS / "full_features.pt"))
    p.add_argument("--mechanism-edges", default=str(DEFAULT_INPUTS / "mechanism_edges.csv"))
    p.add_argument("--tag", default="model3")
    p.add_argument("--results-dir", default="results")
    p.add_argument("--checkpoint-dir", default="models/checkpoints")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--severe-frac", type=float, default=0.15)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=8192)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--pair-mechanism", action="store_true",
                   help="LEAK ABLATION: give the model the scored pair's own mechanism tags")
    p.add_argument("--no-edge-mechanism", action="store_true",
                   help="ablation: zero the mechanism features on message-passing edges")
    return p


def main(argv=None):
    run(build_parser().parse_args(argv))


if __name__ == "__main__":
    main()
