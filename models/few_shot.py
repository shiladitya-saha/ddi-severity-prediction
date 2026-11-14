"""
few_shot.py
Owner: M4. New-drug generalization for Model 3 (Day 3: first experiment).

    python -m models.few_shot

Protocol
  M5's eval.split.drug_disjoint_split partitions DRUGS 70/15/15. Model 3 trains
  on interactions among training drugs only and selects its epoch on validation
  drugs scored zero-shot. Test interactions are among test drugs, none of which
  appear in training.

  For k in --shots, each test drug gets up to k of its known interactions with
  TRAINING drugs (M5's "bridge" edges) added to the message graph as context,
  with their mechanism tags. Severity labels of those context edges are never
  model inputs, and the model is not retrained or fine-tuned: k=0 is zero-shot
  (fingerprints only) and k>0 measures how much a few known interactions help.
  Meta-learning (MAML-style adaptation) is a later step.

Outputs
  results/few_shot_results.json        per-k test metrics + support coverage
  models/checkpoints/few_shot_model3.pt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from models.mechanism_gnn import MechanismOrdinalGNN
from models.train_mechanism import (
    DEFAULT_INPUTS,
    evaluate_logits,
    inverse_frequency_weights,
    load_inputs,
    predict,
    set_seed,
    train_model,
)


def support_edges(pairs, split_indices, k, seed=0):
    """Pick up to k context interactions per test drug, each linking it to a training drug.

    Returns (indices into pairs, {test_drug: number of context edges given}).
    """
    train_drugs = set(pairs[:, split_indices["train"]].flatten().tolist())
    test_drugs = set(pairs[:, split_indices["test"]].flatten().tolist())
    candidates = {drug: [] for drug in test_drugs}
    for i in split_indices["bridge"].tolist():
        a, b = pairs[0, i].item(), pairs[1, i].item()
        if a in test_drugs and b in train_drugs:
            candidates[a].append(i)
        elif b in test_drugs and a in train_drugs:
            candidates[b].append(i)
    rng = np.random.default_rng(seed)
    chosen, given = [], {}
    for drug in sorted(candidates):
        picked = rng.permutation(candidates[drug])[:k].tolist() if k > 0 else []
        chosen.extend(picked)
        given[drug] = len(picked)
    return torch.tensor(sorted(set(chosen)), dtype=torch.long), given


def run(args):
    from eval.split import drug_disjoint_split

    set_seed(args.seed)
    data = load_inputs(args.proc_dir, args.features, args.mechanism_edges)
    splits = drug_disjoint_split(data.pairs.t().numpy(), data.labels.numpy(), seed=args.seed)
    part = {name: torch.as_tensor(splits[name], dtype=torch.long) for name in splits.names}
    print(splits.summary(data.labels.numpy()))

    train_pairs, train_labels = data.pairs[:, part["train"]], data.labels[part["train"]]
    model = MechanismOrdinalGNN(
        num_node_features=data.x.size(1), hidden_dim=args.hidden, num_layers=args.layers,
        heads=args.heads, dropout=args.dropout,
    )
    history, best_epoch = train_model(
        model, data.x, train_pairs, train_labels, data.table,
        data.pairs[:, part["val"]], data.labels[part["val"]],
        epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
        weight_decay=args.weight_decay, class_weights=inverse_frequency_weights(train_labels),
        patience=args.patience, seed=args.seed,
    )

    test_pairs, test_labels = data.pairs[:, part["test"]], data.labels[part["test"]]
    per_k = {}
    for k in args.shots:
        support, given = support_edges(data.pairs, part, k, seed=args.seed)
        context = torch.cat([train_pairs, data.pairs[:, support]], dim=1)
        logits = predict(model, data.x, context, data.table, test_pairs,
                         use_pair_attr=False, use_edge_attr=True)
        result = evaluate_logits(logits, test_labels, f"model3_{k}shot", "test_new_drugs")
        counts = np.array(list(given.values()))
        per_k[str(k)] = {
            "result": result.to_dict(),
            "support_edges_added": int(support.numel()),
            "test_drugs": int(counts.size),
            "test_drugs_with_full_k_support": int((counts >= k).sum()),
            "mean_support_per_test_drug": float(counts.mean()) if counts.size else 0.0,
        }
        print(f"k={k:2d}  macroF1 {result.macro_f1:.4f}  MAE {result.mae:.4f}  "
              f"support edges {support.numel()}")

    out = {
        "protocol": "drug-disjoint split (M5); k known interactions per test drug with training "
                    "drugs added as message-passing context; no fine-tuning; scored pair's own "
                    "mechanism tags not used",
        "config": vars(args),
        "split": {"strategy": splits.strategy, "sizes": splits.sizes(), "meta": splits.meta,
                  "class_counts": splits.class_counts(data.labels.numpy())},
        "drugs_without_features": data.drugs_without_features,
        "best_epoch": best_epoch,
        "history": history,
        "per_k": per_k,
    }
    results_dir, checkpoint_dir = Path(args.results_dir), Path(args.checkpoint_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "few_shot_results.json").write_text(json.dumps(out, indent=2) + "\n")
    torch.save({"model_config": model.config, "model_state_dict": model.state_dict(),
                "args": vars(args)}, checkpoint_dir / "few_shot_model3.pt")
    return out


def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--proc-dir", default="data/processed")
    p.add_argument("--features", default=str(DEFAULT_INPUTS / "full_features.pt"))
    p.add_argument("--mechanism-edges", default=str(DEFAULT_INPUTS / "mechanism_edges.csv"))
    p.add_argument("--results-dir", default="results")
    p.add_argument("--checkpoint-dir", default="models/checkpoints")
    p.add_argument("--shots", type=int, nargs="+", default=[0, 1, 3, 5, 10])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=8192)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    return p


def main(argv=None):
    run(build_parser().parse_args(argv))


if __name__ == "__main__":
    main()
