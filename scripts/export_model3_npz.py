"""Export M5-compatible prediction .npz files for Model 3's seed-42 checkpoints.

Only seed 42 is exported here: it is the one Model 3 checkpoint verified to
sit on the exact official split (eval/splits/ddi_v1.json). Seeds 7/123/2024/777
were trained on train_mechanism.py's own self-generated per-seed split, which
is NOT the same held-out data as the official split -- scoring them against
official test/severe indices risks leakage, so they are intentionally left
out of the auto-generated eval/results_comparison.md table. See
results/model4_day5_baselines.md for how those seeds are reported instead.

    python -m scripts.export_model3_npz

Writes results/preds_mechanism_gnn.npz (tags-on) and
results/preds_mechanism_gnn_noedge.npz (tags-off).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from eval.split import load_splits
from models.mechanism_gnn import MechanismOrdinalGNN
from models.ordinal_loss import coral_logits_to_probs, coral_predict
from models.train_mechanism import load_inputs, predict

CHECKPOINTS = {
    "mechanism_gnn": ("models/checkpoints/model3.pt", True),
    "mechanism_gnn_noedge": ("models/checkpoints/model3_noedge.pt", False),
}


def main() -> None:
    data = load_inputs(features_path="features/full_features.pt", mechanism_path="features/mechanism_edges.csv")
    splits = load_splits("eval/splits/ddi_v1.json")
    val_idx = np.asarray(splits["val"], dtype=np.int64)
    test_idx = np.asarray(splits["test"], dtype=np.int64)
    severe_idx = np.asarray(splits["severe_benchmark"], dtype=np.int64)
    held_idx = np.concatenate([val_idx, test_idx, severe_idx])
    held_pairs = data.pairs[:, torch.as_tensor(held_idx, dtype=torch.long)]
    train_pairs = data.pairs[:, torch.as_tensor(np.asarray(splits["train"], dtype=np.int64), dtype=torch.long)]
    labels_np = data.labels.numpy()

    for tag, (ckpt_path, use_edge_attr) in CHECKPOINTS.items():
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model = MechanismOrdinalGNN(**ckpt["model_config"])
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()

        logits = predict(
            model, data.x, train_pairs, data.table, held_pairs,
            use_pair_attr=False, use_edge_attr=use_edge_attr,
        )
        pred = coral_predict(logits).numpy()
        proba = coral_logits_to_probs(logits).numpy()

        out_path = Path("results") / f"preds_{tag}.npz"
        np.savez_compressed(
            out_path,
            edge_idx=held_idx.astype(np.int64),
            y_pred=pred.astype(np.int64),
            y_proba=proba.astype(np.float32),
            y_true=labels_np[held_idx].astype(np.int64),
        )
        print(f"wrote {out_path} ({held_idx.size} held-out pairs)", flush=True)


if __name__ == "__main__":
    main()
