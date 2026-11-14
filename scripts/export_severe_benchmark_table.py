"""Pair-by-pair model comparison on the severe-benchmark subset.

Builds a human-readable table: drug A, drug B, true severity, and every
model's predicted severity, for the 2,282 held-out MAJOR-only pairs. Reads
each model's results/preds_<model>.npz (same files scripts/run_eval.py
scores) and the official split, so the drug pairs and predictions line up
exactly with eval/results_comparison.md.

    python -m scripts.export_severe_benchmark_table

Writes results/severe_benchmark_predictions.csv (all 2,282 rows) and
results/severe_benchmark_predictions_preview.md (first 30 rows, for a quick
look without opening a spreadsheet).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from eval.split import load_splits

LABELS = ("MINOR", "MODERATE", "MAJOR")

MODELS = {
    "Model2_GCN": "results/preds_ordinal_gnn.npz",
    "Model3_tagson": "results/preds_mechanism_gnn.npz",
    "Model3_tagsoff": "results/preds_mechanism_gnn_noedge.npz",
    "RF": "results/preds_baseline_rf.npz",
    "GBM": "results/preds_baseline_gbm.npz",
    "MLP": "results/preds_baseline_mlp.npz",
    "DeepDDI": "results/preds_baseline_deepddi.npz",
}


def main() -> None:
    splits = load_splits("eval/splits/ddi_v1.json")
    severe_idx = np.asarray(splits["severe_benchmark"], dtype=np.int64)
    canonical_pairs = np.asarray(splits.meta["canonical_pairs"], dtype=np.int64)

    nodes = pd.read_csv("data/processed/nodes.csv", dtype={"drug_id": str})
    names = nodes["name"].tolist()

    pair_a = canonical_pairs[severe_idx, 0]
    pair_b = canonical_pairs[severe_idx, 1]

    table = pd.DataFrame({
        "drug_a": [names[i] for i in pair_a],
        "drug_b": [names[i] for i in pair_b],
        "true_severity": "MAJOR",  # severe_benchmark is MAJOR-only by construction
    })

    for col_name, path in MODELS.items():
        npz_path = Path(path)
        if not npz_path.exists():
            print(f"skipping {col_name}: {npz_path} not found")
            continue
        with np.load(npz_path) as z:
            edge_idx = z["edge_idx"].astype(np.int64)
            y_pred = z["y_pred"].astype(np.int64)
        pred_by_idx = dict(zip(edge_idx.tolist(), y_pred.tolist()))
        missing = [i for i in severe_idx if i not in pred_by_idx]
        if missing:
            raise SystemExit(f"{col_name}: {len(missing)} severe-benchmark pairs have no prediction")
        table[col_name] = [LABELS[pred_by_idx[i]] for i in severe_idx]

    model_cols = [c for c in MODELS if c in table.columns]
    table["models_correct"] = table[model_cols].apply(lambda row: (row == "MAJOR").sum(), axis=1)
    table = table.sort_values("models_correct").reset_index(drop=True)

    out_csv = Path("results/severe_benchmark_predictions.csv")
    table.to_csv(out_csv, index=False)
    print(f"wrote {out_csv} ({len(table)} rows)")

    preview = table.head(30)
    lines = ["# Severe-benchmark predictions, worst cases first", "",
             "All 2,282 rows are in `severe_benchmark_predictions.csv`. This shows the "
             "30 pairs where the fewest models predicted MAJOR correctly -- the hardest "
             "cases.", "",
             "| " + " | ".join(preview.columns) + " |",
             "| " + " | ".join("---" for _ in preview.columns) + " |"]
    for _, row in preview.iterrows():
        lines.append("| " + " | ".join(str(v) for v in row) + " |")
    out_md = Path("results/severe_benchmark_predictions_preview.md")
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {out_md}")


if __name__ == "__main__":
    main()
