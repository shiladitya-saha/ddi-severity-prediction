"""Pair-by-pair model comparison on the full test split.

Same idea as scripts/export_severe_benchmark_table.py but for all 33,002
test-split pairs (mixed MINOR/MODERATE/MAJOR), not just the MAJOR-only
severe-benchmark subset. Reads each model's results/preds_<model>.npz (same
files scripts/run_eval.py scores), so predictions line up exactly with
eval/results_comparison.md.

    python -m scripts.export_test_predictions

Writes results/test_predictions.csv (all 33,002 rows) and
results/test_predictions_preview.md (30 example rows across severities).
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
    test_idx = np.asarray(splits["test"], dtype=np.int64)
    canonical_pairs = np.asarray(splits.meta["canonical_pairs"], dtype=np.int64)

    nodes = pd.read_csv("data/processed/nodes.csv", dtype={"drug_id": str})
    names = nodes["name"].tolist()

    # y_true is identical across every model's npz (same ground truth); read once.
    with np.load(MODELS["Model2_GCN"]) as z:
        edge_idx = z["edge_idx"].astype(np.int64)
        y_true = z["y_true"].astype(np.int64)
    true_by_idx = dict(zip(edge_idx.tolist(), y_true.tolist()))
    missing = [i for i in test_idx if i not in true_by_idx]
    if missing:
        raise SystemExit(f"{len(missing)} test pairs have no ground truth in the reference npz")

    pair_a = canonical_pairs[test_idx, 0]
    pair_b = canonical_pairs[test_idx, 1]

    table = pd.DataFrame({
        "drug_a": [names[i] for i in pair_a],
        "drug_b": [names[i] for i in pair_b],
        "true_severity": [LABELS[true_by_idx[i]] for i in test_idx],
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
        missing = [i for i in test_idx if i not in pred_by_idx]
        if missing:
            raise SystemExit(f"{col_name}: {len(missing)} test pairs have no prediction")
        table[col_name] = [LABELS[pred_by_idx[i]] for i in test_idx]

    model_cols = [c for c in MODELS if c in table.columns]
    correct = np.zeros(len(table), dtype=int)
    for c in model_cols:
        correct += (table[c].to_numpy() == table["true_severity"].to_numpy()).astype(int)
    table["models_correct"] = correct
    table = table.sort_values(["models_correct", "true_severity"]).reset_index(drop=True)

    out_csv = Path("results/test_predictions.csv")
    table.to_csv(out_csv, index=False)
    print(f"wrote {out_csv} ({len(table)} rows)")

    preview_parts = []
    for sev in ("MAJOR", "MODERATE", "MINOR"):
        preview_parts.append(table[table["true_severity"] == sev].head(10))
    preview = pd.concat(preview_parts, ignore_index=True)
    lines = ["# Test-set predictions, 10 hardest examples per true severity", "",
             "All 33,002 rows are in `test_predictions.csv`. This shows 10 example "
             "pairs per true severity class, the ones fewest models got right.", "",
             "| " + " | ".join(preview.columns) + " |",
             "| " + " | ".join("---" for _ in preview.columns) + " |"]
    for _, row in preview.iterrows():
        lines.append("| " + " | ".join(str(v) for v in row) + " |")
    out_md = Path("results/test_predictions_preview.md")
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {out_md}")


if __name__ == "__main__":
    main()
