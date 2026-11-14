#!/usr/bin/env python3
"""
Score every model that has produced predictions and write the comparison doc.

Owner: M5 (Eval / Explainability Lead) - see eval/README.md

This is the Day-3/Day-4 workhorse. It does not import anyone's model code, so
it cannot break when M3 or M4 refactor: each model writes a predictions file,
this reads whatever files exist and regenerates `eval/results_comparison.md`.

    python scripts/run_eval.py                     # score everything in results/
    python scripts/run_eval.py --preds results/preds_mechanism_gnn.npz
    python scripts/run_eval.py --splits eval/splits/ddi_v1.json --out eval/results_comparison.md

PREDICTIONS CONTRACT (M3 / M4 - this is all you need to give me)
----------------------------------------------------------------
Save one file per model, named `results/preds_<model_name>.npz`:

    np.savez(
        "results/preds_ordinal_gnn.npz",
        edge_idx = idx,        # (N,) int, positions in the deduped edge array
                               #      i.e. the same indices the split JSON uses
        y_pred   = y_pred,     # (N,) int in {0,1,2}
        y_proba  = proba,      # (N,3) float, optional but wanted - enables AUC
        y_true   = y_true,     # (N,) int, optional - taken from labels if absent
    )

For a binary DETECTION model (M3's Model 1), save instead:

    np.savez(
        "results/preds_detection_baseline.npz",
        edge_idx    = idx,
        y_true_bin  = y_true_binary,   # (N,) in {0,1}
        y_score     = scores,          # (N,) float probability of interaction
        task        = "detection",
    )

Detection results are reported in their own table, never mixed with severity.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.benchmark_severe_subset import (
    benchmark_severe,
    severe_subset_markdown,
    severity_miss_profile,
)
from eval.metrics import (
    EvalResult,
    comparison_to_markdown,
    detection_to_markdown,
    evaluate,
    evaluate_detection,
    metrics_to_markdown,
)
from eval.split import load_splits

DEFAULT_RESULTS_DIR = Path("results")
DEFAULT_OUT = Path("eval/results_comparison.md")
PREDS_GLOB = "preds_*.npz"


def discover_prediction_files(results_dir: Path, explicit: Sequence[str]) -> List[Path]:
    if explicit:
        paths = [Path(p) for p in explicit]
        missing = [p for p in paths if not p.exists()]
        if missing:
            raise SystemExit(f"missing prediction file(s): {missing}")
        return paths
    if not results_dir.exists():
        return []
    return sorted(results_dir.glob(PREDS_GLOB))


def model_name_from(path: Path) -> str:
    stem = path.stem
    return stem[len("preds_") :] if stem.startswith("preds_") else stem


def load_predictions(path: Path) -> Dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def score_one(
    path: Path,
    splits,
    labels: Optional[np.ndarray],
) -> Tuple[List[EvalResult], List, Dict[str, object]]:
    """Score one predictions file across every split it covers."""
    data = load_predictions(path)
    name = model_name_from(path)
    task = str(data.get("task", np.array("severity")))

    if "edge_idx" not in data:
        raise SystemExit(
            f"{path} has no 'edge_idx' array - I can't tell which edges these "
            "predictions belong to. See the contract at the top of this script."
        )
    edge_idx = data["edge_idx"].astype(int)

    severity_results: List[EvalResult] = []
    detection_results = []
    extras: Dict[str, object] = {}

    if "detection" in task:
        if "y_score" not in data:
            raise SystemExit(f"{path} is a detection file but has no 'y_score'")
        y_true_bin = data.get("y_true_bin")
        if y_true_bin is None:
            raise SystemExit(f"{path} is a detection file but has no 'y_true_bin'")
        for split_name in ("test", "val"):
            if split_name not in splits:
                continue
            mask = np.isin(edge_idx, splits[split_name])
            if not mask.any():
                continue
            detection_results.append(
                evaluate_detection(
                    y_true_bin[mask],
                    data["y_score"][mask],
                    model_name=name,
                    split=split_name,
                )
            )
        return severity_results, detection_results, extras

    if "y_pred" not in data:
        raise SystemExit(f"{path} has no 'y_pred' array")
    y_pred_sub = data["y_pred"].astype(int)
    proba_sub = data.get("y_proba")

    if "y_true" in data:
        y_true_sub = data["y_true"].astype(int)
    elif labels is not None:
        y_true_sub = labels[edge_idx]
    else:
        raise SystemExit(
            f"{path} has no 'y_true' and no label file was given "
            "(--labels). Add y_true to the npz or pass --labels."
        )

    for split_name in ("test", "val", "severe_benchmark"):
        if split_name not in splits or np.asarray(splits[split_name]).size == 0:
            continue
        mask = np.isin(edge_idx, splits[split_name])
        if not mask.any():
            continue
        severity_results.append(
            evaluate(
                y_true_sub[mask],
                y_pred_sub[mask],
                y_proba=None if proba_sub is None else proba_sub[mask],
                model_name=name,
                split=split_name,
            )
        )

    # miss profile on the severe subset, if it was covered
    if "severe_benchmark" in splits and np.asarray(splits["severe_benchmark"]).size:
        mask = np.isin(edge_idx, splits["severe_benchmark"])
        if mask.any():
            full_true = np.zeros(int(edge_idx.max()) + 1, dtype=int)
            full_pred = np.zeros_like(full_true)
            full_true[edge_idx] = y_true_sub
            full_pred[edge_idx] = y_pred_sub
            covered = np.intersect1d(edge_idx, splits["severe_benchmark"])
            if covered.size:
                shim = type("S", (), {"__contains__": lambda s, k: k == "severe_benchmark",
                                      "__getitem__": lambda s, k: covered})()
                extras["severe_miss_profile"] = severity_miss_profile(
                    full_true, full_pred, shim
                )
    return severity_results, detection_results, extras


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="score models and write the comparison doc")
    ap.add_argument("--preds", nargs="*", default=[], help="specific .npz files")
    ap.add_argument("--results-dir", default=str(DEFAULT_RESULTS_DIR))
    ap.add_argument("--splits", default="eval/splits/ddi_v1.json")
    ap.add_argument("--labels", default=None, help="optional .npy of all severity labels")
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    args = ap.parse_args(argv)

    print("=" * 68)
    print("M5 evaluation run")
    print("=" * 68)

    split_path = Path(args.splits)
    if not split_path.exists():
        raise SystemExit(
            f"{split_path} does not exist.\n"
            "Generate it once M1's labeled edges land:\n"
            "    python scripts/make_splits.py --edges data/processed/edges_labeled.csv"
        )
    splits = load_splits(split_path)
    print(f"\n[1] splits loaded from {split_path}")
    print(f"      {splits.sizes()}")

    labels = None
    if args.labels:
        labels = np.load(args.labels).astype(int)
        print(f"      labels loaded from {args.labels} ({labels.size} edges)")

    files = discover_prediction_files(Path(args.results_dir), args.preds)
    print(f"\n[2] prediction files found: {len(files)}")
    for f in files:
        print(f"      {f}")

    if not files:
        print(
            "\nNothing to score yet. Models write "
            f"{args.results_dir}/preds_<model>.npz - contract is at the top of this script."
        )
        return 0

    severity_results: List[EvalResult] = []
    detection_results = []
    extras_by_model: Dict[str, Dict[str, object]] = {}

    for f in files:
        sev, det, extras = score_one(f, splits, labels)
        severity_results.extend(sev)
        detection_results.extend(det)
        if extras:
            extras_by_model[model_name_from(f)] = extras
        shown = sev[0] if sev else None
        if shown is not None:
            print(
                f"\n      {shown.model_name:<22} macroF1={shown.macro_f1:.3f} "
                f"MAE={shown.mae:.3f} MAJORrecall={shown.major_recall:.3f}"
            )
        for d in det:
            auc = "n/a" if d.roc_auc is None else f"{d.roc_auc:.3f}"
            print(f"      {d.model_name:<22} [detection] AUC={auc} F1={d.f1:.3f}")

    # write the doc
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    test_results = [r for r in severity_results if r.split == "test"]
    severe_results = [r for r in severity_results if r.split == "severe_benchmark"]

    body = [
        "# Model comparison",
        "",
        f"Generated by `scripts/run_eval.py` from `{split_path}`.",
        "Regenerate after any new checkpoint; do not hand-edit the tables.",
        "",
        comparison_to_markdown(test_results, title="Severity prediction (ordinal, 3-class)"),
        "",
    ]
    if severe_results:
        body += [severe_subset_markdown(severe_results), ""]
    if detection_results:
        body += [detection_to_markdown(detection_results), ""]
    if test_results:
        best = max(test_results, key=lambda r: r.macro_f1)
        body += ["## Best model detail", "", metrics_to_markdown(best), ""]
    if extras_by_model:
        body += ["## Severe-subset miss profile", "", "Where MAJOR predictions land when wrong.", ""]
        for name, extras in extras_by_model.items():
            prof = extras.get("severe_miss_profile", {})
            if not prof:
                continue
            body.append(f"**{name}** — " + ", ".join(
                f"{k}: {v['count']} ({v['share']:.0%})"
                for k, v in prof.items() if not k.startswith("_")
            ))
        body.append("")

    out.write_text("\n".join(body), encoding="utf-8")
    for r in severity_results:
        r.save(Path(args.results_dir) / f"{r.model_name}_{r.split}.json")
    for d in detection_results:
        d.save(Path(args.results_dir) / f"{d.model_name}_{d.split}_detection.json")

    print(f"\n[3] wrote {out}")
    print(f"      per-model JSON in {args.results_dir}/")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
