"""
Held-out severe-subset benchmark + OGBL-DDI benchmark environment.

Owner: M5 (Eval / Explainability Lead) - see eval/README.md

Two benchmarks live here, both answering the reviewer question "does it work on
the hard cases, and how does it compare to published work?":

1. SEVERE SUBSET - the MAJOR-only partition `eval.split.make_splits` reserves
   before the 70/15/15. Never trained on, never tuned on. `benchmark_severe()`
   scores a model on it and `severe_subset_markdown()` renders the table.

2. OGBL-DDI - the standard public DDI link-prediction benchmark (Hits@20 is the
   leaderboard metric), so our numbers are comparable to published work.

Why the OGB code is defensive: ogbl-ddi is the standard public benchmark
(Hits@20 is the leaderboard metric), so reporting it makes our numbers
comparable to published work. But `ogb` + its torch dependency may not be
installed on every teammate's machine on Day 1, so this module:

    * checks the environment and says exactly what is missing
    * implements Hits@K natively in numpy, so the metric code is testable
      without downloading a 100MB dataset
    * loads the real dataset and official split when ogb IS available

Nothing here imports ogb at module level - importing this file always works.

CLI
---
    python -m eval.benchmark_severe_subset --check            # environment report
    python -m eval.benchmark_severe_subset --download         # fetch dataset into data/ogb
"""

from __future__ import annotations

import argparse
import importlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

import numpy as np

from .metrics import MAJOR, SEVERITY_LABELS, EvalResult, evaluate

DEFAULT_ROOT = Path("data/ogb")
DATASET_NAME = "ogbl-ddi"
LEADERBOARD_K = 20
REPORT_KS = (10, 20, 50, 100)
SEVERE_SPLIT_KEY = "severe_benchmark"


# --------------------------------------------------------------------------- #
# 1. Held-out severe subset
# --------------------------------------------------------------------------- #


def benchmark_severe(
    labels,
    y_pred,
    splits,
    y_proba=None,
    model_name: str = "unnamed_model",
    split_key: str = SEVERE_SPLIT_KEY,
) -> EvalResult:
    """
    Score a model on the reserved MAJOR-only subset.

    `labels` and `y_pred` are full-length arrays over M1's edge list; the split
    indices select the benchmark rows, so callers never slice by hand and can
    never accidentally score the wrong partition.

    The headline number here is `major_recall` - on a MAJOR-only partition,
    accuracy and recall coincide and "what fraction of dangerous interactions
    did we catch?" is exactly the question. Macro F1 is near-meaningless on one
    class and is reported only for table consistency.
    """
    y = np.asarray(labels).astype(int)
    p = np.asarray(y_pred).astype(int)
    if split_key not in splits:
        raise KeyError(
            f"splits has no '{split_key}' partition - build them with "
            "make_splits(..., severe_benchmark_frac=0.15)"
        )
    idx = np.asarray(splits[split_key], dtype=int)
    if idx.size == 0:
        raise ValueError(f"'{split_key}' is empty - nothing to benchmark")
    if y.size != p.size:
        raise ValueError(f"labels ({y.size}) and y_pred ({p.size}) differ in length")

    held = y[idx]
    if not np.all(held == MAJOR):
        raise ValueError(
            f"'{split_key}' contains non-MAJOR edges - the reserved subset is "
            "contaminated; rebuild the splits"
        )

    proba = None if y_proba is None else np.asarray(y_proba, dtype=float)[idx]
    return evaluate(
        held,
        p[idx],
        y_proba=proba,
        model_name=model_name,
        split=split_key,
        notes={
            "partition": "held-out severe subset (MAJOR only)",
            "never_trained_on": "true",
            "headline_metric": "major_recall",
        },
    )


def severity_miss_profile(labels, y_pred, splits, split_key: str = SEVERE_SPLIT_KEY):
    """
    Where the misses go on the severe subset: a MAJOR called MODERATE is a very
    different clinical failure from a MAJOR called MINOR. The paper should
    report this, not just recall.
    """
    y = np.asarray(labels).astype(int)
    p = np.asarray(y_pred).astype(int)
    idx = np.asarray(splits[split_key], dtype=int)
    pred = p[idx]
    n = int(idx.size)
    out = {}
    for cls, name in SEVERITY_LABELS.items():
        count = int((pred == cls).sum())
        out[name] = {"count": count, "share": float(count / n) if n else 0.0}
    out["_n"] = {"count": n, "share": 1.0}
    return out


def severe_subset_markdown(results: Sequence[EvalResult], labels=None, y_preds=None) -> str:
    """Markdown block for results/final_comparison.md."""
    results = list(results)
    if not results:
        return "## Held-out severe subset\n\n_No results yet._\n"
    lines = [
        "## Held-out severe subset benchmark (MAJOR only, never trained on)",
        "",
        "| Model | n | MAJOR recall | MAE | Within-1 acc |",
        "| --- | --- | --- | --- | --- |",
    ]
    for r in results:
        lines.append(
            f"| {r.model_name} | {r.n_samples} | {r.major_recall:.4f} | "
            f"{r.mae:.4f} | {r.within_one_accuracy:.4f} |"
        )
    lines += [
        "",
        "MAJOR recall is the headline here: the partition is single-class, so it",
        "answers 'what fraction of severe interactions did the model catch?'.",
    ]
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- #
# 2. OGBL-DDI link prediction - Hits@K implemented locally
# --------------------------------------------------------------------------- #


def hits_at_k(pos_scores: Sequence[float], neg_scores: Sequence[float], k: int = 20) -> float:
    """
    Fraction of positive edges scored above the k-th highest negative edge.

    This matches OGB's `Evaluator('ogbl-ddi')` definition: a single global
    threshold taken from the negative distribution, not a per-query ranking.
    """
    pos = np.asarray(pos_scores, dtype=float).ravel()
    neg = np.asarray(neg_scores, dtype=float).ravel()
    if pos.size == 0:
        raise ValueError("pos_scores is empty")
    if neg.size == 0:
        raise ValueError("neg_scores is empty")
    if k <= 0:
        raise ValueError("k must be positive")
    if not np.all(np.isfinite(pos)) or not np.all(np.isfinite(neg)):
        raise ValueError("scores contain NaN/inf")
    if neg.size < k:
        # fewer negatives than k: threshold is the weakest negative
        threshold = neg.min()
    else:
        threshold = np.sort(neg)[-k]
    return float((pos > threshold).mean())


def mean_reciprocal_rank(pos_scores: Sequence[float], neg_scores: Sequence[float]) -> float:
    """MRR of each positive against the full negative pool (secondary metric)."""
    pos = np.asarray(pos_scores, dtype=float).ravel()
    neg = np.sort(np.asarray(neg_scores, dtype=float).ravel())
    if pos.size == 0 or neg.size == 0:
        raise ValueError("scores must be non-empty")
    # rank = 1 + number of negatives scoring strictly higher
    higher = neg.size - np.searchsorted(neg, pos, side="right")
    ranks = 1 + higher
    return float(np.mean(1.0 / ranks))


def evaluate_link_prediction(
    pos_scores: Sequence[float],
    neg_scores: Sequence[float],
    ks: Sequence[int] = REPORT_KS,
    model_name: str = "unnamed_model",
    split: str = "test",
) -> Dict[str, object]:
    """Full ogbl-ddi style report for one model on one split."""
    out: Dict[str, object] = {
        "model_name": model_name,
        "split": split,
        "dataset": DATASET_NAME,
        "n_pos": int(np.asarray(pos_scores).size),
        "n_neg": int(np.asarray(neg_scores).size),
        "mrr": mean_reciprocal_rank(pos_scores, neg_scores),
    }
    for k in ks:
        out[f"hits@{k}"] = hits_at_k(pos_scores, neg_scores, k=k)
    out["leaderboard_metric"] = f"hits@{LEADERBOARD_K}"
    out["leaderboard_value"] = out[f"hits@{LEADERBOARD_K}"]
    return out


def link_prediction_to_markdown(reports: Sequence[Dict[str, object]]) -> str:
    """Markdown table for results/*.md."""
    if not reports:
        return "## OGBL-DDI benchmark\n\n_No results yet._\n"
    ks = [key for key in reports[0] if isinstance(key, str) and key.startswith("hits@")]
    lines = [
        "## OGBL-DDI benchmark (link prediction)",
        "",
        "| Model | Split | " + " | ".join(k.upper() for k in ks) + " | MRR |",
        "| --- " * (3 + len(ks)) + "|",
    ]
    for r in reports:
        cells = " | ".join(f"{float(r[k]):.4f}" for k in ks)
        lines.append(
            f"| {r.get('model_name')} | {r.get('split')} | {cells} | {float(r['mrr']):.4f} |"
        )
    lines += ["", f"Leaderboard metric for {DATASET_NAME} is Hits@{LEADERBOARD_K}."]
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- #
# Environment check
# --------------------------------------------------------------------------- #


@dataclass
class EnvReport:
    available: Dict[str, Optional[str]] = field(default_factory=dict)
    missing: List[str] = field(default_factory=list)
    install_hint: str = ""

    @property
    def ready(self) -> bool:
        return not self.missing

    def __str__(self) -> str:
        lines = ["OGBL-DDI environment check", "-" * 34]
        for pkg, ver in self.available.items():
            lines.append(f"  [ok]      {pkg:<20} {ver or ''}")
        for pkg in self.missing:
            lines.append(f"  [MISSING] {pkg}")
        lines.append("")
        lines.append("READY" if self.ready else f"NOT READY - {self.install_hint}")
        return "\n".join(lines)


def check_environment(packages: Sequence[str] = ("numpy", "sklearn", "torch", "ogb")) -> EnvReport:
    """Report which benchmark dependencies are importable. Never raises."""
    report = EnvReport()
    for pkg in packages:
        try:
            mod = importlib.import_module(pkg)
            report.available[pkg] = getattr(mod, "__version__", "unknown")
        except Exception:
            report.missing.append(pkg)
    if report.missing:
        report.install_hint = "pip install " + " ".join(
            {"sklearn": "scikit-learn"}.get(p, p) for p in report.missing
        )
    return report


def ogb_available() -> bool:
    try:
        importlib.import_module("ogb")
        return True
    except Exception:
        return False


# --------------------------------------------------------------------------- #
# Dataset loading (only touches ogb when asked)
# --------------------------------------------------------------------------- #


def load_ogbl_ddi(root: Union[str, Path] = DEFAULT_ROOT, download: bool = True):
    """
    Load the real ogbl-ddi dataset and its official edge split.

    Returns (dataset, split_edge). Raises RuntimeError with an install hint if
    ogb is not present, so callers can fall back to the local Hits@K path.
    """
    if not ogb_available():
        raise RuntimeError(
            "ogb is not installed - run `pip install ogb` (it needs torch). "
            "Metric functions in this module work without it."
        )
    root = Path(root)
    if not download and not root.exists():
        raise FileNotFoundError(f"{root} does not exist and download=False")
    root.mkdir(parents=True, exist_ok=True)

    from ogb.linkproppred import PygLinkPropPredDataset  # type: ignore

    dataset = PygLinkPropPredDataset(name=DATASET_NAME, root=str(root))
    split_edge = dataset.get_edge_split()
    return dataset, split_edge


def official_evaluator():
    """OGB's own Evaluator, for cross-checking our Hits@K implementation."""
    if not ogb_available():
        raise RuntimeError("ogb is not installed - `pip install ogb`")
    from ogb.linkproppred import Evaluator  # type: ignore

    return Evaluator(name=DATASET_NAME)


def dataset_stats(root: Union[str, Path] = DEFAULT_ROOT) -> Dict[str, object]:
    """Node/edge counts and split sizes, for the paper's dataset table."""
    dataset, split_edge = load_ogbl_ddi(root=root, download=True)
    data = dataset[0]
    stats: Dict[str, object] = {
        "dataset": DATASET_NAME,
        "num_nodes": int(data.num_nodes),
        "num_edges": int(data.edge_index.shape[1]),
    }
    for name in ("train", "valid", "test"):
        part = split_edge.get(name, {})
        stats[f"{name}_pos_edges"] = int(part["edge"].shape[0]) if "edge" in part else 0
        stats[f"{name}_neg_edges"] = (
            int(part["edge_neg"].shape[0]) if "edge_neg" in part else 0
        )
    return stats


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="OGBL-DDI benchmark environment")
    ap.add_argument("--check", action="store_true", help="report dependency status")
    ap.add_argument("--download", action="store_true", help="download the dataset")
    ap.add_argument("--stats", action="store_true", help="print dataset statistics")
    ap.add_argument("--root", default=str(DEFAULT_ROOT))
    args = ap.parse_args(argv)

    if not any([args.check, args.download, args.stats]):
        args.check = True

    if args.check:
        print(check_environment())

    if args.download or args.stats:
        try:
            if args.download:
                load_ogbl_ddi(root=args.root, download=True)
                print(f"\ndataset ready under {args.root}")
            if args.stats:
                print("\n" + json.dumps(dataset_stats(root=args.root), indent=2))
        except Exception as exc:
            print(f"\nfailed: {exc}")
            return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
