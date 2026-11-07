#!/usr/bin/env python3
"""
Smoke-run the whole M5 eval harness on dummy data and write real artifacts.

This is the Day-2 "test eval harness end-to-end on dummy predictions" proof:
it exercises split -> verify -> metrics -> comparison table -> attention
extraction -> figures, and leaves output in results/ that the team can look at
before any model exists.

    python scripts/run_dummy_eval.py
    python scripts/run_dummy_eval.py --edges 5000 --seed 1 --outdir results/dummy
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval import benchmark_severe_subset as ob
from eval.attention_viz import (
    AttentionCapture,
    save_figures,
    top_edges_to_markdown,
)
from tests.dummy_data import FakeAttentionModel, make_dummy_graph, make_dummy_predictions
from eval.metrics import comparison_to_markdown, evaluate, metrics_to_markdown
from eval.split import make_splits, save_splits, verify_splits

MODEL_CONFIGS = [
    # (name, mass on true class, bias toward MAJOR)
    ("baseline_gnn", 0.55, 0.00),
    ("ordinal_gnn", 0.70, 0.10),
    ("mechanism_gnn", 0.82, 0.20),
]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="dummy end-to-end eval run")
    ap.add_argument("--drugs", type=int, default=300)
    ap.add_argument("--edges", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--severe-frac", type=float, default=0.15)
    ap.add_argument("--outdir", default="results/dummy_run")
    args = ap.parse_args(argv)

    outdir = Path(args.outdir)
    figdir = outdir / "explainability_figures"
    outdir.mkdir(parents=True, exist_ok=True)

    print("=" * 68)
    print("M5 eval harness - dummy end-to-end run")
    print("=" * 68)

    # 1. fake data standing in for M1's graph ------------------------------- #
    data = make_dummy_graph(n_drugs=args.drugs, n_edges=args.edges, seed=args.seed)
    print(f"\n[1] dummy graph: {data.n_drugs} drugs, {data.n_edges} edges")
    for cls, name in ((0, "MINOR"), (1, "MODERATE"), (2, "MAJOR")):
        n = int((data.labels == cls).sum())
        print(f"      {name:<9} {n:>6}  ({n / data.n_edges:.1%})")

    # 2. splits ------------------------------------------------------------- #
    splits = make_splits(
        data.edge_index, data.labels, seed=args.seed, severe_benchmark_frac=args.severe_frac
    )
    problems = verify_splits(
        splits, labels=data.labels, edge_index=data.edge_index, strict=False
    )
    print("\n[2] splits")
    print(splits.summary(data.labels))
    print(f"      verification: {'clean' if not problems else problems}")
    save_splits(splits, outdir / "split.json")

    # 3. metrics for three fake models -------------------------------------- #
    print("\n[3] evaluating dummy models")
    results = []
    for name, acc, bias in MODEL_CONFIGS:
        y_pred, proba = make_dummy_predictions(
            data.labels, accuracy=acc, major_bias=bias, seed=args.seed + len(name)
        )
        idx = splits["test"]
        res = evaluate(
            data.labels[idx],
            y_pred[idx],
            y_proba=proba[idx],
            model_name=name,
            split="test",
            notes={"source": "dummy_data", "note": "synthetic - not a real result"},
        )
        results.append(res)
        res.save(outdir / f"{name}_test.json")

        # held-out severe subset, through the benchmark module that owns it
        severe = ob.benchmark_severe(
            data.labels, y_pred, splits, y_proba=proba, model_name=name
        )
        results.append(severe)
        severe.save(outdir / f"{name}_severe_benchmark.json")
        print(
            f"      {name:<16} test: macroF1={res.macro_f1:.3f} MAE={res.mae:.3f} "
            f"MAJORrecall={res.major_recall:.3f}  |  severe subset: "
            f"recall={severe.major_recall:.3f} (n={severe.n_samples})"
        )

    # 4. attention + figures ------------------------------------------------ #
    print("\n[4] attention extraction + figures")
    rng = np.random.default_rng(args.seed)
    alphas = [rng.random((data.n_edges, 4)), rng.random((data.n_edges, 4))]
    model = FakeAttentionModel(data.edge_index, alphas)
    with AttentionCapture(model) as cap:
        model(np.zeros(1))
    print(f"      captured {len(cap)} attention layer(s) -> merged {cap.merged().shape}")

    report = save_figures(
        data.attention,
        data.edge_index,
        mechanism_matrix=data.mechanism_matrix,
        labels=data.labels,
        outdir=figdir,
        model_name="mechanism_gnn",
        drug_names=data.drug_names,
        focus_drug=int(data.edge_index[0, 0]),
        top_k=15,
    )
    for fig in report.figures:
        print(f"      {fig}")
    attr = report.summary["mechanism_attribution"]
    print("      mechanism attention share:")
    for tag, val in sorted(attr.items(), key=lambda kv: -kv[1]):
        print(f"        {tag:<30} {val:.1%}")

    # 5. OGBL-DDI environment ----------------------------------------------- #
    print("\n[5] OGBL-DDI benchmark environment")
    env = ob.check_environment()
    print("      " + str(env).replace("\n", "\n      "))
    pos = rng.normal(1.0, 0.5, 500)
    neg = rng.normal(0.0, 0.5, 5000)
    lp = ob.evaluate_link_prediction(pos, neg, model_name="mechanism_gnn (dummy scores)")
    print(f"      local Hits@20 on dummy scores: {lp['hits@20']:.4f}")

    # 6. write the comparison doc ------------------------------------------- #
    comparison_path = outdir / "results_comparison.md"
    body = [
        "# Dummy-data eval run (M5 harness smoke test)",
        "",
        "> Synthetic data. These numbers mean nothing scientifically - this file",
        "> exists to prove the harness runs end-to-end before real checkpoints land.",
        "",
        comparison_to_markdown(results, title="Model comparison (dummy)"),
        "",
        ob.severe_subset_markdown([r for r in results if r.split == "severe_benchmark"]),
        "",
        ob.link_prediction_to_markdown([lp]),
        "",
        "## Best model detail",
        "",
        metrics_to_markdown(next(r for r in results if r.model_name == "mechanism_gnn")),
        "",
        "## Highest-attention drug pairs",
        "",
        top_edges_to_markdown(report.top_edges[:15]),
        "",
        "## Mechanism attention share",
        "",
        "| Mechanism | Share |",
        "| --- | --- |",
    ]
    for tag, val in sorted(attr.items(), key=lambda kv: -kv[1]):
        body.append(f"| {tag.replace('_', ' ')} | {val:.1%} |")
    body += ["", "## Split sizes", "", "```", splits.summary(data.labels), "```", ""]
    comparison_path.write_text("\n".join(body))

    (outdir / "attention_summary.json").write_text(json.dumps(report.summary, indent=2))

    print(f"\n[6] wrote {comparison_path}")
    print(f"      figures in {figdir}")
    print("\ndone - harness is green end-to-end")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
