#!/usr/bin/env python3
"""
Generate the canonical split artifact everyone trains against.

Owner: M5 (Eval / Explainability Lead) - see eval/README.md

Run this ONCE when M1's labeled edges land, commit the JSON, and have every
model load it. Nobody should call make_splits() in their own training script:
if two people generate splits independently the comparison table is void.

    python scripts/make_splits.py --edges data/processed/edges_labeled.csv
    python scripts/make_splits.py --edges ... --out eval/splits/ddi_v1.json

Input: a CSV with one row per interaction. Column names are flexible - the
script looks for the first match of each:
    drug A      : drug_a, drug_1, head, source, src, u, node_idx_a
    drug B      : drug_b, drug_2, tail, target, dst, v, node_idx_b
    severity    : severity, label, y, severity_label, class
Severity may be ints (0/1/2) or strings (minor/moderate/major, any case).

Output: eval/splits/ddi_v1.json plus a human-readable summary next to it.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.metrics import MAJOR, MINOR, MODERATE, SEVERITY_LABELS
from eval.split import (
    dedupe_undirected,
    make_splits,
    map_split_to_directed,
    save_splits,
    verify_splits,
)

DRUG_A_KEYS = ["drug_a", "drug_1", "head", "source", "src", "u", "node_idx_a", "a"]
DRUG_B_KEYS = ["drug_b", "drug_2", "tail", "target", "dst", "v", "node_idx_b", "b"]
LABEL_KEYS = ["severity", "label", "y", "severity_label", "class", "severity_class"]

STRING_LABELS = {
    "minor": MINOR,
    "moderate": MODERATE,
    "major": MAJOR,
    "mild": MINOR,
    "severe": MAJOR,
    "0": MINOR,
    "1": MODERATE,
    "2": MAJOR,
}


def _pick(fieldnames: Sequence[str], candidates: Sequence[str], what: str) -> str:
    lower = {f.lower().strip(): f for f in fieldnames}
    for cand in candidates:
        if cand in lower:
            return lower[cand]
    raise SystemExit(
        f"Could not find the {what} column. Looked for {candidates}; "
        f"the file has {list(fieldnames)}. Rename the column or tell M5 and "
        "the lookup list gets extended."
    )


def read_edges(path: Path) -> Tuple[np.ndarray, np.ndarray, Dict[str, int]]:
    """Read M1's labeled edges into (edge_index (E,2), labels (E,), id map)."""
    if not path.exists():
        raise SystemExit(
            f"{path} does not exist yet.\n"
            "This is the M1 dependency: splits cannot be generated until the "
            "labeled edge list lands. Nothing else in eval/ is blocked."
        )

    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        if not reader.fieldnames:
            raise SystemExit(f"{path} has no header row")
        col_a = _pick(reader.fieldnames, DRUG_A_KEYS, "first drug")
        col_b = _pick(reader.fieldnames, DRUG_B_KEYS, "second drug")
        col_y = _pick(reader.fieldnames, LABEL_KEYS, "severity")

        ids: Dict[str, int] = {}
        pairs: List[Tuple[int, int]] = []
        labels: List[int] = []
        skipped = 0

        def node_id(raw: str) -> int:
            key = raw.strip()
            if key.isdigit():  # already node_idx
                return int(key)
            if key not in ids:
                ids[key] = len(ids)
            return ids[key]

        for row_no, row in enumerate(reader, start=2):
            raw_y = str(row[col_y]).strip().lower()
            if raw_y not in STRING_LABELS:
                try:
                    y = int(float(raw_y))
                except (TypeError, ValueError):
                    skipped += 1
                    continue
                if y not in (MINOR, MODERATE, MAJOR):
                    skipped += 1
                    continue
            else:
                y = STRING_LABELS[raw_y]
            try:
                pairs.append((node_id(row[col_a]), node_id(row[col_b])))
            except (KeyError, AttributeError):
                skipped += 1
                continue
            labels.append(y)

    if not pairs:
        raise SystemExit(f"No usable rows in {path}")
    if skipped:
        print(f"  warning: skipped {skipped} row(s) with unreadable severity", file=sys.stderr)

    return np.array(pairs, dtype=int), np.array(labels, dtype=int), ids


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="generate the canonical DDI split artifact")
    ap.add_argument("--edges", required=True, help="M1's labeled edge CSV")
    ap.add_argument("--out", default="eval/splits/ddi_v1.json")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--severe-frac", type=float, default=0.15)
    ap.add_argument(
        "--keep",
        choices=["max", "first"],
        default="max",
        help="label to keep when the same pair appears twice with different severities",
    )
    args = ap.parse_args(argv)

    print("=" * 68)
    print("M5 split generation")
    print("=" * 68)

    edge_index, labels, id_map = read_edges(Path(args.edges))
    print(f"\n[1] read {edge_index.shape[0]} rows from {args.edges}")
    if id_map:
        print(f"      built a drug_id -> node_idx map for {len(id_map)} drugs")

    # canonical undirected handling, before anything else
    deduped_edges, deduped_labels, kept = dedupe_undirected(
        edge_index, labels, keep=args.keep
    )
    dropped = edge_index.shape[0] - deduped_edges.shape[0]
    print(f"\n[2] canonicalized to (min_id, max_id) and deduped")
    print(f"      {dropped} duplicate/self-loop row(s) removed -> {deduped_edges.shape[0]} unique pairs")
    if dropped:
        print(f"      conflicting labels resolved by keep='{args.keep}'")

    for cls in (MINOR, MODERATE, MAJOR):
        n = int((deduped_labels == cls).sum())
        print(f"      {SEVERITY_LABELS[cls]:<9} {n:>8}  ({n / deduped_labels.size:.1%})")

    splits = make_splits(
        deduped_edges,
        deduped_labels,
        seed=args.seed,
        severe_benchmark_frac=args.severe_frac,
    )
    print(f"\n[3] splits (seed={args.seed}, severe_benchmark_frac={args.severe_frac})")
    print(splits.summary(deduped_labels))

    problems = verify_splits(
        splits, labels=deduped_labels, edge_index=deduped_edges, strict=False
    )
    print(f"\n[4] leakage verification: {'clean' if not problems else 'PROBLEMS'}")
    for p in problems:
        print(f"      - {p}")
    if problems:
        print("\nrefusing to write a leaking split artifact")
        return 1

    splits.meta.update(
        {
            "source_file": str(args.edges),
            "rows_in_source": int(edge_index.shape[0]),
            "unique_pairs": int(deduped_edges.shape[0]),
            "duplicates_removed": int(dropped),
            "dedupe_keep": args.keep,
            "kept_source_rows": kept.tolist()[:0],  # indices omitted for size; see below
        }
    )

    out = Path(args.out)
    save_splits(splits, out)
    np.save(out.with_suffix(".kept_rows.npy"), kept)

    # map the canonical split back onto the source rows, both orientations
    # together, so training can use a bidirectional graph without leaking
    directed = map_split_to_directed(splits, edge_index)
    directed_path = out.with_suffix(".directed.json")
    directed_path.write_text(
        json.dumps(
            {k: v.tolist() for k, v in directed.items() if not k.startswith("_")},
            indent=2,
        )
    )
    unassigned = int(directed.get("_unassigned_count", np.array([0]))[0])
    print(f"\n[5] directed mapping: {directed_path}")
    for name in ("train", "val", "test", "severe_benchmark"):
        if name in directed:
            print(f"      {name:<18}{directed[name].size:>8} source rows")
    if unassigned:
        print(f"      {unassigned} source row(s) unassigned (self-loops)")
    print(f"\n[6] wrote {out}")
    print(f"      wrote {out.with_suffix('.kept_rows.npy')} (row indices kept from the source CSV)")

    summary = out.with_suffix(".summary.md")
    lines = [
        "# Canonical split artifact",
        "",
        f"- source: `{args.edges}`",
        f"- seed: `{args.seed}`",
        f"- severe_benchmark_frac: `{args.severe_frac}`",
        f"- rows in source: {edge_index.shape[0]}",
        f"- unique undirected pairs after dedupe: {deduped_edges.shape[0]} "
        f"({dropped} removed, keep='{args.keep}')",
        "",
        "```",
        splits.summary(deduped_labels),
        "```",
        "",
        "Load it with:",
        "",
        "```python",
        "from eval.split import load_splits",
        f'splits = load_splits("{args.out}")',
        "train_idx, val_idx, test_idx = splits.train, splits.val, splits.test",
        "```",
        "",
        "Indices are positions in the **deduped** edge array, not rows of the source CSV.",
        f"`{out.with_suffix('.kept_rows.npy').name}` maps them back to source rows.",
        "",
        "`severe_benchmark` is MAJOR-only, reserved before the 70/15/15, and must not be",
        "trained or tuned on.",
    ]
    summary.write_text("\n".join(lines))
    print(f"      wrote {summary}")
    print("\ndone - commit the JSON so every model trains on the same partition")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
