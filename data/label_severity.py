"""Attach minor / moderate / major severity labels to every edge.

Reads : data/raw/drugbank_graph_nodes.csv, data/raw/drugbank_graph_edges.csv
        data/severity_overrides.csv (optional: description,severity  -> wins over rules)
Writes: data/processed/edges_labeled.csv   src,dst,description,severity,severity_name,rule
        data/processed/nodes.csv
        data/processed/severity_mapping.csv  one row per distinct interaction type (REVIEW THIS)
        data/processed/graph.pt              only if torch is installed

HEURISTIC LABELS. The source gives an interaction *type* (no direction, no drug-specific
context), so severity is derived from the clinical endpoint named in the type:
  major    : life-threatening endpoints (QT/arrhythmia, serotonin syndrome, bleeding,
             respiratory depression, organ toxicity, seizures, ...)
  minor    : pharmacokinetic-only changes (absorption, metabolism, serum concentration, ...)
  moderate : everything else (pharmacodynamic effects, efficacy changes, generic adverse
             effects) -- also the fallback.
Order 0 < 1 < 2 is meaningful (ordinal). Rules are checked major -> minor -> moderate.
Merged edges ("a || b") take the MAX severity of their parts.
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent
RAW_DIR = DATA_DIR / "raw"
PROC_DIR = DATA_DIR / "processed"
OVERRIDES = DATA_DIR / "severity_overrides.csv"
SEVERITY_NAMES = ["minor", "moderate", "major"]

MAJOR = [
    "qtc", "qt prolong", "torsade", "arrhythm", "serotonin", "serotonerg",
    "bleeding", "hemorrhag", "haemorrhag", "thrombo", "respiratory depress",
    "cardiotox", "hyperkal", "rhabdomyolysis", "myopathy", "agranulocyt",
    "neutropen", "myelosuppress", "hepatotox", "nephrotox", "neurotox",
    "ototox", "seizure", "convuls", "lactic acidosis", "methemoglobin",
    "immunosuppress", "hypertensive crisis", "hyperglyc", "hypoglyc",
]
MINOR = [
    "absorption", "bioavailab", "excretion", "serum concentration", "metabolism",
    "protein binding", "distribution", "elimination", "clearance", "half-life",
    "cmax", "auc", "active metabolite", "urinary",
]


def load_overrides(path=OVERRIDES):
    path = Path(path)
    if not path.exists():
        return {}
    df = pd.read_csv(path, dtype=str).fillna("")
    out = {}
    for d, s in zip(df.get("description", []), df.get("severity", [])):
        s = s.strip().lower()
        if s in SEVERITY_NAMES:
            out[d.strip().lower()] = SEVERITY_NAMES.index(s)
        elif s in ("0", "1", "2"):
            out[d.strip().lower()] = int(s)
    return out


def classify_one(text, overrides=None):
    t = (text or "").strip().lower()
    if not t:
        return 1, "empty->moderate"
    if overrides and t in overrides:
        return overrides[t], "override"
    for p in MAJOR:
        if p in t:
            return 2, f"major:{p}"
    for p in MINOR:
        if p in t:
            return 0, f"minor:{p}"
    return 1, "default:moderate"


def classify(description, overrides=None):
    parts = [p for p in str(description).split(" || ") if p.strip()] or [""]
    results = [classify_one(p, overrides) for p in parts]
    return max(results, key=lambda r: r[0])  # most severe part wins


def label_edges(edges, overrides=None):
    types = edges["description"].astype(str).unique()
    lut = {t: classify(t, overrides) for t in types}
    out = edges.copy()
    out["severity"] = out["description"].astype(str).map(lambda t: lut[t][0]).astype(int)
    out["rule"] = out["description"].astype(str).map(lambda t: lut[t][1])
    out["severity_name"] = out["severity"].map(dict(enumerate(SEVERITY_NAMES)))
    return out[["src", "dst", "description", "severity", "severity_name", "rule"]]


def main(raw_dir=RAW_DIR, out_dir=PROC_DIR, overrides_path=OVERRIDES):
    raw_dir, out_dir = Path(raw_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    nodes = pd.read_csv(raw_dir / "drugbank_graph_nodes.csv", dtype={"drug_id": str, "name": str, "smiles": str, "groups": str}).fillna("")
    edges = pd.read_csv(raw_dir / "drugbank_graph_edges.csv").fillna("")
    lab = label_edges(edges, load_overrides(overrides_path))
    lab.to_csv(out_dir / "edges_labeled.csv", index=False)
    nodes.to_csv(out_dir / "nodes.csv", index=False)

    mapping = (lab.groupby(["description", "severity", "severity_name", "rule"])
               .size().reset_index(name="n_edges").sort_values("n_edges", ascending=False))
    mapping.to_csv(out_dir / "severity_mapping.csv", index=False)

    counts = lab["severity_name"].value_counts().reindex(SEVERITY_NAMES).fillna(0).astype(int)
    print("edges:", len(lab), "| severity counts:", counts.to_dict())
    print("share  :", (counts / counts.sum()).round(3).to_dict())
    print(f"fallback-to-moderate edges: {(lab['rule'] == 'default:moderate').mean():.1%} "
          f"-> review {out_dir / 'severity_mapping.csv'}")

    try:
        import torch
        ei = np.stack([np.concatenate([lab["src"], lab["dst"]]),
                       np.concatenate([lab["dst"], lab["src"]])]).astype(np.int64)
        sev = np.concatenate([lab["severity"], lab["severity"]]).astype(np.int64)
        torch.save({"edge_index": torch.from_numpy(ei), "edge_severity": torch.from_numpy(sev),
                    "num_nodes": len(nodes), "severity_names": SEVERITY_NAMES},
                   out_dir / "graph.pt")
        print("wrote graph.pt")
    except ImportError:
        print("torch not installed: skipped graph.pt (edges_labeled.csv is the canonical output)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", default=str(RAW_DIR))
    ap.add_argument("--out-dir", default=str(PROC_DIR))
    a = ap.parse_args()
    main(a.raw_dir, a.out_dir)