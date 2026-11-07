"""Prep TWOSIDES as the cross-dataset validation set (consumed by M5).

Reads : data/raw/twosides_raw.csv.gz          (full TWOSIDES, streamed in chunks)
        data/processed/nodes.csv, data/processed/edges_labeled.csv   (Day 2 output)
        --mapping optional CSV (rxnorm_id,drugbank_id) to supplement name matching
Writes: data/processed/twosides_processed.csv   all pairs w/ signal counts + mapping flags
        data/processed/twosides_validation.csv  mapped pairs with a heuristic severity (M5 uses this)
        data/processed/twosides_summary.json    coverage / thresholds / overlap stats

WHAT THIS IS (and is not). TWOSIDES lists side-effect *signals* mined from FAERS for drug
pairs; it has NO severity labels. We derive a heuristic pair severity from the signals:
  signal      : PRR >= --min-prr  and  A >= --min-count   (A = co-report count, if present)
  serious AE  : signal whose condition name matches SERIOUS_TERMS (QT, haemorrhage, ...)
  major       : >= --major-min-serious serious AE signals
  moderate    : no serious AE, but >= --moderate-min-sig signals
  minor       : 1..(moderate-min-sig - 1) signals
  (none)      : no signal -> excluded from the validation file
Drugs are linked to the DrugBank graph by NORMALISED NAME (RxNorm id != DrugBank id);
coverage is reported in the summary. FAERS reports are confounded (indication, polypharmacy),
so use this for cross-dataset sanity checks (rank agreement on overlapping pairs, enrichment of
predicted-major among serious-AE pairs), not as ground truth.
"""
import argparse
import json
import re
from pathlib import Path
import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent
RAW_DIR = DATA_DIR / "raw"
PROC_DIR = DATA_DIR / "processed"
SEVERITY_NAMES = ["minor", "moderate", "major"]

SERIOUS_TERMS = [
    "haemorrhag", "hemorrhag", "qt ", "qt-", "long qt", "torsade", "ventricular", "cardiac arrest",
    "myocardial infarction", "serotonin syndrome", "rhabdomyolysis", "respiratory depress",
    "respiratory failure", "respiratory arrest", "anaphyla", "stevens-johnson",
    "toxic epidermal", "agranulocytosis", "pancytopenia", "bone marrow", "hepatic failure",
    "liver failure", "renal failure", "kidney injury", "acute kidney", "convulsion", "seizure",
    "coma", "sudden death", "death", "stroke", "cerebrovascular", "pulmonary embolism",
    "thrombosis", "lactic acidosis", "hyperkalaemia", "hyperkalemia", "hypoglyc", "sepsis",
    "angioedema", "cardiac failure", "cardiogenic shock", "torsades",
]
SERIOUS_RE = "|".join(re.escape(t) for t in SERIOUS_TERMS)

SALTS = {"hydrochloride", "hcl", "sodium", "potassium", "sulfate", "sulphate", "mesylate",
         "maleate", "tartrate", "acetate", "phosphate", "citrate", "succinate", "besylate",
         "bromide", "chloride", "calcium", "fumarate", "hydrobromide", "nitrate", "magnesium"}


def norm_name(s):
    s = re.sub(r"\(.*?\)", " ", str(s).lower())
    s = re.sub(r"[^a-z0-9\s\-]", " ", s)
    toks = s.split()
    return " ".join(toks), " ".join(t for t in toks if t not in SALTS)


def _col(cols, *must, exclude=()):
    for c in cols:
        lc = c.lower().replace(" ", "_")
        if all(m in lc for m in must) and not any(e in lc for e in exclude):
            return c
    return None


def detect_columns(cols):
    d = {
        "id1": _col(cols, "drug_1", "id"), "id2": _col(cols, "drug_2", "id"),
        "n1": _col(cols, "drug_1", "name"), "n2": _col(cols, "drug_2", "name"),
        "cond": _col(cols, "condition", "name") or _col(cols, "condition", exclude=("id",)),
        "prr": next((c for c in cols if c.lower() == "prr"), None),
        "a": next((c for c in cols if c == "A"), None),
    }
    missing = [k for k in ("id1", "id2", "cond", "prr") if not d[k]]
    if missing:
        raise SystemExit(f"Could not detect TWOSIDES columns {missing} in {list(cols)}")
    return d


def aggregate_raw(path, min_prr=2.0, min_count=5, chunksize=500_000):
    head = [c.strip() for c in pd.read_csv(path, nrows=0).columns]
    c = detect_columns(head)
    raw_head = list(pd.read_csv(path, nrows=0).columns)
    strip_map = dict(zip(raw_head, head))
    dtypes = {k: str for k, v in strip_map.items() if v in (c["id1"], c["id2"])}
    parts, names = [], {}
    for ch in pd.read_csv(path, chunksize=chunksize, dtype=dtypes, low_memory=False):
        ch.columns = [x.strip() for x in ch.columns]
        prr = pd.to_numeric(ch[c["prr"]], errors="coerce")
        sig = prr >= min_prr
        if c["a"]:
            sig &= pd.to_numeric(ch[c["a"]], errors="coerce") >= min_count
        serious = ch[c["cond"]].astype(str).str.lower().str.contains(SERIOUS_RE, regex=True)
        if c["n1"] and c["n2"]:
            for idc, nc in ((c["id1"], c["n1"]), (c["id2"], c["n2"])):
                for i, n in ch[[idc, nc]].drop_duplicates().itertuples(index=False):
                    names.setdefault(i, n)
        g = pd.DataFrame({"i1": ch[c["id1"]], "i2": ch[c["id2"]],
                          "sig": sig.astype(int), "ser": (sig & serious).astype(int),
                          "prr_sig": prr.where(sig), "prr_ser": prr.where(sig & serious)})
        parts.append(g.groupby(["i1", "i2"]).agg(
            n_total=("sig", "size"), n_sig=("sig", "sum"), n_serious=("ser", "sum"),
            max_prr_sig=("prr_sig", "max"), max_prr_serious=("prr_ser", "max")).reset_index())
    cat = pd.concat(parts)
    pairs = cat.groupby(["i1", "i2"]).agg(
        n_total=("n_total", "sum"), n_sig=("n_sig", "sum"), n_serious=("n_serious", "sum"),
        max_prr_sig=("max_prr_sig", "max"), max_prr_serious=("max_prr_serious", "max")).reset_index()
    pairs = pairs.rename(columns={"i1": "rxnorm_1", "i2": "rxnorm_2"})
    return pairs, names


def build_name_index(nodes):
    full, stripped = {}, {}
    for idx, name in zip(nodes["node_idx"], nodes["name"]):
        if not str(name).strip():
            continue
        f, s = norm_name(name)
        full.setdefault(f, set()).add(idx)
        stripped.setdefault(s, set()).add(idx)
    uniq = lambda d: {k: next(iter(v)) for k, v in d.items() if len(v) == 1}
    return uniq(full), uniq(stripped)


def map_drugs(names, nodes, mapping_path=None):
    full, stripped = build_name_index(nodes)
    rx2node, how = {}, {}
    if mapping_path:
        m = pd.read_csv(mapping_path, dtype=str)
        db2node = dict(zip(nodes["drug_id"], nodes["node_idx"]))
        for r, d in zip(m["rxnorm_id"], m["drugbank_id"]):
            if d in db2node:
                rx2node[r], how[r] = db2node[d], "mapping_file"
    for r, n in names.items():
        if r in rx2node:
            continue
        f, s = norm_name(n)
        if f in full:
            rx2node[r], how[r] = full[f], "name_exact"
        elif s and s in stripped:
            rx2node[r], how[r] = stripped[s], "name_salt_stripped"
    return rx2node, how


def heuristic_severity(n_sig, n_serious, moderate_min_sig=3, major_min_serious=1):
    if n_sig <= 0:
        return -1
    if n_serious >= major_min_serious:
        return 2
    return 1 if n_sig >= moderate_min_sig else 0


def run(raw_path, proc_dir=PROC_DIR, min_prr=2.0, min_count=5, moderate_min_sig=3,
        mapping_path=None, chunksize=500_000, major_min_serious=1):
    proc_dir = Path(proc_dir)
    nodes = pd.read_csv(proc_dir / "nodes.csv", dtype=str).fillna("")
    nodes["node_idx"] = nodes["node_idx"].astype(int)
    edges = pd.read_csv(proc_dir / "edges_labeled.csv").fillna("")

    pairs, names = aggregate_raw(raw_path, min_prr, min_count, chunksize)
    rx2node, how = map_drugs(names, nodes, mapping_path)
    pairs["name_1"] = pairs["rxnorm_1"].map(names)
    pairs["name_2"] = pairs["rxnorm_2"].map(names)
    n1 = pairs["rxnorm_1"].map(rx2node)
    n2 = pairs["rxnorm_2"].map(rx2node)
    ok = n1.notna() & n2.notna() & (n1 != n2)
    pairs["mapped"] = ok
    pairs["src"] = np.where(ok, np.minimum(n1, n2), -1).astype(int)
    pairs["dst"] = np.where(ok, np.maximum(n1, n2), -1).astype(int)
    pairs.to_csv(proc_dir / "twosides_processed.csv", index=False)

    m = pairs[ok].groupby(["src", "dst"]).agg(
        n_total=("n_total", "sum"), n_sig=("n_sig", "sum"), n_serious=("n_serious", "sum"),
        max_prr_sig=("max_prr_sig", "max"), max_prr_serious=("max_prr_serious", "max")).reset_index()
    m["serious_frac"] = (m["n_serious"] / m["n_sig"].clip(lower=1)).round(4)
    m["twosides_severity"] = [
        heuristic_severity(a, b, moderate_min_sig, major_min_serious)
        for a, b in zip(m["n_sig"], m["n_serious"])
    ]
    m = m[m["twosides_severity"] >= 0].copy()
    m["twosides_severity_name"] = m["twosides_severity"].map(dict(enumerate(SEVERITY_NAMES)))
    e = edges[["src", "dst", "severity", "severity_name", "description"]].rename(
        columns={"severity": "drugbank_severity", "severity_name": "drugbank_severity_name",
                 "description": "drugbank_description"})
    m = m.merge(e, on=["src", "dst"], how="left")
    m["in_drugbank_graph"] = m["drugbank_severity"].notna()
    nm = nodes.set_index("node_idx")["drug_id"]
    m.insert(2, "drugbank_id_1", m["src"].map(nm))
    m.insert(3, "drugbank_id_2", m["dst"].map(nm))
    m.to_csv(proc_dir / "twosides_validation.csv", index=False)

    ov = m[m["in_drugbank_graph"]]
    summary = {
        "thresholds": {"min_prr": min_prr, "min_count": min_count,
                       "moderate_min_sig": moderate_min_sig,
                       "major_min_serious": major_min_serious},
        "raw_pairs": int(len(pairs)),
        "raw_pairs_with_signal": int((pairs["n_sig"] > 0).sum()),
        "twosides_drugs": len(names),
        "twosides_drugs_mapped": len(rx2node),
        "drug_mapping_methods": pd.Series(how).value_counts().to_dict(),
        "pairs_both_drugs_mapped": int(ok.sum()),
        "validation_pairs": int(len(m)),
        "validation_in_drugbank_graph": int(len(ov)),
        "validation_novel_pairs": int(len(m) - len(ov)),
        "twosides_severity_counts": m["twosides_severity_name"].value_counts().to_dict(),
        "overlap_crosstab_drugbank_vs_twosides": (
            pd.crosstab(ov["drugbank_severity_name"], ov["twosides_severity_name"]).to_dict()
            if len(ov) else {}),
    }
    (proc_dir / "twosides_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps(summary, indent=2, default=str))
    return m


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--raw", default=str(RAW_DIR / "twosides_raw.csv.gz"))
    ap.add_argument("--out-dir", default=str(PROC_DIR))
    ap.add_argument("--min-prr", type=float, default=2.0)
    ap.add_argument("--min-count", type=int, default=5)
    ap.add_argument("--moderate-min-sig", type=int, default=3)
    ap.add_argument("--major-min-serious", type=int, default=1,
                    help="serious-AE signals needed for major (default 1; try 3 or 5)")
    ap.add_argument("--mapping", help="optional CSV rxnorm_id,drugbank_id")
    a = ap.parse_args()
    run(a.raw, a.out_dir, a.min_prr, a.min_count, a.moderate_min_sig, a.mapping,
        major_min_serious=a.major_min_serious)