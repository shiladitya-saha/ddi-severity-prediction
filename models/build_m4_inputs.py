"""
build_m4_inputs.py
Owner: M4. Stand-in input builder so Model 3 can train before M2's
features/build_features.py runs end to end.

M2's pipeline is blocked on two stubs: features/fingerprints.load_smiles_lookup()
and the root data_loading.load_graph(). M1's source table (Mendeley DDI_data.csv)
has no SMILES, so this script:

  1. looks up a SMILES string for every graph drug in PubChem by DrugBank ID
     (data/raw/pubchem_smiles.csv, resumable, git-ignored)
  2. fingerprints them with M2's own features.fingerprints.build_fingerprints
     (ECFP4, 1024 bits, no ChemBERTa)
  3. tags every edge with M2's own features.mechanism_extraction, using M1's
     interaction description as the raw mechanism tag

Outputs go to models/checkpoints/m4_inputs/ (git-ignored), not features/, so
they are never mistaken for M2's deliverables:
  full_features.pt     {drug_id: 1024-dim fingerprint}; drugs without a usable
                       SMILES are omitted, as in M2's contract
  mechanism_edges.csv  drug_a, drug_b, mechanism_tag, confidence
  summary.json         coverage numbers

Run from the repo root after M1's pipeline (data/processed/nodes.csv and
edges_labeled.csv must exist):
    python -m models.build_m4_inputs
"""

import argparse
import csv
import json
import time
import urllib.error
import urllib.request
from pathlib import Path

import pandas as pd
import torch

PUBCHEM_URL = (
    "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/xref/RegistryID/"
    "{}/property/SMILES/JSON"
)
SMILES_PATH = Path("data/raw/pubchem_smiles.csv")
OUT_DIR = Path("models/checkpoints/m4_inputs")


def fetch_smiles(drug_ids, out_path=SMILES_PATH, delay=0.25):
    """Append drug_id,cid,smiles,status rows to out_path for IDs not yet fetched."""
    out_path = Path(out_path)
    done = set()
    if out_path.exists():
        done = set(pd.read_csv(out_path, dtype=str)["drug_id"])
    todo = [d for d in drug_ids if d not in done]
    new_file = not out_path.exists()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if new_file:
            writer.writerow(["drug_id", "cid", "smiles", "status"])
        for i, drug_id in enumerate(todo, 1):
            request = urllib.request.Request(
                PUBCHEM_URL.format(drug_id), headers={"User-Agent": "ddi-research-client"}
            )
            for attempt in range(4):
                try:
                    with urllib.request.urlopen(request, timeout=30) as response:
                        props = json.load(response)["PropertyTable"]["Properties"][0]
                    writer.writerow([drug_id, props["CID"], props.get("SMILES", ""), "ok"])
                    break
                except urllib.error.HTTPError as e:
                    if e.code == 404:
                        writer.writerow([drug_id, "", "", "not_found"])
                        break
                    time.sleep(2 ** attempt)
                except (urllib.error.URLError, TimeoutError, KeyError, ValueError):
                    time.sleep(2 ** attempt)
            else:
                writer.writerow([drug_id, "", "", "error"])
            f.flush()
            time.sleep(delay)  # PubChem allows at most 5 requests per second
            if i % 200 == 0:
                print(f"PubChem: {i}/{len(todo)}", flush=True)


def build(proc_dir="data/processed", smiles_path=SMILES_PATH, out_dir=OUT_DIR, fetch=True):
    from features.fingerprints import build_fingerprints
    from features.mechanism_extraction import extract_all_mechanism_edges

    proc_dir, out_dir = Path(proc_dir), Path(out_dir)
    nodes = pd.read_csv(proc_dir / "nodes.csv", dtype={"drug_id": str})
    edges = pd.read_csv(proc_dir / "edges_labeled.csv").fillna("")
    drug_ids = nodes["drug_id"].tolist()

    if fetch:
        fetch_smiles(drug_ids, smiles_path)
    lookup = pd.read_csv(smiles_path, dtype=str).fillna("")
    smiles_of = dict(zip(lookup["drug_id"], lookup["smiles"]))
    fingerprints, failures = build_fingerprints({d: smiles_of.get(d, "") for d in drug_ids})

    graph = {
        "drug_id": drug_ids,
        "edge_index": [edges["src"].tolist(), edges["dst"].tolist()],
        "mechanism_tag": edges["description"].tolist(),
    }
    rows, total, confident, unknown = extract_all_mechanism_edges(graph)

    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(fingerprints, out_dir / "full_features.pt")
    with open(out_dir / "mechanism_edges.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["drug_a", "drug_b", "mechanism_tag", "confidence"])
        writer.writerows(rows)

    summary = {
        "drugs": len(drug_ids),
        "drugs_with_fingerprint": len(fingerprints),
        "drugs_without_fingerprint": len(failures),
        "pubchem_status_counts": lookup["status"].value_counts().to_dict(),
        "edges": total,
        "edges_with_mechanism_tag": confident,
        "edges_untagged": unknown,
        "smiles_source": "PubChem PUG REST, xref RegistryID = DrugBank ID",
        "fingerprint": "M2 features.fingerprints (ECFP4, radius 2, 1024 bits), no ChemBERTa",
        "mechanism_tags": "M2 features.mechanism_extraction on M1 interaction descriptions",
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--proc-dir", default="data/processed")
    parser.add_argument("--smiles", default=str(SMILES_PATH))
    parser.add_argument("--out-dir", default=str(OUT_DIR))
    parser.add_argument("--no-fetch", dest="fetch", action="store_false",
                        help="use the existing SMILES file without contacting PubChem")
    args = parser.parse_args(argv)
    build(args.proc_dir, args.smiles, args.out_dir, fetch=args.fetch)


if __name__ == "__main__":
    main()
