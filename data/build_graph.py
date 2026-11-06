"""Build the raw DrugBank interaction graph (nodes=drugs, edges=interactions).

Reads : data/raw/drugbank_drugs_raw.csv, data/raw/drugbank_raw.csv
Writes: data/raw/drugbank_graph_nodes.csv  (node_idx, drug_id, name, smiles, groups)
        data/raw/drugbank_graph_edges.csv  (src, dst, description)  undirected, deduped
        data/raw/drugbank_graph.npz        edge_index (2, 2E), both directions
Edges with an endpoint not in the drug table are dropped. A->B / B->A descriptions
are merged with ' || ' so Day-2 severity labelling sees all of them.
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

RAW_DIR = Path(__file__).resolve().parent / "raw"


def build(raw_dir=RAW_DIR):
    raw_dir = Path(raw_dir)
    drugs = pd.read_csv(raw_dir / "drugbank_drugs_raw.csv", dtype=str).fillna("")
    drugs = drugs.drop_duplicates("drug_id").reset_index(drop=True)
    drugs.insert(0, "node_idx", np.arange(len(drugs)))
    idx = dict(zip(drugs["drug_id"], drugs["node_idx"]))

    e = pd.read_csv(raw_dir / "drugbank_raw.csv", dtype=str).fillna("")
    n0 = len(e)
    e["src"] = e["drug_id_1"].map(idx)
    e["dst"] = e["drug_id_2"].map(idx)
    e = e.dropna(subset=["src", "dst"])
    e = e[e["src"] != e["dst"]].copy()
    e["src"], e["dst"] = e["src"].astype(int), e["dst"].astype(int)
    a = e[["src", "dst"]].min(axis=1)
    b = e[["src", "dst"]].max(axis=1)
    e["src"], e["dst"] = a, b
    e = (e.groupby(["src", "dst"])["description"]
         .agg(lambda s: " || ".join(dict.fromkeys(x for x in s if x)))
         .reset_index())
    print(f"edges: {n0} raw -> {len(e)} undirected unique ({n0 - len(e)} dropped/merged)")

    drugs.to_csv(raw_dir / "drugbank_graph_nodes.csv", index=False)
    e.to_csv(raw_dir / "drugbank_graph_edges.csv", index=False)
    ei = np.stack([np.concatenate([e["src"], e["dst"]]),
                   np.concatenate([e["dst"], e["src"]])]).astype(np.int64)
    np.savez_compressed(raw_dir / "drugbank_graph.npz", edge_index=ei, num_nodes=len(drugs))
    print(f"{len(drugs)} nodes written to {raw_dir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", default=str(RAW_DIR))
    build(ap.parse_args().raw_dir)