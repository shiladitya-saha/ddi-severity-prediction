"""Shared loaders for downstream code: `from data.data_loading import load_graph`."""
from pathlib import Path
import numpy as np
import pandas as pd

RAW_DIR = Path(__file__).resolve().parent / "raw"


def _need(p):
    p = Path(p)
    if not p.exists():
        raise FileNotFoundError(f"{p} missing. Run data/download_*.py and data/build_graph.py first.")
    return p


def load_drugbank(raw_dir=RAW_DIR):
    raw_dir = Path(raw_dir)
    drugs = pd.read_csv(_need(raw_dir / "drugbank_drugs_raw.csv"), dtype=str).fillna("")
    edges = pd.read_csv(_need(raw_dir / "drugbank_raw.csv"), dtype=str).fillna("")
    return drugs, edges


def load_twosides(pairs=True, raw_dir=RAW_DIR):
    raw_dir = Path(raw_dir)
    if pairs:
        return pd.read_csv(_need(raw_dir / "twosides_pairs_raw.csv"))
    return pd.read_csv(_need(raw_dir / "twosides_raw.csv.gz"), low_memory=False)


def load_graph(raw_dir=RAW_DIR):
    raw_dir = Path(raw_dir)
    nodes = pd.read_csv(_need(raw_dir / "drugbank_graph_nodes.csv"), dtype={"drug_id": str}).fillna("")
    edges = pd.read_csv(_need(raw_dir / "drugbank_graph_edges.csv")).fillna("")
    edge_index = np.load(_need(raw_dir / "drugbank_graph.npz"))["edge_index"]
    return nodes, edges, edge_index


def to_pyg(raw_dir=RAW_DIR):
    import torch
    from torch_geometric.data import Data
    nodes, _, ei = load_graph(raw_dir)
    return Data(edge_index=torch.from_numpy(ei), num_nodes=len(nodes))