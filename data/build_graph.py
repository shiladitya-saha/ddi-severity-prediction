"""
build_graph.py
Owner: M1 (Data Lead)

Constructs the raw drug-drug interaction graph from `data/raw/drugbank_raw.parquet`:
- Assigns contiguous integer node indices [0, ..., N-1] to unique drugs.
- Builds the graph edge index (shape [2, E]), representing interactions.
- Preserves raw mechanism interaction texts and drug metadata.
- Outputs:
    - data/raw/raw_graph.pt (PyTorch Data / dictionary containing edge_index and node IDs)
    - data/raw/drug_mapping.json (drug_id <-> node_idx mapping)
    - data/raw/raw_graph_stats.json (graph statistics summary)
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd
import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("build_graph")

RAW_DIR = Path(__file__).resolve().parent / "raw"
DEFAULT_INPUT_PARQUET = RAW_DIR / "drugbank_raw.parquet"
DEFAULT_INPUT_CSV = RAW_DIR / "drugbank_raw.csv"
DEFAULT_OUTPUT_GRAPH = RAW_DIR / "raw_graph.pt"
DEFAULT_MAPPING_FILE = RAW_DIR / "drug_mapping.json"
DEFAULT_STATS_FILE = RAW_DIR / "raw_graph_stats.json"


def load_raw_interactions(input_path: Path) -> pd.DataFrame:
    """Load raw DrugBank interactions from parquet or csv."""
    if not input_path.exists():
        if DEFAULT_INPUT_CSV.exists():
            input_path = DEFAULT_INPUT_CSV
        else:
            raise FileNotFoundError(
                f"Raw data not found at {input_path}. Please run `download_drugbank.py` first."
            )
    
    logger.info(f"Loading raw interactions from {input_path}")
    if input_path.suffix == ".parquet":
        return pd.read_parquet(input_path)
    return pd.read_csv(input_path)


def build_raw_graph(
    df: pd.DataFrame,
    make_bidirectional: bool = True
) -> Tuple[torch.Tensor, List[str], Dict[str, int], List[str]]:
    """
    Constructs node mapping and edge index from raw interactions dataframe.
    """
    logger.info(f"Total raw interaction records: {len(df):,}")

    # Clean and extract unique drugs
    d1 = df["drug_id_1"].astype(str).str.strip()
    d2 = df["drug_id_2"].astype(str).str.strip()

    unique_drugs = sorted(list(set(d1).union(set(d2))))
    logger.info(f"Identified {len(unique_drugs):,} unique drug nodes.")

    drug_to_idx = {drug: i for i, drug in enumerate(unique_drugs)}

    # Map edges to node indices
    src_nodes = [drug_to_idx[u] for u in d1]
    dst_nodes = [drug_to_idx[v] for v in d2]
    mechanism_texts = df["interaction_text"].fillna("").astype(str).tolist() if "interaction_text" in df.columns else [""] * len(df)

    if make_bidirectional:
        # For DDI, interactions are physically reciprocal unless directional kinetics apply
        # We ensure both (u, v) and (v, u) are represented or track them consistently
        src_all = src_nodes + dst_nodes
        dst_all = dst_nodes + src_nodes
        mechanisms_all = mechanism_texts + mechanism_texts
    else:
        src_all = src_nodes
        dst_all = dst_nodes
        mechanisms_all = mechanism_texts

    edge_index = torch.tensor([src_all, dst_all], dtype=torch.long)
    logger.info(f"Constructed edge_index with shape: {tuple(edge_index.shape)}")

    return edge_index, unique_drugs, drug_to_idx, mechanisms_all


def save_graph_artifacts(
    edge_index: torch.Tensor,
    drug_list: List[str],
    drug_to_idx: Dict[str, int],
    mechanism_texts: List[str],
    output_graph_path: Path,
    mapping_path: Path,
    stats_path: Path
):
    """
    Saves the graph using PyTorch and optionally PyTorch Geometric Data.
    """
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    # Build PyG Data if torch_geometric is available, otherwise use standardized dict
    try:
        from torch_geometric.data import Data
        graph_data = Data(
            edge_index=edge_index,
            drug_id=drug_list,
            raw_mechanism_texts=mechanism_texts,
            num_nodes=len(drug_list)
        )
        torch.save(graph_data, output_graph_path)
        logger.info(f"Saved PyG Data object to {output_graph_path}")
    except ImportError:
        logger.warning("torch_geometric not installed. Saving standard PyTorch dictionary.")
        graph_data = {
            "edge_index": edge_index,
            "drug_id": drug_list,
            "raw_mechanism_texts": mechanism_texts,
            "num_nodes": len(drug_list),
        }
        torch.save(graph_data, output_graph_path)
        logger.info(f"Saved PyTorch dict to {output_graph_path}")

    # Save mapping for downstream components (M2, M3, M4)
    with open(mapping_path, "w") as f:
        json.dump(drug_to_idx, f, indent=2)
    logger.info(f"Saved drug index mapping ({len(drug_to_idx):,} drugs) to {mapping_path}")

    # Compute graph statistics
    num_nodes = len(drug_list)
    num_edges = edge_index.shape[1]
    avg_degree = float(num_edges) / num_nodes if num_nodes > 0 else 0.0

    stats = {
        "num_nodes": num_nodes,
        "num_directed_edges": num_edges,
        "num_unique_pairs": num_edges // 2 if num_edges > 0 else 0,
        "average_node_degree": round(avg_degree, 2),
        "graph_file": str(output_graph_path.name),
        "mapping_file": str(mapping_path.name)
    }

    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    logger.info(f"Saved graph statistics to {stats_path}")


def main():
    parser = argparse.ArgumentParser(description="Build raw DDI graph from DrugBank interactions.")
    parser.add_argument("--input", type=str, default=str(DEFAULT_INPUT_PARQUET), help="Input parquet or csv path")
    parser.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT_GRAPH), help="Output graph path (.pt)")
    parser.add_argument("--no-bidirectional", action="store_true", help="Do not duplicate edges to make undirected")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    df = load_raw_interactions(input_path)
    edge_index, drug_list, drug_to_idx, mechanisms = build_raw_graph(
        df,
        make_bidirectional=not args.no_bidirectional
    )

    save_graph_artifacts(
        edge_index=edge_index,
        drug_list=drug_list,
        drug_to_idx=drug_to_idx,
        mechanism_texts=mechanisms,
        output_graph_path=output_path,
        mapping_path=DEFAULT_MAPPING_FILE,
        stats_path=DEFAULT_STATS_FILE
    )
    logger.info("Raw graph construction complete.")


if __name__ == "__main__":
    main()
