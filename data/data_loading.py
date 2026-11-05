"""
data_loading.py
Owner: M1 (Data Lead)

Central data loader module consumed across the team (M2, M3, M4, M5).
Provides standardized accessors for:
  - Raw DrugBank & TWOSIDES records
  - Raw DDI interaction graph
  - Processed severity graph (`processed/graph.pt`) adhering to the team contract
  - Class weights for ordinal loss functions
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
import pandas as pd
import torch

logger = logging.getLogger("data_loading")

DATA_DIR = Path(__file__).resolve().parent
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"


def load_raw_drugbank(raw_dir: Optional[Path] = None) -> pd.DataFrame:
    """
    Loads raw DrugBank interactions as a pandas DataFrame.
    """
    directory = Path(raw_dir) if raw_dir else RAW_DIR
    parquet_path = directory / "drugbank_raw.parquet"
    csv_path = directory / "drugbank_raw.csv"

    if parquet_path.exists():
        return pd.read_parquet(parquet_path)
    elif csv_path.exists():
        return pd.read_csv(csv_path)
    else:
        raise FileNotFoundError(
            f"No raw DrugBank file found in {directory}. Run `download_drugbank.py` first."
        )


def load_raw_twosides(raw_dir: Optional[Path] = None) -> pd.DataFrame:
    """
    Loads raw TWOSIDES dataset as a pandas DataFrame.
    """
    directory = Path(raw_dir) if raw_dir else RAW_DIR
    parquet_path = directory / "twosides_raw.parquet"
    csv_path = directory / "twosides_raw.csv"

    if parquet_path.exists():
        return pd.read_parquet(parquet_path)
    elif csv_path.exists():
        return pd.read_csv(csv_path)
    else:
        raise FileNotFoundError(
            f"No raw TWOSIDES file found in {directory}. Run `download_twosides.py` first."
        )


def load_drug_mapping(raw_dir: Optional[Path] = None) -> Dict[str, int]:
    """
    Loads drug ID to node index mapping.
    """
    directory = Path(raw_dir) if raw_dir else RAW_DIR
    mapping_path = directory / "drug_mapping.json"
    if not mapping_path.exists():
        raise FileNotFoundError(f"Mapping file not found at {mapping_path}. Run `build_graph.py` first.")
    
    with open(mapping_path, "r") as f:
        return json.load(f)


def load_raw_graph(raw_dir: Optional[Path] = None) -> Any:
    """
    Loads raw interaction graph (saved by `build_graph.py`).
    """
    directory = Path(raw_dir) if raw_dir else RAW_DIR
    graph_path = directory / "raw_graph.pt"
    if not graph_path.exists():
        raise FileNotFoundError(f"Raw graph file not found at {graph_path}. Run `build_graph.py` first.")
    
    return torch.load(graph_path, weights_only=False)


def load_processed_graph(processed_dir: Optional[Path] = None, enforce_contract: bool = True) -> Any:
    """
    Loads the processed severity graph (`processed/graph.pt`).
    
    File Contract verification:
      The object must expose:
        - `drug_id` (node drug IDs)
        - `edge_index` (tensor of shape [2, num_edges])
        - `severity_label` (tensor of shape [num_edges] or [num_pairs], values in {0, 1, 2})
        - mechanism tags for edge feature engineering
    """
    directory = Path(processed_dir) if processed_dir else PROCESSED_DIR
    graph_path = directory / "graph.pt"
    if not graph_path.exists():
        raise FileNotFoundError(
            f"Processed graph not found at {graph_path}. Day 2 `label_severity.py` must run first."
        )

    graph_data = torch.load(graph_path, weights_only=False)

    if enforce_contract:
        # Check required attributes whether graph_data is PyG Data or dict
        has_drug_id = hasattr(graph_data, "drug_id") or (isinstance(graph_data, dict) and "drug_id" in graph_data)
        has_edge_index = hasattr(graph_data, "edge_index") or (isinstance(graph_data, dict) and "edge_index" in graph_data)
        has_severity = hasattr(graph_data, "severity_label") or (isinstance(graph_data, dict) and "severity_label" in graph_data)

        if not (has_drug_id and has_edge_index and has_severity):
            missing = []
            if not has_drug_id: missing.append("drug_id")
            if not has_edge_index: missing.append("edge_index")
            if not has_severity: missing.append("severity_label")
            raise ValueError(
                f"Processed graph violates file contract! Missing required attributes: {missing}"
            )

    return graph_data


def load_class_weights(processed_dir: Optional[Path] = None) -> Dict[str, Union[float, List[float]]]:
    """
    Loads class balance weights for focal loss or weighted CE.
    """
    directory = Path(processed_dir) if processed_dir else PROCESSED_DIR
    weights_path = directory / "class_weights.json"
    if not weights_path.exists():
        raise FileNotFoundError(
            f"Class weights not found at {weights_path}. Day 2 `class_weights.py` must run first."
        )

    with open(weights_path, "r") as f:
        return json.load(f)


if __name__ == "__main__":
    print(f"Data loading module initialized.")
    print(f"Raw directory: {RAW_DIR}")
    print(f"Processed directory: {PROCESSED_DIR}")

