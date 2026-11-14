"""
Shared data loader. Everyone downstream (features/, models/, eval/) imports from here.
Owner: M1. Do not change the output schema without telling the team (see data/README.md).

IMPLEMENTED (M2, unblocking features/fingerprints.py and
features/mechanism_extraction.py): M1's real data/processed/nodes.csv and
edges_labeled.csv now exist (confirmed columns: nodes.csv has node_idx,
drug_id, name, smiles, groups; edges_labeled.csv has src, dst, description,
severity, severity_name, rule). load_graph() below reads those directly via
data.data_loading.load_labeled_graph() rather than data/processed/graph.pt,
so it has no torch dependency and matches what's actually on disk.
"""
from pathlib import Path

_HERE = Path(__file__).resolve().parent


def load_graph(proc_dir=None):
    """Return the processed drug-drug interaction graph.

    Schema:
      - drug_id: list[str], one per node, in node_idx order
      - edge_index: [[src_idx, ...], [dst_idx, ...]] (undirected, one direction per row)
      - severity_label: [num_edges] in {0: minor, 1: moderate, 2: major}
      - mechanism_tag: [num_edges] (raw tag = edges_labeled.csv's free-text
        `description` column; refined into 4 categories by
        features/mechanism_extraction.py)
    """
    import sys
    sys.path.insert(0, str(_HERE))
    from data.data_loading import load_labeled_graph

    proc_dir = proc_dir or (_HERE / "data" / "processed")
    nodes, edges = load_labeled_graph(proc_dir)

    drug_id = nodes.sort_values("node_idx")["drug_id"].tolist()
    edge_index = [edges["src"].astype(int).tolist(), edges["dst"].astype(int).tolist()]
    severity_label = edges["severity"].astype(int).tolist()
    mechanism_tag = edges["description"].tolist()

    return {
        "drug_id": drug_id,
        "edge_index": edge_index,
        "severity_label": severity_label,
        "mechanism_tag": mechanism_tag,
    }


def load_class_weights(path=None):
    import json
    path = Path(path) if path else (_HERE / "data" / "processed" / "class_weights.json")
    return json.loads(path.read_text())
