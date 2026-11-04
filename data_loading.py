"""
Shared data loader. Everyone downstream (features/, models/, eval/) imports from here.
Owner: M1. Do not change the output schema without telling the team (see data/README.md).
"""

def load_graph(path="data/processed/graph.pt"):
    """Return the processed drug-drug interaction graph.

    Expected schema:
      - drug_id: list[str]
      - edge_index: [2, num_edges]
      - severity_label: [num_edges] in {0: minor, 1: moderate, 2: major}
      - mechanism_tag: [num_edges] (raw tag, refined later in features/mechanism_extraction.py)
    """
    raise NotImplementedError


def load_class_weights(path="data/processed/class_weights.json"):
    raise NotImplementedError
