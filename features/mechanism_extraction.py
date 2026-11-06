"""
mechanism_extraction.py
Owner: M2 — see features/README.md for day/deadline and output contract.

Day 1: function stubs only, to confirm the structure of mechanism tagging
before data/processed/graph.pt exists.

Day 2: each stub reads the raw mechanism_tag field off the edges in
data_loading.load_graph() and turns it into the refined per-mechanism
signal described below, then build_features.py assembles the results
into mechanism_edges.csv (drug_a, drug_b, mechanism_tag, confidence).
"""


def extract_cyp450_inhibition(drug_id):
    """Determine CYP450 enzyme inhibition/induction activity for a drug.

    Input:
        drug_id (str): identifier matching data_loading.load_graph().drug_id

    Output:
        dict: {"cyp450_enzymes": list[str], "effect": str, "confidence": float}
            e.g. {"cyp450_enzymes": ["CYP3A4"], "effect": "inhibitor", "confidence": 0.9}

    TODO (Day 2): derive this from the raw mechanism_tag field on
    data/processed/graph.pt's edges once that file exists.
    """
    raise NotImplementedError


def extract_protein_binding_competition(drug_id):
    """Determine plasma protein binding competition risk for a drug.

    Input:
        drug_id (str): identifier matching data_loading.load_graph().drug_id

    Output:
        dict: {"protein_binding_pct": float, "competes": bool, "confidence": float}
            e.g. {"protein_binding_pct": 98.0, "competes": True, "confidence": 0.8}

    TODO (Day 2): derive this from the raw mechanism_tag field on
    data/processed/graph.pt's edges once that file exists.
    """
    raise NotImplementedError


def extract_cns_depression_flag(drug_id):
    """Flag whether a drug contributes to CNS depression.

    Input:
        drug_id (str): identifier matching data_loading.load_graph().drug_id

    Output:
        dict: {"cns_depressant": bool, "confidence": float}
            e.g. {"cns_depressant": True, "confidence": 0.85}

    TODO (Day 2): derive this from the raw mechanism_tag field on
    data/processed/graph.pt's edges once that file exists.
    """
    raise NotImplementedError


def extract_qt_prolongation_flag(drug_id):
    """Flag whether a drug carries QT interval prolongation risk.

    Input:
        drug_id (str): identifier matching data_loading.load_graph().drug_id

    Output:
        dict: {"qt_risk": bool, "confidence": float}
            e.g. {"qt_risk": True, "confidence": 0.75}

    TODO (Day 2): derive this from the raw mechanism_tag field on
    data/processed/graph.pt's edges once that file exists.
    """
    raise NotImplementedError


def main():
    dummy_drug_id = "aspirin"

    stubs = [
        extract_cyp450_inhibition,
        extract_protein_binding_competition,
        extract_cns_depression_flag,
        extract_qt_prolongation_flag,
    ]

    print(f"Running mechanism extraction stubs on dummy drug_id={dummy_drug_id!r}")
    for stub in stubs:
        try:
            result = stub(dummy_drug_id)
        except NotImplementedError:
            result = "NotImplementedError (Day 2 TODO)"
        print(f"  {stub.__name__}: {result}")


if __name__ == "__main__":
    main()
