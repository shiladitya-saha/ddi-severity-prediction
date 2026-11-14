"""
mechanism_extraction.py
Owner: M2 — see features/README.md for day/deadline and output contract.

Day 1: function stubs only, to confirm the structure before data/processed/graph.pt
existed.

Day 2: each extract_* function takes an edge (drug_a, drug_b, raw_mechanism_tag)
and refines the raw mechanism_tag data_loading.load_graph() exposes per edge into
one of four categories, with a confidence score. main() runs this across every
edge in the graph and writes mechanism_edges.csv per the contract in
features/README.md: drug_a, drug_b, mechanism_tag, confidence.

KNOWN GAP (same shape as the one in fingerprints.py): classification here is
keyword/pattern matching on the raw mechanism_tag string only. There is no
documented source of richer per-drug metadata yet (known CYP450 substrate
lists, plasma protein-binding percentages, CNS-depressant drug classes,
QT-risk lists) -- data/README.md's contract for graph.pt doesn't include it.
If/when such metadata becomes available, wire it into _match() below as an
extra signal; until then, matches are only as good as the raw tag text.
"""

import csv

import data_loading

CONFIDENCE_THRESHOLD = 0.5
OUTPUT_CSV_PATH = "features/mechanism_edges.csv"

# (strong keywords -> confidence 0.9, weak keywords -> confidence 0.55)
_CYP450_STRONG = ("cyp450", "cyp3a4", "cyp2d6", "cyp2c9", "cyp2c19", "cyp1a2", "cyp inhibit", "cyp induc")
_CYP450_WEAK = ("cyp", "cytochrome", "hepatic metabolism", "metaboli")

_PROTEIN_BINDING_STRONG = ("protein binding", "plasma protein", "albumin binding", "protein-bound displacement")
_PROTEIN_BINDING_WEAK = ("protein bound", "displacement", "albumin")

_CNS_DEPRESSION_STRONG = ("cns depress", "central nervous system depress", "respiratory depress")
_CNS_DEPRESSION_WEAK = ("sedation", "sedative", "drowsiness", "somnolence", "opioid", "benzodiazepine", "alcohol")

_QT_PROLONGATION_STRONG = ("qt prolong", "qtc prolong", "qtc-prolong", "torsades")
_QT_PROLONGATION_WEAK = ("arrhythm", "cardiac conduction", "ventricular")


def _match(raw_mechanism_tag, strong_keywords, weak_keywords):
    """Keyword-match raw_mechanism_tag against strong/weak keyword sets.

    Returns (matched: bool, confidence: float). Strong keywords indicate an
    explicit, specific mention (confidence 0.9); weak keywords indicate a
    plausible but less specific mention (confidence 0.55); no match is
    (False, 0.0).
    """
    if not raw_mechanism_tag:
        return False, 0.0
    text = str(raw_mechanism_tag).lower()
    if any(kw in text for kw in strong_keywords):
        return True, 0.9
    if any(kw in text for kw in weak_keywords):
        return True, 0.55
    return False, 0.0


def extract_cyp450_inhibition(drug_a, drug_b, raw_mechanism_tag):
    """Flag CYP450 enzyme inhibition/induction between a drug pair.

    Input:
        drug_a, drug_b (str): identifiers matching data_loading.load_graph().drug_id
        raw_mechanism_tag: the raw per-edge tag from data_loading.load_graph()

    Output:
        dict: {"mechanism_tag": "cyp450_inhibition" | "unknown", "confidence": float}
    """
    matched, confidence = _match(raw_mechanism_tag, _CYP450_STRONG, _CYP450_WEAK)
    return {"mechanism_tag": "cyp450_inhibition" if matched else "unknown", "confidence": confidence}


def extract_protein_binding_competition(drug_a, drug_b, raw_mechanism_tag):
    """Flag plasma protein binding competition between a drug pair.

    Input:
        drug_a, drug_b (str): identifiers matching data_loading.load_graph().drug_id
        raw_mechanism_tag: the raw per-edge tag from data_loading.load_graph()

    Output:
        dict: {"mechanism_tag": "protein_binding_competition" | "unknown", "confidence": float}
    """
    matched, confidence = _match(raw_mechanism_tag, _PROTEIN_BINDING_STRONG, _PROTEIN_BINDING_WEAK)
    return {"mechanism_tag": "protein_binding_competition" if matched else "unknown", "confidence": confidence}


def extract_cns_depression_flag(drug_a, drug_b, raw_mechanism_tag):
    """Flag combined CNS depression risk between a drug pair.

    Input:
        drug_a, drug_b (str): identifiers matching data_loading.load_graph().drug_id
        raw_mechanism_tag: the raw per-edge tag from data_loading.load_graph()

    Output:
        dict: {"mechanism_tag": "cns_depression" | "unknown", "confidence": float}
    """
    matched, confidence = _match(raw_mechanism_tag, _CNS_DEPRESSION_STRONG, _CNS_DEPRESSION_WEAK)
    return {"mechanism_tag": "cns_depression" if matched else "unknown", "confidence": confidence}


def extract_qt_prolongation_flag(drug_a, drug_b, raw_mechanism_tag):
    """Flag combined QT interval prolongation risk between a drug pair.

    Input:
        drug_a, drug_b (str): identifiers matching data_loading.load_graph().drug_id
        raw_mechanism_tag: the raw per-edge tag from data_loading.load_graph()

    Output:
        dict: {"mechanism_tag": "qt_prolongation" | "unknown", "confidence": float}
    """
    matched, confidence = _match(raw_mechanism_tag, _QT_PROLONGATION_STRONG, _QT_PROLONGATION_WEAK)
    return {"mechanism_tag": "qt_prolongation" if matched else "unknown", "confidence": confidence}


EXTRACTORS = (
    extract_cyp450_inhibition,
    extract_protein_binding_competition,
    extract_cns_depression_flag,
    extract_qt_prolongation_flag,
)


def classify_edge(drug_a, drug_b, raw_mechanism_tag):
    """Run all extractors on one edge.

    Returns a list of {"mechanism_tag", "confidence"} dicts whose confidence
    is >= CONFIDENCE_THRESHOLD. If none meet the threshold, returns a single
    fallback [{"mechanism_tag": "unknown", "confidence": best_confidence}].
    """
    results = [fn(drug_a, drug_b, raw_mechanism_tag) for fn in EXTRACTORS]
    confident = [r for r in results if r["confidence"] >= CONFIDENCE_THRESHOLD]
    if confident:
        return confident
    best_confidence = max((r["confidence"] for r in results), default=0.0)
    return [{"mechanism_tag": "unknown", "confidence": best_confidence}]


def _iter_edges(graph):
    """Yield (drug_a, drug_b, raw_mechanism_tag) for every edge in a loaded graph."""
    if isinstance(graph, dict):
        drug_id = graph["drug_id"]
        edge_index = graph["edge_index"]
        mechanism_tag = graph.get("mechanism_tag")
    else:
        drug_id = graph.drug_id
        edge_index = graph.edge_index
        mechanism_tag = getattr(graph, "mechanism_tag", None)

    num_edges = len(edge_index[0])
    for i in range(num_edges):
        drug_a = drug_id[int(edge_index[0][i])]
        drug_b = drug_id[int(edge_index[1][i])]
        raw_tag = mechanism_tag[i] if mechanism_tag is not None else None
        yield drug_a, drug_b, raw_tag


def extract_all_mechanism_edges(graph=None):
    """Full Day-2 extraction: classify every edge in a loaded graph.

    If `graph` is not given, loads one via data_loading.load_graph(). Lets
    NotImplementedError / FileNotFoundError (graph not ready) propagate to
    the caller -- callers decide how to report a block.

    Returns (rows, total_edges, edges_with_confident_tag, edges_fell_back_to_unknown):
      - rows: [(drug_a, drug_b, mechanism_tag, confidence), ...]
    """
    if graph is None:
        graph = data_loading.load_graph()

    total_edges = 0
    edges_with_confident_tag = 0
    edges_fell_back_to_unknown = 0
    rows = []

    for drug_a, drug_b, raw_tag in _iter_edges(graph):
        total_edges += 1
        tags = classify_edge(drug_a, drug_b, raw_tag)
        if tags[0]["mechanism_tag"] == "unknown":
            edges_fell_back_to_unknown += 1
        else:
            edges_with_confident_tag += 1
        for tag in tags:
            rows.append((drug_a, drug_b, tag["mechanism_tag"], tag["confidence"]))

    return rows, total_edges, edges_with_confident_tag, edges_fell_back_to_unknown


def main():
    try:
        rows, total_edges, edges_with_confident_tag, edges_fell_back_to_unknown = extract_all_mechanism_edges()
    except NotImplementedError as e:
        reason = str(e) or ("data_loading.load_graph() is not implemented yet "
                             "(data/processed/graph.pt is not available)")
        print(f"BLOCKED: {reason}. Not proceeding with mechanism extraction.")
        return
    except FileNotFoundError as e:
        print(f"BLOCKED: {e}. Not proceeding with mechanism extraction.")
        return

    with open(OUTPUT_CSV_PATH, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["drug_a", "drug_b", "mechanism_tag", "confidence"])
        writer.writerows(rows)

    print(f"Total edges: {total_edges}")
    print(f"Confident tag: {edges_with_confident_tag}")
    print(f"Fell back to unknown: {edges_fell_back_to_unknown}")
    print(f"Saved output to {OUTPUT_CSV_PATH}")


if __name__ == "__main__":
    main()
