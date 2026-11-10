"""
validate_outputs.py
Owner: M2 — see features/README.md for the full_features.pt / mechanism_edges.csv contract.

Loads features/full_features.pt and features/mechanism_edges.csv back in and
runs sanity checks:
  1. Every drug_id in data/processed/nodes.csv (the canonical node list
     models/train.py loads when graph.pt has no drug_id field -- see
     models/train.py's _load_node_ids) has a corresponding entry in
     full_features.pt. Reports the exact count of any gap.
  2. No NaN, empty, or all-zero fingerprint vectors in full_features.pt
     (all-zero is distinct from an intentional zero-padded embedding tail --
     see build_features.merge_embeddings -- this flags a vector that is
     zero in EVERY dimension, which means fingerprinting itself produced
     nothing usable).
  3. mechanism_edges.csv's edge coverage is consistent with
     data_loading.load_graph()'s edge count: every graph edge must produce
     at least one row (classify_edge() always returns >=1 row per edge --
     one or more confident tags, or a single "unknown" fallback), so the
     number of unique (drug_a, drug_b) pairs in mechanism_edges.csv must
     equal the graph's edge count exactly, and the total row count must be
     >= that (extra rows are fine -- one edge can carry several different
     confident mechanism_tags by design; see mechanism_extraction.classify_edge).
  4. No duplicate (drug_a, drug_b, mechanism_tag) rows in mechanism_edges.csv
     (duplicate mechanism_tags for the SAME pair are a bug; a pair having
     several DIFFERENT mechanism_tags is intentional).
  5. Every drug_id used in full_features.pt / mechanism_edges.csv matches a
     drug_id models/train.py would load (data/processed/nodes.csv, via the
     same _load_node_ids() function models/train.py itself calls) exactly
     -- the classic ID-mismatch bug: case, whitespace, or prefix differences.

main() prints a PASS/FAIL summary per check -- never just "done".
"""

import csv
import sys
from pathlib import Path

import torch

import data_loading

FULL_FEATURES_PATH = "features/full_features.pt"
MECHANISM_EDGES_PATH = "features/mechanism_edges.csv"
NODES_PATH = Path("data/processed/nodes.csv")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _load_canonical_node_ids(nodes_path=NODES_PATH):
    """Return the drug_id list models/train.py would load from nodes.csv.

    Delegates to models.train._load_node_ids so this check uses the exact
    same loader models/train.py calls (same validation: no blanks, no
    duplicates, node_idx must match row order if present) -- not a
    reimplementation that could silently drift from it.
    """
    from models.train import _load_node_ids
    return _load_node_ids(nodes_path)


def check_full_coverage(canonical_ids, feature_dict):
    """Flag every canonical drug_id missing from full_features.pt.

    Returns (passed: bool, issues: list[str], missing_count: int).
    """
    missing = [drug_id for drug_id in canonical_ids if drug_id not in feature_dict]
    issues = [f"{drug_id}: no entry in full_features.pt" for drug_id in missing]
    return len(issues) == 0, issues, len(missing)


def check_vector_sanity(feature_dict):
    """Flag drug_ids whose fingerprint vector is empty, NaN/Inf, or all-zero.

    An all-zero vector is flagged even though a zero-padded ChemBERTa tail is
    intentional for some drugs (see merge_embeddings) -- that padding only
    ever covers part of the vector; the fingerprint half should never be
    entirely zero, so a fully-zero vector means fingerprinting failed
    silently rather than raising.

    Returns (passed: bool, issues: list[str]).
    """
    issues = []
    for drug_id, vec in feature_dict.items():
        if vec.numel() == 0:
            issues.append(f"{drug_id}: empty vector")
            continue
        if torch.isnan(vec).any():
            issues.append(f"{drug_id}: contains NaN")
        if torch.isinf(vec).any():
            issues.append(f"{drug_id}: contains Inf")
        if torch.all(vec == 0):
            issues.append(f"{drug_id}: vector is all-zero")
    return len(issues) == 0, issues


def check_edge_coverage(mech_rows, expected_edge_count):
    """Flag a mismatch between mechanism_edges.csv coverage and the graph's edge count.

    Every graph edge must produce exactly one entry in the set of unique
    (drug_a, drug_b) pairs (classify_edge() never returns zero rows for an
    edge), so that count must equal expected_edge_count exactly. The raw row
    count may exceed it -- edges with multiple confident tags add rows --
    but may never fall short of it.

    Returns (passed: bool, issues: list[str]).
    """
    pairs = {(drug_a, drug_b) for drug_a, drug_b, _tag, _conf in mech_rows}
    issues = []
    if len(pairs) != expected_edge_count:
        issues.append(
            f"unique (drug_a, drug_b) pairs = {len(pairs)}, expected exactly "
            f"{expected_edge_count} (one per graph edge)"
        )
    if len(mech_rows) < expected_edge_count:
        issues.append(
            f"total rows = {len(mech_rows)}, which is fewer than the graph's "
            f"{expected_edge_count} edges -- at least one edge produced no row"
        )
    return len(issues) == 0, issues


def check_no_duplicate_edges(rows):
    """Flag (drug_a, drug_b, mechanism_tag) combinations that appear more than once.

    Multiple DIFFERENT mechanism_tags for the same (drug_a, drug_b) pair are
    intentional (one edge can match several mechanism categories); only an
    exact repeat of (drug_a, drug_b, mechanism_tag) is flagged as a bug.

    Returns (passed: bool, issues: list[str]).
    """
    seen = {}
    for drug_a, drug_b, mechanism_tag, _confidence in rows:
        key = (drug_a, drug_b, mechanism_tag)
        seen[key] = seen.get(key, 0) + 1

    issues = [f"{key}: appears {count} times" for key, count in seen.items() if count > 1]
    return len(issues) == 0, issues


def check_id_consistency(used_ids, canonical_ids):
    """Flag any drug_id used in the outputs that doesn't exactly match a
    canonical drug_id models/train.py would load from nodes.csv.

    A case/whitespace-insensitive match against the canonical set is
    reported specifically as a likely ID-format mismatch (vs. a drug_id
    that doesn't exist in the graph at all).

    Returns (passed: bool, issues: list[str]).
    """
    canonical_set = set(canonical_ids)
    canonical_folded = {str(cid).strip().lower(): cid for cid in canonical_ids}

    issues = []
    for used_id in used_ids:
        if used_id in canonical_set:
            continue
        folded = str(used_id).strip().lower()
        if folded in canonical_folded:
            issues.append(f"{used_id!r}: likely ID-format mismatch vs. canonical {canonical_folded[folded]!r}")
        else:
            issues.append(f"{used_id!r}: not found in graph drug_id list at all")
    return len(issues) == 0, issues


def _load_full_features():
    return torch.load(FULL_FEATURES_PATH)


def _load_mechanism_edges():
    rows = []
    with open(MECHANISM_EDGES_PATH, newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        assert header == ["drug_a", "drug_b", "mechanism_tag", "confidence"], \
            f"Unexpected mechanism_edges.csv header: {header}"
        for drug_a, drug_b, mechanism_tag, confidence in reader:
            rows.append((drug_a, drug_b, mechanism_tag, float(confidence)))
    return rows


def _report(name, passed, issues, limit=10):
    status = "PASS" if passed else "FAIL"
    print(f"[{status}] {name}")
    if not passed:
        for issue in issues[:limit]:
            print(f"    - {issue}")
        if len(issues) > limit:
            print(f"    ... and {len(issues) - limit} more")


def main():
    try:
        feature_dict = _load_full_features()
    except FileNotFoundError:
        print(f"[FAIL] load {FULL_FEATURES_PATH}: file does not exist. "
              "Run features/build_features.py first.")
        feature_dict = None

    try:
        mech_rows = _load_mechanism_edges()
    except FileNotFoundError:
        print(f"[FAIL] load {MECHANISM_EDGES_PATH}: file does not exist. "
              "Run features/build_features.py first.")
        mech_rows = None

    try:
        canonical_ids = _load_canonical_node_ids()
    except (FileNotFoundError, ValueError) as e:
        print(f"[SKIP] checks needing data/processed/nodes.csv ({e})")
        canonical_ids = None

    if feature_dict is not None and canonical_ids is not None:
        passed, issues, missing_count = check_full_coverage(canonical_ids, feature_dict)
        _report(
            f"Full node coverage in full_features.pt "
            f"({len(canonical_ids) - missing_count}/{len(canonical_ids)} covered, {missing_count} missing)",
            passed, issues,
        )
    else:
        print("[SKIP] Full node coverage in full_features.pt (nodes.csv or full_features.pt not loaded)")

    if feature_dict is not None:
        passed, issues = check_vector_sanity(feature_dict)
        _report("No NaN/empty/all-zero fingerprint vectors", passed, issues)
    else:
        print("[SKIP] No NaN/empty/all-zero fingerprint vectors (full_features.pt not loaded)")

    if mech_rows is not None:
        try:
            graph = data_loading.load_graph()
            expected_edge_count = len(graph["edge_index"][0]) if isinstance(graph, dict) else len(graph.edge_index[0])
        except (NotImplementedError, FileNotFoundError) as e:
            print(f"[SKIP] mechanism_edges.csv edge-count consistency ({e})")
            expected_edge_count = None

        if expected_edge_count is not None:
            passed, issues = check_edge_coverage(mech_rows, expected_edge_count)
            _report(
                f"mechanism_edges.csv edge coverage ({len(mech_rows)} rows vs. "
                f"{expected_edge_count} graph edges)",
                passed, issues,
            )
    else:
        print("[SKIP] mechanism_edges.csv edge-count consistency (mechanism_edges.csv not loaded)")

    if mech_rows is not None:
        passed, issues = check_no_duplicate_edges(mech_rows)
        _report("No duplicate (drug_a, drug_b, mechanism_tag) rows", passed, issues)
    else:
        print("[SKIP] No duplicate (drug_a, drug_b, mechanism_tag) rows (mechanism_edges.csv not loaded)")

    if canonical_ids is not None and (feature_dict is not None or mech_rows is not None):
        used_ids = set()
        if feature_dict is not None:
            used_ids.update(feature_dict.keys())
        if mech_rows is not None:
            for drug_a, drug_b, _tag, _conf in mech_rows:
                used_ids.update([drug_a, drug_b])
        passed, issues = check_id_consistency(used_ids, canonical_ids)
        _report("drug_id format matches what models/train.py loads from nodes.csv", passed, issues)
    else:
        print("[SKIP] drug_id format consistency (no outputs or canonical IDs loaded)")


if __name__ == "__main__":
    main()
