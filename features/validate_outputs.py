"""
validate_outputs.py
Owner: M2 — see features/README.md for the full_features.pt / mechanism_edges.csv contract.

Loads features/full_features.pt and features/mechanism_edges.csv back in and
runs basic sanity checks:
  1. No NaN or empty fingerprint vectors in full_features.pt
  2. No duplicate (drug_a, drug_b, mechanism_tag) rows in mechanism_edges.csv
     (duplicate mechanism_tags for the SAME pair are a bug; a pair having
     several DIFFERENT mechanism_tags is intentional -- see
     mechanism_extraction.classify_edge)
  3. Every drug_id used in full_features.pt / mechanism_edges.csv matches a
     drug_id in data_loading.load_graph() exactly (the classic ID-mismatch
     bug: case, whitespace, or prefix differences)

main() prints a PASS/FAIL summary per check.
"""

import csv

import torch

import data_loading

FULL_FEATURES_PATH = "features/full_features.pt"
MECHANISM_EDGES_PATH = "features/mechanism_edges.csv"


def check_no_nan_or_empty(feature_dict):
    """Flag drug_ids whose fingerprint vector is empty or contains NaN/Inf.

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
    canonical drug_id from data_loading.load_graph().

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

    if feature_dict is not None:
        passed, issues = check_no_nan_or_empty(feature_dict)
        _report("No NaN/empty fingerprint vectors", passed, issues)
    else:
        print("[SKIP] No NaN/empty fingerprint vectors (full_features.pt not loaded)")

    if mech_rows is not None:
        passed, issues = check_no_duplicate_edges(mech_rows)
        _report("No duplicate (drug_a, drug_b, mechanism_tag) rows", passed, issues)
    else:
        print("[SKIP] No duplicate (drug_a, drug_b, mechanism_tag) rows (mechanism_edges.csv not loaded)")

    if feature_dict is not None or mech_rows is not None:
        try:
            graph = data_loading.load_graph()
            canonical_ids = graph["drug_id"] if isinstance(graph, dict) else graph.drug_id
        except NotImplementedError:
            print("[SKIP] drug_id format consistency (data_loading.load_graph() not implemented yet)")
            canonical_ids = None
        except FileNotFoundError as e:
            print(f"[SKIP] drug_id format consistency ({e})")
            canonical_ids = None

        if canonical_ids is not None:
            used_ids = set()
            if feature_dict is not None:
                used_ids.update(feature_dict.keys())
            if mech_rows is not None:
                for drug_a, drug_b, _tag, _conf in mech_rows:
                    used_ids.update([drug_a, drug_b])
            passed, issues = check_id_consistency(used_ids, canonical_ids)
            _report("drug_id format matches data_loading.load_graph()", passed, issues)
    else:
        print("[SKIP] drug_id format consistency (no outputs loaded)")


if __name__ == "__main__":
    main()
