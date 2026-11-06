"""
build_features.py
Owner: M2 — see features/README.md for day/deadline and output contract.

Orchestrates the full Day 2 feature pipeline:
1. fingerprints.extract_all_fingerprints() -> {drug_id: fingerprint_vector}
2. mechanism_extraction.extract_all_mechanism_edges() -> mechanism_edges rows
3. optionally merges embeddings.get_pretrained_embedding() into each drug's
   feature vector if ChemBERTa is available (see features/embeddings.py --
   optional, deferred to Day 3/4 in this environment today)
4. saves features/full_features.pt ({drug_id: fingerprint_vector}) and
   features/mechanism_edges.csv (drug_a, drug_b, mechanism_tag, confidence),
   exactly matching the contract in features/README.md
5. validates every drug_id in the graph has a corresponding fingerprint and
   reports any gaps

The graph is loaded once here and passed into both extractors so they don't
each call data_loading.load_graph() independently.
"""

import time

import torch

import data_loading
from features import embeddings, fingerprints, mechanism_extraction

FULL_OUTPUT_PATH = "features/full_features.pt"
MECHANISM_OUTPUT_PATH = "features/mechanism_edges.csv"


def merge_embeddings(feature_dict, drug_to_smiles):
    """Optionally append a pretrained embedding onto each drug's fingerprint vector.

    Mutates nothing; returns a new {drug_id: vector} dict. For a drug_id whose
    embedding is available, the fingerprint and embedding are concatenated
    into one vector (full_features.pt stays {drug_id: fingerprint_vector} --
    just a richer vector per drug). If no embedding succeeds for any drug
    (e.g. ChemBERTa unavailable, as noted in features/embeddings.py), returns
    feature_dict unchanged.
    """
    merged = {}
    any_embedded = False
    for drug_id, fp in feature_dict.items():
        smiles = drug_to_smiles.get(drug_id)
        emb = embeddings.get_pretrained_embedding(smiles) if smiles else None
        if emb is not None:
            merged[drug_id] = torch.cat([fp, emb])
            any_embedded = True
        else:
            merged[drug_id] = fp

    if not any_embedded:
        print("[build_features] No pretrained embeddings available; full_features.pt "
              "contains fingerprints only (see features/embeddings.py STATUS note).")
    return merged


def validate_coverage(drug_ids, feature_dict):
    """Confirm every drug_id in the graph has a corresponding fingerprint.

    Returns the list of drug_ids missing from feature_dict (empty if full
    coverage).
    """
    missing = [drug_id for drug_id in drug_ids if drug_id not in feature_dict]
    return missing


def main():
    start = time.time()

    try:
        graph = data_loading.load_graph()
    except NotImplementedError:
        print("BLOCKED: data_loading.load_graph() is not implemented yet "
              "(data/processed/graph.pt is not available). Not proceeding with "
              "feature pipeline.")
        return
    except FileNotFoundError as e:
        print(f"BLOCKED: {e}. Not proceeding with feature pipeline.")
        return

    try:
        feature_dict, fp_failures, drug_ids = fingerprints.extract_all_fingerprints(graph)
    except NotImplementedError as e:
        print(f"BLOCKED (fingerprints): {e}. Not proceeding with feature pipeline.")
        return

    mech_rows, total_edges, mech_confident, mech_unknown = mechanism_extraction.extract_all_mechanism_edges(graph)

    try:
        smiles_lookup = fingerprints.load_smiles_lookup()
        drug_to_smiles = {drug_id: smiles_lookup.get(drug_id) for drug_id in drug_ids}
        feature_dict = merge_embeddings(feature_dict, drug_to_smiles)
    except NotImplementedError:
        print("[build_features] Skipping embedding merge: no drug_id -> SMILES source "
              "wired up yet (same gap noted in features/fingerprints.py).")

    torch.save(feature_dict, FULL_OUTPUT_PATH)

    import csv
    with open(MECHANISM_OUTPUT_PATH, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["drug_a", "drug_b", "mechanism_tag", "confidence"])
        writer.writerows(mech_rows)

    missing = validate_coverage(drug_ids, feature_dict)

    runtime = time.time() - start

    print("=== build_features.py summary ===")
    print(f"Total drugs in graph: {len(drug_ids)}")
    print(f"Fingerprints succeeded: {len(feature_dict)}")
    print(f"Fingerprints failed: {len(fp_failures)}")
    print(f"Total edges: {total_edges}")
    print(f"Mechanism edges with confident tag: {mech_confident}")
    print(f"Mechanism edges fell back to unknown: {mech_unknown}")
    print(f"Coverage validation: {len(drug_ids) - len(missing)}/{len(drug_ids)} drug_ids have a fingerprint")
    if missing:
        print(f"  GAP: {len(missing)} drug_id(s) missing a fingerprint: {missing[:10]}"
              f"{' ...' if len(missing) > 10 else ''}")
    print(f"Runtime: {runtime:.2f}s")
    print(f"Saved {FULL_OUTPUT_PATH} and {MECHANISM_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
