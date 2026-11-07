"""
build_features.py
Owner: M2 — see features/README.md for day/deadline and output contract.

Orchestrates the full Day 2 feature pipeline:
1. fingerprints.extract_all_fingerprints() -> {drug_id: fingerprint_vector}
2. mechanism_extraction.extract_all_mechanism_edges() -> mechanism_edges rows
3. optionally merges embeddings.get_pretrained_embedding() into each drug's
   feature vector (see features/embeddings.py; on by default, --no-embeddings
   to skip). A drug whose embedding fails gets a zero-vector placeholder of
   the same width instead of being left fingerprint-only, so every drug's
   final vector has the same width regardless of per-drug embedding failures
4. saves features/full_features.pt ({drug_id: fingerprint_vector}) and
   features/mechanism_edges.csv (drug_a, drug_b, mechanism_tag, confidence),
   exactly matching the contract in features/README.md
5. validates every drug_id in the graph has a corresponding fingerprint and
   reports any gaps

The graph is loaded once here and passed into both extractors so they don't
each call data_loading.load_graph() independently.
"""

import argparse
import time

import torch

import data_loading
from features import embeddings, fingerprints, mechanism_extraction

FULL_OUTPUT_PATH = "features/full_features.pt"
MECHANISM_OUTPUT_PATH = "features/mechanism_edges.csv"

USE_EMBEDDINGS_DEFAULT = True


def merge_embeddings(feature_dict, drug_to_smiles, *, use_embeddings=USE_EMBEDDINGS_DEFAULT):
    """Optionally append a pretrained ChemBERTa embedding onto each drug's fingerprint.

    Mutates nothing; returns a new {drug_id: vector} dict plus (embedded_count,
    placeholder_count). Controlled by `use_embeddings` -- pass False (or run
    build_features.py with --no-embeddings) to skip entirely and keep
    full_features.pt fingerprint-only.

    Every drug's output vector must stay the same width (models/train.py's
    align_node_features rejects inconsistent widths), so a drug whose
    embedding fails -- missing SMILES, model unavailable, or any per-drug
    inference error -- gets a zero-vector placeholder of the same embedding
    width rather than being left fingerprint-only. Never raises: unexpected
    errors are caught per-drug and counted as placeholders so one bad input
    can't abort the whole build.
    """
    if not use_embeddings:
        return feature_dict, 0, 0

    try:
        embedding_dim = embeddings.get_embedding_dim()
    except Exception as e:
        print(f"[build_features] Could not determine embedding dimension ({e}); "
              "skipping embedding merge -- full_features.pt stays fingerprint-only.")
        return feature_dict, 0, 0

    merged = {}
    embedded = 0
    placeholders = 0
    for drug_id, fp in feature_dict.items():
        smiles = drug_to_smiles.get(drug_id)
        emb = None
        if smiles:
            try:
                emb = embeddings.get_pretrained_embedding(smiles)
            except Exception as e:
                print(f"[build_features] Embedding failed for {drug_id!r} ({e}); "
                      "using a zero placeholder so feature width stays consistent.")
                emb = None
        if emb is None:
            emb = torch.zeros(embedding_dim, dtype=fp.dtype)
            placeholders += 1
        else:
            embedded += 1
        merged[drug_id] = torch.cat([fp, emb])

    if embedded == 0:
        print("[build_features] No pretrained embeddings succeeded; full_features.pt "
              "gained the embedding dimension as zero-padding only (see "
              "features/embeddings.py STATUS note).")
    return merged, embedded, placeholders


def validate_coverage(drug_ids, feature_dict):
    """Confirm every drug_id in the graph has a corresponding fingerprint.

    Returns the list of drug_ids missing from feature_dict (empty if full
    coverage).
    """
    missing = [drug_id for drug_id in drug_ids if drug_id not in feature_dict]
    return missing


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--no-embeddings", dest="use_embeddings", action="store_false", default=USE_EMBEDDINGS_DEFAULT,
        help="skip the optional ChemBERTa embedding merge; full_features.pt stays fingerprint-only",
    )
    args = parser.parse_args(argv)

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

    embedded_count = placeholder_count = 0
    try:
        smiles_lookup = fingerprints.load_smiles_lookup()
        drug_to_smiles = {drug_id: smiles_lookup.get(drug_id) for drug_id in drug_ids}
        feature_dict, embedded_count, placeholder_count = merge_embeddings(
            feature_dict, drug_to_smiles, use_embeddings=args.use_embeddings
        )
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
    if args.use_embeddings:
        print(f"Embeddings merged: {embedded_count} real, {placeholder_count} zero-placeholder")
    print(f"Coverage validation: {len(drug_ids) - len(missing)}/{len(drug_ids)} drug_ids have a fingerprint")
    if missing:
        print(f"  GAP: {len(missing)} drug_id(s) missing a fingerprint: {missing[:10]}"
              f"{' ...' if len(missing) > 10 else ''}")
    print(f"Runtime: {runtime:.2f}s")
    print(f"Saved {FULL_OUTPUT_PATH} and {MECHANISM_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
