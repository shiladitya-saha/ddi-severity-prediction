"""
fingerprints.py
Owner: M2 — see features/README.md for day/deadline and output contract.

Day 1: converted a hardcoded 18-drug test set (drug_id -> SMILES) into Morgan/ECFP4
fingerprints. Output: features/sample_fingerprints.pt (kept untouched as a reference;
nothing below writes to that path anymore).

Day 2: main() now loads the full drug list from data_loading.load_graph() and
generates a fingerprint for every drug, saving features/full_features.pt per the
full_features.pt contract: {drug_id: fingerprint_vector}.

KNOWN GAP: data_loading.load_graph()'s documented schema is drug_id, edge_index,
severity_label, mechanism_tag -- no SMILES field. Fingerprinting needs a
drug_id -> SMILES lookup that doesn't exist in the current data/ contract.
load_smiles_lookup() below is a stub marking where that needs to be wired up
(e.g. parsed from data/raw/drugbank_raw.* by download_drugbank.py). Until
that's resolved, main() will report the block clearly rather than guessing.
"""

import time

import torch
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator

import data_loading

FINGERPRINT_RADIUS = 2
FINGERPRINT_NBITS = 1024

FAILURE_LOG_PATH = "features/fingerprint_failures.log"
FULL_OUTPUT_PATH = "features/full_features.pt"

# Day-1 reference set: real drugs with accurate SMILES strings. Kept for
# smoke-testing smiles_to_fingerprint() independent of data_loading; main()
# no longer uses this and no longer writes features/sample_fingerprints.pt.
TEST_DRUGS = {
    "aspirin": "CC(=O)OC1=CC=CC=C1C(=O)O",
    "ibuprofen": "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",
    "acetaminophen": "CC(=O)NC1=CC=C(O)C=C1",
    "warfarin": "CC(=O)CC(C1=CC=CC=C1)C1=C(O)C2=CC=CC=C2OC1=O",
    "metformin": "CN(C)C(=N)NC(=N)N",
    "atorvastatin": "CC(C)C1=C(C(=O)NC2=CC=CC=C2)C(=C(N1CCC(O)CC(O)CC(=O)O)C1=CC=C(F)C=C1)C1=CC=CC=C1",
    "simvastatin": "CCC(C)(C)C(=O)OC1CC(C)C=C2C=CC(C)C(CCC3CC(O)CC(=O)O3)C12",
    "omeprazole": "CC1=CN=C(CS(=O)C2=NC3=CC=C(OC)C=C3N2)C(OC)=C1C",
    "fluoxetine": "CNCCC(Oc1ccc(cc1)C(F)(F)F)c1ccccc1",
    "sertraline": "CNC1CCC(c2ccc(Cl)c(Cl)c2)c2cccc1c2",
    "amoxicillin": "CC1(C)S[C@@H]2[C@H](NC(=O)[C@H](N)c3ccc(O)cc3)C(=O)N2[C@H]1C(=O)O",
    "ciprofloxacin": "OC(=O)c1cn(C2CC2)c2cc(N3CCNCC3)c(F)cc2c1=O",
    "diazepam": "CN1c2ccc(Cl)cc2C(=NCC1=O)c1ccccc1",
    "losartan": "CCCCc1nc(Cl)c(CO)n1Cc1ccc(-c2ccccc2-c2nnn[nH]2)cc1",
    "digoxin": "CC1OC(CC(O)C1O)OC1CCC2(C)C(CCC3C2CCC2(C)C3CCC2C2=CC(=O)OC2)C1",
    "clopidogrel": "COC(=O)C(c1ccccc1Cl)N1CCc2sccc2C1",
    "lisinopril": "NCCCCC(NC(CCc1ccccc1)C(=O)N1CCCC1C(=O)O)C(=O)O",
    "levothyroxine": "C1=C(C=C(C(=C1I)OC2=CC(=C(C(=C2)I)O)I)I)CC(C(=O)O)N",
}


def smiles_to_fingerprint(smiles, radius=FINGERPRINT_RADIUS, n_bits=FINGERPRINT_NBITS):
    """Convert one SMILES string to an ECFP4 (Morgan) fingerprint tensor.

    Returns a torch.FloatTensor of shape [n_bits]. Raises ValueError if the
    SMILES could not be parsed or fingerprinted -- callers should catch this
    per-drug rather than letting a bad SMILES crash a batch run.
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
    except Exception as e:
        raise ValueError(f"RDKit raised parsing SMILES {smiles!r}: {e}") from e
    if mol is None:
        raise ValueError(f"RDKit could not parse SMILES {smiles!r}")
    try:
        generator = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
        fp = generator.GetFingerprint(mol)
    except Exception as e:
        raise ValueError(f"Fingerprint generation failed for SMILES {smiles!r}: {e}") from e
    return torch.tensor(list(fp), dtype=torch.float32)


def build_fingerprints(drug_to_smiles):
    """Build the {drug_id: fingerprint_vector} dict from a drug_id -> SMILES mapping.

    Matches the full_features.pt contract in features/README.md. Never raises
    on a per-drug failure: missing or invalid SMILES are collected into
    `failures` as (drug_id, reason) pairs instead of crashing the whole run.

    Returns (fingerprints, failures).
    """
    fingerprints = {}
    failures = []
    for drug_id, smiles in drug_to_smiles.items():
        if not smiles:
            failures.append((drug_id, "missing SMILES"))
            continue
        try:
            fingerprints[drug_id] = smiles_to_fingerprint(smiles)
        except ValueError as e:
            failures.append((drug_id, str(e)))
    return fingerprints, failures


def load_smiles_lookup():
    """Return a {drug_id: smiles} mapping for every drug_id data_loading.load_graph()
    can produce.

    TODO (M1/M2 handoff): no such mapping exists yet. data/README.md's
    file contract for processed/graph.pt is drug_id, edge_index,
    severity_label, mechanism_tag -- it does not include SMILES. This needs
    to come from somewhere (most likely parsed out of data/raw/drugbank_raw.*
    by download_drugbank.py) before full-graph fingerprinting can run.
    """
    raise NotImplementedError(
        "No drug_id -> SMILES source is wired up yet (see load_smiles_lookup docstring)."
    )


def _extract_drug_ids(graph):
    """Pull the drug_id list off whatever object data_loading.load_graph() returns."""
    if isinstance(graph, dict):
        return list(graph["drug_id"])
    return list(graph.drug_id)


def extract_all_fingerprints(graph=None):
    """Full Day-2 extraction: fingerprint every drug in a loaded graph.

    If `graph` is not given, loads one via data_loading.load_graph(). Lets
    NotImplementedError / FileNotFoundError (graph not ready) and
    NotImplementedError from load_smiles_lookup() (no SMILES source wired up)
    propagate to the caller -- callers decide how to report a block.

    Returns (fingerprints, failures, drug_ids):
      - fingerprints: {drug_id: fingerprint_vector}
      - failures: [(drug_id, reason), ...]
      - drug_ids: every drug_id in the graph (success + failure should cover it)
    """
    if graph is None:
        graph = data_loading.load_graph()
    drug_ids = _extract_drug_ids(graph)

    smiles_lookup = load_smiles_lookup()
    drug_to_smiles = {drug_id: smiles_lookup.get(drug_id) for drug_id in drug_ids}
    fingerprints, failures = build_fingerprints(drug_to_smiles)
    return fingerprints, failures, drug_ids


def main():
    start = time.time()

    try:
        fingerprints, failures, drug_ids = extract_all_fingerprints()
    except NotImplementedError as e:
        reason = str(e) or ("data_loading.load_graph() is not implemented yet "
                             "(data/processed/graph.pt is not available)")
        print(f"BLOCKED: {reason}. Not proceeding with full-graph fingerprint extraction.")
        return
    except FileNotFoundError as e:
        print(f"BLOCKED: {e}. Not proceeding with full-graph fingerprint extraction.")
        return

    if failures:
        with open(FAILURE_LOG_PATH, "w") as f:
            for drug_id, reason in failures:
                f.write(f"{drug_id}\t{reason}\n")

    torch.save(fingerprints, FULL_OUTPUT_PATH)

    runtime = time.time() - start
    print(f"Total drugs processed: {len(drug_ids)}")
    print(f"Succeeded: {len(fingerprints)}")
    print(f"Failed: {len(failures)}")
    print(f"Runtime: {runtime:.2f}s")
    print(f"Saved output to {FULL_OUTPUT_PATH}")
    if failures:
        print(f"Failure details logged to {FAILURE_LOG_PATH}")


if __name__ == "__main__":
    main()
