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


SMILES_SOURCE_PATH = "data/raw/drugbank_graph_nodes.csv"
SMILES_CACHE_PATH = "data/raw/pubchem_smiles.csv"
PUBCHEM_URL = (
    "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/xref/RegistryID/"
    "{}/property/SMILES/JSON"
)


def _load_local_smiles_csv(path):
    """Read a {drug_id: smiles} mapping from a CSV with drug_id/smiles columns, if present."""
    import csv
    from pathlib import Path

    csv_path = Path(path)
    if not csv_path.exists():
        return {}
    lookup = {}
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            drug_id = row.get("drug_id")
            smiles = row.get("smiles", "")
            if drug_id and smiles:
                lookup[drug_id] = smiles
    return lookup


def fetch_smiles_from_pubchem(drug_ids, cache_path=SMILES_CACHE_PATH, delay=0.25):
    """Fetch SMILES for drug_ids missing from the local cache, via PubChem PUG REST
    (xref RegistryID = DrugBank ID), and append results to cache_path.

    Resumable: already-cached drug_ids are skipped. Same approach M4 used in
    models/build_m4_inputs.py to unblock Model 3 while this was pending --
    kept here so features/ (M2's actual deliverable) no longer depends on
    that stand-in script.

    M1's data/raw/drugbank_drugs_raw.csv smiles column is empty for all
    1868 drugs (the Mendeley DDI_data.csv source has no structure data),
    so this is the real SMILES source, not a fallback.
    """
    import csv
    import json
    import time
    import urllib.error
    import urllib.request
    from pathlib import Path

    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if cache_path.exists():
        with open(cache_path, newline="", encoding="utf-8") as f:
            done = {row["drug_id"] for row in csv.DictReader(f)}
    todo = [d for d in drug_ids if d not in done]
    if not todo:
        print(f"[fingerprints] PubChem cache already covers all {len(drug_ids)} drugs.")
        return
    new_file = not cache_path.exists()
    with open(cache_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if new_file:
            writer.writerow(["drug_id", "cid", "smiles", "status"])
        for i, drug_id in enumerate(todo, 1):
            request = urllib.request.Request(
                PUBCHEM_URL.format(drug_id), headers={"User-Agent": "ddi-research-client"}
            )
            for attempt in range(4):
                try:
                    with urllib.request.urlopen(request, timeout=30) as response:
                        props = json.load(response)["PropertyTable"]["Properties"][0]
                    writer.writerow([drug_id, props["CID"], props.get("SMILES", ""), "ok"])
                    break
                except urllib.error.HTTPError as e:
                    if e.code == 404:
                        writer.writerow([drug_id, "", "", "not_found"])
                        break
                    time.sleep(2 ** attempt)
                except (urllib.error.URLError, TimeoutError, KeyError, ValueError):
                    time.sleep(2 ** attempt)
            else:
                writer.writerow([drug_id, "", "", "error"])
            f.flush()
            time.sleep(delay)  # PubChem allows at most 5 requests per second
            if i % 200 == 0:
                print(f"[fingerprints] PubChem: {i}/{len(todo)}", flush=True)


def load_smiles_lookup(drug_ids=None, path=SMILES_SOURCE_PATH, cache_path=SMILES_CACHE_PATH,
                        fetch=True):
    """Return a {drug_id: smiles} mapping.

    Tries, in order:
      1. data/raw/drugbank_graph_nodes.csv (M1's build_graph.py output) --
         currently all-empty smiles (confirmed against the real file: 0/1868
         rows have a SMILES string; the Mendeley DDI_data.csv source M1
         downloads from has no structure data), kept first in case that
         changes.
      2. data/raw/pubchem_smiles.csv cache, topped up via PubChem PUG REST
         (xref RegistryID = DrugBank ID) for any drug_ids not yet cached,
         when fetch=True and drug_ids is given.

    Raises FileNotFoundError only if neither source has anything usable and
    drug_ids wasn't given to fetch with.
    """
    lookup = _load_local_smiles_csv(path)

    if drug_ids is not None and fetch:
        missing = [d for d in drug_ids if d not in lookup]
        if missing:
            fetch_smiles_from_pubchem(missing, cache_path=cache_path)
        lookup.update(_load_local_smiles_csv(cache_path))

    if not lookup:
        raise FileNotFoundError(
            f"No usable SMILES found in {path} or {cache_path}, and no drug_ids "
            "were given to fetch via PubChem. Call load_smiles_lookup(drug_ids=...) "
            "with the graph's drug_id list, or ensure data/raw/pubchem_smiles.csv "
            "already has entries."
        )
    return lookup


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

    smiles_lookup = load_smiles_lookup(drug_ids=drug_ids)
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
