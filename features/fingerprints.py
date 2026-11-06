"""
fingerprints.py
Owner: M2 — see features/README.md for day/deadline and output contract.

Day 1: converts a hardcoded set of drugs (drug_id -> SMILES) into Morgan/ECFP4
fingerprints, matching the full_features.pt contract: {drug_id: fingerprint_vector}.

Day 2: swap TEST_DRUGS for the drug_id list coming out of data_loading.load_graph()
(and a drug_id -> SMILES lookup built from DrugBank data), then call
build_fingerprints() on that instead. generate_fingerprint() and
build_fingerprints() do not need to change.
"""

import torch
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator

FINGERPRINT_RADIUS = 2
FINGERPRINT_NBITS = 1024

# Hardcoded Day-1 test set: real drugs with accurate SMILES strings.
# Day 2 replaces this dict with one derived from data_loading.load_graph().drug_id.
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

    Returns a torch.FloatTensor of shape [n_bits], or None if the SMILES
    could not be parsed.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    fp = generator.GetFingerprint(mol)
    return torch.tensor(list(fp), dtype=torch.float32)


def build_fingerprints(drug_to_smiles):
    """Build the {drug_id: fingerprint_vector} dict from a drug_id -> SMILES mapping.

    Matches the full_features.pt contract in features/README.md. Returns
    (fingerprints, failed) where failed is a list of drug_ids whose SMILES
    could not be parsed.
    """
    fingerprints = {}
    failed = []
    for drug_id, smiles in drug_to_smiles.items():
        fp = smiles_to_fingerprint(smiles)
        if fp is None:
            failed.append(drug_id)
        else:
            fingerprints[drug_id] = fp
    return fingerprints, failed


def main():
    fingerprints, failed = build_fingerprints(TEST_DRUGS)

    print(f"Succeeded: {len(fingerprints)}/{len(TEST_DRUGS)}")
    print(f"Failed: {len(failed)}/{len(TEST_DRUGS)}")
    if failed:
        print(f"  Failed drug_ids: {failed}")

    out_path = "features/sample_fingerprints.pt"
    torch.save(fingerprints, out_path)
    print(f"Saved sample output to {out_path}")


if __name__ == "__main__":
    main()
