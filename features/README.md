# features/ — Owner: M2

## Day 1
- `fingerprints.py` — SMILES → RDKit molecular fingerprints, tested on a 50-drug subset
- `mechanism_extraction.py` (scaffold) — parse CYP450 pathways / protein targets from DrugBank fields

## Day 2 (blocks M3/M4 — push as soon as `data/processed/graph.pt` lands)
- Full fingerprint extraction for all drugs in `data/processed/graph.pt`
- `mechanism_extraction.py` — complete edge-level mechanism tags: CYP450 inhibition,
  protein binding competition, CNS depression, QT prolongation
- `embeddings.py` — optional pretrained drug2vec / ChemBERTa embeddings
- `build_features.py` — assembles everything into one feature bundle
- Output: `full_features.pt`, `mechanism_edges.csv`

## File contract
`full_features.pt`: `{drug_id: fingerprint_vector}`
`mechanism_edges.csv`: `drug_a, drug_b, mechanism_tag, confidence`
