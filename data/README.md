# data/ — Owner: M1

## Day 1
- `download_drugbank.py` — pull DrugBank 5.0 (15,000+ drugs, 2M+ labeled interactions)
- `download_twosides.py` — pull TWOSIDES (HuggingFace, 63K drug pairs) for later cross-dataset validation
- `build_graph.py` — build raw interaction graph (nodes=drugs, edges=interactions) → `raw/`
- Output: `raw/drugbank_raw.*`, `raw/twosides_raw.*`, `data_loading.py` (loader used by everyone downstream)

## Day 2 (hard deadline — M2/M3/M4 block on this)
- `label_severity.py` — attach minor/moderate/major severity labels to every edge
- `class_weights.py` — compute focal-loss / weighted-sampling weights for class imbalance
- Output: `processed/graph.pt` (or `.csv`), `processed/class_weights.json`

## Day 3
- `twosides_processed` — prep TWOSIDES as the cross-dataset validation set for M5

## File contract (do not change without telling the team)
`processed/graph.pt` must expose: `drug_id`, `edge_index`, `severity_label {0,1,2}`,
plus any raw mechanism tags needed by `features/mechanism_extraction.py`.
**Source data (not in repo):** `DDI_data.csv` from Mendeley Data, "data of multiple-type drug-drug interactions" (DrugBank v5.1-derived): https://data.mendeley.com/datasets/md5czfsfnd. Download it manually, check its licence, and run `python data/download_drugbank.py --csv path/to/DDI_data.csv`.
