# Drug-Drug Interaction Severity Prediction

Ordinal, mechanism-aware graph neural network for predicting DDI severity
(minor / moderate / major) instead of binary interact/no-interact.

See the team plan doc for the full 4-day schedule and task ownership.

## Folder ownership

| Folder | Owner | Purpose |
|---|---|---|
| `data/` | M1 | Raw + processed DrugBank/TWOSIDES graphs, severity labels, class weights |
| `features/` | M2 | Molecular fingerprints, mechanism edge features, embeddings |
| `models/` | M3, M4 | Baseline GNN, Ordinal GNN (M3); Mechanism-Aware Ordinal GNN + few-shot (M4) |
| `eval/` | M5 | Metrics, severe-subset benchmark, explainability, literature validation |
| `results/` | shared (written by M3/M4/M5) | JSON/markdown result tables, saved checkpoints go here or in `models/checkpoints/` |
| `notebooks/` | shared | Exploratory / dry-run notebooks, not part of the pipeline |
| `docs/` | shared | Paper outline, bibliography, write-up notes for the later paper |

## Setup

```bash
pip install -r requirements.txt
# or
conda env create -f environment.yml
```

## Pipeline order

1. `data/download_drugbank.py`, `data/download_twosides.py` → raw data
2. `data/build_graph.py`, `data/label_severity.py`, `data/class_weights.py` → `data/processed/`
3. `features/fingerprints.py`, `features/mechanism_extraction.py`, `features/embeddings.py` → `features/build_features.py`
4. `models/train.py` (trains baseline / ordinal / mechanism-aware models using `eval/split.py`)
5. `eval/metrics.py`, `eval/attention_viz.py`, `eval/benchmark_severe_subset.py`, `eval/literature_validation.py` → `results/`

Each stage reads only from the previous stage's output directory — see each
subfolder's README for exact input/output file names and the day each piece
is due.
