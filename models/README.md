# models/ — Owners: M3 (baseline + ordinal), M4 (mechanism-aware + few-shot)

## Day 1
- M3: `baseline_gnn.py` — binary GNN architecture using a reusable GCN node/pair encoder and symmetric drug-pair logits.
- M4: `ordinal_loss.py` (OrdinalCrossEntropy / CORAL-style), `mechanism_gnn.py` skeleton

## Day 2
- M3: `train.py` — reproducible binary training, validation, metrics, checkpoint handling, and leakage-safe pair splitting; dry-run the baseline when M1/M2 artifacts are available.
- M4: complete `mechanism_gnn.py` with edge features wired in, unit test on dummy data
- Model 2 (`ordinal_gnn.py`) is deferred on this clean M3 submission branch because its current implementation depends on M4's `CoralHead`/`CoralLoss`, which are not yet present on `dev`.

## Binary baseline interface

`BinaryBaselineGNN` accepts either explicit tensors or a PyG `Data` object:

```python
logits = model(x, message_edge_index, pair_index)
# Or: logits = model(data, pair_index), where data.x and data.edge_index exist.
```

`x` has shape `[num_drugs, num_node_features]`, `edge_index` has shape
`[2, num_message_edges]`, and `pair_index` has shape `[2, num_pairs]`. The model
returns one raw logit per pair with shape `[num_pairs]`; pass it to
`BCEWithLogitsLoss` without applying sigmoid first. Pair features combine the
sum, absolute difference, and element-wise product of node embeddings, so
swapping pair order leaves the prediction unchanged.

The M3 binary task is **interaction detection**: positive means a known drug
interaction and negative means a sampled non-interacting pair. Its output is
not one of the three severity classes. Binary metrics must be reported
separately from M5's minor/moderate/major metrics, with a common split and
evaluation protocol agreed before Day 3 comparisons.

Run the real-data dry run from the repository root after M1 and M2 deliver
their artifacts:

```bash
python -m models.train --graph data/processed/graph.pt \
  --features features/full_features.pt --epochs 20 --seed 17
```

The loader accepts M1's documented target form (`drug_id` plus `edge_index`)
and its current Day 2 serialization (`edge_index`, `edge_severity`,
`num_nodes`, `severity_names`) when `data/processed/nodes.csv` is alongside
`graph.pt`. For that serialization, row order in `nodes.csv` and its optional
contiguous `node_idx` column define the stable node-ID order. M1's severity
labels must be aligned to edge columns and lie in `{0, 1, 2}`; they validate
the graph but do not turn this binary task into severity classification.
M2's feature bundle contract is `{drug_id: feature_vector}`. Features are
stacked in graph ID order; missing IDs, duplicate IDs, invalid edge indices,
non-finite values, and inconsistent feature widths fail with explicit errors.

The split canonicalizes duplicate/reversed undirected interactions and samples
reproducible uniform unordered non-edges as approximate negatives. The default
negative ratio is 1:1, so BCE uses balanced sampled examples without an extra
class weight; `--negative-ratio` controls that sampling ratio. Unobserved true
interactions may still be sampled as false negatives. Validation and test
positive targets are removed from message passing in both directions. Training
positives remain in the training context, so training metrics are diagnostic;
held-out validation/test metrics are reported separately.

When project artifacts are unavailable, run an explicitly synthetic pipeline
check. The fixture uses generated two-group node features and graph pairs and
does not represent DrugBank, M1 labels, or M2 molecular features:

```bash
python -m models.train --synthetic --epochs 20 --seed 17 --device cpu
```

This exercises training, validation, held-out evaluation, checkpoint save and
reload through the same pipeline. Results are marked `synthetic_fixture` in
the JSON record and written to `results/model1_synthetic_dry_run.json`.

The default checkpoint is saved under `models/checkpoints/` (ignored by Git),
and lightweight loss history and metrics go to
`results/model1_dry_run.json`. Generated datasets and checkpoints are not
committed. No real-data run has been recorded on this branch: `graph.pt`,
`nodes.csv`, `edges_labeled.csv`, and `features/full_features.pt` are absent
from `origin/dev`; M1 raw DrugBank-derived data is also not tracked. M2's
feature builder still has unfinished dependencies (`load_smiles_lookup()` and
the root `data_loading.py` stub), so the feature bundle cannot currently be
generated from the repository alone.

Run synthetic tests with:

```bash
python -m unittest tests.test_baseline_gnn -v
python -m unittest discover -v
```

Day 2 validation on the M3 clean submission branch: all 12 M3 tests and all 3
available M1 tests pass on CPU, including synthetic end-to-end training,
graph-schema adapter checks, leakage and negative-sampling checks, and
checkpoint round trip. The 16-node synthetic fixture (64 positive pairs) had
training loss 0.6931 at epoch 1 and 0.5010 at epoch 20; best validation loss
was 0.5202, test loss 0.5072, test accuracy 0.7222, F1 0.8000, ROC-AUC 1.0000,
and average precision 1.0000. These are explicitly **synthetic fixture
metrics**, not project performance or evidence of real-data generalization.
CUDA was not available in this environment. The repository CI targets Python 3.10;
M3's code uses Python 3.10-compatible syntax, while local execution used Python
3.13 and Torch 2.15.0 / PyG 2.8.1. CI currently runs import checks only and
does not invoke the unittest suite.

### Day 2 status and remaining inputs

- Completed: binary GCN baseline, symmetric pair scorer, reproducible pair
  splitting, training/validation/test evaluation, checkpoint round trip, and
  a synthetic-only pipeline fixture.
- Blocked for real-data dry run: M1's processed graph tables/tensor and M2's
  `full_features.pt` are not present on `dev`; no licensed raw data is included.
- Day 3 dependency: M4's ordinal components are not yet available on `dev`.
  Integrate them before implementing/running Model 2. Coordinate M5's split
  and metric protocol before comparing binary interaction detection with
  three-class severity results.

## Day 3 (needs `data/processed/` + `features/full_features.pt`)
- M3: run Model 1 and Model 2, then save `results/model1_results.json` and `results/model2_results.json` when Model 2's ordinal dependency is integrated.
- M4: run Model 3 (mechanism-aware ordinal) + `few_shot.py` (MAML/contrastive for new-drug generalization) → `results/model3_results.json`

## Day 4
- Finalize result write-ups: `results/baseline_vs_ordinal.md` (M3), `results/mechanism_aware_final.md` + `results/few_shot_results.md` (M4)

Checkpoints saved to `models/checkpoints/` (gitignored — too large for git; document
the download/regeneration path in each model's results file instead).
