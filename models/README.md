# Models

M3 owns the binary interaction baseline and the ordinal severity model. M4 owns
the mechanism-aware ordinal model, CORAL components, and few-shot experiments.
M3 reuses M4's `CoralHead`, `CoralLoss`, and CORAL decoding helpers from
`ordinal_loss.py`; it does not maintain a second CORAL implementation.

## Shared data and split contract

- `data/processed/graph.pt` contains `edge_index`, `edge_severity`, and
  `num_nodes`. `data/processed/nodes.csv` defines the stable node-index to
  `drug_id` mapping in row order, with `node_idx` contiguous from zero.
- `features/full_features.pt` is a mapping from `drug_id` to one finite feature
  vector. Both models align features to the graph node order and reject missing
  IDs or inconsistent feature widths.
- M5 generates `eval/splits/ddi_v1.json` once from
  `data/processed/edges_labeled.csv`. Its canonical pair order and train/val/test
  indices are shared by the ordinal model and by the positive examples in the
  binary task. The `severe_benchmark` partition is never used for training or
  checkpoint selection.
- M1 labels are `0=minor`, `1=moderate`, `2=major`. These are severity classes;
  they are not the binary interaction-detection labels.

The graph context passed to either GNN contains training positives only, in
both directions. Validation, test, and severe-benchmark target pairs are
excluded from message passing. M3 Model 2 uses drug features and pair indices;
it does not use target-edge descriptions or mechanism pair attributes.

## Model 1: binary interaction detection

`BinaryBaselineGNN` uses GCN node embeddings and a symmetric pair representation
(sum, absolute difference, and element-wise product). It returns a raw logit
per pair for `BCEWithLogitsLoss`. Positive examples follow the canonical M5
folds. Uniform unordered non-edges are sampled once and assigned within those
same folds at the selected negative ratio. Unobserved true interactions can
still be sampled as false negatives.

```bash
python -m models.train --graph data/processed/graph.pt \
  --features features/full_features.pt \
  --splits eval/splits/ddi_v1.json --epochs 20 --seed 17
```

This writes `results/model1_results.json`,
`results/preds_detection_baseline.npz`,
`results/model1_detection_splits.json`, and
`models/checkpoints/model1_binary_baseline.pt`. Detection is evaluated
separately from ordinal severity:

```bash
python scripts/run_eval.py \
  --preds results/preds_detection_baseline.npz \
  --splits results/model1_detection_splits.json
```

The prediction file stores `edge_idx`, `y_true_bin`, `y_score`, and
`task="detection"`. Do not map these binary scores to three-class severity
metrics.

## Model 2: ordinal severity prediction

`OrdinalGNN` reuses the binary model's graph encoder and symmetric pair
representation, then applies M4's `CoralHead`. For three ordered severity
classes it returns two cumulative logits with shape `[num_pairs, 2]`. Training
uses M4's `CoralLoss`; `coral_logits_to_probs` converts logits to three class
probabilities and `coral_predict` returns labels in `{0,1,2}`.

```bash
python -m models.ordinal_gnn --epochs 20 --seed 42
python scripts/run_eval.py --preds results/preds_ordinal_gnn.npz
```

The real-data command writes `results/model2_results.json`,
`results/preds_ordinal_gnn.npz`, and
`models/checkpoints/model2_ordinal_gnn.pt`. The prediction file stores M5
canonical `edge_idx` values, `y_true`, `y_pred`, and `y_proba` with three
columns for validation, test, and the reserved severe benchmark. Checkpoints
are selected by validation loss. Test and severe-benchmark labels do not affect
training or checkpoint selection.

## Synthetic checks

Synthetic fixtures exercise model behavior and file interfaces. They do not
represent DrugBank, M1 labels, or M2 molecular features, and their metrics must
not be reported as real-data performance.

```bash
python -m models.train --synthetic --epochs 20 --seed 17 --device cpu
python -m models.ordinal_gnn --synthetic --epochs 4 --seed 17 --device cpu
python -m unittest tests.test_baseline_gnn tests.test_ordinal_gnn -v
python -m unittest discover -s tests -v
```

Checkpoints are ignored under `models/checkpoints/`. Result JSON/NPZ files are
small evaluation artifacts; datasets and large checkpoints should not be
committed. Real-data runs require the local M1 graph, node table, labeled edge
CSV, an M5 split artifact, and complete M2 feature coverage.

## M4 model interface

`MechanismOrdinalGNN` accepts node features, a message-passing `edge_index` and
four-channel `edge_attr`, plus a separate `pair_index` and optional `pair_attr`.
Its mechanism channel order is CYP450 inhibition, protein-binding competition,
CNS depression, and QT prolongation. M4 must ensure attributes derived from a
held-out interaction description are not supplied for that target pair; the
edge description is also an input to M1's heuristic severity labels.

## Model 3: Mechanism-Aware Ordinal GNN (M4, Day 3)

Results and caveats: `results/model3_day3_summary.md`.

| File | Purpose |
|---|---|
| `mechanism_gnn.py` | GATv2 over drug fingerprints, mechanism tags as edge features, CORAL head |
| `train_mechanism.py` | Trains and evaluates Model 3; writes `results/<tag>_results.json` + `_details.json` |
| `few_shot.py` | New-drug experiment on M5's drug-disjoint split → `results/few_shot_results.json` |
| `build_m4_inputs.py` | Builds fingerprints + mechanism edges as a stand-in while M2's pipeline was blocked (superseded now that `features/full_features.pt` and `features/mechanism_edges.csv` are real — pass those paths explicitly, see below) |

```bash
python -m models.train_mechanism --features features/full_features.pt \
  --mechanism-edges features/mechanism_edges.csv --tag model3 --epochs 100 --patience 5
python -m models.few_shot --features features/full_features.pt \
  --mechanism-edges features/mechanism_edges.csv --epochs 100 --patience 5
python -m unittest tests.test_train_mechanism tests.test_mechanism_model
```

Protocol (matches Model 2 so the comparison is fair): M5's `make_splits(seed=42)` (confirmed
identical, row for row, to `eval/splits/ddi_v1.json`'s canonical pairs and indices); inverse-frequency
class weights on the **training** labels; best validation macro F1 with early stopping (patience 5, cap
100); test scored once. Message passing uses training edges only, with a training pair's own edge
removed while that pair is scored. **The scored pair's own mechanism tags are not an input** — M1's
labels and M2's tags are both keyword rules over the same interaction text, so those tags alone predict
severity (74% vs 54% majority class); `--pair-mechanism` turns them on for the leak ablation only.
