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
