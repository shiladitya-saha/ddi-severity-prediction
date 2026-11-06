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

Run the real-data dry run from the repository root after M1 and M2 deliver
their artifacts:

```bash
python -m models.train --graph data/processed/graph.pt \
  --features features/full_features.pt --epochs 20 --seed 17
```

The graph must provide `drug_id` in node order and `edge_index` for observed
positive interactions. The feature bundle must be `{drug_id: feature_vector}`;
features are stacked in graph ID order, with missing IDs or inconsistent
widths rejected. The split canonicalizes undirected pairs and samples uniform
unordered non-edges as approximate negatives. Validation and test positive
targets are removed from message passing in both directions. Training
positives remain in the training context, so training metrics are diagnostic;
held-out validation/test metrics are reported separately.

The default checkpoint is saved under `models/checkpoints/` (ignored by Git),
and lightweight loss history and metrics go to
`results/model1_dry_run.json`. No real-data run has been recorded on this
branch because `data/processed/graph.pt` and `features/full_features.pt` are
not available on `dev`.

Run synthetic tests with:

```bash
python -m unittest tests.test_baseline_gnn -v
```

## Day 3 (needs `data/processed/` + `features/full_features.pt`)
- M3: run Model 1 and Model 2, then save `results/model1_results.json` and `results/model2_results.json` when Model 2's ordinal dependency is integrated.
- M4: run Model 3 (mechanism-aware ordinal) + `few_shot.py` (MAML/contrastive for new-drug generalization) → `results/model3_results.json`

## Day 4
- Finalize result write-ups: `results/baseline_vs_ordinal.md` (M3), `results/mechanism_aware_final.md` + `results/few_shot_results.md` (M4)

Checkpoints saved to `models/checkpoints/` (gitignored — too large for git; document
the download/regeneration path in each model's results file instead).
