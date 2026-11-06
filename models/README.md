# Models — Owners: M3 (binary baseline and ordinal GNN), M4 (mechanism-aware and few-shot)

## M3, Day 1: Binary Baseline GNN

`baseline_gnn.py` defines `BinaryBaselineGNN`, a PyTorch Geometric model with a
reusable `DrugPairEncoder`, a stack of GCN layers, and an order-invariant
drug-pair classifier. It accepts
either explicit tensors or a PyG `Data` object:

```python
logits = model(x, message_edge_index, pair_index)
# Or: logits = model(data, pair_index), where data.x and data.edge_index exist.
```

`x` has shape `[num_drugs, num_node_features]`, `edge_index` has shape
`[2, num_message_edges]`, and `pair_index` has shape `[2, num_pairs]`. The model
returns raw logits with shape `[num_pairs]`; use `BCEWithLogitsLoss` without
applying sigmoid first. Pair features combine the sum, absolute difference,
and element-wise product of the two node embeddings, so swapping drug order
does not change the prediction.

The graph supplied to the model is message-passing context. Data preparation
must remove validation/test target pairs in both directions before passing it
to the model.

## M3, Day 2: Training and dry run

`train.py` contains seed/device helpers, uniform non-edge sampling, canonical
undirected pair splitting, ID-keyed node-feature alignment, a minibatch binary
training loop, validation, checkpoint restore/save, and a CLI. The split uses
only training positive pairs as message-passing edges. Validation/test positive
targets are excluded in both directions. Negatives are sampled uniformly from
unordered non-edges; because unlabeled pairs may contain true interactions,
negative labels are an approximation. Training metrics are diagnostic because
training positive edges are also graph context; validation/test metrics are
computed on held-out pairs.

Once the processed graph and M2 feature bundle exist, run from the repository
root:

```bash
python -m models.train --graph data/processed/graph.pt \
  --features features/full_features.pt --epochs 20 --seed 17
```

The default checkpoint is written to `models/checkpoints/` (ignored by Git),
and lightweight configuration, loss history, and held-out metrics are written
to `results/model1_dry_run.json`. Both paths can be overridden with
`--checkpoint` and `--results`. No result file should be committed unless the
real-data run completed; synthetic test results are not project experiment
results.

The CLI expects a PyG graph or mapping with `drug_id` in node-index order and
`edge_index` containing observed positive interaction edges, plus M2's
`{drug_id: feature_vector}` mapping. It aligns features by drug ID and rejects
missing IDs or inconsistent feature dimensions. It does not invent labels or
features. M1's current processed graph and M2's full feature bundle are not
present in this checkout, so a real-data dry run remains blocked until both
deliverables are supplied.

Run synthetic validation with:

```bash
python -m unittest tests.test_baseline_gnn -v
```

## M3, Day 2/next stage: Ordinal Model 2 interface

`ordinal_gnn.py` implements Model 2 with the same `DrugPairEncoder` and M4's
CORAL interface. It returns two cumulative logits for the three labels
`{0: minor, 1: moderate, 2: major}`; train with M4's `CoralLoss` and decode with
`coral_predict` (the number of thresholds cleared). CORAL's threshold-based
ordinal prediction is not guaranteed to equal the maximum-probability class.
This matches the interface used by M4's mechanism-aware
ordinal model. `OrdinalCrossEntropyLoss` remains available for models that
instead emit three ordinary class logits. M3's current `train.py` handles the
binary task only; a shared ordinal training loop remains to be coordinated
before Model 2 training.

## M4 responsibilities

- `mechanism_gnn.py`: mechanism-aware ordinal GNN, using node features, graph
  edges with edge attributes, pair indices, and pair attributes.
- `few_shot.py`: few-shot/new-drug generalization work.
- `ordinal_loss.py`: shared ordinal heads and losses. M3 should import these
  components rather than create duplicate losses.

## Day 3 and Day 4

- Day 3: train Model 1 and Model 2 after `data/processed/` and
  `features/full_features.pt` are available. Save compact summaries as
  `results/model1_results.json` and `results/model2_results.json`.
- Day 4: write `results/baseline_vs_ordinal.md` and finalize comparisons with
  M4/M5 outputs.
- Checkpoints belong in `models/checkpoints/`; document how to reproduce them
  instead of committing large model files.
