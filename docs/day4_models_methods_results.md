# Day 4 Models: Methods and Results Status

**Status:** model runs are blocked pending delivery of M1's graph artifacts
and a working PyTorch/PyG runtime. A temporary Python 3.10 environment was
created from `environment.yml`, but importing PyTorch fails while loading
`fbgemm.dll` on this host. No project-data metrics are reported in this
document. The implementation details below are from the repository code; the
split figures are from M5's committed split summary.

## Methods

### Data and graph

M3's loaders expect a PyTorch graph with `edge_index`, interaction severity
labels, and either stable `drug_id` values or a companion `nodes.csv` mapping.
The graph is not present in this checkout. The feature bundle is keyed by drug
ID and is aligned to graph node order by the model loader. The intended feature
width in `features/README.md` is 1,792 (1,024 Morgan/ECFP4 bits plus a 768
dimensional ChemBERTa embedding when embeddings are enabled); the actual graph
and feature tensors could not be loaded in the available environment.

### Canonical split and leakage controls

M5's `eval/splits/ddi_v1.json` uses seed 42, a 70/15/15 train/validation/test
split, and a 15% MAJOR-only severe benchmark reserved before those partitions.
The committed summary reports 222,271 unique undirected pairs: 153,991 train,
32,996 validation, 33,002 test, and 2,282 severe benchmark pairs. Canonical
pairs in the artifact are unique, and the partition index sets are disjoint.
The summary reports all severe benchmark labels as MAJOR. The source labeled
edge table and graph are absent here, so graph-to-artifact label and ordering
checks remain outstanding.

Both model loaders use training positive edges for message passing and exclude
held-out positive pairs in both directions. Model 1's sampled negatives are
allocated to the corresponding positive folds; severe benchmark positives and
their assigned negatives remain outside training. Model 2 uses the canonical
severity pairs directly and reserves validation, test, and severe pairs from
the message-passing graph.

### Model 1: binary interaction detection

`BinaryBaselineGNN` applies a GCN encoder and creates an order-invariant pair
representation from the endpoint sum, absolute difference, and elementwise
product. A binary prediction head returns one raw logit per pair. Training uses
`BCEWithLogitsLoss`, uniform unordered non-edge sampling at a default 1:1
negative ratio, Adam with learning rate 0.001 and weight decay 0.0001, and a
default batch size of 256. This is unweighted BCE; class balance comes from the
sampled negative ratio, not a `pos_weight` setting. The requested run is 60
epochs with seed 17. It keeps the lowest validation-loss checkpoint while
running the fixed epoch count; it does not implement patience-based early
stopping. M5 evaluates this as binary detection, separately from severity.

### Model 2: ordinal severity prediction

`OrdinalGNN` uses the same GCN encoder and symmetric pair representation,
followed by M4's `CoralHead` and `CoralLoss`. Three ordered labels are encoded
as 0=minor, 1=moderate, and 2=major; the head emits two cumulative logits and
decodes three class probabilities. Training uses Adam with learning rate
0.001, weight decay 0.0001, and batch size 256. The run uses inverse-frequency
class weights when the configured weight file provides them. The requested
configuration is 200 epochs and seeds 42, 7, and 123 on the same canonical
split. Checkpoint selection uses validation loss over the fixed epoch count;
there is no patience-based early stopping. Each seed requires distinct output
paths to retain its checkpoint, predictions, and metrics.

### Evaluation and outputs

M5's detection metrics are ROC-AUC and average precision from scores, plus
precision, recall, F1, and accuracy at the configured threshold (0.5 by
default). Ordinal evaluation reports macro-F1, MAE, major recall, QWK, and
probability-based AUC/AP when available. The M3 runners export edge indices
and task-specific values in compressed NPZ files; detection outputs use
`y_true_bin`, `y_score`, and `task="detection"`, while ordinal outputs use
`y_true`, `y_pred`, and `y_proba` with three columns. The two tasks must be
reported separately.

## Results

### Model 1

**Not run.** `data/processed/graph.pt` and `data/processed/nodes.csv` are
missing, and the installed Python 3.10 environment cannot import PyTorch due to
the `fbgemm.dll` load error. No checkpoint, prediction file, training log, or
final test metrics were produced.

### Model 2

**Not run.** The same graph and environment blockers prevent all three seeds.
There are no per-seed metrics, checkpoints, or prediction files, so no mean or
standard deviation is calculated.

### Model 3 comparison

No Model 3 result artifacts were available in the repository at audit time.
The reported 0.11–0.12 macro-F1 improvement over Model 2 therefore remains
unverified context and is not presented as a result.

### DB08913 and limitations

The integrated M2 feature bundle contains DB08913, but
`features/fingerprint_placeholders.log` documents a zero-vector placeholder
because no usable SMILES was found. It is not a valid molecular fingerprint
and supplies no molecular feature signal for that drug. The project feature
contract explicitly allows this documented placeholder to preserve complete
node coverage; the impact on model performance cannot be assessed without a
real-data run. The graph/node alignment and actual tensor finiteness checks
also remain pending because the graph is absent and PyTorch is not installed.

### Reproducibility and pending actions

The integration commit for this branch is `e955dba` (full commit ID should be
recorded with any future experiment). Before reporting results, M1 must deliver
`data/processed/graph.pt`, `data/processed/nodes.csv`, and the matching
`data/processed/edges_labeled.csv`. Then validate graph, features, labels, and
split ordering; install the project-supported Python/PyTorch/PyG dependencies;
run the model and evaluation tests; and run each experiment with unique output
paths. No pre-fix reference metrics are final results.
