# `eval/` — M5 (Evaluation & Explainability)

Everything in here is **model-agnostic**. M3 and M4 do not import each other's
code; they both import this, so the comparison table in the paper is
apples-to-apples.

## Data contract (agreed at the Day-1 sync)

| Thing | Shape | Notes |
| --- | --- | --- |
| `edge_index` | `(2, E)` or `(E, 2)` int | drug node ids; either orientation accepted |
| `labels` | `(E,)` int | `0 = MINOR`, `1 = MODERATE`, `2 = MAJOR` — **ordered** |
| `mechanism_matrix` | `(E, 4)` float | M2's columns, in order: CYP450 inhibition, protein binding competition, CNS depression, QT prolongation |
| predictions | `(N,)` int | same label encoding |
| probabilities | `(N, 3)` float | rows sum to 1 |

If M1's final schema differs, tell M5 — only `eval/split.py` and this table
need changing, not the model code.

## Files

| File | Day | What it does |
| --- | --- | --- |
| `metrics.py` | 1 | Macro F1, MAE, MAJOR-class recall, QWK, per-class P/R/F1, confusion matrix, AUC/AP; ordinal decoders; markdown report + comparison table |
| `split.py` | 1 | 70/15/15 stratified split, reserved severe benchmark subset, drug-disjoint (cold-start) split, leakage verification, JSON save/load |
| `benchmark_severe_subset.py` | 1 | Held-out severe-subset benchmark + OGBL-DDI: native Hits@K / MRR, environment check, dataset loader |
| `attention_viz.py` | 2 | Attention extraction via forward hooks, mechanism attribution, explainability figures |
| `../tests/dummy_data.py` | 2 | Synthetic graph + fake model outputs so the harness is testable before M1's data lands |

## Quick use

### Metrics (M3 / M4 call this)

```python
from eval.metrics import evaluate, comparison_to_markdown

res = evaluate(y_true, y_pred, y_proba=proba, model_name="mechanism_gnn", split="test")
res.save("results/model3_results.json")
print(res.macro_f1, res.mae, res.major_recall)
```

Decoding helpers, so every head type lands in the same label space:

```python
from eval.metrics import softmax_to_labels, expected_severity, coral_to_labels, binary_to_severity

y = softmax_to_labels(proba)      # argmax, (N, 3)
y = expected_severity(proba)      # ordinal-aware: round E[severity]
y = coral_to_labels(cum_probs)    # CORAL head, (N, 2) = P(y>0), P(y>1)
y = binary_to_severity(y_binary)  # lift M3's binary baseline into 3 classes
```

### Splits (run once, commit the JSON, everyone trains on it)

```python
from eval.split import make_splits, save_splits, verify_splits

splits = make_splits(edge_index, labels, seed=42, severe_benchmark_frac=0.15)
verify_splits(splits, labels=labels, edge_index=edge_index)   # raises on leakage
save_splits(splits, "eval/splits/ddi_v1.json")
```

Keys: `train`, `val`, `test`, `severe_benchmark`.

**`severe_benchmark` is pulled out before the 70/15/15 and never trained on** —
it is the held-out severe subset the paper benchmarks on. Do not train on it,
do not tune on it, do not look at it until the final run.

For M4's few/zero-shot claim:

```python
from eval.split import drug_disjoint_split
cold = drug_disjoint_split(edge_index, labels, seed=42)   # no drug in both train and test
```

### Explainability (M4's model)

```python
from eval.attention_viz import AttentionCapture, save_figures

with AttentionCapture(model) as cap:        # works on any torch module
    logits = model.encode(x, edge_index, edge_attr)   # M4's model exposes encode()
att = cap.merged()                          # (E,) attention per edge

save_figures(att, edge_index, mechanism_matrix=mech, labels=labels,
             outdir="results/explainability_figures", model_name="mechanism_gnn",
             drug_names=names, focus_drug=42)
```

**M4: the attention layers need `return_attention_weights=True`** (PyG GATConv /
TransformerConv). Without it there is nothing to hook and `cap.merged()` raises.

### Severe-subset benchmark

```python
from eval.benchmark_severe_subset import benchmark_severe, severity_miss_profile, severe_subset_markdown

res = benchmark_severe(labels, y_pred, splits, y_proba=proba, model_name="mechanism_gnn")
print(res.major_recall)                                   # the headline number
print(severity_miss_profile(labels, y_pred, splits))      # where the misses land
open("results/final_comparison.md", "a").write(severe_subset_markdown([res]))
```

Pass the **full-length** arrays over M1's edge list — the function slices by the
split indices itself, so nobody can accidentally score the wrong partition. It
raises if the partition has been contaminated with non-MAJOR edges.

### OGBL-DDI

```bash
python -m eval.benchmark_severe_subset --check       # what's installed / missing
python -m eval.benchmark_severe_subset --download    # fetch dataset (needs torch + ogb)
```

`hits_at_k()` and `mean_reciprocal_rank()` are pure numpy and work without `ogb`
installed, so the metric path is testable on any machine.

## Integration contract (answers to M3's Day-3 asks)

### 1. The split artifact

One artifact, generated once, committed, loaded by everyone:

```bash
python scripts/make_splits.py --edges data/processed/edges_labeled.csv
# writes eval/splits/ddi_v1.json + .summary.md + .kept_rows.npy
```

```python
from eval.split import load_splits
splits = load_splits("eval/splits/ddi_v1.json")
train_idx, val_idx, test_idx = splits.train, splits.val, splits.test
```

**Do not call `make_splits()` in your own training script.** Two independently
generated splits make the comparison table meaningless. Indices address the
*deduped* edge array; `.kept_rows.npy` maps them back to rows of M1's CSV.

Blocked on M1's `edges_labeled.csv`. Everything else below is usable now.

### 2. Canonical undirected-pair handling

DDI is symmetric, so `(a,b)` and `(b,a)` are one interaction. If both survive,
one can land in train and its mirror in test, and the model scores a pair it
memorised.

```python
from eval.split import canonicalize_edges, dedupe_undirected

edges = canonicalize_edges(edge_index)                    # every pair -> (min, max)
edges, labels, kept = dedupe_undirected(edge_index, labels, keep="max")
```

`keep="max"` resolves a pair labelled twice with different severities by taking
the worst — the clinically safe default. Self-loops are dropped.
`make_splits.py` does this before splitting, and `verify_splits()` fails loudly
if a mirrored pair ever straddles two partitions.

**Why this matters, measured:** on a graph that stores both directions, a
row-wise 70/15/15 puts ~70% of test pairs' mirrors into train. Every number
computed on such a split is inflated. The canonical pipeline measures 0%.

**Training on a bidirectional graph without re-introducing the leak:** split on
canonical pairs, then expand.

```python
from eval.split import map_split_to_directed, both_directions, assert_no_undirected_leakage

directed = map_split_to_directed(splits, original_edge_index)
train_edges = original_edge_index[:, directed["train"]]   # (2, N), PyG layout
```

Both orientations of a pair always land in the same partition. `make_splits.py`
writes this mapping to `eval/splits/ddi_v1.directed.json` alongside the split
itself, so M3/M4 can load it directly. `assert_no_undirected_leakage(splits,
edges)` is the hard check to run after any manual split surgery.

### 3. Macro F1 / MAE interface

```python
from eval.metrics import evaluate
res = evaluate(y_true, y_pred, y_proba=proba, model_name="ordinal_gnn", split="test")
res.macro_f1, res.mae, res.major_recall, res.qwk
res.save("results/ordinal_gnn_test.json")
```

`y_true`/`y_pred` are `(N,)` ints in `{0,1,2}`; `y_proba` is `(N,3)` and
optional (it enables AUC/AP). Decoders for every head type — `softmax_to_labels`,
`expected_severity`, `coral_to_labels`, `binary_to_severity` — are in the same
module, so M3's softmax head and M4's CORAL head land in one label space.

### 4. Severe-subset protocol

`severe_benchmark` holds 15% of MAJOR edges, pulled out **before** the 70/15/15.
Never train on it, never tune on it, don't look at it until the final run.

```python
from eval.benchmark_severe_subset import benchmark_severe, severity_miss_profile
res = benchmark_severe(labels, y_pred, splits, y_proba=proba, model_name="ordinal_gnn")
res.major_recall                       # the headline number for this partition
severity_miss_profile(labels, y_pred, splits)   # where the misses land
```

Pass **full-length** arrays over the edge list; the function slices by split
indices itself and raises if the partition has been contaminated.

### 5. Results format — what to hand me

Save one `.npz` per model into `results/` and I score everything with one command:

```python
np.savez("results/preds_ordinal_gnn.npz",
         edge_idx=idx,        # (N,) positions in the deduped edge array
         y_pred=y_pred,       # (N,) int in {0,1,2}
         y_proba=proba,       # (N,3) float — optional, enables AUC
         y_true=y_true)       # (N,) int — optional, taken from labels otherwise
```

```bash
python scripts/run_eval.py      # regenerates eval/results_comparison.md
```

No model code is imported, so refactors on your side can't break scoring.

### 6. Binary detection vs ordinal severity — reported separately

They are different tasks on different label spaces, so they get **separate
tables** and are never read across. A 0.95 detection AUC next to a 0.62 macro
F1 in one table invites exactly the wrong comparison.

```python
np.savez("results/preds_detection_baseline.npz",
         edge_idx=idx, y_true_bin=y_true_binary, y_score=scores, task="detection")
```

```python
from eval.metrics import evaluate_detection
res = evaluate_detection(y_true_bin, y_score, model_name="detection_baseline")
# ROC-AUC, average precision, precision/recall/F1 at threshold
```

`evaluate_detection()` raises if handed severity labels, and the rendered table
carries a line saying its numbers aren't comparable to the ordinal ones. If you
*do* want the binary baseline inside the severity table, map it explicitly with
`binary_to_severity()` and state the mapping in the paper — a reviewer will ask.

## Tests

```bash
python -m unittest discover -s tests -v    # 53 tests
python scripts/run_dummy_eval.py           # end-to-end smoke run, writes results/dummy_run/
```

`tests/dummy_data.py` is the fixture: a synthetic graph at the real class
imbalance (~60/30/10) plus fake model outputs, so the harness can be developed
and regression-tested before M1's `data/processed_graph.pt` lands.
