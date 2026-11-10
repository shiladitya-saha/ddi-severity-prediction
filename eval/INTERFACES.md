# `eval/` interface reference

Owner: M5. Signatures below are copied from the shipped code — if this file and
the code ever disagree, the code wins and this is a bug; tell M5.

Nothing in `eval/` imports model code, so none of this constrains how M3/M4
build or refactor their models. Call it after inference with plain arrays.

**Import surface:** everything is re-exported from the package, so
`from eval import evaluate, evaluate_detection, load_splits` works, as does the
longer `from eval.metrics import evaluate`.

---

## Label contract

```
0 = MINOR   <   1 = MODERATE   <   2 = MAJOR      (ordered — MAE and QWK depend on it)
```

Detection labels are a **different space**: `0 = no interaction`, `1 = interaction`.
Never pass one where the other is expected; `evaluate_detection()` raises if
handed `{0,1,2}` labels.

---

## 1. Ordinal severity

```python
evaluate(y_true, y_pred, y_proba=None, model_name="unnamed_model",
         split="test", notes=None) -> EvalResult
```

| Arg | Shape / type | Required | Notes |
| --- | --- | --- | --- |
| `y_true` | `(N,)` int in `{0,1,2}` | yes | |
| `y_pred` | `(N,)` int in `{0,1,2}` | yes | same length as `y_true` |
| `y_proba` | `(N,3)` float, rows sum to 1 | no | enables `macro_auc`, `major_auc`, `major_ap`; rows are renormalized with a warning if they don't sum to 1 |
| `model_name` | str | no | free text; becomes the row label in the comparison table |
| `split` | str | no | `"test"`, `"val"`, `"severe_benchmark"`, `"twosides"` … |
| `notes` | `dict[str,str]` | no | carried into the saved JSON |

Raises `ValueError` on: empty input, shape mismatch, NaN/inf, any label outside
`{0,1,2}`, or `y_proba` that isn't `(N,3)`.

**`EvalResult` fields**

```
model_name, split, n_samples,
macro_f1, mae, major_recall, qwk,                      # headline
micro_f1, weighted_f1, accuracy, balanced_accuracy,
within_one_accuracy, rmse,
per_class, confusion, support,                         # per-class detail
macro_auc, major_auc, major_ap,                        # None unless y_proba given
notes
```

`per_class` is `{"MINOR"|"MODERATE"|"MAJOR": {"precision","recall","f1","support"}}`.
`confusion` is a 3×3 nested list, **rows = true, columns = predicted**, class
order `[MINOR, MODERATE, MAJOR]`.

`.to_dict()`, `.save(path)`, `EvalResult.load(path)` — plain JSON, no pickle.

---

## 2. Binary detection — separate table, always

```python
evaluate_detection(y_true, y_score, threshold=0.5, model_name="unnamed_model",
                   split="test", notes=None) -> DetectionResult
```

| Arg | Shape / type | Required | Notes |
| --- | --- | --- | --- |
| `y_true` | `(N,)` int in `{0,1}` | yes | 1 = the pair interacts |
| `y_score` | `(N,)` float | yes | probabilities, **or** hard 0/1 predictions |
| `threshold` | float | no | applied only when `y_score` looks like probabilities |

If `y_score` is probabilities you get `roc_auc` and `average_precision`; with
hard 0/1 predictions both are `None` and only P/R/F1/accuracy are computed.

**`DetectionResult` fields**

```
model_name, split, n_samples, n_positive,
roc_auc, average_precision,          # None if y_score is hard labels
f1, precision, recall, accuracy, threshold, notes
```

Detection and severity results render as **separate tables**
(`detection_to_markdown` vs `comparison_to_markdown`), and the detection table
carries a line stating the numbers aren't comparable to the ordinal ones. To put
a binary model inside the severity table instead, map it explicitly:

```python
binary_to_severity(y_binary, positive_class=2) -> np.ndarray   # (N,) in {0,2}
```

and state that mapping in the paper.

---

## 3. Head decoders — pick one, they all return `(N,)` ints in `{0,1,2}`

```python
softmax_to_labels(proba)                 # (N,3) -> argmax
expected_severity(proba)                 # (N,3) -> round(E[severity]); ordinal-aware
coral_to_labels(cum_probs, threshold=0.5)# (N,2) = [P(y>0), P(y>1)] -> count thresholds passed
binary_to_severity(y_binary, positive_class=2)
```

`expected_severity` differs from `argmax` where a distribution is spread:
`[0.40, 0.35, 0.25]` → argmax `MINOR`, expected severity `MODERATE` (E = 0.85).
Report whichever you use, consistently, and say which in the paper.

---

## 4. Splits

```python
load_splits(path) -> SplitSet
splits.train / .val / .test            # np.ndarray of int indices
splits["severe_benchmark"]             # MAJOR-only, reserved
splits.sizes() / splits.class_counts(labels) / splits.summary(labels)
```

Indices address the **deduped canonical-pair** array. For a bidirectional graph:

```python
map_split_to_directed(splits, original_edge_index) -> dict[str, np.ndarray]
# row indices into the ORIGINAL edge array; both orientations of a pair always
# land in the same partition. make_splits.py writes this as ddi_v1.directed.json
```

Leak-prevention helpers, if you build any edge list yourself:

```python
canonicalize_edges(edge_index) -> (E,2)          # every pair -> (min, max)
dedupe_undirected(edge_index, labels=None, keep="max") -> (edges, labels, kept_idx)
assert_no_undirected_leakage(splits, edge_index) -> None   # raises on failure
verify_splits(splits, labels=None, edge_index=None, strict=True) -> list[str]
```

`edge_index` is accepted as `(2,E)` or `(E,2)` — both orientations handled.

---

## 5. Severe subset

```python
benchmark_severe(labels, y_pred, splits, y_proba=None,
                 model_name="unnamed_model", split_key="severe_benchmark") -> EvalResult
severity_miss_profile(labels, y_pred, splits, split_key="severe_benchmark") -> dict
```

Pass **full-length** arrays over the whole edge list — the function slices by
split indices itself. Raises if the partition contains non-MAJOR edges.
`major_recall` is the headline on this partition; `macro_f1` is near-meaningless
on one class and is reported only for table consistency.

---

## 6. Prediction files — what M5 reads

One `.npz` per model in `results/`, named `preds_<model_name>.npz`:

```python
np.savez("results/preds_ordinal_gnn.npz",
         edge_idx = idx,       # (N,) int, positions in the deduped edge array
         y_pred   = y_pred,    # (N,) int in {0,1,2}
         y_proba  = proba,     # (N,3) float   [optional]
         y_true   = y_true)    # (N,) int      [optional]
```

Detection model:

```python
np.savez("results/preds_detection_baseline.npz",
         edge_idx   = idx,
         y_true_bin = y_true_binary,   # (N,) in {0,1}
         y_score    = scores,          # (N,) float
         task       = "detection")
```

`edge_idx` is the only mandatory extra: without it there's no way to know which
edges the predictions belong to. Use the same indices the split artifact uses.

```bash
python scripts/run_eval.py                  # scores everything in results/
python scripts/run_eval.py --preds results/preds_ordinal_gnn.npz
```

Writes `eval/results_comparison.md` plus per-model JSON. Safe to re-run; it
regenerates rather than appends.

---

## Stability

These signatures are fixed for the rest of the project. If something needs to
change, M5 raises it before changing it. Arguments may be *added* with defaults;
existing ones won't be renamed or reordered.
