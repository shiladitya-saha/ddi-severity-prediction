# Canonical split artifact

- source: `data\processed\edges_labeled.csv`
- seed: `42`
- severe_benchmark_frac: `0.15`
- rows in source: 222271
- unique undirected pairs after dedupe: 222271 (0 removed, keep='max')

```
SplitSet(strategy=stratified_edge+severe_benchmark, seed=42, ratios=(0.7, 0.15, 0.15), severe_benchmark_frac=0.15)
  split                    n       MINOR  MODERATE     MAJOR
  train               153991       60305     84634      9052
  val                  32996       12922     18135      1939
  test                 33002       12924     18137      1941
  severe_benchmark      2282           0         0      2282
```

Load it with:

```python
from eval.split import load_splits
splits = load_splits("eval/splits/ddi_v1.json")
train_idx, val_idx, test_idx = splits.train, splits.val, splits.test
```

Indices are positions in the **deduped** edge array, not rows of the source CSV.
`ddi_v1.kept_rows.npy` maps them back to source rows.

`severe_benchmark` is MAJOR-only, reserved before the 70/15/15, and must not be
trained or tuned on.