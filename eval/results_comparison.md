# Model comparison

**Status: awaiting model outputs.** This file is generated — regenerate it with

```bash
python scripts/run_eval.py
```

and do not hand-edit the tables. It refreshes in seconds, so run it every time a
checkpoint lands rather than updating numbers by hand.

## What it will contain

1. **Severity prediction (ordinal, 3-class)** — Macro F1, MAE, MAJOR recall, QWK,
   balanced accuracy, within-1 accuracy, per model, best value bolded per column.
2. **Held-out severe subset** — MAJOR-only partition reserved before the
   70/15/15 and never trained on. MAJOR recall is the headline there.
3. **Binary interaction detection** — a separate table, because detection and
   severity are different tasks on different label spaces and their numbers are
   not comparable.
4. **Best model detail** — full per-class precision/recall/F1 and confusion matrix.
5. **Severe-subset miss profile** — where MAJOR predictions land when wrong; a
   MAJOR called MODERATE is a different clinical failure from one called MINOR.

## What I need to fill it

| From | What | Status |
| --- | --- | --- |
| M1 | `data/processed/edges_labeled.csv` | **blocking** — no split artifact without it |
| M3 | `results/preds_<model>.npz` per model | waiting |
| M4 | `results/preds_mechanism_gnn.npz` | waiting |

Predictions contract — one file per model:

```python
np.savez("results/preds_ordinal_gnn.npz",
         edge_idx=idx,        # (N,) positions in the deduped edge array
         y_pred=y_pred,       # (N,) int, 0=MINOR 1=MODERATE 2=MAJOR
         y_proba=proba,       # (N,3) float, optional — enables AUC/AP
         y_true=y_true)       # (N,) int, optional
```

For a binary detection model, save `y_true_bin`, `y_score` and `task="detection"`
instead; it lands in the detection table.

Once M1's edges exist:

```bash
python scripts/make_splits.py --edges data/processed/edges_labeled.csv
python scripts/run_eval.py
```

The harness itself is verified end-to-end on synthetic data — 65 tests, including
a full `make_splits.py` → `run_eval.py` run — so when real predictions arrive the
only unknown is the model, not the scoring.
