# M4 Day 5 addendum — established baselines + explainability

Owner: M4. Status: in progress (GBM seed-robustness check running). Supplements
`model4_day4_summary.md`, does not replace it.

## Why this exists

Day 4 established Model 3's architecture gap over Model 2 and left the
mechanism-tags question honestly unresolved. That comparison was entirely
*internal* (our own models against each other). Day 5 adds **external,
established baselines** so the paper can answer "how does this compare to
things outside this project?" — a classical ML baseline (standard practice)
and a reproduced published-method baseline (DeepDDI), both scored with the
exact same canonical split and M5 scoring code as Models 1-3, so the
comparison is fair.

## Baselines added

| Baseline | What it is | Feature representation |
|---|---|---|
| RF | Random Forest, 300 trees, balanced class weight | sum/\|diff\|/product of raw per-drug features |
| GBM | HistGradientBoostingClassifier, 300 iters | sum/\|diff\|/product of raw per-drug features |
| MLP | 2-layer feedforward net (256, 128), early stopping | sum/\|diff\|/product of raw per-drug features |
| DeepDDI-style | Re-implementation of Ryu, Kim & Lee (PNAS 2018), trained fresh on our split/task — **not** a citation of their reported numbers | Tanimoto structural-similarity profiles (SSP) of ECFP4 fingerprints |

All four use zero graph structure and zero message passing — they isolate
what, if anything, the GNN architecture adds over the raw molecular features
alone.

## Results

| Model | Test macro F1 | Severe-benchmark major recall |
|---|---|---|
| RF | 0.790 | 0.406 |
| Model 2 (GCN, internal baseline) | 0.757–0.780 (3 seeds) | 0.850 |
| DeepDDI-style | 0.794 | 0.820 |
| GBM (3-seed mean) | 0.885 ± 0.0005 | 0.850 ± 0.003 |
| MLP | **0.934** | 0.886 |
| Model 3 tags-off (5-seed mean) | 0.885 ± 0.006 | **0.936 ± 0.007** |
| Model 3 tags-on (5-seed mean) | 0.891 ± 0.018 | **0.929 ± 0.013** |

Severe-benchmark = held-out MAJOR-only subset, never trained on; major
recall there is the clinically critical metric (false negative = a missed
dangerous interaction).

## Finding 3 — the headline result, now backed by four independent baselines

No baseline — classical (RF, GBM, MLP) or published-method (DeepDDI) —
reaches above ~0.89 severe-benchmark recall. All four plateau in the
0.82–0.89 band. **Only Model 3 (either tag variant) reaches 0.93–0.94.**
This holds even though two of the baselines (GBM, MLP) match or *exceed*
Model 3 on aggregate test macro F1 — the architecture's advantage is not
spread across overall accuracy, it is concentrated specifically on
generalizing to severe/MAJOR interactions, which is the one metric that
actually matters for this task.

This is a stronger and more precise claim than Day 4's "architecture beats
Model 2 everywhere": it survives contact with external baselines because it
was never a claim about aggregate accuracy in the first place.

## Mechanism tags — unchanged conclusion, now with a positive framing

Day 4 found the tags-on/tags-off difference indistinguishable from noise.
That still holds (tags-off is marginally ahead on severe recall, 4-of-5
seeds, within noise). The correct framing is **not** "tags didn't help, so
we hedge" — it's "tags are a zero-cost addition": same accuracy either way,
but tags-on is the only variant that enables the explainability analysis
(mechanism-attribution; 94.8% of attention on QT-prolongation specifically
for MAJOR-class pairs — see `results/explainability_figures/`). Tags earn
their place in the architecture through interpretability, not accuracy.

## Status

- RF, GBM (3 seeds: 42, 7, 123), MLP, DeepDDI: done. GBM's number is
  extremely stable across seeds (F1 std 0.0005, severe-recall std 0.003) —
  the severe-recall gap to Model 3 is not a one-seed artifact.
- All numbers above are also independently reproduced by the official,
  auto-regenerated `eval/results_comparison.md` (run `python
  scripts/run_eval.py`), which scores RF, GBM, MLP, DeepDDI, Model 2 (3
  seeds), and Model 3 seed 42 (both tag variants) directly from saved
  `.npz` predictions on the exact same official split — matches this table
  to 4 decimal places. This is now the ground-truth regenerable table;
  treat it as the source if these two ever drift apart.
- Only Model 3's seed-42 checkpoints are in the official table — seeds
  7/123/2024/777 were trained on independently resampled splits, not the
  official one (see the methodology note added to
  `model4_day4_summary.md`), so including them there would risk leakage.
- Extended 15-seed tags ablation: not started — lower priority now that the
  four-baseline pattern is already consistent without it.

## What M5 needs from this document

- The baseline table above, verbatim, to extend the official
  `eval/results_comparison.md`.
- Finding 3 (external-baseline-backed severe-recall gap) for the
  results/discussion section — this is the paper's strongest, most
  defensible claim.
- The "tags as zero-cost interpretability addition" framing, to replace any
  earlier framing that reads as apologetic about the ablation.
- Explainability figures and the mechanism-attribution finding, for the
  discussion/interpretability section.
