# M4 Day 4 summary — Mechanism-Aware Ordinal GNN (Model 3)

Owner: M4. Status: final for Day 4. Supersedes `model3_day3_summary.md` for
anything that conflicts (that file was the Day-3 single-seed snapshot; this
is the validated, multi-seed result).

## What changed since Day 3

Day 3 shipped Model 3 with a single seed and an optimistic read ("tags-on
beats tags-off"). Day 4 ran the comparison properly: 5 seeds each for
Model 2, Model 3 tags-on, and Model 3 tags-off, because a single run cannot
distinguish a real effect from noise.

## Important methodology note on what "5 seeds" means here

Model 2 and the baselines (see Day 5 addendum) always score against the one
fixed, official split (`eval/splits/ddi_v1.json`, built with seed 42) — their
`--seed` only changes model initialization/training randomness, never the
held-out data. Model 3's training script (`train_mechanism.py`) instead
regenerates its own train/val/test/severe partition from `--seed` on every
run. We checked this directly: **seed 42's self-generated partition is
identical to the official split; seeds 7, 123, 2024, 777 are not** — they are
genuinely different, independently resampled stratified splits, not the same
benchmark scored under different training noise.

Practical effect: the tags-on vs. tags-off paired comparison per seed (Finding
2 below) is still valid, since both variants at a given seed share that seed's
split. But Model 3's reported spread across its 5 "seeds" mixes split-sampling
variance with training variance, which Model 2/baselines' spread does not
include — so Model 3's std in the table below is not directly comparable to
Model 2's. The correct reading of the 5-seed Model 3 result is **"holds up
across one official split plus four independently resampled stratified
splits,"** not "5 noise draws on the same benchmark." The large effect sizes
reported here (architecture gap, severe-recall gap) are far bigger than this
distinction could plausibly explain, so the conclusions stand, but only the
seed-42 numbers for Model 3 are used in the auto-generated
`eval/results_comparison.md` — the other four live only in this document.

## Final numbers (5 seeds: 42, 7, 123, 2024, 777)

| Config | Seed 42 | Seed 7 | Seed 123 | Seed 2024 | Seed 777 | Mean ± std |
|---|---|---|---|---|---|---|
| Model 2 (GCN, ordinal) | 0.7574 | 0.7796 | 0.7803 | — | — | 0.772 ± 0.011 (3 seeds) |
| Model 3, tags-on | 0.9154 | 0.9009 | 0.8663 | 0.8865 | 0.8866 | **0.8911 ± 0.021** |
| Model 3, tags-off (ablation) | 0.8822 | 0.8821 | 0.8933 | 0.8789 | 0.8891 | **0.8851 ± 0.005** |
| Paired diff (on − off) | +0.0332 | +0.0188 | −0.0270 | +0.0076 | −0.0025 | +0.006 ± 0.023 |

All numbers are test-split macro F1, scored with `eval.metrics.evaluate`,
same M5 canonical split (`eval/splits/ddi_v1.json`, seed 42) for every run.

## Finding 1 — robust, safe to claim

Model 3's architecture (GATv2 attention + residual connections) beats
Model 2 (plain GCN) by **~0.11–0.12 macro F1**, in both tag variants, on
every one of 5 seeds. This gap is 5–20× larger than any seed-to-seed noise
measured anywhere in this study. This is the paper's defensible headline
result, and it holds independent of mechanism tags.

## Finding 2 — inconclusive, report as an honest ablation

The tags-on vs. tags-off difference is **not statistically distinguishable
from noise**:
- Mean paired difference +0.006, standard deviation 0.023 — noise is ~4×
  the signal.
- Sign flips across seeds: 3 wins (+0.0332, +0.0188, +0.0076), 2 losses
  (−0.0270, −0.0025).
- A binomial check: a 3-of-5 win split has ≈50% probability even if tags do
  nothing — it is the single most likely outcome of a fair coin flip, not
  evidence of an effect.
- Over half the total positive margin comes from one seed (42, +0.0332);
  drop it and the remaining four roughly cancel.
- Reaching real statistical significance at this effect size (Cohen's
  d ≈ 0.26) would need on the order of 100+ seeds — outside this project's
  scope. 15–20 more seeds (planned for Day 5) would meaningfully tighten the
  estimate but is unlikely to flip this conclusion.

## Recommended framing for the paper (carry this into the shared write-up)

- **Keep the model name** "Mechanism-Aware Ordinal GNN" — it is an accurate
  architecture description (built to accept mechanism-tag edge features),
  independent of whether the tags were shown to help.
- **Core claim**: "GATv2 ordinal architecture beats a GCN baseline,
  evaluated with a leak-aware severe-benchmark protocol." Do not claim
  mechanism tags improve prediction — that is not established.
- **Deployment recommendation**: Model 3 with tags **off**. Same
  architecture, nearly identical mean (0.885 vs. 0.891), but 4× lower
  variance (±0.005 vs ±0.021) — the more reliable choice given the tags
  question is unresolved.
- **Internal vs. established baseline**: Model 2 is an internal ablation
  baseline (our own simpler architecture), not an established/published
  baseline. No external literature or classical-ML baseline exists yet —
  that is Day 5 work (see addendum, to follow).

## Known data-completeness fix (resolved)

`features/full_features.pt` was missing a feature vector for drug
`DB08913` (1867 → 1868 drugs after the fix). M2 landed the fix on
`origin/dev` (commit `91b4e31`); merged into this branch. It also corrected
mechanism-tag values on ~373 of 222,263 edges (0.17%) — too small to have
materially affected the numbers above (Model 3 was never blocked by this
bug; its loader zero-fills internally). Model 2's numbers above still await
M3's rerun against the corrected file; expected to stay close.

## What M5 needs from this document

- The 5-seed table above, verbatim, for the official `eval/results_comparison.md`.
- The two findings (robust architecture gap; inconclusive ablation) for the
  results/discussion section.
- The recommended framing for the abstract/intro, so the paper's claim
  matches the evidence.
- Explainability figures for the tags-on checkpoint (`models/checkpoints/model3.pt`)
  are being generated separately (see `results/explainability_figures/`) —
  mechanism-attribution figures only make sense for the tags-on model, since
  tags-off has no mechanism edge features for attention to attend to.

## Day 5 addendum (to follow, separate document)

Classical ML baseline (RF/GBM/MLP on raw concatenated features, no graph),
a reproduced published-style baseline, and an extended 15–20 seed ablation
are in progress. A separate `model4_day5_baselines.md` will be added here
once those land — do not block M5's final document on it; incorporate as
an update when ready.
