# eval/ — Owner: M5

## Day 1
- `metrics.py` — Macro F1, MAE on severity scale, per-class precision/recall
- `split.py` — 70/15/15 train/val/test, severe class reserved for benchmark
- OGBL-DDI benchmark environment set up

## Day 2
- `attention_viz.py` — attention-weight extraction utilities, tested on dummy predictions

## Day 3 (consumes M3/M4 checkpoints as they land — partial OK)
- Run `metrics.py` on completed checkpoints
- Start `attention_viz.py` visualizations for Model 3 (mechanism-aware)
- Output: `results/results_comparison.md` (early draft)

## Day 4
- `benchmark_severe_subset.py` — held-out severe-interaction benchmark
- `literature_validation.py` — check top predictions against published literature
- Cross-dataset validation on TWOSIDES (`data/processed/twosides_processed.pt`)
- Final outputs: `results/final_comparison.md`, `eval/explainability_figures/`, `results/literature_validation.md`
