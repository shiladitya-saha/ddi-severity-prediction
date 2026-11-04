# models/ — Owners: M3 (baseline + ordinal), M4 (mechanism-aware + few-shot)

## Day 1
- M3: `baseline_gnn.py` — binary GNN architecture skeleton (untrained)
- M4: `ordinal_loss.py` (OrdinalCrossEntropy / CORAL-style), `mechanism_gnn.py` skeleton

## Day 2
- M3: dry-run train of baseline on partial Day-1 graph, fix bugs
- M4: complete `mechanism_gnn.py` with edge features wired in, unit test on dummy data
- New: `ordinal_gnn.py` — Model 2 (ordinal loss, no mechanism features)

## Day 3 (needs `data/processed/` + `features/full_features.pt`)
- M3: `train.py` run for Model 1 (baseline) and Model 2 (ordinal) → `results/model1_results.json`, `results/model2_results.json`
- M4: `train.py` run for Model 3 (mechanism-aware ordinal) + `few_shot.py` (MAML/contrastive for new-drug generalization) → `results/model3_results.json`

## Day 4
- Finalize result write-ups: `results/baseline_vs_ordinal.md` (M3), `results/mechanism_aware_final.md` + `results/few_shot_results.md` (M4)

Checkpoints saved to `models/checkpoints/` (gitignored — too large for git; document
the download/regeneration path in each model's results file instead).
