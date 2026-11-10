# Model 3 (Mechanism-Aware Ordinal GNN): Day 3 results

Owner: M4. All numbers are from a single run (seed 42) on CPU. Differences of a few thousandths are
within run-to-run noise and must not be read as effects. Reproduce with the commands in `models/README.md`.

## Setup
- Data: Mendeley `DDI_data.csv` (DrugBank v5.1-derived, CC BY 4.0, doi:10.17632/md5czfsfnd.1), through M1's pipeline.
  1,868 drugs, 222,271 undirected interactions.
- Labels: M1's keyword rules on the interaction type (minor 38.8%, moderate 54.4%, major 6.8%).
  **These are heuristic labels, not clinical ground truth.**
- Node features: ECFP4 fingerprints (1024 bits) built from PubChem SMILES with M2's `fingerprints.py`; no ChemBERTa.
  1,867 of 1,868 drugs have a fingerprint; one gets zeros plus a missing-feature flag.
- Mechanism tags: M2's `mechanism_extraction.py` run on the interaction type. 52,297 of 222,271 edges (24%) are tagged.
- Split: M5 `make_splits(seed=42)`, 70/15/15 stratified, 15% of MAJOR edges (2,282) held out as the severe subset.
- Model: GATv2 (2 layers, 4 heads, hidden 128) + CORAL head. Loss weighted by inverse training-class frequency.
- Stopping: best validation macro F1, patience 5, cap 100 epochs. Test scored once.
- Leakage controls: the scored pair's own mechanism tags are not an input, and a training pair's own edge is removed
  from the message graph while it is scored. Validation/test pairs are never in the message graph.

## Test results (known drugs, random edge split)

| Run | Epochs (best) | Macro F1 | MAE | Accuracy | MAJOR recall | MAJOR precision | Severe-subset MAJOR recall |
|---|---|---|---|---|---|---|---|
| **model3** (official) | 63 (58) | **0.916** | 0.066 | 0.939 | 0.934 | 0.794 | 0.922 |
| model3_30epoch (capped, not converged) | 30 (30) | 0.776 | 0.191 | 0.815 | 0.947 | 0.522 | 0.939 |
| model3_noedge (mechanism tags zeroed) | 63 (58) | 0.920 | 0.061 | 0.945 | 0.940 | 0.793 | 0.919 |
| model3_pairleak (scored pair's own tags given) | 100 (96) | 0.941 | 0.041 | 0.962 | 0.954 | 0.832 | 0.942 |

- The severe subset has only MAJOR edges, so its macro F1 is meaningless; use MAJOR recall.
- `model3_30epoch` was still improving at its cap. A fixed small epoch budget understates this model by 0.14 macro F1;
  any comparison model must use the same stopping rule, not the same epoch count.

## Findings
1. **Mechanism tags on edges gave no measurable gain** (0.916 with, 0.920 without; one seed). The claim that mechanism
   features help is **not supported** by this run. The tags are keyword matches on the same text that defines the
   labels and cover 24% of edges. Several seeds and real drug-level mechanism data are needed before claiming anything.
2. **Giving the model the scored pair's own tags raised macro F1 by 0.025** (0.941 vs 0.916) and MAJOR precision from
   0.79 to 0.83. This run hit the 100-epoch cap while still improving, so the true gap may be larger. The tags are
   derived from the label's own text, so this is a shortcut and is excluded from the main result.
3. **New drugs are hard** (`few_shot_results.json`, drug-disjoint split, 266 test drugs, 6,390 test edges):

   | Known interactions per new drug (k) | Macro F1 | MAE | MAJOR recall |
   |---|---|---|---|
   | 0 (zero-shot) | 0.435 | 0.481 | 0.163 |
   | 1 | 0.420 | 0.524 | 0.383 |
   | 3 | 0.450 | 0.499 | 0.383 |
   | 5 | 0.440 | 0.497 | 0.325 |
   | 10 | 0.450 | 0.475 | 0.290 |

   The model is chosen at epoch 1 (validation falls afterwards): it overfits to drug identity. Zero-shot MAJOR recall
   is 16%. Extra known interactions are added as message-passing context only (no retraining), and the
   differences between k are within noise. Meta-learning is not attempted yet.
4. **The 0.92 on known drugs largely reflects seeing both drugs in training.** On unseen drugs the score halves.

## Caveats the paper must state
- Labels are rule-derived from interaction text; the model partly learns those rules.
- Single seed per configuration.
- Model 1 (binary) and Model 2 (ordinal, no mechanism features) are not available, so there is no
  baseline comparison. `model3_noedge` is an M4 control, not M3's Model 2.
- M2's tagger misses large groups (e.g. "QTc-prolonging activities", 5,802 edges) because its keyword is "qt prolong".

## Files
`results/model3_results.json` (M5 `EvalResult`, test split), `results/model3_details.json` (config, history, all splits),
`results/model3_30epoch_*`, `results/model3_noedge_*`, `results/model3_pairleak_*`, `results/few_shot_results.json`.
Checkpoints are in `models/checkpoints/` (git-ignored).
