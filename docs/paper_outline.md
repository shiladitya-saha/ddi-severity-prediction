# Paper Outline (IEEE format) — fill in after the sprint

1. **Abstract**
2. **Introduction** — binary DDI framing problem, severity labels ignored, mechanism-blind models
3. **Related Work**
4. **Problem Formulation** — ordinal severity classification, pharmacological mechanism integration
5. **Method**
   - 5.1 Data & graph construction ← `data/README.md`, `data/processed/`
   - 5.2 Feature extraction (fingerprints, mechanism edges) ← `features/README.md`
   - 5.3 Model architectures: Baseline / Ordinal GNN / Mechanism-Aware Ordinal GNN ← `models/`
   - 5.4 Few-shot / zero-shot generalization ← `models/few_shot.py`
6. **Experiments**
   - 6.1 Setup, splits, benchmark ← `eval/split.py`, OGBL-DDI
   - 6.2 Results ← `results/final_comparison.md`
   - 6.3 Explainability ← `eval/explainability_figures/`
   - 6.4 Cross-dataset validation (TWOSIDES) ← `results/literature_validation.md`
7. **Discussion / Limitations**
8. **Conclusion**
9. **References** ← `docs/references.bib`
