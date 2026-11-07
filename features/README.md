# features/ — Owner: M2

## Day 1
- `fingerprints.py` — SMILES → RDKit molecular fingerprints, tested on a 50-drug subset
- `mechanism_extraction.py` (scaffold) — parse CYP450 pathways / protein targets from DrugBank fields

## Day 2 (blocks M3/M4 — push as soon as `data/processed/graph.pt` lands)
- Full fingerprint extraction for all drugs in `data/processed/graph.pt`
- `mechanism_extraction.py` — complete edge-level mechanism tags: CYP450 inhibition,
  protein binding competition, CNS depression, QT prolongation
- `embeddings.py` — optional pretrained drug2vec / ChemBERTa embeddings
- `build_features.py` — assembles everything into one feature bundle
- Output: `full_features.pt`, `mechanism_edges.csv`

## File contract
`full_features.pt`: `{drug_id: fingerprint_vector}`
`mechanism_edges.csv`: `drug_a, drug_b, mechanism_tag, confidence`

Each `fingerprint_vector` is 1024-dim (fingerprint only) or 1792-dim
(fingerprint + ChemBERTa embedding, the default — see Vector layout below).
Every drug in `full_features.pt` has the same width; a drug is either
fingerprinted and present, or fingerprinting failed and it is omitted
entirely (reported via `build_features.py`'s coverage check), never
zero-padded at the fingerprint stage.

## How the pipeline works (for M5 / paper Methods section)

This section documents the actual Day 2 implementation of each file below,
so M5 can map GNN attention weights back to concrete chemical features for
the Day 4 explainability work, and so it can be adapted into the paper's
Methods with light editing.

### Molecular fingerprints (`fingerprints.py`)

Each drug's SMILES string is converted to an ECFP4 fingerprint (Extended
Connectivity Fingerprint, Morgan radius 2 / diameter 4) with RDKit's
`rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024)`
(`FINGERPRINT_RADIUS = 2`, `FINGERPRINT_NBITS = 1024`). Each of the 1024 bits
is the hashed presence of a circular substructure (an atom and its
neighborhood up to two bonds out). `smiles_to_fingerprint()` raises
`ValueError` on an unparseable SMILES; `build_fingerprints()` catches that
per-drug, so one bad SMILES fails only that drug (logged to
`features/fingerprint_failures.log`) rather than aborting the batch.

**Caveat for explainability:** these bits are hashed, not an injective
substructure index. The generator call here doesn't retain RDKit's
`bitInfo` map (bit → originating atom/fragment), so a bit can't currently be
traced back to an exact chemical fragment. If attention attribution needs
that, `GetMorganGenerator(...)` calls need to be extended with
`AdditionalOutput.CollectBitInfoMap()` — not implemented yet.

### Mechanism tagging (`mechanism_extraction.py`)

Each labeled edge `(drug_a, drug_b)` carries a raw `mechanism_tag` string
from `data_loading.load_graph()`. `classify_edge()` runs four independent
keyword classifiers against that string (case-insensitive substring match,
via `_match()`):

| Category | Output tag | Strong keywords → confidence 0.9 | Weak keywords → confidence 0.55 |
|---|---|---|---|
| CYP450 | `cyp450_inhibition` | cyp450, cyp3a4, cyp2d6, cyp2c9, cyp2c19, cyp1a2, "cyp inhibit", "cyp induc" | cyp, cytochrome, "hepatic metabolism", metaboli |
| Protein binding | `protein_binding_competition` | "protein binding", "plasma protein", "albumin binding", "protein-bound displacement" | "protein bound", displacement, albumin |
| CNS depression | `cns_depression` | "cns depress", "central nervous system depress", "respiratory depress" | sedation, sedative, drowsiness, somnolence, opioid, benzodiazepine, alcohol |
| QT prolongation | `qt_prolongation` | "qt prolong", "qtc prolong", torsades | arrhythmia, "cardiac conduction", ventricular |

A category is kept only if confidence ≥ `CONFIDENCE_THRESHOLD = 0.5`; since
both strong (0.9) and weak (0.55) hits already clear that bar, in practice
every keyword match is kept and only a complete non-match is discarded. An
edge can match more than one category — e.g. an opioid/CYP3A4 interaction
produces both a `cns_depression` and a `cyp450_inhibition` row — so
`mechanism_edges.csv` can (by design) have multiple rows for the same
`(drug_a, drug_b)` pair, each a different `mechanism_tag`; that's distinct
from a true duplicate (same pair, same tag, repeated), which is a bug.
If no category clears the threshold, the edge gets exactly one fallback
row, `("unknown", best_confidence)`, where `best_confidence` is the
highest score any of the four classifiers produced (0.0 for a clean
non-match, a near-miss score otherwise, useful for triage).

**Known limitation** (see the module's own KNOWN GAP note): classification
is pattern matching over the raw tag string only. There's no curated
reference list yet (CYP450 substrate tables, measured plasma
protein-binding percentages, a CNS-depressant drug class list, or a
QT-risk drug list) to check against, so confidence reflects how explicit
the raw tag's wording was, not an independently validated mechanism
probability.

### ChemBERTa embeddings (`embeddings.py`, optional)

`get_pretrained_embedding(smiles)` tokenizes a SMILES string with
HuggingFace `transformers` and runs it through
`seyonec/ChemBERTa-zinc-base-v1`, then mean-pools `last_hidden_state` over
the token dimension into one 768-dim vector per molecule.
`get_embedding_dim()` returns that width from the loaded model's config, or
the hardcoded fallback `768` if the model can't load.

As of Day 2, `transformers` is pinned `>=4.40,<5` (upgraded from a stray
2.1.1 install that predated Hub repo-id resolution and raised `ValueError`
on load) and has been verified to produce real, finite, non-degenerate
embeddings for sample SMILES in `embeddings.py`'s own smoke test — not the
`None` fallback.

`build_features.py`'s `merge_embeddings()` concatenates this onto each
drug's fingerprint, giving the layout in "Vector layout" below. It's on by
default; pass `--no-embeddings` to `build_features.py` (or
`use_embeddings=False` to `merge_embeddings()`) to skip it and keep
`full_features.pt` fingerprint-only. Any drug whose embedding fails for any
reason (missing SMILES, model unavailable, a per-drug inference exception)
gets a 768-dim **zero vector** in that slot instead of being dropped or
left at a different width — every row of `full_features.pt` keeps the same
total width regardless of per-drug embedding failures. The per-run count of
real vs. zero-placeholder embeddings is printed in `build_features.py`'s
summary but is **not currently persisted per-drug** in `full_features.pt`
itself.

**Caveat for M5:** if attention/importance is attributed to the embedding
region (last 768 dims) of a drug's feature vector, a zero region there is
indistinguishable from "the model learned to ignore this" unless you also
know whether that drug's embedding was real or a placeholder. If that
distinction matters for the explainability writeup, flag it — we'd add a
persisted per-drug flag (e.g. a parallel `{drug_id: bool}` file) to
`full_features.pt`'s neighborhood; not implemented yet.

### Vector layout

| Slice | Width | Source | Always present? |
|---|---|---|---|
| `[0:1024]` | 1024 | Morgan/ECFP4 fingerprint (`fingerprints.py`) | Yes — a drug with no fingerprint is omitted from `full_features.pt` entirely, never zero-padded |
| `[1024:1792]` | 768 | ChemBERTa embedding, real or zero-placeholder (`embeddings.py`) | Only when embeddings are enabled (default); absent entirely when run with `--no-embeddings` |

`mechanism_edges.csv` is a separate, edge-indexed table
(`drug_a, drug_b, mechanism_tag, confidence`); it is not folded into
`full_features.pt`'s per-drug vectors and is meant to be joined against
edges by `(drug_a, drug_b)` at train/eval time.
