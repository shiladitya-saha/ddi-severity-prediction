# features/ — Owner: M2

This documents the feature pipeline as it actually runs against the real
project data (M1's `data/processed/graph.pt`, 1,868 drugs / 222,271 labeled
interaction edges) — not the Day 1/2 planning sketch. It's written so M5 can
cite it directly in the explainability section and so it can be dropped into
`docs/paper_outline.md`'s Methods section with light editing.

## Files and what each one produces

| File | Produces |
|---|---|
| `fingerprints.py` | ECFP4/Morgan fingerprints for every drug, via RDKit, with SMILES fetched from PubChem |
| `mechanism_extraction.py` | Per-edge mechanism tags (CYP450 / protein binding / CNS depression / QT prolongation), keyword-matched against DrugBank's interaction text |
| `embeddings.py` | Optional pretrained ChemBERTa embedding per drug |
| `build_features.py` | Orchestrates all three and writes the two deliverables below |
| `validate_outputs.py` | Self-check: coverage, vector sanity, edge-count consistency, duplicate rows, ID-format consistency |

## File contract

`full_features.pt`: `{drug_id: fingerprint_vector}` — **one entry for every
drug_id in the graph, no exceptions.** Each vector is 1792-dim: 1024
(fingerprint) + 768 (ChemBERTa, real or zero-placeholder — embeddings are on
by default; see below). A drug with no usable SMILES gets an **all-zero
1792-dim vector** rather than being omitted — full coverage of the graph's
drug_ids is a hard contract, since omitting even one breaks any loader that
assumes it (e.g. `models/train.py`'s `align_node_features`, which requires
every graph `drug_id` to have a feature). Which drug_ids got a placeholder,
and why, is logged to `features/fingerprint_placeholders.log` — never hidden
inside a silent zero vector.

`mechanism_edges.csv`: `drug_a, drug_b, mechanism_tag, confidence` — at
least one row per graph edge (an edge with no confident mechanism match gets
one `"unknown"` row; an edge matching multiple categories gets one row per
matching category, by design).

## How the pipeline works

### Molecular fingerprints (`fingerprints.py`)

Each drug's SMILES string is converted to an ECFP4 fingerprint (Extended
Connectivity Fingerprint, Morgan radius 2 / diameter 4) with RDKit's
`rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024)`
(`FINGERPRINT_RADIUS = 2`, `FINGERPRINT_NBITS = 1024`). Each of the 1024 bits
is the hashed presence of a circular substructure (an atom and its
neighborhood up to two bonds out).

**Why SMILES come from PubChem, not DrugBank:** M1's DrugBank source has an
**empty `smiles` column for all 1,868 drugs** — the Mendeley `DDI_data.csv`
source M1 downloads from carries no structure data at all. So
`load_smiles_lookup()` falls back to PubChem's PUG REST API, cross-referencing
each DrugBank ID (`xref RegistryID`) to a PubChem CID and SMILES string, and
caches results to `data/raw/pubchem_smiles.csv` (resumable — already-cached
IDs are skipped on rerun, so an interrupted fetch just picks back up). Of
1,868 real drugs, PubChem resolved SMILES for **1,867**.

**The one exception — DB08913 (Radium Ra 223 Dichloride):** PubChem's xref
lookup returns `not_found` for this ID (confirmed in
`data/raw/pubchem_smiles.csv`: `DB08913,,,not_found`) — there is no SMILES
source for it anywhere in the pipeline, not a parsing failure. Rather than
drop this drug from the graph (which would require renumbering every other
node and was rejected as non-durable — a fresh pipeline run would just
reintroduce it), `build_fingerprints()` gives it a `torch.zeros(1024)`
placeholder fingerprint, and the embedding merge step (see below) likewise
zero-pads its ChemBERTa slot, since there's no SMILES to embed either. The
result: DB08913's full feature vector is legitimately all-zero, and it is
listed in `features/fingerprint_placeholders.log` /
`features/fingerprint_failures.log` (reason: `missing SMILES`) so this is
never mistaken for a silent bug. `validate_outputs.py`'s vector-sanity check
reads that log and excuses exactly this drug_id from its "no unexpected
all-zero vectors" check — any *other* drug_id with an all-zero vector would
still fail that check.

`smiles_to_fingerprint()` raises `ValueError` on an unparseable SMILES;
`build_fingerprints()` catches that per-drug (same zero-placeholder
treatment), so one bad SMILES never aborts the whole run.

**Caveat for explainability:** these bits are hashed, not an injective
substructure index. The generator call here doesn't retain RDKit's
`bitInfo` map (bit → originating atom/fragment), so a bit can't currently be
traced back to an exact chemical fragment. If attention attribution needs
that, the `GetMorganGenerator(...)` call needs to be extended with
`AdditionalOutput.CollectBitInfoMap()` — not implemented.

### Mechanism tagging (`mechanism_extraction.py`)

Each labeled edge `(drug_a, drug_b)` carries a raw interaction description
from `data/processed/edges_labeled.csv` (M1's DrugBank interaction text,
e.g. `"metabolism"`, `"QTc-prolonging activities"`,
`"risk or severity of adverse effects"`). `classify_edge()` runs four
independent keyword classifiers against that string (case-insensitive
substring match, via `_match()`):

| Category | Output tag | Strong keywords → confidence 0.9 | Weak keywords → confidence 0.55 |
|---|---|---|---|
| CYP450 | `cyp450_inhibition` | cyp450, cyp3a4, cyp2d6, cyp2c9, cyp2c19, cyp1a2, "cyp inhibit", "cyp induc" | cyp, cytochrome, "hepatic metabolism", metaboli |
| Protein binding | `protein_binding_competition` | "protein binding", "plasma protein", "albumin binding", "protein-bound displacement" | "protein bound", displacement, albumin |
| CNS depression | `cns_depression` | "cns depress", "central nervous system depress", "respiratory depress" | sedation, sedative, drowsiness, somnolence, opioid, benzodiazepine, alcohol |
| QT prolongation | `qt_prolongation` | "qt prolong", "qtc prolong", "qtc-prolong", torsades | arrhythm, "cardiac conduction", ventricular |

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

**Two keyword fixes applied after auditing the real data** (both were
genuine text-matching misses, not scope gaps — see below):
- DrugBank's actual text is `"QTc-prolonging activities"` (hyphenated, no
  space between `"QTc"` and `"prolonging"`), which didn't match the original
  `"qt prolong"` / `"qtc prolong"` keywords. Adding `"qtc-prolong"` reclaimed
  **5,837 edges** into `qt_prolongation`.
- `"arrhythmogenic activities"` (the adjective form of "causing arrhythmia")
  didn't match the literal keyword `"arrhythmia"` due to the `-ogenic` vs.
  `-ia` suffix. Changing the weak keyword to the stem `"arrhythm"` (still
  matches `"arrhythmia"` too) reclaimed **373 edges**.

**Final split, all 222,271 real edges:** **58,507 confident** (26.3%) /
**163,764 unknown** (73.7%). The unknown share is overwhelmingly a genuine
**scope gap, not a bug**: the bulk of it is DrugBank interaction-type labels
that don't mention any of the 4 target mechanisms at all —
`"risk or severity of adverse effects"` (62,710 edges),
`"serum concentration"` (38,918), `"therapeutic efficacy"` (12,786), and
similar labels for hypotension/anticoagulation/hypoglycemia/etc. This
taxonomy covers 4 mechanisms by design; DrugBank's interaction vocabulary
covers far more. A systematic audit of all 139 unique raw description
strings against all 4 categories' keyword lists (strong + weak) turned up
no further near-miss text variants beyond the two fixed above.

**Known limitation:** classification is pattern matching over the raw text
only. There's no curated reference list (CYP450 substrate tables, measured
plasma protein-binding percentages, a CNS-depressant drug class list, or a
QT-risk drug list) to check against, so confidence reflects how explicit the
raw text's wording was, not an independently validated mechanism
probability.

### ChemBERTa embeddings (`embeddings.py`)

`get_pretrained_embedding(smiles)` tokenizes a SMILES string with
HuggingFace `transformers` and runs it through
`seyonec/ChemBERTa-zinc-base-v1`, then mean-pools `last_hidden_state` over
the token dimension into one 768-dim vector per molecule.
`get_embedding_dim()` returns that width from the loaded model's config, or
the hardcoded fallback `768` if the model can't load (used to size
placeholders consistently even when the model is unavailable).

`transformers` is pinned `>=4.40,<5` (upgraded from a stray 2.1.1 install
that predated Hub repo-id resolution and raised `ValueError` on load);
verified to produce real, finite, non-degenerate embeddings end-to-end, not
the `None` fallback.

`build_features.py`'s `merge_embeddings()` concatenates this onto each
drug's fingerprint. It's **on by default**; pass `--no-embeddings` to
`build_features.py` (or `use_embeddings=False` to `merge_embeddings()`) to
skip it and keep `full_features.pt` fingerprint-only (1024-dim). Any drug
whose embedding fails for any reason — missing SMILES (same set as the
fingerprint placeholders, e.g. DB08913), model unavailable, or a per-drug
inference exception (a handful of real drugs here are large
peptides/oligonucleotides whose SMILES exceed ChemBERTa's 512-token limit)
— gets a 768-dim **zero vector** in that slot instead of being dropped or
left at a different width. On the real data: **1,862 drugs got a real
embedding, 6 got a zero-placeholder** (DB08913 plus 5 oversized
macromolecules). Every row of `full_features.pt` keeps the same total width
regardless of per-drug failures.

**Caveat for M5:** if attention/importance is attributed to the embedding
region (last 768 dims) of a drug's feature vector, a zero region there is
indistinguishable from "the model learned to ignore this" unless you also
know whether that drug's embedding was real or a placeholder. That
distinction is **not currently persisted per-drug** inside `full_features.pt`
itself — only the aggregate real/placeholder counts are printed by
`build_features.py`. Cross-reference against
`features/fingerprint_placeholders.log` (same drug_ids get a placeholder at
both the fingerprint and embedding stage) if this matters for a specific
figure; flag it if a persisted per-drug flag file is needed.

### Vector layout

| Slice | Width | Source | Always present? |
|---|---|---|---|
| `[0:1024]` | 1024 | Morgan/ECFP4 fingerprint, or `torch.zeros(1024)` placeholder (`fingerprints.py`) | Yes — every graph drug_id has an entry; a placeholder is zero, not absent |
| `[1024:1792]` | 768 | ChemBERTa embedding, real or zero-placeholder (`embeddings.py`) | Only when embeddings are enabled (default); absent entirely when run with `--no-embeddings` |

`mechanism_edges.csv` is a separate, edge-indexed table
(`drug_a, drug_b, mechanism_tag, confidence`); it is not folded into
`full_features.pt`'s per-drug vectors and is meant to be joined against
edges by `(drug_a, drug_b)` at train/eval time. See `eval/README.md`'s
"Mapping attention back to a mechanism_tag" section for the bridge that
turns this sparse table into the dense `(E, 4)` matrix M5's explainability
code expects.
