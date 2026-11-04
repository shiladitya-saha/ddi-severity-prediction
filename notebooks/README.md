# notebooks/ — shared

Exploratory / scratch notebooks only (EDA, dry-run visualizations, debugging).
Nothing here should be a required input to the pipeline in `data/`, `features/`,
`models/`, or `eval/` — if a notebook produces something the pipeline needs,
move that logic into a `.py` script in the right folder.

Suggested naming: `<member>_<topic>.ipynb`, e.g. `m1_graph_eda.ipynb`.
