# TWOSIDES cross-dataset validation

TWOSIDES severity is a heuristic FAERS co-report signal (confounded by indication/polypharmacy), not ground truth -- these are agreement/enrichment numbers against an independent signal, not accuracy.

Group sizes: {'novel': 62077, 'excluded_train': 18842, 'held_out_overlap': 8342}

`major_enrichment` = of pairs TWOSIDES flags as heuristic-MAJOR (serious-AE signal), what fraction does the model also call MAJOR. `novel` = pairs never in our training data at all (both drugs known, pair unseen).

| Model | Group | n | Agreement | MAJOR enrichment |
| --- | --- | --- | --- | --- |
| Model2_GCN | held_out_overlap | 8342 | 0.3313 | 0.1588 |
| Model2_GCN | novel | 62077 | 0.3239 | 0.1766 |
| Model3_tagson | held_out_overlap | 8342 | 0.3258 | 0.1194 |
| Model3_tagson | novel | 62077 | 0.3273 | 0.0827 |
| Model3_tagsoff | held_out_overlap | 8342 | 0.3183 | 0.1258 |
| Model3_tagsoff | novel | 62077 | 0.3098 | 0.0919 |
| RF | held_out_overlap | 8342 | 0.3335 | 0.0450 |
| RF | novel | 62077 | 0.3570 | 0.0154 |
| GBM | held_out_overlap | 8342 | 0.3340 | 0.1188 |
| GBM | novel | 62077 | 0.3410 | 0.0934 |
| DeepDDI | held_out_overlap | 8342 | 0.3322 | 0.1358 |
| DeepDDI | novel | 62077 | 0.3364 | 0.1265 |
