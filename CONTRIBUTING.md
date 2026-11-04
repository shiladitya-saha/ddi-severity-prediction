# Contributing

## Responsibilities

| Branch | Role | Responsibilities |
|---|---|---|
| `member/m1-data` | Data Lead | Dataset acquisition and documentation, preprocessing, graph construction, severity labels, and class weights. Do not commit restricted DrugBank data. |
| `member/m2-features` | Feature Engineer | Molecular fingerprints, mechanism features, embeddings, and feature-building pipeline. |
| `member/m3-models` | Model Lead A | Baseline and ordinal GNN models and training integration. |
| `member/m4-mechanism` | Model Lead B | Mechanism-aware ordinal models and few-shot methods. |
| `member/m5-evaluation` | Evaluation and Explainability Lead | Splits, metrics, severe-subset benchmarks, explainability, and literature validation. |

No member names or GitHub usernames are assigned by this document.

## Branches and workflow

- `main` is the stable, reviewed research version.
- `dev` is the shared integration branch.
- Each contributor works on their assigned `member/<id>-<area>` branch. Use the existing branch names above; any future task branch should use a descriptive prefix such as `feature/`, `fix/`, or `docs/` and include a short topic.
- Before starting new work, fetch and pull the latest `dev` into your member branch:

  ```bash
  git fetch origin
  git switch member/m1-data  # replace with your assigned branch
  git merge origin/dev
  ```

- Commit focused changes with a clear message, then push your branch:

  ```bash
  git add <files>
  git commit -m "Describe the change"
  git push -u origin member/m1-data  # replace with your assigned branch
  ```

- Open a Pull Request from your member branch into `dev`. Summarize the change, note relevant data or model assumptions, and include the checks or experiments run. Request review from the relevant teammate before merging.
- Member branches merge into `dev` through reviewed Pull Requests. Do not merge a member branch directly into `main`.
- Only reviewed and tested code from `dev` is merged into `main`, through a Pull Request. Confirm the CI workflow and any relevant research checks pass before release.
- Keep raw datasets, restricted DrugBank data, credentials, and generated model checkpoints out of Git. Follow the repository's `.gitignore` and document reproducible download steps instead.
