"""Cross-dataset validation against TWOSIDES (independent of our DrugBank-derived labels).

TWOSIDES severity is a HEURISTIC derived from FAERS co-report signals (see
data/twosides_processed.py's docstring) -- confounded by indication and
polypharmacy, NOT ground truth. This script therefore reports agreement /
enrichment against an independent signal, not accuracy.

Three pair groups, scored separately:
  excluded_train : the pair's canonical index is in our official training
                    set -- a model that trained on it isn't being tested for
                    generalization here, so these are reported as a count
                    only, never scored.
  held_out_overlap: the pair IS in our DrugBank graph (edges_labeled.csv) but
                    its canonical index is in val/test/severe_benchmark, so
                    scoring it is fair (same as the main eval, just compared
                    against a different label source).
  novel           : the pair is NOT in edges_labeled.csv at all (both drugs
                    are in our graph, the specific pair never was) -- the
                    genuine cross-dataset generalization test.

Model 2 / Model 3 (both tag variants) are scored from their existing
checkpoints (zero retrain cost). RF, GBM and a DeepDDI-style model are
retrained here (cheap) so they can be scored in the same run. MLP is
skipped deliberately -- it took ~3.5h to train and isn't central to the
paper's headline claim.

    python -m scripts.validate_twosides

Writes results/twosides_validation_results.json and
results/twosides_validation_summary.md.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier

from eval.split import load_splits
from models.baseline_deepddi import DeepDDINet, tanimoto_similarity_matrix
from models.mechanism_gnn import MechanismOrdinalGNN, load_mechanism_table, mechanism_matrix
from models.ordinal_gnn import OrdinalGNN, load_ordinal_inputs, predict_ordinal
from models.ordinal_loss import coral_logits_to_probs, coral_predict
from models.train_mechanism import load_inputs as load_mech_inputs
from models.train_mechanism import message_graph, predict as mech_predict

LABELS = ("MINOR", "MODERATE", "MAJOR")


def load_twosides_pairs(splits):
    df = pd.read_csv("data/processed/twosides_validation.csv")
    canonical_pairs = np.asarray(splits.meta["canonical_pairs"], dtype=np.int64)
    pair_to_idx = {(int(a), int(b)): i for i, (a, b) in enumerate(canonical_pairs)}

    train_set = set(np.asarray(splits["train"]).tolist())
    group = []
    official_idx = []
    for src, dst in zip(df["src"], df["dst"]):
        idx = pair_to_idx.get((int(src), int(dst)))
        official_idx.append(idx if idx is not None else -1)
        if idx is None:
            group.append("novel")
        elif idx in train_set:
            group.append("excluded_train")
        else:
            group.append("held_out_overlap")
    df["official_idx"] = official_idx
    df["group"] = group
    return df


def enrichment(pred_labels, twosides_severity):
    """Fraction of heuristic-MAJOR (TWOSIDES severity==2) pairs the model also calls MAJOR."""
    is_major = twosides_severity == 2
    if is_major.sum() == 0:
        return None
    return float((pred_labels[is_major] == 2).mean())


def agreement(pred_labels, twosides_severity):
    return float((pred_labels == twosides_severity).mean())


def score_group(name, df_group, pred_labels):
    if len(df_group) == 0:
        return None
    sev = df_group["twosides_severity"].to_numpy()
    return {
        "n": int(len(df_group)),
        "agreement": agreement(pred_labels, sev),
        "major_enrichment": enrichment(pred_labels, sev),
        "pred_severity_counts": {LABELS[i]: int((pred_labels == i).sum()) for i in range(3)},
    }


def main() -> None:
    splits = load_splits("eval/splits/ddi_v1.json")
    df = load_twosides_pairs(splits)
    print(df["group"].value_counts(), flush=True)

    results = {}

    # ---- Model 2 (OrdinalGNN) and Model 3 (MechanismOrdinalGNN, both variants) ----
    drug_ids, x, split, _ = load_ordinal_inputs(
        "data/processed/graph.pt", "features/full_features.pt", "eval/splits/ddi_v1.json",
    )
    target_pairs_all = torch.tensor(df[["src", "dst"]].to_numpy().T, dtype=torch.long)

    ckpt2 = torch.load("models/checkpoints/model2_ordinal_gnn.pt", map_location="cpu", weights_only=False)
    model2 = OrdinalGNN(**ckpt2["model_config"])
    model2.load_state_dict(ckpt2["model_state_dict"])
    model2.eval()
    _, _, pred2 = predict_ordinal(model2, x, split.message_edge_index, target_pairs_all)
    results["Model2_GCN"] = {g: score_group(g, df[df["group"] == g], pred2.numpy()[df["group"] == g])
                              for g in ("held_out_overlap", "novel")}

    mech_data = load_mech_inputs(features_path="features/full_features.pt", mechanism_path="features/mechanism_edges.csv")
    train_pairs = mech_data.pairs[:, torch.as_tensor(np.asarray(splits["train"], dtype=np.int64), dtype=torch.long)]
    for tag, ckpt_path, use_edge_attr in (
        ("Model3_tagson", "models/checkpoints/model3.pt", True),
        ("Model3_tagsoff", "models/checkpoints/model3_noedge.pt", False),
    ):
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model = MechanismOrdinalGNN(**ckpt["model_config"])
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        logits = mech_predict(model, mech_data.x, train_pairs, mech_data.table, target_pairs_all,
                              use_pair_attr=False, use_edge_attr=use_edge_attr)
        pred = coral_predict(logits).numpy()
        results[tag] = {g: score_group(g, df[df["group"] == g], pred[df["group"] == g])
                        for g in ("held_out_overlap", "novel")}
    print("scored Model2/Model3 from existing checkpoints", flush=True)

    # ---- classical baselines, retrained here for the extra scoring step ----
    def pair_features(xf, pairs):
        a, b = xf[pairs[0]], xf[pairs[1]]
        return torch.cat([a + b, (a - b).abs(), a * b], dim=1).numpy()

    X_train = pair_features(x, split.train_pairs)
    y_train = split.train_labels.numpy()
    X_target = pair_features(x, target_pairs_all)

    rf = RandomForestClassifier(n_estimators=300, class_weight="balanced", n_jobs=-1, random_state=42)
    rf.fit(X_train, y_train)
    pred_rf = rf.predict(X_target)
    results["RF"] = {g: score_group(g, df[df["group"] == g], pred_rf[df["group"] == g])
                     for g in ("held_out_overlap", "novel")}
    print("scored RF", flush=True)

    gbm = HistGradientBoostingClassifier(max_iter=300, class_weight="balanced", random_state=42)
    gbm.fit(X_train, y_train)
    pred_gbm = gbm.predict(X_target)
    results["GBM"] = {g: score_group(g, df[df["group"] == g], pred_gbm[df["group"] == g])
                      for g in ("held_out_overlap", "novel")}
    print("scored GBM", flush=True)

    fp = x[:, :1024].numpy()
    ssp = torch.as_tensor(tanimoto_similarity_matrix(fp), dtype=torch.float32)

    def pair_ssp(pairs):
        a, b = ssp[pairs[0]], ssp[pairs[1]]
        return torch.cat([a, b], dim=1)

    X_train_ssp, y_train_t = pair_ssp(split.train_pairs), split.train_labels
    deep_model = DeepDDINet(input_dim=X_train_ssp.size(1))
    counts = torch.bincount(y_train_t, minlength=3).float()
    class_weights = counts.sum() / counts.clamp(min=1)
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(deep_model.parameters(), lr=1e-3, weight_decay=1e-5)
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X_train_ssp, y_train_t), batch_size=4096, shuffle=True,
    )
    deep_model.train()
    for epoch in range(15):
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = criterion(deep_model(xb), yb)
            loss.backward()
            optimizer.step()
    deep_model.eval()
    with torch.no_grad():
        pred_deep = deep_model(pair_ssp(target_pairs_all)).argmax(dim=1).numpy()
    results["DeepDDI"] = {g: score_group(g, df[df["group"] == g], pred_deep[df["group"] == g])
                          for g in ("held_out_overlap", "novel")}
    print("scored DeepDDI-style", flush=True)

    summary = {
        "note": "TWOSIDES severity is a heuristic FAERS-signal proxy, not ground truth. "
                "Numbers are agreement/enrichment against an independent signal.",
        "group_sizes": df["group"].value_counts().to_dict(),
        "results": results,
    }
    Path("results/twosides_validation_results.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("wrote results/twosides_validation_results.json", flush=True)

    lines = ["# TWOSIDES cross-dataset validation", "",
             "TWOSIDES severity is a heuristic FAERS co-report signal (confounded by "
             "indication/polypharmacy), not ground truth -- these are agreement/enrichment "
             "numbers against an independent signal, not accuracy.", "",
             f"Group sizes: {df['group'].value_counts().to_dict()}", "",
             "`major_enrichment` = of pairs TWOSIDES flags as heuristic-MAJOR (serious-AE "
             "signal), what fraction does the model also call MAJOR. `novel` = pairs never "
             "in our training data at all (both drugs known, pair unseen).", "",
             "| Model | Group | n | Agreement | MAJOR enrichment |",
             "| --- | --- | --- | --- | --- |"]
    for name, groups in results.items():
        for g, r in groups.items():
            if r is None:
                continue
            enrich_str = "n/a" if r["major_enrichment"] is None else f"{r['major_enrichment']:.4f}"
            lines.append(f"| {name} | {g} | {r['n']} | {r['agreement']:.4f} | {enrich_str} |")
    Path("results/twosides_validation_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("wrote results/twosides_validation_summary.md", flush=True)


if __name__ == "__main__":
    main()
