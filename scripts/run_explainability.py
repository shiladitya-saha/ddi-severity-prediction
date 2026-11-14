"""M5 explainability pipeline for Model 3 (tags-on), run on M4's behalf.

Loads the final tags-on checkpoint, captures GATv2 attention over the
training message graph, and renders every figure eval.attention_viz offers:
top-attended edges, mechanism-channel attribution, attribution-by-severity-
class heatmap, and an ego-network figure for one illustrative drug.

    python -m scripts.run_explainability

Writes results/explainability_figures/*.png and *_attention_summary.json.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import torch

from eval.attention_viz import AttentionCapture, save_figures, top_edges_to_markdown
from eval.split import make_splits
from models.mechanism_gnn import MechanismOrdinalGNN, mechanism_matrix
from models.train_mechanism import load_inputs, message_graph

CHECKPOINT = "models/checkpoints/model3.pt"
OUTDIR = "results/explainability_figures"


def main() -> None:
    data = load_inputs(features_path="features/full_features.pt", mechanism_path="features/mechanism_edges.csv")
    edge_rows = data.pairs.t().numpy()
    splits = make_splits(edge_rows, data.labels.numpy(), seed=42, severe_benchmark_frac=0.15)
    part = {name: torch.as_tensor(splits[name], dtype=torch.long) for name in splits.names}
    train_pairs = data.pairs[:, part["train"]]
    train_labels = data.labels[part["train"]]

    ckpt = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    model = MechanismOrdinalGNN(**{k: v for k, v in ckpt["model_config"].items()})
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    edge_index, edge_attr = message_graph(train_pairs, mechanism_matrix(train_pairs, data.table), True)

    with AttentionCapture(model, include=("convs",)) as cap:
        with torch.no_grad():
            model.encode(data.x, edge_index, edge_attr)
    att = cap.merged()
    print(f"captured attention for {att.shape[0]} directed training edges", flush=True)

    nodes = pd.read_csv("data/processed/nodes.csv", dtype={"drug_id": str})
    drug_names = {i: n for i, n in enumerate(nodes["name"].tolist())}

    # Labels aligned to the directed edge_index: train_labels repeated for both directions.
    directed_labels = torch.cat([train_labels, train_labels]).numpy()

    # Pick an illustrative focus drug: the one with the most training edges.
    deg = torch.bincount(edge_index[0], minlength=data.x.size(0))
    focus_drug = int(torch.argmax(deg).item())

    report = save_figures(
        att, edge_index.numpy(), mechanism_matrix=edge_attr.numpy(), labels=directed_labels,
        outdir=OUTDIR, model_name="model3_tagson", drug_names=drug_names,
        top_k=20, focus_drug=focus_drug,
    )
    print(f"wrote {len(report.figures)} figures + {report.summary_path}", flush=True)

    md_path = Path(OUTDIR) / "model3_tagson_top_edges.md"
    md_path.write_text(top_edges_to_markdown(report.top_edges), encoding="utf-8")
    print(f"wrote {md_path}", flush=True)


if __name__ == "__main__":
    main()
