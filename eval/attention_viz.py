"""
Attention extraction + explainability visualizations.

Owner: M5 (Eval / Explainability Lead) - see eval/README.md

This is the module that answers the reviewer question "why should I trust this
prediction?". It does two separable jobs:

1. EXTRACTION - pull attention weights out of M4's mechanism-aware GNN without
   M4 having to change their forward signature. `AttentionCapture` registers
   forward hooks and picks up anything shaped like `(edge_index, alpha)`, which
   is what PyG's GATConv/TransformerConv return with
   `return_attention_weights=True`. torch is imported lazily, so this module
   imports fine on a machine with no torch.

2. ATTRIBUTION + FIGURES - pure numpy/matplotlib. Given attention weights and
   M2's mechanism edge features, report how much attention each mechanism
   channel (CYP450, protein binding, CNS depression, QT) received, and render
   the figures that go into results/explainability_figures/.

Usage
-----
    from eval.attention_viz import AttentionCapture, mechanism_attribution, save_figures

    with AttentionCapture(model) as cap:
        logits = model.encode(data.x, data.edge_index, data.edge_attr)
    att = cap.merged()                      # (E,) attention per edge

    attr = mechanism_attribution(att, mech_matrix, MECHANISM_TAGS)
    save_figures(att, data.edge_index, mech_matrix, outdir="results/explainability_figures")
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np

import matplotlib

matplotlib.use("Agg")  # headless: figures are files, never windows
import matplotlib.pyplot as plt  # noqa: E402

ArrayLike = Union[Sequence[float], np.ndarray]

# M2's mechanism edge-feature channels, in column order. Keep in sync with
# features/mechanism_edges.csv.
MECHANISM_TAGS: List[str] = [
    "CYP450_inhibition",
    "protein_binding_competition",
    "CNS_depression",
    "QT_prolongation",
]

PALETTE = ["#2B6CB0", "#C05621", "#2F855A", "#6B46C1", "#B83280", "#4A5568"]


# --------------------------------------------------------------------------- #
# 1. Extraction
# --------------------------------------------------------------------------- #


@dataclass
class AttentionRecord:
    """One layer's attention output."""

    module_name: str
    edge_index: Optional[np.ndarray]  # (2, E) or None
    alpha: np.ndarray  # (E, heads) or (E,)

    @property
    def n_edges(self) -> int:
        return int(self.alpha.shape[0])

    @property
    def n_heads(self) -> int:
        return int(self.alpha.shape[1]) if self.alpha.ndim == 2 else 1


def _to_numpy(obj) -> np.ndarray:
    """torch.Tensor -> ndarray without importing torch at module level."""
    if hasattr(obj, "detach"):
        obj = obj.detach()
    if hasattr(obj, "cpu"):
        obj = obj.cpu()
    if hasattr(obj, "numpy"):
        return np.asarray(obj.numpy())
    return np.asarray(obj)


def _looks_like_attention(value) -> bool:
    """A PyG attention return is a 2-tuple of (edge_index, alpha)."""
    if not isinstance(value, tuple) or len(value) != 2:
        return False
    ei, alpha = value
    try:
        ei_arr, al_arr = _to_numpy(ei), _to_numpy(alpha)
    except Exception:
        return False
    return (
        ei_arr.ndim == 2
        and ei_arr.shape[0] == 2
        and al_arr.ndim in (1, 2)
        and al_arr.shape[0] == ei_arr.shape[1]
    )


class AttentionCapture:
    """
    Context manager that records attention from every attention-ish submodule.

    Works with any torch module. If a layer does not emit attention it is simply
    not recorded, so this is safe to wrap around M3's baseline too.

        with AttentionCapture(model, include=("gat", "attn")) as cap:
            out = model(...)
        cap.records          # list[AttentionRecord]
        cap.merged()         # (E,) head- and layer-averaged attention
    """

    def __init__(
        self,
        model,
        include: Optional[Iterable[str]] = None,
        exclude: Optional[Iterable[str]] = None,
    ):
        self.model = model
        self.include = tuple(include) if include else None
        self.exclude = tuple(exclude) if exclude else ()
        self.records: List[AttentionRecord] = []
        self._handles: List[object] = []

    # - hook plumbing ------------------------------------------------------- #

    def _wanted(self, name: str) -> bool:
        if any(x in name for x in self.exclude):
            return False
        if self.include is None:
            return True
        return any(x in name for x in self.include)

    def _make_hook(self, name: str):
        def hook(_module, _inputs, output):
            candidates = output if isinstance(output, tuple) else (output,)
            # a GATConv with return_attention_weights=True returns
            # (out, (edge_index, alpha)); scan every element to be tolerant
            for item in candidates:
                if _looks_like_attention(item):
                    ei, alpha = item
                    self.records.append(
                        AttentionRecord(
                            module_name=name,
                            edge_index=_to_numpy(ei).astype(int),
                            alpha=_to_numpy(alpha).astype(float),
                        )
                    )
                    return
            if _looks_like_attention(candidates):
                ei, alpha = candidates  # type: ignore[misc]
                self.records.append(
                    AttentionRecord(
                        module_name=name,
                        edge_index=_to_numpy(ei).astype(int),
                        alpha=_to_numpy(alpha).astype(float),
                    )
                )

        return hook

    def __enter__(self) -> "AttentionCapture":
        if not hasattr(self.model, "named_modules"):
            raise TypeError("AttentionCapture expects a torch nn.Module")
        for name, module in self.model.named_modules():
            if name and self._wanted(name) and hasattr(module, "register_forward_hook"):
                self._handles.append(module.register_forward_hook(self._make_hook(name)))
        return self

    def __exit__(self, *exc) -> None:
        for h in self._handles:
            try:
                h.remove()  # type: ignore[attr-defined]
            except Exception:
                pass
        self._handles.clear()

    # - results ------------------------------------------------------------- #

    def __len__(self) -> int:
        return len(self.records)

    def merged(self, reduce: str = "mean") -> np.ndarray:
        """
        Collapse heads then layers into one attention weight per edge.

        All recorded layers must agree on edge count (true for a stack of
        GATConvs over the same edge_index).
        """
        if not self.records:
            raise RuntimeError(
                "no attention recorded - did the model run, and is "
                "return_attention_weights=True on the attention layers?"
            )
        sizes = {r.n_edges for r in self.records}
        if len(sizes) != 1:
            raise ValueError(
                f"layers disagree on edge count {sorted(sizes)}; merge them "
                "yourself or capture one layer at a time with include=(...)"
            )
        per_layer = [aggregate_heads(r.alpha, reduce=reduce) for r in self.records]
        stacked = np.vstack(per_layer)
        return stacked.mean(axis=0) if reduce == "mean" else stacked.max(axis=0)

    def layer_matrix(self, reduce: str = "mean") -> Tuple[np.ndarray, List[str]]:
        """(n_layers, E) attention matrix plus layer names, for the heatmap."""
        if not self.records:
            raise RuntimeError("no attention recorded")
        rows = [aggregate_heads(r.alpha, reduce=reduce) for r in self.records]
        return np.vstack(rows), [r.module_name for r in self.records]


def aggregate_heads(alpha: ArrayLike, reduce: str = "mean") -> np.ndarray:
    """(E, heads) -> (E,). reduce in {mean, max, sum}."""
    a = np.asarray(alpha, dtype=float)
    if a.ndim == 1:
        return a
    if a.ndim != 2:
        raise ValueError(f"alpha must be 1-D or 2-D, got {a.shape}")
    if reduce == "mean":
        return a.mean(axis=1)
    if reduce == "max":
        return a.max(axis=1)
    if reduce == "sum":
        return a.sum(axis=1)
    raise ValueError(f"unknown reduce '{reduce}'")


def normalize_attention(att: ArrayLike) -> np.ndarray:
    """Scale to [0, 1] for comparable figures across models."""
    a = np.asarray(att, dtype=float).ravel()
    if a.size == 0:
        return a
    lo, hi = float(np.min(a)), float(np.max(a))
    if np.isclose(hi, lo):
        return np.zeros_like(a)
    return (a - lo) / (hi - lo)


# --------------------------------------------------------------------------- #
# 2. Attribution
# --------------------------------------------------------------------------- #


def top_k_attended_edges(
    att: ArrayLike,
    edge_index: ArrayLike,
    k: int = 20,
    drug_names: Optional[Dict[int, str]] = None,
    labels: Optional[ArrayLike] = None,
) -> List[Dict[str, object]]:
    """
    The k highest-attention edges, as rows ready for a markdown table or for
    literature validation on Day 4.
    """
    a = np.asarray(att, dtype=float).ravel()
    ei = np.asarray(edge_index)
    if ei.ndim != 2:
        raise ValueError("edge_index must be 2-D")
    if ei.shape[0] != 2:
        ei = ei.T
    if ei.shape[1] != a.size:
        raise ValueError(f"attention length {a.size} != edges {ei.shape[1]}")
    k = int(min(max(k, 1), a.size))

    order = np.argsort(-a)[:k]
    names = drug_names or {}
    y = np.asarray(labels) if labels is not None else None

    rows: List[Dict[str, object]] = []
    for rank, i in enumerate(order, start=1):
        src, dst = int(ei[0, i]), int(ei[1, i])
        row: Dict[str, object] = {
            "rank": rank,
            "edge_idx": int(i),
            "drug_a": names.get(src, f"drug_{src}"),
            "drug_b": names.get(dst, f"drug_{dst}"),
            "attention": float(a[i]),
        }
        if y is not None:
            from .metrics import SEVERITY_LABELS

            row["severity"] = SEVERITY_LABELS.get(int(y[i]), str(y[i]))
        rows.append(row)
    return rows


def mechanism_attribution(
    att: ArrayLike,
    mechanism_matrix: ArrayLike,
    tags: Sequence[str] = MECHANISM_TAGS,
    normalize: bool = True,
) -> Dict[str, float]:
    """
    How much attention each mechanism channel received.

    `mechanism_matrix` is (E, n_tags) from M2's mechanism_edges.csv - binary
    flags or continuous strengths. Score per tag is sum(attention * flag); with
    `normalize=True` scores sum to 1 and read as "share of attention".

    This is the number that supports the mechanism-awareness novelty claim: if
    the mechanism-aware model is working, attention should concentrate on the
    channels that are pharmacologically responsible for severe interactions.
    """
    a = np.asarray(att, dtype=float).ravel()
    m = np.asarray(mechanism_matrix, dtype=float)
    if m.ndim == 1:
        m = m.reshape(-1, 1)
    if m.shape[0] != a.size:
        raise ValueError(f"mechanism_matrix has {m.shape[0]} rows but attention has {a.size}")
    if m.shape[1] != len(tags):
        raise ValueError(f"mechanism_matrix has {m.shape[1]} columns but {len(tags)} tags given")

    scores = (a[:, None] * m).sum(axis=0)
    if normalize:
        total = scores.sum()
        scores = scores / total if total > 0 else np.zeros_like(scores)
    return {tag: float(s) for tag, s in zip(tags, scores)}


def mechanism_attribution_by_class(
    att: ArrayLike,
    mechanism_matrix: ArrayLike,
    labels: ArrayLike,
    tags: Sequence[str] = MECHANISM_TAGS,
) -> Dict[str, Dict[str, float]]:
    """Per-severity-class attribution: the MAJOR row is the interesting one."""
    from .metrics import ORDERED_CLASSES, SEVERITY_LABELS

    a = np.asarray(att, dtype=float).ravel()
    m = np.asarray(mechanism_matrix, dtype=float)
    y = np.asarray(labels).astype(int)
    if y.size != a.size:
        raise ValueError("labels and attention must be the same length")

    out: Dict[str, Dict[str, float]] = {}
    for cls in ORDERED_CLASSES:
        mask = y == cls
        name = SEVERITY_LABELS[cls]
        if not mask.any():
            out[name] = {tag: 0.0 for tag in tags}
            continue
        out[name] = mechanism_attribution(a[mask], m[mask], tags=tags, normalize=True)
    return out


def attention_summary(
    att: ArrayLike,
    mechanism_matrix: Optional[ArrayLike] = None,
    labels: Optional[ArrayLike] = None,
    tags: Sequence[str] = MECHANISM_TAGS,
) -> Dict[str, object]:
    """Compact JSON-serializable summary for results/."""
    a = np.asarray(att, dtype=float).ravel()
    summary: Dict[str, object] = {
        "n_edges": int(a.size),
        "mean": float(a.mean()),
        "std": float(a.std()),
        "min": float(a.min()),
        "max": float(a.max()),
        "p90": float(np.percentile(a, 90)),
        "p99": float(np.percentile(a, 99)),
        "gini": float(_gini(a)),
    }
    if mechanism_matrix is not None:
        summary["mechanism_attribution"] = mechanism_attribution(a, mechanism_matrix, tags)
        if labels is not None:
            summary["mechanism_attribution_by_class"] = mechanism_attribution_by_class(
                a, mechanism_matrix, labels, tags
            )
    return summary


def _gini(x: np.ndarray) -> float:
    """Concentration of attention: 0 = uniform, ->1 = a few edges dominate."""
    a = np.sort(np.abs(np.asarray(x, dtype=float).ravel()))
    if a.size == 0 or a.sum() == 0:
        return 0.0
    n = a.size
    idx = np.arange(1, n + 1)
    return float((2 * (idx * a).sum()) / (n * a.sum()) - (n + 1) / n)


# --------------------------------------------------------------------------- #
# 3. Figures
# --------------------------------------------------------------------------- #


def _finish(fig, outpath: Union[str, Path], dpi: int = 160) -> Path:
    outpath = Path(outpath)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(outpath, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return outpath


def plot_mechanism_attribution(
    attribution: Dict[str, float],
    outpath: Union[str, Path],
    title: str = "Attention share by mechanism channel",
) -> Path:
    tags = list(attribution)
    vals = [attribution[t] for t in tags]
    fig, ax = plt.subplots(figsize=(7.2, 0.7 * len(tags) + 1.8))
    ax.barh(range(len(tags)), vals, color=PALETTE[: len(tags)], height=0.6)
    ax.set_yticks(range(len(tags)))
    ax.set_yticklabels([t.replace("_", " ") for t in tags])
    ax.invert_yaxis()
    ax.set_xlabel("share of total attention")
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.25, linestyle=":")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for i, v in enumerate(vals):
        ax.text(v + max(vals) * 0.01, i, f"{v:.1%}", va="center", fontsize=9)
    return _finish(fig, outpath)


def plot_mechanism_heatmap(
    by_class: Dict[str, Dict[str, float]],
    outpath: Union[str, Path],
    title: str = "Mechanism attention by severity class",
) -> Path:
    classes = list(by_class)
    tags = list(next(iter(by_class.values())))
    mat = np.array([[by_class[c][t] for t in tags] for c in classes], dtype=float)

    fig, ax = plt.subplots(figsize=(1.6 * len(tags) + 2.4, 0.8 * len(classes) + 2.0))
    im = ax.imshow(mat, cmap="BuPu", aspect="auto", vmin=0.0, vmax=max(mat.max(), 1e-9))
    ax.set_xticks(range(len(tags)))
    ax.set_xticklabels([t.replace("_", "\n") for t in tags], fontsize=9)
    ax.set_yticks(range(len(classes)))
    ax.set_yticklabels(classes)
    ax.set_title(title)
    for i in range(len(classes)):
        for j in range(len(tags)):
            v = mat[i, j]
            ax.text(
                j,
                i,
                f"{v:.0%}",
                ha="center",
                va="center",
                fontsize=9,
                color="white" if v > mat.max() * 0.6 else "#1A202C",
            )
    fig.colorbar(im, ax=ax, shrink=0.8, label="attention share")
    return _finish(fig, outpath)


def plot_top_edges(
    rows: Sequence[Dict[str, object]],
    outpath: Union[str, Path],
    title: str = "Highest-attention drug pairs",
) -> Path:
    rows = list(rows)[::-1]
    labels = [f"{r['drug_a']} - {r['drug_b']}" for r in rows]
    vals = [float(r["attention"]) for r in rows]
    colors = PALETTE[0]
    if rows and "severity" in rows[0]:
        cmap = {"MINOR": PALETTE[2], "MODERATE": PALETTE[1], "MAJOR": PALETTE[4]}
        colors = [cmap.get(str(r.get("severity")), PALETTE[5]) for r in rows]

    fig, ax = plt.subplots(figsize=(8.0, 0.36 * len(rows) + 1.8))
    ax.barh(range(len(rows)), vals, color=colors, height=0.68)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("attention weight")
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.25, linestyle=":")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    if rows and "severity" in rows[0]:
        from matplotlib.patches import Patch

        handles = [
            Patch(facecolor=PALETTE[2], label="MINOR"),
            Patch(facecolor=PALETTE[1], label="MODERATE"),
            Patch(facecolor=PALETTE[4], label="MAJOR"),
        ]
        ax.legend(handles=handles, frameon=False, fontsize=8, loc="lower right")
    return _finish(fig, outpath)


def plot_layer_heatmap(
    matrix: ArrayLike,
    layer_names: Sequence[str],
    outpath: Union[str, Path],
    max_edges: int = 60,
    title: str = "Attention per layer (top edges)",
) -> Path:
    m = np.asarray(matrix, dtype=float)
    if m.ndim != 2:
        raise ValueError("matrix must be (n_layers, E)")
    order = np.argsort(-m.mean(axis=0))[: min(max_edges, m.shape[1])]
    sub = m[:, order]

    fig, ax = plt.subplots(figsize=(min(0.18 * sub.shape[1] + 3.0, 14), 0.6 * sub.shape[0] + 2.2))
    im = ax.imshow(sub, cmap="viridis", aspect="auto")
    ax.set_yticks(range(len(layer_names)))
    ax.set_yticklabels(layer_names, fontsize=8)
    ax.set_xlabel(f"edge (top {sub.shape[1]} by mean attention)")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, shrink=0.8, label="attention")
    return _finish(fig, outpath)


def plot_pair_subgraph(
    att: ArrayLike,
    edge_index: ArrayLike,
    focus_drug: int,
    outpath: Union[str, Path],
    drug_names: Optional[Dict[int, str]] = None,
    max_neighbors: int = 12,
) -> Path:
    """
    Ego network of one drug, edge thickness = attention. This is the
    per-prediction explanation figure for a case study in the paper.
    """
    a = np.asarray(att, dtype=float).ravel()
    ei = np.asarray(edge_index)
    if ei.shape[0] != 2:
        ei = ei.T
    mask = (ei[0] == focus_drug) | (ei[1] == focus_drug)
    if not mask.any():
        raise ValueError(f"drug {focus_drug} has no edges")

    idx = np.where(mask)[0]
    idx = idx[np.argsort(-a[idx])][:max_neighbors]
    neighbors = [int(ei[1, i]) if int(ei[0, i]) == focus_drug else int(ei[0, i]) for i in idx]
    weights = normalize_attention(a[idx])
    names = drug_names or {}

    n = len(neighbors)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False) - np.pi / 2
    xs, ys = np.cos(angles), np.sin(angles)

    fig, ax = plt.subplots(figsize=(6.4, 6.4))
    for x, y, w, nb in zip(xs, ys, weights, neighbors):
        ax.plot([0, x], [0, y], color=PALETTE[0], alpha=0.35 + 0.6 * w, linewidth=0.8 + 5.0 * w,
                solid_capstyle="round", zorder=1)
        ax.scatter([x], [y], s=240, color="white", edgecolor=PALETTE[0], linewidth=1.4, zorder=2)
        ax.annotate(
            names.get(nb, f"d{nb}"),
            (x, y),
            textcoords="offset points",
            xytext=(0, -16 if y < 0 else 14),
            ha="center",
            fontsize=8,
        )
    ax.scatter([0], [0], s=420, color=PALETTE[4], zorder=3)
    ax.annotate(
        names.get(int(focus_drug), f"drug_{focus_drug}"),
        (0, 0),
        ha="center",
        va="center",
        fontsize=9,
        color="white",
        zorder=4,
    )
    ax.set_xlim(-1.45, 1.45)
    ax.set_ylim(-1.45, 1.45)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(
        f"Attention around {names.get(int(focus_drug), f'drug_{focus_drug}')}\n"
        "(line width = attention weight)",
        fontsize=10,
    )
    return _finish(fig, outpath)


# --------------------------------------------------------------------------- #
# One-call report
# --------------------------------------------------------------------------- #


@dataclass
class ExplainabilityReport:
    figures: List[Path] = field(default_factory=list)
    summary: Dict[str, object] = field(default_factory=dict)
    top_edges: List[Dict[str, object]] = field(default_factory=list)
    summary_path: Optional[Path] = None


def save_figures(
    att: ArrayLike,
    edge_index: ArrayLike,
    mechanism_matrix: Optional[ArrayLike] = None,
    labels: Optional[ArrayLike] = None,
    outdir: Union[str, Path] = "results/explainability_figures",
    model_name: str = "model",
    drug_names: Optional[Dict[int, str]] = None,
    tags: Sequence[str] = MECHANISM_TAGS,
    top_k: int = 20,
    focus_drug: Optional[int] = None,
) -> ExplainabilityReport:
    """
    Render every explainability figure for one model and write a JSON summary.
    Missing inputs just skip their figure instead of failing.
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    report = ExplainabilityReport()

    rows = top_k_attended_edges(att, edge_index, k=top_k, drug_names=drug_names, labels=labels)
    report.top_edges = rows
    report.figures.append(
        plot_top_edges(
            rows,
            outdir / f"{model_name}_top_edges.png",
            title=f"{model_name}: highest-attention drug pairs",
        )
    )

    if mechanism_matrix is not None:
        attr = mechanism_attribution(att, mechanism_matrix, tags=tags)
        report.figures.append(
            plot_mechanism_attribution(
                attr,
                outdir / f"{model_name}_mechanism_attribution.png",
                title=f"{model_name}: attention share by mechanism",
            )
        )
        if labels is not None:
            by_class = mechanism_attribution_by_class(att, mechanism_matrix, labels, tags=tags)
            report.figures.append(
                plot_mechanism_heatmap(
                    by_class,
                    outdir / f"{model_name}_mechanism_by_class.png",
                    title=f"{model_name}: mechanism attention by severity",
                )
            )

    if focus_drug is not None:
        report.figures.append(
            plot_pair_subgraph(
                att,
                edge_index,
                focus_drug,
                outdir / f"{model_name}_ego_drug{focus_drug}.png",
                drug_names=drug_names,
            )
        )

    report.summary = attention_summary(att, mechanism_matrix, labels, tags=tags)
    report.summary["model_name"] = model_name
    report.summary["figures"] = [str(p) for p in report.figures]
    report.summary["top_edges"] = rows[:10]
    path = outdir / f"{model_name}_attention_summary.json"
    path.write_text(json.dumps(report.summary, indent=2))
    report.summary_path = path
    return report


def top_edges_to_markdown(rows: Sequence[Dict[str, object]]) -> str:
    """Table for results/literature_validation.md on Day 4."""
    if not rows:
        return "_No attended edges._\n"
    has_sev = "severity" in rows[0]
    header = "| Rank | Drug A | Drug B | Attention |" + (" Severity |" if has_sev else "")
    sep = "| --- | --- | --- | --- |" + (" --- |" if has_sev else "")
    lines = [header, sep]
    for r in rows:
        line = f"| {r['rank']} | {r['drug_a']} | {r['drug_b']} | {float(r['attention']):.4f} |"
        if has_sev:
            line += f" {r.get('severity')} |"
        lines.append(line)
    return "\n".join(lines) + "\n"
