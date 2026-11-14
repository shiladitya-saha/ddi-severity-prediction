"""
Synthetic DDI data that mimics M1's graph contract.

Owner: M5 (Eval / Explainability Lead) - see eval/README.md

Lets the eval harness be developed and tested end-to-end before M1's real
`data/processed_graph.pt` lands on Day 2. Deliberately class-imbalanced
(~60/30/10 MINOR/MODERATE/MAJOR) because that is the shape of real DDI data and
the reason Macro F1 and MAJOR recall are our headline metrics rather than
accuracy.

Also used by tests/ as the fixture, so the harness is exercised on the same
shape of data every run.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.attention_viz import MECHANISM_TAGS
from eval.metrics import MAJOR, MINOR, MODERATE, ORDERED_CLASSES

DEFAULT_CLASS_PROBS = (0.60, 0.30, 0.10)


@dataclass
class DummyDDIData:
    """A fake DDI graph plus fake model outputs."""

    edge_index: np.ndarray  # (2, E)
    labels: np.ndarray  # (E,)
    mechanism_matrix: np.ndarray  # (E, 4)
    drug_names: Dict[int, str]
    attention: np.ndarray  # (E,)
    n_drugs: int

    @property
    def n_edges(self) -> int:
        return int(self.labels.size)


def make_dummy_graph(
    n_drugs: int = 120,
    n_edges: int = 900,
    class_probs: Tuple[float, float, float] = DEFAULT_CLASS_PROBS,
    seed: int = 7,
) -> DummyDDIData:
    """
    Build a synthetic undirected DDI graph with no duplicate pairs.

    Mechanism flags are correlated with severity - MAJOR edges are more likely
    to carry CYP450 inhibition and QT prolongation - so attribution tests have
    real signal to detect rather than noise.
    """
    rng = np.random.default_rng(seed)
    if n_drugs < 4:
        raise ValueError("need at least 4 drugs")
    max_edges = n_drugs * (n_drugs - 1) // 2
    n_edges = int(min(n_edges, max_edges))

    seen: set[Tuple[int, int]] = set()
    pairs: List[Tuple[int, int]] = []
    while len(pairs) < n_edges:
        a, b = rng.integers(0, n_drugs, size=2)
        a, b = int(a), int(b)
        if a == b:
            continue
        key = (min(a, b), max(a, b))
        if key in seen:
            continue
        seen.add(key)
        pairs.append(key)

    edge_index = np.array(pairs, dtype=int).T  # (2, E)
    labels = rng.choice(ORDERED_CLASSES, size=n_edges, p=list(class_probs)).astype(int)

    # mechanism flags, severity-correlated
    base_rate = {
        MINOR: [0.10, 0.25, 0.10, 0.05],
        MODERATE: [0.35, 0.30, 0.25, 0.15],
        MAJOR: [0.70, 0.35, 0.40, 0.55],
    }
    mech = np.zeros((n_edges, len(MECHANISM_TAGS)), dtype=float)
    for i, y in enumerate(labels):
        rates = base_rate[int(y)]
        mech[i] = (rng.random(len(MECHANISM_TAGS)) < rates).astype(float)

    # attention: higher on severe edges with mechanism flags, plus noise
    raw = 0.25 * labels + 0.35 * mech.sum(axis=1) + rng.normal(0, 0.4, size=n_edges)
    attention = np.exp(raw)
    attention = attention / attention.sum()

    names = {i: f"DRUG_{i:03d}" for i in range(n_drugs)}
    return DummyDDIData(
        edge_index=edge_index,
        labels=labels,
        mechanism_matrix=mech,
        drug_names=names,
        attention=attention,
        n_drugs=n_drugs,
    )


def make_dummy_predictions(
    labels: np.ndarray,
    accuracy: float = 0.72,
    major_bias: float = 0.0,
    seed: int = 11,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fake model output: (y_pred, y_proba).

    `accuracy` is the probability mass placed on the true class; the rest leaks
    to ordinally adjacent classes, which is how a working ordinal model errs.
    Labels are SAMPLED from that distribution rather than argmaxed, so the
    realized accuracy really is ~`accuracy` and a weaker model really does
    score worse - otherwise every config decodes to a perfect prediction.
    `major_bias` shifts mass toward MAJOR, so you can simulate a model with
    high MAJOR recall and poor precision - useful for checking that the
    comparison table actually surfaces that trade-off.
    """
    rng = np.random.default_rng(seed)
    y = np.asarray(labels).astype(int)
    n = y.size
    k = len(ORDERED_CLASSES)
    if not 0.0 < accuracy <= 1.0:
        raise ValueError("accuracy must be in (0, 1]")

    proba = np.zeros((n, k), dtype=float)
    for i, true in enumerate(y):
        # distance-decayed mass around the true class
        dist = np.abs(np.asarray(ORDERED_CLASSES) - true).astype(float)
        w = np.where(dist == 0, accuracy, (1.0 - accuracy) * np.exp(-dist))
        w[MAJOR] *= 1.0 + major_bias
        w = w + rng.normal(0, 0.02, size=k).clip(0)
        proba[i] = w / w.sum()

    # sample, don't argmax: argmax would make every config perfect
    cum = np.cumsum(proba, axis=1)
    draws = rng.random((n, 1))
    y_pred = (draws > cum).sum(axis=1).clip(0, k - 1).astype(int)
    return y_pred, proba


class FakeAttentionLayer:
    """
    Stands in for a PyG GATConv in tests - returns (out, (edge_index, alpha))
    exactly like `GATConv(..., return_attention_weights=True)`, so the hook
    plumbing in AttentionCapture is exercised without torch installed.
    """

    def __init__(self, edge_index: np.ndarray, alpha: np.ndarray, name: str = "gat"):
        self.edge_index = np.asarray(edge_index)
        self.alpha = np.asarray(alpha, dtype=float)
        self.name = name
        self._hooks: List[object] = []

    def register_forward_hook(self, fn):
        self._hooks.append(fn)

        class _Handle:
            def __init__(self, owner, f):
                self.owner, self.f = owner, f

            def remove(self):
                if self.f in self.owner._hooks:
                    self.owner._hooks.remove(self.f)

        return _Handle(self, fn)

    def __call__(self, x):
        out = (x, (self.edge_index, self.alpha))
        for fn in list(self._hooks):
            fn(self, (x,), out)
        return out


class FakeAttentionModel:
    """Two fake attention layers, mimicking a stacked GNN."""

    def __init__(self, edge_index: np.ndarray, alphas: List[np.ndarray]):
        self.layers = [
            FakeAttentionLayer(edge_index, a, name=f"gat{i+1}") for i, a in enumerate(alphas)
        ]

    def named_modules(self):
        yield "", self
        for layer in self.layers:
            yield layer.name, layer

    def register_forward_hook(self, fn):  # top-level module emits nothing
        class _Noop:
            def remove(self):
                return None

        return _Noop()

    def forward(self, x: Optional[np.ndarray] = None):
        x = np.zeros(1) if x is None else x
        for layer in self.layers:
            x, _ = layer(x)
        return x

    __call__ = forward
