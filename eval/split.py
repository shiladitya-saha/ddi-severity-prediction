"""
Train/val/test split logic for DDI severity prediction.

Owner: M5 (Eval / Explainability Lead) - see eval/README.md

Three split regimes, all reproducible from a seed and all serializable to JSON so
M3/M4 train on exactly the partition M5 evaluates on:

1. `stratified_split`   - 70/15/15 edge-level, stratified on severity.
2. severe benchmark     - a slice of MAJOR edges pulled out BEFORE the 70/15/15
                          and never trained on; this is the held-out severe
                          subset the paper benchmarks on ("does it work on the
                          rare, dangerous cases?").
3. `drug_disjoint_split`- no drug appears in both train and test, for M4's
                          few-shot / zero-shot new-drug generalization claim.

Data contract (agreed with M1 at the Day-1 sync)
------------------------------------------------
    edge_index : array (2, E) or (E, 2) of integer drug node ids
    labels     : array (E,) in {0=MINOR, 1=MODERATE, 2=MAJOR}

Everything here is index-based: a split is a dict of name -> edge indices into
M1's edge array. Nothing is copied, so this works on a 4M-edge graph too.

Usage
-----
    from eval.split import make_splits, save_splits, load_splits, verify_splits

    splits = make_splits(edge_index, labels, seed=42, severe_benchmark_frac=0.15)
    save_splits(splits, "eval/splits/ddi_v1.json")
    print(splits.summary(labels))
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np

from .metrics import MAJOR, MINOR, MODERATE, ORDERED_CLASSES, SEVERITY_LABELS

ArrayLike = Union[Sequence[int], np.ndarray]

DEFAULT_RATIOS = (0.70, 0.15, 0.15)


# --------------------------------------------------------------------------- #
# Container
# --------------------------------------------------------------------------- #


@dataclass
class SplitSet:
    """Named index partitions over M1's edge array."""

    indices: Dict[str, np.ndarray]
    seed: int
    ratios: Tuple[float, float, float] = DEFAULT_RATIOS
    severe_benchmark_frac: float = 0.0
    strategy: str = "stratified_edge"
    meta: Dict[str, object] = field(default_factory=dict)

    # - access -------------------------------------------------------------- #

    def __getitem__(self, name: str) -> np.ndarray:
        return self.indices[name]

    def __contains__(self, name: str) -> bool:
        return name in self.indices

    @property
    def names(self) -> List[str]:
        return list(self.indices)

    @property
    def train(self) -> np.ndarray:
        return self.indices["train"]

    @property
    def val(self) -> np.ndarray:
        return self.indices["val"]

    @property
    def test(self) -> np.ndarray:
        return self.indices["test"]

    # - reporting ----------------------------------------------------------- #

    def sizes(self) -> Dict[str, int]:
        return {k: int(v.size) for k, v in self.indices.items()}

    def class_counts(self, labels: ArrayLike) -> Dict[str, Dict[str, int]]:
        y = np.asarray(labels)
        out: Dict[str, Dict[str, int]] = {}
        for name, idx in self.indices.items():
            sub = y[idx]
            out[name] = {
                SEVERITY_LABELS[c]: int((sub == c).sum()) for c in ORDERED_CLASSES
            }
        return out

    def summary(self, labels: Optional[ArrayLike] = None) -> str:
        lines = [
            f"SplitSet(strategy={self.strategy}, seed={self.seed}, "
            f"ratios={self.ratios}, severe_benchmark_frac={self.severe_benchmark_frac})"
        ]
        total = sum(self.sizes().values())
        if labels is None:
            for name, n in self.sizes().items():
                lines.append(f"  {name:<18} {n:>8}  ({n / max(total, 1):.1%})")
            return "\n".join(lines)

        counts = self.class_counts(labels)
        header = f"  {'split':<18}{'n':>8}  " + "".join(
            f"{SEVERITY_LABELS[c]:>10}" for c in ORDERED_CLASSES
        )
        lines.append(header)
        for name, n in self.sizes().items():
            row = f"  {name:<18}{n:>8}  " + "".join(
                f"{counts[name][SEVERITY_LABELS[c]]:>10}" for c in ORDERED_CLASSES
            )
            lines.append(row)
        return "\n".join(lines)

    # - io ------------------------------------------------------------------ #

    def to_dict(self) -> dict:
        return {
            "strategy": self.strategy,
            "seed": self.seed,
            "ratios": list(self.ratios),
            "severe_benchmark_frac": self.severe_benchmark_frac,
            "meta": self.meta,
            "indices": {k: np.asarray(v).astype(int).tolist() for k, v in self.indices.items()},
        }


def save_splits(splits: SplitSet, path: Union[str, Path]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(splits.to_dict(), indent=2))
    return path


def load_splits(path: Union[str, Path]) -> SplitSet:
    raw = json.loads(Path(path).read_text())
    return SplitSet(
        indices={k: np.asarray(v, dtype=int) for k, v in raw["indices"].items()},
        seed=int(raw["seed"]),
        ratios=tuple(raw.get("ratios", DEFAULT_RATIOS)),  # type: ignore[arg-type]
        severe_benchmark_frac=float(raw.get("severe_benchmark_frac", 0.0)),
        strategy=raw.get("strategy", "stratified_edge"),
        meta=raw.get("meta", {}),
    )


# --------------------------------------------------------------------------- #
# Input normalization
# --------------------------------------------------------------------------- #


def normalize_edge_index(edge_index: ArrayLike) -> np.ndarray:
    """Accept (2, E) or (E, 2); always return (E, 2) int array."""
    arr = np.asarray(edge_index)
    if arr.ndim != 2:
        raise ValueError(f"edge_index must be 2-D, got shape {arr.shape}")
    if arr.shape[0] == 2 and arr.shape[1] != 2:
        arr = arr.T
    elif arr.shape[1] != 2:
        raise ValueError(f"edge_index must be (2, E) or (E, 2), got {arr.shape}")
    return arr.astype(int)


def _check_labels(labels: ArrayLike, n_edges: int) -> np.ndarray:
    y = np.asarray(labels).astype(int)
    if y.ndim != 1:
        raise ValueError(f"labels must be 1-D, got {y.shape}")
    if y.size != n_edges:
        raise ValueError(f"labels length {y.size} != number of edges {n_edges}")
    bad = set(np.unique(y)) - set(ORDERED_CLASSES)
    if bad:
        raise ValueError(f"labels outside {ORDERED_CLASSES}: {sorted(bad)}")
    return y


def _check_ratios(ratios: Sequence[float]) -> Tuple[float, float, float]:
    if len(ratios) != 3:
        raise ValueError("ratios must be (train, val, test)")
    if any(r < 0 for r in ratios):
        raise ValueError("ratios must be non-negative")
    total = float(sum(ratios))
    if not np.isclose(total, 1.0, atol=1e-6):
        raise ValueError(f"ratios must sum to 1.0, got {total}")
    return tuple(float(r) for r in ratios)  # type: ignore[return-value]


# --------------------------------------------------------------------------- #
# Core splitters
# --------------------------------------------------------------------------- #


def _stratified_indices(
    y: np.ndarray,
    pool: np.ndarray,
    ratios: Tuple[float, float, float],
    rng: np.random.Generator,
) -> Dict[str, np.ndarray]:
    """Per-class proportional allocation, so rare classes appear in every split."""
    train, val, test = [], [], []
    for cls in ORDERED_CLASSES:
        cls_idx = pool[y[pool] == cls]
        if cls_idx.size == 0:
            continue
        cls_idx = rng.permutation(cls_idx)
        n = cls_idx.size
        n_train = int(np.floor(n * ratios[0]))
        n_val = int(np.floor(n * ratios[1]))
        # give the remainder to test, then make sure val/test are non-empty when
        # the class is big enough to afford it (3+ edges)
        if n >= 3:
            n_train = max(1, min(n_train, n - 2))
            n_val = max(1, min(n_val, n - n_train - 1))
        train.append(cls_idx[:n_train])
        val.append(cls_idx[n_train : n_train + n_val])
        test.append(cls_idx[n_train + n_val :])

    out = {
        "train": np.sort(np.concatenate(train)) if train else np.array([], dtype=int),
        "val": np.sort(np.concatenate(val)) if val else np.array([], dtype=int),
        "test": np.sort(np.concatenate(test)) if test else np.array([], dtype=int),
    }
    return out


def stratified_split(
    labels: ArrayLike,
    ratios: Sequence[float] = DEFAULT_RATIOS,
    seed: int = 42,
    pool: Optional[ArrayLike] = None,
) -> SplitSet:
    """Plain 70/15/15 edge-level split, stratified on severity."""
    y = np.asarray(labels).astype(int)
    ratios_t = _check_ratios(ratios)
    rng = np.random.default_rng(seed)
    pool_arr = np.arange(y.size) if pool is None else np.asarray(pool, dtype=int)
    idx = _stratified_indices(y, pool_arr, ratios_t, rng)
    return SplitSet(indices=idx, seed=seed, ratios=ratios_t, strategy="stratified_edge")


def make_splits(
    edge_index: ArrayLike,
    labels: ArrayLike,
    ratios: Sequence[float] = DEFAULT_RATIOS,
    seed: int = 42,
    severe_benchmark_frac: float = 0.15,
) -> SplitSet:
    """
    The project's canonical split.

    Steps
    -----
    1. Pull `severe_benchmark_frac` of MAJOR edges into `severe_benchmark`.
       These are removed from the pool entirely - never trained on, never in
       val/test, so the benchmark number cannot be contaminated.
    2. 70/15/15 stratified split over everything that is left.

    Returns a SplitSet with keys: train, val, test, severe_benchmark.
    """
    ei = normalize_edge_index(edge_index)
    y = _check_labels(labels, ei.shape[0])
    ratios_t = _check_ratios(ratios)
    if not 0.0 <= severe_benchmark_frac < 1.0:
        raise ValueError("severe_benchmark_frac must be in [0, 1)")

    rng = np.random.default_rng(seed)
    all_idx = np.arange(y.size)

    major_idx = all_idx[y == MAJOR]
    n_reserve = int(np.floor(major_idx.size * severe_benchmark_frac))
    # never reserve so much that MAJOR disappears from training
    n_reserve = min(n_reserve, max(major_idx.size - len(ORDERED_CLASSES), 0))
    reserved = (
        np.sort(rng.permutation(major_idx)[:n_reserve])
        if n_reserve > 0
        else np.array([], dtype=int)
    )

    pool = np.setdiff1d(all_idx, reserved, assume_unique=True)
    idx = _stratified_indices(y, pool, ratios_t, rng)
    idx["severe_benchmark"] = reserved

    splits = SplitSet(
        indices=idx,
        seed=seed,
        ratios=ratios_t,
        severe_benchmark_frac=float(severe_benchmark_frac),
        strategy="stratified_edge+severe_benchmark",
        meta={
            "n_edges": int(y.size),
            "n_major_total": int(major_idx.size),
            "n_major_reserved": int(reserved.size),
        },
    )
    return splits


def drug_disjoint_split(
    edge_index: ArrayLike,
    labels: ArrayLike,
    ratios: Sequence[float] = DEFAULT_RATIOS,
    seed: int = 42,
) -> SplitSet:
    """
    Cold-start / new-drug split for M4's few-shot and zero-shot claim.

    Drugs are partitioned, then an edge lands in a split only if BOTH endpoints
    belong to that split's drug set. Cross-partition edges go to `bridge` and
    are excluded from training and evaluation (they would leak drug identity).

    `zero_shot` = test edges where both drugs are unseen in train
    `few_shot`  = test edges with exactly one unseen drug, which here is empty
                  by construction; it is kept as a key so M4's reporting code
                  does not have to special-case the shape.
    """
    ei = normalize_edge_index(edge_index)
    y = _check_labels(labels, ei.shape[0])
    ratios_t = _check_ratios(ratios)
    rng = np.random.default_rng(seed)

    drugs = np.unique(ei)
    drugs = rng.permutation(drugs)
    n = drugs.size
    n_train = int(np.floor(n * ratios_t[0]))
    n_val = int(np.floor(n * ratios_t[1]))
    train_d = set(drugs[:n_train].tolist())
    val_d = set(drugs[n_train : n_train + n_val].tolist())
    test_d = set(drugs[n_train + n_val :].tolist())

    buckets: Dict[str, List[int]] = {"train": [], "val": [], "test": [], "bridge": []}
    for i, (a, b) in enumerate(ei):
        a, b = int(a), int(b)
        if a in train_d and b in train_d:
            buckets["train"].append(i)
        elif a in val_d and b in val_d:
            buckets["val"].append(i)
        elif a in test_d and b in test_d:
            buckets["test"].append(i)
        else:
            buckets["bridge"].append(i)

    idx = {k: np.array(sorted(v), dtype=int) for k, v in buckets.items()}
    idx["zero_shot"] = idx["test"].copy()
    idx["few_shot"] = np.array([], dtype=int)

    return SplitSet(
        indices=idx,
        seed=seed,
        ratios=ratios_t,
        strategy="drug_disjoint",
        meta={
            "n_drugs": int(n),
            "n_train_drugs": len(train_d),
            "n_val_drugs": len(val_d),
            "n_test_drugs": len(test_d),
            "n_bridge_edges": int(idx["bridge"].size),
        },
    )


# --------------------------------------------------------------------------- #
# Leakage verification - run this before anyone trains
# --------------------------------------------------------------------------- #


def verify_splits(
    splits: SplitSet,
    labels: Optional[ArrayLike] = None,
    edge_index: Optional[ArrayLike] = None,
    require_all_classes: Iterable[str] = ("train", "val", "test"),
    strict: bool = True,
) -> List[str]:
    """
    Check a SplitSet for the mistakes that silently inflate results.

    Checks: pairwise index overlap, duplicate indices, empty splits, every
    severity class present in the named splits, severe_benchmark purity
    (all MAJOR), and - when edge_index is given - undirected duplicate leakage
    (edge (a,b) in train and (b,a) in test).

    Returns the list of problems. Raises AssertionError if `strict` and any found.
    """
    problems: List[str] = []
    names = splits.names

    for name, idx in splits.indices.items():
        arr = np.asarray(idx)
        if arr.size != np.unique(arr).size:
            problems.append(f"{name}: contains duplicate indices")
        if arr.size == 0 and name in require_all_classes:
            problems.append(f"{name}: is empty")

    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            overlap = np.intersect1d(splits[a], splits[b])
            if overlap.size:
                problems.append(
                    f"index leakage: {a} and {b} share {overlap.size} edges "
                    f"(e.g. {overlap[:5].tolist()})"
                )

    if labels is not None:
        y = np.asarray(labels).astype(int)
        for name in require_all_classes:
            if name not in splits:
                continue
            present = set(np.unique(y[splits[name]]).tolist())
            missing = [SEVERITY_LABELS[c] for c in ORDERED_CLASSES if c not in present]
            if missing:
                problems.append(f"{name}: missing class(es) {missing}")
        if "severe_benchmark" in splits and splits["severe_benchmark"].size:
            sb = y[splits["severe_benchmark"]]
            if not np.all(sb == MAJOR):
                problems.append("severe_benchmark: contains non-MAJOR edges")

    if edge_index is not None:
        ei = normalize_edge_index(edge_index)
        canon = np.sort(ei, axis=1)
        keys = {
            name: {tuple(canon[i]) for i in np.asarray(idx)}
            for name, idx in splits.indices.items()
            if np.asarray(idx).size
        }
        ordered = list(keys)
        for i, a in enumerate(ordered):
            for b in ordered[i + 1 :]:
                shared = keys[a] & keys[b]
                if shared:
                    problems.append(
                        f"undirected duplicate leakage: {a} and {b} share "
                        f"{len(shared)} drug pairs regardless of direction"
                    )

    if strict and problems:
        raise AssertionError("split verification failed:\n  - " + "\n  - ".join(problems))
    return problems
