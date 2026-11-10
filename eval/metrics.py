"""
Evaluation metrics for DDI severity prediction (ordinal 3-class task).

Owner: M5 (Eval / Explainability Lead) - see eval/README.md
Contract: severity labels are integer-encoded and ORDERED.

    0 = MINOR  <  1 = MODERATE  <  2 = MAJOR

Every model (M3 binary baseline, M3 ordinal GNN, M4 mechanism-aware ordinal GNN)
reports through `evaluate()` so the final comparison table is apples-to-apples.

Headline metrics for the paper:
    * macro_f1        - class-imbalance-robust headline number
    * mae             - ordinal distance (predicting MINOR for a MAJOR costs 2, not 1)
    * major_recall    - "do we catch the dangerous ones?" (reviewer question #1)
    * qwk             - quadratic weighted kappa, standard for ordinal tasks

Usage
-----
    from eval.metrics import evaluate, metrics_to_markdown

    res = evaluate(y_true, y_pred, y_proba=proba, model_name="mechanism_gnn")
    print(metrics_to_markdown(res))
"""

from __future__ import annotations

import contextlib
import json
import warnings
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Union

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    precision_recall_fscore_support,
    roc_auc_score,
)

# --------------------------------------------------------------------------- #
# Label contract - do not change without telling M1/M3/M4
# --------------------------------------------------------------------------- #

MINOR, MODERATE, MAJOR = 0, 1, 2
SEVERITY_LABELS: Dict[int, str] = {MINOR: "MINOR", MODERATE: "MODERATE", MAJOR: "MAJOR"}
ORDERED_CLASSES: List[int] = [MINOR, MODERATE, MAJOR]
SEVERE_CLASS = MAJOR  # the reserved / benchmarked class

ArrayLike = Union[Sequence[int], Sequence[float], np.ndarray]


# --------------------------------------------------------------------------- #
# Result container
# --------------------------------------------------------------------------- #


@dataclass
class EvalResult:
    """Serializable evaluation result for one model on one split."""

    model_name: str
    split: str
    n_samples: int

    # headline
    macro_f1: float
    mae: float
    major_recall: float
    qwk: float

    # secondary
    micro_f1: float
    weighted_f1: float
    accuracy: float
    balanced_accuracy: float
    within_one_accuracy: float
    rmse: float

    # per class
    per_class: Dict[str, Dict[str, float]] = field(default_factory=dict)
    confusion: List[List[int]] = field(default_factory=list)
    support: Dict[str, int] = field(default_factory=dict)

    # optional probabilistic
    macro_auc: Optional[float] = None
    major_auc: Optional[float] = None
    major_ap: Optional[float] = None

    notes: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, path: Union[str, Path]) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2))
        return path

    @classmethod
    def load(cls, path: Union[str, Path]) -> "EvalResult":
        return cls(**json.loads(Path(path).read_text()))


# --------------------------------------------------------------------------- #
# Input validation
# --------------------------------------------------------------------------- #


def _as_label_array(y: ArrayLike, name: str) -> np.ndarray:
    arr = np.asarray(y)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1-D, got shape {arr.shape}")
    if arr.size == 0:
        raise ValueError(f"{name} is empty - nothing to evaluate")
    if not np.all(np.isfinite(arr.astype(float))):
        raise ValueError(f"{name} contains NaN/inf")
    arr = arr.astype(int)
    bad = set(np.unique(arr)) - set(ORDERED_CLASSES)
    if bad:
        raise ValueError(
            f"{name} contains labels outside the severity contract {ORDERED_CLASSES}: "
            f"{sorted(bad)}. Expected 0=MINOR, 1=MODERATE, 2=MAJOR."
        )
    return arr


# --------------------------------------------------------------------------- #
# Ordinal decoding helpers (M4's CORAL head, M3's softmax head)
# --------------------------------------------------------------------------- #


def softmax_to_labels(proba: ArrayLike) -> np.ndarray:
    """Standard argmax decoding for a (N, 3) softmax output."""
    p = np.asarray(proba, dtype=float)
    if p.ndim != 2 or p.shape[1] != len(ORDERED_CLASSES):
        raise ValueError(f"expected (N, 3) probability matrix, got {p.shape}")
    return p.argmax(axis=1).astype(int)


def expected_severity(proba: ArrayLike) -> np.ndarray:
    """
    Ordinal-aware decoding: round the expected severity instead of taking argmax.

    For [0.4, 0.35, 0.25] argmax says MINOR but the expected severity is 0.85
    -> MODERATE, which is the ordinally sane call. Report both in the paper.
    """
    p = np.asarray(proba, dtype=float)
    if p.ndim != 2 or p.shape[1] != len(ORDERED_CLASSES):
        raise ValueError(f"expected (N, 3) probability matrix, got {p.shape}")
    exp = p @ np.asarray(ORDERED_CLASSES, dtype=float)
    return np.clip(np.rint(exp), MINOR, MAJOR).astype(int)


def coral_to_labels(cum_probs: ArrayLike, threshold: float = 0.5) -> np.ndarray:
    """
    Decode CORAL / cumulative-link outputs.

    `cum_probs` is (N, K-1) = (N, 2): column k holds P(y > k).
    Label = number of thresholds passed, which keeps predictions monotone.
    """
    c = np.asarray(cum_probs, dtype=float)
    if c.ndim != 2 or c.shape[1] != len(ORDERED_CLASSES) - 1:
        raise ValueError(f"expected (N, 2) cumulative matrix, got {c.shape}")
    return (c > threshold).sum(axis=1).astype(int)


def binary_to_severity(y_binary: ArrayLike, positive_class: int = MAJOR) -> np.ndarray:
    """
    Lift M3's binary baseline (interaction / no-interaction, or severe / not)
    into the 3-class space so it can sit in the same comparison table.
    Non-positive predictions are mapped to MINOR.
    """
    b = np.asarray(y_binary).astype(int)
    if not set(np.unique(b)) <= {0, 1}:
        raise ValueError("binary predictions must be 0/1")
    return np.where(b == 1, positive_class, MINOR).astype(int)


# --------------------------------------------------------------------------- #
# Core metric computation
# --------------------------------------------------------------------------- #


def evaluate(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    y_proba: Optional[ArrayLike] = None,
    model_name: str = "unnamed_model",
    split: str = "test",
    notes: Optional[Dict[str, str]] = None,
) -> EvalResult:
    """
    Compute the full metric suite for one model on one split.

    Parameters
    ----------
    y_true, y_pred : 1-D int arrays over {0, 1, 2}
    y_proba        : optional (N, 3) class probabilities, enables AUC metrics
    """
    yt = _as_label_array(y_true, "y_true")
    yp = _as_label_array(y_pred, "y_pred")
    if yt.shape != yp.shape:
        raise ValueError(f"shape mismatch: y_true {yt.shape} vs y_pred {yp.shape}")

    labels = ORDERED_CLASSES

    # The severe-subset benchmark is MAJOR-only by construction, so sklearn
    # warns about a single class being present. That is expected here, and we
    # always pass `labels=` explicitly, so the shapes stay correct.
    single_class = np.unique(yt).size == 1 or np.unique(yp).size == 1
    with contextlib.ExitStack() as stack:
        if single_class:
            stack.enter_context(warnings.catch_warnings())
            warnings.simplefilter("ignore", UserWarning)
        prec, rec, f1, sup = precision_recall_fscore_support(
            yt, yp, labels=labels, zero_division=0
        )
        return _assemble(yt, yp, y_proba, prec, rec, f1, sup, labels, model_name, split, notes)


def _assemble(
    yt: np.ndarray,
    yp: np.ndarray,
    y_proba: Optional[ArrayLike],
    prec: np.ndarray,
    rec: np.ndarray,
    f1: np.ndarray,
    sup: np.ndarray,
    labels: List[int],
    model_name: str,
    split: str,
    notes: Optional[Dict[str, str]],
) -> EvalResult:
    """Build the EvalResult; split out so the warning context stays narrow."""

    per_class = {
        SEVERITY_LABELS[c]: {
            "precision": float(prec[i]),
            "recall": float(rec[i]),
            "f1": float(f1[i]),
            "support": int(sup[i]),
        }
        for i, c in enumerate(labels)
    }

    diff = np.abs(yt - yp)
    result = EvalResult(
        model_name=model_name,
        split=split,
        n_samples=int(yt.size),
        macro_f1=float(f1_score(yt, yp, labels=labels, average="macro", zero_division=0)),
        mae=float(mean_absolute_error(yt, yp)),
        major_recall=float(rec[labels.index(MAJOR)]),
        qwk=float(_safe_qwk(yt, yp, labels)),
        micro_f1=float(f1_score(yt, yp, labels=labels, average="micro", zero_division=0)),
        weighted_f1=float(
            f1_score(yt, yp, labels=labels, average="weighted", zero_division=0)
        ),
        accuracy=float(accuracy_score(yt, yp)),
        balanced_accuracy=float(_safe_balanced_accuracy(yt, yp)),
        within_one_accuracy=float((diff <= 1).mean()),
        rmse=float(np.sqrt(np.mean(diff.astype(float) ** 2))),
        per_class=per_class,
        confusion=confusion_matrix(yt, yp, labels=labels).tolist(),
        support={SEVERITY_LABELS[c]: int(sup[i]) for i, c in enumerate(labels)},
        notes=dict(notes or {}),
    )

    if y_proba is not None:
        _attach_probabilistic(result, yt, y_proba, labels)

    return result


def _safe_balanced_accuracy(yt: np.ndarray, yp: np.ndarray) -> float:
    """
    sklearn warns when y_pred contains a class absent from y_true - routine on
    the MAJOR-only severe subset, where a model may still predict MODERATE.
    Those classes are correctly excluded from the average; the warning is noise.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return balanced_accuracy_score(yt, yp)


def _safe_qwk(yt: np.ndarray, yp: np.ndarray, labels: List[int]) -> float:
    """QWK is undefined when everything lands in one cell; return 0.0 there."""
    if np.unique(yt).size == 1 and np.unique(yp).size == 1:
        return 0.0
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            return cohen_kappa_score(yt, yp, labels=labels, weights="quadratic")
    except ValueError:
        return 0.0


def _attach_probabilistic(
    result: EvalResult, yt: np.ndarray, y_proba: ArrayLike, labels: List[int]
) -> None:
    from sklearn.metrics import average_precision_score

    p = np.asarray(y_proba, dtype=float)
    if p.ndim != 2 or p.shape[1] != len(labels):
        raise ValueError(f"y_proba must be (N, 3), got {p.shape}")
    if p.shape[0] != yt.size:
        raise ValueError("y_proba rows must match y_true length")

    row_sums = p.sum(axis=1)
    if not np.allclose(row_sums, 1.0, atol=1e-3):
        result.notes["proba_warning"] = "rows did not sum to 1; renormalized"
        p = p / np.clip(row_sums[:, None], 1e-12, None)

    present = np.unique(yt)
    if present.size < 2:
        result.notes["auc_skipped"] = "only one class present in y_true"
        return

    try:
        if present.size == len(labels):
            result.macro_auc = float(
                roc_auc_score(yt, p, multi_class="ovr", average="macro", labels=labels)
            )
    except ValueError as exc:  # pragma: no cover - defensive
        result.notes["macro_auc_error"] = str(exc)

    is_major = (yt == MAJOR).astype(int)
    if is_major.min() != is_major.max():
        major_scores = p[:, labels.index(MAJOR)]
        result.major_auc = float(roc_auc_score(is_major, major_scores))
        result.major_ap = float(average_precision_score(is_major, major_scores))


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #

HEADLINE_FIELDS = [
    ("macro_f1", "Macro F1", "up"),
    ("mae", "MAE", "down"),
    ("major_recall", "MAJOR recall", "up"),
    ("qwk", "QWK", "up"),
    ("balanced_accuracy", "Bal. acc", "up"),
    ("within_one_accuracy", "Within-1 acc", "up"),
]


@dataclass
class DetectionResult:
    """
    Binary interaction-detection result - kept in a SEPARATE type from
    EvalResult on purpose.

    Detection ("do these two drugs interact at all?") and severity ("how badly?")
    are different tasks on different label spaces, and mixing them in one table
    is how a paper ends up comparing a 0.95 detection AUC against a 0.62 macro
    F1 as if they measured the same thing. They get separate tables; the text
    says which is which.
    """

    model_name: str
    split: str
    n_samples: int
    n_positive: int

    roc_auc: Optional[float]
    average_precision: Optional[float]
    f1: float
    precision: float
    recall: float
    accuracy: float
    threshold: float
    notes: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, path: Union[str, Path]) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2))
        return path

    @classmethod
    def load(cls, path: Union[str, Path]) -> "DetectionResult":
        return cls(**json.loads(Path(path).read_text()))


def evaluate_detection(
    y_true: ArrayLike,
    y_score: ArrayLike,
    threshold: float = 0.5,
    model_name: str = "unnamed_model",
    split: str = "test",
    notes: Optional[Dict[str, str]] = None,
) -> DetectionResult:
    """
    Score binary interaction detection (M3's Model 1 / the link-prediction task).

    Parameters
    ----------
    y_true    : (N,) in {0, 1} - 1 means "these drugs interact"
    y_score   : (N,) float probabilities, or (N,) hard 0/1 predictions
    threshold : cut applied to probabilities to get hard labels

    Pass the DETECTION labels here, not severity labels. To fold a detection
    model into the severity comparison table instead, use `binary_to_severity()`
    and `evaluate()` - and say in the paper how you mapped it.
    """
    from sklearn.metrics import average_precision_score

    yt = np.asarray(y_true).astype(int).ravel()
    ys = np.asarray(y_score, dtype=float).ravel()
    if yt.size == 0:
        raise ValueError("y_true is empty")
    if yt.shape != ys.shape:
        raise ValueError(f"shape mismatch: y_true {yt.shape} vs y_score {ys.shape}")
    bad = set(np.unique(yt)) - {0, 1}
    if bad:
        raise ValueError(
            f"detection labels must be 0/1, got {sorted(bad)}. If these are severity "
            "labels, use evaluate() instead - detection and severity are separate tables."
        )
    if not np.all(np.isfinite(ys)):
        raise ValueError("y_score contains NaN/inf")

    is_prob = bool(ys.min() >= 0.0 and ys.max() <= 1.0 and not np.array_equal(np.unique(ys), np.array([0, 1])))
    y_hat = (ys >= threshold).astype(int) if is_prob else ys.astype(int)

    prec, rec, f1, _ = precision_recall_fscore_support(
        yt, y_hat, labels=[1], average="binary", zero_division=0
    )

    roc = ap = None
    if np.unique(yt).size == 2:
        if is_prob:
            roc = float(roc_auc_score(yt, ys))
            ap = float(average_precision_score(yt, ys))
    else:
        notes = dict(notes or {})
        notes["auc_skipped"] = "only one class present in y_true"

    return DetectionResult(
        model_name=model_name,
        split=split,
        n_samples=int(yt.size),
        n_positive=int(yt.sum()),
        roc_auc=roc,
        average_precision=ap,
        f1=float(f1),
        precision=float(prec),
        recall=float(rec),
        accuracy=float(accuracy_score(yt, y_hat)),
        threshold=float(threshold),
        notes=dict(notes or {}),
    )


def detection_to_markdown(
    results: Iterable["DetectionResult"],
    title: str = "Binary interaction detection (separate task)",
) -> str:
    """
    Detection table. Rendered apart from the severity table, with a line saying
    why, so nobody reads across the two.
    """
    results = list(results)
    if not results:
        return f"## {title}\n\n_No results yet._\n"
    lines = [
        f"## {title}",
        "",
        "_Different task and label space from severity prediction: this answers "
        '"do these drugs interact?", not "how severe?". Numbers here are not '
        "comparable to the ordinal table above._",
        "",
        "| Model | Split | n | Positives | ROC-AUC | AP | Precision | Recall | F1 |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in results:
        auc = f"{r.roc_auc:.4f}" if r.roc_auc is not None else "n/a"
        ap = f"{r.average_precision:.4f}" if r.average_precision is not None else "n/a"
        lines.append(
            f"| {r.model_name} | {r.split} | {r.n_samples} | {r.n_positive} | {auc} | {ap} | "
            f"{r.precision:.4f} | {r.recall:.4f} | {r.f1:.4f} |"
        )
    return "\n".join(lines) + "\n"


def metrics_to_markdown(result: EvalResult) -> str:
    """Single-model markdown block for results/*.md."""
    lines = [
        f"### {result.model_name} - `{result.split}` (n={result.n_samples})",
        "",
        "| Metric | Value |",
        "| --- | --- |",
    ]
    for key, label, direction in HEADLINE_FIELDS:
        arrow = "higher better" if direction == "up" else "lower better"
        lines.append(f"| {label} ({arrow}) | {getattr(result, key):.4f} |")
    for key, label in (
        ("macro_auc", "Macro AUC"),
        ("major_auc", "MAJOR AUC"),
        ("major_ap", "MAJOR AP"),
    ):
        val = getattr(result, key)
        if val is not None:
            lines.append(f"| {label} | {val:.4f} |")

    lines += [
        "",
        "| Class | Precision | Recall | F1 | Support |",
        "| --- | --- | --- | --- | --- |",
    ]
    for cls in ORDERED_CLASSES:
        name = SEVERITY_LABELS[cls]
        m = result.per_class[name]
        lines.append(
            f"| {name} | {m['precision']:.4f} | {m['recall']:.4f} | "
            f"{m['f1']:.4f} | {m['support']} |"
        )

    lines += ["", "Confusion matrix (rows = true, cols = predicted):", "", "| | " + " | ".join(
        SEVERITY_LABELS[c] for c in ORDERED_CLASSES) + " |", "| --- " * (len(ORDERED_CLASSES) + 1) + "|"]
    for i, cls in enumerate(ORDERED_CLASSES):
        row = result.confusion[i]
        lines.append(f"| **{SEVERITY_LABELS[cls]}** | " + " | ".join(str(v) for v in row) + " |")
    return "\n".join(lines) + "\n"


def comparison_to_markdown(
    results: Iterable[EvalResult], title: str = "Cross-model comparison"
) -> str:
    """
    The table that becomes the paper's Results section.
    Best value per column is bolded; MAE is bolded at the minimum.
    """
    results = list(results)
    if not results:
        return f"## {title}\n\n_No results yet._\n"

    cols = [(k, lbl, d) for k, lbl, d in HEADLINE_FIELDS]
    best: Dict[str, float] = {}
    for key, _, direction in cols:
        vals = [getattr(r, key) for r in results]
        best[key] = min(vals) if direction == "down" else max(vals)

    header = "| Model | Split | n | " + " | ".join(lbl for _, lbl, _ in cols) + " |"
    sep = "| --- " * (3 + len(cols)) + "|"
    lines = [f"## {title}", "", header, sep]
    for r in results:
        cells = []
        for key, _, _ in cols:
            v = getattr(r, key)
            cells.append(f"**{v:.4f}**" if np.isclose(v, best[key]) else f"{v:.4f}")
        lines.append(f"| {r.model_name} | {r.split} | {r.n_samples} | " + " | ".join(cells) + " |")
    lines += [
        "",
        "Arrows: Macro F1 / MAJOR recall / QWK / Bal. acc / Within-1 higher is better; MAE lower is better.",
    ]
    return "\n".join(lines) + "\n"


def save_comparison(
    results: Iterable[EvalResult],
    path: Union[str, Path],
    title: str = "Cross-model comparison",
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(comparison_to_markdown(results, title=title))
    return path
