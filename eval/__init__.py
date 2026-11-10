"""
M5 evaluation + explainability package for DDI severity prediction.

Severity label contract (shared with M1/M2/M3/M4):
    0 = MINOR  <  1 = MODERATE  <  2 = MAJOR
"""

from .metrics import (  # noqa: F401
    MAJOR,
    MINOR,
    MODERATE,
    ORDERED_CLASSES,
    SEVERITY_LABELS,
    EvalResult,
    binary_to_severity,
    comparison_to_markdown,
    coral_to_labels,
    evaluate,
    expected_severity,
    metrics_to_markdown,
    save_comparison,
    softmax_to_labels,
)
from .benchmark_severe_subset import (  # noqa: F401
    benchmark_severe,
    evaluate_link_prediction,
    hits_at_k,
    severe_subset_markdown,
    severity_miss_profile,
)
from .split import (  # noqa: F401
    SplitSet,
    drug_disjoint_split,
    load_splits,
    make_splits,
    save_splits,
    stratified_split,
    verify_splits,
)

__all__ = [
    "MINOR",
    "MODERATE",
    "MAJOR",
    "ORDERED_CLASSES",
    "SEVERITY_LABELS",
    "EvalResult",
    "evaluate",
    "metrics_to_markdown",
    "comparison_to_markdown",
    "save_comparison",
    "softmax_to_labels",
    "expected_severity",
    "coral_to_labels",
    "binary_to_severity",
    "SplitSet",
    "make_splits",
    "stratified_split",
    "drug_disjoint_split",
    "verify_splits",
    "save_splits",
    "load_splits",
    "benchmark_severe",
    "severity_miss_profile",
    "severe_subset_markdown",
    "hits_at_k",
    "evaluate_link_prediction",
]

__version__ = "0.2.0"
