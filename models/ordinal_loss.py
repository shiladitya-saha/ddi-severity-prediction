"""
ordinal_loss.py
Owner: M4 — see models/README.md for day/deadline and output contract.

Ordinal losses for severity labels {0: minor, 1: moderate, 2: major}. Plain
cross-entropy treats minor -> major as no worse than minor -> moderate; both
options below penalise predictions by how far they land from the true severity.

Option A: CORAL-style (Cao et al., 2020). Pair `CoralHead` with `CoralLoss`.
    The model outputs K-1 logits, one for each question "is severity > k?".
    Convert them with `coral_logits_to_probs` / `coral_predict`.

Option B: `OrdinalCrossEntropyLoss`. Works with any model that outputs K
    ordinary class logits.

Both losses accept `class_weights` of shape [K], one weight per severity class
(M1's data/processed/class_weights.json).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def _check_labels(labels, num_classes):
    if labels.numel() and (labels.min() < 0 or labels.max() >= num_classes):
        raise ValueError(
            f"labels must be in [0, {num_classes - 1}], "
            f"got range [{labels.min().item()}, {labels.max().item()}]"
        )


def _weighted_mean(per_example_loss, labels, class_weights):
    """Average normalised by total weight, the same way F.cross_entropy does."""
    if class_weights is None:
        return per_example_loss.mean()
    weights = class_weights.to(per_example_loss)[labels]
    return (weights * per_example_loss).sum() / weights.sum()


def levels_from_labels(labels, num_classes=3):
    """Encode labels as cumulative targets: with K=3, 0 -> [0,0], 1 -> [1,0], 2 -> [1,1]."""
    thresholds = torch.arange(num_classes - 1, device=labels.device)
    return (labels.unsqueeze(1) > thresholds).float()


class CoralHead(nn.Module):
    """Maps features to K-1 rank-consistent logits, shape [N, K-1].

    All thresholds share one score and differ only by a bias. Original CORAL
    leaves the biases free, so the ordering is only guaranteed at the optimum.
    Here the gaps pass through softplus, which keeps the biases strictly
    decreasing throughout training, so `coral_logits_to_probs` is always valid.
    """

    def __init__(self, in_dim, num_classes=3):
        super().__init__()
        if num_classes < 2:
            raise ValueError("num_classes must be at least 2")
        self.score = nn.Linear(in_dim, 1, bias=False)
        self.first_bias = nn.Parameter(torch.zeros(1))
        self.raw_gaps = nn.Parameter(torch.zeros(num_classes - 2))

    def biases(self):
        offsets = torch.cumsum(F.softplus(self.raw_gaps), dim=0)
        return self.first_bias - torch.cat([offsets.new_zeros(1), offsets])

    def forward(self, x):
        return self.score(x) + self.biases()


class CoralLoss(nn.Module):
    """Binary cross-entropy summed over the K-1 "severity > k" questions."""

    def __init__(self, num_classes=3, class_weights=None):
        super().__init__()
        self.num_classes = num_classes
        self.register_buffer("class_weights", class_weights)

    def forward(self, logits, labels):
        _check_labels(labels, self.num_classes)
        levels = levels_from_labels(labels, self.num_classes)
        per_task = F.binary_cross_entropy_with_logits(logits, levels, reduction="none")
        return _weighted_mean(per_task.sum(dim=1), labels, self.class_weights)


class OrdinalCrossEntropyLoss(nn.Module):
    """Cross-entropy plus the expected ordinal distance from the true class.

    loss = CE + distance_weight * E[|predicted - true|] / (K - 1)

    The distance term uses the predicted probabilities, not argmax, so it stays
    differentiable. distance_weight=0 reduces to ordinary cross-entropy.
    """

    def __init__(self, num_classes=3, class_weights=None, distance_weight=1.0):
        super().__init__()
        self.num_classes = num_classes
        self.distance_weight = distance_weight
        self.register_buffer("class_weights", class_weights)

    def forward(self, logits, labels):
        _check_labels(labels, self.num_classes)
        ce = F.cross_entropy(logits, labels, reduction="none")
        classes = torch.arange(self.num_classes, device=labels.device)
        distance = (classes.unsqueeze(0) - labels.unsqueeze(1)).abs().to(logits)
        expected = (F.softmax(logits, dim=1) * distance).sum(dim=1)
        per_example = ce + self.distance_weight * expected / (self.num_classes - 1)
        return _weighted_mean(per_example, labels, self.class_weights)


def coral_logits_to_probs(logits):
    """Convert CORAL logits [N, K-1] to class probabilities [N, K]."""
    above = torch.sigmoid(logits)
    return torch.cat(
        [1 - above[:, :1], above[:, :-1] - above[:, 1:], above[:, -1:]], dim=1
    )


def coral_predict(logits):
    """Predicted severity: the number of thresholds the example clears."""
    return (logits > 0).sum(dim=1)
