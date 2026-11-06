"""Dummy-data tests for M4's models. Run from the repo root:

    python -m unittest tests/test_mechanism_model.py -v
"""

import unittest

import torch

from models.mechanism_gnn import MECHANISM_TAGS, MechanismOrdinalGNN
from models.ordinal_loss import (
    CoralHead,
    CoralLoss,
    OrdinalCrossEntropyLoss,
    coral_logits_to_probs,
    coral_predict,
    levels_from_labels,
)

NUM_TAGS = len(MECHANISM_TAGS)


class OrdinalLossTests(unittest.TestCase):
    def test_levels_encoding(self):
        levels = levels_from_labels(torch.tensor([0, 1, 2]))
        self.assertEqual(levels.tolist(), [[0, 0], [1, 0], [1, 1]])

    def test_coral_probs_valid_even_for_extreme_parameters(self):
        torch.manual_seed(0)
        head = CoralHead(8, num_classes=5)
        head.raw_gaps.data = torch.tensor([-20.0, 5.0, 0.0])
        head.first_bias.data = torch.tensor([3.0])
        probs = coral_logits_to_probs(head(torch.randn(64, 8) * 10))
        self.assertTrue((probs >= 0).all())
        self.assertTrue(torch.allclose(probs.sum(1), torch.ones(64)))

    def test_coral_predict_matches_probs_argmax(self):
        logits = torch.tensor([[-1.0, -2.0], [2.0, -2.0], [3.0, 1.0]])
        self.assertEqual(coral_predict(logits).tolist(), [0, 1, 2])
        self.assertEqual(coral_logits_to_probs(logits).argmax(1).tolist(), [0, 1, 2])

    def test_coral_penalises_far_mistakes_more(self):
        predicts_minor = torch.tensor([[-4.0, -6.0]])
        loss = CoralLoss()
        near = loss(predicts_minor, torch.tensor([1]))
        far = loss(predicts_minor, torch.tensor([2]))
        self.assertGreater(far, near)

    def test_ordinal_ce_penalises_far_mistakes_more(self):
        logits = torch.tensor([[1.0, 0.5, 0.0]])
        no_distance = OrdinalCrossEntropyLoss(distance_weight=0.0)
        loss = OrdinalCrossEntropyLoss()
        near_extra = loss(logits, torch.tensor([1])) - no_distance(logits, torch.tensor([1]))
        far_extra = loss(logits, torch.tensor([2])) - no_distance(logits, torch.tensor([2]))
        self.assertGreater(far_extra, near_extra)

    def test_zero_distance_weight_equals_cross_entropy(self):
        logits = torch.randn(16, 3)
        labels = torch.randint(0, 3, (16,))
        weights = torch.tensor([0.2, 1.0, 5.0])
        ours = OrdinalCrossEntropyLoss(class_weights=weights, distance_weight=0.0)
        expected = torch.nn.functional.cross_entropy(logits, labels, weight=weights)
        self.assertTrue(torch.isclose(ours(logits, labels), expected))

    def test_class_weights_raise_cost_of_weighted_class(self):
        logits = torch.tensor([[-4.0, -6.0], [-4.0, -6.0]])
        labels = torch.tensor([0, 2])
        plain = CoralLoss()(logits, labels)
        major_heavy = CoralLoss(class_weights=torch.tensor([1.0, 1.0, 10.0]))(logits, labels)
        self.assertGreater(major_heavy, plain)

    def test_invalid_labels_rejected(self):
        with self.assertRaises(ValueError):
            CoralLoss()(torch.zeros(1, 2), torch.tensor([3]))
        with self.assertRaises(ValueError):
            OrdinalCrossEntropyLoss()(torch.zeros(1, 3), torch.tensor([-1]))


def dummy_graph():
    """Six drugs in a chain; drug 5 has no edges and stands in for a new drug."""
    torch.manual_seed(0)
    x = torch.randn(6, 32)
    edge_index = torch.tensor([[0, 1, 1, 2, 2, 3, 3, 4], [1, 0, 2, 1, 3, 2, 4, 3]])
    edge_attr = torch.rand(edge_index.size(1), NUM_TAGS)
    pair_index = torch.tensor([[0, 1, 2, 5], [2, 3, 4, 0]])
    pair_attr = torch.rand(pair_index.size(1), NUM_TAGS)
    return x, edge_index, edge_attr, pair_index, pair_attr


class MechanismGNNTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.graph = dummy_graph()
        self.model = MechanismOrdinalGNN(32, hidden_dim=16, heads=2).eval()

    def test_output_shape_and_new_drug_is_finite(self):
        logits = self.model(*self.graph)
        self.assertEqual(logits.shape, (4, 2))
        self.assertTrue(torch.isfinite(logits).all())

    def test_pair_order_does_not_change_prediction(self):
        x, edge_index, edge_attr, pair_index, pair_attr = self.graph
        forward = self.model(x, edge_index, edge_attr, pair_index, pair_attr)
        swapped = self.model(x, edge_index, edge_attr, pair_index.flip(0), pair_attr)
        self.assertTrue(torch.allclose(forward, swapped))

    def test_edge_mechanism_features_affect_output(self):
        x, edge_index, edge_attr, pair_index, pair_attr = self.graph
        base = self.model(x, edge_index, edge_attr, pair_index, pair_attr)
        changed = self.model(x, edge_index, edge_attr + 1.0, pair_index, pair_attr)
        self.assertFalse(torch.allclose(base, changed))

    def test_pair_mechanism_features_affect_output(self):
        x, edge_index, edge_attr, pair_index, pair_attr = self.graph
        base = self.model(x, edge_index, edge_attr, pair_index, pair_attr)
        changed = self.model(x, edge_index, edge_attr, pair_index, pair_attr + 1.0)
        self.assertFalse(torch.allclose(base, changed))

    def test_attention_returned_for_every_layer(self):
        _, attention = self.model(*self.graph, return_attention=True)
        self.assertEqual(len(attention), 2)
        edge_index, alpha = attention[0]
        # 8 directed edges + 6 self-loops, 2 heads.
        self.assertEqual(edge_index.shape, (2, 14))
        self.assertEqual(alpha.shape, (14, 2))

    def test_every_parameter_receives_a_gradient(self):
        self.model.train()
        CoralLoss()(self.model(*self.graph), torch.tensor([0, 1, 2, 1])).backward()
        for name, parameter in self.model.named_parameters():
            self.assertIsNotNone(parameter.grad, name)

    def test_can_overfit_small_dummy_set(self):
        model = MechanismOrdinalGNN(32, hidden_dim=16, heads=2, dropout=0.0)
        labels = torch.tensor([0, 1, 2, 1])
        loss_fn = CoralLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        for _ in range(300):
            optimizer.zero_grad()
            loss = loss_fn(model(*self.graph), labels)
            loss.backward()
            optimizer.step()
        model.eval()
        self.assertEqual(coral_predict(model(*self.graph)).tolist(), labels.tolist())


if __name__ == "__main__":
    unittest.main()
