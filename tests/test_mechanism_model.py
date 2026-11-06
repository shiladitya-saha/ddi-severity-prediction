"""Dummy-data tests for M4's models. Run from the repo root:

    python -m unittest tests/test_mechanism_model.py -v
"""

import os
import tempfile
import unittest

import torch

from models.mechanism_gnn import (
    MECHANISM_TAGS,
    MechanismOrdinalGNN,
    load_mechanism_table,
    mechanism_matrix,
)
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
        for returned_edges, alpha in attention:
            # One row per input edge, in input order, so M5 can line attention
            # up with the (E, 4) mechanism matrix. 8 directed edges, 2 heads.
            self.assertTrue(torch.equal(returned_edges, self.graph[1]))
            self.assertEqual(alpha.shape, (8, 2))

    def test_attention_sums_to_one_over_each_drugs_incoming_edges(self):
        _, attention = self.model(*self.graph, return_attention=True)
        edge_index, alpha = attention[0]
        totals = torch.zeros(6, 2).index_add_(0, edge_index[1], alpha)
        receiving = edge_index[1].unique()
        self.assertTrue(torch.allclose(totals[receiving], torch.ones(len(receiving), 2)))

    def test_encode_runs_without_any_pairs(self):
        x, edge_index, edge_attr, _, _ = self.graph
        h, attention = self.model.encode(x, edge_index, edge_attr)
        self.assertEqual(h.shape, (6, 16))
        self.assertEqual(len(attention), 2)

    def test_missing_pair_attr_means_all_zeros(self):
        x, edge_index, edge_attr, pair_index, _ = self.graph
        without = self.model(x, edge_index, edge_attr, pair_index)
        zeros = self.model(x, edge_index, edge_attr, pair_index, torch.zeros(4, NUM_TAGS))
        self.assertTrue(torch.equal(without, zeros))

    def test_bad_inputs_rejected(self):
        x, edge_index, edge_attr, pair_index, pair_attr = self.graph
        cases = {
            "edge_attr row count": (x, edge_index, edge_attr[:-1], pair_index, pair_attr),
            "pair_attr row count": (x, edge_index, edge_attr, pair_index, pair_attr[:-1]),
            "pair_attr width": (x, edge_index, edge_attr, pair_index, pair_attr[:, :-1]),
            "negative pair index": (x, edge_index, edge_attr, pair_index - 1, pair_attr),
            "pair index too large": (x, edge_index, edge_attr, pair_index + 6, pair_attr),
            "edge index too large": (x, edge_index + 6, edge_attr, pair_index, pair_attr),
            "pair index wrong shape": (x, edge_index, edge_attr, pair_index.t(), pair_attr),
        }
        for name, args in cases.items():
            with self.subTest(name), self.assertRaises(ValueError):
                self.model(*args)

    def test_config_rebuilds_an_identical_model(self):
        rebuilt = MechanismOrdinalGNN(**self.model.config).eval()
        rebuilt.load_state_dict(self.model.state_dict())
        self.assertTrue(torch.equal(rebuilt(*self.graph), self.model(*self.graph)))

    def test_learns_severity_from_mechanism_on_held_out_pairs(self):
        # Every drug has the same fingerprint and severity depends only on the
        # mechanism features, so held-out accuracy requires those features to
        # reach the output. Random fingerprints would let a model this size
        # memorise individual drugs on so few pairs. Message passing uses
        # training pairs only.
        torch.manual_seed(0)
        x = torch.ones(40, 16)
        low = torch.randint(0, 20, (300,))
        pairs = torch.stack([low, low + torch.randint(1, 20, (300,))])
        attr = (torch.rand(300, NUM_TAGS) > 0.5).float() * 0.9
        labels = (attr[:, 0] > 0).long() + (attr[:, 3] > 0).long()
        train, test = torch.arange(240), torch.arange(240, 300)
        edge_index = torch.cat([pairs[:, train], pairs[:, train].flip(0)], dim=1)
        edge_attr = attr[train].repeat(2, 1)

        model = MechanismOrdinalGNN(16, hidden_dim=16, heads=2, dropout=0.0)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        loss_fn = CoralLoss()
        for _ in range(200):
            optimizer.zero_grad()
            logits = model(x, edge_index, edge_attr, pairs[:, train], attr[train])
            loss_fn(logits, labels[train]).backward()
            optimizer.step()

        model.eval()
        logits = model(x, edge_index, edge_attr, pairs[:, test], attr[test])
        accuracy = (coral_predict(logits) == labels[test]).float().mean().item()
        self.assertGreater(accuracy, 0.85)

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


HEADER = "drug_a,drug_b,mechanism_tag,confidence\n"
DRUG_IDS = ["DB01", "DB02", "DB03", "DB04"]


class MechanismFeatureTests(unittest.TestCase):
    def load(self, body):
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "mechanism_edges.csv")
            with open(path, "w", newline="", encoding="utf-8") as f:
                f.write(HEADER + body)
            return load_mechanism_table(path, DRUG_IDS)

    def test_both_directions_get_the_same_features(self):
        table = self.load("DB02,DB01,qt_prolongation,0.9\n")
        rows = mechanism_matrix(torch.tensor([[0, 1], [1, 0]]), table)
        self.assertTrue(torch.equal(rows[0], rows[1]))
        self.assertAlmostEqual(rows[0, MECHANISM_TAGS.index("qt_prolongation")].item(), 0.9, 5)

    def test_rows_follow_input_order_and_tag_columns(self):
        table = self.load(
            "DB01,DB02,cyp450_inhibition,0.9\n"
            "DB01,DB02,cns_depression,0.55\n"
            "DB03,DB04,protein_binding_competition,0.55\n"
        )
        rows = mechanism_matrix(torch.tensor([[2, 0], [3, 1]]), table)
        expected = torch.tensor([[0.0, 0.55, 0.0, 0.0], [0.9, 0.0, 0.55, 0.0]])
        self.assertTrue(torch.allclose(rows, expected))

    def test_unknown_tag_and_absent_pair_are_all_zeros(self):
        table = self.load("DB01,DB02,unknown,0.0\n")
        rows = mechanism_matrix(torch.tensor([[0, 2], [1, 3]]), table)
        self.assertTrue(torch.equal(rows, torch.zeros(2, NUM_TAGS)))

    def test_repeated_rows_keep_highest_confidence(self):
        table = self.load(
            "DB01,DB02,qt_prolongation,0.55\n"
            "DB02,DB01,qt_prolongation,0.9\n"
            "DB01,DB02,qt_prolongation,0.55\n"
        )
        rows = mechanism_matrix(torch.tensor([[0], [1]]), table)
        self.assertAlmostEqual(rows[0, MECHANISM_TAGS.index("qt_prolongation")].item(), 0.9, 5)

    def test_bad_rows_rejected(self):
        for body in (
            "DB01,DB02,QT_prolongation,0.9\n",  # wrong tag spelling
            "DB01,DB99,qt_prolongation,0.9\n",  # drug not in graph
            "DB01,DB02,qt_prolongation,1.5\n",  # confidence out of range
            "DB01,DB02,qt_prolongation,nan\n",
        ):
            with self.subTest(body), self.assertRaises(ValueError):
                self.load(body)

    def test_empty_pair_list_keeps_feature_width(self):
        rows = mechanism_matrix(torch.empty((2, 0), dtype=torch.long), {})
        self.assertEqual(rows.shape, (0, NUM_TAGS))

    def test_csv_features_flow_through_the_model(self):
        torch.manual_seed(0)
        table = self.load("DB01,DB02,qt_prolongation,0.9\nDB02,DB03,cns_depression,0.55\n")
        edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])
        pair_index = torch.tensor([[0, 2], [2, 3]])
        model = MechanismOrdinalGNN(8, hidden_dim=16, heads=2).eval()
        logits = model(
            torch.randn(4, 8),
            edge_index,
            mechanism_matrix(edge_index, table),
            pair_index,
            mechanism_matrix(pair_index, table),
        )
        self.assertEqual(logits.shape, (2, 2))


if __name__ == "__main__":
    unittest.main()
