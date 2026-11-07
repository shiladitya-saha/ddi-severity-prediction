"""Synthetic-data tests for M4's Model 3 training and few-shot scripts.

    python -m unittest tests.test_train_mechanism -v
"""

import json
import os
import tempfile
import unittest

import pandas as pd
import torch

from models import few_shot, train_mechanism
from models.mechanism_gnn import MechanismOrdinalGNN
from models.train_mechanism import load_inputs, train_model

TAGS = ("cyp450_inhibition", "cns_depression", "qt_prolongation")


def write_synthetic_project(folder, num_drugs=60, num_edges=500, seed=0):
    """M1/M2-shaped files: nodes.csv, edges_labeled.csv, features .pt, mechanism csv."""
    g = torch.Generator().manual_seed(seed)
    drug_ids = [f"DB{i:05d}" for i in range(num_drugs)]
    pairs = set()
    while len(pairs) < num_edges:
        a, b = torch.randint(0, num_drugs, (2,), generator=g).tolist()
        if a != b:
            pairs.add((min(a, b), max(a, b)))
    pairs = sorted(pairs)
    severity = torch.randint(0, 3, (len(pairs),), generator=g).tolist()

    proc = os.path.join(folder, "processed")
    os.makedirs(proc)
    pd.DataFrame({"node_idx": range(num_drugs), "drug_id": drug_ids}).to_csv(
        os.path.join(proc, "nodes.csv"), index=False)
    pd.DataFrame({"src": [a for a, _ in pairs], "dst": [b for _, b in pairs],
                  "description": "x", "severity": severity}).to_csv(
        os.path.join(proc, "edges_labeled.csv"), index=False)

    features = {d: torch.rand(16, generator=g) for d in drug_ids[5:]}  # 5 drugs lack features
    features_path = os.path.join(folder, "full_features.pt")
    torch.save(features, features_path)

    rows = [(drug_ids[a], drug_ids[b], TAGS[s], 0.9) for (a, b), s in zip(pairs, severity)]
    mechanism_path = os.path.join(folder, "mechanism_edges.csv")
    pd.DataFrame(rows, columns=["drug_a", "drug_b", "mechanism_tag", "confidence"]).to_csv(
        mechanism_path, index=False)
    return proc, features_path, mechanism_path


class SpyModel(MechanismOrdinalGNN):
    """Records every (message edges, scored pairs) the training loop passes in."""

    calls = []

    def forward(self, x, edge_index, edge_attr, pair_index, pair_attr=None, return_attention=False):
        if self.training:
            SpyModel.calls.append((edge_index.clone(), pair_index.clone(), pair_attr))
        return super().forward(x, edge_index, edge_attr, pair_index, pair_attr, return_attention)


class TrainMechanismTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.folder = self.tmp.name
        self.proc, self.features, self.mechanism = write_synthetic_project(self.folder)

    def tearDown(self):
        self.tmp.cleanup()

    def common_args(self):
        return ["--proc-dir", self.proc, "--features", self.features,
                "--mechanism-edges", self.mechanism,
                "--results-dir", os.path.join(self.folder, "results"),
                "--checkpoint-dir", os.path.join(self.folder, "ckpt"),
                "--epochs", "2", "--batch-size", "128", "--hidden", "16", "--heads", "2"]

    def test_missing_features_get_zero_vector_and_flag(self):
        data = load_inputs(self.proc, self.features, self.mechanism)
        self.assertEqual(data.x.shape, (60, 17))
        self.assertEqual(data.drugs_without_features, 5)
        self.assertTrue(torch.equal(data.x[:5, -1], torch.ones(5)))
        self.assertTrue(torch.equal(data.x[:5, :16], torch.zeros(5, 16)))
        self.assertTrue(torch.equal(data.x[5:, -1], torch.zeros(55)))

    def test_scored_pairs_never_in_message_graph_and_no_pair_mechanism(self):
        data = load_inputs(self.proc, self.features, self.mechanism)
        train, val = data.pairs[:, :400], data.pairs[:, 400:]
        SpyModel.calls = []
        model = SpyModel(17, hidden_dim=16, heads=2)
        train_model(model, data.x, train, data.labels[:400], data.table, val, data.labels[400:],
                     epochs=1, batch_size=100, seed=0, log=lambda *_: None)
        self.assertEqual(len(SpyModel.calls), 4)
        for edge_index, scored, pair_attr in SpyModel.calls:
            self.assertIsNone(pair_attr)
            message = {tuple(p) for p in edge_index.t().tolist()}
            for a, b in scored.t().tolist():
                self.assertNotIn((a, b), message)
                self.assertNotIn((b, a), message)
            # Every other training pair stays in the message graph, both directions.
            self.assertEqual(edge_index.size(1), 2 * (400 - scored.size(1)))

    def test_pair_mechanism_flag_feeds_pair_tags(self):
        data = load_inputs(self.proc, self.features, self.mechanism)
        SpyModel.calls = []
        model = SpyModel(17, hidden_dim=16, heads=2)
        train_model(model, data.x, data.pairs[:, :400], data.labels[:400], data.table,
                    data.pairs[:, 400:], data.labels[400:], epochs=1, batch_size=400,
                    use_pair_attr=True, seed=0, log=lambda *_: None)
        self.assertIsNotNone(SpyModel.calls[0][2])

    def test_run_writes_results_and_records_leakage_controls(self):
        train_mechanism.main(self.common_args())
        results = os.path.join(self.folder, "results")
        test = json.load(open(os.path.join(results, "model3_results.json")))
        self.assertEqual(test["split"], "test")
        details = json.load(open(os.path.join(results, "model3_details.json")))
        self.assertFalse(details["leakage_controls"]["scored_pair_mechanism_input"])
        self.assertTrue(details["leakage_controls"]["scored_training_pairs_removed_from_message_graph"])
        self.assertIn("severe_benchmark", details["results"])
        self.assertTrue(os.path.exists(os.path.join(self.folder, "ckpt", "model3.pt")))

    def test_few_shot_run_writes_each_k(self):
        few_shot.main(self.common_args() + ["--shots", "0", "2"])
        out = json.load(open(os.path.join(self.folder, "results", "few_shot_results.json")))
        self.assertEqual(sorted(out["per_k"]), ["0", "2"])
        self.assertEqual(out["per_k"]["0"]["support_edges_added"], 0)


class SupportEdgeTests(unittest.TestCase):
    def setUp(self):
        # drugs 0-2 train, 3-4 test, 5 val
        self.pairs = torch.tensor([[0, 1, 3, 0, 1, 2, 0, 3, 5],
                                   [1, 2, 4, 3, 3, 3, 4, 5, 4]])
        self.part = {"train": torch.tensor([0, 1]), "test": torch.tensor([2]),
                     "bridge": torch.tensor([3, 4, 5, 6, 7, 8])}

    def test_only_test_to_train_edges_and_at_most_k(self):
        chosen, given = few_shot.support_edges(self.pairs, self.part, k=2, seed=0)
        self.assertEqual(given, {3: 2, 4: 1})
        for i in chosen.tolist():
            a, b = self.pairs[:, i].tolist()
            self.assertTrue({a, b} & {3, 4} and {a, b} & {0, 1, 2})

    def test_zero_shot_adds_nothing(self):
        chosen, given = few_shot.support_edges(self.pairs, self.part, k=0)
        self.assertEqual(chosen.numel(), 0)
        self.assertEqual(set(given.values()), {0})


if __name__ == "__main__":
    unittest.main()
