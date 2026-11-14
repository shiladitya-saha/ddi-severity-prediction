"""Synthetic tests for M3's ordinal GNN and M5 split integration."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from eval.split import make_splits, save_splits
from models.ordinal_gnn import (
    OrdinalGNN,
    build_ordinal_edge_split,
    load_checkpoint,
    predict_ordinal,
    run_ordinal_experiment,
    train_ordinal_model,
)
from models.ordinal_loss import CoralLoss


def fixture_data(seed: int = 31):
    generator = torch.Generator().manual_seed(seed)
    num_nodes = 20
    x = torch.randn(num_nodes, 10, generator=generator)
    pairs = torch.tensor(
        [
            (a, b)
            for a in range(num_nodes)
            for b in range(a + 1, num_nodes)
            if (a * 5 + b * 7) % 4 != 0
        ],
        dtype=torch.long,
    ).t().contiguous()
    labels = torch.tensor(
        [(a + 2 * b) % 3 for a, b in pairs.t().tolist()], dtype=torch.long
    )
    splits = make_splits(
        pairs.t().numpy(), labels.numpy(), seed=seed, severe_benchmark_frac=0.15
    )
    edge_split = build_ordinal_edge_split(pairs, labels, splits, num_nodes)
    return x, pairs, labels, splits, edge_split


class OrdinalGNNTests(unittest.TestCase):
    def setUp(self):
        self.x, self.pairs, self.labels, self.splits, self.edge_split = fixture_data()
        self.model = OrdinalGNN(10, hidden_dim=12, num_layers=2, dropout=0.0)

    def test_coral_dimensions_probabilities_and_predictions(self):
        logits = self.model(self.x, self.edge_split.message_edge_index, self.pairs[:, :7])
        self.assertEqual(tuple(logits.shape), (7, 2))
        probabilities = self.model.probabilities(logits)
        predictions = self.model.predict(logits)
        self.assertEqual(tuple(probabilities.shape), (7, 3))
        self.assertEqual(tuple(predictions.shape), (7,))
        self.assertTrue(torch.allclose(probabilities.sum(dim=1), torch.ones(7), atol=1e-6))
        self.assertTrue(torch.all(probabilities >= 0))
        self.assertTrue(torch.all((predictions >= 0) & (predictions <= 2)))

    def test_pair_order_is_symmetric(self):
        pairs = self.pairs[:, :9]
        forward = self.model(self.x, self.edge_split.message_edge_index, pairs)
        reverse = self.model(self.x, self.edge_split.message_edge_index, pairs.flip(0))
        self.assertTrue(torch.allclose(forward, reverse, atol=1e-6))

    def test_coral_loss_is_finite_and_differentiable(self):
        logits = torch.randn(12, 2, requires_grad=True)
        labels = torch.tensor([0, 1, 2] * 4)
        loss = CoralLoss(num_classes=3)(logits, labels)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())

    def test_invalid_dimensions_labels_and_self_pairs_fail_clearly(self):
        with self.assertRaisesRegex(ValueError, "x must have shape"):
            self.model(self.x[:, :8], self.edge_split.message_edge_index, self.pairs[:, :1])
        with self.assertRaisesRegex(ValueError, "self-pairs"):
            self.model(
                self.x,
                self.edge_split.message_edge_index,
                torch.tensor([[0], [0]], dtype=torch.long),
            )
        with self.assertRaisesRegex(ValueError, "labels must use"):
            build_ordinal_edge_split(
                self.pairs,
                torch.full_like(self.labels, 3),
                self.splits,
                self.x.size(0),
            )
        with self.assertRaisesRegex(ValueError, "integer severity labels"):
            build_ordinal_edge_split(
                self.pairs,
                self.labels.float() + 0.5,
                self.splits,
                self.x.size(0),
            )
        with self.assertRaisesRegex(TypeError, "integer node indices"):
            build_ordinal_edge_split(
                self.pairs.float() + 0.5,
                self.labels,
                self.splits,
                self.x.size(0),
            )

    def test_m5_split_excludes_reverse_held_out_edges_from_context(self):
        context = {
            tuple(sorted(pair))
            for pair in self.edge_split.message_edge_index.t().tolist()
        }
        for idx in (
            self.edge_split.validation_indices,
            self.edge_split.test_indices,
            self.edge_split.severe_indices,
        ):
            held = {tuple(pair) for pair in self.pairs[:, idx].t().tolist()}
            self.assertFalse(context & held)
        expected_context = 2 * self.edge_split.train_indices.numel()
        self.assertEqual(self.edge_split.message_edge_index.size(1), expected_context)

    def test_training_validation_checkpoint_and_inference(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint = Path(temp_dir) / "ordinal.pt"
            result = train_ordinal_model(
                self.model,
                self.x,
                self.edge_split.message_edge_index,
                self.edge_split.train_pairs,
                self.edge_split.train_labels,
                self.edge_split.validation_pairs,
                self.edge_split.validation_labels,
                epochs=2,
                batch_size=32,
                seed=31,
                device="cpu",
                checkpoint_path=checkpoint,
            )
            self.assertTrue(checkpoint.exists())
            self.assertGreaterEqual(result["best_epoch"], 1)
            restored = OrdinalGNN(**self.model.config)
            data = load_checkpoint(checkpoint, restored)
            self.assertGreaterEqual(data["epoch"], 1)
            resumed = train_ordinal_model(
                restored,
                self.x,
                self.edge_split.message_edge_index,
                self.edge_split.train_pairs,
                self.edge_split.train_labels,
                self.edge_split.validation_pairs,
                self.edge_split.validation_labels,
                epochs=3,
                batch_size=32,
                seed=31,
                device="cpu",
                checkpoint_path=checkpoint,
                resume_from=checkpoint,
            )
            self.assertGreaterEqual(resumed["best_epoch"], data["epoch"])
            logits, probabilities, predictions = predict_ordinal(
                restored,
                self.x,
                self.edge_split.message_edge_index,
                self.edge_split.test_pairs,
                device="cpu",
            )
            self.assertEqual(tuple(logits.shape), (self.edge_split.test_labels.numel(), 2))
            self.assertEqual(tuple(probabilities.shape), (self.edge_split.test_labels.numel(), 3))
            self.assertEqual(tuple(predictions.shape), (self.edge_split.test_labels.numel(),))

    def test_project_runner_exports_m5_severity_predictions_and_results(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            node_ids = [f"SYN-{index}" for index in range(self.x.size(0))]
            directed_pairs = torch.cat((self.pairs, self.pairs.flip(0)), dim=1)
            directed_labels = torch.cat((self.labels, self.labels))
            graph_path = root / "graph.pt"
            features_path = root / "features.pt"
            split_path = save_splits(self.splits, root / "splits.json")
            torch.save(
                {
                    "drug_id": node_ids,
                    "edge_index": directed_pairs,
                    "edge_severity": directed_labels,
                    "num_nodes": len(node_ids),
                },
                graph_path,
            )
            torch.save(
                {drug_id: self.x[index] for index, drug_id in enumerate(node_ids)},
                features_path,
            )
            result = run_ordinal_experiment(
                graph_path,
                features_path,
                split_path,
                class_weights_path=root / "missing_weights.json",
                epochs=1,
                batch_size=64,
                seed=31,
                device="cpu",
                checkpoint_path=root / "model2.pt",
                results_path=root / "model2_results.json",
                predictions_path=root / "preds_ordinal_gnn.npz",
                dataset_kind="synthetic_fixture",
            )
            self.assertEqual(result["dataset_kind"], "synthetic_fixture")
            self.assertEqual(result["held_out_edges_in_message_graph"], 0)
            self.assertIn("test", result["metrics"])
            self.assertIn("severe_benchmark", result["metrics"])
            self.assertTrue((root / "model2_results.json").is_file())
            self.assertTrue((root / "model2.pt").is_file())
            with __import__("numpy").load(root / "preds_ordinal_gnn.npz") as values:
                self.assertEqual(values["y_proba"].shape[1], 3)
                self.assertEqual(values["y_pred"].shape, values["y_true"].shape)
                self.assertEqual(values["edge_idx"].size, values["y_true"].size)
            from scripts.run_eval import main as run_eval_main

            code = run_eval_main(
                [
                    "--preds", str(root / "preds_ordinal_gnn.npz"),
                    "--results-dir", str(root),
                    "--splits", str(split_path),
                    "--out", str(root / "ordinal_comparison.md"),
                ]
            )
            self.assertEqual(code, 0)
            report = (root / "ordinal_comparison.md").read_text(encoding="cp1252")
            self.assertIn("Severity prediction", report)
            self.assertIn("Held-out severe subset", report)

    def test_m5_canonical_pair_order_mismatch_is_rejected(self):
        bad_pairs = self.pairs.flip(1).contiguous()
        with self.assertRaisesRegex(ValueError, "canonical pair order"):
            build_ordinal_edge_split(bad_pairs, self.labels, self.splits, self.x.size(0))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
