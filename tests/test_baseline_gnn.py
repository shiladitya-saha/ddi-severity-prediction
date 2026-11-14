"""Synthetic graph checks for M3's binary baseline and training utilities."""

import tempfile
import unittest
from pathlib import Path

import torch
from torch_geometric.data import Data

from eval.split import load_splits, make_splits, save_splits
from models.baseline_gnn import BinaryBaselineGNN
from models.train import (
    align_node_features,
    build_binary_edge_split,
    build_m5_detection_split,
    load_checkpoint,
    load_binary_inputs,
    run_m5_detection_experiment,
    run_dry_run,
    run_synthetic_dry_run,
    train_binary_model,
)


def toy_positive_pairs(num_nodes=12):
    pairs = [(i, (i + 1) % num_nodes) for i in range(num_nodes)]
    pairs.extend((i, (i + 3) % num_nodes) for i in range(num_nodes // 2))
    return torch.tensor(pairs, dtype=torch.long).t().contiguous()


class BinaryBaselineTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(3)
        self.x = torch.randn(12, 16)
        self.pairs = toy_positive_pairs()
        self.edge_index = torch.cat((self.pairs, self.pairs.flip(0)), dim=1)
        self.model = BinaryBaselineGNN(16, hidden_dim=8, num_layers=2, dropout=0.0)

    def test_forward_shape_finite_loss_and_gradients(self):
        logits = self.model(self.x, self.edge_index, self.pairs[:, :5])
        self.assertEqual(tuple(logits.shape), (5,))
        self.assertTrue(torch.isfinite(logits).all())
        labels = torch.tensor([1.0, 0.0, 1.0, 0.0, 1.0])
        loss = torch.nn.BCEWithLogitsLoss()(logits, labels)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(all(parameter.grad is not None for parameter in self.model.parameters()))

    def test_pair_order_is_symmetric(self):
        pair_index = self.pairs[:, :6]
        forward = self.model(self.x, self.edge_index, pair_index)
        reverse = self.model(self.x, self.edge_index, pair_index.flip(0))
        self.assertTrue(torch.allclose(forward, reverse, atol=1e-6))

    def test_accepts_pyg_data(self):
        graph = Data(x=self.x, edge_index=self.edge_index)
        logits = self.model(graph, self.pairs[:, :4])
        self.assertEqual(tuple(logits.shape), (4,))

    def test_rejects_bad_input_shapes_and_indices(self):
        with self.assertRaises(ValueError):
            self.model(self.x, self.edge_index, torch.zeros((3, 2), dtype=torch.long))
        with self.assertRaises(ValueError):
            self.model(self.x, self.edge_index, torch.tensor([[0], [12]]))

    def test_training_updates_parameters_and_checkpoint_round_trips(self):
        split = build_binary_edge_split(self.pairs, self.x.size(0), seed=9)
        before = [parameter.detach().clone() for parameter in self.model.parameters()]
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint = Path(temp_dir) / "baseline.pt"
            result = train_binary_model(
                self.model,
                self.x,
                split.message_edge_index,
                split.train_pairs,
                split.train_labels,
                split.validation_pairs,
                split.validation_labels,
                epochs=2,
                batch_size=8,
                seed=9,
                device="cpu",
                checkpoint_path=checkpoint,
            )
            self.assertTrue(checkpoint.exists())
            self.assertEqual(len(result["history"]), 2)
            self.assertTrue(any(
                not torch.equal(old, new)
                for old, new in zip(before, self.model.parameters())
            ))

            restored = BinaryBaselineGNN(**self.model.config)
            checkpoint_data = load_checkpoint(checkpoint, restored)
            self.assertGreaterEqual(checkpoint_data["epoch"], 1)
            self.model.eval()
            restored.eval()
            expected = self.model(self.x, self.edge_index, self.pairs[:, :5])
            actual = restored(self.x, self.edge_index, self.pairs[:, :5])
            self.assertTrue(torch.allclose(expected, actual, atol=1e-6))

    def test_training_is_reproducible_with_a_fixed_seed(self):
        split = build_binary_edge_split(self.pairs, self.x.size(0), seed=27)

        def fit_once():
            torch.manual_seed(101)
            model = BinaryBaselineGNN(16, hidden_dim=8, dropout=0.0)
            result = train_binary_model(
                model,
                self.x,
                split.message_edge_index,
                split.train_pairs,
                split.train_labels,
                split.validation_pairs,
                split.validation_labels,
                epochs=2,
                batch_size=8,
                seed=27,
                device="cpu",
            )
            return model, result

        first_model, first_result = fit_once()
        second_model, second_result = fit_once()
        self.assertEqual(first_result["history"], second_result["history"])
        for first, second in zip(first_model.parameters(), second_model.parameters()):
            self.assertTrue(torch.equal(first, second))

    def test_split_is_reproducible_and_excludes_validation_test_edges(self):
        first = build_binary_edge_split(self.pairs, 12, seed=44)
        second = build_binary_edge_split(self.pairs.flip(0), 12, seed=44)
        self.assertTrue(torch.equal(first.train_pairs, second.train_pairs))
        self.assertTrue(torch.equal(first.validation_pairs, second.validation_pairs))
        context = {tuple(pair) for pair in first.context_positive_pairs.t().tolist()}
        for pairs, labels in (
            (first.validation_pairs, first.validation_labels),
            (first.test_pairs, first.test_labels),
        ):
            held_out_positive = pairs[:, labels.bool()].t().tolist()
            for pair in held_out_positive:
                key = tuple(sorted(pair))
                self.assertNotIn(key, context)
                self.assertFalse(torch.any(
                    (first.message_edge_index[0] == key[0])
                    & (first.message_edge_index[1] == key[1])
                ))
                self.assertFalse(torch.any(
                    (first.message_edge_index[0] == key[1])
                    & (first.message_edge_index[1] == key[0])
                ))
        # No canonical unordered pair appears in more than one label split.
        sets = []
        for pairs in (first.train_pairs, first.validation_pairs, first.test_pairs):
            sets.append({tuple(pair) for pair in pairs.t().tolist()})
        self.assertFalse(sets[0] & sets[1])
        self.assertFalse(sets[0] & sets[2])
        self.assertFalse(sets[1] & sets[2])

    def test_feature_alignment_follows_graph_id_order(self):
        aligned = align_node_features(
            ["drug-b", "drug-a"],
            {"drug-a": torch.tensor([1.0, 2.0]), "drug-b": torch.tensor([3.0, 4.0])},
        )
        self.assertEqual(aligned.tolist(), [[3.0, 4.0], [1.0, 2.0]])
        with self.assertRaises(ValueError):
            align_node_features(["missing"], {})

    def test_loads_m1_day2_graph_using_companion_nodes_csv(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            ids = ["DB-B", "DB-A", "DB-C"]
            (root / "nodes.csv").write_text(
                "node_idx,drug_id\n0,DB-B\n1,DB-A\n2,DB-C\n", encoding="utf-8"
            )
            edges = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
            torch.save({
                "edge_index": edges,
                "edge_severity": torch.tensor([2, 2, 0, 0]),
                "num_nodes": 3,
                "severity_names": ["minor", "moderate", "major"],
            }, root / "graph.pt")
            torch.save({
                "DB-C": torch.tensor([30.0]),
                "DB-A": torch.tensor([20.0]),
                "DB-B": torch.tensor([10.0]),
            }, root / "features.pt")
            drug_ids, x, positives = load_binary_inputs(root / "graph.pt", root / "features.pt")
        self.assertEqual(drug_ids, ids)
        self.assertEqual(x[:, 0].tolist(), [10.0, 20.0, 30.0])
        self.assertEqual({tuple(pair) for pair in positives.t().tolist()}, {(0, 1), (1, 2)})

    def test_m1_adapter_rejects_bad_severity_alignment_and_ranges(self):
        edges = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
        with self.assertRaisesRegex(ValueError, "one value per edge"):
            load_graph = {"edge_index": edges, "edge_severity": torch.tensor([1])}
            from models.train import _validated_graph_edges
            _validated_graph_edges(load_graph, edges, 2)
        with self.assertRaisesRegex(ValueError, r"in \{0, 1, 2\}"):
            from models.train import _validated_graph_edges
            _validated_graph_edges(
                {"edge_index": edges, "edge_severity": torch.tensor([3, 3])}, edges, 2
            )

    def test_synthetic_end_to_end_is_explicitly_labeled_and_finite(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            result = run_synthetic_dry_run(
                epochs=2,
                batch_size=32,
                seed=13,
                results_path=Path(temp_dir) / "synthetic.json",
                device="cpu",
            )
            self.assertEqual(result["dataset_kind"], "synthetic_fixture")
            self.assertTrue(torch.isfinite(torch.tensor(result["test_loss"])))
            self.assertTrue(torch.isfinite(torch.tensor(result["best_validation_loss"])))

    def test_file_based_dry_run_writes_only_local_synthetic_artifacts(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            graph_path = root / "graph.pt"
            features_path = root / "features.pt"
            checkpoint_path = root / "checkpoints" / "baseline.pt"
            results_path = root / "results" / "dry_run.json"
            drug_ids = [f"drug-{index}" for index in range(12)]
            torch.save({"drug_id": drug_ids, "edge_index": self.pairs}, graph_path)
            torch.save(
                {drug_id: self.x[index] for index, drug_id in enumerate(drug_ids)},
                features_path,
            )
            result = run_dry_run(
                graph_path,
                features_path,
                epochs=1,
                batch_size=16,
                seed=37,
                checkpoint_path=checkpoint_path,
                results_path=results_path,
                device="cpu",
            )
            self.assertEqual(result["num_drugs"], 12)
            self.assertTrue(checkpoint_path.exists())
            self.assertTrue(results_path.exists())
            self.assertEqual(result["held_out_positive_edges_in_message_graph"], 0)

    def test_m5_binary_split_keeps_positive_folds_and_negatives_disjoint(self):
        labels = torch.tensor([index % 3 for index in range(self.pairs.size(1))])
        m5 = make_splits(
            self.pairs.t().numpy(), labels.numpy(), seed=13, severe_benchmark_frac=0.15
        )
        split, detection_splits = build_m5_detection_split(
            self.pairs, self.x.size(0), m5, negative_ratio=1.0, seed=13
        )
        for name, source_indices, detection_indices in (
            ("train", m5.train, split.train_indices),
            ("val", m5.val, split.validation_indices),
            ("test", m5.test, split.test_indices),
            ("severe_benchmark", m5["severe_benchmark"], split.severe_indices),
        ):
            source_pairs = {
                tuple(pair)
                for pair in torch.as_tensor(m5.meta["canonical_pairs"])[source_indices].tolist()
            }
            candidates = split.pairs[:, detection_indices]
            observed = {
                tuple(pair)
                for pair, label in zip(candidates.t().tolist(), split.labels[detection_indices].tolist())
                if label == 1
            }
            self.assertEqual(observed, source_pairs, name)
        self.assertEqual(detection_splits.strategy, "m5_positive_folds_with_partitioned_sampled_negatives")
        self.assertEqual(split.message_edge_index.size(1), 2 * m5.train.size)
        negative_pairs = {
            tuple(sorted(pair))
            for pair, label in zip(split.pairs.t().tolist(), split.labels.tolist())
            if label == 0
        }
        self.assertEqual(len(negative_pairs), int((split.labels == 0).sum()))
        held = set()
        for index in (split.validation_indices, split.test_indices, split.severe_indices):
            held.update(tuple(sorted(pair)) for pair in split.pairs[:, index].t().tolist())
        context = {tuple(sorted(pair)) for pair in split.message_edge_index.t().tolist()}
        self.assertFalse(context & held)

    def test_m5_detection_runner_exports_predictions_and_scores_via_m5(self):
        labels = torch.tensor([index % 3 for index in range(self.pairs.size(1))])
        m5 = make_splits(
            self.pairs.t().numpy(), labels.numpy(), seed=19, severe_benchmark_frac=0.15
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            graph = root / "graph.pt"
            features = root / "features.pt"
            split_path = save_splits(m5, root / "severity_splits.json")
            node_ids = [f"SYN-{index}" for index in range(self.x.size(0))]
            directed_pairs = torch.cat((self.pairs, self.pairs.flip(0)), dim=1)
            directed_labels = torch.cat((labels, labels))
            torch.save(
                {
                    "drug_id": node_ids,
                    "edge_index": directed_pairs,
                    "edge_severity": directed_labels,
                    "num_nodes": len(node_ids),
                },
                graph,
            )
            torch.save(
                {drug_id: self.x[index] for index, drug_id in enumerate(node_ids)}, features
            )
            results = run_m5_detection_experiment(
                graph,
                features,
                split_path,
                epochs=1,
                batch_size=32,
                seed=19,
                checkpoint_path=root / "model.pt",
                results_path=root / "model1_results.json",
                predictions_path=root / "preds_detection_baseline.npz",
                detection_splits_path=root / "detection_splits.json",
                device="cpu",
                dataset_kind="synthetic_fixture",
            )
            self.assertEqual(results["dataset_kind"], "synthetic_fixture")
            self.assertTrue((root / "model1_results.json").is_file())
            self.assertTrue((root / "model.pt").is_file())
            detection = load_splits(root / "detection_splits.json")
            with __import__("numpy").load(root / "preds_detection_baseline.npz") as values:
                self.assertEqual(str(values["task"]), "detection")
                self.assertEqual(values["y_true_bin"].shape, values["y_score"].shape)
                self.assertTrue(set(values["edge_idx"].tolist()).issubset(
                    set(detection.val.tolist()) | set(detection.test.tolist())
                ))
            from scripts.run_eval import main as run_eval_main

            code = run_eval_main(
                [
                    "--preds", str(root / "preds_detection_baseline.npz"),
                    "--results-dir", str(root),
                    "--splits", str(root / "detection_splits.json"),
                    "--out", str(root / "detection_comparison.md"),
                ]
            )
            self.assertEqual(code, 0)
            comparison = (root / "detection_comparison.md").read_text(encoding="utf-8")
            self.assertIn("Binary interaction detection", comparison)


if __name__ == "__main__":
    unittest.main()
