"""
End-to-end tests for the M5 eval harness (Day 2 deliverable: "passing test on
dummy data").

Run:
    python -m unittest discover -s tests -v
    # or, if pytest is around:
    pytest tests -q
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval import attention_viz as av  # noqa: E402
from eval import benchmark_severe_subset as ob  # noqa: E402
from tests.dummy_data import (  # noqa: E402
    FakeAttentionModel,
    make_dummy_graph,
    make_dummy_predictions,
)
from eval.metrics import (  # noqa: E402
    MAJOR,
    MINOR,
    MODERATE,
    EvalResult,
    binary_to_severity,
    comparison_to_markdown,
    coral_to_labels,
    evaluate,
    expected_severity,
    metrics_to_markdown,
    softmax_to_labels,
)
from eval.split import (  # noqa: E402
    drug_disjoint_split,
    load_splits,
    make_splits,
    normalize_edge_index,
    save_splits,
    stratified_split,
    verify_splits,
)


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #


class TestMetrics(unittest.TestCase):
    def test_perfect_prediction(self):
        y = np.array([MINOR, MODERATE, MAJOR, MAJOR, MINOR])
        res = evaluate(y, y, model_name="oracle")
        self.assertAlmostEqual(res.macro_f1, 1.0)
        self.assertAlmostEqual(res.mae, 0.0)
        self.assertAlmostEqual(res.major_recall, 1.0)
        self.assertAlmostEqual(res.qwk, 1.0)
        self.assertAlmostEqual(res.within_one_accuracy, 1.0)

    def test_mae_is_ordinal(self):
        # calling a MAJOR a MINOR must cost 2, not 1
        far = evaluate([MAJOR], [MINOR])
        near = evaluate([MAJOR], [MODERATE])
        self.assertAlmostEqual(far.mae, 2.0)
        self.assertAlmostEqual(near.mae, 1.0)

    def test_major_recall_hand_computed(self):
        y_true = [MAJOR, MAJOR, MAJOR, MAJOR, MINOR]
        y_pred = [MAJOR, MAJOR, MODERATE, MINOR, MINOR]
        res = evaluate(y_true, y_pred)
        self.assertAlmostEqual(res.major_recall, 0.5)  # 2 of 4 caught
        self.assertAlmostEqual(res.per_class["MAJOR"]["precision"], 1.0)

    def test_accuracy_is_misleading_under_imbalance(self):
        # 90 MINOR, 10 MAJOR; model predicts MINOR always
        y_true = np.array([MINOR] * 90 + [MAJOR] * 10)
        y_pred = np.zeros_like(y_true)
        res = evaluate(y_true, y_pred, model_name="all_minor")
        self.assertAlmostEqual(res.accuracy, 0.90)
        self.assertAlmostEqual(res.major_recall, 0.0)
        self.assertLess(res.macro_f1, 0.5)  # headline metric exposes it

    def test_confusion_matrix_orientation(self):
        res = evaluate([MINOR, MINOR, MAJOR], [MINOR, MAJOR, MAJOR])
        cm = np.array(res.confusion)
        self.assertEqual(cm.shape, (3, 3))
        self.assertEqual(cm[MINOR, MINOR], 1)  # rows = true
        self.assertEqual(cm[MINOR, MAJOR], 1)
        self.assertEqual(cm[MAJOR, MAJOR], 1)

    def test_probabilistic_metrics_attach(self):
        data = make_dummy_graph(n_drugs=60, n_edges=400, seed=3)
        y_pred, proba = make_dummy_predictions(data.labels, accuracy=0.8, seed=4)
        res = evaluate(data.labels, y_pred, y_proba=proba, model_name="probs")
        self.assertIsNotNone(res.major_auc)
        self.assertGreater(res.major_auc, 0.5)
        self.assertIsNotNone(res.macro_auc)

    def test_unnormalized_proba_is_renormalized_not_rejected(self):
        y = np.array([MINOR, MODERATE, MAJOR, MAJOR])
        proba = np.array([[2.0, 1.0, 1.0], [1.0, 3.0, 1.0], [1.0, 1.0, 4.0], [0.5, 1.0, 3.0]])
        res = evaluate(y, softmax_to_labels(proba), y_proba=proba)
        self.assertIn("proba_warning", res.notes)

    def test_bad_labels_rejected(self):
        with self.assertRaises(ValueError):
            evaluate([0, 1, 3], [0, 1, 2])  # label 3 is outside the contract
        with self.assertRaises(ValueError):
            evaluate([0, 1], [0, 1, 2])  # shape mismatch
        with self.assertRaises(ValueError):
            evaluate([], [])

    def test_result_roundtrip(self):
        res = evaluate([MINOR, MAJOR], [MINOR, MAJOR], model_name="rt")
        with tempfile.TemporaryDirectory() as d:
            p = res.save(Path(d) / "r.json")
            back = EvalResult.load(p)
        self.assertEqual(back.model_name, "rt")
        self.assertAlmostEqual(back.macro_f1, res.macro_f1)
        json.dumps(res.to_dict())  # must stay JSON-serializable


class TestDecoders(unittest.TestCase):
    def test_expected_severity_beats_argmax_ordinally(self):
        proba = np.array([[0.40, 0.35, 0.25]])
        self.assertEqual(softmax_to_labels(proba)[0], MINOR)
        self.assertEqual(expected_severity(proba)[0], MODERATE)  # E[y] = 0.85

    def test_coral_decoding_is_monotone(self):
        cum = np.array([[0.9, 0.8], [0.9, 0.1], [0.2, 0.05]])  # P(y>0), P(y>1)
        np.testing.assert_array_equal(coral_to_labels(cum), [MAJOR, MODERATE, MINOR])

    def test_binary_lift(self):
        np.testing.assert_array_equal(binary_to_severity([1, 0, 1]), [MAJOR, MINOR, MAJOR])
        with self.assertRaises(ValueError):
            binary_to_severity([0, 2])

    def test_decoder_shape_validation(self):
        with self.assertRaises(ValueError):
            expected_severity(np.zeros((4, 2)))
        with self.assertRaises(ValueError):
            coral_to_labels(np.zeros((4, 3)))


# --------------------------------------------------------------------------- #
# Splits
# --------------------------------------------------------------------------- #


class TestSplits(unittest.TestCase):
    def setUp(self):
        self.data = make_dummy_graph(n_drugs=150, n_edges=1200, seed=5)

    def test_edge_index_orientation_tolerated(self):
        ei_t = normalize_edge_index(self.data.edge_index)  # (2, E) in
        ei_f = normalize_edge_index(self.data.edge_index.T)  # (E, 2) in
        self.assertEqual(ei_t.shape, (self.data.n_edges, 2))
        np.testing.assert_array_equal(ei_t, ei_f)

    def test_ratios_are_roughly_70_15_15(self):
        s = make_splits(self.data.edge_index, self.data.labels, seed=1, severe_benchmark_frac=0.0)
        total = s.train.size + s.val.size + s.test.size
        self.assertEqual(total, self.data.n_edges)
        self.assertAlmostEqual(s.train.size / total, 0.70, delta=0.03)
        self.assertAlmostEqual(s.val.size / total, 0.15, delta=0.03)
        self.assertAlmostEqual(s.test.size / total, 0.15, delta=0.03)

    def test_no_index_leakage_and_all_classes_present(self):
        s = make_splits(self.data.edge_index, self.data.labels, seed=1)
        problems = verify_splits(
            s, labels=self.data.labels, edge_index=self.data.edge_index, strict=False
        )
        self.assertEqual(problems, [])

    def test_severe_benchmark_is_reserved_and_pure(self):
        s = make_splits(
            self.data.edge_index, self.data.labels, seed=1, severe_benchmark_frac=0.2
        )
        sb = s["severe_benchmark"]
        self.assertGreater(sb.size, 0)
        self.assertTrue(np.all(self.data.labels[sb] == MAJOR))
        # never seen in training or evaluation
        for name in ("train", "val", "test"):
            self.assertEqual(np.intersect1d(sb, s[name]).size, 0)
        n_major = int((self.data.labels == MAJOR).sum())
        self.assertAlmostEqual(sb.size / n_major, 0.2, delta=0.02)

    def test_major_still_trainable_after_reserving(self):
        s = make_splits(
            self.data.edge_index, self.data.labels, seed=1, severe_benchmark_frac=0.2
        )
        self.assertGreater((self.data.labels[s.train] == MAJOR).sum(), 0)

    def test_stratification_preserves_class_ratio(self):
        s = make_splits(self.data.edge_index, self.data.labels, seed=1, severe_benchmark_frac=0.0)
        overall = (self.data.labels == MAJOR).mean()
        for name in ("train", "val", "test"):
            self.assertAlmostEqual((self.data.labels[s[name]] == MAJOR).mean(), overall, delta=0.05)

    def test_same_seed_same_split(self):
        a = make_splits(self.data.edge_index, self.data.labels, seed=99)
        b = make_splits(self.data.edge_index, self.data.labels, seed=99)
        c = make_splits(self.data.edge_index, self.data.labels, seed=100)
        np.testing.assert_array_equal(a.train, b.train)
        self.assertFalse(np.array_equal(a.train, c.train))

    def test_split_roundtrip(self):
        s = make_splits(self.data.edge_index, self.data.labels, seed=2)
        with tempfile.TemporaryDirectory() as d:
            p = save_splits(s, Path(d) / "split.json")
            back = load_splits(p)
        self.assertEqual(set(back.names), set(s.names))
        for name in s.names:
            np.testing.assert_array_equal(back[name], s[name])
        self.assertEqual(back.seed, s.seed)

    def test_drug_disjoint_has_no_shared_drugs(self):
        s = drug_disjoint_split(self.data.edge_index, self.data.labels, seed=8)
        ei = normalize_edge_index(self.data.edge_index)
        train_drugs = set(ei[s.train].ravel().tolist())
        test_drugs = set(ei[s.test].ravel().tolist())
        self.assertEqual(train_drugs & test_drugs, set())
        self.assertGreater(s.test.size, 0)
        np.testing.assert_array_equal(s["zero_shot"], s.test)

    def test_verify_catches_injected_leakage(self):
        s = make_splits(self.data.edge_index, self.data.labels, seed=1)
        s.indices["test"] = np.concatenate([s.test, s.train[:5]])
        problems = verify_splits(s, labels=self.data.labels, strict=False)
        self.assertTrue(any("leakage" in p for p in problems))
        with self.assertRaises(AssertionError):
            verify_splits(s, labels=self.data.labels, strict=True)

    def test_bad_ratios_rejected(self):
        with self.assertRaises(ValueError):
            stratified_split(self.data.labels, ratios=(0.8, 0.3, 0.1))
        with self.assertRaises(ValueError):
            make_splits(self.data.edge_index, self.data.labels, severe_benchmark_frac=1.0)
        with self.assertRaises(ValueError):
            make_splits(self.data.edge_index, self.data.labels[:-3])  # length mismatch


# --------------------------------------------------------------------------- #
# OGBL-DDI
# --------------------------------------------------------------------------- #


class TestSevereSubsetBenchmark(unittest.TestCase):
    def setUp(self):
        self.data = make_dummy_graph(n_drugs=150, n_edges=1500, seed=31)
        self.splits = make_splits(
            self.data.edge_index, self.data.labels, seed=31, severe_benchmark_frac=0.2
        )
        self.y_pred, self.proba = make_dummy_predictions(
            self.data.labels, accuracy=0.8, seed=32
        )

    def test_benchmark_severe_scores_only_reserved_rows(self):
        res = ob.benchmark_severe(
            self.data.labels, self.y_pred, self.splits, y_proba=self.proba, model_name="m3"
        )
        self.assertEqual(res.n_samples, self.splits["severe_benchmark"].size)
        self.assertEqual(res.split, "severe_benchmark")
        self.assertEqual(res.support["MINOR"], 0)
        self.assertEqual(res.support["MODERATE"], 0)
        self.assertGreater(res.support["MAJOR"], 0)
        self.assertEqual(res.notes["headline_metric"], "major_recall")

    def test_benchmark_severe_rejects_contaminated_partition(self):
        self.splits.indices["severe_benchmark"] = np.concatenate(
            [self.splits["severe_benchmark"], self.splits.train[:3]]
        )
        with self.assertRaises(ValueError):
            ob.benchmark_severe(self.data.labels, self.y_pred, self.splits)

    def test_benchmark_severe_requires_the_partition(self):
        plain = make_splits(
            self.data.edge_index, self.data.labels, seed=1, severe_benchmark_frac=0.0
        )
        with self.assertRaises(ValueError):  # present but empty
            ob.benchmark_severe(self.data.labels, self.y_pred, plain)
        del plain.indices["severe_benchmark"]
        with self.assertRaises(KeyError):
            ob.benchmark_severe(self.data.labels, self.y_pred, plain)

    def test_miss_profile_sums_to_one(self):
        prof = ob.severity_miss_profile(self.data.labels, self.y_pred, self.splits)
        shares = sum(v["share"] for k, v in prof.items() if not k.startswith("_"))
        self.assertAlmostEqual(shares, 1.0, places=6)
        self.assertEqual(prof["_n"]["count"], self.splits["severe_benchmark"].size)

    def test_severe_markdown_renders(self):
        res = ob.benchmark_severe(
            self.data.labels, self.y_pred, self.splits, model_name="mechanism_gnn"
        )
        md = ob.severe_subset_markdown([res])
        self.assertIn("mechanism_gnn", md)
        self.assertIn("MAJOR recall", md)


class TestOgblDdi(unittest.TestCase):
    def test_hits_at_k_hand_computed(self):
        pos = [0.9, 0.8, 0.2]
        neg = [0.5, 0.4, 0.3, 0.1]
        # k=2 -> threshold is 2nd highest negative = 0.4; pos above: 0.9, 0.8
        self.assertAlmostEqual(ob.hits_at_k(pos, neg, k=2), 2 / 3)
        # k=4 -> threshold 0.1; 0.2 also clears
        self.assertAlmostEqual(ob.hits_at_k(pos, neg, k=4), 1.0)

    def test_hits_at_k_bounds(self):
        self.assertAlmostEqual(ob.hits_at_k([1.0] * 5, [0.0] * 50, k=20), 1.0)
        self.assertAlmostEqual(ob.hits_at_k([0.0] * 5, [1.0] * 50, k=20), 0.0)

    def test_hits_at_k_more_k_than_negatives(self):
        self.assertAlmostEqual(ob.hits_at_k([0.9], [0.5, 0.4], k=20), 1.0)

    def test_mrr_range(self):
        mrr = ob.mean_reciprocal_rank([0.9, 0.1], [0.5, 0.4, 0.3])
        self.assertGreater(mrr, 0.0)
        self.assertLessEqual(mrr, 1.0)

    def test_report_shape_and_markdown(self):
        rng = np.random.default_rng(0)
        pos = rng.normal(1.0, 0.5, 200)
        neg = rng.normal(0.0, 0.5, 2000)
        rep = ob.evaluate_link_prediction(pos, neg, model_name="mechanism_gnn")
        self.assertEqual(rep["leaderboard_metric"], "hits@20")
        self.assertIn("hits@20", rep)
        md = ob.link_prediction_to_markdown([rep])
        self.assertIn("mechanism_gnn", md)
        self.assertIn("HITS@20", md)

    def test_environment_check_never_raises(self):
        rep = ob.check_environment()
        self.assertIn("numpy", rep.available)
        self.assertIsInstance(str(rep), str)

    def test_loader_fails_loudly_without_ogb(self):
        if ob.ogb_available():
            self.skipTest("ogb installed; nothing to assert about the fallback")
        with self.assertRaises(RuntimeError):
            ob.load_ogbl_ddi(download=False)

    def test_bad_inputs_rejected(self):
        with self.assertRaises(ValueError):
            ob.hits_at_k([], [1.0])
        with self.assertRaises(ValueError):
            ob.hits_at_k([1.0], [0.1], k=0)
        with self.assertRaises(ValueError):
            ob.hits_at_k([np.nan], [0.1])


# --------------------------------------------------------------------------- #
# Attention extraction + attribution
# --------------------------------------------------------------------------- #


class TestAttention(unittest.TestCase):
    def setUp(self):
        self.data = make_dummy_graph(n_drugs=80, n_edges=500, seed=6)

    def test_capture_collects_every_attention_layer(self):
        e = self.data.n_edges
        rng = np.random.default_rng(0)
        alphas = [rng.random((e, 4)), rng.random((e, 4))]
        model = FakeAttentionModel(self.data.edge_index, alphas)
        with av.AttentionCapture(model) as cap:
            model(np.zeros(1))
        self.assertEqual(len(cap), 2)
        merged = cap.merged()
        self.assertEqual(merged.shape, (e,))
        mat, names = cap.layer_matrix()
        self.assertEqual(mat.shape, (2, e))
        self.assertEqual(names, ["gat1", "gat2"])

    def test_capture_hooks_removed_on_exit(self):
        alphas = [np.random.random((self.data.n_edges, 2))]
        model = FakeAttentionModel(self.data.edge_index, alphas)
        with av.AttentionCapture(model) as cap:
            model(np.zeros(1))
        n_after_first = len(cap)
        model(np.zeros(1))  # outside the context: must not record again
        self.assertEqual(len(cap), n_after_first)

    def test_capture_include_filter(self):
        alphas = [np.random.random((self.data.n_edges, 2))] * 2
        model = FakeAttentionModel(self.data.edge_index, alphas)
        with av.AttentionCapture(model, include=("gat2",)) as cap:
            model(np.zeros(1))
        self.assertEqual(len(cap), 1)
        self.assertEqual(cap.records[0].module_name, "gat2")

    def test_merged_raises_when_nothing_captured(self):
        model = FakeAttentionModel(self.data.edge_index, [])
        with av.AttentionCapture(model) as cap:
            model(np.zeros(1))
        with self.assertRaises(RuntimeError):
            cap.merged()

    def test_head_aggregation(self):
        a = np.array([[0.0, 1.0], [2.0, 4.0]])
        np.testing.assert_allclose(av.aggregate_heads(a, "mean"), [0.5, 3.0])
        np.testing.assert_allclose(av.aggregate_heads(a, "max"), [1.0, 4.0])
        np.testing.assert_allclose(av.aggregate_heads(a, "sum"), [1.0, 6.0])
        with self.assertRaises(ValueError):
            av.aggregate_heads(a, "median")

    def test_normalize_attention_handles_constant_input(self):
        np.testing.assert_allclose(av.normalize_attention([0.3, 0.3, 0.3]), [0, 0, 0])
        out = av.normalize_attention([1.0, 2.0, 3.0])
        self.assertAlmostEqual(out.min(), 0.0)
        self.assertAlmostEqual(out.max(), 1.0)

    def test_top_k_edges_sorted_and_named(self):
        rows = av.top_k_attended_edges(
            self.data.attention,
            self.data.edge_index,
            k=10,
            drug_names=self.data.drug_names,
            labels=self.data.labels,
        )
        self.assertEqual(len(rows), 10)
        vals = [r["attention"] for r in rows]
        self.assertEqual(vals, sorted(vals, reverse=True))
        self.assertTrue(rows[0]["drug_a"].startswith("DRUG_"))
        self.assertIn(rows[0]["severity"], {"MINOR", "MODERATE", "MAJOR"})
        md = av.top_edges_to_markdown(rows)
        self.assertIn("| Rank |", md)

    def test_mechanism_attribution_sums_to_one(self):
        attr = av.mechanism_attribution(self.data.attention, self.data.mechanism_matrix)
        self.assertEqual(set(attr), set(av.MECHANISM_TAGS))
        self.assertAlmostEqual(sum(attr.values()), 1.0, places=6)

    def test_mechanism_attribution_detects_planted_signal(self):
        # all attention on edges flagged CYP450 only
        e = self.data.n_edges
        mech = np.zeros((e, 4))
        mech[:, 0] = 1.0
        att = np.ones(e)
        attr = av.mechanism_attribution(att, mech)
        self.assertAlmostEqual(attr["CYP450_inhibition"], 1.0)
        self.assertAlmostEqual(attr["QT_prolongation"], 0.0)

    def test_attribution_shape_validation(self):
        with self.assertRaises(ValueError):
            av.mechanism_attribution(np.ones(10), np.zeros((9, 4)))
        with self.assertRaises(ValueError):
            av.mechanism_attribution(np.ones(10), np.zeros((10, 3)))

    def test_attribution_by_class_covers_all_classes(self):
        by_class = av.mechanism_attribution_by_class(
            self.data.attention, self.data.mechanism_matrix, self.data.labels
        )
        self.assertEqual(set(by_class), {"MINOR", "MODERATE", "MAJOR"})
        for row in by_class.values():
            self.assertAlmostEqual(sum(row.values()), 1.0, places=6)

    def test_summary_is_json_serializable(self):
        s = av.attention_summary(
            self.data.attention, self.data.mechanism_matrix, self.data.labels
        )
        json.dumps(s)
        self.assertGreater(s["gini"], 0.0)
        self.assertEqual(s["n_edges"], self.data.n_edges)


class TestFigures(unittest.TestCase):
    def test_all_figures_render_to_disk(self):
        data = make_dummy_graph(n_drugs=60, n_edges=400, seed=12)
        with tempfile.TemporaryDirectory() as d:
            report = av.save_figures(
                data.attention,
                data.edge_index,
                mechanism_matrix=data.mechanism_matrix,
                labels=data.labels,
                outdir=d,
                model_name="mechanism_gnn",
                drug_names=data.drug_names,
                focus_drug=int(data.edge_index[0, 0]),
                top_k=12,
            )
            self.assertGreaterEqual(len(report.figures), 4)
            for fig in report.figures:
                self.assertTrue(Path(fig).exists(), f"missing {fig}")
                self.assertGreater(Path(fig).stat().st_size, 1000)
            self.assertTrue(report.summary_path.exists())
            json.loads(report.summary_path.read_text())

    def test_figures_degrade_without_mechanism_features(self):
        data = make_dummy_graph(n_drugs=40, n_edges=200, seed=13)
        with tempfile.TemporaryDirectory() as d:
            report = av.save_figures(
                data.attention, data.edge_index, outdir=d, model_name="baseline"
            )
            self.assertEqual(len(report.figures), 1)  # top-edges only, no crash

    def test_ego_plot_rejects_isolated_drug(self):
        data = make_dummy_graph(n_drugs=40, n_edges=200, seed=14)
        missing = data.n_drugs + 50
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ValueError):
                av.plot_pair_subgraph(
                    data.attention, data.edge_index, missing, Path(d) / "x.png"
                )


# --------------------------------------------------------------------------- #
# The end-to-end run the Day-2 deliverable is actually about
# --------------------------------------------------------------------------- #


class TestEndToEnd(unittest.TestCase):
    def test_full_harness_on_dummy_predictions(self):
        data = make_dummy_graph(n_drugs=200, n_edges=2000, seed=21)

        splits = make_splits(
            data.edge_index, data.labels, seed=42, severe_benchmark_frac=0.15
        )
        self.assertEqual(
            verify_splits(
                splits, labels=data.labels, edge_index=data.edge_index, strict=False
            ),
            [],
        )

        results = []
        configs = [
            ("baseline_gnn", 0.55, 0.0),
            ("ordinal_gnn", 0.70, 0.1),
            ("mechanism_gnn", 0.82, 0.2),
        ]
        for name, acc, bias in configs:
            y_pred, proba = make_dummy_predictions(
                data.labels, accuracy=acc, major_bias=bias, seed=hash(name) % 1000
            )
            for split_name in ("test", "severe_benchmark"):
                idx = splits[split_name]
                if idx.size == 0:
                    continue
                res = evaluate(
                    data.labels[idx],
                    y_pred[idx],
                    y_proba=proba[idx],
                    model_name=name,
                    split=split_name,
                )
                results.append(res)
                self.assertEqual(res.n_samples, idx.size)
                self.assertTrue(0.0 <= res.macro_f1 <= 1.0)
                self.assertTrue(0.0 <= res.mae <= 2.0)
                self.assertIn("MAJOR", res.per_class)

        # the stronger model should win on the test split
        test_res = {r.model_name: r for r in results if r.split == "test"}
        self.assertGreater(
            test_res["mechanism_gnn"].macro_f1, test_res["baseline_gnn"].macro_f1
        )
        self.assertLess(test_res["mechanism_gnn"].mae, test_res["baseline_gnn"].mae)

        # severe-benchmark rows exist and are MAJOR-only
        severe = [r for r in results if r.split == "severe_benchmark"]
        self.assertTrue(severe)
        for r in severe:
            self.assertEqual(r.support["MINOR"], 0)
            self.assertEqual(r.support["MODERATE"], 0)
            self.assertGreater(r.support["MAJOR"], 0)

        # explainability on the best model
        rng = np.random.default_rng(1)
        alphas = [rng.random((data.n_edges, 4)), rng.random((data.n_edges, 4))]
        model = FakeAttentionModel(data.edge_index, alphas)
        with av.AttentionCapture(model) as cap:
            model(np.zeros(1))
        att = cap.merged()
        self.assertEqual(att.shape, (data.n_edges,))

        with tempfile.TemporaryDirectory() as d:
            out = Path(d)
            md = metrics_to_markdown(test_res["mechanism_gnn"])
            comp = comparison_to_markdown(results, title="Dummy-data comparison")
            (out / "results_comparison.md").write_text(md + "\n" + comp)
            report = av.save_figures(
                data.attention,
                data.edge_index,
                mechanism_matrix=data.mechanism_matrix,
                labels=data.labels,
                outdir=out / "figs",
                model_name="mechanism_gnn",
                drug_names=data.drug_names,
                focus_drug=int(data.edge_index[0, 0]),
            )
            save_splits(splits, out / "split.json")

            self.assertTrue((out / "results_comparison.md").exists())
            self.assertIn("mechanism_gnn", comp)
            self.assertIn("Macro F1", comp)
            self.assertTrue(all(Path(f).exists() for f in report.figures))
            self.assertEqual(
                set(load_splits(out / "split.json").names), set(splits.names)
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
