import unittest
from contextlib import ExitStack
from unittest.mock import patch
from pathlib import Path
from tempfile import TemporaryDirectory
from functools import partialmethod
import json
import numpy as np
from scripts import personalize
from scripts.personalize import _sample_k, _stratified_split
from src import synthetic as synthetic_data


class PersonalizeTests(unittest.TestCase):

    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.tmp_path = Path(self.stack.enter_context(TemporaryDirectory()))

    def test_split_is_disjoint_and_covers_all(self):
        rng = np.random.RandomState(0)
        y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        ev, pool = _stratified_split(y, 0.5, rng)
        self.assertTrue(set(ev).isdisjoint(pool))
        self.assertEqual(sorted([*ev, *pool]), list(range(len(y))))

    def test_split_keeps_both_classes_each_side(self):
        rng = np.random.RandomState(1)
        y = np.array([0] * 6 + [1] * 6)
        ev, pool = _stratified_split(y, 0.5, rng)
        self.assertTrue({0, 1} <= set(y[ev]) and {0, 1} <= set(y[pool]))

    def test_sample_k_balances_classes(self):
        rng = np.random.RandomState(2)
        y_pool = np.array([0] * 10 + [1] * 10)
        pick = _sample_k(y_pool, 6, rng)
        self.assertLessEqual({0, 1}, set(y_pool[pick]))
        self.assertLessEqual(len(pick), 6)
        counts = np.bincount(y_pool[pick], minlength=2)
        self.assertTrue(counts[0] == counts[1] == 3)

    def test_sampling_empty_pool_and_budget_one_does_not_exceed_budget(self):
        empty = _sample_k(np.array([], dtype=int), 5, np.random.RandomState(0))
        self.assertTrue(empty.dtype.kind == "i" and (not len(empty)))
        pick = _sample_k(np.array([0, 0, 1, 1]), 1, np.random.RandomState(0))
        self.assertEqual(len(pick), 1)

    def test_sampling_keeps_classes_balanced_when_one_has_fewer_windows(self):
        for counts, budget, expected in (
            ([12, 2], 10, 2),
            ([12, 1], 5, 1),
            ([2, 12, 4], 12, 2),
            ([10, 10], 5, 2),
        ):
            with self.subTest(counts=counts, budget=budget):
                labels = np.repeat(np.arange(len(counts)), counts)
                original = labels.copy()
                first = _sample_k(labels, budget, np.random.RandomState(0))
                second = _sample_k(labels, budget, np.random.RandomState(0))
                selected = np.bincount(labels[first], minlength=len(counts))
                np.testing.assert_array_equal(selected, [expected] * len(counts))
                np.testing.assert_array_equal(first, second)
                np.testing.assert_array_equal(labels, original)
                self.assertEqual(len(np.unique(first)), len(first))
                self.assertLessEqual(len(first), budget)

    def test_sampling_rejects_invalid_budget(self):
        for budget in [0, -1, 1.5, True]:
            with self.subTest(budget=budget):
                with self.assertRaisesRegex(ValueError, "positive integers"):
                    _sample_k(np.array([0, 1]), budget, np.random.RandomState(0))

    def test_tiny_personalization_pool_falls_back_without_crashing(self):
        from sklearn.dummy import DummyClassifier
        from sklearn.pipeline import Pipeline

        self.stack.enter_context(
            patch.object(
                personalize,
                "build_pipeline",
                lambda model: Pipeline([("clf", DummyClassifier())]),
            )
        )
        y = np.tile([0, 0, 1, 1], 3)
        groups = np.repeat(["S0", "S1", "S2"], 4)
        result = personalize.compute(np.zeros((12, 1)), y, groups, k_values=[5])
        self.assertEqual(result["n_subjects"], 3)
        self.assertEqual(result["fewshot"]["5"], result["uncalibrated"])

    def test_personalization_rejects_no_eligible_targets(self):
        from sklearn.dummy import DummyClassifier
        from sklearn.pipeline import Pipeline

        self.stack.enter_context(
            patch.object(
                personalize,
                "build_pipeline",
                lambda model: Pipeline([("clf", DummyClassifier())]),
            )
        )
        with self.assertRaisesRegex(ValueError, "target subject"):
            personalize.compute(
                np.zeros((6, 1)),
                np.zeros(6, dtype=int),
                np.repeat(["S0", "S1", "S2"], 2),
            )

    def _case_personalization_output_paths_protect_benchmark(self, synthetic):
        tmp_path = self.tmp_path
        results_dir = tmp_path / "outputs" / "results"
        figures_dir = tmp_path / "outputs" / "generated" / "figures"
        demo_dir = tmp_path / "outputs" / "generated" / "demo"
        results_dir.mkdir(parents=True)
        figures_dir.mkdir(parents=True)
        real_result = results_dir / "personalization.json"
        real_figure = figures_dir / "personalization.png"
        real_result.write_bytes(b"committed benchmark snapshot")
        real_figure.write_bytes(b"committed benchmark plot")
        self.stack.enter_context(patch.object(personalize, "RESULTS_DIR", results_dir))
        self.stack.enter_context(patch.object(personalize, "FIGURES_DIR", figures_dir))
        self.stack.enter_context(patch.object(personalize, "DEMO_DIR", demo_dir))

        def cached_features():
            self.assertFalse(synthetic, "Synthetic demo read the real dataset cache")
            return (None, None)

        def generated_features(**kwargs):
            self.assertTrue(synthetic)
            self.assertIs(kwargs["cache"], False)
            return (None, None, None)

        self.stack.enter_context(
            patch.object(personalize, "load_cached", cached_features)
        )
        self.stack.enter_context(
            patch.object(synthetic_data, "features", generated_features)
        )
        self.stack.enter_context(
            patch.object(personalize, "prepare_task", lambda *args: (None,) * 5)
        )
        metrics = {
            "k_values": [5],
            "uncalibrated": {"ece": 0.1, "brier": 0.1},
            "global": {"ece": 0.08, "brier": 0.08},
            "fewshot": {"5": {"ece": 0.05, "brier": 0.05}},
        }
        self.stack.enter_context(
            patch.object(personalize, "compute", lambda *args, **kwargs: metrics)
        )
        self.stack.enter_context(
            patch.object(
                personalize,
                "analysis_provenance",
                lambda *args: {"git_sha": "test", "generated_at": "test"},
            )
        )
        self.stack.enter_context(
            patch.object(
                personalize, "_plot", lambda out, path: path.write_bytes(b"new plot")
            )
        )
        personalize.run(synthetic=synthetic)
        target_results = demo_dir / "results" if synthetic else results_dir
        target_figures = demo_dir / "figures" if synthetic else figures_dir
        self.assertEqual(
            json.loads((target_results / "personalization.json").read_text()), metrics
        )
        self.assertEqual(
            (target_figures / "personalization.png").read_bytes(), b"new plot"
        )
        if synthetic:
            self.assertEqual(real_result.read_bytes(), b"committed benchmark snapshot")
            self.assertEqual(real_figure.read_bytes(), b"committed benchmark plot")
        else:
            self.assertFalse(demo_dir.exists())

    def test_personalization_serialization_failure_preserves_previous_snapshot(self):
        tmp_path = self.tmp_path
        path = tmp_path / "personalization.json"
        path.write_bytes(b"previous usable snapshot")
        self.stack.enter_context(patch.object(personalize, "RESULTS_DIR", tmp_path))
        self.stack.enter_context(
            patch.object(personalize, "FIGURES_DIR", tmp_path / "figures")
        )
        self.stack.enter_context(
            patch.object(personalize, "load_cached", lambda: (None, None))
        )
        self.stack.enter_context(
            patch.object(personalize, "prepare_task", lambda *args: (None,) * 5)
        )
        self.stack.enter_context(
            patch.object(
                personalize, "compute", lambda *args, **kwargs: {"ece": float("nan")}
            )
        )
        self.stack.enter_context(
            patch.object(
                personalize, "_plot", lambda *args: self.fail("Plotted invalid metrics")
            )
        )
        with self.assertRaisesRegex(ValueError, "JSON compliant"):
            personalize.run()
        self.assertEqual(path.read_bytes(), b"previous usable snapshot")


for index, synthetic in enumerate([False, True]):
    setattr(
        PersonalizeTests,
        f"test_personalization_output_paths_protect_benchmark_{index}",
        partialmethod(
            PersonalizeTests._case_personalization_output_paths_protect_benchmark,
            synthetic=synthetic,
        ),
    )
