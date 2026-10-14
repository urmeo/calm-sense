import unittest
from unittest.mock import patch
import numpy as np
from src.calibration import (
    brier_score,
    expected_calibration_error,
    maximum_calibration_error,
    net_benefit,
    normalize_binary_calibration,
    reliability_curve,
)


class CalibrationTests(unittest.TestCase):

    def test_decision_curve_rejects_divergent_label_order(self):
        from scripts import calibration

        y = np.tile([0, 0, 1, 1], 3)
        groups = np.repeat(["S0", "S1", "S2"], 4)
        X = np.zeros((len(y), 1))

        def probabilities(factory, values, labels, subjects):
            return labels, np.tile([0.4, 0.6], (len(labels), 1)), subjects

        def recalibrated(factory, values, labels, subjects, method):
            return labels[::-1], np.tile([0.4, 0.6], (len(labels), 1))

        with (
            patch.object(calibration, "loso_proba", probabilities),
            patch.object(calibration, "within_subject_proba", probabilities),
            patch.object(calibration, "loso_recalibrated_proba", recalibrated),
        ):
            with self.assertRaisesRegex(ValueError, "label order diverged"):
                calibration.compute(X, y, groups)

    def test_binary_brier_is_independent_of_probability_representation(self):
        y = np.array([0, 1, 1, 0])
        p = np.array([0.1, 0.9, 0.8, 0.3])
        np.testing.assert_allclose(brier_score(y, p), 0.0375, rtol=1e-06, atol=1e-12)
        np.testing.assert_allclose(
            brier_score(y, np.column_stack([1 - p, p])), 0.0375, rtol=1e-06, atol=1e-12
        )

    def test_multiclass_brier_remains_summed(self):
        y = np.array([0, 2])
        p = np.array([[0.8, 0.1, 0.1], [0.1, 0.1, 0.8]])
        np.testing.assert_allclose(brier_score(y, p), 0.06, rtol=1e-06, atol=1e-12)

    def test_legacy_calibration_conversion_preserves_source_and_is_idempotent(self):
        original = {
            "loso": {"ece": 0.07, "brier": 0.136},
            "gap_significance": {
                "mean_brier_gap": 0.06,
                "ci95": [0.02, 0.1],
                "wilcoxon_p": 0.001,
                "per_subject": {"S2": {"loso": 0.2, "within": 0.1}},
            },
            "provenance": {"git_sha": "historical", "generated_at": "original"},
        }
        out = normalize_binary_calibration(original)
        self.assertEqual(out["loso"], {"ece": 0.07, "brier": 0.068})
        self.assertEqual(out["gap_significance"]["mean_brier_gap"], 0.03)
        self.assertEqual(out["gap_significance"]["ci95"], [0.01, 0.05])
        self.assertEqual(
            out["gap_significance"]["per_subject"]["S2"], {"loso": 0.1, "within": 0.05}
        )
        self.assertEqual(
            out["gap_significance"]["wilcoxon_p"],
            original["gap_significance"]["wilcoxon_p"],
        )
        self.assertEqual(out["provenance"], original["provenance"])
        self.assertEqual(original["loso"]["brier"], 0.136)
        self.assertEqual(normalize_binary_calibration(out), out)

    def test_unknown_brier_scale_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unknown binary Brier definition"):
            normalize_binary_calibration({"brier_definition": "unknown"})

    def test_net_benefit_rejects_one_for_every_prediction_pattern(self):
        for labels, probabilities in (
            ([1, 0], [1.0, 0.0]),
            ([1, 0], [0.0, 1.0]),
            ([1, 0], [0.8, 0.2]),
            ([0, 0], [1.0, 1.0]),
            ([1, 1], [1.0, 1.0]),
        ):
            for thresholds in ([1.0], [0.5, 1.0]):
                with self.subTest(labels=labels, probabilities=probabilities):
                    with self.assertRaisesRegex(ValueError, r"\[0, 1\)"):
                        net_benefit(labels, probabilities, thresholds)

    def test_net_benefit_at_zero_is_prevalence_including_zero_probabilities(self):
        np.testing.assert_array_equal(
            net_benefit([1, 0, 1, 0], [0.0, 0.0, 1.0, 1.0], [0.0]), [0.5]
        )

    def test_net_benefit_just_below_one_keeps_the_false_positive_cost(self):
        threshold = np.nextafter(1.0, 0.0)
        expected = 0.5 - 0.5 * threshold / (1.0 - threshold)
        actual = net_benefit([1, 0], [1.0, 1.0], [threshold])
        np.testing.assert_array_equal(actual, [expected])
        self.assertTrue(np.isfinite(actual).all())
        self.assertLess(actual[0], 0)
        np.testing.assert_array_equal(
            net_benefit([1, 0], [1.0, 0.0], [threshold]), [0.5]
        )

    def test_net_benefit_uses_inclusive_thresholds_and_preserves_input_order(self):
        labels = np.array([1, 0, 1, 0])
        probabilities = np.array([0.8, 0.8, 0.5, 0.2])
        thresholds = np.array([0.8, 0.5, 0.0, 0.5])
        originals = [values.copy() for values in (labels, probabilities, thresholds)]
        np.testing.assert_allclose(
            net_benefit(labels, probabilities, thresholds),
            [-0.75, 0.25, 0.5, 0.25],
        )
        for values, original in zip((labels, probabilities, thresholds), originals):
            np.testing.assert_array_equal(values, original)

    def test_net_benefit_accepts_an_empty_threshold_vector(self):
        result = net_benefit([0, 1], [0.1, 0.9], [])
        self.assertEqual(result.shape, (0,))
        self.assertEqual(result.dtype, np.dtype(float))

    def test_perfect_calibration_has_zero_ece(self):
        y = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0])
        proba = np.full(10, 0.5)
        self.assertLess(expected_calibration_error(y, proba, n_bins=5), 1e-09)

    def test_overconfident_model_has_large_ece(self):
        y = np.array([1, 0] * 50)
        proba = np.full(100, 0.99)
        ece = expected_calibration_error(y, proba, n_bins=10)
        self.assertLess(abs(ece - 0.49), 0.05)

    def test_brier_rewards_confident_correct(self):
        y = np.array([1, 0])
        confident = brier_score(y, np.array([0.9, 0.1]))
        unsure = brier_score(y, np.array([0.5, 0.5]))
        self.assertLess(confident, unsure)

    def test_brier_perfect_is_zero(self):
        y = np.array([1, 0, 1])
        self.assertLess(brier_score(y, np.array([1.0, 0.0, 1.0])), 1e-12)

    def test_reliability_bins_sum_to_n(self):
        rng = np.random.RandomState(0)
        y = rng.randint(0, 2, 200)
        proba = rng.rand(200)
        rows = reliability_curve(y, proba, n_bins=10)
        self.assertEqual(sum((r["count"] for r in rows)), 200)

    def test_mce_at_least_ece(self):
        rng = np.random.RandomState(1)
        y = rng.randint(0, 2, 300)
        proba = rng.rand(300)
        self.assertGreaterEqual(
            maximum_calibration_error(y, proba),
            expected_calibration_error(y, proba) - 1e-09,
        )

    def test_net_benefit_treat_none_baseline(self):
        y = np.array([1, 0, 1, 0])
        nb = net_benefit(y, np.zeros(4), np.array([0.2, 0.5]))
        self.assertTrue(np.allclose(nb, 0.0))

    def test_gap_significance_detects_consistent_gap(self):
        from scripts.calibration import gap_significance

        loso = {f"S{i}": 0.2 for i in range(12)}
        within = {f"S{i}": 0.1 for i in range(12)}
        sig = gap_significance(loso, within)
        self.assertEqual(sig["n_subjects"], 12)
        self.assertLess(abs(sig["mean_brier_gap"] - 0.1), 1e-09)
        self.assertGreater(sig["ci95"][0], 0)
        self.assertLess(sig["wilcoxon_p"], 0.05)

    def test_calibration_rejects_invalid_probability_contract(self):
        for y, proba in [
            ([], []),
            ([0, 1], [0.2]),
            ([0], [np.nan]),
            ([1], [1.1]),
            ([2], [0.3]),
            ([0.5], [0.5]),
            ([0], [[0.2, 0.2]]),
            ([0], [[1.0]]),
            ([[0]], [0.1]),
        ]:
            with self.subTest(y=y, proba=proba):
                for metric in (
                    brier_score,
                    expected_calibration_error,
                    reliability_curve,
                ):
                    with self.assertRaises(ValueError):
                        metric(y, proba)

    def test_calibration_requires_positive_integer_bin_count(self):
        for bins in [0, -1, 2.5, True]:
            with self.subTest(bins=bins):
                with self.assertRaisesRegex(ValueError, "n_bins"):
                    expected_calibration_error([0, 1], [0.2, 0.8], bins)

    def test_binary_reliability_ties_match_two_column_predictions(self):
        y = [0, 0, 1]
        p = np.array([0.5, 0.5, 0.5])
        self.assertEqual(
            reliability_curve(y, p), reliability_curve(y, np.column_stack([1 - p, p]))
        )

    def test_net_benefit_rejects_invalid_thresholds(self):
        for thresholds in [
            [-0.1],
            [1.0],
            [1.1],
            [np.nan],
            [np.inf],
            [-np.inf],
            0.5,
            [[0.5]],
        ]:
            with self.subTest(thresholds=thresholds):
                with self.assertRaisesRegex(ValueError, "Thresholds"):
                    net_benefit([0, 1], [0.1, 0.9], np.array(thresholds))

    def test_single_class_loso_probabilities_keep_baseline_stress_columns(self):
        from sklearn.dummy import DummyClassifier
        from sklearn.pipeline import Pipeline
        from scripts.calibration import loso_proba

        X = np.zeros((4, 1))
        y = np.array([0, 0, 1, 1])
        groups = np.array(["S0", "S0", "S1", "S1"])
        factory = lambda: Pipeline([("clf", DummyClassifier())])
        true, proba, subjects = loso_proba(factory, X, y, groups)
        np.testing.assert_array_equal(true, y)
        np.testing.assert_array_equal(subjects, groups)
        np.testing.assert_array_equal(proba, [[0, 1], [0, 1], [1, 0], [1, 0]])

    def test_sigmoid_single_class_calibrator_returns_class_constant(self):
        from scripts.calibration import _apply_calibrator, _fit_calibrator

        for label in [0, 1]:
            with self.subTest(label=label):
                model = _fit_calibrator(
                    np.array([0.2, 0.4, 0.8]), np.full(3, label), "sigmoid"
                )
                np.testing.assert_array_equal(
                    _apply_calibrator(model, np.array([0.1, 0.9]), "sigmoid"),
                    [label, label],
                )

    def test_gap_significance_rejects_missing_or_unpaired_subject_scores(self):
        from scripts.calibration import gap_significance

        with self.assertRaisesRegex(ValueError, "three subjects"):
            gap_significance({}, {})
        with self.assertRaisesRegex(ValueError, "three subjects"):
            gap_significance(
                {"S0": 0.1, "S1": 0.2, "S2": 0.3}, {"S0": 0.1, "S1": 0.2, "S3": 0.3}
            )

    def test_estimator_probability_contract_rejects_invalid_folds(self):
        from scripts.calibration import _pos_proba

        for classes, probabilities in [
            ([0], [[np.nan]]),
            ([0], [[0.5]]),
            ([0, 1], [[0.2, 0.9]]),
            ([0, 1], [[-0.1, 1.1]]),
            ([0, 1], [[0.2]]),
            ([0, 0], [[0.5, 0.5]]),
        ]:
            with self.subTest(classes=classes, probabilities=probabilities):

                class Estimator:
                    classes_ = classes

                    def predict_proba(self, X):
                        return probabilities

                with self.assertRaisesRegex(ValueError, "(probabilities|classes)"):
                    _pos_proba(Estimator(), np.zeros((1, 1)))

    def test_estimator_positive_probabilities_follow_reversed_class_order(self):
        from scripts.calibration import _pos_proba

        class Estimator:
            classes_ = np.array([1, 0])

            def predict_proba(self, X):
                return [[0.8, 0.2], [0.3, 0.7]]

        np.testing.assert_array_equal(
            _pos_proba(Estimator(), np.zeros((2, 1))), [0.8, 0.3]
        )
