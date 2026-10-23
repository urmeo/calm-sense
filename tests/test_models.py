from contextlib import contextmanager
from copy import deepcopy
from importlib import import_module
import json
import os
import random
from unittest.mock import Mock, patch
import weakref

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from scripts import ablation, calibration, cross_dataset, personalize, tuning
from scripts.personalize import _sample_k, _stratified_split
from scripts.run_experiment import (
    build_pipeline,
    cnn_loso,
    kfold_accuracy,
    loso_evaluate,
    nonoverlap_mask,
    prepare_task,
)
from src import synthetic
from src.calibration import (
    brier_score,
    expected_calibration_error,
    maximum_calibration_error,
    net_benefit,
    normalize_binary_calibration,
    reliability_curve,
    validate_bin_count,
)
from src.dataset import WindowedDataset
from src.dataset_wrist import WristDataset
from src.features.feature_pipeline import (
    FEATURE_SCHEMA_VERSION,
    FeatureExtractionPipeline,
)
from src.models.ml.classifiers import HYPERPARAMS, get_classifier
from src.synthetic import features
from src.utils import set_seed

import test_data as fixtures


@contextmanager
def preserve_random_state():
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    hash_seed = os.environ.get("PYTHONHASHSEED")
    try:
        torch = import_module("torch")
    except ImportError:
        torch = None
    if torch is not None:
        cpu_state = torch.get_rng_state()
        cuda_state = (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        )
        deterministic = torch.are_deterministic_algorithms_enabled()
        warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
        cudnn_deterministic = torch.backends.cudnn.deterministic
        cudnn_benchmark = torch.backends.cudnn.benchmark
    try:
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        if hash_seed is None:
            os.environ.pop("PYTHONHASHSEED", None)
        else:
            os.environ["PYTHONHASHSEED"] = hash_seed
        if torch is not None:
            torch.set_rng_state(cpu_state)
            if cuda_state is not None:
                torch.cuda.set_rng_state_all(cuda_state)
            torch.use_deterministic_algorithms(deterministic, warn_only=warn_only)
            torch.backends.cudnn.deterministic = cudnn_deterministic
            torch.backends.cudnn.benchmark = cudnn_benchmark


class RandomStateTestCase(fixtures.PatchedTestCase):
    def setUp(self):
        self.enterContext(preserve_random_state())

    def _torch(self):
        try:
            return import_module("torch")
        except ImportError:
            self.skipTest("torch unavailable")


class DeterminismTests(RandomStateTestCase):

    def test_set_seed_makes_rngs_reproducible(self):
        set_seed(123)
        first = (random.random(), np.random.rand(3).tolist())
        set_seed(123)
        second = (random.random(), np.random.rand(3).tolist())
        self.assertEqual(first, second)

    def test_set_seed_can_disable_a_previous_deterministic_setting(self):
        torch = self._torch()
        set_seed(7, deterministic=True)
        self.assertTrue(torch.are_deterministic_algorithms_enabled())
        self.assertTrue(torch.backends.cudnn.deterministic)
        self.assertFalse(torch.backends.cudnn.benchmark)
        set_seed(7, deterministic=False)
        self.assertFalse(torch.are_deterministic_algorithms_enabled())
        self.assertFalse(torch.backends.cudnn.deterministic)

    def test_random_state_is_restored_after_an_exception(self):
        python_state = random.getstate()
        numpy_state = np.random.get_state()
        hash_seed = os.environ.get("PYTHONHASHSEED")
        try:
            torch = import_module("torch")
        except ImportError:
            torch = None
        if torch is not None:
            cpu_state = torch.get_rng_state()
            cuda_state = (
                torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
            )
            backend = (
                torch.are_deterministic_algorithms_enabled(),
                torch.is_deterministic_algorithms_warn_only_enabled(),
                torch.backends.cudnn.deterministic,
                torch.backends.cudnn.benchmark,
            )
        with self.assertRaisesRegex(RuntimeError, "test failed"):
            with preserve_random_state():
                set_seed(91, deterministic=False)
                random.random()
                np.random.random(3)
                if torch is not None:
                    torch.rand(3)
                    for device in range(len(cuda_state)):
                        torch.rand(3, device=f"cuda:{device}")
                    torch.use_deterministic_algorithms(True, warn_only=False)
                    torch.backends.cudnn.deterministic = True
                    torch.backends.cudnn.benchmark = True
                raise RuntimeError("test failed")
        self.assertEqual(random.getstate(), python_state)
        for actual, expected in zip(np.random.get_state(), numpy_state):
            np.testing.assert_equal(actual, expected)
        self.assertEqual(os.environ.get("PYTHONHASHSEED"), hash_seed)
        if torch is not None:
            self.assertTrue(torch.equal(torch.get_rng_state(), cpu_state))
            current_cuda = torch.cuda.get_rng_state_all() if cuda_state else []
            self.assertEqual(len(current_cuda), len(cuda_state))
            for actual, expected in zip(current_cuda, cuda_state):
                self.assertTrue(torch.equal(actual, expected))
            self.assertEqual(
                (
                    torch.are_deterministic_algorithms_enabled(),
                    torch.is_deterministic_algorithms_warn_only_enabled(),
                    torch.backends.cudnn.deterministic,
                    torch.backends.cudnn.benchmark,
                ),
                backend,
            )

    def test_feature_extraction_is_deterministic(self):
        feature_df_a, _, _ = features(n_subjects=2, block_sec=80, seed=5)
        feature_df_b, _, _ = features(n_subjects=2, block_sec=80, seed=5)
        meta = ("subject_id", "window_id", "label", "label_name")
        cols = [c for c in feature_df_a.columns if c not in meta]
        a = feature_df_a[cols].to_numpy(dtype=float)
        b = feature_df_b[cols].to_numpy(dtype=float)
        self.assertEqual(a.shape, b.shape)
        both_nan = np.isnan(a) & np.isnan(b)
        self.assertTrue(
            (np.isclose(a, b, equal_nan=False) | both_nan).all(),
            "Feature extraction is not reproducible with a fixed seed",
        )

    def test_cnn_training_is_deterministic(self):
        self._torch()
        from src.models.dl.cnn_1d import CNN1DClassifier

        rng = np.random.RandomState(0)
        n, channels, length = (24, 3, 256)
        X = rng.randn(n, channels, length).astype("float32")
        y = np.array([0, 1] * (n // 2))

        def train_and_predict():
            set_seed(7)
            clf = CNN1DClassifier(
                in_channels=channels, max_epochs=2, batch_size=8, random_state=7
            )
            clf.fit(X, y)
            return clf.predict_proba(X)

        self.assertTrue(
            np.allclose(train_and_predict(), train_and_predict(), atol=1e-06),
            "CNN training is not reproducible with a fixed seed",
        )


class CalibrationTests(fixtures.PatchedTestCase):

    def test_decision_curve_rejects_divergent_label_order(self):
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
            self.assertRaisesRegex(ValueError, "label order diverged"),
        ):
            calibration.compute(X, y, groups)

    def test_binary_brier_is_independent_of_probability_representation(self):
        y = np.array([0, 1, 1, 0])
        p = np.array([0.1, 0.9, 0.8, 0.3])
        fixtures.assert_close(brier_score(y, p), 0.0375)
        fixtures.assert_close(brier_score(y, np.column_stack([1 - p, p])), 0.0375)

    def test_multiclass_brier_remains_summed(self):
        y = np.array([0, 2])
        p = np.array([[0.8, 0.1, 0.1], [0.1, 0.1, 0.8]])
        fixtures.assert_close(brier_score(y, p), 0.06)

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
                with (
                    self.subTest(labels=labels, probabilities=probabilities),
                    self.assertRaisesRegex(ValueError, "\\[0, 1\\)"),
                ):
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
                    self.assertRaises(ValueError, metric, y, proba)

    def test_calibration_requires_positive_integer_bin_count(self):
        for bins in [0, -1, 2.5, True]:
            with (
                self.subTest(bins=bins),
                self.assertRaisesRegex(ValueError, "n_bins"),
            ):
                expected_calibration_error([0, 1], [0.2, 0.8], bins)

    def test_bin_count_validation_accepts_positive_integer_types(self):
        for bins in [1, 15, np.int64(3)]:
            with self.subTest(bins=bins):
                self.assertIsNone(validate_bin_count(bins))

    def test_invalid_bins_fail_before_calibration_training(self):
        for bins in [0, -1, 2.5, True, np.nan, np.inf, "15", None]:
            with self.subTest(bins=bins):
                self.assertRaisesRegex(ValueError, "n_bins", validate_bin_count, bins)
                with patch.object(calibration, "loso_proba") as predictions:
                    with self.assertRaisesRegex(ValueError, "n_bins"):
                        calibration.compute(None, None, None, n_bins=bins)
                    predictions.assert_not_called()

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
            with (
                self.subTest(thresholds=thresholds),
                self.assertRaisesRegex(ValueError, "Thresholds"),
            ):
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

        self.assertRaisesRegex(ValueError, "three subjects", gap_significance, {}, {})
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


class ClassifierTests(fixtures.PatchedTestCase):
    def test_lightgbm_default_preserves_fits_with_inactive_subsampling(self):
        for classes in (2, 3):
            with self.subTest(classes=classes):
                X, y = make_classification(
                    n_samples=360,
                    n_features=10,
                    n_informative=6,
                    n_redundant=2,
                    n_classes=classes,
                    n_clusters_per_class=1,
                    random_state=17,
                )
                X = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
                X_train, X_test, y_train, _ = train_test_split(
                    X, y, test_size=0.25, stratify=y, random_state=29
                )
                previous = get_classifier("lgbm", subsample=0.8, n_jobs=1)
                cleaned = get_classifier("lgbm", n_jobs=1)
                self.assertEqual(previous.get_params()["subsample_freq"], 0)
                self.assertEqual(cleaned.get_params()["subsample_freq"], 0)
                self.assertEqual(cleaned.get_params()["subsample"], 1.0)
                previous.fit(X_train, y_train)
                cleaned.fit(X_train, y_train)
                self.assertEqual(
                    previous.booster_.dump_model()["tree_info"],
                    cleaned.booster_.dump_model()["tree_info"],
                )
                np.testing.assert_array_equal(
                    previous.predict_proba(X_test), cleaned.predict_proba(X_test)
                )
                np.testing.assert_array_equal(
                    previous.predict(X_test), cleaned.predict(X_test)
                )

    def test_sampling_overrides_and_xgboost_default_are_preserved(self):
        self.assertEqual(get_classifier("xgb").get_params()["subsample"], 0.8)
        overridden = get_classifier(
            "lightgbm", subsample=0.6, subsample_freq=2, random_state=7, n_jobs=1
        )
        parameters = overridden.get_params()
        self.assertEqual(parameters["subsample"], 0.6)
        self.assertEqual(parameters["subsample_freq"], 2)
        self.assertEqual(parameters["random_state"], 7)
        self.assertEqual(parameters["n_jobs"], 1)


class CnnTests(RandomStateTestCase):

    def test_cnn_generalizes_to_held_out_split(self):
        self._torch()
        from src.models.dl.cnn_1d import CNN1DClassifier

        rng = np.random.RandomState(0)
        n, c, length = (160, 5, 256)
        X = rng.randn(n, c, length).astype("float32")
        y = (rng.randn(n) > 0).astype(int)
        X[y == 1, 0, :] += 2.0
        X_tr, y_tr = (X[:120], y[:120])
        X_te, y_te = (X[120:], y[120:])
        model = CNN1DClassifier(in_channels=c, max_epochs=25, random_state=0)
        model.fit(X_tr, y_tr)
        proba = model.predict_proba(X_te)
        self.assertEqual(proba.shape, (len(y_te), 2))
        self.assertTrue(np.allclose(proba.sum(axis=1), 1.0, atol=0.0001))
        self.assertGreater((model.predict(X_te) == y_te).mean(), 0.7)

    def test_cnn_subject_holdout_normalizes_and_weights_training_only(self):
        self._torch()
        from src.models.dl import cnn_1d

        groups = np.repeat(["S0", "S1", "S2", "S3"], 6)
        y = np.tile([0, 0, 0, 0, 1, 1], 4)
        rng = np.random.RandomState(1)
        X = rng.randn(24, 1, 128).astype("float32")
        X += np.repeat([0, 10, 100, 1000], 6)[:, None, None]
        model = cnn_1d.CNN1DClassifier(
            in_channels=1, max_epochs=1, batch_size=8, random_state=3, device="cpu"
        )
        train, validation = model._validation_split(y, groups)
        self.assertTrue(set(groups[train]).isdisjoint(groups[validation]))
        np.testing.assert_array_equal(model._validation_split(y, groups)[0], train)
        seen_weights = []
        original = cnn_1d.compute_class_weight

        def weights(class_weight, *, classes, y):
            seen_weights.append(y.copy())
            return original(class_weight, classes=classes, y=y)

        self.patch_object(cnn_1d, "compute_class_weight", weights)
        model.fit(X, y, groups=groups)
        np.testing.assert_allclose(
            model._mean, X[train].mean(axis=(0, 2), keepdims=True)
        )
        np.testing.assert_allclose(
            model._std, X[train].std(axis=(0, 2), keepdims=True) + 1e-08
        )
        self.assertFalse(np.allclose(model._mean, X.mean(axis=(0, 2), keepdims=True)))
        np.testing.assert_array_equal(seen_weights[0], y[train])
        self.assertEqual(
            model.predict_proba(np.empty((0, 1, 128), dtype="float32")).shape, (0, 2)
        )

    def test_cnn_rejects_unfitted_and_nonfinite_predictions(self):
        self._torch()
        from sklearn.exceptions import NotFittedError
        from src.models.dl.cnn_1d import CNN1DClassifier

        model = CNN1DClassifier(in_channels=1, max_epochs=1)
        self.assertRaises(NotFittedError, model.predict_proba, np.zeros((2, 1, 128)))
        with self.assertRaisesRegex(ValueError, "finite"):
            model.fit(np.full((8, 1, 128), np.nan), np.tile([0, 1], 4))

    def test_failed_cnn_fit_cannot_predict_partial_or_stale_weights(self):
        torch = self._torch()
        from sklearn.exceptions import NotFittedError
        from src.models.dl import cnn_1d

        class SmallNetwork(torch.nn.Module):
            def __init__(self, in_channels, n_classes):
                super().__init__()
                self.linear = torch.nn.Linear(in_channels, n_classes)

            def forward(self, X):
                return self.linear(X.mean(dim=2))

        rng = np.random.RandomState(3)
        X = rng.randn(8, 1, 61).astype("float32")
        y = np.tile([0, 1], 4)
        self.patch_object(cnn_1d, "_Net", SmallNetwork)
        model = cnn_1d.CNN1DClassifier(
            in_channels=1, max_epochs=1, val_fraction=0.25, device="cpu"
        )
        for previously_fitted in (False, True):
            with self.subTest(previously_fitted=previously_fitted):
                if previously_fitted:
                    model.fit(X, y)
                    self.assertEqual(model.predict_proba(X).shape, (8, 2))
                with (
                    patch.object(
                        torch.optim.AdamW,
                        "step",
                        side_effect=RuntimeError("fit failed"),
                    ),
                    self.assertRaisesRegex(RuntimeError, "fit failed"),
                ):
                    model.fit(X, y + 2)
                self.assertRaises(NotFittedError, model.predict_proba, X)
                self.assertRaises(NotFittedError, model.predict, X)

    def test_cnn_retains_singleton_minimum_length_training_batch(self):
        self._torch()
        from src.models.dl.cnn_1d import CNN1DClassifier

        rng = np.random.RandomState(2)
        X = rng.randn(8, 1, 61).astype("float32")
        y = np.tile([0, 1], 4)
        model = CNN1DClassifier(
            in_channels=1,
            max_epochs=1,
            batch_size=5,
            val_fraction=0.25,
            random_state=2,
            device="cpu",
        )
        model.fit(X, y)
        proba = model.predict_proba(X)
        self.assertTrue(proba.shape == (8, 2) and np.isfinite(proba).all())

    def test_cnn_rejects_nonfinite_normalization_or_inference(self):
        torch = self._torch()
        from src.models.dl.cnn_1d import CNN1DClassifier

        for failure in ("overflow", "invalid_scale", "invalid_network"):
            with self.subTest(failure=failure):

                class FixedNetwork(torch.nn.Module):

                    def forward(self, X):
                        fill = float("nan") if failure == "invalid_network" else 0.0
                        return torch.full((len(X), 2), fill)

                model = CNN1DClassifier(in_channels=1, device="cpu")
                model.classes_ = np.array([0, 1])
                model.model = FixedNetwork()
                model._mean = np.zeros((1, 1, 1), dtype="float32")
                model._std = np.full((1, 1, 1), 1e-08, dtype="float32")
                X = np.zeros((2, 1, 61), dtype="float32")
                if failure == "overflow":
                    X.fill(1e38)
                elif failure == "invalid_scale":
                    model._std.fill(np.inf)
                with self.assertRaisesRegex(
                    ValueError, "CNN (normalization|inference)"
                ):
                    model.predict_proba(X)

    def test_cnn_group_split_fallback_finds_rare_class_cover_deterministically(self):
        self._torch()
        from src.models.dl.cnn_1d import CNN1DClassifier

        y = np.array([0] * 28 + [1, 2])
        groups = np.arange(30)
        model = CNN1DClassifier(val_fraction=0.9, random_state=42)
        train, validation = model._validation_split(y, groups)
        self.assertTrue(len(train) == 3 and len(validation) == 27)
        self.assertTrue(set(groups[train]).isdisjoint(groups[validation]))
        np.testing.assert_array_equal(np.unique(y[train]), [0, 1, 2])
        np.testing.assert_array_equal(model._validation_split(y, groups)[0], train)

    def test_cnn_group_split_rejects_impossible_class_cover_at_requested_fraction(self):
        self._torch()
        from src.models.dl.cnn_1d import CNN1DClassifier

        with self.assertRaisesRegex(ValueError, "retain every training class"):
            CNN1DClassifier(val_fraction=0.5)._validation_split(
                np.array([0, 1, 2]), np.arange(3)
            )


class MethodologyTests(fixtures.TemporaryWorkspaceTestCase):

    def setUp(self):
        self.enterContext(preserve_random_state())
        super().setUp()

    def test_pipeline_imputes_and_scales_inside_the_fold(self):
        pipe = build_pipeline("rf")
        self.assertEqual([name for name, _ in pipe.steps], ["impute", "scale", "clf"])

    def test_task_preparation_sanitizes_a_copy_under_pandas_copy_on_write(self):
        frame = pd.DataFrame(
            {
                "subject_id": ["S0", "S1"],
                "window_id": [0, 0],
                "label": [1, 2],
                "label_name": ["baseline", "stress"],
                "HRV_test": [1.0, np.inf],
            }
        )
        original = frame.copy(deep=True)
        raw = np.zeros((2, 1, 4))
        with pd.option_context("mode.copy_on_write", True):
            X, y, subjects, columns, selected_raw = prepare_task(frame, raw, [1, 2])
        np.testing.assert_array_equal(X, [[1.0], [np.nan]])
        np.testing.assert_array_equal(y, [0, 1])
        np.testing.assert_array_equal(subjects, ["S0", "S1"])
        np.testing.assert_array_equal(selected_raw, raw)
        self.assertEqual(columns, ["HRV_test"])
        pd.testing.assert_frame_equal(frame, original)

    def test_loso_evaluate_holds_out_each_subject(self):
        rng = np.random.RandomState(0)
        n_subjects, per = (4, 40)
        groups = np.repeat([f"S{i}" for i in range(n_subjects)], per)
        X = rng.randn(len(groups), 6)
        y = rng.randint(0, 2, len(groups))
        res = loso_evaluate(lambda: build_pipeline("lr"), X, y, groups)
        self.assertEqual(len(res["per_subject"]), n_subjects)
        self.assertEqual(set(res["per_subject"]["subject"]), set(groups))

    def test_cnn_loso_reports_window_weighted_accuracy(self):
        case = self
        from src.models.dl import cnn_1d

        groups = np.array(["S0"] * 2 + ["S1"] * 6)
        y = np.tile([0, 1], 4)
        x_raw = np.zeros((8, 1, 2))
        x_raw[:, 0, 0] = [0, 1, 1, 0, 1, 0, 1, 0]
        x_raw[:, 0, 1] = np.arange(8)
        train_sizes = []

        class FixedCNN:

            def __init__(self, in_channels, random_state):
                self.training_ids = set()

            def fit(self, X, labels, groups=None):
                self.training_ids = set(X[:, 0, 1])
                case.assertEqual(len(groups), len(labels))
                train_sizes.append(len(labels))

            def predict(self, X):
                case.assertTrue(self.training_ids.isdisjoint(X[:, 0, 1]))
                return X[:, 0, 0].astype(int)

        self.patch_object(cnn_1d, "CNN1DClassifier", FixedCNN)
        result = cnn_loso(x_raw, y, groups)
        self.assertEqual(train_sizes, [6, 2])
        fixtures.assert_close(result["accuracy_mean"], 0.5)
        fixtures.assert_close(result["pooled_accuracy"], 0.25)
        np.testing.assert_array_equal(result["y_true"], y)

    def test_loso_splits_have_disjoint_subjects(self):
        from sklearn.model_selection import LeaveOneGroupOut

        groups = np.repeat([f"S{i}" for i in range(5)], 30)
        X = np.zeros((len(groups), 3))
        y = np.zeros(len(groups))
        for train_idx, test_idx in LeaveOneGroupOut().split(X, y, groups):
            train_subjects = set(groups[train_idx])
            test_subjects = set(groups[test_idx])
            self.assertTrue(train_subjects.isdisjoint(test_subjects))
            self.assertEqual(len(test_subjects), 1)

    def test_nonoverlap_mask_keeps_every_other_window_per_subject(self):
        groups = np.array(["S0", "S0", "S0", "S0", "S0", "S1", "S1", "S1", "S1"])
        mask = nonoverlap_mask(groups)
        self.assertEqual(list(np.where(mask)[0]), [0, 2, 4, 5, 7])
        for g in np.unique(groups):
            idx = np.where(groups == g)[0]
            self.assertEqual(list(idx[mask[idx]]), list(idx[::2]))

    def test_kfold_gap_uses_non_overlapping_windows(self):
        rng = np.random.RandomState(0)
        groups = np.repeat(["S0", "S1"], 40)
        X = rng.randn(len(groups), 5)
        y = np.tile(np.concatenate([np.zeros(20), np.ones(20)]).astype(int), 2)
        self.assertEqual(int(nonoverlap_mask(groups).sum()), 40)
        acc = kfold_accuracy(lambda: build_pipeline("lr"), X, y, groups)
        self.assertTrue(0.0 <= acc <= 1.0)

    def test_window_label_rejects_impure_and_out_of_set_windows(self):
        ds = WindowedDataset.__new__(WindowedDataset)
        ds.purity = 0.9
        self.assertEqual(ds._window_label(np.full(100, 2)), 2)
        self.assertIs(
            ds._window_label(np.concatenate([np.full(85, 2), np.full(15, 1)])), None
        )
        self.assertEqual(
            ds._window_label(np.concatenate([np.full(95, 2), np.full(5, 1)])), 2
        )
        self.assertIs(ds._window_label(np.full(100, 4)), None)

    def test_scaler_imputer_fit_per_fold_never_on_held_out_subject(self):
        from sklearn.model_selection import LeaveOneGroupOut

        rng = np.random.RandomState(0)
        subjects = ["S0", "S1", "S2"]
        shifts = {"S0": 0.0, "S1": 50.0, "S2": 200.0}
        groups = np.repeat(subjects, 30)
        X = np.vstack([rng.randn(30, 4) + shifts[s] for s in subjects])
        y = np.tile([0, 1], 45)
        logo = LeaveOneGroupOut()
        fold_scaler_means = {}
        global_mean = X.mean(axis=0)
        for train_idx, test_idx in logo.split(X, y, groups):
            held = groups[test_idx][0]
            pipe = build_pipeline("lr")
            pipe.fit(X[train_idx], y[train_idx])
            scaler_mean = pipe.named_steps["scale"].mean_
            imputer_stats = pipe.named_steps["impute"].statistics_
            np.testing.assert_allclose(
                scaler_mean, X[train_idx].mean(axis=0), rtol=1e-06, atol=1e-06
            )
            np.testing.assert_allclose(
                imputer_stats, np.median(X[train_idx], axis=0), rtol=1e-06, atol=1e-06
            )
            self.assertFalse(np.allclose(scaler_mean, global_mean, atol=1.0))
            fold_scaler_means[held] = scaler_mean
        self.assertFalse(
            np.allclose(fold_scaler_means["S0"], fold_scaler_means["S1"], atol=1.0)
        )
        self.assertFalse(
            np.allclose(fold_scaler_means["S1"], fold_scaler_means["S2"], atol=1.0)
        )

    def test_loso_macro_f1_uses_the_full_task_class_set(self):
        from sklearn.dummy import DummyClassifier
        from sklearn.pipeline import Pipeline

        groups = np.repeat(["S0", "S1", "S2"], 4)
        y = np.array([0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1])
        factory = lambda: Pipeline(
            [("clf", DummyClassifier(strategy="constant", constant=0))]
        )
        result = loso_evaluate(factory, np.zeros((12, 1)), y, groups)
        s0 = result["per_subject"].set_index("subject").loc["S0"]
        self.assertEqual(s0["accuracy"], 1)
        fixtures.assert_close(s0["f1_macro"], 0.5)

    def test_recalibration_predictions_hold_out_outer_and_inner_subjects(self):
        case = self
        from scripts.calibration import loso_recalibrated_proba

        groups = np.repeat(np.arange(4), 6)
        X = np.column_stack([groups, np.arange(24)])
        y = np.tile([0, 1], 12)
        predictions = []

        class RecordingEstimator:
            classes_ = np.array([0, 1])

            def __init__(self):
                self.named_steps = {"clf": self}

            def fit(self, values, labels):
                self.trained_subjects = set(values[:, 0])
                return self

            def predict_proba(self, values):
                held_subjects = set(values[:, 0])
                case.assertTrue(self.trained_subjects.isdisjoint(held_subjects))
                predictions.append((self.trained_subjects, held_subjects))
                return np.tile([0.4, 0.6], (len(values), 1))

        true, proba = loso_recalibrated_proba(
            RecordingEstimator, X, y, groups, method="sigmoid"
        )
        np.testing.assert_array_equal(true, y)
        self.assertEqual(len(predictions), 16)
        self.assertTrue(proba.shape == (24, 2) and np.isfinite(proba).all())

    def test_threshold_pass_balances_xgboost_on_training_labels(self):
        from sklearn.utils.class_weight import compute_sample_weight
        from scripts import threshold_metrics

        fitted = []

        class XGBClassifier:
            classes_ = np.array([0, 1])

            def __init__(self):
                self.named_steps = {"clf": self}

            def fit(self, values, labels, **kwargs):
                np.testing.assert_array_equal(
                    kwargs["clf__sample_weight"],
                    compute_sample_weight("balanced", labels),
                )
                fitted.append(len(labels))

            def predict_proba(self, values):
                return np.tile([0.3, 0.7], (len(values), 1))

        self.patch_object(
            threshold_metrics, "build_pipeline", lambda key: XGBClassifier()
        )
        true, proba = threshold_metrics.loso_pos_proba(
            "xgb",
            np.zeros((12, 1)),
            np.tile([0, 0, 0, 1], 3),
            np.repeat(["S0", "S1", "S2"], 4),
        )
        self.assertEqual(fitted, [8, 8, 8])
        self.assertTrue(len(true) == len(proba) == 12)

    def test_constant_predictions_select_a_finite_operating_point(self):
        from scripts.threshold_metrics import operating_point

        point = operating_point([0, 1, 0, 1], [0.5, 0.5, 0.5, 0.5])
        self.assertEqual(point["threshold"], 0.5)
        self.assertTrue(point["ppv"] == 0.5 and point["npv"] is None)
        json.dumps(point, allow_nan=False)

    def test_threshold_results_save_exact_roc_and_finite_thresholds(self):
        from sklearn.metrics import auc
        from scripts import threshold_metrics

        y = np.array([0, 0, 1, 1])
        for scores, expected_auc in (
            ([0.1, 0.4, 0.35, 0.8], 0.75),
            ([0.5, 0.5, 0.5, 0.5], 0.5),
        ):
            with self.subTest(scores=scores):
                p1 = np.array(scores)
                with (
                    patch.object(threshold_metrics, "RESULTS_DIR", self.tmp_path),
                    patch.object(threshold_metrics, "FEATURE_MODELS", ["lr", "rf"]),
                    patch.object(
                        threshold_metrics, "load_cached", lambda: (None, None)
                    ),
                    patch.object(
                        threshold_metrics,
                        "prepare_task",
                        lambda *args: (None, y, None, None, None),
                    ),
                    patch.object(
                        threshold_metrics, "loso_pos_proba", lambda *args: (y, p1)
                    ),
                ):
                    threshold_metrics.run()
                result = json.loads(
                    (self.tmp_path / "threshold_metrics.json").read_text()
                )
                json.dumps(result, allow_nan=False)
                self.assertEqual(result["n_windows"], 4)
                self.assertEqual(len(result["models"]), 2)
                for row in result["models"]:
                    self.assertIs(row["available"], True)
                    curve = row["roc_curve"]
                    fpr, tpr, thresholds = (
                        curve["fpr"],
                        curve["tpr"],
                        curve["thresholds"],
                    )
                    self.assertEqual(len(fpr), len(tpr))
                    self.assertEqual(len(fpr), len(thresholds))
                    self.assertEqual((fpr[0], tpr[0]), (0.0, 0.0))
                    self.assertEqual((fpr[-1], tpr[-1]), (1.0, 1.0))
                    self.assertIsNone(thresholds[0])
                    self.assertAlmostEqual(auc(fpr, tpr), expected_auc)
                    self.assertAlmostEqual(row["auroc"], expected_auc)
                    for false_positive_rate, true_positive_rate, threshold in zip(
                        fpr[1:], tpr[1:], thresholds[1:]
                    ):
                        self.assertTrue(np.isfinite(threshold))
                        predicted_stress = p1 >= threshold
                        self.assertEqual(
                            false_positive_rate, predicted_stress[y == 0].mean()
                        )
                        self.assertEqual(
                            true_positive_rate, predicted_stress[y == 1].mean()
                        )
                self.assertNotIn("operating_point", result["models"][0])
                self.assertIn("operating_point", result["models"][1])

    def _case_threshold_results_mark_undefined_scores_unavailable(self, invalid_case):
        from scripts import threshold_metrics

        self.patch_object(threshold_metrics, "RESULTS_DIR", self.tmp_path)
        self.patch_object(threshold_metrics, "FEATURE_MODELS", ["lr", "rf"])
        self.patch_object(threshold_metrics, "load_cached", lambda: (None, None))
        y = np.array([0, 1])
        self.patch_object(
            threshold_metrics, "prepare_task", lambda *args: (None, y, None, None, None)
        )

        def probabilities(key, *args):
            labels = (
                np.array([0, 0]) if key == "lr" and invalid_case == "one_class" else y
            )
            return labels, np.array([0.2, 0.8])

        self.patch_object(threshold_metrics, "loso_pos_proba", probabilities)
        if invalid_case == "nonfinite_auc":
            self.patch_object(
                threshold_metrics,
                "roc_auc_score",
                Mock(side_effect=[float("nan"), 1.0]),
            )
        threshold_metrics.run()
        result = json.loads((self.tmp_path / "threshold_metrics.json").read_text())
        self.assertIs(result["models"][0]["available"], False)
        self.assertIn("require", result["models"][0]["reason"])
        self.assertIs(result["models"][1]["available"], True)

    def test_threshold_failure_preserves_previous_results_and_signals_failure(self):
        from scripts import threshold_metrics

        path = self.tmp_path / "threshold_metrics.json"
        path.write_bytes(b"previous usable ROC results")
        self.patch_object(threshold_metrics, "RESULTS_DIR", self.tmp_path)
        self.patch_object(threshold_metrics, "FEATURE_MODELS", ["lr", "rf"])
        self.patch_object(threshold_metrics, "load_cached", lambda: (None, None))
        self.patch_object(
            threshold_metrics,
            "prepare_task",
            lambda *args: (None, np.array([0, 1]), None, None, None),
        )
        self.patch_object(
            threshold_metrics,
            "loso_pos_proba",
            Mock(side_effect=RuntimeError("estimator failed")),
        )
        with self.assertRaisesRegex(RuntimeError, "all models failed"):
            threshold_metrics.run()
        self.assertEqual(path.read_bytes(), b"previous usable ROC results")

    def _case_wrist_rejects_incompatible_chest_results_before_fitting(
        self, schema, subjects
    ):
        from scripts import wrist

        (self.tmp_path / "metrics.json").write_text(
            json.dumps(
                {
                    "binary": {
                        "feature_schema_version": schema,
                        "per_subject": [{"subject": s} for s in subjects],
                    }
                }
            )
        )
        self.patch_object(wrist, "RESULTS_DIR", self.tmp_path)
        self.patch_object(wrist, "FIGURES_DIR", self.tmp_path / "figures")
        self.patch_object(wrist, "load_wrist", lambda: object())
        self.patch_object(
            wrist,
            "prepare_binary",
            lambda *args: (np.zeros((2, 1)), np.array([0, 1]), np.array(["S0", "S1"])),
        )
        self.patch_object(
            wrist,
            "loso_evaluate",
            lambda *args: self.fail("Fitted an incompatible comparison"),
        )
        with self.assertRaisesRegex(
            ValueError, "run_experiment.py --rebuild --subjects S0 S1"
        ):
            wrist.run()

    def test_wrist_accepts_matching_chest_schema_and_subjects_in_any_order(self):
        from scripts.wrist import _validate_chest_comparison

        _validate_chest_comparison(
            {
                "feature_schema_version": FEATURE_SCHEMA_VERSION,
                "per_subject": [{"subject": "S1"}, {"subject": "S0"}],
            },
            np.array(["S0", "S0", "S1", "S1"]),
        )

    def test_all_tied_model_statistics_remain_finite(self):
        from scripts.stats import _friedman, bootstrap_ci, holm_bonferroni

        scores = [np.array([0.6, 0.8, 0.9])] * 4
        self.assertEqual(_friedman(scores), (0.0, 1.0))
        self.assertEqual(bootstrap_ci(np.ones(3), n=10), (1.0, 1.0))
        self.assertEqual(
            holm_bonferroni([("a", 0.01), ("b", 0.04), ("c", 0.03)]),
            {"a": 0.03, "c": 0.06, "b": 0.06},
        )

    def test_subject_bootstrap_rejects_insufficient_or_nonfinite_data(self):
        from scripts.stats import bootstrap_ci

        for scores in [[], [0.1, 0.2], [0.1, 0.2, np.nan]]:
            with self.subTest(scores=scores):
                self.assertRaisesRegex(ValueError, "three finite", bootstrap_ci, scores)

    def _case_experiment_exports_zero_gap_and_feature_schema(self, cnn_wins):
        from scripts import run_experiment

        y = np.tile([0, 1], 6)
        frame = pd.DataFrame(
            {
                "subject_id": np.repeat(["S0", "S1", "S2"], 4),
                "window_id": np.tile(np.arange(4), 3),
                "label": y + 1,
                "label_name": np.where(y, "stress", "baseline"),
                "HRV_test": y,
            }
        )
        raw = np.zeros((12, 1, 64))
        saved_models = []

        class InverseEstimator:

            def __init__(self):
                self.named_steps = {"clf": self}

            def fit(self, X, labels):
                return self

            def predict(self, X):
                return 1 - X[:, 0].astype(int)

        self.patch_object(run_experiment, "load_cached", lambda: (frame, raw))
        self.patch_object(
            run_experiment, "build_pipeline", lambda key: InverseEstimator()
        )
        self.patch_object(run_experiment, "CLASSIFIERS", ["rf"])
        self.patch_object(
            run_experiment, "TASKS", {"binary": run_experiment.TASKS["binary"]}
        )
        for name in ("RESULTS_DIR", "FIGURES_DIR", "MODELS_DIR"):
            self.patch_object(run_experiment, name, self.tmp_path / name.lower())
        generated_gap = run_experiment.FIGURES_DIR / "binary_optimism_gap.png"
        generated_gap.parent.mkdir(parents=True)
        generated_gap.write_bytes(b"stale generated comparison")
        archived_gap = self.tmp_path / "outputs" / "figures" / generated_gap.name
        archived_gap.parent.mkdir(parents=True)
        archived_gap.write_bytes(b"archived benchmark comparison")
        for name in (
            "plot_model_comparison",
            "plot_confusion",
            "plot_per_subject",
            "plot_embedding",
        ):
            self.patch_object(run_experiment, name, lambda *args: None)
        self.patch_object(
            run_experiment,
            "plot_gap",
            lambda loso, kfold, path: path.write_bytes(b"current comparison"),
        )
        self.patch_object(
            run_experiment,
            "shap_analysis",
            lambda *args: pd.DataFrame(
                {"feature": ["HRV_test"], "mean_abs_shap": [0.1]}
            ),
        )

        def save_model(model, path):
            saved_models.append(model)
            path.write_bytes(b"temporary test model")

        self.patch_object(run_experiment, "save_verified_joblib", save_model)
        if cnn_wins:
            groups = frame["subject_id"].to_numpy()
            folds = [
                (np.flatnonzero(groups == group), y[groups == group])
                for group in np.unique(groups)
            ]
            cnn_result = run_experiment._loso_result(y, groups, folds)
            self.patch_object(run_experiment, "cnn_loso", lambda *args: cnn_result)
        self.patch_object(
            run_experiment.sys,
            "argv",
            ["run_experiment.py"] + ([] if cnn_wins else ["--no-cnn"]),
        )
        run_experiment.run()
        result = json.loads((run_experiment.RESULTS_DIR / "metrics.json").read_text())
        self.assertEqual(archived_gap.read_bytes(), b"archived benchmark comparison")
        if cnn_wins:
            self.assertFalse(generated_gap.exists())
            self.assertEqual(result["binary"]["best_model"], "1D-CNN")
            self.assertIs(result["binary"]["loso_matched_accuracy"], None)
            self.assertIs(result["binary"]["within_subject_accuracy"], None)
            self.assertIs(result["binary"]["optimism_gap_pts"], None)
            self.assertEqual(
                result["methodology"]["cnn_validation"], "training_subject_holdout"
            )
        else:
            self.assertEqual(generated_gap.read_bytes(), b"current comparison")
            self.assertEqual(result["binary"]["loso_matched_accuracy"], 0)
            self.assertEqual(result["binary"]["within_subject_accuracy"], 0)
            self.assertEqual(result["binary"]["optimism_gap_pts"], 0)
            self.assertIs(result["methodology"]["cnn_validation"], None)
        self.assertEqual(result["binary"]["n_subjects"], 3)
        self.assertEqual(
            saved_models[0]["feature_schema_version"],
            result["binary"]["feature_schema_version"],
        )
        self.assertTrue(
            saved_models[0]["model_name"]
            == result["binary"]["inference_model"]
            == "Random Forest"
        )
        self.assertEqual(
            result["artifacts"]["shap"],
            {
                "model": "XGBoost",
                "scope": "full_data_binary_fit",
                "path": "shap_top_features.csv",
                "sha256": run_experiment.sha256_file(
                    run_experiment.RESULTS_DIR / "shap_top_features.csv"
                ),
            },
        )
        self.assertEqual(
            result["artifacts"]["inference_model"],
            {
                "model": "Random Forest",
                "path": "stress_classifier.joblib",
                "sha256": run_experiment.sha256_file(
                    run_experiment.MODELS_DIR / "stress_classifier.joblib"
                ),
            },
        )
        self.assertEqual(result["benchmark_protocol_version"], 2)

    def test_benchmark_rejects_missing_amusement_before_any_model_fits(self):
        from scripts import run_experiment

        frame = pd.DataFrame(
            {"subject_id": ["S0", "S0", "S1", "S1"], "label": [1, 2, 1, 2]}
        )
        self.patch_object(
            run_experiment, "load_cached", lambda: (frame, np.zeros((4, 1, 64)))
        )
        self.patch_object(run_experiment.sys, "argv", ["run_experiment.py", "--no-cnn"])
        for name in ("RESULTS_DIR", "FIGURES_DIR", "MODELS_DIR"):
            self.patch_object(run_experiment, name, self.tmp_path / name.lower())
        self.patch_object(
            run_experiment,
            "loso_evaluate",
            lambda *args: self.fail("Fitted before task validation"),
        )
        with self.assertRaisesRegex(
            ValueError, "multiclass benchmark is missing required classes: amusement"
        ):
            run_experiment.run()

    def _case_descriptive_pca_handles_available_feature_dimensions(self, n_features):
        from scripts import run_experiment

        X = np.random.RandomState(4).randn(8, n_features)
        y = np.tile([0, 1], 4)
        plotted_vertical = []
        original_scatter = run_experiment.plt.scatter

        def scatter(x, vertical, **kwargs):
            plotted_vertical.extend(vertical)
            return original_scatter(x, vertical, **kwargs)

        self.patch_object(run_experiment.plt, "scatter", scatter)
        path = self.tmp_path / "pca.png"
        run_experiment.plot_embedding(X, y, ["baseline", "stress"], path)
        self.assertTrue(path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n"))
        self.assertEqual(len(plotted_vertical), len(y))
        self.assertTrue(np.isfinite(plotted_vertical).all())
        self.assertEqual(bool(np.any(np.array(plotted_vertical) != 0)), n_features > 1)

    def test_descriptive_pca_rejects_class_name_mismatch(self):
        from scripts.run_experiment import plot_embedding

        with self.assertRaisesRegex(ValueError, "class names"):
            plot_embedding(
                np.ones((4, 2)),
                np.array([0, 1, 0, 1]),
                ["baseline", "stress", "amusement"],
                self.tmp_path / "pca.png",
            )


fixtures.add_cases(
    MethodologyTests,
    "threshold_results_mark_undefined_scores_unavailable",
    [("one_class",), ("nonfinite_auc",)],
)


fixtures.add_cases(
    MethodologyTests,
    "wrist_rejects_incompatible_chest_results_before_fitting",
    [
        (1, ["S0", "S1"]),
        (2, ["S0", "S2"]),
        (2, ["S0", "S0"]),
        (2, ["S0", ""]),
        (2, ["S0", 1]),
        (2, []),
    ],
)


fixtures.add_cases(
    MethodologyTests,
    "experiment_exports_zero_gap_and_feature_schema",
    [(False,), (True,)],
)


fixtures.add_cases(
    MethodologyTests,
    "descriptive_pca_handles_available_feature_dimensions",
    [(1,), (3,)],
)


class PersonalizeTests(fixtures.TemporaryWorkspaceTestCase):

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
            with (
                self.subTest(budget=budget),
                self.assertRaisesRegex(ValueError, "positive integers"),
            ):
                _sample_k(np.array([0, 1]), budget, np.random.RandomState(0))

    def test_tiny_personalization_pool_falls_back_without_crashing(self):
        from sklearn.dummy import DummyClassifier
        from sklearn.pipeline import Pipeline

        self.patch_object(
            personalize,
            "build_pipeline",
            lambda model: Pipeline([("clf", DummyClassifier())]),
        )
        y = np.tile([0, 0, 1, 1], 3)
        groups = np.repeat(["S0", "S1", "S2"], 4)
        result = personalize.compute(np.zeros((12, 1)), y, groups, k_values=[5])
        self.assertEqual(result["n_subjects"], 3)
        self.assertEqual(result["fewshot"]["5"], result["uncalibrated"])

    def test_personalization_rejects_no_eligible_targets(self):
        from sklearn.dummy import DummyClassifier
        from sklearn.pipeline import Pipeline

        self.patch_object(
            personalize,
            "build_pipeline",
            lambda model: Pipeline([("clf", DummyClassifier())]),
        )
        with self.assertRaisesRegex(ValueError, "target subject"):
            personalize.compute(
                np.zeros((6, 1)),
                np.zeros(6, dtype=int),
                np.repeat(["S0", "S1", "S2"], 2),
            )

    def _case_personalization_output_paths_protect_benchmark(self, use_synthetic):
        results_dir = self.tmp_path / "outputs" / "results"
        figures_dir = self.tmp_path / "outputs" / "generated" / "figures"
        demo_dir = self.tmp_path / "outputs" / "generated" / "demo"
        results_dir.mkdir(parents=True)
        figures_dir.mkdir(parents=True)
        real_result = results_dir / "personalization.json"
        real_figure = figures_dir / "personalization.png"
        real_result.write_bytes(b"committed benchmark snapshot")
        real_figure.write_bytes(b"committed benchmark plot")
        self.patch_object(personalize, "RESULTS_DIR", results_dir)
        self.patch_object(personalize, "FIGURES_DIR", figures_dir)
        self.patch_object(personalize, "DEMO_DIR", demo_dir)

        def cached_features():
            self.assertFalse(
                use_synthetic, "Synthetic demo read the real dataset cache"
            )
            return (None, None)

        def generated_features(**kwargs):
            self.assertTrue(use_synthetic)
            self.assertIs(kwargs["cache"], False)
            return (None, None, None)

        self.patch_object(personalize, "load_cached", cached_features)
        self.patch_object(synthetic, "features", generated_features)
        self.patch_object(personalize, "prepare_task", lambda *args: (None,) * 5)
        metrics = {
            "k_values": [5],
            "uncalibrated": {"ece": 0.1, "brier": 0.1},
            "global": {"ece": 0.08, "brier": 0.08},
            "fewshot": {"5": {"ece": 0.05, "brier": 0.05}},
        }
        self.patch_object(personalize, "compute", lambda *args, **kwargs: metrics)
        self.patch_object(
            personalize,
            "analysis_provenance",
            lambda *args: {"git_sha": "test", "generated_at": "test"},
        )
        self.patch_object(
            personalize, "_plot", lambda out, path: path.write_bytes(b"new plot")
        )
        personalize.run(synthetic=use_synthetic)
        target_results = demo_dir / "results" if use_synthetic else results_dir
        target_figures = demo_dir / "figures" if use_synthetic else figures_dir
        self.assertEqual(
            json.loads((target_results / "personalization.json").read_text()), metrics
        )
        self.assertEqual(
            (target_figures / "personalization.png").read_bytes(), b"new plot"
        )
        if use_synthetic:
            self.assertEqual(real_result.read_bytes(), b"committed benchmark snapshot")
            self.assertEqual(real_figure.read_bytes(), b"committed benchmark plot")
        else:
            self.assertFalse(demo_dir.exists())

    def test_personalization_serialization_failure_preserves_previous_snapshot(self):
        path = self.tmp_path / "personalization.json"
        path.write_bytes(b"previous usable snapshot")
        self.patch_object(personalize, "RESULTS_DIR", self.tmp_path)
        self.patch_object(personalize, "FIGURES_DIR", self.tmp_path / "figures")
        self.patch_object(personalize, "load_cached", lambda: (None, None))
        self.patch_object(personalize, "prepare_task", lambda *args: (None,) * 5)
        self.patch_object(
            personalize, "compute", lambda *args, **kwargs: {"ece": float("nan")}
        )
        self.patch_object(
            personalize, "_plot", lambda *args: self.fail("Plotted invalid metrics")
        )
        self.assertRaisesRegex(ValueError, "JSON compliant", personalize.run)
        self.assertEqual(path.read_bytes(), b"previous usable snapshot")


fixtures.add_cases(
    PersonalizeTests,
    "personalization_output_paths_protect_benchmark",
    [(False,), (True,)],
)


class PipelineSmokeTests(fixtures.TemporaryWorkspaceTestCase):

    @classmethod
    def setUpClass(cls):
        cls.feature_frame, cls.raw_windows, _ = features(
            n_subjects=4, block_sec=100, seed=3
        )

    def test_pipeline_builds_features(self):
        df = self.feature_frame
        self.assertGreater(len(df), 0)
        self.assertEqual(df["subject_id"].nunique(), 4)
        feat = [
            c
            for c in df.columns
            if c not in ("subject_id", "window_id", "label", "label_name")
        ]
        self.assertGreaterEqual(len(feat), 50)

    def test_classifier_aliases_preserve_defaults_and_allow_overrides(self):
        defaults = deepcopy(HYPERPARAMS)
        for name, alias in (
            ("lr", "logistic"),
            ("rf", "random_forest"),
            ("xgb", "xgboost"),
            ("lgbm", "lightgbm"),
        ):
            with self.subTest(classifier=name, alias=alias):
                model = get_classifier(name)
                aliased = get_classifier(alias.upper(), random_state=7, n_jobs=1)
                self.assertIs(type(aliased), type(model))
                for parameter, expected in defaults[name].items():
                    self.assertEqual(model.get_params()[parameter], expected)
                self.assertEqual(model.get_params()["random_state"], 42)
                self.assertEqual(model.get_params()["n_jobs"], -1)
                self.assertEqual(aliased.get_params()["random_state"], 7)
                self.assertEqual(aliased.get_params()["n_jobs"], 1)
        self.assertEqual(HYPERPARAMS, defaults)
        with self.assertRaisesRegex(ValueError, "Unknown classifier"):
            get_classifier("unknown")

    def test_loso_runs_end_to_end(self):
        df, x_raw = self.feature_frame, self.raw_windows
        X, y, groups, _, _ = prepare_task(df, x_raw, [1, 2])
        res = loso_evaluate(lambda: build_pipeline("lr"), X, y, groups)
        self.assertTrue(0.0 <= res["accuracy_mean"] <= 1.0)
        self.assertEqual(len(res["per_subject"]), 4)

    def test_calibration_outputs_are_valid(self):
        df, x_raw = self.feature_frame, self.raw_windows
        X, y, groups, _, _ = prepare_task(df, x_raw, [1, 2])
        out = calibration.compute(X, y, groups, model="lr", n_bins=10)
        for key in ("loso", "within_subject", "recalibrated_isotonic"):
            self.assertTrue(0.0 <= out[key]["ece"] <= 1.0)
            self.assertTrue(0.0 <= out[key]["brier"] <= 2.0)
        dc = out["decision_curve"]
        self.assertEqual(len(dc["thresholds"]), len(dc["net_benefit_uncalibrated"]))

    def test_chest_dataset_rejects_rates_that_reinterpret_native_wesad(self):
        for fs in (350, 1000):
            with self.subTest(fs=fs):
                tmp_path = self.tmp_path
                with self.assertRaisesRegex(
                    ValueError, "WESAD chest signals and labels are sampled at 700"
                ):
                    WindowedDataset(fs=fs, data_path=tmp_path / "missing")

    def test_subject_recordings_are_released_before_loading_the_next(self):
        for device in ("chest", "wrist"):
            with self.subTest(device=device):
                builder_type = WindowedDataset if device == "chest" else WristDataset
                builder = builder_type.__new__(builder_type)
                builder.loader = fixtures.source_loader(
                    self.tmp_path, ["S2", "S3", "S4"]
                )
                builder.features = FeatureExtractionPipeline(
                    {
                        group: group == "temperature"
                        for group in FeatureExtractionPipeline.DEFAULT_CONFIG
                    }
                )
                builder.purity = 0.9
                builder.fs, builder.window_samples, builder.step, builder.cnn_length = (
                    700,
                    42000,
                    21000,
                    16,
                )
                builder.window_sec, builder.overlap = (60.0, 0.5)
                recordings = []
                seen = []

                def process(subject):
                    self.assertTrue(
                        all((reference() is None for reference in recordings))
                    )
                    seen.append(subject)
                    if subject == "S4":
                        return ([], [], []) if device == "chest" else ([], [])
                    label = 1 if subject == "S2" else 2
                    recording = np.full(1000, 35.0 + label / 10)
                    recordings.append(weakref.ref(recording))
                    windows = [
                        {
                            "temperature": recording[:240],
                            "subject_id": subject,
                            "window_id": label * 21000,
                            "label": label,
                        }
                    ]
                    return (
                        (windows, [np.full((5, 16), label)], [label])
                        if device == "chest"
                        else (windows, [label])
                    )

                builder._process_subject = process
                built = builder.build(subjects=["S2", "S4", "S3"], cache=False)
                frame = built[0] if device == "chest" else built
                self.assertEqual(seen, ["S2", "S4", "S3"])
                self.assertTrue(all((reference() is None for reference in recordings)))
                self.assertEqual(frame["subject_id"].tolist(), ["S2", "S3"])
                self.assertEqual(frame["window_id"].tolist(), [21000, 42000])
                self.assertEqual(frame["label_name"].tolist(), ["baseline", "stress"])
                np.testing.assert_allclose(frame["TEMP_mean"], [35.1, 35.2])
                self.assertEqual(
                    frame.attrs["feature_schema_version"], FEATURE_SCHEMA_VERSION
                )
                self.assertEqual(
                    frame.attrs["dataset_parameters"]["subjects"], ["S2", "S3", "S4"]
                )
                if device == "chest":
                    _, raw, labels = built
                    np.testing.assert_array_equal(labels, frame["label"])
                    np.testing.assert_array_equal(raw[:, 0, 0], labels)

    def test_dataset_without_retained_windows_keeps_its_empty_schema(self):
        for device in ("chest", "wrist"):
            with self.subTest(device=device):
                builder_type = WindowedDataset if device == "chest" else WristDataset
                builder = builder_type.__new__(builder_type)
                builder.loader = fixtures.source_loader(self.tmp_path, ["S2"])
                builder.features = FeatureExtractionPipeline()
                builder.purity = 0.9
                builder.fs, builder.window_samples, builder.step, builder.cnn_length = (
                    700,
                    42000,
                    21000,
                    16,
                )
                builder.window_sec, builder.overlap = (60.0, 0.5)
                builder._process_subject = lambda _: (
                    ([], [], []) if device == "chest" else ([], [])
                )
                built = builder.build(subjects=["S2"], cache=False)
                frame = built[0] if device == "chest" else built
                self.assertTrue(frame.empty)
                self.assertEqual(
                    list(frame),
                    [
                        "subject_id",
                        "window_id",
                        "label",
                        *builder.features.get_feature_names(),
                        "label_name",
                    ],
                )
                if device == "chest":
                    _, raw, labels = built
                    self.assertEqual(raw.shape, (0, 5, 16))
                    self.assertEqual(labels.size, 0)


class AnalysisPublicationTests(fixtures.TemporaryWorkspaceTestCase):
    def test_plotting_failure_preserves_metrics_and_figures_for_each_analysis(self):
        for module, result_name, figure_names in (
            (
                calibration,
                "calibration.json",
                [
                    "calibration_reliability.png",
                    "calibration_gap.png",
                    "calibration_decision_curve.png",
                ],
            ),
            (personalize, "personalization.json", ["personalization.png"]),
            (ablation, "ablation.csv", ["ablation.png"]),
            (cross_dataset, "cross_dataset.json", ["cross_dataset.png"]),
        ):
            with self.subTest(analysis=module.__name__):
                results = self.tmp_path / module.__name__ / "results"
                figures = self.tmp_path / module.__name__ / "figures"
                results.mkdir(parents=True)
                figures.mkdir()
                previous = {
                    results / result_name: b"previous metrics",
                    **{figures / name: b"previous figure" for name in figure_names},
                }
                for path, contents in previous.items():
                    path.write_bytes(contents)

                def fail_plot(*args):
                    args[-1].write_bytes(b"partial new figure")
                    raise OSError("plot failed")

                self.patch_object(module, "RESULTS_DIR", results)
                self.patch_object(module, "FIGURES_DIR", figures)
                self.patch_object(
                    module, "benchmark_reference", lambda *args, **kwargs: None
                )
                self.patch_object(module, "analysis_provenance", lambda *args: {})

                if module is cross_dataset:
                    frame = pd.DataFrame(
                        {
                            "subject": ["S0", "S1"],
                            "label": [0, 1],
                            "TEMP_mean": [1.0, 2.0],
                        }
                    )
                    self.patch_object(
                        module,
                        "_load_or_build_cache",
                        lambda *args, **kwargs: (
                            frame,
                            {"cache_sha256": "verified"},
                        ),
                    )
                    score = {
                        "accuracy": 0.5,
                        "balanced_accuracy": 0.5,
                        "f1_macro": 0.5,
                    }
                    self.patch_object(module, "within", lambda *args: score)
                    self.patch_object(module, "transfer", lambda *args: score)
                    self.patch_object(module, "_generation_context", lambda: {})
                    self.patch_object(module, "save_figure", fail_plot)
                else:
                    self.patch_object(
                        module,
                        "load_feature_data",
                        lambda *args, **kwargs: (None, None),
                    )
                    columns = [
                        "HRV_test",
                        "EDA_test",
                        "TEMP_test",
                        "RESP_test",
                        "ACC_test",
                    ]
                    self.patch_object(
                        module,
                        "prepare_task",
                        lambda *args: (
                            np.zeros((2, 5)),
                            np.array([0, 1]),
                            np.array(["S0", "S1"]),
                            columns,
                            None,
                        ),
                    )
                    if module is ablation:
                        score = {
                            "accuracy_mean": 0.5,
                            "accuracy_std": 0.0,
                            "f1_macro_mean": 0.5,
                        }
                        self.patch_object(module, "loso_evaluate", lambda *args: score)
                        self.patch_object(module, "save_figure", fail_plot)
                    else:
                        self.patch_object(module, "compute", lambda *args, **kwargs: {})
                        if module is calibration:
                            self.patch_object(
                                module,
                                "_plot_reliability",
                                lambda out, path: path.write_bytes(b"new reliability"),
                            )
                            self.patch_object(module, "_plot_gap", fail_plot)
                        else:
                            self.patch_object(module, "_plot", fail_plot)
                try:
                    with self.assertRaisesRegex(OSError, "plot failed"):
                        module.run()
                finally:
                    module.plt.close("all")

                for path, contents in previous.items():
                    self.assertEqual(path.read_bytes(), contents)
                self.assertEqual(set(results.iterdir()), {results / result_name})
                self.assertEqual(
                    set(figures.iterdir()), {figures / name for name in figure_names}
                )


class TuningTests(fixtures.TemporaryWorkspaceTestCase):

    def _case_synthetic_tuning_isolates_outputs_and_reads_only_demo_defaults(
        self, demo_accuracy
    ):
        output_dir = self.tmp_path / "outputs"
        results_dir = output_dir / "results"
        figures_dir = output_dir / "generated" / "figures"
        demo_dir = output_dir / "generated" / "demo"
        results_dir.mkdir(parents=True)
        figures_dir.mkdir(parents=True)
        model = "Random Forest"
        real_metrics = results_dir / "metrics.json"
        real_metrics.write_text(
            json.dumps(
                {"binary": {"models": [{"model": model, "accuracy_mean": 0.91}]}}
            )
        )
        real_result = results_dir / "tuning.json"
        real_figure = figures_dir / "tuning.png"
        real_result.write_bytes(b"existing real tuning result")
        real_figure.write_bytes(b"existing real tuning plot")
        originals = {
            path: path.read_bytes() for path in (real_metrics, real_result, real_figure)
        }
        if demo_accuracy is not None:
            demo_results = demo_dir / "results"
            demo_results.mkdir(parents=True)
            (demo_results / "metrics.json").write_text(
                json.dumps(
                    {
                        "binary": {
                            "models": [{"model": model, "accuracy_mean": demo_accuracy}]
                        }
                    }
                )
            )
        self.patch_object(tuning, "RESULTS_DIR", results_dir)
        self.patch_object(tuning, "FIGURES_DIR", figures_dir)
        self.patch_object(tuning, "DEMO_DIR", demo_dir)

        def refuse_real_cache():
            self.fail("Synthetic tuning read the real dataset cache")

        def synthetic_features(**kwargs):
            self.assertIs(kwargs["cache"], False)
            return (None, None, None)

        self.patch_object(tuning, "load_cached", refuse_real_cache)
        self.patch_object(synthetic, "features", synthetic_features)
        self.patch_object(tuning, "prepare_task", lambda *args: (None,) * 5)
        scores = {model: {"accuracy_mean": 0.72, "best_params": {}}}
        self.patch_object(tuning, "compute", lambda *args: scores)
        provenance = {"git_sha": "test", "generated_at": "test"}
        self.patch_object(tuning, "analysis_provenance", lambda *args: provenance)
        original_plot = tuning._plot
        plotted_defaults = []

        def plot(tuned, defaults, path):
            plotted_defaults.append(defaults)
            original_plot(tuned, defaults, path)

        self.patch_object(tuning, "_plot", plot)
        tuning.run(synthetic=True)
        self.assertTrue(
            all(
                (
                    path.read_bytes() == contents
                    for (path, contents) in originals.items()
                )
            )
        )
        self.assertEqual(
            json.loads((demo_dir / "results" / "tuning.json").read_text()),
            {
                model: {"accuracy_mean": 0.72, "best_params": {}},
                "provenance": provenance,
            },
        )
        demo_figure = demo_dir / "figures" / "tuning.png"
        if demo_accuracy is None:
            self.assertEqual(plotted_defaults, [])
            self.assertFalse(demo_figure.exists())
        else:
            self.assertEqual(plotted_defaults, [{model: demo_accuracy}])
            self.assertTrue(demo_figure.read_bytes().startswith(b"\x89PNG\r\n\x1a\n"))

    def test_nested_tuning_fails_on_invalid_candidate_instead_of_selecting_around_it(
        self,
    ):
        self.enterContext(patch.dict(tuning.GRIDS, {"lr": {"clf__C": [-1.0, 1.0]}}))
        groups = np.repeat(["S0", "S1", "S2", "S3"], 6)
        y = np.tile([0, 1], 12)
        X = np.random.RandomState(0).randn(24, 2)
        with self.assertRaisesRegex(ValueError, "C"):
            tuning.tune_model("lr", X, y, groups, inner_splits=2)

    def test_nested_tuning_requires_valid_subject_counts(self):
        for inner_splits, groups in [(1, ["S0", "S1", "S2"]), (2, ["S0", "S1"])]:
            with (
                self.subTest(inner_splits=inner_splits, groups=groups),
                self.assertRaisesRegex(ValueError, "requires"),
            ):
                tuning.tune_model(
                    "lr",
                    np.zeros((len(groups), 1)),
                    np.zeros(len(groups)),
                    np.array(groups),
                    inner_splits,
                )

    def test_tuning_plot_does_not_show_missing_default_as_zero(self):
        bars = []
        original = tuning.plt.bar

        def bar(x, values, *args, **kwargs):
            bars.append(list(values))
            return original(x, values, *args, **kwargs)

        self.patch_object(tuning.plt, "bar", bar)
        tuning._plot(
            {
                "Random Forest": {"accuracy_mean": 0.8},
                "LightGBM": {"accuracy_mean": 0.7},
            },
            {"Random Forest": 0.9},
            self.tmp_path / "plot.png",
        )
        self.assertEqual(bars, [[0.9], [0.8]])

    def _case_tuning_default_comparison_requires_matching_protocol_schema_and_cohort(
        self, mismatch
    ):

        summary = {
            "benchmark_protocol_version": tuning.BENCHMARK_PROTOCOL_VERSION,
            "binary": {
                "feature_schema_version": FEATURE_SCHEMA_VERSION,
                "models": [{"model": "Random Forest", "accuracy_mean": 0.91}],
                "per_subject": [{"subject": "S0"}, {"subject": "S1"}],
            },
        }
        if mismatch == "schema":
            summary["binary"]["feature_schema_version"] -= 1
        elif mismatch == "protocol":
            summary["benchmark_protocol_version"] -= 1
        elif mismatch == "cohort":
            summary["binary"]["per_subject"][1]["subject"] = "S2"
        elif mismatch == "duplicates":
            summary["binary"]["per_subject"][1]["subject"] = "S0"
        (self.tmp_path / "metrics.json").write_text(json.dumps(summary))
        defaults = tuning._defaults(self.tmp_path, groups=np.array(["S1", "S0", "S0"]))
        self.assertEqual(defaults, {"Random Forest": 0.91} if mismatch is None else {})

    def test_tuning_retires_only_generated_stale_plot_when_defaults_are_incompatible(
        self,
    ):
        results, figures = (
            self.tmp_path / "results",
            self.tmp_path / "generated" / "figures",
        )
        results.mkdir()
        figures.mkdir(parents=True)
        (results / "metrics.json").write_text(
            json.dumps({"benchmark_protocol_version": 1})
        )
        stale = figures / "tuning.png"
        stale.write_bytes(b"stale generated comparison")
        historical = self.tmp_path / "figures" / "tuning.png"
        historical.parent.mkdir()
        historical.write_bytes(b"preserved historical figure")
        self.patch_object(tuning, "RESULTS_DIR", results)
        self.patch_object(tuning, "FIGURES_DIR", figures)
        self.patch_object(tuning, "load_cached", lambda: (None, None))
        self.patch_object(
            tuning,
            "prepare_task",
            lambda *args: (None, None, np.array(["S0", "S1"]), None, None),
        )
        self.patch_object(
            tuning, "compute", lambda *args: {"Random Forest": {"accuracy_mean": 0.6}}
        )
        tuning.run()
        self.assertTrue((results / "tuning.json").exists())
        self.assertFalse(stale.exists())
        self.assertEqual(historical.read_bytes(), b"preserved historical figure")


fixtures.add_cases(
    TuningTests,
    "synthetic_tuning_isolates_outputs_and_reads_only_demo_defaults",
    [(None,), (0.42,)],
)


fixtures.add_cases(
    TuningTests,
    "tuning_default_comparison_requires_matching_protocol_schema_and_cohort",
    [(None,), ("schema",), ("protocol",), ("cohort",), ("duplicates",)],
)
