import unittest
from contextlib import ExitStack
from unittest.mock import patch
from pathlib import Path
from tempfile import TemporaryDirectory
from functools import partialmethod
import numpy as np
from scripts.run_experiment import (
    build_pipeline,
    cnn_loso,
    kfold_accuracy,
    loso_evaluate,
    nonoverlap_mask,
)
from src.dataset import WindowedDataset


class MethodologyTests(unittest.TestCase):

    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.tmp_path = Path(self.stack.enter_context(TemporaryDirectory()))

    def test_pipeline_imputes_and_scales_inside_the_fold(self):
        pipe = build_pipeline("rf")
        self.assertEqual([name for name, _ in pipe.steps], ["impute", "scale", "clf"])

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

        self.stack.enter_context(patch.object(cnn_1d, "CNN1DClassifier", FixedCNN))
        result = cnn_loso(x_raw, y, groups)
        self.assertEqual(train_sizes, [6, 2])
        np.testing.assert_allclose(result["accuracy_mean"], 0.5, rtol=1e-06, atol=1e-12)
        np.testing.assert_allclose(
            result["pooled_accuracy"], 0.25, rtol=1e-06, atol=1e-12
        )
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
        np.testing.assert_allclose(s0["f1_macro"], 0.5, rtol=1e-06, atol=1e-12)

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

        self.stack.enter_context(
            patch.object(
                threshold_metrics, "build_pipeline", lambda key: XGBClassifier()
            )
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
        import json
        from scripts.threshold_metrics import operating_point

        point = operating_point([0, 1, 0, 1], [0.5, 0.5, 0.5, 0.5])
        self.assertEqual(point["threshold"], 0.5)
        self.assertTrue(point["ppv"] == 0.5 and point["npv"] is None)
        json.dumps(point, allow_nan=False)

    def test_threshold_results_save_exact_roc_and_finite_thresholds(self):
        import json
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
        tmp_path = self.tmp_path
        import json
        from scripts import threshold_metrics

        self.stack.enter_context(
            patch.object(threshold_metrics, "RESULTS_DIR", tmp_path)
        )
        self.stack.enter_context(
            patch.object(threshold_metrics, "FEATURE_MODELS", ["lr"])
        )
        self.stack.enter_context(
            patch.object(threshold_metrics, "load_cached", lambda: (None, None))
        )
        y = np.array([0, 0] if invalid_case == "one_class" else [0, 1])
        self.stack.enter_context(
            patch.object(
                threshold_metrics,
                "prepare_task",
                lambda *args: (None, y, None, None, None),
            )
        )
        self.stack.enter_context(
            patch.object(
                threshold_metrics,
                "loso_pos_proba",
                lambda *args: (y, np.array([0.2, 0.8])),
            )
        )
        if invalid_case == "nonfinite_auc":
            self.stack.enter_context(
                patch.object(
                    threshold_metrics, "roc_auc_score", lambda *args: float("nan")
                )
            )
        threshold_metrics.run()
        result = json.loads((tmp_path / "threshold_metrics.json").read_text())
        self.assertIs(result["models"][0]["available"], False)
        self.assertIn("require", result["models"][0]["reason"])

    def _case_wrist_rejects_incompatible_chest_results_before_fitting(
        self, schema, subjects
    ):
        tmp_path = self.tmp_path
        import json
        from scripts import wrist

        (tmp_path / "metrics.json").write_text(
            json.dumps(
                {
                    "binary": {
                        "feature_schema_version": schema,
                        "per_subject": [{"subject": s} for s in subjects],
                    }
                }
            )
        )
        self.stack.enter_context(patch.object(wrist, "RESULTS_DIR", tmp_path))
        self.stack.enter_context(
            patch.object(wrist, "FIGURES_DIR", tmp_path / "figures")
        )
        self.stack.enter_context(patch.object(wrist, "load_wrist", lambda: object()))
        self.stack.enter_context(
            patch.object(
                wrist,
                "prepare_binary",
                lambda *args: (
                    np.zeros((2, 1)),
                    np.array([0, 1]),
                    np.array(["S0", "S1"]),
                ),
            )
        )
        self.stack.enter_context(
            patch.object(
                wrist,
                "loso_evaluate",
                lambda *args: self.fail("Fitted an incompatible comparison"),
            )
        )
        with self.assertRaisesRegex(
            ValueError, "run_experiment.py --rebuild --subjects S0 S1"
        ):
            wrist.run()

    def test_wrist_accepts_matching_chest_schema_and_subjects_in_any_order(self):
        from scripts.wrist import _validate_chest_comparison
        from src.features.feature_pipeline import FEATURE_SCHEMA_VERSION

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

    def _case_subject_bootstrap_rejects_insufficient_or_nonfinite_data(self, scores):
        from scripts.stats import bootstrap_ci

        with self.assertRaisesRegex(ValueError, "three finite"):
            bootstrap_ci(scores)

    def _case_experiment_exports_zero_gap_and_feature_schema(self, cnn_wins):
        tmp_path = self.tmp_path
        import json
        import pandas as pd
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

        self.stack.enter_context(
            patch.object(run_experiment, "load_cached", lambda: (frame, raw))
        )
        self.stack.enter_context(
            patch.object(
                run_experiment, "build_pipeline", lambda key: InverseEstimator()
            )
        )
        self.stack.enter_context(patch.object(run_experiment, "CLASSIFIERS", ["rf"]))
        self.stack.enter_context(
            patch.object(
                run_experiment, "TASKS", {"binary": run_experiment.TASKS["binary"]}
            )
        )
        for name in ("RESULTS_DIR", "FIGURES_DIR", "MODELS_DIR"):
            self.stack.enter_context(
                patch.object(run_experiment, name, tmp_path / name.lower())
            )
        for name in (
            "plot_model_comparison",
            "plot_confusion",
            "plot_per_subject",
            "plot_embedding",
            "plot_gap",
        ):
            self.stack.enter_context(
                patch.object(run_experiment, name, lambda *args: None)
            )
        self.stack.enter_context(
            patch.object(
                run_experiment,
                "shap_analysis",
                lambda *args: pd.DataFrame(
                    {"feature": ["HRV_test"], "mean_abs_shap": [0.1]}
                ),
            )
        )

        def save_model(model, path):
            saved_models.append(model)
            path.write_bytes(b"temporary test model")

        self.stack.enter_context(
            patch.object(run_experiment, "save_verified_joblib", save_model)
        )
        if cnn_wins:
            groups = frame["subject_id"].to_numpy()
            folds = [
                (np.flatnonzero(groups == group), y[groups == group])
                for group in np.unique(groups)
            ]
            cnn_result = run_experiment._loso_result(y, groups, folds)
            self.stack.enter_context(
                patch.object(run_experiment, "cnn_loso", lambda *args: cnn_result)
            )
        self.stack.enter_context(
            patch.object(
                run_experiment.sys,
                "argv",
                ["run_experiment.py"] + ([] if cnn_wins else ["--no-cnn"]),
            )
        )
        run_experiment.run()
        result = json.loads((run_experiment.RESULTS_DIR / "metrics.json").read_text())
        if cnn_wins:
            self.assertEqual(result["binary"]["best_model"], "1D-CNN")
            self.assertIs(result["binary"]["loso_matched_accuracy"], None)
            self.assertIs(result["binary"]["within_subject_accuracy"], None)
            self.assertIs(result["binary"]["optimism_gap_pts"], None)
            self.assertEqual(
                result["methodology"]["cnn_validation"], "training_subject_holdout"
            )
        else:
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
        tmp_path = self.tmp_path
        import pandas as pd
        from scripts import run_experiment

        frame = pd.DataFrame(
            {"subject_id": ["S0", "S0", "S1", "S1"], "label": [1, 2, 1, 2]}
        )
        self.stack.enter_context(
            patch.object(
                run_experiment, "load_cached", lambda: (frame, np.zeros((4, 1, 64)))
            )
        )
        self.stack.enter_context(
            patch.object(run_experiment.sys, "argv", ["run_experiment.py", "--no-cnn"])
        )
        for name in ("RESULTS_DIR", "FIGURES_DIR", "MODELS_DIR"):
            self.stack.enter_context(
                patch.object(run_experiment, name, tmp_path / name.lower())
            )
        self.stack.enter_context(
            patch.object(
                run_experiment,
                "loso_evaluate",
                lambda *args: self.fail("Fitted before task validation"),
            )
        )
        with self.assertRaisesRegex(
            ValueError, "multiclass benchmark is missing required classes: amusement"
        ):
            run_experiment.run()

    def _case_descriptive_pca_handles_available_feature_dimensions(self, n_features):
        tmp_path = self.tmp_path
        from scripts import run_experiment

        X = np.random.RandomState(4).randn(8, n_features)
        y = np.tile([0, 1], 4)
        plotted_vertical = []
        original_scatter = run_experiment.plt.scatter

        def scatter(x, vertical, **kwargs):
            plotted_vertical.extend(vertical)
            return original_scatter(x, vertical, **kwargs)

        self.stack.enter_context(patch.object(run_experiment.plt, "scatter", scatter))
        path = tmp_path / "pca.png"
        run_experiment.plot_embedding(X, y, ["baseline", "stress"], path)
        self.assertTrue(path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n"))
        self.assertEqual(len(plotted_vertical), len(y))
        self.assertTrue(np.isfinite(plotted_vertical).all())
        self.assertEqual(bool(np.any(np.array(plotted_vertical) != 0)), n_features > 1)

    def test_descriptive_pca_rejects_class_name_mismatch(self):
        tmp_path = self.tmp_path
        from scripts.run_experiment import plot_embedding

        with self.assertRaisesRegex(ValueError, "class names"):
            plot_embedding(
                np.ones((4, 2)),
                np.array([0, 1, 0, 1]),
                ["baseline", "stress", "amusement"],
                tmp_path / "pca.png",
            )


for index, invalid_case in enumerate(["one_class", "nonfinite_auc"]):
    setattr(
        MethodologyTests,
        f"test_threshold_results_mark_undefined_scores_unavailable_{index}",
        partialmethod(
            MethodologyTests._case_threshold_results_mark_undefined_scores_unavailable,
            invalid_case=invalid_case,
        ),
    )
for index, (schema, subjects) in enumerate(
    [
        (1, ["S0", "S1"]),
        (2, ["S0", "S2"]),
        (2, ["S0", "S0"]),
        (2, ["S0", ""]),
        (2, ["S0", 1]),
        (2, []),
    ]
):
    setattr(
        MethodologyTests,
        f"test_wrist_rejects_incompatible_chest_results_before_fitting_{index}",
        partialmethod(
            MethodologyTests._case_wrist_rejects_incompatible_chest_results_before_fitting,
            schema=schema,
            subjects=subjects,
        ),
    )
for index, scores in enumerate([[], [0.1, 0.2], [0.1, 0.2, np.nan]]):
    setattr(
        MethodologyTests,
        f"test_subject_bootstrap_rejects_insufficient_or_nonfinite_data_{index}",
        partialmethod(
            MethodologyTests._case_subject_bootstrap_rejects_insufficient_or_nonfinite_data,
            scores=scores,
        ),
    )
for index, cnn_wins in enumerate([False, True]):
    setattr(
        MethodologyTests,
        f"test_experiment_exports_zero_gap_and_feature_schema_{index}",
        partialmethod(
            MethodologyTests._case_experiment_exports_zero_gap_and_feature_schema,
            cnn_wins=cnn_wins,
        ),
    )
for index, n_features in enumerate([1, 3]):
    setattr(
        MethodologyTests,
        f"test_descriptive_pca_handles_available_feature_dimensions_{index}",
        partialmethod(
            MethodologyTests._case_descriptive_pca_handles_available_feature_dimensions,
            n_features=n_features,
        ),
    )
