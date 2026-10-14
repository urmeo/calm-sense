import unittest
from copy import deepcopy
from contextlib import ExitStack
from pathlib import Path
from tempfile import TemporaryDirectory
import weakref
import numpy as np
from scripts import calibration
from scripts.run_experiment import build_pipeline, loso_evaluate, prepare_task
from src.dataset import WindowedDataset
from src.dataset_wrist import WristDataset
from src.features.feature_pipeline import (
    FEATURE_SCHEMA_VERSION,
    FeatureExtractionPipeline,
)
from src.models.ml.classifiers import HYPERPARAMS, get_classifier
from src.synthetic import features
from tests.cache_fixtures import source_loader


class PipelineSmokeTests(unittest.TestCase):

    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.tmp_path = Path(self.stack.enter_context(TemporaryDirectory()))

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
                builder.loader = source_loader(self.tmp_path, ["S2", "S3", "S4"])
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
                builder.loader = source_loader(self.tmp_path, ["S2"])
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
