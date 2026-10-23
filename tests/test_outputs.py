import hashlib
import json
import logging
import math
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path
import re
from stat import S_IMODE
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch
from urllib.parse import unquote, urlsplit

import joblib
from matplotlib.figure import Figure
import numpy as np
from numpy.testing import assert_array_equal
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve
import structlog

from scripts import (
    build_dashboard_data,
    export_signals,
    plot_results as charts,
    run_experiment as benchmark,
    stamp_provenance,
    update_readme_tables,
)
from scripts.plot_results import (
    _roc_points,
    _roc_source,
    plot_calibration,
    plot_roc_auc,
    plt,
)
from src import logging_config, utils
from src.calibration import BINARY_BRIER_DEFINITION
from src.config import RESULTS_DIR
from src.utils import (
    analysis_provenance,
    atomic_write_text,
    benchmark_reference,
    feature_frame_sha256,
    load_verified_joblib,
    paired_effect_size,
    pipeline_source_sha256,
    provenance,
    save_verified_joblib,
    sha256_file,
    write_json,
)

import test_data as fixtures


class FixtureEstimator:
    def __init__(self):
        self.named_steps = {"clf": self}

    def fit(self, values, labels):
        return self


class BenchmarkOutputTests(fixtures.PatchedTestCase):
    def setUp(self):
        self.addCleanup(benchmark.plt.close, "all")
        self.root = Path(self.enterContext(TemporaryDirectory()))
        self.results = self.root / "outputs" / "results"
        self.figures = self.root / "outputs" / "generated" / "figures"
        self.models = self.root / "outputs" / "models"
        self.archive = self.root / "outputs" / "figures"
        for directory in (self.results, self.figures, self.models, self.archive):
            directory.mkdir(parents=True)
        result_names = ["metrics.json", "shap_top_features.csv", "calibration.json"]
        figure_names = ["shap_beeswarm.png", "unrelated.png"]
        for task in benchmark.TASKS:
            result_names.extend(
                (f"{task}_model_comparison.csv", f"{task}_per_subject.csv")
            )
            figure_names.extend(
                f"{task}_{plot}.png"
                for plot in (
                    "model_comparison",
                    "confusion",
                    "per_subject",
                    "pca",
                    "optimism_gap",
                )
            )
            (self.archive / f"{task}_optimism_gap.png").write_bytes(b"Archived curve")
        for name in result_names:
            (self.results / name).write_bytes(b"Previous " + name.encode())
        for name in figure_names:
            (self.figures / name).write_bytes(b"Previous " + name.encode())
        for name in (
            "stress_classifier.joblib",
            "stress_classifier.joblib.sha256",
            "unrelated.joblib",
        ):
            (self.models / name).write_bytes(b"Previous " + name.encode())
        (self.results / "metrics.json").chmod(0o640)
        (self.models / "stress_classifier.joblib").chmod(0o600)

        labels = np.tile([1, 1, 2, 2, 3, 3], 3)
        self.frame = pd.DataFrame(
            {
                "subject_id": np.repeat(["S0", "S1", "S2"], 6),
                "window_id": np.tile(np.arange(6), 3),
                "label": labels,
                "label_name": [
                    benchmark.TASKS["multiclass"]["names"][label - 1]
                    for label in labels
                ],
                "feature": labels - 1,
            }
        )
        self.raw = np.zeros((len(labels), 1, 16))
        self.classical_correct = True
        for name, directory in (
            ("RESULTS_DIR", self.results),
            ("FIGURES_DIR", self.figures),
            ("MODELS_DIR", self.models),
            ("PROCESSED_DATA_DIR", self.root / "processed"),
        ):
            self.patch(name, directory)
        self.patch("load_cached", return_value=(self.frame, self.raw))
        self.patch("set_seed")
        self.patch("CLASSIFIERS", ["rf"])
        self.patch("build_pipeline", side_effect=lambda key: FixtureEstimator())
        self.evaluate = self.patch("loso_evaluate", side_effect=self.evaluate_features)
        self.patch("kfold_accuracy", return_value=0.75)
        self.source_hash = self.patch(
            "pipeline_source_sha256",
            return_value={"scripts/run_experiment.py": "a" * 64},
        )
        self.patch(
            "provenance",
            return_value={
                "git_sha": "a" * 40,
                "working_tree_dirty": False,
                "feature_schema_version": benchmark.FEATURE_SCHEMA_VERSION,
            },
        )
        for name in (
            "plot_model_comparison",
            "plot_confusion",
            "plot_per_subject",
            "plot_embedding",
            "plot_gap",
        ):
            self.patch(name, side_effect=self.write_plot)
        self.patch("shap_analysis", side_effect=self.shap_snapshot)
        self.patch_object(benchmark.sys, "argv", ["run_experiment.py", "--no-cnn"])

    def patch(self, name, *args, **kwargs):
        return self.patch_object(benchmark, name, *args, **kwargs)

    @staticmethod
    def write_plot(*args):
        args[-1].write_bytes(b"Generated chart")

    @staticmethod
    def shap_snapshot(values, labels, names, directory):
        (directory / "shap_beeswarm.png").write_bytes(b"Generated SHAP chart")
        return pd.DataFrame({"feature": names, "mean_abs_shap": [0.1] * len(names)})

    @staticmethod
    def evaluation(labels, groups, correct):
        predicted = labels if correct else (labels + 1) % len(np.unique(labels))
        folds = [
            (np.flatnonzero(groups == group), predicted[groups == group])
            for group in np.unique(groups)
        ]
        return benchmark._loso_result(labels, groups, folds)

    def evaluate_features(self, factory, values, labels, groups):
        return self.evaluation(labels, groups, self.classical_correct)

    def use_winning_cnn(self):
        self.classical_correct = False
        self.patch(
            "cnn_loso",
            side_effect=lambda raw, labels, groups: self.evaluation(
                labels, groups, True
            ),
        )
        self.patch_object(benchmark.sys, "argv", ["run_experiment.py"])

    def snapshot(self):
        return (
            {
                path.relative_to(self.root): (
                    path.read_bytes(),
                    S_IMODE(path.stat().st_mode),
                )
                for path in self.root.rglob("*")
                if path.is_file()
            },
            {
                path.relative_to(self.root)
                for path in self.root.rglob("*")
                if path.is_dir()
            },
        )

    def assert_run_failure(self, exception, message):
        before = self.snapshot()
        with self.assertRaisesRegex(exception, message):
            benchmark.run()
        self.assertEqual(self.snapshot(), before)

    def test_late_task_failure_preserves_complete_previous_benchmark(self):
        external = benchmark.plt.figure()

        def fail_multiclass(factory, values, labels, groups):
            if len(np.unique(labels)) == 3:
                benchmark.plt.figure()
                raise ValueError("Multiclass fitting failed")
            return self.evaluate_features(factory, values, labels, groups)

        self.evaluate.side_effect = fail_multiclass
        self.assert_run_failure(ValueError, "Multiclass fitting failed")
        self.assertEqual(benchmark.plt.get_fignums(), [external.number])

    def test_source_change_preserves_results_models_and_figures(self):
        self.source_hash.side_effect = [
            {"scripts/run_experiment.py": "a" * 64},
            {"scripts/run_experiment.py": "b" * 64},
        ]
        self.assert_run_failure(ValueError, "Pipeline source changed")

    def test_invalid_final_metrics_preserve_previous_files(self):
        def nonfinite_metrics(factory, values, labels, groups):
            result = self.evaluate_features(factory, values, labels, groups)
            if len(np.unique(labels)) == 3:
                result["accuracy_mean"] = float("nan")
            return result

        self.evaluate.side_effect = nonfinite_metrics
        self.assert_run_failure(ValueError, "Out of range float")

    def test_model_serialization_failure_preserves_previous_files(self):
        self.patch("save_verified_joblib", side_effect=OSError("Model write failed"))
        self.assert_run_failure(OSError, "Model write failed")

    def test_publication_failure_restores_existing_and_new_artifacts(self):
        (self.results / "binary_per_subject.csv").unlink()
        (self.figures / "multiclass_pca.png").unlink()
        replace = Path.replace
        checksum = self.models / "stress_classifier.joblib.sha256"

        def fail_checksum(source, destination):
            if destination == checksum:
                raise OSError("Checksum publication failed")
            return replace(source, destination)

        with patch.object(Path, "replace", fail_checksum):
            self.assert_run_failure(OSError, "Checksum publication failed")

    def test_failed_second_gap_removal_restores_first_removal_and_all_files(self):
        self.use_winning_cnn()
        before = self.snapshot()
        unlink = Path.unlink
        second_gap = self.figures / "multiclass_optimism_gap.png"
        first_gap = self.figures / "binary_optimism_gap.png"
        removed_first_gap = []

        def fail_second_gap(path, *args, **kwargs):
            if path == second_gap:
                self.assertFalse(first_gap.exists())
                removed_first_gap.append(True)
                raise OSError("Gap removal failed")
            return unlink(path, *args, **kwargs)

        with patch.object(Path, "unlink", fail_second_gap):
            with self.assertRaisesRegex(OSError, "Gap removal failed"):
                benchmark.run()
        self.assertEqual(removed_first_gap, [True])
        self.assertEqual(self.snapshot(), before)

    def test_success_publishes_manifest_last_and_preserves_unmanaged_files(self):
        before_files, before_dirs = self.snapshot()
        replace = Path.replace
        published = []
        output_dirs = {self.results, self.figures, self.models}

        def record_publication(source, destination):
            if destination.parent in output_dirs:
                published.append(destination)
            return replace(source, destination)

        with patch.object(Path, "replace", record_publication):
            benchmark.run()
        after_files, after_dirs = self.snapshot()
        self.assertEqual(after_dirs, before_dirs)
        self.assertEqual(after_files.keys(), before_files.keys())
        self.assertEqual(published[-1], self.results / "metrics.json")
        for relative, (content, mode) in before_files.items():
            self.assertEqual(after_files[relative][1], mode)
            if relative.parts[1] == "figures" or relative.name in (
                "calibration.json",
                "unrelated.png",
                "unrelated.joblib",
            ):
                self.assertEqual(after_files[relative][0], content)
            else:
                self.assertNotEqual(after_files[relative][0], content)
        metrics = json.loads((self.results / "metrics.json").read_text())
        model_path = self.models / "stress_classifier.joblib"
        digest = hashlib.sha256(model_path.read_bytes()).hexdigest()
        self.assertEqual(metrics["artifacts"]["inference_model"]["sha256"], digest)
        self.assertEqual(
            (self.models / "stress_classifier.joblib.sha256").read_text().split()[0],
            digest,
        )
        self.assertEqual(
            set(metrics) & {"binary", "multiclass"}, {"binary", "multiclass"}
        )

    def test_successful_cnn_run_removes_only_generated_obsolete_gap_plots(self):
        self.use_winning_cnn()
        before_files, before_dirs = self.snapshot()
        benchmark.run()
        after_files, after_dirs = self.snapshot()
        self.assertEqual(after_dirs, before_dirs)
        removed = before_files.keys() - after_files.keys()
        self.assertEqual(
            removed,
            {
                (self.figures / f"{task}_optimism_gap.png").relative_to(self.root)
                for task in benchmark.TASKS
            },
        )
        for task in benchmark.TASKS:
            archived = (self.archive / f"{task}_optimism_gap.png").relative_to(
                self.root
            )
            self.assertEqual(after_files[archived], before_files[archived])
        metrics = json.loads((self.results / "metrics.json").read_text())
        for task in benchmark.TASKS:
            self.assertIsNone(metrics[task]["optimism_gap_pts"])
            self.assertEqual(metrics[task]["best_model"], "1D-CNN")

    def test_synthetic_run_publishes_only_to_demo_directories(self):
        self.patch("DEMO_DIR", self.root / "demo")
        before_files, _ = self.snapshot()

        def demo_run(args, results, figures, models):
            self.assertTrue(args.synthetic)
            (results / "metrics.json").write_text("{}")
            return []

        self.patch("_run_benchmark", side_effect=demo_run)
        with patch.object(
            benchmark.sys, "argv", ["run_experiment.py", "--synthetic", "--no-cnn"]
        ):
            benchmark.run()
        after_files, _ = self.snapshot()
        for path, content in before_files.items():
            self.assertEqual(after_files[path], content)
        self.assertEqual(
            (self.root / "demo" / "results" / "metrics.json").read_text(), "{}"
        )


def _read_module(path):
    text = path.read_text(encoding="utf-8")
    prefix, suffix = ("const data = ", ";\n\nexport default data;\n")
    assert text.startswith(prefix) and text.endswith(suffix)
    return json.loads(text[len(prefix) : -len(suffix)])


def _benchmark_fixture(protocol=1):
    metrics = {"benchmark_protocol_version": protocol}
    for task in ("binary", "multiclass"):
        metrics[task] = {
            "n_windows": 8,
            "n_features": 2,
            "n_subjects": 2,
            "classes": ["baseline", "stress"]
            + (["amusement"] if task == "multiclass" else []),
            "models": [
                {
                    "model": "Random Forest",
                    "accuracy_mean": 0.75,
                    "accuracy_std": 0.1,
                    "f1_macro_mean": 0.7,
                    "balanced_accuracy": 0.72,
                }
            ],
            "best_model": "Random Forest",
            "loso_accuracy": 0.75,
            "loso_pooled_accuracy": 0.75,
            "loso_matched_accuracy": 0.625,
            "within_subject_accuracy": 0.875,
            "optimism_gap_pts": 25.0,
        }
    return metrics


class DashboardDataTests(fixtures.TemporaryWorkspaceTestCase):
    def dashboard_output(self, path=None):
        output = path or self.tmp_path / "results.ts"
        self.patch_object(build_dashboard_data, "DASHBOARD_RESULTS", output)
        return output

    def patch_metrics(self, changes):
        load_json = build_dashboard_data._load_json

        def load(name):
            result = load_json(name)
            if name == "metrics.json":
                for path, value in changes.items():
                    target = result
                    for field in path[:-1]:
                        target = target[field]
                    target[path[-1]] = value
            return result

        self.patch_object(build_dashboard_data, "_load_json", load)

    def assert_export_rejected(self, output, message):
        before = output.read_text(encoding="utf-8")
        with self.assertRaisesRegex(ValueError, message):
            build_dashboard_data.run()
        self.assertEqual(output.read_text(encoding="utf-8"), before)

    def test_export_preserves_matched_metrics_and_normalizes_historical_brier(self):
        results_dir = self.tmp_path / "results"
        results_dir.mkdir()
        metrics = _benchmark_fixture()
        calibration = {
            name: {"ece": 0.1, "mce": 0.2, "brier": brier, "reliability": []}
            for name, brier in {
                "loso": 0.4,
                "loso_matched": 0.3,
                "within_subject": 0.1,
                "recalibrated_isotonic": 0.2,
                "recalibrated_sigmoid": 0.25,
            }.items()
        }
        calibration["gap_significance"] = {
            "mean_brier_gap": 0.2,
            "ci95": [0.1, 0.3],
            "per_subject": {"S2": {"loso": 0.3, "within": 0.1}},
        }
        saved_results = {
            "metrics.json": metrics,
            "calibration.json": calibration,
            "personalization.json": {"uncalibrated": {"ece": 0.1, "brier": 0.2}},
        }
        for name, value in saved_results.items():
            (results_dir / name).write_text(json.dumps(value))
        originals = {path: path.read_bytes() for path in results_dir.glob("*.json")}
        output = self.dashboard_output(self.tmp_path / "dashboard" / "results.ts")
        self.patch_object(build_dashboard_data, "RESULTS_DIR", results_dir)
        build_dashboard_data.run()
        exported = _read_module(output)
        for task in ("binary", "multiclass"):
            data = exported[task]
            self.assertEqual(
                data["loso_matched_accuracy"], metrics[task]["loso_matched_accuracy"]
            )
            self.assertEqual(
                round(
                    (data["within_subject_accuracy"] - data["loso_matched_accuracy"])
                    * 100,
                    1,
                ),
                data["optimism_gap_pts"],
            )
        self.assertEqual(
            exported["calibration"]["loso_matched"],
            {
                **calibration["loso_matched"],
                "brier": calibration["loso_matched"]["brier"] / 2,
            },
        )
        fixtures.assert_close(
            exported["calibration"]["loso"]["brier"], calibration["loso"]["brier"] / 2
        )
        self.assertEqual(
            exported["calibration"]["brier_definition"], BINARY_BRIER_DEFINITION
        )
        self.assertEqual(
            exported["personalization"]["brier_definition"], BINARY_BRIER_DEFINITION
        )
        self.assertEqual(
            exported["calibration"]["gap_significance"],
            {
                "mean_brier_gap": 0.1,
                "ci95": [0.05, 0.15],
                "per_subject": {"S2": {"loso": 0.15, "within": 0.05}},
            },
        )
        self.assertTrue(
            all((path.read_bytes() == before for path, before in originals.items()))
        )

    def _case_export_rejects_nonfinite_values_without_replacing_snapshot(
        self, invalid_number
    ):
        output = self.dashboard_output()
        before = "const data = {};\n\nexport default data;\n"
        output.write_text(before, encoding="utf-8")
        self.patch_object(
            build_dashboard_data,
            "_load_json",
            lambda name: (
                {"binary": {"n_windows": invalid_number}}
                if name == "metrics.json"
                else None
            ),
        )
        self.patch_object(build_dashboard_data, "_load_csv", lambda name: None)
        with self.assertRaisesRegex(ValueError, "Out of range float values"):
            build_dashboard_data.run()
        self.assertEqual(output.read_text(encoding="utf-8"), before)

    def test_signal_export_creates_output_folder_and_preserves_aligned_samples(self):
        samples = np.arange(6, dtype=float)
        recording = {
            "chest": {
                "ECG": samples,
                "EDA": samples + 10,
                "Temp": samples + 20,
                "ACC": np.column_stack((samples + 30, samples + 40, samples + 50)),
            },
            "label": np.repeat([1, 2, 3], 2),
        }
        loader = SimpleNamespace(load_subject=lambda subject: recording)
        output = self.tmp_path / "dashboard" / "signals.ts"
        self.patch_object(export_signals, "DASHBOARD_SIGNALS", output)
        self.patch_object(export_signals, "WESADLoader", lambda: loader)
        self.patch_object(export_signals, "FS", SimpleNamespace(CHEST=2))
        self.patch_object(export_signals, "SUBJECTS", ["S2"])
        self.patch_object(export_signals, "SECONDS", 1)
        self.patch_object(export_signals, "OUT_FS", 2)
        export_signals.run()
        data = _read_module(output)["S2"]
        self.assertEqual(data["time"], [0, 0.5, 1, 1.5, 2, 2.5])
        self.assertEqual(
            data["conditions"],
            ["Baseline", "Baseline", "Stress", "Stress", "Amusement", "Amusement"],
        )
        for name, offset in {
            "ecg": 0,
            "eda": 10,
            "temp": 20,
            "accX": 30,
            "accY": 40,
            "accZ": 50,
        }.items():
            self.assertEqual(data[name], (samples + offset).tolist())

    def test_signal_clips_do_not_span_separate_condition_blocks(self):
        labels = np.array([1, 1, 2, 1, 1])
        self.assertIs(
            export_signals._slice(np.arange(5), labels, label=1, want=3), None
        )
        labels = np.array([0, 1, 1, 1, 1, 1, 0])
        clip = export_signals._slice(np.arange(7), labels, label=1, want=3)
        self.assertEqual(clip.tolist(), [2, 3, 4])
        self.assertTrue(np.all(labels[clip] == 1))

    def test_signal_clips_reject_misaligned_samples(self):
        with self.assertRaisesRegex(ValueError, "lengths differ"):
            export_signals._slice(np.arange(4), np.ones(5), label=1, want=3)

    def test_export_rejects_partial_benchmark_without_replacing_snapshot(self):
        output = self.dashboard_output()
        output.write_text("existing snapshot", encoding="utf-8")
        self.patch_object(
            build_dashboard_data,
            "_load_json",
            lambda name: {} if name == "metrics.json" else None,
        )
        self.patch_object(build_dashboard_data, "_load_csv", lambda name: None)
        with self.assertRaisesRegex(ValueError, "non-empty model comparison"):
            build_dashboard_data.run()
        self.assertEqual(output.read_text(encoding="utf-8"), "existing snapshot")

    def test_export_copies_protocol_metadata_from_the_benchmark_run(self):
        output = self.dashboard_output()
        self.patch_metrics({("benchmark_protocol_version",): 2})
        build_dashboard_data.run()
        self.assertEqual(_read_module(output)["benchmark_protocol_version"], 2)

    def test_export_copies_recorded_subject_counts(self):
        output = self.dashboard_output()
        self.patch_metrics(
            {("binary", "n_subjects"): 3, ("multiclass", "n_subjects"): 4}
        )
        build_dashboard_data.run()
        exported = _read_module(output)
        self.assertEqual(exported["binary"]["n_subjects"], 3)
        self.assertEqual(exported["multiclass"]["n_subjects"], 4)

    def _case_export_rejects_invalid_subject_counts(self, count):
        output = self.dashboard_output()
        output.write_text("existing snapshot", encoding="utf-8")
        self.patch_metrics({("binary", "n_subjects"): count})
        self.assert_export_rejected(output, "n_subjects must be an integer >= 2")

    def _case_export_rejects_malformed_or_inconsistent_scores_before_replacing_snapshot(
        self, field_path, value, message
    ):
        output = self.dashboard_output()
        output.write_text("existing snapshot")
        self.patch_metrics({("binary", *field_path): value})
        self.assert_export_rejected(output, message)

    def test_task_validation_rejects_nonfinite_optimism_gaps(self):
        for gap in (np.nan, np.inf, -np.inf):
            with self.subTest(gap=gap):
                metrics = _benchmark_fixture()
                metrics["binary"]["optimism_gap_pts"] = gap
                with self.assertRaisesRegex(ValueError, "optimism gap must match"):
                    build_dashboard_data._validate_tasks(metrics)

    def test_export_accepts_absent_matched_comparison_without_inventing_scores(self):
        output = self.dashboard_output()
        keys = ("loso_matched_accuracy", "within_subject_accuracy", "optimism_gap_pts")
        self.patch_metrics({("binary", key): None for key in keys})
        build_dashboard_data.run()
        data = _read_module(output)["binary"]
        for key in keys:
            self.assertIs(data[key], None)

    def test_environment_snapshot_fingerprints_results_and_excludes_itself(self):
        (self.tmp_path / "metrics.json").write_bytes(b'{"binary": {}}')
        (self.tmp_path / "provenance.json").write_text("old snapshot", encoding="utf-8")
        self.patch_object(stamp_provenance, "RESULTS_DIR", self.tmp_path)
        stamp_provenance.run()
        snapshot = json.loads((self.tmp_path / "provenance.json").read_text())
        self.assertEqual(snapshot["provenance_kind"], "environment_snapshot")
        self.assertEqual(snapshot["data"]["n_reference_subjects"], 15)
        self.assertNotIn("n_subjects", snapshot["data"])
        self.assertEqual(
            snapshot["result_sha256"],
            {"metrics.json": hashlib.sha256(b'{"binary": {}}').hexdigest()},
        )
        self.assertEqual(
            (self.tmp_path / "metrics.json").read_bytes(), b'{"binary": {}}'
        )

    def _case_readme_metrics_reject_invalid_values(self, value):
        with self.assertRaisesRegex(ValueError, "README metrics must be finite"):
            update_readme_tables._format_metric(value)

    def test_readme_metrics_accept_valid_unit_interval_values(self):
        for value, expected in [(0, "0.000"), (1, "1.000"), (0.1234, "0.123")]:
            with self.subTest(value=value):
                self.assertEqual(update_readme_tables._format_metric(value), expected)

    def test_invalid_later_metrics_preserve_existing_readme(self):
        readme = self.tmp_path / "README.md"
        before = (
            "<!-- AUTOGEN:calibration START -->\nprevious calibration table\n"
            "<!-- AUTOGEN:calibration END -->\n"
            "<!-- AUTOGEN:personalization START -->\nprevious enrollment table\n"
            "<!-- AUTOGEN:personalization END -->\n"
        )
        readme.write_text(before)
        for name in ("calibration.json", "personalization.json"):
            data = json.loads((RESULTS_DIR / name).read_text())
            if name == "personalization.json":
                data["uncalibrated"]["ece"] = True
            (self.tmp_path / name).write_text(json.dumps(data))
        self.patch_object(update_readme_tables, "RESULTS", self.tmp_path)
        self.patch_object(update_readme_tables, "README", readme)
        with self.assertRaisesRegex(ValueError, "finite numeric values"):
            update_readme_tables.main()
        self.assertEqual(readme.read_text(), before)

    def test_personalization_table_rejects_an_unknown_brier_scale(self):
        (self.tmp_path / "personalization.json").write_text(
            json.dumps({"brier_definition": "unknown_scale"}), encoding="utf-8"
        )
        self.patch_object(update_readme_tables, "RESULTS", self.tmp_path)
        with self.assertRaisesRegex(ValueError, "positive-class MSE"):
            update_readme_tables._personalization_table()

    def test_new_primary_protocol_does_not_certify_saved_ancillary_runs(self):
        results_dir = self.tmp_path / "results"
        results_dir.mkdir()
        saved_results = {
            "metrics.json": _benchmark_fixture(protocol=2),
            "calibration.json": {"brier_definition": BINARY_BRIER_DEFINITION},
            "personalization.json": {"brier_definition": BINARY_BRIER_DEFINITION},
            **{
                f"{name}.json": {}
                for name in ("stats", "wrist", "cross_dataset", "tuning")
            },
        }
        for name, value in saved_results.items():
            (results_dir / name).write_text(json.dumps(value))
        (results_dir / "shap_top_features.csv").write_text(
            "feature,mean_abs_shap\nHRV_MeanNN,0.2\n"
        )
        (results_dir / "ablation.csv").write_text(
            "subset,n_features,accuracy_mean\nAll features,2,0.75\n"
        )
        output = self.dashboard_output()
        self.patch_object(build_dashboard_data, "RESULTS_DIR", results_dir)
        build_dashboard_data.run()
        self.assertEqual(
            _read_module(output)["unverified_sections"],
            [
                "ablation",
                "calibration",
                "cross_dataset",
                "personalization",
                "shap",
                "stats",
                "tuning",
                "wrist",
            ],
        )

    def _case_shap_link_requires_the_recorded_benchmark_artifact(self, fault):
        loader = build_dashboard_data._load_json

        def current_primary(name):
            value = loader(name)
            if name == "metrics.json":
                artifact = {
                    "model": "XGBoost",
                    "scope": "full_data_binary_fit",
                    "path": "shap_top_features.csv",
                    "sha256": sha256_file(RESULTS_DIR / "shap_top_features.csv"),
                }
                if fault:
                    artifact["sha256" if fault == "checksum" else fault] = "incorrect"
                value["benchmark_protocol_version"] = 2
                value["artifacts"] = {"shap": artifact}
            return value

        output = self.dashboard_output()
        output.write_text("previous snapshot")
        self.patch_object(build_dashboard_data, "_load_json", current_primary)
        if fault:
            with self.assertRaisesRegex(ValueError, "SHAP snapshot does not match"):
                build_dashboard_data.run()
            self.assertEqual(output.read_text(), "previous snapshot")
        else:
            build_dashboard_data.run()
            self.assertNotIn("shap", _read_module(output)["unverified_sections"])


class BenchmarkLinkageTests(fixtures.PatchedTestCase):

    def setUp(self):
        self.root = Path(self.enterContext(TemporaryDirectory()))
        self.cache = self.root / "cache"
        self.cache.mkdir()
        (self.cache / "features.parquet").write_bytes(b"feature cache")
        (self.cache / "raw_windows.npz").write_bytes(b"raw windows")
        self.enterContext(patch("src.config.PROCESSED_DATA_DIR", self.cache))
        self.frame = pd.DataFrame({"subject": ["S2", "S3"], "value": [1.0, 2.0]})
        self.frame.attrs["feature_schema_version"] = 2
        self.metrics = {
            "benchmark_protocol_version": 2,
            "provenance": {
                "git_sha": "a" * 40,
                "source_file_sha256": pipeline_source_sha256(),
            },
            "inputs": {
                "dataset": "WESAD",
                "feature_frame_sha256": feature_frame_sha256(self.frame),
                "feature_cache_sha256": sha256_file(self.cache / "features.parquet"),
                "raw_windows_sha256": sha256_file(self.cache / "raw_windows.npz"),
            },
        }
        for task in ("binary", "multiclass"):
            self.metrics[task] = {
                "feature_schema_version": 2,
                "n_windows": 3,
                "n_features": 1,
                "n_subjects": 2,
                "classes": ["baseline", "stress"]
                + (["amusement"] if task == "multiclass" else []),
                "models": [
                    {
                        "model": "Logistic Regression",
                        "accuracy_mean": 0.9,
                        "accuracy_std": 0.1,
                        "f1_macro_mean": 0.8,
                        "balanced_accuracy": 0.85,
                    }
                ],
                "best_model": "Logistic Regression",
                "inference_model": "Logistic Regression",
                "loso_accuracy": 0.9,
                "loso_pooled_accuracy": 0.88,
            }
        self._write_metrics()

    def _write_metrics(self):
        (self.root / "metrics.json").write_text(json.dumps(self.metrics))

    def _bind(self):
        reference = benchmark_reference(self.root, self.frame)
        return analysis_provenance(self.root, reference)

    def _export(self):
        output = self.root / "results.ts"
        with patch.object(build_dashboard_data, "RESULTS_DIR", self.root), patch.object(
            build_dashboard_data, "DASHBOARD_RESULTS", output
        ):
            build_dashboard_data.run()
        return _read_module(output)

    def test_frame_identity_covers_values_order_and_schema(self):
        expected = feature_frame_sha256(self.frame)
        self.assertEqual(expected, feature_frame_sha256(self.frame.copy()))
        changed = self.frame.copy()
        changed.loc[0, "value"] = 3.0
        self.assertNotEqual(expected, feature_frame_sha256(changed))
        self.assertNotEqual(expected, feature_frame_sha256(self.frame.iloc[::-1]))
        changed = self.frame.copy()
        changed.attrs["feature_schema_version"] = 1
        self.assertNotEqual(expected, feature_frame_sha256(changed))

    def test_shared_inputs_require_the_recorded_frame_and_cache(self):
        context = self._bind()
        self.assertEqual(
            context["primary_benchmark_sha256"], sha256_file(self.root / "metrics.json")
        )
        self.assertEqual(
            context["source_file_sha256"],
            self.metrics["provenance"]["source_file_sha256"],
        )
        changed = self.frame.copy()
        changed.loc[0, "value"] = 3.0
        with self.assertRaisesRegex(ValueError, "feature frame differs"):
            benchmark_reference(self.root, changed)
        (self.cache / "raw_windows.npz").write_bytes(b"changed raw windows")
        with self.assertRaisesRegex(ValueError, "cache differs"):
            benchmark_reference(self.root, self.frame)
        self.assertEqual(
            benchmark_reference(self.root, shared_cache=False),
            context["primary_benchmark_sha256"],
        )

    def test_historical_synthetic_and_incomplete_contexts_remain_unverified(self):
        cases = [
            ("benchmark_protocol_version", None),
            ("inputs", {**self.metrics["inputs"], "dataset": "synthetic"}),
            ("inputs", {"dataset": "WESAD"}),
            ("provenance", {"git_sha": "a" * 40}),
            ("provenance", {**self.metrics["provenance"], "git_sha": "unknown"}),
        ]
        for key, value in cases:
            with self.subTest(key=key, value=value):
                original = self.metrics[key]
                self.metrics[key] = value
                self._write_metrics()
                self.assertIsNone(benchmark_reference(self.root, self.frame))
                self.assertNotIn(
                    "primary_benchmark_sha256", analysis_provenance(self.root, None)
                )
                self.metrics[key] = original
        self._write_metrics()

    def test_primary_source_mismatch_prevents_linking(self):
        self.metrics["provenance"]["source_file_sha256"]["src/utils.py"] = "0" * 64
        self._write_metrics()
        with self.assertRaisesRegex(ValueError, "source differs"):
            benchmark_reference(self.root, self.frame)

    def test_export_preserves_recorded_source_while_new_analysis_rejects_it(self):
        context = self._bind()
        (self.root / "stats.json").write_text(json.dumps({"provenance": context}))
        original = (self.root / "metrics.json").read_bytes()
        changed = {**pipeline_source_sha256(), "src/utils.py": "0" * 64}
        with patch(
            "src.utils.pipeline_source_sha256", return_value=changed
        ), patch.object(
            build_dashboard_data, "pipeline_source_sha256", return_value=changed
        ):
            data = self._export()
            self.assertEqual(data["unverified_sections"], [])
            self.assertEqual(data["benchmark_provenance"], self.metrics["provenance"])
            self.assertIs(data["benchmark_source_matches_current"], False)
            with self.assertRaisesRegex(ValueError, "source differs"):
                benchmark_reference(self.root, self.frame)
            with self.assertRaisesRegex(ValueError, "source differs"):
                analysis_provenance(self.root, context["primary_benchmark_sha256"])
        self.assertEqual((self.root / "metrics.json").read_bytes(), original)

    def test_recorded_reference_rejects_current_inputs_and_malformed_source_hashes(
        self,
    ):
        for kwargs in ({}, {"shared_cache": False, "frame": self.frame}):
            with self.subTest(kwargs=kwargs), self.assertRaisesRegex(
                ValueError, "cannot validate current inputs"
            ):
                benchmark_reference(self.root, recorded=True, **kwargs)
        context = self._bind()
        output = self.root / "results.ts"
        for sources in ({}, {"src/utils.py": "not a hash"}, {"../utils.py": "a" * 64}):
            with self.subTest(sources=sources):
                self.metrics["provenance"]["source_file_sha256"] = sources
                self._write_metrics()
                saved_context = {**context, "source_file_sha256": sources}
                saved_context["primary_benchmark_sha256"] = sha256_file(
                    self.root / "metrics.json"
                )
                (self.root / "stats.json").write_text(
                    json.dumps({"provenance": saved_context})
                )
                output.write_text("previous snapshot")
                with self.assertRaisesRegex(ValueError, "does not match"):
                    self._export()
                self.assertEqual(output.read_text(), "previous snapshot")

    def test_primary_changes_during_analysis_are_rejected(self):
        reference = benchmark_reference(self.root, self.frame)
        self.metrics["binary"]["n_windows"] = 4
        self._write_metrics()
        with self.assertRaisesRegex(ValueError, "changed during the analysis"):
            analysis_provenance(self.root, reference)

    def test_export_accepts_verified_json_csv_and_shap_links(self):
        shap = self.root / "shap_top_features.csv"
        shap.write_text("feature,mean_abs_shap\nvalue,0.1\n")
        self.metrics["artifacts"] = {
            "shap": {
                "model": "XGBoost",
                "scope": "full_data_binary_fit",
                "path": shap.name,
                "sha256": sha256_file(shap),
            }
        }
        self._write_metrics()
        context = self._bind()
        for filename in (
            "stats.json",
            "wrist.json",
            "cross_dataset.json",
            "tuning.json",
        ):
            (self.root / filename).write_text(json.dumps({"provenance": context}))
        pd.DataFrame(
            [
                {
                    "subset": "ECG",
                    "accuracy": 0.9,
                    "benchmark_sha256": context["primary_benchmark_sha256"],
                }
            ]
        ).to_csv(self.root / "ablation.csv", index=False)
        for path in self.cache.iterdir():
            path.unlink()
        data = self._export()
        self.assertEqual(data["unverified_sections"], [])
        self.assertEqual(data["shap_model"], "XGBoost")
        self.assertEqual(data["shap_scope"], "full_data_binary_fit")
        self.assertEqual(data["binary"]["inference_model"], "Logistic Regression")
        self.assertIs(data["benchmark_source_matches_current"], True)
        self.assertNotIn("benchmark_sha256", data["ablation"][0])

    def test_stale_and_mixed_links_preserve_the_existing_dashboard(self):
        output = self.root / "results.ts"
        for fault in ("json_reference", "json_source", "csv_reference", "csv_mixed"):
            with self.subTest(fault=fault):
                output.write_text("previous snapshot")
                context = self._bind()
                if fault == "json_reference":
                    context["primary_benchmark_sha256"] = "0" * 64
                elif fault == "json_source":
                    context["source_file_sha256"] = {}
                if fault.startswith("json"):
                    (self.root / "stats.json").write_text(
                        json.dumps({"provenance": context})
                    )
                else:
                    references = (
                        ["0" * 64]
                        if fault == "csv_reference"
                        else [context["primary_benchmark_sha256"], None]
                    )
                    pd.DataFrame(
                        [
                            {"subset": "ECG", "benchmark_sha256": reference}
                            for reference in references
                        ]
                    ).to_csv(self.root / "ablation.csv", index=False)
                with self.assertRaisesRegex(ValueError, "does not match"):
                    self._export()
                self.assertEqual(output.read_text(), "previous snapshot")
                (self.root / "stats.json").unlink(missing_ok=True)
                (self.root / "ablation.csv").unlink(missing_ok=True)


fixtures.add_cases(
    DashboardDataTests,
    "export_rejects_nonfinite_values_without_replacing_snapshot",
    [(float("nan"),), (float("inf"),), (-float("inf"),)],
)


fixtures.add_cases(
    DashboardDataTests,
    "export_rejects_invalid_subject_counts",
    [(None,), (0,), (1,), (2.5,), ("15",)],
)


fixtures.add_cases(
    DashboardDataTests,
    "export_rejects_malformed_or_inconsistent_scores_before_replacing_snapshot",
    [
        (("n_windows",), True, "positive integer"),
        (("n_features",), False, "positive integer"),
        (("classes",), ["stress", "baseline"], "classes must be"),
        (("models", 0, "model"), "", "nonempty strings"),
        (("models", 1, "model"), "Logistic Regression", "must be unique"),
        (("models", 0, "accuracy_mean"), "0.9", "numeric value"),
        (("models", 0, "f1_macro_mean"), 1.2, "numeric value"),
        (("models", 0, "balanced_accuracy"), -0.1, "numeric value"),
        (("models", 0, "accuracy_std"), None, "numeric value"),
        (("loso_accuracy",), 0.1, "disagrees with"),
        (("optimism_gap_pts",), 4.0, "must match"),
        (("loso_matched_accuracy",), None, "must match"),
    ],
)


fixtures.add_cases(
    DashboardDataTests,
    "readme_metrics_reject_invalid_values",
    [
        (float("nan"),),
        (float("inf"),),
        (-float("inf"),),
        (True,),
        (False,),
        (-0.1,),
        (1.2,),
        ("0.5",),
        (None,),
    ],
)


fixtures.add_cases(
    DashboardDataTests,
    "shap_link_requires_the_recorded_benchmark_artifact",
    [(None,), ("checksum",), ("model",), ("scope",), ("path",)],
)


ROOT = Path(__file__).resolve().parent.parent


README = (ROOT / "README.md").read_text()


PHANTOM_MODELS = [
    "Transformer",
    "BiLSTM",
    "CatBoost",
    "EfficientNet",
    "cross-modal attention",
]


class DocsTests(fixtures.TemporaryWorkspaceTestCase):

    def test_readme_names_no_untrained_models(self):
        lower = README.lower()
        found = [name for name in PHANTOM_MODELS if name.lower() in lower]
        self.assertFalse(
            found, f"README names models the project never trains: {found}"
        )

    def test_docs_make_no_best_overall_ranking_claim(self):
        self.assertNotIn(
            "best overall",
            README.lower(),
            "README makes a 'Best overall' ranking claim",
        )

    def test_readme_local_links_and_images_exist(self):
        refs = re.findall(r"\]\(([^\s)]+)\)", README) + re.findall(
            '(?:src|href)="([^"]+)"', README
        )
        for ref in refs:
            url = urlsplit(ref)
            if url.scheme or not url.path:
                continue
            self.assertTrue(
                (ROOT / unquote(url.path)).exists(), f"Missing README asset: {ref}"
            )

    def test_readme_table_update_is_idempotent(self):
        from scripts import update_readme_tables as tables

        path = self.tmp_path / "README.md"
        path.write_text(README)
        self.patch_object(tables, "README", path)
        tables.main()
        self.assertEqual(path.read_text(), README)
        tables.main()
        self.assertEqual(path.read_text(), README)

    def test_readme_table_update_rejects_missing_markers(self):
        from scripts.update_readme_tables import _replace

        with self.assertRaisesRegex(SystemExit, "not found in README.md"):
            _replace("overview without generated tables", "calibration", "replacement")

    def test_referenced_notebooks_exist(self):
        refs = sorted(set(re.findall(r"notebooks/[\w./-]+\.ipynb", README)))
        missing = [ref for ref in refs if not (ROOT / ref).exists()]
        self.assertFalse(
            missing, f"Docs reference notebooks that do not exist: {missing}"
        )

    def test_no_nonexistent_numbered_notebook_series(self):
        self.assertFalse(
            re.search("notebooks/0[1-8]_", README),
            "README references a nonexistent numbered 01-08 notebook series",
        )


class LoggingTests(fixtures.TemporaryWorkspaceTestCase):
    def setUp(self):
        super().setUp()
        self.project = logging.getLogger("calmsense")
        self.saved = (
            self.project.handlers[:],
            self.project.level,
            self.project.propagate,
        )
        self.project.handlers = []
        self.addCleanup(self.restore_logging)
        self.patch_object(logging_config, "_configured_settings", None)

    def restore_logging(self):
        for handler in self.project.handlers:
            handler.close()
        self.project.handlers, self.project.level, self.project.propagate = self.saved

    def test_logging_reconfigures_destinations_without_resetting_host(self):
        host = logging.getLogger()
        host_handler = logging.NullHandler()
        host.addHandler(host_handler)
        self.addCleanup(host.removeHandler, host_handler)
        structlog_before = structlog.get_config().copy()
        first, second = self.tmp_path / "first.log", self.tmp_path / "second.log"
        logging_config.setup_logging(log_file=first, console=False)
        logger = logging_config.get_logger("src.test")
        logger.info("first", value=1)
        old_handler = self.project.handlers[0]
        logging_config.setup_logging(log_file=second, console=False)
        logger.info("second", value=2)
        self.assertIs(old_handler.stream, None)
        self.assertEqual(json.loads(first.read_text())["event"], "first")
        self.assertEqual(json.loads(second.read_text())["event"], "second")
        self.assertIn(host_handler, host.handlers)
        self.assertEqual(structlog.get_config(), structlog_before)

    def test_invalid_logging_level_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unknown logging level"):
            logging_config.setup_logging(level="invalid", log_file=None)

    def test_failed_reconfiguration_keeps_the_working_logger(self):
        destination = self.tmp_path / "working.log"
        blocker = self.tmp_path / "file-not-directory"
        blocker.write_text("keep")
        logging_config.setup_logging(log_file=destination, console=False)
        handler = self.project.handlers[0]
        settings = logging_config._configured_settings
        logger = logging_config.get_logger("src.test")
        logger.info("before")
        with self.assertRaises(OSError):
            logging_config.setup_logging(level="DEBUG", log_file=blocker / "new.log")
        self.assertEqual(self.project.handlers, [handler])
        self.assertEqual(self.project.level, logging.INFO)
        self.assertEqual(logging_config._configured_settings, settings)
        self.assertIsNot(handler.stream, None)
        logger.info("after failure")
        logging_config.setup_logging(log_file=destination, console=False)
        logger.info("after retry")
        self.assertEqual(
            [
                json.loads(line)["event"]
                for line in destination.read_text().splitlines()
            ],
            ["before", "after failure", "after retry"],
        )


class CalibrationPlotTests(fixtures.PatchedTestCase):
    def tearDown(self):
        plt.close("all")

    def calibration_data(self, model="rf"):
        summary = {
            "ece": 0.35,
            "reliability": [
                {"confidence": 0.6, "accuracy": 0.0},
                {"confidence": 0.9, "accuracy": 1.0},
            ],
        }
        return {
            "model": model,
            "n_windows": 4,
            "n_bins": 2,
            "loso": summary,
            "recalibrated_isotonic": summary,
            "decision_curve": {
                "thresholds": [0.1, 0.6],
                "net_benefit_uncalibrated": [0.2, -0.1],
                "net_benefit_recalibrated": [0.3, 0.1],
                "treat_all": [0.4, -0.25],
            },
        }

    def test_calibration_labels_the_recorded_model(self):
        for model, label in (
            ("lr", "Logistic Regression"),
            ("rf", "Random Forest"),
            ("xgb", "XGBoost"),
            ("lgbm", "LightGBM"),
            ("custom model", "custom model"),
        ):
            with self.subTest(model=model):
                figure = plot_calibration(self.calibration_data(model))
                self.assertTrue(figure.texts[1].get_text().startswith(f"{label} |"))

    def test_calibration_does_not_invent_an_unrecorded_model(self):
        data = self.calibration_data()
        data.pop("model")
        figure = plot_calibration(data)
        self.assertIn("Model not recorded", figure.texts[1].get_text())

    def test_reliability_plot_shows_low_accuracy_bins_and_full_unit_range(self):
        figure = plot_calibration(self.calibration_data())
        axis = figure.axes[0]
        lower, upper = axis.get_ylim()
        self.assertLessEqual(lower, 0)
        self.assertGreaterEqual(upper, 1)
        assert_array_equal(axis.lines[1].get_ydata(), [0.0, 1.0])


class RocPlotTests(fixtures.PatchedTestCase):
    def tearDown(self):
        plt.close("all")

    def test_scores_without_points_never_create_a_curve(self):
        data = {
            "n_windows": 4,
            "models": [{"model": "Example", "available": True, "auroc": 0.75}],
        }
        fig = plot_roc_auc(data)
        self.assertFalse(fig.axes[0].lines)
        self.assertTrue(
            any(
                "ROC points were not saved" in text.get_text()
                for text in fig.axes[0].texts
            )
        )
        with TemporaryDirectory() as directory:
            path = Path(directory) / "roc-auc.svg"
            fig.savefig(path)
            self.assertIn("ROC points were not saved", path.read_text())

    def test_curve_uses_actual_rates_from_known_binary_labels(self):
        labels = [0, 0, 1, 1]
        probabilities = [0.1, 0.4, 0.35, 0.8]
        fpr, tpr, _ = roc_curve(labels, probabilities)
        model = {
            "model": "Test fixture",
            "available": True,
            "auroc": roc_auc_score(labels, probabilities),
            "roc_curve": {"fpr": fpr.tolist(), "tpr": tpr.tolist()},
        }
        fig = plot_roc_auc({"n_windows": len(labels), "models": [model]})
        line = fig.axes[0].lines[1]
        assert_array_equal(line.get_xdata(), [0, 0, 0.5, 0.5, 1])
        assert_array_equal(line.get_ydata(), [0, 0.5, 0.5, 1, 1])
        self.assertAlmostEqual(model["auroc"], 0.75)

    def test_malformed_or_inconsistent_curves_are_rejected(self):
        invalid = [
            {"fpr": [0, 1], "tpr": [0]},
            {"fpr": [0, -0.1, 1], "tpr": [0, 0.5, 1]},
            {"fpr": [0, 0.5, 1], "tpr": [0, np.nan, 1]},
            {"fpr": [0, 0.8, 0.5, 1], "tpr": [0, 0.5, 0.5, 1]},
            {"fpr": [0, 1], "tpr": [0, 1]},
        ]
        for curve in invalid:
            with self.subTest(curve=curve):
                model = {"model": "Example", "auroc": 0.75, "roc_curve": curve}
                with self.assertRaisesRegex(ValueError, "ROC points"):
                    _roc_points(model)

    def test_supplemental_curves_require_current_artifact_and_matching_scores(self):
        current = {
            "task": "binary",
            "n_windows": 4,
            "models": [
                {"model": "Example", "available": True, "auroc": 0.75, "auprc": 5 / 6}
            ],
        }
        supplemental = {
            **current,
            "models": [
                {
                    **current["models"][0],
                    "roc_curve": {
                        "fpr": [0, 0, 0.5, 0.5, 1],
                        "tpr": [0, 0.5, 0.5, 1, 1],
                    },
                }
            ],
        }
        with TemporaryDirectory() as directory:
            path = Path(directory)
            current_path = path / "threshold_metrics.json"
            supplemental_path = path / "roc_curves.json"
            current_path.write_text(json.dumps(current))
            supplemental["provenance"] = {
                "threshold_metrics_sha256": hashlib.sha256(
                    current_path.read_bytes()
                ).hexdigest()
            }
            supplemental_path.write_text(json.dumps(supplemental))
            self.assertEqual(_roc_source(path), "roc_curves.json")

            current["models"][0]["auroc"] = 0.6
            current_path.write_text(json.dumps(current))
            self.assertEqual(_roc_source(path), "threshold_metrics.json")

            supplemental["provenance"]["threshold_metrics_sha256"] = hashlib.sha256(
                current_path.read_bytes()
            ).hexdigest()
            supplemental_path.write_text(json.dumps(supplemental))
            self.assertEqual(_roc_source(path), "threshold_metrics.json")

            current["models"] = supplemental["models"]
            current_path.write_text(json.dumps(current))
            self.assertEqual(_roc_source(path), "threshold_metrics.json")


class ChartRefreshTests(fixtures.PatchedTestCase):
    sources = (
        "metrics.json",
        "calibration.json",
        "threshold_metrics.json",
        "cross_dataset.json",
        "binary_per_subject.csv",
        "multiclass_per_subject.csv",
        "ablation.csv",
        "wrist.json",
        "personalization.json",
        "shap_top_features.csv",
    )
    chart_names = (
        "research-overview.svg",
        "calibration-curves.svg",
        "roc-auc.svg",
        "transfer.svg",
        "subject-results.svg",
        "validation-gap.svg",
        "sensor-analysis.svg",
        "personalization-results.svg",
        "feature-contributions.svg",
    )

    def setUp(self):
        self.addCleanup(plt.close, "all")
        directory = Path(self.enterContext(TemporaryDirectory()))
        self.results = directory / "results"
        self.output = directory / "assets"
        self.results.mkdir()
        self.output.mkdir()
        for source in self.sources:
            shutil.copyfile(RESULTS_DIR / source, self.results / source)
        for name in self.chart_names:
            (self.output / name).write_text("Previous " + name)
        (self.output / "pipeline.svg").write_text("Existing pipeline")
        (self.output / "notes.txt").write_text("Unmanaged file")
        self.plotters = {
            name: self.patch_object(charts, name, side_effect=self.draw_chart)
            for name in (
                "plot_models",
                "plot_calibration",
                "plot_roc_auc",
                "plot_transfer",
                "plot_subject_results",
                "plot_validation_gap",
                "plot_sensor_analysis",
                "plot_personalization",
                "plot_feature_contributions",
            )
        }

    @staticmethod
    def draw_chart(*data):
        figure = plt.figure(figsize=(2, 1))
        figure.text(0.1, 0.5, "Fixture chart")
        return figure

    def snapshot(self):
        return {
            path.name: (path.read_bytes(), S_IMODE(path.stat().st_mode))
            for path in self.output.iterdir()
        }

    def assert_run_failure(self, exception, message):
        before = self.snapshot()
        with self.assertRaisesRegex(exception, message):
            charts.refresh_charts(self.results, self.output)
        self.assertEqual(self.snapshot(), before)

    def test_missing_source_preserves_all_charts(self):
        (self.results / "calibration.json").unlink()
        before = self.snapshot()
        with self.assertRaises(FileNotFoundError):
            charts.refresh_charts(self.results, self.output)
        self.assertEqual(self.snapshot(), before)
        for plotter in self.plotters.values():
            plotter.assert_not_called()

    def test_malformed_late_source_is_rejected_before_rendering(self):
        (self.results / "cross_dataset.json").write_text("{")
        before = self.snapshot()
        with self.assertRaises(json.JSONDecodeError):
            charts.refresh_charts(self.results, self.output)
        self.assertEqual(self.snapshot(), before)
        for plotter in self.plotters.values():
            plotter.assert_not_called()

    def test_late_plot_failure_preserves_files_and_closes_owned_figures(self):
        external = plt.figure()

        def broken_plot(*data):
            plt.figure()
            raise ValueError("Invalid SHAP results")

        self.plotters["plot_feature_contributions"].side_effect = broken_plot
        self.assert_run_failure(ValueError, "Invalid SHAP results")
        self.assertEqual(plt.get_fignums(), [external.number])

    def test_late_save_failure_preserves_all_charts(self):
        savefig = Figure.savefig

        def fail_transfer(figure, destination, **kwargs):
            if Path(destination).name == "feature-contributions.svg":
                raise OSError("Cannot write SHAP chart")
            savefig(figure, destination, **kwargs)

        with patch.object(Figure, "savefig", fail_transfer):
            self.assert_run_failure(OSError, "Cannot write SHAP chart")
        self.assertEqual(plt.get_fignums(), [])

    def test_publication_failure_restores_replaced_and_new_files(self):
        (self.output / "calibration-curves.svg").unlink()
        (self.output / "research-overview.svg").chmod(0o640)
        replace = Path.replace

        def fail_transfer(source, destination):
            if destination == self.output / "feature-contributions.svg":
                raise OSError("Cannot publish SHAP chart")
            return replace(source, destination)

        with patch.object(Path, "replace", fail_transfer):
            self.assert_run_failure(OSError, "Cannot publish SHAP chart")
        self.assertEqual(plt.get_fignums(), [])

    def test_success_replaces_only_managed_charts_and_preserves_permissions(self):
        (self.output / "research-overview.svg").chmod(0o640)
        before = self.snapshot()
        charts.refresh_charts(self.results, self.output)
        after = self.snapshot()
        self.assertEqual(after.keys(), before.keys())
        for name in self.chart_names:
            self.assertIn(b"<svg", after[name][0])
            self.assertNotEqual(after[name][0], before[name][0])
            self.assertEqual(after[name][1], before[name][1])
        for name in ("pipeline.svg", "notes.txt"):
            self.assertEqual(after[name], before[name])
        self.assertEqual(plt.get_fignums(), [])

    def test_new_sources_are_required_before_rendering(self):
        for name in (
            "binary_per_subject.csv",
            "ablation.csv",
            "personalization.json",
            "shap_top_features.csv",
        ):
            with self.subTest(source=name):
                payload = (self.results / name).read_bytes()
                (self.results / name).unlink()
                self.assert_run_failure(FileNotFoundError, name)
                (self.results / name).write_bytes(payload)
        for plotter in self.plotters.values():
            plotter.assert_not_called()

    def test_mixed_study_or_shap_sources_are_rejected_before_rendering(self):
        for name in (
            "ablation.csv",
            "wrist.json",
            "personalization.json",
            "shap_top_features.csv",
        ):
            with self.subTest(source=name):
                path = self.results / name
                payload = path.read_bytes()
                if name.endswith(".json"):
                    data = json.loads(payload)
                    data["provenance"]["primary_benchmark_sha256"] = "0" * 64
                    path.write_text(json.dumps(data))
                elif name == "ablation.csv":
                    frame = pd.read_csv(path)
                    frame["benchmark_sha256"] = "0" * 64
                    frame.to_csv(path, index=False)
                else:
                    path.write_bytes(payload + b"Other,0.5\n")
                self.assert_run_failure(ValueError, "does not match the benchmark")
                path.write_bytes(payload)
        for plotter in self.plotters.values():
            plotter.assert_not_called()

    def test_result_changes_during_rendering_never_publish(self):
        def changed(*data):
            path = self.results / "binary_per_subject.csv"
            path.write_bytes(path.read_bytes() + b"\n")
            return self.draw_chart(*data)

        self.plotters["plot_feature_contributions"].side_effect = changed
        self.assert_run_failure(RuntimeError, "sources changed")
        self.assertEqual(plt.get_fignums(), [])


class RecordedChartTests(fixtures.TemporaryWorkspaceTestCase):
    def setUp(self):
        super().setUp()
        self.addCleanup(plt.close, "all")
        self.metrics = self.load("metrics.json")

    @staticmethod
    def load(name):
        path = RESULTS_DIR / name
        return (
            pd.read_csv(path) if name.endswith(".csv") else json.loads(path.read_text())
        )

    def test_subject_points_keep_each_recorded_person_and_mean(self):
        frames = [
            self.load(task + "_per_subject.csv") for task in ("binary", "multiclass")
        ]
        figure = charts.plot_subject_results(self.metrics, *frames)
        for task, axis, frame in zip(("binary", "multiclass"), figure.axes, frames):
            ordered = frame.iloc[np.argsort(frame.subject.str[1:].astype(int))]
            np.testing.assert_allclose(
                axis.collections[1].get_offsets()[:, 0], ordered.accuracy * 100
            )
            if axis.get_yticklabels():
                self.assertEqual(
                    [tick.get_text() for tick in axis.get_yticklabels()],
                    ordered.subject.tolist(),
                )
            self.assertAlmostEqual(
                axis.lines[0].get_xdata()[0], self.metrics[task]["loso_accuracy"] * 100
            )
        frames[0].loc[0, "accuracy"] = np.nan
        self.assertRaisesRegex(
            ValueError,
            "finite values",
            charts.plot_subject_results,
            self.metrics,
            *frames,
        )

    def test_validation_bars_use_matched_pooled_scores(self):
        figure = charts.plot_validation_gap(self.metrics)
        for task, axis in zip(("binary", "multiclass"), figure.axes):
            expected = [
                self.metrics[task][key] * 100
                for key in ("loso_matched_accuracy", "within_subject_accuracy")
            ]
            np.testing.assert_allclose(
                [bar.get_width() for bar in axis.patches], expected
            )
        self.metrics["binary"]["optimism_gap_pts"] += 1
        self.assertRaisesRegex(
            ValueError, "gap disagrees", charts.plot_validation_gap, self.metrics
        )

    def test_sensor_panels_keep_ablation_and_same_classifier_placement_separate(self):
        ablation, wrist = self.load("ablation.csv"), self.load("wrist.json")
        figure = charts.plot_sensor_analysis(ablation, wrist)
        np.testing.assert_allclose(
            [bar.get_width() for bar in figure.axes[0].patches],
            ablation.accuracy_mean * 100,
        )
        np.testing.assert_allclose(
            [bar.get_width() for bar in figure.axes[1].patches],
            [wrist["same_model_rf"][key] * 100 for key in ("chest", "wrist")],
        )
        self.assertIn("separate study", figure.texts[-2].get_text())
        wrist["same_model_rf"]["drop_pts"] += 1
        self.assertRaisesRegex(
            ValueError,
            "difference disagrees",
            charts.plot_sensor_analysis,
            ablation,
            wrist,
        )

    def test_personalization_keeps_probability_metrics_and_requested_budgets(self):
        data = self.load("personalization.json")
        figure = charts.plot_personalization(data)
        variants = [
            data["uncalibrated"],
            data["global"],
            *[data["fewshot"][str(k)] for k in data["k_values"]],
        ]
        for metric, axis in zip(("ece", "brier"), figure.axes):
            np.testing.assert_allclose(
                [bar.get_width() for bar in axis.patches],
                [row[metric] for row in variants],
            )
        self.assertIn("at most 4 windows", figure.texts[-1].get_text())
        data["provenance"]["git_sha"] = "a" * 40
        current = charts.plot_personalization(data)
        self.assertNotIn("at most 4", current.texts[-1].get_text())
        data["brier_definition"] = "two_class_sum"
        self.assertRaisesRegex(ValueError, "Brier", charts.plot_personalization, data)

    def test_feature_contributions_use_recorded_top_ten_without_causal_claims(self):
        data = self.load("shap_top_features.csv")
        figure = charts.plot_feature_contributions(data, self.metrics)
        np.testing.assert_allclose(
            [bar.get_width() for bar in figure.axes[0].patches],
            sorted(data.mean_abs_shap, reverse=True)[:10],
        )
        self.assertIn("not held-out accuracy", figure.texts[-1].get_text())
        self.metrics["artifacts"]["shap"]["scope"] = "held_out"
        self.assertRaisesRegex(
            ValueError,
            "full-data",
            charts.plot_feature_contributions,
            data,
            self.metrics,
        )

    def test_refresh_records_sources_scope_and_plot_values_in_new_svgs(self):
        charts.refresh_charts(RESULTS_DIR, self.tmp_path)
        for name in (
            "subject-results.svg",
            "validation-gap.svg",
            "sensor-analysis.svg",
            "personalization-results.svg",
            "feature-contributions.svg",
        ):
            with self.subTest(chart=name):
                root = ET.parse(self.tmp_path / name).getroot()
                description = root.find(
                    ".//{http://purl.org/dc/elements/1.1/}description"
                )
                self.assertIsNotNone(description)
                metadata = json.loads(description.text)
                self.assertEqual(
                    metadata["benchmark_source"], self.metrics["provenance"]["git_sha"]
                )
                self.assertTrue(metadata["plotted_data"])
                self.assertEqual(len(metadata["scope"]), 2)
                for path, digest in metadata["source_files"].items():
                    self.assertEqual(digest, sha256_file(RESULTS_DIR / Path(path).name))
                self.assertFalse(
                    any(
                        element.tag.rsplit("}", 1)[-1]
                        in {"script", "foreignObject", "image"}
                        for element in root.iter()
                    )
                )


class UtilsTests(fixtures.TemporaryWorkspaceTestCase):

    def test_provenance_has_sha_and_timestamp(self):
        p = provenance()
        self.assertEqual(
            set(p),
            {"git_sha", "working_tree_dirty", "feature_schema_version", "generated_at"},
        )
        self.assertTrue(isinstance(p["git_sha"], str) and len(p["git_sha"]) >= 7)
        self.assertIn("T", p["generated_at"])
        self.assertTrue(isinstance(p["working_tree_dirty"], bool))
        self.assertEqual(p["feature_schema_version"], 2)

    def test_exported_source_does_not_borrow_parent_repository_provenance(self):
        repo = Path(utils.__file__).resolve().parents[1]
        generated = repo / "outputs" / "generated"
        generated.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(dir=generated) as directory:
            source = Path(directory) / "src" / "utils.py"
            with patch.object(utils, "__file__", str(source)):
                result = provenance()
        self.assertEqual(result["git_sha"], "unknown")
        self.assertIsNone(result["working_tree_dirty"])

    def test_paired_effect_size_direction_and_small_sample_correction(self):
        a = [0.9, 0.8, 0.95, 0.7]
        b = [0.85, 0.78, 0.9, 0.72]
        es = paired_effect_size(a, b)
        self.assertEqual(es["n"], 4)
        self.assertGreater(es["cohens_d"], 0)
        self.assertLess(abs(es["hedges_g"]), abs(es["cohens_d"]))

    def test_paired_effect_size_zero_when_identical(self):
        es = paired_effect_size([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        self.assertTrue(es["cohens_d"] == 0.0 and es["hedges_g"] == 0.0)

    def test_paired_hedges_correction_uses_difference_degrees_of_freedom(self):
        es = paired_effect_size([1.0, 2.0, 3.0, 4.0], [0.0] * 4)
        d = 2.5 / math.sqrt(5 / 3)
        fixtures.assert_close(es["cohens_d"], d)
        fixtures.assert_close(es["hedges_g"], d * math.sqrt(math.pi / 6))

    def _case_effect_size_rejects_unaligned_or_invalid_pairs(self, a, b):
        with self.assertRaises(ValueError):
            paired_effect_size(a, b)

    def test_constant_nonzero_difference_has_undefined_effect_size(self):
        es = paired_effect_size([2.0, 3.0, 4.0], [1.0, 2.0, 3.0])
        self.assertEqual(es, {"cohens_d": None, "hedges_g": None, "n": 3})

    def _case_paired_effect_size_is_unit_invariant_without_variance_underflow(
        self, scale
    ):
        expected = paired_effect_size([1.0, 2.0, 3.0], [0.0] * 3)
        scaled = paired_effect_size([scale, 2 * scale, 3 * scale], [0.0] * 3)
        fixtures.assert_close(scaled["cohens_d"], expected["cohens_d"])
        fixtures.assert_close(scaled["hedges_g"], expected["hedges_g"])

    def test_file_sha256_streams_without_read_bytes(self):
        path = self.tmp_path / "recording.bin"
        contents = b"recording" * 100000
        path.write_bytes(contents)
        self.patch_object(
            Path,
            "read_bytes",
            lambda path: self.fail("Hashing buffered the whole file"),
        )
        self.assertEqual(sha256_file(path), hashlib.sha256(contents).hexdigest())

    def test_saved_model_and_checksum_stay_in_sync(self):
        path = self.tmp_path / "models" / "stress_classifier.joblib"
        first = {"features": ["ECG"], "classes": ["baseline", "stress"]}
        save_verified_joblib(first, path)
        first_digest = hashlib.sha256(path.read_bytes()).hexdigest()
        self.assertEqual(load_verified_joblib(path), first)
        updated = {"features": ["ECG", "EDA"], "classes": ["baseline", "stress"]}
        save_verified_joblib(updated, path)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        self.assertNotEqual(digest, first_digest)
        self.assertEqual(
            path.with_name(path.name + ".sha256").read_text(),
            f"{digest}  {path.name}\n",
        )
        self.assertEqual(load_verified_joblib(path), updated)

    def test_tampered_model_is_rejected_before_unpickling(self):
        path = self.tmp_path / "stress_classifier.joblib"
        save_verified_joblib({"features": ["ECG"]}, path)
        path.write_bytes(b"tampered model")

        def refuse_unpickling(*args, **kwargs):
            self.fail("Tampered artifact reached joblib.load")

        self.patch_object(joblib, "load", refuse_unpickling)
        with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
            load_verified_joblib(path)

    def test_model_load_uses_verified_snapshot_if_path_is_replaced(self):
        path = self.tmp_path / "model.joblib"
        trusted = {"trusted": True}
        save_verified_joblib(trusted, path)
        original_load = joblib.load

        def replace_then_load(snapshot):
            path.write_bytes(b"untrusted replacement")
            return original_load(snapshot)

        self.patch_object(joblib, "load", replace_then_load)
        self.assertEqual(load_verified_joblib(path), trusted)

    def test_failed_model_serialization_keeps_existing_verified_model(self):
        path = self.tmp_path / "model.joblib"
        save_verified_joblib({"version": 1}, path)
        original_bytes = path.read_bytes()
        checksum = path.with_name(path.name + ".sha256")
        original_checksum = checksum.read_bytes()

        def failed_dump(bundle, destination):
            destination.write_bytes(b"partial serialization")
            raise OSError("serialization failed")

        self.patch_object(joblib, "dump", failed_dump)
        with self.assertRaisesRegex(OSError, "serialization failed"):
            save_verified_joblib({"version": 2}, path)
        self.assertEqual(path.read_bytes(), original_bytes)
        self.assertEqual(checksum.read_bytes(), original_checksum)
        self.assertEqual(load_verified_joblib(path), {"version": 1})

    def _case_failed_model_sidecar_publication_restores_previous_pair(self, existing):
        path = self.tmp_path / "model.joblib"
        sidecar = path.with_name(path.name + ".sha256")
        before = None
        if existing:
            save_verified_joblib({"version": 1}, path)
            before = (path.read_bytes(), sidecar.read_bytes())
        replace = Path.replace

        def fail_sidecar(source, destination):
            if source.name == sidecar.name:
                raise OSError("sidecar publication failed")
            return replace(source, destination)

        self.patch_object(Path, "replace", fail_sidecar)
        with self.assertRaisesRegex(OSError, "sidecar publication failed"):
            save_verified_joblib({"version": 2}, path)
        if existing:
            self.assertEqual((path.read_bytes(), sidecar.read_bytes()), before)
            self.assertEqual(load_verified_joblib(path), {"version": 1})
        else:
            self.assertTrue(not path.exists() and (not sidecar.exists()))
        self.assertEqual(
            set(self.tmp_path.iterdir()), {path, sidecar} if existing else set()
        )

    def _case_invalid_model_sidecar_fails_before_loading(self, contents):
        path = self.tmp_path / "model.joblib"
        save_verified_joblib({}, path)
        path.with_name(path.name + ".sha256").write_text(contents)
        with self.assertRaisesRegex(ValueError, "Invalid SHA-256 sidecar"):
            load_verified_joblib(path)

    def test_atomic_text_creates_utf8_output_and_preserves_permissions(self):
        path = self.tmp_path / "nested" / "report.txt"
        atomic_write_text(path, "Temperature: 30°C\n")
        self.assertEqual(path.read_text(encoding="utf-8"), "Temperature: 30°C\n")
        path.chmod(416)
        atomic_write_text(path, "updated\n")
        self.assertEqual(path.read_text(), "updated\n")
        self.assertEqual(path.stat().st_mode & 511, 416)
        self.assertEqual(list(path.parent.iterdir()), [path])

    def _case_output_io_failure_keeps_previous_snapshot_and_cleans_temporary_files(
        self, failure
    ):
        path = self.tmp_path / "results.json"
        path.write_text("previous snapshot")
        original_write = Path.write_text

        def partial_write(destination, text, **kwargs):
            original_write(destination, "partial", **kwargs)
            raise OSError("disk failure")

        def failed_replace(*args):
            raise OSError("disk failure")

        self.patch_object(
            Path,
            "write_text" if failure == "write" else "replace",
            partial_write if failure == "write" else failed_replace,
        )
        with self.assertRaisesRegex(OSError, "disk failure"):
            write_json(path, {"score": 0.9})
        self.assertEqual(path.read_text(), "previous snapshot")
        self.assertEqual(list(self.tmp_path.iterdir()), [path])

    def _case_invalid_json_cannot_replace_previous_results(self, value):
        path = self.tmp_path / "results.json"
        path.write_text("previous snapshot")
        with self.assertRaises((TypeError, ValueError)):
            write_json(path, {"score": value})
        self.assertEqual(path.read_text(), "previous snapshot")
        self.assertEqual(list(self.tmp_path.iterdir()), [path])


fixtures.add_cases(
    UtilsTests,
    "effect_size_rejects_unaligned_or_invalid_pairs",
    [
        ([], []),
        ([1, 2], [0, 0]),
        ([1, 2, 3], [0]),
        ([[1, 2, 3]], [[0, 0, 0]]),
        ([1, 2, float("nan")], [0, 0, 0]),
    ],
)


fixtures.add_cases(
    UtilsTests,
    "paired_effect_size_is_unit_invariant_without_variance_underflow",
    [(1e-200,), (1e-17,), (1.0,), (1e150,)],
)


fixtures.add_cases(
    UtilsTests,
    "failed_model_sidecar_publication_restores_previous_pair",
    [(False,), (True,)],
)


fixtures.add_cases(
    UtilsTests,
    "invalid_model_sidecar_fails_before_loading",
    [("",), ("not a hash",), ("0" * 63,)],
)


fixtures.add_cases(
    UtilsTests,
    "output_io_failure_keeps_previous_snapshot_and_cleans_temporary_files",
    [("write",), ("replace",)],
)


fixtures.add_cases(
    UtilsTests,
    "invalid_json_cannot_replace_previous_results",
    [(float("nan"),), (float("inf"),), (object(),)],
)
