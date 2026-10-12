import unittest
from contextlib import ExitStack
from unittest.mock import patch
from pathlib import Path
from tempfile import TemporaryDirectory
from functools import partialmethod
import json
from types import SimpleNamespace
import numpy as np
import pandas as pd
from scripts import (
    build_dashboard_data,
    export_signals,
    stamp_provenance,
    update_readme_tables,
)
from src.calibration import BINARY_BRIER_DEFINITION
from src.config import RESULTS_DIR
from src.utils import (
    analysis_provenance,
    benchmark_reference,
    feature_frame_sha256,
    pipeline_source_sha256,
    sha256_file,
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


class DashboardDataTests(unittest.TestCase):

    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.tmp_path = Path(self.stack.enter_context(TemporaryDirectory()))

    def test_export_preserves_matched_metrics_and_normalizes_historical_brier(self):
        tmp_path = self.tmp_path
        results_dir = tmp_path / "results"
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
        fixtures = {
            "metrics.json": metrics,
            "calibration.json": calibration,
            "personalization.json": {"uncalibrated": {"ece": 0.1, "brier": 0.2}},
        }
        for name, value in fixtures.items():
            (results_dir / name).write_text(json.dumps(value))
        originals = {path: path.read_bytes() for path in results_dir.glob("*.json")}
        output = tmp_path / "dashboard" / "results.ts"
        self.stack.enter_context(
            patch.object(build_dashboard_data, "RESULTS_DIR", results_dir)
        )
        self.stack.enter_context(
            patch.object(build_dashboard_data, "DASHBOARD_RESULTS", output)
        )
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
        np.testing.assert_allclose(
            exported["calibration"]["loso"]["brier"],
            calibration["loso"]["brier"] / 2,
            rtol=1e-06,
            atol=1e-12,
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
        tmp_path = self.tmp_path
        output = tmp_path / "results.ts"
        before = "const data = {};\n\nexport default data;\n"
        output.write_text(before, encoding="utf-8")
        self.stack.enter_context(
            patch.object(build_dashboard_data, "DASHBOARD_RESULTS", output)
        )
        self.stack.enter_context(
            patch.object(
                build_dashboard_data,
                "_load_json",
                lambda name: (
                    {"binary": {"n_windows": invalid_number}}
                    if name == "metrics.json"
                    else None
                ),
            )
        )
        self.stack.enter_context(
            patch.object(build_dashboard_data, "_load_csv", lambda name: None)
        )
        with self.assertRaisesRegex(ValueError, "Out of range float values"):
            build_dashboard_data.run()
        self.assertEqual(output.read_text(encoding="utf-8"), before)

    def test_signal_export_creates_output_folder_and_preserves_aligned_samples(self):
        tmp_path = self.tmp_path
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
        output = tmp_path / "dashboard" / "signals.ts"
        self.stack.enter_context(
            patch.object(export_signals, "DASHBOARD_SIGNALS", output)
        )
        self.stack.enter_context(
            patch.object(export_signals, "WESADLoader", lambda: loader)
        )
        self.stack.enter_context(
            patch.object(export_signals, "FS", SimpleNamespace(CHEST=2))
        )
        self.stack.enter_context(patch.object(export_signals, "SUBJECTS", ["S2"]))
        self.stack.enter_context(patch.object(export_signals, "SECONDS", 1))
        self.stack.enter_context(patch.object(export_signals, "OUT_FS", 2))
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
        tmp_path = self.tmp_path
        output = tmp_path / "results.ts"
        output.write_text("existing snapshot", encoding="utf-8")
        self.stack.enter_context(
            patch.object(build_dashboard_data, "DASHBOARD_RESULTS", output)
        )
        self.stack.enter_context(
            patch.object(
                build_dashboard_data,
                "_load_json",
                lambda name: {} if name == "metrics.json" else None,
            )
        )
        self.stack.enter_context(
            patch.object(build_dashboard_data, "_load_csv", lambda name: None)
        )
        with self.assertRaisesRegex(ValueError, "non-empty model comparison"):
            build_dashboard_data.run()
        self.assertEqual(output.read_text(encoding="utf-8"), "existing snapshot")

    def test_export_copies_protocol_metadata_from_the_benchmark_run(self):
        tmp_path = self.tmp_path
        load_json = build_dashboard_data._load_json

        def current_run(name):
            value = load_json(name)
            if name == "metrics.json":
                value["benchmark_protocol_version"] = 2
            return value

        output = tmp_path / "results.ts"
        self.stack.enter_context(
            patch.object(build_dashboard_data, "DASHBOARD_RESULTS", output)
        )
        self.stack.enter_context(
            patch.object(build_dashboard_data, "_load_json", current_run)
        )
        build_dashboard_data.run()
        self.assertEqual(_read_module(output)["benchmark_protocol_version"], 2)

    def test_export_copies_recorded_subject_counts(self):
        tmp_path = self.tmp_path
        load_json = build_dashboard_data._load_json

        def subset_run(name):
            value = load_json(name)
            if name == "metrics.json":
                value["binary"]["n_subjects"] = 3
                value["multiclass"]["n_subjects"] = 4
            return value

        output = tmp_path / "results.ts"
        self.stack.enter_context(
            patch.object(build_dashboard_data, "DASHBOARD_RESULTS", output)
        )
        self.stack.enter_context(
            patch.object(build_dashboard_data, "_load_json", subset_run)
        )
        build_dashboard_data.run()
        exported = _read_module(output)
        self.assertEqual(exported["binary"]["n_subjects"], 3)
        self.assertEqual(exported["multiclass"]["n_subjects"], 4)

    def _case_export_rejects_invalid_subject_counts(self, count):
        tmp_path = self.tmp_path
        load_json = build_dashboard_data._load_json

        def invalid_run(name):
            value = load_json(name)
            if name == "metrics.json":
                value["binary"]["n_subjects"] = count
            return value

        output = tmp_path / "results.ts"
        output.write_text("existing snapshot", encoding="utf-8")
        self.stack.enter_context(
            patch.object(build_dashboard_data, "DASHBOARD_RESULTS", output)
        )
        self.stack.enter_context(
            patch.object(build_dashboard_data, "_load_json", invalid_run)
        )
        with self.assertRaisesRegex(ValueError, "n_subjects must be an integer >= 2"):
            build_dashboard_data.run()
        self.assertEqual(output.read_text(encoding="utf-8"), "existing snapshot")

    def _case_export_rejects_malformed_or_inconsistent_scores_before_replacing_snapshot(
        self, field_path, value, message
    ):
        tmp_path = self.tmp_path
        original_load = build_dashboard_data._load_json

        def malformed_run(name):
            data = original_load(name)
            if name == "metrics.json":
                target = data["binary"]
                for field in field_path[:-1]:
                    target = target[field]
                target[field_path[-1]] = value
            return data

        output = tmp_path / "results.ts"
        output.write_text("existing snapshot")
        self.stack.enter_context(
            patch.object(build_dashboard_data, "DASHBOARD_RESULTS", output)
        )
        self.stack.enter_context(
            patch.object(build_dashboard_data, "_load_json", malformed_run)
        )
        with self.assertRaisesRegex(ValueError, message):
            build_dashboard_data.run()
        self.assertEqual(output.read_text(), "existing snapshot")

    def test_export_accepts_absent_matched_comparison_without_inventing_scores(self):
        tmp_path = self.tmp_path
        original_load = build_dashboard_data._load_json

        def unmatched_run(name):
            data = original_load(name)
            if name == "metrics.json":
                for key in (
                    "loso_matched_accuracy",
                    "within_subject_accuracy",
                    "optimism_gap_pts",
                ):
                    data["binary"][key] = None
            return data

        output = tmp_path / "results.ts"
        self.stack.enter_context(
            patch.object(build_dashboard_data, "DASHBOARD_RESULTS", output)
        )
        self.stack.enter_context(
            patch.object(build_dashboard_data, "_load_json", unmatched_run)
        )
        build_dashboard_data.run()
        data = _read_module(output)["binary"]
        self.assertIs(data["loso_matched_accuracy"], None)
        self.assertIs(data["within_subject_accuracy"], None)
        self.assertIs(data["optimism_gap_pts"], None)

    def test_environment_snapshot_fingerprints_results_and_excludes_itself(self):
        tmp_path = self.tmp_path
        import hashlib

        (tmp_path / "metrics.json").write_bytes(b'{"binary": {}}')
        (tmp_path / "provenance.json").write_text("old snapshot", encoding="utf-8")
        self.stack.enter_context(
            patch.object(stamp_provenance, "RESULTS_DIR", tmp_path)
        )
        stamp_provenance.run()
        snapshot = json.loads((tmp_path / "provenance.json").read_text())
        self.assertEqual(snapshot["provenance_kind"], "environment_snapshot")
        self.assertEqual(snapshot["data"]["n_reference_subjects"], 15)
        self.assertNotIn("n_subjects", snapshot["data"])
        self.assertEqual(
            snapshot["result_sha256"],
            {"metrics.json": hashlib.sha256(b'{"binary": {}}').hexdigest()},
        )
        self.assertEqual((tmp_path / "metrics.json").read_bytes(), b'{"binary": {}}')

    def _case_readme_metrics_reject_nonfinite_values(self, value):
        with self.assertRaisesRegex(ValueError, "README metrics must be finite"):
            update_readme_tables._format_metric(value)

    def test_personalization_table_rejects_an_unknown_brier_scale(self):
        tmp_path = self.tmp_path
        (tmp_path / "personalization.json").write_text(
            json.dumps({"brier_definition": "unknown_scale"}), encoding="utf-8"
        )
        self.stack.enter_context(
            patch.object(update_readme_tables, "RESULTS", tmp_path)
        )
        with self.assertRaisesRegex(ValueError, "positive-class MSE"):
            update_readme_tables._personalization_table()

    def test_new_primary_protocol_does_not_certify_saved_ancillary_runs(self):
        tmp_path = self.tmp_path
        results_dir = tmp_path / "results"
        results_dir.mkdir()
        fixtures = {
            "metrics.json": _benchmark_fixture(protocol=2),
            "calibration.json": {"brier_definition": BINARY_BRIER_DEFINITION},
            "personalization.json": {"brier_definition": BINARY_BRIER_DEFINITION},
            **{
                f"{name}.json": {}
                for name in ("stats", "wrist", "cross_dataset", "tuning")
            },
        }
        for name, value in fixtures.items():
            (results_dir / name).write_text(json.dumps(value))
        (results_dir / "shap_top_features.csv").write_text(
            "feature,mean_abs_shap\nHRV_MeanNN,0.2\n"
        )
        (results_dir / "ablation.csv").write_text(
            "subset,n_features,accuracy_mean\nAll features,2,0.75\n"
        )
        output = tmp_path / "results.ts"
        self.stack.enter_context(
            patch.object(build_dashboard_data, "DASHBOARD_RESULTS", output)
        )
        self.stack.enter_context(
            patch.object(build_dashboard_data, "RESULTS_DIR", results_dir)
        )
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
        tmp_path = self.tmp_path
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

        output = tmp_path / "results.ts"
        output.write_text("previous snapshot")
        self.stack.enter_context(
            patch.object(build_dashboard_data, "DASHBOARD_RESULTS", output)
        )
        self.stack.enter_context(
            patch.object(build_dashboard_data, "_load_json", current_primary)
        )
        if fault:
            with self.assertRaisesRegex(ValueError, "SHAP snapshot does not match"):
                build_dashboard_data.run()
            self.assertEqual(output.read_text(), "previous snapshot")
        else:
            build_dashboard_data.run()
            self.assertNotIn("shap", _read_module(output)["unverified_sections"])


class BenchmarkLinkageTests(unittest.TestCase):

    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.root = Path(self.stack.enter_context(TemporaryDirectory()))
        self.cache = self.root / "cache"
        self.cache.mkdir()
        (self.cache / "features.parquet").write_bytes(b"feature cache")
        (self.cache / "raw_windows.npz").write_bytes(b"raw windows")
        self.stack.enter_context(patch("src.config.PROCESSED_DATA_DIR", self.cache))
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


for index, invalid_number in enumerate([float("nan"), float("inf"), -float("inf")]):
    setattr(
        DashboardDataTests,
        f"test_export_rejects_nonfinite_values_without_replacing_snapshot_{index}",
        partialmethod(
            DashboardDataTests._case_export_rejects_nonfinite_values_without_replacing_snapshot,
            invalid_number=invalid_number,
        ),
    )
for index, count in enumerate([None, 0, 1, 2.5, "15"]):
    setattr(
        DashboardDataTests,
        f"test_export_rejects_invalid_subject_counts_{index}",
        partialmethod(
            DashboardDataTests._case_export_rejects_invalid_subject_counts, count=count
        ),
    )
for index, (field_path, value, message) in enumerate(
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
    ]
):
    setattr(
        DashboardDataTests,
        f"test_export_rejects_malformed_or_inconsistent_scores_before_replacing_snapshot_{index}",
        partialmethod(
            DashboardDataTests._case_export_rejects_malformed_or_inconsistent_scores_before_replacing_snapshot,
            field_path=field_path,
            value=value,
            message=message,
        ),
    )
for index, value in enumerate([float("nan"), float("inf"), -float("inf")]):
    setattr(
        DashboardDataTests,
        f"test_readme_metrics_reject_nonfinite_values_{index}",
        partialmethod(
            DashboardDataTests._case_readme_metrics_reject_nonfinite_values, value=value
        ),
    )
for index, fault in enumerate([None, "checksum", "model", "scope", "path"]):
    setattr(
        DashboardDataTests,
        f"test_shap_link_requires_the_recorded_benchmark_artifact_{index}",
        partialmethod(
            DashboardDataTests._case_shap_link_requires_the_recorded_benchmark_artifact,
            fault=fault,
        ),
    )
