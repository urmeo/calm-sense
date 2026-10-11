import unittest
from contextlib import ExitStack
from unittest.mock import patch
from pathlib import Path
from tempfile import TemporaryDirectory
from functools import partialmethod
import json
import numpy as np
from scripts import tuning
from src import synthetic


class TuningTests(unittest.TestCase):

    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.tmp_path = Path(self.stack.enter_context(TemporaryDirectory()))

    def _case_synthetic_tuning_isolates_outputs_and_reads_only_demo_defaults(
        self, demo_accuracy
    ):
        tmp_path = self.tmp_path
        output_dir = tmp_path / "outputs"
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
        self.stack.enter_context(patch.object(tuning, "RESULTS_DIR", results_dir))
        self.stack.enter_context(patch.object(tuning, "FIGURES_DIR", figures_dir))
        self.stack.enter_context(patch.object(tuning, "DEMO_DIR", demo_dir))

        def refuse_real_cache():
            self.fail("Synthetic tuning read the real dataset cache")

        def synthetic_features(**kwargs):
            self.assertIs(kwargs["cache"], False)
            return (None, None, None)

        self.stack.enter_context(patch.object(tuning, "load_cached", refuse_real_cache))
        self.stack.enter_context(
            patch.object(synthetic, "features", synthetic_features)
        )
        self.stack.enter_context(
            patch.object(tuning, "prepare_task", lambda *args: (None,) * 5)
        )
        scores = {model: {"accuracy_mean": 0.72, "best_params": {}}}
        self.stack.enter_context(patch.object(tuning, "compute", lambda *args: scores))
        provenance = {"git_sha": "test", "generated_at": "test"}
        self.stack.enter_context(
            patch.object(tuning, "analysis_provenance", lambda *args: provenance)
        )
        original_plot = tuning._plot
        plotted_defaults = []

        def plot(tuned, defaults, path):
            plotted_defaults.append(defaults)
            original_plot(tuned, defaults, path)

        self.stack.enter_context(patch.object(tuning, "_plot", plot))
        tuning.run(synthetic=True)
        self.assertTrue(
            all((path.read_bytes() == contents for path, contents in originals.items()))
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
        self.stack.enter_context(
            patch.dict(tuning.GRIDS, {"lr": {"clf__C": [-1.0, 1.0]}})
        )
        groups = np.repeat(["S0", "S1", "S2", "S3"], 6)
        y = np.tile([0, 1], 12)
        X = np.random.RandomState(0).randn(24, 2)
        with self.assertRaisesRegex(ValueError, "C"):
            tuning.tune_model("lr", X, y, groups, inner_splits=2)

    def test_nested_tuning_requires_valid_subject_counts(self):
        for inner_splits, groups in [(1, ["S0", "S1", "S2"]), (2, ["S0", "S1"])]:
            with self.subTest(inner_splits=inner_splits, groups=groups):
                with self.assertRaisesRegex(ValueError, "requires"):
                    tuning.tune_model(
                        "lr",
                        np.zeros((len(groups), 1)),
                        np.zeros(len(groups)),
                        np.array(groups),
                        inner_splits,
                    )

    def test_tuning_plot_does_not_show_missing_default_as_zero(self):
        tmp_path = self.tmp_path
        bars = []
        original = tuning.plt.bar

        def bar(x, values, *args, **kwargs):
            bars.append(list(values))
            return original(x, values, *args, **kwargs)

        self.stack.enter_context(patch.object(tuning.plt, "bar", bar))
        tuning._plot(
            {
                "Random Forest": {"accuracy_mean": 0.8},
                "LightGBM": {"accuracy_mean": 0.7},
            },
            {"Random Forest": 0.9},
            tmp_path / "plot.png",
        )
        self.assertEqual(bars, [[0.9], [0.8]])

    def _case_tuning_default_comparison_requires_matching_protocol_schema_and_cohort(
        self, mismatch
    ):
        tmp_path = self.tmp_path
        from src.features.feature_pipeline import FEATURE_SCHEMA_VERSION

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
        (tmp_path / "metrics.json").write_text(json.dumps(summary))
        defaults = tuning._defaults(tmp_path, groups=np.array(["S1", "S0", "S0"]))
        self.assertEqual(defaults, {"Random Forest": 0.91} if mismatch is None else {})

    def test_tuning_retires_only_generated_stale_plot_when_defaults_are_incompatible(
        self,
    ):
        tmp_path = self.tmp_path
        results, figures = (tmp_path / "results", tmp_path / "generated" / "figures")
        results.mkdir()
        figures.mkdir(parents=True)
        (results / "metrics.json").write_text(
            json.dumps({"benchmark_protocol_version": 1})
        )
        stale = figures / "tuning.png"
        stale.write_bytes(b"stale generated comparison")
        historical = tmp_path / "figures" / "tuning.png"
        historical.parent.mkdir()
        historical.write_bytes(b"preserved historical figure")
        self.stack.enter_context(patch.object(tuning, "RESULTS_DIR", results))
        self.stack.enter_context(patch.object(tuning, "FIGURES_DIR", figures))
        self.stack.enter_context(
            patch.object(tuning, "load_cached", lambda: (None, None))
        )
        self.stack.enter_context(
            patch.object(
                tuning,
                "prepare_task",
                lambda *args: (None, None, np.array(["S0", "S1"]), None, None),
            )
        )
        self.stack.enter_context(
            patch.object(
                tuning,
                "compute",
                lambda *args: {"Random Forest": {"accuracy_mean": 0.6}},
            )
        )
        tuning.run()
        self.assertTrue((results / "tuning.json").exists())
        self.assertFalse(stale.exists())
        self.assertEqual(historical.read_bytes(), b"preserved historical figure")


for index, demo_accuracy in enumerate([None, 0.42]):
    setattr(
        TuningTests,
        f"test_synthetic_tuning_isolates_outputs_and_reads_only_demo_defaults_{index}",
        partialmethod(
            TuningTests._case_synthetic_tuning_isolates_outputs_and_reads_only_demo_defaults,
            demo_accuracy=demo_accuracy,
        ),
    )
for index, mismatch in enumerate([None, "schema", "protocol", "cohort", "duplicates"]):
    setattr(
        TuningTests,
        f"test_tuning_default_comparison_requires_matching_protocol_schema_and_cohort_{index}",
        partialmethod(
            TuningTests._case_tuning_default_comparison_requires_matching_protocol_schema_and_cohort,
            mismatch=mismatch,
        ),
    )
