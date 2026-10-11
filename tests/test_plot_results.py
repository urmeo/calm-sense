import unittest
import hashlib
import json
from tempfile import TemporaryDirectory
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

from scripts.plot_results import _roc_points, _roc_source, plot_roc_auc, plt


class RocPlotTests(unittest.TestCase):
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
        np.testing.assert_array_equal(line.get_xdata(), [0, 0, 0.5, 0.5, 1])
        np.testing.assert_array_equal(line.get_ydata(), [0, 0.5, 0.5, 1, 1])
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
