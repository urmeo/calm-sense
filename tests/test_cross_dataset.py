import unittest
from contextlib import ExitStack
from unittest.mock import patch
from tempfile import TemporaryDirectory
from functools import partialmethod
import json
from pathlib import Path
from unittest.mock import Mock
import pandas as pd
import numpy as np
from scripts import cross_dataset
from src.portable import (
    PORTABLE_FEATURE_COLUMNS,
    PORTABLE_SCHEMA_VERSION,
    PORTABLE_SIGNAL_UNITS,
)


def features():
    frame = pd.DataFrame({column: [1.0, 2.0] for column in PORTABLE_FEATURE_COLUMNS})
    frame["subject"] = ["S2", "S3"]
    frame["label"] = [0, 1]
    return frame


class CrossDatasetTests(unittest.TestCase):

    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.tmp_path = Path(self.stack.enter_context(TemporaryDirectory()))
        self.features = features()

    def test_legacy_cache_is_ignored_and_current_cache_is_verified(self):
        tmp_path = self.tmp_path
        features = self.features
        directory = tmp_path / "processed"
        directory.mkdir()
        legacy = directory / "portable_wesad.parquet"
        legacy.write_bytes(b"legacy sample-based slopes")
        previous = directory / f"portable_wesad_v{PORTABLE_SCHEMA_VERSION - 1}.parquet"
        previous.write_bytes(b"previous HR boundary convention")
        self.stack.enter_context(
            patch.object(cross_dataset, "PROCESSED_DATA_DIR", directory)
        )
        builder = Mock(return_value=features)
        frame, metadata = cross_dataset._load_or_build_cache("wesad", builder)
        builder.assert_called_once_with()
        self.assertEqual(legacy.read_bytes(), b"legacy sample-based slopes")
        self.assertEqual(previous.read_bytes(), b"previous HR boundary convention")
        self.assertEqual(metadata["schema_version"], PORTABLE_SCHEMA_VERSION)
        self.assertEqual(metadata["sampling_rates_hz"], {"eda": 4, "temp": 4})
        self.assertEqual(metadata["slope_units"], "per_second")
        self.assertEqual(
            metadata["finite_sample_timestamps"], "original_sample_positions"
        )
        self.assertEqual(metadata["signal_units"], PORTABLE_SIGNAL_UNITS["wesad"])
        self.assertEqual(
            metadata["hr_window_boundary"], "include_timestamps_in_[start,end)"
        )
        self.assertIn("scipy", metadata["extraction_packages"])
        builder.reset_mock()
        cached, again = cross_dataset._load_or_build_cache("wesad", builder)
        builder.assert_not_called()
        pd.testing.assert_frame_equal(frame, cached)
        self.assertEqual(metadata, again)

    def test_cache_creates_parent_directories(self):
        tmp_path = self.tmp_path
        features = self.features
        directory = tmp_path / "nested" / "processed"
        self.stack.enter_context(
            patch.object(cross_dataset, "PROCESSED_DATA_DIR", directory)
        )
        _, metadata = cross_dataset._load_or_build_cache("noneeg", lambda: features)
        self.assertTrue(
            (directory / f"portable_noneeg_v{PORTABLE_SCHEMA_VERSION}.parquet").exists()
        )
        self.assertEqual(metadata["sampling_rates_hz"], {"eda": 8, "temp": 8})
        self.assertEqual(metadata["signal_units"], PORTABLE_SIGNAL_UNITS["noneeg"])

    def _case_unverified_current_cache_is_rejected(self, fault):
        tmp_path = self.tmp_path
        features = self.features
        self.stack.enter_context(
            patch.object(cross_dataset, "PROCESSED_DATA_DIR", tmp_path)
        )
        cross_dataset._load_or_build_cache("wesad", lambda: features)
        sidecar = tmp_path / f"portable_wesad_v{PORTABLE_SCHEMA_VERSION}.json"
        if fault == "missing_metadata":
            sidecar.unlink()
        else:
            metadata = json.loads(sidecar.read_text())
            if fault == "old_schema":
                metadata["schema_version"] = PORTABLE_SCHEMA_VERSION - 1
            else:
                metadata["cache_sha256"] = "0" * 64
            sidecar.write_text(json.dumps(metadata))
        builder = Mock(return_value=features)
        with self.assertRaisesRegex(ValueError, "metadata|schema|checksum"):
            cross_dataset._load_or_build_cache("wesad", builder)
        builder.assert_not_called()

    def _case_cache_rejects_changed_extractor_or_environment_and_can_rebuild(
        self, field
    ):
        tmp_path = self.tmp_path
        features = self.features
        self.stack.enter_context(
            patch.object(cross_dataset, "PROCESSED_DATA_DIR", tmp_path)
        )
        _, old = cross_dataset._load_or_build_cache("wesad", lambda: features)
        schema = cross_dataset._cache_schema("wesad")
        schema[field] = {"changed": "new version"}
        self.stack.enter_context(
            patch.object(cross_dataset, "_cache_schema", lambda dataset: schema)
        )
        builder = Mock(return_value=features)
        with self.assertRaisesRegex(ValueError, "schema mismatch"):
            cross_dataset._load_or_build_cache("wesad", builder)
        builder.assert_not_called()
        _, new = cross_dataset._load_or_build_cache("wesad", builder, rebuild=True)
        builder.assert_called_once_with()
        self.assertNotEqual(new[field], old[field])

    def test_cross_dataset_labels_allow_integer_valued_float_storage(self):
        features = self.features
        features["label"] = features["label"].astype(float)
        _, y, _ = cross_dataset._xy(features, list(PORTABLE_FEATURE_COLUMNS))
        self.assertIn(y.dtype.kind, "iu")
        self.assertEqual(y.tolist(), [0, 1])

    def test_feature_preparation_accepts_read_only_arrays_without_changing_frame(self):
        frame = self.features
        frame.loc[0, PORTABLE_FEATURE_COLUMNS[0]] = np.nan
        original = frame.copy(deep=True)
        with pd.option_context("mode.copy_on_write", True):
            X, y, subjects = cross_dataset._xy(frame, list(PORTABLE_FEATURE_COLUMNS))
        self.assertTrue(np.isnan(X[0, 0]))
        np.testing.assert_array_equal(y, [0, 1])
        np.testing.assert_array_equal(subjects, ["S2", "S3"])
        pd.testing.assert_frame_equal(frame, original)

    def test_cross_dataset_rejects_metadata_as_features(self):
        for columns in [[], ["label"], ["subject"], ["eda_mean", "eda_mean"]]:
            with self.subTest(columns=columns):
                with self.assertRaisesRegex(ValueError, "feature columns"):
                    cross_dataset._xy(self.features, columns)

    def test_failed_rebuild_preserves_verified_cache(self):
        tmp_path = self.tmp_path
        features = self.features
        self.stack.enter_context(
            patch.object(cross_dataset, "PROCESSED_DATA_DIR", tmp_path)
        )
        cross_dataset._load_or_build_cache("wesad", lambda: features)
        cache = tmp_path / f"portable_wesad_v{PORTABLE_SCHEMA_VERSION}.parquet"
        sidecar = cache.with_suffix(".json")
        before = (cache.read_bytes(), sidecar.read_bytes())
        changed = features.copy()
        changed[PORTABLE_FEATURE_COLUMNS[0]] += 1
        with self.assertRaises(TypeError):
            cross_dataset._load_or_build_cache(
                "wesad",
                lambda: changed,
                source_provenance={"invalid": object()},
                rebuild=True,
            )
        self.assertEqual((cache.read_bytes(), sidecar.read_bytes()), before)

    def _case_failed_cache_sidecar_publication_restores_previous_pair(self, existing):
        tmp_path = self.tmp_path
        features = self.features
        self.stack.enter_context(
            patch.object(cross_dataset, "PROCESSED_DATA_DIR", tmp_path)
        )
        cache = tmp_path / f"portable_wesad_v{PORTABLE_SCHEMA_VERSION}.parquet"
        sidecar = cache.with_suffix(".json")
        before = None
        if existing:
            cross_dataset._load_or_build_cache("wesad", lambda: features)
            before = (cache.read_bytes(), sidecar.read_bytes())
        changed = features.copy()
        changed[PORTABLE_FEATURE_COLUMNS[0]] += 1
        replace = Path.replace

        def fail_sidecar(source, destination):
            if source.name == sidecar.name:
                raise OSError("sidecar publication failed")
            return replace(source, destination)

        self.stack.enter_context(patch.object(Path, "replace", fail_sidecar))
        with self.assertRaisesRegex(OSError, "sidecar publication failed"):
            cross_dataset._load_or_build_cache("wesad", lambda: changed, rebuild=True)
        if existing:
            self.assertEqual((cache.read_bytes(), sidecar.read_bytes()), before)
            cached, _ = cross_dataset._load_or_build_cache(
                "wesad", lambda: self.fail("Rebuilt instead of recovering")
            )
            pd.testing.assert_frame_equal(cached, features)
        else:
            self.assertTrue(not cache.exists() and (not sidecar.exists()))
        self.assertEqual(
            set(tmp_path.iterdir()), {cache, sidecar} if existing else set()
        )


for index, fault in enumerate(["missing_metadata", "old_schema", "bad_checksum"]):
    setattr(
        CrossDatasetTests,
        f"test_unverified_current_cache_is_rejected_{index}",
        partialmethod(
            CrossDatasetTests._case_unverified_current_cache_is_rejected, fault=fault
        ),
    )
for index, field in enumerate(["extractor_sha256", "extraction_packages"]):
    setattr(
        CrossDatasetTests,
        f"test_cache_rejects_changed_extractor_or_environment_and_can_rebuild_{index}",
        partialmethod(
            CrossDatasetTests._case_cache_rejects_changed_extractor_or_environment_and_can_rebuild,
            field=field,
        ),
    )
for index, existing in enumerate([False, True]):
    setattr(
        CrossDatasetTests,
        f"test_failed_cache_sidecar_publication_restores_previous_pair_{index}",
        partialmethod(
            CrossDatasetTests._case_failed_cache_sidecar_publication_restores_previous_pair,
            existing=existing,
        ),
    )
