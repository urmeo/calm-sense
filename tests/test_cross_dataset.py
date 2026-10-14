import unittest
import hashlib
import os
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
        raw_path = Path(self.stack.enter_context(TemporaryDirectory()))
        self.wesad_path, self.noneeg_path = raw_path / "wesad", raw_path / "noneeg"
        for subject in ("S2", "S3"):
            directory = self.wesad_path / subject
            directory.mkdir(parents=True)
            (directory / f"{subject}.pkl").write_bytes(subject.encode())
        self.noneeg_path.mkdir()
        for subject in range(1, 21):
            for record, extensions in (
                ("AccTempEDA", ("hea", "dat", "atr")),
                ("SpO2HR", ("hea", "dat")),
            ):
                for extension in extensions:
                    name = f"Subject{subject}_{record}.{extension}"
                    (self.noneeg_path / name).write_bytes(name.encode())
        self.stack.enter_context(
            patch.object(cross_dataset, "WESAD_DIR", self.wesad_path)
        )
        self.stack.enter_context(
            patch.object(cross_dataset, "VALID_SUBJECTS", ["S2", "S3"])
        )
        self.stack.enter_context(
            patch.object(cross_dataset.non_eeg, "DATA_DIR", self.noneeg_path)
        )

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

    def test_raw_manifests_cover_every_consumed_file(self):
        for dataset, count in (("wesad", 2), ("noneeg", 100)):
            with self.subTest(dataset=dataset):
                source = cross_dataset._raw_provenance(dataset)
                self.assertEqual(len(source["raw_file_sha256"]), count)
                self.assertEqual(
                    cross_dataset._raw_identity(source)[1], source["manifest_sha256"]
                )
        with patch.object(cross_dataset, "VALID_SUBJECTS", []):
            with self.assertRaisesRegex(ValueError, "empty"):
                cross_dataset._raw_provenance("wesad")

    def test_same_size_raw_changes_with_preserved_mtime_require_rebuilding(self):
        for dataset, raw in (
            ("wesad", self.wesad_path / "S3/S3.pkl"),
            ("noneeg", self.noneeg_path / "Subject20_SpO2HR.dat"),
        ):
            with self.subTest(dataset=dataset), patch.object(
                cross_dataset, "PROCESSED_DATA_DIR", self.tmp_path / dataset
            ):
                _, before = cross_dataset._load_or_build_cache(
                    dataset, lambda: self.features
                )
                state = raw.stat()
                payload = raw.read_bytes()
                raw.write_bytes(bytes([payload[0] ^ 1]) + payload[1:])
                os.utime(raw, ns=(state.st_atime_ns, state.st_mtime_ns))
                self.assertEqual(raw.stat().st_size, state.st_size)
                self.assertEqual(raw.stat().st_mtime_ns, state.st_mtime_ns)
                builder = Mock(return_value=self.features)
                with self.assertRaisesRegex(ValueError, "raw inputs changed"):
                    cross_dataset._load_or_build_cache(dataset, builder)
                builder.assert_not_called()
                _, after = cross_dataset._load_or_build_cache(
                    dataset, builder, rebuild=True
                )
                builder.assert_called_once_with()
                self.assertNotEqual(
                    before["source_provenance"]["manifest_sha256"],
                    after["source_provenance"]["manifest_sha256"],
                )

    def test_missing_or_unreadable_raw_files_prevent_cache_reuse(self):
        for dataset, raw in (
            ("wesad", self.wesad_path / "S3/S3.pkl"),
            ("noneeg", self.noneeg_path / "Subject20_SpO2HR.dat"),
        ):
            with self.subTest(dataset=dataset), patch.object(
                cross_dataset, "PROCESSED_DATA_DIR", self.tmp_path / dataset
            ):
                cross_dataset._load_or_build_cache(dataset, lambda: self.features)
                builder = Mock(return_value=self.features)
                payload = raw.read_bytes()
                raw.unlink()
                with self.assertRaisesRegex(FileNotFoundError, "Required raw input"):
                    cross_dataset._load_or_build_cache(dataset, builder)
                raw.write_bytes(payload)
                verified_input = cross_dataset.verified_input

                def unreadable(path):
                    if path == raw:
                        raise PermissionError(f"Unreadable raw file: {path}")
                    return verified_input(path)

                with patch.object(cross_dataset, "verified_input", unreadable):
                    with self.assertRaises(PermissionError):
                        cross_dataset._load_or_build_cache(dataset, builder)
                builder.assert_not_called()

    def test_raw_changes_during_extraction_never_publish_a_cache(self):
        raw = self.wesad_path / "S3/S3.pkl"
        for existing in (False, True):
            with self.subTest(existing=existing), patch.object(
                cross_dataset, "PROCESSED_DATA_DIR", self.tmp_path / str(existing)
            ):
                cache = (
                    cross_dataset.PROCESSED_DATA_DIR
                    / f"portable_wesad_v{PORTABLE_SCHEMA_VERSION}.parquet"
                )
                sidecar = cache.with_suffix(".json")
                if existing:
                    cross_dataset._load_or_build_cache("wesad", lambda: self.features)
                before = (
                    (cache.read_bytes(), sidecar.read_bytes()) if existing else None
                )

                def changed_raw():
                    payload = raw.read_bytes()
                    raw.write_bytes(bytes([payload[0] ^ 1]) + payload[1:])
                    return self.features

                with self.assertRaisesRegex(ValueError, "changed during.*extraction"):
                    cross_dataset._load_or_build_cache(
                        "wesad", changed_raw, rebuild=True
                    )
                if existing:
                    self.assertEqual((cache.read_bytes(), sidecar.read_bytes()), before)
                else:
                    self.assertFalse(cache.exists() or sidecar.exists())

    def test_raw_identity_ignores_generation_context_changes(self):
        def current_source():
            source = cross_dataset._raw_provenance("wesad")
            source["extraction_context"] = {
                "git_sha": str(provider.call_count),
                "generated_at": str(provider.call_count),
            }
            return source

        provider = Mock(side_effect=current_source)
        with patch.object(cross_dataset, "PROCESSED_DATA_DIR", self.tmp_path):
            expected, metadata = cross_dataset._load_or_build_cache(
                "wesad", lambda: self.features, provider
            )
            builder = Mock(return_value=self.features)
            actual, again = cross_dataset._load_or_build_cache(
                "wesad", builder, provider
            )
        self.assertEqual(provider.call_count, 3)
        builder.assert_not_called()
        pd.testing.assert_frame_equal(actual, expected)
        self.assertEqual(metadata, again)
        self.assertEqual(
            metadata["source_provenance"]["extraction_context"]["git_sha"], "1"
        )

    def test_static_provenance_cannot_bypass_raw_input_verification(self):
        builder = Mock(return_value=self.features)
        with self.assertRaisesRegex(TypeError, "must be callable"):
            cross_dataset._load_or_build_cache(
                "wesad", builder, cross_dataset._raw_provenance("wesad")
            )
        builder.assert_not_called()

    def test_missing_partial_or_invalid_raw_metadata_is_rejected(self):
        with patch.object(cross_dataset, "PROCESSED_DATA_DIR", self.tmp_path):
            _, original = cross_dataset._load_or_build_cache(
                "wesad", lambda: self.features
            )
            sidecar = self.tmp_path / f"portable_wesad_v{PORTABLE_SCHEMA_VERSION}.json"
            for fault in ("missing", "empty", "partial", "bad_digest"):
                with self.subTest(fault=fault):
                    metadata = json.loads(json.dumps(original))
                    source = metadata["source_provenance"]
                    if fault == "missing":
                        metadata.pop("source_provenance")
                    elif fault == "empty":
                        source["raw_file_sha256"] = {}
                    elif fault == "partial":
                        source["raw_file_sha256"].pop("S3/S3.pkl")
                        source["manifest_sha256"] = hashlib.sha256(
                            json.dumps(
                                source["raw_file_sha256"], sort_keys=True
                            ).encode()
                        ).hexdigest()
                    else:
                        source["manifest_sha256"] = "0" * 64
                    sidecar.write_text(json.dumps(metadata))
                    builder = Mock(return_value=self.features)
                    with self.assertRaisesRegex(ValueError, "raw input"):
                        cross_dataset._load_or_build_cache("wesad", builder)
                    builder.assert_not_called()

    def test_raw_replacement_while_hashing_is_rejected(self):
        raw = self.wesad_path / "S3/S3.pkl"
        file_digest = hashlib.file_digest

        def replace_after_hash(stream, algorithm):
            digest = file_digest(stream, algorithm)
            if Path(stream.name) == raw:
                replacement = raw.with_suffix(".replacement")
                replacement.write_bytes(raw.read_bytes())
                replacement.replace(raw)
            return digest

        with patch.object(hashlib, "file_digest", replace_after_hash):
            with self.assertRaisesRegex(RuntimeError, "Input changed"):
                cross_dataset._raw_provenance("wesad")

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
                source_provenance=lambda: {
                    **cross_dataset._raw_provenance("wesad"),
                    "invalid": object(),
                },
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
