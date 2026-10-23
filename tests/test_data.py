from copy import deepcopy
from dataclasses import replace
from functools import partialmethod
import hashlib
import io
from itertools import product
import json
import os
from pathlib import Path
import pickle
import shutil
import struct
from tempfile import TemporaryDirectory
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import zipfile

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
import pandas as pd
from scipy.integrate import trapezoid

from scripts import cross_dataset, download_data
from scripts.download_data import _download, _safe_extract
from src import cache, config, dataset, dataset_wrist, synthetic
from src.cache import cache_fingerprint, raw_input_sha256
from src.config import (
    FILTER_PARAMS,
    FS,
    RR_CORRECTION_VERSION,
    SCR_DETECTION_VERSION,
    VALID_SUBJECTS,
)
from src.data import loader
from src.data.loader import WESADLoader
from src.dataset import WindowedDataset
from src.dataset_wrist import WristDataset
from src.datasets import non_eeg
from src.datasets.records import read_annotations, read_record, record_files
from src.features.accelerometer_features import AccelerometerFeatureExtractor
from src.features.eda_features import EDAFeatureExtractor
from src.features.feature_pipeline import (
    FEATURE_SCHEMA_VERSION,
    FeatureExtractionPipeline,
)
from src.features.hrv_frequency_domain import HRVFrequencyDomainExtractor
from src.features.hrv_nonlinear import HRVNonlinearExtractor
from src.features.hrv_time_domain import HRVTimeDomainExtractor
from src.features.respiration_features import RespirationFeatureExtractor
from src.features.temperature_features import TemperatureFeatureExtractor
from src.portable import (
    PORTABLE_FEATURE_COLUMNS,
    PORTABLE_SCHEMA_VERSION,
    PORTABLE_SIGNAL_UNITS,
    portable_features,
    wesad_portable,
)
from src.preprocessing import eda_processor
from src.preprocessing.ecg_processor import ECGProcessor
from src.preprocessing.eda_processor import EDAProcessor
from src.preprocessing.filters import SignalProcessor
from src.utils import feature_frame_sha256, sha256_file


class PatchedTestCase(unittest.TestCase):
    def patch_object(self, *args, **kwargs):
        return self.enterContext(patch.object(*args, **kwargs))


class TemporaryWorkspaceTestCase(PatchedTestCase):
    def setUp(self):
        self.tmp_path = Path(self.enterContext(TemporaryDirectory()))


def add_cases(test_class, method_name, arguments):
    method = getattr(test_class, f"_case_{method_name}")
    for index, values in enumerate(arguments):
        setattr(
            test_class, f"test_{method_name}_{index}", partialmethod(method, *values)
        )


def assert_close(actual, expected):
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-12)


def source_loader(directory, subjects=VALID_SUBJECTS):
    subjects = list(subjects)
    data_path = Path(directory) / "WESAD"
    record = {
        "signal": {
            "chest": {"ECG": np.zeros((700, 1))},
            "wrist": {
                "BVP": np.zeros((64, 1)),
                "EDA": np.zeros((4, 1)),
                "TEMP": np.zeros((4, 1)),
                "ACC": np.zeros((32, 3)),
            },
        },
        "label": np.ones(700, dtype=int),
    }
    payload = pickle.dumps(record, protocol=4)
    for subject in subjects:
        path = data_path / subject / f"{subject}.pkl"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
    return SimpleNamespace(
        subjects=subjects,
        data_path=data_path,
        raw_input_sha256=raw_input_sha256(data_path, subjects),
    )


class CacheFingerprintTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(self.enterContext(TemporaryDirectory()))
        self.path = self.root / "S2" / "S2.pkl"
        self.path.parent.mkdir()
        self.path.write_bytes(b"original recording")

    def fingerprint(self):
        return cache.cache_fingerprint("chest", self.root, ["S2"])

    def test_full_content_hash_and_portable_paths(self):
        fingerprint = self.fingerprint()
        self.assertEqual(
            fingerprint["raw_input_sha256"],
            {"S2/S2.pkl": hashlib.sha256(self.path.read_bytes()).hexdigest()},
        )
        relocated = self.root / "relocated"
        shutil.copytree(self.path.parent, relocated / "S2")
        self.assertEqual(
            fingerprint, cache.cache_fingerprint("chest", relocated, ["S2"])
        )

    def test_raw_change_with_same_size_and_mtime_changes_identity(self):
        before = self.fingerprint()
        stat = self.path.stat()
        self.path.write_bytes(b"modified recording")
        os.utime(self.path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
        self.assertEqual(stat.st_size, self.path.stat().st_size)
        self.assertEqual(stat.st_mtime_ns, self.path.stat().st_mtime_ns)
        self.assertNotEqual(before, self.fingerprint())

    def test_missing_raw_input_cannot_be_verified(self):
        self.path.unlink()
        self.assertRaises(FileNotFoundError, self.fingerprint)

    def test_runtime_filter_changes_require_restart(self):
        before = self.fingerprint()
        settings = replace(config.FILTER_PARAMS, ECG_BANDPASS_HIGH=35.0)
        with (
            patch.object(config, "FILTER_PARAMS", settings),
            self.assertRaisesRegex(RuntimeError, "settings changed; restart Python"),
        ):
            self.fingerprint()
        self.assertEqual(before, self.fingerprint())

    def test_module_filter_alias_changes_cannot_mislabel_extracted_values(self):
        processor = eda_processor.EDAProcessor(sampling_rate=700)
        time = np.arange(7000) / 700
        eda = 2 + np.sin(2 * np.pi * 3 * time)
        expected = processor.lowpass_filter(eda)
        settings = replace(config.FILTER_PARAMS, EDA_LOWPASS=1.0)
        with patch.object(eda_processor, "FILTER_PARAMS", settings):
            self.assertFalse(np.array_equal(expected, processor.lowpass_filter(eda)))
            with self.assertRaisesRegex(
                RuntimeError, "settings changed; restart Python"
            ):
                self.fingerprint()

    def test_dependency_changes_change_identity(self):
        before = self.fingerprint()
        actual_version = cache.version
        with patch.object(
            cache,
            "version",
            side_effect=lambda name: (
                "changed" if name == "scipy" else actual_version(name)
            ),
        ):
            self.assertNotEqual(before, self.fingerprint())

    def test_optional_dependency_absence_is_recorded(self):
        actual_version = cache.version

        def installed_version(name):
            if name == "neurokit2":
                raise cache.PackageNotFoundError(name)
            return actual_version(name)

        with patch.object(cache, "version", side_effect=installed_version):
            self.assertIsNone(
                cache.extraction_context("chest")["packages"]["neurokit2"]
            )

    def test_source_changes_require_restart_and_invalidate_identity(self):
        source_root = self.root / "source"
        for relative in cache.source_sha256():
            target = source_root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(cache.SOURCE_ROOT / relative, target)
        with patch.object(cache, "SOURCE_ROOT", source_root):
            initial = cache.source_sha256()
            with patch.object(cache, "IMPORTED_SOURCE_SHA256", initial):
                before = self.fingerprint()
                source = source_root / "src/preprocessing/filters.py"
                source.write_bytes(source.read_bytes() + b"\n# changed filter\n")
                self.assertRaisesRegex(RuntimeError, "restart Python", self.fingerprint)
                with patch.object(
                    cache, "IMPORTED_SOURCE_SHA256", cache.source_sha256()
                ):
                    self.assertNotEqual(before, self.fingerprint())

    def test_unrelated_files_do_not_change_extraction_identity(self):
        source_root = self.root / "source"
        for relative in cache.source_sha256():
            target = source_root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(cache.SOURCE_ROOT / relative, target)
        with patch.object(cache, "SOURCE_ROOT", source_root):
            before = cache.source_sha256()
            (source_root / "README.md").write_text("updated documentation")
            models = source_root / "src/models"
            models.mkdir()
            (models / "classifier.py").write_text("updated model")
            self.assertEqual(before, cache.source_sha256())

    def test_build_rejects_changed_settings_or_loaded_recordings(self):
        context = cache.extraction_context("chest")
        with (
            patch.object(
                config, "FILTER_PARAMS", replace(config.FILTER_PARAMS, EDA_LOWPASS=4.0)
            ),
            self.assertRaisesRegex(RuntimeError, "settings changed"),
        ):
            cache.cache_fingerprint("chest", self.root, ["S2"], context=context)
        loaded = cache.raw_input_sha256(self.root, ["S2"])
        self.path.write_bytes(b"modified recording")
        with self.assertRaisesRegex(RuntimeError, "recordings changed"):
            cache.cache_fingerprint("chest", self.root, ["S2"], loaded_inputs=loaded)

    def test_invalid_subjects_and_device_are_rejected(self):
        for subjects in ([], ["S2", "S2"], ["../S2"], ["S12"]):
            with (
                self.subTest(subjects=subjects),
                self.assertRaises(ValueError),
            ):
                cache.raw_input_sha256(self.root, subjects)
        self.assertRaises(ValueError, cache.extraction_context, "unknown")

    def test_changes_during_hashing_and_reading_are_rejected(self):
        digest_file = hashlib.file_digest

        def changed_while_hashing(stream, algorithm):
            digest = digest_file(stream, algorithm)
            self.path.write_bytes(b"modified recording")
            return digest

        with (
            patch.object(hashlib, "file_digest", side_effect=changed_while_hashing),
            self.assertRaisesRegex(RuntimeError, "while hashing"),
        ):
            cache.raw_input_sha256(self.root, ["S2"])
        with self.assertRaisesRegex(RuntimeError, "while reading"):
            with cache.verified_input(self.path):
                replacement = self.path.with_suffix(".replacement")
                replacement.write_bytes(self.path.read_bytes())
                replacement.replace(self.path)


class LoaderFingerprintTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(self.enterContext(TemporaryDirectory()))
        self.path = self.root / "S2" / "S2.pkl"
        self.path.parent.mkdir()
        self.data = {
            "signal": {
                "chest": {"ECG": np.zeros((700, 1))},
                "wrist": {"EDA": np.zeros((4, 1))},
            },
            "label": np.ones(700, dtype=int),
        }
        self.path.write_bytes(pickle.dumps(self.data))
        self.loader = loader.WESADLoader(self.root)

    def test_loaded_recording_matches_full_file_digest(self):
        result = self.loader.load_subject("S2")
        np.testing.assert_array_equal(result["label"], self.data["label"])
        self.assertEqual(
            self.loader.raw_input_sha256, cache.raw_input_sha256(self.root, ["S2"])
        )

    def test_change_during_unpickling_is_rejected_without_recording_digest(self):
        load = pickle.load

        def changed_during_load(stream, **kwargs):
            result = load(stream, **kwargs)
            self.path.write_bytes(self.path.read_bytes() + b"changed")
            return result

        with (
            patch.object(pickle, "load", side_effect=changed_during_load),
            self.assertRaisesRegex(RuntimeError, "while reading"),
        ):
            self.loader.load_subject("S2")
        self.assertEqual(self.loader.raw_input_sha256, {})

    def test_invalid_recording_does_not_receive_a_digest(self):
        self.path.write_bytes(pickle.dumps({"signal": {}}))
        with self.assertRaisesRegex(ValueError, "Invalid WESAD structure"):
            self.loader.load_subject("S2")
        self.assertEqual(self.loader.raw_input_sha256, {})


class DatasetCacheTests(PatchedTestCase):
    def setUp(self):
        self.directory = Path(self.enterContext(TemporaryDirectory()))
        self.loader = source_loader(self.enterContext(TemporaryDirectory()))
        for module in (dataset, dataset_wrist):
            for name, value in (
                ("PROCESSED_DATA_DIR", self.directory),
                ("WESAD_DIR", self.loader.data_path),
            ):
                self.patch_object(module, name, value)
        self.feature_path = self.directory / "features.parquet"
        self.raw_path = self.directory / "raw_windows.npz"
        self.frame = pd.DataFrame(
            {
                "subject_id": ["S2", "S2"],
                "window_id": [0, 42000],
                "label": [1, 2],
                **{
                    name: [0.0, 1.0]
                    for name in dataset.FeatureExtractionPipeline().get_feature_names()
                },
                "label_name": ["baseline", "stress"],
            }
        )
        self.frame.attrs = {
            "feature_schema_version": FEATURE_SCHEMA_VERSION,
            "rr_correction_version": RR_CORRECTION_VERSION,
            "scr_detection_version": SCR_DETECTION_VERSION,
            "extraction_fingerprint": cache_fingerprint(
                "chest", self.loader.data_path, VALID_SUBJECTS
            ),
            "dataset_parameters": {
                "fs": 700.0,
                "window_samples": 42000,
                "step": 21000,
                "purity": 0.9,
                "cnn_length": 1024,
                "subjects": sorted(VALID_SUBJECTS),
            },
        }
        self.raw = np.zeros((2, 5, 1024), dtype=np.float32)
        self.builder = dataset.WindowedDataset.__new__(dataset.WindowedDataset)
        self.builder.loader = self.loader

    def snapshot(self):
        return {
            path.name: path.read_bytes()
            for path in (self.feature_path, self.raw_path)
            if path.exists()
        }

    def save_caches(self):
        self.builder._save(self.frame, self.raw)
        self.wrist_builder().build()
        self.assertIsNotNone(dataset.load_cached())
        self.assertIsNotNone(dataset_wrist.load_wrist())
        return {path.name: path.read_bytes() for path in self.directory.iterdir()}

    def test_cache_loaders_accept_an_explicit_source_path(self):
        self.save_caches()
        missing = self.loader.data_path / "missing"
        with (
            patch.object(dataset, "WESAD_DIR", missing),
            patch.object(dataset_wrist, "WESAD_DIR", missing),
        ):
            self.assertIsNotNone(dataset.load_cached(self.loader.data_path))
            self.assertIsNotNone(dataset_wrist.load_wrist(str(self.loader.data_path)))
            self.assertIsNone(dataset.load_cached())
            self.assertIsNone(dataset_wrist.load_wrist())

    def test_integer_acceleration_is_finite_in_actual_chest_and_wrist_windows(self):
        seconds = 60
        chest_count = int(seconds * FS.CHEST)
        chest_time = np.arange(chest_count) / FS.CHEST
        wrist_time = np.arange(int(seconds * FS.WRIST_BVP)) / FS.WRIST_BVP
        recording = {
            "signal": {
                "chest": {
                    "ECG": np.sin(2 * np.pi * 1.2 * chest_time).reshape(-1, 1),
                    "EDA": np.full((chest_count, 1), 2.0),
                    "Temp": np.full((chest_count, 1), 33.0),
                    "Resp": np.sin(2 * np.pi * 0.25 * chest_time).reshape(-1, 1),
                    "ACC": np.full((chest_count, 3), 200, dtype=np.int16),
                },
                "wrist": {
                    "BVP": np.sin(2 * np.pi * 1.2 * wrist_time).reshape(-1, 1),
                    "EDA": np.full((int(seconds * FS.WRIST_EDA), 1), 2.0),
                    "TEMP": np.full((int(seconds * FS.WRIST_TEMP), 1), 33.0),
                    "ACC": np.full(
                        (int(seconds * FS.WRIST_ACC), 3), 200, dtype=np.int16
                    ),
                },
            },
            "label": np.ones(chest_count, dtype=int),
        }
        path = self.loader.data_path / "S2" / "S2.pkl"
        path.write_bytes(pickle.dumps(recording, protocol=4))
        expected = np.sqrt(3 * 200.0**2)
        for builder_type in (dataset.WindowedDataset, dataset_wrist.WristDataset):
            with self.subTest(dataset=builder_type.__name__):
                builder = builder_type(data_path=self.loader.data_path)
                loaded = builder.loader.load_subject("S2")
                self.assertEqual(loaded["chest"]["ACC"].dtype, np.dtype(np.int16))
                self.assertEqual(loaded["wrist"]["ACC"].dtype, np.dtype(np.int16))
                result = builder._process_subject("S2")
                windows = result[0]
                self.assertEqual(len(windows), 1)
                magnitude = windows[0]["accelerometer"]["magnitude"]
                self.assertTrue(np.isfinite(magnitude).all())
                np.testing.assert_allclose(magnitude, expected)
                if builder_type is dataset.WindowedDataset:
                    cnn_acceleration = result[1][0][4]
                    self.assertTrue(np.isfinite(cnn_acceleration).all())
                    np.testing.assert_allclose(cnn_acceleration, expected, rtol=1e-6)

    def test_caches_require_a_matching_extraction_fingerprint(self):
        self.save_caches()
        for path, load in (
            (self.feature_path, dataset.load_cached),
            (self.directory / "features_wrist.parquet", dataset_wrist.load_wrist),
        ):
            for fingerprint in (None, {"different": True}):
                with self.subTest(cache=path.name, fingerprint=fingerprint):
                    frame = pd.read_parquet(path)
                    if fingerprint is None:
                        frame.attrs.pop("extraction_fingerprint", None)
                    else:
                        frame.attrs["extraction_fingerprint"] = fingerprint
                    frame.to_parquet(path, index=False)
                    self.assertIsNone(load())

    def test_caches_reject_raw_byte_changes_with_unchanged_size_and_mtime(self):
        before = self.save_caches()
        path = self.loader.data_path / "S2" / "S2.pkl"
        state = path.stat()
        record = pickle.loads(path.read_bytes())
        record["signal"]["wrist"]["EDA"][0, 0] = 1.0
        path.write_bytes(pickle.dumps(record, protocol=4))
        os.utime(path, ns=(state.st_atime_ns, state.st_mtime_ns))
        self.assertEqual(path.stat().st_size, state.st_size)
        self.assertEqual(path.stat().st_mtime_ns, state.st_mtime_ns)
        self.assertIsNone(dataset.load_cached())
        self.assertIsNone(dataset_wrist.load_wrist())
        self.assertEqual(
            {path.name: path.read_bytes() for path in self.directory.iterdir()}, before
        )

    def test_caches_reject_missing_raw_recordings(self):
        before = self.save_caches()
        (self.loader.data_path / "S2" / "S2.pkl").unlink()
        self.assertIsNone(dataset.load_cached())
        self.assertIsNone(dataset_wrist.load_wrist())
        self.assertEqual(
            {path.name: path.read_bytes() for path in self.directory.iterdir()}, before
        )

    def test_caches_reject_changed_filter_parameters(self):
        self.save_caches()
        changed = replace(config.FILTER_PARAMS, EDA_FILTER_ORDER=6)
        with patch.object(config, "FILTER_PARAMS", changed):
            self.assertIsNone(dataset.load_cached())
            self.assertIsNone(dataset_wrist.load_wrist())

    def test_caches_reject_extraction_source_changes(self):
        with TemporaryDirectory() as directory:
            source_root = Path(directory)
            for relative in cache.IMPORTED_SOURCE_SHA256:
                path = source_root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes((cache.SOURCE_ROOT / relative).read_bytes())
            with patch.object(cache, "SOURCE_ROOT", source_root):
                self.save_caches()
                source = source_root / "src" / "features" / "eda_features.py"
                source.write_text(
                    source.read_text() + "\n# Changed extraction source.\n"
                )
                self.assertIsNone(dataset.load_cached())
                self.assertIsNone(dataset_wrist.load_wrist())

    def test_build_rejects_raw_changes_after_loading_before_saving(self):
        before = self.save_caches()
        path = self.loader.data_path / "S2" / "S2.pkl"
        original = path.read_bytes()
        for builder_type in (dataset.WindowedDataset, dataset_wrist.WristDataset):
            with self.subTest(dataset=builder_type.__name__):
                builder = builder_type(data_path=self.loader.data_path)

                def process(subject):
                    builder.loader.load_subject(subject)
                    record = pickle.loads(original)
                    record["signal"]["wrist"]["EDA"][0, 0] = 1.0
                    path.write_bytes(pickle.dumps(record, protocol=4))
                    return (
                        ([], [], [])
                        if builder_type is dataset.WindowedDataset
                        else ([], [])
                    )

                try:
                    with (
                        patch.object(builder, "_process_subject", process),
                        patch.object(
                            pd.DataFrame,
                            "to_parquet",
                            side_effect=AssertionError("published changed inputs"),
                        ),
                        self.assertRaisesRegex(RuntimeError, "Raw recordings changed"),
                    ):
                        builder.build(subjects=["S2"])
                finally:
                    path.write_bytes(original)
                self.assertEqual(
                    {path.name: path.read_bytes() for path in self.directory.iterdir()},
                    before,
                )

    def test_build_rejects_filter_changes_during_processing_before_saving(self):
        before = self.save_caches()
        original = config.FILTER_PARAMS
        for builder_type in (dataset.WindowedDataset, dataset_wrist.WristDataset):
            with self.subTest(dataset=builder_type.__name__):
                builder = builder_type(data_path=self.loader.data_path)

                def process(subject):
                    builder.loader.load_subject(subject)
                    config.FILTER_PARAMS = replace(original, EDA_FILTER_ORDER=6)
                    return (
                        ([], [], [])
                        if builder_type is dataset.WindowedDataset
                        else ([], [])
                    )

                with (
                    patch.object(config, "FILTER_PARAMS", original),
                    patch.object(builder, "_process_subject", process),
                    patch.object(
                        pd.DataFrame,
                        "to_parquet",
                        side_effect=AssertionError("published changed settings"),
                    ),
                    self.assertRaisesRegex(RuntimeError, "Extraction settings changed"),
                ):
                    builder.build(subjects=["S2"])
                self.assertEqual(
                    {path.name: path.read_bytes() for path in self.directory.iterdir()},
                    before,
                )

    def test_save_binds_files_without_changing_caller_or_feature_identity(self):
        before = deepcopy(self.frame.attrs)
        identity = feature_frame_sha256(self.frame)
        self.builder._save(self.frame, self.raw)
        self.assertEqual(self.frame.attrs, before)
        persisted = pd.read_parquet(self.feature_path)
        self.assertEqual(
            persisted.attrs["raw_windows_sha256"], sha256_file(self.raw_path)
        )
        frame, raw = dataset.load_cached()
        pd.testing.assert_frame_equal(frame, self.frame)
        self.assertEqual(frame.attrs, before)
        self.assertEqual(feature_frame_sha256(frame), identity)
        np.testing.assert_array_equal(raw, self.raw)

    def test_serialization_failures_preserve_existing_pair_or_leave_no_cache(self):
        for existing in (False, True):
            for serializer in ("raw", "features"):
                with self.subTest(existing=existing, serializer=serializer):
                    self.feature_path.unlink(missing_ok=True)
                    self.raw_path.unlink(missing_ok=True)
                    if existing:
                        self.builder._save(self.frame, self.raw)
                    before = self.snapshot()

                    def fail_serialization(*args, **kwargs):
                        path = args[0] if serializer == "raw" else args[1]
                        Path(path).write_bytes(b"incomplete serialization")
                        raise OSError("serialization failed")

                    target = (
                        patch.object(
                            dataset.np,
                            "savez_compressed",
                            side_effect=fail_serialization,
                        )
                        if serializer == "raw"
                        else patch.object(
                            pd.DataFrame,
                            "to_parquet",
                            autospec=True,
                            side_effect=fail_serialization,
                        )
                    )
                    with (
                        target,
                        self.assertRaisesRegex(OSError, "serialization failed"),
                    ):
                        self.builder._save(self.frame, self.raw + 1)
                    self.assertEqual(self.snapshot(), before)
                    self.assertEqual(
                        {path.name for path in self.directory.iterdir()}, set(before)
                    )
                    if existing:
                        np.testing.assert_array_equal(
                            dataset.load_cached()[1], self.raw
                        )
                    else:
                        self.assertIsNone(dataset.load_cached())

    def test_second_publication_failure_restores_previous_pair(self):
        for existing in (False, True):
            with self.subTest(existing=existing):
                self.feature_path.unlink(missing_ok=True)
                self.raw_path.unlink(missing_ok=True)
                if existing:
                    self.builder._save(self.frame, self.raw)
                before = self.snapshot()
                replace = Path.replace

                def fail_raw_publication(source, destination):
                    if source.name == self.raw_path.name:
                        raise OSError("raw publication failed")
                    return replace(source, destination)

                with (
                    patch.object(Path, "replace", fail_raw_publication),
                    self.assertRaisesRegex(OSError, "raw publication failed"),
                ):
                    self.builder._save(self.frame, self.raw + 1)
                self.assertEqual(self.snapshot(), before)
                self.assertEqual(
                    {path.name for path in self.directory.iterdir()}, set(before)
                )
                if existing:
                    np.testing.assert_array_equal(dataset.load_cached()[1], self.raw)
                else:
                    self.assertIsNone(dataset.load_cached())

    def test_legacy_cache_without_pair_checksum_requires_rebuild(self):
        self.builder._save(self.frame, self.raw)
        self.frame.to_parquet(self.feature_path, index=False)
        self.assertIsNone(dataset.load_cached())

    def assert_stale_versions(
        self, path, load, attribute, version, *, include_future=False
    ):
        versions = [None, version - 1]
        if include_future:
            versions.append(version + 1)
        for invalid in versions:
            with self.subTest(version=invalid):
                frame = pd.read_parquet(path)
                if invalid is None:
                    frame.attrs.pop(attribute, None)
                else:
                    frame.attrs[attribute] = invalid
                frame.to_parquet(path, index=False)
                self.assertIsNone(load())

    def assert_wrist_failure_preserves_cache(self, failure, message):
        path = self.directory / "features_wrist.parquet"
        builder = self.wrist_builder()
        for existing in (False, True):
            with self.subTest(existing=existing):
                path.unlink(missing_ok=True)
                expected = builder.build() if existing else None
                before = path.read_bytes() if existing else None
                with failure:
                    self.assertRaisesRegex(OSError, message, builder.build)
                self.assertEqual(path.read_bytes() if path.exists() else None, before)
                if existing:
                    pd.testing.assert_frame_equal(dataset_wrist.load_wrist(), expected)
                else:
                    self.assertIsNone(dataset_wrist.load_wrist())
                self.assertEqual(
                    set(self.directory.iterdir()), {path} if existing else set()
                )

    def test_chest_cache_requires_current_rr_correction(self):
        self.builder._save(self.frame, self.raw)
        self.assert_stale_versions(
            self.feature_path,
            dataset.load_cached,
            "rr_correction_version",
            RR_CORRECTION_VERSION,
        )

    def test_chest_cache_requires_current_scr_detection(self):
        self.builder._save(self.frame, self.raw)
        self.assertIsNotNone(dataset.load_cached())
        self.assert_stale_versions(
            self.feature_path,
            dataset.load_cached,
            "scr_detection_version",
            SCR_DETECTION_VERSION,
            include_future=True,
        )

    def test_fresh_builds_record_processing_versions(self):
        loader = SimpleNamespace(
            subjects=["S2"],
            data_path=self.loader.data_path,
            raw_input_sha256=raw_input_sha256(self.loader.data_path, ["S2"]),
        )
        with (
            patch.object(dataset, "WESADLoader", return_value=loader),
            patch.object(dataset_wrist, "WESADLoader", return_value=loader),
        ):
            for builder, subject_data in (
                (dataset.WindowedDataset(), ([], [], [])),
                (dataset_wrist.WristDataset(), ([], [])),
            ):
                with self.subTest(dataset=type(builder).__name__):
                    with patch.object(
                        builder, "_process_subject", return_value=subject_data
                    ):
                        result = builder.build(subjects=["S2"], cache=False)
                    frame = result[0] if isinstance(result, tuple) else result
                    self.assertEqual(
                        frame.attrs["rr_correction_version"], RR_CORRECTION_VERSION
                    )
                    self.assertEqual(
                        frame.attrs["scr_detection_version"], SCR_DETECTION_VERSION
                    )
                    self.assertEqual(
                        frame.attrs["feature_schema_version"], FEATURE_SCHEMA_VERSION
                    )
                    self.assertEqual(
                        frame.attrs["extraction_fingerprint"],
                        cache_fingerprint(
                            (
                                "chest"
                                if isinstance(builder, dataset.WindowedDataset)
                                else "wrist"
                            ),
                            loader.data_path,
                            ["S2"],
                        ),
                    )

    def test_interrupted_publication_cannot_load_mixed_generations(self):
        self.builder._save(self.frame, self.raw)
        old = self.snapshot()
        changed = self.frame.copy(deep=True)
        changed["TEMP_mean"] += 2
        self.builder._save(changed, self.raw + 1)
        new = self.snapshot()
        for feature_bytes, raw_bytes in (
            (old["features.parquet"], new["raw_windows.npz"]),
            (new["features.parquet"], old["raw_windows.npz"]),
        ):
            with self.subTest(new_features=feature_bytes == new["features.parquet"]):
                self.feature_path.write_bytes(feature_bytes)
                self.raw_path.write_bytes(raw_bytes)
                self.assertIsNone(dataset.load_cached())

    def test_checksum_failure_is_rejected_before_npz_parsing(self):
        self.builder._save(self.frame, self.raw)
        self.raw_path.write_bytes(b"incomplete raw cache")
        with patch.object(dataset.np, "load", side_effect=AssertionError("parsed NPZ")):
            self.assertIsNone(dataset.load_cached())

    def test_verified_npz_uses_the_same_open_file_when_path_is_replaced(self):
        self.builder._save(self.frame, self.raw)
        digest = dataset.hashlib.file_digest

        def replace_after_hash(stream, algorithm):
            result = digest(stream, algorithm)
            if Path(stream.name) == self.raw_path:
                replacement = self.directory / "replacement.npz"
                replacement.write_bytes(b"replacement after hash")
                replacement.replace(self.raw_path)
            return result

        with patch.object(dataset.hashlib, "file_digest", replace_after_hash):
            np.testing.assert_array_equal(dataset.load_cached()[1], self.raw)
        self.assertEqual(self.raw_path.read_bytes(), b"replacement after hash")

    def wrist_builder(self):
        builder = dataset_wrist.WristDataset.__new__(dataset_wrist.WristDataset)
        builder.loader = self.loader
        builder.features = SimpleNamespace(
            extract_all_features=lambda windows, **kwargs: self.frame.assign(
                subject_id=windows[0]["subject_id"]
            )
        )
        builder._process_subject = lambda subject: ([{"subject_id": subject}], [])
        builder.window_sec, builder.overlap, builder.purity = 60.0, 0.5, 0.9
        return builder

    def test_wrist_cache_rejects_nonnumeric_features(self):
        path = self.directory / "features_wrist.parquet"
        expected = self.wrist_builder().build()
        for value in ("invalid", "0.5"):
            with self.subTest(value=value):
                frame = expected.copy(deep=True)
                frame["TEMP_mean"] = value
                frame.to_parquet(path, index=False)
                with self.assertRaisesRegex(ValueError, "current schema"):
                    dataset_wrist.load_wrist()

    def test_wrist_cache_requires_current_rr_correction(self):
        self.wrist_builder().build()
        self.assertIsNotNone(dataset_wrist.load_wrist())
        self.assert_stale_versions(
            self.directory / "features_wrist.parquet",
            dataset_wrist.load_wrist,
            "rr_correction_version",
            RR_CORRECTION_VERSION,
        )

    def test_wrist_cache_requires_current_scr_detection(self):
        self.wrist_builder().build()
        self.assertIsNotNone(dataset_wrist.load_wrist())
        self.assert_stale_versions(
            self.directory / "features_wrist.parquet",
            dataset_wrist.load_wrist,
            "scr_detection_version",
            SCR_DETECTION_VERSION,
            include_future=True,
        )

    def test_wrist_cache_requires_current_eda_filtering(self):
        cache = self.directory / "features_wrist.parquet"
        filtering = {
            "eda_sampling_rate": FS.WRIST_EDA,
            "eda_lowpass_hz": FILTER_PARAMS.WRIST_EDA_LOWPASS,
            "eda_filter_order": FILTER_PARAMS.EDA_FILTER_ORDER,
        }
        with patch.object(dataset_wrist, "PROCESSED_DATA_DIR", self.directory):
            expected = self.wrist_builder().build()
            for name, value in filtering.items():
                self.assertEqual(expected.attrs["dataset_parameters"][name], value)
            pd.testing.assert_frame_equal(dataset_wrist.load_wrist(), expected)

            for name, value in filtering.items():
                for replacement in (None, value + 1):
                    with self.subTest(parameter=name, value=replacement):
                        frame = expected.copy(deep=True)
                        frame.attrs = deepcopy(expected.attrs)
                        parameters = frame.attrs["dataset_parameters"]
                        if replacement is None:
                            parameters.pop(name)
                        else:
                            parameters[name] = replacement
                        frame.to_parquet(cache, index=False)
                        self.assertIsNone(dataset_wrist.load_wrist())

    def test_wrist_serialization_failure_preserves_previous_cache(self):
        def fail_serialization(frame, path, **kwargs):
            Path(path).write_bytes(b"incomplete parquet")
            raise OSError("wrist serialization failed")

        self.assert_wrist_failure_preserves_cache(
            patch.object(
                pd.DataFrame,
                "to_parquet",
                autospec=True,
                side_effect=fail_serialization,
            ),
            "wrist serialization failed",
        )

    def test_wrist_publication_failure_preserves_previous_cache(self):
        self.assert_wrist_failure_preserves_cache(
            patch.object(
                Path, "replace", side_effect=OSError("wrist publication failed")
            ),
            "wrist publication failed",
        )


class DownloadResponse(io.BytesIO):

    def __init__(self, payload, *, length=None, url="https://example.com/data.zip"):
        super().__init__(payload)
        self.headers = {
            "Content-Length": str(len(payload) if length is None else length)
        }
        self.url = url

    def geturl(self):
        return self.url


class DownloadTests(TemporaryWorkspaceTestCase):

    def test_safe_extract_unpacks_benign_zip(self):
        archive = self.tmp_path / "ok.zip"
        with zipfile.ZipFile(archive, "w") as z:
            z.writestr("a/b.txt", "hello")
        out = self.tmp_path / "out"
        _safe_extract(archive, out)
        self.assertEqual((out / "a" / "b.txt").read_text(), "hello")

    def test_safe_extract_blocks_zip_slip(self):
        archive = self.tmp_path / "evil.zip"
        with zipfile.ZipFile(archive, "w") as z:
            z.writestr("../escape.txt", "pwned")
        with self.assertRaisesRegex(RuntimeError, "unsafe path"):
            _safe_extract(archive, self.tmp_path / "out")

    def test_safe_extract_blocks_zip_bomb(self):
        archive = self.tmp_path / "big.zip"
        with zipfile.ZipFile(archive, "w") as z:
            z.writestr("data.bin", b"x" * 4096)
        with self.assertRaisesRegex(RuntimeError, "size cap"):
            _safe_extract(archive, self.tmp_path / "out", max_bytes=1024)

    def test_wesad_download_accepts_recorded_uncompressed_size(self):
        provenance = (
            Path(__file__).resolve().parents[1] / "outputs/results/provenance.json"
        )
        expanded_bytes = json.loads(provenance.read_text())["input_verification"][
            "wesad"
        ]["bytes"]
        payload = b"verified recording"
        self.patch_object(download_data, "RAW_DATA_DIR", self.tmp_path)
        self.patch_object(
            download_data,
            "WESAD_SHA256",
            {"S2": hashlib.sha256(payload).hexdigest()},
        )

        def download_archive(url, destination):
            destination.write_bytes(b"archive placeholder")

        self.patch_object(download_data, "_download", download_archive)
        archive = self.patch_object(download_data.zipfile, "ZipFile")
        contents = archive.return_value.__enter__.return_value
        entry = zipfile.ZipInfo("WESAD/S2/S2.pkl")
        entry.file_size = expanded_bytes
        contents.infolist.return_value = [entry]

        def extract_recording(destination):
            subject = destination / "WESAD" / "S2"
            subject.mkdir(parents=True)
            (subject / "S2.pkl").write_bytes(payload)

        contents.extractall.side_effect = extract_recording
        download_data.download_wesad()
        self.assertEqual((self.tmp_path / "WESAD/S2/S2.pkl").read_bytes(), payload)
        self.assertFalse((self.tmp_path / "WESAD.zip").exists())

    def test_download_refuses_non_https(self):
        with self.assertRaisesRegex(ValueError, "non-HTTPS"):
            _download("http://example.com/x.zip", self.tmp_path / "x.zip")

    def test_link_check_refuses_non_https_urls_and_redirects(self):
        with self.assertRaisesRegex(ValueError, "non-HTTPS"):
            download_data._check("http://example.com/data.zip")
        response = DownloadResponse(b"", url="http://example.com/data.zip")
        with (
            patch.object(download_data, "urlopen", return_value=response),
            self.assertRaisesRegex(ValueError, "non-HTTPS"),
        ):
            download_data._check("https://example.com/data.zip")

    def _case_existing_wesad_must_be_complete_and_verified(self, missing_subject):
        self.patch_object(download_data, "RAW_DATA_DIR", self.tmp_path)
        reference = {}
        for sid in ("S2", "S3"):
            payload = sid.encode()
            reference[sid] = hashlib.sha256(payload).hexdigest()
            if sid == missing_subject:
                continue
            subject = self.tmp_path / "WESAD" / sid
            subject.mkdir(parents=True)
            (subject / f"{sid}.pkl").write_bytes(payload)
        self.patch_object(download_data, "WESAD_SHA256", reference)
        self.patch_object(
            download_data, "_download", lambda *args: self.fail("Unexpected download")
        )
        if missing_subject:
            with self.assertRaisesRegex(SystemExit, f"{missing_subject}: missing"):
                download_data.download_wesad()
        else:
            download_data.download_wesad()

    def test_download_streams_to_complete_file(self):
        response = DownloadResponse(b"verified archive")
        calls = []

        def open_response(url, *, timeout):
            calls.append((url, timeout))
            return response

        self.patch_object(download_data, "urlopen", open_response)
        dest = self.tmp_path / "nested" / "data.zip"
        _download("https://example.com/data.zip", str(dest))
        self.assertEqual(dest.read_bytes(), b"verified archive")
        self.assertEqual(
            calls, [("https://example.com/data.zip", download_data.DOWNLOAD_TIMEOUT)]
        )
        self.assertFalse(dest.with_suffix(".zip.part").exists())

    def test_parallel_downloads_use_independent_partial_files(self):
        destination = self.tmp_path / "data.zip"
        first_written = threading.Event()
        second_written = threading.Event()
        first_published = threading.Event()
        outcomes = {}

        class Response(DownloadResponse):
            def __init__(self, name):
                super().__init__(name.encode() * 8)
                self.name = name
                self.started = False

            def read(self, size):
                if not self.started:
                    self.started = True
                    if self.name == "second" and not first_written.wait(10):
                        raise RuntimeError("First download did not write")
                    return super().read(size)
                if self.name == "first":
                    first_written.set()
                    if not second_written.wait(10):
                        raise RuntimeError("Second download did not write")
                else:
                    second_written.set()
                    if not first_published.wait(10):
                        raise RuntimeError("First download did not publish")
                return b""

        replace = Path.replace

        def publish(source, target):
            result = replace(source, target)
            if target == destination and threading.current_thread().name == "first":
                first_published.set()
            return result

        def download(name):
            try:
                _download(f"https://example.com/{name}", destination)
                outcomes[name] = None
            except Exception as error:
                outcomes[name] = error

        threads = [
            threading.Thread(target=download, args=(name,), name=name)
            for name in ("first", "second")
        ]
        with (
            patch.object(
                download_data,
                "urlopen",
                side_effect=lambda url, **kwargs: Response(url.rsplit("/", 1)[-1]),
            ),
            patch.object(Path, "replace", publish),
        ):
            try:
                for thread in threads:
                    thread.start()
                for thread in threads:
                    thread.join(15)
            finally:
                for event in (first_written, second_written, first_published):
                    event.set()
                for thread in threads:
                    thread.join(2)
        self.assertTrue(all(not thread.is_alive() for thread in threads))
        self.assertEqual(outcomes, {"first": None, "second": None})
        self.assertIn(destination.read_bytes(), (b"first" * 8, b"second" * 8))
        self.assertEqual(set(self.tmp_path.iterdir()), {destination})

    def _case_download_failure_preserves_previous_file(self, redirect):
        response = DownloadResponse(
            b"short body",
            length=100,
            url=(
                "http://example.com/data.zip"
                if redirect
                else "https://example.com/data.zip"
            ),
        )
        self.patch_object(download_data, "urlopen", lambda *args, **kwargs: response)
        dest = self.tmp_path / "data.zip"
        dest.write_bytes(b"previous archive")
        error = ValueError if redirect else RuntimeError
        with self.assertRaisesRegex(error, "non-HTTPS|incomplete"):
            _download("https://example.com/data.zip", dest)
        self.assertEqual(dest.read_bytes(), b"previous archive")
        self.assertFalse(dest.with_suffix(".zip.part").exists())

    def _case_existing_noneeg_requires_complete_publisher_manifest(self, problem):
        self.patch_object(download_data, "NONEEG_DIR", self.tmp_path)
        names = ("Subject1_AccTempEDA.dat", "Subject1_SpO2HR.hea")
        self.patch_object(download_data, "NONEEG_FILES", names)
        target = (
            self.tmp_path
            / "non-eeg-dataset-for-assessment-of-neurological-status-1.0.0"
        )
        target.mkdir()
        references = []
        for name in names:
            payload = name.encode()
            (target / name).write_bytes(payload)
            references.append(f"{hashlib.sha256(payload).hexdigest()}  {name}")
        (target / "SHA256SUMS.txt").write_text("\n".join(references) + "\n")
        if problem == "missing":
            (target / names[1]).unlink()
        elif problem == "changed":
            (target / names[1]).write_bytes(b"changed")
        elif problem == "manifest":
            (target / "SHA256SUMS.txt").write_text("invalid digest\n")
        self.patch_object(
            download_data, "_download", lambda *args: self.fail("Unexpected download")
        )
        if problem:
            with self.assertRaisesRegex(SystemExit, "integrity check failed"):
                download_data.download_noneeg()
        else:
            download_data.download_noneeg()

    def _case_dataset_is_installed_only_after_integrity_verification_and_retry_succeeds(
        self, dataset
    ):
        payload = b"verified recording"
        digest = hashlib.sha256(payload).hexdigest()
        valid = False
        if dataset == "wesad":
            self.patch_object(download_data, "RAW_DATA_DIR", self.tmp_path)
            self.patch_object(download_data, "WESAD_SHA256", {"S2": digest})
            target = self.tmp_path / "WESAD"
            file_name = "S2/S2.pkl"
            download = download_data.download_wesad
        else:
            self.patch_object(download_data, "NONEEG_DIR", self.tmp_path)
            self.patch_object(download_data, "NONEEG_FILES", ("record.dat",))
            target = (
                self.tmp_path
                / "non-eeg-dataset-for-assessment-of-neurological-status-1.0.0"
            )
            file_name = "record.dat"
            download = download_data.download_noneeg

        def download_archive(url, destination):
            with zipfile.ZipFile(destination, "w") as archive:
                archive.writestr(
                    f"{target.name}/{file_name}", payload if valid else b"corrupt"
                )
                if dataset == "noneeg":
                    archive.writestr(
                        f"{target.name}/SHA256SUMS.txt", f"{digest}  {file_name}\n"
                    )

        self.patch_object(download_data, "_download", download_archive)
        self.assertRaisesRegex(SystemExit, "checksum mismatch", download)
        self.assertFalse(target.exists())
        self.assertEqual([p.suffix for p in self.tmp_path.iterdir()], [".zip"])
        valid = True
        download()
        self.assertEqual((target / file_name).read_bytes(), payload)
        self.assertEqual(list(self.tmp_path.iterdir()), [target])


add_cases(
    DownloadTests,
    "existing_wesad_must_be_complete_and_verified",
    [(None,), ("S2",), ("S3",)],
)


add_cases(
    DownloadTests, "download_failure_preserves_previous_file", [(False,), (True,)]
)


add_cases(
    DownloadTests,
    "existing_noneeg_requires_complete_publisher_manifest",
    [(None,), ("missing",), ("changed",), ("manifest",)],
)


add_cases(
    DownloadTests,
    "dataset_is_installed_only_after_integrity_verification_and_retry_succeeds",
    [("wesad",), ("noneeg",)],
)


def word(kind, value=0):
    return struct.pack("<H", (kind << 10) | value)


def skip(interval):
    return word(59) + struct.pack("<hH", interval >> 16, interval & 65535)


class RecordTests(unittest.TestCase):
    def setUp(self):
        self.directory = Path(self.enterContext(TemporaryDirectory()))
        self.folder = self.directory / "records"
        self.folder.mkdir()

    def signal(self, values=((1,), (2,)), scalings=None, zero=0):
        digital = np.asarray(values, dtype="<i2")
        samples, channels = digital.shape
        scalings = scalings or ["1/mV"] * channels
        record = self.folder / "sample"
        lines = [f"sample {channels} 8 {samples}"]
        for channel, scaling in enumerate(scalings):
            checksum = int(digital[:, channel].sum(dtype=np.int64))
            checksum = (checksum + 32768) % 65536 - 32768
            lines.append(
                f"signal.dat 16 {scaling} 16 {zero} {digital[0, channel]} "
                f"{checksum} 0 lead {channel}"
            )
        record.with_suffix(".hea").write_text("\n".join(lines) + "\n")
        (self.folder / "signal.dat").write_bytes(digital.tobytes())
        return record

    def change_header(self, record, line, field, value):
        path = record.with_suffix(".hea")
        lines = path.read_text().splitlines()
        fields = lines[line].split()
        fields[field] = str(value)
        lines[line] = " ".join(fields)
        path.write_text("\n".join(lines) + "\n")

    def test_interleaving_scaling_metadata_and_invalid_sample(self):
        record = self.signal([[102, 204], [106, -32768]], ["2(100)/mV", "4(200)/C"])
        result = read_record(record)
        np.testing.assert_allclose(
            result.p_signal, [[1, 1], [3, np.nan]], equal_nan=True
        )
        self.assertEqual(result.fs, 8)
        self.assertEqual(result.units, ["mV", "C"])
        self.assertEqual(result.sig_name, ["lead 0", "lead 1"])

    def test_record_files_uses_header_reference_without_loading_samples(self):
        record = self.signal()
        data = self.folder / "signal.dat"
        data.unlink()
        self.assertEqual(
            record_files(record), (record.with_suffix(".hea"), data.resolve())
        )
        self.change_header(record, 1, 0, "../outside.dat")
        self.assertRaisesRegex(ValueError, "record directory", record_files, record)

    def test_little_endian_signed_samples_and_negative_checksum(self):
        record = self.signal([[0x1234], [-0x2345]])
        self.assertEqual((self.folder / "signal.dat").read_bytes(), b"\x34\x12\xbb\xdc")
        np.testing.assert_array_equal(read_record(record).p_signal[:, 0], [4660, -9029])

    def test_baselines_outside_adc_range_do_not_overflow(self):
        record = self.signal([[0, 0], [10, -10]], ["2(-100000)/C", "2(100000)/C"])
        np.testing.assert_array_equal(
            read_record(record).p_signal, [[50000, -50000], [50005, -50005]]
        )

    def test_missing_baseline_uses_adc_zero(self):
        record = self.signal([[104], [108]], ["2/mV"], zero=100)
        np.testing.assert_array_equal(read_record(record).p_signal[:, 0], [2, 4])

    def test_header_comments_and_blank_lines_preserve_data(self):
        record = self.signal()
        header = record.with_suffix(".hea")
        header.write_text("\n# acquisition\n" + header.read_text() + "  # metadata\n\n")
        np.testing.assert_array_equal(read_record(record).p_signal[:, 0], [1, 2])

    def test_uncalibrated_nonpositive_and_nonfinite_gains_are_rejected(self):
        for gain in ("0", "-1", "nan", "inf", "-inf"):
            with self.subTest(gain=gain):
                record = self.signal(scalings=[f"{gain}/mV"])
                with self.assertRaisesRegex(
                    ValueError, "gain must be finite and positive"
                ):
                    read_record(record)

    def test_unsupported_encoding_modifiers_resolution_and_blocks_are_rejected(self):
        cases = [(1, value) for value in ("212", "16x2", "16:1", "16+2")]
        cases += [(3, "12"), (7, "512")]
        for field, value in cases:
            with self.subTest(field=field, value=value):
                record = self.signal()
                self.change_header(record, 1, field, value)
                with self.assertRaisesRegex(
                    ValueError, "format-16|resolution|block size"
                ):
                    read_record(record)

    def test_invalid_record_dimensions_and_rates_are_rejected(self):
        for field, value in ((1, 0), (1, -1), (2, 0), (2, "nan"), (2, "inf"), (3, 0)):
            with self.subTest(field=field, value=value):
                record = self.signal()
                self.change_header(record, 0, field, value)
                with self.assertRaisesRegex(
                    ValueError, "channel count|sample count|sampling rate"
                ):
                    read_record(record)

    def test_multisegment_and_inconsistent_signal_headers_are_rejected(self):
        for fault in ("segment", "count", "files", "scaling", "empty"):
            with self.subTest(fault=fault):
                record = self.signal([[1, 2], [3, 4]])
                if fault == "segment":
                    self.change_header(record, 0, 0, "sample/2")
                elif fault == "count":
                    self.change_header(record, 0, 1, 1)
                elif fault == "files":
                    self.change_header(record, 2, 0, "other.dat")
                elif fault == "scaling":
                    self.change_header(record, 1, 2, "1(bad)/mV")
                else:
                    record.with_suffix(".hea").write_text("")
                with self.assertRaisesRegex(ValueError, "Invalid Non-EEG header"):
                    read_record(record)

    def test_corrupt_samples_fail_the_header_checksum(self):
        record = self.signal()
        (self.folder / "signal.dat").write_bytes(struct.pack("<hh", 1, 3))
        self.assertRaisesRegex(ValueError, "checksum mismatch", read_record, record)

    def test_truncated_or_extra_signal_bytes_are_rejected(self):
        for data in (b"\x01\x00\x02", b"\x01\x00\x02\x00\x00\x00"):
            with self.subTest(data=data):
                record = self.signal()
                (self.folder / "signal.dat").write_bytes(data)
                self.assertRaisesRegex(ValueError, "byte count", read_record, record)

    def test_signal_paths_cannot_escape_the_record_directory(self):
        outside = self.directory / "outside.dat"
        outside.write_bytes(struct.pack("<hh", 1, 2))
        (self.folder / "linked.dat").symlink_to(outside)
        for filename in ("../outside.dat", str(outside), "linked.dat"):
            with self.subTest(filename=filename):
                record = self.signal()
                self.change_header(record, 1, 0, filename)
                with self.assertRaisesRegex(ValueError, "record directory"):
                    read_record(record)


class AnnotationTests(unittest.TestCase):
    def setUp(self):
        self.record = Path(self.enterContext(TemporaryDirectory())) / "sample"

    def annotations(self, data, extension="atr"):
        self.record.with_suffix(f".{extension}").write_bytes(data)
        return read_annotations(self.record, extension)

    def test_long_and_signed_skips_use_pdp11_word_order(self):
        result = self.annotations(
            skip(70000) + word(22, 10) + skip(-65539) + word(22, 6) + word(0)
        )
        np.testing.assert_array_equal(result.sample, [70010, 4477])
        self.assertEqual(result.aux_note, ["", ""])

    def test_aux_odd_padding_latin1_and_modifiers_do_not_advance_time(self):
        data = word(22, 7) + word(63, 5) + b"Relax\x00"
        data += word(60, 3) + word(61, 2) + word(62, 1)
        data += word(22, 9) + word(63, 2) + b"\xe9C" + word(0)
        result = self.annotations(data)
        np.testing.assert_array_equal(result.sample, [7, 16])
        self.assertEqual(result.aux_note, ["Relax", "\xe9C"])

    def test_empty_notes_and_custom_extension(self):
        result = self.annotations(
            word(22, 0) + word(63, 0) + word(22, 8) + word(0), "labels"
        )
        np.testing.assert_array_equal(result.sample, [0, 8])
        self.assertEqual(result.aux_note, ["", ""])

    def test_empty_annotation_file_requires_a_terminator(self):
        with self.assertRaisesRegex(ValueError, "Missing annotation terminator"):
            self.annotations(b"")
        empty = self.annotations(word(0))
        self.assertEqual(empty.sample.dtype, np.dtype("int64"))
        self.assertEqual(empty.sample.size, 0)
        self.assertEqual(empty.aux_note, [])

    def test_truncated_words_skips_and_notes_are_rejected(self):
        cases = (
            (b"\x01", "Truncated annotation word"),
            (word(59) + b"\x00\x00\x01", "annotation skip"),
            (word(59, 1) + struct.pack("<i", 1), "annotation skip"),
            (word(22) + word(63, 4) + b"ab", "annotation note"),
            (word(22) + word(63, 3) + b"abc", "annotation note"),
        )
        for data, message in cases:
            with self.subTest(data=data):
                self.assertRaisesRegex(ValueError, message, self.annotations, data)

    def test_unattached_aux_and_modifiers_are_rejected(self):
        for kind in (60, 61, 62, 63):
            with (
                self.subTest(kind=kind),
                self.assertRaisesRegex(ValueError, "[Uu]nattached"),
            ):
                self.annotations(word(kind) + word(0))

    def test_missing_terminator_unsupported_code_and_trailing_data_are_rejected(self):
        for data, message in (
            (word(22, 4), "Missing annotation terminator"),
            (word(0, 1) + word(0), "Unsupported annotation code"),
            (word(22) + word(0) + word(22), "after annotation terminator"),
        ):
            with self.subTest(data=data):
                self.assertRaisesRegex(ValueError, message, self.annotations, data)

    def test_nonzero_odd_note_padding_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "padding"):
            self.annotations(word(22) + word(63, 3) + b"abc!" + word(0))


class LoaderValidationTests(unittest.TestCase):
    def setUp(self):
        self.directory = Path(self.enterContext(TemporaryDirectory()))
        self.path = self.directory / "S2" / "S2.pkl"
        self.path.parent.mkdir()
        self.loader = WESADLoader(self.directory)

    def recording(self):
        return {
            "signal": {
                "chest": {"ECG": np.zeros((700, 1))},
                "wrist": {"EDA": np.zeros((4, 1))},
            },
            "label": np.ones(700, dtype=int),
        }

    def load(self, recording):
        self.path.write_bytes(pickle.dumps(recording))
        return self.loader.load_subject("S2")

    def test_complex_labels_are_rejected_before_recording_digest(self):
        for dtype in (np.complex64, np.complex128):
            for imaginary in (0, 1):
                with self.subTest(dtype=dtype, imaginary=imaginary):
                    recording = self.recording()
                    recording["label"] = np.full(700, 1 + imaginary * 1j, dtype=dtype)
                    with self.assertRaisesRegex(ValueError, "labels must be"):
                        self.load(recording)
                    self.assertEqual(self.loader.raw_input_sha256, {})

    def test_complex_signals_are_rejected_before_recording_digest(self):
        for device, channel, samples in (("chest", "ECG", 700), ("wrist", "EDA", 4)):
            for dtype in (np.complex64, np.complex128):
                for imaginary in (0, 1):
                    with self.subTest(device=device, dtype=dtype, imaginary=imaginary):
                        recording = self.recording()
                        recording["signal"][device][channel] = np.full(
                            (samples, 1), 1 + imaginary * 1j, dtype=dtype
                        )
                        with self.assertRaisesRegex(
                            ValueError, "signal shape or dtype"
                        ):
                            self.load(recording)
                        self.assertEqual(self.loader.raw_input_sha256, {})

    def test_real_arrays_preserve_dtypes_and_missing_signal_samples(self):
        for label_dtype in (np.int16, np.float64):
            for signal_dtype in (np.int16, np.float32, np.float64):
                with self.subTest(label_dtype=label_dtype, signal_dtype=signal_dtype):
                    recording = self.recording()
                    recording["label"] = recording["label"].astype(label_dtype)
                    for channels in recording["signal"].values():
                        for name, values in channels.items():
                            values = values.astype(signal_dtype)
                            if np.issubdtype(signal_dtype, np.floating):
                                values[0, 0] = np.nan
                            channels[name] = values
                    loaded = self.load(recording)
                    np.testing.assert_array_equal(loaded["label"], recording["label"])
                    self.assertEqual(loaded["label"].dtype, label_dtype)
                    for device, channels in recording["signal"].items():
                        for name, values in channels.items():
                            np.testing.assert_array_equal(loaded[device][name], values)
                            self.assertEqual(loaded[device][name].dtype, signal_dtype)
                    self.assertEqual(
                        self.loader.raw_input_sha256,
                        raw_input_sha256(self.directory, ["S2"]),
                    )


class ECGFilterTests(PatchedTestCase):
    def test_invalid_cutoff_order_and_nyquist_are_rejected_for_any_input(self):
        processor = ECGProcessor(sampling_rate=700)
        for ecg in ([], np.zeros(100)):
            for low, high in ((10, 5), (5, 5), (5, 350), (5, 400)):
                with self.subTest(samples=len(ecg), low=low, high=high):
                    with self.assertRaisesRegex(ValueError, "0 < low < high < Nyquist"):
                        processor.bandpass_filter(ecg, low=low, high=high)

    def test_empty_input_with_valid_cutoffs_stays_empty(self):
        processor = ECGProcessor(sampling_rate=700)
        filtered = processor.bandpass_filter([], low=5, high=15)
        assert_array_equal(filtered, [])
        self.assertEqual(filtered.dtype, np.dtype(float))


class RRCorrectionTests(PatchedTestCase):
    def setUp(self):
        self.processor = ECGProcessor(sampling_rate=700)

    def test_ectopic_correction_holds_endpoints_instead_of_extrapolating(self):
        for method in ("linear", "cubic"):
            for rr, expected in (
                ([100, 600, 800, 750, 850], [600, 600, 800, 750, 850]),
                ([850, 750, 800, 600, 100], [850, 750, 800, 600, 600]),
            ):
                with self.subTest(method=method, rr=rr):
                    _, mask = self.processor.remove_ectopic_beats(rr)
                    corrected = self.processor.interpolate_artifacts(rr, mask, method)
                    assert_array_equal(corrected, expected)

    def test_linear_interpolation_preserves_anchors_and_fills_interior_gaps(self):
        rr = np.array([0, 600, -1, np.nan, 900, np.inf])
        original = rr.copy()
        corrected = self.processor.interpolate_artifacts(rr)
        assert_array_equal(corrected, [600, 600, 700, 800, 900, 900])
        assert_array_equal(rr, original)

    def test_internal_gaps_cannot_overshoot_neighboring_valid_intervals(self):
        for method in ("linear", "cubic"):
            for rr, expected in (
                ([2500, 200, 700, 200, 2500], 200),
                ([200, 2500, 700, 2500, 200], 2500),
            ):
                with self.subTest(method=method, rr=rr):
                    corrected = self.processor.interpolate_artifacts(
                        rr, [True, True, False, True, True], method
                    )
                    self.assertEqual(corrected[2], expected)
                    assert_array_equal(
                        corrected[[0, 1, 3, 4]], np.array(rr)[[0, 1, 3, 4]]
                    )

    def test_interpolation_stays_within_each_gap_and_preserves_valid_anchors(self):
        rng = np.random.default_rng(112)
        for method in ("linear", "cubic"):
            for _ in range(30):
                rr = rng.uniform(200, 2500, 40)
                mask = rng.random(40) > 0.5
                mask[[3, 36]] = True
                mask[:3] = mask[37:] = False
                anchors = np.flatnonzero(mask)
                corrected = self.processor.interpolate_artifacts(rr, mask, method)
                with self.subTest(method=method, anchors=anchors.tolist()):
                    self.assertEqual(corrected.shape, rr.shape)
                    self.assertTrue(np.isfinite(corrected).all())
                    assert_array_equal(corrected[mask], rr[mask])
                    assert_array_equal(corrected[:3], np.full(3, rr[3]))
                    assert_array_equal(corrected[37:], np.full(3, rr[36]))
                    for left, right in zip(anchors[:-1], anchors[1:]):
                        gap = corrected[left : right + 1]
                        self.assertTrue(np.all(gap >= min(rr[left], rr[right])))
                        self.assertTrue(np.all(gap <= max(rr[left], rr[right])))

    def test_cubic_roundoff_at_a_physiological_bound_does_not_reject_valid_input(self):
        edge = np.nextafter(200.0, np.inf)
        rr = np.array([edge, 0, 0, 0, 0, 0, 0, 200, edge])
        corrected = self.processor.interpolate_artifacts(rr, method="cubic")
        self.assertTrue(np.isfinite(corrected).all())
        self.assertTrue(np.all(corrected >= 200))
        self.assertTrue(np.all(corrected <= edge))
        assert_array_equal(corrected[[0, 7, 8]], rr[[0, 7, 8]])

    def test_physiological_bounds_apply_without_a_mask_or_with_an_all_true_mask(self):
        rr = [100, 199.999, 200, 800, 2500, 2500.001, 3000, np.inf, np.nan]
        for mask in (None, np.ones(len(rr), dtype=bool)):
            with self.subTest(mask=mask):
                corrected = self.processor.interpolate_artifacts(rr, mask)
                assert_array_equal(
                    corrected, [200, 200, 200, 800, 2500, 2500, 2500, 2500, 2500]
                )
        retained, mask = self.processor.remove_ectopic_beats(rr, threshold=100)
        assert_array_equal(retained, [200, 800, 2500])
        assert_array_equal(
            mask, [False, False, True, True, True, False, False, False, False]
        )

    def test_invalid_intervals_do_not_distort_the_local_median(self):
        for rr in ([100, 100, 800], [3000, 3000, 800], [np.nan, np.inf, 800]):
            with self.subTest(rr=rr):
                retained, mask = self.processor.remove_ectopic_beats(rr)
                assert_array_equal(retained, [800])
                assert_array_equal(mask, [False, False, True])

    def test_no_valid_anchor_returns_nan_without_inventing_intervals(self):
        for method in ("linear", "cubic"):
            for rr, mask in (
                ([0, -1, 100, 3000, np.nan, np.inf], None),
                ([800, 810], [False, False]),
            ):
                with self.subTest(method=method, rr=rr):
                    corrected = self.processor.interpolate_artifacts(rr, mask, method)
                    self.assertEqual(len(corrected), len(rr))
                    self.assertTrue(np.isnan(corrected).all())

    def test_one_or_two_anchors_work_for_both_methods(self):
        for method in ("linear", "cubic"):
            for rr, expected in (
                ([0, 800, 0], [800, 800, 800]),
                ([0, 600, 0, 800, 0], [600, 600, 700, 800, 800]),
            ):
                with self.subTest(method=method, rr=rr):
                    assert_array_equal(
                        self.processor.interpolate_artifacts(rr, method=method),
                        expected,
                    )

    def test_valid_input_and_empty_input_keep_their_values(self):
        for method in ("linear", "cubic"):
            for rr in ([], [800], [200, 2500], [600, 800, 750, 850]):
                with self.subTest(method=method, rr=rr):
                    assert_array_equal(
                        self.processor.interpolate_artifacts(rr, method=method), rr
                    )

    def test_units_match_rr_extraction_and_preserve_the_same_bounds(self):
        peaks = np.array([0, 70, 630, 1190, 1260])
        for unit, scale in (("ms", 1), ("s", 0.001), ("samples", 0.7)):
            with self.subTest(unit=unit):
                rr = self.processor.extract_rr_intervals(peaks, unit=unit)
                retained, mask = self.processor.remove_ectopic_beats(rr, unit=unit)
                assert_allclose(retained, np.array([800, 800]) * scale)
                assert_array_equal(mask, [False, True, True, False])
                assert_allclose(
                    self.processor.interpolate_artifacts(rr, mask, unit=unit),
                    np.full(4, 800) * scale,
                )

    def test_local_outlier_threshold_is_consistent_across_units(self):
        for unit, scale in (("ms", 1), ("s", 0.001), ("samples", 0.7)):
            with self.subTest(unit=unit):
                rr = np.array([100, 600, 800, 750, 850]) * scale
                _, mask = self.processor.remove_ectopic_beats(rr, unit=unit)
                assert_array_equal(mask, [False, True, True, True, True])
                assert_allclose(
                    self.processor.interpolate_artifacts(rr, mask, unit=unit),
                    np.array([600, 600, 800, 750, 850]) * scale,
                )

    def test_invalid_options_are_rejected_even_when_no_interpolation_is_needed(self):
        for rr in ([], [800], [0]):
            with self.subTest(rr=rr):
                with self.assertRaisesRegex(ValueError, "method"):
                    self.processor.interpolate_artifacts(rr, method="quadratic")
                with self.assertRaisesRegex(ValueError, "unit"):
                    self.processor.interpolate_artifacts(rr, unit="minutes")
                with self.assertRaisesRegex(ValueError, "unit"):
                    self.processor.remove_ectopic_beats(rr, unit="minutes")
        with self.assertRaisesRegex(ValueError, "Mask length"):
            self.processor.interpolate_artifacts([800, 810], [True])
        for mask in ([True, 2], [True, np.nan], [True, "false"]):
            with self.subTest(mask=mask):
                with self.assertRaisesRegex(ValueError, "boolean"):
                    self.processor.interpolate_artifacts([800, 810], mask)

    def test_corrected_intervals_reach_hrv_extractors_without_losing_beats(self):
        rr = np.full(60, 800.0)
        rr[[0, 1, 15, 16, 40, 59]] = [100, 100, -200, np.inf, 3000, 3000]
        _, mask = self.processor.remove_ectopic_beats(rr)
        corrected = self.processor.interpolate_artifacts(rr, mask)
        assert_array_equal(corrected, np.full(60, 800.0))
        time_features = HRVTimeDomainExtractor().extract_all(corrected)
        self.assertEqual(time_features["MeanNN"], 800)
        self.assertEqual(time_features["SDNN"], 0)
        self.assertEqual(time_features["RMSSD"], 0)
        frequency_features = HRVFrequencyDomainExtractor().extract_all(corrected)
        self.assertAlmostEqual(frequency_features["Total_power"], 0, delta=1e-12)


def component_amplitude(values, time, frequency):
    centered = values - np.mean(values)
    return 2 * abs(np.mean(centered * np.exp(-2j * np.pi * frequency * time)))


class EDAFilterTests(PatchedTestCase):
    def setUp(self):
        self.time = np.arange(480) / FS.WRIST_EDA
        self.eda = (
            2
            + 0.5 * np.sin(2 * np.pi * 0.2 * self.time)
            + 0.25 * np.sin(2 * np.pi * 1.5 * self.time)
        )

    def test_wrist_default_preserves_slow_variation_and_attenuates_high_frequencies(
        self,
    ):
        original = self.eda.copy()
        filtered = EDAProcessor().lowpass_filter(self.eda)
        time, filtered_center = self.time[40:-40], filtered[40:-40]
        self.assertEqual(filtered.shape, self.eda.shape)
        self.assertTrue(np.isfinite(filtered).all())
        self.assertFalse(np.array_equal(filtered, self.eda))
        self.assertAlmostEqual(
            component_amplitude(filtered_center, time, 0.2), 0.5, delta=0.005
        )
        self.assertLess(component_amplitude(filtered_center, time, 1.5), 0.0025)
        assert_array_equal(self.eda, original)

    def test_chest_default_keeps_the_existing_five_hertz_cutoff(self):
        time = np.arange(7000) / FS.CHEST
        eda = 2 + np.sin(2 * np.pi * time) + np.sin(2 * np.pi * 20 * time)
        processor = EDAProcessor(sampling_rate=FS.CHEST)
        assert_array_equal(
            processor.lowpass_filter(eda), processor.lowpass_filter(eda, cutoff=5)
        )

    def test_explicit_cutoff_overrides_the_wrist_default(self):
        eda = 2 + np.sin(2 * np.pi * 0.75 * self.time)
        processor = EDAProcessor()
        default = processor.lowpass_filter(eda)
        lower_cutoff = processor.lowpass_filter(eda, cutoff=0.5)
        default_amplitude = component_amplitude(
            default[40:-40], self.time[40:-40], 0.75
        )
        lower_amplitude = component_amplitude(
            lower_cutoff[40:-40], self.time[40:-40], 0.75
        )
        self.assertGreater(default_amplitude, 0.9)
        self.assertLess(lower_amplitude, default_amplitude / 10)

    def test_invalid_cutoffs_raise_instead_of_returning_unfiltered_data(self):
        for eda in (self.eda, []):
            for cutoff in (2, 5):
                with self.subTest(cutoff=cutoff, samples=len(eda)):
                    with self.assertRaisesRegex(ValueError, "Nyquist"):
                        EDAProcessor().lowpass_filter(eda, cutoff=cutoff)
            for cutoff in (0, -1, np.nan, np.inf):
                with self.subTest(cutoff=cutoff, samples=len(eda)):
                    with self.assertRaisesRegex(ValueError, "cutoff"):
                        EDAProcessor().lowpass_filter(eda, cutoff=cutoff)

    def test_empty_and_constant_inputs_keep_their_shape_and_level(self):
        processor = EDAProcessor()
        assert_array_equal(processor.lowpass_filter([]), [])
        assert_allclose(processor.lowpass_filter(np.full(80, 3.0)), 3.0)

    def test_wrist_windows_receive_filtered_eda_before_decomposition(self):
        recording = {
            "label": np.ones(120 * int(FS.CHEST), dtype=int),
            "wrist": {
                "BVP": np.zeros(120 * int(FS.WRIST_BVP)),
                "EDA": self.eda[:, None],
                "TEMP": np.full((480, 1), 35.0),
                "ACC": np.zeros((120 * int(FS.WRIST_ACC), 3)),
            },
        }
        loader = SimpleNamespace(load_subject=lambda subject: recording)
        with patch("src.dataset_wrist.WESADLoader", return_value=loader):
            builder = WristDataset()
        peaks = np.arange(32, 120 * int(FS.WRIST_BVP), int(FS.WRIST_BVP))
        with patch.object(builder, "_bvp_peaks", return_value=peaks):
            windows, labels = builder._process_subject("S2")
        self.assertEqual(labels, [1, 1, 1])
        middle = windows[1]
        filtered = middle["eda_raw"]
        time = self.time[120:360]
        self.assertEqual(filtered.shape, (240,))
        self.assertAlmostEqual(
            component_amplitude(filtered, time, 0.2), 0.5, delta=0.005
        )
        self.assertLess(component_amplitude(filtered, time, 1.5), 0.0025)
        assert_allclose(
            middle["eda_tonic"] + middle["eda_phasic"], filtered, rtol=1e-12
        )


class SCRDetectionTests(PatchedTestCase):
    def setUp(self):
        self.processor = EDAProcessor(sampling_rate=4)
        self.response = np.array(
            [0.02, 0.01, 0, 0.005, 0.01, 0.015, 0.02, 0.015, 0.01, 0.005, 0, 0.01, 0.02]
        )

    def test_baseline_shifts_preserve_the_known_response_and_its_timing(self):
        original = self.response.copy()
        for sampling_rate in (4, 700):
            samples_per_knot = sampling_rate // 4
            phasic = np.interp(
                np.arange((len(self.response) - 1) * samples_per_knot + 1)
                / samples_per_knot,
                np.arange(len(self.response)),
                self.response,
            )
            processor = EDAProcessor(sampling_rate=sampling_rate)
            for offset in (-1, -0.04, -0.015, 0, 0.04, 1):
                with self.subTest(sampling_rate=sampling_rate, offset=offset):
                    shifted = phasic + offset
                    peaks, responses = processor.detect_scr_peaks(shifted)
                    assert_array_equal(peaks, [6 * samples_per_knot])
                    self.assertEqual(len(responses), 1)
                    self.assertEqual(responses[0]["peak_idx"], 6 * samples_per_knot)
                    self.assertEqual(responses[0]["onset_idx"], 2 * samples_per_knot)
                    self.assertAlmostEqual(responses[0]["amplitude"], 0.02)
                    self.assertEqual(responses[0]["rise_time"], 1)
                    self.assertEqual(responses[0]["recovery_time"], 0.5)
                    self.assertEqual(responses[0]["peak_time"], 1.5)
                    assert_array_equal(shifted, phasic + offset)
        assert_array_equal(self.response, original)

    def test_scr_feature_values_are_invariant_to_a_constant_baseline_shift(self):
        phasic = np.array(
            [
                0.02,
                0.01,
                0,
                0.01,
                0.03,
                0.06,
                0.03,
                0.01,
                0,
                0.01,
                0.02,
                0.04,
                0.02,
                0.01,
                0,
            ]
        )
        extractor = EDAFeatureExtractor(sampling_rate=4)
        _, responses = self.processor.detect_scr_peaks(phasic)
        expected = extractor.extract_phasic_features(responses, len(phasic) / 4)
        self.assertEqual(expected["SCR_count"], 2)
        for offset in (-2, -0.1, 0.1, 2):
            with self.subTest(offset=offset):
                peaks, responses = self.processor.detect_scr_peaks(phasic + offset)
                assert_array_equal(peaks, [5, 11])
                actual = extractor.extract_phasic_features(responses, len(phasic) / 4)
                self.assertEqual(list(actual), list(expected))
                assert_allclose(list(actual.values()), list(expected.values()))

    def test_minimum_amplitude_boundary_tolerates_subtraction_roundoff(self):
        for offset in (-1, -0.1, 0, 0.1, 1):
            with self.subTest(offset=offset):
                peaks, responses = self.processor.detect_scr_peaks(
                    self.response / 2 + offset
                )
                assert_array_equal(peaks, [6])
                self.assertAlmostEqual(responses[0]["amplitude"], 0.01)

    def test_small_responses_are_rejected_even_on_a_high_positive_baseline(self):
        for amplitude in (0.0075, 0.00999999):
            for offset in (-1, 0, 1):
                with self.subTest(amplitude=amplitude, offset=offset):
                    peaks, responses = self.processor.detect_scr_peaks(
                        self.response * (amplitude / 0.02) + offset
                    )
                    self.assertEqual(peaks.size, 0)
                    self.assertEqual(responses, [])

    def test_prominence_boundary_tolerates_subtraction_roundoff(self):
        phasic = np.array(
            [0.02, 0.01, 0, 0.005, 0.01, 0.015, 0.02, 0.018, 0.017, 0.016, 0.015]
        )
        for offset in (-1, -0.04, 0, 0.04, 1):
            with self.subTest(offset=offset):
                peaks, responses = self.processor.detect_scr_peaks(phasic + offset)
                assert_array_equal(peaks, [6])
                self.assertAlmostEqual(responses[0]["amplitude"], 0.02)
                self.assertIsNone(responses[0]["recovery_time"])

    def test_fast_rises_remain_rejected_after_a_baseline_shift(self):
        phasic = np.array([0.02, 0.01, 0, 0.02, 0.01, 0])
        for offset in (-1, 0, 1):
            with self.subTest(offset=offset):
                peaks, responses = self.processor.detect_scr_peaks(phasic + offset)
                self.assertEqual(peaks.size, 0)
                self.assertEqual(responses, [])

    def test_missing_recovery_remains_missing_after_a_baseline_shift(self):
        phasic = np.array([0.02, 0.01, 0, 0.01, 0.03, 0.02, 0.02, 0.02])
        for offset in (-1, 0, 1):
            with self.subTest(offset=offset):
                peaks, responses = self.processor.detect_scr_peaks(phasic + offset)
                assert_array_equal(peaks, [4])
                self.assertIsNone(responses[0]["recovery_time"])


class FeaturesTests(TemporaryWorkspaceTestCase):

    def test_missing_eda_preserves_feature_order_and_zero_event_counts(self):
        extractor = EDAFeatureExtractor()
        features = extractor.extract_all({})
        self.assertEqual(
            list(features),
            [
                "SCL_mean",
                "SCL_std",
                "SCL_slope",
                "SCL_min",
                "SCL_max",
                "SCR_count",
                "SCR_rate",
                "SCR_amplitude_mean",
                "SCR_amplitude_max",
                "SCR_rise_time_mean",
                "SCR_recovery_time_mean",
                "SCR_AUC",
                "EDA_mean",
                "EDA_range",
                "EDA_kurtosis",
            ],
        )
        self.assertTrue(
            all(
                (
                    value == 0.0 if key.startswith("SCR_") else np.isnan(value)
                    for key, value in features.items()
                )
            )
        )
        features["SCL_mean"] = 3.0
        self.assertTrue(np.isnan(extractor.extract_all({})["SCL_mean"]))

    def test_compute_failure_preserves_earlier_values_and_remaining_nans(self):
        extractor = HRVTimeDomainExtractor()

        def fail(_):
            raise RuntimeError("synthetic compute failure")

        self.patch_object(extractor, "compute_sdnn", fail)
        features = extractor.extract_all(np.full(60, 800.0))
        self.assertEqual(features["MeanNN"], 800.0)
        self.assertTrue(
            all((np.isnan(value) for key, value in features.items() if key != "MeanNN"))
        )

    def test_rmssd_constant_rr_is_zero(self):
        rr = np.full(60, 800.0)
        features = HRVTimeDomainExtractor().extract_all(rr)
        self.assertEqual(features["RMSSD"], 0.0)
        self.assertLess(abs(features["MeanNN"] - 800.0), 1e-06)

    def test_triangular_index_uses_fixed_rr_bins_and_handles_constant_intervals(self):
        extractor = HRVTimeDomainExtractor()
        self.assertEqual(extractor.compute_hrvti(np.full(60, 800.0)), 1.0)
        self.assertEqual(extractor.compute_hrvti(np.tile([800.0, 807.0], 30)), 2.0)

    def test_frequency_features_nan_below_min_rr(self):
        feats = HRVFrequencyDomainExtractor().extract_all(np.full(5, 800.0))
        self.assertTrue(all((np.isnan(v) for v in feats.values())))

    def test_nonlinear_features_finite_with_enough_rr(self):
        rng = np.random.RandomState(0)
        rr = 800 + 30 * rng.randn(120)
        feats = HRVNonlinearExtractor().extract_all(rr)
        for key in ("SD1", "SD2", "SampEn"):
            self.assertTrue(
                np.isfinite(feats[key]),
                f"{key} should be finite on a well-formed RR series",
            )

    def test_poincare_geometry_and_cvi_use_the_paired_ellipse_axes(self):
        extractor = HRVNonlinearExtractor()
        trending = extractor.compute_poincare(np.array([800.0, 820.0, 840.0]))
        self.assertEqual(trending["SD1"], 0.0)
        assert_close(trending["SD2"], 20.0)
        features = extractor.compute_poincare(np.array([800.0, 810.0, 790.0, 820.0]))
        assert_close(features["SD1"], np.sqrt(950 / 3))
        assert_close(features["SD2"], np.sqrt(50 / 3))
        assert_close(features["CVI"], np.log10(16 * np.sqrt(950 / 3) * np.sqrt(50 / 3)))

    def test_hrv_matches_known_sequence(self):
        rr = np.tile([800.0, 820.0], 30)
        f = HRVTimeDomainExtractor().extract_all(rr)
        self.assertLess(abs(f["MeanNN"] - 810.0), 1e-06)
        self.assertLess(abs(f["RMSSD"] - 20.0), 1e-06)
        self.assertLess(abs(f["SDNN"] - np.std(rr, ddof=1)), 1e-06)

    def test_too_few_rr_returns_nan(self):
        features = HRVTimeDomainExtractor().extract_all(np.array([800.0, 810.0]))
        self.assertTrue(np.isnan(features["SDNN"]))

    def test_rpeaks_recover_known_rate(self):
        fs = 700
        t = np.arange(0, 30, 1 / fs)
        ecg = np.sin(2 * np.pi * 1.0 * t) ** 21
        peaks = ECGProcessor(sampling_rate=fs).detect_r_peaks(ecg)
        rate = len(peaks) / 30
        self.assertTrue(0.8 < rate < 1.2)

    def test_tonic_and_temperature_slopes_preserve_missing_sample_times(self):
        time = np.arange(40) / 4.0
        tonic, temperature = (3 + 0.25 * time, 35 - 0.01 * time)
        tonic[[1, 4, 7, 15]] = np.nan
        temperature[[2, 6, 14, 29]] = np.inf
        assert_close(
            EDAFeatureExtractor(4).extract_tonic_features(tonic)["SCL_slope"], 0.25
        )
        assert_close(
            TemperatureFeatureExtractor(4).extract_all(temperature)["TEMP_slope"], -0.01
        )

    def test_rr_interpolation_uses_its_declared_sampling_rate(self):
        time, interpolated = HRVFrequencyDomainExtractor(
            interpolation_rate=4
        )._interpolate_rr(np.full(40, 810.0))
        assert_allclose(np.diff(time), 0.25)
        assert_allclose(interpolated, 810.0)

    def test_band_power_includes_unsampled_endpoints(self):
        extractor = HRVFrequencyDomainExtractor()
        freqs = np.arange(0, 2.01, 0.05)
        density = np.full(len(freqs), 2.0)
        assert_close(extractor.compute_vlf_power(freqs, density), 2 * (0.04 - 0.0033))
        assert_close(extractor.compute_lf_power(freqs, density), 2 * (0.15 - 0.04))
        assert_close(extractor.compute_hf_power(freqs, density), 2 * (0.4 - 0.15))

    def test_lomb_density_integrates_to_rr_variance_and_handles_constant_rr(self):
        extractor = HRVFrequencyDomainExtractor()
        rr = 800 + 30 * np.sin(np.arange(100) * 0.6)
        freqs, density = extractor.compute_psd(rr, method="lomb")
        assert_close(trapezoid(density, freqs), np.var(rr))
        features = extractor.extract_all(np.full(60, 800.0), method="lomb")
        self.assertEqual(features["Total_power"], 0.0)
        self.assertTrue(np.isnan(features["LF_peak_freq"]))
        with self.assertRaisesRegex(ValueError, "PSD method"):
            extractor.compute_psd(rr, method="typo")

    def test_lomb_default_grid_and_density_keep_the_existing_computation(self):
        from scipy.signal import lombscargle

        rr = 800 + 30 * np.sin(np.arange(100) * 0.6)
        time = np.r_[0.0, np.cumsum(rr[:-1]) / 1000.0]
        centered = rr - rr.mean()
        expected_freqs = np.linspace(0.001, 0.5, 512)
        expected = np.maximum(
            lombscargle(time, centered, 2 * np.pi * expected_freqs, normalize=False),
            0.0,
        )
        expected *= np.var(centered) / trapezoid(expected, expected_freqs)
        freqs, density = HRVFrequencyDomainExtractor().compute_psd(rr, method="lomb")
        assert_array_equal(freqs, expected_freqs)
        assert_array_equal(density, expected)

    def test_lomb_grid_covers_custom_bands_above_half_a_hertz(self):
        rr = 500 + 20 * np.sin(2 * np.pi * 0.8 * np.arange(200) * 0.5)
        extractor = HRVFrequencyDomainExtractor(hf_band=(0.7, 0.9))
        freqs, density = extractor.compute_psd(rr, method="lomb")
        self.assertEqual(freqs[-1], 0.9)
        self.assertGreater(extractor.compute_hf_power(freqs, density), 100.0)
        self.assertAlmostEqual(freqs[np.argmax(density)], 0.8, delta=0.005)
        assert_allclose(trapezoid(density, freqs), np.var(rr))

    def test_portable_integer_acceleration_preserves_magnitude(self):
        duration = 60
        recording = {
            "label": np.ones(duration * 700, dtype=int),
            "signal": {
                "chest": {"ECG": np.zeros(duration * 700)},
                "wrist": {
                    "BVP": np.zeros(duration * 64),
                    "EDA": np.ones(duration * 4),
                    "TEMP": np.full(duration * 4, 35.0),
                    "ACC": np.full((duration * 32, 3), 200, dtype=np.int16),
                },
            },
        }
        subject = self.tmp_path / "S2"
        subject.mkdir()
        (subject / "S2.pkl").write_bytes(pickle.dumps(recording))
        with (
            patch("neurokit2.ppg_clean", side_effect=lambda values, **kwargs: values),
            patch(
                "neurokit2.ppg_findpeaks",
                return_value={"PPG_Peaks": np.arange(0, duration * 64, 64)},
            ),
        ):
            portable = wesad_portable(data_path=self.tmp_path)

        expected = np.sqrt(3 * 200.0**2)
        self.assertAlmostEqual(portable.iloc[0]["ACC_mean"], expected)
        self.assertAlmostEqual(portable.iloc[0]["ACC_energy"], expected**2)
        acceleration = recording["signal"]["wrist"]["ACC"]
        self.assertEqual(acceleration.dtype, np.int16)
        assert_array_equal(acceleration, 200)

    def test_sample_entropy_is_zero_when_every_matching_pattern_extends(self):
        for rr in (np.full(20, 800.0), np.tile([800.0, 820.0], 10)):
            with self.subTest(rr=rr.tolist()):
                self.assertEqual(
                    HRVNonlinearExtractor().compute_sample_entropy(rr), 0.0
                )

    def test_sample_entropy_reports_zero_continuation_probability(self):
        rr = np.array([800, 800, 900, 800, 800, 1000])
        self.assertTrue(
            np.isinf(HRVNonlinearExtractor().compute_sample_entropy(rr, m=2, r=0.01))
        )

    def test_recurrence_determinism_uses_the_requested_embedding(self):
        extractor = HRVNonlinearExtractor()
        rr = np.tile([800.0, 820.0], 3)
        assert_close(extractor.compute_rqa_determinism(rr, embedding_dim=2), 6 / 8)
        assert_close(
            extractor.compute_rqa_determinism(rr, embedding_dim=2, time_delay=2), 1.0
        )
        self.assertTrue(
            np.isnan(
                extractor.compute_rqa_determinism(rr, embedding_dim=4, time_delay=2)
            )
        )
        self.assertTrue(np.isnan(extractor.compute_dfa(np.full(100, 800.0))).all())

    def test_flat_signals_have_no_detected_respiration_or_motion_frequency(self):
        respiration = RespirationFeatureExtractor(10).extract_all(np.ones(200))
        self.assertTrue(np.isnan(respiration["RESP_rate"]))
        self.assertEqual(respiration["RESP_amplitude"], 0.0)
        self.assertEqual(respiration["RESP_apnea_index"], 100.0)
        movement = AccelerometerFeatureExtractor(32).extract_from_magnitude(
            np.ones(320)
        )
        self.assertTrue(np.isnan(movement["ACC_peak_freq"]))
        self.assertEqual(movement["ACC_zero_crossings"], 0.0)

    def test_missing_accelerometer_samples_do_not_compress_event_time(self):
        magnitude = np.tile([0.0, 2.0], 50)
        magnitude[20:30] = np.nan
        features = AccelerometerFeatureExtractor(10).extract_from_magnitude(magnitude)
        assert_close(features["ACC_zero_crossings"], 88 / 10)
        self.assertTrue(np.isnan(features["ACC_peak_freq"]))

    def test_ectopic_correction_excludes_nonfinite_and_nonpositive_intervals(self):
        processor = ECGProcessor()
        _, valid = processor.remove_ectopic_beats(
            np.array([800.0, np.nan, -10.0, 810.0, 800.0])
        )
        assert_array_equal(valid, [True, False, False, True, True])
        corrected = processor.interpolate_artifacts(
            [800.0, np.nan, -10.0, 810.0, 800.0], valid
        )
        self.assertTrue(np.isfinite(corrected).all())
        self.assertTrue(
            np.isnan(
                processor.interpolate_artifacts([4000, 5000], [False, False])
            ).all()
        )

    def test_rr_extraction_rejects_invalid_peak_order_or_values(self):
        for peaks in ([700, 350], [350, 350], [0, np.nan], [-1, 350]):
            with self.subTest(peaks=peaks):
                with self.assertRaisesRegex(ValueError, "strictly increasing"):
                    ECGProcessor().extract_rr_intervals(np.asarray(peaks))

    def test_ecg_empty_or_constant_input_has_no_peaks(self):
        processor = ECGProcessor()
        self.assertEqual(processor.detect_r_peaks(np.array([])).size, 0)
        self.assertEqual(processor.detect_r_peaks(np.ones(100)).size, 0)
        with self.assertRaisesRegex(ValueError, "Nyquist"):
            processor.bandpass_filter(np.ones(100), low=10, high=5)

    def test_feature_extractors_reject_invalid_sampling_rates(self):
        extractors = (
            EDAFeatureExtractor,
            TemperatureFeatureExtractor,
            RespirationFeatureExtractor,
            AccelerometerFeatureExtractor,
        )
        for extractor, rate in product(extractors, (0, -1, np.inf, np.nan, True)):
            with self.subTest(extractor=extractor.__name__, sampling_rate=rate):
                with self.assertRaisesRegex(ValueError, "finite and positive"):
                    extractor(sampling_rate=rate)

    def test_window_configuration_is_rejected_before_loading_data(self):
        configurations = (
            {"window_sec": 0},
            {"overlap": 1},
            {"overlap": -0.1},
            {"purity": 0},
            {"purity": np.nan},
            {"window_sec": 0.0001},
        )
        for parameters, builder in product(
            configurations, (WindowedDataset, WristDataset)
        ):
            with self.subTest(builder=builder.__name__, parameters=parameters):
                with self.assertRaises(ValueError):
                    builder(**parameters)

    def test_feature_schema_does_not_compute_dummy_signals(self):
        pipeline = FeatureExtractionPipeline()
        self.patch_object(
            pipeline,
            "extract_window_features",
            lambda _: self.fail("Schema discovery must not compute signals"),
        )
        names = pipeline.get_feature_names()
        self.assertEqual(names, list(pipeline.get_feature_descriptions()))
        names.clear()
        self.assertEqual(pipeline.get_feature_count(), 60)
        self.assertEqual(
            list(pipeline.extract_all_features([])),
            ["subject_id", "window_id", "label", *pipeline.get_feature_names()],
        )

    def test_feature_names_follow_the_active_groups(self):
        pipeline = FeatureExtractionPipeline()
        original = pipeline.get_feature_names()
        pipeline.feature_config["eda"] = False
        names = pipeline.get_feature_names()
        expected = list(pipeline.get_feature_descriptions())
        self.assertLess(len(names), len(original))
        self.assertEqual(names, expected)
        self.assertEqual(names, list(pipeline.extract_window_features({})))
        names.clear()
        self.assertEqual(pipeline.get_feature_names(), expected)

    def test_feature_groups_preserve_schema_order_and_disabled_groups(self):
        for enabled in (
            {},
            {"hrv_time": False, "eda": False},
            dict.fromkeys(FeatureExtractionPipeline.DEFAULT_CONFIG, False),
        ):
            with self.subTest(configuration=enabled):
                pipeline = FeatureExtractionPipeline(enabled)
                grouped = pipeline.get_feature_groups()
                self.assertEqual(
                    [name for names in grouped.values() for name in names],
                    pipeline.get_feature_names(),
                )
                for group, (_, label) in pipeline._GROUPS.items():
                    if not pipeline.feature_config[group]:
                        self.assertEqual(grouped[label], [])

    def test_synthetic_subjects_skip_unavailable_wesad_ids(self):
        self.patch_object(synthetic, "_subject", lambda seed, block_sec: {"seed": seed})
        data_path = synthetic.write_dataset(self.tmp_path, n_subjects=15, block_sec=1)
        self.assertEqual(WESADLoader(data_path).subjects, VALID_SUBJECTS)
        self.assertFalse((data_path / "S12").exists())
        with self.assertRaisesRegex(ValueError, "real-data caches"):
            synthetic.features(cache=True)

    def test_wesad_loader_rejects_misaligned_and_malformed_channels(self):
        subject = self.tmp_path / "S2"
        subject.mkdir()
        record = {
            "signal": {
                "chest": {"ECG": np.zeros((700, 1))},
                "wrist": {"EDA": np.zeros((4, 1))},
            },
            "label": np.ones(700),
        }
        path = subject / "S2.pkl"
        path.write_bytes(pickle.dumps(record))
        loader = WESADLoader(self.tmp_path)
        self.assertEqual(len(loader.load_subject("S2")["label"]), 700)
        record["signal"]["chest"]["ECG"] = np.zeros((650, 1))
        path.write_bytes(pickle.dumps(record))
        with self.assertRaisesRegex(ValueError, "duration"):
            loader.load_subject("S2")
        record["signal"]["chest"]["ECG"] = np.zeros((700, 3))
        path.write_bytes(pickle.dumps(record))
        with self.assertRaisesRegex(ValueError, "signal shape"):
            loader.load_subject("S2")

    def test_cached_chest_features_require_current_protocol_and_aligned_rows(self):
        self.patch_object(dataset, "PROCESSED_DATA_DIR", self.tmp_path)
        source_path = self.enterContext(TemporaryDirectory())
        loader = source_loader(source_path)
        self.patch_object(dataset, "WESAD_DIR", loader.data_path)
        columns = dataset.FeatureExtractionPipeline().get_feature_names()
        frame = pd.DataFrame(
            {
                "subject_id": ["S2"],
                "window_id": [0],
                "label": [1],
                **{name: [0.0] for name in columns},
            }
        )
        frame.to_parquet(self.tmp_path / "features.parquet", index=False)
        np.savez(self.tmp_path / "raw_windows.npz", x=np.zeros((1, 5, 1024)))
        self.assertIs(dataset.load_cached(), None)
        frame.attrs.update(
            {
                "feature_schema_version": FEATURE_SCHEMA_VERSION,
                "rr_correction_version": dataset.RR_CORRECTION_VERSION,
                "scr_detection_version": dataset.SCR_DETECTION_VERSION,
                "extraction_fingerprint": cache_fingerprint(
                    "chest", loader.data_path, VALID_SUBJECTS
                ),
                "dataset_parameters": {
                    "fs": 700.0,
                    "window_samples": 42000,
                    "step": 21000,
                    "purity": 0.9,
                    "cnn_length": 1024,
                    "subjects": sorted(VALID_SUBJECTS),
                },
            }
        )
        builder = dataset.WindowedDataset.__new__(dataset.WindowedDataset)
        builder.loader = loader
        builder._save(frame, np.zeros((1, 5, 1024)))
        self.assertEqual(dataset.load_cached()[1].shape, (1, 5, 1024))
        frame = pd.read_parquet(self.tmp_path / "features.parquet")
        frame["window_id"] = 700
        frame.to_parquet(self.tmp_path / "features.parquet", index=False)
        with self.assertRaisesRegex(ValueError, "not aligned"):
            dataset.load_cached()
        frame.attrs["dataset_parameters"]["purity"] = 0.7
        frame.to_parquet(self.tmp_path / "features.parquet", index=False)
        self.assertIs(dataset.load_cached(), None)


class ButterworthFilterTests(PatchedTestCase):
    @classmethod
    def setUpClass(cls):
        cls.processor = SignalProcessor(fs=FS.CHEST)
        cls.time = np.arange(int(400 * FS.CHEST)) / FS.CHEST
        cls.center = slice(int(100 * FS.CHEST), int(300 * FS.CHEST))

    def amplitude(self, values, frequency):
        return component_amplitude(
            values[self.center], self.time[self.center], frequency
        )

    def test_low_and_high_pass_keep_the_requested_order_at_small_cutoffs(self):
        for cutoff in (0.1, 3.5):
            frequencies = (cutoff / 2, 2 * cutoff)
            data = sum(np.sin(2 * np.pi * f * self.time) for f in frequencies)
            for order in (3, 4, 6):
                for kind in ("low", "high"):
                    with self.subTest(cutoff=cutoff, order=order, kind=kind):
                        filtered = self.processor.butterworth_filter(
                            data, cutoff, order=order, btype=kind
                        )
                        self.assertTrue(np.isfinite(filtered).all())
                        for frequency in frequencies:
                            ratio = np.tan(np.pi * frequency / FS.CHEST) / np.tan(
                                np.pi * cutoff / FS.CHEST
                            )
                            if kind == "high":
                                ratio = 1 / ratio
                            expected = 1 / (1 + ratio ** (2 * order))
                            self.assertAlmostEqual(
                                self.amplitude(filtered, frequency),
                                expected,
                                delta=2e-5,
                            )

    def test_low_cutoff_band_filters_keep_the_requested_design_order(self):
        frequencies = (0.2, 0.8)
        data = sum(np.sin(2 * np.pi * f * self.time) for f in frequencies)
        low, high = np.tan(np.pi * np.array([0.1, 0.5]) / FS.CHEST)
        for order in (3, 4, 6):
            for kind in ("band", "bandstop"):
                with self.subTest(order=order, kind=kind):
                    filtered = self.processor.butterworth_filter(
                        data, (0.1, 0.5), order=order, btype=kind
                    )
                    self.assertEqual(filtered.shape, data.shape)
                    self.assertTrue(np.isfinite(filtered).all())
                    for frequency in frequencies:
                        warped = np.tan(np.pi * frequency / FS.CHEST)
                        ratio = abs((warped**2 - low * high) / ((high - low) * warped))
                        if kind == "bandstop":
                            ratio = 1 / ratio
                        expected = 1 / (1 + ratio ** (2 * order))
                        self.assertAlmostEqual(
                            self.amplitude(filtered, frequency), expected, delta=2e-5
                        )

    def test_respiration_uses_its_configured_fourth_order_design(self):
        data = np.sin(2 * np.pi * 0.8 * self.time)
        filtered = self.processor.process_respiration(data)
        self.assertEqual(FILTER_PARAMS.RESP_FILTER_ORDER, 4)
        self.assertLess(self.amplitude(filtered, 0.8), 0.01)
        lower_order = self.processor.butterworth_filter(
            data, (0.1, 0.5), order=2, btype="band"
        )
        self.assertGreater(self.amplitude(lower_order, 0.8), 0.07)

    def test_real_window_lengths_stay_finite_without_mutating_input(self):
        for duration in (60, 300):
            time = np.arange(int(duration * FS.CHEST)) / FS.CHEST
            respiration = np.sin(2 * np.pi * 0.25 * time)
            temperature = np.full(len(time), 35.0)
            original = respiration.copy()
            filtered = self.processor.process_respiration(respiration)
            self.assertEqual(filtered.shape, respiration.shape)
            self.assertTrue(np.isfinite(filtered).all())
            assert_array_equal(respiration, original)
            assert_allclose(
                self.processor.process_temperature(temperature), 35.0, atol=1e-6
            )

    def test_short_signals_raise_instead_of_lowering_the_requested_order(self):
        for length in (10, 20):
            with self.subTest(length=length):
                with self.assertRaisesRegex(ValueError, "length.*padlen"):
                    self.processor.butterworth_filter(np.ones(length), 0.1, order=6)

    def test_empty_and_invalid_inputs_keep_explicit_validation(self):
        assert_array_equal(self.processor.butterworth_filter([], 0.1, order=4), [])
        for order in (0, -1, 2.5, True):
            with self.subTest(order=order), self.assertRaisesRegex(ValueError, "order"):
                self.processor.butterworth_filter(np.ones(100), 0.1, order=order)
        for cutoff in (0, -0.1, FS.CHEST / 2, np.inf, np.nan, (0.5, 0.1)):
            with self.subTest(cutoff=cutoff), self.assertRaisesRegex(
                ValueError, "Cutoff"
            ):
                self.processor.butterworth_filter(np.ones(100), cutoff)
        with self.assertRaisesRegex(ValueError, "finite samples"):
            self.processor.butterworth_filter([0, np.nan, 1], 0.1)


class HRVInterpolationTests(PatchedTestCase):
    def setUp(self):
        self.extractor = HRVFrequencyDomainExtractor()

    def test_corrected_ecg_intervals_remain_physiological_after_resampling(self):
        rr = np.r_[np.full(30, 200.0), np.full(30, 2500.0)]
        original = rr.copy()
        ecg = ECGProcessor()
        _, valid = ecg.remove_ectopic_beats(rr)
        assert_array_equal(valid, True)
        corrected = ecg.interpolate_artifacts(rr, valid)

        time, resampled = self.extractor._interpolate_rr(corrected)

        self.assertTrue(np.isfinite(resampled).all())
        self.assertGreaterEqual(resampled.min(), 200)
        self.assertLessEqual(resampled.max(), 2500)
        assert_allclose(np.diff(time), 0.25)
        self.assertLess(time[-1], np.sum(rr[:-1]) / 1000)
        assert_array_equal(rr, original)
        freqs, psd = self.extractor.compute_psd(corrected)
        self.assertGreater(len(freqs), 0)
        self.assertTrue(np.isfinite(psd).all())
        self.assertTrue(np.all(psd >= 0))

    def test_cubic_resampling_stays_between_each_pair_of_anchors(self):
        rng = np.random.default_rng(42)
        rr = rng.choice([200.0, 300.0, 800.0, 1500.0, 2500.0], 60)
        anchor_time = np.r_[0.0, np.cumsum(rr[:-1]) / 1000]
        time, resampled = self.extractor._interpolate_rr(rr, method="cubic")
        for index, (left, right) in enumerate(zip(anchor_time[:-1], anchor_time[1:])):
            with self.subTest(interval=index):
                values = resampled[(time >= left) & (time <= right)]
                lower, upper = sorted(rr[index : index + 2])
                self.assertTrue(np.all(values >= lower - 1e-10))
                self.assertTrue(np.all(values <= upper + 1e-10))

    def test_linear_resampling_preserves_its_existing_computation(self):
        rr = np.tile([600.0, 800.0, 1000.0], 20)
        anchor_time = np.r_[0.0, np.cumsum(rr[:-1]) / 1000]
        time, resampled = self.extractor._interpolate_rr(rr, method="linear")
        assert_array_equal(resampled, np.interp(time, anchor_time, rr))

    def test_two_anchors_use_their_bounded_linear_segment(self):
        for method in ("linear", "cubic"):
            with self.subTest(method=method):
                time, resampled = self.extractor._interpolate_rr(
                    [2500.0, 200.0], method=method
                )
                assert_allclose(resampled, 2500 - 2300 * time / 2.5)

    def test_empty_or_short_recordings_have_no_resampled_signal(self):
        for rr in ([], [800.0], [800.0, 800.0]):
            with self.subTest(rr=rr):
                self.assertEqual(self.extractor._interpolate_rr(rr), (None, None))

    def test_nonphysiological_anchors_are_rejected(self):
        for invalid in (np.nan, np.inf, -np.inf, -1, 0, 199.9, 2500.1):
            with self.subTest(invalid=invalid):
                with self.assertRaisesRegex(ValueError, "finite intervals"):
                    self.extractor._interpolate_rr([800.0, invalid, 800.0])

    def test_unknown_interpolation_method_is_rejected_for_any_input(self):
        for rr in ([], [800.0], np.full(60, 800.0)):
            with self.subTest(length=len(rr)):
                with self.assertRaisesRegex(ValueError, "interpolation method"):
                    self.extractor._interpolate_rr(rr, method="quadratic")


class PortableTests(TemporaryWorkspaceTestCase):

    def test_portable_features_keys_and_finite(self):
        rng = np.random.RandomState(0)
        feats = portable_features(
            eda=rng.rand(240) + 5,
            temp=rng.rand(240) + 33,
            acc_mag=rng.rand(1920) + 1,
            hr=rng.rand(70) * 10 + 70,
            eda_fs=4,
            temp_fs=4,
        )
        self.assertEqual(len(feats), 18)
        self.assertTrue(any((k.startswith("EDA_") for k in feats)))
        self.assertTrue(any((k.startswith("HR_") for k in feats)))
        self.assertTrue(np.isfinite(feats["EDA_mean"]) and np.isfinite(feats["HR_std"]))

    def test_portable_features_handle_empty(self):
        feats = portable_features(
            eda=np.array([]),
            temp=np.array([1.0]),
            acc_mag=np.array([]),
            hr=np.array([]),
            eda_fs=4,
            temp_fs=4,
        )
        self.assertEqual(len(feats), 18)
        self.assertTrue(np.isnan(feats["EDA_mean"]))

    def test_same_columns_both_datasets(self):
        rng = np.random.RandomState(1)
        a = portable_features(
            rng.rand(240),
            rng.rand(240),
            rng.rand(1920),
            rng.rand(70),
            eda_fs=4,
            temp_fs=4,
        )
        b = portable_features(
            rng.rand(480),
            rng.rand(480),
            rng.rand(480),
            rng.rand(60),
            eda_fs=8,
            temp_fs=8,
        )
        self.assertEqual(set(a), set(b))

    def _case_slopes_match_physical_units_across_sampling_rates(self, missing):
        for fs in (4, 8):
            t = np.arange(60 * fs) / fs
            eda = 3.0 + 0.25 * t
            temp = 35.0 - 0.01 * t
            if missing:
                eda[[1, 7, 20, 90]] = np.nan
                temp[[4, 8, 35]] = np.inf
            features = portable_features(eda, temp, [], [], eda_fs=fs, temp_fs=fs)
            assert_close(features["EDA_slope"], 0.25)
            assert_close(features["TEMP_slope"], -0.01)

    def _case_invalid_sampling_rates_are_rejected(self, rate, channel):
        rates = {"eda_fs": 4, "temp_fs": 4, channel: rate}
        with self.assertRaisesRegex(ValueError, "finite and positive"):
            portable_features([1, 2], [35, 36], [], [], **rates)

    def test_sampling_rates_must_be_explicit(self):
        with self.assertRaisesRegex(TypeError, "eda_fs"):
            portable_features([1, 2], [35, 36], [], [])

    def _case_noneeg_windows_align_annotations_sensor_channels_and_hr(
        self, sensor_offset
    ):
        self.patch_object(non_eeg, "DATA_DIR", self.tmp_path)
        (self.tmp_path / "Subject1_AccTempEDA.hea").touch()
        sensor_time = np.arange(180 * 8 + sensor_offset) / 8
        sensor = np.column_stack(
            [
                np.zeros((len(sensor_time), 2)),
                np.ones(len(sensor_time)),
                35 - 0.01 * sensor_time,
                3 + 0.25 * sensor_time,
            ]
        )
        hr = np.column_stack([np.zeros(181), 70 + np.arange(181)])
        annotations = SimpleNamespace(
            sample=np.array([0, 480, 960]) + sensor_offset,
            aux_note=["Relax", "CognitiveStress", "PhysicalStress"],
        )

        def read(record):
            return (
                SimpleNamespace(fs=1, p_signal=hr)
                if record.endswith("SpO2HR")
                else SimpleNamespace(fs=8, p_signal=sensor)
            )

        self.patch_object(non_eeg, "read_record", read)
        self.patch_object(non_eeg, "read_annotations", lambda *args: annotations)
        frame = non_eeg.build(["Subject1"])
        self.assertEqual(frame["label"].tolist(), [0, 1])
        assert_allclose(frame["EDA_slope"], 0.25)
        assert_allclose(frame["TEMP_slope"], -0.01)
        first_hr = int(sensor_offset > 0)
        assert_allclose(frame["HR_mean"], [99.5 + first_hr, 159.5 + first_hr])
        assert_allclose(frame["HR_min"], [70 + first_hr, 130 + first_hr])
        assert_allclose(frame["HR_max"], [129 + first_hr, 189 + first_hr])

    def test_noneeg_rejects_annotations_beyond_available_samples(self):
        self.patch_object(non_eeg, "DATA_DIR", self.tmp_path)
        (self.tmp_path / "Subject1_AccTempEDA.hea").touch()
        annotations = SimpleNamespace(
            sample=np.array([0, 900]), aux_note=["Relax", "CognitiveStress"]
        )
        self.patch_object(non_eeg, "read_annotations", lambda *args: annotations)
        self.patch_object(
            non_eeg,
            "read_record",
            lambda record: SimpleNamespace(
                fs=1 if record.endswith("SpO2HR") else 8,
                p_signal=np.ones(
                    (
                        60 if record.endswith("SpO2HR") else 480,
                        2 if record.endswith("SpO2HR") else 5,
                    )
                ),
            ),
        )
        with self.assertRaisesRegex(ValueError, "outside the sensor record"):
            non_eeg.build(["Subject1"])


add_cases(
    PortableTests,
    "slopes_match_physical_units_across_sampling_rates",
    [(False,), (True,)],
)


add_cases(
    PortableTests,
    "invalid_sampling_rates_are_rejected",
    product(
        [0, -1, np.nan, np.inf, True, np.bool_(True), None, "invalid"],
        ["eda_fs", "temp_fs"],
    ),
)


add_cases(
    PortableTests,
    "noneeg_windows_align_annotations_sensor_channels_and_hr",
    [(0,), (1,), (7,)],
)


class SyntheticTests(PatchedTestCase):
    def test_short_blocks_preserve_finite_aligned_sensor_samples(self):
        for seconds in (1, 2, 3):
            with self.subTest(seconds=seconds):
                subject = synthetic._subject(seed=42, block_sec=seconds)
                assert_array_equal(
                    subject["label"],
                    np.repeat([1, 2, 3, 1, 2], seconds * synthetic.CHEST_FS),
                )
                sampling_rates = {
                    "chest": {
                        channel: synthetic.CHEST_FS
                        for channel in ("ECG", "EDA", "Temp", "Resp", "EMG", "ACC")
                    },
                    "wrist": synthetic.WRIST,
                }
                for sensor, channels in sampling_rates.items():
                    for channel, sampling_rate in channels.items():
                        samples = subject["signal"][sensor][channel]
                        expected_columns = 3 if channel == "ACC" else 1
                        self.assertEqual(
                            samples.shape,
                            (5 * seconds * sampling_rate, expected_columns),
                        )
                        self.assertTrue(np.isfinite(samples).all())

    def test_default_block_keeps_original_respiration(self):
        import neurokit2 as nk

        seconds = 120
        seed = 42
        expected = synthetic._fit_length(
            nk.rsp_simulate(
                duration=seconds,
                sampling_rate=synthetic.CHEST_FS,
                respiratory_rate=synthetic.RESP_RATE["baseline"],
                random_state=seed,
            ),
            seconds * synthetic.CHEST_FS,
        )
        with patch.object(nk, "rsp_simulate", wraps=nk.rsp_simulate) as simulate:
            chest = synthetic._chest_block(
                "baseline", seconds, np.random.RandomState(seed), seed
            )
        simulate.assert_called_once_with(
            duration=seconds,
            sampling_rate=synthetic.CHEST_FS,
            respiratory_rate=synthetic.RESP_RATE["baseline"],
            random_state=seed,
        )
        assert_array_equal(chest["Resp"].ravel(), expected)


def features():
    frame = pd.DataFrame({column: [1.0, 2.0] for column in PORTABLE_FEATURE_COLUMNS})
    frame["subject"] = ["S2", "S3"]
    frame["label"] = [0, 1]
    return frame


class CrossDatasetTests(TemporaryWorkspaceTestCase):

    def setUp(self):
        super().setUp()
        self.features = features()
        raw_path = Path(self.enterContext(TemporaryDirectory()))
        self.wesad_path, self.noneeg_path = raw_path / "wesad", raw_path / "noneeg"
        for subject in ("S2", "S3"):
            directory = self.wesad_path / subject
            directory.mkdir(parents=True)
            (directory / f"{subject}.pkl").write_bytes(subject.encode())
        self.noneeg_path.mkdir()
        for subject in range(1, 21):
            for record in ("AccTempEDA", "SpO2HR"):
                name = f"Subject{subject}_{record}"
                (self.noneeg_path / f"{name}.hea").write_text(
                    f"{name} 1 8 2\n{name}.dat 16 1/mV 16 0 1 3 0 signal\n"
                )
                (self.noneeg_path / f"{name}.dat").write_bytes(b"\x01\x00\x02\x00")
                if record == "AccTempEDA":
                    (self.noneeg_path / f"{name}.atr").write_bytes(b"\x00\x00")
        self.patch_object(cross_dataset, "WESAD_DIR", self.wesad_path)
        self.patch_object(cross_dataset, "VALID_SUBJECTS", ["S2", "S3"])
        self.patch_object(cross_dataset.non_eeg, "DATA_DIR", self.noneeg_path)

    def test_legacy_cache_is_ignored_and_current_cache_is_verified(self):
        directory = self.tmp_path / "processed"
        directory.mkdir()
        legacy = directory / "portable_wesad.parquet"
        legacy.write_bytes(b"legacy sample-based slopes")
        previous = directory / f"portable_wesad_v{PORTABLE_SCHEMA_VERSION - 1}.parquet"
        previous.write_bytes(b"previous HR boundary convention")
        self.patch_object(cross_dataset, "PROCESSED_DATA_DIR", directory)
        builder = Mock(return_value=self.features)
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
        directory = self.tmp_path / "nested" / "processed"
        self.patch_object(cross_dataset, "PROCESSED_DATA_DIR", directory)
        _, metadata = cross_dataset._load_or_build_cache(
            "noneeg", lambda: self.features
        )
        self.assertTrue(
            (directory / f"portable_noneeg_v{PORTABLE_SCHEMA_VERSION}.parquet").exists()
        )
        self.assertEqual(metadata["sampling_rates_hz"], {"eda": 8, "temp": 8})
        self.assertEqual(metadata["signal_units"], PORTABLE_SIGNAL_UNITS["noneeg"])

    def test_raw_manifests_cover_every_consumed_file(self):
        for dataset_name, count in (("wesad", 2), ("noneeg", 100)):
            with self.subTest(dataset=dataset_name):
                source = cross_dataset._raw_provenance(dataset_name)
                self.assertEqual(len(source["raw_file_sha256"]), count)
                self.assertEqual(
                    cross_dataset._raw_identity(source)[1], source["manifest_sha256"]
                )
        with (
            patch.object(cross_dataset, "VALID_SUBJECTS", []),
            self.assertRaisesRegex(ValueError, "empty"),
        ):
            cross_dataset._raw_provenance("wesad")

    def test_dataset_alias_reuses_the_verified_cache(self):
        for dataset_name, directory, module, attribute in (
            ("wesad", self.wesad_path, cross_dataset, "WESAD_DIR"),
            ("noneeg", self.noneeg_path, cross_dataset.non_eeg, "DATA_DIR"),
        ):
            with self.subTest(dataset=dataset_name), patch.object(
                cross_dataset, "PROCESSED_DATA_DIR", self.tmp_path / dataset_name
            ):
                expected, metadata = cross_dataset._load_or_build_cache(
                    dataset_name, lambda: self.features
                )
                alias = self.tmp_path / f"{dataset_name}-alias"
                alias.symlink_to(directory, target_is_directory=True)
                builder = Mock(return_value=self.features)
                with patch.object(module, attribute, alias):
                    source = cross_dataset._raw_provenance(dataset_name)
                    self.assertEqual(
                        cross_dataset._raw_identity(source),
                        cross_dataset._raw_identity(metadata["source_provenance"]),
                    )
                    actual, again = cross_dataset._load_or_build_cache(
                        dataset_name, builder
                    )
                builder.assert_not_called()
                pd.testing.assert_frame_equal(actual, expected)
                self.assertEqual(again, metadata)

    def test_same_size_raw_changes_with_preserved_mtime_require_rebuilding(self):
        for dataset_name, raw in (
            ("wesad", self.wesad_path / "S3/S3.pkl"),
            ("noneeg", self.noneeg_path / "Subject20_SpO2HR.dat"),
        ):
            with self.subTest(dataset=dataset_name), patch.object(
                cross_dataset, "PROCESSED_DATA_DIR", self.tmp_path / dataset_name
            ):
                _, before = cross_dataset._load_or_build_cache(
                    dataset_name, lambda: self.features
                )
                state = raw.stat()
                payload = raw.read_bytes()
                raw.write_bytes(bytes([payload[0] ^ 1]) + payload[1:])
                os.utime(raw, ns=(state.st_atime_ns, state.st_mtime_ns))
                self.assertEqual(raw.stat().st_size, state.st_size)
                self.assertEqual(raw.stat().st_mtime_ns, state.st_mtime_ns)
                builder = Mock(return_value=self.features)
                with self.assertRaisesRegex(ValueError, "raw inputs changed"):
                    cross_dataset._load_or_build_cache(dataset_name, builder)
                builder.assert_not_called()
                _, after = cross_dataset._load_or_build_cache(
                    dataset_name, builder, rebuild=True
                )
                builder.assert_called_once_with()
                self.assertNotEqual(
                    before["source_provenance"]["manifest_sha256"],
                    after["source_provenance"]["manifest_sha256"],
                )

    def test_header_referenced_data_changes_invalidate_portable_cache(self):
        header = self.noneeg_path / "Subject20_SpO2HR.hea"
        canonical = self.noneeg_path / "Subject20_SpO2HR.dat"
        alternate = self.noneeg_path / "alternate.bin"
        alternate.write_bytes(canonical.read_bytes())
        header.write_text(header.read_text().replace(canonical.name, alternate.name))
        with patch.object(cross_dataset, "PROCESSED_DATA_DIR", self.tmp_path):
            _, before = cross_dataset._load_or_build_cache(
                "noneeg", lambda: self.features
            )
            manifest = before["source_provenance"]["raw_file_sha256"]
            self.assertIn(alternate.name, manifest)
            self.assertNotIn(canonical.name, manifest)
            state = alternate.stat()
            alternate.write_bytes(b"\x02\x00\x01\x00")
            os.utime(alternate, ns=(state.st_atime_ns, state.st_mtime_ns))
            self.assertEqual(alternate.stat().st_size, state.st_size)
            self.assertEqual(alternate.stat().st_mtime_ns, state.st_mtime_ns)
            builder = Mock(return_value=self.features)
            with self.assertRaisesRegex(ValueError, "raw inputs changed"):
                cross_dataset._load_or_build_cache("noneeg", builder)
            builder.assert_not_called()

    def test_missing_or_unreadable_raw_files_prevent_cache_reuse(self):
        for dataset_name, raw in (
            ("wesad", self.wesad_path / "S3/S3.pkl"),
            ("noneeg", self.noneeg_path / "Subject20_SpO2HR.dat"),
        ):
            with self.subTest(dataset=dataset_name), patch.object(
                cross_dataset, "PROCESSED_DATA_DIR", self.tmp_path / dataset_name
            ):
                cross_dataset._load_or_build_cache(dataset_name, lambda: self.features)
                builder = Mock(return_value=self.features)
                payload = raw.read_bytes()
                raw.unlink()
                with self.assertRaisesRegex(FileNotFoundError, "Required raw input"):
                    cross_dataset._load_or_build_cache(dataset_name, builder)
                raw.write_bytes(payload)
                verified_input = cross_dataset.verified_input

                def unreadable(path):
                    if path.resolve() == raw.resolve():
                        raise PermissionError(f"Unreadable raw file: {path}")
                    return verified_input(path)

                with (
                    patch.object(cross_dataset, "verified_input", unreadable),
                    self.assertRaises(PermissionError),
                ):
                    cross_dataset._load_or_build_cache(dataset_name, builder)
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

        with (
            patch.object(hashlib, "file_digest", replace_after_hash),
            self.assertRaisesRegex(RuntimeError, "Input changed"),
        ):
            cross_dataset._raw_provenance("wesad")

    def _case_unverified_current_cache_is_rejected(self, fault):
        self.patch_object(cross_dataset, "PROCESSED_DATA_DIR", self.tmp_path)
        cross_dataset._load_or_build_cache("wesad", lambda: self.features)
        sidecar = self.tmp_path / f"portable_wesad_v{PORTABLE_SCHEMA_VERSION}.json"
        if fault == "missing_metadata":
            sidecar.unlink()
        else:
            metadata = json.loads(sidecar.read_text())
            if fault == "old_schema":
                metadata["schema_version"] = PORTABLE_SCHEMA_VERSION - 1
            else:
                metadata["cache_sha256"] = "0" * 64
            sidecar.write_text(json.dumps(metadata))
        builder = Mock(return_value=self.features)
        with self.assertRaisesRegex(ValueError, "metadata|schema|checksum"):
            cross_dataset._load_or_build_cache("wesad", builder)
        builder.assert_not_called()

    def _case_cache_rejects_changed_extractor_or_environment_and_can_rebuild(
        self, field
    ):
        self.patch_object(cross_dataset, "PROCESSED_DATA_DIR", self.tmp_path)
        _, old = cross_dataset._load_or_build_cache("wesad", lambda: self.features)
        schema = cross_dataset._cache_schema("wesad")
        schema[field] = {"changed": "new version"}
        self.patch_object(cross_dataset, "_cache_schema", lambda dataset: schema)
        builder = Mock(return_value=self.features)
        with self.assertRaisesRegex(ValueError, "schema mismatch"):
            cross_dataset._load_or_build_cache("wesad", builder)
        builder.assert_not_called()
        _, new = cross_dataset._load_or_build_cache("wesad", builder, rebuild=True)
        builder.assert_called_once_with()
        self.assertNotEqual(new[field], old[field])

    def test_cross_dataset_labels_allow_integer_valued_float_storage(self):
        self.features["label"] = self.features["label"].astype(float)
        _, y, _ = cross_dataset._xy(self.features, list(PORTABLE_FEATURE_COLUMNS))
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
            with (
                self.subTest(columns=columns),
                self.assertRaisesRegex(ValueError, "feature columns"),
            ):
                cross_dataset._xy(self.features, columns)

    def test_failed_rebuild_preserves_verified_cache(self):
        self.patch_object(cross_dataset, "PROCESSED_DATA_DIR", self.tmp_path)
        cross_dataset._load_or_build_cache("wesad", lambda: self.features)
        cache = self.tmp_path / f"portable_wesad_v{PORTABLE_SCHEMA_VERSION}.parquet"
        sidecar = cache.with_suffix(".json")
        before = (cache.read_bytes(), sidecar.read_bytes())
        changed = self.features.copy()
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
        self.patch_object(cross_dataset, "PROCESSED_DATA_DIR", self.tmp_path)
        cache = self.tmp_path / f"portable_wesad_v{PORTABLE_SCHEMA_VERSION}.parquet"
        sidecar = cache.with_suffix(".json")
        before = None
        if existing:
            cross_dataset._load_or_build_cache("wesad", lambda: self.features)
            before = (cache.read_bytes(), sidecar.read_bytes())
        changed = self.features.copy()
        changed[PORTABLE_FEATURE_COLUMNS[0]] += 1
        replace = Path.replace

        def fail_sidecar(source, destination):
            if source.name == sidecar.name:
                raise OSError("sidecar publication failed")
            return replace(source, destination)

        self.patch_object(Path, "replace", fail_sidecar)
        with self.assertRaisesRegex(OSError, "sidecar publication failed"):
            cross_dataset._load_or_build_cache("wesad", lambda: changed, rebuild=True)
        if existing:
            self.assertEqual((cache.read_bytes(), sidecar.read_bytes()), before)
            cached, _ = cross_dataset._load_or_build_cache(
                "wesad", lambda: self.fail("Rebuilt instead of recovering")
            )
            pd.testing.assert_frame_equal(cached, self.features)
        else:
            self.assertTrue(not cache.exists() and (not sidecar.exists()))
        self.assertEqual(
            set(self.tmp_path.iterdir()), {cache, sidecar} if existing else set()
        )


add_cases(
    CrossDatasetTests,
    "unverified_current_cache_is_rejected",
    [("missing_metadata",), ("old_schema",), ("bad_checksum",)],
)


add_cases(
    CrossDatasetTests,
    "cache_rejects_changed_extractor_or_environment_and_can_rebuild",
    [("extractor_sha256",), ("extraction_packages",)],
)


add_cases(
    CrossDatasetTests,
    "failed_cache_sidecar_publication_restores_previous_pair",
    [(False,), (True,)],
)
