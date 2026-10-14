import unittest
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd

from src import dataset, dataset_wrist
from src.config import VALID_SUBJECTS
from src.features.feature_pipeline import FEATURE_SCHEMA_VERSION
from src.utils import feature_frame_sha256, sha256_file


class DatasetCacheTests(unittest.TestCase):
    def setUp(self):
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.directory = Path(directory.name)
        patcher = patch.object(dataset, "PROCESSED_DATA_DIR", self.directory)
        patcher.start()
        self.addCleanup(patcher.stop)
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

    def snapshot(self):
        return {
            path.name: path.read_bytes()
            for path in (self.feature_path, self.raw_path)
            if path.exists()
        }

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
                    with target, self.assertRaisesRegex(
                        OSError, "serialization failed"
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

                with patch.object(Path, "replace", fail_raw_publication):
                    with self.assertRaisesRegex(OSError, "raw publication failed"):
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
            replacement = self.directory / "replacement.npz"
            replacement.write_bytes(b"replacement after hash")
            replacement.replace(self.raw_path)
            return result

        with patch.object(dataset.hashlib, "file_digest", replace_after_hash):
            np.testing.assert_array_equal(dataset.load_cached()[1], self.raw)
        self.assertEqual(self.raw_path.read_bytes(), b"replacement after hash")

    def wrist_builder(self):
        builder = dataset_wrist.WristDataset.__new__(dataset_wrist.WristDataset)
        builder.loader = SimpleNamespace(subjects=VALID_SUBJECTS)
        builder.features = SimpleNamespace(
            extract_all_features=lambda windows, **kwargs: self.frame.assign(
                subject_id=windows[0]["subject_id"]
            )
        )
        builder._process_subject = lambda subject: ([{"subject_id": subject}], [])
        builder.window_sec, builder.overlap, builder.purity = 60.0, 0.5, 0.9
        return builder

    def test_wrist_serialization_failure_preserves_previous_cache(self):
        cache = self.directory / "features_wrist.parquet"
        with patch.object(dataset_wrist, "PROCESSED_DATA_DIR", self.directory):
            builder = self.wrist_builder()
            for existing in (False, True):
                with self.subTest(existing=existing):
                    cache.unlink(missing_ok=True)
                    if existing:
                        expected = builder.build()
                    before = cache.read_bytes() if existing else None

                    def fail_serialization(frame, path, **kwargs):
                        Path(path).write_bytes(b"incomplete parquet")
                        raise OSError("wrist serialization failed")

                    with patch.object(
                        pd.DataFrame,
                        "to_parquet",
                        autospec=True,
                        side_effect=fail_serialization,
                    ):
                        with self.assertRaisesRegex(
                            OSError, "wrist serialization failed"
                        ):
                            builder.build()
                    self.assertEqual(
                        cache.read_bytes() if cache.exists() else None, before
                    )
                    if existing:
                        pd.testing.assert_frame_equal(
                            dataset_wrist.load_wrist(), expected
                        )
                    else:
                        self.assertIsNone(dataset_wrist.load_wrist())
                    self.assertEqual(
                        set(self.directory.iterdir()), {cache} if existing else set()
                    )

    def test_wrist_publication_failure_preserves_previous_cache(self):
        cache = self.directory / "features_wrist.parquet"
        with patch.object(dataset_wrist, "PROCESSED_DATA_DIR", self.directory):
            builder = self.wrist_builder()
            for existing in (False, True):
                with self.subTest(existing=existing):
                    cache.unlink(missing_ok=True)
                    if existing:
                        expected = builder.build()
                    before = cache.read_bytes() if existing else None
                    with patch.object(
                        Path, "replace", side_effect=OSError("wrist publication failed")
                    ):
                        with self.assertRaisesRegex(
                            OSError, "wrist publication failed"
                        ):
                            builder.build()
                    self.assertEqual(
                        cache.read_bytes() if cache.exists() else None, before
                    )
                    if existing:
                        pd.testing.assert_frame_equal(
                            dataset_wrist.load_wrist(), expected
                        )
                    else:
                        self.assertIsNone(dataset_wrist.load_wrist())
                    self.assertEqual(
                        set(self.directory.iterdir()), {cache} if existing else set()
                    )
