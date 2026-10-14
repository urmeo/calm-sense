import hashlib
import os
import pickle
import shutil
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np

from src import cache, config
from src.data import loader
from src.preprocessing import eda_processor


class CacheFingerprintTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
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
        with self.assertRaises(FileNotFoundError):
            self.fingerprint()

    def test_runtime_filter_changes_require_restart(self):
        before = self.fingerprint()
        settings = replace(config.FILTER_PARAMS, ECG_BANDPASS_HIGH=35.0)
        with patch.object(config, "FILTER_PARAMS", settings):
            with self.assertRaisesRegex(
                RuntimeError, "settings changed; restart Python"
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
                with self.assertRaisesRegex(RuntimeError, "restart Python"):
                    self.fingerprint()
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
        with patch.object(
            config, "FILTER_PARAMS", replace(config.FILTER_PARAMS, EDA_LOWPASS=4.0)
        ):
            with self.assertRaisesRegex(RuntimeError, "settings changed"):
                cache.cache_fingerprint("chest", self.root, ["S2"], context=context)
        loaded = cache.raw_input_sha256(self.root, ["S2"])
        self.path.write_bytes(b"modified recording")
        with self.assertRaisesRegex(RuntimeError, "recordings changed"):
            cache.cache_fingerprint("chest", self.root, ["S2"], loaded_inputs=loaded)

    def test_invalid_subjects_and_device_are_rejected(self):
        for subjects in ([], ["S2", "S2"], ["../S2"], ["S12"]):
            with self.subTest(subjects=subjects), self.assertRaises(ValueError):
                cache.raw_input_sha256(self.root, subjects)
        with self.assertRaises(ValueError):
            cache.extraction_context("unknown")

    def test_changes_during_hashing_and_reading_are_rejected(self):
        digest_file = hashlib.file_digest

        def changed_while_hashing(stream, algorithm):
            digest = digest_file(stream, algorithm)
            self.path.write_bytes(b"modified recording")
            return digest

        with patch.object(hashlib, "file_digest", side_effect=changed_while_hashing):
            with self.assertRaisesRegex(RuntimeError, "while hashing"):
                cache.raw_input_sha256(self.root, ["S2"])
        with self.assertRaisesRegex(RuntimeError, "while reading"):
            with cache.verified_input(self.path):
                replacement = self.path.with_suffix(".replacement")
                replacement.write_bytes(self.path.read_bytes())
                replacement.replace(self.path)


class LoaderFingerprintTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
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

        with patch.object(pickle, "load", side_effect=changed_during_load):
            with self.assertRaisesRegex(RuntimeError, "while reading"):
                self.loader.load_subject("S2")
        self.assertEqual(self.loader.raw_input_sha256, {})

    def test_invalid_recording_does_not_receive_a_digest(self):
        self.path.write_bytes(pickle.dumps({"signal": {}}))
        with self.assertRaisesRegex(ValueError, "Invalid WESAD structure"):
            self.loader.load_subject("S2")
        self.assertEqual(self.loader.raw_input_sha256, {})


if __name__ == "__main__":
    unittest.main()
