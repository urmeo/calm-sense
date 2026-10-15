import unittest
from contextlib import ExitStack
from unittest.mock import patch
from tempfile import TemporaryDirectory
from functools import partialmethod
import numpy as np
import hashlib
import math
from pathlib import Path
import joblib
from src import utils
from src.utils import (
    atomic_write_text,
    load_verified_joblib,
    paired_effect_size,
    provenance,
    save_verified_joblib,
    sha256_file,
    write_json,
)


class UtilsTests(unittest.TestCase):

    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.tmp_path = Path(self.stack.enter_context(TemporaryDirectory()))

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
        np.testing.assert_allclose(es["cohens_d"], d, rtol=1e-06, atol=1e-12)
        np.testing.assert_allclose(
            es["hedges_g"], d * math.sqrt(math.pi / 6), rtol=1e-06, atol=1e-12
        )

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
        np.testing.assert_allclose(
            scaled["cohens_d"], expected["cohens_d"], rtol=1e-06, atol=1e-12
        )
        np.testing.assert_allclose(
            scaled["hedges_g"], expected["hedges_g"], rtol=1e-06, atol=1e-12
        )

    def test_file_sha256_streams_without_read_bytes(self):
        tmp_path = self.tmp_path
        path = tmp_path / "recording.bin"
        contents = b"recording" * 100000
        path.write_bytes(contents)
        self.stack.enter_context(
            patch.object(
                Path,
                "read_bytes",
                lambda path: self.fail("Hashing buffered the whole file"),
            )
        )
        self.assertEqual(sha256_file(path), hashlib.sha256(contents).hexdigest())

    def test_saved_model_and_checksum_stay_in_sync(self):
        tmp_path = self.tmp_path
        path = tmp_path / "models" / "stress_classifier.joblib"
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
        tmp_path = self.tmp_path
        path = tmp_path / "stress_classifier.joblib"
        save_verified_joblib({"features": ["ECG"]}, path)
        path.write_bytes(b"tampered model")

        def refuse_unpickling(*args, **kwargs):
            self.fail("Tampered artifact reached joblib.load")

        self.stack.enter_context(patch.object(joblib, "load", refuse_unpickling))
        with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
            load_verified_joblib(path)

    def test_model_load_uses_verified_snapshot_if_path_is_replaced(self):
        tmp_path = self.tmp_path
        path = tmp_path / "model.joblib"
        trusted = {"trusted": True}
        save_verified_joblib(trusted, path)
        original_load = joblib.load

        def replace_then_load(snapshot):
            path.write_bytes(b"untrusted replacement")
            return original_load(snapshot)

        self.stack.enter_context(patch.object(joblib, "load", replace_then_load))
        self.assertEqual(load_verified_joblib(path), trusted)

    def test_failed_model_serialization_keeps_existing_verified_model(self):
        tmp_path = self.tmp_path
        path = tmp_path / "model.joblib"
        save_verified_joblib({"version": 1}, path)
        original_bytes = path.read_bytes()
        checksum = path.with_name(path.name + ".sha256")
        original_checksum = checksum.read_bytes()

        def failed_dump(bundle, destination):
            destination.write_bytes(b"partial serialization")
            raise OSError("serialization failed")

        self.stack.enter_context(patch.object(joblib, "dump", failed_dump))
        with self.assertRaisesRegex(OSError, "serialization failed"):
            save_verified_joblib({"version": 2}, path)
        self.assertEqual(path.read_bytes(), original_bytes)
        self.assertEqual(checksum.read_bytes(), original_checksum)
        self.assertEqual(load_verified_joblib(path), {"version": 1})

    def _case_failed_model_sidecar_publication_restores_previous_pair(self, existing):
        tmp_path = self.tmp_path
        path = tmp_path / "model.joblib"
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

        self.stack.enter_context(patch.object(Path, "replace", fail_sidecar))
        with self.assertRaisesRegex(OSError, "sidecar publication failed"):
            save_verified_joblib({"version": 2}, path)
        if existing:
            self.assertEqual((path.read_bytes(), sidecar.read_bytes()), before)
            self.assertEqual(load_verified_joblib(path), {"version": 1})
        else:
            self.assertTrue(not path.exists() and (not sidecar.exists()))
        self.assertEqual(
            set(tmp_path.iterdir()), {path, sidecar} if existing else set()
        )

    def _case_invalid_model_sidecar_fails_before_loading(self, contents):
        tmp_path = self.tmp_path
        path = tmp_path / "model.joblib"
        save_verified_joblib({}, path)
        path.with_name(path.name + ".sha256").write_text(contents)
        with self.assertRaisesRegex(ValueError, "Invalid SHA-256 sidecar"):
            load_verified_joblib(path)

    def test_atomic_text_creates_utf8_output_and_preserves_permissions(self):
        tmp_path = self.tmp_path
        path = tmp_path / "nested" / "report.txt"
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
        tmp_path = self.tmp_path
        path = tmp_path / "results.json"
        path.write_text("previous snapshot")
        original_write = Path.write_text

        def partial_write(destination, text, **kwargs):
            original_write(destination, "partial", **kwargs)
            raise OSError("disk failure")

        def failed_replace(*args):
            raise OSError("disk failure")

        self.stack.enter_context(
            patch.object(
                Path,
                "write_text" if failure == "write" else "replace",
                partial_write if failure == "write" else failed_replace,
            )
        )
        with self.assertRaisesRegex(OSError, "disk failure"):
            write_json(path, {"score": 0.9})
        self.assertEqual(path.read_text(), "previous snapshot")
        self.assertEqual(list(tmp_path.iterdir()), [path])

    def _case_invalid_json_cannot_replace_previous_results(self, value):
        tmp_path = self.tmp_path
        path = tmp_path / "results.json"
        path.write_text("previous snapshot")
        with self.assertRaises((TypeError, ValueError)):
            write_json(path, {"score": value})
        self.assertEqual(path.read_text(), "previous snapshot")
        self.assertEqual(list(tmp_path.iterdir()), [path])


for index, (a, b) in enumerate(
    [
        ([], []),
        ([1, 2], [0, 0]),
        ([1, 2, 3], [0]),
        ([[1, 2, 3]], [[0, 0, 0]]),
        ([1, 2, float("nan")], [0, 0, 0]),
    ]
):
    setattr(
        UtilsTests,
        f"test_effect_size_rejects_unaligned_or_invalid_pairs_{index}",
        partialmethod(
            UtilsTests._case_effect_size_rejects_unaligned_or_invalid_pairs, a=a, b=b
        ),
    )
for index, scale in enumerate([1e-200, 1e-17, 1.0, 1e150]):
    setattr(
        UtilsTests,
        f"test_paired_effect_size_is_unit_invariant_without_variance_underflow_{index}",
        partialmethod(
            UtilsTests._case_paired_effect_size_is_unit_invariant_without_variance_underflow,
            scale=scale,
        ),
    )
for index, existing in enumerate([False, True]):
    setattr(
        UtilsTests,
        f"test_failed_model_sidecar_publication_restores_previous_pair_{index}",
        partialmethod(
            UtilsTests._case_failed_model_sidecar_publication_restores_previous_pair,
            existing=existing,
        ),
    )
for index, contents in enumerate(["", "not a hash", "0" * 63]):
    setattr(
        UtilsTests,
        f"test_invalid_model_sidecar_fails_before_loading_{index}",
        partialmethod(
            UtilsTests._case_invalid_model_sidecar_fails_before_loading,
            contents=contents,
        ),
    )
for index, failure in enumerate(["write", "replace"]):
    setattr(
        UtilsTests,
        f"test_output_io_failure_keeps_previous_snapshot_and_cleans_temporary_files_{index}",
        partialmethod(
            UtilsTests._case_output_io_failure_keeps_previous_snapshot_and_cleans_temporary_files,
            failure=failure,
        ),
    )
for index, value in enumerate([float("nan"), float("inf"), object()]):
    setattr(
        UtilsTests,
        f"test_invalid_json_cannot_replace_previous_results_{index}",
        partialmethod(
            UtilsTests._case_invalid_json_cannot_replace_previous_results, value=value
        ),
    )
