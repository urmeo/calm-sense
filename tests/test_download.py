import unittest
from contextlib import ExitStack
from unittest.mock import patch
from pathlib import Path
from tempfile import TemporaryDirectory
from functools import partialmethod
import hashlib
import io
import json
import zipfile
from scripts import download_data
from scripts.download_data import _download, _safe_extract


class DownloadResponse(io.BytesIO):

    def __init__(self, payload, *, length=None, url="https://example.com/data.zip"):
        super().__init__(payload)
        self.headers = {
            "Content-Length": str(len(payload) if length is None else length)
        }
        self.url = url

    def geturl(self):
        return self.url


class DownloadTests(unittest.TestCase):

    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.tmp_path = Path(self.stack.enter_context(TemporaryDirectory()))

    def test_safe_extract_unpacks_benign_zip(self):
        tmp_path = self.tmp_path
        archive = tmp_path / "ok.zip"
        with zipfile.ZipFile(archive, "w") as z:
            z.writestr("a/b.txt", "hello")
        out = tmp_path / "out"
        _safe_extract(archive, out)
        self.assertEqual((out / "a" / "b.txt").read_text(), "hello")

    def test_safe_extract_blocks_zip_slip(self):
        tmp_path = self.tmp_path
        archive = tmp_path / "evil.zip"
        with zipfile.ZipFile(archive, "w") as z:
            z.writestr("../escape.txt", "pwned")
        with self.assertRaisesRegex(RuntimeError, "unsafe path"):
            _safe_extract(archive, tmp_path / "out")

    def test_safe_extract_blocks_zip_bomb(self):
        tmp_path = self.tmp_path
        archive = tmp_path / "big.zip"
        with zipfile.ZipFile(archive, "w") as z:
            z.writestr("data.bin", b"x" * 4096)
        with self.assertRaisesRegex(RuntimeError, "size cap"):
            _safe_extract(archive, tmp_path / "out", max_bytes=1024)

    def test_wesad_download_accepts_recorded_uncompressed_size(self):
        provenance = (
            Path(__file__).resolve().parents[1] / "outputs/results/provenance.json"
        )
        expanded_bytes = json.loads(provenance.read_text())["input_verification"][
            "wesad"
        ]["bytes"]
        payload = b"verified recording"
        self.stack.enter_context(
            patch.object(download_data, "RAW_DATA_DIR", self.tmp_path)
        )
        self.stack.enter_context(
            patch.object(
                download_data,
                "WESAD_SHA256",
                {"S2": hashlib.sha256(payload).hexdigest()},
            )
        )

        def download_archive(url, destination):
            destination.write_bytes(b"archive placeholder")

        self.stack.enter_context(
            patch.object(download_data, "_download", download_archive)
        )
        archive = self.stack.enter_context(
            patch.object(download_data.zipfile, "ZipFile")
        )
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
        tmp_path = self.tmp_path
        with self.assertRaisesRegex(ValueError, "non-HTTPS"):
            _download("http://example.com/x.zip", tmp_path / "x.zip")

    def test_link_check_refuses_non_https_urls_and_redirects(self):
        with self.assertRaisesRegex(ValueError, "non-HTTPS"):
            download_data._check("http://example.com/data.zip")
        response = DownloadResponse(b"", url="http://example.com/data.zip")
        with patch.object(download_data, "urlopen", return_value=response):
            with self.assertRaisesRegex(ValueError, "non-HTTPS"):
                download_data._check("https://example.com/data.zip")

    def _case_existing_wesad_must_be_complete_and_verified(self, missing_subject):
        tmp_path = self.tmp_path
        self.stack.enter_context(patch.object(download_data, "RAW_DATA_DIR", tmp_path))
        reference = {}
        for sid in ("S2", "S3"):
            payload = sid.encode()
            reference[sid] = hashlib.sha256(payload).hexdigest()
            if sid == missing_subject:
                continue
            subject = tmp_path / "WESAD" / sid
            subject.mkdir(parents=True)
            (subject / f"{sid}.pkl").write_bytes(payload)
        self.stack.enter_context(patch.object(download_data, "WESAD_SHA256", reference))
        self.stack.enter_context(
            patch.object(
                download_data,
                "_download",
                lambda *args: self.fail("Unexpected download"),
            )
        )
        if missing_subject:
            with self.assertRaisesRegex(SystemExit, f"{missing_subject}: missing"):
                download_data.download_wesad()
        else:
            download_data.download_wesad()

    def test_download_streams_to_complete_file(self):
        tmp_path = self.tmp_path
        response = DownloadResponse(b"verified archive")
        calls = []

        def open_response(url, *, timeout):
            calls.append((url, timeout))
            return response

        self.stack.enter_context(patch.object(download_data, "urlopen", open_response))
        dest = tmp_path / "nested" / "data.zip"
        _download("https://example.com/data.zip", str(dest))
        self.assertEqual(dest.read_bytes(), b"verified archive")
        self.assertEqual(
            calls, [("https://example.com/data.zip", download_data.DOWNLOAD_TIMEOUT)]
        )
        self.assertFalse(dest.with_suffix(".zip.part").exists())

    def _case_download_failure_preserves_previous_file(self, redirect):
        tmp_path = self.tmp_path
        response = DownloadResponse(
            b"short body",
            length=100,
            url=(
                "http://example.com/data.zip"
                if redirect
                else "https://example.com/data.zip"
            ),
        )
        self.stack.enter_context(
            patch.object(download_data, "urlopen", lambda *args, **kwargs: response)
        )
        dest = tmp_path / "data.zip"
        dest.write_bytes(b"previous archive")
        error = ValueError if redirect else RuntimeError
        with self.assertRaisesRegex(error, "non-HTTPS|incomplete"):
            _download("https://example.com/data.zip", dest)
        self.assertEqual(dest.read_bytes(), b"previous archive")
        self.assertFalse(dest.with_suffix(".zip.part").exists())

    def _case_existing_noneeg_requires_complete_publisher_manifest(self, problem):
        tmp_path = self.tmp_path
        self.stack.enter_context(patch.object(download_data, "NONEEG_DIR", tmp_path))
        names = ("Subject1_AccTempEDA.dat", "Subject1_SpO2HR.hea")
        self.stack.enter_context(patch.object(download_data, "NONEEG_FILES", names))
        target = (
            tmp_path / "non-eeg-dataset-for-assessment-of-neurological-status-1.0.0"
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
        self.stack.enter_context(
            patch.object(
                download_data,
                "_download",
                lambda *args: self.fail("Unexpected download"),
            )
        )
        if problem:
            with self.assertRaisesRegex(SystemExit, "integrity check failed"):
                download_data.download_noneeg()
        else:
            download_data.download_noneeg()

    def _case_dataset_is_installed_only_after_integrity_verification_and_retry_succeeds(
        self, dataset
    ):
        tmp_path = self.tmp_path
        payload = b"verified recording"
        digest = hashlib.sha256(payload).hexdigest()
        valid = False
        if dataset == "wesad":
            self.stack.enter_context(
                patch.object(download_data, "RAW_DATA_DIR", tmp_path)
            )
            self.stack.enter_context(
                patch.object(download_data, "WESAD_SHA256", {"S2": digest})
            )
            target = tmp_path / "WESAD"
            file_name = "S2/S2.pkl"
            download = download_data.download_wesad
        else:
            self.stack.enter_context(
                patch.object(download_data, "NONEEG_DIR", tmp_path)
            )
            self.stack.enter_context(
                patch.object(download_data, "NONEEG_FILES", ("record.dat",))
            )
            target = (
                tmp_path / "non-eeg-dataset-for-assessment-of-neurological-status-1.0.0"
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

        self.stack.enter_context(
            patch.object(download_data, "_download", download_archive)
        )
        with self.assertRaisesRegex(SystemExit, "checksum mismatch"):
            download()
        self.assertFalse(target.exists())
        self.assertEqual([p.suffix for p in tmp_path.iterdir()], [".zip"])
        valid = True
        download()
        self.assertEqual((target / file_name).read_bytes(), payload)
        self.assertEqual(list(tmp_path.iterdir()), [target])


for index, missing_subject in enumerate([None, "S2", "S3"]):
    setattr(
        DownloadTests,
        f"test_existing_wesad_must_be_complete_and_verified_{index}",
        partialmethod(
            DownloadTests._case_existing_wesad_must_be_complete_and_verified,
            missing_subject=missing_subject,
        ),
    )
for index, redirect in enumerate([False, True]):
    setattr(
        DownloadTests,
        f"test_download_failure_preserves_previous_file_{index}",
        partialmethod(
            DownloadTests._case_download_failure_preserves_previous_file,
            redirect=redirect,
        ),
    )
for index, problem in enumerate([None, "missing", "changed", "manifest"]):
    setattr(
        DownloadTests,
        f"test_existing_noneeg_requires_complete_publisher_manifest_{index}",
        partialmethod(
            DownloadTests._case_existing_noneeg_requires_complete_publisher_manifest,
            problem=problem,
        ),
    )
for index, dataset in enumerate(["wesad", "noneeg"]):
    setattr(
        DownloadTests,
        f"test_dataset_is_installed_only_after_integrity_verification_and_retry_succeeds_{index}",
        partialmethod(
            DownloadTests._case_dataset_is_installed_only_after_integrity_verification_and_retry_succeeds,
            dataset=dataset,
        ),
    )
