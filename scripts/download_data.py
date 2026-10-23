"""WESAD requires its source research agreement."""

import argparse
import re
import sys
import zipfile
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.parse import urlsplit
from urllib.request import urlopen

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import EXTERNAL_DATA_DIR, RAW_DATA_DIR
from src.utils import sha256_file

NONEEG_URL = (
    "https://physionet.org/static/published-projects/noneeg/"
    "non-eeg-dataset-for-assessment-of-neurological-status-1.0.0.zip"
)
NONEEG_DIR = EXTERNAL_DATA_DIR / "noneeg"
WESAD_URL = "https://uni-siegen.sciebo.de/s/HGdUkoNlW1Ub0Gx/download"
MAX_UNCOMPRESSED = 10 * 1024**3
# Archive cap
WESAD_MAX_UNCOMPRESSED = 20 * 1024**3
DOWNLOAD_TIMEOUT = 30
NONEEG_FILES = tuple(
    f"Subject{subject}_{record}.{extension}"
    for subject in range(1, 21)
    for record, extensions in (
        ("AccTempEDA", ("hea", "dat", "atr")),
        ("SpO2HR", ("hea", "dat")),
    )
    for extension in extensions
)

WESAD_SHA256 = {
    "S2": "36ef5e8afc0f91998eefba7c12fc9fa97b7b07198cbec0126917d7abb436ca23",
    "S3": "5c8bd4a82af029c082e610bca28a011fca2ae3b23e14a18458ebb5990be4015e",
    "S4": "0f0740a79388723360ff12b4f47c465665ea7827d1399b18ac43908daac17900",
    "S5": "74bd187e3a9c1ca4259af52d04974c8e7ff7dc49ceea7e269f499ca98fe6d8ec",
    "S6": "8aa9bf57b69f4fe5bce06c550230857627c3f05befa2f787151646bb29ee8f62",
    "S7": "9cb62705ae7f53dca327a9a00a6f9fdabf5128d449174ab37594658e912cb6d8",
    "S8": "dac1141dac11d56b3641be982f45da63f05e9d74154f59e6ea0cdcf47fc72710",
    "S9": "24dc004e201bd541f092989443f0a29ebf89e4a227a80bb6b6d1987255039544",
    "S10": "41da29c68366f33650f3d41a6be78107bf6942929c3bb0ef46238078ddddee9f",
    "S11": "f39557a8d660b10154936f51debf2926aea7ebb9b26a168858f59502f914d8f7",
    "S13": "772fb490f19b279e49367271e009fc10d3a3ca1e3456df0d68b9063a73992066",
    "S14": "e7bd33c57538319a25c6d53e6a9fb6c1abd12800cfc64bb63275d89de8d2fd60",
    "S15": "1ea573bc6b45ba79fb134f9460d691b86176f60dce23420dc514c28017d4049c",
    "S16": "f65cf40cada75c3e9f5813276d7dcc90359c3b06dec41d68656c0a6e61dbc575",
    "S17": "3315796a75227d54d7b0056736f671484fd2fb85afffa65818fd76aeff2920fa",
}


def verify_wesad(target=None) -> None:
    root = Path(target) if target is not None else RAW_DATA_DIR / "WESAD"
    problems = []
    for sid, expected in WESAD_SHA256.items():
        path = root / sid / f"{sid}.pkl"
        if not path.exists():
            problems.append(f"{sid}: missing")
            continue
        got = sha256_file(path)
        if got != expected:
            problems.append(f"{sid}: checksum mismatch")
    if problems:
        raise SystemExit(
            "WESAD integrity check FAILED:\n  "
            + "\n  ".join(problems)
            + f"\nDownload source: {WESAD_URL} (research agreement applies)."
        )
    print(f"WESAD integrity OK: {len(WESAD_SHA256)} subjects verified.")


def _progress(downloaded, total):
    percent = min(100, 100 * downloaded / total) if total > 0 else 0
    print(f"\r  {downloaded // (1024 * 1024)} MB ({percent:.0f}%)", end="", flush=True)


def _download(url, dest):
    if urlsplit(url).scheme != "https":
        raise ValueError(f"refusing non-HTTPS download: {url}")
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {url}")
    with TemporaryDirectory(dir=dest.parent) as directory:
        part = Path(directory) / dest.name
        with urlopen(url, timeout=DOWNLOAD_TIMEOUT) as response, part.open(
            "wb"
        ) as stream:
            if urlsplit(response.geturl()).scheme != "https":
                raise ValueError("refusing a redirect to a non-HTTPS download")
            total = int(response.headers.get("Content-Length", 0))
            downloaded = 0
            while block := response.read(1024 * 1024):
                stream.write(block)
                downloaded += len(block)
                _progress(downloaded, total)
            if total > 0 and downloaded != total:
                raise RuntimeError("incomplete dataset download; retry the command")
        part.replace(dest)
    print()


def _safe_extract(zip_path, dest, max_bytes=MAX_UNCOMPRESSED):
    dest = Path(dest).resolve()
    with zipfile.ZipFile(zip_path) as z:
        total = 0
        for info in z.infolist():
            resolved = (dest / info.filename).resolve()
            if resolved != dest and dest not in resolved.parents:
                raise RuntimeError(f"unsafe path in archive: {info.filename}")
            total += info.file_size
            if total > max_bytes:
                raise RuntimeError(
                    "archive expands beyond the size cap; refusing to extract"
                )
        z.extractall(dest)


def _install_archive(zip_path, target, verify, *, max_bytes=MAX_UNCOMPRESSED):
    target = Path(target)
    with TemporaryDirectory(dir=target.parent) as directory:
        staging = Path(directory)
        _safe_extract(zip_path, staging, max_bytes=max_bytes)
        candidate = staging / target.name
        verify(candidate)
        candidate.replace(target)


def download_noneeg() -> None:
    target = NONEEG_DIR / "non-eeg-dataset-for-assessment-of-neurological-status-1.0.0"
    if target.exists():
        verify_noneeg(target)
        print(f"Non-EEG already present at {target}")
        return
    NONEEG_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = NONEEG_DIR / "noneeg.zip"
    _download(NONEEG_URL, zip_path)
    _install_archive(zip_path, target, verify_noneeg)
    zip_path.unlink()
    print(f"Non-EEG ready at {target}")


def verify_noneeg(target=None) -> None:
    target = (
        Path(target)
        if target is not None
        else NONEEG_DIR / "non-eeg-dataset-for-assessment-of-neurological-status-1.0.0"
    )
    manifest = target / "SHA256SUMS.txt"
    if not manifest.is_file():
        raise SystemExit(
            "Non-EEG integrity check failed: missing SHA256SUMS.txt; re-download the dataset"
        )
    references = {}
    for line in manifest.read_text().splitlines():
        parts = line.split(maxsplit=1)
        if len(parts) != 2 or not re.fullmatch(r"[0-9a-fA-F]{64}", parts[0]):
            raise SystemExit("Non-EEG integrity check failed: invalid SHA256SUMS.txt")
        references[parts[1].lstrip("*")] = parts[0].lower()
    problems = []
    for name in NONEEG_FILES:
        path = target / name
        if not path.is_file() or name not in references:
            problems.append(f"{name}: missing file or checksum")
        elif sha256_file(path) != references[name]:
            problems.append(f"{name}: checksum mismatch")
    if problems:
        raise SystemExit("Non-EEG integrity check failed:\n  " + "\n  ".join(problems))
    print(f"Non-EEG integrity OK: {len(NONEEG_FILES)} required files verified.")


def download_wesad() -> None:
    target = RAW_DATA_DIR / "WESAD"
    if target.exists():
        verify_wesad(target)
        print(f"WESAD already present at {target}")
        return
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = RAW_DATA_DIR / "WESAD.zip"
    print("WESAD is ~2 GB and covered by a research-only agreement.")
    _download(WESAD_URL, zip_path)
    _install_archive(zip_path, target, verify_wesad, max_bytes=WESAD_MAX_UNCOMPRESSED)
    zip_path.unlink()
    print(f"WESAD ready at {target}")


def _check(url) -> None:
    if urlsplit(url).scheme != "https":
        raise ValueError(f"refusing non-HTTPS request: {url}")
    with urlopen(url, timeout=DOWNLOAD_TIMEOUT) as r:
        if urlsplit(r.geturl()).scheme != "https":
            raise ValueError("refusing a redirect to a non-HTTPS request")
        size = r.headers.get("Content-Length")
        print(f"{r.status}  {url}  ({int(size) // (1024 * 1024) if size else '?'} MB)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch CalmSense datasets.")
    parser.add_argument(
        "--wesad", action="store_true", help="fetch WESAD (~2 GB, primary dataset)"
    )
    parser.add_argument(
        "--noneeg",
        action="store_true",
        help="fetch PhysioNet Non-EEG (cross-dataset transfer)",
    )
    parser.add_argument(
        "--check", action="store_true", help="only verify the download links"
    )
    parser.add_argument(
        "--verify-wesad",
        action="store_true",
        help="check downloaded WESAD .pkl SHA-256 and exit",
    )
    parser.add_argument(
        "--verify-noneeg",
        action="store_true",
        help="check Non-EEG SHA-256 manifest and exit",
    )
    args = parser.parse_args()

    if args.verify_wesad:
        verify_wesad()
    elif args.verify_noneeg:
        verify_noneeg()
    elif args.check:
        _check(NONEEG_URL)
        _check(WESAD_URL)
    elif not args.wesad and not args.noneeg:
        download_noneeg()
    else:
        if args.noneeg:
            download_noneeg()
        if args.wesad:
            download_wesad()
