import hashlib
import os
import sys
from contextlib import contextmanager
from dataclasses import asdict
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import BinaryIO, Dict, Iterable, Iterator, Optional, Tuple

from . import config

SOURCE_ROOT = Path(__file__).resolve().parents[1]
EXTRACTION_PACKAGES = ("numpy", "pandas", "scipy", "neurokit2")
IMPORTED_SETTINGS = {
    name: getattr(config, name)
    for name in ("FS", "FILTER_PARAMS", "FEATURE_PARAMS", "RR_MIN_MS", "RR_MAX_MS")
}


def _file_state(stat: os.stat_result) -> Tuple[int, ...]:
    return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)


@contextmanager
def verified_input(path: Path) -> Iterator[Tuple[BinaryIO, str]]:
    """Hash and read one open file; reject changes during either operation."""
    with path.open("rb") as stream:
        initial = _file_state(os.fstat(stream.fileno()))
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
        stream.seek(0)
        if initial != _file_state(os.fstat(stream.fileno())):
            raise RuntimeError(f"Input changed while hashing: {path}")
        yield stream, digest
        if initial != _file_state(os.fstat(stream.fileno())) or initial != _file_state(
            path.stat()
        ):
            raise RuntimeError(f"Input changed while reading: {path}")


def source_sha256() -> Dict[str, str]:
    files = [
        SOURCE_ROOT / "src" / name
        for name in (
            "cache.py",
            "config.py",
            "utils.py",
            "data/loader.py",
            "dataset.py",
            "dataset_wrist.py",
        )
    ]
    for directory in ("features", "preprocessing"):
        files.extend((SOURCE_ROOT / "src" / directory).rglob("*.py"))
    hashes = {}
    for path in sorted(files):
        with verified_input(path) as (_, digest):
            hashes[path.relative_to(SOURCE_ROOT).as_posix()] = digest
    return hashes


IMPORTED_SOURCE_SHA256 = source_sha256()


def extraction_context(device: str) -> Dict:
    if device not in {"chest", "wrist"}:
        raise ValueError("device must be chest or wrist")
    source = source_sha256()
    if source != IMPORTED_SOURCE_SHA256:
        raise RuntimeError("Extraction code changed; restart Python before rebuilding")
    for name, original in IMPORTED_SETTINGS.items():
        current = getattr(config, name)
        if current != original:
            raise RuntimeError(
                "Extraction settings changed; restart Python before rebuilding"
            )
        for path in source:
            module_name = path.removesuffix(".py").replace("/", ".")
            module = sys.modules.get(module_name)
            if module is not None and getattr(module, name, current) != current:
                raise RuntimeError(
                    "Extraction settings changed; restart Python before rebuilding"
                )
    packages = {}
    for package in EXTRACTION_PACKAGES:
        try:
            packages[package] = version(package)
        except PackageNotFoundError:
            packages[package] = None
    return {
        "version": 1,
        "device": device,
        "source_sha256": source,
        "sampling_rates": asdict(config.FS),
        "filter_parameters": asdict(config.FILTER_PARAMS),
        "feature_parameters": asdict(config.FEATURE_PARAMS),
        "rr_bounds_ms": [config.RR_MIN_MS, config.RR_MAX_MS],
        "packages": packages,
    }


def raw_input_sha256(data_path: Path, subjects: Iterable[str]) -> Dict[str, str]:
    subjects = list(subjects)
    if (
        not subjects
        or len(subjects) != len(set(subjects))
        or any(subject not in config.VALID_SUBJECTS for subject in subjects)
    ):
        raise ValueError("subjects must be a nonempty list of unique WESAD subjects")
    hashes = {}
    for subject in sorted(subjects):
        relative = f"{subject}/{subject}.pkl"
        with verified_input(Path(data_path) / relative) as (_, digest):
            hashes[relative] = digest
    return hashes


def cache_fingerprint(
    device: str,
    data_path: Path,
    subjects: Iterable[str],
    *,
    context: Optional[Dict] = None,
    loaded_inputs: Optional[Dict[str, str]] = None,
) -> Dict:
    current_context = extraction_context(device)
    if context is not None and context != current_context:
        raise RuntimeError("Extraction settings changed during the build")
    raw = raw_input_sha256(data_path, subjects)
    if loaded_inputs is not None and any(
        loaded_inputs.get(path) != digest for path, digest in raw.items()
    ):
        raise RuntimeError("Raw recordings changed during the build; rebuild the cache")
    if current_context != extraction_context(device):
        raise RuntimeError("Extraction settings changed while verifying raw recordings")
    return {"extraction": current_context, "raw_input_sha256": raw}
