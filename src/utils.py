import os
import random
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Generator, Union, cast

from .artifacts import publish_files


def set_seed(seed: int = 42, deterministic: bool = True) -> None:
    """PYTHONHASHSEED affects child processes."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)

    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass

    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = deterministic
        if deterministic:
            torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(deterministic, warn_only=True)
    except ImportError:
        pass


@contextmanager
def timer(name: str = "Operation") -> Generator[None, None, None]:
    from .logging_config import get_logger

    logger = get_logger(__name__)
    start = time.perf_counter()
    try:
        yield
    finally:
        logger.info(f"{name} finished in {time.perf_counter() - start:.2f} seconds")


def atomic_write_text(path: Union[str, Path], text: str) -> None:
    """Replace complete text; preserve permissions."""
    from stat import S_IMODE
    from tempfile import TemporaryDirectory

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=path.parent) as directory:
        temporary = Path(directory) / path.name
        temporary.write_text(text, encoding="utf-8")
        if path.exists():
            temporary.chmod(S_IMODE(path.stat().st_mode))
        temporary.replace(path)


def write_json(path: Union[str, Path], value) -> None:
    """Reject nonfinite JSON before replacement."""
    import json

    atomic_write_text(path, json.dumps(value, indent=2, allow_nan=False))


def provenance() -> dict:
    import subprocess
    from datetime import datetime, timezone

    from .features.feature_pipeline import FEATURE_SCHEMA_VERSION

    root = Path(__file__).resolve().parent.parent
    sha, dirty = "unknown", None
    try:
        git_root = subprocess.check_output(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=root,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        if Path(git_root).resolve() == root:
            sha = subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=root,
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
            dirty = bool(
                subprocess.check_output(
                    ["git", "status", "--porcelain"],
                    cwd=root,
                    text=True,
                    stderr=subprocess.DEVNULL,
                ).strip()
            )
    except (OSError, subprocess.CalledProcessError):
        sha = "unknown"
        dirty = None
    return {
        "git_sha": sha,
        "working_tree_dirty": dirty,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


def sha256_file(path: Union[str, Path]) -> str:
    import hashlib

    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def feature_frame_sha256(frame) -> str:
    import hashlib
    import json
    from pandas.util import hash_pandas_object

    metadata = {
        "columns": list(frame.columns),
        "dtypes": [str(dtype) for dtype in frame.dtypes],
        "attrs": frame.attrs,
    }
    digest = hashlib.sha256(
        json.dumps(metadata, sort_keys=True, allow_nan=False).encode()
    )
    hash_frame = cast(Callable[..., Any], hash_pandas_object)
    digest.update(hash_frame(frame, index=True).to_numpy().astype("<u8").tobytes())
    return digest.hexdigest()


def pipeline_source_sha256() -> dict:
    root = Path(__file__).resolve().parent.parent
    files = sorted((root / "src").rglob("*.py")) + sorted(
        (root / "scripts").glob("*.py")
    )
    return {str(path.relative_to(root)): sha256_file(path) for path in files}


def benchmark_reference(
    results_dir: Union[str, Path], frame=None, *, shared_cache=True, recorded=False
):
    import hashlib
    import json
    import re

    from .config import PROCESSED_DATA_DIR
    from .features.feature_pipeline import FEATURE_SCHEMA_VERSION

    path = Path(results_dir) / "metrics.json"
    if not path.exists():
        return None
    payload = path.read_bytes()
    metrics = json.loads(payload)
    if not isinstance(metrics, dict) or metrics.get("benchmark_protocol_version") != 2:
        return None
    context, inputs = metrics.get("provenance", {}), metrics.get("inputs", {})
    if not isinstance(context, dict) or not isinstance(inputs, dict):
        return None
    if (
        inputs.get("dataset") != "WESAD"
        or not isinstance(context.get("git_sha"), str)
        or not re.fullmatch(r"[0-9a-f]{40,64}", context["git_sha"])
        or any(
            not isinstance(metrics.get(task), dict)
            or metrics[task].get("feature_schema_version") != FEATURE_SCHEMA_VERSION
            for task in ("binary", "multiclass")
        )
    ):
        return None
    sources = context.get("source_file_sha256")
    if (
        not isinstance(sources, dict)
        or not sources
        or any(
            not isinstance(name, str)
            or not re.fullmatch(r"(?:src|scripts)/(?:[\w]+/)*[\w]+\.py", name)
            or not isinstance(digest, str)
            or not re.fullmatch(r"[0-9a-f]{64}", digest)
            for name, digest in sources.items()
        )
    ):
        return None
    if recorded and (shared_cache or frame is not None):
        raise ValueError(
            "Recorded snapshot verification cannot validate current inputs"
        )
    if not recorded and sources != pipeline_source_sha256():
        raise ValueError(
            "Primary benchmark source differs from the current pipeline; rerun it"
        )
    keys = ("feature_frame_sha256", "feature_cache_sha256", "raw_windows_sha256")
    if any(
        not isinstance(inputs.get(key), str)
        or not re.fullmatch(r"[0-9a-f]{64}", inputs[key])
        for key in keys
    ):
        return None
    if shared_cache:
        for name, key in (
            ("features.parquet", "feature_cache_sha256"),
            ("raw_windows.npz", "raw_windows_sha256"),
        ):
            cache = PROCESSED_DATA_DIR / name
            if not cache.exists() or sha256_file(cache) != inputs[key]:
                raise ValueError(
                    "Primary benchmark cache differs from the current analysis inputs"
                )
        if (
            frame is not None
            and feature_frame_sha256(frame) != inputs["feature_frame_sha256"]
        ):
            raise ValueError(
                "Analysis feature frame differs from the primary benchmark inputs"
            )
    return hashlib.sha256(payload).hexdigest()


def analysis_provenance(results_dir: Union[str, Path], benchmark_sha256) -> dict:
    context = provenance()
    if benchmark_sha256 is not None:
        path = Path(results_dir) / "metrics.json"
        if not path.exists() or sha256_file(path) != benchmark_sha256:
            raise ValueError("Primary benchmark changed during the analysis")
        if benchmark_reference(results_dir, shared_cache=False) != benchmark_sha256:
            raise ValueError("Primary benchmark context changed during the analysis")
        context["primary_benchmark_sha256"] = benchmark_sha256
        context["source_file_sha256"] = pipeline_source_sha256()
    return context


def save_verified_joblib(bundle, path: Union[str, Path]) -> None:
    from tempfile import TemporaryDirectory

    import joblib

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=path.parent) as directory:
        temporary = Path(directory) / path.name
        checksum = temporary.with_name(path.name + ".sha256")
        joblib.dump(bundle, temporary)
        digest = sha256_file(temporary)
        checksum.write_text(f"{digest}  {path.name}\n")
        publish_files(
            {path: temporary, path.with_name(path.name + ".sha256"): checksum}
        )


def load_verified_joblib(path: Union[str, Path]):
    """Hash-check trusted bundles before deserialization."""
    import hashlib
    import io
    import re

    import joblib

    path = Path(path)
    checksum = path.with_name(path.name + ".sha256").read_text().split()
    if not checksum or not re.fullmatch(r"[0-9a-fA-F]{64}", checksum[0]):
        raise ValueError(f"Invalid SHA-256 sidecar for {path.name}")
    expected = checksum[0].lower()
    payload = path.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    if digest != expected:
        raise ValueError(
            f"SHA-256 mismatch for {path.name}: refusing to load an unverified pickle."
        )
    return joblib.load(io.BytesIO(payload))


def paired_effect_size(a, b) -> dict:
    """Paired effects; constant nonzero differences are undefined."""
    from math import exp, lgamma, log

    import numpy as np

    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if a.ndim != 1 or a.shape != b.shape or len(a) < 3:
        raise ValueError("Paired effect size requires at least three aligned 1D pairs")
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("Paired observations must be finite")
    with np.errstate(over="ignore", invalid="ignore"):
        diff = a - b
    n = len(diff)
    if not np.isfinite(diff).all():
        raise ValueError("Paired differences exceed the supported numerical range")
    magnitude = float(np.abs(diff).max())
    if magnitude == 0:
        return {"cohens_d": 0.0, "hedges_g": 0.0, "n": int(n)}
    scaled = diff / magnitude
    sd = scaled.std(ddof=1)
    if sd <= np.finfo(float).eps:
        return {"cohens_d": None, "hedges_g": None, "n": int(n)}
    d = float(scaled.mean() / sd)
    df = n - 1
    correction = exp(lgamma(df / 2) - 0.5 * log(df / 2) - lgamma((df - 1) / 2))
    g = d * correction
    return {"cohens_d": d, "hedges_g": float(g), "n": int(n)}
