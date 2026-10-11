"""Generate WESAD-shaped signals for testing, not performance evaluation."""

import pickle
from pathlib import Path
from typing import Any, cast, Dict, Tuple

import numpy as np

from .config import VALID_SUBJECTS
from .preprocessing.filters import _positive_integer

CHEST_FS = 700
WRIST = {"BVP": 64, "EDA": 4, "TEMP": 4, "ACC": 32}

CONDITIONS = {"baseline": 1, "stress": 2, "amusement": 3}
HR = {"baseline": 74, "stress": 86, "amusement": 80}
SCR = {"baseline": 2, "stress": 4, "amusement": 3}
EDA_LEVEL = {"baseline": 4.5, "stress": 6.5, "amusement": 5.2}
RESP_RATE = {"baseline": 15, "stress": 19, "amusement": 16}
MOTION = {"baseline": 0.03, "stress": 0.07, "amusement": 0.05}
TEMP_BASE = {"baseline": 33.1, "stress": 33.5, "amusement": 33.3}
NOISE = 0.06


def _resample(x: np.ndarray, n: int) -> np.ndarray:
    src = np.linspace(0.0, 1.0, len(x))
    dst = np.linspace(0.0, 1.0, n)
    return np.interp(dst, src, x)


def _fixlen(x: np.ndarray, n: int) -> np.ndarray:
    x = np.asarray(x)
    return x[:n] if len(x) >= n else np.pad(x, (0, n - len(x)), mode="edge")


def _chest_block(
    cond: str, seconds: int, rng: np.random.RandomState, seed: int
) -> Dict:
    import neurokit2 as nk

    n = seconds * CHEST_FS
    hr = HR[cond] + rng.uniform(-6, 6)
    ecg = _fixlen(
        cast(
            np.ndarray,
            nk.ecg_simulate(
                duration=seconds,
                sampling_rate=CHEST_FS,
                heart_rate=cast(Any, hr),
                noise=NOISE,
                method="simple",
                random_state=seed,
            ),
        ),
        n,
    )
    eda = EDA_LEVEL[cond] + _fixlen(
        cast(
            np.ndarray,
            nk.eda_simulate(
                duration=seconds,
                sampling_rate=CHEST_FS,
                scr_number=SCR[cond],
                noise=NOISE,
                random_state=seed,
            ),
        ),
        n,
    )
    resp = _fixlen(
        nk.rsp_simulate(
            duration=seconds,
            sampling_rate=CHEST_FS,
            respiratory_rate=RESP_RATE[cond],
            random_state=seed,
        ),
        n,
    )
    temp = TEMP_BASE[cond] + rng.normal(0, 0.02, n).cumsum() / CHEST_FS
    acc = rng.normal(0, MOTION[cond], (n, 3)) + np.array([0.0, 0.0, 1.0])
    emg = rng.normal(0, 0.01, n)

    return {
        "ECG": ecg.reshape(-1, 1),
        "EDA": eda.reshape(-1, 1),
        "Temp": temp.reshape(-1, 1),
        "Resp": resp.reshape(-1, 1),
        "EMG": emg.reshape(-1, 1),
        "ACC": acc,
    }


def _wrist_block(
    cond: str, seconds: int, chest: Dict, rng: np.random.RandomState
) -> Dict:
    """BVP is sinusoidal."""
    bvp_n = int(seconds * WRIST["BVP"])
    eda_n = int(seconds * WRIST["EDA"])
    acc_n = int(seconds * WRIST["ACC"])
    t = np.arange(bvp_n) / WRIST["BVP"]
    bvp = np.sin(2 * np.pi * (HR[cond] / 60.0) * t) + rng.normal(0, 0.05, bvp_n)
    return {
        "BVP": bvp.reshape(-1, 1),
        "EDA": _resample(chest["EDA"].ravel(), eda_n).reshape(-1, 1),
        "TEMP": _resample(chest["Temp"].ravel(), eda_n).reshape(-1, 1),
        "ACC": rng.normal(0, MOTION[cond], (acc_n, 3)) + np.array([0.0, 0.0, 1.0]),
    }


def _subject(seed: int, block_sec: int) -> Dict:
    rng = np.random.RandomState(seed)
    order = ["baseline", "stress", "amusement", "baseline", "stress"]

    chest_parts: Dict[str, list] = {
        k: [] for k in ["ECG", "EDA", "Temp", "Resp", "EMG", "ACC"]
    }
    wrist_parts: Dict[str, list] = {k: [] for k in WRIST}
    labels = []

    for i, cond in enumerate(order):
        chest = _chest_block(cond, block_sec, rng, seed + i)
        wrist = _wrist_block(cond, block_sec, chest, rng)
        for k in chest_parts:
            chest_parts[k].append(chest[k])
        for k in wrist_parts:
            wrist_parts[k].append(wrist[k])
        labels.append(np.full(len(chest["ECG"]), CONDITIONS[cond]))

    return {
        "signal": {
            "chest": {k: np.concatenate(v) for k, v in chest_parts.items()},
            "wrist": {k: np.concatenate(v) for k, v in wrist_parts.items()},
        },
        "label": np.concatenate(labels),
    }


def write_dataset(
    out_dir: Path, n_subjects: int = 4, block_sec: int = 120, seed: int = 0
) -> Path:
    n_subjects = _positive_integer(n_subjects, "n_subjects")
    block_sec = _positive_integer(block_sec, "block_sec")
    if n_subjects > len(VALID_SUBJECTS):
        raise ValueError(f"n_subjects must not exceed {len(VALID_SUBJECTS)} WESAD IDs")
    root = Path(out_dir) / "WESAD"
    for i, sid in enumerate(VALID_SUBJECTS[:n_subjects]):
        data = _subject(seed + i * 100, block_sec)
        subj_dir = root / sid
        subj_dir.mkdir(parents=True, exist_ok=True)
        with open(subj_dir / f"{sid}.pkl", "wb") as f:
            pickle.dump(data, f, protocol=4)
    return root


def features(
    n_subjects: int = 4, block_sec: int = 120, seed: int = 0, cache: bool = False
) -> Tuple:
    if cache:
        raise ValueError(
            "Synthetic features cannot replace real-data caches; use cache=False"
        )
    from tempfile import TemporaryDirectory

    from .dataset import WindowedDataset

    with TemporaryDirectory(
        prefix="calmsense_synth_", ignore_cleanup_errors=True
    ) as directory:
        root = write_dataset(
            Path(directory), n_subjects=n_subjects, block_sec=block_sec, seed=seed
        )
        dataset = WindowedDataset(data_path=root)
        return dataset.build(subjects=dataset.loader.subjects, cache=cache)
