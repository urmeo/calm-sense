"""Generate WESAD-shaped signals for testing, not performance evaluation."""

import pickle
from math import ceil
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


def _resample(signal: np.ndarray, sample_count: int) -> np.ndarray:
    source_time = np.linspace(0.0, 1.0, len(signal))
    target_time = np.linspace(0.0, 1.0, sample_count)
    return np.interp(target_time, source_time, signal)


def _fit_length(signal: np.ndarray, sample_count: int) -> np.ndarray:
    signal = np.asarray(signal)
    if len(signal) >= sample_count:
        return signal[:sample_count]
    return np.pad(signal, (0, sample_count - len(signal)), mode="edge")


def _chest_block(
    condition: str, seconds: int, rng: np.random.RandomState, seed: int
) -> Dict:
    import neurokit2 as nk

    sample_count = seconds * CHEST_FS
    heart_rate = HR[condition] + rng.uniform(-6, 6)
    ecg = _fit_length(
        cast(
            np.ndarray,
            nk.ecg_simulate(
                duration=seconds,
                sampling_rate=CHEST_FS,
                heart_rate=cast(Any, heart_rate),
                noise=NOISE,
                method="simple",
                random_state=seed,
            ),
        ),
        sample_count,
    )
    eda = EDA_LEVEL[condition] + _fit_length(
        cast(
            np.ndarray,
            nk.eda_simulate(
                duration=seconds,
                sampling_rate=CHEST_FS,
                scr_number=SCR[condition],
                noise=NOISE,
                random_state=seed,
            ),
        ),
        sample_count,
    )
    # Minimum duration
    resp = _fit_length(
        nk.rsp_simulate(
            duration=max(seconds, ceil(2 * 60 / RESP_RATE[condition])),
            sampling_rate=CHEST_FS,
            respiratory_rate=RESP_RATE[condition],
            random_state=seed,
        ),
        sample_count,
    )
    temp = TEMP_BASE[condition] + rng.normal(0, 0.02, sample_count).cumsum() / CHEST_FS
    acc = rng.normal(0, MOTION[condition], (sample_count, 3)) + np.array(
        [0.0, 0.0, 1.0]
    )
    emg = rng.normal(0, 0.01, sample_count)

    return {
        "ECG": ecg.reshape(-1, 1),
        "EDA": eda.reshape(-1, 1),
        "Temp": temp.reshape(-1, 1),
        "Resp": resp.reshape(-1, 1),
        "EMG": emg.reshape(-1, 1),
        "ACC": acc,
    }


def _wrist_block(
    condition: str, seconds: int, chest: Dict, rng: np.random.RandomState
) -> Dict:
    """Approximate wrist BVP with a noisy sinusoid."""
    bvp_n = int(seconds * WRIST["BVP"])
    eda_n = int(seconds * WRIST["EDA"])
    temp_n = int(seconds * WRIST["TEMP"])
    acc_n = int(seconds * WRIST["ACC"])
    t = np.arange(bvp_n) / WRIST["BVP"]
    bvp = np.sin(2 * np.pi * (HR[condition] / 60.0) * t) + rng.normal(0, 0.05, bvp_n)
    return {
        "BVP": bvp.reshape(-1, 1),
        "EDA": _resample(chest["EDA"].ravel(), eda_n).reshape(-1, 1),
        "TEMP": _resample(chest["Temp"].ravel(), temp_n).reshape(-1, 1),
        "ACC": rng.normal(0, MOTION[condition], (acc_n, 3)) + np.array([0.0, 0.0, 1.0]),
    }


def _subject(seed: int, block_sec: int) -> Dict:
    rng = np.random.RandomState(seed)
    order = ["baseline", "stress", "amusement", "baseline", "stress"]

    chest_parts: Dict[str, list] = {
        channel: [] for channel in ("ECG", "EDA", "Temp", "Resp", "EMG", "ACC")
    }
    wrist_parts: Dict[str, list] = {channel: [] for channel in WRIST}
    labels = []

    for block_index, condition in enumerate(order):
        chest = _chest_block(condition, block_sec, rng, seed + block_index)
        wrist = _wrist_block(condition, block_sec, chest, rng)
        for channel, parts in chest_parts.items():
            parts.append(chest[channel])
        for channel, parts in wrist_parts.items():
            parts.append(wrist[channel])
        labels.append(np.full(len(chest["ECG"]), CONDITIONS[condition]))

    return {
        "signal": {
            "chest": {
                channel: np.concatenate(parts) for channel, parts in chest_parts.items()
            },
            "wrist": {
                channel: np.concatenate(parts) for channel, parts in wrist_parts.items()
            },
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
    for subject_index, subject_id in enumerate(VALID_SUBJECTS[:n_subjects]):
        data = _subject(seed + subject_index * 100, block_sec)
        subject_dir = root / subject_id
        subject_dir.mkdir(parents=True, exist_ok=True)
        with (subject_dir / f"{subject_id}.pkl").open("wb") as stream:
            pickle.dump(data, stream, protocol=4)
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
        return dataset.build(subjects=dataset.loader.subjects, cache=False)
