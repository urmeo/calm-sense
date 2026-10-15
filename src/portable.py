"""Native units; EDA/TEMP slopes per second."""

from pathlib import Path
from typing import Any, cast, Dict, List, Optional, Union

import numpy as np
import pandas as pd

from .config import FS
from .data.loader import WESADLoader
from .preprocessing.filters import _positive_number

WINDOW_SEC = 60.0
OVERLAP = 0.5
PURITY = 0.9
PORTABLE_SCHEMA_VERSION = 3
PORTABLE_SIGNAL_UNITS = {
    "wesad": {"EDA": "uS", "TEMP": "degC", "ACC": "1/64 g", "HR": "bpm"},
    "noneeg": {"EDA": "NU", "TEMP": "degC", "ACC": "NU", "HR": "bpm"},
}
FEATURE_KEYS = {
    "EDA": ["mean", "std", "min", "max", "range", "slope"],
    "TEMP": ["mean", "std", "min", "max", "slope"],
    "ACC": ["mean", "std", "energy"],
    "HR": ["mean", "std", "min", "max"],
}
PORTABLE_FEATURE_COLUMNS = [
    f"{prefix}_{key}" for prefix, keys in FEATURE_KEYS.items() for key in keys
]


def _stats(
    x: np.ndarray, prefix: str, keys: List[str], sampling_rate: float = 1.0
) -> Dict[str, float]:
    sampling_rate = _positive_number(sampling_rate, "sampling_rate")
    x = np.asarray(x, dtype=float).ravel()
    t = np.arange(len(x), dtype=float) / sampling_rate
    finite = np.isfinite(x)
    x, t = x[finite], t[finite]
    out = {f"{prefix}_{k}": np.nan for k in keys}
    if len(x) < 2:
        return out
    funcs = {
        "mean": lambda: np.mean(x),
        "std": lambda: np.std(x),
        "min": lambda: np.min(x),
        "max": lambda: np.max(x),
        "range": lambda: np.ptp(x),
        "slope": lambda: np.polyfit(t, x, 1)[0],
        "energy": lambda: np.mean(x**2),
    }
    for k in keys:
        out[f"{prefix}_{k}"] = float(funcs[k]())
    return out


def portable_features(
    eda, temp, acc_mag, hr, *, eda_fs: float, temp_fs: float
) -> Dict[str, float]:
    feats = {}
    feats.update(_stats(eda, "EDA", FEATURE_KEYS["EDA"], eda_fs))
    feats.update(_stats(temp, "TEMP", FEATURE_KEYS["TEMP"], temp_fs))
    feats.update(_stats(acc_mag, "ACC", FEATURE_KEYS["ACC"]))
    feats.update(_stats(hr, "HR", FEATURE_KEYS["HR"]))
    return feats


def _window_label(labels: np.ndarray, keep: set) -> Optional[int]:
    if not len(labels):
        return None
    values, counts = np.unique(labels, return_counts=True)
    dominant = values[counts.argmax()]
    if dominant not in keep:
        return None
    if counts.max() / counts.sum() < PURITY:
        return None
    return int(dominant)


def wesad_portable(
    subjects: Optional[List[str]] = None, data_path: Optional[Union[str, Path]] = None
) -> pd.DataFrame:
    import neurokit2 as nk

    loader = WESADLoader(data_path=data_path)
    subjects = loader.subjects if subjects is None else subjects
    if not subjects or len(set(subjects)) != len(subjects):
        raise ValueError("subjects must be a nonempty list without duplicates")
    step = WINDOW_SEC * (1 - OVERLAP)
    rows = []

    for subject_id in subjects:
        recording = loader.load_subject(subject_id)
        wrist = recording["wrist"]
        labels = np.asarray(recording["label"]).flatten()
        bvp = np.asarray(wrist["BVP"]).flatten()
        eda = np.asarray(wrist["EDA"]).flatten()
        temp = np.asarray(wrist["TEMP"]).flatten()
        acc_magnitude = np.sqrt(
            np.sum(np.asarray(wrist["ACC"], dtype=float) ** 2, axis=1)
        )

        peaks = np.asarray(
            nk.ppg_findpeaks(
                nk.ppg_clean(bvp, sampling_rate=cast(Any, FS.WRIST_BVP)),
                sampling_rate=cast(Any, FS.WRIST_BVP),
            )["PPG_Peaks"]
        )
        beat_hr = 60.0 / (np.diff(peaks) / FS.WRIST_BVP)
        beat_times = peaks[1:] / FS.WRIST_BVP

        duration = len(labels) / FS.CHEST
        window_start = 0.0
        while window_start + WINDOW_SEC <= duration:
            window_end = window_start + WINDOW_SEC
            lab = _window_label(
                labels[int(window_start * FS.CHEST) : int(window_end * FS.CHEST)],
                {1, 2},
            )
            if lab is not None:
                eda_start = int(window_start * FS.WRIST_EDA)
                eda_end = int(window_end * FS.WRIST_EDA)
                temp_start = int(window_start * FS.WRIST_TEMP)
                temp_end = int(window_end * FS.WRIST_TEMP)
                acc_start = int(window_start * FS.WRIST_ACC)
                acc_end = int(window_end * FS.WRIST_ACC)
                hr_win = beat_hr[
                    (beat_times >= window_start) & (beat_times < window_end)
                ]
                row: Dict[str, Any] = portable_features(
                    eda[eda_start:eda_end],
                    temp[temp_start:temp_end],
                    acc_magnitude[acc_start:acc_end],
                    hr_win,
                    eda_fs=FS.WRIST_EDA,
                    temp_fs=FS.WRIST_TEMP,
                )
                row["subject"] = subject_id
                row["label"] = 0 if lab == 1 else 1
                rows.append(row)
            window_start += step

    return pd.DataFrame(
        rows, columns=cast(Any, [*PORTABLE_FEATURE_COLUMNS, "subject", "label"])
    )
