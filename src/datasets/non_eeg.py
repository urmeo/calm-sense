"""Build Relax versus cognitive or emotional stress windows from Non-EEG records."""

from pathlib import Path
from typing import Any, cast, Optional

import numpy as np
import pandas as pd

from ..config import EXTERNAL_DATA_DIR
from ..portable import OVERLAP, PORTABLE_FEATURE_COLUMNS, WINDOW_SEC, portable_features
from .records import read_annotations, read_record

DATA_DIR = (
    EXTERNAL_DATA_DIR
    / "noneeg"
    / "non-eeg-dataset-for-assessment-of-neurological-status-1.0.0"
)
ACC_FS = 8
EDA_FS = 8
TEMP_FS = 8
HR_FS = 1
STRESS = {"CognitiveStress", "EmotionalStress"}
RELAX = {"Relax"}


def _segments(record: str):
    ann = read_annotations(record, "atr")
    if (
        len(ann.sample) != len(ann.aux_note)
        or np.any(np.asarray(ann.sample) < 0)
        or np.any(np.diff(ann.sample) <= 0)
    ):
        raise ValueError(
            "Non-EEG annotations require aligned notes and increasing sample indices"
        )
    bounds = list(ann.sample) + [None]
    for i, note in enumerate(ann.aux_note):
        yield int(ann.sample[i]), bounds[i + 1], note.strip("\x00 \t\r\n")


def build(subjects: Optional[list] = None) -> pd.DataFrame:
    valid_subjects = [f"Subject{i}" for i in range(1, 21)]
    subjects = valid_subjects if subjects is None else subjects
    if (
        not subjects
        or len(set(subjects)) != len(subjects)
        or not set(subjects) <= set(valid_subjects)
    ):
        raise ValueError(
            "subjects must contain distinct Non-EEG IDs from Subject1 to Subject20"
        )
    win = int(WINDOW_SEC * ACC_FS)
    step = int(win * (1 - OVERLAP))
    rows = []

    for sid in subjects:
        acc_rec = str(DATA_DIR / f"{sid}_AccTempEDA")
        hr_rec = str(DATA_DIR / f"{sid}_SpO2HR")
        if not Path(acc_rec + ".hea").exists():
            continue
        sensor_record = read_record(acc_rec)
        hr_record = read_record(hr_rec)
        if sensor_record.fs != ACC_FS or hr_record.fs != HR_FS:
            raise ValueError(f"Unexpected Non-EEG sampling rates for {sid}")
        if (
            sensor_record.p_signal.ndim != 2
            or sensor_record.p_signal.shape[1] != 5
            or hr_record.p_signal.ndim != 2
            or hr_record.p_signal.shape[1] < 2
        ):
            raise ValueError(f"Unexpected Non-EEG channel shapes for {sid}")
        sig = sensor_record.p_signal
        hr = hr_record.p_signal[:, 1]
        acc_mag = np.sqrt(np.sum(sig[:, 0:3] ** 2, axis=1))
        temp, eda = sig[:, 3], sig[:, 4]

        for s0, s1, note in _segments(acc_rec):
            s1 = s1 if s1 is not None else len(eda)
            if not 0 <= s0 < s1 <= len(eda):
                raise ValueError(
                    f"Non-EEG annotation interval outside the sensor record for {sid}"
                )
            if note in STRESS:
                label = 1
            elif note in RELAX:
                label = 0
            else:
                continue
            for w0 in range(s0, s1 - win + 1, step):
                w1 = w0 + win
                h0 = (w0 * HR_FS + ACC_FS - 1) // ACC_FS
                h1 = (w1 * HR_FS + ACC_FS - 1) // ACC_FS
                hr_win = hr[h0:h1]
                if len(hr_win) != h1 - h0:
                    raise ValueError(
                        f"Non-EEG HR record is shorter than the sensor window for {sid}"
                    )
                row: dict[str, Any] = portable_features(
                    eda[w0:w1],
                    temp[w0:w1],
                    acc_mag[w0:w1],
                    hr_win,
                    eda_fs=EDA_FS,
                    temp_fs=TEMP_FS,
                )
                row["subject"] = sid
                row["label"] = label
                rows.append(row)

    return pd.DataFrame(
        rows, columns=cast(Any, [*PORTABLE_FEATURE_COLUMNS, "subject", "label"])
    )
