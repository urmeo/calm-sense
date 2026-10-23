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
    window_samples = int(WINDOW_SEC * ACC_FS)
    step_samples = int(window_samples * (1 - OVERLAP))
    rows = []

    for subject_id in subjects:
        sensor_path = str(DATA_DIR / f"{subject_id}_AccTempEDA")
        hr_path = str(DATA_DIR / f"{subject_id}_SpO2HR")
        if not Path(sensor_path + ".hea").exists():
            continue
        sensor_record = read_record(sensor_path)
        hr_record = read_record(hr_path)
        if sensor_record.fs != ACC_FS or hr_record.fs != HR_FS:
            raise ValueError(f"Unexpected Non-EEG sampling rates for {subject_id}")
        if (
            sensor_record.p_signal.ndim != 2
            or sensor_record.p_signal.shape[1] != 5
            or hr_record.p_signal.ndim != 2
            or hr_record.p_signal.shape[1] < 2
        ):
            raise ValueError(f"Unexpected Non-EEG channel shapes for {subject_id}")
        signals = sensor_record.p_signal
        heart_rate = hr_record.p_signal[:, 1]
        acc_mag = np.sqrt(np.sum(signals[:, 0:3] ** 2, axis=1))
        temperature, eda = signals[:, 3], signals[:, 4]

        for segment_start, segment_end, note in _segments(sensor_path):
            segment_end = segment_end if segment_end is not None else len(eda)
            if not 0 <= segment_start < segment_end <= len(eda):
                raise ValueError(
                    f"Non-EEG annotation interval outside the sensor record for {subject_id}"
                )
            if note in STRESS:
                label = 1
            elif note in RELAX:
                label = 0
            else:
                continue
            for window_start in range(
                segment_start, segment_end - window_samples + 1, step_samples
            ):
                window_end = window_start + window_samples
                hr_start = (window_start * HR_FS + ACC_FS - 1) // ACC_FS
                hr_end = (window_end * HR_FS + ACC_FS - 1) // ACC_FS
                hr_window = heart_rate[hr_start:hr_end]
                if len(hr_window) != hr_end - hr_start:
                    raise ValueError(
                        f"Non-EEG HR record is shorter than the sensor window for {subject_id}"
                    )
                row: dict[str, Any] = portable_features(
                    eda[window_start:window_end],
                    temperature[window_start:window_end],
                    acc_mag[window_start:window_end],
                    hr_window,
                    eda_fs=EDA_FS,
                    temp_fs=TEMP_FS,
                )
                row["subject"] = subject_id
                row["label"] = label
                rows.append(row)

    return pd.DataFrame(
        rows, columns=cast(Any, [*PORTABLE_FEATURE_COLUMNS, "subject", "label"])
    )
