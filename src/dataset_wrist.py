from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, cast, List, Optional, Tuple

import numpy as np
import pandas as pd

from .config import FS, PROCESSED_DATA_DIR, RR_CORRECTION_VERSION, VALID_SUBJECTS
from .data.loader import WESADLoader
from .dataset import CONDITION_LABELS, window_label
from .features.feature_pipeline import FEATURE_SCHEMA_VERSION, FeatureExtractionPipeline
from .logging_config import LoggerMixin
from .preprocessing.ecg_processor import ECGProcessor
from .preprocessing.eda_processor import EDAProcessor
from .preprocessing.filters import _window_parameters


class WristDataset(LoggerMixin):
    """Build features from WESAD wrist windows using the chest label timeline."""

    def __init__(
        self, window_sec: float = 60.0, overlap: float = 0.5, purity: float = 0.9
    ):
        _window_parameters(window_sec, overlap, purity, FS.WRIST_EDA)
        self.window_sec = float(window_sec)
        self.overlap = float(overlap)
        self.purity = float(purity)
        self.label_fs = FS.CHEST
        self.loader = WESADLoader()
        self.eda = EDAProcessor(sampling_rate=FS.WRIST_EDA)
        self.rr = ECGProcessor(sampling_rate=FS.WRIST_BVP)
        self.features = FeatureExtractionPipeline(
            chest_fs=FS.CHEST, wrist_eda_fs=FS.WRIST_EDA, wrist_acc_fs=FS.WRIST_ACC
        )

    def _window_label(self, labels: np.ndarray) -> Optional[int]:
        return window_label(labels, self.purity)

    def _bvp_peaks(self, bvp: np.ndarray) -> np.ndarray:
        import neurokit2 as nk

        clean = nk.ppg_clean(bvp, sampling_rate=int(FS.WRIST_BVP))
        info = nk.ppg_findpeaks(clean, sampling_rate=int(FS.WRIST_BVP))
        return np.asarray(info["PPG_Peaks"])

    def _process_subject(self, subject_id: str) -> Tuple[List[dict], List[int]]:
        data = self.loader.load_subject(subject_id)
        wrist = data["wrist"]
        labels = np.asarray(data["label"]).flatten()

        bvp = np.asarray(wrist["BVP"]).flatten()
        eda = np.asarray(wrist["EDA"]).flatten()
        temp = np.asarray(wrist["TEMP"]).flatten()
        acc = np.asarray(wrist["ACC"])
        acc_mag = np.sqrt(np.sum(acc**2, axis=1))

        peaks = self._bvp_peaks(bvp)
        eda_filt = self.eda.lowpass_filter(eda)
        tonic, phasic = self.eda.decompose_eda(eda_filt)
        _, scr_features = self.eda.detect_scr_peaks(phasic)
        scr_peak_indices = np.array([scr["peak_idx"] for scr in scr_features])

        step = self.window_sec * (1 - self.overlap)
        duration = len(labels) / self.label_fs
        windows, window_labels = [], []
        window_start = 0.0
        while window_start + self.window_sec <= duration:
            window_end = window_start + self.window_sec
            label = self._window_label(
                labels[
                    int(window_start * self.label_fs) : int(window_end * self.label_fs)
                ]
            )
            if label is not None:
                bvp_start, bvp_end = int(window_start * FS.WRIST_BVP), int(
                    window_end * FS.WRIST_BVP
                )
                in_window = (peaks >= bvp_start) & (peaks < bvp_end)
                rr = self.rr.extract_rr_intervals(peaks[in_window], unit="ms")
                _, valid = self.rr.remove_ectopic_beats(rr)
                rr_clean = self.rr.interpolate_artifacts(rr, valid)

                eda_start, eda_end = int(window_start * FS.WRIST_EDA), int(
                    window_end * FS.WRIST_EDA
                )
                acc_start, acc_end = int(window_start * FS.WRIST_ACC), int(
                    window_end * FS.WRIST_ACC
                )
                window_scr = [
                    scr
                    for scr, peak_index in zip(scr_features, scr_peak_indices)
                    if eda_start <= peak_index < eda_end
                ]

                windows.append(
                    {
                        "rr_intervals": rr_clean,
                        "eda_tonic": tonic[eda_start:eda_end],
                        "eda_phasic": phasic[eda_start:eda_end],
                        "eda_raw": eda_filt[eda_start:eda_end],
                        "scr_peaks": window_scr,
                        "temperature": temp[eda_start:eda_end],
                        "accelerometer": {"magnitude": acc_mag[acc_start:acc_end]},
                        "subject_id": subject_id,
                        "window_id": int(window_start * self.label_fs),
                        "label": label,
                    }
                )
                window_labels.append(label)
            window_start += step

        self.logger.info(f"{subject_id} (wrist): {len(window_labels)} windows")
        return windows, window_labels

    def build(
        self, subjects: Optional[List[str]] = None, cache: bool = True
    ) -> pd.DataFrame:
        subjects = self.loader.subjects if subjects is None else subjects
        if not subjects or len(set(subjects)) != len(subjects):
            raise ValueError("subjects must be a nonempty list without duplicates")
        frames = []
        for subject_id in subjects:
            windows, _ = self._process_subject(subject_id)
            if windows:
                frames.append(
                    self.features.extract_all_features(windows, show_progress=False)
                )
            del windows
        features_df = (
            pd.concat(frames, ignore_index=True)
            if frames
            else self.features.extract_all_features([], show_progress=False)
        )
        features_df["label_name"] = features_df["label"].map(
            cast(Any, CONDITION_LABELS)
        )
        features_df.attrs["feature_schema_version"] = FEATURE_SCHEMA_VERSION
        features_df.attrs["rr_correction_version"] = RR_CORRECTION_VERSION
        features_df.attrs["dataset_parameters"] = {
            "window_sec": self.window_sec,
            "overlap": self.overlap,
            "purity": self.purity,
            "subjects": sorted(subjects),
        }
        if cache:
            PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
            with TemporaryDirectory(dir=PROCESSED_DATA_DIR) as directory:
                staged = Path(directory) / "features_wrist.parquet"
                features_df.to_parquet(staged, index=False)
                staged.replace(PROCESSED_DATA_DIR / staged.name)
            self.logger.info(f"Cached wrist features ({len(features_df)} windows)")
        return features_df


def load_wrist() -> Optional[pd.DataFrame]:
    path = Path(PROCESSED_DATA_DIR) / "features_wrist.parquet"
    if not path.exists():
        return None
    frame = pd.read_parquet(path)
    parameters = {
        "window_sec": 60.0,
        "overlap": 0.5,
        "purity": 0.9,
        "subjects": sorted(VALID_SUBJECTS),
    }
    if (
        frame.attrs.get("feature_schema_version") != FEATURE_SCHEMA_VERSION
        or frame.attrs.get("rr_correction_version") != RR_CORRECTION_VERSION
        or frame.attrs.get("dataset_parameters") != parameters
    ):
        return None
    columns = [
        column
        for column in frame
        if column not in {"subject_id", "window_id", "label", "label_name"}
    ]
    if columns != FeatureExtractionPipeline().get_feature_names() or not {
        "subject_id",
        "window_id",
        "label",
    } <= set(frame.columns):
        raise ValueError("Cached wrist feature columns do not match the current schema")
    return frame
