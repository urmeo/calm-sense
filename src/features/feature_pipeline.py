import pickle
from pathlib import Path
from typing import Any, cast, Dict, List, Optional, Union

import numpy as np
import pandas as pd

from ..config import FS
from ..logging_config import LoggerMixin
from ..utils import ensure_directory, timer
from .accelerometer_features import AccelerometerFeatureExtractor
from .eda_features import EDAFeatureExtractor
from .hrv_frequency_domain import HRVFrequencyDomainExtractor
from .hrv_nonlinear import HRVNonlinearExtractor
from .hrv_time_domain import HRVTimeDomainExtractor
from .respiration_features import RespirationFeatureExtractor
from .temperature_features import TemperatureFeatureExtractor

FEATURE_SCHEMA_VERSION = 2


class FeatureExtractionPipeline(LoggerMixin):
    """Extract a fixed feature schema from preprocessed sensor windows."""

    _GROUPS = {
        "hrv_time": ("HRV_", "HRV Time-Domain"),
        "hrv_frequency": ("HRV_", "HRV Frequency-Domain"),
        "hrv_nonlinear": ("HRV_", "HRV Nonlinear"),
        "eda": ("EDA_", "EDA"),
        "temperature": ("TEMP_", "Temperature"),
        "respiration": ("RESP_", "Respiration"),
        "accelerometer": ("ACC_", "Accelerometer"),
    }
    DEFAULT_CONFIG = dict.fromkeys(_GROUPS, True)

    def __init__(
        self,
        feature_config: Optional[Dict[str, bool]] = None,
        chest_fs: float = FS.CHEST,
        wrist_eda_fs: float = FS.WRIST_EDA,
        wrist_acc_fs: float = FS.WRIST_ACC,
    ):
        if feature_config:
            unknown = set(feature_config) - set(self.DEFAULT_CONFIG)
            if unknown:
                raise ValueError(
                    f"Unknown feature_config keys: {sorted(unknown)}. "
                    f"Valid keys: {sorted(self.DEFAULT_CONFIG)}"
                )
            if any(not isinstance(value, bool) for value in feature_config.values()):
                raise ValueError("feature_config values must be booleans")
        self.feature_config = {**self.DEFAULT_CONFIG, **(feature_config or {})}

        self.extractors: Dict[str, Any] = {
            "hrv_time": HRVTimeDomainExtractor(),
            "hrv_frequency": HRVFrequencyDomainExtractor(),
            "hrv_nonlinear": HRVNonlinearExtractor(),
            "eda": EDAFeatureExtractor(sampling_rate=wrist_eda_fs),
            "temperature": TemperatureFeatureExtractor(sampling_rate=wrist_eda_fs),
            "respiration": RespirationFeatureExtractor(sampling_rate=chest_fs),
            "accelerometer": AccelerometerFeatureExtractor(sampling_rate=wrist_acc_fs),
        }

        self._feature_names: Optional[List[str]] = None

    def extract_window_features(self, window_data: Dict[str, Any]) -> Dict[str, float]:
        features: Dict[str, float] = {}
        for group, (prefix, _) in self._GROUPS.items():
            if not self.feature_config.get(group, True):
                continue
            result = self._extract_group_features(group, window_data)
            if result is None:
                result = dict.fromkeys(
                    self.extractors[group].get_feature_descriptions(), np.nan
                )
            features.update(
                {
                    name if name.startswith(prefix) else prefix + name: value
                    for name, value in result.items()
                }
            )
        return features

    def _extract_group_features(
        self, group: str, window_data: Dict[str, Any]
    ) -> Optional[Dict[str, float]]:
        extractor = self.extractors[group]
        if group.startswith("hrv_"):
            rr = window_data.get("rr_intervals")
            return None if rr is None else extractor.extract_all(rr)

        if group == "temperature":
            temperature = window_data.get("temperature")
            return None if temperature is None else extractor.extract_all(temperature)

        if group == "eda":
            tonic, raw = window_data.get("eda_tonic"), window_data.get("eda_raw")
            if tonic is None and raw is None:
                return None
            decomposed = {"tonic": tonic, "phasic": window_data.get("eda_phasic")}
            return extractor.extract_all(decomposed, window_data.get("scr_peaks"), raw)

        if group == "respiration":
            respiration = window_data.get("respiration")
            if respiration is None:
                return None
            return extractor.extract_all(
                respiration,
                breath_peaks=window_data.get("breath_peaks"),
                breath_troughs=window_data.get("breath_troughs"),
                breath_intervals=window_data.get("breath_intervals"),
            )

        accelerometer = window_data.get("accelerometer")
        if accelerometer is None:
            return None
        if not isinstance(accelerometer, dict):
            return extractor.extract_from_magnitude(accelerometer)
        if "magnitude" in accelerometer:
            return extractor.extract_from_magnitude(accelerometer["magnitude"])
        empty = np.array([])
        return extractor.extract_all(
            accelerometer.get("x", empty),
            accelerometer.get("y", empty),
            accelerometer.get("z", empty),
        )

    def extract_all_features(
        self,
        processed_data: Union[pd.DataFrame, List[Dict]],
        show_progress: bool = True,
    ) -> pd.DataFrame:
        if isinstance(processed_data, pd.DataFrame):
            windows = processed_data.to_dict("records")
        else:
            windows = processed_data

        n_windows = len(windows)
        self.logger.info(f"Extracting features from {n_windows} windows")

        all_features = []

        with timer("Feature extraction"):
            for i, window_data in enumerate(windows):
                if show_progress and (i + 1) % 100 == 0:
                    self.logger.info(f"Processing window {i + 1}/{n_windows}")

                features = self.extract_window_features(window_data)

                for key in ["subject_id", "window_id", "label"]:
                    if key in window_data:
                        features[key] = window_data[key]

                all_features.append(features)

        if not all_features:
            return pd.DataFrame(
                columns=cast(
                    Any, ["subject_id", "window_id", "label", *self.get_feature_names()]
                )
            )
        features_df = pd.DataFrame(all_features)

        metadata_cols = ["subject_id", "window_id", "label"]
        existing_meta = [c for c in metadata_cols if c in features_df.columns]
        feature_cols = [c for c in features_df.columns if c not in metadata_cols]
        features_df = cast(pd.DataFrame, features_df[existing_meta + feature_cols])

        self.logger.info(
            f"Feature extraction complete: {len(features_df)} windows, {len(feature_cols)} features"
        )

        return features_df

    def get_feature_names(self) -> List[str]:
        if self._feature_names is None:
            self._feature_names = list(self.get_feature_descriptions())
        return self._feature_names.copy()

    def get_feature_count(self) -> int:
        return len(self.get_feature_names())

    def get_feature_descriptions(self) -> Dict[str, str]:
        descriptions = {}
        for group, (prefix, _) in self._GROUPS.items():
            if not self.feature_config.get(group, True):
                continue
            for name, description in (
                self.extractors[group].get_feature_descriptions().items()
            ):
                full_name = name if name.startswith(prefix) else prefix + name
                descriptions[full_name] = description
        return descriptions

    def save_features(
        self,
        features_df: pd.DataFrame,
        output_path: Union[str, Path],
        file_format: str = "csv",
    ) -> Path:
        output_path = Path(output_path)
        ensure_directory(output_path.parent)

        if file_format == "csv":
            features_df.to_csv(output_path, index=False)
        elif file_format == "parquet":
            features_df.to_parquet(output_path, index=False)
        elif file_format == "pickle":
            with open(output_path, "wb") as f:
                pickle.dump(features_df, f, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            raise ValueError(f"Unknown format: {file_format}")

        self.logger.info(f"Features saved to {output_path}")
        return output_path

    def load_features(
        self, input_path: Union[str, Path], file_format: Optional[str] = None
    ) -> pd.DataFrame:
        input_path = Path(input_path)

        if file_format is None:
            file_format = input_path.suffix.lstrip(".")

        if file_format == "csv":
            return pd.read_csv(input_path)
        if file_format == "parquet":
            return pd.read_parquet(input_path)
        if file_format in ("pickle", "pkl"):
            with open(input_path, "rb") as f:
                return pickle.load(f)
        raise ValueError(f"Unknown format: {file_format}")

    def get_feature_groups(self) -> Dict[str, List[str]]:
        enabled = set(self.get_feature_names())
        groups = {}
        for group, (prefix, label) in self._GROUPS.items():
            names = [
                name if name.startswith(prefix) else prefix + name
                for name in self.extractors[group].get_feature_descriptions()
            ]
            groups[label] = [name for name in names if name in enabled]
        return groups
