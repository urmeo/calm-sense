from typing import cast, Dict, Optional

import numpy as np
from scipy import stats

from ..logging_config import LoggerMixin
from ..preprocessing.filters import _positive_number


class TemperatureFeatureExtractor(LoggerMixin):
    def __init__(self, sampling_rate: float = 4.0):
        self.sampling_rate = _positive_number(sampling_rate, "sampling_rate")

    def _validate_signal(self, signal: np.ndarray) -> Optional[np.ndarray]:
        if signal is None:
            return None

        signal = np.asarray(signal, dtype=float).flatten()
        signal = signal[np.isfinite(signal)]

        if len(signal) < 10:
            self.logger.warning("Temperature signal too short")
            return None

        mean_temperature = np.mean(signal)
        if mean_temperature < 15 or mean_temperature > 45:
            self.logger.warning(
                f"Temperature out of physiological range: mean={mean_temperature:.1f}"
            )

        return signal

    def extract_all(self, temp: np.ndarray) -> Dict[str, float]:
        features = dict.fromkeys(self.get_feature_descriptions(), np.nan)

        original = np.asarray(temp, dtype=float).flatten()
        validated = self._validate_signal(temp)
        if validated is None:
            self.logger.warning("Invalid temperature signal, returning NaN features")
            return features
        temp = validated

        try:
            features["TEMP_mean"] = float(np.mean(temp))
            features["TEMP_std"] = float(np.std(temp))
            features["TEMP_min"] = float(np.min(temp))
            features["TEMP_max"] = float(np.max(temp))

            x = np.arange(len(original))[np.isfinite(original)] / self.sampling_rate
            slope, _, _, _, _ = stats.linregress(x, temp)
            features["TEMP_slope"] = float(cast(float, slope))

        except Exception as e:
            self.logger.error(f"Temperature feature extraction failed: {e}")

        return features

    def get_feature_descriptions(self) -> Dict[str, str]:
        return {
            "TEMP_mean": "Mean skin temperature (°C)",
            "TEMP_std": "Temperature standard deviation (°C)",
            "TEMP_slope": "Temperature linear trend (°C/s)",
            "TEMP_min": "Minimum temperature (°C)",
            "TEMP_max": "Maximum temperature (°C)",
        }
