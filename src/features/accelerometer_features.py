from typing import Dict, Optional

import numpy as np
from scipy import signal as scipy_signal

from ..logging_config import LoggerMixin
from ..preprocessing.filters import _positive_number


class AccelerometerFeatureExtractor(LoggerMixin):
    def __init__(self, sampling_rate: float = 32.0):
        self.sampling_rate = _positive_number(sampling_rate, "sampling_rate")

    def _validate_signal(self, signal: np.ndarray) -> Optional[np.ndarray]:
        if signal is None:
            return None

        signal = np.asarray(signal, dtype=float).flatten()
        signal = signal[np.isfinite(signal)]

        if len(signal) < 10:
            self.logger.warning("Accelerometer signal too short")
            return None

        return signal

    def compute_magnitude(
        self, acc_x: np.ndarray, acc_y: np.ndarray, acc_z: np.ndarray
    ) -> np.ndarray:
        acc_x, acc_y, acc_z = (
            np.asarray(axis, dtype=float) for axis in (acc_x, acc_y, acc_z)
        )
        return np.sqrt(acc_x**2 + acc_y**2 + acc_z**2)

    def extract_all(
        self, acc_x: np.ndarray, acc_y: np.ndarray, acc_z: np.ndarray
    ) -> Dict[str, float]:
        if acc_x is None or acc_y is None or acc_z is None:
            self.logger.warning("Invalid accelerometer data")
            return self.extract_from_magnitude(np.array([]))

        ax = np.asarray(acc_x).flatten()
        ay = np.asarray(acc_y).flatten()
        az = np.asarray(acc_z).flatten()
        if not ax.size == ay.size == az.size:
            raise ValueError("Accelerometer axes must have equal lengths")
        return self.extract_from_magnitude(self.compute_magnitude(ax, ay, az))

    def extract_from_magnitude(self, magnitude: np.ndarray) -> Dict[str, float]:
        features = dict.fromkeys(self.get_feature_descriptions(), np.nan)

        original = np.asarray(magnitude, dtype=float).flatten()
        validated = self._validate_signal(magnitude)
        if validated is None:
            return features
        magnitude = validated

        try:
            features["ACC_magnitude"] = float(np.mean(magnitude))
            features["ACC_std"] = float(np.std(magnitude))

            mag_centered = original - np.mean(magnitude)
            pairs_finite = np.isfinite(original[:-1]) & np.isfinite(original[1:])
            zero_crossings = np.count_nonzero(
                np.diff(np.signbit(mag_centered)) & pairs_finite
            )
            duration = len(original) / self.sampling_rate
            features["ACC_zero_crossings"] = float(zero_crossings / duration)
            features["ACC_energy"] = float(np.sum(magnitude**2) / len(magnitude))

            if (
                len(magnitude) >= 64
                and np.isfinite(original).all()
                and np.std(magnitude) > 1e-10
            ):
                nperseg = int(min(len(magnitude), max(256, self.sampling_rate * 4)))
                freqs, psd = scipy_signal.welch(
                    mag_centered,
                    fs=self.sampling_rate,
                    nperseg=nperseg,
                )
                mask = (freqs >= 0.1) & (freqs <= 10.0)
                if np.any(mask):
                    mov_freqs = freqs[mask]
                    mov_psd = psd[mask]
                    features["ACC_peak_freq"] = float(mov_freqs[np.argmax(mov_psd)])

        except Exception as e:
            self.logger.error(f"Accelerometer feature extraction failed: {e}")

        return features

    def get_feature_descriptions(self) -> Dict[str, str]:
        return {
            "ACC_magnitude": "Mean vector magnitude (input units)",
            "ACC_std": "Magnitude standard deviation",
            "ACC_zero_crossings": "Zero-crossing rate (Hz)",
            "ACC_energy": "Mean squared magnitude (energy)",
            "ACC_peak_freq": "Dominant frequency in movement band (Hz)",
        }
