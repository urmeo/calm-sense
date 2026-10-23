from typing import Dict, Optional

import numpy as np

from ..config import FEATURE_PARAMS
from ..logging_config import LoggerMixin
from ..preprocessing.filters import _positive_number


class RespirationFeatureExtractor(LoggerMixin):
    def __init__(self, sampling_rate: float = 700.0):
        self.sampling_rate = _positive_number(sampling_rate, "sampling_rate")

    def _validate_signal(self, signal: np.ndarray) -> Optional[np.ndarray]:
        if signal is None:
            return None

        signal = np.asarray(signal, dtype=float).flatten()

        if len(signal) < 100 or not np.isfinite(signal).all():
            self.logger.warning("Respiratory signal too short")
            return None

        return signal

    def _breath_indices(self, indices, length):
        indices = np.asarray(indices, dtype=float).flatten()
        return np.unique(
            indices[np.isfinite(indices) & (indices >= 0) & (indices < length)]
        )

    def extract_all(
        self,
        resp: np.ndarray,
        breath_peaks: Optional[np.ndarray] = None,
        breath_troughs: Optional[np.ndarray] = None,
        breath_intervals: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        features = dict.fromkeys(self.get_feature_descriptions(), np.nan)

        validated = self._validate_signal(resp)
        if validated is None:
            self.logger.warning("Invalid respiratory signal, returning NaN features")
            return features
        resp = validated

        try:
            if breath_intervals is not None and len(breath_intervals) > 0:
                breath_intervals = np.asarray(breath_intervals, dtype=float).flatten()
                breath_intervals = breath_intervals[
                    np.isfinite(breath_intervals) & (breath_intervals > 0)
                ]

                if len(breath_intervals) > 0:
                    mean_interval = np.mean(breath_intervals)
                    if mean_interval > 0:
                        features["RESP_rate"] = float(60.0 / mean_interval)

                    std_interval = np.std(breath_intervals)
                    if mean_interval > FEATURE_PARAMS.EPSILON:
                        features["RESP_variability"] = float(
                            std_interval / mean_interval
                        )

            if np.isnan(features["RESP_rate"]):
                features["RESP_rate"] = self._estimate_breathing_rate(resp)

            if breath_peaks is not None and breath_troughs is not None:
                breath_peaks = self._breath_indices(breath_peaks, len(resp))
                breath_troughs = self._breath_indices(breath_troughs, len(resp))

                amplitudes = []
                for peak_idx in breath_peaks:
                    pi = int(peak_idx)
                    troughs_before = breath_troughs[breath_troughs < peak_idx]
                    troughs_after = breath_troughs[breath_troughs > peak_idx]
                    if len(troughs_before) > 0 and len(troughs_after) > 0:
                        tb, ta = int(troughs_before[-1]), int(troughs_after[0])
                        amplitudes.append(resp[pi] - 0.5 * (resp[tb] + resp[ta]))

                if len(amplitudes) > 0:
                    features["RESP_amplitude"] = float(np.mean(amplitudes))

            if np.isnan(features["RESP_amplitude"]):
                features["RESP_amplitude"] = float(np.std(resp))

            if breath_peaks is not None and breath_troughs is not None:
                features["RESP_inhale_exhale_ratio"] = self._compute_ie_ratio(
                    breath_peaks, breath_troughs
                )

            features["RESP_apnea_index"] = self._compute_apnea_index(
                resp, breath_intervals
            )

        except Exception as e:
            self.logger.error(f"Respiration feature extraction failed: {e}")

        return features

    def _estimate_breathing_rate(self, resp: np.ndarray) -> float:
        from scipy import signal as scipy_signal

        if np.std(resp) <= FEATURE_PARAMS.EPSILON:
            return np.nan
        nperseg = min(int(30 * self.sampling_rate), len(resp) // 2)
        if nperseg < 64:
            return np.nan

        freqs, psd = scipy_signal.welch(resp, fs=self.sampling_rate, nperseg=nperseg)

        mask = (freqs >= 0.1) & (freqs <= 0.5)
        if not np.any(mask):
            return np.nan

        resp_freqs = freqs[mask]
        resp_psd = psd[mask]

        return float(resp_freqs[np.argmax(resp_psd)] * 60.0)

    def _compute_ie_ratio(self, peaks: np.ndarray, troughs: np.ndarray) -> float:
        inspiration_times = []
        expiration_times = []

        for peak_idx in peaks:
            troughs_before = troughs[troughs < peak_idx]
            if len(troughs_before) > 0:
                insp_start = troughs_before[-1]
                insp_time = (peak_idx - insp_start) / self.sampling_rate
                inspiration_times.append(insp_time)

            troughs_after = troughs[troughs > peak_idx]
            if len(troughs_after) > 0:
                exp_end = troughs_after[0]
                exp_time = (exp_end - peak_idx) / self.sampling_rate
                expiration_times.append(exp_time)

        if len(inspiration_times) > 0 and len(expiration_times) > 0:
            mean_insp = np.mean(inspiration_times)
            mean_exp = np.mean(expiration_times)
            if mean_exp > FEATURE_PARAMS.EPSILON:
                return float(mean_insp / mean_exp)

        return np.nan

    def _compute_apnea_index(
        self, resp: np.ndarray, breath_intervals: Optional[np.ndarray]
    ) -> float:
        if breath_intervals is None or len(breath_intervals) < 3:
            window_sec = 10
            window_samples = int(window_sec * self.sampling_rate)

            if len(resp) < window_samples:
                return 0.0

            n_windows = len(resp) // window_samples
            threshold = max(FEATURE_PARAMS.EPSILON, 0.1 * np.var(resp))
            windows = resp[: n_windows * window_samples].reshape(
                n_windows, window_samples
            )
            return float(100.0 * np.mean(np.var(windows, axis=1) <= threshold))

        breath_intervals = np.asarray(breath_intervals)
        breath_intervals = breath_intervals[
            np.isfinite(breath_intervals) & (breath_intervals > 0)
        ]

        if len(breath_intervals) == 0:
            return 0.0

        apnea_threshold = 10.0
        n_apnea = np.sum(breath_intervals > apnea_threshold)

        return float(100.0 * n_apnea / len(breath_intervals))

    def get_feature_descriptions(self) -> Dict[str, str]:
        return {
            "RESP_rate": "Breathing rate (breaths/min)",
            "RESP_amplitude": "Mean breath amplitude, or signal standard deviation fallback",
            "RESP_variability": "CV of breath intervals (dimensionless)",
            "RESP_inhale_exhale_ratio": "Inspiration/Expiration time ratio",
            "RESP_apnea_index": "Long-interval or low-variance window proxy (%)",
        }
