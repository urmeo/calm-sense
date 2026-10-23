from typing import cast, Dict, List, Optional

import numpy as np
from scipy import stats

from ..logging_config import LoggerMixin
from ..preprocessing.filters import _positive_number


class EDAFeatureExtractor(LoggerMixin):
    def __init__(self, sampling_rate: float = 4.0):
        self.sampling_rate = _positive_number(sampling_rate, "sampling_rate")

    def _validate_signal(self, signal: Optional[np.ndarray]) -> Optional[np.ndarray]:
        if signal is None:
            return None

        signal = np.asarray(signal, dtype=float).flatten()
        signal = signal[np.isfinite(signal)]

        if len(signal) < 10:
            self.logger.warning("EDA signal too short")
            return None

        return signal

    def _empty_features(self, prefix: str) -> Dict[str, float]:
        return dict.fromkeys(
            (key for key in self.get_feature_descriptions() if key.startswith(prefix)),
            np.nan,
        )

    def extract_tonic_features(self, scl: Optional[np.ndarray]) -> Dict[str, float]:
        features = self._empty_features("SCL_")

        original = np.asarray(scl, dtype=float).flatten()
        validated = self._validate_signal(scl)
        if validated is None:
            return features
        scl = validated

        try:
            features["SCL_mean"] = float(np.mean(scl))
            features["SCL_std"] = float(np.std(scl))
            features["SCL_min"] = float(np.min(scl))
            features["SCL_max"] = float(np.max(scl))

            x = np.arange(len(original))[np.isfinite(original)] / self.sampling_rate
            slope, _, _, _, _ = stats.linregress(x, scl)
            features["SCL_slope"] = float(cast(float, slope))

        except Exception as e:
            self.logger.warning(f"Tonic feature extraction failed: {e}")

        return features

    def extract_phasic_features(
        self, scr_peaks: Optional[List[Dict]], signal_duration: float
    ) -> Dict[str, float]:
        features = self._empty_features("SCR_")

        if scr_peaks is None or len(scr_peaks) == 0:
            return dict.fromkeys(features, 0.0)

        try:
            n_scr = len(scr_peaks)
            features["SCR_count"] = float(n_scr)

            duration_min = signal_duration / 60.0
            features["SCR_rate"] = (
                float(n_scr / duration_min) if duration_min > 0 else 0.0
            )

            amplitudes = [scr["amplitude"] for scr in scr_peaks if "amplitude" in scr]
            if len(amplitudes) > 0:
                features["SCR_amplitude_mean"] = float(np.mean(amplitudes))
                features["SCR_amplitude_max"] = float(np.max(amplitudes))

            rise_times = [scr["rise_time"] for scr in scr_peaks if "rise_time" in scr]
            if len(rise_times) > 0:
                features["SCR_rise_time_mean"] = float(np.mean(rise_times))

            recovery_times = [
                scr["recovery_time"]
                for scr in scr_peaks
                if scr.get("recovery_time") is not None
            ]
            if len(recovery_times) > 0:
                features["SCR_recovery_time_mean"] = float(np.mean(recovery_times))

            auc_total = 0.0
            for scr in scr_peaks:
                amp = scr.get("amplitude", 0)
                rise = scr.get("rise_time", 0)
                recovery = scr.get("recovery_time")
                if recovery is None:
                    recovery = rise * 2
                auc_total += 0.5 * amp * (rise + recovery)
            features["SCR_AUC"] = float(auc_total)

        except Exception as e:
            self.logger.warning(f"Phasic feature extraction failed: {e}")

        return features

    def extract_statistical_features(
        self, eda: Optional[np.ndarray]
    ) -> Dict[str, float]:
        features = self._empty_features("EDA_")

        validated = self._validate_signal(eda)
        if validated is None:
            return features
        eda = validated

        try:
            features["EDA_mean"] = float(np.mean(eda))
            features["EDA_range"] = float(np.ptp(eda))
            if np.std(eda) > 0:
                features["EDA_kurtosis"] = float(stats.kurtosis(eda))
        except Exception as e:
            self.logger.warning(f"Statistical feature extraction failed: {e}")

        return features

    def extract_all(
        self,
        eda_decomposed: Dict,
        scr_peaks: Optional[List[Dict]] = None,
        raw_eda: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        tonic = eda_decomposed.get("tonic") if eda_decomposed else None
        features = self.extract_tonic_features(tonic)
        duration_source = tonic if tonic is not None else raw_eda
        signal_duration = (
            len(duration_source) / self.sampling_rate
            if duration_source is not None
            else 60.0
        )
        features.update(self.extract_phasic_features(scr_peaks, signal_duration))

        if raw_eda is None and eda_decomposed:
            tonic = eda_decomposed.get("tonic")
            phasic = eda_decomposed.get("phasic")
            if tonic is not None and phasic is not None:
                raw_eda = np.asarray(tonic, dtype=float) + np.asarray(
                    phasic, dtype=float
                )

        features.update(self.extract_statistical_features(raw_eda))

        return features

    def get_feature_descriptions(self) -> Dict[str, str]:
        return {
            "SCL_mean": "Mean skin conductance level (µS)",
            "SCL_std": "Standard deviation of SCL (µS)",
            "SCL_slope": "Linear trend of SCL (µS/s)",
            "SCL_min": "Minimum SCL (µS)",
            "SCL_max": "Maximum SCL (µS)",
            "SCR_count": "Number of SCR events",
            "SCR_rate": "SCR events per minute",
            "SCR_amplitude_mean": "Mean SCR amplitude (µS)",
            "SCR_amplitude_max": "Maximum SCR amplitude (µS)",
            "SCR_rise_time_mean": "Mean SCR rise time (s)",
            "SCR_recovery_time_mean": "Mean SCR half-recovery time (s)",
            "SCR_AUC": "Triangular SCR area estimate (µS·s)",
            "EDA_mean": "Overall EDA mean (µS)",
            "EDA_range": "EDA dynamic range (µS)",
            "EDA_kurtosis": "EDA distribution kurtosis",
        }
