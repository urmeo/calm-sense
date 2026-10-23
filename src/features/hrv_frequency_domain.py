from typing import Dict, Optional, Tuple

import numpy as np
from scipy import signal as scipy_signal
from scipy.interpolate import PchipInterpolator

from ..config import RR_MAX_MS, RR_MIN_MS
from ..preprocessing.filters import _positive_integer, _positive_number
from .hrv_base import BaseHRVExtractor

_trapz = getattr(np, "trapezoid", None) or getattr(np, "trapz")


class HRVFrequencyDomainExtractor(BaseHRVExtractor):
    VLF_BAND = (0.0033, 0.04)
    LF_BAND = (0.04, 0.15)
    HF_BAND = (0.15, 0.40)

    def __init__(
        self,
        vlf_band: Tuple[float, float] = VLF_BAND,
        lf_band: Tuple[float, float] = LF_BAND,
        hf_band: Tuple[float, float] = HF_BAND,
        interpolation_rate: float = 4.0,
        min_rr_count: int = 30,
    ):
        self.vlf_band = vlf_band
        self.lf_band = lf_band
        self.hf_band = hf_band
        self.interpolation_rate = _positive_number(
            interpolation_rate, "interpolation_rate"
        )
        self.min_rr_count = _positive_integer(min_rr_count, "min_rr_count")
        for band in (vlf_band, lf_band, hf_band):
            if (
                len(band) != 2
                or not 0 <= band[0] < band[1] < self.interpolation_rate / 2
            ):
                raise ValueError(
                    "HRV frequency bands must be ordered below the interpolation Nyquist"
                )

    def _interpolate_rr(
        self, rr: np.ndarray, method: str = "cubic"
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Resample RR intervals without overshoot or extrapolation."""
        if method not in {"linear", "cubic"}:
            raise ValueError("RR interpolation method must be 'linear' or 'cubic'")
        rr = np.asarray(rr, dtype=float).flatten()
        if (
            not np.isfinite(rr).all()
            or np.any(rr < RR_MIN_MS)
            or np.any(rr > RR_MAX_MS)
        ):
            raise ValueError(
                f"RR interpolation requires finite intervals in [{RR_MIN_MS}, {RR_MAX_MS}] ms"
            )
        if len(rr) < 2:
            return None, None

        t_rr = np.r_[0.0, np.cumsum(rr[:-1]) / 1000.0]
        if t_rr[-1] * self.interpolation_rate < 10:
            return None, None

        t_uniform = np.arange(0.0, t_rr[-1], 1.0 / self.interpolation_rate)
        if method == "linear":
            rr_interpolated = np.interp(t_uniform, t_rr, rr)
        else:
            interpolator = PchipInterpolator(t_rr, rr, extrapolate=False)
            rr_interpolated = interpolator(t_uniform)
            rr_interpolated = np.clip(rr_interpolated, rr.min(), rr.max())

        if (
            not np.isfinite(rr_interpolated).all()
            or np.any(rr_interpolated < RR_MIN_MS)
            or np.any(rr_interpolated > RR_MAX_MS)
        ):
            raise ValueError("Resampled RR intervals exceed physiological bounds")
        return t_uniform, rr_interpolated

    def compute_psd(
        self, rr_intervals: np.ndarray, method: str = "welch"
    ) -> Tuple[np.ndarray, np.ndarray]:
        if method not in {"welch", "lomb"}:
            raise ValueError("PSD method must be 'welch' or 'lomb'")
        rr = self._validate_input(rr_intervals)
        if rr is None:
            return np.array([]), np.array([])

        if method == "lomb":
            return self._compute_psd_lomb(rr)
        return self._compute_psd_welch(rr)

    def _compute_psd_welch(self, rr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        _, rr_interp = self._interpolate_rr(rr)
        if rr_interp is None:
            return np.array([]), np.array([])

        rr_detrend = rr_interp - np.mean(rr_interp)

        nperseg = min(256, len(rr_detrend) // 2)
        if nperseg < 16:
            return np.array([]), np.array([])

        freqs, psd = scipy_signal.welch(
            rr_detrend,
            fs=self.interpolation_rate,
            nperseg=nperseg,
            noverlap=nperseg // 2,
            window="hann",
            scaling="density",
        )

        return freqs, np.maximum(psd, 0)

    def _compute_psd_lomb(self, rr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        from scipy.signal import lombscargle

        t = np.r_[0.0, np.cumsum(rr[:-1]) / 1000.0]

        rr_centered = rr - np.mean(rr)

        f_max = max(0.5, self.vlf_band[1], self.lf_band[1], self.hf_band[1])
        n_freqs = 512
        freqs = np.linspace(0.001, f_max, n_freqs)

        variance = float(np.var(rr_centered))
        if variance == 0:
            return freqs, np.zeros_like(freqs)

        angular_freqs = 2 * np.pi * freqs

        try:
            psd = np.maximum(
                lombscargle(t, rr_centered, angular_freqs, normalize=False), 0.0
            )
            area = float(_trapz(psd, freqs))
            if not np.isfinite(area) or area <= 0:
                return np.array([]), np.array([])
            psd *= variance / area
        except Exception as e:
            self.logger.warning(f"Lomb-Scargle failed: {e}")
            return np.array([]), np.array([])

        return freqs, psd

    def _compute_band_power(
        self, freqs: np.ndarray, psd: np.ndarray, band: Tuple[float, float]
    ) -> float:
        if len(freqs) == 0 or len(psd) == 0:
            return np.nan

        lo, hi = max(band[0], freqs[0]), min(band[1], freqs[-1])
        if lo >= hi:
            return 0.0
        inside = freqs[(freqs > lo) & (freqs < hi)]
        band_freqs = np.r_[lo, inside, hi]
        return float(_trapz(np.interp(band_freqs, freqs, psd), band_freqs))

    def compute_vlf_power(self, freqs: np.ndarray, psd: np.ndarray) -> float:
        return self._compute_band_power(freqs, psd, self.vlf_band)

    def compute_lf_power(self, freqs: np.ndarray, psd: np.ndarray) -> float:
        return self._compute_band_power(freqs, psd, self.lf_band)

    def compute_hf_power(self, freqs: np.ndarray, psd: np.ndarray) -> float:
        return self._compute_band_power(freqs, psd, self.hf_band)

    def _power_ratio(self, numerator, denominator, *powers):
        if (
            not np.isfinite([numerator, denominator, *powers]).all()
            or denominator < 1e-6
        ):
            return np.nan
        return float(numerator / denominator)

    def compute_lf_peak_freq(self, freqs: np.ndarray, psd: np.ndarray) -> float:
        if len(freqs) == 0 or len(psd) == 0:
            return np.nan

        mask = (freqs >= self.lf_band[0]) & (freqs <= self.lf_band[1])
        if not np.any(mask):
            return np.nan

        lf_freqs = freqs[mask]
        lf_psd = psd[mask]

        if not np.isfinite(lf_psd).all() or np.max(lf_psd) <= 0:
            return np.nan

        peak_idx = np.argmax(lf_psd)
        return float(lf_freqs[peak_idx])

    def extract_all(
        self, rr_intervals: np.ndarray, method: str = "welch"
    ) -> Dict[str, float]:
        features = dict.fromkeys(self.get_feature_descriptions(), np.nan)

        freqs, psd = self.compute_psd(rr_intervals, method=method)

        if len(freqs) == 0:
            self.logger.warning("PSD computation failed, returning NaN features")
            return features

        try:
            features["VLF_power"] = self.compute_vlf_power(freqs, psd)
            features["LF_power"] = self.compute_lf_power(freqs, psd)
            features["HF_power"] = self.compute_hf_power(freqs, psd)
            band_powers = [
                features["VLF_power"],
                features["LF_power"],
                features["HF_power"],
            ]
            finite_powers = [power for power in band_powers if np.isfinite(power)]
            features["Total_power"] = (
                float(sum(finite_powers)) if finite_powers else np.nan
            )

            lf, hf = features["LF_power"], features["HF_power"]
            features["LF_HF_ratio"] = self._power_ratio(lf, hf)
            features["LFn"] = self._power_ratio(100.0 * lf, lf + hf, lf, hf)
            features["HFn"] = self._power_ratio(100.0 * hf, lf + hf, lf, hf)
            features["LF_peak_freq"] = self.compute_lf_peak_freq(freqs, psd)

        except Exception as e:
            self.logger.error(f"Feature extraction failed: {e}")

        return features

    def get_feature_descriptions(self) -> Dict[str, str]:
        return {
            "VLF_power": f"VLF power {self.vlf_band} Hz (ms²)",
            "LF_power": f"LF power {self.lf_band} Hz (ms²)",
            "HF_power": f"HF power {self.hf_band} Hz (ms²)",
            "Total_power": "Total power VLF+LF+HF (ms²)",
            "LF_HF_ratio": "Ratio of LF to HF spectral power",
            "LFn": "Normalized LF power (%)",
            "HFn": "Normalized HF power (%)",
            "LF_peak_freq": "Peak frequency in LF band (Hz)",
        }
