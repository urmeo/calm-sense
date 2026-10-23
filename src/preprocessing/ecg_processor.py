from typing import Optional, Tuple

import numpy as np
from scipy import signal
from scipy.interpolate import PchipInterpolator

from ..config import FILTER_PARAMS, FS, RR_MAX_MS, RR_MIN_MS
from ..logging_config import LoggerMixin
from .filters import _positive_integer, _positive_number


class ECGProcessor(LoggerMixin):
    """Detect R-peaks with NeuroKit2, or a simplified Pan-Tompkins fallback."""

    def __init__(self, sampling_rate: float = FS.CHEST):
        self.sampling_rate = _positive_number(sampling_rate, "sampling_rate")

    def bandpass_filter(
        self,
        ecg: np.ndarray,
        low: float = FILTER_PARAMS.ECG_BANDPASS_LOW,
        high: float = FILTER_PARAMS.ECG_BANDPASS_HIGH,
        order: int = FILTER_PARAMS.ECG_FILTER_ORDER,
    ) -> np.ndarray:
        ecg = np.asarray(ecg, dtype=float).flatten()
        order = _positive_integer(order, "order")
        low, high = _positive_number(low, "low"), _positive_number(high, "high")
        if not np.isfinite(ecg).all():
            raise ValueError("ECG filter input must contain only finite samples")
        nyq = 0.5 * self.sampling_rate
        low_norm = low / nyq
        high_norm = high / nyq

        if not 0 < low_norm < high_norm < 1:
            raise ValueError(
                f"ECG cutoffs must satisfy 0 < low < high < Nyquist ({nyq} Hz)"
            )
        if len(ecg) == 0:
            return ecg

        sos = signal.butter(order, [low_norm, high_norm], btype="band", output="sos")
        return signal.sosfiltfilt(sos, ecg)

    def detect_r_peaks(self, ecg: np.ndarray) -> np.ndarray:
        ecg = np.asarray(ecg, dtype=float).flatten()
        if not np.isfinite(ecg).all():
            raise ValueError("R-peak input must contain only finite samples")
        if len(ecg) < 2 or np.std(ecg) == 0:
            return np.array([], dtype=int)
        duration_sec = len(ecg) / self.sampling_rate

        try:
            import neurokit2 as nk

            _, info = nk.ecg_peaks(ecg, sampling_rate=int(self.sampling_rate))
            r_peaks = np.array(info["ECG_R_Peaks"])
        except ImportError:
            self.logger.debug("neurokit2 not available, using Pan-Tompkins")
            r_peaks = self._pan_tompkins(ecg)

        n_peaks = len(r_peaks)
        expected_min = int(0.5 * duration_sec)
        expected_max = int(3.5 * duration_sec)

        if n_peaks < expected_min:
            self.logger.warning(
                f"Suspiciously few R-peaks: {n_peaks} in {duration_sec:.1f}s "
                f"(expected >={expected_min})"
            )
        elif n_peaks > expected_max:
            self.logger.warning(
                f"Suspiciously many R-peaks: {n_peaks} in {duration_sec:.1f}s "
                f"(expected <={expected_max})"
            )

        return r_peaks

    def _pan_tompkins(self, ecg: np.ndarray) -> np.ndarray:
        diff_ecg = np.diff(ecg)
        squared = diff_ecg**2

        window_size = max(1, int(0.150 * self.sampling_rate))
        integrated = np.convolve(
            squared, np.ones(window_size) / window_size, mode="same"
        )

        init_samples = int(2 * self.sampling_rate)
        threshold = 0.5 * np.max(integrated[:init_samples])

        min_rr = max(1, int(0.2 * self.sampling_rate))

        r_peaks = []
        search_start = 0

        while search_start < len(integrated) - min_rr:
            search_window = integrated[
                search_start : search_start + int(self.sampling_rate)
            ]

            if len(search_window) == 0:
                break

            peaks, _ = signal.find_peaks(
                search_window, height=threshold, distance=min_rr
            )

            if len(peaks) > 0:
                peak_idx = search_start + peaks[0]

                refine_window = max(1, int(0.05 * self.sampling_rate))
                start = max(0, peak_idx - refine_window)
                end = min(len(ecg), peak_idx + refine_window)

                refined_peak = start + np.argmax(ecg[start:end])
                r_peaks.append(refined_peak)

                threshold = 0.5 * (threshold + 0.25 * integrated[peak_idx])

                search_start = refined_peak + min_rr
            else:
                threshold *= 0.8
                search_start += max(1, int(0.5 * self.sampling_rate))

        return np.array(r_peaks, dtype=int)

    def extract_rr_intervals(self, r_peaks: np.ndarray, unit: str = "ms") -> np.ndarray:
        if unit not in ("ms", "s", "samples"):
            raise ValueError(f"unit must be 'ms', 's', or 'samples', got {unit!r}")

        r_peaks = np.asarray(r_peaks, dtype=float).flatten()
        if (
            not np.isfinite(r_peaks).all()
            or np.any(r_peaks < 0)
            or np.any(np.diff(r_peaks) <= 0)
        ):
            raise ValueError(
                "R-peak indices must be finite, nonnegative, and strictly increasing"
            )

        if len(r_peaks) < 2:
            self.logger.warning("Less than 2 R-peaks, cannot compute RR intervals")
            return np.array([])

        rr_samples = np.diff(r_peaks)

        if unit == "ms":
            rr_intervals = (rr_samples / self.sampling_rate) * 1000
        elif unit == "s":
            rr_intervals = rr_samples / self.sampling_rate
        else:
            rr_intervals = rr_samples.astype(float)

        return rr_intervals

    def remove_ectopic_beats(
        self, rr_intervals: np.ndarray, threshold: float = 0.2, *, unit: str = "ms"
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Reject intervals outside 200–2500 ms and local median outliers."""
        rr = np.asarray(rr_intervals, dtype=float).flatten()
        threshold = _positive_number(threshold, "threshold")
        lower, upper = self._rr_bounds(unit)
        physiological_mask = np.isfinite(rr) & (rr >= lower) & (rr <= upper)
        valid_mask = physiological_mask.copy()

        if len(rr) < 3:
            return rr[valid_mask], valid_mask

        for i in np.flatnonzero(physiological_mask):
            start = max(0, i - 2)
            end = min(len(rr), i + 3)

            local = rr[start:end]
            local = local[physiological_mask[start:end]]
            local_median = np.median(local)
            relative_diff = abs(rr[i] - local_median) / local_median
            valid_mask[i] = relative_diff <= threshold or np.isclose(
                relative_diff, threshold, rtol=1e-12, atol=0
            )

        return rr[valid_mask], valid_mask

    def interpolate_artifacts(
        self,
        rr_intervals: np.ndarray,
        valid_mask: Optional[np.ndarray] = None,
        method: str = "linear",
        *,
        unit: str = "ms",
    ) -> np.ndarray:
        """Fill RR gaps within the 200–2500 ms bounds, converted to ``unit``.

        Linear interpolation is the default; cubic uses shape-preserving PCHIP.
        End gaps hold the nearest valid interval. No valid intervals yields NaNs.
        """
        if method not in ("linear", "cubic"):
            raise ValueError("method must be 'linear' or 'cubic'")
        lower, upper = self._rr_bounds(unit)
        rr = np.asarray(rr_intervals, dtype=float).flatten()
        physiological_mask = np.isfinite(rr) & (rr >= lower) & (rr <= upper)

        if valid_mask is None:
            valid_mask = physiological_mask
        else:
            valid_mask = np.asarray(valid_mask).flatten()
            if not np.isin(valid_mask, [False, True]).all():
                raise ValueError("valid_mask must contain only boolean values")
            if len(valid_mask) != len(rr):
                raise ValueError(
                    f"Mask length {len(valid_mask)} doesn't match RR length {len(rr)}"
                )
            valid_mask = valid_mask.astype(bool) & physiological_mask

        if np.all(valid_mask):
            return rr
        if not np.any(valid_mask):
            self.logger.warning("All intervals marked as invalid, returning NaN")
            return np.full(len(rr), np.nan)

        x_valid = np.flatnonzero(valid_mask)
        x_all = np.arange(len(rr))
        valid_rr = rr[valid_mask]

        if method == "linear" or len(x_valid) < 3:
            corrected = np.interp(x_all, x_valid, valid_rr)
        else:
            interpolator = PchipInterpolator(x_valid, valid_rr, extrapolate=False)
            corrected = interpolator(np.clip(x_all, x_valid[0], x_valid[-1]))
            corrected = np.clip(corrected, valid_rr.min(), valid_rr.max())

        corrected[: x_valid[0]] = valid_rr[0]
        corrected[x_valid[-1] + 1 :] = valid_rr[-1]
        corrected[valid_mask] = valid_rr
        if not (
            np.isfinite(corrected).all()
            and np.all(corrected >= lower)
            and np.all(corrected <= upper)
        ):
            raise ValueError("Corrected RR intervals exceed physiological bounds")
        return corrected

    def _rr_bounds(self, unit: str) -> Tuple[float, float]:
        if unit == "ms":
            scale = 1.0
        elif unit == "s":
            scale = 0.001
        elif unit == "samples":
            scale = self.sampling_rate / 1000
        else:
            raise ValueError(f"unit must be 'ms', 's', or 'samples', got {unit!r}")
        return RR_MIN_MS * scale, RR_MAX_MS * scale
