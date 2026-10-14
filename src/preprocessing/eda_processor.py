from typing import cast, Dict, List, Optional, Tuple

import numpy as np
from scipy import signal
from scipy.ndimage import median_filter

from ..config import FILTER_PARAMS, FS
from ..logging_config import LoggerMixin
from .filters import _positive_integer, _positive_number


class EDAProcessor(LoggerMixin):
    def __init__(self, sampling_rate: float = FS.WRIST_EDA):
        self.sampling_rate = _positive_number(sampling_rate, "sampling_rate")

    def lowpass_filter(
        self,
        eda: np.ndarray,
        cutoff: Optional[float] = None,
        order: int = FILTER_PARAMS.EDA_FILTER_ORDER,
    ) -> np.ndarray:
        """Default to 1 Hz for 4 Hz wrist EDA and 5 Hz at other sampling rates."""
        eda = np.asarray(eda, dtype=float).flatten()
        if cutoff is None:
            cutoff = (
                FILTER_PARAMS.WRIST_EDA_LOWPASS
                if self.sampling_rate == FS.WRIST_EDA
                else FILTER_PARAMS.EDA_LOWPASS
            )
        cutoff = _positive_number(cutoff, "cutoff")
        order = _positive_integer(order, "order")
        if not np.isfinite(eda).all():
            raise ValueError("EDA filter input must contain only finite samples")

        nyq = 0.5 * self.sampling_rate
        if cutoff >= nyq:
            raise ValueError(
                f"EDA cutoff must be below Nyquist ({nyq} Hz), got {cutoff} Hz"
            )
        if len(eda) == 0:
            return eda

        sos = signal.butter(order, cutoff / nyq, btype="low", output="sos")
        return signal.sosfiltfilt(sos, eda)

    def remove_artifacts(
        self, eda: np.ndarray, median_kernel: int = FILTER_PARAMS.EDA_MEDIAN_SIZE
    ) -> np.ndarray:
        eda = np.asarray(eda, dtype=float).flatten()
        median_kernel = _positive_integer(median_kernel, "median_kernel")
        return median_filter(eda, size=median_kernel)

    def decompose_eda(
        self, eda: np.ndarray, method: str = "highpass"
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Split EDA into tonic and phasic signals; cvxEDA failures use a median filter."""
        if method not in {"highpass", "median", "cvxeda"}:
            raise ValueError(
                "EDA decomposition method must be 'highpass', 'median', or 'cvxeda'"
            )
        eda = np.asarray(eda, dtype=float).flatten()
        if not np.isfinite(eda).all():
            raise ValueError("EDA decomposition input must contain only finite samples")
        if len(eda) == 0:
            return eda.copy(), eda.copy()

        if method == "cvxeda":
            return self._decompose_cvxeda(eda)
        if method == "median":
            return self._decompose_median(eda)
        return self._decompose_highpass(eda)

    def _decompose_highpass(
        self, eda: np.ndarray, cutoff: float = 0.05
    ) -> Tuple[np.ndarray, np.ndarray]:
        nyq = 0.5 * self.sampling_rate
        cutoff_norm = cutoff / nyq

        if cutoff_norm >= 1.0:
            self.logger.warning(
                "Cutoff too high for Nyquist, returning original as tonic"
            )
            return eda.copy(), np.zeros_like(eda)

        sos = signal.butter(2, cutoff_norm, btype="low", output="sos")
        tonic = signal.sosfiltfilt(sos, eda)
        return tonic, eda - tonic

    def _decompose_median(
        self, eda: np.ndarray, window_sec: float = 4.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        window_samples = int(window_sec * self.sampling_rate)
        if window_samples % 2 == 0:
            window_samples += 1
        window_samples = max(3, window_samples)

        tonic = median_filter(eda, size=window_samples)
        return tonic, eda - tonic

    def _decompose_cvxeda(self, eda: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        try:
            import neurokit2 as nk

            decomposed = nk.eda_phasic(
                eda, sampling_rate=int(self.sampling_rate), method="cvxeda"
            )
            tonic = cast(np.ndarray, decomposed["EDA_Tonic"].values)
            phasic = cast(np.ndarray, decomposed["EDA_Phasic"].values)
            return tonic, phasic
        except ImportError:
            self.logger.debug("neurokit2 not available, using median decomposition")
            return self._decompose_median(eda)
        except Exception as e:
            self.logger.warning(f"cvxEDA failed: {e}, using median decomposition")
            return self._decompose_median(eda)

    def detect_scr_peaks(
        self,
        phasic: np.ndarray,
        min_amplitude: float = 0.01,
        min_rise_time: float = 0.5,
        max_rise_time: float = 4.0,
    ) -> Tuple[np.ndarray, List[Dict]]:
        phasic = np.asarray(phasic, dtype=float).flatten()
        min_amplitude = _positive_number(min_amplitude, "min_amplitude")
        min_rise_time = _positive_number(min_rise_time, "min_rise_time")
        max_rise_time = _positive_number(max_rise_time, "max_rise_time")
        if min_rise_time > max_rise_time:
            raise ValueError("min_rise_time must not exceed max_rise_time")
        if not np.isfinite(phasic).all():
            raise ValueError("SCR input must contain only finite samples")

        min_rise_samples = int(min_rise_time * self.sampling_rate)
        max_rise_samples = int(max_rise_time * self.sampling_rate)
        min_distance = max(1, min_rise_samples)

        peaks, _ = signal.find_peaks(
            phasic,
            height=min_amplitude,
            distance=min_distance,
            prominence=min_amplitude / 2,
        )

        scr_features = []
        valid_peaks = []

        for peak_idx in peaks:
            search_start = max(0, peak_idx - max_rise_samples)
            onset_region = phasic[search_start:peak_idx]

            if len(onset_region) == 0:
                continue

            onset_local_idx = np.argmin(onset_region)
            onset_idx = search_start + onset_local_idx

            rise_samples = peak_idx - onset_idx
            rise_time = rise_samples / self.sampling_rate

            if rise_time < min_rise_time or rise_time > max_rise_time:
                continue

            amplitude = phasic[peak_idx] - phasic[onset_idx]

            if amplitude < min_amplitude:
                continue

            recovery_target = phasic[peak_idx] - amplitude / 2
            search_end = min(len(phasic), peak_idx + max_rise_samples * 2)
            recovery_region = phasic[peak_idx:search_end]

            recovered = np.flatnonzero(recovery_region <= recovery_target)
            recovery_time = (
                float(recovered[0] / self.sampling_rate) if len(recovered) else None
            )

            valid_peaks.append(peak_idx)
            scr_features.append(
                {
                    "peak_idx": peak_idx,
                    "onset_idx": onset_idx,
                    "amplitude": float(amplitude),
                    "rise_time": float(rise_time),
                    "recovery_time": float(recovery_time) if recovery_time else None,
                    "peak_time": float(peak_idx / self.sampling_rate),
                }
            )

        return np.array(valid_peaks, dtype=int), scr_features
