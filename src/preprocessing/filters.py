from typing import Tuple, Union

import numpy as np
from scipy import signal

from ..config import FILTER_PARAMS, FS
from ..logging_config import LoggerMixin


def _positive_number(value, name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be finite and positive") from error
    if isinstance(value, (bool, np.bool_)) or not np.isfinite(number) or number <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return number


def _positive_integer(value, name: str) -> int:
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
        or value < 1
    ):
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _window_parameters(window_sec, overlap, purity, fs):
    window_sec = _positive_number(window_sec, "window_sec")
    fs = _positive_number(fs, "sampling_rate")
    original_overlap, original_purity = overlap, purity
    try:
        overlap, purity = float(overlap), float(purity)
    except (TypeError, ValueError) as error:
        raise ValueError("overlap and purity must be numeric") from error
    if (
        not np.isfinite(overlap)
        or isinstance(original_overlap, (bool, np.bool_))
        or not 0 <= overlap < 1
    ):
        raise ValueError("overlap must be finite and in [0, 1)")
    if (
        not np.isfinite(purity)
        or isinstance(original_purity, (bool, np.bool_))
        or not 0 < purity <= 1
    ):
        raise ValueError("purity must be finite and in (0, 1]")
    window_samples = int(window_sec * fs)
    step = int(window_samples * (1 - overlap))
    if window_samples < 1 or step < 1:
        raise ValueError("Window size and stride must each span at least one sample")
    return fs, window_samples, step


class SignalProcessor(LoggerMixin):
    def __init__(self, fs: float = FS.CHEST):
        self.fs = _positive_number(fs, "fs")

    def butterworth_filter(
        self,
        data: np.ndarray,
        cutoff: Union[float, Tuple[float, float]],
        order: int = 4,
        btype: str = "low",
    ) -> np.ndarray:
        data = np.asarray(data, dtype=float).flatten()
        order = _positive_integer(order, "order")
        nyq = 0.5 * self.fs

        normalized_cutoff: Union[float, Tuple[float, float]]
        if isinstance(cutoff, tuple):
            normalized_cutoff = (cutoff[0] / nyq, cutoff[1] / nyq)
            if not 0 < normalized_cutoff[0] < normalized_cutoff[1] < 1.0:
                raise ValueError(
                    f"Cutoff frequencies {cutoff} exceed Nyquist frequency {nyq} Hz"
                )
        else:
            normalized_cutoff = cutoff / nyq
            if not np.isfinite(normalized_cutoff) or not 0 < normalized_cutoff < 1.0:
                raise ValueError(
                    f"Cutoff frequency {cutoff} exceeds Nyquist frequency {nyq} Hz"
                )

        if len(data) == 0:
            return data
        if not np.isfinite(data).all():
            raise ValueError("Filter input must contain only finite samples")

        sos = signal.butter(order, normalized_cutoff, btype=btype, output="sos")
        return signal.sosfiltfilt(sos, data)

    def process_respiration(self, resp: np.ndarray) -> np.ndarray:
        return self.butterworth_filter(
            resp,
            cutoff=(FILTER_PARAMS.RESP_BANDPASS_LOW, FILTER_PARAMS.RESP_BANDPASS_HIGH),
            order=FILTER_PARAMS.RESP_FILTER_ORDER,
            btype="band",
        )

    def process_temperature(self, temp: np.ndarray) -> np.ndarray:
        return self.butterworth_filter(
            temp,
            cutoff=FILTER_PARAMS.TEMP_LOWPASS,
            order=FILTER_PARAMS.TEMP_FILTER_ORDER,
            btype="low",
        )
