from typing import Dict

import numpy as np

from ..preprocessing.filters import _positive_integer
from .hrv_base import BaseHRVExtractor


class HRVTimeDomainExtractor(BaseHRVExtractor):
    def __init__(self, min_rr_count: int = 10):
        self.min_rr_count = _positive_integer(min_rr_count, "min_rr_count")

    def compute_sdnn(self, rr: np.ndarray) -> float:
        return float(np.std(rr, ddof=1))

    def compute_hrvti(self, rr: np.ndarray, bin_width: float = 7.8125) -> float:
        if not np.isfinite(bin_width) or bin_width <= 0:
            raise ValueError("bin_width must be finite and positive")
        rr = np.asarray(rr, dtype=float).flatten()
        if not len(rr) or not np.isfinite(rr).all():
            return np.nan
        _, counts = np.unique(np.floor(rr / bin_width), return_counts=True)
        return float(len(rr) / counts.max())

    def extract_all(self, rr_intervals: np.ndarray) -> Dict[str, float]:
        features = dict.fromkeys(self.get_feature_descriptions(), np.nan)

        rr = self._validate_input(rr_intervals)
        if rr is None:
            self.logger.warning("Invalid input, returning NaN features")
            return features

        try:
            features["MeanNN"] = float(np.mean(rr))
            features["SDNN"] = self.compute_sdnn(rr)
            differences = np.diff(rr)
            if len(differences):
                features["RMSSD"] = float(np.sqrt(np.mean(differences**2)))
                for threshold in (50, 20):
                    features[f"pNN{threshold}"] = float(
                        100.0
                        * np.sum(np.abs(differences) > threshold)
                        / len(differences)
                    )
            features["MedianNN"] = float(np.median(rr))
            features["CVNN"] = features["SDNN"] / features["MeanNN"]
            features["CVSD"] = features["RMSSD"] / features["MeanNN"]
            features["MadNN"] = float(np.median(np.abs(rr - features["MedianNN"])))
            features["MCVNN"] = features["MadNN"] / features["MedianNN"]
            q75, q25 = np.percentile(rr, [75, 25])
            features["IQRNN"] = float(q75 - q25)
            features["HRVTI"] = self.compute_hrvti(rr)

        except Exception as e:
            self.logger.error(f"Feature extraction failed: {e}")

        return features

    def get_feature_descriptions(self) -> Dict[str, str]:
        return {
            "MeanNN": "Mean of NN intervals (ms)",
            "SDNN": "Standard deviation of NN intervals (ms)",
            "RMSSD": "Root mean square of successive differences (ms)",
            "pNN50": "Percentage of intervals differing >50ms (%)",
            "pNN20": "Percentage of intervals differing >20ms (%)",
            "MedianNN": "Median of NN intervals (ms)",
            "CVNN": "Coefficient of variation (SDNN/MeanNN)",
            "CVSD": "CV of successive differences (RMSSD/MeanNN)",
            "MadNN": "Median absolute deviation (ms)",
            "MCVNN": "MadNN/MedianNN ratio",
            "IQRNN": "Interquartile range (ms)",
            "HRVTI": "HRV Triangular Index",
        }
