from typing import Any, cast, Dict, Tuple

import numpy as np
from scipy import stats
from scipy.spatial.distance import cdist

from ..config import FEATURE_PARAMS
from ..preprocessing.filters import _positive_integer, _positive_number
from .hrv_base import BaseHRVExtractor


class HRVNonlinearExtractor(BaseHRVExtractor):
    def __init__(self, min_rr_count: int = 50):
        self.min_rr_count = _positive_integer(min_rr_count, "min_rr_count")

    def _entropy_input(self, rr, m, r):
        rr = np.asarray(rr, dtype=float).flatten()
        m = _positive_integer(m, "m")
        r = _positive_number(r, "r")
        if len(rr) < m + 2 or not np.isfinite(rr).all():
            return None, m, r
        return rr, m, r * np.std(rr)

    def compute_sample_entropy(
        self, rr: np.ndarray, m: int = 2, r: float = 0.2
    ) -> float:
        validated, m, tolerance = self._entropy_input(rr, m, r)
        if validated is None:
            return np.nan

        patterns = np.lib.stride_tricks.sliding_window_view(validated, m + 1)
        b = np.count_nonzero(
            np.triu(
                cdist(patterns[:, :m], patterns[:, :m], "chebyshev") <= tolerance, 1
            )
        )
        a = np.count_nonzero(
            np.triu(cdist(patterns, patterns, "chebyshev") <= tolerance, 1)
        )

        if b == 0:
            return np.nan
        if a == 0:
            return np.inf
        return float(-np.log(a / b))

    def compute_approximate_entropy(
        self, rr: np.ndarray, m: int = 2, r: float = 0.2
    ) -> float:
        validated, m, tolerance = self._entropy_input(rr, m, r)
        if validated is None:
            return np.nan

        def _phi(template_len: int) -> float:
            patterns = np.lib.stride_tricks.sliding_window_view(validated, template_len)
            counts = np.count_nonzero(
                cdist(patterns, patterns, "chebyshev") <= tolerance, axis=1
            )
            return float(np.mean(np.log(counts / len(patterns))))

        return float(_phi(m) - _phi(m + 1))

    def compute_dfa(
        self, rr: np.ndarray, scale_min: int = 4, scale_max: int = 64
    ) -> Tuple[float, float]:
        scale_min = _positive_integer(scale_min, "scale_min")
        scale_max = _positive_integer(scale_max, "scale_max")
        if scale_min >= scale_max:
            raise ValueError("scale_min must be below scale_max")
        rr = np.asarray(rr, dtype=float).flatten()
        n = len(rr)
        if n < scale_max or not np.isfinite(rr).all() or np.std(rr) == 0:
            return np.nan, np.nan

        y = np.cumsum(rr - np.mean(rr))

        scales = np.logspace(np.log10(scale_min), np.log10(scale_max), 15).astype(int)
        scales = np.unique(scales)
        scales = scales[scales >= 4]

        fluctuations = []

        for scale in scales:
            n_windows = n // scale

            if n_windows < 2:
                continue

            segments = y[: n_windows * scale].reshape(n_windows, scale)
            x = np.arange(scale)
            slopes, intercepts = np.polyfit(x, segments.T, 1)
            trends = slopes[:, None] * x + intercepts[:, None]
            fluctuations.append((scale, np.sqrt(np.mean((segments - trends) ** 2))))

        if len(fluctuations) < 4:
            return np.nan, np.nan

        scales_used = np.array([f[0] for f in fluctuations])
        fluct_values = np.array([f[1] for f in fluctuations])

        log_scales = np.log10(scales_used)
        log_fluct = np.log10(fluct_values + FEATURE_PARAMS.EPSILON)

        alphas = []
        for low, high in ((4, 16), (16, 64)):
            mask = (scales_used >= low) & (scales_used <= high)
            alpha = (
                float(
                    cast(Any, stats.linregress(log_scales[mask], log_fluct[mask])).slope
                )
                if np.sum(mask) >= 2
                else np.nan
            )
            alphas.append(alpha)
        return alphas[0], alphas[1]

    def compute_poincare(self, rr: np.ndarray) -> Dict[str, float]:
        rr = np.asarray(rr, dtype=float).flatten()
        if len(rr) < 3:
            return dict.fromkeys(("SD1", "SD2", "SD1_SD2_ratio", "CSI", "CVI"), np.nan)

        sd1 = float(np.std(np.diff(rr), ddof=1) / np.sqrt(2))
        sd2 = float(np.std((rr[:-1] + rr[1:]) / np.sqrt(2), ddof=1))

        return {
            "SD1": sd1,
            "SD2": sd2,
            "SD1_SD2_ratio": (
                float(sd1 / sd2) if sd2 > FEATURE_PARAMS.EPSILON else np.nan
            ),
            "CSI": float(sd2 / sd1) if sd1 > FEATURE_PARAMS.EPSILON else np.nan,
            "CVI": float(np.log10(16 * sd1 * sd2)) if sd1 > 0 and sd2 > 0 else np.nan,
        }

    def compute_rqa_determinism(
        self,
        rr: np.ndarray,
        embedding_dim: int = 10,
        time_delay: int = 1,
        radius: float = 0.2,
        min_line_length: int = 2,
    ) -> float:
        embedding_dim = _positive_integer(embedding_dim, "embedding_dim")
        time_delay = _positive_integer(time_delay, "time_delay")
        min_line_length = _positive_integer(min_line_length, "min_line_length")
        radius = _positive_number(radius, "radius")
        rr = np.asarray(rr, dtype=float).flatten()
        n = len(rr)
        n_vectors = n - (embedding_dim - 1) * time_delay
        if n_vectors < 2 or not np.isfinite(rr).all():
            return np.nan

        starts = np.arange(n_vectors)[:, None]
        offsets = np.arange(embedding_dim) * time_delay
        embedded = rr[starts + offsets]
        dist_matrix = cdist(embedded, embedded, "chebyshev")
        threshold = radius * np.max(dist_matrix)
        recurrence_matrix = (dist_matrix <= threshold).astype(int)
        np.fill_diagonal(recurrence_matrix, 0)

        total_recurrence = np.sum(recurrence_matrix)

        if total_recurrence == 0:
            return 0.0

        diagonal_points = 0

        for k in range(1, n_vectors):
            diag = np.diag(recurrence_matrix, k)
            runs = np.diff(np.concatenate([[0], diag, [0]]))
            run_starts = np.where(runs == 1)[0]
            run_ends = np.where(runs == -1)[0]
            run_lengths = run_ends - run_starts

            diagonal_points += np.sum(run_lengths[run_lengths >= min_line_length])

        diagonal_points *= 2

        return float(diagonal_points / total_recurrence)

    def extract_all(self, rr_intervals: np.ndarray) -> Dict[str, float]:
        features = dict.fromkeys(self.get_feature_descriptions(), np.nan)

        rr = self._validate_input(rr_intervals)
        if rr is None:
            self.logger.warning("Invalid input, returning NaN features")
            return features

        try:
            features["SampEn"] = self.compute_sample_entropy(rr, m=2, r=0.2)
            features["ApEn"] = self.compute_approximate_entropy(rr, m=2, r=0.2)

            features["DFA_alpha1"], features["DFA_alpha2"] = self.compute_dfa(rr)
            features.update(self.compute_poincare(rr))

            features["RQA_DET"] = self.compute_rqa_determinism(rr[:200])

        except Exception as e:
            self.logger.error(f"Feature extraction failed: {e}")

        return features

    def get_feature_descriptions(self) -> Dict[str, str]:
        return {
            "SampEn": "Sample entropy (complexity, m=2, r=0.2*SD)",
            "ApEn": "Approximate entropy (regularity, m=2, r=0.2*SD)",
            "DFA_alpha1": "DFA short-term scaling exponent (4-16 beats)",
            "DFA_alpha2": "DFA long-term scaling exponent (16-64 beats)",
            "SD1": "Poincaré plot width - short-term variability (ms)",
            "SD2": "Poincaré plot length - long-term variability (ms)",
            "SD1_SD2_ratio": "SD1/SD2 ratio",
            "CSI": "Cardiac Sympathetic Index (SD2/SD1)",
            "CVI": "Cardiac Vagal Index (log10(16*SD1*SD2))",
            "RQA_DET": "Embedded recurrence determinism (m=10, delay=1; 0-1)",
        }
