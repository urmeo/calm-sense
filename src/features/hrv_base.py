from typing import Optional

import numpy as np

from ..config import RR_MAX_MS, RR_MIN_MS
from ..logging_config import LoggerMixin


class BaseHRVExtractor(LoggerMixin):
    min_rr_count: int = 10

    def _validate_input(self, rr_intervals: np.ndarray) -> Optional[np.ndarray]:
        rr = np.asarray(rr_intervals, dtype=float).flatten()
        rr = rr[np.isfinite(rr)]

        if len(rr) < self.min_rr_count:
            self.logger.warning(
                f"Insufficient RR intervals: {len(rr)} < {self.min_rr_count}"
            )
            return None

        rr = rr[(rr >= RR_MIN_MS) & (rr <= RR_MAX_MS)]

        if len(rr) < self.min_rr_count:
            self.logger.warning("Too many invalid RR intervals removed")
            return None

        return rr
