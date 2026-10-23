import pickle
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np

from ..cache import verified_input
from ..config import FS, LABEL_NAMES, VALID_SUBJECTS, WESAD_DIR
from ..logging_config import LoggerMixin


class WESADLoader(LoggerMixin):
    """Load trusted WESAD pickles at their recorded sampling rates."""

    VALID_SUBJECTS = VALID_SUBJECTS
    LABELS = LABEL_NAMES

    CHEST_FS = FS.CHEST
    WRIST_ACC_FS = FS.WRIST_ACC
    WRIST_BVP_FS = FS.WRIST_BVP
    WRIST_EDA_FS = FS.WRIST_EDA
    WRIST_TEMP_FS = FS.WRIST_TEMP

    def __init__(self, data_path: Optional[Union[str, Path]] = None):
        self.data_path = Path(data_path) if data_path else WESAD_DIR
        self._validate_path()
        self.subjects = self._discover_subjects()
        self.raw_input_sha256: Dict[str, str] = {}

    def _validate_path(self) -> None:
        if not self.data_path.is_dir():
            self.logger.error(f"WESAD data path not found: {self.data_path}")
            raise FileNotFoundError(
                f"WESAD data path not found: {self.data_path}\n"
                "Download with: python scripts/download_data.py --wesad\n"
                "Use the official source under its research agreement."
            )

    def _discover_subjects(self) -> List[str]:
        subjects = []
        for subj_id in self.VALID_SUBJECTS:
            pkl_path = self.data_path / subj_id / f"{subj_id}.pkl"
            if pkl_path.is_file():
                subjects.append(subj_id)

        if not subjects:
            self.logger.warning("No subjects found in dataset directory")

        return subjects

    def load_subject(
        self, subject_id: str, signals: Optional[List[str]] = None
    ) -> Dict:
        if subject_id not in self.VALID_SUBJECTS:
            raise ValueError(
                f"Invalid subject ID: {subject_id}. Valid subjects: {self.VALID_SUBJECTS}"
            )

        pkl_path = self.data_path / subject_id / f"{subject_id}.pkl"
        if not pkl_path.exists():
            raise FileNotFoundError(f"Subject data not found: {pkl_path}")

        self.logger.info(f"Loading subject {subject_id} from {pkl_path}")

        try:
            with verified_input(pkl_path) as (stream, digest):
                data = pickle.load(stream, encoding="latin1")
        except Exception as e:
            self.logger.error(f"Failed to load {subject_id}: {e}")
            raise

        try:
            chest_signals = data["signal"]["chest"]
            wrist_signals = data["signal"]["wrist"]
            labels = np.asarray(data["label"])
        except (KeyError, TypeError) as error:
            raise ValueError(f"Invalid WESAD structure for {subject_id}") from error
        if (
            labels.ndim != 1
            or not len(labels)
            or not np.issubdtype(labels.dtype, np.number)
            or np.issubdtype(labels.dtype, np.complexfloating)
            or not np.isfinite(labels).all()
        ):
            raise ValueError(
                f"{subject_id}: labels must be a nonempty finite real numeric vector"
            )
        duration = len(labels) / self.CHEST_FS
        wrist_rates = {
            "ACC": self.WRIST_ACC_FS,
            "BVP": self.WRIST_BVP_FS,
            "EDA": self.WRIST_EDA_FS,
            "TEMP": self.WRIST_TEMP_FS,
        }
        for device, channels in (("chest", chest_signals), ("wrist", wrist_signals)):
            if not isinstance(channels, dict) or not channels:
                raise ValueError(f"{subject_id}: missing {device} channel dictionary")
            for name, values in channels.items():
                values = np.asarray(values)
                expected_axes = 3 if name.upper() == "ACC" else 1
                if (
                    values.ndim not in (1, 2)
                    or (values.ndim == 1 and expected_axes != 1)
                    or (values.ndim == 2 and values.shape[1] != expected_axes)
                    or not np.issubdtype(values.dtype, np.number)
                    or np.issubdtype(values.dtype, np.complexfloating)
                ):
                    raise ValueError(
                        f"{subject_id}: invalid {device}/{name} signal shape or dtype"
                    )
                rate = (
                    self.CHEST_FS
                    if device == "chest"
                    else wrist_rates.get(name.upper())
                )
                if rate is None:
                    raise ValueError(f"{subject_id}: unknown wrist channel {name}")
                if abs(len(values) - duration * rate) > 1.01:
                    raise ValueError(
                        f"{subject_id}: {device}/{name} duration does not match labels"
                    )

        if signals is not None:
            signals_upper = {s.upper() for s in signals}
            chest_signals = {
                k: v for k, v in chest_signals.items() if k.upper() in signals_upper
            }
            wrist_signals = {
                k: v for k, v in wrist_signals.items() if k.upper() in signals_upper
            }

        result = {
            "subject": subject_id,
            "chest": chest_signals,
            "wrist": wrist_signals,
            "label": labels,
        }
        self.raw_input_sha256[f"{subject_id}/{subject_id}.pkl"] = digest

        self.logger.info(
            f"Loaded {subject_id}: {len(result['label'])} samples "
            f"({len(result['label']) / self.CHEST_FS:.1f}s)"
        )

        return result

    def __repr__(self) -> str:
        return f"WESADLoader(path='{self.data_path}', subjects={len(self.subjects)})"

    def __len__(self) -> int:
        return len(self.subjects)
