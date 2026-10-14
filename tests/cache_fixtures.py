import pickle
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from src.cache import raw_input_sha256
from src.config import VALID_SUBJECTS


def source_loader(directory, subjects=VALID_SUBJECTS):
    """Write small source records for cache tests without extracting signals."""
    subjects = list(subjects)
    data_path = Path(directory) / "WESAD"
    record = {
        "signal": {
            "chest": {"ECG": np.zeros((700, 1))},
            "wrist": {
                "BVP": np.zeros((64, 1)),
                "EDA": np.zeros((4, 1)),
                "TEMP": np.zeros((4, 1)),
                "ACC": np.zeros((32, 3)),
            },
        },
        "label": np.ones(700, dtype=int),
    }
    payload = pickle.dumps(record, protocol=4)
    for subject in subjects:
        path = data_path / subject / f"{subject}.pkl"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
    return SimpleNamespace(
        subjects=subjects,
        data_path=data_path,
        raw_input_sha256=raw_input_sha256(data_path, subjects),
    )
