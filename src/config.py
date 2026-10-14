from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

PROJECT_ROOT = Path(__file__).parent.parent.absolute()

SEED: int = 42

RR_MIN_MS: float = 200.0
RR_MAX_MS: float = 2500.0
RR_CORRECTION_VERSION: int = 1

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
EXTERNAL_DATA_DIR = DATA_DIR / "external"
WESAD_DIR = RAW_DATA_DIR / "WESAD"

OUTPUT_DIR = PROJECT_ROOT / "outputs"
RESULTS_DIR = OUTPUT_DIR / "results"
MODELS_DIR = OUTPUT_DIR / "models"
GENERATED_DIR = OUTPUT_DIR / "generated"
FIGURES_DIR = GENERATED_DIR / "figures"
PROCESSED_DATA_DIR = GENERATED_DIR / "cache"
DEMO_DIR = GENERATED_DIR / "demo"
LOGS_DIR = GENERATED_DIR / "logs"

VALID_SUBJECTS: List[str] = [f"S{i}" for i in range(2, 18) if i != 12]

LABEL_NAMES: Dict[int, str] = {
    0: "not_defined",
    1: "baseline",
    2: "stress",
    3: "amusement",
    4: "meditation",
    5: "ignore",
    6: "ignore",
    7: "ignore",
}


@dataclass(frozen=True)
class SamplingRates:
    CHEST: float = 700.0

    WRIST_ACC: float = 32.0
    WRIST_BVP: float = 64.0
    WRIST_EDA: float = 4.0
    WRIST_TEMP: float = 4.0


FS = SamplingRates()


@dataclass(frozen=True)
class FilterParams:
    ECG_BANDPASS_LOW: float = 0.5
    ECG_BANDPASS_HIGH: float = 40.0
    ECG_FILTER_ORDER: int = 4

    EDA_LOWPASS: float = 5.0
    EDA_MEDIAN_SIZE: int = 5
    EDA_FILTER_ORDER: int = 4

    RESP_BANDPASS_LOW: float = 0.1
    RESP_BANDPASS_HIGH: float = 0.5
    RESP_FILTER_ORDER: int = 4

    TEMP_LOWPASS: float = 0.1
    TEMP_FILTER_ORDER: int = 2


FILTER_PARAMS = FilterParams()


@dataclass(frozen=True)
class FeatureParams:
    WINDOW_SIZE_SEC: float = 60.0
    WINDOW_OVERLAP: float = 0.5
    EPSILON: float = 1e-10


FEATURE_PARAMS = FeatureParams()

CONDITION_COLORS: Dict[str, str] = {
    "baseline": "#2ecc71",
    "stress": "#e74c3c",
    "amusement": "#3498db",
    "meditation": "#9b59b6",
    "not_defined": "#95a5a6",
    "ignore": "#bdc3c7",
}

LOG_LEVEL: str = "INFO"
LOG_FILE: Path = LOGS_DIR / "calmsense.log"
