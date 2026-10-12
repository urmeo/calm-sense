import json
import math
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.calibration import BINARY_BRIER_DEFINITION, normalize_binary_calibration
from src.config import RESULTS_DIR
from src.utils import atomic_write_text

ROOT = Path(__file__).resolve().parent.parent
README = ROOT / "README.md"
RESULTS = RESULTS_DIR


def _format_metric(value: float) -> str:
    if not math.isfinite(value):
        raise ValueError("README metrics must be finite")
    return f"{value:.3f}"


def _replace(text: str, name: str, body: str) -> str:
    start, end = f"<!-- AUTOGEN:{name} START -->", f"<!-- AUTOGEN:{name} END -->"
    pattern = re.compile(re.escape(start) + r".*?" + re.escape(end), re.DOTALL)
    if not pattern.search(text):
        raise SystemExit(f"AUTOGEN markers for '{name}' not found in README.md")
    return pattern.sub(f"{start}\n{body}\n{end}", text)


def _calibration_table() -> str:
    calibration = normalize_binary_calibration(
        json.loads((RESULTS / "calibration.json").read_text())
    )
    head = "| Evaluation | ECE | MCE | Brier |\n| --- | :-: | :-: | :-: |"
    rows = [
        ("Subject-mixed 5-fold, non-overlapping", "within_subject"),
        ("LOSO, matched non-overlapping", "loso_matched"),
        ("LOSO, all windows", "loso"),
        ("LOSO, all windows + isotonic", "recalibrated_isotonic"),
    ]
    lines = [head]
    for label, key in rows:
        metrics = calibration[key]
        lines.append(
            f"| {label} | {_format_metric(metrics['ece'])} | "
            f"{_format_metric(metrics['mce'])} | {_format_metric(metrics['brier'])} |"
        )
    return "\n".join(lines)


def _personalization_table() -> str:
    personalization = json.loads((RESULTS / "personalization.json").read_text())
    if (
        personalization.get("brier_definition", BINARY_BRIER_DEFINITION)
        != BINARY_BRIER_DEFINITION
    ):
        raise ValueError("Personalization Brier values must use positive-class MSE")
    head = "| Recalibration / requested enrollment budget | ECE | Brier |\n| --- | :-: | :-: |"
    rows = [
        ("None (LOSO)", personalization["uncalibrated"]),
        ("Global (training subjects)", personalization["global"]),
    ]
    for budget in sorted(personalization["fewshot"], key=int):
        rows.append(
            (f"Per-subject, budget {budget}", personalization["fewshot"][budget])
        )
    lines = [head]
    for label, metrics in rows:
        lines.append(
            f"| {label} | {_format_metric(metrics['ece'])} | "
            f"{_format_metric(metrics['brier'])} |"
        )
    return "\n".join(lines)


def main() -> None:
    if (
        not (RESULTS / "calibration.json").exists()
        or not (RESULTS / "personalization.json").exists()
    ):
        sys.exit(
            "Missing results: run scripts/calibration.py and scripts/personalize.py first."
        )
    text = README.read_text()
    text = _replace(text, "calibration", _calibration_table())
    text = _replace(text, "personalization", _personalization_table())
    atomic_write_text(README, text)
    print(f"README calibration and personalization tables updated from {RESULTS}.")


if __name__ == "__main__":
    main()
