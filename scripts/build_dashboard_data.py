import json
import sys
from math import isclose
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from src.calibration import BINARY_BRIER_DEFINITION, normalize_binary_calibration
from src.config import OUTPUT_DIR, RESULTS_DIR
from src.utils import (
    atomic_write_text,
    benchmark_reference,
    pipeline_source_sha256,
    sha256_file,
)

DASHBOARD_RESULTS = OUTPUT_DIR / "dashboard" / "results.ts"

TASK_KEYS = [
    "n_windows",
    "n_features",
    "classes",
    "models",
    "best_model",
    "loso_accuracy",
    "loso_pooled_accuracy",
    "loso_matched_accuracy",
    "within_subject_accuracy",
    "optimism_gap_pts",
]


def _load_json(name):
    path = RESULTS_DIR / name
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def _load_csv(name):
    path = RESULTS_DIR / name
    return pd.read_csv(path).to_dict("records") if path.exists() else None


def _validate_tasks(out):
    for task in ("binary", "multiclass"):
        data = out.get(task)
        if (
            not isinstance(data, dict)
            or not isinstance(data.get("models"), list)
            or not data["models"]
        ):
            raise ValueError(f"{task}: dashboard requires a non-empty model comparison")
        for key in ("n_windows", "n_features"):
            if type(data.get(key)) is not int or data[key] <= 0:
                raise ValueError(f"{task}: {key} must be a positive integer")
        if "n_subjects" in data and (
            type(data["n_subjects"]) is not int
            or not 2 <= data["n_subjects"] <= data["n_windows"]
        ):
            raise ValueError(
                f"{task}: n_subjects must be an integer >= 2 and <= n_windows"
            )
        expected_classes = ["baseline", "stress"] + (
            ["amusement"] if task == "multiclass" else []
        )
        if data.get("classes") != expected_classes:
            raise ValueError(f"{task}: classes must be {expected_classes}")
        names = []
        for model in data["models"]:
            if (
                not isinstance(model, dict)
                or not isinstance(model.get("model"), str)
                or not model["model"].strip()
            ):
                raise ValueError(f"{task}: model names must be nonempty strings")
            names.append(model["model"])
            for key in (
                "accuracy_mean",
                "accuracy_std",
                "f1_macro_mean",
                "balanced_accuracy",
            ):
                _unit_metric(model.get(key), f"{task}: {model['model']} {key}")
        if len(names) != len(set(names)):
            raise ValueError(f"{task}: model names must be unique")
        if data.get("best_model") not in names:
            raise ValueError(f"{task}: best_model is absent from model comparisons")
        for key in ("loso_accuracy", "loso_pooled_accuracy"):
            _unit_metric(data.get(key), f"{task}: {key}")
        selected = data["models"][names.index(data["best_model"])]
        if abs(selected["accuracy_mean"] - data["loso_accuracy"]) > 1e-9:
            raise ValueError(f"{task}: LOSO accuracy disagrees with the selected model")
        for key in ("loso_matched_accuracy", "within_subject_accuracy"):
            if data.get(key) is not None:
                _unit_metric(data[key], f"{task}: {key}")
        gap = data.get("optimism_gap_pts")
        if gap is not None:
            matched, within = data.get("loso_matched_accuracy"), data.get(
                "within_subject_accuracy"
            )
            if (
                type(gap) not in (int, float)
                or matched is None
                or within is None
                or not isclose(
                    gap, round((within - matched) * 100, 1), rel_tol=0, abs_tol=1e-9
                )
            ):
                raise ValueError(
                    f"{task}: optimism gap must match the recorded comparison"
                )


def _unit_metric(value, name):
    if type(value) not in (int, float) or not 0 <= value <= 1:
        raise ValueError(f"{name} must be a numeric value in [0, 1]")


def _unverified_sections(metrics, out):
    """Protocol versions apply only to the recorded analysis."""
    sections = set(out) & {
        "shap",
        "stats",
        "wrist",
        "cross_dataset",
        "calibration",
        "personalization",
        "tuning",
        "ablation",
    }
    artifacts = metrics.get("artifacts", {})
    if not isinstance(artifacts, dict):
        raise ValueError("Benchmark artifact metadata must be an object")
    artifact = artifacts.get("shap")
    if "shap" in sections and artifact is not None:
        path = RESULTS_DIR / "shap_top_features.csv"
        if (
            not isinstance(artifact, dict)
            or artifact.get("model") != "XGBoost"
            or artifact.get("scope") != "full_data_binary_fit"
            or artifact.get("path") != path.name
            or artifact.get("sha256") != sha256_file(path)
        ):
            raise ValueError(
                "SHAP snapshot does not match the primary benchmark artifact"
            )
        sections.remove("shap")
    for section in sorted(sections):
        data = out[section]
        if section == "ablation":
            references = [row.get("benchmark_sha256") for row in data]
            if not any(reference is not None for reference in references):
                continue
        else:
            context = data.get("provenance", {}) if isinstance(data, dict) else {}
            if (
                not isinstance(context, dict)
                or "primary_benchmark_sha256" not in context
            ):
                continue
            references = [context["primary_benchmark_sha256"]]
            primary_context = metrics.get("provenance", {})
            if not isinstance(primary_context, dict) or context.get(
                "source_file_sha256"
            ) != primary_context.get("source_file_sha256"):
                raise ValueError(
                    f"{section} snapshot source does not match the primary benchmark"
                )
        expected = benchmark_reference(RESULTS_DIR, shared_cache=False, recorded=True)
        if expected is None or any(reference != expected for reference in references):
            raise ValueError(f"{section} snapshot does not match the primary benchmark")
        sections.remove(section)
    return sorted(sections)


def run():
    metrics = _load_json("metrics.json")
    if metrics is None:
        raise SystemExit(
            f"{RESULTS_DIR / 'metrics.json'} missing. Run scripts/run_experiment.py first."
        )
    if not isinstance(metrics, dict):
        raise ValueError("Benchmark results must be an object")
    artifacts = metrics.get("artifacts", {})
    if not isinstance(artifacts, dict):
        raise ValueError("Benchmark artifact metadata must be an object")

    out = {}
    if "benchmark_protocol_version" in metrics:
        if (
            type(metrics["benchmark_protocol_version"]) is not int
            or metrics["benchmark_protocol_version"] < 1
        ):
            raise ValueError("Benchmark protocol version must be a positive integer")
        out["benchmark_protocol_version"] = metrics["benchmark_protocol_version"]
    for task in ("binary", "multiclass"):
        if task in metrics:
            if not isinstance(metrics[task], dict):
                raise ValueError(f"{task}: benchmark task must be an object")
            out[task] = {k: metrics[task].get(k) for k in TASK_KEYS}
            for key in ("n_subjects", "inference_model"):
                if key in metrics[task]:
                    out[task][key] = metrics[task][key]

    shap = _load_csv("shap_top_features.csv")
    if shap:
        out["shap"] = sorted(shap, key=lambda row: row["mean_abs_shap"], reverse=True)[
            :12
        ]
        artifact = artifacts.get("shap", {})
        if isinstance(artifact, dict) and artifact:
            out["shap_model"] = artifact.get("model")
            out["shap_scope"] = artifact.get("scope")
    for key, fname in [
        ("stats", "stats.json"),
        ("wrist", "wrist.json"),
        ("cross_dataset", "cross_dataset.json"),
        ("calibration", "calibration.json"),
        ("personalization", "personalization.json"),
        ("tuning", "tuning.json"),
    ]:
        data = _load_json(fname)
        if data is not None:
            if key == "calibration":
                data = normalize_binary_calibration(data)
            elif key == "personalization":
                data.setdefault("brier_definition", BINARY_BRIER_DEFINITION)
                if data["brier_definition"] != BINARY_BRIER_DEFINITION:
                    raise ValueError(
                        "Personalization Brier values must use positive-class MSE"
                    )
            out[key] = data
    ablation = _load_csv("ablation.csv")
    if ablation:
        out["ablation"] = ablation

    if out.get("benchmark_protocol_version", 1) >= 2:
        out["unverified_sections"] = _unverified_sections(metrics, out)
        if benchmark_reference(RESULTS_DIR, shared_cache=False, recorded=True):
            out["benchmark_provenance"] = metrics["provenance"]
            out["benchmark_source_matches_current"] = (
                metrics["provenance"]["source_file_sha256"] == pipeline_source_sha256()
            )
    if "ablation" in out:
        out["ablation"] = [
            {key: value for key, value in row.items() if key != "benchmark_sha256"}
            for row in out["ablation"]
        ]

    payload = json.dumps(out, indent=2, allow_nan=False)
    _validate_tasks(out)
    atomic_write_text(
        DASHBOARD_RESULTS, f"const data = {payload};\n\nexport default data;\n"
    )
    print(f"Wrote {DASHBOARD_RESULTS} with sections: {sorted(out)}")


if __name__ == "__main__":
    run()
