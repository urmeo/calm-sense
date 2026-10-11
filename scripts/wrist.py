import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from scripts.run_experiment import (
    CLASSIFIERS,
    CLF_NAMES,
    _prepare_features,
    build_pipeline,
    loso_evaluate,
)
from src.config import FIGURES_DIR, RESULTS_DIR
from src.dataset_wrist import WristDataset, load_wrist
from src.features.feature_pipeline import FEATURE_SCHEMA_VERSION
from src.utils import (
    feature_frame_sha256,
    write_json,
    analysis_provenance,
    benchmark_reference,
)

META = ["subject_id", "window_id", "label", "label_name"]


def prepare_binary(df):
    X, y, groups, _, _ = _prepare_features(df, [1, 2], META)
    return X, y, groups


def _validate_chest_comparison(chest, groups):
    rows = chest.get("per_subject")
    subjects = (
        [row.get("subject") if isinstance(row, dict) else None for row in rows]
        if isinstance(rows, list)
        else []
    )
    if (
        chest.get("feature_schema_version") != FEATURE_SCHEMA_VERSION
        or not subjects
        or any(
            not isinstance(subject, str) or not subject.strip() for subject in subjects
        )
        or len(set(subjects)) != len(subjects)
        or set(subjects) != set(groups)
    ):
        raise ValueError(
            "Chest and wrist comparison requires the current feature schema and identical "
            "subject cohorts. Rebuild chest metrics for these subjects with "
            "python scripts/run_experiment.py --rebuild --subjects "
            + " ".join(sorted(map(str, set(groups))))
        )


def run():
    benchmark_sha = benchmark_reference(RESULTS_DIR, shared_cache=False)
    chest_path = RESULTS_DIR / "metrics.json"
    if not chest_path.exists():
        raise SystemExit(
            "No chest benchmark results. Run scripts/run_experiment.py first."
        )
    with open(chest_path) as f:
        chest = json.load(f)["binary"]
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    df = load_wrist()
    if df is None:
        print("Building wrist features...")
        df = WristDataset().build()
    X, y, groups = prepare_binary(df)
    _validate_chest_comparison(chest, groups)
    print(
        f"Wrist: {len(y)} windows, {X.shape[1]} features, {len(np.unique(groups))} subjects"
    )

    rows = []
    for key in CLASSIFIERS:
        res = loso_evaluate(lambda k=key: build_pipeline(k), X, y, groups)
        rows.append(
            {
                "model": CLF_NAMES[key],
                "accuracy_mean": res["accuracy_mean"],
                "f1_macro_mean": res["f1_macro_mean"],
            }
        )
        print(
            f"  {CLF_NAMES[key]:20s} acc={res['accuracy_mean']:.3f} f1={res['f1_macro_mean']:.3f}"
        )

    best = max(rows, key=lambda r: r["accuracy_mean"])
    chest_best = max(chest["models"], key=lambda r: r["accuracy_mean"])
    chest_rf = next(
        m["accuracy_mean"] for m in chest["models"] if m["model"] == "Random Forest"
    )
    wrist_rf = next(m["accuracy_mean"] for m in rows if m["model"] == "Random Forest")

    out = {
        "wrist_models": rows,
        "wrist_best": best,
        "chest_best": {
            "model": chest_best["model"],
            "accuracy_mean": chest_best["accuracy_mean"],
        },
        "same_model_rf": {
            "chest": chest_rf,
            "wrist": wrist_rf,
            "drop_pts": (chest_rf - wrist_rf) * 100,
        },
        "best_per_arm_drop_pts": (chest_best["accuracy_mean"] - best["accuracy_mean"])
        * 100,
    }
    out["provenance"] = analysis_provenance(RESULTS_DIR, benchmark_sha)
    out["provenance"]["feature_frame_sha256"] = feature_frame_sha256(df)
    write_json(RESULTS_DIR / "wrist.json", out)

    plt.figure(figsize=(4.5, 4))
    bars = plt.bar(
        ["Chest\n(RespiBAN)", "Wrist\n(Empatica E4)"],
        [chest_rf, wrist_rf],
        color=["#3498db", "#9b59b6"],
    )
    for b, v in zip(bars, [chest_rf, wrist_rf]):
        plt.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.3f}", ha="center")
    plt.ylabel("LOSO accuracy (Random Forest, binary)")
    plt.ylim(0, 1)
    plt.title(
        f"Chest vs wrist (same model): {out['same_model_rf']['drop_pts']:.1f} pt drop"
    )
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "chest_vs_wrist.png", dpi=150)
    plt.close()
    print(
        f"\nSame-model (RF): chest {chest_rf:.3f} vs wrist {wrist_rf:.3f} "
        f"({out['same_model_rf']['drop_pts']:.1f} pt drop). "
        f"Best-per-arm: chest {chest_best['accuracy_mean']:.3f} vs wrist {best['accuracy_mean']:.3f}."
    )


if __name__ == "__main__":
    run()
