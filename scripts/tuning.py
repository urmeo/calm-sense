import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import GridSearchCV, GroupKFold, LeaveOneGroupOut

from scripts.analysis import analysis_directories, load_feature_data, save_figure
from scripts.run_experiment import (
    BENCHMARK_PROTOCOL_VERSION,
    CLASSIFIERS,
    CLF_NAMES,
    build_pipeline,
    load_cached,
    prepare_task,
)
from src.config import DEMO_DIR, FIGURES_DIR, RESULTS_DIR, SEED
from src.features.feature_pipeline import FEATURE_SCHEMA_VERSION
from src.utils import write_json, analysis_provenance, benchmark_reference

GRIDS = {
    "lr": {"clf__C": [0.1, 1.0, 10.0]},
    "rf": {
        "clf__n_estimators": [200, 400],
        "clf__max_depth": [6, 10, None],
        "clf__min_samples_leaf": [1, 5],
    },
    "xgb": {
        "clf__max_depth": [3, 5, 7],
        "clf__learning_rate": [0.05, 0.1],
        "clf__scale_pos_weight": [1, 3],
    },
    "lgbm": {
        "clf__num_leaves": [31, 63],
        "clf__learning_rate": [0.05, 0.1],
        "clf__n_estimators": [200, 400],
    },
}


def tune_model(key, X, y, groups, inner_splits=3):
    if key not in GRIDS:
        raise ValueError(f"No tuning grid for classifier: {key}")
    if (
        isinstance(inner_splits, bool)
        or not isinstance(inner_splits, (int, np.integer))
        or inner_splits < 2
    ):
        raise ValueError("Nested tuning requires at least two inner subject splits")
    if len(np.unique(groups)) < 3:
        raise ValueError("Nested tuning requires at least three subjects")
    logo = LeaveOneGroupOut()
    rows, chosen = [], []
    for train_idx, test_idx in logo.split(X, y, groups):
        gtr = groups[train_idx]
        k = min(inner_splits, len(np.unique(gtr)))
        search = GridSearchCV(
            build_pipeline(key),
            GRIDS[key],
            cv=GroupKFold(n_splits=k),
            scoring="balanced_accuracy",
            error_score="raise",
        )
        search.fit(X[train_idx], y[train_idx], groups=gtr)
        if not np.isfinite(search.cv_results_["mean_test_score"]).all():
            raise ValueError("Nested tuning produced a nonfinite validation score")
        pred = search.predict(X[test_idx])
        rows.append(
            {
                "subject": groups[test_idx][0],
                "accuracy": accuracy_score(y[test_idx], pred),
                "f1_macro": f1_score(
                    y[test_idx],
                    pred,
                    labels=np.unique(y),
                    average="macro",
                    zero_division=0,
                ),
                "balanced_accuracy": balanced_accuracy_score(y[test_idx], pred),
            }
        )
        chosen.append(tuple(sorted(search.best_params_.items())))
    df = pd.DataFrame(rows)
    mode_params = dict(Counter(chosen).most_common(1)[0][0])
    return df, mode_params


def compute(X, y, groups, inner_splits=3):
    out = {}
    for key in CLASSIFIERS:
        df, params = tune_model(key, X, y, groups, inner_splits)
        out[CLF_NAMES[key]] = {
            "accuracy_mean": float(df["accuracy"].mean()),
            "accuracy_std": float(df["accuracy"].std()),
            "f1_macro_mean": float(df["f1_macro"].mean()),
            "balanced_accuracy_mean": float(df["balanced_accuracy"].mean()),
            "best_params": {k.replace("clf__", ""): v for k, v in params.items()},
        }
    return out


def _defaults(results_dir=None, *, groups=None):
    path = (RESULTS_DIR if results_dir is None else results_dir) / "metrics.json"
    if not path.exists():
        return {}
    with open(path) as f:
        summary = json.load(f)
    binary = summary.get("binary", {})
    if groups is not None:
        rows = binary.get("per_subject")
        subjects = (
            [row.get("subject") if isinstance(row, dict) else None for row in rows]
            if isinstance(rows, list)
            else []
        )
        if (
            summary.get("benchmark_protocol_version") != BENCHMARK_PROTOCOL_VERSION
            or binary.get("feature_schema_version") != FEATURE_SCHEMA_VERSION
            or not subjects
            or any(
                not isinstance(subject, str) or not subject.strip()
                for subject in subjects
            )
            or len(set(subjects)) != len(subjects)
            or set(subjects) != set(groups)
        ):
            return {}
    models = binary.get("models", [])
    return {m["model"]: m["accuracy_mean"] for m in models}


def _plot(tuned, defaults, path):
    names = [name for name in tuned if name != "provenance" and name in defaults]
    if not names:
        return
    x = np.arange(len(names))
    plt.figure(figsize=(7, 4))
    plt.bar(
        x - 0.2, [defaults[n] for n in names], 0.4, label="default", color="#95a5a6"
    )
    plt.bar(
        x + 0.2,
        [tuned[n]["accuracy_mean"] for n in names],
        0.4,
        label="tuned",
        color="#3498db",
    )
    plt.xticks(x, names, rotation=20, ha="right")
    plt.ylabel("LOSO accuracy")
    plt.ylim(0, 1)
    plt.title("Default vs nested-CV tuned")
    plt.legend()
    save_figure(path)


def run(synthetic=False, inner_splits=3):
    results_dir, figures_dir = analysis_directories(
        synthetic, DEMO_DIR, RESULTS_DIR, FIGURES_DIR
    )
    features_df, x_raw = load_feature_data(
        load_cached, synthetic=synthetic, n_subjects=6, seed=SEED
    )

    benchmark_sha = benchmark_reference(results_dir, features_df)
    X, y, groups, _, _ = prepare_task(features_df, x_raw, [1, 2])
    tuned = compute(X, y, groups, inner_splits)
    defaults = _defaults(results_dir, groups=groups)
    if not defaults and (results_dir / "metrics.json").exists():
        print("No compatible default benchmark; the comparison is omitted.")

    tuned["provenance"] = analysis_provenance(results_dir, benchmark_sha)
    write_json(results_dir / "tuning.json", tuned)
    if defaults:
        _plot(tuned, defaults, figures_dir / "tuning.png")
    else:
        (figures_dir / "tuning.png").unlink(missing_ok=True)

    print(f"\n{'Model':20s} {'default':>8s} {'tuned':>8s}")
    for name, r in tuned.items():
        if name == "provenance":
            continue
        d = defaults.get(name)
        d_str = "n/a" if d is None else f"{d:.3f}"
        print(f"{name:20s} {d_str:>8} {r['accuracy_mean']:>8.3f}")
    print(f"\nWrote {results_dir / 'tuning.json'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--inner-splits", type=int, default=3)
    args = parser.parse_args()
    run(synthetic=args.synthetic, inner_splits=args.inner_splits)
