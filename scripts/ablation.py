import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from scripts.analysis import load_feature_data, save_figure, stage_analysis_outputs
from scripts.run_experiment import (
    build_pipeline,
    load_cached,
    loso_evaluate,
    prepare_task,
)
from src.config import FIGURES_DIR, RESULTS_DIR
from src.utils import atomic_write_text, analysis_provenance, benchmark_reference

FEATURE_GROUPS = ["HRV", "EDA", "TEMP", "RESP", "ACC"]

SUBSETS = {
    "All features": FEATURE_GROUPS,
    "No motion (HRV+EDA+TEMP+RESP)": ["HRV", "EDA", "TEMP", "RESP"],
    "Autonomic (HRV+EDA)": ["HRV", "EDA"],
    "HRV only": ["HRV"],
    "EDA only": ["EDA"],
    "Motion only (ACC)": ["ACC"],
}


def run():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    features_df, x_raw = load_feature_data(load_cached)
    benchmark_sha = benchmark_reference(RESULTS_DIR, features_df)

    X, y, groups, feature_cols, _ = prepare_task(features_df, x_raw, [1, 2])

    feature_groups = [column.partition("_")[0] for column in feature_cols]
    unknown = sorted(set(feature_groups) - set(FEATURE_GROUPS))
    if unknown:
        raise SystemExit(f"Feature groups not in FEATURE_GROUPS: {unknown}")

    rows = []
    for name, prefixes in SUBSETS.items():
        cols = [i for i, group in enumerate(feature_groups) if group in prefixes]
        if not cols:
            raise ValueError(
                f"No feature columns available for ablation subset: {name}"
            )
        res = loso_evaluate(lambda: build_pipeline("rf"), X[:, cols], y, groups)
        rows.append(
            {
                "subset": name,
                "n_features": len(cols),
                "accuracy_mean": res["accuracy_mean"],
                "accuracy_std": res["accuracy_std"],
                "f1_macro_mean": res["f1_macro_mean"],
            }
        )
        print(f"  {name:32s} ({len(cols):2d} feat)  acc={res['accuracy_mean']:.3f}")

    analysis_provenance(RESULTS_DIR, benchmark_sha)
    df = pd.DataFrame(rows)
    if benchmark_sha is not None:
        df["benchmark_sha256"] = benchmark_sha
    with stage_analysis_outputs(RESULTS_DIR, FIGURES_DIR) as (results, figures):
        atomic_write_text(results / "ablation.csv", df.to_csv(index=False))

        order = df.iloc[::-1]
        plt.figure(figsize=(7, 4))
        plt.barh(
            order["subset"],
            order["accuracy_mean"],
            xerr=order["accuracy_std"],
            color="#3498db",
        )
        plt.axvline(
            df.iloc[0]["accuracy_mean"], color="#e74c3c", ls="--", label="all features"
        )
        plt.xlabel("LOSO accuracy (Random Forest, binary)")
        plt.title("Feature-group ablation")
        plt.xlim(0, 1)
        plt.legend()
        save_figure(figures / "ablation.png")
    print(f"\nWrote {RESULTS_DIR / 'ablation.csv'} and {FIGURES_DIR / 'ablation.png'}")


if __name__ == "__main__":
    run()
