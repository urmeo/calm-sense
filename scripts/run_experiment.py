import argparse
import sys
from contextlib import ExitStack
from pathlib import Path
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import LeaveOneGroupOut, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

from scripts.analysis import save_figure
from src.artifacts import publish_files
from src.config import (
    DEMO_DIR,
    FIGURES_DIR,
    MODELS_DIR,
    PROCESSED_DATA_DIR,
    RESULTS_DIR,
    SEED,
)
from src.dataset import WindowedDataset, load_cached
from src.features.feature_pipeline import FEATURE_SCHEMA_VERSION
from src.models.ml.classifiers import get_classifier
from src.utils import (
    atomic_write_text,
    feature_frame_sha256,
    pipeline_source_sha256,
    provenance,
    save_verified_joblib,
    set_seed,
    sha256_file,
    write_json,
)

TASKS = {
    "binary": {"keep": [1, 2], "names": ["baseline", "stress"]},
    "multiclass": {"keep": [1, 2, 3], "names": ["baseline", "stress", "amusement"]},
}

CLASSIFIERS = ["lr", "rf", "xgb", "lgbm"]
BENCHMARK_PROTOCOL_VERSION = 2
CLF_NAMES = {
    "lr": "Logistic Regression",
    "rf": "Random Forest",
    "xgb": "XGBoost",
    "lgbm": "LightGBM",
    "cnn": "1D-CNN",
}


def build_pipeline(clf_key: str) -> Pipeline:
    estimator = get_classifier(clf_key)
    return Pipeline(
        [
            ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scale", StandardScaler()),
            ("clf", estimator),
        ]
    )


def _fit_params(pipe, y_train):
    if pipe.named_steps["clf"].__class__.__name__ == "XGBClassifier":
        return {"clf__sample_weight": compute_sample_weight("balanced", y_train)}
    return {}


def _loso_result(y, groups, folds):
    classes = np.unique(y)
    indices, predictions = zip(*folds)
    pooled_true = y[np.concatenate(indices)]
    pooled_pred = np.concatenate(predictions)
    subj_df = pd.DataFrame(
        {
            "subject": groups[test_idx][0],
            "n": len(test_idx),
            "accuracy": accuracy_score(y[test_idx], pred),
            "f1_macro": f1_score(
                y[test_idx], pred, labels=classes, average="macro", zero_division=0
            ),
        }
        for test_idx, pred in folds
    )
    return {
        "accuracy_mean": float(subj_df["accuracy"].mean()),
        "accuracy_std": float(subj_df["accuracy"].std()),
        "f1_macro_mean": float(subj_df["f1_macro"].mean()),
        "f1_macro_std": float(subj_df["f1_macro"].std()),
        "balanced_accuracy": float(balanced_accuracy_score(pooled_true, pooled_pred)),
        "pooled_accuracy": float(accuracy_score(pooled_true, pooled_pred)),
        "per_subject": subj_df,
        "y_true": pooled_true,
        "y_pred": pooled_pred,
        "classes": classes,
    }


def loso_evaluate(pipeline_factory, X, y, groups):
    folds = []
    for train_idx, test_idx in LeaveOneGroupOut().split(X, y, groups):
        pipe = pipeline_factory()
        pipe.fit(X[train_idx], y[train_idx], **_fit_params(pipe, y[train_idx]))
        folds.append((test_idx, np.array(pipe.predict(X[test_idx]), copy=True)))
    return _loso_result(y, groups, folds)


def cnn_loso(x_raw, y, groups):
    from src.models.dl.cnn_1d import CNN1DClassifier

    folds = []
    n_folds = len(np.unique(groups))

    for fold, (train_idx, test_idx) in enumerate(
        LeaveOneGroupOut().split(x_raw, y, groups), 1
    ):
        print(f"    1D-CNN fold {fold}/{n_folds}", flush=True)
        model = CNN1DClassifier(in_channels=x_raw.shape[1], random_state=SEED)
        model.fit(x_raw[train_idx], y[train_idx], groups=groups[train_idx])
        folds.append((test_idx, model.predict(x_raw[test_idx])))
    return _loso_result(y, groups, folds)


def nonoverlap_mask(groups):
    """Alternating windows remove direct overlap at the default 50%."""
    keep = np.zeros(len(groups), dtype=bool)
    for g in np.unique(groups):
        idx = np.where(groups == g)[0]
        keep[idx[::2]] = True
    return keep


def kfold_accuracy(pipeline_factory, X, y, groups) -> float:
    keep = nonoverlap_mask(groups)
    Xk, yk = X[keep], y[keep]

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    pooled_true, pooled_pred = [], []
    for train_idx, test_idx in skf.split(Xk, yk):
        pipe = pipeline_factory()
        pipe.fit(Xk[train_idx], yk[train_idx], **_fit_params(pipe, yk[train_idx]))
        pooled_true.extend(yk[test_idx])
        pooled_pred.extend(pipe.predict(Xk[test_idx]))
    return float(accuracy_score(pooled_true, pooled_pred))


def plot_confusion(result, names, title, path):
    cm = confusion_matrix(
        result["y_true"], result["y_pred"], labels=result["classes"], normalize="true"
    )
    plt.figure(figsize=(5, 4))
    sns.heatmap(
        cm, annot=True, fmt=".2f", cmap="Blues", xticklabels=names, yticklabels=names
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(title)
    save_figure(path)


def plot_model_comparison(rows, path):
    df = pd.DataFrame(rows).sort_values("accuracy_mean")
    plt.figure(figsize=(7, 4))
    plt.barh(df["model"], df["accuracy_mean"], xerr=df["accuracy_std"], color="#3498db")
    plt.xlabel("LOSO accuracy")
    plt.title("Subject-independent model comparison")
    plt.xlim(0, 1)
    for i, acc in enumerate(df["accuracy_mean"]):
        plt.text(acc + 0.01, i, f"{acc:.3f}", va="center")
    save_figure(path)


def plot_per_subject(subj_df, path):
    df = subj_df.sort_values("accuracy")
    plt.figure(figsize=(8, 4))
    plt.bar(df["subject"], df["accuracy"], color="#2ecc71")
    plt.axhline(df["accuracy"].mean(), color="#e74c3c", ls="--", label="mean")
    plt.ylabel("LOSO accuracy")
    plt.xlabel("Held-out subject")
    plt.title("Per-subject performance (best model)")
    plt.xticks(rotation=45)
    plt.ylim(0, 1)
    plt.legend()
    save_figure(path)


def plot_gap(loso_acc, kfold_acc, path):
    plt.figure(figsize=(4.5, 4))
    bars = plt.bar(
        ["LOSO\n(subject-independent)", "5-fold\n(subject-mixed)"],
        [loso_acc, kfold_acc],
        color=["#3498db", "#e67e22"],
    )
    for b, v in zip(bars, [loso_acc, kfold_acc]):
        plt.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.3f}", ha="center")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1)
    plt.title(f"Optimism gap: {(kfold_acc - loso_acc) * 100:.1f} pts")
    save_figure(path)


def plot_embedding(X, y, names, path):
    """Descriptive full-data PCA; excluded from classifier scoring."""
    from sklearn.decomposition import PCA

    classes = np.unique(y)
    if len(classes) != len(names):
        raise ValueError("PCA class names must match the observed task classes")
    Xi = SimpleImputer(strategy="median").fit_transform(X)
    Xs = StandardScaler().fit_transform(Xi)
    n_components = min(2, *Xs.shape)
    coords = PCA(n_components=n_components, random_state=SEED).fit_transform(Xs)
    vertical = coords[:, 1] if n_components == 2 else np.zeros(len(coords))
    plt.figure(figsize=(6, 5))
    for cls, name in zip(classes, names):
        m = y == cls
        plt.scatter(coords[m, 0], vertical[m], s=8, alpha=0.5, label=name)
    plt.xlabel("PC1")
    if n_components == 2:
        plt.ylabel("PC2")
    else:
        plt.yticks([])
    plt.title("Feature space (PCA)")
    plt.legend()
    save_figure(path)


def shap_analysis(X, y, feature_names, fig_dir):
    """Full-data explanatory fit; separate from LOSO scoring."""
    import shap

    pipe = build_pipeline("xgb")
    pipe.fit(X, y, **_fit_params(pipe, y))
    Xt = pipe.named_steps["scale"].transform(pipe.named_steps["impute"].transform(X))
    explainer = shap.TreeExplainer(pipe.named_steps["clf"])
    values = explainer.shap_values(Xt)
    if isinstance(values, list):
        values = values[-1]
    shap_vals = np.asarray(values)
    if shap_vals.ndim == 3:
        shap_vals = shap_vals[:, :, -1]

    shap.summary_plot(
        shap_vals, Xt, feature_names=feature_names, show=False, max_display=15
    )
    plt.tight_layout()
    plt.savefig(fig_dir / "shap_beeswarm.png", dpi=150, bbox_inches="tight")
    plt.close()

    arr = np.abs(shap_vals).mean(axis=0)
    importance = (
        pd.DataFrame({"feature": feature_names, "mean_abs_shap": arr})
        .sort_values("mean_abs_shap", ascending=False)
        .head(15)
    )
    return importance


def _prepare_features(features_df, keep, meta):
    mask = features_df["label"].isin(keep).to_numpy()
    sub = features_df[mask].reset_index(drop=True)
    if sub.empty:
        raise ValueError("No feature windows match the task labels")
    feature_cols = [c for c in sub.columns if c not in meta]
    if not feature_cols or sub["subject_id"].isna().any():
        raise ValueError(
            "Task windows require features and nonmissing subject identifiers"
        )
    X = sub[feature_cols].to_numpy(dtype=float, copy=True)
    X[~np.isfinite(X)] = np.nan
    keep_cols = ~np.isnan(X).all(axis=0)
    X = X[:, keep_cols]
    feature_cols = [c for c, k in zip(feature_cols, keep_cols) if k]
    if not feature_cols:
        raise ValueError("Task contains no finite feature values")
    remap = {label: i for i, label in enumerate(keep)}
    y = sub["label"].map(remap).to_numpy()
    groups = sub["subject_id"].to_numpy()
    return X, y, groups, feature_cols, mask


def prepare_task(features_df, x_raw, keep):
    if len(x_raw) != len(features_df) or len(set(keep)) != len(keep) or not keep:
        raise ValueError(
            "Task labels must be distinct and raw windows must align with feature rows"
        )
    X, y, groups, feature_cols, mask = _prepare_features(
        features_df, keep, ["subject_id", "window_id", "label", "label_name"]
    )
    return X, y, groups, feature_cols, x_raw[mask]


def _run_benchmark(args, results_dir, figures_dir, models_dir):
    set_seed(SEED)

    if args.synthetic:
        from src.synthetic import features as synth_features

        print("Building synthetic dataset (demo only)...")
        features_df, x_raw, _ = synth_features(n_subjects=8, block_sec=150, cache=False)
    else:
        cached = None if (args.rebuild or args.subjects) else load_cached()
        if cached is None:
            print("Building dataset from raw WESAD...")
            features_df, x_raw, _ = WindowedDataset().build(subjects=args.subjects)
        else:
            print("Loaded cached dataset.")
            features_df, x_raw = cached

    print(
        f"Windows: {len(features_df)} | subjects: {features_df['subject_id'].nunique()}"
    )
    observed_labels = set(features_df["label"])
    for task, cfg in TASKS.items():
        missing = [
            name
            for label, name in zip(cfg["keep"], cfg["names"])
            if label not in observed_labels
        ]
        if missing:
            raise ValueError(
                f"{task} benchmark is missing required classes: {', '.join(missing)}"
            )
    summary = {}
    obsolete_figures = []
    source_context = {**provenance(), "source_file_sha256": pipeline_source_sha256()}
    inputs = {
        "dataset": "synthetic" if args.synthetic else "WESAD",
        "feature_frame_sha256": feature_frame_sha256(features_df),
    }
    if not args.synthetic:
        for name, key in (
            ("features.parquet", "feature_cache_sha256"),
            ("raw_windows.npz", "raw_windows_sha256"),
        ):
            path = PROCESSED_DATA_DIR / name
            if path.exists():
                inputs[key] = sha256_file(path)

    for task, cfg in TASKS.items():
        print(f"\n=== Task: {task} ===")
        X, y, groups, feature_cols, x_raw_task = prepare_task(
            features_df, x_raw, cfg["keep"]
        )
        rows, results = [], {}

        model_keys = [*CLASSIFIERS, *([] if args.no_cnn else ["cnn"])]
        for key in model_keys:
            res = (
                cnn_loso(x_raw_task, y, groups)
                if key == "cnn"
                else loso_evaluate(lambda k=key: build_pipeline(k), X, y, groups)
            )
            results[key] = res
            rows.append(
                {
                    "model": CLF_NAMES[key],
                    **{
                        metric: res[metric]
                        for metric in (
                            "accuracy_mean",
                            "accuracy_std",
                            "f1_macro_mean",
                            "balanced_accuracy",
                        )
                    },
                }
            )
            print(
                f"  {CLF_NAMES[key]:20s} acc={res['accuracy_mean']:.3f} f1={res['f1_macro_mean']:.3f}"
            )

        best_key, best_result = max(
            results.items(), key=lambda item: item[1]["accuracy_mean"]
        )
        best_name = CLF_NAMES[best_key]

        plot_model_comparison(rows, figures_dir / f"{task}_model_comparison.png")
        plot_confusion(
            best_result,
            cfg["names"],
            f"{task} ({best_name})",
            figures_dir / f"{task}_confusion.png",
        )
        plot_per_subject(
            best_result["per_subject"], figures_dir / f"{task}_per_subject.png"
        )
        plot_embedding(X, y, cfg["names"], figures_dir / f"{task}_pca.png")

        if best_key in CLASSIFIERS:
            m = nonoverlap_mask(groups)
            gap_factory = lambda k=best_key: build_pipeline(k)
            loso_matched = loso_evaluate(gap_factory, X[m], y[m], groups[m])[
                "pooled_accuracy"
            ]
            kf_acc = kfold_accuracy(gap_factory, X, y, groups)
            plot_gap(loso_matched, kf_acc, figures_dir / f"{task}_optimism_gap.png")
        else:
            loso_matched = None
            kf_acc = None
            obsolete_figures.append(f"{task}_optimism_gap.png")

        for suffix, frame in (
            ("model_comparison", pd.DataFrame(rows)),
            ("per_subject", best_result["per_subject"]),
        ):
            atomic_write_text(
                results_dir / f"{task}_{suffix}.csv", frame.to_csv(index=False)
            )

        summary[task] = {
            "n_windows": int(len(y)),
            "n_subjects": int(len(np.unique(groups))),
            "n_features": int(X.shape[1]),
            "feature_schema_version": FEATURE_SCHEMA_VERSION,
            "classes": cfg["names"],
            "models": rows,
            "best_model": best_name,
            "loso_accuracy": best_result["accuracy_mean"],
            "loso_pooled_accuracy": best_result["pooled_accuracy"],
            "loso_matched_accuracy": loso_matched,
            "within_subject_accuracy": kf_acc,
            "optimism_gap_pts": (
                round((kf_acc - loso_matched) * 100, 1)
                if kf_acc is not None and loso_matched is not None
                else None
            ),
            "per_subject": best_result["per_subject"].to_dict("records"),
        }

        if task == "binary":
            top_clf = max(CLASSIFIERS, key=lambda key: results[key]["accuracy_mean"])
            importance = shap_analysis(X, y, feature_cols, figures_dir)
            shap_path = results_dir / "shap_top_features.csv"
            atomic_write_text(shap_path, importance.to_csv(index=False))
            shap_record = {
                "model": "XGBoost",
                "scope": "full_data_binary_fit",
                "path": shap_path.name,
                "sha256": sha256_file(shap_path),
            }
            final = build_pipeline(top_clf)
            final.fit(X, y, **_fit_params(final, y))
            model_path = models_dir / "stress_classifier.joblib"
            save_verified_joblib(
                {
                    "pipeline": final,
                    "model_name": CLF_NAMES[top_clf],
                    "features": feature_cols,
                    "classes": cfg["names"],
                    "feature_schema_version": FEATURE_SCHEMA_VERSION,
                },
                model_path,
            )
            summary[task]["inference_model"] = CLF_NAMES[top_clf]
            summary["artifacts"] = {
                "shap": shap_record,
                "inference_model": {
                    "model": CLF_NAMES[top_clf],
                    "path": model_path.name,
                    "sha256": sha256_file(model_path),
                },
            }
            print(f"  Saved inference model ({CLF_NAMES[top_clf]}) + SHAP.")

    if source_context["source_file_sha256"] != pipeline_source_sha256():
        raise ValueError("Pipeline source changed during the primary benchmark")
    summary["provenance"] = source_context
    summary["inputs"] = inputs
    summary["benchmark_protocol_version"] = BENCHMARK_PROTOCOL_VERSION
    summary["methodology"] = {
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "cnn_validation": None if args.no_cnn else "training_subject_holdout",
        "cnn_normalization": None if args.no_cnn else "inner_training_windows_only",
        "f1_classes": "full_task_class_set",
    }
    write_json(results_dir / "metrics.json", summary)

    return obsolete_figures


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subjects", nargs="+", default=None)
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument("--no-cnn", action="store_true")
    parser.add_argument(
        "--synthetic", action="store_true", help="run on generated data, no WESAD"
    )
    args = parser.parse_args()

    results_dir = DEMO_DIR / "results" if args.synthetic else RESULTS_DIR
    figures_dir = DEMO_DIR / "figures" if args.synthetic else FIGURES_DIR
    models_dir = DEMO_DIR / "models" if args.synthetic else MODELS_DIR
    destinations = (results_dir, figures_dir, models_dir)
    with ExitStack() as stack:
        staging = []
        for destination in destinations:
            destination.mkdir(parents=True, exist_ok=True)
            staging.append(
                Path(stack.enter_context(TemporaryDirectory(dir=destination)))
            )
        existing_figures = set(plt.get_fignums())
        try:
            obsolete_figures = _run_benchmark(args, *staging)
        finally:
            for number in set(plt.get_fignums()) - existing_figures:
                plt.close(number)
        changes: dict[Path, Path | None] = {
            destination / path.name: path
            for directory, destination in zip(staging, destinations)
            for path in sorted(directory.iterdir())
            if path.is_file()
        }
        changes.update((figures_dir / name, None) for name in obsolete_figures)
        metrics_path = results_dir / "metrics.json"
        changes[metrics_path] = changes.pop(metrics_path)
        publish_files(changes)

    print(f"\nResults written to {results_dir}")
    if not args.synthetic:
        print("Run scripts/build_dashboard_data.py to refresh the dashboard.")


if __name__ == "__main__":
    run()
