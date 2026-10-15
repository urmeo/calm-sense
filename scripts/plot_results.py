"""Draw documentation charts from saved results without retraining models."""

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import PercentFormatter

ROOT = Path(__file__).resolve().parents[1]
BLUE = "#2563eb"
TEAL = "#0f766e"
AMBER = "#b45309"
INK = "#17243b"
MUTED = "#52627a"
GRID = "#e2e8f0"


def _style():
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "text.color": INK,
            "axes.labelcolor": MUTED,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "axes.edgecolor": GRID,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.spines.left": False,
            "axes.titleweight": "bold",
            "axes.titlepad": 14,
            "axes.axisbelow": True,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "legend.frameon": False,
            "svg.fonttype": "none",
            "svg.hashsalt": "calmsense-results",
        }
    )


def _header(fig, title, subtitle):
    fig.text(0.05, 0.95, title, fontsize=20, weight="bold", va="top")
    fig.text(0.05, 0.89, subtitle, fontsize=11, color=MUTED, va="top")


def plot_models(metrics):
    fig, axes = plt.subplots(1, 2, figsize=(12, 6.3))
    _header(
        fig,
        "Accuracy on held-out subjects",
        "WESAD | Each subject is tested after training on the other subjects",
    )
    for ax, task, title, color in zip(
        axes,
        ("binary", "multiclass"),
        ("Baseline / stress", "Baseline / stress / amusement"),
        (BLUE, TEAL),
    ):
        data = metrics[task]
        models = data["models"]
        positions = np.arange(len(models))
        means = [model["accuracy_mean"] * 100 for model in models]
        deviations = [model["accuracy_std"] * 100 for model in models]
        ax.barh(
            positions,
            means,
            xerr=deviations,
            height=0.58,
            color=color,
            alpha=0.88,
            error_kw={"ecolor": INK, "elinewidth": 1.25, "capsize": 4},
        )
        for position, mean in zip(positions, means):
            ax.text(
                4,
                position,
                f"{mean:.1f}%",
                color="white",
                weight="bold",
                ha="left",
                va="center",
            )
        ax.set_yticks(positions, [model["model"] for model in models])
        ax.invert_yaxis()
        ax.set_xlim(0, 112)
        ax.set_xticks([0, 25, 50, 75, 100])
        ax.set_xlabel("Mean subject accuracy (%)", labelpad=12)
        ax.set_title(title, loc="left", fontsize=13)
        ax.grid(axis="x")
        ax.tick_params(axis="y", length=0, pad=8)
        ax.text(
            0,
            -0.22,
            f"{data['n_subjects']} subjects | {data['n_windows']:,} windows",
            transform=ax.transAxes,
            fontsize=10,
            color=MUTED,
        )
    fig.text(
        0.05,
        0.075,
        "Bars show subject means. Whiskers show subject SD, not confidence intervals.",
        fontsize=10,
        color=MUTED,
    )
    fig.text(
        0.05,
        0.035,
        f"Feature models use {metrics['binary']['n_features']} features."
        " The CNN uses five signal channels. Results use the recorded evaluation settings.",
        fontsize=10,
        color=MUTED,
    )
    fig.subplots_adjust(left=0.18, right=0.97, top=0.75, bottom=0.23, wspace=0.7)
    return fig


def plot_calibration(data):
    fig, (reliability, decision) = plt.subplots(1, 2, figsize=(12, 6.8))
    model = data.get("model")
    model_name = {
        "lr": "Logistic Regression",
        "rf": "Random Forest",
        "xgb": "XGBoost",
        "lgbm": "LightGBM",
    }.get(model, model or "Model not recorded")
    _header(
        fig,
        "Calibration and decision curves",
        f"{model_name} | Binary WESAD task | {data['n_windows']:,} held-out windows",
    )
    reliability.plot(
        [0.5, 1], [0.5, 1], color=MUTED, linestyle=":", label="Perfect calibration"
    )
    for key, label, color in (
        ("loso", "Original", BLUE),
        ("recalibrated_isotonic", "Isotonic", TEAL),
    ):
        rows = data[key]["reliability"]
        reliability.plot(
            [row["confidence"] for row in rows],
            [row["accuracy"] for row in rows],
            marker="o",
            linewidth=2,
            markersize=5,
            color=color,
            label=f"{label} | ECE {data[key]['ece']:.3f}",
        )
    reliability.set_title("Confidence and observed accuracy", loc="left", fontsize=13)
    reliability.set_xlabel("Mean predicted-class confidence", labelpad=12)
    reliability.set_ylabel("Accuracy within confidence bin", labelpad=12)
    reliability.set_xlim(0.48, 1.02)
    reliability.set_ylim(0, 1.04)
    reliability.xaxis.set_major_formatter(PercentFormatter(1))
    reliability.yaxis.set_major_formatter(PercentFormatter(1))
    reliability.grid()
    reliability.legend(loc="lower right", fontsize=10)

    curves = data["decision_curve"]
    thresholds = curves["thresholds"]
    for key, label, color, linestyle in (
        ("net_benefit_uncalibrated", "Original", BLUE, "-"),
        ("net_benefit_recalibrated", "Isotonic", TEAL, "-"),
        ("treat_all", "Alert everyone", AMBER, "--"),
    ):
        decision.plot(
            thresholds,
            curves[key],
            label=label,
            color=color,
            linestyle=linestyle,
            linewidth=2,
            marker="o" if linestyle == "-" else None,
            markersize=4,
        )
    decision.axhline(0, color=MUTED, linestyle=":", label="Alert no one")
    decision.set_title("Net benefit across alert thresholds", loc="left", fontsize=13)
    decision.set_xlabel("Stress-probability threshold", labelpad=12)
    decision.set_ylabel("Net benefit", labelpad=12)
    decision.set_xticks([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    decision.xaxis.set_major_formatter(PercentFormatter(1))
    decision.grid()
    decision.legend(loc="lower left", fontsize=10)
    fig.text(
        0.05,
        0.075,
        f"Reliability uses nonempty bins from {data['n_bins']} confidence bins."
        " Isotonic calibration uses training-subject out-of-fold predictions.",
        fontsize=10,
        color=MUTED,
    )
    fig.text(
        0.05,
        0.035,
        "Curves join saved observations without smoothing."
        " Decision curves are an offline analysis; no clinical alert threshold is validated.",
        fontsize=10,
        color=MUTED,
    )
    fig.subplots_adjust(left=0.08, right=0.97, top=0.75, bottom=0.23, wspace=0.35)
    return fig


def _roc_points(model):
    curve = model.get("roc_curve")
    if curve is None:
        return None
    if not isinstance(curve, dict):
        raise ValueError(f"{model['model']}: ROC points must be an object")
    fpr = np.asarray(curve.get("fpr"), dtype=float)
    tpr = np.asarray(curve.get("tpr"), dtype=float)
    if (
        fpr.ndim != 1
        or tpr.shape != fpr.shape
        or len(fpr) < 2
        or not np.isfinite([fpr, tpr]).all()
        or np.any((fpr < 0) | (fpr > 1) | (tpr < 0) | (tpr > 1))
        or np.any(np.diff(fpr) < 0)
        or np.any(np.diff(tpr) < 0)
        or not np.allclose([fpr[0], tpr[0], fpr[-1], tpr[-1]], [0, 0, 1, 1])
    ):
        raise ValueError(f"{model['model']}: invalid ROC points")
    area = float(np.sum(np.diff(fpr) * (tpr[:-1] + tpr[1:]) / 2))
    if not np.isclose(area, model["auroc"], rtol=1e-6, atol=1e-6):
        raise ValueError(f"{model['model']}: ROC points disagree with AUROC")
    return fpr, tpr


def _roc_source(results_dir):
    current_path = results_dir / "threshold_metrics.json"
    supplemental_path = results_dir / "roc_curves.json"
    if not current_path.exists() or not supplemental_path.exists():
        return current_path.name
    current_bytes = current_path.read_bytes()
    current = json.loads(current_bytes)
    models = [row for row in current["models"] if row.get("available")]
    if models and all(row.get("roc_curve") is not None for row in models):
        return current_path.name
    supplemental = json.loads(supplemental_path.read_text())
    if supplemental.get("provenance", {}).get(
        "threshold_metrics_sha256"
    ) != hashlib.sha256(current_bytes).hexdigest() or any(
        supplemental.get(key) != current.get(key) for key in ("task", "n_windows")
    ):
        return current_path.name
    supplemental_models = {
        row["model"]: row for row in supplemental["models"] if row.get("available")
    }
    if set(supplemental_models) != {row["model"] for row in models}:
        return current_path.name
    for model in models:
        saved = supplemental_models[model["model"]]
        if saved.get("roc_curve") is None or any(
            not np.isclose(saved[key], model[key], rtol=0, atol=1e-12)
            for key in ("auroc", "auprc")
        ):
            return current_path.name
    return supplemental_path.name


def plot_roc_auc(data):
    models = [model for model in data["models"] if model.get("available")]
    if not models or any(
        not np.isfinite(model["auroc"]) or not 0 <= model["auroc"] <= 1
        for model in models
    ):
        raise ValueError("ROC/AUC charts require available models with valid AUROC")
    points = [_roc_points(model) for model in models]
    n_curves = sum(point is not None for point in points)
    fig, (roc, scores) = plt.subplots(1, 2, figsize=(12, 6.8))
    subjects = f"{data['n_subjects']} subjects | " if "n_subjects" in data else ""
    _header(
        fig,
        "Saved ROC / AUC results",
        f"Binary WESAD task | {subjects}{data['n_windows']:,} pooled held-out windows"
        " | Stress is the positive class",
    )
    colors = (BLUE, TEAL, AMBER, "#7c3aed")
    if n_curves:
        roc.plot([0, 1], [0, 1], color=MUTED, linestyle=":", label="Chance reference")
        for index, (model, point) in enumerate(zip(models, points)):
            if point is not None:
                roc.plot(
                    *point,
                    color=colors[index % len(colors)],
                    linewidth=2,
                    label=f"{model['model']} | {model['auroc']:.4f}",
                )
        roc.set_xlabel("False positive rate", labelpad=12)
        roc.set_ylabel("True positive rate", labelpad=12)
        roc.set_xlim(0, 1)
        roc.set_ylim(0, 1.02)
        roc.xaxis.set_major_formatter(PercentFormatter(1))
        roc.yaxis.set_major_formatter(PercentFormatter(1))
        roc.set_aspect("equal", adjustable="box")
        roc.grid()
        roc.legend(loc="lower right", fontsize=9)
    else:
        roc.set_axis_off()
        roc.text(
            0.5,
            0.65,
            "ROC points were not saved.",
            transform=roc.transAxes,
            ha="center",
            fontsize=15,
            weight="bold",
        )
        roc.text(
            0.5,
            0.48,
            "Rerun threshold_metrics to draw curves.\n\n"
            "python -m scripts.threshold_metrics",
            transform=roc.transAxes,
            ha="center",
            fontsize=11,
            color=MUTED,
        )
    roc.set_title("ROC curves", loc="left", fontsize=13)
    for index, model in enumerate(models):
        scores.bar(
            index,
            model["auroc"],
            width=0.58,
            color=colors[index % len(colors)],
            alpha=0.88,
        )
        scores.text(
            index,
            model["auroc"] + 0.025,
            f"{model['auroc']:.4f}",
            ha="center",
            fontsize=10,
            weight="bold",
        )
    scores.set_xticks(
        range(len(models)),
        [model["model"].replace(" ", "\n") for model in models],
        fontsize=10,
    )
    scores.tick_params(axis="x", length=0, pad=10)
    scores.set_ylim(0, 1.1)
    scores.set_yticks([0, 0.25, 0.5, 0.75, 1])
    scores.set_ylabel("Area under the ROC curve", labelpad=12)
    scores.set_title("Recorded AUROC", loc="left", fontsize=13)
    scores.grid(axis="y")
    replay = data.get("methodology", {}).get("replay")
    fig.text(
        0.05,
        0.075,
        (
            "Replay of the recorded benchmark: original source and verified cached inputs."
            " Feature models only; no CNN curve."
            if replay
            else f"ROC points are available for {n_curves} of {len(models)} models."
            " Curves require saved false-positive and true-positive rates."
        ),
        fontsize=10,
        color=MUTED,
    )
    fig.text(
        0.05,
        0.035,
        "AUROC summarizes ranking across thresholds."
        " It does not measure probability calibration or validate an alert threshold.",
        fontsize=10,
        color=MUTED,
    )
    fig.subplots_adjust(left=0.08, right=0.97, top=0.75, bottom=0.23, wspace=0.35)
    return fig


def plot_transfer(data):
    fig, ax = plt.subplots(figsize=(12, 5.6))
    _header(
        fig,
        "Performance across datasets",
        f"Random Forest | Binary task | {data['n_shared_features']} shared features"
        " | Balanced accuracy weights classes equally",
    )
    comparisons = (
        ("within_wesad", "WESAD\nSubject held out", BLUE),
        ("within_noneeg", "Non-EEG\nSubject held out", TEAL),
        ("wesad_to_noneeg", "WESAD to Non-EEG\nDataset held out", AMBER),
        ("noneeg_to_wesad", "Non-EEG to WESAD\nDataset held out", AMBER),
    )
    for position, (key, label, color) in enumerate(comparisons):
        value = data[key]["balanced_accuracy"]
        ax.bar(position, value, width=0.58, color=color, alpha=0.88)
        ax.text(
            position,
            value + 0.025,
            f"{value:.1%}",
            ha="center",
            weight="bold",
            color=INK,
        )
    ax.axhline(0.5, color=MUTED, linestyle=":", linewidth=1.2, label="50% reference")
    ax.legend(loc="upper right", fontsize=10)
    ax.set_xticks(range(len(comparisons)), [row[1] for row in comparisons])
    ax.tick_params(axis="x", length=0, pad=10)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Balanced accuracy", labelpad=12)
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.grid(axis="y")
    fig.text(
        0.05,
        0.045,
        "Balanced accuracy pools held-out predictions. Transfer uses the full target dataset."
        " Sensor units and label protocols differ.",
        fontsize=10,
        color=MUTED,
    )
    fig.subplots_adjust(left=0.08, right=0.97, top=0.73, bottom=0.25)
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir", type=Path, default=ROOT / "outputs" / "results"
    )
    parser.add_argument("--output-dir", type=Path, default=ROOT / "docs" / "assets")
    args = parser.parse_args()
    _style()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    roc_source = _roc_source(args.results_dir)
    for source, filename, plot in (
        ("metrics.json", "research-overview.svg", plot_models),
        ("calibration.json", "calibration-curves.svg", plot_calibration),
        (roc_source, "roc-auc.svg", plot_roc_auc),
        ("cross_dataset.json", "transfer.svg", plot_transfer),
    ):
        source_path = args.results_dir / source
        if not source_path.exists():
            print(f"Skipped {filename}: {source_path} is missing")
            continue
        fig = plot(json.loads(source_path.read_text()))
        target = args.output_dir / filename
        fig.savefig(target, metadata={"Date": None})
        plt.close(fig)
        target.write_text(
            "\n".join(line.rstrip() for line in target.read_text().splitlines()) + "\n"
        )
        print(target)


if __name__ == "__main__":
    main()
