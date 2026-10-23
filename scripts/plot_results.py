"""Draw documentation charts from saved results without retraining models."""

import argparse
import hashlib
import json
from datetime import date
from io import StringIO
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter

from src.artifacts import publish_files

ROOT = Path(__file__).resolve().parents[1]
BLUE = "#2563eb"
TEAL = "#0f766e"
AMBER = "#b45309"
PURPLE = "#7c3aed"
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


def _result_header(fig, title, subtitle):
    fig.text(0.045, 0.955, title, fontsize=20.5, weight="bold", va="top")
    fig.add_artist(
        Line2D(
            [0.045, 0.955],
            [0.900, 0.900],
            transform=fig.transFigure,
            color=GRID,
            linewidth=1,
        )
    )
    fig.text(0.045, 0.86, subtitle, color=MUTED, fontsize=11, va="top")


def _result_notes(fig, payload, notes):
    for y, note in zip((0.115, 0.075), notes):
        fig.text(0.045, y, note, color=MUTED, fontsize=9.5, va="top")
    fig._calmsense_metadata = {"plotted_data": payload, "scope": notes}
    return fig


def _unit_values(values, name):
    values = np.asarray(values, dtype=float)
    if (
        not values.size
        or not np.isfinite(values).all()
        or np.any((values < 0) | (values > 1))
    ):
        raise ValueError(f"{name} requires finite values in [0, 1]")
    return values


def plot_subject_results(metrics, binary, multiclass):
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 8.2), sharey=True)
    _result_header(
        fig,
        "Per-subject accuracy",
        f"WESAD chest  |  Binary: {metrics['binary']['n_subjects']} subjects  |  Three-class: {metrics['multiclass']['n_subjects']} subjects  |  One person held out per fold",
    )
    payload = {}
    for ax, task, color in zip(
        axes,
        ("binary", "multiclass"),
        (BLUE, TEAL),
    ):
        rows = {"binary": binary, "multiclass": multiclass}[task].copy()
        if len(rows) != metrics[task]["n_subjects"] or rows.subject.duplicated().any():
            raise ValueError(
                f"{task}: per-subject rows must match the recorded subject count"
            )
        _unit_values(rows.accuracy, "Subject accuracy")
        if (
            not np.isfinite(rows.n).all()
            or np.any(rows.n <= 0)
            or np.any(rows.n != np.floor(rows.n))
            or rows.n.sum() != metrics[task]["n_windows"]
        ):
            raise ValueError(
                f"{task}: per-subject window counts do not match the benchmark"
            )
        rows["subject_number"] = rows.subject.str[1:].astype(int)
        rows = rows.sort_values("subject_number")
        values = rows.accuracy.to_numpy() * 100
        mean = metrics[task]["loso_accuracy"] * 100
        if not np.isclose(values.mean(), mean, rtol=0, atol=1e-12):
            raise ValueError(f"{task}: per-subject mean disagrees with the benchmark")
        y = np.arange(len(rows))
        ax.hlines(y, 0, 100, color=GRID, linewidth=0.7)
        ax.scatter(values, y, color=color, s=48, zorder=3)
        ax.axvline(mean, color=color, linestyle="--", linewidth=1.1, alpha=0.65)
        for value, pos in zip(values, y):
            ax.text(value + 1.9, pos, f"{value:.1f}", va="center", fontsize=9.3)
        ax.set_yticks(y, rows.subject)
        ax.set_ylim(len(rows) - 0.3, -0.7)
        ax.set_xlim(0, 112)
        ax.set_xticks([0, 25, 50, 75, 100])
        ax.set_xlabel("Accuracy (%)")
        ax.set_title(
            f"{'Binary' if task == 'binary' else 'Three-class'} | {metrics[task]['best_model']}",
            loc="left",
            fontsize=13,
        )
        ax.tick_params(axis="y", length=0, pad=9)
        ax.text(
            0,
            -0.13,
            f"Subject mean: {mean:.1f}%  |  {metrics[task]['n_windows']:,} windows",
            transform=ax.transAxes,
            color=color,
            fontsize=10,
            weight="bold",
        )
        payload[task] = rows[["subject", "n", "accuracy"]].to_dict("records")
    fig.subplots_adjust(left=0.10, right=0.96, top=0.77, bottom=0.24, wspace=0.26)
    return _result_notes(
        fig,
        payload,
        [
            "Each point is one subject's accuracy. Dashed lines show the unweighted subject means.",
            "The tasks use different labels, classifiers and window sets; the panels are separate evaluations.",
        ],
    )


def plot_validation_gap(data):
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 6.3))
    _result_header(
        fig,
        "Matched windows, different validation",
        "Pooled accuracy  |  The same non-overlapping window subset is used within each task",
    )
    payload = {}
    for ax, task, title in zip(
        axes, ("binary", "multiclass"), ("Binary", "Three-class")
    ):
        row = data[task]
        values = (
            _unit_values(
                [row["loso_matched_accuracy"], row["within_subject_accuracy"]],
                "Matched accuracy",
            )
            * 100
        )
        if not np.isclose(
            round(values[1] - values[0], 1), row["optimism_gap_pts"], rtol=0, atol=1e-9
        ):
            raise ValueError(f"{task}: validation gap disagrees with matched scores")
        bars = ax.barh([0, 1], values, color=[BLUE, AMBER], height=0.48)
        for bar, value in zip(bars, values):
            ax.text(
                value - 2,
                bar.get_y() + bar.get_height() / 2,
                f"{value:.1f}%",
                color="white",
                ha="right",
                va="center",
                fontsize=12,
                weight="bold",
            )
        ax.set_yticks([0, 1], ["LOSO\nPerson held out", "Subject-mixed\n5-fold"])
        ax.set_ylim(1.8, -0.7)
        ax.set_xlim(0, 100)
        ax.set_xticks([0, 25, 50, 75, 100])
        ax.set_xlabel("Pooled accuracy (%)")
        ax.set_title(f"{title} | {row['best_model']}", loc="left", fontsize=13)
        ax.tick_params(axis="y", length=0, pad=9)
        ax.grid(axis="x")
        gap = values[1] - values[0]
        ax.text(
            0,
            -0.25,
            f"Subject-mixed difference: {gap:+.1f} percentage points",
            transform=ax.transAxes,
            color=AMBER,
            fontsize=10.5,
            weight="bold",
        )
        payload[task] = {
            key: row[key]
            for key in (
                "best_model",
                "loso_matched_accuracy",
                "within_subject_accuracy",
                "optimism_gap_pts",
            )
        }
    fig.subplots_adjust(left=0.15, right=0.97, top=0.73, bottom=0.31, wspace=0.52)
    return _result_notes(
        fig,
        payload,
        [
            "Subject-mixed folds share people across training and testing; they do not test new-person performance.",
            "These pooled matched-window scores differ from the subject-mean accuracies in the main model table.",
        ],
    )


def plot_sensor_analysis(ablation, wrist):
    fig, axes = plt.subplots(
        1, 2, figsize=(12.8, 7), gridspec_kw={"width_ratios": [1.5, 1]}
    )
    _result_header(
        fig,
        "Sensors and feature subsets",
        "Random Forest  |  Binary WESAD task  |  Mean accuracy across held-out subjects",
    )
    label_map = {
        "No motion (HRV+EDA+TEMP+RESP)": "Without motion",
        "Autonomic (HRV+EDA)": "HRV + EDA",
        "Motion only (ACC)": "Motion only",
    }
    labels = [label_map.get(label, label) for label in ablation.subset]
    _unit_values(ablation.accuracy_mean, "Ablation accuracy")
    if (
        not np.isfinite(ablation.n_features).all()
        or np.any(ablation.n_features <= 0)
        or np.any(ablation.n_features != np.floor(ablation.n_features))
    ):
        raise ValueError("Ablation feature counts must be positive")
    positions = np.arange(len(ablation))
    values = ablation.accuracy_mean.to_numpy() * 100
    axes[0].barh(
        positions,
        values,
        color=[AMBER if label == "Motion only" else BLUE for label in labels],
        height=0.54,
    )
    axes[0].set_yticks(
        positions,
        [f"{label} ({count})" for label, count in zip(labels, ablation.n_features)],
    )
    axes[0].set_ylim(len(ablation) - 0.3, -0.7)
    axes[0].set_title("Feature subsets | Chest", loc="left", fontsize=13)
    for pos, value in zip(positions, values):
        axes[0].text(
            value - 2,
            pos,
            f"{value:.1f}%",
            color="white",
            ha="right",
            va="center",
            weight="bold",
            fontsize=10,
        )
    placement = (
        _unit_values(
            [wrist["same_model_rf"][key] for key in ("chest", "wrist")],
            "Placement accuracy",
        )
        * 100
    )
    if not np.isclose(
        placement[0] - placement[1],
        wrist["same_model_rf"]["drop_pts"],
        rtol=0,
        atol=1e-9,
    ):
        raise ValueError("Placement difference disagrees with chest and wrist scores")
    axes[1].barh([0, 1], placement, color=[BLUE, TEAL], height=0.48)
    axes[1].set_yticks([0, 1], ["Chest", "Wrist"])
    axes[1].set_ylim(2.5, -0.8)
    axes[1].set_title("Placement | Same classifier", loc="left", fontsize=13)
    for pos, value in enumerate(placement):
        axes[1].text(
            value - 2,
            pos,
            f"{value:.1f}%",
            color="white",
            ha="right",
            va="center",
            weight="bold",
            fontsize=11,
        )
    axes[1].text(
        0,
        0.12,
        f"Recorded chest–wrist difference:\n{wrist['same_model_rf']['drop_pts']:.1f} percentage points",
        transform=axes[1].transAxes,
        fontsize=10.5,
        color=TEAL,
        linespacing=1.6,
    )
    for ax in axes:
        ax.set_xlim(0, 100)
        ax.set_xticks([0, 25, 50, 75, 100])
        ax.set_xlabel("Mean subject accuracy (%)")
        ax.grid(axis="x")
        ax.tick_params(axis="y", length=0, pad=9)
    fig.subplots_adjust(left=0.21, right=0.97, top=0.75, bottom=0.23, wspace=0.36)
    return _result_notes(
        fig,
        {
            "ablation": ablation[["subset", "n_features", "accuracy_mean"]].to_dict(
                "records"
            ),
            "same_model_rf": wrist["same_model_rf"],
        },
        [
            "Numbers in parentheses count retained features. Sensor placement was evaluated in a separate study.",
            "Feature subsets do not establish causal effects; task motion may contribute. No field equivalence is established.",
        ],
    )


def plot_personalization(data):
    if (
        data.get("model") != "rf"
        or data.get("brier_definition") != "positive_class_mse"
    ):
        raise ValueError(
            "Personalization chart requires RF stress-probability Brier scores"
        )
    variants = [data["uncalibrated"], data["global"]] + [
        data["fewshot"][str(k)] for k in data["k_values"]
    ]
    _unit_values(
        [[row[metric] for metric in ("ece", "brier")] for row in variants],
        "Personalization errors",
    )
    labels = ["No recalibration", "Global: training subjects"] + [
        f"Requested budget: {k}" for k in data["k_values"]
    ]
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 6.8), sharey=True)
    _result_header(
        fig,
        "Personalization: probability error",
        f"Random Forest  |  Means across {data['n_subjects']} subjects  |  Evaluation fraction: {data['eval_frac']:.0%}  |  Lower is better",
    )
    for ax, metric, title, upper, ticks in zip(
        axes,
        ("ece", "brier"),
        ("Expected calibration error (ECE)", "Brier score (stress probability)"),
        (0.20, 0.095),
        ([0, 0.05, 0.10, 0.15, 0.20], [0, 0.03, 0.06, 0.09]),
    ):
        values = [row[metric] for row in variants]
        upper = max(upper, max(values) * 1.3)
        positions = np.arange(len(values))
        ax.barh(
            positions,
            values,
            color=["#64748b", BLUE] + [TEAL] * len(data["k_values"]),
            height=0.5,
        )
        for pos, value in zip(positions, values):
            ax.text(
                value + upper * 0.025,
                pos,
                f"{value:.4f}",
                fontsize=10,
                va="center",
                weight="bold",
            )
        ax.set_yticks(positions, labels)
        ax.set_ylim(len(variants) - 0.4, -0.7)
        ax.set_xlim(0, upper)
        ax.set_xticks(ticks)
        ax.set_title(title, loc="left", fontsize=12)
        ax.tick_params(axis="y", length=0, pad=10)
        ax.grid(axis="x")
    fig.subplots_adjust(left=0.26, right=0.96, top=0.74, bottom=0.23, wspace=0.22)
    return _result_notes(
        fig,
        {
            "model": data["model"],
            "n_subjects": data["n_subjects"],
            "eval_frac": data["eval_frac"],
            "labels": labels,
            "variants": variants,
            "requested_budgets": data["k_values"],
        },
        [
            "Enrollment recalibrates probabilities on labeled windows; it does not retrain the classifier.",
            (
                "Budget 5 uses at most 4 windows. Class availability can reduce any requested budget."
                if data.get("provenance", {}).get("git_sha")
                == "4f5c02f05f6492bd5da4483b7a5328946969914c"
                else "Labels are requested enrollment budgets; class availability can reduce enrollment."
            ),
        ],
    )


def plot_feature_contributions(shap, metrics):
    data = shap.sort_values("mean_abs_shap", ascending=False).head(10)
    if (
        data.empty
        or not np.isfinite(data.mean_abs_shap).all()
        or np.any(data.mean_abs_shap < 0)
    ):
        raise ValueError("SHAP contributions must be finite and nonnegative")
    artifact = metrics.get("artifacts", {}).get("shap", {})
    if (
        artifact.get("model") != "XGBoost"
        or artifact.get("scope") != "full_data_binary_fit"
    ):
        raise ValueError("SHAP chart requires a recorded full-data binary XGBoost fit")
    families = {
        "ACC": ("Motion", AMBER),
        "HRV": ("HRV", BLUE),
        "EDA": ("EDA", TEAL),
        "RESP": ("Respiration", PURPLE),
    }
    labels = {
        "ACC_zero_crossings": "ACC · zero crossings",
        "HRV_MedianNN": "HRV · median NN interval",
        "ACC_std": "ACC · magnitude SD",
        "EDA_SCR_recovery_time_mean": "EDA · mean SCR half-recovery time",
        "EDA_SCR_amplitude_max": "EDA · maximum SCR amplitude",
        "HRV_MeanNN": "HRV · mean NN interval",
        "RESP_rate": "RESP · breathing rate",
        "EDA_SCL_max": "EDA · maximum tonic level",
        "ACC_peak_freq": "ACC · peak frequency",
        "ACC_magnitude": "ACC · mean magnitude",
    }
    fig, ax = plt.subplots(figsize=(12.8, 8))
    _result_header(
        fig,
        "XGBoost feature contributions",
        "Binary full-data fit  |  Top 10 saved mean absolute SHAP contributions",
    )
    positions = np.arange(len(data))
    colors = [
        families.get(key.split("_")[0], ("Other", MUTED))[1] for key in data.feature
    ]
    ax.barh(positions, data.mean_abs_shap, color=colors, height=0.6)
    ax.set_yticks(positions, [labels.get(key, key) for key in data.feature])
    ax.invert_yaxis()
    ax.set_xlim(0, max(1.55, float(data.mean_abs_shap.max()) * 1.17))
    ax.set_xlabel("Mean |SHAP|", labelpad=10)
    ax.set_xticks([0, 0.5, 1, 1.5])
    ax.tick_params(axis="y", length=0, pad=10)
    ax.grid(axis="x")
    for pos, value in zip(positions, data.mean_abs_shap):
        ax.text(
            value + 0.025, pos, f"{value:.3f}", va="center", fontsize=10, weight="bold"
        )
    fig.legend(
        handles=[
            Line2D([], [], color=color, marker="s", linestyle="", label=name)
            for name, color in families.values()
        ],
        loc="upper right",
        bbox_to_anchor=(0.97, 0.84),
        ncol=4,
        fontsize=9,
        handlelength=1,
        columnspacing=1.3,
    )
    fig.subplots_adjust(left=0.36, right=0.96, top=0.74, bottom=0.23)
    return _result_notes(
        fig,
        {
            "model": "XGBoost",
            "scope": "full_data_binary_fit",
            "features": data.to_dict("records"),
            "display_labels": labels,
        },
        [
            "NN means normal-to-normal heartbeat interval; SCR is a skin conductance response.",
            "These magnitudes describe the fitted model, not held-out accuracy, signed effects or causality.",
        ],
    )


def refresh_charts(results_dir: Path, output_dir: Path) -> None:
    roc_source = _roc_source(results_dir)
    registry = (
        (("metrics.json",), "research-overview.svg", plot_models),
        (("calibration.json",), "calibration-curves.svg", plot_calibration),
        ((roc_source,), "roc-auc.svg", plot_roc_auc),
        (("cross_dataset.json",), "transfer.svg", plot_transfer),
        (
            ("metrics.json", "binary_per_subject.csv", "multiclass_per_subject.csv"),
            "subject-results.svg",
            plot_subject_results,
        ),
        (("metrics.json",), "validation-gap.svg", plot_validation_gap),
        (("ablation.csv", "wrist.json"), "sensor-analysis.svg", plot_sensor_analysis),
        (
            ("personalization.json",),
            "personalization-results.svg",
            plot_personalization,
        ),
        (
            ("shap_top_features.csv", "metrics.json"),
            "feature-contributions.svg",
            plot_feature_contributions,
        ),
    )
    payloads = {
        name: (results_dir / name).read_bytes()
        for sources, _, _ in registry
        for name in sources
    }
    data = {
        name: (
            pd.read_csv(StringIO(payload.decode()))
            if name.endswith(".csv")
            else json.loads(payload)
        )
        for name, payload in payloads.items()
    }
    for name, value in data.items():
        if name.endswith(".json"):
            json.dumps(value, allow_nan=False)
    primary = hashlib.sha256(payloads["metrics.json"]).hexdigest()
    if not all(data["ablation.csv"]["benchmark_sha256"] == primary):
        raise ValueError("Ablation snapshot does not match the benchmark")
    for name in ("wrist.json", "personalization.json"):
        context = data[name].get("provenance", {})
        if context.get("primary_benchmark_sha256") != primary or context.get(
            "source_file_sha256"
        ) != data["metrics.json"].get("provenance", {}).get("source_file_sha256"):
            raise ValueError(f"{name}: snapshot does not match the benchmark source")
    artifact = data["metrics.json"].get("artifacts", {}).get("shap", {})
    if (
        artifact.get("path") != "shap_top_features.csv"
        or artifact.get("sha256")
        != hashlib.sha256(payloads["shap_top_features.csv"]).hexdigest()
    ):
        raise ValueError("SHAP snapshot does not match the benchmark artifact")
    _style()
    output_dir.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output_dir) as directory:
        staging = Path(directory)
        replacements = {}
        for sources, filename, plot in registry:
            staged = staging / filename
            existing_figures = set(plt.get_fignums())
            try:
                figure = plot(*(data[source] for source in sources))
                metadata = {"Date": None}
                if hasattr(figure, "_calmsense_metadata"):
                    context = data["metrics.json"]["provenance"]
                    recorded = date.fromisoformat(context["generated_at"][:10])
                    figure.text(
                        0.045,
                        0.027,
                        "Source: "
                        + "; ".join(sources)
                        + f"  |  Benchmark: {recorded.day} {recorded:%b %Y}; source {context['git_sha'][:7]}.",
                        color=MUTED,
                        fontsize=8.3,
                        va="bottom",
                    )
                    metadata.update(
                        Title=figure.texts[0].get_text(),
                        Description=json.dumps(
                            {
                                **figure._calmsense_metadata,
                                "source_files": {
                                    f"outputs/results/{source}": hashlib.sha256(
                                        payloads[source]
                                    ).hexdigest()
                                    for source in sources
                                },
                                "benchmark_source": context["git_sha"],
                            },
                            ensure_ascii=False,
                            sort_keys=True,
                            allow_nan=False,
                        ),
                    )
                    figure.canvas.draw()
                    bounds, renderer = figure.bbox, figure.canvas.get_renderer()
                    for text in figure.findobj(matplotlib.text.Text):
                        if text.get_visible() and text.get_text():
                            box = text.get_window_extent(renderer)
                            if (
                                box.width
                                and box.height
                                and any(
                                    not bounds.fully_contains(*point)
                                    for point in box.get_points()
                                )
                            ):
                                raise ValueError(
                                    f"{filename}: text extends beyond the figure"
                                )
                figure.savefig(staged, metadata=metadata)
            finally:
                for number in set(plt.get_fignums()) - existing_figures:
                    plt.close(number)
            staged.write_text(
                "\n".join(line.rstrip() for line in staged.read_text().splitlines())
                + "\n"
            )
            replacements[output_dir / filename] = staged
        if _roc_source(results_dir) != roc_source or any(
            (results_dir / name).read_bytes() != payload
            for name, payload in payloads.items()
        ):
            raise RuntimeError("Result sources changed while rendering charts")
        publish_files(replacements)
    for target in replacements:
        print(target)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir", type=Path, default=ROOT / "outputs" / "results"
    )
    parser.add_argument("--output-dir", type=Path, default=ROOT / "docs" / "assets")
    args = parser.parse_args()
    refresh_charts(args.results_dir, args.output_dir)


if __name__ == "__main__":
    main()
