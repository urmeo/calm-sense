import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import wilcoxon
from sklearn.dummy import DummyClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut, StratifiedKFold

from scripts.run_experiment import (
    _fit_params,
    build_pipeline,
    load_cached,
    nonoverlap_mask,
    prepare_task,
)
from scripts.stats import bootstrap_ci
from src import calibration as cal
from src.config import DEMO_DIR, FIGURES_DIR, RESULTS_DIR, SEED
from src.utils import (
    paired_effect_size,
    provenance,
    set_seed,
    write_json,
    analysis_provenance,
    benchmark_reference,
)

POSITIVE = "stress"
N_BINS = 15


def _probability_folds(factory, X, y, splits):
    for train_idx, test_idx in splits:
        pipe = factory()
        pipe.fit(X[train_idx], y[train_idx], **_fit_params(pipe, y[train_idx]))
        yield test_idx, _pos_proba(pipe, X[test_idx]).copy()


def _pooled_proba(factory, X, y, groups, splits):
    indices, predictions = zip(*_probability_folds(factory, X, y, splits))
    order, positive = np.concatenate(indices), np.concatenate(predictions)
    return y[order], np.column_stack([1.0 - positive, positive]), groups[order]


def loso_proba(factory, X, y, groups):
    return _pooled_proba(factory, X, y, groups, LeaveOneGroupOut().split(X, y, groups))


def within_subject_proba(factory, X, y, groups):
    keep = nonoverlap_mask(groups)
    Xk, yk, gk = X[keep], y[keep], groups[keep]

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    return _pooled_proba(factory, Xk, yk, gk, skf.split(Xk, yk))


def _subject_brier(y, proba, g):
    return {s: cal.brier_score(y[g == s], proba[g == s]) for s in np.unique(g)}


def gap_significance(loso, within):
    if len(loso) < 3 or set(loso) != set(within):
        raise ValueError(
            "Paired calibration scores require the same set of at least three subjects"
        )
    subjects = sorted(loso)
    a = np.array([loso[s] for s in subjects])
    b = np.array([within[s] for s in subjects])
    gap = a - b
    if not np.isfinite(gap).all():
        raise ValueError("Paired calibration scores must be finite")
    pval = 1.0 if np.all(gap == 0) else float(wilcoxon(a, b).pvalue)
    lo, hi = bootstrap_ci(gap, seed=SEED)
    return {
        "n_subjects": len(subjects),
        "mean_brier_gap": float(gap.mean()),
        "ci95": [lo, hi],
        "wilcoxon_p": pval,
        "effect_size": paired_effect_size(a, b),
        "per_subject": {s: {"loso": loso[s], "within": within[s]} for s in subjects},
    }


def _fit_calibrator(raw, y, method):
    if method not in {"isotonic", "sigmoid"}:
        raise ValueError(f"Unknown calibration method: {method}")
    raw = np.asarray(raw, dtype=float)
    y = np.asarray(y)
    cal.brier_score(y, raw)
    if method == "isotonic":
        return IsotonicRegression(out_of_bounds="clip").fit(raw, y)
    if len(np.unique(y)) == 1:
        return DummyClassifier(strategy="constant", constant=y[0]).fit(
            raw.reshape(-1, 1), y
        )
    return LogisticRegression().fit(raw.reshape(-1, 1), y)


def _apply_calibrator(model, raw, method):
    if method not in {"isotonic", "sigmoid"}:
        raise ValueError(f"Unknown calibration method: {method}")
    raw = np.asarray(raw, dtype=float)
    if method == "isotonic":
        return model.transform(raw)
    return _pos_proba(model, raw.reshape(-1, 1))


def _pos_proba(estimator, X):
    proba = np.asarray(estimator.predict_proba(X))
    classes = np.asarray(estimator.classes_)
    if (
        classes.ndim != 1
        or not 1 <= len(classes) <= 2
        or len(np.unique(classes)) != len(classes)
        or not np.isin(classes, [0, 1]).all()
    ):
        raise ValueError(
            "Binary calibration requires estimator classes drawn from {0, 1}"
        )
    if (
        proba.shape != (len(X), len(classes))
        or not np.isfinite(proba).all()
        or np.any((proba < 0) | (proba > 1))
        or not np.allclose(proba.sum(axis=1), 1.0)
    ):
        raise ValueError(
            "Estimator probabilities must be finite, normalized, and match its classes"
        )
    if 1 in classes:
        return proba[:, np.flatnonzero(classes == 1)[0]]
    return np.zeros(len(X))


def _global_calibrator(factory, Xtr, ytr, gtr, method):
    n_groups = len(np.unique(gtr))
    if n_groups < 2:
        return None
    oof = np.zeros(len(ytr))
    inner = GroupKFold(n_splits=min(5, n_groups))
    for held, positive in _probability_folds(
        factory, Xtr, ytr, inner.split(Xtr, ytr, gtr)
    ):
        oof[held] = positive
    return _fit_calibrator(oof, ytr, method)


def loso_recalibrated_proba(factory, X, y, groups, method="isotonic"):
    """Training-subject OOF probabilities fit calibrators."""
    logo = LeaveOneGroupOut()
    yt, pp = [], []
    for train_idx, test_idx in logo.split(X, y, groups):
        Xtr, ytr, gtr = X[train_idx], y[train_idx], groups[train_idx]
        base = factory()
        base.fit(Xtr, ytr, **_fit_params(base, ytr))
        raw_te = _pos_proba(base, X[test_idx])

        calibrator = _global_calibrator(factory, Xtr, ytr, gtr, method)
        cal_pos = (
            raw_te
            if calibrator is None
            else _apply_calibrator(calibrator, raw_te, method)
        )

        pp.append(np.column_stack([1.0 - cal_pos, cal_pos]))
        yt.append(y[test_idx])
    return np.concatenate(yt), np.concatenate(pp)


def compute(X, y, groups, model="rf", n_bins=N_BINS):
    factory = lambda: build_pipeline(model)

    y_loso, p_loso, _ = loso_proba(factory, X, y, groups)
    y_iso, p_iso = loso_recalibrated_proba(factory, X, y, groups, "isotonic")
    y_sig, p_sig = loso_recalibrated_proba(factory, X, y, groups, "sigmoid")

    m = nonoverlap_mask(groups)
    y_within, p_within, g_within = within_subject_proba(factory, X, y, groups)
    y_lm, p_lm, g_lm = loso_proba(factory, X[m], y[m], groups[m])

    loso = cal.summary(y_loso, p_loso, n_bins)
    loso_matched = cal.summary(y_lm, p_lm, n_bins)
    within = cal.summary(y_within, p_within, n_bins)
    iso = cal.summary(y_iso, p_iso, n_bins)
    sig = cal.summary(y_sig, p_sig, n_bins)
    significance = gap_significance(
        _subject_brier(y_lm, p_lm, g_lm),
        _subject_brier(y_within, p_within, g_within),
    )

    if not np.array_equal(y_loso, y_iso):
        raise ValueError("LOSO label order diverged across passes")
    thresholds = np.round(np.arange(0.05, 0.61, 0.05), 2)
    prevalence = float(np.mean(y_loso == 1))
    decision = {
        "thresholds": thresholds.tolist(),
        "net_benefit_uncalibrated": cal.net_benefit(
            y_loso, p_loso[:, 1], thresholds
        ).tolist(),
        "net_benefit_recalibrated": cal.net_benefit(
            y_loso, p_iso[:, 1], thresholds
        ).tolist(),
        "treat_all": [
            prevalence - (1 - prevalence) * (t / (1 - t)) for t in thresholds
        ],
    }

    return {
        "model": model,
        "positive_class": POSITIVE,
        "n_windows": int(len(y_loso)),
        "n_bins": n_bins,
        "brier_definition": cal.BINARY_BRIER_DEFINITION,
        "loso": loso,
        "loso_matched": loso_matched,
        "within_subject": within,
        "recalibrated_isotonic": iso,
        "recalibrated_sigmoid": sig,
        "calibration_optimism_gap_ece": round(loso_matched["ece"] - within["ece"], 4),
        "recalibration_reduction_ece": round(loso["ece"] - iso["ece"], 4),
        "gap_significance": significance,
        "decision_curve": decision,
        "provenance": provenance(),
    }


def _plot_reliability(out, path):
    plt.figure(figsize=(5, 5))
    plt.plot([0, 1], [0, 1], "k:", label="perfect")
    for key, color, label in [
        ("loso", "#3498db", "LOSO, all windows"),
        ("recalibrated_isotonic", "#2ecc71", "LOSO, all windows + isotonic"),
    ]:
        rows = out[key]["reliability"]
        plt.plot(
            [r["confidence"] for r in rows],
            [r["accuracy"] for r in rows],
            "o-",
            color=color,
            label=f"{label} (ECE {out[key]['ece']:.3f})",
        )
    plt.xlabel("Confidence")
    plt.ylabel("Accuracy")
    plt.title("Reliability diagram")
    plt.legend(loc="upper left", fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def _plot_gap(out, path):
    keys = ["within_subject", "loso_matched"]
    labels = ["Subject-mixed", "LOSO matched"]
    eces = [out[k]["ece"] for k in keys]
    plt.figure(figsize=(4.5, 4))
    bars = plt.bar(labels, eces, color=["#e67e22", "#3498db"])
    for b, v in zip(bars, eces):
        plt.text(b.get_x() + b.get_width() / 2, v + 0.002, f"{v:.3f}", ha="center")
    plt.ylabel("Expected calibration error")
    plt.title("Calibration gap on matched non-overlapping windows")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def _plot_decision(out, path):
    d = out["decision_curve"]
    t = d["thresholds"]
    plt.figure(figsize=(5.5, 4))
    plt.plot(
        t, d["net_benefit_uncalibrated"], "o-", color="#3498db", label="uncalibrated"
    )
    plt.plot(
        t, d["net_benefit_recalibrated"], "o-", color="#2ecc71", label="recalibrated"
    )
    plt.plot(t, d["treat_all"], "--", color="gray", label="alert everyone")
    plt.axhline(0, color="black", lw=0.8, label="alert no one")
    plt.xlabel("Alert threshold")
    plt.ylabel("Net benefit")
    plt.title("Decision-curve analysis")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def run(synthetic=False, model="rf", n_bins=N_BINS):
    set_seed(SEED)
    results_dir = DEMO_DIR / "results" if synthetic else RESULTS_DIR
    figures_dir = DEMO_DIR / "figures" if synthetic else FIGURES_DIR
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    if synthetic:
        from src.synthetic import features

        print("Using synthetic data (demo only).")
        features_df, x_raw, _ = features(
            n_subjects=6, block_sec=150, seed=SEED, cache=False
        )
    else:
        cached = load_cached()
        if cached is None:
            raise SystemExit("No cached features. Run scripts/run_experiment.py first.")
        features_df, x_raw = cached

    benchmark_sha = benchmark_reference(results_dir, features_df)
    X, y, groups, _, _ = prepare_task(features_df, x_raw, [1, 2])
    out = compute(X, y, groups, model=model, n_bins=n_bins)
    out["provenance"] = analysis_provenance(results_dir, benchmark_sha)

    _plot_reliability(out, figures_dir / "calibration_reliability.png")
    _plot_gap(out, figures_dir / "calibration_gap.png")
    _plot_decision(out, figures_dir / "calibration_decision_curve.png")

    write_json(results_dir / "calibration.json", out)

    print(
        f"LOSO ECE {out['loso']['ece']:.3f} | within-subject ECE "
        f"{out['within_subject']['ece']:.3f} | recalibrated ECE "
        f"{out['recalibrated_isotonic']['ece']:.3f}"
    )
    print(
        f"Calibration optimism gap {out['calibration_optimism_gap_ece']:+.3f} ECE | "
        f"recalibration cuts ECE by {out['recalibration_reduction_ece']:+.3f}"
    )
    sig = out["gap_significance"]
    print(
        f"Per-subject Brier gap {sig['mean_brier_gap']:+.4f} "
        f"95% CI [{sig['ci95'][0]:.4f}, {sig['ci95'][1]:.4f}] Wilcoxon p={sig['wilcoxon_p']:.3f}"
    )
    if synthetic:
        print(
            "Note: synthetic stress is near-separable, so ECE is ~0 and the optimism gap is not "
            "meaningful. Run `python scripts/run_experiment.py` on real WESAD for benchmark results."
        )
    print(f"Wrote {results_dir / 'calibration.json'} and 3 figures.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--synthetic", action="store_true", help="run on generated demo data"
    )
    parser.add_argument("--model", default="rf")
    parser.add_argument("--bins", type=int, default=N_BINS)
    args = parser.parse_args()
    run(synthetic=args.synthetic, model=args.model, n_bins=args.bins)
