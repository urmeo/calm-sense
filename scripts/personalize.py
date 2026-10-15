"""Enrollment labels calibrate probabilities; evaluation windows remain separate."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import LeaveOneGroupOut

from scripts.calibration import (
    _apply_calibrator,
    _fit_calibrator,
    _global_calibrator,
    _pos_proba,
)
from scripts.run_experiment import (
    _fit_params,
    build_pipeline,
    load_cached,
    prepare_task,
)
from src import calibration as cal
from src.config import DEMO_DIR, FIGURES_DIR, RESULTS_DIR, SEED
from src.utils import write_json, analysis_provenance, benchmark_reference

K_VALUES = [5, 10, 20]
METHOD = "isotonic"


def _stratified_split(y, frac, rng):
    y = np.asarray(y)
    if y.ndim != 1 or not len(y) or not 0 < frac < 1:
        raise ValueError(
            "Enrollment split requires labels and an evaluation fraction in (0, 1)"
        )
    ev, pool = [], []
    for c in np.unique(y):
        idx = np.where(y == c)[0]
        rng.shuffle(idx)
        n_eval = min(max(1, int(round(len(idx) * frac))), max(1, len(idx) - 1))
        ev.extend(idx[:n_eval])
        pool.extend(idx[n_eval:])
    return np.array(ev, dtype=int), np.array(pool, dtype=int)


def _sample_k(y_pool, k, rng):
    """Balanced enrollment may yield fewer than k windows."""
    if isinstance(k, bool) or not isinstance(k, (int, np.integer)) or k < 1:
        raise ValueError("Enrollment budgets must be positive integers")
    y_pool = np.asarray(y_pool)
    if y_pool.ndim != 1:
        raise ValueError("Enrollment labels must be one-dimensional")
    classes, counts = np.unique(y_pool, return_counts=True)
    if not len(classes):
        return np.array([], dtype=int)
    if k < len(classes):
        classes = rng.choice(classes, k, replace=False)
    per = min(max(1, k // len(classes)), int(counts.min()))
    picks = []
    for c in classes:
        idx = np.where(y_pool == c)[0]
        rng.shuffle(idx)
        picks.extend(idx[:per])
    return np.array(picks, dtype=int)


def _metrics(y, p_pos):
    proba = np.column_stack([1.0 - p_pos, p_pos])
    return cal.expected_calibration_error(y, proba), cal.brier_score(y, p_pos)


def compute(X, y, groups, model="rf", k_values=K_VALUES):
    k_values = list(k_values)
    if not k_values or len(set(k_values)) != len(k_values):
        raise ValueError("Provide at least one distinct positive enrollment budget")
    for k in k_values:
        _sample_k(np.array([], dtype=int), k, np.random.RandomState(SEED))
    factory = lambda: build_pipeline(model)
    logo = LeaveOneGroupOut()
    rng = np.random.RandomState(SEED)
    scores = {"uncalibrated": [], "global": [], **{k: [] for k in k_values}}

    for train_idx, test_idx in logo.split(X, y, groups):
        Xtr, ytr, gtr = X[train_idx], y[train_idx], groups[train_idx]
        base = factory()
        base.fit(Xtr, ytr, **_fit_params(base, ytr))
        raw = _pos_proba(base, X[test_idx])
        y_s = y[test_idx]
        raw, y_s = raw[::2], y_s[::2]
        if len(np.unique(y_s)) < 2:
            continue

        ev, pool = _stratified_split(y_s, 0.5, rng)
        raw_ev, y_ev = raw[ev], y_s[ev]
        global_calibrator = _global_calibrator(factory, Xtr, ytr, gtr, METHOD)
        global_probabilities = (
            raw_ev
            if global_calibrator is None
            else _apply_calibrator(global_calibrator, raw_ev, METHOD)
        )

        scores["uncalibrated"].append(_metrics(y_ev, raw_ev))
        scores["global"].append(_metrics(y_ev, global_probabilities))

        for k in k_values:
            pick = _sample_k(y_s[pool], k, rng)
            if len(np.unique(y_s[pool][pick])) < 2:
                scores[k].append(_metrics(y_ev, raw_ev))
                continue
            calib = _fit_calibrator(raw[pool][pick], y_s[pool][pick], METHOD)
            scores[k].append(_metrics(y_ev, _apply_calibrator(calib, raw_ev, METHOD)))

    if not scores["uncalibrated"]:
        raise ValueError(
            "Personalization requires a target subject with both baseline and stress windows"
        )

    def mean(rows):
        arr = np.array(rows)
        return {"ece": float(arr[:, 0].mean()), "brier": float(arr[:, 1].mean())}

    return {
        "model": model,
        "brier_definition": cal.BINARY_BRIER_DEFINITION,
        "eval_frac": 0.5,
        "k_values": k_values,
        "n_subjects": len(scores["uncalibrated"]),
        "uncalibrated": mean(scores["uncalibrated"]),
        "global": mean(scores["global"]),
        "fewshot": {str(k): mean(scores[k]) for k in k_values},
    }


def _plot(out, path):
    ks = out["k_values"]
    plt.figure(figsize=(6, 4))
    plt.axhline(
        out["uncalibrated"]["ece"], color="#95a5a6", ls=":", label="uncalibrated"
    )
    plt.axhline(
        out["global"]["ece"], color="#3498db", ls="--", label="global recalibration"
    )
    plt.plot(
        ks,
        [out["fewshot"][str(k)]["ece"] for k in ks],
        "o-",
        color="#2ecc71",
        label="few-shot",
    )
    plt.xlabel("Requested enrollment budget (windows)")
    plt.ylabel("ECE (mean over subjects)")
    plt.title("Calibration by enrollment budget")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def run(synthetic=False, model="rf"):
    results_dir = DEMO_DIR / "results" if synthetic else RESULTS_DIR
    figures_dir = DEMO_DIR / "figures" if synthetic else FIGURES_DIR
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    if synthetic:
        from src.synthetic import features

        print("Using synthetic data (demo only).")
        features_df, x_raw, _ = features(
            n_subjects=8, block_sec=150, seed=SEED, cache=False
        )
    else:
        cached = load_cached()
        if cached is None:
            raise SystemExit("No cached features. Run scripts/run_experiment.py first.")
        features_df, x_raw = cached

    benchmark_sha = benchmark_reference(results_dir, features_df)
    X, y, groups, _, _ = prepare_task(features_df, x_raw, [1, 2])
    out = compute(X, y, groups, model=model)

    out["provenance"] = analysis_provenance(results_dir, benchmark_sha)
    write_json(results_dir / "personalization.json", out)
    _plot(out, figures_dir / "personalization.png")

    print(f"\n{'condition':18s} {'ECE':>7s} {'Brier':>7s}")
    print(
        f"{'uncalibrated':18s} {out['uncalibrated']['ece']:>7.3f} {out['uncalibrated']['brier']:>7.3f}"
    )
    print(f"{'global':18s} {out['global']['ece']:>7.3f} {out['global']['brier']:>7.3f}")
    for k in out["k_values"]:
        f = out["fewshot"][str(k)]
        print(f"{'few-shot k=' + str(k):18s} {f['ece']:>7.3f} {f['brier']:>7.3f}")
    print(f"\nWrote {results_dir / 'personalization.json'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--model", default="rf")
    args = parser.parse_args()
    run(synthetic=args.synthetic, model=args.model)
