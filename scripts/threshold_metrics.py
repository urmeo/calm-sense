"""Save pooled LOSO ROC curves.

Youden-J selection and scoring share labels; operating points are exploratory.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve
from sklearn.model_selection import LeaveOneGroupOut

from scripts.calibration import _pooled_proba
from scripts.run_experiment import CLF_NAMES, build_pipeline, load_cached, prepare_task
from src.calibration import brier_score
from src.config import RESULTS_DIR
from src.utils import write_json, analysis_provenance, benchmark_reference

FEATURE_MODELS = ["lr", "rf", "xgb", "lgbm"]
POINT_MODEL = "rf"


def loso_pos_proba(key, X, y, groups):
    true, proba, _ = _pooled_proba(
        lambda: build_pipeline(key),
        X,
        y,
        groups,
        LeaveOneGroupOut().split(X, y, groups),
    )
    return true, proba[:, 1]


def operating_point(y_true, p1):
    y_true = np.asarray(y_true)
    p1 = np.asarray(p1, dtype=float)
    brier_score(y_true, p1)
    if len(np.unique(y_true)) != 2:
        raise ValueError(
            "An ROC operating point requires both baseline and stress labels"
        )
    fpr, tpr, thr = roc_curve(y_true, p1)
    candidates = np.flatnonzero(np.isfinite(thr))
    j = int(candidates[np.argmax((tpr - fpr)[candidates])])
    t = float(thr[j])
    pred = (p1 >= t).astype(int)
    tp = int(np.sum((pred == 1) & (y_true == 1)))
    fp = int(np.sum((pred == 1) & (y_true == 0)))
    tn = int(np.sum((pred == 0) & (y_true == 0)))
    fn = int(np.sum((pred == 0) & (y_true == 1)))
    safe = lambda num, den: float(num / den) if den else None
    return {
        "rule": "Youden J (max sensitivity + specificity - 1)",
        "threshold": t,
        "sensitivity": safe(tp, tp + fn),
        "specificity": safe(tn, tn + fp),
        "ppv": safe(tp, tp + fp),
        "npv": safe(tn, tn + fn),
    }


def run():
    cached = load_cached()
    if cached is None:
        raise SystemExit("No cached features. Run scripts/run_experiment.py first.")
    features_df, x_raw = cached
    benchmark_sha = benchmark_reference(RESULTS_DIR, features_df)
    X, y, groups, _, _ = prepare_task(features_df, x_raw, [1, 2])

    out = {"task": "binary", "n_windows": int(len(y)), "models": []}
    for key in FEATURE_MODELS:
        name = CLF_NAMES[key]
        try:
            y_true, p1 = loso_pos_proba(key, X, y, groups)
            brier_score(y_true, p1)
            if len(np.unique(y_true)) != 2:
                raise ValueError(
                    "AUROC and AUPRC require both baseline and stress labels"
                )
            row = {
                "model": name,
                "available": True,
                "auroc": float(roc_auc_score(y_true, p1)),
                "auprc": float(average_precision_score(y_true, p1)),
            }
            if not np.isfinite([row["auroc"], row["auprc"]]).all():
                raise ValueError(
                    "AUROC and AUPRC require finite scores and both binary classes"
                )
            fpr, tpr, thresholds = roc_curve(y_true, p1)
            row["roc_curve"] = {
                "fpr": fpr.tolist(),
                "tpr": tpr.tolist(),
                "thresholds": [
                    float(threshold) if np.isfinite(threshold) else None
                    for threshold in thresholds
                ],
            }
            if key == POINT_MODEL:
                row["operating_point"] = operating_point(y_true, p1)
        except Exception as e:
            out["models"].append(
                {"model": name, "available": False, "reason": str(e)[:80]}
            )
            print(f"  {name:20s} skipped ({str(e)[:40]})")
            continue
        out["models"].append(row)
        print(f"  {name:20s} AUROC={row['auroc']:.3f}  AUPRC={row['auprc']:.3f}")

    out["provenance"] = analysis_provenance(RESULTS_DIR, benchmark_sha)
    out["methodology"] = {
        "xgboost_balancing": "training_fold_sample_weights",
        "roc_curve": "pooled held-out LOSO probabilities; stress is positive",
        "roc_thresholds": "null denotes an infinite threshold (predict no stress)",
    }
    path = RESULTS_DIR / "threshold_metrics.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json(path, out)
    print(f"\nWrote {path}")


if __name__ == "__main__":
    run()
