import sys
from itertools import combinations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from scipy.stats import friedmanchisquare, wilcoxon

from scripts.analysis import load_feature_data
from scripts.run_experiment import (
    CLASSIFIERS,
    CLF_NAMES,
    build_pipeline,
    load_cached,
    loso_evaluate,
    prepare_task,
)
from src.config import RESULTS_DIR, SEED
from src.utils import (
    paired_effect_size,
    write_json,
    analysis_provenance,
    benchmark_reference,
)


def per_subject_acc(res) -> dict:
    df = res["per_subject"]
    if df.empty or df["subject"].isna().any() or df["subject"].duplicated().any():
        raise ValueError("Statistics require one nonmissing score per subject")
    values = df["accuracy"].to_numpy(dtype=float)
    if not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
        raise ValueError(
            "Subject accuracy scores must be finite values between zero and one"
        )
    return dict(zip(df["subject"], df["accuracy"]))


def bootstrap_ci(values, n=10000, seed=SEED):
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or len(values) < 3 or not np.isfinite(values).all():
        raise ValueError("Subject bootstrap requires at least three finite scores")
    if isinstance(n, bool) or not isinstance(n, (int, np.integer)) or n < 1:
        raise ValueError("Bootstrap repetitions must be a positive integer")
    rng = np.random.RandomState(seed)
    means = [rng.choice(values, len(values), replace=True).mean() for _ in range(n)]
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def holm_bonferroni(pairs):
    if len({key for key, _ in pairs}) != len(pairs) or any(
        not np.isfinite(p) or not 0 <= p <= 1 for _, p in pairs
    ):
        raise ValueError(
            "Holm correction requires distinct comparisons and finite p-values in [0, 1]"
        )
    ordered = sorted(pairs, key=lambda kv: kv[1])
    m = len(ordered)
    corrected, running = {}, 0.0
    for rank, (key, p) in enumerate(ordered):
        running = max(running, min((m - rank) * p, 1.0))
        corrected[key] = running
    return corrected


def _friedman(vecs):
    scores = np.asarray(vecs, dtype=float)
    if scores.ndim != 2 or min(scores.shape) < 3 or not np.isfinite(scores).all():
        raise ValueError(
            "Friedman comparison requires at least three models and three finite paired subjects"
        )
    if np.all(scores == scores[0]):
        return 0.0, 1.0
    chi2, p_value = friedmanchisquare(*scores)
    return float(chi2), float(p_value)


def run():
    features_df, x_raw = load_feature_data(load_cached)
    benchmark_sha = benchmark_reference(RESULTS_DIR, features_df)
    X, y, groups, _, _ = prepare_task(features_df, x_raw, [1, 2])

    scores = {}
    for key in CLASSIFIERS:
        scores[key] = per_subject_acc(
            loso_evaluate(lambda k=key: build_pipeline(k), X, y, groups)
        )

    subject_sets = [set(scores[k]) for k in CLASSIFIERS]
    if any(subjects != subject_sets[0] for subjects in subject_sets[1:]):
        raise ValueError("Model statistics require identical held-out subject sets")
    subjects = sorted(subject_sets[0])
    vecs = {k: np.array([scores[k][s] for s in subjects]) for k in CLASSIFIERS}

    chi2, omnibus_p = _friedman([vecs[k] for k in CLASSIFIERS])

    raw = {}
    for a, b in combinations(CLASSIFIERS, 2):
        if np.array_equal(vecs[a], vecs[b]):
            raw[(a, b)] = 1.0
        else:
            raw[(a, b)] = float(wilcoxon(vecs[a], vecs[b]).pvalue)
    corrected = holm_bonferroni(list(raw.items()))

    best = max(CLASSIFIERS, key=lambda k: vecs[k].mean())
    lo, hi = bootstrap_ci(vecs[best])

    out = {
        "best_model": CLF_NAMES[best],
        "best_accuracy_mean": float(vecs[best].mean()),
        "best_ci95": [lo, hi],
        "omnibus_friedman": {"chi2": float(chi2), "p_value": float(omnibus_p)},
        "pairwise_holm": {
            f"{CLF_NAMES[a]} vs {CLF_NAMES[b]}": {
                "delta_mean": float(vecs[a].mean() - vecs[b].mean()),
                "p_raw": raw[(a, b)],
                "p_holm": corrected[(a, b)],
                "effect_size": paired_effect_size(vecs[a], vecs[b]),
            }
            for a, b in combinations(CLASSIFIERS, 2)
        },
        "provenance": analysis_provenance(RESULTS_DIR, benchmark_sha),
    }

    print(
        f"Best: {CLF_NAMES[best]}  acc={vecs[best].mean():.3f}  95% CI [{lo:.3f}, {hi:.3f}]"
    )
    print(f"Friedman omnibus: chi2={chi2:.2f}  p={omnibus_p:.3f}")
    print("Pairwise Wilcoxon (Holm-corrected):")
    for a, b in combinations(CLASSIFIERS, 2):
        d = out["pairwise_holm"][f"{CLF_NAMES[a]} vs {CLF_NAMES[b]}"]
        print(
            f"  {CLF_NAMES[a]:20s} vs {CLF_NAMES[b]:20s} Δ={d['delta_mean']:+.3f}  p_holm={d['p_holm']:.3f}"
        )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    write_json(RESULTS_DIR / "stats.json", out)
    print(f"\nWrote {RESULTS_DIR / 'stats.json'}")


if __name__ == "__main__":
    run()
