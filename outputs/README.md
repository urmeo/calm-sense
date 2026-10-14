# Saved results and charts

The saved benchmark was recorded on 4 October 2026. [provenance.json](results/provenance.json) records its source revision, environment, dataset checksums, and artifact hashes. The [methodology](../docs/methodology.md) explains the evaluation conditions.

![Recorded binary and three-class model results, with subject variation](../docs/assets/research-overview.svg)

## Plot index

| Question | Saved plot | Values |
| :-- | :-- | :-- |
| How do the five models compare? | [Binary](figures/binary_model_comparison.png), [three-class](figures/multiclass_model_comparison.png) | [metrics.json](results/metrics.json) |
| How well do stress probabilities rank the two classes? | [ROC curves and AUROC](../docs/assets/roc-auc.svg) | [roc_curves.json](results/roc_curves.json), [threshold_metrics.json](results/threshold_metrics.json) |
| Which classes are confused? | [Binary confusion matrix](figures/binary_confusion.png), [three-class confusion matrix](figures/multiclass_confusion.png) | [metrics.json](results/metrics.json) |
| How much do subjects differ? | [Binary subject scores](figures/binary_per_subject.png) | [binary_per_subject.csv](results/binary_per_subject.csv) |
| What changes with subject-mixed testing? | [Evaluation comparison](figures/binary_optimism_gap.png) | [metrics.json](results/metrics.json) |
| Which feature groups help? | [Feature ablation](figures/ablation.png) | [ablation.csv](results/ablation.csv) |
| Do predicted probabilities match outcomes? | [Reliability curves](figures/calibration_reliability.png), [decision curves](figures/calibration_decision_curve.png) | [calibration.json](results/calibration.json) |
| What changes after enrollment? | [Personalization](figures/personalization.png) | [personalization.json](results/personalization.json) |
| What changes between chest and wrist? | [Device comparison](figures/chest_vs_wrist.png) | [wrist.json](results/wrist.json) |
| How well do models transfer between datasets? | [Transfer comparison](figures/cross_dataset.png) | [cross_dataset.json](results/cross_dataset.json) |
| Which features affect the full-data XGBoost fit? | [SHAP beeswarm](figures/shap_beeswarm.png) | [shap_top_features.csv](results/shap_top_features.csv) |

The [calibration curves](../docs/assets/calibration-curves.svg) and [transfer chart](../docs/assets/transfer.svg) also summarize the saved metrics. Regenerate these SVG charts with `python -m scripts.plot_results` from the repository root.

The ROC curves use [roc_curves.json](results/roc_curves.json), a supplemental replay made on 12 October 2026 (Asia/Dhaka). It uses archived source `4f5c02f` and verified cached WESAD features. AUROC and AUPRC match the original threshold file exactly for all four models. Its provenance records the replay source, cache hashes, and comparisons. The original 33 research files are unchanged. This replay does not measure later code changes.

After running a new compatible benchmark, save its measured ROC points and redraw the charts:

```bash
python -m scripts.threshold_metrics
python -m scripts.plot_results
```

The first command replaces `results/threshold_metrics.json` and requires cached real-data features. The chart generator uses a supplemental ROC file only when it matches the current threshold file. Follow the [reproduction guide](../docs/reproduction.md#additional-analyses) to keep the analyses consistent with the benchmark.

Binary AUROC, AUPRC, and the selected Random Forest threshold are recorded in [threshold_metrics.json](results/threshold_metrics.json). [stats.json](results/stats.json) contains bootstrap intervals and paired model tests; [tuning.json](results/tuning.json) contains nested tuning results. These analyses use different aggregation and evaluation settings, as described in the [methodology](../docs/methodology.md).

## Files by purpose

```text
outputs/
├── results/       CSV and JSON research measurements
├── figures/       Recorded plots
├── models/        Uncalibrated chest Random Forest and its checksum
├── dashboard/     Saved TypeScript results and recorded signal excerpts
└── generated/     Local files; ignored by Git
    ├── cache/     Extracted features
    ├── demo/      Synthetic examples
    ├── figures/   New plots from real-data reruns
    ├── logs/      Python logs
    └── site/      Dashboard build
```

The [saved model](models/stress_classifier.joblib) was refitted on all 869 binary windows and 58 features with scikit-learn 1.6.1. Its training fit is separate from the held-out benchmark. Use `load_verified_joblib` in [src/utils.py](../src/utils.py) to check its SHA-256 file before loading it from a trusted checkout.

The dashboard reads [results.ts](dashboard/results.ts) and [signals.ts](dashboard/signals.ts). Build it with `node frontend/tooling.mjs build` from the repository root. The signal excerpts are recorded WESAD data, resampled for display.

Real-data experiment scripts can replace `results/` and `models/`. New plots go to `generated/figures/`; synthetic examples go to `generated/demo/`. Follow the [reproduction guide](../docs/reproduction.md) to rerun analyses and refresh dashboard exports.
