# Running the experiments

Start with the Python installation steps in the [README](../README.md#run-locally). Run the commands below from the repository root with the virtual environment active. Defaults, paths, and seed `42` are defined in [src/config.py](../src/config.py).

The saved benchmark was generated on 4 October 2026 from source revision `4f5c02f`. Its [provenance record](../outputs/results/provenance.json) lists the package versions used. The dependency ranges in `requirements.txt` permit other versions; matching the protocol alone does not guarantee identical numbers across environments. The saved scikit-learn model requires version 1.6.1.

## Data

Raw recordings are not part of the repository. Use the official dataset sources and follow their terms. WESAD is distributed under a research agreement. Its archive is about 2 GB; the recorded subject pickles occupy about 13.8 GB after extraction. Allow additional disk space for the archive, feature caches, and outputs.

```bash
python scripts/download_data.py --wesad
python scripts/download_data.py --verify-wesad
```

The downloader installs WESAD at `data/raw/WESAD/` and checks each subject pickle against the repository's SHA-256 reference. You can extract the official archive yourself, preserving paths such as `data/raw/WESAD/S2/S2.pkl`, then run verification. WESAD files use Python pickle; load only files from a trusted source after verification.

The optional transfer analysis also needs PhysioNet Non-EEG:

```bash
python scripts/download_data.py --noneeg
python scripts/download_data.py --verify-noneeg
```

Non-EEG is installed at `data/external/noneeg/non-eeg-dataset-for-assessment-of-neurological-status-1.0.0/`. Verification checks the required files against the archive's checksum manifest. The project includes its own reader for the record and annotation formats; a separate WFDB package is not required.

## Primary benchmark

```bash
python scripts/run_experiment.py --rebuild
```

This extracts chest features, evaluates all five models on both tasks, writes summaries and figures, computes full-data XGBoost SHAP values, and refits a binary feature model on all available training windows. With the recorded results, that refit is Random Forest. The script saves the model and its SHA-256 file.

Later runs can reuse a compatible cache by omitting `--rebuild`. `--no-cnn` skips the CNN. `--subjects S2 S3 S4` limits extraction to specified subjects. These options change the experiment and do not reproduce the complete five-model, 15-subject benchmark.

Real-data scripts write to `outputs/results/` and can replace the tracked research artifacts. The primary script also replaces the saved model in `outputs/models/`. Use a separate checkout when preserving the original artifacts matters. Newly generated plots go to `outputs/generated/figures/`; they do not replace the tracked figure snapshots in `outputs/figures/`.

## Additional analyses

Run the primary benchmark first. Most analysis scripts require its compatible cached features and record the primary benchmark's hash. Run them from the same source revision as the primary benchmark so the provenance checks can match their inputs.

| Command | Output or purpose |
| :-- | :-- |
| `python scripts/threshold_metrics.py` | Pooled binary ROC coordinates, AUROC, AUPRC, and threshold metrics |
| `python scripts/stats.py` | Subject bootstrap intervals and paired model tests |
| `python scripts/ablation.py` | Binary Random Forest feature-subset comparisons |
| `python scripts/calibration.py` | Binary probability calibration and decision curves |
| `python scripts/personalize.py` | Recalibration with held-out-subject enrollment labels |
| `python scripts/tuning.py` | Nested grouped hyperparameter search for binary models |
| `python scripts/wrist.py` | Wrist LOSO evaluation and a chest comparison |
| `python scripts/cross_dataset.py --rebuild` | WESAD wrist and Non-EEG feature extraction and transfer |

The tuning script uses three inner subject folds by default. Calibration, personalization, and tuning also accept `--synthetic` for a dataset-free check. The primary benchmark supports `--synthetic --no-cnn` for a feature-model check. Synthetic artifacts go under `outputs/generated/demo/` and are separate from research results.

After a real-data rerun, record its environment before changing source code or dependencies:

```bash
python scripts/stamp_provenance.py
```

This command records an environment snapshot and hashes the current result files. It does not validate the experiments or prove how older files were generated. The tracked `validated_reproduction` record includes additional evidence from its original run.

## ROC and documentation charts

To draw the documentation charts from the saved CSV and JSON files, without training models:

```bash
python -m scripts.plot_results
```

This writes SVGs to `docs/assets/`. Use `--results-dir` and `--output-dir` to select other folders. The charts cover subject accuracy, ROC/AUROC, reliability, decision curves, and dataset transfer. They join measured points without smoothing.

The original `threshold_metrics.json` contains AUROC and AUPRC scores but no ROC coordinates. The supplemental [roc_curves.json](../outputs/results/roc_curves.json) records a later replay of its four feature models, using the original source revision, matching package versions, and cached inputs with matching file and frame hashes. Its provenance records those checks and the comparison with the original scores. This replay leaves the original benchmark artifacts intact.

For a new experiment, run the primary benchmark and threshold analysis from the same source revision:

```bash
python scripts/run_experiment.py --rebuild
python scripts/threshold_metrics.py
python -m scripts.plot_results
```

The threshold analysis saves pooled held-out false-positive rates, true-positive rates, and thresholds for each available feature model. Stress is the positive class. A `null` threshold represents the initial infinite threshold, which predicts no stress. The plotting script uses supplemental curves only when their reference and model scores match the current threshold results; otherwise it reads the current threshold file. If no coordinates are available, it displays AUROC scores and states that curves are missing.

## Refresh and build the dashboard

The dashboard imports saved TypeScript exports from `outputs/dashboard/`. It needs neither a Python server nor raw data to display them.

After rerunning analyses, regenerate the results export and README calibration tables:

```bash
python scripts/build_dashboard_data.py
python scripts/update_readme_tables.py
```

The results exporter checks artifact and benchmark references. It records whether the saved benchmark source matches the current source. If an optional analysis belongs to another benchmark, rerun that analysis or remove its stale local artifact before exporting. Retain the README's `AUTOGEN` markers; the table updater uses them.

To refresh the recorded signal excerpts, run `python scripts/export_signals.py` with WESAD present. This exports 40 seconds per eligible condition for S2, S3, and S4, resampled to 30 Hz for display. These excerpts are recorded data, not live sensor input. Dataset terms still apply to exported recordings.

Use Node.js 24 and npm for the frontend:

```bash
node frontend/tooling.mjs install
node frontend/tooling.mjs test
node frontend/tooling.mjs build
node frontend/tooling.mjs preview
```

The production build is written to `outputs/generated/site/`. Preview serves it at [http://127.0.0.1:4173/calm-sense/](http://127.0.0.1:4173/calm-sense/). For source changes, use `node frontend/tooling.mjs dev`, which serves [http://127.0.0.1:5173/calm-sense/](http://127.0.0.1:5173/calm-sense/) and rebuilds on changes.

The tooling writes local `package.json`, `package-lock.json`, and `tsconfig.json` files from the tracked files in `frontend/config/`. Edit those tracked sources when changing dependencies or TypeScript settings. See [CONTRIBUTING.md](../CONTRIBUTING.md) for the full development checks.
