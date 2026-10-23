"""Cross-dataset generalization: WESAD <-> PhysioNet Non-EEG on a shared feature space."""

import argparse
import hashlib
import json
import sys
from importlib.metadata import version
from pathlib import Path
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

from scripts.run_experiment import build_pipeline, loso_evaluate
from scripts.analysis import save_figure, stage_analysis_outputs
from src.artifacts import publish_files
from src.cache import verified_input
from src.config import (
    FIGURES_DIR,
    FS,
    PROCESSED_DATA_DIR,
    PROJECT_ROOT,
    RESULTS_DIR,
    VALID_SUBJECTS,
    WESAD_DIR,
)
from src.datasets import non_eeg
from src.datasets.records import record_files
from src.portable import (
    OVERLAP,
    PORTABLE_FEATURE_COLUMNS,
    PORTABLE_SCHEMA_VERSION,
    PORTABLE_SIGNAL_UNITS,
    PURITY,
    WINDOW_SEC,
    wesad_portable,
)
from src.utils import (
    provenance,
    sha256_file,
    write_json,
    analysis_provenance,
    benchmark_reference,
)

META = ["subject", "label"]


def _cache_schema(dataset):
    if dataset not in {"wesad", "noneeg"}:
        raise ValueError(f"Unknown portable dataset: {dataset}")
    rates = (
        {"eda": FS.WRIST_EDA, "temp": FS.WRIST_TEMP}
        if dataset == "wesad"
        else {"eda": non_eeg.EDA_FS, "temp": non_eeg.TEMP_FS}
    )
    return {
        "schema_version": PORTABLE_SCHEMA_VERSION,
        "dataset": dataset,
        "slope_units": "per_second",
        "finite_sample_timestamps": "original_sample_positions",
        "signal_units": PORTABLE_SIGNAL_UNITS[dataset],
        "hr_window_boundary": "include_timestamps_in_[start,end)",
        "sampling_rates_hz": rates,
        "window_seconds": WINDOW_SEC,
        "overlap": OVERLAP,
        "wesad_label_purity": PURITY,
        "feature_columns": PORTABLE_FEATURE_COLUMNS,
        "extractor_sha256": {
            path: sha256_file(PROJECT_ROOT / path)
            for path in (
                "src/portable.py",
                "src/config.py",
                "src/preprocessing/filters.py",
                (
                    "src/data/loader.py"
                    if dataset == "wesad"
                    else "src/datasets/non_eeg.py"
                ),
                *(() if dataset == "wesad" else ("src/datasets/records.py",)),
            )
        },
        "extraction_packages": {
            name: version(name) for name in ("numpy", "scipy", "neurokit2")
        },
    }


def _validate_frame(frame):
    expected = PORTABLE_FEATURE_COLUMNS + META
    if (
        not isinstance(frame, pd.DataFrame)
        or frame.empty
        or len(frame.columns) != len(expected)
        or set(frame.columns) != set(expected)
    ):
        raise ValueError(
            "Portable cache requires 18 feature columns, subject, label and nonempty rows"
        )
    if (
        frame["subject"].isna().any()
        or frame["subject"].astype(str).str.strip().eq("").any()
        or not frame["label"].isin([0, 1]).all()
    ):
        raise ValueError(
            "Portable cache requires nonmissing subjects and binary labels"
        )
    if not all(
        pd.api.types.is_numeric_dtype(frame[column])
        for column in PORTABLE_FEATURE_COLUMNS
    ):
        raise ValueError("Portable features must be numeric")
    if np.isinf(frame[PORTABLE_FEATURE_COLUMNS].to_numpy(dtype=float)).any():
        raise ValueError("Portable features may contain NaN, but not infinity")


def _load_or_build_cache(dataset, builder, source_provenance=None, *, rebuild=False):
    if source_provenance is not None and not callable(source_provenance):
        raise TypeError("source_provenance must be callable to recheck raw inputs")
    read_source = (
        source_provenance
        if source_provenance is not None
        else lambda: _raw_provenance(dataset)
    )
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    cache = (
        PROCESSED_DATA_DIR / f"portable_{dataset}_v{PORTABLE_SCHEMA_VERSION}.parquet"
    )
    sidecar = cache.with_suffix(".json")
    schema = _cache_schema(dataset)
    source = read_source()
    raw_identity = _raw_identity(source)
    if cache.exists() and not rebuild:
        try:
            metadata = json.loads(sidecar.read_text())
        except (OSError, ValueError) as error:
            raise ValueError(
                f"Missing or invalid portable cache metadata: {sidecar}"
            ) from error
        if not isinstance(metadata, dict) or any(
            metadata.get(key) != value for key, value in schema.items()
        ):
            raise ValueError(
                f"Portable cache schema mismatch: {cache}; rebuild from raw data"
            )
        if _raw_identity(metadata.get("source_provenance")) != raw_identity:
            raise ValueError(
                f"Portable cache raw inputs changed: {cache}; rebuild from raw data"
            )
        with verified_input(cache) as (stream, digest):
            if metadata.get("cache_sha256") != digest:
                raise ValueError(f"Portable cache checksum mismatch: {cache}")
            frame = pd.read_parquet(stream)
    else:
        frame = builder()
        _validate_frame(frame)
        if _raw_identity(read_source()) != raw_identity:
            raise ValueError("Raw inputs changed during portable feature extraction")
        if _cache_schema(dataset) != schema:
            raise ValueError("Pipeline changed during portable feature extraction")
        with TemporaryDirectory(dir=PROCESSED_DATA_DIR) as temporary:
            new_cache = Path(temporary) / cache.name
            new_sidecar = Path(temporary) / sidecar.name
            frame.to_parquet(new_cache, index=False)
            metadata = {
                **schema,
                "cache_sha256": sha256_file(new_cache),
                "source_provenance": source,
            }
            new_sidecar.write_text(json.dumps(metadata, indent=2) + "\n")
            publish_files({cache: new_cache, sidecar: new_sidecar})
    _validate_frame(frame)
    return frame, metadata


def _raw_identity(source):
    if not isinstance(source, dict):
        raise ValueError("Missing portable raw input metadata; rebuild from raw data")
    manifest = source.get("raw_file_sha256")
    if not isinstance(manifest, dict) or not manifest:
        raise ValueError("Missing portable raw input manifest; rebuild from raw data")
    digest = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    if source.get("manifest_sha256") != digest:
        raise ValueError("Invalid portable raw input manifest; rebuild from raw data")
    return manifest, digest


def _raw_provenance(dataset):
    if dataset == "wesad":
        directory = WESAD_DIR
        files = [directory / sid / f"{sid}.pkl" for sid in VALID_SUBJECTS]
    elif dataset == "noneeg":
        directory = non_eeg.DATA_DIR
        files = []
        for subject in range(1, 21):
            for name in ("AccTempEDA", "SpO2HR"):
                record = directory / f"Subject{subject}_{name}"
                files.extend(record_files(record))
                if name == "AccTempEDA":
                    files.append(record.with_suffix(".atr"))
    else:
        raise ValueError(f"Unknown portable dataset: {dataset}")
    if not files:
        raise ValueError("Cannot verify an empty portable raw input manifest")
    manifest = {}
    resolved_directory = directory.resolve()
    for path in files:
        if not path.is_file():
            raise FileNotFoundError(f"Required raw input not found: {path}")
        with verified_input(path) as (_, digest):
            manifest[path.resolve().relative_to(resolved_directory).as_posix()] = digest
    return {
        "feature_source": "raw_reextraction",
        "raw_file_sha256": manifest,
        "manifest_sha256": hashlib.sha256(
            json.dumps(manifest, sort_keys=True).encode()
        ).hexdigest(),
        "extraction_context": _generation_context(),
    }


def _generation_context():
    files = (
        "src/portable.py",
        "src/datasets/non_eeg.py",
        "src/datasets/records.py",
        "scripts/cross_dataset.py",
        "scripts/run_experiment.py",
        "src/models/ml/classifiers.py",
        "src/config.py",
    )
    return {
        **provenance(),
        "source_file_sha256": {
            path: sha256_file(PROJECT_ROOT / path) for path in files
        },
    }


def _xy(df, feature_cols):
    _validate_frame(df)
    if (
        not feature_cols
        or len(feature_cols) != len(set(feature_cols))
        or not set(feature_cols).issubset(PORTABLE_FEATURE_COLUMNS)
    ):
        raise ValueError(
            "Select distinct portable feature columns, excluding subject and label"
        )
    X = df[feature_cols].to_numpy(dtype=float)
    return X, df["label"].to_numpy(dtype=int), df["subject"].to_numpy()


def transfer(train_df, test_df, feature_cols):
    Xtr, ytr, _ = _xy(train_df, feature_cols)
    Xte, yte, _ = _xy(test_df, feature_cols)
    pipe = build_pipeline("rf")
    pipe.fit(Xtr, ytr)
    pred = pipe.predict(Xte)
    return {
        "accuracy": float(accuracy_score(yte, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(yte, pred)),
        "f1_macro": float(
            f1_score(yte, pred, labels=[0, 1], average="macro", zero_division=0)
        ),
    }


def within(df, feature_cols):
    X, y, groups = _xy(df, feature_cols)
    res = loso_evaluate(lambda: build_pipeline("rf"), X, y, groups)
    return {
        "accuracy": res["accuracy_mean"],
        "f1_macro": res["f1_macro_mean"],
        "balanced_accuracy": res["balanced_accuracy"],
    }


def run(*, rebuild=False):
    benchmark_sha = benchmark_reference(RESULTS_DIR, shared_cache=False)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    wesad, wesad_cache = _load_or_build_cache("wesad", wesad_portable, rebuild=rebuild)
    noneeg, noneeg_cache = _load_or_build_cache(
        "noneeg", non_eeg.build, rebuild=rebuild
    )

    feature_cols = sorted((set(wesad.columns) & set(noneeg.columns)) - set(META))
    print(
        f"WESAD: {len(wesad)} windows | Non-EEG: {len(noneeg)} windows | shared features: {len(feature_cols)}"
    )
    print(
        f"WESAD balance: {np.bincount(wesad['label'].to_numpy(dtype=int))} | Non-EEG balance: {np.bincount(noneeg['label'].to_numpy(dtype=int))}"
    )

    out = {
        "n_shared_features": len(feature_cols),
        "methodology": {
            "portable_schema_version": PORTABLE_SCHEMA_VERSION,
            "slope_units": "per_second",
            "finite_sample_timestamps": "original_sample_positions",
            "signal_units": PORTABLE_SIGNAL_UNITS,
            "hr_window_boundary": "include_timestamps_in_[start,end)",
            "window_seconds": WINDOW_SEC,
            "overlap": OVERLAP,
            "packages": {
                name: version(name)
                for name in ("numpy", "scipy", "pandas", "scikit-learn", "neurokit2")
            },
        },
        "datasets": {
            name: {
                "n_windows": len(frame),
                "n_subjects": int(frame["subject"].nunique()),
                "label_counts": {
                    str(label): int(count)
                    for label, count in frame["label"]
                    .value_counts()
                    .sort_index()
                    .items()
                },
                "cache_sha256": metadata["cache_sha256"],
                "source_provenance": metadata.get("source_provenance", {}),
            }
            for name, frame, metadata in (
                ("wesad", wesad, wesad_cache),
                ("noneeg", noneeg, noneeg_cache),
            )
        },
        "within_wesad": within(wesad, feature_cols),
        "within_noneeg": within(noneeg, feature_cols),
        "wesad_to_noneeg": transfer(wesad, noneeg, feature_cols),
        "noneeg_to_wesad": transfer(noneeg, wesad, feature_cols),
    }
    out["provenance"] = {
        **_generation_context(),
        **analysis_provenance(RESULTS_DIR, benchmark_sha),
    }
    with stage_analysis_outputs(RESULTS_DIR, FIGURES_DIR) as (results, figures):
        write_json(results / "cross_dataset.json", out)

        print("\n              within-LOSO   cross-dataset")
        print(
            f"WESAD          acc={out['within_wesad']['accuracy']:.3f}     -> Non-EEG f1={out['wesad_to_noneeg']['f1_macro']:.3f} (bal-acc {out['wesad_to_noneeg']['balanced_accuracy']:.3f})"
        )
        print(
            f"Non-EEG        acc={out['within_noneeg']['accuracy']:.3f}     -> WESAD   f1={out['noneeg_to_wesad']['f1_macro']:.3f} (bal-acc {out['noneeg_to_wesad']['balanced_accuracy']:.3f})"
        )

        labels = [
            "WESAD\n(within)",
            "WESAD→\nNon-EEG",
            "Non-EEG\n(within)",
            "Non-EEG→\nWESAD",
        ]
        vals = [
            out["within_wesad"]["balanced_accuracy"],
            out["wesad_to_noneeg"]["balanced_accuracy"],
            out["within_noneeg"]["balanced_accuracy"],
            out["noneeg_to_wesad"]["balanced_accuracy"],
        ]
        colors = ["#3498db", "#e74c3c", "#2ecc71", "#e74c3c"]
        plt.figure(figsize=(6, 4))
        bars = plt.bar(labels, vals, color=colors)
        for b, v in zip(bars, vals):
            plt.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.2f}", ha="center")
        plt.axhline(0.5, color="gray", ls=":", label="chance")
        plt.ylabel("Balanced accuracy")
        plt.ylim(0, 1)
        plt.title("Within- vs cross-dataset generalization")
        plt.legend()
        save_figure(figures / "cross_dataset.png")
    print(
        f"\nWrote {RESULTS_DIR / 'cross_dataset.json'} and {FIGURES_DIR / 'cross_dataset.png'}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="re-extract portable caches from raw data",
    )
    run(rebuild=parser.parse_args().rebuild)
