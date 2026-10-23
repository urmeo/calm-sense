from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory

from src.artifacts import publish_files
from src.config import SEED


def load_feature_data(cache_loader, *, synthetic=False, n_subjects=6, seed=SEED):
    if synthetic:
        from src.synthetic import features

        print("Using synthetic data (demo only).")
        frame, raw, _ = features(
            n_subjects=n_subjects, block_sec=150, seed=seed, cache=False
        )
        return frame, raw
    cached = cache_loader()
    if cached is None:
        raise SystemExit("No cached features. Run scripts/run_experiment.py first.")
    return cached


def analysis_directories(synthetic, demo_dir, results_dir, figures_dir):
    if synthetic:
        results_dir, figures_dir = demo_dir / "results", demo_dir / "figures"
    for directory in (results_dir, figures_dir):
        directory.mkdir(parents=True, exist_ok=True)
    return results_dir, figures_dir


@contextmanager
def stage_analysis_outputs(results_dir, figures_dir):
    with (
        TemporaryDirectory(dir=results_dir) as results,
        TemporaryDirectory(dir=figures_dir) as figures,
    ):
        staged = (Path(results), Path(figures))
        yield staged
        publish_files(
            {
                destination / path.name: path
                for directory, destination in zip(staged, (results_dir, figures_dir))
                for path in sorted(directory.iterdir())
                if path.is_file()
            }
        )


def save_figure(path: Path):
    import matplotlib.pyplot as plt

    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
