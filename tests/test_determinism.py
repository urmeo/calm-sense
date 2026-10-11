import unittest
from importlib import import_module
import random
import numpy as np
from src.utils import set_seed


class DeterminismTests(unittest.TestCase):

    def _torch(self):
        try:
            return import_module("torch")
        except ImportError:
            self.skipTest("torch unavailable")

    def test_set_seed_makes_rngs_reproducible(self):
        set_seed(123)
        first = (random.random(), np.random.rand(3).tolist())
        set_seed(123)
        second = (random.random(), np.random.rand(3).tolist())
        self.assertEqual(first, second)

    def test_feature_extraction_is_deterministic(self):
        from src.synthetic import features

        feature_df_a, _, _ = features(n_subjects=2, block_sec=80, seed=5)
        feature_df_b, _, _ = features(n_subjects=2, block_sec=80, seed=5)
        meta = ("subject_id", "window_id", "label", "label_name")
        cols = [c for c in feature_df_a.columns if c not in meta]
        a = feature_df_a[cols].to_numpy(dtype=float)
        b = feature_df_b[cols].to_numpy(dtype=float)
        self.assertEqual(a.shape, b.shape)
        both_nan = np.isnan(a) & np.isnan(b)
        self.assertTrue(
            (np.isclose(a, b, equal_nan=False) | both_nan).all(),
            "Feature extraction is not reproducible with a fixed seed",
        )

    def test_cnn_training_is_deterministic(self):
        self._torch()
        from src.models.dl.cnn_1d import CNN1DClassifier

        rng = np.random.RandomState(0)
        n, channels, length = (24, 3, 256)
        X = rng.randn(n, channels, length).astype("float32")
        y = np.array([0, 1] * (n // 2))

        def train_and_predict():
            set_seed(7)
            clf = CNN1DClassifier(
                in_channels=channels, max_epochs=2, batch_size=8, random_state=7
            )
            clf.fit(X, y)
            return clf.predict_proba(X)

        self.assertTrue(
            np.allclose(train_and_predict(), train_and_predict(), atol=1e-06),
            "CNN training is not reproducible with a fixed seed",
        )
