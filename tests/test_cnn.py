import unittest
from contextlib import ExitStack
from unittest.mock import patch
from importlib import import_module
import numpy as np


class CnnTests(unittest.TestCase):

    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)

    def _torch(self):
        try:
            return import_module("torch")
        except ImportError:
            self.skipTest("torch unavailable")

    def test_cnn_generalizes_to_held_out_split(self):
        self._torch()
        from src.models.dl.cnn_1d import CNN1DClassifier

        rng = np.random.RandomState(0)
        n, c, length = (160, 5, 256)
        X = rng.randn(n, c, length).astype("float32")
        y = (rng.randn(n) > 0).astype(int)
        X[y == 1, 0, :] += 2.0
        X_tr, y_tr = (X[:120], y[:120])
        X_te, y_te = (X[120:], y[120:])
        model = CNN1DClassifier(in_channels=c, max_epochs=25, random_state=0)
        model.fit(X_tr, y_tr)
        proba = model.predict_proba(X_te)
        self.assertEqual(proba.shape, (len(y_te), 2))
        self.assertTrue(np.allclose(proba.sum(axis=1), 1.0, atol=0.0001))
        self.assertGreater((model.predict(X_te) == y_te).mean(), 0.7)

    def test_cnn_subject_holdout_normalizes_and_weights_training_only(self):
        self._torch()
        from src.models.dl import cnn_1d

        groups = np.repeat(["S0", "S1", "S2", "S3"], 6)
        y = np.tile([0, 0, 0, 0, 1, 1], 4)
        rng = np.random.RandomState(1)
        X = rng.randn(24, 1, 128).astype("float32")
        X += np.repeat([0, 10, 100, 1000], 6)[:, None, None]
        model = cnn_1d.CNN1DClassifier(
            in_channels=1, max_epochs=1, batch_size=8, random_state=3, device="cpu"
        )
        train, validation = model._validation_split(y, groups)
        self.assertTrue(set(groups[train]).isdisjoint(groups[validation]))
        np.testing.assert_array_equal(model._validation_split(y, groups)[0], train)
        seen_weights = []
        original = cnn_1d.compute_class_weight

        def weights(class_weight, *, classes, y):
            seen_weights.append(y.copy())
            return original(class_weight, classes=classes, y=y)

        self.stack.enter_context(patch.object(cnn_1d, "compute_class_weight", weights))
        model.fit(X, y, groups=groups)
        np.testing.assert_allclose(
            model._mean, X[train].mean(axis=(0, 2), keepdims=True)
        )
        np.testing.assert_allclose(
            model._std, X[train].std(axis=(0, 2), keepdims=True) + 1e-08
        )
        self.assertFalse(np.allclose(model._mean, X.mean(axis=(0, 2), keepdims=True)))
        np.testing.assert_array_equal(seen_weights[0], y[train])
        self.assertEqual(
            model.predict_proba(np.empty((0, 1, 128), dtype="float32")).shape, (0, 2)
        )

    def test_cnn_rejects_unfitted_and_nonfinite_predictions(self):
        self._torch()
        from sklearn.exceptions import NotFittedError
        from src.models.dl.cnn_1d import CNN1DClassifier

        model = CNN1DClassifier(in_channels=1, max_epochs=1)
        with self.assertRaises(NotFittedError):
            model.predict_proba(np.zeros((2, 1, 128)))
        with self.assertRaisesRegex(ValueError, "finite"):
            model.fit(np.full((8, 1, 128), np.nan), np.tile([0, 1], 4))

    def test_failed_cnn_fit_cannot_predict_partial_or_stale_weights(self):
        torch = self._torch()
        from sklearn.exceptions import NotFittedError
        from src.models.dl import cnn_1d

        class SmallNetwork(torch.nn.Module):
            def __init__(self, in_channels, n_classes):
                super().__init__()
                self.linear = torch.nn.Linear(in_channels, n_classes)

            def forward(self, X):
                return self.linear(X.mean(dim=2))

        rng = np.random.RandomState(3)
        X = rng.randn(8, 1, 61).astype("float32")
        y = np.tile([0, 1], 4)
        self.stack.enter_context(patch.object(cnn_1d, "_Net", SmallNetwork))
        model = cnn_1d.CNN1DClassifier(
            in_channels=1, max_epochs=1, val_fraction=0.25, device="cpu"
        )
        for previously_fitted in (False, True):
            with self.subTest(previously_fitted=previously_fitted):
                if previously_fitted:
                    model.fit(X, y)
                    self.assertEqual(model.predict_proba(X).shape, (8, 2))
                with patch.object(
                    torch.optim.AdamW, "step", side_effect=RuntimeError("fit failed")
                ):
                    with self.assertRaisesRegex(RuntimeError, "fit failed"):
                        model.fit(X, y + 2)
                with self.assertRaises(NotFittedError):
                    model.predict_proba(X)
                with self.assertRaises(NotFittedError):
                    model.predict(X)

    def test_cnn_retains_singleton_minimum_length_training_batch(self):
        self._torch()
        from src.models.dl.cnn_1d import CNN1DClassifier

        rng = np.random.RandomState(2)
        X = rng.randn(8, 1, 61).astype("float32")
        y = np.tile([0, 1], 4)
        model = CNN1DClassifier(
            in_channels=1,
            max_epochs=1,
            batch_size=5,
            val_fraction=0.25,
            random_state=2,
            device="cpu",
        )
        model.fit(X, y)
        proba = model.predict_proba(X)
        self.assertTrue(proba.shape == (8, 2) and np.isfinite(proba).all())

    def test_cnn_rejects_nonfinite_normalization_or_inference(self):
        torch = self._torch()
        from src.models.dl.cnn_1d import CNN1DClassifier

        for failure in ("overflow", "invalid_scale", "invalid_network"):
            with self.subTest(failure=failure):

                class FixedNetwork(torch.nn.Module):

                    def forward(self, X):
                        fill = float("nan") if failure == "invalid_network" else 0.0
                        return torch.full((len(X), 2), fill)

                model = CNN1DClassifier(in_channels=1, device="cpu")
                model.classes_ = np.array([0, 1])
                model.model = FixedNetwork()
                model._mean = np.zeros((1, 1, 1), dtype="float32")
                model._std = np.full((1, 1, 1), 1e-08, dtype="float32")
                X = np.zeros((2, 1, 61), dtype="float32")
                if failure == "overflow":
                    X.fill(1e38)
                elif failure == "invalid_scale":
                    model._std.fill(np.inf)
                with self.assertRaisesRegex(
                    ValueError, "CNN (normalization|inference)"
                ):
                    model.predict_proba(X)

    def test_cnn_group_split_fallback_finds_rare_class_cover_deterministically(self):
        self._torch()
        from src.models.dl.cnn_1d import CNN1DClassifier

        y = np.array([0] * 28 + [1, 2])
        groups = np.arange(30)
        model = CNN1DClassifier(val_fraction=0.9, random_state=42)
        train, validation = model._validation_split(y, groups)
        self.assertTrue(len(train) == 3 and len(validation) == 27)
        self.assertTrue(set(groups[train]).isdisjoint(groups[validation]))
        np.testing.assert_array_equal(np.unique(y[train]), [0, 1, 2])
        np.testing.assert_array_equal(model._validation_split(y, groups)[0], train)

    def test_cnn_group_split_rejects_impossible_class_cover_at_requested_fraction(self):
        self._torch()
        from src.models.dl.cnn_1d import CNN1DClassifier

        with self.assertRaisesRegex(ValueError, "retain every training class"):
            CNN1DClassifier(val_fraction=0.5)._validation_split(
                np.array([0, 1, 2]), np.arange(3)
            )
