import unittest

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from src.models.ml.classifiers import get_classifier


class ClassifierTests(unittest.TestCase):
    def test_lightgbm_default_preserves_fits_with_inactive_subsampling(self):
        for classes in (2, 3):
            with self.subTest(classes=classes):
                X, y = make_classification(
                    n_samples=360,
                    n_features=10,
                    n_informative=6,
                    n_redundant=2,
                    n_classes=classes,
                    n_clusters_per_class=1,
                    random_state=17,
                )
                X = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
                X_train, X_test, y_train, _ = train_test_split(
                    X, y, test_size=0.25, stratify=y, random_state=29
                )
                previous = get_classifier("lgbm", subsample=0.8, n_jobs=1)
                cleaned = get_classifier("lgbm", n_jobs=1)
                self.assertEqual(previous.get_params()["subsample_freq"], 0)
                self.assertEqual(cleaned.get_params()["subsample_freq"], 0)
                self.assertEqual(cleaned.get_params()["subsample"], 1.0)
                previous.fit(X_train, y_train)
                cleaned.fit(X_train, y_train)
                self.assertEqual(
                    previous.booster_.dump_model()["tree_info"],
                    cleaned.booster_.dump_model()["tree_info"],
                )
                np.testing.assert_array_equal(
                    previous.predict_proba(X_test), cleaned.predict_proba(X_test)
                )
                np.testing.assert_array_equal(
                    previous.predict(X_test), cleaned.predict(X_test)
                )

    def test_sampling_overrides_and_xgboost_default_are_preserved(self):
        self.assertEqual(get_classifier("xgb").get_params()["subsample"], 0.8)
        overridden = get_classifier(
            "lightgbm", subsample=0.6, subsample_freq=2, random_state=7, n_jobs=1
        )
        parameters = overridden.get_params()
        self.assertEqual(parameters["subsample"], 0.6)
        self.assertEqual(parameters["subsample_freq"], 2)
        self.assertEqual(parameters["random_state"], 7)
        self.assertEqual(parameters["n_jobs"], 1)
