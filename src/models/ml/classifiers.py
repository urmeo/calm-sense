"""Fixed defaults; tuning runs are separate."""

from importlib import import_module
from typing import Any, Dict

from ...config import SEED

HYPERPARAMS: Dict[str, Dict[str, Any]] = {
    "lr": {
        "C": 1.0,
        "penalty": "l2",
        "solver": "lbfgs",
        "max_iter": 1000,
        "class_weight": "balanced",
    },
    "rf": {
        "n_estimators": 200,
        "max_depth": 10,
        "min_samples_leaf": 5,
        "min_samples_split": 5,
        "max_features": "sqrt",
        "class_weight": "balanced",
    },
    "xgb": {
        "n_estimators": 200,
        "max_depth": 7,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "gamma": 0,
        "reg_alpha": 0,
        "reg_lambda": 1,
        "scale_pos_weight": 1,
        "eval_metric": "mlogloss",
    },
    "lgbm": {
        "num_leaves": 50,
        "n_estimators": 200,
        "learning_rate": 0.1,
        "max_depth": -1,
        "colsample_bytree": 0.8,
        "reg_alpha": 0,
        "reg_lambda": 0,
        "min_child_samples": 20,
        "class_weight": "balanced",
        "verbose": -1,
    },
}


_CLASSIFIERS = {
    "lr": ("sklearn.linear_model", "LogisticRegression"),
    "rf": ("sklearn.ensemble", "RandomForestClassifier"),
    "xgb": ("xgboost", "XGBClassifier"),
    "lgbm": ("lightgbm", "LGBMClassifier"),
}
_ALIASES = {
    "logistic": "lr",
    "random_forest": "rf",
    "xgboost": "xgb",
    "lightgbm": "lgbm",
}


def get_classifier(name: str, **kwargs: Any) -> Any:
    key = name.lower()
    key = _ALIASES.get(key, key)
    if key not in _CLASSIFIERS:
        available = sorted([*_CLASSIFIERS, *_ALIASES])
        raise ValueError(f"Unknown classifier: {name}. Available: {available}")
    module_name, class_name = _CLASSIFIERS[key]
    classifier = getattr(import_module(module_name), class_name)
    parameters = {**HYPERPARAMS[key], "random_state": SEED, "n_jobs": -1, **kwargs}
    return classifier(**parameters)
