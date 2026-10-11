from .ml import get_classifier

__all__ = ["get_classifier"]

try:
    from .dl import CNN1DClassifier
except ImportError:
    pass
else:
    __all__ = ["get_classifier", "CNN1DClassifier"]
