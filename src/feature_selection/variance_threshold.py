import numpy as np
from base import BaseEstimator, TransformerMixin


class VarianceThreshold(BaseEstimator, TransformerMixin):
    def __init__(self, threshold=0.0):
        self.threshold = threshold

    def fit(self, X, y=None):

        if self.threshold < 0:
            raise ValueError(f"threshold must be non-negative, got {self.threshold}")

        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        self.n_features_in_ = X.shape[1]
        self.variances_ = np.var(X, axis=0, ddof=0)
        self.support_ = self.variances_ > self.threshold

        if not np.any(self.support_):
            raise ValueError(
                f"No feature has variance > {self.threshold}. "
                f"All {self.n_features_in_} features would be removed. "
                "Lower the threshold or check your data."
            )

        return self

    def transform(self, X):

        self._check_is_fitted(["support_", "variances_"])
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features, but VarianceThreshold expects "
                f"{self.n_features_in_} features."
            )

        return X[:, self.support_]

    def get_support(self):
        self._check_is_fitted(["support_"])
        return self.support_
