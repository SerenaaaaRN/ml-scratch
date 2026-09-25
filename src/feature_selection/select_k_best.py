import numpy as np
from base import BaseEstimator, TransformerMixin
from ._score_functions import SCORE_FUNC_MAP


class SelectKBest(BaseEstimator, TransformerMixin):
    def __init__(self, k=10, score_func="correlation"):
        self.k = k
        self.score_func = score_func

    def fit(self, X, y):

        if self.score_func not in SCORE_FUNC_MAP:
            raise ValueError(
                f"Unknown score_func='{self.score_func}'. "
                f"Available: {list(SCORE_FUNC_MAP.keys())}"
            )

        if not isinstance(self.k, int) or self.k < 1:
            raise ValueError(f"k must be a positive integer, got {self.k}")

        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        self.n_features_in_ = X.shape[1]
        self.k_ = min(self.k, self.n_features_in_)

        # Compute scores
        score_fn = SCORE_FUNC_MAP[self.score_func]
        self.scores_ = score_fn(X, y)

        top_indices = np.argsort(self.scores_)[::-1][: self.k_]

        # Build boolean support mask
        self.support_ = np.zeros(self.n_features_in_, dtype=bool)
        self.support_[top_indices] = True

        return self

    def transform(self, X):

        self._check_is_fitted(["support_", "scores_"])
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features, but SelectKBest expects "
                f"{self.n_features_in_} features."
            )

        return X[:, self.support_]

    def get_support(self):

        self._check_is_fitted(["support_"])
        return self.support_
