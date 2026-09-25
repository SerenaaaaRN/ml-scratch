import numpy as np
from base import BaseEstimator, RegressorMixin
from ._utils import CRITERION_MAP, best_split, predict_single


class DecisionTreeRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, max_depth=None, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split

    def _build_tree(self, X, y, depth):

        n_samples = len(y)

        if self.max_depth is not None and depth >= self.max_depth:
            return {"value": np.mean(y)}

        if n_samples < self.min_samples_split:
            return {"value": np.mean(y)}

        if np.var(y) == 0:
            return {"value": np.mean(y)}

        criterion_func = CRITERION_MAP["mse"]
        feat_idx, threshold, gain = best_split(X, y, criterion_func)

        if feat_idx is None or gain <= 0:
            return {"value": np.mean(y)}

        left_mask = X[:, feat_idx] <= threshold
        right_mask = ~left_mask

        return {
            "feature": feat_idx,
            "threshold": threshold,
            "left": self._build_tree(X[left_mask], y[left_mask], depth + 1),
            "right": self._build_tree(X[right_mask], y[right_mask], depth + 1),
        }

    def fit(self, X, y):

        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)

        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if y.ndim > 1:
            y = y.ravel()

        self.n_features_in_ = X.shape[1]
        self.tree_ = self._build_tree(X, y, depth=0)

        return self

    def predict(self, X):

        self._check_is_fitted(["tree_"])
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features, but DecisionTreeRegressor expects "
                f"{self.n_features_in_} features."
            )

        return np.array([predict_single(self.tree_, x) for x in X])
