import numpy as np
from base import BaseEstimator, ClassifierMixin
from ._utils import CRITERION_MAP, best_split, predict_single


class DecisionTreeClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, max_depth=None, min_samples_split=2, criterion="gini"):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.criterion = criterion

    def _build_tree(self, X, y, depth):
        n_samples = len(y)

        unique_classes = np.unique(y)
        if len(unique_classes) == 1:
            return {"value": unique_classes[0]}

        if self.max_depth is not None and depth >= self.max_depth:
            values, counts = np.unique(y, return_counts=True)
            return {"value": values[np.argmax(counts)]}

        if n_samples < self.min_samples_split:
            values, counts = np.unique(y, return_counts=True)
            return {"value": values[np.argmax(counts)]}

        criterion_func = CRITERION_MAP[self.criterion]
        feat_idx, threshold, gain = best_split(X, y, criterion_func)

        if feat_idx is None or gain <= 0:
            values, counts = np.unique(y, return_counts=True)
            return {"value": values[np.argmax(counts)]}

        left_mask = X[:, feat_idx] <= threshold
        right_mask = ~left_mask

        return {
            "feature": feat_idx,
            "threshold": threshold,
            "left": self._build_tree(X[left_mask], y[left_mask], depth + 1),
            "right": self._build_tree(X[right_mask], y[right_mask], depth + 1),
        }

    def fit(self, X, y):

        if self.criterion not in ("gini", "entropy"):
            raise ValueError(
                f"criterion must be 'gini' or 'entropy', got '{self.criterion}'"
            )

        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        self.n_features_in_ = X.shape[1]
        self.classes_ = np.unique(y)
        self.tree_ = self._build_tree(X, y, depth=0)

        return self

    def predict(self, X):

        self._check_is_fitted(["tree_", "classes_"])
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features, but DecisionTreeClassifier expects "
                f"{self.n_features_in_} features."
            )

        return np.array([predict_single(self.tree_, x) for x in X])
