import numpy as np
from base import BaseEstimator, RegressorMixin
from ._utils import compute_distances, top_k_indices


class KNeighborsRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, n_neighbors=5):
        self.n_neighbors = n_neighbors

    def fit(self, X, y):

        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)

        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if y.ndim > 1:
            y = y.ravel()

        if self.n_neighbors > X.shape[0]:
            raise ValueError(
                f"n_neighbors={self.n_neighbors} cannot be greater than "
                f"n_samples={X.shape[0]}"
            )

        self.X_train_ = X
        self.y_train_ = y
        self.n_features_in_ = X.shape[1]

        return self

    def predict(self, X):

        self._check_is_fitted(["X_train_", "y_train_"])
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features, but KNeighborsRegressor expects "
                f"{self.n_features_in_} features."
            )

        distances = compute_distances(self.X_train_, X)
        knn_indices = top_k_indices(distances, self.n_neighbors)
        knn_targets = self.y_train_[knn_indices]  # (n_test, k)

        return np.mean(knn_targets, axis=1)
