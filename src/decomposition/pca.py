import numpy as np
from base import BaseEstimator, TransformerMixin


class PCA(BaseEstimator, TransformerMixin):
    def __init__(self, n_components=None):
        self.n_components = n_components

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=np.float64)

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        n_samples, n_features = X.shape
        self.n_features_in_ = n_features

        if self.n_components is None:
            self.n_components = min(n_samples, n_features)

        elif isinstance(self.n_components, int):
            if self.n_components < 1:
                raise ValueError(f"n_components must be >= 1, got {self.n_components}")

            max_possible = min(n_samples, n_features)
            if self.n_components > max_possible:
                raise ValueError(
                    f"n_components={self.n_components} exceeds maximum possible components=min(n_samples, n_features)={max_possible}"
                )
            self.n_components_ = self.n_components

        else:
            raise TypeError(
                f"n_components must be int or None, got {type(self.n_components)}"
            )

        # center data
        self.mean_ = np.mean(X, axis=0)
        X_centered = X - self.mean_

        # covariance matrix (using n-1 for unbiased estimate)
        cov_matrix = (X_centered.T @ X_centered) / (n_samples - 1)

        # eigendecomposition
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

        # eigh returns ascending order → reverse to get descending
        idx_sorted = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx_sorted]
        eigenvectors = eigenvectors[:, idx_sorted]

        # clamp tiny negative eigenvalues from numerical errors
        eigenvalues = np.maximum(eigenvalues, 0.0)

        # select top n_components
        self.components_ = eigenvectors[
            :, : self.n_components_
        ].T  # (n_components, n_features)
        self.explained_variance_ = eigenvalues[: self.n_components_]

        total_variance = np.sum(eigenvalues)
        if total_variance > 0:
            self.explained_variance_ratio_ = self.explained_variance_ / total_variance
        else:
            self.explained_variance_ratio_ = np.zeros(self.n_components_)

        return self

    def transform(self, X):
        self._check_is_fitted(["components_", "mean_"])
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features, but PCA expects "
                f"{self.n_features_in_} features."
            )

        X_centered = X - self.mean_
        return X_centered @ self.components_.T

    def inverse_transform(self, X_transformed):
        self._check_is_fitted(["components_", "mean_"])
        X_transformed = np.asarray(X_transformed, dtype=np.float64)
        if X_transformed.ndim == 1:
            X_transformed = X_transformed.reshape(-1, 1)

        if X_transformed.shape[1] != self.n_components_:
            raise ValueError(
                f"X_transformed has {X_transformed.shape[1]} components, but PCA "
                f"has {self.n_components_} components."
            )

        return X_transformed @ self.components_ + self.mean_
