import warnings
import numpy as np

from base import BaseEstimator, RegressorMixin


class LinearRegression(BaseEstimator, RegressorMixin):
    def __init__(self, fit_intercept=True):
        self.fit_intercept = fit_intercept

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        if y.ndim == 1:
            y = y.reshape(-1, 1)
            self._y_ndim = 1
        else:
            self._y_ndim = y.ndim

        if X.shape[0] != y.shape[0]:
            raise ValueError(
                f"X and y have incompatible shapes: X has {X.shape[0]} samples, "
                f"y has {y.shape[0]} samples"
            )
        self.n_features_in_ = X.shape[1]

        if self.fit_intercept:
            X_aug = np.hstack([np.ones((X.shape[0], 1)), X])
        else:
            X_aug = X

        theta = np.linalg.pinv(X_aug) @ y

        if self.fit_intercept:
            self.intercept_ = theta[0].squeeze()
            self.coef_ = theta[1:].squeeze()
        else:
            self.intercept_ = 0.0
            self.coef_ = theta.squeeze()

        return self

    def predict(self, X):
        self._check_is_fitted(["coef_", "intercept_"])
        X = np.asarray(X, dtype=np.float64)

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features, but LinearRegression is expecting "
                f"{self.n_features_in_} features"
            )

        y_pred = X @ self.coef_ + self.intercept_

        if getattr(self, "_y_ndim", 1) == 1 and y_pred.ndim > 1:
            y_pred = y_pred.ravel()

        return y_pred


class Ridge(BaseEstimator, RegressorMixin):
    def __init__(self, alpha=0.1, fit_intercept=True):
        self.alpha = alpha
        self.fit_intercept = fit_intercept

    def fit(self, X, y):
        if self.alpha < 0:
            raise ValueError(f"alpha must be non-negative, got {self.alpha}")

        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        if y.ndim > 1:
            y = y.ravel()

        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y have incompatible shapes")

        self.n_features_in_ = X.shape[1]

        if self.fit_intercept:
            X_mean = np.mean(X, axis=0)
            y_mean = np.mean(y)
            X_centered = X - X_mean
            y_centered = y - y_mean
        else:
            X_centered = X
            y_centered = y
            X_mean = np.zeros(X.shape[1])
            y_mean = 0.0

        XtX = X_centered.T @ X_centered
        reg_matrix = self.alpha * np.eye(X.shape[1])
        A = XtX + reg_matrix

        try:
            self.coef_ = np.linalg.solve(A, X_centered.T @ y_centered)
        except np.linalg.LinAlgError:
            warnings.warn("Singular matrix in Ridge, falling back to pseudo-inverse")
            self.coef_ = np.linalg.pinv(A) @ (X_centered.T @ y_centered)

        if self.fit_intercept:
            self.intercept_ = y_mean - X_mean @ self.coef_
        else:
            self.intercept_ = 0.0

        return self

    def predict(self, X):
        self._check_is_fitted(["coef_", "intercept_"])
        X = np.asarray(X, dtype=np.float64)

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features, but Ridge is expecting "
                f"{self.n_features_in_} features"
            )

        return X @ self.coef_ + self.intercept_


class Lasso(BaseEstimator, RegressorMixin):
    def __init__(self, alpha=1.0, max_iter=1000, tol=1e-4, fit_intercept=True):
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol
        self.fit_intercept = fit_intercept

    @staticmethod
    def _soft_threshold(rho, lam):
        return np.sign(rho) * np.maximum(np.abs(rho) - lam, 0.0)

    def fit(self, X, y):
        if self.alpha < 0:
            raise ValueError(f"alpha must be non-negative, got {self.alpha}")

        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        if y.ndim > 1:
            y = y.ravel()

        n_samples, n_features = X.shape

        if n_samples != y.shape[0]:
            raise ValueError("X and y have incompatible shapes")

        self.n_features_in_ = n_features

        if self.fit_intercept:
            X_mean = np.mean(X, axis=0)
            y_mean = np.mean(y)
            X_c = X - X_mean
            y_c = y - y_mean
        else:
            X_c = X
            y_c = y
            X_mean = np.zeros(n_features)
            y_mean = 0.0

        col_norms_sq = np.sum(X_c**2, axis=0) / n_samples
        col_norms_sq[col_norms_sq == 0] = 1.0

        coef = np.zeros(n_features)

        coef_change = float("inf")
        converged = False

        for i in range(self.max_iter):
            coef_old = coef.copy()

            for j in range(n_features):
                residual = y_c - X_c @ coef + X_c[:, j] * coef[j]
                rho_j = X_c[:, j] @ residual / n_samples
                coef[j] = self._soft_threshold(rho_j, self.alpha) / col_norms_sq[j]

            coef_change = np.max(np.abs(coef - coef_old))
            if coef_change < self.tol:
                converged = True
                self.n_iter_ = i + 1
                break

        if not converged:
            warnings.warn(
                f"Lasso did not converge after {self.max_iter} iterations. "
                f"Max coefficient change: {coef_change:.2e}, tol: {self.tol:.2e}. "
                "Consider increasing max_iter or decreasing tol.",
                UserWarning,
            )
            self.n_iter_ = self.max_iter

        self.coef_ = coef

        if self.fit_intercept:
            self.intercept_ = y_mean - X_mean @ self.coef_
        else:
            self.intercept_ = 0.0

        return self

    def predict(self, X):
        self._check_is_fitted(["coef_", "intercept_"])
        X = np.asarray(X, dtype=np.float64)

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features, but Lasso is expecting "
                f"{self.n_features_in_} features."
            )

        return X @ self.coef_ + self.intercept_
