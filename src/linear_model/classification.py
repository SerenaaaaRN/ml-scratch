import numpy as np
import warnings
from base import BaseEstimator, ClassifierMixin


class LogisticRegression(BaseEstimator, ClassifierMixin):
    def __init__(self, learning_rate=0.01, max_iter=1000, tol=1e-4, fit_intercept=True):
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.tol = tol
        self.fit_intercept = fit_intercept

    @staticmethod
    def _sigmoid(z):
        z = np.clip(z, -500, 500)
        return np.where(z >= 0, 1.0 / (1.0 + np.exp(-z)), np.exp(z) / (1.0 + np.exp(z)))

    @staticmethod
    def _binary_cross_entropy(y, y_prob):
        eps = 1e-15
        y_prob = np.clip(y_prob, eps, 1 - eps)
        return -np.mean(y * np.log(y_prob) + (1 - y) * np.log(1 - y_prob))

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)

        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if y.ndim > 1:
            y = y.ravel()

        unique_classes = np.unique(y)
        if len(unique_classes) != 2:
            raise ValueError(
                f"LogisticRegression supports only binary classification"
                f"Found {len(unique_classes)} classes: {unique_classes}"
            )
        if not np.all(np.isin(unique_classes, [0, 1])):
            raise ValueError(
                f"Labels must be 0 or 1 for binary logistic regression"
                f"Found: {unique_classes}"
            )

        n_samples, n_features = X.shape
        self.n_features_in_ = n_features
        self.classes_ = unique_classes

        if self.fit_intercept:
            X_aug = np.hstack([np.ones((n_samples, 1)), X])
        else:
            X_aug = X

        theta = np.zeros(X_aug.shape[1])

        self.loss_history_ = []

        loss_change = float("inf")
        converged = False

        for iteration in range(self.max_iter):
            z = X_aug @ theta
            y_prob = self._sigmoid(z)

            loss = self._binary_cross_entropy(y, y_prob)
            self.loss_history_.append(loss)

            gradient = X_aug.T @ (y_prob - y) / n_samples

            theta -= self.learning_rate * gradient

            if len(self.loss_history_) >= 2:
                loss_change = abs(self.loss_history_[-2] - self.loss_history_[-1])
                if loss_change < self.tol:
                    converged = True
                    self.n_iter_ = iteration + 1
                    break

        if not converged:
            warnings.warn(
                f"LogisticRegression did not converge after {self.max_iter} iterations"
                f"Last loss change: {loss_change:.2e}, tol: {self.tol:.2e}"
                "Consider adjusting learning_rate or increasing max_iter.",
                UserWarning,
            )
            self.n_iter_ = self.max_iter

        if self.fit_intercept:
            self.intercept_ = theta[0]
            self.coef_ = theta[1:]
        else:
            self.intercept_ = 0.0
            self.coef_ = theta

        return self

    def predict_proba(self, X):
        self._check_is_fitted(["coef_", "intercept_"])
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features, but LogisticRegression is expecting "
                f"{self.n_features_in_} features."
            )

        z = X @ self.coef_ + self.intercept_
        p1 = self._sigmoid(z)
        return np.column_stack([1 - p1, p1])

    def predict(self, X):
        proba = self.predict_proba(X)
        return (proba[:, 1] >= 0.5).astype(int)
