import numpy as np

from base import BaseEstimator, ClassifierMixin


class SVC(BaseEstimator, ClassifierMixin):
    def __init__(self, C=1.0, learning_rate=0.001, n_iters=1000):
        self.C = C
        self.learning_rate = learning_rate
        self.n_iters = n_iters

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        unique_classes = np.unique(y)
        if len(unique_classes) != 2:
            raise ValueError(
                f"SVC support only binary classificationFound {len(unique_classes)} classess: {unique_classes}"
            )

        self.classes_ = unique_classes
        self.n_features_in_ = X.shape[1]

        y_internal = np.where(y == unique_classes[1], 1.0, -1.0)
        n_samples, n_features = X.shape

        w = np.zeros(n_features)
        b = 0.0

        self.loss_history_ = []

        for i in range(self.n_iters):
            margins = y_internal * (X @ w - b)

            violated = margins < 1.0  # boolean mask

            dw = w - self.C * (y_internal[violated] @ X[violated]) / n_samples
            db = self.C * np.sum(y_internal[violated]) / n_samples

            w -= self.learning_rate * dw
            b -= self.learning_rate * db

            hinge_loss = np.mean(np.maximum(0, 1.0 - margins))
            total_loss = 0.5 * np.dot(w, w) + self.C * hinge_loss
            self.loss_history_.append(total_loss)

        self.w_ = w
        self.b_ = b

        return self

    def decision_function(self, X):
        self._check_is_fitted(["w_", "b_"])
        X = np.asarray(X, dtype=np.float64)

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features, but SVC expect {self.n_features_in_} features"
            )

        return X @ self.w_ - self.b_

    def predict(self, X):
        decisions = self.decision_function(X)

        # sign: +1 -> classes_[1], -1 atau 0 -> classes_[0]
        predicted_internal = np.sign(decisions)
        predicted_internal[predicted_internal == 0] = -1.0

        y_pred = np.where(predicted_internal == 1.0, self.classes_[1], self.classes_[0])

        return y_pred
