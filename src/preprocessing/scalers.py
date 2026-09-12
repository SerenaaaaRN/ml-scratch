import numpy as np
from base import BaseEstimator, TransformerMixin


class StandardScaler(BaseEstimator, TransformerMixin):
    def __init__(self, with_mean=True, with_std=True):
        self.with_mean = with_mean
        self.with_std = with_std

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        self.n_features_in_ = X.shape[1]
        self.mean_ = np.mean(X, axis=0) if self.with_mean else np.zeros(X.shape[1])

        if self.with_std:
            self.std_ = np.std(X, axis=0, ddof=0)
            self.std_[self.std_ == 0] = 1.0
        else:
            self.std_ = np.ones(X.shape[1])

        return self

    def transform(self, X):
        self._check_is_fitted(["mean_", "std_"])
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} feature, but StandardScaler is expecting"
                f"{self.n_features_in_} features"
            )

        X_scaled = X.copy()
        if self.with_mean:
            X_scaled = X_scaled - self.mean_
        if self.with_std:
            X_scaled = X_scaled / self.std_

        return X_scaled

    def inverse_transform(self, X):
        self._check_is_fitted(["mean_", "std_"])
        X = np.asarray(X, dtype=np.float64)

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        X_original = X.copy()
        if self.with_std:
            X_original = X_original * self.std_
        if self.with_mean:
            X_original = X_original + self.mean_

        return X_original


class MinMaxScaler(BaseEstimator, TransformerMixin):
    def __init__(self, feature_range=(0, 1)):
        self.feature_range = feature_range

    def fit(self, X, y=None):
        if self.feature_range[0] >= self.feature_range[1]:
            raise ValueError(
                "Minimum of feature_range must be less than maxiumum"
                f"Got {self.feature_range}"
            )
        X = np.array(X, dtype=np.float64)

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        self.n_features_in_ = X.shape[1]
        self.data_min_ = np.min(X, axis=0)
        self.data_max_ = np.max(X, axis=0)
        self.data_range_ = self.data_max_ - self.data_min_

        self.data_range_[self.data_range_ == 0] = 1.0

        feature_min, feature_max = self.feature_range
        self.scale_ = (feature_max - feature_min) / self.data_range_
        self.min_offset_ = feature_min - self.data_min_ * self.scale_

        self.min_ = self.data_min_
        self.max_ = self.data_max_

        return self

    def transform(self, X):
        self._check_is_fitted(["scale_", "min_offset_"])
        X = np.asarray(X, dtype=np.float64)

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} feature, but MinMaxScaler is expecting"
                f"{self.n_features_in_} feature"
            )

        return X * self.scale_ + self.min_offset_

    def inverse_transform(self, X):
        self._check_is_fitted(["scale_", "min_offset_"])
        X = np.asarray(X, dtype=np.float64)

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        return (X - self.min_offset_) / self.scale_
