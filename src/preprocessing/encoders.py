import numpy as np
from base import BaseEstimator, TransformerMixin


class LabelEncoder(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        X = np.asarray(X)
        self.classes_ = np.unique(X)
        return self

    def transform(self, X):
        self._check_is_fitted(["classes_"])
        X = np.asarray(X)

        class_to_idx = {cls: idx for idx, cls in enumerate(self.classes_)}

        encoded = np.array([class_to_idx.get(val, -1) for val in X])

        if np.any(encoded == -1):
            unseen = set(X[encoded == -1]) - set(self.classes_)
            raise ValueError(
                f"X contains previously unseen labels: {unseen}. Call fit() first."
            )

        return encoded.astype(int)

    def inverse_transform(self, X):
        self._check_is_fitted(["classes_"])
        X = np.asarray(X, dtype=int)
        if np.any((X < 0) | (X >= len(self.classes_))):
            raise ValueError(
                f"X contains indices outside valid range [0, {len(self.classes_) - 1}]"
            )
        return self.classes_[X]

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)


class OneHotEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, handle_unknown="error"):
        self.handle_unknown = handle_unknown

    def fit(self, X, y=None):
        if self.handle_unknown not in ("error", "ignore"):
            raise ValueError(
                f"handle_unknown mmust be 'error' or 'ignore', got '{self.handle_unknown}'"
            )

        X = np.asarray(X)

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        self.n_features_in_ = X.shape[1]
        self.categories_ = [np.unique(X[:, i]) for i in range(X.shape[1])]

        return self

    def transform(self, X):
        self._check_is_fitted(["categories_"])
        X = np.asarray(X)

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features, but OneHotEncoding is expecting"
                f"{self.n_features_in_} features"
            )

        n_samples = X.shape[0]
        output_cols = []

        for col_idx, categories in enumerate(self.categories_):
            col_data = X[:, col_idx]

            indices = np.searchsorted(categories, col_data)

            valid_mask = (indices < len(categories)) & (categories[indices] == col_data)

            if not np.all(valid_mask):
                if self.handle_unknown == "error":
                    unseen = set(col_data[~valid_mask])
                    raise ValueError(
                        f"Found unknown categories {unseen} in column {col_idx} during transform. "
                        "Set handle_unknown='ignore' to skip unknowns."
                    )

            n_cats = len(categories)
            one_hot_block = np.zeros((n_samples, n_cats), dtype=np.float64)

            valid_indices = indices[valid_mask]
            valid_rows = np.where(valid_mask)[0]
            one_hot_block[valid_rows, valid_indices] = 1.0

            output_cols.append(one_hot_block)

        return np.hstack(output_cols)

    def get_feature_names_out(self):
        self._check_is_fitted(["categories_"])
        names = []
        for col_idx, categories in enumerate(self.categories_):
            for cat in categories:
                names.append(f"x{col_idx}_{cat}")
            return names
