import numpy as np
from base import BaseEstimator, ClassifierMixin, RegressorMixin
from tree import DecisionTreeClassifier, DecisionTreeRegressor
from ._utils import bootstrap_sample, resolve_max_features


class RandomForestClassifier(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        n_estimators=100,
        max_depth=None,
        min_samples_split=2,
        max_features="sqrt",
        criterion="gini",
        random_state=None,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.criterion = criterion
        self.random_state = random_state

    def fit(self, X, y):

        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        self.n_features_in_ = X.shape[1]
        self.classes_ = np.unique(y)

        n_feats = resolve_max_features(self.max_features, self.n_features_in_)

        rng = np.random.RandomState(self.random_state)
        self.trees_ = []
        self.feature_indices_ = []

        for i in range(self.n_estimators):
            seed_i = rng.randint(0, 2**31)
            X_boot, y_boot = bootstrap_sample(X, y, random_state=seed_i)

            feat_indices = rng.choice(self.n_features_in_, size=n_feats, replace=False)
            feat_indices.sort()

            X_boot_sub = X_boot[:, feat_indices]

            tree = DecisionTreeClassifier(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                criterion=self.criterion,
            )
            tree.fit(X_boot_sub, y_boot)

            self.trees_.append(tree)
            self.feature_indices_.append(feat_indices)

        return self

    def predict(self, X):

        self._check_is_fitted(["trees_", "classes_", "feature_indices_"])
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features, but RandomForestClassifier expects "
                f"{self.n_features_in_} features."
            )

        all_preds = np.array(
            [
                tree.predict(X[:, feat_idx])
                for tree, feat_idx in zip(self.trees_, self.feature_indices_)
            ]
        )

        n_samples = X.shape[0]
        y_pred = np.empty(n_samples, dtype=self.classes_.dtype)

        class_to_idx = {c: i for i, c in enumerate(self.classes_)}
        idx_to_class = {i: c for c, i in class_to_idx.items()}

        mapped_preds = np.vectorize(class_to_idx.get)(all_preds)

        for s in range(n_samples):
            counts = np.bincount(
                mapped_preds[:, s].astype(int), minlength=len(self.classes_)
            )

            y_pred[s] = idx_to_class[int(np.argmax(counts))]

        return y_pred


class RandomForestRegressor(BaseEstimator, RegressorMixin):
    def __init__(
        self,
        n_estimators=100,
        max_depth=None,
        min_samples_split=2,
        max_features=1.0,
        random_state=None,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.random_state = random_state

    def fit(self, X, y):

        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)

        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if y.ndim > 1:
            y = y.ravel()

        self.n_features_in_ = X.shape[1]
        n_feats = resolve_max_features(self.max_features, self.n_features_in_)

        rng = np.random.RandomState(self.random_state)
        self.trees_ = []
        self.feature_indices_ = []

        for i in range(self.n_estimators):
            seed_i = rng.randint(0, 2**31)
            X_boot, y_boot = bootstrap_sample(X, y, random_state=seed_i)

            feat_indices = rng.choice(self.n_features_in_, size=n_feats, replace=False)
            feat_indices.sort()

            X_boot_sub = X_boot[:, feat_indices]

            tree = DecisionTreeRegressor(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
            )
            tree.fit(X_boot_sub, y_boot)

            self.trees_.append(tree)
            self.feature_indices_.append(feat_indices)

        return self

    def predict(self, X):

        self._check_is_fitted(["trees_", "feature_indices_"])
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features, but RandomForestRegressor expects "
                f"{self.n_features_in_} features."
            )

        all_preds = np.array(
            [
                tree.predict(X[:, feat_idx])
                for tree, feat_idx in zip(self.trees_, self.feature_indices_)
            ]
        )

        return np.mean(all_preds, axis=0)
