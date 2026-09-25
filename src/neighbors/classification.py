import numpy as np
from base import BaseEstimator, ClassifierMixin
from ._utils import compute_distances, top_k_indices


class KNeighborsClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, n_neighbors=5):
        self.n_neighbors = n_neighbors

    def fit(self, X, y):

        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        if self.n_neighbors > X.shape[0]:
            raise ValueError(
                f"n_neighbors={self.n_neighbors} cannot be greater than "
                f"n_samples={X.shape[0]}"
            )

        self.X_train_ = X
        self.y_train_ = y
        self.classes_ = np.unique(y)
        self.n_features_in_ = X.shape[1]

        return self

    def predict(self, X):

        self._check_is_fitted(["X_train_", "y_train_", "classes_"])
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features, but KNeighborsClassifier expects "
                f"{self.n_features_in_} features."
            )

        distances = compute_distances(self.X_train_, X)  # (n_test, n_train)
        knn_indices = top_k_indices(distances, self.n_neighbors)  # (n_test, k)
        knn_distances = distances[
            np.arange(X.shape[0])[:, np.newaxis], knn_indices
        ]  # (n_test, k)
        knn_labels = self.y_train_[knn_indices]  # (n_test, k)

        n_test = X.shape[0]
        y_pred = np.empty(n_test, dtype=self.classes_.dtype)

        for i in range(n_test):
            labels_i = knn_labels[i]  # (k,)
            dists_i = knn_distances[i]  # (k,) — already sorted nearest-first

            # Count votes per class
            unique_labels, counts = np.unique(labels_i, return_counts=True)
            max_count = np.max(counts)
            tied_classes = unique_labels[counts == max_count]

            if len(tied_classes) == 1:
                y_pred[i] = tied_classes[0]
            else:
                # Tie-break: among tied classes, pick the one whose
                # nearest occurrence is closest to the query point
                best_class = tied_classes[0]
                best_dist = np.inf

                for cls in tied_classes:
                    # Find minimum distance among neighbors with this class
                    cls_mask = labels_i == cls
                    min_dist = np.min(dists_i[cls_mask])
                    if min_dist < best_dist:
                        best_dist = min_dist
                        best_class = cls

                y_pred[i] = best_class

        return y_pred
