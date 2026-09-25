import numpy as np
from base import BaseEstimator


class KMeans(BaseEstimator):
    def __init__(self, n_clusters=3, max_iter=300, n_init=10, random_state=False):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.n_init = n_init
        self.random_state = random_state

    @staticmethod
    def _compute_inertia(X, labels, centroids):
        assigned_centroids = centroids[labels]
        sq_distance = np.sum((X - assigned_centroids) ** 2, axis=1)
        return float(np.sum(sq_distance))

    @staticmethod
    def _assign_clusters(X, centroids):
        x_sq = np.sum(X**2, axis=1, keepdims=True)  # (n_samples, 1)
        c_sq = np.sum(centroids**2, axis=1, keepdims=True)  # (n_clusters, 1)
        cross = X @ centroids.T  # (n_samples, n_clusters)

        dist_sq = x_sq + c_sq.T - 2 * cross  # (n_samples, n_clusters)
        dist_sq = np.maximum(dist_sq, 0.0)

        return np.argmin(dist_sq, axis=1)

    def _single_run(self, X, rng):
        n_samples = X.shape[0]

        init_indices = rng.choice(n_samples, size=self.n_clusters, replace=False)
        centroids = X[init_indices].copy()

        labels = np.zeros(n_samples, dtype=int)
        n_iter = 0

        for iteration in range(self.max_iter):
            # assignment step
            new_labels = self._assign_clusters(X, centroids)

            # update step
            new_centroids = np.empty_like(centroids)
            empty_clusters = []

            for k in range(self.n_clusters):
                mask = new_labels == k
                if np.sum(mask) == 0:
                    empty_clusters.append(k)
                    new_centroids[k] = X[rng.randint(0, n_samples)]
                else:
                    new_centroids[k] = np.mean(X[mask], axis=0)

            centroid_shift = np.sqrt(np.sum((new_centroids - centroids) ** 2))
            centroids = new_centroids
            labels = new_labels
            n_iter = iteration + 1

            if centroid_shift < 1e-10:
                break

        inertia = self._compute_inertia(X, labels, centroids)
        return centroids, labels, inertia, n_iter

    def fit(self, X, y=None):

        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        if self.n_clusters > X.shape[0]:
            raise ValueError(
                f"n_clusters={self.n_clusters} cannot be greater than n_samples={X.shape[0]}"
            )

        self.n_features_in_ = X.shape[1]
        rng = np.random.RandomState(self.random_state)

        best_inertia = np.inf
        best_centroids = None
        best_labels = None
        best_n_iter = 0

        for _ in range(self.n_init):
            centroids, labels, inertia, n_iter = self._single_run(X, rng)

            if inertia < best_inertia:
                best_inertia = inertia
                best_centroids = centroids
                best_labels = labels
                best_n_iter = n_iter

        self.centroids_ = best_centroids
        self.labels_ = best_labels
        self.inertia_ = best_inertia
        self.n_iter_ = best_n_iter

        return self

    def predict(self, X):
        self._check_is_fitted(["centroids_"])
        X = np.asarray(X, dtype=np.float64)

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features, but KMeans expects {self.n_features_in_} features"
            )

        return self._assign_clusters(X, self.centroids_)

    def fit_predict(self, X, y=None):
        self.fit(X, y)
        return self.labels_
