import numpy as np


def compute_distances(X_train, X_test):

    train_sq = np.sum(X_train**2, axis=1)
    test_sq = np.sum(X_test**2, axis=1)

    cross = X_test @ X_train.T

    # ||a-b||² = ||a||² + ||b||² - 2*a·b
    dist_sq = test_sq[:, np.newaxis] + train_sq[np.newaxis, :] - 2 * cross

    dist_sq = np.maximum(dist_sq, 0.0)

    return np.sqrt(dist_sq)


def top_k_indices(distances, k):

    if k >= distances.shape[1]:
        return np.argsort(distances, axis=1)[:, :k]

    partitioned = np.argpartition(distances, k, axis=1)[:, :k]

    row_idx = np.arange(distances.shape[0])[:, np.newaxis]
    selected_dists = distances[row_idx, partitioned]
    sorted_within = np.argsort(selected_dists, axis=1)

    return partitioned[row_idx, sorted_within]
