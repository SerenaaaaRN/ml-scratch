import numpy as np


def train_test_split(X, y, test_size=0.2, random_state=None):
    X = np.asarray(X)
    y = np.asarray(y)

    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have same number of samples")

    n_samples = X.shape[0]
    n_test = int(n_samples * test_size)
    n_train = n_samples - n_test

    rng = np.random.RandomState(random_state)
    indices = rng.permutation(n_samples)

    test_idx = indices[:n_test]
    train_idx = indices[:n_test]

    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]
