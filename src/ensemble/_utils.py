import numpy as np


def bootstrap_sample(X, y, random_state=None):
    rng = np.random.RandomState(random_state)
    n_samples = X.shape[0]
    indices = rng.choice(n_samples, size=n_samples, replace=True)
    return X[indices], y[indices]


def resolve_max_features(max_features, n_features):

    if max_features is None:
        return n_features
    elif isinstance(max_features, str):
        if max_features == "sqrt":
            return max(1, int(np.sqrt(n_features)))
        elif max_features == "log2":
            return max(1, int(np.log2(n_features)))
        else:
            raise ValueError(
                f"Invalid max_features='{max_features}'. "
                "Use 'sqrt', 'log2', int, float, or None."
            )
    elif isinstance(max_features, float):
        if not (0.0 < max_features <= 1.0):
            raise ValueError(
                f"max_features as float must be in (0, 1], got {max_features}"
            )
        return max(1, int(max_features * n_features))
    elif isinstance(max_features, int):
        if max_features < 1 or max_features > n_features:
            raise ValueError(
                f"max_features={max_features} out of range [1, {n_features}]"
            )
        return max_features
    else:
        raise TypeError(
            f"max_features must be str, int, float, or None, got {type(max_features)}"
        )
