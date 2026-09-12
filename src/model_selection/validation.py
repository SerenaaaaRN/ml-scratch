import numpy as np
import copy


class KFold:
    def __init__(self, n_split=5, shuffle=True, random_state=None):
        if n_split < 2:
            raise ValueError("n_splits must be at least 2")
        self.n_splits = n_split
        self.shuffle = shuffle
        self.random_state = random_state

    def split(self, X):
        n_samples = np.asarray(X).shape[0]
        if self.n_splits > n_samples:
            raise ValueError(
                f"Cannot have n_split={self.n_splits} greater than n_samples={n_samples}"
            )

        indices = np.arange(n_samples)
        if self.shuffle:
            rng = np.random.RandomState(self.random_state)
            rng.shuffle(indices)

        fold_sizes = np.full(self.n_splits, n_samples // self.n_splits, dtype=int)
        fold_sizes[: n_samples % self.n_splits] += 1

        current = 0
        for fold_size in fold_sizes:
            start, stop = current, current + fold_size
            val_idx = indices[start:stop]
            train_idx = np.concatenate([indices[:start], indices[stop:]])
            yield train_idx, val_idx
            current = stop

    def get_n_split(self):
        return self.n_splits


def cross_val_score(model, X, y, cv=5):
    if isinstance(cv, int):
        kfold = KFold(n_split=cv, shuffle=True, random_state=42)
    elif isinstance(cv, KFold):
        kfold = cv
    else:
        raise TypeError("cv must be int or KFold instance")

    X = np.asarray(X)
    y = np.asarray(y)

    scores = []

    for train_idx, val_idx in kfold.split(X):
        model_clone = copy.deepcopy(model)
        model_clone.fit(X[train_idx], y[train_idx])
        score = model_clone.score(X[val_idx], y[val_idx])
        scores.append(score)

    return np.array(scores)
