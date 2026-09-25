import numpy as np


def correlation_score(X, y):
    y = y.astype(np.float64)
    y_centered = y - np.mean(y)
    y_std = np.std(y, ddof=0)

    if y_std == 0:
        return np.zeros(X.shape[1])

    X_centered = X - np.mean(X, axis=0)
    X_std = np.std(X, axis=0, ddof=0)

    # avoid division by zero for constant features
    X_std[X_std == 0] = 1.0

    correlations = (X_centered.T @ y_centered) / (len(y) * X_std * y_std)
    return np.abs(correlations)


def f_classif_score(X, y):

    classes = np.unique(y)
    n_classes = len(classes)
    n_samples = X.shape[0]

    if n_classes < 2:
        return np.zeros(X.shape[1])

    overall_mean = np.mean(X, axis=0)  # (n_features,)

    ss_between = np.zeros(X.shape[1])

    ss_within = np.zeros(X.shape[1])

    for cls in classes:
        mask = y == cls
        n_k = np.sum(mask)
        if n_k == 0:
            continue

        group_mean = np.mean(X[mask], axis=0)
        ss_between += n_k * (group_mean - overall_mean) ** 2
        ss_within += np.sum((X[mask] - group_mean) ** 2, axis=0)

    # degrees of freedom
    df_between = n_classes - 1
    df_within = n_samples - n_classes

    # avoid division by zero
    if df_between == 0 or df_within <= 0:
        return np.zeros(X.shape[1])

    ms_between = ss_between / df_between
    ms_within = ss_within / df_within

    # handle zero within-group variance
    ms_within[ms_within == 0] = 1e-15

    f_scores = ms_between / ms_within
    return f_scores


SCORE_FUNC_MAP = {
    "correlation": correlation_score,
    "f_classif": f_classif_score,
}
