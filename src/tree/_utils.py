import numpy as np


def gini_impurity(y):
    if len(y) == 0:
        return 0.0

    _, counts = np.unique(y, return_counts=True)
    probs = counts / len(y)
    return 1.0 - np.sum(probs**2)


def entropy_impurity(y):
    if len(y) == 0:
        return 0.0

    _, counts = np.unique(y, return_counts=True)
    probs = counts / len(y)
    probs = probs[probs > 0]
    return -np.sum(probs * np.log2(probs))


def mse_criterion(y):
    if len(y) == 0:
        return 0.0

    return np.var(y)


CRITERION_MAP = {
    "gini": gini_impurity,
    "entropy": entropy_impurity,
    "mse": mse_criterion,
}


def best_split(X, y, criterion_func):
    n_samples, n_features = X.shape
    parent_impurity = criterion_func(y)

    best_gain = 0.0
    best_feature = None
    best_threshold = None

    for feature_idx in range(n_features):
        col = X[:, feature_idx]

        unique_vals = np.unique(col)
        if len(unique_vals) <= 1:
            continue

        thresholds = (unique_vals[:-1] + unique_vals[1:]) / 2.0

        for threshold in thresholds:
            left_mask = col <= threshold
            right_mask = ~left_mask

            n_left = np.sum(left_mask)
            n_right = np.sum(right_mask)

            if n_left == 0 or n_right == 0:
                continue

            child_impurity = (n_left / n_samples) * criterion_func(y[left_mask]) + (
                n_right / n_samples
            ) * criterion_func(y[right_mask])

            gain = parent_impurity - child_impurity

            if gain > best_gain:
                best_gain = gain
                best_feature = feature_idx
                best_threshold = threshold

    return best_feature, best_threshold, best_gain


def predict_single(node, x):
    if "value" in node:
        return node["value"]

    if x[node["feature"]] <= node["threshold"]:
        return predict_single(node["left"], x)
    else:
        return predict_single(node["right"], x)
