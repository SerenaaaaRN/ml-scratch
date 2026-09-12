import numpy as np


def accuracy_score(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have same shape")
    return np.mean(y_true == y_pred)


def confusion_matrix(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    classes = np.unique(np.concatenate([y_true, y_pred]))
    n_classes = len(classes)
    class_to_idx = {c: i for i, c in enumerate(classes)}

    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[class_to_idx[t], class_to_idx[p]] += 1
    return cm


def precision_recall_f1(y_true, y_pred, average="macro"):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    match average.lower():
        case "binary":
            tp = np.sum((y_true == 1) & (y_pred == 1))
            fp = np.sum((y_true == 0) & (y_pred == 1))
            fn = np.sum((y_true == 1) & (y_pred == 0))

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (
                2 * precision * recall / (precision + recall)
                if (precision + recall) > 0
                else 0.0
            )
            return float(precision), float(recall), float(f1)

        case "macro":
            classes = np.unique(np.concatenate([y_true, y_pred]))
            precision, recalls, f1s = [], [], []

            for cls in classes:
                tp = np.sum((y_true == cls) & (y_pred == cls))
                fp = np.sum((y_true != cls) & (y_pred == cls))
                fn = np.sum((y_true == cls) & (y_pred != cls))

                p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0

                precision.append(p)
                recalls.append(r)
                f1s.append(f)

                return (
                    float(np.mean(precision)),
                    float(np.mean(recalls)),
                    float(np.mean(f1s)),
                )

        case "micro":
            classes = np.unique(np.concatenate([y_true, y_pred]))
            tp_total, fp_total, fn_total = 0, 0, 0

            for cls in classes:
                tp_total += np.sum((y_true == cls) & (y_pred == cls))
                fp_total += np.sum((y_true != cls) & (y_pred == cls))
                fn_total += np.sum((y_true == cls) & (y_pred != cls))

            precision = (
                tp_total / (tp_total + fp_total) if (tp_total + fp_total) > 0 else 0.0
            )
            recall = (
                tp_total / (tp_total + fn_total) if (tp_total + fn_total) > 0 else 0.0
            )
            f1 = (
                2 * precision * recall / (precision + recall)
                if (precision + recall) > 0
                else 0.0
            )

            return float(precision), float(recall), float(f1)

        case _:
            raise ValueError(
                f"Unknown average= '{average}'. Use 'macro' 'micro' or 'binary'"
            )
