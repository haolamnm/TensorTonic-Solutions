import numpy as np

def compute_monitoring_metrics(system_type, y_true, y_pred):
    """
    Compute the appropriate monitoring metrics for the given system type.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    metrics = {}

    if system_type == "classification":
        # binary
        TP = np.sum((y_true == 1) & (y_pred == 1))
        TN = np.sum((y_true == 0) & (y_pred == 0))
        FP = np.sum((y_true == 0) & (y_pred == 1))
        FN = np.sum((y_true == 1) & (y_pred == 0))
        n  = len(y_true)

        precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        accuracy = (TP + TN) / n
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        metrics["precision"] = precision
        metrics["recall"] = recall
        metrics["accuracy"] = accuracy
        metrics["f1"] = f1

    elif system_type == "regression":
        # floats
        diff = y_true - y_pred
        metrics["mae"] = np.mean(np.abs(diff))
        metrics["rmse"] = np.sqrt(np.mean(diff ** 2))

    elif system_type == "ranking":
        # y_true: binary, y_pred: floats
        k = 3
        top_k_idx = np.argsort(y_pred)[::-1][:k]
        top_k_relevant = y_true[top_k_idx].sum()
        total_relevant = y_true.sum()

        metrics["precision_at_3"] = top_k_relevant / k
        metrics["recall_at_3"] = top_k_relevant / total_relevant if total_relevant > 0 else 0.0

    else:
        raise ValueError(f"Unsupported system type. Got {system_type}")

    return sorted(metrics.items())