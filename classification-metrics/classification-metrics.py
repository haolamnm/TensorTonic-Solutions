import numpy as np

def classification_metrics(y_true, y_pred, average="micro", pos_label=1):
    """
    Compute accuracy, precision, recall, F1 for single-label classification.
    Averages: 'micro' | 'macro' | 'weighted' | 'binary' (uses pos_label).
    Return dict with float values.
    """
    y_t = np.asarray(y_true, dtype=int)
    y_p = np.asarray(y_pred, dtype=float)

    classes = np.unique(y_t)
    n_classes = len(classes)

    # compute confusion matrix
    label_map = {val: i for i, val in enumerate(classes)}
    print(label_map)

    y_t_idx = np.array([label_map[c] for c in y_t])
    y_p_idx = np.array([label_map.get(c, -1) for c in y_p])
    print(y_t_idx)
    print(y_p_idx)

    # filter out predictions that aren't in y_true classes
    mask = y_p_idx != -1

    # compute confusion matrix
    cm = np.bincount(n_classes * y_t_idx[mask] + y_p_idx[mask], 
                     minlength=n_classes**2).reshape(n_classes, n_classes)
    print(cm)

    tp = np.diag(cm)
    fp = np.sum(cm, axis=0) - tp
    fn = np.sum(cm, axis=1) - tp
    support = np.sum(cm, axis=1)

    # use np.errstate to handle divide-by-zero
    with np.errstate(divide="ignore", invalid="ignore"):
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1 = 2 * (precision * recall) / (precision + recall)
    
    # fill nans with 0
    precision[np.isnan(precision)] = 0
    recall[np.isnan(recall)] = 0
    f1[np.isnan(f1)] = 0
    
    # averaging logic
    if average == "binary":
        idx = label_map[pos_label]
        return {"accuracy": np.mean(y_t == y_p), "precision": precision[idx], "recall": recall[idx], "f1": f1[idx]}
    
    if average == "macro":
        return {"accuracy": np.mean(y_t == y_p), "precision": np.mean(precision), "recall": np.mean(recall), "f1": np.mean(f1)}
    
    if average == "weighted":
        weights = support / np.sum(support)
        return {"accuracy": np.mean(y_t == y_p), "precision": np.sum(precision * weights), "recall": np.sum(recall * weights), "f1": np.sum(f1 * weights)}
    
    if average == "micro":
        global_tp = np.sum(tp)
        global_fp = np.sum(fp)
        global_fn = np.sum(fn)
        p_micro = global_tp / (global_tp + global_fp)
        r_micro = global_tp / (global_tp + global_fn)
        f1_micro = 2 * (p_micro * r_micro) / (p_micro + r_micro)
        return {"accuracy": np.mean(y_t == y_p), "precision": p_micro, "recall": r_micro, "f1": f1_micro}

    return None