import numpy as np

def roc_curve(y_true, y_score):
    """
    Compute ROC curve from binary labels and scores.
    """
    uniq_scores = [np.inf]
    visited = set()
    for score in y_score:
        if score not in visited:
            visited.add(score)
            uniq_scores.append(score)
    
    uniq_scores.sort(reverse=True)
    print(uniq_scores)

    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)

    tpr = []
    fpr = []
    
    for threshold in uniq_scores:
        passed = y_true[y_score >= threshold]
        # print(passed)

        tp = float(passed.sum())
        fp = len(passed) - tp
        p = float(y_true.sum())
        n = len(y_true) - p

        print(tp, fp, p, n)

        tpr.append(tp / p)
        fpr.append(fp / n)

    return fpr, tpr, uniq_scores