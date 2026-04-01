import numpy as np

def auc(fpr, tpr):
    """
    Compute AUC (Area Under ROC Curve) using trapezoidal rule.
    """
    if len(fpr) < 2 or len(tpr) < 2:
        return None

    if len(fpr) != len(tpr):
        return None    

    return float(np.trapezoid(tpr, fpr))
