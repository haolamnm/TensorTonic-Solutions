import numpy as np

def detect_drift(reference_counts, production_counts, threshold):
    """
    Compare reference and production distributions to detect data drift.
    """
    ref = np.asarray(reference_counts, dtype=float)
    ref_total = np.sum(ref)
    ref_norm = ref / ref_total
    
    prod = np.asarray(production_counts, dtype=float)
    prod_total = np.sum(prod)
    prod_norm = prod / prod_total
    
    tvd_score = float(0.5 * np.sum(np.abs(ref_norm - prod_norm)))
    drift_detected = tvd_score > threshold

    return {
        "score": tvd_score,
        "drift_detected": drift_detected
    }