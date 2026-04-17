import numpy as np

def detect_mode_collapse(generated_samples, threshold=0.1):
    """
    Returns: dict with "diversity_score" (float) and "is_collapsed" (bool)
    """
    std = np.std(generated_samples, axis=0)
    score = np.mean(std)

    is_collapsed = score < threshold

    return {
        "diversity_score": score,
        "is_collapsed": is_collapsed
    }