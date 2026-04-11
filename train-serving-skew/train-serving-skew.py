import numpy as np

def detect_skew(train_dist, serving_dist, threshold=0.2, eps=1e-10):
    """
    Detect train-serving skew using PSI with high numerical precision.
    """
    results = {}
    
    for feature in train_dist:
        p_train = np.array(train_dist[feature])
        p_serving = np.array(serving_dist[feature])
        
        p_t_eps = p_train + eps
        p_s_eps = p_serving + eps
        
        psi_value = np.sum((p_s_eps - p_t_eps) * np.log(p_s_eps / p_t_eps))
        
        results[feature] = {
            "psi": float(psi_value),
            "skewed": bool(psi_value >= threshold)
        }
        
    return results