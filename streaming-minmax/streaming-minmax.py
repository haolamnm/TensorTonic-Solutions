import numpy as np

def streaming_minmax_init(D):
    """
    Initialize state dict with min, max arrays of shape (D,).
    """
    return {
        "min": np.full(D, np.inf).tolist(),
        "max": np.full(D, -np.inf).tolist()
    }

def streaming_minmax_update(state, X_batch, eps=1e-8):
    """
    Update state's min/max with X_batch, return normalized batch.
    """
    X_batch = np.asarray(X_batch, dtype=float)
    state_min = np.asarray(state["min"], dtype=float)
    state_max = np.asarray(state["max"], dtype=float)

    batch_min = np.min(X_batch, axis=0)
    batch_max = np.max(X_batch, axis=0)

    state_min = np.minimum(state_min, batch_min)
    state_max = np.maximum(state_max, batch_max)

    X_norm = (X_batch - state_min) / (state_max - state_min + eps)

    state["min"] = state_min.tolist()
    state["max"] = state_max.tolist()

    return X_norm.tolist()