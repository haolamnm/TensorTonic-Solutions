import numpy as np


def relu(x):
    return np.maximum(0, x)


def norm(x, gamma, beta, eps=1e-9):
    x = np.asarray(x)
    mean = np.mean(x, axis=0, keepdims=True)
    var = np.var(x, axis=0, keepdims=True)
    print(mean, var)

    x_hat = (x - mean) / np.sqrt(var + eps)
    out = x_hat * gamma + beta
    return out


def batch_norm_block(x, W1, W2, gamma1, beta1, gamma2, beta2, mode):
    """
    Returns: np.ndarray of same shape as input with batch-normalized and skip-connected output
    """
    W1 = np.asarray(W1)
    W2 = np.asarray(W2)
    gamma1 = np.asarray(gamma1)
    gamma2 = np.asarray(gamma2)
    beta1 = np.asarray(beta1)
    beta2 = np.asarray(beta2)
    

    if mode == "post":
        h1 = relu(norm(x @ W1, gamma1, beta1))
        h2 = norm(h1 @ W2, gamma2, beta2)
        val = relu(h2 + x)

    elif mode == "pre":
        h1 = relu(norm(x, gamma1, beta1)) @ W1
        h2 = relu(norm(h1, gamma2, beta2)) @ W2
        val = (h2 + x)

    else:
        raise ValueError("Invalid mode!")

    return {
        "output": np.round(val, 4),
        "mode": mode
    }