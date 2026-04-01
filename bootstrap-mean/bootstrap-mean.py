import numpy as np

def bootstrap_mean(x, n_bootstrap=1000, ci=0.95, rng=None):
    """
    Returns: (boot_means, lower, upper)
    """
    if rng is None:
        rng = np.random.default_rng()

    x = np.asanyarray(x)
    n = len(x)

    # 2D array of indices (random integers)
    resample_indices = rng.integers(0, n, size=(n_bootstrap, n))

    resamples = x[resample_indices]
    boot_means = np.mean(resamples, axis=1)

    alpha = 1 - ci
    lower_pct = (alpha / 2) * 100
    upper_pct = (1 - alpha / 2) * 100
    
    lower, upper = np.percentile(boot_means, [lower_pct, upper_pct])

    return boot_means, float(lower), float(upper)