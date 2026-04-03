import numpy as np

def winsorize(values, lower_pct, upper_pct):
    """
    Clip values at the given percentile bounds.
    """
    values = np.asarray(values, dtype=float)

    lower_bound = np.percentile(values, lower_pct)
    upper_bound = np.percentile(values, upper_pct)
    winsorized_values = np.clip(values, lower_bound, upper_bound)

    return winsorized_values.tolist()