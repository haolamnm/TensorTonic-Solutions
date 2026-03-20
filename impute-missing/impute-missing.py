import numpy as np

def impute_missing(X, strategy='mean'):
    """
    Fill NaN values in each feature column using column mean or median.
    """
    fn = {
        "mean": np.nanmean,
        "median": np.nanmedian
    }
    
    X = np.asarray(X)
    mask = np.isnan(X)
    cols = np.all(mask, axis=0)
    X = np.where(cols, 0, X)
    
    values = fn[strategy](X, axis=0, keepdims=True)
    X = np.where(mask, values, X)
    
    return X