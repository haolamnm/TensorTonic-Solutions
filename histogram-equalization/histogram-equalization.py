import numpy as np

def histogram_equalize(image):
    """
    Apply histogram equalization to enhance image contrast.
    """
    image = np.asarray(image)
    flat = image.flatten()
    bins = np.bincount(flat)
    total_pixels = len(flat)
    
    cdf = np.cumsum(bins)
    print(cdf)

    cdf_min = np.min(cdf[cdf != 0])
    print(cdf_min)

    if total_pixels == cdf_min:
        return np.zeros_like(image).tolist()

    new_val = np.round((cdf[flat] - cdf_min) / (total_pixels - cdf_min) * 255).reshape(image.shape)
    return new_val.tolist()