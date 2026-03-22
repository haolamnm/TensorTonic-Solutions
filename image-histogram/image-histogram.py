import numpy as np

def image_histogram(image):
    """
    Compute the intensity histogram of a grayscale image.
    """
    image = np.asarray(image)

    # good for integer values
    return np.bincount(image.flatten(), minlength=256).tolist()