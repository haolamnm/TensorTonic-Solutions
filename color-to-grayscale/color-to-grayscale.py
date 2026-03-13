import numpy as np

def color_to_grayscale(image):
    """
    Convert an RGB image to grayscale using luminance weights.
    """
    image = np.asarray(image, dtype=int)

    gray = np.array([0.299, 0.587, 0.114], dtype=float)

    return np.dot(image, gray).tolist()