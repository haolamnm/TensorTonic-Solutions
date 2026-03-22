def image_histogram(image):
    """
    Compute the intensity histogram of a grayscale image.
    """
    n_colors = 256
    bins = [0] * n_colors

    for i in range(len(image)):
        for j in range(len(image[0])):
            bins[image[i][j]] += 1

    return bins