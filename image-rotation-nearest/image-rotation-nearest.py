import math as m

def rotate_image(image, angle_degrees):
    """
    Rotate the image counterclockwise by the given angle using nearest neighbor interpolation.
    """
    new_image = [[0 for _ in row] for row in image]

    # forgot to convert to rad
    alpha = m.sin(m.radians(angle_degrees))
    beta = m.cos(m.radians(angle_degrees))

    H = len(image)
    W = len(image[0])

    cy = (H - 1) / 2
    cx = (W - 1) / 2

    for i in range(H):
        for j in range(W):
            dy = i - cy
            dx = j - cx

            src_y = round(cy + dy * beta + dx * alpha)
            src_x = round(cx - dy * alpha + dx * beta)

            if 0 <= src_y < H and 0 <= src_x < W:
                new_image[i][j] = image[src_y][src_x]

    return new_image