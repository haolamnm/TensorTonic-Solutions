def bilinear_resize(image, new_h, new_w):
    """
    Resize a 2D grid using bilinear interpolation.
    """
    H = len(image)
    W = len(image[0])

    resized_grid = [[0.0 for _ in range(new_w)] for _ in range(new_h)]

    for i in range(new_h):
        for j in range(new_w):
            if new_h > 1:
                src_y = i * (H - 1) / (new_h - 1)
            else:
                src_y = 0.0
                
            if new_w > 1:
                src_x = j * (W - 1) / (new_w - 1)
            else:
                src_x = 0.0
            
            y0 = int(src_y)
            x0 = int(src_x)
            dy = src_y - y0
            dx = src_x - x0
            
            y1 = min(y0 + 1, H - 1)
            x1 = min(x0 + 1, W - 1)
            
            # apply the interpolation formula
            # (y0, x0), (y1, x0), (y0, x1), (y1, x1)
            top_left = image[y0][x0] * (1 - dy) * (1 - dx)
            bottom_left = image[y1][x0] * dy * (1 - dx)
            top_right = image[y0][x1] * (1 - dy) * dx
            bottom_right = image[y1][x1] * dy * dx
            
            resized_grid[i][j] = top_left + bottom_left + top_right + bottom_right
            
    return resized_grid