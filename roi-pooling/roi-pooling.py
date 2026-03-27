import math as m
import numpy as np

def roi_pool(feature_map, rois, output_size):
    """
    Apply ROI Pooling to extract fixed-size features.
    """
    # create bin indices
    indices = []
    for i in range(output_size):
        for j in range(output_size):
            indices.append((i, j))

    # create result container
    results = []
    feature_map = np.asarray(feature_map, dtype=int)

    # pooling
    for x1, y1, x2, y2 in rois:
        roi_h, roi_w = y2 - y1, x2 - x1
        result = np.zeros((output_size, output_size))
        for i, j in indices:
            h_start = y1 + m.floor(i       * roi_h / output_size)
            h_end   = y1 + m.floor((i + 1) * roi_h / output_size)

            w_start = x1 + m.floor(j       * roi_w / output_size)
            w_end   = x1 + m.floor((j + 1) * roi_w / output_size)

            h_end = max(h_end, h_start + 1)
            w_end = max(w_end, w_start + 1)

            view = feature_map[h_start:h_end, w_start:w_end] 
            print(view)
            result[i][j] = np.max(view)

        results.append(result.tolist())

    return results