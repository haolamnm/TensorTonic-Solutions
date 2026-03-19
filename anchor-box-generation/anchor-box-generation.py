import numpy as np

def generate_anchors(feature_size, image_size, scales, aspect_ratios):
    """
    Generate anchor boxes for object detection.
    """
    stride = image_size / feature_size

    idx = np.arange(feature_size)
    cx = (idx + 0.5) * stride # (feature_size,)
    cy = (idx + 0.5) * stride

    cx, cy = np.meshgrid(cx, cy) # (feature_size, feature_size)
    cx = cx.reshape(-1) # (feature_size^2,)
    cy = cy.reshape(-1)

    scales = np.asarray(scales)
    ratios = np.asarray(aspect_ratios)

    r, s = np.meshgrid(ratios, scales)
    w = (s * r**0.5).reshape(-1) # (n_anchors_per_cell,)
    h = (s / r**0.5).reshape(-1)

    # (feature_size^2, n_anchors_per_cell)
    x1 = cx[:, None] - w[None, :] / 2
    y1 = cy[:, None] - h[None, :] / 2
    x2 = cx[:, None] + w[None, :] / 2
    y2 = cy[:, None] + h[None, :] / 2

    # (feature_size^2, n_anchors_per_cell, 4)
    anchors = np.stack([x1, y1, x2, y2], axis=-1)
    return anchors.reshape(-1, 4).tolist()
    