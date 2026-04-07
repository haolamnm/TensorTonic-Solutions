import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

def morphological_op(image, kernel, operation):
    """
    Apply morphological erosion or dilation to a binary image.
    """
    image = np.asarray(image)
    kernel = np.asarray(kernel)
    mask = kernel.astype(bool)

    ih, iw = image.shape
    kh, kw = kernel.shape
    # print(kh, kw)
    
    pad_width = ((kh // 2,), (kw // 2,))
    padded = np.pad(image, pad_width=pad_width, mode="constant")
    # print(padded)

    windows = sliding_window_view(padded, (kh, kw))
    # print(windows.shape)

    handler = {
        "erode": np.all,
        "dilate": np.any
    }

    vh, vw, _, _ = windows.shape
    out = []
    for i in range(vh):
        if i >= ih:
            break
        line = []
        for j in range(vw):
            if j >= iw:
                break
            view = windows[i, j, :, :][mask]
            o = int(handler[operation](view))
            line.append(o)
        out.append(line)

    return out
            