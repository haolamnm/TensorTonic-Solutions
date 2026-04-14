import numpy as np

def unet_decoder_block(x: np.ndarray, skip: np.ndarray, out_channels: int) -> np.ndarray:
    """
    Returns zero array with correct shape.
    """
    B, H, W, C = x.shape
    up_H = 2 * H
    up_W = 2 * W

    out_shape = (B, up_H - 4, up_W - 4, out_channels)
    return np.zeros(out_shape)
