import numpy as np

def crop_and_concat(encoder_features: np.ndarray, decoder_features: np.ndarray) -> np.ndarray:
    """
    Crop encoder features to match decoder spatial dims, then concatenate along channels.
    """
    B_enc, H_enc, W_enc, C_enc = encoder_features.shape
    B_dec, H_dec, W_dec, C_dec = decoder_features.shape

    H_diff = H_enc - H_dec
    W_diff = W_enc - W_dec

    H_start = H_diff // 2
    W_start = W_diff // 2
    
    cropped = encoder_features[:, H_start:H_start+H_dec, W_start:W_start+W_dec, :]
    return np.concatenate([cropped, decoder_features], axis=3)