import numpy as np

def encoder(in_shape, out_channels):
    B, H, W, C = in_shape
    pool_shape = (B, (H - 4) // 2, (W - 4) // 2, out_channels)
    skip_shape = (B, H - 4, W - 4, out_channels)
    return (pool_shape, skip_shape)

def bottleneck(in_shape, out_channels):
    B, H, W, C = in_shape
    out_shape = (B, H - 4, W - 4, out_channels)
    return out_shape

def decoder(in_shape, skip_shape, out_channels):
    B, H, W, C = in_shape
    up_H = 2 * H
    up_W = 2 * W
    out_shape = (B, up_H - 4, up_W - 4, out_channels)
    return out_shape

def cnc(dec_shape, enc_shape):
    B, H_enc, W_enc, C_enc = enc_shape
    _, H_dec, W_dec, C_dec = dec_shape
    out_shape = (B, H_dec, W_dec, C_enc + C_dec)
    return out_shape

def output(in_shape, num_classes):
    B, H, W, C = in_shape
    out_shape = (B, H, W, num_classes)
    return out_shape

def unet(x: np.ndarray, num_classes: int = 2) -> np.ndarray:
    """
    Complete U-Net: trace shape through 4 encoder blocks, bottleneck, 4 decoder blocks, output.
    Each block: two 3x3 unpadded convs (reduce by 4), encoder pools (halve), decoder upsamples (double).
    Returns zero array with correct output shape.
    """
    pool1_shape, skip1_shape = encoder(x.shape, 64)
    pool2_shape, skip2_shape = encoder(pool1_shape, 128)
    pool3_shape, skip3_shape = encoder(pool2_shape, 256)
    pool4_shape, skip4_shape = encoder(pool3_shape, 512)
    
    bn_shape = bottleneck(pool4_shape, 1024)
    
    dec4_shape = decoder(bn_shape, skip4_shape, 512)
    cnc4_shape = cnc(dec4_shape, skip4_shape)
    
    dec3_shape = decoder(cnc4_shape, skip3_shape, 256)
    cnc3_shape = cnc(dec3_shape, skip3_shape)
    
    dec2_shape = decoder(cnc3_shape, skip2_shape, 128)
    cnc2_shape = cnc(dec2_shape, skip2_shape)
    
    dec1_shape = decoder(cnc2_shape, skip1_shape, 64)
    cnc1_shape = cnc(dec1_shape, skip1_shape)
    
    out = output(cnc1_shape, num_classes)

    return np.zeros(out)
    