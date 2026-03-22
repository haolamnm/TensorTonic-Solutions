import numpy as np
import math as m

def cyclic_encoding(values, period):
    """
    Encode cyclic features as sin/cos pairs.
    """
    values = np.asarray(values)
    theta = 2 * m.pi * values / period

    # vectorized sin and cos
    theta_sin = np.sin(theta).tolist()
    theta_cos = np.cos(theta).tolist()

    encoded = []
    for sin, cos in zip(theta_sin, theta_cos):
        encoded.append([sin, cos])

    return encoded
    