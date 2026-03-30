import numpy as np

def relu(x):
    return np.maximum(0, x)

class IdentityBlock:
    """
    Identity Block: F(x) + x
    Used when input and output dimensions match.
    """
    
    def __init__(self, channels: int):
        self.channels = channels
        # Simplified: using dense layers instead of conv for demo
        self.W1 = np.random.randn(channels, channels) * 0.01
        self.W2 = np.random.randn(channels, channels) * 0.01
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass: y = ReLU(W2 @ ReLU(W1 @ x)) + x
        """
        # (B * H * W, C)
        x = np.asarray(x)

        # (C_out, C_in) @ (C, B * H * W)
        z_t = relu(self.W1 @ x.T)
        F_x = relu(self.W2 @ z_t)
        y = F_x.T + x
        
        return y
    