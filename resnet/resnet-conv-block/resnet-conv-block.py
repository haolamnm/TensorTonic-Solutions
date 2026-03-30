import numpy as np

def relu(x):
    return np.maximum(0, x)

class ConvBlock:
    """
    Convolutional Block with projection shortcut.
    Used when input/output dimensions differ.
    """
    
    def __init__(self, in_channels: int, out_channels: int):
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # Main path weights
        self.W1 = np.random.randn(in_channels, out_channels) * 0.01
        self.W2 = np.random.randn(out_channels, out_channels) * 0.01
        
        # Shortcut projection (1x1 conv equivalent)
        self.Ws = np.random.randn(in_channels, out_channels) * 0.01
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass with projection shortcut.
        """
        # (N, C_in)
        x = np.asarray(x)

        # main path
        # (C_out, C_in) @ (C_in, N)
        h_t = relu(self.W1.T @ x.T)
        z_t = self.W2.T @ h_t

        # shortcut
        # (C_out, C_in) @ (C_in, N)
        s_t = self.Ws.T @ x.T

        y = relu(z_t + s_t).T
        
        return y