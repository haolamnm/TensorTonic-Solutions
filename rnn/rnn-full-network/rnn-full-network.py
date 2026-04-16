import numpy as np

class VanillaRNN:
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        self.hidden_dim = hidden_dim

        # Xavier initialization
        self.W_xh = np.random.randn(hidden_dim, input_dim) * np.sqrt(2.0 / (input_dim + hidden_dim))
        self.W_hh = np.random.randn(hidden_dim, hidden_dim) * np.sqrt(2.0 / (2 * hidden_dim))
        self.W_hy = np.random.randn(output_dim, hidden_dim) * np.sqrt(2.0 / (hidden_dim + output_dim))
        self.b_h = np.zeros(hidden_dim)
        self.b_y = np.zeros(output_dim)

    def forward(self, X: np.ndarray, h_0: np.ndarray = None) -> tuple:
        """
        Forward pass through entire sequence.
        Returns (y_seq, h_final).
        """
        batch_size, T, input_dim = X.shape
        h_prev = h_0
        H = []
        Y = []
        
        for t in range(T):
            x_t = X[:, t, :]
            h_t = np.tanh(x_t @ self.W_xh.T + h_prev @ self.W_hh.T + self.b_h)
            y_t = h_t @ self.W_hy.T + self.b_y
            
            H.append(h_t)
            Y.append(y_t)
            
            h_prev = h_t

        H = np.stack(H, axis=1)
        Y = np.stack(Y, axis=1)

        return (Y, H[:, -1, :])