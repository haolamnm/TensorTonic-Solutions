import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

def gru_cell(x_t: np.ndarray, h_prev: np.ndarray,
             W_r: np.ndarray, W_z: np.ndarray, W_h: np.ndarray,
             b_r: np.ndarray, b_z: np.ndarray, b_h: np.ndarray) -> np.ndarray:
    """
    Complete GRU cell forward pass.
    """
    h_prev = np.atleast_2d(h_prev)
    x_t = np.atleast_2d(x_t)
    
    in_state = np.concatenate([h_prev, x_t], axis=-1)
    
    r_t = sigmoid((W_r @ in_state.T).T + b_r)
    z_t = sigmoid((W_z @ in_state.T).T + b_z)

    fo_state = np.concatenate([r_t * h_prev, x_t], axis=-1)
    
    h_tilde = np.tanh((W_h @ fo_state.T).T + b_h)
    h_t = z_t * h_prev + (1 - z_t) * h_tilde

    # trouble maker!
    # h_t = h_t.squeeze(0) if h_t.shape[0] == 1 else h_t

    return h_t

class GRU:
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        self.hidden_dim = hidden_dim
        scale = np.sqrt(2.0 / (input_dim + hidden_dim))

        self.W_r = np.random.randn(hidden_dim, hidden_dim + input_dim) * scale
        self.W_z = np.random.randn(hidden_dim, hidden_dim + input_dim) * scale
        self.W_h = np.random.randn(hidden_dim, hidden_dim + input_dim) * scale
        self.b_r = np.zeros(hidden_dim)
        self.b_z = np.zeros(hidden_dim)
        self.b_h = np.zeros(hidden_dim)

        self.W_y = np.random.randn(output_dim, hidden_dim) * np.sqrt(2.0 / (hidden_dim + output_dim))
        self.b_y = np.zeros(output_dim)

    def forward(self, X: np.ndarray) -> tuple:
        """
        Forward pass. Returns (y, h_last).
        """
        N, T, _ = X.shape
        H_seq = []
        h_curr = np.zeros((N, self.hidden_dim))

        for t in range(T):
            x_t = X[:, t, :]
            h_curr = gru_cell(
                x_t, h_curr,
                self.W_r, self.W_z, self.W_h,
                self.b_r, self.b_z, self.b_h
            )
            H_seq.append(h_curr)

        H = np.stack(H_seq, axis=1) # (N, T, hidden_dim)
        
        Y = H @ self.W_y.T + self.b_y # (N, T, output_dim)

        return Y, h_curr