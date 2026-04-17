import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

class LSTM:
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        self.hidden_dim = hidden_dim
        scale = np.sqrt(2.0 / (input_dim + hidden_dim))

        self.W_f = np.random.randn(hidden_dim, hidden_dim + input_dim) * scale
        self.W_i = np.random.randn(hidden_dim, hidden_dim + input_dim) * scale
        self.W_c = np.random.randn(hidden_dim, hidden_dim + input_dim) * scale
        self.W_o = np.random.randn(hidden_dim, hidden_dim + input_dim) * scale
        self.b_f = np.zeros(hidden_dim)
        self.b_i = np.zeros(hidden_dim)
        self.b_c = np.zeros(hidden_dim)
        self.b_o = np.zeros(hidden_dim)

        self.W_y = np.random.randn(output_dim, hidden_dim) * np.sqrt(2.0 / (hidden_dim + output_dim))
        self.b_y = np.zeros(output_dim)

    def forward(self, X: np.ndarray) -> tuple:
        """
        Forward pass. Returns (y, h_last, C_last).
        """
        N, T, D = X.shape

        shape = (N, self.hidden_dim)
        h_prev = np.atleast_2d(np.zeros(shape))
        C_prev = np.atleast_2d(np.zeros(shape))

        H = []
        C = []

        for t in range(T):
            x_t = X[:, t, :]
            input = np.concatenate([h_prev, x_t], axis=-1)
    
            f_t = sigmoid((self.W_f @ input.T).T + self.b_f)
    
            i_t = sigmoid((self.W_i @ input.T).T + self.b_i)
            C_t_tilde = np.tanh((self.W_c @ input.T).T + self.b_c)
    
            o_t = sigmoid((self.W_o @ input.T).T + self.b_o)
            C_t = f_t * C_prev + i_t * C_t_tilde
    
            h_t = o_t * np.tanh(C_t)

            H.append(h_t)
            C.append(C_t)

            h_prev = h_t
            C_prev = C_t

        H = np.array(np.stack(H, axis=1))
        C = np.array(np.stack(C, axis=1))

        y = H @ self.W_y.T + self.b_y 
    
        return (y, H[:, -1, :], C[:, -1, :])