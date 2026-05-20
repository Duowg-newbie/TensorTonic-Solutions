import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

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
        # YOUR CODE HERE
        N = X.shape[0]
        T = X.shape[1]
        my_list = []
        h_t = np.zeros((N, self.hidden_dim))

        for t in range(T):
            x_t = X[:, t, :]
            r_t = sigmoid(np.dot(np.concatenate([h_t, x_t], axis = -1), self.W_r.T) + self.b_r)
            z_t = sigmoid(np.dot(np.concatenate([h_t, x_t], axis = -1), self.W_z.T) + self.b_z)
            h_tidle = np.tanh(np.dot(np.concatenate([r_t * h_t, x_t], axis = -1), self.W_h.T) + self.b_h)
            h_t = (z_t * h_t) + ((1 - z_t) * h_tidle)
            y_t = np.dot(h_t, self.W_y.T) + self.b_y
            my_list.append(y_t)

        y_t = np.stack(my_list, axis = 1)
        h_last = h_t
        return y_t, h_last
        pass