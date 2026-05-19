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
        # YOUR CODE HERE
        N = X.shape[0]
        T = X.shape[1]
        my_list = []
        h_t = np.zeros((N, self.hidden_dim))
        C_t = np.zeros((N, self.hidden_dim))

        for t in range(T):
            x_t = X[:, t, :]
            f_t = sigmoid(np.dot(np.concatenate([h_t, x_t], axis = -1), self.W_f.T) + self.b_f)
            i_t = sigmoid(np.dot(np.concatenate([h_t, x_t], axis = -1), self.W_i.T) + self.b_i)
            c_tidle = np.tanh(np.dot(np.concatenate([h_t, x_t], axis = -1), self.W_c.T) + self.b_c)
            o_t = sigmoid(np.dot(np.concatenate([h_t, x_t], axis = -1), self.W_o.T) + self.b_o)
            C_t = (f_t * C_t) + (i_t * c_tidle)
            h_t = o_t * np.tanh(C_t)
            y_t = np.dot(h_t, self.W_y.T) + self.b_y
            my_list.append(y_t)

        y = np.stack(my_list, axis = 1)
        h_last = h_t
        C_last = C_t
        return y, h_last, C_last
        pass