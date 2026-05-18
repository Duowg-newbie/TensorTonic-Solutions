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
        # YOUR CODE HERE
        T = X.shape[1]

        if h_0 is None:
            h_0 = np.zeros((batch_size, self.hidden_dim))
        
        h_prev = h_0
        my_list = []
        for t in range(T):
            x_t = X[:, t, :]
            h_t = np.tanh(np.dot(x_t, self.W_xh.T) + np.dot(h_prev, self.W_hh.T) + self.b_h)
            y_t = np.dot(h_t, self.W_hy.T) + self.b_y
            my_list.append(y_t)
            h_prev = h_t
            
        y_seq = np.stack(my_list, axis = 1)
        h_final = h_t
        return y_seq, h_final
        pass