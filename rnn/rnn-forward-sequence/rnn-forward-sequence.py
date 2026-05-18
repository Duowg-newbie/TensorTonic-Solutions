import numpy as np

def rnn_forward(X: np.ndarray, h_0: np.ndarray,
                W_xh: np.ndarray, W_hh: np.ndarray, b_h: np.ndarray) -> tuple:
    """
    Forward pass through entire sequence.
    """
    # YOUR CODE HERE
    T = X.shape[1]
    h_prev = h_0
    my_list = []
    for t in range(0, T):
        x_t = X[:, t, :]
        input_part = np.dot(x_t, W_xh.T)
        hidden_part = np.dot(h_prev, W_hh.T)
        linear_combination = input_part + hidden_part + b_h
        h_t = np.tanh(linear_combination)
        my_list.append(h_t)
        h_prev = h_t
    hidden_state = np.stack(my_list, axis = 1)
    h_final = h_prev
    return hidden_state, h_final
    pass