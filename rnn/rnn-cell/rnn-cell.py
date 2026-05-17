import numpy as np

def rnn_cell(x_t: np.ndarray, h_prev: np.ndarray, 
             W_xh: np.ndarray, W_hh: np.ndarray, b_h: np.ndarray) -> np.ndarray:
    """
    Single RNN cell forward pass.
    """
    # YOUR CODE HERE
    input_part = np.dot(h_prev, W_hh.T)
    hidden_part = np.dot(x_t, W_xh.T)
    linear_combination = input_part + hidden_part + b_h
    h_t = np.tanh(linear_combination)
    return h_t
    pass