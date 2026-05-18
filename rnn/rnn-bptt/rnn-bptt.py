import numpy as np

def bptt_single_step(dh_next: np.ndarray, h_t: np.ndarray, h_prev: np.ndarray,
                     x_t: np.ndarray, W_hh: np.ndarray) -> tuple:
    """
    Backprop through one RNN time step.
    Returns (dh_prev, dW_hh).
    """
    # YOUR CODE HERE
    d_tanh = dh_next * (1 - h_t*h_t)
    dh_prev = np.dot(d_tanh, W_hh)
    dW_hh = np.dot(d_tanh.T, h_prev)
    return dh_prev, dW_hh
    pass