import numpy as np

def candidate_hidden(h_prev: np.ndarray, x_t: np.ndarray, r_t: np.ndarray,
                     W_h: np.ndarray, b_h: np.ndarray) -> np.ndarray:
    """
    Compute candidate: h_tilde = tanh(W_h @ [r*h, x] + b_h)
    """
    # YOUR CODE HERE
    h_tidle = np.tanh(np.dot(np.concatenate([r_t * h_prev, x_t], axis = -1), W_h.T) + b_h)
    return h_tidle
    pass