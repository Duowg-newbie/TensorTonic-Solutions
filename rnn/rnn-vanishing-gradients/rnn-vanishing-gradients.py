import numpy as np

def compute_gradient_norm_decay(T: int, W_hh: np.ndarray) -> list:
    """
    Simulate gradient norm decay over T time steps.
    Returns list of gradient norms.
    """
    # YOUR CODE HERE
    w_norm = np.linalg.norm(W_hh, ord = 2)
    my_list = []
    d_t = 1.0
    my_list.append(d_t)
    for t in range(1, T):
        d_t = d_t * w_norm
        my_list.append(d_t)
    return my_list
    pass