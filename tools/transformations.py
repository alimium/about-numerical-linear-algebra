import numpy as np


def householder(u: np.ndarray):
    """
    Compute the Householder transformation matrix.

    Parameters:
        u (np.ndarray): The vector used to compute the Householder transformation.

    Returns:
        np.ndarray: The Householder transformation matrix.
    """
    I = np.eye(u.shape[0])
    H = I - 2 / (u.T @ u) * np.outer(u, u)
    return H
