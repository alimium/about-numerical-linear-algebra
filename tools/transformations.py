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


def upper_hessenberg(A: np.ndarray):
    """
    Compute the upper Hessenberg form of a matrix.

    Parameters:
        A (np.ndarray): The matrix to be transformed.

    Returns:
        np.ndarray: The upper Hessenberg form of the matrix.
    """
    n = A.shape[0]
    for j in range(n - 2):
        x = A[j + 1 :, j]
        e1 = np.zeros_like(x)
        e1[0] = 1
        v = x + (1 if x[0] >= 0 else -1) * np.linalg.norm(x, ord=2) * e1
        Hj = np.eye(n)
        Hj[j + 1 :, j + 1 :] = householder(v)
        A = Hj @ A @ Hj.T
    return A
