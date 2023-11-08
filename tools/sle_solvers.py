"""
This module provides several algorithms for solving systems of linear equations.
"""

import numpy as np


def thomas(mat: np.ndarray, b: np.ndarray):
    """
    Solves tri-diagonal system of linear equeations using the Thomas algorithm.

    Parameters
    ----------
        mat(numpy.ndarray): the coefficient matrix of the system (usually denoted by A)
            `This must be a tri-diagonal square matrix`
        f(numpy.ndarray): constant vector (the right side of the equation, usually denoted by b)

    Returns
    -------
        x(numpy.ndarray): sulution vector to the system.
    """

    # TODO: chack if the matrix is tri-diagonal and square
    #
    # ====================================================

    # change matrix representation for better memory compensation
    n = mat.shape[0]

    # TODO: THIS BIT NEEDS OPTIMIZATION ==================
    a = np.array([0] + [mat[i, i - 1] for i in range(1, n)])
    f = np.array([mat[i, i] for i in range(n)])
    c = np.array([mat[i, i + 1] for i in range(n - 1)] + [0])
    # ====================================================
    alpha = np.ndarray(shape=n)
    beta = np.ndarray(shape=n)
    y = np.ndarray(shape=n)
    x = np.ndarray(shape=n)

    alpha[0], y[0] = f[0], b[0]
    for i in range(1, n):
        beta[i] = a[i] / alpha[i - 1]
        alpha[i] = f[i] - beta[i] * c[i - 1]
        y[i] = b[i] - beta[i] * y[i - 1]

    x[n - 1] = y[n - 1] / alpha[n - 1]
    for i in range(n - 2, -1, -1):
        x[i] = (y[i] - c[i] * x[i + 1]) / alpha[i]

    return x


def gauss_seidel(
    mat: np.ndarray,
    b: np.ndarray,
    x_0: np.ndarray = None,
    eps: float = 1e-10,
    stop_cond=lambda a, b, x: np.linalg.norm(b - np.matmul(a, x)) / np.norm(b),
):
    x = np.zeros_like(b) if x_0 is None else x_0
