"""
This module provides several algorithms for solving systems of linear equations.
"""

import numpy as np


def thomas(mat: np.ndarray, f: np.ndarray):
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
    b = np.array([mat[i, i] for i in range(n)])
    c = np.array([mat[i, i + 1] for i in range(n - 1)] + [0])
    # ====================================================
    alpha = np.ndarray(shape=n)
    beta = np.ndarray(shape=n)
    y = np.ndarray(shape=n)
    x = np.ndarray(shape=n)

    alpha[0], y[0] = b[0], f[0]
    for i in range(1, n):
        beta[i] = a[i] / alpha[i - 1]
        alpha[i] = b[i] - beta[i] * c[i - 1]
        y[i] = f[i] - beta[i] * y[i - 1]

    x[n - 1] = y[n - 1] / alpha[n - 1]
    for i in range(n - 2, -1, -1):
        x[i] = (y[i] - c[i] * x[i + 1]) / alpha[i]

    return x
