"""
This module holds functions that generate different NLA stuff such as matrices and vectors
"""
import numpy as np


def hilbert(n: int):
    """
    generate Hilbert matrices of size n.

    Parameters
    ----------
        n(int): dimention of the matrix

    Returns
    -------
        h(numpy.ndarray): generated Hilbert matrix

    Notes
    -----
        This currently returns Hilbert matrices of maximum size 10000 in a reasonable time.
    """

    def _element_calc(i, j):
        return np.ones_like(i) / (i + j - np.ones_like(i))

    calc_func = np.vectorize(_element_calc)

    i_s = np.array([(i + 1) * np.ones(shape=(n)) for i in range(n)])
    j_s = np.array([(j + 1) * np.ones(shape=(n)) for j in range(n)]).T
    h = calc_func(i_s, j_s)

    return h


def vandermonde(x):
    """
    Generate a Vandermonde matrix based on the given input vector x.

    Parameters
    ----------
    x : numpy.ndarray
        Input vector.

    Returns
    -------
    v : numpy.ndarray
        Vandermonde matrix.

    """
    n = len(x)
    powers = np.arange(n)
    v = np.column_stack([x**powers])
    return v
