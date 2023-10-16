"""
This module holds several matrix factorization/decomposition
functions. It is yet to be comlete though.
"""

import numpy as np


def qr_factorize(mat: np.ndarray, intensive_mode=False):
    """
    Factorizes a rectangular matrix A into a rectangular isometric matrix Q and an upper
    triangular matrix R. (This is quite slow and doesn't work on singular matrices as I'm
    learning along. But I'll make it better!)

    Parameters
    ----------
        `mat:numpy.ndarray` - the matrix to be factorized
        `intensive_mode:bool` - a fancy word that returns all the internal matrices in
                                calculations if True. (more like debug mode)

    Returns
    -------
        `Q:numpy.ndarray` - the isometric matrix
        `R:numpy.ndarray` - the upper triangular matrix
        `results:dict(Q, R, Q_hat, R_hat, D, D_inv)` - details (if `intensive_mode=True`)
    """
    # mxn
    q_hat = np.zeros_like(mat, dtype=np.float32)

    # nxn
    r_hat = np.triu(
        np.ones(shape=(mat.shape[1], mat.shape[1]), dtype=np.float32))

    # nxn
    d = np.identity(n=mat.shape[1], dtype=np.float32)

    q_hat.T[0] = mat.T[0]
    d[0, 0] = np.linalg.norm(q_hat.T[0])
    for j in range(1, mat.shape[1]):
        q_hat.T[j] = mat.T[j]
        for i in range(j+1):
            if i == j:
                d[i, j] = np.linalg.norm(q_hat.T[j])
            else:
                r_hat[i, j] = np.matmul(q_hat.T[i], mat.T[j]) / \
                    np.matmul(q_hat.T[i], q_hat.T[i])
                q_hat.T[j] -= (r_hat[i, j]*q_hat.T[i])

    d_inv = np.diag([1/d[i, i] for i in range(d.shape[0])])
    q = np.matmul(q_hat, d_inv)
    r = np.matmul(d, r_hat)

    if intensive_mode:
        results = {
            'Q': q,
            'R': r,
            'Q_hat': q_hat,
            'R_hat': r_hat,
            'D': d,
            'D_inv': d_inv,
        }
        return results

    return q, r


def validate(mat, q, r, threshold=1e-8):
    """
    Validates the accuracy of a decomposition by checking if all elements'
    difference with the original matrix(`mat`) are less than the error threshold.

    Parameters
    ----------
        `mat:numpy.ndarray` - the original matrix
        `Q:numpy.ndarray` - decomposition output Q
        `R:numpy.ndarray` - decomposition output R

    Returns
    -------
        Tuple[bool, float] - (if all of the elements have passed the accuracy
                              test, maximum absolute error within the obtained matrix)
    """
    res = np.matmul(q, r)
    err = np.absolute(mat-res)
    return np.all(err <= threshold), np.max(err)
