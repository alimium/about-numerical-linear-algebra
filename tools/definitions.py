"""
This module holds functions to calculate definitions of Linear Algebra entities.
"""
import numpy as np


def condition_number(mat: np.ndarray, norm=2):
    """
    Calculate the condition number for a given matrix.

    NOTE: THIS IS NOT OPTIMIZED YET. USE WITH CAUTION!

    Parameters
    ----------
        mat(numpy.ndarray): input matrix (must be non singular)
        norm(str): norm used in calculations (`1`, `2`, `np.inf`, `'fro'`)

    Returns
    -------
        k(float): condition number for the input matrix
    """
    if len(mat.shape) != 2:
        raise ValueError(
            f"Shape Error: {len(mat.shape)} is not valid for input matrix dimention. Must be 2."
        )
    if mat.shape[0] != mat.shape[1]:
        raise ValueError("Shape Error: Input matrix must be square.")
    if np.linalg.det(mat) == 0:
        raise ValueError("Singularity Error: Input matrix should not be singular.")

    n_mat = np.linalg.norm(mat, ord=norm)
    n_mat_inv = np.linalg.norm(np.linalg.inv(mat), ord=norm)

    return n_mat * n_mat_inv
