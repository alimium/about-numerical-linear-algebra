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


def eigenvalue_scaled_power(
    A: np.ndarray, v_0: np.ndarray, eps: float = 5e-5, max_iter: int = 100
):
    def stop_condition(l_k, l_k_1, eps):
        return np.abs(l_k - l_k_1) <= eps

    l = 0
    iter_ = 1
    v_prev = v_0
    while max_iter > iter_:
        v = A @ v_prev
        new_l = np.linalg.norm(v, 1) / np.linalg.norm(v_prev, 1)
        if stop_condition(new_l, l, eps):
            return new_l
        l = new_l
        v_prev = v
        iter_ += 1
    print("reached max iter")
    return l


def eigenvalue_inverse_scaled_power(
    A: np.ndarray, v_0: np.ndarray, eps: float = 5e-5, max_iter: int = 100
):
    def stop_condition(l_k, l_k_1, eps):
        return np.abs(l_k - l_k_1) <= eps

    l = 0
    iter_ = 1
    v_prev = v_0
    while max_iter > iter_:
        v = np.linalg.solve(A, v_0)
        new_l = np.linalg.norm(v, 1) / np.linalg.norm(v_prev, 1)
        if stop_condition(new_l, l, eps):
            return new_l
        l = new_l
        v_prev = v
        iter_ += 1
    print("reached max iter")
    return l
