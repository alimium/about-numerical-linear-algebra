"""
This module provides several algorithms for solving systems of linear equations.
"""

import numpy as np
from dataclasses import dataclass


class StopCondition:
    def __init__(self, mode: str):
        if mode == "absolute":
            self.f = lambda xn, xt: np.max(np, abs(xn - xt))
        if mode == "relative":
            self.f = lambda xn, xt: np.max(np.abs(xn - xt) / np.abs(xt))
        if mode == "absremainder":
            self.f = lambda A, b, xk: np.linalg.norm(b - (A @ xk))
        if mode == "relremainder":
            self.f = lambda A, b, xk: np.linalg.norm(b - (A @ xk)) / np.linalg.norm(b)


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


def successive_over_relaxation(
    mat: np.ndarray,
    b: np.ndarray,
    x_0: np.ndarray = None,
    omega: float = 1,
    eps: float = 1e-10,
    stop_cond=lambda a, b, x: np.linalg.norm(b - np.matmul(a, x), 2)
    / np.linalg.norm(b, None),
    max_iter: int = 500,
    verbose: bool = False,
) -> np.ndarray:
    """
    Solve a system of linear equations (Ax=b) using SOR method. Things to consider:
        - if no initial guess is provided, zero vector will be used.
        - if no `omega` is provided, Gauss-Seidel method will be used (`omega=1`)

    Parameters
    ----------
        mat(numpy.ndarray): coefficient matrix of the SLE
        b(numpy.ndarray): right hand side of the SLE
        x_0(numpy.ndarray): initial guess
        omega(float): omega hyperparameter
        eps(float): epsilon for error bound
        stop_cond(function): for calculation of the stop condition
        max_iter(int): maximum iterations
        vrebose(bool): whether to print results at each step. good for debugging

    Returns
    -------
        x(numpy.ndarray): solution to the SLE

    Notes
    -----
        There is no automatic way of opitimizing `omega` yet.

        you need to provide the optimized value yourself but I

        think I will add this functionality soon
    """
    assert (
        0 < omega < 2
    ), f"Omega out of bounds: {omega} not in (0,2)! SOR will not converge."
    assert stop_cond is not None, "Stop condition cannot be None"

    x = x_0
    iter_num = 1

    while stop_cond(mat, b, x) >= eps and iter_num <= max_iter:
        for i in range(x.shape[0]):
            alpha = 0
            beta = 0
            for j in range(i - 1):
                alpha += mat[i, j] * x[j]
            for j in range(i - 1, x.shape[0]):
                beta += mat[i, j] * x[j]
            x[i] += omega / mat[i, i] * (b[i] - alpha - beta)
        if verbose:
            print(
                f"Iteration {iter_num:>2} {'|'} Stop Condition = {stop_cond(mat,b,x):.10f} | X = {x}"
            )

        iter_num += 1

    if iter_num > max_iter:
        print("Maximum iterations reached")
    else:
        print(f"Solution found on iteration {iter_num}")

    return x


def vectorized_jacobean(
    coef: np.ndarray,
    rhs: np.ndarray,
    x_0: np.ndarray = None,
    stop_cond: StopCondition = StopCondition("absremainder"),
    eps: float = 1e-8,
    max_iter: int = 500,
):
    """
    TODO: REFACTOR THIS FUNCTION TO UTILIZE VECTORIZATION.
    """
    x = np.zeros_like(rhs) if x_0 is None else x_0

    def calc_func(A, b, x, stp_cnd, eps, max_iter):
        d = np.diag(np.diag(A))
        d_inv = np.linalg.inv(d)
        lu = A - d
        m = -d_inv @ lu
        c = d_inv @ b
        i = 0
        while stp_cnd.f(A, b, x) >= eps and i < max_iter:
            x = m @ x + c
            i += 1
        return x

    # calc_func = np.vectorize(calc_func)
    return calc_func(coef, rhs, x, stop_cond, eps, max_iter)
