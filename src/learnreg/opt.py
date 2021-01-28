"""
functions for optimization
"""

import cvxpy as cp
import functools
import torch
import numpy as np

def eval_lasso(A, x, y, beta, W):
    """
    compute J(x) = 1/2 || Ax - y ||_2^2 + beta ||W x||_1

    """
    A, x, y, W = (torch.as_tensor(X) for X in (A, x, y, W))

    return 0.5 * ((A @ x - y)**2).sum() + beta * (W @ x).abs().sum()


def solve_lasso(A, y, beta, W, method, **opts):
    """
    argmin_x 1/2 || Ax - y ||_2^2 + beta ||W x||_1

    using different possible solvers
    """

    if method == 'ADMM':
        return solve_lasso_ADMM(A, y, beta, W, **opts)
    elif method == 'cvxpy':
        return solve_lasso_cvxpy(A, y, beta, W)
    else:
        raise ValueError(method)


def solve_lasso_ADMM(A, y, beta, W, num_steps, rho):
    """
    argmin_x 1/2 || Ax - y ||_2^2 + beta ||W x||_1

    based on https://web.stanford.edu/~boyd/papers/pdf/admm_distr_stats.pdf
    6.4.1  Generalized Lasso

    A, y, W - pytorch tensors
    beta, rho - scalars

    """

    # force everything to be tensors
    # no copying is done if already tensors
    A, y, W = (torch.as_tensor(X) for X in (A, y, W))

    z = torch.zeros((W.shape[0], y.shape[1]), dtype=y.dtype)
    u = torch.zeros_like(z)

    # precomputations
    Q = (A.T @ A + rho * W.T @ W)
    Q_LU = torch.lu(Q)  # for useful for solving Qx = b, later
    # see https://pytorch.org/docs/stable/generated/torch.lu_solve.html

    ATy = A @ y

    threshold = beta/rho

    for step in range(num_steps):
        x = torch.lu_solve(ATy + rho * W.T @ (z - u), *Q_LU)
        Wx = W @ x
        z = torch.nn.functional.softshrink(Wx + u, threshold)
        u = u + Wx - z

    return x


def solve_lasso_cvxpy(A, y, beta, W):
    """
    argmin_x 1/2 || Ax - y ||_2^2 + beta ||W x||_1

    using cvxpy
    """

    k = W.shape[0]
    num = y.shape[1]
    problem, params = setup_cvxpy_problem(*A.shape, k, num)
    params['A'].value = np.array(A)
    params['y'].value = np.array(y)
    params['beta'].value = np.array(beta)
    params['W'].value = W

    problem.solve()

    return params['x'].value


@functools.lru_cache()  # caching because making the problem object is slow
def setup_cvxpy_problem(m, n, k, batch_size=1):
    """
    sets up a cvxpy Problem representing

    argmin_x 1/2||Ax - y||_2^2 + beta||Wx||_1

    where A, y, beta, and W are left as free parameters

    A: m x n
    x: n x batch_size
    y: m x batch_size
    beta: scalar
    W: k x n

    """
    A = cp.Parameter((m,n), name='A')
    x = cp.Variable((n, batch_size), name='x')
    y = cp.Parameter((m, batch_size), name='y')
    beta = cp.Parameter(name='beta', nonneg=True)
    W = cp.Parameter((k, n), name='W')
    z = cp.Variable((k, batch_size), name='z')

    objective_fn = 0.5 * cp.sum_squares(A @ x - y) + beta*cp.sum(cp.abs(z))
    constraint = [W @ x == z]  # writing with this splitting makes is_dpp True
    problem = cp.Problem(cp.Minimize(objective_fn), constraint)

    params = {
        'A':A,
        'y':y,
        'beta':beta,
        'W':W,
        'z':z,
        'x':x
    }

    return problem, params
