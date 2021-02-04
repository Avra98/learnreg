"""
functions for optimization
"""

import cvxpy as cp
import functools
import torch
import numpy as np
import scipy.optimize
import scipy.linalg

def eval_lasso(A, x, y, beta, W):
    """
    compute J(x) = 1/2 || Ax - y ||_2^2 + beta ||W x||_1

    """

    return 0.5 * np.sum((A @ x - y)**2) + beta * np.sum(np.abs(W @ x))


def solve_lasso(A, y, beta, W, method, **opts):
    """
    argmin_x 1/2 || Ax - y ||_2^2 + beta ||W x||_1

    using different possible solvers
    """

    if method == 'ADMM':
        x = solve_lasso_ADMM(A, y, beta, W, **opts)
    elif method == 'cvxpy':
        x = solve_lasso_cvxpy(A, y, beta, W)
    elif method == 'dual_scipy':
        x = solve_lasso_dual_scipy(A, y, beta, W)
    elif method == 'dual_PGD':
        x = solve_lasso_dual_PGD(A, y, beta, W, **opts)
    else:
        raise ValueError(method)

    return x

def solve_lasso_dual_scipy(A, y, beta, W):
    """
    exploratory at this point, assumes A = I
    """

    x = np.zeros((A.shape[1], y.shape[1]))

    for i in range(y.shape[1]):
        u = scipy.optimize.lsq_linear(W.T, y[:,i], (-beta, beta)).x
        x[:,i] = y[:,i] - W.T @ u

    return x

def solve_lasso_dual_PGD(A, y, beta, W, num_steps=100, step_size=None):
    """
    exploratory at this point, assumes A = I

    || y - WT u || st -beta <= ui <= beta
    """

    Wy = W @ y

    u = Wy

    WWT = W @ W.T

    if step_size is None:
        step_size = 1 / np.linalg.norm(WWT, 2)

        # optimal step size given by 1/L (see globalbioim)

    for step in range(num_steps):
        u = u - step_size * (WWT @ u - Wy)
        u[u > beta] = beta
        u[u < -beta] = -beta



    info = {
        'step_size': step_size,
        'u': u}

    x = np.zeros_like(y)
    for i in range(y.shape[1]):
        s = np.sign(u[np.abs(u[:, i]) == beta, i])
        Wpm = W[np.abs(u[:, i]) == beta, :]
        W0 = W[np.abs(u[:, i]) != beta, :]
        yterm = y[:, i] - beta * Wpm.T @ s
        x[:, i] = yterm - (np.linalg.pinv(W0) @ W0) @ yterm


    return x #y - W.T @ u




def solve_lasso_ADMM(A, y, beta, W, num_steps, rho):
    """
    argmin_x 1/2 || Ax - y ||_2^2 + beta ||W x||_1

    based on https://web.stanford.edu/~boyd/papers/pdf/admm_distr_stats.pdf
    6.4.1  Generalized Lasso

    plus the penalty parameter updates from 3.4.1

    A, y, W - np arrays
    beta, rho - scalars

    """
    # parameters for rho updates
    t_incr = 2.0
    t_decr = 2.0
    mu_sq = 10.0**2

    # over-relaxation (from https://web.stanford.edu/~boyd/papers/admm/total_variation/total_variation.html)
    alpha = 1.8  #1.5  # typically 1.0 to 1.8

    z = np.zeros((W.shape[0], y.shape[1]), dtype=y.dtype)
    u = np.zeros_like(z)

    # precomputations
    Q = (A.T @ A + rho * W.T @ W)
    Q_LU = scipy.linalg.lu_factor(Q)  # for useful for solving Qx = b, later
    # see https://pytorch.org/docs/stable/generated/torch.lu_solve.html

    threshold = beta/rho

    ATy = A @ y



    for step in range(num_steps):
        x = scipy.linalg.lu_solve(Q_LU, ATy + rho * W.T @ (z - u))
        Wx = alpha * W @ x + (1 - alpha) * z
        z_new = Wx + u
        z_new = np.sign(z_new) * np.maximum(np.abs(z_new)-threshold, 0)  # this is soft threshold

        s = rho * W.T @ (z_new - z)  # dual residual
        z = z_new
        r = Wx - z  # primal residual
        u = u + r

        s_norm_sq = np.sum(s**2)
        r_norm_sq = np.sum(r**2)

        if r_norm_sq > mu_sq * s_norm_sq:
            rho = t_incr * rho
            u = u / t_incr
        elif s_norm_sq > mu_sq * r_norm_sq:
            rho = rho / t_decr
            u = u * t_decr
        else:
            continue

        # this happens whenever we updated rho:
        Q = (A.T @ A + rho * W.T @ W)
        Q_LU = scipy.linalg.lu_factor(Q)  # for useful for solving Qx = b, later
        threshold = beta/rho

    return x


def solve_lasso_cvxpy(A, y, beta, W):
    """
    argmin_x 1/2 || Ax - y ||_2^2 + beta ||W x||_1

    using cvxpy
    """
    y = y.reshape(y.shape[0], -1)  # add trailing dim if needed

    k = W.shape[0]
    num = y.shape[1]
    problem, params = setup_cvxpy_problem(*A.shape, k, num)
    params['A'].value = np.array(A)
    params['y'].value = np.array(y)
    params['beta'].value = np.array(beta)
    params['W'].value = W

    problem.solve()

    return params['x'].value


#@functools.lru_cache()  # caching because making the problem object is slow
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
