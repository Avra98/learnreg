"""
functions for optimization and cost functions
"""

import cvxpy as cp
import functools
import torch
import numpy as np
import scipy.optimize
import scipy.linalg


def MSE(x, x_gt):
    return ((x - x_gt)**2).mean()


def eval_lasso(A, x, y, beta, W):
    """
    compute J(x) = 1/2 || Ax - y ||_2^2 + beta ||W x||_1
    """

    return 0.5 * np.sum((A @ x - y)**2) + beta * np.sum(np.abs(W @ x))


def solve_lasso(A, y, beta, W, method='cvxpy', **opts):
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


class CvxpySolver():
    """
    use this for time-critical repeated solves
    assumes A is not changing
    """

    def __init__(self, A, k, threshold):
        self.threshold = threshold
        self.prob = make_cvxpy_problem(A, k)


    def eval_upper(self, A, y, W, x_GT, requires_grad=False):
        x_hat = solve_cvxpy_problem(self.prob, y, 1.0, W)
        Q = MSE(x_hat, x_GT)

        if not requires_grad:
            return Q

        grad = compute_grad(x_hat, y, W, x_GT, self.threshold)

        return Q, grad

def solve_cvxpy_problem(prob, y, beta, W):
    prob.y.value = y
    prob.W.value = beta * W

    prob.solve(eps_abs=1e-4)
    return prob.x.value


def null_proj(X):
    """
    N = I - X^+ X
      = I - V E+ U* U E V*
      = I - V E+E V*

    where E+...
    """
    rcond = 1e-15

    if X.shape[0] == 0:
        return np.eye(X.shape[1])

    U, S, Vh = np.linalg.svd(X)
    S = np.where(S >= rcond, 1.0, 0.0)

    Vh = Vh[:len(S), :]

    XpX = Vh.T @ np.diag(S) @ Vh  #

    return np.eye(X.shape[1]) - XpX


def compute_grad(x_hat, y, W, x, threshold):
    is_zero = np.abs(W@x_hat)[:, 0] < threshold

    W0 = W[is_zero, :]
    s = np.sign(W@x_hat)[~is_zero]
    Wpm = W[~is_zero, :]

    gradJ = 2 * (x_hat - x) / x.size  # careful here! this is correct for MSE
    grad = np.zeros_like(W)

    # grad for the Wpm part
    N = null_proj(W0)
    grad[~is_zero, :] = - s @ gradJ.T @ N

    # grad for the W0 part
    W0p = np.linalg.pinv(W0)
    q = y - Wpm.T @ s
    grad_W0 = N @ q @ gradJ.T @ W0p
    grad_W0 += N @ gradJ @ q.T @ W0p
    grad_W0 = -grad_W0.T
    grad[is_zero, :] = grad_W0

    return grad


def make_cvxpy_problem(A, k):
    """
    a cvxpy problem representing

    argmin_x 1/2||Ax - y||_2^2 + ||Wx||_1

    where A is a constant and y and W are parameters

    can set this with, e.g.,  prob.y.value

    use this when you plan to solve the same problem
    repeatedly and speed is important, see https://www.cvxpy.org/tutorial/advanced/index.html#disciplined-parametrized-programming

    putting beta loses the DPP structure,
    so handle it by scaling W
    """

    m, n = A.shape
    A = cp.Constant(A)
    x = cp.Variable((n, 1), name='x')
    y = cp.Parameter((m, 1), name='y')
    W = cp.Parameter((k, n), name='W')

    obj = 0.5 * cp.sum_squares(A @ x - y) + cp.norm1(W @ x)

    prob = cp.Problem(cp.Minimize(obj))

    prob.x = prob.variables()[0]
    prob.y = prob.parameters()[0]
    prob.W = prob.parameters()[1]

    return prob


#  solvers ---------------------------------------
def solve_lasso_cvxpy(A, y, beta, W, prob=None):
    """
    argmin_x 1/2 || Ax - y ||_2^2 + beta ||W x||_1

    using cvxpy
    """
    y = y.reshape(y.shape[0], -1)  # add trailing dim if needed

    x_hat = np.zeros_like(y)

    if prob is None:
        prob = make_cvxpy_problem(A, W.shape[0])

    for i in range(y.shape[1]):
        x_hat[:, i] = solve_cvxpy_problem(prob, y[:, [i]], beta, W)[:, 0]

    return x_hat


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

    argmin_u || y - WT u || st -beta <= ui <= beta
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

    return x


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
