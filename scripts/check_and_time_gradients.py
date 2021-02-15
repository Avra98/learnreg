"""
compare cvxpy  and hand gradient to numerical gradient

michael.thompson.mccann@gmail.com
2020-02-09

"""

import numpy as np
import cvxpy as cp
from timeit import repeat
import pprint

import learnreg as lr

n = 32
signal_type = 'piecewise_constant'
num_signals = 1

k = n - 1  # just to keep track of shapes
forward_model_type = 'identity'
noise_sigma = 0.1

transform_type = 'random'
transform_scale = 1e-1

threshold = 1e-6

random_seed = 1

np.random.seed(random_seed)
A = lr.make_foward_model(forward_model_type, n)
W = lr.make_transform(transform_type, n, k, transform_scale)
x, y = lr.make_dataset(signal_type, A, noise_sigma, num_signals)



def solve(prob):
    prob.solve()
    return prob.variables()[0].value

def compute_Q(prob, x):
    x_hat = solve(prob)
    return MSE(x_hat, x)

def MSE(x, y):
    return 0.5 * np.sum((x - y)**2)

def numerical_grad(prob, W, x, dx=1e-8):
    grad = np.zeros_like(W)
    x_hat = solve(prob)
    is_zero_0 = np.abs(W@x_hat)[:, 0] < threshold
    for i, j in np.ndindex(W.shape):
        val = W[i, j]
        W[i, j] = val + dx
        x_hat = solve(prob)
        is_zero = np.abs(W@x_hat)[:, 0] < threshold
        if not np.all(is_zero == is_zero_0):
            print('sign patterned changed')
        Q_foward = MSE(x_hat, x)

        W[i, j] = val - dx
        x_hat = solve(prob)
        is_zero = np.abs(W@x_hat)[:, 0] < threshold
        if not np.all(is_zero == is_zero_0):
            print('sign patterned changed')

        Q_backward = MSE(x_hat, x)
        grad[i, j] = (Q_foward - Q_backward)/(2*dx)
        W[i, j] = val
    return grad


# hand gradient funcs

def null_proj(X):
    """
    N = I - X^+ X
      = I - V E+ U* U E V*
      = I - V E+E V*

    where E+
    """
    rcond = 1e-15

    if X.shape[0] == 0:
        return np.eye(X.shape[1])

    U, S, Vh = np.linalg.svd(X)
    S = np.where(S >= rcond, 1.0, 0.0)

    Vh = Vh[:len(S), :]

    XpX = Vh.T @ np.diag(S) @ Vh  #

    return np.eye(X.shape[1]) - XpX

def hand_grad(prob, y, W, x):
    x_hat = solve(prob)

    is_zero = np.abs(W@x_hat)[:, 0] < threshold

    W0 = W[is_zero, :]
    s = np.sign(W@x_hat)[~is_zero]
    Wpm = W[~is_zero, :]

    gradJ = x_hat - x
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

def cvxpy_grad(prob, x):
    prob.solve(requires_grad=True)
    x_hat = prob.variables()[0].value

    prob.variables()[0].gradient = x_hat - x
    prob.backward()

    return prob.parameters()[1].gradient

# gradients

prob = make_problem(A, y, W, k)
grad_num = numerical_grad(prob, W, x)
grad_hand = hand_grad(prob, y, W, x)
grad_cvxpy = cvxpy_grad(prob, x)

# other useful calcs

x_hat = solve(prob)
is_zero = np.abs(W@x_hat)[:, 0] < threshold

# plots

fig1, fig2 = lr.reports.compare_matrices(
    (grad_num, grad_hand), ('numerical', 'hand'))
fig2.show()
fig1.show()

fig1, fig2 = lr.reports.compare_matrices(
    (grad_num, grad_cvxpy), ('numerical', 'cvxpy'))
fig2.show()
fig1.show()

fig1, fig2 = lr.reports.compare_matrices(
    (grad_hand, grad_cvxpy), ('hand', 'cvxpy'))
fig2.show()
fig1.show()


# timing
time_hand = repeat(lambda: hand_grad(prob, y, W, x), number=100)
time_cvxpy = repeat(lambda: cvxpy_grad(prob, x), number=100)

print('time by hand:')
pprint.pprint(time_hand)

print('time by cvxpy:')
pprint.pprint(time_cvxpy)
