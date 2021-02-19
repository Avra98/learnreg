"""
compare cvxpy  and hand gradient to numerical gradient

michael.thompson.mccann@gmail.com
2020-02-09

"""

import numpy as np
from timeit import repeat
import matplotlib.pyplot as plt
import torch

import learnreg as lr
import learnreg.opt as opt

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

plt.close('all')

np.random.seed(random_seed)
A = lr.make_foward_model(forward_model_type, n)
W = lr.make_transform(transform_type, n, k, transform_scale)
x, y = lr.make_dataset(signal_type, A, noise_sigma, num_signals)


def compute_Q(prob, y, beta, W, x):
    x_hat = opt.solve_cvxpy_problem(prob, y, beta, W)
    return opt.MSE(x_hat, x)


def numerical_grad(prob, W, x, dx=1e-8):
    grad = np.zeros_like(W)

    x_hat = opt.solve_cvxpy_problem(prob, y, 1.0, W)
    is_zero_0 = np.abs(W@x_hat)[:, 0] < threshold

    for i, j in np.ndindex(W.shape):
        val = W[i, j]
        prob.W.value[i, j] = val + dx
        prob.solve()
        x_hat = prob.x.value
        is_zero = np.abs(W@x_hat)[:, 0] < threshold
        if not np.all(is_zero == is_zero_0):
            print('sign patterned changed')
        Q_foward = opt.MSE(x_hat, x)

        prob.W.value[i, j] = val - dx
        prob.solve()
        x_hat = prob.x.value
        is_zero = np.abs(W@x_hat)[:, 0] < threshold
        if not np.all(is_zero == is_zero_0):
            print('sign patterned changed')

        Q_backward = opt.MSE(x_hat, x)
        grad[i, j] = (Q_foward - Q_backward)/(2*dx)
        prob.W.value[i, j] = val
    return grad


def cvxpy_grad(prob, x):
    prob.solve(requires_grad=True)
    x_hat = prob.x.value

    prob.x.gradient = 2 * (x_hat - x) / x.size
    prob.backward()

    return prob.parameters()[1].gradient


def hand_grad(prob, y, W, x):
    prob.solve()
    x_hat = prob.x.value

    return lr.opt.compute_grad(x_hat, y, W, x, threshold)


def pytorch_grad(prob, y, W, x):
    x = torch.as_tensor(x)
    prob.solve()
    x_hat = prob.x.value
    W = torch.tensor(W, requires_grad=True)
    W0, Wpm, s = find_signs(torch.as_tensor(x_hat), W)
    x_closed = closed_form(W0, Wpm, s, y, beta=1.0)
    loss = opt.MSE(x_closed, x)
    loss.backward()
    return np.array(W.grad)


def find_signs(x, W, threshold=1e-6):
    """
    given x* and W, find the necessary sign matrices:
    W_0, W_pm, and s
    """
    W = torch.as_tensor(W)

    Wx = W @ x
    is_zero = (Wx.abs() < threshold).squeeze()
    W0 = W[is_zero, :]
    Wpm = W[~is_zero, :]
    s = (Wpm @ x).sign()

    return W0, Wpm, s


def closed_form(W0, Wpm, s, y, beta):
    """
    implemention of (XXX) from "XXXXX" Tibshi...
    https://arxiv.org/pdf/1805.07682.pdf

    """
    rcond = 1e-15  # cutoff for small singular values

    W0 = torch.as_tensor(W0)
    y = torch.as_tensor(y)

    y_term = y - beta * Wpm.T @ s

    """
    if W0.shape[0] == 0:
        return y_term

    U, S, V = torch.svd(W0)
    S = torch.where(S >= rcond, torch.ones_like(S), torch.zeros_like(S))
    proj = V @ torch.diag(S) @ V.T
    """

    proj = W0.T @ (W0 @ W0.T).inverse() @ W0
    return y_term - proj @ y_term


# gradients
prob = lr.opt.make_cvxpy_problem(A, k)
prob.y.value = y
prob.W.value = W
grad_num = numerical_grad(prob, W, x)
grad_hand = hand_grad(prob, y, W, x)
grad_cvxpy = cvxpy_grad(prob, x)
grad_pytorch = pytorch_grad(prob, y, W, x)

# other useful calcs
x_hat = prob.x.value
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
    (grad_num, grad_pytorch), ('numerical', 'pytorch'))
fig2.show()
fig1.show()


# timing
time_hand = repeat(lambda: hand_grad(prob, y, W, x), number=100)
time_cvxpy = repeat(lambda: cvxpy_grad(prob, x), number=100)
time_pytorch = repeat(lambda: pytorch_grad(prob, y, W, x), number=100)

print('time by hand:')
print(min(time_hand))

print('time by cvxpy:')
print(min(time_cvxpy))

print('time by pytorch')
print(min(time_pytorch))
