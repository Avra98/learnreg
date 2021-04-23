"""
explore using the dual problem to pick a good threshold or
otherwise find W_pm and W_0

Conclusion: no obvious answer here, thresholding is not too bad.
Maybe if we solved using duality, something could happen
"""

import torch

import learnreg as lr

torch.manual_seed(0)  # make repeatable

n = 64
noise_sigma = 1e-1
beta = 1.0e-2
A = np.eye(n, n)
x, y = lr.make_dataset('piecewise_constant', A, num_signals=1, sigma=noise_sigma)

W = lr.make_conv(np.ones(1), n)
W = W - W.mean(axis=1, keepdims=True)
W = W[1:, :]  # to let W be full row rank

x_star = solve_lasso(A, y, beta, W)

# note how some values definitely are 0, others definitely are not,
# and some are in between
print('(Wx*)[i]\t\t ==0 ?')
print(np.concatenate((W@x_star, W@x_star == 0), axis=1))

W0, Wpm, s = lr.find_signs(torch.tensor(x_star), torch.tensor(W))
x_closed = lr.closed_form(W0, Wpm, s, torch.tensor(y), beta)

print(np.abs(x_closed-x_star).max().item())
