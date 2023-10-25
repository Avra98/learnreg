"""
compare the solution to a l1 problem returned by cvxpy
to the one returned by our closed-form expression.
"""
import torch

import learnreg as lr

# parameters
n = 5 # signal length, W is n x n
sigma = 1e-1
zero_thresh = 1e-6
SEED = 0

# init
torch.manual_seed(SEED)  # make repeatable


# make TV
#W = lr.make_TV(n)
W = lr.make_transform("TV", n, scale=1.0)
print(W)
#W = W + 0.001 * torch.randn_like(W)

# make dataset
A = torch.eye(n,n)
x, y = lr.make_dataset(A, num_signals=1, noise_sigma=sigma)

beta = lr.find_optimal_beta(A, x, y, W, upper=2)

J = lambda x : lr.compute_loss(x, y, beta, W)

# solve with cvxpy
x_cvxpy = lr.optimize(W, y, beta)

# solve in closed form
z = W @ x_cvxpy
S0,Sp,s = lr.find_sign_pattern(z, threshold=zero_thresh)
x_closed=lr.closed_form(S0, Sp, s, W, beta, y)

# solve with closed form alt
W0, Wpm, s = lr.find_signs_alt(x_cvxpy, W, threshold=zero_thresh)
x_closed_alt = lr.closed_form_alt(W0, Wpm, s, y, beta)

# compared
max_diff = (x_closed - x_cvxpy).abs().max()
print(f'max abs difference: {max_diff}')
print(f'J(x_cvxpy) = {J(x_cvxpy)}')
print(f'J(x_closed) = {J(x_closed)}')
print(f'J(x_closed_alt) = {J(x_closed_alt)}')
