"""
compare the solution to a l1 problem returned by cvxpy
to the one returned by our closed-form expression.
"""
import torch

import learnreg

# parameters
n = 100 # signal length, W is n x n
sigma = 1e-1
sign_thresh = 1e-18
SEED = 0

# init
torch.manual_seed(SEED)  # make repeatable

# make TV
W = lr.make_TV(n)
W = W + 0.001 * torch.randn_like(W)

# make dataset
A = torch.eye(n,n)
x, y = learnreg.make_dataset(A, num_signals=1, sigma=sigma)

beta = lr.find_optimal_beta(A, x, y, W, upper=2)

J = lambda x : learnreg.compute_loss(x, y, beta, W)

# solve with cvxpy
x_cvxpy = learnreg.optimize(W, y, beta)

# solve in closed form
z = W @ x_cvxpy
S0,Sp,s = learnreg.find_sign_pattern(z, threshold=sign_thresh)
x_closed=learnreg.closed_form(S0, Sp, s, W, beta, y)

# solve with closed form alt
W0, Wpm, s = learnreg.find_signs_alt(x_cvxpy, W)
x_closed_alt = learnreg.closed_form_alt(W0, Wpm, s, y, beta)

# compared
max_diff = (x_closed - x_cvxpy).abs().max()
print(f'max abs difference: {max_diff}')
print(f'J(x_cvxpy) = {J(x_cvxpy)}')
print(f'J(x_closed) = {J(x_closed)}')
print(f'J(x_closed_alt) = {J(x_closed_alt)}')
