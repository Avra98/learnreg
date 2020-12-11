"""
compare the solution to a l1 problem returned by cvxpy
to the one returned by our closed-form expression.
"""

import torch

import learnreg

# parameters
n = 5 # signal length, W is n x n
sigma = 0.25
sign_thresh = 1e-18
beta = 2.0
SEED = 0

# init
torch.manual_seed(SEED)  # make repeatable

# make TV
tv=torch.zeros(n)
tv[0]=1.0
tv[1]=-1.0
TV=learnreg.create_circulant(tv)

# make dataset
A = torch.eye(n,n)
x, y = learnreg.make_set(A, num_signals=1, sigma=sigma)

# solve with cvxpy
J = lambda x : learnreg.compute_loss(x, y, beta, TV)
x_cvxpy = learnreg.optimize(TV, y, beta)

# solve in closed form
z = TV @ x_cvxpy
S0,Sp,s = learnreg.find_sign_pattern(z, threshold=sign_thresh)
x_closed=learnreg.closed_form(S0, Sp, s, TV, beta, y)

# compared
max_diff = (x_closed - x_cvxpy).abs().max()
print(f'max abs difference: {max_diff}')
print(f'J(x_cvxpy) = {J(x_cvxpy)}')
print(f'J(x_closed) = {J(x_closed)}')
