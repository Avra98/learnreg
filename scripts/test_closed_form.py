"""
compare the solution to a l1 problem returned by cvxpy
to the one returned by our closed-form expression.
"""
import torch

import learnreg

# parameters
n = 50 # signal length, W is n x n
sigma = 0.25
sign_thresh = 1e-18
beta = 0.5  # fine when beta is 0.5, breaks when 1.0 or higher
SEED = 0

# init
torch.manual_seed(SEED)  # make repeatable

# make TV
tv=torch.zeros(n)
tv[0]=1.0
tv[1]=-1.0
W = learnreg.create_circulant(tv)
#W += torch.rand_like(W)


# make dataset
A = torch.eye(n,n)
x, y = learnreg.make_set(A, num_signals=1, sigma=sigma)

J = lambda x : learnreg.compute_loss(x, y, beta, W)

# solve with cvxpy
x_cvxpy = learnreg.optimize(W, y, beta)

# solve with pylayers
_, solve_lasso = learnreg.setup_cvxpy_problem(n, n, n)
x_pylayers = solve_lasso(A, y, beta * W)

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
print(f'J(x_pylayers) = {J(x_pylayers)}')
print(f'J(x_closed) = {J(x_closed)}')
print(f'J(x_closed_alt) = {J(x_closed_alt)}')
