import numpy as np
import itertools

import learnreg as lr

n = 64
k = 64
beta = 2.0
num_signals = 3
np.random.seed(1)
A = lr.make_foward_model('identity', n)
x, y = lr.make_dataset('piecewise_constant', A, 0.1, num_signals)
W = lr.make_transform('random', n, k, scale=1.0e-2)

opts_dict = {
    'ADMM': {
        'num_steps': 1000, 'rho': 1e0},
    'cvxpy': {}
}

x_star = {}
for method, opts in opts_dict.items():
    x_star[method] = lr.opt.solve_lasso(A, y, beta, W, method=method, **opts)

for k1, k2 in itertools.combinations(opts_dict.keys(), 2):
    print(k1, k2)
    np.testing.assert_allclose(x_star[k1], x_star[k2])
