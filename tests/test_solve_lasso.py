import numpy as np
import itertools

import learnreg as lr

n = 64
k = 64
beta = 2.0
num_signals = 10

for seed in range(10):
    np.random.seed(seed)
    A = lr.make_foward_model('identity', n)
    x, y = lr.make_dataset('piecewise_constant', A, 0.1, num_signals)
    W = lr.make_transform('random', n, k, scale=1.0e-2)

    opts_dict = {
        'ADMM': {
            'num_steps': 500, 'rho': 1},
        'cvxpy': {},
        'dual' : {},
    }

    x_star = {}
    for method, opts in opts_dict.items():
        x_star[method] = lr.opt.solve_lasso(A, y, beta, W, method=method, **opts)

    for k1, k2 in itertools.combinations(opts_dict.keys(), 2):
        val1 = lr.opt.eval_lasso(A, x_star[k1], y, beta, W)
        val2 = lr.opt.eval_lasso(A, x_star[k2], y, beta, W)

        if val2 < val1:  # swap
            val1, val2 = val2, val1
            k1, k2 = k2, k1
        print(f'{k1} (J(x)={val1:.3f}) beats {k2} (J(x)={val2:.3f})')
