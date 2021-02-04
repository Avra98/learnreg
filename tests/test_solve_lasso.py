import numpy as np

import learnreg as lr

n = 64
k = 64
beta = 2.0
num_signals = 10


for seed in range(10):
    np.random.seed(seed)
    A = lr.make_foward_model('identity', n)
    x, y = lr.make_dataset('piecewise_constant', A, 0.1, num_signals)
    W = lr.make_transform('random', n, k, scale=1.0e-1)

    opts_dict = {
        'ADMM': {
            'num_steps': 1000, 'rho': 1},
        'cvxpy': {},
        'dual_PGD' : {'num_steps':10000, 'step_size':None},
    }

    x_star = {}
    J = {}
    signs = {}
    for method, opts in opts_dict.items():
        x_star[method] = lr.opt.solve_lasso(A, y, beta, W, method=method, **opts)
        J[method] = lr.opt.eval_lasso(A, x_star[method], y, beta, W)
        signs[method] = np.sign(x_star[method])
        signs[method][x_star[method] < 1e-7] = 0


    min_method = min(J, key=J.get)

    for m, val in J.items():
        J_diff = val-J[min_method]
        x_diff = np.max(np.array(np.abs(x_star[min_method]-x_star[m])))
        sign_diff = np.sum(signs[m] != signs[min_method])
        if sign_diff > 0:
            pass
            #1/0
        print(sign_diff)

        print(f'{m}: {J_diff:.3e}, {x_diff:.3e}')
    print('---')
