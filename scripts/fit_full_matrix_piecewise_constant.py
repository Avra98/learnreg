"""
learn a W on a training set of piecewise constant signals
"""

import matplotlib.pyplot as plt
import torch
import collections
import numpy as np

import time
import learnreg as lr

# todo: get serious: set this up so you can systematically compare

# TV is ~2.00e-03
#best so far: 6.234e-02, seems like a local min with constnt output
learn_opts = dict(
    learning_rate=1e0,
    num_steps=int(1e5), #1e5 takes a few minutes, gives good results
    print_interval=100,
    sign_threshold=1e-6)

signal_type = 'piecewise_constant'
n = 64
noise_sigma = 1e-1

train, A, W0, W, beta = lr.learn_for_denoising(
    signal_type=signal_type,
    n=n,
    num_signals=10000,
    noise_sigma=noise_sigma,
    SEED=0,
    learn_opts=learn_opts)

# to learn more with the same W
#learn_opts['learning_rate'] = 1e-1
#W = lr.main(A, beta, W, train, **learn_opts)

# testing -------------------------

# make a small test set to evaluate
test = lr.make_dataset(signal_type, A, num_signals=10, sigma=noise_sigma)

# refit W
#beta_W = lr.find_optimal_beta(A, test.x, test.y, W, 1e1)
beta_W = beta
MSE_learned = lr.eval_upper(A, test.x, test.y, beta_W, W).item()

# show results
plt.close('all')

fig, ax = lr.plot_denoising(test, beta_W, W)
fig.suptitle(f'learnred reconstruction results, MSE:{MSE_learned:.3e}')
fig.tight_layout()
fig.show()

# compare to TV
TV = lr.make_TV(n)

beta_TV = lr.find_optimal_beta(A, test.x, test.y, TV, 1e1)

MSE_TV = lr.eval_upper(A, test.x, test.y, beta_TV, TV).item()

fig, ax = lr.plot_denoising(test, beta_TV, TV)
fig.suptitle(f'TV reconstruction results, MSE:{MSE_TV:.3e}')
fig.tight_layout()
fig.show()

# show W0 and W
lr.show_W(W0, W)

# todo: we are getting a lot of wrong x_closed,
# can we fix this via something clever with the dual problem?

# are the reconstructions actually sparse?
fig, ax = lr.plot_recon_sparsity(A, test, beta_W, W)
fig.suptitle('W_learned @ x*(W_learned)')
fig.show()
fig, ax = lr.plot_recon_sparsity(A, test, beta_TV, TV)
fig.suptitle('W_TV @ x*(W_TV)')
fig.show()

print(MSE_learned, MSE_TV)
