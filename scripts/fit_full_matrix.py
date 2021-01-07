"""
learn a W on random batches of piecewise constant signals
"""

import matplotlib.pyplot as plt
import torch
import collections
import numpy as np

import time
import learnreg as lr


learn_opts = dict(
    learning_rate=1e0,
    num_steps=int(1e5), #1e5 takes a few minutes, gives good results
    print_interval=100,)

n = 64
noise_sigma = 1e0

train, A, W0, W, beta = lr.learn_for_denoising(
    n=n,
    num_signals=1000,
    noise_sigma=noise_sigma,
    SEED=0,
    learn_opts=learn_opts)

# make a small test set to evaluate
test = lr.make_dataset(A, num_signals=10, sigma=noise_sigma)
beta_W = lr.find_optimal_beta(A, train_small.x, train_small.y, W, 1.0)

MSE_learned = lr.eval_upper(A, test.x, test.y, beta_W, W).item()

# show results
fig, ax = lr.plot_denoising(test, beta_W, W)
fig.suptitle(f'learnred reconstruction results, MSE:{MSE_learned:.3e}')
fig.tight_layout()
fig.show()

# compare to TV
TV = lr.make_TV(n)
A = torch.eye(n)

beta_TV = lr.find_optimal_beta(A, test.x, test.y, TV, 1.0)

MSE_TV = lr.eval_upper(A, test.x, test.y, beta_TV, TV).item()

fig, ax = lr.plot_denoising(test, beta_TV, TV)
fig.suptitle(f'TV reconstruction results, MSE:{MSE_TV:.3e}')
fig.tight_layout()
fig.show()

# show W0 and W
lr.show_W(W0, W)

# todo: do this on a validation set instead, *shouldn't* change the conclusion

# todo: we are getting a lot of wrong x_closed,
# can we fix this via something clever with the dual problem?

# are the reconstructions actually sparse?
fig, ax = lr.plot_recon_sparsity(A, test, beta_W, W)
fig.suptitle('W_learned @ x_star(W_learned)')
fig.show()
fig, ax = lr.plot_recon_sparsity(A, test, beta_TV, TV)
fig.suptitle('W_TV @ x_star(W_TV)')
fig.show()
