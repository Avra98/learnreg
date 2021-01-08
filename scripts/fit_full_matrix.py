"""
learn a W on random batches of piecewise constant signals
"""

import matplotlib.pyplot as plt
import torch
import collections
import numpy as np

import time
import learnreg as lr

# todo: better signal generation (random values rather than random jumps,
# so the regions are not correlated)
# todo: get serious: set this up so you can systematically compare

# TV is roughly 1.5e-1
# best learned I've seen is roughly 1.6e-1 (avg filter init)
# 1.7e-1 from random init
# 1.595e-01 from eye init, lr=1e-2 (@10000, 2.53e-01)
# 1.484 from eye init lr=1e-2
# 1.365 (longer run) ->
# with 1000 signals, there is evidence of overfitting
# with 10,000 signals, got to .152 (us) vs .138 (TV) with lr 1e-1,
# ... continuing more training at lr 1e-2
# with 10,000 signals, there is overfitting
# ... (.188 vs .200 on training, .168 vs .161 on testing)
learn_opts = dict(
    learning_rate=1e-1, # 1e0 for random init
    num_steps=int(1e5), #1e5 takes a few minutes, gives good results
    print_interval=100,)

n = 64
noise_sigma = 1e0

train, A, W0, W, beta = lr.learn_for_denoising(
    n=n,
    num_signals=10000,
    noise_sigma=noise_sigma,
    SEED=0,
    learn_opts=learn_opts)

# to learn more with the same W
learn_opts['learning_rate'] = 1e-2
W = lr.main(A, beta, W, train, **learn_opts)

# make a small test set to evaluate
test = lr.make_dataset(A, num_signals=10, sigma=noise_sigma)
#test = lr.Dataset(train.x[:, :10], train.y[:, :10])
beta_W = lr.find_optimal_beta(A, test.x, test.y, W, 1e1)

MSE_learned = lr.eval_upper(A, test.x, test.y, beta_W, W).item()

# show results
plt.close('all')

fig, ax = lr.plot_denoising(test, beta_W, W)
fig.suptitle(f'learnred reconstruction results, MSE:{MSE_learned:.3e}')
fig.tight_layout()
fig.show()

# compare to TV
TV = lr.make_TV(n)
A = torch.eye(n)

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
fig.suptitle('W_learned @ x_star(W_learned)')
fig.show()
fig, ax = lr.plot_recon_sparsity(A, test, beta_TV, TV)
fig.suptitle('W_TV @ x_star(W_TV)')
fig.show()

print(MSE_learned, MSE_TV)
