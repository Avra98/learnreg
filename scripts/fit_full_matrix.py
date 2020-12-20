"""
learn a W on random batches of piecewise constant signals
"""

import matplotlib.pyplot as plt
import torch
import collections
import numpy as np

import time
import learnreg as lr

#1.6e-02 is the best I have seen
learn_opts = dict(
    learning_rate=1e1,
    num_steps=int(1.5e4),  # 1.5e4 todo: try lr=1e-1 and int(1e6) (overnight)?
    print_interval=100,)

beta = 0.1
n = 50

train, W0, W = lr.learn_for_denoising(
    n=n,
    num_signals=1000,
    noise_sigma=1e-1,
    beta=beta,
    SEED=0,
    learn_opts=learn_opts)


# show results
fig_name = 'reconstruction results'
# try this W on another problem
train_small = lr.Dataset(train.x[:,:5], train.y[:, :5])
fig, ax = lr.plot_denoising(train_small, beta, W)
ax.set_title('training (first 5)')
fig.tight_layout()
fig.show()

TV = lr.make_TV(n)
A = torch.eye(n)
beta_TV = lr.find_optimal_beta(A, train_small.x, train_small.y, TV, 1.0)

# show W0 and W
lr.show_W(W0, W)
