"""
learn a W on random batches of piecewise constant signals
"""

import matplotlib.pyplot as plt
import torch
import collections

import time

import learnreg as lr

n = 100  # signal length
sigma = 5e-1
learning_rate = 1e0

# todo: longer validation interval, smaller max batch (67 gives memory error)

A = torch.eye(n,n)

train = lr.make_dataset(A, num_signals=2000, sigma=sigma)
val   = lr.make_dataset(A, num_signals=1,    sigma=sigma)

W0 = lr.make_TV(n)

beta = lr.find_optimal_beta(A, val.x, val.y, beta, W, upper=2.0)

W = lr.main(A, beta, W0, train, val=val,
    batch_size=1, opti_type='SGD', opti_opts=dict(lr=learning_rate),
    num_steps=1000, val_interval=10, print_interval=10,
    history_length=20, max_batch_size=64)

# show results
fig_name = 'reconstruction results'
# try this W on another problem
fig, ax = learnreg.solve_and_plot(A, val, W)
ax.set_title('validation signal')


fig.tight_layout()
fig.show()

# show W0 and W
learnreg.show_W(W0, W)
