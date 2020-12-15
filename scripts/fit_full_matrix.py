"""
learn a W on random batches of piecewise constant signals
"""

import matplotlib.pyplot as plt
import torch
import collections

import time

import learnreg as lr

n = 100  # signal length
sigma = 1e-1
learning_rate = 1e-6
# todo: longer validation interval, smaller max batch (67 gives memory error)

A = torch.eye(n,n)

train = lr.make_dataset(A, num_signals=1, sigma=sigma)
val   = lr.make_dataset(A, num_signals=10,    sigma=sigma)

W0 = lr.make_TV(n)
W0 = W0 + 0.001 * torch.randn_like(W0)

beta = lr.find_optimal_beta(A, val.x, val.y, W0, upper=2)

W = lr.main(A, beta, W0, train,
            batch_size=1, opti_type='SGD', opti_opts=dict(lr=learning_rate),
            num_steps=100000, print_interval=100)

# show results
fig_name = 'reconstruction results'
# try this W on another problem
fig, ax = lr.solve_and_plot(A, val, beta, W)
ax.set_title('validation signal')


fig.tight_layout()
fig.show()

# show W0 and W
learnreg.show_W(W0, W)
