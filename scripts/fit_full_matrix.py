"""
learn a W on random batches of piecewise constant signals
"""

import matplotlib.pyplot as plt
import torch
import collections

import learnreg

sigma = 5e-1
learning_rate = 1e0

# todo: longer validation interval, smaller max batch (67 gives memory error)

train, val, A = learnreg.make_data(
    n=100, m=100, sigma=sigma, train_size=2000, val_size=2)

# x_tv = learnreg.TV_recon(val, A, beta='auto')  # todo: write this function

W, W0 = learnreg.main(
    A, k=100, train=train,  beta0=1e-1, val=val,
    batch_size=1, opti_opts=('SGD', dict(lr=learning_rate)),
    num_steps=10000, val_interval=10, print_interval=10,
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
