"""
learn a W on random batches of piecewise constant signals
"""

import matplotlib.pyplot as plt
import torch
import collections

import learnreg
sigma = 5e-1

train, val, A = learnreg.make_data(n=100, m=100, sigma=sigma, train_size=20, val_size=2)
train = None  # for autogen

# x_tv = learnreg.TV_recon(val, A, beta='auto')  # todo: write this function

W, W0 = learnreg.main(A, k=100, train=train,  beta0=1e-1, val=val,
         batch_size=1, opti_opts=('SGD', dict(lr=1e-1)),
         num_steps=5000, val_interval=10, sigma=sigma, print_interval=10,
         history_length=10)

# show results
fig_name = 'reconstruction results'
# try this W on another problem
fig, ax = learnreg.solve_and_plot(A, val, W)
ax.set_title('validation signal')


fig.tight_layout()
fig.show()

# show W0 and W
learnreg.show_W(W0, W)
