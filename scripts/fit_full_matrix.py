"""
learn a W on random batches of piecewise constant signals
"""

import matplotlib.pyplot as plt
import torch
import collections
import numpy as np

import time
nn
import learnreg as lr

n = 5  # signal length
sigma = 1e-1
#opti_type = 'LBFGS'
#learning_rate = 1e-1
#num_steps = 20
opti_type = 'SGD'
learning_rate = 1e-1
num_steps = 1000
SEED = 0


# init
torch.manual_seed(SEED)  # make repeatable

A = torch.eye(n,n)

train = lr.make_dataset(A, num_signals=25, sigma=sigma)
val   = lr.make_dataset(A, num_signals=1,    sigma=sigma)

#W0 = lr.make_TV(n)
#W0 = W0 + 0.001 * torch.randn_like(W0)
W0 = torch.randn(n-1, n)


beta = 0.1

W = W0.clone()
#beta = lr.find_optimal_beta(A, train.x[:,:5], train.y[:,:5], W0, upper=2)

W = lr.main(A, beta, W, train, val=val,
            batch_size=1, opti_type=opti_type, opti_opts=dict(lr=learning_rate),
            num_steps=25*num_steps, val_interval=100, print_interval=np.ceil(num_steps/20))

# show results
fig_name = 'reconstruction results'
# try this W on another problem
t_small = lr.Dataset(train.x[:,:5], train.y[:, :5])
fig, ax = lr.solve_and_plot(A, val, beta, W)
ax.set_title('validation signal')

fig.tight_layout()
fig.show()

# show W0 and W
lr.show_W(W0, W)
