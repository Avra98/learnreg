import matplotlib.pyplot as plt
import torch
import collections

import learnreg
sigma = 5e-1

torch.manual_seed(0)  # make repeatable

n = 100

train, val, A = learnreg.make_data(n=n, m=100, sigma=sigma, train_size=3, val_size=1)

k = 2*n-1

W, W0 = learnreg.main(A, k=k, train=train,  beta0=1e-1, W_type='conv', val=val,
                      batch_size=3, opti_opts=('LBFGS', dict(lr=1e-2)),
                      num_steps=400)

# show results
fig_name = 'reconstruction results'

plt.close(fig_name)
fig, axes = plt.subplots(2, 1, num=fig_name, sharex=True, sharey=True)

_, ax = learnreg.solve_and_plot(A, train, W, ax=axes[0])
ax.set_title('training signal')


# try this W on another problem
_, ax = learnreg.solve_and_plot(A, val, W, ax=axes[1])
ax.set_title('testing signal')

fig.tight_layout()
fig.show()

fig, ax = plt.subplots()
x_star = learnreg.solve_lasso(A, train.y, W)
ax.plot(W@x_star, '.')
ax.set_title('W @ x_star (training)')
fig.show()

# show W0 and W
learnreg.show_W(W0, W)
