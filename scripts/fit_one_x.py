import matplotlib.pyplot as plt
import torch
import collections

import learnreg
sigma = 5e-1

torch.manual_seed(0)  # make repeatable

train, val, A = learnreg.make_data(n=100, m=100, sigma=sigma, train_size=1, val_size=1)


W, W0 = learnreg.main(A, k=100, train=train,  beta0=1e-1, val=val,
         batch_size=1, opti_opts=('LBFGS', dict(lr=1e-1)),
         num_steps=20)

# show results
fig_name = 'reconstruction results'

plt.close(fig_name)
fig, axes = plt.subplots(2, 1, num=fig_name, sharex=True, sharey=True)

_, ax = learnreg.solve_and_plot(A, train, W, ax=axes[0])
ax.plot(W@train.x, '.')
ax.set_title('training signal')


# try this W on another problem
_, ax = learnreg.solve_and_plot(A, val, W, ax=axes[1])
ax.set_title('testing signal')

fig.tight_layout()
fig.show()

# show W0 and W
learnreg.show_W(W0, W)
