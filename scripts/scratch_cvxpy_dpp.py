"""

There is nothing useful here, except maybe plotting code


exploring Differentiable Convex Optimization Layers
https://web.stanford.edu/~boyd/papers/pdf/diff_cvxpy.pdf

especially the question of whether our L1 problem fits in their framework
"""


import torch
import numpy as np

import matplotlib.pyplot as plt
import collections
import scipy.spatial
import scipy.cluster

"""

n = 100
m = 100

sigma = 5e-1

beta_0 = 1e-1
#filter_width = 100

learning_rate = 1e-1
num_steps = 2000
batch_size = 1
max_batch_size = 128
momentum = 0.9
validation_step = 10  # compute validation this often
history_length = 10  # needs to be a min in this many validations or batch_size doubles

# init
torch.manual_seed(0)




# setup
A = torch.eye(m, n)

x_val = make_signal(n)
y_val = make_measurement(x_val, A, sigma)
loss_0 = MSE(y_val, x_val).item()

problem, layer = setup_problem(m, n)

W_0 = beta_0 * torch.randn(n, n)
#W_0 = torch.triu(W_0, -(filter_width//2) )
#W_0 = torch.tril(W_0, (filter_width-1)//2)
W = W_0.clone()

opti = torch.optim.SGD((W,), lr=learning_rate, momentum=momentum)

# main loop
W.requires_grad_(True)
loss_history = collections.deque(history_length*[np.inf], history_length)
best_loss = np.inf

for step in range(num_steps):
    opti.zero_grad()
    for batch_ind in range(batch_size):
        x_gt = make_signal(n)
        y = make_measurement(x_gt, A, sigma)
        x_star, = layer(A, y, W)  # solves the l1 problem
        loss = MSE(x_star, x_gt) / batch_size
        loss.backward()
    opti.step()

    if step % validation_step == 0 or step == num_steps - 1:
        with torch.no_grad():
            x_val_star, = layer(A, y_val, W)
            val_loss = MSE(x_val_star, x_val)
            print(step, loss_0, val_loss.item())
            loss_history.appendleft(val_loss)
            if val_loss < best_loss:
                best_loss = val_loss

            if best_loss not in loss_history:
                if batch_size < max_batch_size:
                    batch_size *= 2
                print(f'validation fail, batch_size={batch_size}')
                best_loss = np.inf

W.requires_grad_(False)
"""

# plot results
def plot_recon(x_gt, y, A, W):
    x_star, = layer(A, y, W)
    fig, ax = plt.subplots()
    ax.plot(x_gt, color='k')
    ax.plot(y, color='tab:blue')
    ax.plot(x_star, color='tab:orange', linestyle='dashed')
    fig.show()

plot_recon(x_val, y_val, A, W)
#plot_recon(x_gt, y, A, W)

def permute_for_display(W):
    W_out = W.clone()
    max_inds = (W.abs()).argmax(dim=1)

    max_sign = W[range(len(W)), max_inds].sign()
    W_out[max_sign==-1] *= -1  # flip sign on negative rows

    idx = max_inds.argsort()

    W_out = W_out[idx]

    return W_out

def sort_for_display(W):
    # todo: what if distance ignored shifts?
    W_out = W.clone()
    dists = scipy.spatial.distance.pdist(W)
    linkage = scipy.cluster.hierarchy.linkage(dists)
    inds = scipy.cluster.hierarchy.leaves_list(linkage)
    return W_out[np.int64(inds)]

def plot_sparsity(W):
    fig, axes = plt.subplots(2,1)
    for ax in axes.flatten():
        x_gt = make_signal(n)
        y = make_measurement(x_gt, A, sigma)
        x_star, = layer(A, y, W)
        ax.plot(x_star)
        ax2 = ax.twinx()
        ax2.plot(W @ x_star, 'o', color='tab:orange')
        ax2.plot(W @ y, 'o', color='tab:green')
    fig.tight_layout()
    fig.show()


show_W(W, W_0)
plot_sparsity(W)
