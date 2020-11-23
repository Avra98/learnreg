"""
batch learnreg in 1D

# idea: D as explicit conv matrix
"""


import torch
import cvxpy as cp
import matplotlib.pyplot as plt

from  cvxpylayers.torch  import  CvxpyLayer


num_signals = 100
n = 100
m = 100
sigma = 5e-1

learning_rate = 1e0
num_steps = 30


# init
torch.manual_seed(0)

def make_signal(n, num_signals, jump_freq=0.1):
    jumps = torch.rand(n, num_signals) <= jump_freq
    jumps[0] = True  # so they don't all start at zero
    heights = torch.randn(n, num_signals)
    heights[~jumps] = 0
    return heights.cumsum(dim=0)


# setup problem
x = cp.Variable ((n, num_signals))
z = cp.Variable ((n, num_signals))
A = cp.Parameter ((m, n))
y = cp.Parameter ((m, num_signals))
W = cp.Parameter ((n, n))
#lambd = cp.Parameter ((1), nonneg=True)
objective_fn = cp.sum_squares(A @ x - y) + cp.sum(cp.abs(z))
constraints = [W @ x == z]
problem = cp.Problem(cp.Minimize(objective_fn), constraints)

x_gt = make_signal(n, num_signals)
y_t = x_gt + sigma * torch.randn_like(x_gt)

# setup operators
A_t = torch.eye(m, n)
W_0 = torch.randn(n, n)
W_t = W_0.clone()

# set warm start at A_t.T @ y_t
x0 = A_t.T @ y_t
x.value = x0.numpy()
z.value = (W_t @ x0).numpy()

x0_loss = ((x0 - x_gt)**2).mean().item()

layer = CvxpyLayer(
    problem , parameters =[A, y, W], variables =[x])

#opti = torch.optim.SGD((W_t,), lr=learning_rate)
opti = torch.optim.LBFGS((W_t,), lr=learning_rate)

W_t.requires_grad_(True)
last_loss = [None]
for step in range(num_steps):
    def closure():
        opti.zero_grad()
        x_star, = layer(A_t, y_t, W_t)
        loss = ((x_star - x_gt)**2).mean()
        loss.backward()
        last_loss[0] = loss.item()
        return loss

    opti.step(closure)
    print(step, x0_loss, last_loss[0])

W_t.requires_grad_(False)
x_star, = layer(A_t, y_t, W_t)

def plot_recons(x_gt, y_t, x_star):
    fig, axes = plt.subplots(3, 2)
    for idx, ax in enumerate(axes.flatten()):
        ax.plot(x_gt[:, idx], color='k')
        ax.plot(y_t[:, idx], color='tab:blue')
        ax.plot(x_star[:, idx], color='tab:orange', linestyle='dashed')
    fig.show()
plot_recons(x_gt, y_t, x_star)

def permute_for_display(W):
    W_out = W.clone()
    max_inds = (W.abs()).argmax(dim=1)

    max_sign = W[range(len(W)), max_inds].sign()
    W_out[max_sign==-1] *= -1  # flip sign on negative rows

    idx = max_inds.argsort()

    W_out = W_out[idx]

    return W_out

fig, ax = plt.subplots(1, 2)
#ax.plot(W_t[:, :3])
ax[0].imshow(permute_for_display(W_0))
ax[1].imshow(permute_for_display(W_t))
fig.show()

# test this W on a new problem, interesting result!
x_new = make_signal(n, num_signals)
y_new = x_new + sigma * torch.randn_like(x_new)
x_star_new, = layer(A_t, y_new, W_t)
plot_recons(x_new, y_new, x_star_new)


print(((y_new - x_new)**2).mean())
print(((layer(A_t, y_new, W_t)[0] - x_new)**2).mean())
