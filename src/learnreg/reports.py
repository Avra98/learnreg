"""
code for examining the results of experiments,
including making plots and text reports
"""
import matplotlib.pyplot as plt


"""

# refit W
#beta_W = lr.find_optimal_beta(A, test.x, test.y, W, 1e1)
beta_W = beta
MSE_learned = lr.eval_upper(A, test.x, test.y, beta_W, W).item()

# show results
plt.close('all')

fig, ax = lr.plot_denoising(test, beta_W, W)
fig.suptitle(f'learnred reconstruction results, MSE:{MSE_learned:.3e}')
fig.tight_layout()
fig.show()

# compare to TV
TV = lr.make_TV(n)

beta_TV = lr.find_optimal_beta(A, test.x, test.y, TV, 1e1)

MSE_TV = lr.eval_upper(A, test.x, test.y, beta_TV, TV).item()

fig, ax = lr.plot_denoising(test, beta_TV, TV)
fig.suptitle(f'TV reconstruction results, MSE:{MSE_TV:.3e}')
fig.tight_layout()
fig.show()

# show W0 and W
lr.show_W(W0, W)

# todo: we are getting a lot of wrong x_closed,
# can we fix this via something clever with the dual problem?

# are the reconstructions actually sparse?
fig, ax = lr.plot_recon_sparsity(A, test, beta_W, W)
fig.suptitle('W_learned @ x*(W_learned)')
fig.show()
fig, ax = lr.plot_recon_sparsity(A, test, beta_TV, TV)
fig.suptitle('W_TV @ x*(W_TV)')
fig.show()

print(MSE_learned, MSE_TV)

"""


# plotting ------------------------
def plot_denoising(data, beta, W, max_signals=3, **kwargs):
    """
    plot examples of the denoising results obtained with W
    """
    A = np.eye(data.x.shape[0])
    x_star = solve_lasso(A, data.y[:, :max_signals], beta, W)
    x_gt = data.x
    fig, axes = plt.subplots(x_star.shape[1], 1)
    for i, ax in enumerate(axes):
        plot_recon(
            x_gt[:, i],
            data.y[:, i:i+1],
            x_star[:, i:i+1], ax=ax, **kwargs)
    return fig, ax

def plot_recon_sparsity(A, data, beta, W, max_signals=3, **kwargs):
    """
    make a plot to evaluate if W@x_star is sparse
    """
    x_star = solve_lasso(A, data.y[:, :max_signals], beta, W)
    Wx_star = W @ x_star

    fig, axes = plt.subplots(x_star.shape[1], 1)

    for i, ax in enumerate(axes):
        ax.stem(Wx_star[:, i])
    return fig, ax


def plot_recon(x_gt, y, x_star, ax=None, **kwargs):
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()

    ax.plot(x_gt, color='k', label='x_GT')
    ax.plot(y, label='y', color='tab:blue')
    ax.plot(x_star, color='tab:orange', linestyle='dashed', label='x*')

    ax.legend(('x_GT', 'y', 'x*'))

    return fig, ax


def show_W(W_0, W):
    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(W_0)
    ax[0].set_title('W_0')
    ax[1].imshow(W)
    ax[1].set_title('W*')
    fig.show()
