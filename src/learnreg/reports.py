"""
code for examining the results of experiments,
including making plots and text reports
"""
import matplotlib.pyplot as plt
import pymongo
import jsonpickle
import jsonpickle.ext.numpy as jsonpickle_numpy
from argparse import Namespace
import numpy as np
import scipy.optimize
import seaborn as sns

import learnreg as lr

transform_cmap = sns.diverging_palette(250, 30, l=65, center="dark", as_cmap=True)

jsonpickle_numpy.register_handlers()


def get_runs_from_db():
    client = pymongo.MongoClient()
    db = client.sacred
    return db.runs



def get_array_from_run(run, array_name):
    X = jsonpickle.unpickler.Unpickler().restore(run['info'][array_name])
    return X


def make_plots(exp_id, prefix=None):
    if prefix is not None:
        prefix = prefix + ' '

    runs = get_runs_from_db()
    run = runs.find_one({'_id': exp_id})

    cfg = Namespace(**run['config'])

    W = get_array_from_run(run, 'W')
    beta = run['info']['beta']
    A = get_array_from_run(run, 'A')
    W0 = get_array_from_run(run, 'W0')
    MSE = run['info']['MSE']

    np.random.seed(cfg.SEED)

    test = lr.make_dataset(cfg.signal_type, A, cfg.noise_sigma, cfg.num_testing)

    fig, ax = plot_denoising(test, beta, W)
    fig.suptitle(prefix + f'reconstructions\nbeta={beta:.3e}, testing MSE={MSE:.3e}')
    fig.tight_layout()
    fig.show()

    fig, ax = plot_recon_sparsity(A, test, beta, W)
    fig.suptitle(prefix + 'W @ x*(W)')
    fig.show()

    fig, ax = show_W(W0, W)
    fig.suptitle(prefix)
    fig.show()

    return W


# plotting ------------------------
def plot_denoising(data, beta, W, max_signals=3, **kwargs):
    """
    plot examples of the denoising results obtained with W
    """
    A = np.eye(data.x.shape[0])
    x_star = lr.solve_lasso(A, data.y[:, :max_signals], beta, W)
    x_gt = data.x
    fig, axes = plt.subplots(x_star.shape[1], 1)
    if data.y.shape[1] == 1: # make axes a list always
        axes = [axes]
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
    x_star = lr.solve_lasso(A, data.y[:, :max_signals], beta, W)
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
    fig.tight_layout()
    return fig, ax

def show_W_patch(W):
    k, n = W.shape
    p = int(np.sqrt(n))  # patch side length
    m = int(np.ceil(np.sqrt(k)))  # determine number of subplots
    fig, axes = plt.subplots(m, m, constrained_layout=True)

    vmin = W.min()
    vmax = W.max()
    for i in range(k):
        ax = axes.flatten()[i]
        im = ax.imshow(
            W[i, :].T.reshape(p, p),
            vmin=vmin, vmax=vmax, cmap=transform_cmap)

    for ax in axes.flatten():
        ax.axis('off')

    fig.colorbar(im, ax=axes, location='bottom')
    return fig, axes



def rearrange_to_baseline(W, B):
    """
    return a copy of W with rows rearranged
    to match with B


    """

    # normalize rows to remove effect of scaling
    Wnorm = W / np.sqrt(np.sum(W**2, axis=1, keepdims=True))
    Bnorm = B / np.sqrt(np.sum(B**2, axis=1, keepdims=True))

    corr = Wnorm @ Bnorm.T  # shape: (k1, k2)

    row_inds, col_inds = scipy.optimize.linear_sum_assignment(np.abs(corr), maximize=True)

    Wout = np.full((max(W.shape[0], B.shape[0]), B.shape[1]), np.nan)

    # swap signs where corr was negative
    Wout[col_inds] = W[row_inds] * np.sign(corr[row_inds, col_inds])[:, np.newaxis]

    if B.shape[0] < W.shape[0]:  # put leftover rows at the bottom
        leftover_inds = list(set(range(W.shape[0])) - set(row_inds))
        Wout[B.shape[0]:] = W[leftover_inds]


    return Wout


def compare_matrices(xs, titles):
    x1, x2 = xs

    fig, axes = plt.subplots(1, 2, sharey=True, sharex=True)
    ax = axes[0] # useful later for sharing axes
    args = dict(vmin=-np.max(np.abs((x1, x2))),
                vmax=np.max(np.abs((x1, x2))),
                cmap='BrBG',
                )
    ax = axes[0]
    axes[0].imshow(x1.copy(), **args)
    axes[0].set_title(titles[0])
    im = axes[1].imshow(x2.copy(), **args)
    axes[1].set_title(titles[1])
    fig.tight_layout()
    fig.colorbar(im, ax=axes, location='bottom')
    fig1 = fig

    fig, axes = plt.subplots(1, 2, sharex=True, sharey=True)
    im0 = axes[0].imshow(np.abs(x1 - x2))
    axes[0].set_title('absolute error')

    # join to previous graphs
    axes[0].get_shared_x_axes().join(axes[0], ax)
    axes[0].get_shared_y_axes().join(axes[0], ax)

    im1 = axes[1].imshow(np.abs(x1 - x2) / np.abs(x1))
    axes[1].set_title('relative error')
    fig.suptitle(f'{titles[0]} vs. {titles[1]}')
    fig.tight_layout()
    fig.colorbar(im0, ax=[axes[0]], location='bottom')
    fig.colorbar(im1, ax=[axes[1]], location='bottom')


    return fig1, fig
