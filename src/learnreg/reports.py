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

import learnreg as lr

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

    W = get_array_from_run(run, 'W')

    cfg = Namespace(**run['config'])
    beta = run['info']['beta_W']

    np.random.seed(cfg.SEED)

    A = lr.make_foward_model(cfg.forward_model_type, cfg.n)
    W0 = lr.make_transform(cfg.transform_type, cfg.n, cfg.n, cfg.transform_scale)
    test = lr.make_dataset(cfg.signal_type, A, cfg.noise_sigma, cfg.num_testing)

    beta = lr.find_optimal_beta(A, test.x, test.y, W, 2e1).item()
    MSE = lr.eval_upper(A, test.x, test.y, beta, W).item()

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


# plotting ------------------------
def plot_denoising(data, beta, W, max_signals=3, **kwargs):
    """
    plot examples of the denoising results obtained with W
    """
    A = np.eye(data.x.shape[0])
    x_star = lr.solve_lasso(A, data.y[:, :max_signals], beta, W)
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
