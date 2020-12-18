import numpy as np
import torch
import cvxpy as cp
import matplotlib.pyplot as plt
import collections
import random
import functools

from cvxpylayers.torch import CvxpyLayer

# datatypes
Dataset = collections.namedtuple('Dataset', ['x', 'y'])

# top-level driver code
def learn_for_denoising(n, num_signals, noise_sigma, beta, SEED, learn_opts):
    # init
    torch.manual_seed(SEED)  # make repeatable

    # setup data
    A = torch.eye(n, n)
    train = make_dataset(A, num_signals=num_signals, sigma=noise_sigma)

    # setup transform
    W = torch.randn(n-1, n)
    W0 = W.clone()

    # learn
    W = main(A, beta, W, train, **learn_opts)

    return train, W0, W


def make_opti(algo, W, opts):
    if algo == 'LBFGS':
        opti = torch.optim.LBFGS((W,), **opts)
    elif algo == 'SGD':
        opti = torch.optim.SGD((W,), **opts)
    else:
        raise ValueError(algo)
    return opti


# problem setup

def eval_lasso(A, x, y, beta, W):
    k = W.shape[0]
    num = y.shape[1]
    problem, params = setup_cvxpy_problem(A, k, num)
    params['y'].value = y.numpy()
    params['beta'].value = beta
    params['W'].value = W.numpy()
    params['x'].value = x.numpy()
    params['z'].value = (W @ x).numpy()

    return problem.objective.expr.value


def solve_lasso(A, y, beta, W):
    k = W.shape[0]
    num = y.shape[1]
    problem, params = setup_cvxpy_problem(A, k, num)
    params['y'].value = y.numpy()
    params['beta'].value = beta
    params['W'].value = W.numpy()

    problem.solve()

    return torch.tensor(params['x'].value, dtype=torch.float)

@functools.lru_cache  # caching because making the problem object is slow
def setup_cvxpy_problem(A, k, batch_size=1):
    """
    sets up a cvxpy Problem representing
    argmin_x 1/2||Ax - y||_2^2 + beta||Wx||_1
    where A and beta are fixed and
    y and W are left as free parameters

    A: m x n
    x: n x batch_size
    y: m x batch_size

    beta: scalar
    W: k x n
    """
    m, n = A.shape
    x = cp.Variable((n, batch_size), name='x')
    y = cp.Parameter((m, batch_size), name='y')
    beta = cp.Parameter(name='beta', nonneg=True)
    W = cp.Parameter((k, n), name='W')
    z = cp.Variable((k, batch_size), name='z')

    objective_fn = 0.5 * cp.sum_squares(A @ x - y) + beta*cp.sum(cp.abs(z))
    constraint = [W @ x == z]  # writing with this splitting makes is_dpp True
    problem = cp.Problem(cp.Minimize(objective_fn), constraint)

    params = {'y':y,
              'beta':beta,
              'W':W,
              'z':z,
              'x':x}

    return problem, params


def make_signal(n, jump_freq=0.1, num_signals=1):
    jumps = torch.rand(n, num_signals) <= jump_freq
    heights = torch.randn(n, num_signals)
    heights[~jumps] = 0
    sigs = heights.cumsum(dim=0)
    sigs = sigs - sigs.mean(dim=0) + torch.randn(num_signals)
    return sigs


def make_measurement(x, A, sigma):
    y = A @ x + sigma * torch.randn_like(x)
    return y


def make_dataset(A, num_signals, sigma):
    x = make_signal(A.shape[1], num_signals=num_signals)
    y = make_measurement(x, A, sigma)
    return Dataset(x=x, y=y)


def main(A, beta, W0, train,
         learning_rate, num_steps, print_interval=1):
    """
    k : sparse code length

    train : NamedTupe with fields x and y
    """

    x, y = train
    train_length = x.shape[1]

    W = W0.clone()
    W.requires_grad_(True)

    MSE_history = torch.full((train_length,), np.nan)

    # main loop
    opti = torch.optim.SGD((W,), learning_rate)

    print(f'{"step":6s}{"epoch":6s}{"index":6s}'
          f'{"cur loss":15s}{"epoch avg loss":15s}')
    for step in range(num_steps):
        epoch = step // train_length
        index = step % train_length

        # compute grad and take a opti step
        opti.zero_grad()
        y_cur = y[:, index:index+1]
        x_cur = x[:, index:index+1]
        with torch.no_grad():
            x_star = solve_lasso(A, y_cur, beta, W)
        W0, Wpm, s = find_signs_alt(x_star, W)
        x_closed = closed_form_alt(W0, Wpm, s, y_cur, beta)

        # check that x_closed is accurate
        with torch.no_grad():
            J_star = eval_lasso(A, x_star, y_cur, beta, W)
            J_closed = eval_lasso(A, x_closed, y_cur, beta, W)
            gap = J_closed - J_star
            if gap > 1e-4:
                print(f'large gap, J_closed={J_closed:.3e}'
                      f'J_star={J_star:.3e}')

        loss = MSE(x_closed, x_cur)
        last_loss = loss.item()
        MSE_history[index] = last_loss
        epoch_loss = MSE_history.mean()
        loss.backward()
        opti.step()

        # print status line
        if step % print_interval == 0 or step == num_steps-1:
            print(f'{step:<6d}{epoch:<6d}{index:<6d}'
                  f'{last_loss:<15.3e}{epoch_loss:<15.3e}')

    return W.detach()





# utilities


def MSE(x, x_gt):
    return ((x - x_gt)**2).mean()


def permute_for_display(W):
    """
    sort the rows of W by starting at the top
    and picking as the next row, the one that is most similar to a shift by one pixel right
    (assumption here is that we are expecting to see filters)

    does not seem to give us much, and makes spurious patterns from noise
    """
    W = W.copy()
    out = np.zeros_like(W)
    out[0, :] = W[0,:]
    ind = 0
    inds_taken = [0]
    for i in range(1, W.shape[0]):
        dists_pos = np.sum((np.roll(W[ind,:], 1) - W)**2, axis=1)
        dists_neg = np.sum((np.roll(W[ind,:], 1) + W)**2, axis=1)

        dists_pos[inds_taken] = np.inf
        dists_neg[inds_taken] = np.inf

        min_pos = dists_pos.min()
        min_neg = dists_neg.min()

        if min_pos < min_neg:
            ind = dists_pos.argmin()
            out[i] = W[ind]
        else:
            ind = dists_neg.argmin()
            W = -W
            out[i] = W[ind]
        inds_taken.append(ind)

    return out

    #FW = np.fft.rfft(W, axis=1)
    #corr = np.fft.irfft( np.conj(FW[0:1, :]) * FW, axis=1)


def find_optimal_beta(A, x_GT, y, W, upper, lower=0):
    def J(beta):
        x_star = solve_lasso(A, y, beta, W)
        return MSE(x_star, x_GT)

    a, b = min_golden(J, lower, upper)
    return (a+b)/2
          
          
          
          
def find_optimal_thres(x_cvx,W,y,beta,lower=1e-16,upper=1e-1):
    def J(thres):
        W0,Wpm,s=find_signs_alt(x_cvx, W, thres)
        x_close=closed_form_alt(W0, Wpm, s, y, beta)
        return MSE(x_close, x_cvx)
    
    a, b = min_golden(J, lower, upper)
    return (a+b)/2          

def min_golden(f, a, b, tol=1e-5):
    '''
    Minimize a unimodal function via Golden section search.
    Useful for picking optimal regularization parameters
    source: wikipedia

    aka "gss" golden section search

    Given a function f with a single local minimum in
    the interval [a,b], gss returns a subset interval
    [c,d] that contains the minimum with d-c <= tol.

    example:
    >>> f = lambda x: (x-2)**2
    >>> a = 1
    >>> b = 5
    >>> tol = 1e-5
    >>> (c,d) = gss(f, a, b, tol)
    >>> print (c,d)
    (1.9999959837979107, 2.0000050911830893)

    adapted from https://en.wikipedia.org/wpiki/Golden-section_search
    '''

    invphi = (np.sqrt(5) - 1) / 2  # 1/phi
    invphi2 = (3 - np.sqrt(5)) / 2  # 1/phi^2

    a, b = (min(a, b), max(a, b))
    h = b - a
    if h <= tol:
        return (a, b)

    # required steps to achieve tolerance
    n = int(np.ceil(np.log(tol / h) / np.log(invphi)))

    c = a + invphi2 * h
    d = a + invphi * h
    yc = f(c)
    yd = f(d)

    for k in range(n - 1):
        if yc < yd:
            best_val, x = yc, c
            b = d
            d = c
            yd = yc
            h = invphi * h
            c = a + invphi2 * h
            yc = f(c)
        else:
            best_val, x = yd, d
            a = c
            c = d
            yc = yd
            h = invphi * h
            d = a + invphi * h
            yd = f(d)

        #logging.info(f"iter {k}, f({x}) = {best_val}")

    if yc < yd:
        return (a, d)
    else:
        return (c, b)


# plotting
def plot_denoising(data, beta, W, **kwargs):
    A = torch.eye(data.x.shape[0])
    x_star = solve_lasso(A, data.y, beta, W)
    x_gt = data.x
    return plot_recon(x_gt, data.y, x_star, **kwargs)

def plot_recon(x_gt, y, x_star, ax=None, **kwargs):
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()

    for a, b, c in zip(x_gt.T, y.T, x_star.T):
        ax.plot(a, color='k', label='x_GT')
        ax.plot(b, label='y', color='tab:blue')
        ax.plot(c, color='tab:orange', linestyle='dashed', label='x*')

    ax.legend(('x_GT', 'y', 'x*'))

    return fig, ax


def show_W(W_0, W):
    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(W_0)
    ax[0].set_title('W_0')
    ax[1].imshow(W)
    ax[1].set_title('W*')
    fig.show()


def find_sign_pattern(z, threshold=1e-10):
    """
    z: (k,), tensor - D@x_opt

    returns
    S_0 (k_0, k) tensor - selection matrix
    S_pm (k_{+-}, k) tensor - selection matrix
    s (k_{+-},) tensor
    """
    is_zero = z.abs() < threshold  # boolean, (k,)
    i=is_zero.reshape(is_zero.shape[0])
    signs = torch.sign(z)  # {-1, 0, 1}, (k,)
    s = signs[~i]  # {-1, 1}, (k_{+-})
    I = torch.eye(len(z))
    S_0 = I[i, :]
    S_pm = I[~i, :]

    return S_0, S_pm, s

def find_signs_alt(x, W, threshold=1e-4):
    """
    given x* and W, find the necessary sign matrices:
    W_0, W_pm, and s
    """
    Wx = W @ x
    is_zero = (Wx.abs() < threshold).squeeze()
    W0 = W[is_zero, :]
    Wpm = W[~is_zero, :]
    s = (Wpm @ x).sign()

    return W0, Wpm, s

def closed_form_alt(W0, Wpm, s, y, beta):
    y_term = y - beta * Wpm.T @ s
    # proj = W0.pinverse() @ W0
    proj = W0.T @ (W0 @ W0.T).inverse() @ W0
    return y_term - proj @ y_term


def closed_form(S_0, S_pm, s, W, l, b):
    """
    implemention of (XXX) from "XXXXX" Tibshi...
    https://arxiv.org/pdf/1805.07682.pdf

    """
    W=W.float()
    S_0=S_0.float()
    S_pm=S_pm.float()
    W_b=torch.matmul(S_0,W)
    Wb=torch.matmul(S_pm,W)
    W_bt=torch.transpose(W_b,0,1)
    Wbt=torch.transpose(Wb,0,1)
    s=s.float()
    proj = (W_bt @ torch.pinverse(W_b @ W_bt) @ W_b)
    Pnull=torch.eye(W_b.shape[1])-proj
    temp= (b - l * Wbt @ s)
    beta=Pnull @ temp
    return beta

def compute_loss(x, y, beta, W):
    assert x.shape == y.shape
    return MSE(x, y) + beta * torch.sum(torch.abs((W@x)))


def optimize(D,bh,beta):
    n = D.shape[1]
    x_l1 = cp.Variable(shape=(n,1))
    # Form objective.
    obj = cp.Minimize( 0.5*cp.sum_squares(x_l1-bh) + beta*cp.norm(D@x_l1, 1))
    # Form and solve problem.
    prob = cp.Problem(obj)
    prob.solve()
    #print("optimal objective value: {}".format(obj.value))
    return torch.tensor(x_l1.value, dtype=torch.float)


def make_conv(h, n):
    """
    Return a matrix, H, that implements convolution of a length-n signal by h

    if h is length-m, the length of the (valid) convoluation result
    is n-m+1, so H has shape (n-m+1, n)

    h = [1.0, -1.0], n = 4 ->
    H =
    [[-1, 1, 0, 0,],
     [0, -1, 1, 0,],
     [0, 0, -1, 1,]]

    """
    assert h.ndim == 1

    m = len(h)
    pad = n-m # adds to beginning and end
    h_repeat = torch.nn.functional.unfold(
        h.view(1, 1, -1, 1), (n, 1),
        padding=(pad, 0))
    return h_repeat[0].T.flip(1)

def make_TV(n):
    return make_conv(torch.tensor([1.0, -1.0]), n)

def TV_denoise(m,x1,y1,b_opt):
    """
    b_opt is the penalty strength
    y1 is the noisy version of x1
    xrec is the TV reconstructed denoise signal
    """
    tv=torch.zeros(m)
    tv[0]=1.0
    tv[1]=-1.0
    TV=b_opt*create_circulant(tv)
    xrec=optimize(TV,y1,1)
    return xrec,src.MSE(x1,xrec)


# deprecated, don't use ----------------------------
def create_circulant(r):
    DeprecationWarning('use make_conv instead')
    A=torch.zeros(r.shape[0],r.shape[0])
    rn=r/torch.norm(r,2)  # normalize rows
    for i in range(r.shape[0]):
        A[i,:]=torch.roll(rn,i)
    return A
