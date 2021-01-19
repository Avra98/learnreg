"""
Assume everything is a numpy array.
Try to keep everything a numpy array.
"""
import numpy as np
import scipy
import torch
import cvxpy as cp
import collections
import random
import functools


# datatypes
Dataset = collections.namedtuple('Dataset', ['x', 'y'])

# top-level driver code


#
def main(signal_type,
         n,
         k,
         forward_model_type,
         noise_sigma,
         num_training,
         transform_type,
         transform_scale,
         num_testing,
         SEED,
         learning_rate,
         num_steps,
         sign_threshold,
         _run=None):

    np.random.seed(SEED)
    A = make_foward_model(forward_model_type, n)
    train = make_dataset(signal_type, A, noise_sigma, num_training)
    W = make_transform(transform_type, n, k, transform_scale)
    W0 = W.copy()

    print_interval = 100
    W = do_learning(A, 1.0, W, train, learning_rate, num_steps, print_interval, sign_threshold, logger=_run)

    test = make_dataset(signal_type, A, noise_sigma, num_testing)

    beta = find_optimal_beta(A, test.x, test.y, W)
    MSE = eval_upper(A, test.x, test.y, beta, W)

    if _run is not None:
        _run.info['A'] = A
        _run.info['MSE'] = float(MSE)
        _run.info['beta'] = float(beta)
        _run.info['W0'] = W0
        _run.info['W'] = W
    else:
        return MSE, beta, W


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
    """
    compute J(x) for the lower-level  prblem
    """
    k = W.shape[0]
    num = y.shape[1]
    problem, params = setup_cvxpy_problem(*A.shape, k, num)
    params['A'].value = A
    params['y'].value = y
    params['beta'].value = beta
    params['W'].value = W
    params['x'].value = x
    params['z'].value = (W @ x)

    return problem.objective.expr.value


def eval_upper(A, x_GT, y, beta, W):
    """
    return the upper cost value at point W
    """
    x = solve_lasso(A, y, beta, W)
    return MSE(x, x_GT)


def solve_lasso(A, y, beta, W):
    k = W.shape[0]
    num = y.shape[1]
    problem, params = setup_cvxpy_problem(*A.shape, k, num)
    params['A'].value = A
    params['y'].value = y
    params['beta'].value = beta
    params['W'].value = W

    problem.solve()

    return params['x'].value

@functools.lru_cache()  # caching because making the problem object is slow
def setup_cvxpy_problem(m, n, k, batch_size=1):
    """
    sets up a cvxpy Problem representing

    argmin_x 1/2||Ax - y||_2^2 + beta||Wx||_1

    where A, y, beta, and W are left as free parameters

    A: m x n
    x: n x batch_size
    y: m x batch_size
    beta: scalar
    W: k x n

    """
    A = cp.Parameter((m,n), name='A')
    x = cp.Variable((n, batch_size), name='x')
    y = cp.Parameter((m, batch_size), name='y')
    beta = cp.Parameter(name='beta', nonneg=True)
    W = cp.Parameter((k, n), name='W')
    z = cp.Variable((k, batch_size), name='z')

    objective_fn = 0.5 * cp.sum_squares(A @ x - y) + beta*cp.sum(cp.abs(z))
    constraint = [W @ x == z]  # writing with this splitting makes is_dpp True
    problem = cp.Problem(cp.Minimize(objective_fn), constraint)

    params = {
        'A':A,
        'y':y,
        'beta':beta,
        'W':W,
        'z':z,
        'x':x
    }

    return problem, params

def make_signal(sig_type, n, **kwargs):
    if sig_type == 'piecewise_constant':
        sigs = make_piecewise_const_signal(n, **kwargs)
    elif sig_type == 'DCT-sparse':
        sigs = make_DCT_signal(n, **kwargs)
    else:
        raise ValueError(sig_type)

    return sigs

def make_piecewise_const_signal(n, jump_freq=0.1, num_signals=1):
    """
    piecewise constant signals in the range [0, 1)
    each piece height is chosen uniformly at random
    """
    jumps = np.random.rand(n, num_signals) <= jump_freq
    inds = np.cumsum(jumps, axis=0)
    heights = np.random.rand(n, num_signals)

    sigs = heights[inds, range(num_signals)]

    return sigs

def make_DCT_signal(n, nonzero_freq=0.1, num_signals=1):
    """
    make signals that are well-sparsified by the DCT,

    specifically, scipy.fft.dct(sigs, axis=0) will be sparse
    so will
    W = lr.make_transform('DCT', n)
    W @ sigs
    """
    support = np.random.rand(n, num_signals) <= nonzero_freq
    coeffs = 2 * (np.random.rand(n, num_signals) - 0.5) * support
    sigs = scipy.fft.idct(coeffs, axis=0, norm="ortho")
    return sigs


def make_measurement(x, A, sigma):
    y = A @ x + sigma * np.random.randn(*x.shape)
    return y


def make_dataset(signal_type, A, noise_sigma, num_signals):

    x = make_signal(signal_type, A.shape[1], num_signals=num_signals)
    y = make_measurement(x, A, noise_sigma)
    return Dataset(x=x, y=y)

def patch_dataset(num_signals,sigma):
    import hdf5storage
    ##Import data
    mat = hdf5storage.loadmat('patch.mat')
    x = mat['impatc']
    y = mat['impatn']   ##Generate two types of measurements, 1) noise added in image domain (y), 2) noise added in patch domain (y1)
    ##Subtract mean
    x=x-np.mean(x,axis=0)
    y=y-np.mean(y,axis=0)
    ##Convert to torch
    #x=torch.from_numpy(x)
    #y=torch.from_numpy(y)
    ##Generate y1
    y1=x + np.random.normal(0, sigma, x.shape)
    nData=x.shape[1]
    ##Scramble data
    perm = np.random.permutation(nData)
    x = x[:,perm]
    y = y[:,perm]
    y1 = y1[:,perm]
    ##Select subset of data to work with
    x=x[:,:num_signals]
    y1=y1[:,:num_signals]

    return Dataset(x=x, y=y1)




def do_learning(A, beta, W0, train,
                learning_rate, num_steps, print_interval=1,
                sign_threshold=1e-6, logger=None):
    """
    k : sparse code length

    train : NamedTupe with fields x and y

    logger : sacred.Run object for storing online metrics
    """

    x, y = train
    train_length = x.shape[1]

    W = torch.tensor(W0)
    W.requires_grad_(True)

    MSE_history = torch.full((train_length,), np.nan)

    # main loop
    opti = torch.optim.SGD((W,), learning_rate)

    print(f'{"step":12s}{"epoch":6s}{"index":6s}'
          f'{"cur loss":15s}{"epoch avg loss":15s}')
    for step in range(num_steps):
        epoch = step // train_length
        index = step % train_length

        # compute grad and take a opti step
        opti.zero_grad()
        y_cur = y[:, index:index+1]
        x_cur = x[:, index:index+1]
        with torch.no_grad():
            x_star = solve_lasso(A, y_cur, beta, W.numpy())
            # threshold = find_optimal_thres(x_star, W, y_cur , beta)
        W0, Wpm, s = find_signs(torch.tensor(x_star), W, threshold=sign_threshold)
        x_closed = closed_form(W0, Wpm, s, torch.tensor(y_cur), beta)

        # check that x_closed is accurate
        with torch.no_grad():
            J_star = eval_lasso(A, x_star, y_cur, beta, W.numpy())
            J_closed = eval_lasso(A, x_closed.numpy(), y_cur, beta, W.numpy())
            gap = np.abs(J_closed - J_star)

            if gap/J_star > 1e-2:
                print(f'large gap: J_closed={J_closed:.3e}, '
                      f'J_star={J_star:.3e}')
                if logger is not None:
                    logger.log_scalar('gap', gap, step)

        loss = MSE(x_closed, torch.tensor(x_cur))
        last_loss = loss.item()
        MSE_history[index] = last_loss
        epoch_loss = np.nanmean(MSE_history)
        loss.backward()
        opti.step()

        # print status line
        if step % print_interval == 0 or step == num_steps-1:
            print(f'{step:<12d}{epoch:<6d}{index:<6d}'
                  f'{last_loss:<15.3e}{epoch_loss:<15.3e}')

            #if logger is not None:
            #    logger.log_scalar('train.loss', last_loss, step)

    return np.array(W.detach())


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


def find_optimal_beta(A, x_GT, y, W, lower=0, upper=None):
    # heuristic to pick the upper limit
    if upper is None:
        cost_zero = eval_lasso(A, np.zeros_like(x_GT), y, 0.0, W)
        data_GT = eval_lasso(A, x_GT, y, 0.0, W)
        reg_GT = eval_lasso(np.zeros_like(A), x_GT, np.zeros_like(y), 1.0, W)

        upper = (cost_zero - data_GT) / reg_GT

        upper = max(upper, lower)


    def J(beta):
        x_star = solve_lasso(A, y, beta, W)
        return MSE(x_star, x_GT)

    a, b = min_golden(J, lower, upper)
    val = (a+b)/2
    if np.abs(val-lower)/upper < 1e-2 or np.abs(val-upper)/upper < 1e-2:
        print("warning, optimal beta is close to one of the limits")

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


def find_signs(x, W, threshold=1e-6):
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

def closed_form(W0, Wpm, s, y, beta):
    """
    implemention of (XXX) from "XXXXX" Tibshi...
    https://arxiv.org/pdf/1805.07682.pdf

    """
    y_term = y - beta * Wpm.T @ s
    # proj = W0.pinverse() @ W0  # todo: do we prefer this or the next line?
    proj = W0.T @ (W0 @ W0.T).inverse() @ W0
    return y_term - proj @ y_term

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
    return xrec, src.MSE(x1,xrec)

# forward models
def make_foward_model(forward_model_type, n):
    if forward_model_type == 'identity':
        A = np.eye(n)
    else:
        raise ValueError(forward_model_type)

    return A

# transforms ----------------------


def make_transform(transform_type, n, k, scale=1.0):
    k = int(k)
    if transform_type == 'identity':
        #assert k <= n
        W = np.eye(k, n)
        W = W - W.mean(axis=1, keepdims=True)
    elif transform_type == 'TV':
        W = make_conv(np.array([1.0, -1.0]), n)
        #assert k <= W.shape[0]
        W = W[:k]
    elif transform_type == 'DCT':
        #assert k <= n
        W = scipy.fft.dct(np.eye(n), axis=0, norm='ortho')
        W = W[:k]
    elif transform_type == 'random':
        W = np.random.randn(k, n)
    else:
        raise ValueError(transform_type)

    return scale * W


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
        torch.from_numpy(h).view(1, 1, -1, 1), (n, 1),
        padding=(pad, 0))
    return h_repeat[0].T.flip(1).numpy()
