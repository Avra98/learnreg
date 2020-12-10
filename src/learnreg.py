import numpy as np
import torch
import cvxpy as cp
import matplotlib.pyplot as plt
import collections
import random


from cvxpylayers.torch import CvxpyLayer

# datatypes
Dataset = collections.namedtuple('Dataset', ['x', 'y'])

#

def make_opti(opts, W):
    if opts[0] == 'LBFGS':
        opti = torch.optim.LBFGS((W,), **opts[1])
    elif opts[0] == 'SGD':
        opti = torch.optim.SGD((W,), **opts[1])
    else:
        raise ValueError(opts[0])
    return opti


# problem setup

def solve_lasso(A, y, W):
    m, n = A.shape
    k = W.shape[0]
    num = y.shape[1]
    _, solve = setup_cvxpy_problem(m, n, k, num)

    return solve(A, y, W)

def setup_cvxpy_problem(m, n, k, batch_size=1):
    """
    argmin_x ||Ax - y||_2^2 + ||Wx||_1

    CvxpyLayer from Differentiable Convex Optimization Layers
    https://web.stanford.edu/~boyd/papers/pdf/diff_cvxpy.pdf
    """
    A = cp.Parameter((m, n), name='A')
    x = cp.Variable((n, batch_size), name='x')
    z = cp.Variable((k, batch_size), name='z')
    y = cp.Parameter((m, batch_size), name='y')
    W = cp.Parameter((k, n), name='W')
    objective_fn = cp.sum_squares(A @ x - y) + cp.sum(cp.abs(z))
    constraints = [W @ x == z]
    problem = cp.Problem(cp.Minimize(objective_fn), constraints)

    layer = CvxpyLayer(
        problem, parameters=[A, y, W], variables=[x])

    def solve_lasso(A, y, W):
        return layer(A, y, W)[0]  # because layer returns a list

    return problem, solve_lasso

def setup_conv_problem(m, n, filter_length):
    raise ValueError('careful, this code does not seem to work, solutions are always zero')
    """
    argmin_x ||Ax - y||_2^2 + ||W * x||_1
    """
    A = cp.Parameter((m, n))
    x = cp.Variable((n, 1))
    z = cp.Variable((n+filter_length-1, 1))
    y = cp.Parameter((m, 1))
    W = cp.Parameter((filter_length, 1))
    objective_fn = cp.sum_squares(A @ x - y) + 1e-4 * cp.sum(cp.abs(z))
    constraints = [z == cp.conv(W, x)]
    problem = cp.Problem(cp.Minimize(objective_fn), constraints)

    layer = CvxpyLayer(
        problem, parameters=[A, y, W], variables=[x])

    def solve_lasso(A, y, W):
        return layer(A, y, W)[0]  # because layer returns a list

    return problem, solve_lasso


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


def make_set(A, num_signals, sigma):
    x = make_signal(A.shape[1], num_signals=num_signals)
    y = make_measurement(x, A, sigma)

    return Dataset(x=x, y=y)


def make_data(n, m, sigma, train_size, val_size=0):
    do_val = val_size > 0

    A = torch.eye(m, n)

    train = make_set(A, train_size, sigma)

    if do_val:
        val = make_set(A, val_size, sigma)
    else:
        val = None

    return train, val, A


def main(A, k, W_type, train, beta0,
         opti_opts, num_steps,
         batch_size=1, val=None, val_interval=1, print_interval=1,
         history_length=None, max_batch_size=None):
    """
    k : sparse code length

    train : NamedTupe with fields x and y
    """

    batch_increment = 2  # add this on val fail

    # infer some problem specifics from parameters
    do_val = val is not None
    decrease_batch_size = history_length is not None

    train_size = train.x.shape[1]
    assert batch_size <= train_size

    if max_batch_size is None:
        max_batch_size = train_size

    # setup
    m, n = A.shape

    if do_val:
        val_size = val.x.shape[1]
        _, solve_val = setup_cvxpy_problem(m, n, k, val_size)

    _, solve_batch = setup_cvxpy_problem(m, n, k, batch_size)

    if W_type == 'full':
        params0 = beta0 * torch.randn(k, n)
        def make_W(params):
            return params
    elif W_type == 'conv':
        params0 = beta0 * torch.randn(n)
        def make_W(params):
            pad = n-1 # always want to add 2n-2 values
            params = torch.nn.functional.pad(params, (pad, pad))
            return torch.nn.functional.unfold(params.view(1, 1, -1, 1), (n, 1)).squeeze().T
    else:
        raise ValueError(W_type)

    params = params0.clone()
    params.requires_grad_(True)
    W = make_W(params)

    # main loop
    opti = make_opti(opti_opts, params)

    last_loss = [None]  # use this list to get losses out of the closure

    if decrease_batch_size:
        val_history = collections.deque(history_length*[float('inf')], history_length)
        best_loss = float('inf')
    val_fail = False

    # setup batches
    batch_ind = 0
    shuffled_inds = random.sample(range(train_size), train_size)

    print(f'{"step":6s}{"batch ind":12s}{"batch loss":15s}{"val loss":15s}')
    for step in range(num_steps):
        # compute grad and take a opti step
        def closure():  # why "closure"? see https://pytorch.org/docs/stable/optim.html
            opti.zero_grad()
            batch_slice = slice(batch_ind, batch_ind+batch_size)
            x_star = solve_batch(A, train.y[:, shuffled_inds[batch_slice]], W)
            loss = MSE(x_star, train.x[:, shuffled_inds[batch_slice]])
            loss.backward()
            last_loss[0] = loss.item()
            return loss
        opti.step(closure)
        W = make_W(params)

        # do validation
        if do_val and step % val_interval == 0 or step == num_steps-1:
            x_star = solve_val(A, val.y, W)
            val_loss = MSE(x_star, val.x)

            # handle batch size increase if val loss not decreasing
            if decrease_batch_size:
                val_history.appendleft(val_loss)
                if val_loss < best_loss:
                    best_loss = val_loss

                if best_loss not in val_history:
                    if batch_size + batch_increment < max_batch_size:
                        batch_size += batch_increment
                        _, solve_batch = setup_cvxpy_problem(m, n, k, batch_size)
                    val_fail = True
                    best_loss = float('inf')
        else:
            val_loss = float('nan')

        do_print = step % print_interval == 0 or step == num_steps-1
        if do_print or val_fail:
            print(f'{step:<6d}{batch_ind:<12d}'
                  f'{last_loss[0]:<15.3e}{val_loss:<15.3e}', end='')
            if val_fail:
                print(f'validation fail -> batch_size={batch_size}')
                val_fail = False
            else:
                print('')

        # update batch index for next step
        batch_ind += batch_size
        if batch_ind + batch_size > train_size:
            batch_ind = 0
            shuffled_inds = random.sample(range(train_size), train_size)


    return W.detach(), make_W(params0)





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


def find_optimal_beta(W, y, x_GT, upper, lower=0):
    def J(beta):
        return MSE(optimize(W, y, beta), x_GT)

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
def solve_and_plot(A, data, W, **kwargs):
    x_star = solve_lasso(A, data.y, W)
    x_gt = data.x
    return plot_recon(x_gt, data.y, x_star, **kwargs)

def plot_recon(x_gt, y, x_star, ax=None, **kwargs):
    if ax is None:
        fig, ax = plt.subplots()

    for a, b, c in zip(x_gt.T, y.T, x_star.T):
        ax.plot(a, color='k', label='x_GT')
        ax.plot(b, label='y', color='tab:blue')
        ax.plot(c, color='tab:orange', linestyle='dashed', label='x*')

    ax.legend(('x_GT', 'y', 'x*'))
    return ax.get_figure(), ax


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
    Pnull=torch.eye(W_b.shape[1])-(W_bt @ torch.pinverse(W_b @ W_bt) @ W_b)
    temp= (b - l * Wbt @ s)
    temp=temp.float()
    beta=Pnull @ temp
    return beta

def compute_loss(x, y, beta, W):
    return MSE(x, y) + beta * torch.sum(torch.abs((W@x)))


def optimize(D,bh,l):
    n=100
    x_l1 = cp.Variable(shape=(n,1))
    # Form objective.
    obj = cp.Minimize( 0.5*cp.sum_squares(x_l1-bh) + l*cp.norm(D@x_l1, 1))
    # Form and solve problem.
    prob = cp.Problem(obj)
    prob.solve()
    #print("optimal objective value: {}".format(obj.value))
    return torch.tensor(x_l1.value, dtype=torch.float)


def create_circulant(r):
    A=torch.zeros(r.shape[0],r.shape[0])
    rn=r/torch.norm(r,2)  # normalize rows
    for i in range(r.shape[0]):
        A[i,:]=torch.roll(rn,i)
    return A

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
