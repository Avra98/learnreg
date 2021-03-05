"""
Assume everything is a numpy array.
Try to keep everything a numpy array.
"""
import numpy as np
import scipy
import torch
import collections
import random
import math
import shortuuid
import time
import pathlib
import imageio
import learnreg.opt as opt



# datatypes
Dataset = collections.namedtuple('Dataset', ['x', 'y'])

# top-level driver code
def main(signal_type,
         n,
         forward_model_type,
         noise_sigma,
         transform_type,
         transform_opts,
         transform_scale,
         learning_rate,
         num_steps,
         batch_size,
         sign_threshold,
         seed,
         _run=None,
        num_training=10000,
        num_testing=1000):
    """


    """

    A = make_forward_model(forward_model_type, n)
    train = make_dataset(signal_type, A, noise_sigma, num_training)

    if signal_type == 'image_patch':
        test = train
    else :
        test = make_dataset(signal_type, A, noise_sigma, num_testing)

    W = make_transform(transform_type, n, transform_scale, **transform_opts)

    solver = opt.CvxpySolver(A, W.shape[0], sign_threshold)

    W = do_learning(
        A, W, train, solver.eval_upper,
        learning_rate, num_steps, batch_size, print_interval=100)

    if signal_type == 'image_patch':
        test.x = test.x[:,:num_testing]
        test.y = test.y[:,:num_testing]
        fig, ax = reports.show_W_patch(W)


    beta = find_optimal_beta(A, test.x, test.y, W)

    x_hat = opt.solve_lasso(A, test.y, beta, W)
    MSE = opt.MSE(x_hat, test.x)

    print(f'beta: {beta:.3e}')
    print(f'test MSE: {MSE:.3e}')

    if _run is not None:
        _run.info['MSE'] = MSE
        _run.info['beta'] = beta
        _run.info['W'] = W
    else:
        return W
        #return MSE, beta, W


def make_dataset(signal_type, A, noise_sigma, num_signals, signal_opts=None):
    oned_signal_types = ['piecewise_constant', 'DCT-sparse',
                       'constant_patch']
    twod_signal_types = ['image_patch']
    patch_size= int(np.sqrt(A.shape[1]))

    if signal_type in oned_signal_types:
        x = make_signal(signal_type, A.shape[1], num_signals=num_signals,
                    signal_opts=signal_opts)
        y = make_measurement(x, A, noise_sigma)

    elif signal_type in twod_signal_types:
        [x, y,_,_,_] = image2patchset(noise_sigma,patch_size,filename='barbara.png')

    else:
        raise ValueError(signal_type)

    train = datasetconv(x,y)
    return train


##Driver code for training patches from images. Denoising is on the same patches of the images
def main_image(filename,patch_size,
         forward_model_type,
         noise_sigma,
         transform_type,
         transform_scale,
         SEED,
         learning_rate,
         num_steps,
         sign_threshold,
         batch_size,
         _run=None):

    np.random.seed(SEED)
    n=patch_size**2
    k=n-1
    A = make_forward_model(forward_model_type, n)
    train,origin,img,noise_img=image2patchset(noise_sigma,patch_size,filename='barbara.png')
    W = make_transform(transform_type, n, k, transform_scale)
    W0 = W.copy()
    beta = 1.0
    print_interval = 100
    #W = do_learning(A, beta, W, train, learning_rate, num_steps, print_interval, sign_threshold, logger=_run)

    solver = opt.CvxpySolver(A, W.shape[0], sign_threshold)

    W=do_learning(
      A, W, train, solver.eval_upper,
      learning_rate, num_steps, batch_size, print_interval=100)



    beta_W = 1.0
    denoised=np.zeros((patch_size**2,train.x.shape[1]))
    #beta_W = find_optimal_beta(A, test.x, test.y, W, 1e2).item()
    batch_size2=64  ## Solving lasso on all the patches faces memory issue, so solve lasso by batches
    for i in range(0,train.x.shape[1],batch_size2):
        denoised[:,i:i+batch_size2]=opt.solve_lasso(A, train.y[:,i:i+batch_size2], beta_W, W)

    denoised_image=patchset2image(denoised,origin)

    if _run is not None:
        _run.info['MSE'] = MSE
        _run.info['beta_W'] = beta_W
        _run.info['W'] = W
    else:
         return denoised_image,W,img,noise_img





def test_image(testnoise,W,beta,images = ['cameraman.tif','house.tif'
            ,'jetplane.tif','lake.tif'
            ,'livingroom.tif','mandril_gray.tif'
            ,'peppers_gray.tif'
            ,'pirate.tif','walkbridge.tif'
            ,'woman_blonde.tif','woman_darkhair.tif']):

    d2=np.zeros([512,512,len(images)])
    c2=np.zeros([512,512,len(images)])
    n2=np.zeros([512,512,len(images)])
    pn2=np.zeros([1,len(images)])
    pd2=np.zeros([1,len(images)])

    for i in range(len(images)):
        [d,c,n,pn,pd]=denoise_with_W(testnoise,W,beta,images[i])
        d2[:,:,i]=d
        c2[:,:,i]=c
        n2[:,:,i]=n
        pn2[:,i]=pn
        pd2[:,i]=pd

    for i in range(0,len(images)):
        fig, (ax1, ax2,ax3) = plt.subplots(1, 3,figsize=(15,15))
        ax1.imshow(c2[:,:,i],'gray')
        ax1.set_title('Clean image')
        ax2.imshow(n2[:,:,i],'gray')
        ax2.set_title('Noisy image ')
        ax3.imshow(d2[:,:,i],'gray')
        ax3.set_title('Recon image')
        plt.pause(0.2)
    return d2,c2,n2,pn2,pd2






# problem setup


def make_signal(sig_type, n, num_signals, signal_opts=None):
    if signal_opts is None:
        signal_opts = {}

    if sig_type == 'piecewise_constant':
        sigs = make_piecewise_const_signal(n, num_signals, **signal_opts)
    elif sig_type == 'DCT-sparse':
        sigs = make_DCT_signal(n, num_signals, **signal_opts)
    elif sig_type == 'constant_patch':
        sigs = make_constant_patch_signal(n, num_signals, **signal_opts)
    else:
        raise ValueError(sig_type)
    return sigs


def make_constant_patch_signal(n, num_signals, num_jumps=None):
    """
    separable sum of piecewise constant signals
    reshaped into vectors
    """
    m = math.isqrt(n)
    assert m**2 == n

    if num_jumps is None:
        num_jumps = m/4

    sigs_a = make_piecewise_const_signal(m, num_signals, num_jumps)
    sigs_b = make_piecewise_const_signal(m, num_signals, num_jumps)

    sigs = sigs_a[:, np.newaxis, :] + sigs_b[np.newaxis, :, :]

    return sigs.reshape(n, num_signals)/2  # so the max is 1


def make_piecewise_const_signal(n, num_signals=1, num_jumps=None):
    """
    piecewise constant signals in the range [0, 1)
    each piece height is chosen uniformly at random

    num_jumps - expected number of jumps
    """
    if num_jumps is None:
        jump_freq = 0.1
    else:
        jump_freq = num_jumps / n


    jumps = np.random.rand(n, num_signals) <= jump_freq
    inds = np.cumsum(jumps, axis=0)-1
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





def minibatcher(N, batch_size):
    """
    gives you the numbers 0, 1, 2, ..., N-1
    in random groups of size batch_size
    without replacement

    usesage
    shuffler = minibatcher(100, 3)

    while something:
        try:
            batch_inds = next(shuffler)
        except StopIteration:
            shuffler = minibatcher(100, 3)
    """
    inds = list(range(N))
    random.shuffle(inds)
    for batch_ind in range(N // batch_size):
        yield inds[batch_ind*batch_size:(batch_ind+1)*batch_size]


def do_learning(A, W0, train, eval_upper_fcn,
                learning_rate, num_steps, batch_size, print_interval=1,
                checkpoint_dir='checkpoints',
                checkpoint_frequency=None):
    """


    train : NamedTupe with fields x and y
    eval_upper_fcn : computes value and gradient of upper-level optimization problem

    checkpoint_frequency: save results every X seconds
    """

    outpath = pathlib.Path(checkpoint_dir, 'W_' + shortuuid.uuid())
    if checkpoint_frequency is not None:
        print(f'Saving to {outpath}')
        time_last_save = time.time()

    x, y = train
    train_length = x.shape[1]

    W = W0.copy()

    # setup main loop
    MSE_history = torch.full((train_length,), np.nan)
    shuffler = minibatcher(train_length, batch_size)
    epoch = 0

    print(f'{"step":12s}{"epoch":6s}'
          f'{"cur loss":15s}{"avg loss":15s}')

    # main loop
    for step in range(num_steps):
        try:
            batch_indices = next(shuffler)
        except StopIteration:
            shuffler = minibatcher(train_length, batch_size)
            epoch += 1

        # compute batch gradient
        grad = np.zeros_like(W)
        for idx in batch_indices:
            x_cur = x[:, [idx]]
            y_cur = y[:, [idx]]

            loss_cur, grad_cur = eval_upper_fcn(A, y_cur, W, x_cur,
                                                requires_grad=True)
            MSE_history[idx] = loss_cur
            grad += grad_cur / batch_size

        # gradient step
        W -= learning_rate * grad

        # print status line
        if step % print_interval == 0 or step == num_steps-1:
            epoch_loss = np.nanmean(MSE_history)
            print(f'{step:<12d}{epoch:<6d}'
                  f'{loss_cur:<15.3e}{epoch_loss:<15.3e}')

            #if logger is not None:
            #    logger.log_scalar('train.loss', last_loss, step)

        # save current W
        if (

                checkpoint_frequency is not None and
                time.time() - time_last_save > checkpoint_frequency
        ):
            np.save(pathlib.Path(str(outpath) + f'_{step}'), W)
            time_last_save = time.time()

    # save the final W
    np.save(outpath, W)
    print(f'Saved to {outpath}')

    return np.array(W)



def denoise_with_W(noise_sigma,W,beta,image_file):
    patch_size=int(np.sqrt(W.shape[1]))
    x,y,origin,image,noise_img=image2patchset(noise_sigma,patch_size,image_file)
    #beta_W = 1.0
    denoised=np.zeros((patch_size**2,x.shape[1]))
    batch_size=64  ## Solving lasso on all the patches faces memory issue, so solve lasso by batches
    A = make_forward_model('identity', patch_size**2)
    for i in range(0,x.shape[1],batch_size):
        denoised[:,i:i+batch_size]=opt.solve_lasso(A, y[:,i:i+batch_size], beta, W)
    denoised_image=patchset2image(denoised,origin)
    psnr_n = PSNR(image,noise_img)
    psnr_d = PSNR(image,denoised_image)
    return denoised_image,image,noise_img,psnr_n,psnr_d


def PSNR(original, compressed):
    mse = np.mean((original - compressed) ** 2)
    if(mse == 0):
        return 100
    max_pixel = 1.0
    psnr = 20 * np.log10(max_pixel / sqrt(mse))
    return psnr





# utilities
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
        cost_zero = opt.eval_lasso(A, np.zeros_like(x_GT), y, 0.0, W)
        data_GT = opt.eval_lasso(A, x_GT, y, 0.0, W)
        reg_GT = opt.eval_lasso(np.zeros_like(A), x_GT, np.zeros_like(y), 1.0, W)

        upper = (cost_zero - data_GT) / reg_GT

        upper = float(max(upper, lower))

    prob = opt.make_cvxpy_problem(A, W.shape[0])

    def J(beta):
        x_star = opt.solve_lasso(A, y, beta, W, prob=prob)
        return opt.MSE(x_star, x_GT)

    a, b = min_golden(J, lower, upper)
    val = (a+b)/2
    if np.abs(val-lower)/upper < 1e-2 or np.abs(val-upper)/upper < 1e-2:
        print(f'warning, optimal beta ({val:.3e}) '
              f'is close to one of the limits ({lower:.3e}, {upper:.3e})')

    return val


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


def make_forward_model(forward_model_type, n):
    if forward_model_type == 'identity':
        A = np.eye(n)
    else:
        raise ValueError(forward_model_type)

    return A

# transforms ----------------------



def make_transform(transform_type, n, scale=1.0, **opts):
    if transform_type == 'identity':
        W = np.eye(n, n)
        W = W - W.mean(axis=1, keepdims=True)

    elif transform_type == 'TV':
        W = make_conv(np.array([1.0, -1.0]), n)

    elif transform_type == 'DCT':
        W = scipy.fft.dct(np.eye(n), axis=0, norm='ortho')

    elif transform_type == 'random':
        W = np.random.randn(opts['k'], n)

    elif transform_type == 'TV-2D':
        m = math.isqrt(n)
        assert m**2 == n  # n must be perfect square
        W_horizontal = make_conv(np.array([1.0, -1.0]), n)
        W_horizontal = np.delete(W_horizontal, slice(m-1, None, m), axis=0)

        h = np.zeros(m+1)
        h[0] = -1.0
        h[-1] = 1.0
        W_vertical = make_conv(h, n)

        W = np.concatenate((W_horizontal, W_vertical), axis=0)


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


def image2patchset(noise_sigma,patch_size,filename):
    img = imageio.imread(filename).sum(2)/3
    img=img/255
    noise_sigma=noise_sigma/255
    noise_img = img + np.random.normal(0,noise_sigma,[img.shape[0],img.shape[1]])
    p, origin = extract_grayscale_patches( img, (patch_size,patch_size), stride=(1,1))
    p1, origin1 = extract_grayscale_patches( noise_img, (patch_size,patch_size), stride=(1,1))
    x, y = (patch2vector(t) for t in (p, p1))
    #train = datasetconv(x=x,y=y)
    return x,y,origin,img,noise_img




def patchset2image(vector,origin):
    patch=vector2patch(vector)
    denoised_image,wgt=reconstruct_from_grayscale_patches( patch, origin, epsilon=1e-12 )
    return denoised_image



def patch2vector(p):
    clean=np.zeros((p.shape[1]*p.shape[1],p.shape[0]))
    for i in np.arange(0,p.shape[0],1):
        clean[:,i] = np.reshape(p[i,:,:],[p.shape[1]*p.shape[1]])
    return clean


def datasetconv(x,y):
    return Dataset(x=x,y=y)

def vector2patch(p):
    denoised=np.zeros((p.shape[1],int(np.sqrt(p.shape[0])),int(np.sqrt(p.shape[0]))))
    for i in np.arange(0,p.shape[1],1):
        denoised[i,:,:] = np.reshape(p[:,i],[int(np.sqrt(p.shape[0])),int(np.sqrt(p.shape[0]))])
    return denoised

###Image to patches and vice versa     [Code source :http://jamesgregson.ca/extract-image-patches-in-python.html]
def extract_grayscale_patches( img, shape, offset=(0,0), stride=(1,1) ):
    """Extracts (typically) overlapping regular patches from a grayscale image

    Changing the offset and stride parameters will result in images
    reconstructed by reconstruct_from_grayscale_patches having different
    dimensions! Callers should pad and unpad as necessary!

    Args:
        img (HxW ndarray): input image from which to extract patches

        shape (2-element arraylike): shape of that patches as (h,w)

        offset (2-element arraylike): offset of the initial point as (y,x)

        stride (2-element arraylike): vertical and horizontal strides

    Returns:
        patches (ndarray): output image patches as (N,shape[0],shape[1]) array

        origin (2-tuple): array of top and array of left coordinates
    """
    px, py = np.meshgrid( np.arange(shape[1]),np.arange(shape[0]))
    l, t = np.meshgrid(
        np.arange(offset[1],img.shape[1]-shape[1]+1,stride[1]),
        np.arange(offset[0],img.shape[0]-shape[0]+1,stride[0]) )
    l = l.ravel()
    t = t.ravel()
    x = np.tile( px[None,:,:], (t.size,1,1)) + np.tile( l[:,None,None], (1,shape[0],shape[1]))
    y = np.tile( py[None,:,:], (t.size,1,1)) + np.tile( t[:,None,None], (1,shape[0],shape[1]))
    return img[y.ravel(),x.ravel()].reshape((t.size,shape[0],shape[1])), (t,l)



def reconstruct_from_grayscale_patches( patches, origin, epsilon=1e-12 ):
    """Rebuild an image from a set of patches by averaging

    The reconstructed image will have different dimensions than the
    original image if the strides and offsets of the patches were changed
    from the defaults!

    Args:
        patches (ndarray): input patches as (N,patch_height,patch_width) array

        origin (2-tuple): top and left coordinates of each patch

        epsilon (scalar): regularization term for averaging when patches
            some image pixels are not covered by any patch

    Returns:
        image (ndarray): output image reconstructed from patches of
            size ( max(origin[0])+patches.shape[1], max(origin[1])+patches.shape[2])

        weight (ndarray): output weight matrix consisting of the count
            of patches covering each pixel
    """
    patch_width  = patches.shape[2]
    patch_height = patches.shape[1]
    img_width    = np.max( origin[1] ) + patch_width
    img_height   = np.max( origin[0] ) + patch_height

    out = np.zeros( (img_height,img_width) )
    wgt = np.zeros( (img_height,img_width) )
    for i in range(patch_height):
        for j in range(patch_width):
            out[origin[0]+i,origin[1]+j] += patches[:,i,j]
            wgt[origin[0]+i,origin[1]+j] += 1.0

    return out/np.maximum( wgt, epsilon ), wgt
