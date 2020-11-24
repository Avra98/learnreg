import numpy as np
import torch
import cvxpy as cp
import matplotlib.pyplot as plt
import collections

import learnreg


# parameters

sigma=0.08   ##Noise level
m=100 ##Number of elements in vector
rg=torch.randn(m)  ## Random kernel initialisation. Try scaling this if ECOS solver failed.

steps=250   ##Steps of iteration
opti_opts=('LBFGS', dict(lr=0.2))  ## Specify optimizer type and learning rate. Decrease lr if ECOS fails.

plots_while_training = False

# setup

opti = learnreg.make_opti(opti_opts, rg)
# use this list to get losses out of the closure
last_loss = [None]  #weighted combination of fit and l1 loss
fit = [None]   ## loss of xreconstructed to ground truth
l1 = [None]     ##l1 norm of the kernel
l2 = [None]     ## To check if the rows of dictionary are normalized. This is an enforced constraint.

##Generate a pair of clean and noisy signal
A = torch.eye(m,m)  # for now, denoising
x1, y1 = learnreg.make_set(A, num_signals=1, sigma=sigma)

##The dictionary is constrained to be circulant and each rows is contrained to have l2 norm=1.
##Also dictionary rows assumed to have a prior of minimum l1 norm which is added as a penalty constraint
for out_step in range(steps):
    ##Define the circulant matrix outside the closure function so as to plot later
    rg.requires_grad_(False)
    Wg = learnreg.create_circulant(rg)
    ##Define closure function
    def closure():
        ##Make circulant
        rg.requires_grad_(False)
        Wg = learnreg.create_circulant(rg)
        ##Do a parameter search of penalty parameter to get the best signs (may slow the iteration)
        def f(bet):
            xopt = learnreg.optimize(Wg, y1, bet)
            return learnreg.MSE(x1,xopt)
        [c1,d1]=learnreg.min_golden(f, 1e-2, 1, tol=1e-5)  ##beta assumed to lie in (1e-2,1)
        bpot=(c1+d1)/2   ##Optimum beta obtained
        xopt = learnreg.optimize(Wg, y1, bpot)   ##Find the denoised recovered x using a randomly initialised dictionary
        z= Wg @ xopt         ## Find the sparse rerpresentation of the recovered signal in the random dictionary
        [S0,Sp,s]= learnreg.find_sign_pattern(z, threshold=1e-6)  ##Finding sign pattern of the sporse vector. The threshold parameter is sensitive. 1e-6 seems like an optimum value.
        opti.zero_grad()  ##Taking the gradeint zero before updating the kernel for a new iteration


        rg.requires_grad_(True)
        Wg=learnreg.create_circulant(rg)
        X_close = learnreg.closed_form(S0, Sp, s, Wg, bpot, y1)  #closed form solution predicted from sign patterns and variable dictioanry
        rg.requires_grad_(False)
        lamb= learnreg.MSE(x1,X_close)/torch.norm(rg/torch.norm(rg,2),1)  ##Weight associated in front of the l1 loss. This is done to ensure that the l1 loss does not dominate the total loss.
        rg.requires_grad_(True)
        loss = learnreg.MSE(x1,X_close) + lamb*torch.norm(rg/torch.norm(rg,2),1)  ##Minimise l1 loss of l2 normalised kernel + the fit. Scale lamb if you want to have more spikes in kernel
        loss.backward()
        rg.requires_grad_(False)
        last_loss[0] = loss.item()
        fit[0] = learnreg.MSE(x1,X_close).item()
        l1[0] = torch.norm(rg/torch.norm(rg,2),1).item()
        l2[0] = torch.norm(Wg[0,:],2).item()   ##Just to check if dictionary has normalised rows. Will always be 1 .
        return loss
    opti.step(closure)
    rg.requires_grad_(False)
    print('Steps: {:d}|Loss: {:.10f}| fit : {:.10f}|l1 : {:.10f}|l2 : {:.10f} '.format(out_step,last_loss[0],fit[0],l1[0],l2[0]))

    if plots_while_training:
        Dv1=Wg.cpu().detach().numpy()
        Dv=learnreg.permute_for_display(Dv1)
        rg1=rg.detach().numpy()
        plt.plot(Dv[0,:])
        plt.show()
