import torch
import matplotlib.pyplot as plt

import learnreg


##Reconstruct using a TV matrix
##
n = 100 # signal length, W is n x n
sigma = 0.25
sign_thresh = 1e-18

# init
plt.close('all')
torch.manual_seed(0)  # make repeatable


# make TV
tv=torch.zeros(n)
tv[0]=1.0
tv[1]=-1.0
TV=learnreg.create_circulant(tv)

# make dataset
A = torch.eye(n,n)
x, y = learnreg.make_set(A, num_signals=1, sigma=sigma)

# find best beta for TV
beta_opt = learnreg.find_optimal_beta(TV, y, x, 1.0)

x_TV = learnreg.optimize(TV,y,beta_opt)

fig, ax = learnreg.plot_recon(x, y, x_TV)
ax.set_title('TV reconstruction')
fig.show()

##Get the sign patterns
z = TV @ x
[S0,Sp,s]=learnreg.find_sign_pattern(z, threshold=sign_thresh)  ##Thresholds off values less than threshold in z.



##Use the closed form expression
x_cl=learnreg.closed_form(S0, Sp, s, TV, beta_opt, y)
fig, ax = plt.subplots()
ax.plot(x_cl, label='x closed')
ax.plot(x_TV, label='x_TV')
ax.plot(x, label='x_GT')
ax.set_title('')
ax.legend()
fig.show()

for z, name in zip((x, x_TV, x_cl), ('GT', 'TV', 'closed')):
    loss = learnreg.compute_loss(z, y, beta_opt, TV)
    print(f'loss of {name} is {loss:.3f}')
