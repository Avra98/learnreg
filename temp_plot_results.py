import numpy as np
import matplotlib.pyplot as plt

import learnreg as lr

plt.close('all')

#lr.reports.make_plots(45, prefix='learned (random init)')

#lr.reports.make_plots(40, prefix='DCT')
#lr.reports.make_plots(43, prefix='learnreg')

#lr.reports.make_plots(39, prefix='TV')
W_baseline = lr.reports.make_plots(8, prefix='baseline')
W = lr.reports.make_plots(7, prefix='learnreg, identity init')
W = lr.reports.make_plots(5, prefix='learnreg, random init')

fig, ax = lr.reports.show_W(W_baseline, lr.reports.rearrange_to_baseline(W, W_baseline))
fig.show()

W_baseline = lr.reports.make_plots(9, prefix='baseline')
W = lr.reports.make_plots(6, prefix='learnreg, identity init')
fig, ax = lr.reports.show_W(W_baseline, lr.reports.rearrange_to_baseline(W, W_baseline))
fig.show()



#W = lr.reports.make_plots(6, prefix='learnreg, identity init')

#fig, ax = plt.subplots()
#ax.imshow(np.abs(np.fft.fft(W, axis=1)))
#fig.show()
