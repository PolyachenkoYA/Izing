import numpy as np
import matplotlib.pyplot as plt
import os

import mylib as my

Temp = 3.0

h = np.array([0, 0.1, 0.02, 0.05, 0.15, 0.015, 0.005, 0.025, 0.012, 0.011, 0.01])
m = np.array([0, 0.3, 5.438e-2, 0.1649, 0.4176, 0.06, 0.0078, 0.0819, 0.0411, 0.046, 0.033])
d_m = np.array([8e-5, 4e-5, 15e-5, 8e-5, 5e-5, 5e-5, 1e-4, 7e-5, 5e-5, 10e-5, 10e-5])

sort_inds = np.argsort(h)
h = h[sort_inds]
m = m[sort_inds]
d_m = d_m[sort_inds]

#linfit = np.polyfit(h, m, 1)
#xi = linfit[0]
chi = np.sum(h * m / d_m**2) / np.sum((h / d_m)**2)
linfit = np.array([chi, 0])

#fig,ax = my.get_fig('h', 'm', title='m(h); T = ' + str(Temp) + '; xi = ' + my.f2s(xi) + '', xscl='log', yscl='log')
fig,ax = my.get_fig('h', 'm', title='m(h); T/J = ' + str(Temp) + r'; $\partial m / \partial h = \chi = ' + my.f2s(chi) + '$')

ax.errorbar(h, m, yerr=d_m, ls='none', marker='o', label='data')
ax.plot(h, np.polyval(linfit, h), label=r'fit $m \sim \chi h$ ')
ax.legend()

plt.show()

