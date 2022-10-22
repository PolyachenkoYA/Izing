import numpy as np
import matplotlib.pyplot as plt
import os
import scipy
import sys
import scipy.interpolate

import mylib as my
import table_data

BF_data = np.array([[100, 64, 48, 32, 18], \
					[1.9636217582430445e-7, 1.470488459276814e-7, 2.303309502663671e-7, 2.040138418682597e-7, 2.3693171277405201e-7], \
					[2.921273220048962e-7, 1.765474815156926e-7, 2.3970434991833825e-7, 2.527982538321685e-7, 2.760771024843176e-7]])

CS_data = np.array([[100, 56, 24, 18], \
					[3.3355932203389845e-8, 3.4474576271186445e-8, 7.362711864406782e-8, 8.522033898305086e-8], \
					[3.9355932203389846e-8, 3.610169491525424e-8, 8.36949152542373e-8, 9.254237288135595e-8], \
					[0.0005905482041587901, 0.0005905482041587901, 0.0011860113421550095, 0.0015035916824196597], \
					[0.0007096408317580341, 0.0006283553875236294, 0.0013485822306238185, 0.0016623818525519848], \
					[0.00005654545454545454, 0.00005857640232108317, 0.00006219342359767891, 0.00005670019342359768], \
					[0.00006188394584139265, 0.0000615164410058027, 0.0000633733075435203, 0.000060239845261121855]])

M_data = np.array([[100, 50, 24], \
					[1.1476635514018679e-8, 6.8e-8, 9.790654205607477e-8], \
					[1.7906542056074768e-8, 9.760747663551401e-8, 1.0314018691588785e-7], \
					[0.000031111111111111096, 0.00013521367521367525, 0.00034162393162393164], \
					[0.000049059829059829095, 0.00019564102564102569, 0.0003625641025641026], \
					[0.0003687969924812029, 0.0005022556390977443, 0.0002872180451127819], \
					[0.00039812030075187963, 0.0005229323308270676, 0.000294360902255639]])

data = [CS_data, M_data, BF_data]
lbls = ['CS', 'M', 'BF']

N = len(data)
assert(N == len(lbls))

def add_data(ax, data, lbl, i, ax_P=None, ax_F=None):
	x = data[0, :]
	y = data[1, :]
	dy = data[2, :] - data[1, :]
	ax.errorbar(x, y, yerr=dy, label=lbl, fmt='.--')
	
	if(ax_P is not None):
		P = data[3, :]
		dP = data[4, :] - data[3, :]
		ax_P.errorbar(x, P, yerr=dP, label=lbl, fmt='.--')
	
	if(ax_F is not None):
		F = data[5, :]
		dF = data[6, :] - data[5, :]
		ax_F.errorbar(x, F, yerr=dF, label=lbl, fmt='.--')

fig, ax = my.get_fig('L', r'$k_{AB} / L^2$ [trans / (sweep $\cdot l^2$)]', title=r'$k_{AB}(L)$; T/J = 1.5, h/J=0.15', xscl='log', yscl='log')
fig_P, ax_P = my.get_fig('L', r'$P_{AB}$', title=r'$k_{AB}(L)$; T/J = 1.5, h/J=0.15', xscl='log', yscl='log')
fig_F, ax_F = my.get_fig('L', r'$\Phi_{A} / L^2$ [trans / (sweep $\cdot l^2$)]', title=r'$k_{AB}(L)$; T/J = 1.5, h/J=0.15', xscl='log', yscl='log')
for i in range(N):
	add_data(ax, data[i], lbls[i], i, ax_P = ax_P if(i < 2) else None, ax_F = ax_F if(i < 2) else None)

ax.legend()
ax_P.legend()
ax_F.legend()

plt.show()
