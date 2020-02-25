import matplotlib.pyplot as plt
from foggie.utils.consistency import *
import numpy as np
from numpy import *
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = ['serif'] 
plt.ioff()






'''
cmap1 = density_color_map
cmap2 =  plt.cm.Greys_r
zmin = density_proj_min
zmax = density_proj_max



fig, ax = plt.subplots(1,1, figsize = (10,10))


pl = ax.imshow(zeros((100,100)), cmap  = cmap1, vmin = density_proj_min, vmax = density_proj_max)

ax.axis('off')


cbar = fig.colorbar(pl, ax=ax, orientation = 'horizontal')

cbar.set_label('mass surface density (M$_{\odot}$ pc$^{-2}$)') 

fig.savefig('/Users/rsimons/Dropbox/foggie/figures/for_paper/central_projections/colorbars.png', dpi = 400)
'''

'''

Makes standalone colorbars for density, temperature, metallicity,
HI,  SiII, SiIII, SiIV, OVI, CIV. Useful for talks and stuff.

'''

import matplotlib as mpl


fs = 60
xtext = 0.08
ytext = 0.54
figsizex = 24
figsizey = 2.1
fig = plt.figure(figsize=(figsizex, figsizey))
ax = fig.add_axes([0.01, 0.50, 0.98, 0.36])
norm = mpl.colors.Normalize(vmin=np.log10(density_proj_min), vmax=np.log10(density_proj_max))
cb = mpl.colorbar.ColorbarBase(ax, cmap=density_color_map,
                                norm=norm,
                                orientation='horizontal',
                                extend='both')
#ax.annotate('gas', (xtext, ytext), ha = 'left', va = 'center', xycoords = 'axes fraction', color = 'white', fontsize = fs, fontweight = 'bold')
cb.ax.tick_params(labelsize=20.)
cb.ax.set_xticklabels([''])
#cb.set_label(r'$\log$ surface density [M$_{\odot}$ pc$^{-2}$]',fontsize=32.)
plt.savefig('gas_density_proj_colorbar_largelabels.png', dpi = 500)




fig = plt.figure(figsize=(figsizex, figsizey))
ax = fig.add_axes([0.01, 0.54, 0.98, 0.30])
norm = mpl.colors.Normalize(vmin=np.log10(density_proj_min), vmax=np.log10(density_proj_max))
cb = mpl.colorbar.ColorbarBase(ax, cmap=plt.cm.Greys_r,
                                norm=norm,
                                orientation='horizontal',
                                extend='both')
#ax.annotate('stars', (xtext, ytext), ha = 'left', va = 'center', xycoords = 'axes fraction', color = 'white', fontsize = fs, fontweight = 'bold')
cb.ax.tick_params(labelsize=20.)
cb.set_label(r'$\log$ surface density [M$_{\odot}$ pc$^{-2}$]',fontsize=50.)
fig.subplots_adjust(bottom = 0.40)
plt.savefig('star_density_proj_colorbar_largelabels.png.png', dpi = 500)











