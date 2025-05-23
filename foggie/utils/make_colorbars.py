'''

Makes standalone colorbars for density, temperature, metallicity,
HI,  SiII, SiIII, SiIV, OVI, CIV. Useful for talks and stuff.

'''

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid", {'axes.grid' : False})
mpl.rcParams['font.family'] = 'serif'
from consistency import *


fig = plt.figure(figsize=(2.1,12))
ax = fig.add_axes([0.05, 0.01, 0.36, 0.98])
norm = mpl.colors.Normalize(vmin=np.log10(density_slc_min), vmax=np.log10(density_slc_max))
cb = mpl.colorbar.ColorbarBase(ax, cmap=density_color_map,
                                norm=norm,
                                orientation='vertical',
                                extend='both')
cb.ax.tick_params(labelsize=20.)
cb.set_label(r'log density [M$_{\odot}$/pc$^{-3}$]',fontsize=32.)
plt.savefig('density_slice_colorbar.png')
plt.savefig('density_slice_colorbar.pdf')

fig = plt.figure(figsize=(2.1,12))
ax = fig.add_axes([0.05, 0.01, 0.36, 0.98])
norm = mpl.colors.Normalize(vmin=np.log10(temperature_min), vmax=np.log10(temperature_max))
cb = mpl.colorbar.ColorbarBase(ax, cmap=temperature_color_map,
                                norm=norm,
                                orientation='vertical',
                                extend='both')
cb.ax.tick_params(labelsize=20.)
cb.set_label(r'log temperature [K]',fontsize=32.)
plt.savefig('temperature_colorbar.png')
plt.savefig('temperature_colorbar.pdf')

fig = plt.figure(figsize=(2.1,12))
ax = fig.add_axes([0.05, 0.01, 0.36, 0.98])
norm = mpl.colors.Normalize(vmin=np.log10(metal_min), vmax=np.log10(metal_max))
cb = mpl.colorbar.ColorbarBase(ax, cmap=metal_color_map,
                                norm=norm,
                                orientation='vertical',
                                extend='both')
cb.ax.tick_params(labelsize=20.)
cb.set_label(r'log metallicity [Z$_{\odot}$]',fontsize=32.)
plt.savefig('metallicity_colorbar.png')
plt.savefig('metallicity_colorbar.pdf')

fig = plt.figure(figsize=(1.5,4))
ax = fig.add_axes([0.05, 0.025, 0.4, 0.95])
norm = mpl.colors.Normalize(vmin=np.log10(h1_proj_min), vmax=np.log10(h1_proj_max))
cb = mpl.colorbar.ColorbarBase(ax, cmap=h1_color_map,
                                norm=norm,
                                orientation='vertical',
                                extend='both')
cb.ax.tick_params(labelsize=16.)
cb.set_label(r'log HI column density [cm$^{-2}$]',fontsize=20.)
plt.savefig('h1_colorbar_small.png')
plt.savefig('h1_colorbar_small.pdf')

fig = plt.figure(figsize=(1.5,4))
ax = fig.add_axes([0.05, 0.025, 0.4, 0.95])
norm = mpl.colors.Normalize(vmin=np.log10(si2_min), vmax=np.log10(si2_max))
cb = mpl.colorbar.ColorbarBase(ax, cmap=si2_color_map,
                                norm=norm,
                                orientation='vertical',
                                extend='both')
cb.ax.tick_params(labelsize=16.)
cb.set_label(r'log SiII column density [cm$^{-2}$]',fontsize=20.)
plt.savefig('si2_colorbar_small.png')
plt.savefig('si2_colorbar_small.pdf')

fig = plt.figure(figsize=(1.5,4))
ax = fig.add_axes([0.05, 0.025, 0.4, 0.95])
norm = mpl.colors.Normalize(vmin=np.log10(si3_min), vmax=np.log10(si3_max))
cb = mpl.colorbar.ColorbarBase(ax, cmap=si3_color_map,
                                norm=norm,
                                orientation='vertical',
                                extend='both')
cb.ax.tick_params(labelsize=16.)
cb.set_ticks([11, 12, 13, 14, 15])
cb.set_ticklabels(['11','12','13','14','15'])
cb.set_label(r'log SiIII column density [cm$^{-2}$]',fontsize=20.)
plt.savefig('si3_colorbar_small.png')
plt.savefig('si3_colorbar_small.pdf')

fig = plt.figure(figsize=(1.5,4))
ax = fig.add_axes([0.05, 0.025, 0.4, 0.95])
norm = mpl.colors.Normalize(vmin=np.log10(si4_min), vmax=np.log10(si4_max))
cb = mpl.colorbar.ColorbarBase(ax, cmap=si4_color_map,
                                norm=norm,
                                orientation='vertical',
                                extend='both')
cb.ax.tick_params(labelsize=16.)
cb.set_ticks([11, 12, 13, 14, 15])
cb.set_ticklabels(['11','12','13','14','15'])
cb.set_label(r'log SiIV column density [cm$^{-2}$]',fontsize=20.)
plt.savefig('si4_colorbar_small.png')
plt.savefig('si4_colorbar_small.pdf')

fig = plt.figure(figsize=(1.5,4))
ax = fig.add_axes([0.05, 0.025, 0.4, 0.95])
norm = mpl.colors.Normalize(vmin=np.log10(c4_min), vmax=np.log10(c4_max))
cb = mpl.colorbar.ColorbarBase(ax, cmap=c4_color_map,
                                norm=norm,
                                orientation='vertical',
                                extend='both')
cb.ax.tick_params(labelsize=16.)
cb.set_ticks([11, 12, 13, 14, 15])
cb.set_ticklabels(['11','12','13','14','15'])
cb.set_label(r'log CIV column density [cm$^{-2}$]',fontsize=20.)
plt.savefig('c4_colorbar_small.png')
plt.savefig('c4_colorbar_small.pdf')

fig = plt.figure(figsize=(1.5,4))
ax = fig.add_axes([0.05, 0.025, 0.4, 0.95])
norm = mpl.colors.Normalize(vmin=np.log10(o6_min), vmax=np.log10(o6_max))
cb = mpl.colorbar.ColorbarBase(ax, cmap=o6_color_map,
                                norm=norm,
                                orientation='vertical',
                                extend='both')
cb.ax.tick_params(labelsize=16.)
cb.set_ticks([11, 12, 13, 14, 15])
cb.set_ticklabels(['11','12','13','14','15'])
cb.set_label(r'log OVI column density [cm$^{-2}$]',fontsize=20.)
plt.savefig('o6_colorbar_small.png')
plt.savefig('o6_colorbar_small.pdf')

fig = plt.figure(figsize=(1.5,12))
ax = fig.add_axes([0.05, 0.1, 0.42, 0.86])
norm = mpl.colors.Normalize(vmin=np.log10(h1_proj_min), vmax=np.log10(h1_proj_max))
cb = mpl.colorbar.ColorbarBase(ax, cmap=hi_discrete_colormap,
                                norm=norm,
                                orientation='vertical',
                                extend='both')
cb.ax.tick_params(labelsize=20.)
cb.set_label(r'log HI column density [cm$^{-2}$]',fontsize=26.)
plt.savefig('h1_colorbar_discrete.png')
plt.savefig('h1_colorbar_discrete.pdf')
