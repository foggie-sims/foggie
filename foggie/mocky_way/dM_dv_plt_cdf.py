#### plot dM-dv, changing all the times, no need to be fancy. YZ, UCB. ###

import numpy as np
import sys
from astropy.io import fits
from astropy.table import Table
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['font.family'] = 'stixgeneral'

sim_name = 'nref11n_nref10f'
dd_name = 'DD2175'
obs_point = 'halo_center'  # halo_center, offcenter_location
obj_tag = 'all' # halo_only, halo_and_disk, disk

fig_dir = 'figs/dM_dv/'
tb_name = '%s_%s_dMdv_%s_%s.fits'%(sim_name, dd_name, obj_tag, obs_point)
filename = '%s/%s'%(fig_dir, tb_name)

# get dM/dv
data = Table.read(filename, format='fits')
dv_bins = data['v (km/s)']
dM_all = data['dM (Msun/km/s)']
dM_cold = data['dM_cold (Msun/km/s)']
dM_cool = data['dM_cool (Msun/km/s)']
dM_warm = data['dM_warm (Msun/km/s)']
dM_hot = data['dM_hot (Msun/km/s)']

## for sanity check reason
print('%s, %s...'%(obj_tag, obs_point))
dv = dv_bins[1]-dv_bins[0]
print('M all : %.2e Msun'%(dM_all.sum()*dv))
print('M cold: %.2e Msun'%(dM_cold.sum()*dv))
print('M cool: %.2e Msun'%(dM_cool.sum()*dv))
print('M warm: %.2e Msun'%(dM_warm.sum()*dv))
print('M hot : %.2e Msun'%(dM_hot.sum()*dv))

# calculate the cumulative distribution functions for diff components
cdf_all = np.zeros(dv_bins.size)
cdf_cold = np.zeros(dv_bins.size)
cdf_cool = np.zeros(dv_bins.size)
cdf_warm = np.zeros(dv_bins.size)
cdf_hot = np.zeros(dv_bins.size)
for i in range(dv_bins.size):
    cdf_all[i] = dM_all[:i+1].sum()/dM_all.sum()
    cdf_cold[i] = dM_cold[:i+1].sum()/dM_cold.sum()
    cdf_cool[i] = dM_cool[:i+1].sum()/dM_cool.sum()
    cdf_warm[i] = dM_warm[:i+1].sum()/dM_warm.sum()
    cdf_hot[i] = dM_hot[:i+1].sum()/dM_hot.sum()

### now let's plot all the stuff
print("Plotting data from %s..."%(filename))

fs=14
from foggie.utils import consistency
cmap = consistency.temperature_discrete_cmap
c_all = plt.cm.Greys(0.7)
c_cold = cmap(0.05)
c_cool = cmap(0.25)
c_warm = cmap(0.6)
c_hot = cmap(0.9)
lwa = 1.5
lwb = 2.5

fig = plt.figure(figsize=(3.5, 5))
ax1 = fig.add_axes([0.18, 0.40, 0.75, 0.55])
ax2 = fig.add_axes([0.18, 0.11, 0.75, 0.27])
for ax in [ax1, ax2]:
    ax.minorticks_on()
    ax.grid('on', linestyle=':')
    ax.set_xlim(-400, 400)
    ax.set_xticks([-400, -200, 0, 200, 400])
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(fs-2)
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(fs-2)

# first plot dM-dv
ax1.step(dv_bins, dM_cold, color=c_cold, label='cold', lw=lwb)
ax1.step(dv_bins, dM_cool, color=c_cool, label='cool', lw=lwb)
ax1.step(dv_bins, dM_warm, color=c_warm, label='warm', lw=lwb)
ax1.step(dv_bins, dM_hot, color=c_hot, label='hot', lw=lwb)
ax1.step(dv_bins, dM_all, color=c_all, label='All', lw=lwa)

ax1.legend(fontsize=fs-4)
ax1.set_ylim(1e4, 8e8)
ax1.set_yscale("log")
ax1.set_ylabel("dM/dv (Msun per km/s)", fontsize=fs)
ax1.set_xticklabels([])

# then plot CDF
ax2.step(dv_bins, cdf_cold, color=c_cold, label='cold', lw=lwb)
ax2.step(dv_bins, cdf_cool, color=c_cool, label='cool', lw=lwb)
ax2.step(dv_bins, cdf_warm, color=c_warm, label='warm', lw=lwb)
ax2.step(dv_bins, cdf_hot, color=c_hot, label='hot', lw=lwb)
ax2.step(dv_bins, cdf_all, color=c_all, label='All', lw=lwa)
ax2.set_xlabel("Velocity (km/s)", fontsize=fs)
ax2.set_ylabel('M(<v)/M', fontsize=fs)

save_to_fig = '%s.pdf'%(filename[:-5])
fig.savefig(save_to_fig)
print("Hey, I saved the figure to ", save_to_fig)
