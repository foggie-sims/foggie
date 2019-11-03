import matplotlib.pyplot as plt
import numpy as np
from astropy.table import Table
import matplotlib as mpl
mpl.rcParams['font.family'] = 'stixgeneral'
from foggie.mocky_way.core_funcs import calc_mean_median_3sig_2sig_1sig
fontsize = 16

ion_tag = 'SiIV'
ymin, ymax = 10, 13

sim_name = 'nref11n_nref10f'
dd_name = 'DD2175'

figname = 'figs/Nr_inview/%s_%s_N%s_in_ex_view.pdf'%(sim_name, dd_name, ion_tag)

dr = 10
rbins = np.mgrid[5:161:dr]

#### first calculate distribution from inside out view
fitsname = 'figs/Nr_inview/fits/%s_%s_N%s_inview.fits'%(sim_name, dd_name, ion_tag)
tb = Table.read(fitsname, format='fits')
# restrict to |b|>20 sightlines
tb = tb[np.abs(tb['b'])>=20]

N_mean = np.zeros(rbins.size)
N_median = np.zeros(rbins.size)
N_3sig_up = np.zeros(rbins.size)
N_3sig_low = np.zeros(rbins.size)
N_2sig_up = np.zeros(rbins.size)
N_2sig_low = np.zeros(rbins.size)
N_1sig_up = np.zeros(rbins.size)
N_1sig_low = np.zeros(rbins.size)

for ir in range(rbins.size):
    rin = rbins[ir]
    rout = rin+dr
    indr = np.all([tb['r']>=rin, tb['r']<rout], axis=0)
    sub_N = tb['N'][indr]

    data_stat = calc_mean_median_3sig_2sig_1sig(sub_N)
    N_mean[ir] = data_stat['mean']
    N_median[ir] = data_stat['median']
    N_3sig_up[ir] = data_stat['3sig_up']
    N_3sig_low[ir] = data_stat['3sig_low']
    N_2sig_up[ir] = data_stat['2sig_up']
    N_2sig_low[ir] = data_stat['2sig_low']
    N_1sig_up[ir] = data_stat['1sig_up']
    N_1sig_low[ir] = data_stat['1sig_low']

Nin_mean = np.log10(N_mean)
Nin_median = np.log10(N_median)
Nin_3sig_up = np.log10(N_3sig_up)
Nin_3sig_low = np.log10(N_3sig_low)
Nin_2sig_up = np.log10(N_2sig_up)
Nin_2sig_low = np.log10(N_2sig_low)
Nin_1sig_up = np.log10(N_1sig_up)
Nin_1sig_low = np.log10(N_1sig_low)

### then calculate distribution from external view
fitsname = 'figs/Nr_exview/fits/%s_%s_N%s_exview.fits'%(sim_name, dd_name, ion_tag)
tb = Table.read(fitsname, format='fits')

N_mean = np.zeros(rbins.size)
N_median = np.zeros(rbins.size)
N_3sig_up = np.zeros(rbins.size)
N_3sig_low = np.zeros(rbins.size)
N_2sig_up = np.zeros(rbins.size)
N_2sig_low = np.zeros(rbins.size)
N_1sig_up = np.zeros(rbins.size)
N_1sig_low = np.zeros(rbins.size)

for ir in range(rbins.size):
    rin = rbins[ir]
    rout = rin+dr
    indr = np.all([tb['impact_para']>=rin, tb['impact_para']<rout], axis=0)
    sub_N = tb['Nion'][indr]

    data_stat = calc_mean_median_3sig_2sig_1sig(sub_N)
    N_mean[ir] = data_stat['mean']
    N_median[ir] = data_stat['median']
    N_3sig_up[ir] = data_stat['3sig_up']
    N_3sig_low[ir] = data_stat['3sig_low']
    N_2sig_up[ir] = data_stat['2sig_up']
    N_2sig_low[ir] = data_stat['2sig_low']
    N_1sig_up[ir] = data_stat['1sig_up']
    N_1sig_low[ir] = data_stat['1sig_low']

Nex_mean = np.log10(N_mean)
Nex_median = np.log10(N_median)
Nex_3sig_up = np.log10(N_3sig_up)
Nex_3sig_low = np.log10(N_3sig_low)
Nex_2sig_up = np.log10(N_2sig_up)
Nex_2sig_low = np.log10(N_2sig_low)
Nex_1sig_up = np.log10(N_1sig_up)
Nex_1sig_low = np.log10(N_1sig_low)

### now plot things ####

fig = plt.figure(figsize=(10, 5.5))
ax1 = fig.add_subplot(121)
ax1.fill_between(rbins, Nin_3sig_up, Nin_3sig_low, edgecolor=None,
                facecolor='k', alpha=0.1, label=None)
ax1.fill_between(rbins, Nin_2sig_up, Nin_2sig_low, edgecolor=None,
                facecolor='k', alpha=0.15, label=None)
ax1.fill_between(rbins, Nin_1sig_up, Nin_1sig_low, edgecolor=None,
                facecolor='k', alpha=0.25, label=None)
ax1.plot(rbins, Nin_mean, color=plt.cm.Reds(0.6), lw=3, ls='-')
ax1.plot(rbins, Nin_median, color='k', lw=0.8, ls='--')
ax1.set_xlabel('r from LSR observers (kpc)', fontsize=fontsize)
ax1.set_ylabel(r'log [N(< r) (cm$^{-2}$)]', fontsize=fontsize)
ax1.set_title(r'%s (inside-out view)'%(ion_tag), fontsize=fontsize)


ax2 = fig.add_subplot(122)
ax2.fill_between(rbins, Nex_3sig_up, Nex_3sig_low, edgecolor=None,
                 facecolor='k', alpha=0.1, label=None)
ax2.fill_between(rbins, Nex_2sig_up, Nex_2sig_low, edgecolor=None,
                facecolor='k', alpha=0.15, label=None)
ax2.fill_between(rbins, Nex_1sig_up, Nex_1sig_low, edgecolor=None,
                facecolor='k', alpha=0.25, label=None)
ax2.plot(rbins, Nex_mean, color=plt.cm.Reds(0.6), lw=3, ls='-', label='Mean')
ax2.plot(rbins, Nex_median, color='k', lw=0.8, ls='--', label='Median')
ax2.set_xlabel('Impact Paramter (kpc)', fontsize=fontsize)
ax2.set_ylabel(r'log [N(r) (cm$^{-2}$)]', fontsize=fontsize)
ax2.set_title(r'%s (external view)'%(ion_tag), fontsize=fontsize)
ax2.legend(fontsize=fontsize-2)

for ax in [ax1, ax2]:
    # ax.set_yscale('log')
    # ax.set_xscale('log')
    ax.minorticks_on()
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(fontsize-2)
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(fontsize-2)
    ax.set_xlim(0, 160)
    ax.set_ylim(ymin, ymax) # in unit of cm-2
    ax.grid(linestyle='--', color=plt.cm.Greys(0.5), alpha=0.5)
fig.tight_layout()
fig.savefig(figname)
print(figname)
# print(figname)
# plt.close()
