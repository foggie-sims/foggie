import matplotlib.pyplot as plt
import numpy as np
import healpy as hp
import matplotlib as mpl
mpl.rcParams['font.family'] = 'stixgeneral'
from foggie.mocky_way.core_funcs import calc_mean_median_3sig_2sig_1sig

ion_list = ['HI', 'SiII', 'SiIII', 'SiIV', 'NV', 'CII', 'CIV',
            'OVI', 'OVII', 'OVIII', 'NeVII', 'NeVIII']
r_list = ['r0-10', 'r0-20', 'r0-30', 'r0-40', 'r0-50', 'r0-60',
          'r0-70', 'r0-80', 'r0-90', 'r0-100', 'r0-110', 'r0-120']
rbins = np.mgrid[10:121:10]

ion_tag = ion_list[11]

N_median = []
N_mean = []
N_3sig_up = []
N_3sig_low = []
N_2sig_up = []
N_2sig_low = []
N_1sig_up = []
N_1sig_low = []

for r_tag in r_list:
    ddir = 'figs/allsky_diff_ions/%s/fits'%(r_tag)
    fitsfile = '%s/nref11n_nref10f_DD2175_%s_halo_center_%s.fits'%(ddir, r_tag, ion_tag)
    N_ion = hp.read_map(fitsfile)
    nside = hp.get_nside(N_ion)
    lon, lat = hp.pix2ang(nside, np.arange(len(N_ion)), lonlat=True)
    ind_hb = np.any([lat<=-20, lat>=20], axis=0)
    N_ion = N_ion[ind_hb]

    data_stat = calc_mean_median_3sig_2sig_1sig(N_ion)
    N_mean.append(data_stat['mean'])
    N_median.append(data_stat['median'])
    N_3sig_up.append(data_stat['3sig_up'])
    N_3sig_low.append(data_stat['3sig_low'])
    N_2sig_up.append(data_stat['2sig_up'])
    N_2sig_low.append(data_stat['2sig_low'])
    N_1sig_up.append(data_stat['1sig_up'])
    N_1sig_low.append(data_stat['1sig_low'])

#### now plot things ####

fig = plt.figure(figsize=(5, 5))
ax = fig.add_subplot(111)
ax.fill_between(rbins, N_3sig_up, N_3sig_low, edgecolor=None,
                facecolor='k', alpha=0.1, label=None)
ax.fill_between(rbins, N_2sig_up, N_2sig_low, edgecolor=None,
                facecolor='k', alpha=0.15, label=None)
ax.fill_between(rbins, N_1sig_up, N_1sig_low, edgecolor=None,
                facecolor='k', alpha=0.25, label=None)
ax.plot(rbins, N_mean, color=plt.cm.Reds(0.6), lw=3, ls='--', label='Mean')
ax.plot(rbins, N_median, color='k', lw=0.8, ls='-', label='Median')

fontsize = 16
# ax.set_xlim(0, 10) # in unit of kpc
# ax.set_ylim(1e-10, 1e-4) # in unit of cm-3
ax.set_yscale('log')
ax.legend(fontsize=fontsize-2, loc='upper right')
ax.set_xlabel('r (kpc)', fontsize=fontsize)
ax.set_ylabel(r'N$_{\rm %s}$(r) (cm$^{-3}$)'%(ion_tag), fontsize=fontsize)
ax.minorticks_on()

for tick in ax.xaxis.get_major_ticks():
    tick.label.set_fontsize(fontsize-2)
for tick in ax.yaxis.get_major_ticks():
    tick.label.set_fontsize(fontsize-2)
ax.set_title(r'%s (|b|>20$\degree$)'%(ion_tag), fontsize=fontsize)
ax.set_xlim(0, 120)
# ax.set_ylim(1e-20, 1e-3)
fig.tight_layout()
ax.grid(linestyle='--', color=plt.cm.Greys(0.5), alpha=0.5)
figname = 'figs/nr_Nr/nref11n_nref10f_DD2175_all_coldens_%s_b20.pdf'%(ion_tag)
fig.savefig(figname)
print(figname)
plt.close()
