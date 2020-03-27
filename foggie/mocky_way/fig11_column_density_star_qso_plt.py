import matplotlib.pyplot as plt
import numpy as np
from astropy.table import Table
import matplotlib as mpl
mpl.rcParams['font.family'] = 'stixgeneral'
from foggie.mocky_way.core_funcs import calc_mean_median_3sig_2sig_1sig

sim_name = 'nref11n_nref10f'
dd_name = 'DD2175'
ion_list = ['HI', 'SiII', 'CII', 'SiIII', 'SiIV', 'CIV', 'NV',
            'OVI', 'NeVII', 'NeVIII', 'OVII', 'OVIII']
#ion_list = ['HI', 'SiII', 'SiIII', 'SiIV', 'CII', 'CIV', 'OVI', 'NV',
#            'OVII', 'OVIII', 'NeVII', 'NeVIII']
# low_ions = ['HI', 'SiII', 'SiIII', 'SiIV', 'CII']

ion_median = np.zeros(len(ion_list))
ion_mean = np.zeros(len(ion_list))
ion_1sig_up = np.zeros(len(ion_list))
ion_1sig_low = np.zeros(len(ion_list))

np.random.seed(1024)
nlos = 1000
for ii, ion_tag in enumerate(ion_list):
    fitsname = 'figs/Nr_inview/fits/%s_%s_N%s_inview.fits'%(sim_name, dd_name, ion_tag)
    tb = Table.read(fitsname, format='fits')
    # restrict to |b|>20 sightlines
    tb = tb[np.abs(tb['b'])>=20]

    # for [5, 15] kpc range
    rin = 5
    rout = 15
    star_fits = 'figs/Nr_inview/fits/%s_%s_N%s_inview_%d-%d.fits'%(sim_name,
                                                dd_name, ion_tag, rin, rout)
    star_tb = Table.read(star_fits, format='fits')
    print(star_fits, len(star_tb))
    star_tb = star_tb[np.abs(star_tb['b'])>=20]
    star_N = star_tb['N']

    # for [150, 160] range
    rin = 150
    rout = 160
    qso_fits = 'figs/Nr_inview/fits/%s_%s_N%s_inview_%d-%d.fits'%(sim_name,
                                                dd_name, ion_tag, rin, rout)
    qso_tb = Table.read(qso_fits, format='fits')
    print(qso_fits, len(qso_tb))
    qso_tb = qso_tb[np.abs(qso_tb['b'])>=20]
    qso_N = qso_tb['N']

    ### let's use Monte Carlo to propagate the error
    offset_logN = np.zeros(nlos)
    for i in range(nlos):
        istar = np.random.randint(low=0, high=star_N.size)
        iqso = np.random.randint(low=0, high=qso_N.size)

        istar_logN = np.log10(star_N[istar])
        iqso_logN = np.log10(qso_N[iqso])

        offset_logN[i] = iqso_logN - istar_logN

    # now get the mean and median and other stat
    data_stat = calc_mean_median_3sig_2sig_1sig(offset_logN)
    ion_mean[ii] = data_stat['mean']
    ion_median[ii] = data_stat['median']
    ion_1sig_up[ii] = data_stat['1sig_up']
    ion_1sig_low[ii] = data_stat['1sig_low']


#### plot ! ###
fig = plt.figure(figsize=(10, 4))
ax = fig.add_subplot(111)
x = np.arange(len(ion_list))
width = 0.5

fs = 14
ax.grid(linestyle='--')
ax.set_ylim(-1.0, 2.2)
ax.set_xlim(-width, x.size-width)
ax.hlines(0, -width, x.size-width, linestyle=':', lw=1)

for ii in range(x.size):
    xa, xb = x[ii]-width/2., x[ii]+width/2.
    ya, yb = ion_1sig_up[ii], ion_1sig_low[ii]

    if ii == 0:
        labela = r'$1\sigma$'
        labelb = 'Median'
    else:
        labela = None
        labelb = None

    this_ion = ion_list[ii]
    ax.fill_between([xa, xb], ya, yb, color=plt.cm.Reds(0.3), label=labela)
    ax.hlines(ion_median[ii], xa, xb, color=plt.cm.Blues(0.9), lw=3, label=labelb)
ax.set_xticks(x)
ax.set_xticklabels(ion_list)
ax.legend(fontsize=fs, loc='upper left')
for tick in ax.xaxis.get_major_ticks():
    tick.label.set_fontsize(fs)
for tick in ax.yaxis.get_major_ticks():
    tick.label.set_fontsize(fs)
ax.set_ylabel(r'$\delta$ logN (dex)', fontsize=fs+3)
ax.set_title('Differce of logN between QSO and halo star measurements', fontsize=fs+4)
fig.tight_layout()
figname = 'figs/Nr_star_qso/fig_logN_star_qso_ion-eV.pdf'
fig.savefig(figname)

print(figname)
