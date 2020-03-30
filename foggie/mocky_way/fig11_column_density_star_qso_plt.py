# 03/30/2020, now move this code to run on pleiades, otherwise it takes forever
#             to filter through pair sightlines within certain angular separation
#

import os
import sys
import astropy.units as u
import numpy as np
from astropy.table import Table
from astropy.coordinates import SkyCoord
import astropy.units as u

###### if used on pleiades, run this #####
#import mocky_way_modules  # read this in before reading in foggie and yt
#from mocky_way_modules import data_dir_sys_dir
#data_dir, sys_dir = data_dir_sys_dir()
#os.sys.path.insert(0, sys_dir)
#from mocky_way_modules import calc_mean_median_3sig_2sig_1sig

###### if used locally in foggie.mocky_way, then use this
sys_dir = '/Users/Yong/ForkedRepo/foggie/foggie/'
from foggie.mocky_way.core_funcs import calc_mean_median_3sig_2sig_1sig

max_pair_deg = 10 # define as the largest separation allowed for a close pair of qso-star sightlines
nlos = 50 # 1000

np.random.seed(1024)
sim_name = 'nref11n_nref10f'
dd_name = 'DD2175'
ion_list = ['HI', 'SiII', 'CII', 'SiIII', 'SiIV', 'CIV', 'NV',
            'OVI', 'NeVII', 'NeVIII', 'OVII', 'OVIII']
data_dir = '%s/mocky_way'%(sys_dir)
for ii, ion_tag in enumerate(ion_list):
    # fitsname = 'figs/Nr_inview/fits/%s_%s_N%s_inview.fits'%(sim_name, dd_name, ion_tag)
    # table includes  N, l, b, r
    # tb = Table.read(fitsname, format='fits')
    # restrict to |b|>20 sightlines
    # tb = tb[np.abs(tb['b'])>=20]

    # for [5, 15] kpc range
    rin = 5
    rout = 15

    star_fits = '%s/figs/Nr_inview/fits/%s_%s_N%s_inview_%d-%d.fits'%(data_dir, sim_name,
                                                            dd_name, ion_tag, rin, rout)
    star_tb = Table.read(star_fits, format='fits')
    print(star_fits, len(star_tb))
    star_tb = star_tb[np.abs(star_tb['b'])>=20]
    star_coord = SkyCoord(l=star_tb['l'], b=star_tb['b'],
                          unit=(u.deg, u.deg), frame='galactic')
    # star_N = star_tb['N']
    # star_l = star_tb

    # for [150, 160] range
    rin = 150
    rout = 160
    qso_fits = '%s/figs/Nr_inview/fits/%s_%s_N%s_inview_%d-%d.fits'%(data_dir, sim_name,
                                                dd_name, ion_tag, rin, rout)
    qso_tb = Table.read(qso_fits, format='fits')
    print(qso_fits, len(qso_tb))
    qso_tb = qso_tb[np.abs(qso_tb['b'])>=20]
    qso_coord = SkyCoord(l=qso_tb['l'], b=qso_tb['b'],
                          unit=(u.deg, u.deg), frame='galactic')
    # qso_N = qso_tb['N']

    ### let's use Monte Carlo to propagate the error
    offset_logN = np.zeros(nlos)
    offset_deg = np.zeros(nlos)

    for i in range(nlos):
        sep_star_qso = max_pair_deg +1
        while (sep_star_qso>max_pair_deg) is True:
            istar = np.random.randint(low=0, high=len(star_tb))
            iqso = np.random.randint(low=0, high=len(qso_tb))
            istar_coord = star_coord[istar]
            iqso_coord = qso_coord[iqso]
            sep_star_qso = istar_coord.separation(iqso_coord).deg

        istar_logN = np.log10(star_tb['N'][istar])
        iqso_logN = np.log10(qso_tb['N'][iqso])

        offset_logN[i] = iqso_logN - istar_logN
        offset_deg[i] = sep_star_qso

    # now save the data for this ion:
    import astropy.io.fits as fits
    c1 = fits.Column(name='offset_logN', array=offset_logN, format='D')
    c2 = fits.Column(name='offset_deg', array=offset_deg, format='D')
    all_cols = [c1, c2]
    t = fits.BinTableHDU.from_columns(all_cols)
    fig_dir = '%s/mocky_way/figs/Nr_inview/fits'%(sys_dir)
    tb_name = '%s_%s_quastar_logNoffset_maxpairdeg%d_%s.fits'%(sim_name, dd_name, max_pair_deg, ion_tag)
    save_to_file = '%s/%s'%(fig_dir, tb_name)
    # print("%s: I am saving it to %s"%(i, save_to_file))
    t.writeto(save_to_file, overwrite=True)
    print(tb_name)
    break

'''

ion_median = np.zeros(len(ion_list))
ion_mean = np.zeros(len(ion_list))
ion_1sig_up = np.zeros(len(ion_list))
ion_1sig_low = np.zeros(len(ion_list))

    # now get the mean and median and other stat
    data_stat = calc_mean_median_3sig_2sig_1sig(offset_logN)
    ion_mean[ii] = data_stat['mean']
    ion_median[ii] = data_stat['median']
    ion_1sig_up[ii] = data_stat['1sig_up']
    ion_1sig_low[ii] = data_stat['1sig_low']


#### plot ! ###
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['font.family'] = 'stixgeneral'

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
figname = 'figs/Nr_star_qso/fig_logN_star_qso_ion-eV_maxpairdeg%d.pdf'%(max_pair_deg)
fig.savefig(figname)

print(figname)
'''
