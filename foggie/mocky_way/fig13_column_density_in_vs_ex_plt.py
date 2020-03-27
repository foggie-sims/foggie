import matplotlib.pyplot as plt
import numpy as np
from astropy.table import Table
import matplotlib as mpl
mpl.rcParams['font.family'] = 'stixgeneral'
from foggie.mocky_way.core_funcs import calc_mean_median_3sig_2sig_1sig
fontsize = 16

#ion_tag = 'CII'
#ymin, ymax = 10, 18
ion_tag = 'OVI'
ymin, ymax = 11.5, 15.5

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

fig = plt.figure(figsize=(9, 4.5))
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
ax1.set_xticks(np.mgrid[0:160:20])
if ion_tag == 'OVI':
    ax1.set_xlim(0, 185)
else:
    ax1.set_xlim(0, 160)


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
ax2.legend(fontsize=fontsize, loc='lower left')
ax2.set_xlim(0, 160)

for ax in [ax1, ax2]:
    # ax.set_yscale('log')
    # ax.set_xscale('log')
    ax.minorticks_on()
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(fontsize-2)
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(fontsize-2)
    ax.set_ylim(ymin, ymax) # in unit of cm-2
    ax.grid(linestyle='--', color=plt.cm.Greys(0.5), alpha=0.5)

#### for OVI ####
if ion_tag == 'OVI':
    #### inside-out view, add Savage+2003 (QSO, low v), Savage+2009 (star, low v)
    #### and Sembach+2003 (QSO, high v) data
    low_star = Table.read('/Users/Yong/Dropbox/databucket/savage09_table2.txt', format='ascii')
    low_star = low_star[low_star['SQ'] == 'S']
    c_low_star = plt.cm.Purples(0.8)
    m_low_star = 'o'
    l_low_star = 'Savage+2009, Star, low vel'
    s_low_star = 30

    low_qso = Table.read('/Users/Yong/Dropbox/databucket/savage03_table2.fits', format='fits')
    c_low_qso = plt.cm.Greens(0.6)
    m_low_qso = '+'
    l_low_qso = 'Savage+2003, QSO, low vel'
    s_low_qso = 25

    high_qso = Table.read('/Users/Yong/Dropbox/databucket/Sembach03_OVI_tb1.txt', format='ascii')
    c_high_qso = plt.cm.Greys(0.7)
    m_high_qso = '^'
    l_high_qso = 'Sembach+03, QSO, high vel'
    s_high_qso = 20

    # halo star, Savage+2003, |vlsr|<~ 100 km/s
    ax1.scatter(low_star['d(kpc)'], low_star['logNO6'],
                s=s_low_star, edgecolor=c_low_star, facecolor='none',
                marker=m_low_star,
                label=l_low_star)
    #ind_good = np.abs(low_star['elogNO6']) < 999
    #ind_up = low_star['elogNO6'] == 999
    #uplims = [0.1]*len(low_star[ind_up])
    #ind_low = low_star['elogNO6'] == -999
    #lolims = [0.1]*len(low_star[ind_low])
    #ax1.errorbar(low_star['d(kpc)'][ind_good], low_star['logNO6'][ind_good],
    #             yerr=low_star['elogNO6'][ind_good],
    #             fmt=m_low_star, markersize=s_low_star,
    #             color=c_low_star, label=l_low_star)
    #ax1.errorbar(low_star['d(kpc)'][ind_up], low_star['logNO6'][ind_up],
    #             yerr=uplims, uplims=uplims,
    #             fmt=m_low_star, markersize=s_low_star,
    #             color=c_low_star, label=None)
    #ax1.errorbar(low_star['d(kpc)'][ind_low], low_star['logNO6'][ind_low],
    #             yerr=lolims, lolims=lolims,
    #             fmt=m_low_star, markersize=s_low_star,
    #             color=c_low_star, label=None)

    # qso, Savage+2009, |vlsr|<~ 100 km/s
    dd = np.random.uniform(low=162, high=172, size=len(low_qso))
    ax1.scatter(dd, low_qso['logN_OVI_'],
                s=s_low_qso, color=c_low_qso,
                marker=m_low_qso, label=l_low_qso)
    #ax1.errorbar(dd, low_qso['logN_OVI_'], yerr=low_qso['e_sc']+low_qso['e_sys'],
    #             fmt=m_low_qso, markersize=s_low_qso,
    #             color=c_low_qso, label=l_low_qso)

    # qso, Sembach+2003, |vlsr|>~100 km/s
    dd = np.random.uniform(low=172, high=182, size=len(high_qso))
    ax1.scatter(dd, high_qso['logN'],
                s=s_high_qso, edgecolor=c_high_qso, facecolor='none',
                marker=m_high_qso, label=l_high_qso)
    #ax1.errorbar(dd, high_qso['logN'], yerr=high_qso['e1'],
    #             fmt=m_high_qso, markersize=s_high_qso,
    #             color=c_high_qso, label=l_high_qso)

    ax1.vlines(160, 11.5, 15.5, linestyle='--')
    ax1.text(162, 11.2, 'z>0', fontsize=18)
    ax1.legend()

    #### external view, add Tumlinson+2011 data
    ovi_tb = Table.read('/Users/Yong/Dropbox/databucket/Tumlinson11_logNOVI.txt', format='ascii')
    color_a = plt.cm.Blues(0.5)
    color_b = plt.cm.Reds(0.7)

    ind0 = ovi_tb['galaxytype'] == 'active'
    ind1 = ovi_tb['galaxytype'] == 'passive'

    xa = ovi_tb['impact_para'][ind0]
    ya = ovi_tb['NOVI'][ind0]
    ea = ovi_tb['eNOVI'][ind0]
    inda_good = ea!= 0
    ax2.errorbar(xa[inda_good], ya[inda_good], yerr=ea[inda_good], fmt='s', markersize=7,
                 markerfacecolor=color_a, markeredgecolor='k', label='Tumlinson11, star-forming')
    inda_bad = ea==0
    uplims = [0.1]*ea[inda_bad].size
    ax2.errorbar(xa[inda_bad], ya[inda_bad], yerr=uplims, uplims=uplims, fmt='s', markersize=7,
                 markeredgecolor=color_a, markerfacecolor='none', color=color_a, label=None)

    xb = ovi_tb['impact_para'][ind1]
    yb = ovi_tb['NOVI'][ind1]
    eb = ovi_tb['eNOVI'][ind1]
    indb_good = eb!= 0
    ax2.errorbar(xb[indb_good], yb[indb_good], yerr=eb[indb_good], fmt='D', markersize=7,
                 markerfacecolor=color_b, markeredgecolor='k', label='Tumlinson11, passive')
    indb_bad = eb==0
    uplims = [0.1]*eb[indb_bad].size
    ax2.errorbar(xb[indb_bad], yb[indb_bad], yerr=uplims, uplims=uplims, fmt='D', markersize=7,
                 markeredgecolor=color_b, markerfacecolor='none', color=color_b, label=None)

    ax2.legend(loc='lower left')
#########################################

fig.tight_layout()
fig.savefig(figname)
print(figname)
# print(figname)
# plt.close()
