from __future__ import print_function
import numpy as np
from scipy import stats

from astropy.table import Table
from astropy.io import ascii
import pandas

import matplotlib as mpl
import seaborn as sns
sns.set_style("whitegrid", {'axes.grid' : False})
mpl.rcParams['font.family'] = 'stixgeneral'
mpl.rcParams['font.size'] = 22.
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def make_misty_plots():
    ref_color = 'darkorange' ###  r'#4575b4' # purple
    nat_color = r'#4daf4a' # green
    palette = sns.blend_palette((nat_color, ref_color),n_colors=2)
    si2_limit = 11
    c4_limit = 12
    si4_limit = 13
    o6_limit = 13



    # for now, be lazy and hardcode the paths to the files
    nat = Table.read('/Users/molly/Dropbox/foggie-collab/plots_halo_008508/nref11n/natural/spectra/lls/misty_v6_lsf_lls.dat', format='ascii.fixed_width')
    ref = Table.read('/Users/molly/Dropbox/foggie-collab/plots_halo_008508/nref11n/nref11n_nref10f_refine200kpc/spectra/lls/misty_v6_lsf_lls.dat', format='ascii.fixed_width')
    #hires = Table.read('/Users/molly/Dropbox/foggie-collab/plots_halo_008508/nref11n/nref11f_refine200kpc/spectra/misty_v5_rsp.dat', format='ascii.fixed_width')
    kodiaq = Table.read('/Users/molly/Dropbox/kodiaq/kodiaq_spectacle_all.dat', format='ascii.fixed_width')
    print(len(nat), len(ref))
    print(max(nat['O_VI_Nmin']), max(ref['O_VI_Nmin']))
    print(max(nat['Si_II_Nmin']), max(ref['Si_II_Nmin']))

    nmin_range = np.arange(1,25,1)
    for n in nmin_range:
        nat.add_row([-1, -1, 25, n, -1, n, n, n, 25, n, n, n, -10, -10, 25, n, n, n, -10, -10,25, n, n, n, -10, -10, 25, n, n, n, -10, -10])

#    hires_p = hires.to_pandas()
    kod_p = kodiaq.to_pandas()

    nat['simulation'] = 'standard'
    ref['simulation'] = 'refined'
    nat_p = nat.to_pandas()
    ref_p = ref.to_pandas()
    frames = [nat_p, ref_p]
    both_p = pandas.concat(frames)

    idn = [(nat['Si_II_Nmin'] > 0) & (nat['Si_II_col'] > si2_limit)]
    idr = [(ref['Si_II_Nmin'] > 0) & (ref['Si_II_col'] > si2_limit)]
#    idh = [hires['Si_II_col'] > 12]
    nat_si2 = nat[idn].to_pandas()
    ref_si2 = ref[idr].to_pandas()
    frames = [nat_si2, ref_si2]
    si2 = pandas.concat(frames)
#    hires_si2 = hires[idh].to_pandas()

    idn = [(nat['Si_IV_Nmin'] > 0) & (nat['Si_IV_col'] > si4_limit)]
    idr = [(ref['Si_IV_Nmin'] > 0) & (ref['Si_IV_col'] > si4_limit)]
#    idh = [hires['Si_IV_col'] > 11.5]
    nat_si4 = nat[idn].to_pandas()
    ref_si4 = ref[idr].to_pandas()
    frames = [nat_si4, ref_si4]
    si4 = pandas.concat(frames)
    #hires_si2 = hires[idh].to_pandas()

    idn = [(nat['C_IV_Nmin'] > 0) & (nat['C_IV_col'] > c4_limit)]
    idr = [(ref['C_IV_Nmin'] > 0) & (ref['C_IV_col'] > c4_limit)]
    nat_c4 = nat[idn].to_pandas()
    ref_c4 = ref[idr].to_pandas()
    frames = [nat_c4, ref_c4]
    c4 = pandas.concat(frames)

    idn = [(nat['O_VI_Nmin'] > 0) & (nat['O_VI_col'] > o6_limit)]
    idr = [(ref['O_VI_Nmin'] > 0) & (ref['O_VI_col'] > o6_limit)]
    #idh = [hires['O_VI_col'] > 13]
    nat_o6 = nat[idn].to_pandas()
    ref_o6 = ref[idr].to_pandas()
    frames = [nat_o6, ref_o6]
    o6 = pandas.concat(frames)
    #hires_o6 = hires[idh].to_pandas()


    fig = plt.figure(figsize=(9,7))
    ax = fig.add_subplot(111)
    #sns.swarmplot(x="impact", y="Si_II_Nmin", data=nat_p, color=nat_color,alpha=0.7,orient='h')
    #sns.swarmplot(x="impact", y="Si_II_Nmin", data=ref_p, color=ref_color,alpha=0.7,orient='h')
    g = sns.swarmplot(x="impact", y="Si_II_Nmin", data=si2, hue='simulation', palette=palette, alpha=0.7,orient='h')
    lg = g.axes.get_legend()
    new_title = ''
    lg.set_title(new_title)
    ax.set_yticks((0,5,10,15,20))
    ax.set_yticklabels(('0','5','10','15','20'))
    plt.xlim(0.5,62)
    plt.ylim(0.5,23.5)
    plt.xlabel('impact parameter [kpc]', fontsize=34)
    plt.ylabel('number of Si II 1260 minima', fontsize=34)
    fig.tight_layout()
    fig.savefig('SiII_Nmin_vs_impact.png')
    fig.savefig('SiII_Nmin_vs_impact.pdf')

    fig = plt.figure(figsize=(9,7))
    ax = fig.add_subplot(111)
    g = sns.swarmplot(x="impact", y="O_VI_Nmin", data=o6, hue='simulation', palette=palette, alpha=0.7,orient='h')
    lg = g.axes.get_legend()
    new_title = ''
    lg.set_title(new_title)
    ax.set_yticks((0,5,10,15,20))
    ax.set_yticklabels(('0','5','10','15','20'))
    plt.xlim(0.5,62)
    plt.ylim(0.5,23.5)
    plt.xlabel('impact parameter [kpc]', fontsize=34)
    plt.ylabel('number of O VI 1032 minima', fontsize=34)
    fig.tight_layout()
    fig.savefig('OVI_Nmin_vs_impact.png')
    fig.savefig('OVI_Nmin_vs_impact.pdf')

    fig = plt.figure(figsize=(9,7))
    ax = fig.add_subplot(111)
    g = sns.swarmplot(x="impact", y="C_IV_Nmin", data=c4, hue='simulation', palette=palette, alpha=0.7,orient='h')
    lg = g.axes.get_legend()
    new_title = ''
    lg.set_title(new_title)
    ax.set_yticks((0,5,10,15,20))
    ax.set_yticklabels(('0','5','10','15','20'))
    plt.xlim(0.5,62)
    plt.ylim(0.5,23.5)
    plt.xlabel('impact parameter [kpc]', fontsize=34)
    plt.ylabel('number of C IV 1548 minima', fontsize=34)
    fig.tight_layout()
    fig.savefig('CIV_Nmin_vs_impact.png')
    fig.savefig('CIV_Nmin_vs_impact.pdf')

    # fig = plt.figure(figsize=(9,7))
    # ax = fig.add_subplot(111)
    # ax.scatter(nat['Si_IV_col'], nat['C_IV_col'], marker='D', color=nat_color,label='standard')
    # ax.scatter(ref['Si_IV_col'], ref['C_IV_col'], color=ref_color, label='refined')
    # ax.plot([10.0, 14.5], [10.0+0.47712125472, 14.5+0.47712125472],color='black', label = 'CIV/SiIV = 3')
    # ax.plot([10.0, 14.5], [10.0+2*0.47712125472, 14.5+2*0.47712125472],color='black', ls=':',label = 'CIV/SiIV = 6')
    # plt.legend(loc='lower right')
    # plt.xlabel('Si IV column')
    # plt.ylabel('C IV column')
    # fig.tight_layout()
    # fig.savefig('SiIV_vs_CIV_column.png')

    fig = plt.figure(figsize=(9,7))
    ax = fig.add_subplot(111)
    g = sns.swarmplot(x="Si_II_dv90", y="Si_II_Ncomp", data=si2, hue='simulation', palette=palette, alpha=0.7,orient='h')
    ax.scatter(kodiaq['Si_II_dv90'], kodiaq['Si_II_Ncomp'], color='k', marker='*', s=100, label='KODIAQ',zorder=100)
    ax.legend()
    plt.xlim(xmin=0)
    ax.set_yticks((0,5,10,15))
    ax.set_yticklabels(('0','5','10','15'))
    plt.ylim(0.5,15.5)
    plt.xlabel(r'Si II $\Delta v_{90}$ [km/s]', fontsize=34)
    plt.ylabel(r'number of Si II 1260 components', fontsize=34)
    fig.tight_layout()
    fig.savefig('SiII_dv90_vs_Ncomp.png')

    fig = plt.figure(figsize=(9,7))
    ax = fig.add_subplot(111)
    g = sns.swarmplot(x="Si_II_dv90", y="Si_II_Nmin", data=si2, hue='simulation', palette=palette, alpha=0.7,orient='h')
    ax.scatter(kodiaq['Si_II_dv90'], kodiaq['Si_II_Nmin'], color='k', marker='*', s=100, label='KODIAQ',zorder=100)
    ax.legend()
    plt.xlim(xmin=0)
    ax.set_yticks((0,5,10,15,20))
    ax.set_yticklabels(('0','5','10','15','20'))
    plt.ylim(0.5,23.5)
    plt.xlabel(r'Si II $\Delta v_{90}$ [km/s]')
    plt.ylabel(r'number of Si II 1260 minima', fontsize=34)
    fig.tight_layout()
    fig.savefig('SiII_dv90_vs_Nmin.png')


    fig = plt.figure(figsize=(9,7))
    ax = fig.add_subplot(111)
    g = sns.swarmplot(x="Si_II_col", y="Si_II_Nmin", data=si2, hue='simulation', palette=palette, alpha=0.7,orient='h')
    ax.scatter(kodiaq['Si_II_col'], kodiaq['Si_II_Nmin'], color='k', marker='*', s=100, label='KODIAQ',zorder=100)
    ax.legend()
    plt.xlim(si2_limit, 17)
    ax.set_yticks((0,5,10,15,20))
    ax.set_yticklabels(('0','5','10','15','20'))
    plt.ylim(0.5,23.5)
    plt.xlabel(r'log Si II column density', fontsize=34)
    plt.ylabel(r'number of Si II 1260 minima', fontsize=34)
    fig.tight_layout()
    fig.savefig('SiII_col_vs_Nmin.png')
    fig.savefig('SiII_col_vs_Nmin.pdf')

    fig = plt.figure(figsize=(9,7))
    ax = fig.add_subplot(111)
    g = sns.swarmplot(x="C_IV_col", y="C_IV_Nmin", data=c4, hue='simulation', palette=palette, alpha=0.7,orient='h')
    ax.scatter(kodiaq['C_IV_col'], kodiaq['C_IV_Nmin'], color='k', marker='*', s=100, label='KODIAQ',zorder=100)
    ax.legend()
    plt.xlim(c4_limit, 15.5)
    ax.set_xticks((12,13,14,15))
    ax.set_xticklabels(('12','13','14','15'))
    ax.set_yticks((0,5,10,15,20))
    ax.set_yticklabels(('0','5','10','15','20'))
    plt.ylim(0.5,23.5)
    plt.xlabel(r'log C IV column density', fontsize=34)
    plt.ylabel(r'number of C IV 1548 minima', fontsize=34)
    fig.tight_layout()
    fig.savefig('CIV_col_vs_Nmin.png')
    fig.savefig('CIV_col_vs_Nmin.pdf')

    fig = plt.figure(figsize=(9,7))
    ax = fig.add_subplot(111)
    g = sns.swarmplot(x="Si_IV_col", y="Si_IV_Nmin", data=c4, hue='simulation', palette=palette, alpha=0.7,orient='h')
    ax.scatter(kodiaq['Si_IV_col'], kodiaq['Si_IV_Nmin'], color='k', marker='*', s=100, label='KODIAQ',zorder=100)
    ax.legend()
    plt.xlim(si4_limit, 15.5)
    ax.set_xticks((12,13,14,15))
    ax.set_xticklabels(('12','13','14','15'))
    ax.set_yticks((0,5,10,15,20))
    ax.set_yticklabels(('0','5','10','15','20'))
    plt.ylim(0.5,23.5)
    plt.xlabel(r'log Si IV column density', fontsize=34)
    plt.ylabel(r'number of Si IV 1394 minima', fontsize=34)
    fig.tight_layout()
    fig.savefig('SiIV_col_vs_Nmin.png')
    fig.savefig('SiIV_col_vs_Nmin.pdf')

    fig = plt.figure(figsize=(9,7))
    ax = fig.add_subplot(111)
    g = sns.swarmplot(x="O_VI_col", y="O_VI_Nmin", data=o6, hue='simulation', palette=palette, alpha=0.7,orient='h')
    ax.scatter(kodiaq['O_VI_col'], kodiaq['O_VI_Nmin'], color='k', marker='*', s=100, label='KODIAQ',zorder=100)
    ax.legend()
    plt.xlim(o6_limit, 15.5)
    ax.set_xticks((13,14,15))
    ax.set_xticklabels(('13','14','15'))
    ax.set_yticks((0,5,10,15,20))
    ax.set_yticklabels(('0','5','10','15','20'))
    plt.ylim(0.5,23.5)
    plt.xlabel(r'log O VI column density', fontsize=34)
    plt.ylabel(r'number of O VI 1032 minima', fontsize=34)
    fig.tight_layout()
    fig.savefig('OVI_col_vs_Nmin.png')
    fig.savefig('OVI_col_vs_Nmin.pdf')

    fig = plt.figure(figsize=(9,7))
    ax = fig.add_subplot(111)
    sns.swarmplot(x="O_VI_dv90", y="O_VI_Nmin", data=nat_o6, color=nat_color,alpha=0.7,orient='h')
    sns.swarmplot(x="O_VI_dv90", y="O_VI_Nmin", data=ref_o6, color=ref_color,alpha=0.7,orient='h')
#    sns.swarmplot(x="O_VI_dv90", y="O_VI_Nmin", data=hires_o6, color=r'number984ea3',alpha=0.7,orient='h')
    # sns.swarmplot(x="Si_II_dv90", y="Si_II_Ncomp", data=kod_p, color='k',alpha=0.7,orient='h')
    #ax.scatter(nat['Si_II_dv90'],nat['Si_II_Nmin'], marker='D', s=60, color=nat_color,alpha=0.5,label='standard')
    #ax.scatter(ref['Si_II_dv90'],ref['Si_II_Nmin'], color=ref_color, marker='o',s=100, alpha=0.5,label='refined')
    ax.scatter(kodiaq['O_VI_dv90'], kodiaq['O_VI_Nmin'], color='k', marker='*', s=100, label='KODIAQ',zorder=100)
    plt.legend(loc='upper right')
    plt.xlim(xmin=0)
    plt.ylim(0.5,14.5)
    plt.xlabel(r'O VI $\Delta v_{90}$ [km/s]', fontsize=34)
    plt.ylabel(r'number of O VI 1032 minima', fontsize=34)
    fig.tight_layout()
    fig.savefig('OVI_dv90_vs_Nmin.png')


    fig = plt.figure(figsize=(9,7))
    ax = fig.add_subplot(111)
    sns.swarmplot(x="C_IV_dv90", y="C_IV_Nmin", data=nat_c4, color=nat_color,alpha=0.7,orient='h')
    sns.swarmplot(x="C_IV_dv90", y="C_IV_Nmin", data=ref_c4, color=ref_color,alpha=0.7,orient='h')
#    sns.swarmplot(x="O_VI_dv90", y="O_VI_Nmin", data=hires_o6, color=r'number984ea3',alpha=0.7,orient='h')
    # sns.swarmplot(x="Si_II_dv90", y="Si_II_Ncomp", data=kod_p, color='k',alpha=0.7,orient='h')
    #ax.scatter(nat['Si_II_dv90'],nat['Si_II_Nmin'], marker='D', s=60, color=nat_color,alpha=0.5,label='standard')
    #ax.scatter(ref['Si_II_dv90'],ref['Si_II_Nmin'], color=ref_color, marker='o',s=100, alpha=0.5,label='refined')
    ax.scatter(kodiaq['C_IV_dv90'], kodiaq['C_IV_Nmin'], color='k', marker='*', s=100, label='KODIAQ',zorder=100)
    plt.legend(loc='upper right')
    plt.xlim(0,500)
    plt.ylim(0.5,14.5)
    plt.xlabel(r'C IV $\Delta v_{90}$ [km/s]', fontsize=34)
    plt.ylabel(r'number of C IV 1548 minima', fontsize=34)
    fig.tight_layout()
    fig.savefig('CIV_dv90_vs_Nmin.png')


    fig = plt.figure(figsize=(11,7))
    ax = fig.add_subplot(111)
    sns.swarmplot(x="Si_II_EW", y="Si_II_Nmin", data=nat_si2, color=nat_color,alpha=0.7,orient='h')
    sns.swarmplot(x="Si_II_EW", y="Si_II_Nmin", data=ref_si2, color=ref_color,alpha=0.7,orient='h')
    #ax.scatter(nat['Si_II_EW'],nat['Si_II_Nmin'], marker='D', s=60, color=nat_color,label='standard')
    #ax.scatter(ref['Si_II_EW'],ref['Si_II_Nmin'], color=ref_color, marker='o',s=100, label='refined')
    ax.scatter(kodiaq['Si_II_EW'], kodiaq['Si_II_Nmin'], color='k', marker='*', s=100, label='KODIAQ', zorder=200)
    plt.legend(loc='lower right')
    plt.xlim(xmin=0)
    plt.ylim(0.5,14.5)
    plt.xlabel(r'Si II EW', fontsize=34)
    plt.ylabel(r'number of Si II 1260 minima', fontsize=34)
    fig.tight_layout()
    fig.savefig('SiII_EW_vs_Nmin.png')

    fig = plt.figure(figsize=(11,7))
    ax = fig.add_subplot(111)
    sns.swarmplot(x="O_VI_EW", y="O_VI_Nmin", data=nat_o6, color=nat_color,alpha=0.7,orient='h')
    sns.swarmplot(x="O_VI_EW", y="O_VI_Nmin", data=ref_o6, color=ref_color,alpha=0.7,orient='h')
    #ax.scatter(nat['Si_II_EW'],nat['Si_II_Nmin'], marker='D', s=60, color=nat_color,label='standard')
    #ax.scatter(ref['Si_II_EW'],ref['Si_II_Nmin'], color=ref_color, marker='o',s=100, label='refined')
    ax.scatter(kodiaq['O_VI_EW'], kodiaq['O_VI_Nmin'], color='k', marker='*', s=100, label='KODIAQ', zorder=200)
    plt.legend(loc='lower right')
    plt.xlim(xmin=0)
    plt.ylim(0.5,14.5)
    plt.xlabel(r'O VI EW', fontsize=34)
    plt.ylabel(r'number of O VI 1032 minima', fontsize=34)
    fig.tight_layout()
    fig.savefig('OVI_EW_vs_Nmin.png')

    fig = plt.figure(figsize=(11,7))
    ax = fig.add_subplot(111)
    sns.swarmplot(x="Si_II_EW", y="Si_II_Nreg", data=nat_si2, color=nat_color,alpha=0.7,orient='h')
    sns.swarmplot(x="Si_II_EW", y="Si_II_Nreg", data=ref_si2, color=ref_color,alpha=0.7,orient='h')
    #ax.scatter(nat['Si_II_EW'],nat['Si_II_Nmin'], marker='D', s=60, color=nat_color,label='standard')
    #ax.scatter(ref['Si_II_EW'],ref['Si_II_Nmin'], color=ref_color, marker='o',s=100, label='refined')
    ##ax.scatter(kodiaq['Si_II_EW'], kodiaq['Si_II_Nmin'], color='k', marker='*', s=100, label='KODIAQ', zorder=200)
    plt.legend(loc='lower right')
    plt.xlim(xmin=0)
    plt.ylim(0.5,7.5)
    plt.xlabel(r'Si II EW', fontsize=34)
    plt.ylabel(r'number of Si II 1260 regions', fontsize=34)
    fig.tight_layout()
    fig.savefig('SiII_EW_vs_Nreg.png')

    fig = plt.figure(figsize=(11,7))
    ax = fig.add_subplot(111)
    sns.swarmplot(x="O_VI_EW", y="O_VI_Nreg", data=nat_si2, color=nat_color,alpha=0.7,orient='h')
    sns.swarmplot(x="O_VI_EW", y="O_VI_Nreg", data=ref_si2, color=ref_color,alpha=0.7,orient='h')
    #ax.scatter(nat['Si_II_EW'],nat['Si_II_Nmin'], marker='D', s=60, color=nat_color,label='standard')
    #ax.scatter(ref['Si_II_EW'],ref['Si_II_Nmin'], color=ref_color, marker='o',s=100, label='refined')
    ##ax.scatter(kodiaq['Si_II_EW'], kodiaq['Si_II_Nmin'], color='k', marker='*', s=100, label='KODIAQ', zorder=200)
    plt.legend(loc='lower right')
    plt.xlim(xmin=0)
    plt.ylim(0.5,7.5)
    plt.xlabel(r'O VI EW', fontsize=34)
    plt.ylabel(r'number of O VI 1032 regions', fontsize=34)
    fig.tight_layout()
    fig.savefig('OVI_EW_vs_Nreg.png')

    fig = plt.figure(figsize=(9,7))
    ax = fig.add_subplot(111)
    idn = [nat['Si_IV_col'] > si4_limit]
    idr = [ref['Si_IV_col'] > si4_limit]
    ax.scatter(nat['Si_IV_col'][idn],nat['Si_IV_dv90'][idn], marker='D', s=30, color=nat_color,alpha=0.5, label='standard')
    ax.scatter(ref['Si_IV_col'][idr],ref['Si_IV_dv90'][idr], color=ref_color, marker='o',s=50, alpha=0.5, label='refined')
    ax.scatter(kodiaq['Si_IV_col'], kodiaq['Si_IV_dv90'], color='k', marker='*', s=100, label='KODIAQ')
    plt.legend(loc='upper left')
    plt.xlim(xmin=11.5)
    plt.ylim(0,600)
    plt.xlabel('Si IV column density', fontsize=34)
    plt.ylabel(r'Si IV $\Delta v_{90}$', fontsize=34)
    fig.tight_layout()
    fig.savefig('SiIV_col_dv90.png')

    fig = plt.figure(figsize=(9,7))
    ax = fig.add_subplot(111)
    ax.scatter(nat['O_VI_col'],nat['O_VI_dv90'], marker='D', s=30, color=nat_color,alpha=0.5, label='standard')
    ax.scatter(ref['O_VI_col'],ref['O_VI_dv90'], color=ref_color, marker='o',s=50, alpha=0.5, label='refined')
    ax.scatter(kodiaq['O_VI_col'], kodiaq['O_VI_dv90'], color='k', marker='*', s=100, label='KODIAQ')
    plt.legend(loc='upper right')
    plt.xlim(xmin=13)
    plt.ylim(ymin=0)
    plt.xlabel('O VI column density', fontsize=34)
    plt.ylabel(r'O VI $\Delta v_{90}$ [km/s]', fontsize=34)
    fig.tight_layout()
    fig.savefig('OVI_col_dv90.png')

    fig = plt.figure(figsize=(9,7))
    ax = fig.add_subplot(111)
    ax.scatter(nat['HI_col'],nat['HI_1216_EW'], marker='D', s=60, color=nat_color,alpha=0.5, label='standard')
    ax.scatter(ref['HI_col'],ref['HI_1216_EW'], color=ref_color, marker='o',s=100, alpha=0.5, label='refined')
    plt.legend(loc='upper left')
    #plt.xlim(xmin=0)
    plt.ylim(ymin=0)
    plt.xlabel('HI column density', fontsize=34)
    plt.ylabel(r'HI 1216 EW', fontsize=34)
    fig.tight_layout()
    fig.savefig('HI_col_vs_ew.png')


    fig = plt.figure(figsize=(9,7))
    ax = fig.add_subplot(111)
    ax.scatter(nat['O_VI_col'],nat['O_VI_EW'], marker='D', s=60, color=nat_color,alpha=0.5, label='standard')
    ax.scatter(ref['O_VI_col'],ref['O_VI_EW'], color=ref_color, marker='o',s=100, alpha=0.5, label='refined')
    ax.scatter(kodiaq['O_VI_col'], kodiaq['O_VI_EW'], color='k', marker='*', s=100, alpha=0.5, label='KODIAQ')
    plt.legend(loc='upper left')
    #plt.xlim(xmin=0)
    #plt.ylim(ymin=0)
    plt.xlabel('O VI column density', fontsize=34)
    plt.ylabel(r'O VI 1032 EW', fontsize=34)
    fig.tight_layout()
    fig.savefig('OVI_col_ew.png')

    # fig = plt.figure(figsize=(9,7))
    # ax = fig.add_subplot(111)
    # sns.swarmplot(x="Si_IV_col", y="Si_IV_Nmin", data=nat_si4, color=nat_color,alpha=0.7,orient='h')
    # sns.swarmplot(x="Si_IV_col", y="Si_IV_Nmin", data=ref_si4, color=ref_color,alpha=0.7,orient='h')
    # #ax.scatter(nat['HI_col'],nat['Si_II_Nmin'], marker='D', s=60, color=nat_color,alpha=0.5,label='standard')
    # #ax.scatter(ref['HI_col'],ref['Si_II_Nmin'], color=ref_color, marker='o',s=100, alpha=0.5,label='refined')
    # ax.scatter(kodiaq['Si_IV_col'], kodiaq['Si_IV_Nmin'], color='k', marker='*', s=100, alpha=0.7, label='KODIAQ',zorder=100)
    # plt.legend(loc='upper left', frameon=False)
    # plt.xlim(xmin=11.5)
    # plt.ylim(0.5,14.5)
    # plt.xlabel(r'log SiIV column density')
    # plt.ylabel(r'number of Si IV 1394 minima', fontsize=34)
    # fig.tight_layout()
    # fig.savefig('SiIV_col_vs_Nmin.png')
    #
    # fig = plt.figure(figsize=(11,7))
    # ax = fig.add_subplot(111)
    # sns.swarmplot(x="O_VI_col", y="O_VI_Nmin", data=nat_o6, color=nat_color,alpha=0.7,orient='h')
    # sns.swarmplot(x="O_VI_col", y="O_VI_Nmin", data=ref_o6, color=ref_color,alpha=0.7,orient='h')
    # #ax.scatter(nat['HI_col'],nat['Si_II_Nmin'], marker='D', s=60, color=nat_color,alpha=0.5,label='standard')
    # #ax.scatter(ref['HI_col'],ref['Si_II_Nmin'], color=ref_color, marker='o',s=100, alpha=0.5,label='refined')
    # ax.scatter(kodiaq['O_VI_col'], kodiaq['O_VI_Nmin'], color='k', marker='*', s=100, alpha=0.7, label='KODIAQ',zorder=100)
    # plt.legend(loc='upper left', frameon=False)
    # plt.xlim(xmin=13)
    # plt.ylim(0.5,16.5)
    # plt.xlabel(r'log OVI column density')
    # plt.ylabel(r'number of O VI 1032 minima', fontsize=34)
    # fig.tight_layout()
    # fig.savefig('OVI_col_vs_Nmin.png')

    fig = plt.figure(figsize=(9,7))
    ax = fig.add_subplot(111)
    sns.swarmplot(x="HI_col", y="Si_II_Nmin", data=nat_si2, color=nat_color,alpha=0.7,orient='h')
    sns.swarmplot(x="HI_col", y="Si_II_Nmin", data=ref_si2, color=ref_color,alpha=0.7,orient='h')
    #ax.scatter(nat['HI_col'],nat['Si_II_Nmin'], marker='D', s=60, color=nat_color,alpha=0.5,label='standard')
    #ax.scatter(ref['HI_col'],ref['Si_II_Nmin'], color=ref_color, marker='o',s=100, alpha=0.5,label='refined')
    ax.scatter(kodiaq['HI_col'], kodiaq['Si_II_Nmin'], color='k', marker='*', s=100, alpha=0.7, label='KODIAQ',zorder=100)
    plt.legend(loc='upper left', frameon=False)
    plt.xlim(xmin=16)
    plt.ylim(0.5,14)
    plt.xlabel(r'log HI column density', fontsize=34)
    plt.ylabel(r'number of Si II 1260 minima', fontsize=34)
    fig.tight_layout()
    fig.savefig('HI_col_vs_SiII_Nmin.png')

    fig = plt.figure(figsize=(9,7))
    ax = fig.add_subplot(111)
    sns.swarmplot(x="HI_col", y="Si_IV_Nmin", data=nat_si4, color=nat_color,alpha=0.7,orient='h')
    sns.swarmplot(x="HI_col", y="Si_IV_Nmin", data=ref_si4, color=ref_color,alpha=0.7,orient='h')
    ax.scatter(kodiaq['HI_col'], kodiaq['Si_IV_Nmin'], color='k', marker='*', s=100, alpha=0.7, label='KODIAQ',zorder=100)
    plt.legend(loc='upper left', frameon=False)
    plt.xlim(xmin=16)
    plt.ylim(0.5,12)
    plt.xlabel(r'log HI column density', fontsize=34)
    plt.ylabel(r'number of Si IV 1394 minima', fontsize=34)
    fig.tight_layout()
    fig.savefig('HI_col_vs_SiIV_Nmin.png')

    fig = plt.figure(figsize=(9,7))
    ax = fig.add_subplot(111)
    sns.stripplot(x=nat['O_VI_Nmin'], y=nat['Si_II_Nmin']+0.1,jitter=True, color=nat_color, dodge=True,edgecolor='none',s=5, marker='D',alpha=0.5)
    sns.stripplot(x=ref['O_VI_Nmin'], y=ref['Si_II_Nmin']-0.1,jitter=True, color=ref_color, dodge=True,edgecolor='none',s=5, marker='o',alpha=0.5)
    # ax.scatter(nat['O_VI_Nmin'],nat['Si_II_Nmin'], marker='D', s=60, color=nat_color,alpha=0.5,label='standard')
#    ax.scatter(ref['O_VI_Nmin'],ref['Si_II_Nmin'], color=ref_color, marker='o',s=100, alpha=0.5,label='refined')
    ax.scatter(kodiaq['O_VI_Nmin'], kodiaq['Si_II_Nmin'], color='k', marker='*', s=100, alpha=0.7, label='KODIAQ',zorder=100)
    plt.legend(loc='upper left', frameon=False)
    plt.xlim(xmin=0)
    plt.ylim(ymin=0)
    plt.xlabel(r'number of O VI 1032 minima', fontsize=34)
    plt.ylabel(r'number of Si II 1260 minima', fontsize=34)
    fig.tight_layout()
    fig.savefig('OVI_Nmin_vs_SiII_Nmin.png')

    # fig = plt.figure(figsize=(9,7))
    # ax = fig.add_subplot(111)
    # ax.scatter(nat['C_II_dv90'],nat['C_II_Ncomp'], marker='D', s=100, color=nat_color,label='standard')
    # ax.scatter(ref['C_II_dv90'],ref['C_II_Ncomp'], color=ref_color, marker='*',s=100, label='refined')
    # plt.legend(loc='lower right')
    # plt.xlim(xmin=0)
    # plt.ylim(ymin=0)
    # plt.xlabel(r'C II $\Delta v_{90}$')
    # plt.ylabel(r'number of C II 1335 components')
    # fig.tight_layout()
    # fig.savefig('CII_dv90_vs_Ncomp.png')


    bins = np.arange(1,20,1)
    fig = plt.figure(figsize=(11,7))
    ax = fig.add_subplot(111)
    idn = [(nat['O_VI_Nmin'] > 4) & (nat['O_VI_col'] > 13)]
    idr = [(ref['O_VI_Nmin'] > 4) & (ref['O_VI_col'] > 13)]
    #ax.hist(kodiaq['Si_II_Nreg'], color='k',bins=bins,normed=True,histtype='step',lw=3,label='KODIAQ'  )
    ax.hist(nat['O_VI_Nmin'][idn]/nat['O_VI_Nreg'][idn], bins=bins,normed=True, align='left',histtype='step',lw=3, edgecolor=nat_color, hatch='\\\\', label='standard')
    ax.hist(ref['O_VI_Nmin'][idr]/ref['O_VI_Nreg'][idr], bins=bins,normed=True, align='left',histtype='step',lw=3, edgecolor=ref_color, hatch='//', label='refined')
    plt.legend(loc='upper right')
    plt.xlabel(r'number of O VI 1032 minima / regions')
    plt.ylabel('normalized fraction of sightlines')
    ax.set_xticks((1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18))
    ax.set_xticklabels(('1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18'))
    fig.tight_layout()
    fig.savefig('O_VI_regions_fraction_histograms.png')

    bins = np.arange(1,20,1)
    fig = plt.figure(figsize=(11,7))
    ax = fig.add_subplot(111)
    print('min KODIAQ SiII col = ', min(kodiaq['Si_II_col']))
    idn = [(nat['Si_II_Nmin'] > 4) & (nat['Si_II_col'] > 12)]
    idr = [(ref['Si_II_Nmin'] > 4) & (ref['Si_II_col'] > 12)]
    #ax.hist(kodiaq['Si_II_Nreg'], color='k',bins=bins,normed=True,histtype='step',lw=3,label='KODIAQ'  )
    ax.hist(nat['Si_II_Nmin'][idn]/nat['Si_II_Nreg'][idn], bins=bins,normed=True, align='left',histtype='step',lw=3, edgecolor=nat_color, hatch='\\\\', label='standard')
    ax.hist(ref['Si_II_Nmin'][idr]/ref['Si_II_Nreg'][idr], bins=bins,normed=True, align='left',histtype='step',lw=3, edgecolor=ref_color, hatch='//', label='refined')
    plt.legend(loc='upper right')
    plt.xlabel(r'number of Si II 1260 minima / regions')
    plt.ylabel('normalized fraction of sightlines with Nmin > 4')
    ax.set_xticks((1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18))
    ax.set_xticklabels(('1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18'))
    fig.tight_layout()
    fig.savefig('Si_II_regions_fraction_histograms.png')

    bins = np.arange(1,20,1)
    fig = plt.figure(figsize=(11,7))
    ax = fig.add_subplot(111)
    print('min KODIAQ SiII col = ', min(kodiaq['Si_II_col']))
    idn = [(nat['Si_II_Nmin'] > 4) & (nat['Si_II_col'] > 12)]
    idr = [(ref['Si_II_Nmin'] > 4) & (ref['Si_II_col'] > 12)]
    #idh = [hires['Si_II_col'] > 12]
    #ax.hist(kodiaq['Si_II_Nreg'], color='k',bins=bins,normed=True,histtype='step',lw=3,label='KODIAQ'  )
    ax.hist(nat['Si_II_Nreg'][idn], bins=bins,normed=True, histtype='step',lw=3,align='left', edgecolor=nat_color, hatch='\\\\', label='standard')
    ax.hist(ref['Si_II_Nreg'][idr], bins=bins,normed=True, histtype='step',lw=3, align='left',edgecolor=ref_color, hatch='//', label='refined')
    plt.legend(loc='upper right')
    plt.xlabel(r'number of Si II 1260 regions')
    plt.ylabel('fraction of sightlines with > 4 minima', fontsize=34)
    ax.set_xticks((1,2,3,4,5,6,7,8,9,10))
    ax.set_xticklabels(('1','2','3','4','5','6','7','8','9','10'))
    plt.xlim(0,11)
    fig.tight_layout()
    fig.savefig('Si_II_regions_histograms_Nmingt4.png')

    bins = np.arange(1,20,1)
    fig = plt.figure(figsize=(11,7))
    ax = fig.add_subplot(111)
    print('min KODIAQ SiII col = ', min(kodiaq['Si_II_col']))
    idn = [nat['Si_II_col'] > si2_limit]
    idr = [ref['Si_II_col'] > si2_limit]
    #idh = [hires['Si_II_col'] > 12]
    #ax.hist(kodiaq['Si_II_Nreg'], color='k',bins=bins,normed=True,histtype='step',lw=3,label='KODIAQ'  )
    ax.hist(nat['Si_II_Nreg'][idn], bins=bins,normed=True, histtype='step',lw=3,align='left', edgecolor=nat_color, hatch='\\\\', label='standard')
    ax.hist(ref['Si_II_Nreg'][idr], bins=bins,normed=True, histtype='step',lw=3, align='left',edgecolor=ref_color, hatch='//', label='refined')
    plt.legend(loc='upper right')
    plt.xlabel(r'number of Si II 1260 regions')
    plt.ylabel('normalized fraction of sightlines')
    ax.set_xticks((1,2,3,4,5,6,7,8,9,10))
    ax.set_xticklabels(('1','2','3','4','5','6','7','8','9','10'))
    plt.xlim(0,11)
    fig.tight_layout()
    fig.savefig('Si_II_regions_histograms.png')

    bins = np.arange(1,20,1)
    fig = plt.figure(figsize=(11,7))
    ax = fig.add_subplot(111)
    print('min KODIAQ SiII col = ', min(kodiaq['Si_II_col']))
    idn = [(nat['O_VI_Nmin'] > 4) & (nat['O_VI_col'] > 13)]
    idr = [(ref['O_VI_Nmin'] > 4) & (ref['O_VI_col'] > 13)]
    #idh = [hires['Si_II_col'] > 12]
    #ax.hist(kodiaq['Si_II_Nreg'], color='k',bins=bins,normed=True,histtype='step',lw=3,label='KODIAQ'  )
    ax.hist(nat['O_VI_Nreg'][idn], bins=bins,normed=True, histtype='step',lw=3,align='left', edgecolor=nat_color, hatch='\\\\', label='standard')
    ax.hist(ref['O_VI_Nreg'][idr], bins=bins,normed=True, histtype='step',lw=3, align='left',edgecolor=ref_color, hatch='//', label='refined')
    plt.legend(loc='upper right')
    plt.xlabel(r'number of O VI 1032 regions')
    plt.ylabel('fraction of sightlines with > 4 minima', fontsize=34)
    ax.set_xticks((1,2,3,4,5,6,7,8,9,10))
    ax.set_xticklabels(('1','2','3','4','5','6','7','8','9','10'))
    plt.xlim(0,11)
    fig.tight_layout()
    fig.savefig('O_VI_regions_histograms_Nmingt4.png')

    bins = np.arange(1,20,1)
    fig = plt.figure(figsize=(11,7))
    ax = fig.add_subplot(111)
    print('min KODIAQ SiII col = ', min(kodiaq['Si_II_col']))
    idn = [nat['O_VI_col'] > 13]
    idr = [ref['O_VI_col'] > 13]
    #idh = [hires['Si_II_col'] > 12]
    #ax.hist(kodiaq['Si_II_Nreg'], color='k',bins=bins,normed=True,histtype='step',lw=3,label='KODIAQ'  )
    ax.hist(nat['O_VI_Nreg'][idn], bins=bins,normed=True, histtype='step',lw=3,align='left', edgecolor=nat_color, hatch='\\\\', label='standard')
    ax.hist(ref['O_VI_Nreg'][idr], bins=bins,normed=True, histtype='step',lw=3, align='left',edgecolor=ref_color, hatch='//', label='refined')
    plt.legend(loc='upper right')
    plt.xlabel(r'number of O VI 1032 regions')
    plt.ylabel('normalized fraction of sightlines')
    ax.set_xticks((1,2,3,4,5,6,7,8,9,10))
    ax.set_xticklabels(('1','2','3','4','5','6','7','8','9','10'))
    plt.xlim(0,11)
    fig.tight_layout()
    fig.savefig('O_VI_regions_histograms.png')

    bins = np.arange(1,20,1)
    fig = plt.figure(figsize=(11,7))
    ax = fig.add_subplot(111)
    print('min KODIAQ SiII col = ', min(kodiaq['Si_II_col']))
    idn = [nat['Si_II_col'] > si2_limit]
    idr = [ref['Si_II_col'] > si2_limit]
    print('mean SiII Nmin, natural = ', np.mean(nat['Si_II_Nmin'][idn]))
    print('mean SiII Nmin, refined = ', np.mean(ref['Si_II_Nmin'][idr]))
    ax.hist(kodiaq['Si_II_Nmin'], color='k',bins=bins,normed=True,histtype='step',align='left',lw=3,label='KODIAQ'  )
    ax.hist(nat['Si_II_Nmin'][idn], bins=bins,normed=True, histtype='step',lw=3, align='left',edgecolor=nat_color, hatch='\\\\', label='standard')
    ax.hist(ref['Si_II_Nmin'][idr], bins=bins,normed=True, histtype='step',lw=3, align='left',edgecolor=ref_color, hatch='//', label='refined')
    ax.set_xticks((1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18))
    ax.set_xticklabels(('1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18'))
    plt.legend(loc='upper right')
    plt.xlabel(r'number of Si II 1260 minima', fontsize=34)
    plt.ylabel('normalized fraction', fontsize=34)
    fig.tight_layout()
    fig.savefig('Si_II_histograms.png')
    fig.savefig('Si_II_histograms.pdf')
    #
    # fig = plt.figure(figsize=(11,7))
    # ax = fig.add_subplot(111)
    # sns.swarmplot(x="Si_II_col", y="Si_II_Nmin", data=nat_si2, color=nat_color,alpha=0.7,orient='h')
    # sns.swarmplot(x="Si_II_col", y="Si_II_Nmin", data=ref_si2, color=ref_color,alpha=0.7,orient='h')
    # #ax.scatter(nat['HI_col'],nat['Si_II_Nmin'], marker='D', s=60, color=nat_color,alpha=0.5,label='standard')
    # #ax.scatter(ref['HI_col'],ref['Si_II_Nmin'], color=ref_color, marker='o',s=100, alpha=0.5,label='refined')
    # ax.scatter(kodiaq['Si_II_col'], kodiaq['Si_II_Nmin'], color='k', marker='*', s=100, alpha=0.7, label='KODIAQ',zorder=100)
    # plt.legend(loc='upper left', frameon=False)
    # plt.xlim(xmin=12)
    # plt.ylim(0.5,14.5)
    # plt.xlabel(r'log SiII column density')
    # plt.ylabel(r'number of Si II 1260 minima', fontsize=34)
    # fig.tight_layout()
    # fig.savefig('SiII_col_vs_Nmin.png')

    fig = plt.figure(figsize=(11,7))
    ax = fig.add_subplot(111)
    print('min KODIAQ SiIV col = ', min(kodiaq['Si_IV_col'][kodiaq['Si_IV_col'] > 0]))
    idn = [nat['Si_IV_col'] > si4_limit]
    idr = [ref['Si_IV_col'] > si4_limit]
    print('mean SiIV Nmin, natural = ', np.mean(nat['Si_IV_Nmin'][idn]))
    print('mean SiIV Nmin, refined = ', np.mean(ref['Si_IV_Nmin'][idr]))
    ax.hist(kodiaq['Si_IV_Nmin'], color='k',bins=bins,normed=True,histtype='step',align='left', lw=3,label='KODIAQ'  )
    ax.hist(nat['Si_IV_Nmin'][idn], bins=bins,normed=True, histtype='step',align='left', lw=3, edgecolor=nat_color, hatch='\\\\', label='standard')
    ax.hist(ref['Si_IV_Nmin'][idr], bins=bins,normed=True, histtype='step',align='left', lw=3, edgecolor=ref_color, hatch='//', label='refined')
    ax.set_xticks((1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18))
    ax.set_xticklabels(('1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18'))
    plt.legend(loc='upper right')
    plt.xlabel(r'number of Si IV 1394 minima', fontsize=34)
    plt.ylabel('normalized fraction', fontsize=34)
    fig.tight_layout()
    fig.savefig('Si_IV_histograms.png')
    fig.savefig('Si_IV_histograms.pdf')

    fig = plt.figure(figsize=(11,7))
    ax = fig.add_subplot(111)
    print('min KODIAQ CIV col = ', min(kodiaq['C_IV_col'][kodiaq['C_IV_col'] > 0]))
    idn = [nat['C_IV_col'] > c4_limit]
    idr = [ref['C_IV_col'] > c4_limit]
    print('mean CIV Nmin, natural = ', np.mean(nat['C_IV_Nmin'][idn]))
    print('mean CIV Nmin, refined = ', np.mean(ref['C_IV_Nmin'][idr]))
    ax.hist(kodiaq['C_IV_Nmin'], color='k',bins=bins,normed=True,histtype='step',align='left', lw=3,label='KODIAQ' )
    ax.hist(nat['C_IV_Nmin'][idn], bins=bins, normed=True, histtype='step',align='left', lw=3, edgecolor=nat_color, hatch='\\\\', label='standard')
    ax.hist(ref['C_IV_Nmin'][idr], bins=bins,normed=True, histtype='step',align='left', lw=3, edgecolor=ref_color, hatch='//', label='refined')
    ax.set_xticks((1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18))
    ax.set_xticklabels(('1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18'))
    plt.legend(loc='upper right')
    plt.xlabel(r'number of C IV 1548 minima', fontsize=34)
    plt.ylabel('normalized fraction', fontsize=34)
    fig.tight_layout()
    fig.savefig('C_IV_histograms.png')
    fig.savefig('C_IV_histograms.pdf')

    fig = plt.figure(figsize=(11,7))
    ax = fig.add_subplot(111)
    print('min KODIAQ OVI col = ', min(kodiaq['O_VI_col'][kodiaq['O_VI_col'] > 0]))
    idn = [nat['O_VI_col'] > o6_limit] ## 11.2
    idr = [ref['O_VI_col'] > o6_limit]
    print('mean OVI Nmin, natural = ', np.mean(nat['O_VI_Nmin'][idn]))
    print('mean OVI Nmin, refined = ', np.mean(ref['O_VI_Nmin'][idr]))
    ax.hist(kodiaq['O_VI_Nmin'], color='k',bins=bins,normed=True,histtype='step',align='left', lw=3,label='KODIAQ')
    ax.hist(nat['O_VI_Nmin'][idn], bins=bins,normed=True, histtype='step', align='left', lw=3, edgecolor=nat_color, hatch='\\\\', label='standard')
    ax.hist(ref['O_VI_Nmin'][idr], bins=bins,normed=True, histtype='step', align='left', lw=3, edgecolor=ref_color, hatch='//', label='refined')
    ax.set_xticks((1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18))
    ax.set_xticklabels(('1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18'))
    plt.legend(loc='upper right')
    plt.xlabel(r'number of O VI 1032 minima', fontsize=34)
    plt.ylabel('normalized fraction', fontsize=34)
    fig.tight_layout()
    fig.savefig('O_VI_histograms.png')
    fig.savefig('O_VI_histograms.pdf')


    idn = [(nat['Si_II_Nmin'] > 0) & (nat['Si_II_col'] > si2_limit) & (nat['HI_col'] > 0)]
    idr = [(ref['Si_II_Nmin'] > 0) & (ref['Si_II_col'] > si2_limit)]
    nat_si2 = nat[idn]
    ref_si2 = ref[idr]
    key = 'Si_II_Nmin'
    print(key, 'nat vs ref:', stats.ks_2samp(np.array(nat_si2[key]), np.array(ref_si2[key])))


    idn = [(nat['Si_IV_Nmin'] > 0) & (nat['Si_IV_col'] > si4_limit) & (nat['HI_col'] > 0)]
    idr = [(ref['Si_IV_Nmin'] > 0) & (ref['Si_IV_col'] > si4_limit)]
    nat_si4 = nat[idn]
    ref_si4 = ref[idr]
    key = 'Si_IV_Nmin'
    print(key, 'nat vs ref:', stats.ks_2samp(np.array(nat_si4[key]), np.array(ref_si4[key])))

    idn = [(nat['C_IV_Nmin'] > 0) & (nat['C_IV_col'] > c4_limit) & (nat['HI_col'] > 0)]
    idr = [(ref['C_IV_Nmin'] > 0) & (ref['C_IV_col'] > c4_limit)]
    nat_c4 = nat[idn]
    ref_c4 = ref[idr]
    key = 'C_IV_Nmin'
    print(key, 'nat vs ref:', stats.ks_2samp(np.array(nat_c4[key]), np.array(ref_c4[key])))

    idn = [(nat['O_VI_Nmin'] > 0) & (nat['O_VI_col'] > o6_limit) & (nat['HI_col'] > 0)]
    idr = [(ref['O_VI_Nmin'] > 0) & (ref['O_VI_col'] > o6_limit)]
    nat_o6 = nat[idn]
    ref_o6 = ref[idr]
    key = 'O_VI_Nmin'
    print(key, 'nat vs ref:', stats.ks_2samp(np.array(nat_o6[key]), np.array(ref_o6[key])))


if __name__ == "__main__":
    make_misty_plots()
