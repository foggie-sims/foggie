from __future__ import print_function
import numpy as np
from scipy import stats

from astropy.table import Table
from astropy.io import ascii

import matplotlib as mpl
import seaborn as sns
sns.set_style("whitegrid", {'axes.grid' : False})
mpl.rcParams['font.family'] = 'stixgeneral'
mpl.rcParams['font.size'] = 16.
import matplotlib.pyplot as plt


def make_misty_component_plots():
    # for now, be lazy and hardcode the paths to the files
    nat = Table.read('/Users/molly/Dropbox/foggie-collab/plots_halo_008508/nref11n/natural/spectra/lls/misty_si2_reg_v6_lsf_lls.dat', format='ascii.fixed_width')
    ref = Table.read('/Users/molly/Dropbox/foggie-collab/plots_halo_008508/nref11n/nref11n_nref10f_refine200kpc/spectra/lls/misty_si2_reg_v6_lsf_lls.dat', format='ascii.fixed_width')
    # hires = Table.read('/Users/molly/Dropbox/foggie-collab/plots_halo_008508/nref11n/nref11f_refine200kpc/spectra/misty_si2_v5_rsp.dat', format='ascii.fixed_width')
    # kodiaq = Table.read('/Users/molly/Dropbox/kodiaq/kodiaq_spectacle_si2.dat', format='ascii.fixed_width')
    print(len(nat), len(ref))
    #print(ref)
    ref_color = 'darkorange' ###  '#4575b4' # purple
    nat_color = '#4daf4a' # green

    nat_p = nat.to_pandas()
    ref_p = ref.to_pandas()
    #hires_p = hires.to_pandas()
    #kod_p = kodiaq.to_pandas()

    idn = [nat['tot_col'] > 12]
    idr = [ref['tot_col'] > 12]
    #idh = [hires['tot_col'] > 12]
    nat_si2 = nat[idn].to_pandas()
    ref_si2 = ref[idr].to_pandas()
    #hires_si2 = hires[idh].to_pandas()

    fig = plt.figure(figsize=(9,7))
    ax = fig.add_subplot(111)
    ax.scatter(ref['reg_dv90'], ref['Nmin'], color='darkorange', alpha=0.5, s=10, label='refined')
    # sns.kdeplot(ref['comp_col'], np.log10(ref['comp_b']),  n_levels=30, shade=True, shade_lowest=False, cmap='Purples')
    #sns.jointplot(ref['comp_col'], np.log10(ref['comp_b']), kind='hex', color='darkorange')
    ax.scatter(nat['reg_dv90'], nat['Nmin'], marker='D', color='#4daf4a',alpha=0.3, s=10, label='natural')
    # sns.kdeplot(nat['comp_col'], np.log10(nat['comp_b']), n_levels=30, shade=True, shade_lowest=False, cmap='Greens', alpha=0.6)
    # ax.scatter(kodiaq['comp_col'], np.log10(kodiaq['comp_b']), color='k', marker='*', s=100, alpha=0.7, label='KODIAQ', zorder=200)
    plt.legend(loc='upper right')
    plt.xlabel(r'Si II region $\Delta$v$_{90}$')
    plt.ylabel(r'Si II region N$_{min}$')
    #plt.ylim(0,30)
    #plt.xlim(xmin=10.5)
    plt.ylim(ymin=0.5)
    fig.tight_layout()
    fig.savefig('SiII_reg_dv90_Nmin.png')

    bins = np.arange(1,25,1)
    fig = plt.figure(figsize=(9,7))
    ax = fig.add_subplot(111)
    ## print('min KODIAQ SiII col = ', min(kodiaq['Si_II_col']))
    idn = [nat['tot_col'] > 12]
    idr = [ref['tot_col'] > 12]
    print(max(nat['Nmin'][idn]), max(ref['Nmin'][idr]))
    ##idh = [hires['Si_II_col'] > 12]
    ## ax.hist(kodiaq['Si_II_Nmin'], color='k',bins=bins,normed=True,histtype='step',lw=3,label='KODIAQ'  )
    ax.hist(nat['Nmin'][idn], bins=bins,normed=True, cumulative=True, histtype='step',lw=3, edgecolor=nat_color, hatch='\\\\', label='natural')
    ax.hist(ref['Nmin'][idr], bins=bins,normed=True, cumulative=True, histtype='step',lw=3, edgecolor=ref_color, hatch='//', label='refined')
    plt.legend(loc='upper left')
    plt.xlim(xmax=10.5)
    plt.xlabel('cumulative fraction of Si II 1260 minima per region')
    fig.tight_layout()
    fig.savefig('Si_II_region_minima.png')

    bins = np.arange(0,150,2)
    fig = plt.figure(figsize=(9,7))
    ax = fig.add_subplot(111)
    ## print('min KODIAQ SiII col = ', min(kodiaq['Si_II_col']))
    idn = [nat['tot_col'] > 12]
    idr = [ref['tot_col'] > 12]
    print(max(nat['Nmin'][idn]), max(ref['Nmin'][idr]))
    ##idh = [hires['Si_II_col'] > 12]
    ## ax.hist(kodiaq['Si_II_Nmin'], color='k',bins=bins,normed=True,histtype='step',lw=3,label='KODIAQ'  )
    ax.hist(nat['reg_dv90'][idn], bins=300,normed=True, cumulative=-1,histtype='step',lw=3, edgecolor=nat_color, hatch='\\\\', label='natural')
    ax.hist(ref['reg_dv90'][idr], bins=300,normed=True, cumulative=-1,histtype='step',lw=3, edgecolor=ref_color, hatch='//', label='refined')
    plt.legend(loc='upper right')
    plt.xlim(0,40)
    plt.ylabel('fraction less than')
    plt.xlabel('Si II 1260 dv90 per region')
    fig.tight_layout()
    fig.savefig('Si_II_region_dv90.png')

    bins = np.arange(0,150,2)
    fig = plt.figure(figsize=(9,7))
    ax = fig.add_subplot(111)
    ## print('min KODIAQ SiII col = ', min(kodiaq['Si_II_col']))
    idn = [nat['tot_col'] > 12]
    idr = [ref['tot_col'] > 12]
    print(max(nat['Nmin'][idn]), max(ref['Nmin'][idr]))
    ##idh = [hires['Si_II_col'] > 12]
    ## ax.hist(kodiaq['Si_II_Nmin'], color='k',bins=bins,normed=True,histtype='step',lw=3,label='KODIAQ'  )
    ax.hist(nat['reg_EW'][idn], bins=300,normed=True, cumulative=-1,histtype='step',lw=3, edgecolor=nat_color, hatch='\\\\', label='natural')
    ax.hist(ref['reg_EW'][idr], bins=300,normed=True, cumulative=-1,histtype='step',lw=3, edgecolor=ref_color, hatch='//', label='refined')
    plt.legend(loc='upper right')
    #plt.xlim(0,40)
    plt.ylabel('fraction less than')
    plt.xlabel('Si II 1260 EW per region')
    fig.tight_layout()
    fig.savefig('Si_II_region_EW.png')

    nat = Table.read('/Users/molly/Dropbox/foggie-collab/plots_halo_008508/nref11n/natural/spectra/lls/misty_o6_reg_v6_lsf_lls.dat', format='ascii.fixed_width')
    ref = Table.read('/Users/molly/Dropbox/foggie-collab/plots_halo_008508/nref11n/nref11n_nref10f_refine200kpc/spectra/lls/misty_o6_reg_v6_lsf_lls.dat', format='ascii.fixed_width')
    bins = np.arange(1,10,1)
    fig = plt.figure(figsize=(9,7))
    ax = fig.add_subplot(111)
    ## print('min KODIAQ SiII col = ', min(kodiaq['Si_II_col']))
    idn = [nat['tot_col'] > 13]
    idr = [ref['tot_col'] > 13]
    print(max(nat['Nmin'][idn]), max(ref['Nmin'][idr]))
    ##idh = [hires['Si_II_col'] > 12]
    ## ax.hist(kodiaq['Si_II_Nmin'], color='k',bins=bins,normed=True,histtype='step',lw=3,label='KODIAQ'  )
    ax.hist(nat['Nmin'][idn], bins=bins,normed=True, cumulative=True, histtype='step',lw=3, edgecolor=nat_color, hatch='\\\\', label='natural')
    ax.hist(ref['Nmin'][idr], bins=bins,normed=True, cumulative=True, histtype='step',lw=3, edgecolor=ref_color, hatch='//', label='refined')
    plt.legend(loc='upper left')
    plt.xlim(xmax=10.5)
    plt.xlabel('cumulative fraction of O VI 1032 minima per region')
    fig.tight_layout()
    fig.savefig('O_VI_region_minima.png')

    bins = np.arange(0,150,2)
    fig = plt.figure(figsize=(9,7))
    ax = fig.add_subplot(111)
    ## print('min KODIAQ SiII col = ', min(kodiaq['Si_II_col']))
    idn = [nat['tot_col'] > 13]
    idr = [ref['tot_col'] > 13]
    print(max(nat['Nmin'][idn]), max(ref['Nmin'][idr]))
    ##idh = [hires['Si_II_col'] > 12]
    ## ax.hist(kodiaq['Si_II_Nmin'], color='k',bins=bins,normed=True,histtype='step',lw=3,label='KODIAQ'  )
    ax.hist(nat['reg_dv90'][idn], bins=300,normed=True, cumulative=-1,histtype='step',lw=3, edgecolor=nat_color, hatch='\\\\', label='natural')
    ax.hist(ref['reg_dv90'][idr], bins=300,normed=True, cumulative=-1,histtype='step',lw=3, edgecolor=ref_color, hatch='//', label='refined')
    plt.legend(loc='upper right')
    plt.xlim(0,40)
    plt.ylabel('fraction less than')
    plt.xlabel('O VI 1032 dv90 per region')
    fig.tight_layout()
    fig.savefig('O_VI_region_dv90.png')



if __name__ == "__main__":
    make_misty_component_plots()
