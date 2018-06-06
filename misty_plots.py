import numpy as np

from astropy.table import Table
from astropy.io import ascii

import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['font.size'] = 16.


def make_misty_plots():
    # for now, be lazy and hardcode the paths to the files
    nat = Table.read('/Users/molly/Dropbox/foggie-collab/plots_halo_008508/nref11n/natural/spectra/misty_v5_lsf.dat', format='ascii.fixed_width')
    ref = Table.read('/Users/molly/Dropbox/foggie-collab/plots_halo_008508/nref11n/nref11n_nref10f_refine200kpc/spectra/misty_v5_lsf.dat', format='ascii.fixed_width')

    fig = plt.figure(figsize=(9,7))
    ax = fig.add_subplot(111)
    ax.scatter(nat['impact'], nat['Si_III_Nmin'], marker='D', color='#4daf4a',label='natural')
    ax.scatter(ref['impact'], ref['Si_III_Nmin'], color='#984ea3', label='refined')
    #ax.plot([10,45],[10,1],color='#4daf4a')
    #ax.plot([12,45],[17,4],color='#984ea3')
    plt.legend(loc='upper right')
    plt.xlabel('impact parameter [kpc]')
    plt.ylabel('number of Si III minima')
    fig.tight_layout()
    fig.savefig('SiIII_Nmin_vs_impact.png')

    fig = plt.figure(figsize=(9,7))
    ax = fig.add_subplot(111)
    ax.scatter(nat['impact'], nat['O_VI_Nmin'], marker='D', color='#4daf4a',label='natural')
    ax.scatter(ref['impact'], ref['O_VI_Nmin'], color='#984ea3', label='refined')
    #ax.plot([10,45],[10,1],color='#4daf4a')
    #ax.plot([12,45],[17,4],color='#984ea3')
    plt.legend(loc='upper right')
    plt.xlabel('impact parameter [kpc]')
    plt.ylabel('number of O VI minima')
    fig.tight_layout()
    fig.savefig('OVI_Nmin_vs_impact.png')

    fig = plt.figure(figsize=(9,7))
    ax = fig.add_subplot(111)
    ax.scatter(nat['Si_IV_col'], nat['C_IV_col'], marker='D', color='#4daf4a',label='natural')
    ax.scatter(ref['Si_IV_col'], ref['C_IV_col'], color='#984ea3', label='refined')
    ax.plot([10.0, 14.5], [10.0+0.47712125472, 14.5+0.47712125472],color='black', label = 'CIV/SiIV = 3')
    ax.plot([10.0, 14.5], [10.0+2*0.47712125472, 14.5+2*0.47712125472],color='black', ls=':',label = 'CIV/SiIV = 6')
    plt.legend(loc='lower right')
    plt.xlabel('Si IV column')
    plt.ylabel('C IV column')
    fig.tight_layout()
    fig.savefig('SiIV_vs_CIV_column.png')

    fig = plt.figure(figsize=(9,7))
    ax = fig.add_subplot(111)
    ax.scatter(nat['Si_II_dv90'],nat['Si_II_Ncomp'], marker='D', s=100, color='#4daf4a',label='natural')
    ax.scatter(ref['Si_II_dv90'],ref['Si_II_Ncomp'], color='#984ea3', marker='*',s=100, label='refined')
    plt.legend(loc='lower right')
    plt.xlim(xmin=0)
    plt.ylim(ymin=0)
    plt.xlabel(r'Si II $\Delta v_{90}$')
    plt.ylabel('# of Si II 1260 components')
    fig.tight_layout()
    fig.savefig('SiII_dv90_vs_Ncomp.png')

    fig = plt.figure(figsize=(9,7))
    ax = fig.add_subplot(111)
    ax.scatter(nat['C_II_dv90'],nat['C_II_Ncomp'], marker='D', s=100, color='#4daf4a',label='natural')
    ax.scatter(ref['C_II_dv90'],ref['C_II_Ncomp'], color='#984ea3', marker='*',s=100, label='refined')
    plt.legend(loc='lower right')
    plt.xlim(xmin=0)
    plt.ylim(ymin=0)
    plt.xlabel(r'C II $\Delta v_{90}$')
    plt.ylabel('# of C II 1335 components')
    fig.tight_layout()
    fig.savefig('CII_dv90_vs_Ncomp.png')

if __name__ == "__main__":
    make_misty_plots()
