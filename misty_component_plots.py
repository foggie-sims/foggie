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
    nat = Table.read('/Users/molly/Dropbox/foggie-collab/plots_halo_008508/nref11n/natural/spectra/misty_si2_v5_rsp.dat', format='ascii.fixed_width')
    ref = Table.read('/Users/molly/Dropbox/foggie-collab/plots_halo_008508/nref11n/nref11n_nref10f_refine200kpc/spectra/misty_si2_v5_rsp.dat', format='ascii.fixed_width')
    hires = Table.read('/Users/molly/Dropbox/foggie-collab/plots_halo_008508/nref11n/nref11f_refine200kpc/spectra/misty_si2_v5_rsp.dat', format='ascii.fixed_width')
    kodiaq = Table.read('/Users/molly/Dropbox/kodiaq/kodiaq_spectacle_si2.dat', format='ascii.fixed_width')
    print(len(nat), len(ref), len(hires))

    nat_p = nat.to_pandas()
    ref_p = ref.to_pandas()
    hires_p = hires.to_pandas()
    kod_p = kodiaq.to_pandas()

    idn = [nat['tot_col'] > 12]
    idr = [ref['tot_col'] > 12]
    idh = [hires['tot_col'] > 12]
    nat_si2 = nat[idn].to_pandas()
    ref_si2 = ref[idr].to_pandas()
    hires_si2 = hires[idh].to_pandas()

    fig = plt.figure(figsize=(9,7))
    ax = fig.add_subplot(111)
    ax.scatter(ref['comp_col'], np.log10(ref['comp_b']), color='darkorange', alpha=0.5, s=10, label='refined')
    # sns.kdeplot(ref['comp_col'], np.log10(ref['comp_b']),  n_levels=30, shade=True, shade_lowest=False, cmap='Purples')
    #sns.jointplot(ref['comp_col'], np.log10(ref['comp_b']), kind='hex', color='darkorange')
    ax.scatter(nat['comp_col'], np.log10(nat['comp_b']), marker='D', color='#4daf4a',alpha=0.3, s=10, label='natural')
    # sns.kdeplot(nat['comp_col'], np.log10(nat['comp_b']), n_levels=30, shade=True, shade_lowest=False, cmap='Greens', alpha=0.6)
    ax.scatter(kodiaq['comp_col'], np.log10(kodiaq['comp_b']), color='k', marker='*', s=100, alpha=0.7, label='KODIAQ', zorder=200)
    plt.legend(loc='upper right')
    plt.xlabel('Si II component column density')
    plt.ylabel('Si II component log b parameter')
    #plt.ylim(0,30)
    plt.xlim(xmin=10.5)
    plt.ylim(ymax=2.)
    fig.tight_layout()
    fig.savefig('SiII_comp_col_b.png')


if __name__ == "__main__":
    make_misty_component_plots()
