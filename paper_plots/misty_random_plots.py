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
mpl.rcParams['font.size'] = 16.
import matplotlib.pyplot as plt


def make_misty_plots():
    ref_color = 'darkorange' ###  '#4575b4' # purple
    nat_color = '#4daf4a' # green
    palette = sns.blend_palette((nat_color, ref_color),n_colors=2)
    si2_limit = 12
    c4_limit = 12
    si4_limit = 13
    o6_limit = 13

    # for now, be lazy and hardcode the paths to the files
    nat = Table.read('/Users/molly/Dropbox/foggie-collab/plots_halo_008508/nref11n/natural/spectra/random/misty_v6_lsf_random.dat', format='ascii.fixed_width')
    ref = Table.read('/Users/molly/Dropbox/foggie-collab/plots_halo_008508/nref11n/nref11n_nref10f_refine200kpc/spectra/random/misty_v6_lsf_random.dat', format='ascii.fixed_width')
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
    plt.xlabel('impact parameter [kpc]')
    plt.ylabel('number of Si II minima')
    fig.tight_layout()
    fig.savefig('SiII_Nmin_vs_impact_random.png')

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
    plt.xlabel('impact parameter [kpc]')
    plt.ylabel('number of O VI minima')
    fig.tight_layout()
    fig.savefig('OVI_Nmin_vs_impact_random.png')

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
    plt.xlabel('# of Si II 1260 minima')
    plt.ylabel('normalized fraction')
    fig.tight_layout()
    fig.savefig('Si_II_histograms_random.png')

    fig = plt.figure(figsize=(11,7))
    ax = fig.add_subplot(111)
    sns.swarmplot(x="Si_II_col", y="Si_II_Nmin", data=nat_si2, color=nat_color,alpha=0.7,orient='h')
    sns.swarmplot(x="Si_II_col", y="Si_II_Nmin", data=ref_si2, color=ref_color,alpha=0.7,orient='h')
    #ax.scatter(nat['HI_col'],nat['Si_II_Nmin'], marker='D', s=60, color=nat_color,alpha=0.5,label='standard')
    #ax.scatter(ref['HI_col'],ref['Si_II_Nmin'], color=ref_color, marker='o',s=100, alpha=0.5,label='refined')
    ax.scatter(kodiaq['Si_II_col'], kodiaq['Si_II_Nmin'], color='k', marker='*', s=100, alpha=0.7, label='KODIAQ',zorder=100)
    plt.legend(loc='upper left', frameon=False)
    plt.xlim(si2_limit,17)
    plt.ylim(0.5,14.5)
    plt.xlabel(r'log SiII column density')
    plt.ylabel('# of Si II 1260 minima')
    fig.tight_layout()
    fig.savefig('SiII_col_vs_Nmin_random.png')

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
    plt.xlabel('# of Si IV 1260 minima')
    plt.ylabel('normalized fraction')
    fig.tight_layout()
    fig.savefig('Si_IV_histograms_random.png')

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
    plt.xlabel('# of C IV 1548 minima')
    plt.ylabel('normalized fraction')
    fig.tight_layout()
    fig.savefig('C_IV_histograms_random.png')

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
    plt.xlabel('# of O VI 1032 minima')
    plt.ylabel('normalized fraction')
    fig.tight_layout()
    fig.savefig('O_VI_histograms_random.png')

    ## KS-test
#    for key in ['Si_II_Nmin','Si_IV_Nmin', 'C_IV_Nmin', 'O_VI_Nmin']:
#        print(key, 'nat vs. KODIAQ:', stats.ks_2samp(np.array(nat[key][key > 0])[0], np.array(kodiaq[key][key > 0])[0]))
#        print(key, 'ref vs. KODIAQ:', stats.ks_2samp(np.array(ref[key][key > 0])[0], np.array(kodiaq[key][key > 0])[0]))


if __name__ == "__main__":
    make_misty_plots()
