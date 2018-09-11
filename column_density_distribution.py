import numpy as np
import yt
import sys
import os

import pickle

import matplotlib as mpl
import seaborn as sns
sns.set_style("whitegrid", {'axes.grid' : False})
mpl.rcParams['font.family'] = 'stixgeneral'
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import AxesGrid
mpl.rcParams['font.size'] = 20.

from astropy.table import Table
from astropy.io import fits

from get_refine_box import get_refine_box
from get_proper_box_size import get_proper_box_size
from consistency import *

def calc_cddf(**kwargs):
    import trident
    ion = kwargs.get("ion","H I 1216")
    # load the simulation
    forced_ds = yt.load("/Users/molly/foggie/halo_008508/nref11n/nref11n_nref10f_refine200kpc/RD0018/RD0018")
    track_name = "/Users/molly/foggie/halo_008508/nref11n/nref11n_nref10f_refine200kpc/halo_track"
    output_dir = "/Users/molly/Dropbox/foggie-collab/plots_halo_008508/nref11n/comparisons/"
    natural_ds = yt.load("/Users/molly/foggie/halo_008508/nref11n/natural/RD0018/RD0018")
    ## output_dir = "/Users/molly/Dropbox/foggie-collab/plots/halo_008508/natural/nref11/spectra/"
    os.chdir(output_dir)
    track = Table.read(track_name, format='ascii')
    track.sort('col1')
    zsnap = forced_ds.get_parameter('CosmologyCurrentRedshift')

    proper_box_size = get_proper_box_size(forced_ds)
    forced_box, forced_c, width = get_refine_box(forced_ds, zsnap, track)
    natural_box, natural_c, width = get_refine_box(natural_ds, zsnap, track)
    width = width * proper_box_size

    # forced_box = forced_ds.box([xmin, ymin, zmin], [xmax, ymax, zmax])
    # forced_c = forced_ds.arr(halo_center,'code_length')
    # natural_box = natural_ds.box([xmin, ymin, zmin], [xmax, ymax, zmax])
    # natural_c = natural_ds.arr(halo_center,'code_length')
    # width = (197./forced_ds.hubble_constant)/(1+forced_ds.current_redshift)
    print("width = ", width, "kpc")

    res = [1000,1000]

    trident.add_ion_fields(forced_ds, ions=['Si II', 'Si IV', 'C IV', 'O VI'])
    trident.add_ion_fields(natural_ds, ions=['Si II', 'Si IV', 'C IV', 'O VI'])
    # trident.add_ion_fields(forced_ds, ions=['C II'])
    # trident.add_ion_fields(natural_ds, ions=['C II'])

    ## start with HI

    ions = ['H_p0_number_density', 'Si_p1_number_density', 'Si_p3_number_density', 'C_p3_number_density', 'O_p5_number_density']
    fields = []
    for ion in ions:
        field = ('gas', ion)
        fields.append(field)

    dp_forced_x = yt.ProjectionPlot(forced_ds, 'x', fields, center=forced_c, \
                        width=(width,"kpc"), data_source=forced_box)
    dp_forced_y = yt.ProjectionPlot(forced_ds, 'y', fields, center=forced_c, \
                        width=(width,"kpc"), data_source=forced_box)
    dp_forced_z = yt.ProjectionPlot(forced_ds, 'z', fields, center=forced_c, \
                        width=(width,"kpc"), data_source=forced_box)

    dp_natural_x = yt.ProjectionPlot(natural_ds, 'x', fields, center=natural_c, \
                        width=(width,"kpc"), data_source=natural_box)
    dp_natural_y = yt.ProjectionPlot(natural_ds, 'y', fields, center=natural_c, \
                        width=(width,"kpc"), data_source=natural_box)
    dp_natural_z = yt.ProjectionPlot(natural_ds, 'z', fields, center=natural_c, \
                        width=(width,"kpc"), data_source=natural_box)


    for ion in ions:
        colr = np.array([])
        coln = np.array([])
        print("trying ",ion)
        frb = dp_natural_x.frb['gas',ion]
        natural = np.array(np.log10(frb))
        coln = np.append(coln, natural.ravel(), axis=None)

        frb = dp_natural_y.frb['gas',ion]
        natural = np.array(np.log10(frb))
        coln = np.append(coln, natural.ravel(), axis=None)

        frb = dp_natural_z.frb['gas',ion]
        natural = np.array(np.log10(frb))
        coln = np.append(coln, natural.ravel(), axis=None)

        frb = dp_forced_x.frb['gas',ion]
        forced = np.array(np.log10(frb))
        colr = np.append(colr, forced.ravel(), axis=None)

        frb = dp_forced_y.frb['gas',ion]
        forced = np.array(np.log10(frb))
        colr = np.append(colr, forced.ravel(), axis=None)

        frb = dp_forced_z.frb['gas',ion]
        forced = np.array(np.log10(frb))
        colr = np.append(colr, forced.ravel(), axis=None)


        pkl_name = ion + '_nref10f_RD0018_column_densities.pkl'
        print("saving to ", pkl_name)
        pickle.dump(colr, open( pkl_name, "wb" ) )
        pkl_name = ion + '_natural_RD0018_column_densities.pkl'
        print("saving to ", pkl_name)
        pickle.dump(coln, open( pkl_name, "wb" ) )

def plot_cddf():
    kodiaq = Table.read('/Users/molly/Dropbox/kodiaq/kodiaq_all.dat', format='ascii.fixed_width')
    si2_limit = 11.5
    c4_limit = 12
    si4_limit = 13
    o6_limit = 13
    ref_color = 'darkorange' ###  '#4575b4' # purple
    nat_color = '#4daf4a' # green

    output_dir = "/Users/molly/Dropbox/foggie-collab/plots_halo_008508/nref11n/comparisons/cddf/"


    ## screw it, let's brute force this
    ion = 'H_p0_number_density'
    print("trying ",ion)
    pkl_name = ion + '_nref10f_RD0020_column_densities.pkl'
    print("opening ", pkl_name)
    poop = pickle.load(open( pkl_name, "rb" ) )
    pkl_name = ion + '_nref10f_RD0018_column_densities.pkl'
    print("opening ", pkl_name)
    fart = pickle.load(open( pkl_name, "rb" ) )
    hi_colr = np.append(poop,fart,axis=None)

    pkl_name = ion + '_natural_RD0020_column_densities.pkl'
    print("opening ", pkl_name)
    poop = pickle.load(open( pkl_name, "rb" ) )
    pkl_name = ion + '_natural_RD0018_column_densities.pkl'
    print("opening ", pkl_name)
    fart = pickle.load(open( pkl_name, "rb" ) )
    hi_coln = np.append(poop,fart,axis=None)
    hi_colr[hi_colr == -np.inf] = 1
    hi_coln[hi_coln == -np.inf] = 1

    ## now we want just the > 16:
    llsr = [hi_colr > 16]
    llsn = [hi_coln > 16]

    ion = 'Si_p1_number_density'
    print("trying ",ion)
    pkl_name = ion + '_nref10f_RD0020_column_densities.pkl'
    print("opening ", pkl_name)
    poop = pickle.load(open( pkl_name, "rb" ) )
    pkl_name = ion + '_nref10f_RD0018_column_densities.pkl'
    print("opening ", pkl_name)
    fart = pickle.load(open( pkl_name, "rb" ) )
    si2_colr = np.append(poop,fart,axis=None)
    pkl_name = ion + '_natural_RD0020_column_densities.pkl'
    print("opening ", pkl_name)
    poop = pickle.load(open( pkl_name, "rb" ) )
    pkl_name = ion + '_natural_RD0018_column_densities.pkl'
    print("opening ", pkl_name)
    fart = pickle.load(open( pkl_name, "rb" ) )
    si2_coln = np.append(poop,fart,axis=None)
    si2_colr[si2_colr == -np.inf] = 1
    si2_coln[si2_coln == -np.inf] = 1

    ion = 'Si_p3_number_density'
    print("trying ",ion)
    pkl_name = ion + '_nref10f_RD0020_column_densities.pkl'
    print("opening ", pkl_name)
    poop = pickle.load(open( pkl_name, "rb" ) )
    pkl_name = ion + '_nref10f_RD0018_column_densities.pkl'
    print("opening ", pkl_name)
    fart = pickle.load(open( pkl_name, "rb" ) )
    si4_colr = np.append(poop,fart,axis=None)
    pkl_name = ion + '_natural_RD0020_column_densities.pkl'
    print("opening ", pkl_name)
    poop = pickle.load(open( pkl_name, "rb" ) )
    pkl_name = ion + '_natural_RD0018_column_densities.pkl'
    print("opening ", pkl_name)
    fart = pickle.load(open( pkl_name, "rb" ) )
    si4_coln = np.append(poop,fart,axis=None)
    si4_colr[si4_colr == -np.inf] = 1
    si4_coln[si4_coln == -np.inf] = 1

    ion = 'C_p3_number_density'
    print("trying ",ion)
    pkl_name = ion + '_nref10f_RD0020_column_densities.pkl'
    print("opening ", pkl_name)
    poop = pickle.load(open( pkl_name, "rb" ) )
    pkl_name = ion + '_nref10f_RD0018_column_densities.pkl'
    print("opening ", pkl_name)
    fart = pickle.load(open( pkl_name, "rb" ) )
    c4_colr = np.append(poop,fart,axis=None)
    pkl_name = ion + '_natural_RD0020_column_densities.pkl'
    print("opening ", pkl_name)
    poop = pickle.load(open( pkl_name, "rb" ) )
    pkl_name = ion + '_natural_RD0018_column_densities.pkl'
    print("opening ", pkl_name)
    fart = pickle.load(open( pkl_name, "rb" ) )
    c4_coln = np.append(poop,fart,axis=None)
    c4_colr[c4_colr == -np.inf] = 1
    c4_coln[c4_coln == -np.inf] = 1

    ion = 'O_p5_number_density'
    print("trying ",ion)
    pkl_name = ion + '_nref10f_RD0020_column_densities.pkl'
    print("opening ", pkl_name)
    poop = pickle.load(open( pkl_name, "rb" ) )
    pkl_name = ion + '_nref10f_RD0018_column_densities.pkl'
    print("opening ", pkl_name)
    fart = pickle.load(open( pkl_name, "rb" ) )
    o6_colr = np.append(poop,fart,axis=None)
    pkl_name = ion + '_natural_RD0020_column_densities.pkl'
    print("opening ", pkl_name)
    poop = pickle.load(open( pkl_name, "rb" ) )
    pkl_name = ion + '_natural_RD0018_column_densities.pkl'
    print("opening ", pkl_name)
    fart = pickle.load(open( pkl_name, "rb" ) )
    o6_coln = np.append(poop,fart,axis=None)
    o6_colr[o6_colr == -np.inf] = 1
    o6_coln[o6_coln == -np.inf] = 1

    # print(len(o6_colr), len(o6_coln))
    # print(len(np.unique(hi_colr)), len(np.unique(hi_coln)))


    fig, ax = plt.subplots(figsize=(9,6))
    ax.hist(hi_colr,bins=5000,histtype='step',normed=True,cumulative=-1,color=ref_color,label="refined",lw=2)
    ax.hist(hi_coln,bins=5000,histtype='step',normed=True,cumulative=-1,color=nat_color,label="standard",lw=2)
    plt.xlim(15,20)
    plt.ylim(0,0.5)
    plt.legend(loc = 'upper right')
    xlabel = 'log HI column density'
    plt.xlabel(xlabel,fontsize=22)
    plt.ylabel('fraction of sightlines with > N',fontsize=22)
    plt.tight_layout()
    plotname = output_dir + 'HI_cddf.png'
    plt.savefig(plotname)

    fig, ax = plt.subplots(figsize=(9,6))
    ax.hist(si4_colr,bins=5000,histtype='step',normed=True,cumulative=-1,color=ref_color,label="refined",lw=2)
    ax.hist(si4_coln,bins=5000,histtype='step',normed=True,cumulative=-1,color=nat_color,label="standard",lw=2)
    plt.xlim(si4_limit,15)
    plt.ylim(0,0.1)
    plt.legend(loc = 'upper right')
    xlabel = 'log Si IV column density'
    plt.xlabel(xlabel,fontsize=22)
    plt.ylabel('fraction of sightlines with > N',fontsize=22)
    plt.tight_layout()
    plotname = output_dir + 'SiIV_cddf.png'
    plt.savefig(plotname)



    fig = plt.figure(figsize=(9,10))
    axtop = fig.add_axes([0.1, 0.54, 0.88, 0.44],
                   ylim=(0, 0.25), xlim=(si2_limit-1,17.5))
    axbot = fig.add_axes([0.1, 0.1, 0.88, 0.44],
                   ylim=(0, 0.9), xlim=(si2_limit-1,17.5))
    axtop.hist(si2_colr,bins=5000,histtype='step',normed=True,cumulative=-1,color=ref_color,label="refined",lw=2)
    axtop.hist(si2_coln,bins=5000,histtype='step',normed=True,cumulative=-1,color=nat_color,label="standard",lw=2)
    axtop.set_ylabel('fraction of sightlines with > N',fontsize=20)
    axtop.set_xticklabels(())
    axtop.set_yticks((0.0,0.1, 0.2))
    axtop.set_yticklabels(('0','0.1','0.2'))
    axtop.legend(loc = 'upper right')
    axbot.hist(si2_colr[llsr],bins=5000,histtype='step',normed=True,cumulative=-1,color=ref_color,lw=2)
    axbot.hist(si2_coln[llsn],bins=5000,histtype='step',normed=True,cumulative=-1,color=nat_color,lw=2)
    axbot.set_xlabel('log Si II column density',fontsize=22)
    axbot.set_ylabel('fraction of LLS sightlines with > N',fontsize=20)
    plt.tight_layout()
    plotname = output_dir + 'SiII_both_cddf.png'
    plt.savefig(plotname)


    fig = plt.figure(figsize=(9,10))
    axtop = fig.add_axes([0.1, 0.54, 0.88, 0.44],
                   ylim=(0, 0.2), xlim=(si4_limit-1,15.4))
    axbot = fig.add_axes([0.1, 0.1, 0.88, 0.44],
                   ylim=(0, 0.68), xlim=(si4_limit-1,15.4))
    axtop.hist(si4_colr,bins=5000,histtype='step',normed=True,cumulative=-1,color=ref_color,label="refined",lw=2)
    axtop.hist(si4_coln,bins=5000,histtype='step',normed=True,cumulative=-1,color=nat_color,label="standard",lw=2)
    axtop.set_ylabel('fraction of sightlines with > N',fontsize=20)
    axtop.set_xticklabels(())
    axtop.set_yticks((0.0,0.1, 0.2))
    axtop.set_yticklabels(('0','0.1','0.2'))
    axtop.legend(loc = 'upper right')
    axbot.hist(si4_colr[llsr],bins=5000,histtype='step',normed=True,cumulative=-1,color=ref_color,lw=2)
    axbot.hist(si4_coln[llsn],bins=5000,histtype='step',normed=True,cumulative=-1,color=nat_color,lw=2)
    axbot.set_xticks((12,13,14,15))
    axbot.set_xticklabels(('12','13','14','15'))
    axbot.set_xlabel('log Si IV column density',fontsize=22)
    axbot.set_ylabel('fraction of LLS sightlines with > N',fontsize=20)
    plt.tight_layout()
    plotname = output_dir + 'SiIV_both_cddf.png'
    plt.savefig(plotname)


    fig = plt.figure(figsize=(9,10))
    axtop = fig.add_axes([0.1, 0.54, 0.88, 0.44],
                   ylim=(0, 0.85), xlim=(c4_limit-1,15.4))
    axbot = fig.add_axes([0.1, 0.1, 0.88, 0.44],
                   ylim=(0, 1.05), xlim=(c4_limit-1,15.4))
    axtop.hist(c4_colr,bins=5000,histtype='step',normed=True,cumulative=-1,color=ref_color,label="refined",lw=2)
    axtop.hist(c4_coln,bins=5000,histtype='step',normed=True,cumulative=-1,color=nat_color,label="standard",lw=2)
    axtop.set_ylabel('fraction of sightlines with > N',fontsize=20)
    axtop.set_xticklabels(())
    #axtop.set_yticks((0.0,0.1, 0.2))
    #axtop.set_yticklabels(('0','0.1','0.2'))
    axtop.legend(loc = 'upper right')
    axbot.hist(c4_colr[llsr],bins=5000,histtype='step',normed=True,cumulative=-1,color=ref_color,lw=2)
    axbot.hist(c4_coln[llsn],bins=5000,histtype='step',normed=True,cumulative=-1,color=nat_color,lw=2)
    axbot.set_xticks((11,12,13,14,15))
    axbot.set_xticklabels(('11','12','13','14','15'))
    axbot.set_xlabel('log C IV column density',fontsize=22)
    axbot.set_ylabel('fraction of LLS sightlines with > N',fontsize=20)
    plt.tight_layout()
    plotname = output_dir + 'CIV_both_cddf.png'
    plt.savefig(plotname)


    fig = plt.figure(figsize=(9,10))
    axtop = fig.add_axes([0.1, 0.54, 0.88, 0.44],
                   ylim=(0, 1.05), xlim=(o6_limit-1,15))
    axbot = fig.add_axes([0.1, 0.1, 0.88, 0.44],
                   ylim=(0, 1.05), xlim=(o6_limit-1,15))
    axtop.hist(o6_colr,bins=5000,histtype='step',normed=True,cumulative=-1,color=ref_color,label="refined",lw=2)
    axtop.hist(o6_coln,bins=5000,histtype='step',normed=True,cumulative=-1,color=nat_color,label="standard",lw=2)
    axtop.set_ylabel('fraction of sightlines with > N',fontsize=20)
    axtop.set_xticklabels(())
    #axtop.set_yticks((0.0,0.1, 0.2))
    #axtop.set_yticklabels(('0','0.1','0.2'))
    axtop.legend(loc = 'upper right')
    axbot.hist(o6_colr[llsr],bins=5000,histtype='step',normed=True,cumulative=-1,color=ref_color,lw=2)
    axbot.hist(o6_coln[llsn],bins=5000,histtype='step',normed=True,cumulative=-1,color=nat_color,lw=2)
    axbot.set_xticks((12,13,14,15))
    axbot.set_xticklabels(('12','13','14','15'))
    axbot.set_xlabel('log O VI column density',fontsize=22)
    axbot.set_ylabel('fraction of LLS sightlines with > N',fontsize=20)
    plt.tight_layout()
    plotname = output_dir + 'OVI_both_cddf.png'
    plt.savefig(plotname)


    fig, ax = plt.subplots(figsize=(9,6))
    ax.hist(si2_colr,bins=5000,histtype='step',normed=True,cumulative=-1,color=ref_color,label="refined",lw=2)
    ax.hist(si2_coln,bins=5000,histtype='step',normed=True,cumulative=-1,color=nat_color,label="standard",lw=2)
    plt.xlim(si2_limit,19)
    plt.ylim(0,0.3)
    plt.legend(loc = 'upper right')
    xlabel = 'log Si II column density'
    plt.xlabel(xlabel,fontsize=22)
    plt.ylabel('fraction of sightlines with > N',fontsize=22)
    plt.tight_layout()
    plotname = output_dir + 'SiII_cddf.png'
    plt.savefig(plotname)

    fig, ax = plt.subplots(figsize=(9,6))
    ax.hist(si2_colr[llsr],bins=5000,histtype='step',normed=True,cumulative=-1,color=ref_color,label="refined",lw=2)
    ax.hist(si2_coln[llsn],bins=5000,histtype='step',normed=True,cumulative=-1,color=nat_color,label="standard",lw=2)
    plt.xlim(si2_limit,19)
    plt.ylim(0,0.9)
    plt.legend(loc = 'upper right')
    xlabel = 'log Si II column density'
    plt.xlabel(xlabel,fontsize=22)
    plt.ylabel('fraction of LLS sightlines with > N',fontsize=22)
    plt.tight_layout()
    plotname = output_dir + 'SiII_lls_cddf.png'
    plt.savefig(plotname)

    fig, ax = plt.subplots(figsize=(9,6))
    hb = ax.hexbin(hi_colr, si2_colr,cmap='Oranges',extent=(13,22,si2_limit-1.5,19),mincnt=1,gridsize=200, vmin=0, vmax=1000)
    ax.scatter(kodiaq['HI_col'], kodiaq['Si_II_col'], color='k', marker='*', s=100, label='KODIAQ',zorder=100)
    cb = fig.colorbar(hb, ax=ax)
    cb.set_label('counts, refined simulation')
    plt.legend(loc='upper left')
    plt.ylim(ymin=si2_limit-1.5)
    plt.xlabel('log HI column density',fontsize=22)
    plt.ylabel('log Si II column density',fontsize=22)
    plt.tight_layout()
    plotname = output_dir + 'HI_SiII_refined.png'
    plt.savefig(plotname)

    fig, ax = plt.subplots(figsize=(9,6))
    hb = ax.hexbin(hi_coln, si2_coln,cmap='Greens',extent=(13,22,si2_limit-1.5,19),mincnt=1,gridsize=200, vmin=0, vmax=1000)
    ax.scatter(kodiaq['HI_col'], kodiaq['Si_II_col'], color='k', marker='*', s=100, label='KODIAQ',zorder=100)
    cb = fig.colorbar(hb, ax=ax)
    cb.set_label('counts, standard simulation')
    plt.legend(loc='upper left')
    plt.ylim(ymin=si2_limit-1.5)
    plt.xlabel('log HI column density',fontsize=22)
    plt.ylabel('log Si II column density',fontsize=22)
    plt.tight_layout()
    plotname = output_dir + 'HI_SiII_natural.png'
    plt.savefig(plotname)

    fig, ax = plt.subplots(figsize=(9,9))
    hb = ax.hexbin(hi_colr, si4_colr,cmap='Oranges',extent=(13,22,si4_limit-2.5,18),mincnt=1,gridsize=200, vmin=0, vmax=500)
    ax.scatter(kodiaq['HI_col'], kodiaq['Si_IV_col'], color='k', marker='*', s=100, label='KODIAQ',zorder=100)
    cb = fig.colorbar(hb, ax=ax)
    cb.set_label('counts, refined simulation')
    plt.legend(loc='upper left')
    plt.ylim(ymin=si4_limit-2.5)
    plt.xlabel('log HI column density',fontsize=22)
    plt.ylabel('log Si IV column density',fontsize=22)
    plt.tight_layout()
    plotname = output_dir + 'HI_SiIV_refined.png'
    plt.savefig(plotname)

    fig, ax = plt.subplots(figsize=(9,9))
    hb = ax.hexbin(hi_coln, si4_coln,cmap='Greens',extent=(13,22,si4_limit-2.5,18),mincnt=1,gridsize=200, vmin=0, vmax=500)
    ax.scatter(kodiaq['HI_col'], kodiaq['Si_IV_col'], color='k', marker='*', s=100, label='KODIAQ',zorder=100)
    cb = fig.colorbar(hb, ax=ax)
    cb.set_label('counts, standard simulation')
    plt.legend(loc='upper left')
    plt.ylim(ymin=si4_limit-2.5)
    plt.xlabel('log HI column density',fontsize=22)
    plt.ylabel('log Si IV column density',fontsize=22)
    plt.tight_layout()
    plotname = output_dir + 'HI_SiIV_natural.png'
    plt.savefig(plotname)

    fig, ax = plt.subplots(figsize=(9,6))
    hb = ax.hexbin(hi_colr, c4_colr,cmap='Oranges',extent=(13,22,c4_limit-2.5,16),mincnt=1,gridsize=200, vmin=0, vmax=2000)
    ax.scatter(kodiaq['HI_col'], kodiaq['C_IV_col'], color='k', marker='*', s=100, label='KODIAQ',zorder=100)
    cb = fig.colorbar(hb, ax=ax)
    cb.set_label('counts, refined simulation')
    plt.legend(loc='upper left')
    plt.ylim(ymin=c4_limit-2.5)
    plt.xlabel('log HI column density',fontsize=22)
    plt.ylabel('log C IV column density',fontsize=22)
    plt.tight_layout()
    plotname = output_dir + 'HI_CIV_refined.png'
    plt.savefig(plotname)

    fig, ax = plt.subplots(figsize=(9,6))
    hb = ax.hexbin(hi_coln, c4_coln,cmap='Greens',extent=(13,22,c4_limit-2.5,16),mincnt=1,gridsize=200, vmin=0, vmax=2000)
    ax.scatter(kodiaq['HI_col'], kodiaq['C_IV_col'], color='k', marker='*', s=100, label='KODIAQ',zorder=100)
    cb = fig.colorbar(hb, ax=ax)
    cb.set_label('counts, standard simulation')
    plt.legend(loc='upper left')
    plt.ylim(ymin=c4_limit-2.5)
    plt.xlabel('log HI column density',fontsize=22)
    plt.ylabel('log C IV column density',fontsize=22)
    plt.tight_layout()
    plotname = output_dir + 'HI_CIV_natural.png'
    plt.savefig(plotname)

    fig, ax = plt.subplots(figsize=(9,6))
    hb = ax.hexbin(hi_colr, o6_colr,cmap='Oranges',extent=(13,22,o6_limit-1.5,16),mincnt=1,gridsize=200, vmin=0, vmax=2000)
    ax.scatter(kodiaq['HI_col'], kodiaq['O_VI_col'], color='k', marker='*', s=100, label='KODIAQ',zorder=100)
    cb = fig.colorbar(hb, ax=ax)
    cb.set_label('counts, refined simulation')
    plt.legend(loc='upper left')
    plt.ylim(ymin=o6_limit-1.5)
    plt.xlabel('log HI column density',fontsize=22)
    plt.ylabel('log O VI column density',fontsize=22)
    plt.tight_layout()
    plotname = output_dir + 'HI_OVI_refined.png'
    plt.savefig(plotname)

    fig, ax = plt.subplots(figsize=(9,6))
    hb = ax.hexbin(hi_coln, o6_coln,cmap='Greens',extent=(13,22,o6_limit-1.5,16),mincnt=1,gridsize=200, vmin=0, vmax=2000)
    ax.scatter(kodiaq['HI_col'], kodiaq['O_VI_col'], color='k', marker='*', s=100, label='KODIAQ',zorder=100)
    cb = fig.colorbar(hb, ax=ax)
    cb.set_label('counts, standard simulation')
    plt.legend(loc='upper left')
    plt.ylim(ymin=o6_limit-1.5)
    plt.xlabel('log HI column density',fontsize=22)
    plt.ylabel('log O VI column density',fontsize=22)
    plt.tight_layout()
    plotname = output_dir + 'HI_OVI_natural.png'
    plt.savefig(plotname)


    fig, ax = plt.subplots(figsize=(9,6))
    hb = ax.hexbin(si2_colr, si4_colr, cmap='Oranges',extent=(si2_limit-1.5,19,si4_limit-1.5,16),mincnt=1,gridsize=200, vmin=0, vmax=500)
    ax.scatter(kodiaq['Si_II_col'], kodiaq['Si_IV_col'], color='k', marker='*', s=100, label='KODIAQ',zorder=100)
    cb = fig.colorbar(hb, ax=ax)
    cb.set_label('counts, refined simulation')
    plt.legend(loc='upper left')
    plt.ylim(ymin=si4_limit-1.5)
    plt.xlim(xmin=si2_limit-1.5)
    plt.xlabel('log Si II column density',fontsize=22)
    plt.ylabel('log Si IV column density',fontsize=22)
    plt.tight_layout()
    plotname = output_dir + 'SiII_SiIV_refined.png'
    plt.savefig(plotname)

    fig, ax = plt.subplots(figsize=(9,6))
    hb = ax.hexbin(si2_coln, si4_coln,cmap='Greens',extent=(si2_limit-1.5,19,si4_limit-1.5,16),mincnt=1,gridsize=200, vmin=0, vmax=500)
    ax.scatter(kodiaq['Si_II_col'], kodiaq['Si_IV_col'], color='k', marker='*', s=100, label='KODIAQ',zorder=100)
    cb = fig.colorbar(hb, ax=ax)
    cb.set_label('counts, standard simulation')
    plt.legend(loc='upper left')
    plt.ylim(ymin=si4_limit-1.5)
    plt.xlim(xmin=si2_limit-1.5)
    plt.xlabel('log Si II column density',fontsize=22)
    plt.ylabel('log Si IV column density',fontsize=22)
    plt.tight_layout()
    plotname = output_dir + 'SiII_SiIV_natural.png'
    plt.savefig(plotname)


    fig, ax = plt.subplots(figsize=(9,6))
    hb = ax.hexbin(c4_colr, si4_colr, cmap='Oranges',extent=(c4_limit-1.5,16,si4_limit-1.5,16),mincnt=1,gridsize=200, vmin=0, vmax=500)
    ax.scatter(kodiaq['C_IV_col'], kodiaq['Si_IV_col'], color='k', marker='*', s=100, label='KODIAQ',zorder=100)
    cb = fig.colorbar(hb, ax=ax)
    cb.set_label('counts, refined simulation')
    plt.legend(loc='upper left')
    plt.ylim(ymin=si4_limit-1.5)
    plt.xlim(xmin=c4_limit-1.5)
    plt.xlabel('log C IV column density',fontsize=22)
    plt.ylabel('log Si IV column density',fontsize=22)
    plt.tight_layout()
    plotname = output_dir + 'CIV_SiIV_refined.png'
    plt.savefig(plotname)

    fig, ax = plt.subplots(figsize=(9,6))
    hb = ax.hexbin(c4_coln, si4_coln,cmap='Greens',extent=(c4_limit-1.5,16,si4_limit-1.5,16),mincnt=1,gridsize=200, vmin=0, vmax=500)
    ax.scatter(kodiaq['C_IV_col'], kodiaq['Si_IV_col'], color='k', marker='*', s=100, label='KODIAQ',zorder=100)
    cb = fig.colorbar(hb, ax=ax)
    cb.set_label('counts, standard simulation')
    plt.legend(loc='upper left')
    plt.ylim(ymin=si4_limit-1.5)
    plt.xlim(xmin=c4_limit-1.5)
    plt.xlabel('log C IV column density',fontsize=22)
    plt.ylabel('log Si IV column density',fontsize=22)
    plt.tight_layout()
    plotname = output_dir + 'CIV_SiIV_natural.png'
    plt.savefig(plotname)




if __name__ == "__main__":
    ## calc_cddf()
    plot_cddf()
    sys.exit("~~~*~*~*~*~*~all done!!!! yay column densities!")
