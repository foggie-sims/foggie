import numpy as np
import sys
import os

import matplotlib as mpl
import seaborn as sns
sns.set_style("whitegrid", {'axes.grid' : False})
mpl.rcParams['font.family'] = 'serif'
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import AxesGrid
import matplotlib.image as mpimg

from astropy.table import Table
from astropy.io import fits

import pickle
from functools import partial
import datashader as dshader
import datashader.transfer_functions as tf
from datashader import reductions
from datashader.utils import export_image
import pandas as pd

from utils.get_refine_box import get_refine_box
from utils.get_proper_box_size import get_proper_box_size
from consistency import *


def make_metal_pickle():
    import yt
    forced_ds = yt.load("/Users/molly/foggie/halo_008508/nref11n/nref11n_nref10f_refine200kpc/RD0020/RD0020")
    track_name = "/Users/molly/foggie/halo_008508/nref11n/nref11n_nref10f_refine200kpc/halo_track"
    output_dir = "/Users/molly/Dropbox/foggie-collab/plots_halo_008508/nref11n/comparisons/"
    natural_ds = yt.load("/Users/molly/foggie/halo_008508/nref11n/natural/RD0020/RD0020")
    ## output_dir = "/Users/molly/Dropbox/foggie-collab/plots/halo_008508/natural/nref11/spectra/"
    os.chdir(output_dir)
    track = Table.read(track_name, format='ascii')
    track.sort('col1')
    zsnap = forced_ds.get_parameter('CosmologyCurrentRedshift')

    proper_box_size = get_proper_box_size(forced_ds)
    forced_box, forced_c, width = get_refine_box(forced_ds, zsnap, track)
    natural_box, natural_c, width = get_refine_box(natural_ds, zsnap, track)
    width = width * proper_box_size

    print("width = ", width, "kpc")

    resolution = (1048,1048)
    radii = np.zeros(resolution)
    indices = np.indices(resolution)
    for x in range(0, 1048):
        for y in range(0, 1048):
            radii[x,y] = np.sqrt((524-x)**2 + (524-y)**2)
    big_radii = np.concatenate((radii, radii, radii), axis=None)

    temp_metalr = []
    for axis in ('x','y','z'):
        dph_forced = yt.ProjectionPlot(forced_ds,axis,('gas','metallicity'), weight_field=('gas','H_p0_number_density'),
                                center=forced_c, \
                                width=(width,"kpc"), data_source=forced_box)
        frb = dph_forced.data_source.to_frb((width,'kpc'), resolution)
        metal_forced = np.array(np.log10(frb['metallicity']))
        temp_metalr.append(metal_forced)
    metalr = np.concatenate((temp_metalr[0], temp_metalr[1], temp_metalr[2]), axis=None)
    metalr[metalr == -np.inf] = 1

    
def make_pickles():
    import yt
    forced_ds = yt.load("/Users/molly/foggie/halo_008508/nref11n/nref11n_nref10f_refine200kpc/RD0020/RD0020")
    track_name = "/Users/molly/foggie/halo_008508/nref11n/nref11n_nref10f_refine200kpc/halo_track"
    output_dir = "/Users/molly/Dropbox/foggie-collab/plots_halo_008508/nref11n/comparisons/"
    natural_ds = yt.load("/Users/molly/foggie/halo_008508/nref11n/natural/RD0020/RD0020")
    ## output_dir = "/Users/molly/Dropbox/foggie-collab/plots/halo_008508/natural/nref11/spectra/"
    os.chdir(output_dir)
    track = Table.read(track_name, format='ascii')
    track.sort('col1')
    zsnap = forced_ds.get_parameter('CosmologyCurrentRedshift')

    proper_box_size = get_proper_box_size(forced_ds)
    forced_box, forced_c, width = get_refine_box(forced_ds, zsnap, track)
    natural_box, natural_c, width = get_refine_box(natural_ds, zsnap, track)
    width = width * proper_box_size

    print("width = ", width, "kpc")

    resolution = (1048,1048)
    radii = np.zeros(resolution)
    indices = np.indices(resolution)
    for x in range(0, 1048):
        for y in range(0, 1048):
            radii[x,y] = np.sqrt((524-x)**2 + (524-y)**2)
    big_radii = np.concatenate((radii, radii, radii), axis=None)

    import trident
    trident.add_ion_fields(forced_ds, ions=['O VI'])
    trident.add_ion_fields(natural_ds, ions=['O VI'])
    trident.add_ion_fields(forced_ds, ions=['C IV'])
    trident.add_ion_fields(natural_ds, ions=['C IV'])
    trident.add_ion_fields(forced_ds, ions=['Si II'])
    trident.add_ion_fields(natural_ds, ions=['Si II'])
    trident.add_ion_fields(forced_ds, ions=['Si IV'])
    trident.add_ion_fields(natural_ds, ions=['Si IV'])

    ion = 'H_p0_number_density'
    temp_hicoln = []
    for axis in ('x','y','z'):
        dph_natural = yt.ProjectionPlot(natural_ds,axis,('gas',ion), center=natural_c, \
                                width=(width,"kpc"), data_source=natural_box)
        frb = dph_natural.data_source.to_frb((width,'kpc'), resolution)
        hi_natural = np.array(np.log10(frb[ion]))
        temp_hicoln.append(hi_natural)
    hicoln = np.concatenate((temp_hicoln[0], temp_hicoln[1], temp_hicoln[2]), axis=None)
    hicoln[hicoln == -np.inf] = 1
    temp_hicolr = []
    for axis in ('x','y','z'):
        dph_forced = yt.ProjectionPlot(forced_ds,axis,('gas',ion), center=forced_c, \
                                width=(width,"kpc"), data_source=forced_box)
        frb = dph_forced.data_source.to_frb((width,'kpc'), resolution)
        hi_forced = np.array(np.log10(frb[ion]))
        temp_hicolr.append(hi_forced)
    hicolr = np.concatenate((temp_hicolr[0], temp_hicolr[1], temp_hicolr[2]), axis=None)
    hicolr[hicolr == -np.inf] = 1

    ion = 'Si_p1_number_density'
    temp_si2coln = []
    for axis in ('x','y','z'):
        dph_natural = yt.ProjectionPlot(natural_ds,axis,('gas',ion), center=natural_c, \
                                width=(width,"kpc"), data_source=natural_box)
        frb = dph_natural.data_source.to_frb((width,'kpc'), resolution)
        si2_natural = np.array(np.log10(frb[ion]))
        temp_si2coln.append(si2_natural.ravel())
    si2coln = np.concatenate((temp_si2coln[0], temp_si2coln[1], temp_si2coln[2]), axis=None)
    si2coln[si2coln == -np.inf] = 1
    temp_si2colr = []
    for axis in ('x','y','z'):
        dph_forced = yt.ProjectionPlot(forced_ds,axis,('gas',ion), center=forced_c, \
                                width=(width,"kpc"), data_source=forced_box)
        frb = dph_forced.data_source.to_frb((width,'kpc'), resolution)
        si2_forced = np.array(np.log10(frb[ion]))
        temp_si2colr.append(si2_forced.ravel())
    si2colr = np.concatenate((temp_si2colr[0], temp_si2colr[1], temp_si2colr[2]), axis=None)
    si2colr[si2colr == -np.inf] = 1

    ion = 'Si_p3_number_density'
    temp_si4coln = []
    for axis in ('x','y','z'):
        dph_natural = yt.ProjectionPlot(natural_ds,axis,('gas',ion), center=natural_c, \
                                width=(width,"kpc"), data_source=natural_box)
        frb = dph_natural.data_source.to_frb((width,'kpc'), resolution)
        si4_natural = np.array(np.log10(frb[ion]))
        temp_si4coln.append(si4_natural.ravel())
    si4coln = np.concatenate((temp_si4coln[0], temp_si4coln[1], temp_si4coln[2]), axis=None)
    si4coln[si4coln == -np.inf] = 1
    temp_si4colr = []
    for axis in ('x','y','z'):
        dph_forced = yt.ProjectionPlot(forced_ds,axis,('gas',ion), center=forced_c, \
                                width=(width,"kpc"), data_source=forced_box)
        frb = dph_forced.data_source.to_frb((width,'kpc'), resolution)
        si4_forced = np.array(np.log10(frb[ion]))
        temp_si4colr.append(si4_forced.ravel())
    si4colr = np.concatenate((temp_si4colr[0], temp_si4colr[1], temp_si4colr[2]), axis=None)
    si4colr[si4colr == -np.inf] = 1


    ion = 'C_p3_number_density'
    temp_c4coln = []
    for axis in ('x','y','z'):
        dph_natural = yt.ProjectionPlot(natural_ds,axis,('gas',ion), center=natural_c, \
                                width=(width,"kpc"), data_source=natural_box)
        frb = dph_natural.data_source.to_frb((width,'kpc'), resolution)
        c4_natural = np.array(np.log10(frb[ion]))
        temp_c4coln.append(c4_natural.ravel())
    c4coln = np.concatenate((temp_c4coln[0], temp_c4coln[1], temp_c4coln[2]), axis=None)
    c4coln[c4coln == -np.inf] = 1
    temp_c4colr = []
    for axis in ('x','y','z'):
        dph_forced = yt.ProjectionPlot(forced_ds,axis,('gas',ion), center=forced_c, \
                                width=(width,"kpc"), data_source=forced_box)
        frb = dph_forced.data_source.to_frb((width,'kpc'), resolution)
        c4_forced = np.array(np.log10(frb[ion]))
        temp_c4colr.append(c4_forced.ravel())
    c4colr = np.concatenate((temp_c4colr[0], temp_c4colr[1], temp_c4colr[2]), axis=None)
    c4colr[c4colr == -np.inf] = 1

    ion = 'O_p5_number_density'
    temp_o6coln = []
    for axis in ('x','y','z'):
        dph_natural = yt.ProjectionPlot(natural_ds,axis,('gas',ion), center=natural_c, \
                                width=(width,"kpc"), data_source=natural_box)
        frb = dph_natural.data_source.to_frb((width,'kpc'), resolution)
        o6_natural = np.array(np.log10(frb[ion]))
        temp_o6coln.append(o6_natural.ravel())
    o6coln = np.concatenate((temp_o6coln[0], temp_o6coln[1], temp_o6coln[2]), axis=None)
    o6coln[o6coln == -np.inf] = 1
    temp_o6colr = []
    for axis in ('x','y','z'):
        dph_forced = yt.ProjectionPlot(forced_ds,axis,('gas',ion), center=forced_c, \
                                width=(width,"kpc"), data_source=forced_box)
        frb = dph_forced.data_source.to_frb((width,'kpc'), resolution)
        o6_forced = np.array(np.log10(frb[ion]))
        temp_o6colr.append(o6_forced.ravel())
    o6colr = np.concatenate((temp_o6colr[0], temp_o6colr[1], temp_o6colr[2]), axis=None)
    o6colr[o6colr == -np.inf] = 1

def make_plots():
    output_dir = "/Users/molly/Dropbox/foggie-collab/plots_halo_008508/nref11n/comparisons/cddf/"

    ## screw it, let's brute force this
    pkl_name = 'radii_RD0018_physicalkpc.pkl'
    poop = pickle.load(open( pkl_name, "rb" ) )
    pkl_name = 'radii_RD0020_physicalkpc.pkl'
    fart = pickle.load(open( pkl_name, "rb" ) )
    radii = np.append(poop,fart,axis=None)
    print('max radius = ',np.max(radii))

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

    hi_cat_natural = categorize_by_hi(hi_coln)
    data_frame_natural = pd.DataFrame({'hi':hi_cat_natural, 'si2':si2_coln,
                                        'si4':si4_coln, 'c4':c4_coln, 'o6':o6_coln, 'radii':radii})
    data_frame_natural.hi = data_frame_natural.hi.astype('category')
    hi_cat_forced = categorize_by_hi(hi_colr)
    data_frame_forced = pd.DataFrame({'hi':hi_cat_forced, 'si2':si2_colr,
                                        'si4':si4_colr, 'c4':c4_colr, 'o6':o6_colr, 'radii':radii})
    data_frame_forced.hi = data_frame_forced.hi.astype('category')


    fig = plt.figure(figsize=(7,6))
    x_min, x_max, xstep = 0, 65, 10
    y_min, y_max, ystep = 8, 20, 3
    width, height = 600, 600
    cvs = dshader.Canvas(plot_width=width, plot_height=height,
                         y_range=(y_min, y_max),
                         x_range=(x_min, x_max))
    agg = cvs.points(data_frame_natural, 'radii', 'si2', dshader.count_cat('hi'))
    img = tf.shade(agg, color_key=hi_color_key, how='linear',min_alpha=255)
    export = partial(export_image, background='white', export_path="./")
    export(img, 'temp')
    new_img = mpimg.imread('temp.png')
    ax = fig.add_axes([0.13, 0.13, 0.85, 0.85])
    ax.imshow(np.flip(new_img[:,:,0:3],0), alpha=1.)
    #ax.set_ylim(9,19)
    ax.set_xlabel('radius [physical kpc]',fontsize=16)
    ax.set_ylabel('log Si II column density',fontsize=16)
    ax.set_xticks(np.arange((x_max - x_min) + 1., step=xstep) * width / (x_max - x_min))
    ax.set_xticklabels([ str(int(s)) for s in \
        np.arange((x_max - x_min) + 1., step=xstep) +  x_min ], fontsize=20)
    ax.set_yticks(np.arange((y_max - y_min) + 1., step=ystep) * height / (y_max - y_min))
    ax.set_yticklabels([ str(int(s)) for s in \
        np.arange((y_max - y_min) + 1., step=ystep) + y_min], fontsize=20)
    #cb = fig.colorbar(sc)
    #cb.set_label('log HI column density')
    plotname = output_dir + 'SiII_radius_natural.png'
    plt.savefig(plotname)


    fig = plt.figure(figsize=(7,6))
    x_min, x_max, xstep = 0, 65, 10
    y_min, y_max, ystep = 8, 20, 3
    width, height = 600, 600
    cvs = dshader.Canvas(plot_width=width, plot_height=height,
                         y_range=(y_min, y_max),
                         x_range=(x_min, x_max))
    agg = cvs.points(data_frame_forced, 'radii', 'si2', dshader.count_cat('hi'))
    img = tf.shade(agg, color_key=hi_color_key, how='linear',min_alpha=255)
    export = partial(export_image, background='white', export_path="./")
    export(img, 'temp')
    new_img = mpimg.imread('temp.png')
    ax = fig.add_axes([0.13, 0.13, 0.85, 0.85])
    ax.imshow(np.flip(new_img[:,:,0:3],0), alpha=1.)
    #ax.set_ylim(9,19)
    ax.set_xlabel('radius [physical kpc]',fontsize=16)
    ax.set_ylabel('log Si II column density',fontsize=16)
    ax.set_xticks(np.arange((x_max - x_min) + 1., step=xstep) * width / (x_max - x_min))
    ax.set_xticklabels([ str(int(s)) for s in \
        np.arange((x_max - x_min) + 1., step=xstep) +  x_min ], fontsize=20)
    ax.set_yticks(np.arange((y_max - y_min) + 1., step=ystep) * height / (y_max - y_min))
    ax.set_yticklabels([ str(int(s)) for s in \
        np.arange((y_max - y_min) + 1., step=ystep) + y_min], fontsize=20)
    #cb = fig.colorbar(sc)
    #cb.set_label('log HI column density')
    plotname = output_dir + 'SiII_radius_forced.png'
    plt.savefig(plotname)



    fig = plt.figure(figsize=(7,6))
    x_min, x_max, xstep = 0, 65, 10
    y_min, y_max, ystep = 10, 16, 2
    width, height = 600, 600
    cvs = dshader.Canvas(plot_width=width, plot_height=height,
                         y_range=(y_min, y_max),
                         x_range=(x_min, x_max))
    agg = cvs.points(data_frame_natural, 'radii', 'c4', dshader.count_cat('hi'))
    img = tf.shade(agg, color_key=hi_color_key, how='linear',min_alpha=255)
    export = partial(export_image, background='white', export_path="./")
    export(img, 'temp')
    new_img = mpimg.imread('temp.png')
    ax = fig.add_axes([0.13, 0.13, 0.85, 0.85])
    ax.imshow(np.flip(new_img[:,:,0:3],0), alpha=1.)
    #ax.set_ylim(9,19)
    ax.set_xlabel('radius [physical kpc]',fontsize=16)
    ax.set_ylabel('log C IV column density',fontsize=16)
    ax.set_xticks(np.arange((x_max - x_min) + 1., step=xstep) * width / (x_max - x_min))
    ax.set_xticklabels([ str(int(s)) for s in \
        np.arange((x_max - x_min) + 1., step=xstep) +  x_min ], fontsize=20)
    ax.set_yticks(np.arange((y_max - y_min) + 1., step=ystep) * height / (y_max - y_min))
    ax.set_yticklabels([ str(int(s)) for s in \
        np.arange((y_max - y_min) + 1., step=ystep) + y_min], fontsize=20)
    #cb = fig.colorbar(sc)
    #cb.set_label('log HI column density')
    plotname = output_dir + 'CIV_radius_natural.png'
    plt.savefig(plotname)


    fig = plt.figure(figsize=(7,6))
    x_min, x_max, xstep = 0, 65, 10
    y_min, y_max, ystep = 10, 16, 2
    width, height = 600, 600
    cvs = dshader.Canvas(plot_width=width, plot_height=height,
                         y_range=(y_min, y_max),
                         x_range=(x_min, x_max))
    agg = cvs.points(data_frame_forced, 'radii', 'c4', dshader.count_cat('hi'))
    img = tf.shade(agg, color_key=hi_color_key, how='linear',min_alpha=255)
    export = partial(export_image, background='white', export_path="./")
    export(img, 'temp')
    new_img = mpimg.imread('temp.png')
    ax = fig.add_axes([0.13, 0.13, 0.85, 0.85])
    ax.imshow(np.flip(new_img[:,:,0:3],0), alpha=1.)
    #ax.set_ylim(9,19)
    ax.set_xlabel('radius [physical kpc]',fontsize=16)
    ax.set_ylabel('log C IV column density',fontsize=16)
    ax.set_xticks(np.arange((x_max - x_min) + 1., step=xstep) * width / (x_max - x_min))
    ax.set_xticklabels([ str(int(s)) for s in \
        np.arange((x_max - x_min) + 1., step=xstep) +  x_min ], fontsize=20)
    ax.set_yticks(np.arange((y_max - y_min) + 1., step=ystep) * height / (y_max - y_min))
    ax.set_yticklabels([ str(int(s)) for s in \
        np.arange((y_max - y_min) + 1., step=ystep) + y_min], fontsize=20)
    #cb = fig.colorbar(sc)
    #cb.set_label('log HI column density')
    plotname = output_dir + 'CIV_radius_forced.png'
    plt.savefig(plotname)


    ############
    ### OVI ####
    ############
    fig = plt.figure(figsize=(7,12))
    x_min, x_max, xstep = 0, 70, 10
    y_min, y_max, ystep = 12, 16, 1
    width, height = 1000, 1000
    axtop = fig.add_axes([0.12, 0.53, 0.86, 0.43],
                 ylim=(y_min, y_max), xlim=(x_min, x_max))
    axbot = fig.add_axes([0.12, 0.1, 0.86, 0.43],
                   ylim=(y_min, y_max), xlim=(x_min, x_max))
    cvs = dshader.Canvas(plot_width=width, plot_height=height,
                         y_range=(y_min, y_max),
                         x_range=(x_min, x_max))
    agg = cvs.points(data_frame_natural, 'radii', 'o6', dshader.count_cat('hi'))
    img = tf.shade(agg, color_key=hi_color_key, how='linear',min_alpha=255)
    export = partial(export_image, background='white', export_path="./")
    export(img, 'temp')
    new_img = mpimg.imread('temp.png')
    ## ax = fig.add_axes([0.13, 0.13, 0.85, 0.85])
    axtop.imshow(np.flip(new_img[:,:,0:3],0), alpha=1.)
    #axtop.set_ylim(9,19)
    ##axtop.set_xlabel('radius [physical kpc]',fontsize=16)
    #axtop.set_ylabel('log O VI column density',fontsize=16)
    axtop.set_xticks(np.arange((x_max - x_min) + 1., step=xstep) * width / (x_max - x_min))
    axtop.set_xticklabels([ ])
    axtop.set_yticks(np.arange((y_max - y_min) + 1., step=ystep) * height / (y_max - y_min))
    axtop.set_yticklabels([ str(int(s)) for s in \
        np.arange((y_max - y_min) + 1., step=ystep) + y_min], fontsize=20)
    cvs = dshader.Canvas(plot_width=width, plot_height=height,
                         y_range=(y_min, y_max),
                         x_range=(x_min, x_max))
    agg = cvs.points(data_frame_forced, 'radii', 'o6', dshader.count_cat('hi'))
    img = tf.shade(agg, color_key=hi_color_key, how='linear',min_alpha=255)
    export = partial(export_image, background='white', export_path="./")
    export(img, 'temp')
    new_img = mpimg.imread('temp.png')
    axbot.imshow(np.flip(new_img[:,:,0:3],0), alpha=1.)
    axbot.set_xticks(np.arange((x_max - x_min) + 1., step=xstep) * width / (x_max - x_min))
    axbot.set_xticklabels([ str(int(s)) for s in \
        np.arange((x_max - x_min) + 1., step=xstep) +  x_min ], fontsize=20)
    axbot.set_yticks(np.arange((y_max - y_min) + 1., step=ystep) * height / (y_max - y_min))
    axbot.set_yticklabels([ str(int(s)) for s in \
        np.arange((y_max - y_min), step=ystep) + y_min], fontsize=20)
    axbot.set_xlabel('radius [physical kpc]',fontsize=26)
    fig.text(0.95, 0.92, 'standard resolution', fontsize=24, ha='right')
    fig.text(0.95, 0.49, 'high resolution', fontsize=24, ha='right')
    fig.text(0.01, 0.5, r'log O VI column density [cm$^{-2}$]', fontsize=26, va='center', rotation='vertical')
    #cb = fig.colorbar(sc)
    #cb.set_label('log HI column density')
    plotname = output_dir + 'OVI_radius_combined.png'
    plt.savefig(plotname)




if __name__ == "__main__":
    make_plots()
    sys.exit("~~~*~*~*~*~*~all done!!!! yay column densities!")
