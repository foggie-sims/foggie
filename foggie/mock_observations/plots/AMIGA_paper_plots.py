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

from foggie.utils.get_refine_box import get_refine_box
from foggie.utils.get_proper_box_size import get_proper_box_size
from foggie.utils.consistency import *
from foggie.utils.enzoGalaxyProps import find_rvirial
import foggie.utils.get_halo_info as ghi
import yt
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,AutoMinorLocator)
import csv
from collections import OrderedDict as odict
mpl.rcParams['axes.linewidth']=1
mpl.rcParams['axes.edgecolor']='k'


r200sim = 159. # for RD 00 42
#r200sim = 154. # for RD 00 41
#r200sim = 150. # for RD 00 40


def make_only_radius_pickle():
    forced_ds = yt.load("/astro/simulations/FOGGIE/halo_008508/nref11n_nref10f/RD0042/RD0042")
    track_name = "/Users/raugustin/foggie/foggie/halo_tracks/008508/nref11n_selfshield_15/halo_track_200kpc_nref10"
    track = Table.read(track_name, format='ascii')
    track.sort('col1')
    zsnap = forced_ds.get_parameter('CosmologyCurrentRedshift')

    proper_box_size = get_proper_box_size(forced_ds)
    forced_box, forced_c, width = get_refine_box(forced_ds, zsnap, track)
    width = width * proper_box_size
    r_res_over_r_200 = width /2. /r200sim

    output_dir = "/Users/raugustin/WORK/AMIGA/output/halo_track_600kpc_nref8/"

    forced_ds.add_particle_filter('stars')
    forced_ds.add_particle_filter('darkmatter')

    os.chdir(output_dir)
    track = Table.read(track_name, format='ascii')
    track.sort('col1')
    zsnap = forced_ds.get_parameter('CosmologyCurrentRedshift')

    proper_box_size = get_proper_box_size(forced_ds)
    forced_box, forced_c, width = get_refine_box(forced_ds, zsnap, track)
    width = width * proper_box_size
    print("width = ", width, "kpc")
    rvir = ghi.get_halo_info(forced_ds, track_name)
    print(rvir)

    res=1048
    resolution = (res,res)
    radii = np.zeros(resolution)
    indices = np.indices(resolution)
    for x in range(0, res):
        for y in range(0, res):
            radii[x,y] = ((width / res) * np.sqrt((res/2-x)**2 + (res/2-y)**2)) /r200sim
    print(np.max(radii))
    big_radii = np.concatenate((radii, radii, radii), axis=None)
    pkl_name = 'radii_RD0042_physicalkpc.pkl'
    print("saving to ", pkl_name)
    pickle.dump(big_radii, open( pkl_name, "wb" ) )

def make_pickles_subtract_sat():
    forced_ds = yt.load("/astro/simulations/FOGGIE/halo_008508/nref11n_nref10f/RD0042/RD0042")
    track_name = "/Users/raugustin/foggie/foggie/halo_tracks/008508/nref11n_selfshield_15/halo_track_600kpc_nref8"
    output_dir = "/Users/raugustin/WORK/AMIGA/output/halo_track_600kpc_nref8/"

    os.chdir(output_dir)
    track = Table.read(track_name, format='ascii')
    track.sort('col1')
    zsnap = forced_ds.get_parameter('CosmologyCurrentRedshift')

    proper_box_size = get_proper_box_size(forced_ds)
    forced_box, forced_c, width = get_refine_box(forced_ds, zsnap, track)
    width = width * proper_box_size
    print("width = ", width, "kpc")

    res=1048
    resolution = (res,res


    #### SUBSAT
    sat_cat_file = "/Users/raugustin/WORK/Outputs/plots_halo_008508/nref11n_nref10f/satellite_selection_halo_008508_nref11n_nref10f.npy"
    sat_cat = np.load(sat_cat_file, allow_pickle = True)
    for sat in sat_cat:
        x, y, z = sat['x'], sat['y'], sat['z']
        sat_center = forced_ds.arr([x,y,z], 'kpc')
        from yt.units import kpc
        small_sp = forced_ds.sphere(sat_center, 15 * kpc)
        forced_box -= small_sp


    import trident
    for chosenion in ['O VI','C II','C IV','Si II','Si III','Si IV']:
        trident.add_ion_fields(forced_ds, ions=[chosenion])

    for ion in ['Si_p1_number_density', 'Si_p2_number_density', 'Si_p3_number_density', 'C_p1_number_density', 'C_p3_number_density', 'O_p5_number_density']:

        temp_ioncolr = []


        for axis in ('x','y','z'):

            readinghasfinished = False
            while readinghasfinished == False:
                try:
                    dph_forced = yt.ProjectionPlot(forced_ds,axis,('gas',ion), center=forced_c, \
                                        width=(width,"kpc"), data_source=forced_box)
                    dph_forced.save(ion+'_refined_RD0042')
                    readinghasfinished = True
                except OSError:
                    pass

            frb = dph_forced.data_source.to_frb((width,'kpc'), resolution)
            ion_forced = np.array(np.log10(frb[ion]))
            temp_ioncolr.append(ion_forced.ravel())



        ioncolr = np.concatenate((temp_ioncolr[0], temp_ioncolr[1], temp_ioncolr[2]), axis=None)
        ioncolr[ioncolr == -np.inf] = 1
        pkl_name = ion + '_nref10f_RD0042_column_densities_sat.pkl'
        print("saving to ", pkl_name)
        pickle.dump(ioncolr, open( pkl_name, "wb" ) )


def make_pickles():
    forced_ds = yt.load("/astro/simulations/FOGGIE/halo_008508/nref11n_nref10f/RD0042/RD0042")
    track_name = "/Users/raugustin/foggie/foggie/halo_tracks/008508/nref11n_selfshield_15/halo_track_200kpc_nref10"
    track = Table.read(track_name, format='ascii')
    track.sort('col1')
    zsnap = forced_ds.get_parameter('CosmologyCurrentRedshift')

    proper_box_size = get_proper_box_size(forced_ds)
    forced_box, forced_c, width = get_refine_box(forced_ds, zsnap, track)
    width = width * proper_box_size
    r_res_over_r_200 = width /2. /r200sim
    output_dir = "/Users/raugustin/WORK/AMIGA/output/halo_track_600kpc_nref8/"

    os.chdir(output_dir)
    track = Table.read(track_name, format='ascii')
    track.sort('col1')
    zsnap = forced_ds.get_parameter('CosmologyCurrentRedshift')

    proper_box_size = get_proper_box_size(forced_ds)
    forced_box, forced_c, width = get_refine_box(forced_ds, zsnap, track)
    width = width * proper_box_size
    print("width = ", width, "kpc")

    res=1048
    resolution = (res,res)
    radii = np.zeros(resolution)
    indices = np.indices(resolution)
    for x in range(0, res):
        for y in range(0, res):
            radii[x,y] = (width / res) * np.sqrt((res/2-x)**2 + (res/2-y)**2) / r200sim
    print(np.max(radii))
    big_radii = np.concatenate((radii, radii, radii), axis=None)
    pkl_name = 'radii_RD0042_physicalkpc.pkl'
    print("saving to ", pkl_name)
    pickle.dump(big_radii, open( pkl_name, "wb" ) )

    import trident
    for chosenion in ['O VI','C II','C IV','Si II','Si III','Si IV']:
        trident.add_ion_fields(forced_ds, ions=[chosenion])

    ion = 'H_p0_number_density'
    temp_hicolr = []
    for axis in ('x','y','z'):
        yt.ProjectionPlot(forced_ds,axis,('gas',ion), center=forced_c, \
                          width=(width,"kpc"), data_source=forced_box).save(ion+'_refined_RD0042')
        dph_forced = yt.ProjectionPlot(forced_ds,axis,('gas',ion), center=forced_c, \
                                width=(width,"kpc"), data_source=forced_box)
        frb = dph_forced.data_source.to_frb((width,'kpc'), resolution)
        hi_forced = np.array(np.log10(frb[ion]))
        temp_hicolr.append(hi_forced)
    hicolr = np.concatenate((temp_hicolr[0], temp_hicolr[1], temp_hicolr[2]), axis=None)
    hicolr[hicolr == -np.inf] = 1
    pkl_name = ion + '_nref10f_RD0042_column_densities.pkl'
    print("saving to ", pkl_name)
    pickle.dump(hicolr, open( pkl_name, "wb" ) )


    for ion in ['Si_p1_number_density', 'Si_p2_number_density', 'Si_p3_number_density', 'C_p1_number_density', 'C_p3_number_density', 'O_p5_number_density']:
        temp_ioncolr = []
        for axis in ('x','y','z'):
            yt.ProjectionPlot(forced_ds,axis,('gas',ion), center=forced_c, \
                              width=(width,"kpc"), data_source=forced_box).save(ion+'_refined_RD0042')
            dph_forced = yt.ProjectionPlot(forced_ds,axis,('gas',ion), center=forced_c, \
                                    width=(width,"kpc"), data_source=forced_box)
            frb = dph_forced.data_source.to_frb((width,'kpc'), resolution)
            ion_forced = np.array(np.log10(frb[ion]))
            temp_ioncolr.append(ion_forced.ravel())
        ioncolr = np.concatenate((temp_ioncolr[0], temp_ioncolr[1], temp_ioncolr[2]), axis=None)
        ioncolr[ioncolr == -np.inf] = 1
        pkl_name = ion + '_nref10f_RD0042_column_densities.pkl'
        print("saving to ", pkl_name)
        pickle.dump(ioncolr, open( pkl_name, "wb" ) )



def make_plots():
    forced_ds = yt.load("/astro/simulations/FOGGIE/halo_008508/nref11n_nref10f/RD0042/RD0042")
    track_name = "/Users/raugustin/foggie/foggie/halo_tracks/008508/nref11n_selfshield_15/halo_track_200kpc_nref10"
    track = Table.read(track_name, format='ascii')
    track.sort('col1')
    zsnap = forced_ds.get_parameter('CosmologyCurrentRedshift')

    proper_box_size = get_proper_box_size(forced_ds)
    forced_box, forced_c, width = get_refine_box(forced_ds, zsnap, track)
    width = width * proper_box_size
    r_res_over_r_200 = width /2. /r200sim
    output_dir = "/Users/raugustin/WORK/AMIGA/output/halo_track_600kpc_nref8/"
    os.chdir(output_dir)
    pkl_name = 'radii_RD0042_physicalkpc.pkl'
    coffee = pickle.load(open( pkl_name, "rb" ) )
    pkl_name = 'radii_RD0042_physicalkpc.pkl'
    tea = pickle.load(open( pkl_name, "rb" ) )
    radii = np.append(coffee,tea,axis=None)
    print('max radius = ',np.max(radii))
    print(radii)
    for ion in ['H_p0_number_density','Si_p1_number_density','Si_p2_number_density', 'Si_p3_number_density', 'C_p1_number_density', 'C_p3_number_density', 'O_p5_number_density']:
        print("trying ",ion)
        pkl_name = ion + '_nref10f_RD0042_column_densities.pkl'
        print("opening ", pkl_name)
        coffee = pickle.load(open( pkl_name, "rb" ) )
        pkl_name = ion + '_nref10f_RD0042_column_densities.pkl'
        print("opening ", pkl_name)
        tea = pickle.load(open( pkl_name, "rb" ) )
        if ion == 'H_p0_number_density':
            hi_colr = np.append(coffee,tea,axis=None)
        elif ion == 'Si_p1_number_density':
            si2_colr = np.append(coffee,tea,axis=None)
        elif ion == 'Si_p2_number_density':
            si3_colr = np.append(coffee,tea,axis=None)
        elif ion == 'Si_p3_number_density':
            si4_colr = np.append(coffee,tea,axis=None)
        elif ion == 'C_p1_number_density':
            c2_colr = np.append(coffee,tea,axis=None)
        elif ion == 'C_p3_number_density':
            c4_colr = np.append(coffee,tea,axis=None)
        elif ion == 'O_p5_number_density':
            o6_colr = np.append(coffee,tea,axis=None)

    for ion in ['H_p0_number_density','Si_p1_number_density','Si_p2_number_density', 'Si_p3_number_density', 'C_p1_number_density', 'C_p3_number_density', 'O_p5_number_density']:
        print("trying ",ion)
        pkl_name = ion + '_nref10f_RD0042_column_densities_subsat.pkl'
        print("opening ", pkl_name)
        coffee = pickle.load(open( pkl_name, "rb" ) )
        pkl_name = ion + '_nref10f_RD0042_column_densities_subsat.pkl'
        print("opening ", pkl_name)
        tea = pickle.load(open( pkl_name, "rb" ) )
        if ion == 'H_p0_number_density':
            hi_colr_subsat = np.append(coffee,tea,axis=None)
        elif ion == 'Si_p1_number_density':
            si2colr_subsat = np.append(coffee,tea,axis=None)
        elif ion == 'Si_p2_number_density':
            si3colr_subsat = np.append(coffee,tea,axis=None)
        elif ion == 'Si_p3_number_density':
            si4colr_subsat = np.append(coffee,tea,axis=None)
        elif ion == 'C_p1_number_density':
            c2colr_subsat = np.append(coffee,tea,axis=None)
        elif ion == 'C_p3_number_density':
            c4colr_subsat = np.append(coffee,tea,axis=None)
        elif ion == 'O_p5_number_density':
            o6colr_subsat = np.append(coffee,tea,axis=None)
    hi_cat_forced = categorize_by_hi(hi_colr)
    data_frame_forced = pd.DataFrame({'hi':hi_cat_forced, 'si2':si2_colr, 'si3':si3_colr,
                                        'si4':si4_colr, 'c2':c2_colr, 'c4':c4_colr, 'o6':o6_colr, 'radii':radii})
    data_frame_forced.hi = data_frame_forced.hi.astype('category')
    data_frame_subsat = pd.DataFrame({'hisubsat':hi_cat_forced, 'si2subsat':si2colr_subsat, 'si3subsat':si3colr_subsat,
                                        'si4subsat':si4colr_subsat, 'c2subsat':c2colr_subsat, 'c4subsat':c4colr_subsat, 'o6subsat':o6colr_subsat, 'radii':radii})
    data_frame_subsat.hi = data_frame_forced.hi.astype('category')

    [target_CII, rho_CII, ra_CII, dec_CII, gl_CII, gb_CII, lms_CII, bms_CII, vc_CII, ncol_CII, encol1_CII, encol2_CII, flg_col_CII,
      target_CIV, rho_CIV, ra_CIV, dec_CIV, gl_CIV, gb_CIV, lms_CIV, bms_CIV, vc_CIV, ncol_CIV, encol1_CIV, encol2_CIV, flg_col_CIV,
      target_SiII, rho_SiII, ra_SiII, dec_SiII, gl_SiII, gb_SiII, lms_SiII, bms_SiII, vc_SiII, ncol_SiII, encol1_SiII, encol2_SiII, flg_col_SiII,
      target_SiIII, rho_SiIII, ra_SiIII, dec_SiIII, gl_SiIII, gb_SiIII, lms_SiIII, bms_SiIII, vc_SiIII, ncol_SiIII, encol1_SiIII, encol2_SiIII, flg_col_SiIII,
      target_SiIV, rho_SiIV, ra_SiIV, dec_SiIV, gl_SiIV, gb_SiIV, lms_SiIV, bms_SiIV, vc_SiIV, ncol_SiIV, encol1_SiIV, encol2_SiIV, flg_col_SiIV,
      target_OVI, rho_OVI, ra_OVI, dec_OVI, gl_OVI, gb_OVI, lms_OVI, bms_OVI, vc_OVI, ncol_OVI, encol1_OVI, encol2_OVI, flg_col_OVI,
      target_OI, rho_OI, ra_OI, dec_OI, gl_OI, gb_OI, lms_OI, bms_OI, vc_OI, ncol_OI, encol1_OI, encol2_OI, flg_col_OI
      ] = pickle.load(open('/Users/raugustin/WORK/AMIGA/input/Project_AMIGA_total.pkl','rb'),encoding='latin1')

    fig = plt.figure(figsize=(10,15))
    x_min, x_max, xstep = 0, 3, 1
    y_min, y_max, ystep = 11, 16, 1
    width, height = 1000, 400

    axc2 = fig.add_axes([0.1, 0.85, 0.83, 0.15], ylim=(y_min, y_max), xlim=(x_min, x_max))
    axsi2 = fig.add_axes([0.1, 0.70, 0.83, 0.15], ylim=(y_min, y_max), xlim=(x_min, x_max))
    axsi3 = fig.add_axes([0.1, 0.55, 0.83, 0.15], ylim=(y_min, y_max), xlim=(x_min, x_max))
    axsi4 = fig.add_axes([0.1, 0.40, 0.83, 0.15], ylim=(y_min, y_max), xlim=(x_min, x_max))
    axc4 = fig.add_axes([0.1, 0.25, 0.83, 0.15], ylim=(y_min, y_max), xlim=(x_min, x_max))
    axo6 = fig.add_axes([0.1, 0.1, 0.83, 0.15], ylim=(y_min, y_max), xlim=(x_min, x_max))
    for ax in [axc2,axsi2,axsi3,axsi4,axc4,axo6]:
        if ax == axsi2:
            ion='si2'
            ionsubsat = 'si2subsat'
            rho = rho_SiII
            ncol = ncol_SiII
            flgcol = flg_col_SiII
            ylimbottom = 11
            ylimtop = 14.2
        elif ax == axsi3:
            ion='si3'
            ionsubsat = 'si3subsat'
            rho = rho_SiIII
            ncol = ncol_SiIII
            flgcol = flg_col_SiIII
            ylimbottom = 11
            ylimtop = 14.2
        elif ax == axsi4:
            ion='si4'
            ionsubsat = 'si4subsat'
            rho = rho_SiIV
            ncol = ncol_SiIV
            flgcol = flg_col_SiIV
            ylimbottom = 11
            ylimtop = 14.2
        elif ax == axo6:
            ion='o6'
            ionsubsat = 'o6subsat'
            rho = rho_OVI
            ncol = ncol_OVI
            flgcol = flg_col_OVI
            ylimbottom = 12
            ylimtop = 15.2
        elif ax == axc2:
            ion='c2'
            ionsubsat = 'c2subsat'
            rho = rho_CII
            ncol = ncol_CII
            flgcol = flg_col_CII
            ylimbottom = 12
            ylimtop = 15.2
        elif ax == axc4:
            ion='c4'
            ionsubsat = 'c4subsat'
            rho = rho_CIV
            ncol = ncol_CIV
            flgcol = flg_col_CIV
            ylimbottom = 12
            ylimtop = 15.2

        cvs = dshader.Canvas(plot_width=width, plot_height=height, y_range=(y_min, y_max), x_range=(x_min, x_max))
        agg1 = cvs.points(data_frame_forced, 'radii', ion)
        img1 = tf.shade(agg1,cmap=['pink', 'purple'], how='linear',min_alpha=255)
        agg2 = cvs.points(data_frame_subsat, 'radii', ionsubsat)
        img2 = tf.shade(agg2,cmap=['greenyellow', 'green'], how='linear',min_alpha=255)
        img = tf.stack(img1,img2)
        export = partial(export_image, background='white', export_path="./")
        export(img, 'temp')
        new_img = mpimg.imread('temp.png')
        ax.imshow(np.flip(new_img[:,:,0:3],0), alpha=1.0)

        ax.set_xticks(np.arange((x_max - x_min) + 1., step=xstep) * width / (x_max - x_min))
        ax.set_xticklabels([ ])
        ax.set_yticks(np.arange((y_max - y_min) + 1., step=ystep) * height / (y_max - y_min))
        ax.set_yticklabels([ str(int(s)) for s in np.arange((y_max - y_min) + 1., step=ystep) + y_min], fontsize=20)
        ax.errorbar((1/230.)*rho[flgcol==0] * width / (x_max - x_min) ,ncol[flgcol==0] * height / (y_max - y_min) - y_min * height / (y_max  - y_min ), color='#019BD9', fmt='o' ,zorder=10, markeredgecolor='white',markersize=20) # detection
        ax.errorbar((1/230.)*rho[flgcol==-1] * width / (x_max - x_min) ,ncol[flgcol==-1] * height / (y_max - y_min)- y_min * height / (y_max  - y_min ), yerr=25, color='gray', fmt='o', uplims=True ,zorder=10, markeredgecolor='white',markersize=20) # upper limit, nondetection
        ax.errorbar((1/230.)*rho[flgcol==-2] * width / (x_max - x_min) ,ncol[flgcol==-2] * height / (y_max - y_min)- y_min * height / (y_max  - y_min ), yerr=25, color='#019BD9', fmt='o', lolims=True ,zorder=10, markeredgecolor='white',markersize=20) # lower limit, detection

        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        ax.tick_params(axis='both', which='both', direction='in', colors='k', bottom='on', left='on', top='on', right='on', length=7, width=1.5)
        ax.set_ylim(bottom=ylimbottom* height / (y_max - y_min)- y_min * height / (y_max  - y_min ),top=ylimtop* height / (y_max - y_min)- y_min * height / (y_max  - y_min ))
        ax.plot([r_res_over_r_200 * width / (x_max - x_min), r_res_over_r_200 * width / (x_max - x_min)],[0, height], '-k')

    axo6.set_xlabel(r'R / R$_{200}$',fontsize=26)
    axo6.set_xticklabels([ str(int(s)) for s in np.arange((x_max - x_min) + 1., step=xstep) +  x_min ], fontsize=20)
    fig.text(0.9, 0.95, 'C II', fontsize=24, ha='right')
    fig.text(0.9, 0.8, 'Si II', fontsize=24, ha='right')
    fig.text(0.9, 0.65, 'Si III', fontsize=24, ha='right')
    fig.text(0.9, 0.5, 'Si IV', fontsize=24, ha='right')
    fig.text(0.9, 0.35, 'C IV', fontsize=24, ha='right')
    fig.text(0.9, 0.2, 'O VI', fontsize=24, ha='right')

    fig.text(0.01, 0.5, r'log column density [cm$^{-2}$]', fontsize=26, va='center', rotation='vertical')
    plotname = output_dir + 'AMIGA_TOTAL_RD0042_halo_008508_column_densities_R200SCALED.png'
    plt.savefig(plotname)
    plotname = output_dir + 'AMIGA_TOTAL_RD0042_halo_008508_column_densities_R200SCALED.pdf'
    plt.savefig(plotname)



if __name__ == "__main__":
    #make_pickles()
    make_plots()
    sys.exit("done")
