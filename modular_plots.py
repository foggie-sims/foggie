from __future__ import print_function

import numpy as np

import yt
import trident

import argparse
import os
import glob
import sys

try:
    import cPickle as pickle
except ImportError:
    import pickle

from astropy.table import Table

from consistency import *
from get_refine_box import get_refine_box
from get_halo_center import get_halo_center
from get_proper_box_size import get_proper_box_size

import seaborn as sns
sns.set_style("whitegrid", {'axes.grid' : False})
import matplotlib as mpl
mpl.rcParams['font.family'] = 'stixgeneral'


def parse_args():
    '''
    Parse command line arguments.  Returns args object.
    '''
    parser = argparse.ArgumentParser(description="makes a bunch of plots")

    ## optional arguments
    parser.add_argument('--halo', metavar='halo', type=str, action='store',
                        help='which halo? default is 8508 (Tempest)')
    parser.set_defaults(halo="8508")

    ## clobber?
    parser.add_argument('--clobber', dest='clobber', action='store_true')
    parser.add_argument('--no-clobber', dest='clobber', action='store_false', help="default is no clobber")
    parser.set_defaults(clobber=False)

    ## what are we plotting and where is it
    parser.add_argument('--run', metavar='run', type=str, action='store',
                        help='which run? default is natural')
    parser.set_defaults(run="natural")

    parser.add_argument('--output', metavar='output', type=str, action='store',
                        help='which output? default is RD0020')
    parser.set_defaults(output="RD0020")

    parser.add_argument('--system', metavar='system', type=str, action='store',
                        help='which system are you on? default is oak')
    parser.set_defaults(system="oak")

    ## plot groups
    parser.add_argument('--all', dest='all', action='store_true',
                        help='make all plots?, default if not')
    parser.set_defaults(all=False)

    parser.add_argument('--resolution', dest='resolution', action='store_true',
                        help='make resolution slice plot?, default if not')
    parser.set_defaults(ions=False)

    parser.add_argument('--ions', dest='ions', action='store_true',
                        help='make plots of ions?, default if not')
    parser.set_defaults(ions=False)

    parser.add_argument('--physical', dest='physical', action='store_true',
                        help='make plots of physical properties?, default if not')
    parser.set_defaults(physical=False)

    parser.add_argument('--box', dest='box', action='store_true',
                        help='make plots of fixed physical width?, default is no')
    parser.set_defaults(box=False)

    ## individual plots
    parser.add_argument('--hi', dest='hi', action='store_true',
                        help='make HI plot?, default if not')
    parser.set_defaults(hi=False)

    parser.add_argument('--civ', dest='civ', action='store_true',
                        help='make CIV plot?, default if not')
    parser.set_defaults(civ=False)

    parser.add_argument('--ovi', dest='ovi', action='store_true',
                        help='make OVI?, default if not')
    parser.set_defaults(ovi=False)

    parser.add_argument('--mgii', dest='mgii', action='store_true',
                        help='make MgII?, default if not')
    parser.set_defaults(ovi=False)

    parser.add_argument('--neviii', dest='neviii', action='store_true',
                        help='make NeVIII?, default if not')
    parser.set_defaults(ovi=False)

    parser.add_argument('--silicon', dest='silicon', action='store_true',
                        help='make Silicon plots?, default if not')
    parser.set_defaults(ovi=False)

    parser.add_argument('--density', dest='density', action='store_true',
                        help='make density plots?, default if not')
    parser.set_defaults(density=False)

    parser.add_argument('--metals', dest='metals', action='store_true',
                        help='make metallicity plots?, default if not')
    parser.set_defaults(metals=False)

    parser.add_argument('--slices', dest='slices', action='store_true',
                        help='make only slice plots?, default if not')
    parser.set_defaults(slices=False)

    parser.add_argument('--noslices', dest='noslices', action='store_true',
                        help='make no slice plots?, default if not')
    parser.set_defaults(noslices=False)


    args = parser.parse_args()
    return args

#-----------------------------------------------------------------------------------------------------

def make_projection_plot(ds, prefix, field, zmin, zmax, cmap, **kwargs):
    axis = kwargs.get("axis", ['x','y','z']) # if axis not set, do all
    box = kwargs.get("box", "")
    center = kwargs.get("center", "")
    appendix = kwargs.get("appendix", "")
    width = kwargs.get("width", default_width)
    resolution = kwargs.get("resolution", (1048,1048)) # correct for the nref11f box
    ision  = kwargs.get("ision", False)
    if ision:
        basename = prefix + 'ions/' + ds.basename + appendix
        if not (os.path.exists(prefix + 'ions/' )):
            os.system("mkdir " + prefix + 'ions/' )
    else:
        basename = prefix + 'physical/' + ds.basename + appendix
        if not (os.path.exists(prefix + 'physical/' )):
            os.system("mkdir " + prefix + 'physical' )
    for ax in axis:
        if ision:
            print("field = ", species_dict[field])
            p = yt.ProjectionPlot(ds, ax, species_dict[field], center=center, data_source=box, width=(width, 'kpc'))
            p.set_zlim(species_dict[field], zmin, zmax)
            p.set_cmap(field=species_dict[field], cmap=cmap)
        else:
            if field == "density" or field == "metal_density":
                p = yt.ProjectionPlot(ds, ax, field, center=center, data_source=box, width=(width, 'kpc'))
                p.set_unit(('gas','density'),'Msun/pc**2')
                p.set_unit(('gas','metal_density'),'Msun/pc**2')
            else:
                p = yt.ProjectionPlot(ds, ax, field, center=center, data_source=box, weight_field=("gas","density"), width=(width, 'kpc'))
            p.set_zlim(field, zmin, zmax)
            p.set_cmap(field=field, cmap=cmap)
        p.annotate_timestamp(corner='upper_left', redshift=True, draw_inset_box=True)
        if field == 'HI' or field == 'H_p0_number_density':
            plot = p.plots['H_p0_number_density']
            colorbar = plot.cb
            p._setup_plots()
            colorbar.set_ticks([1e13,1e15,1e17,1e19,1e21,1e23])
            colorbar.set_ticklabels(['13','15','17','19','21','23'])
        p.annotate_scale(size_bar_args={'color':'white'})
        p.hide_axes()
        p.save(basename + '_Projection_' + ax + '_' + field + '.png')
        p.save(basename + '_Projection_' + ax + '_' + field + '.pdf')
        # frb = p.data_source.to_frb(width, resolution, center=center)
        # if ision:
        #     pickle.dump(frb[species_dict[field]], open(basename + '_Projection_' + ax + '_' + species_dict[field] + '.cpkl','wb'), protocol=-1)
        # else:
        #     pickle.dump(frb[field], open(basename + '_Projection_' + ax + '_' + field + '.cpkl','wb'), protocol=-1)

#-----------------------------------------------------------------------------------------------------

def make_slice_plot(ds, prefix, field, zmin, zmax, cmap, **kwargs):
    axis = kwargs.get("axis", ['x','y','z']) # if axis not set, do all
    box = kwargs.get("box", "")
    center = kwargs.get("center", "")
    appendix = kwargs.get("appendix", "")
    width = kwargs.get("width", default_width)
    resolution = kwargs.get("resolution", (1048,1048)) # correct for the nref11f box
    ision  = kwargs.get("ision", False)
    if ision:
        basename = prefix + 'ions/' + ds.basename + appendix
        if not (os.path.exists(prefix + 'ions/' )):
            os.system("mkdir " + prefix + 'ions/' )
    else:
        basename = prefix + 'physical/' + ds.basename + appendix
        if not (os.path.exists(prefix + 'physical/' )):
            os.system("mkdir " + prefix + 'physical' )
    for ax in axis:
        if ision:
            print("field = ", species_dict[field])
            s = yt.SlicePlot(ds, ax, species_dict[field], center=center, data_source=box, width=(width, 'kpc'))
            s.set_zlim(species_dict[field], zmin, zmax)
            s.set_cmap(field=species_dict[field], cmap=cmap)
        else:
            s = yt.SlicePlot(ds, ax, field, center=center, data_source=box, width=(width, 'kpc'))
            s.set_zlim(field, zmin, zmax)
            s.set_cmap(field=field, cmap=cmap)
        s.annotate_timestamp(corner='upper_left', redshift=True, draw_inset_box=True)
        if field == "density" or field == "metal_density":
            s.set_unit(('gas','density'),'Msun/pc**3')
        s.annotate_scale(size_bar_args={'color':'white'})
        s.hide_axes()
        s.save(basename + '_Slice_' + ax + '_' + field + '.png')
        s.save(basename + '_Slice_' + ax + '_' + field + '.pdf')
        #frb = s.data_source.to_frb(width, resolution, center=center)
        #cPickle.dump(frb[field], open(basename + '_Slice_' + ax + '_' + field + '.cpkl','wb'), protocol=-1)

#-----------------------------------------------------------------------------------------------------

def make_resolution_slice(ds, prefix, **kwargs):
    box = kwargs.get("box", "")
    center = kwargs.get("center", "")
    appendix = kwargs.get("appendix", "")
    width = kwargs.get("width", default_width)
    basename = prefix + 'physical/' + ds.basename + appendix
    s = yt.SlicePlot(ds, "y", 'dy', center=center, width=(1.5*width, 'kpc'))
    s.set_cmap('dy', discrete_cmap)
    s.set_unit('dy','kpc')
    s.annotate_timestamp(corner='upper_left', redshift=True, draw_inset_box=True)
    #s.set_cmap(('index','grid_level'), discrete_cmap)
    #s.set_zlim(('index','grid_level'),6,11)
    plot = s.plots['dy']
    s._setup_plots()
    colorbar = plot.cb
    colorbar.set_label('cell size (kpc)')
    s.save(basename + '_Slice_y_cellsize_dy.png')


#-----------------------------------------------------------------------------------------------------

def plot_script(halo, foggie_dir, output_dir, run, axis, **kwargs):
    outs = kwargs.get("outs", "all")
    trackname = kwargs.get("trackname", "halo_track")
    if axis == "all":
        axis = ['x','y','z']

    print(foggie_dir)
    track_name = foggie_dir + 'halo_00' + str(halo) + '/' + run + '/' + trackname
    if args.system == "pleiades":
        track_name = foggie_dir + "halo_008508/nref11f_refine200kpc_z4to2/halo_track"

    print("opening track: " + track_name)
    track = Table.read(track_name, format='ascii')
    track.sort('col1')

    ## default is do allll the snaps in the directory
    ## want to add flag for if just one
    run_dir = foggie_dir + 'halo_00' + str(halo) + '/' + run
    if halo == "8508":
        prefix = output_dir + 'plots_halo_008508/' + run + '/'
    else:
        prefix = output_dir + 'other_halo_plots/' + str(halo) + '/' + run + '/'
    if not (os.path.exists(prefix)):
        os.system("mkdir " + prefix)

    if outs == "all":
        print("looking for outputs in ", run_dir)
        outs = glob.glob(os.path.join(run_dir, '?D0???/?D0???'))
    elif outs == "RD":
        print("looking for just the RD outputs in ", run_dir)
        outs = glob.glob(os.path.join(run_dir, 'RD0???/RD0???'))
    else:
        print("outs = ", outs)
        new_outs = [glob.glob(os.path.join(run_dir, snap)) for snap in outs]
        print("new_outs = ", new_outs)
        new_new_outs = [snap[0] for snap in new_outs]
        outs = new_new_outs

    print("making plots for ", axis, " axis in ", outs)

    for snap in outs:
        # load the snapshot
        print('opening snapshot '+ snap)
        ds = yt.load(snap)
        if args.all or args.ions:
            trident.add_ion_fields(ds, ions=['C IV', 'O VI', 'Mg II', 'Si II', 'C II', 'Si III', 'Si IV', 'Ne VIII'])
        if args.mgii or args.ions:
            trident.add_ion_fields(ds, ions=['Mg II'])
        if args.neviii or args.ions:
            trident.add_ion_fields(ds, ions=['Ne VIII'])
        if args.silicon:
            trident.add_ion_fields(ds, ions=['Si II', 'Si III', 'Si IV'])

        ## add metal density
        # ds.add_field(("gas", "metal_density"), function=_metal_density, units="g/cm**2")


        # box = ds.r[ center[0]-wide/143886:center[0]+wide/143886, center[1]-wide/143886.:center[1]+wide/143886., center[2]-wide/143886.:center[2]+wide/143886.]

        #### this was for the off-center box
        # center = [centerx, centery+20. / 143886., centerz]
        # box = ds.r[ center[0]-wide/143886:center[0]+wide/143886, center[1]-wide/143886.:center[1]+wide/143886., center[2]-wide/143886.:center[2]+wide/143886.]

        zsnap = ds.get_parameter('CosmologyCurrentRedshift')
        proper_box_size = get_proper_box_size(ds)

        refine_box, refine_box_center, refine_width = get_refine_box(ds, zsnap, track)
        refine_width = refine_width * proper_box_size

        # center is trying to be the center of the halo
        center, velocity = get_halo_center(ds, refine_box_center)

        ## if want to add to the edges, need to loop over axes so unrefined
        ## region not in foreground / background
        width = default_width
        width_code = width / proper_box_size ## needs to be in code units
        box = ds.r[center[0] - 0.5*width_code : center[0] + 0.5*width_code, \
                  center[1] - 0.5*width_code : center[1] + 0.5*width_code, \
                  center[2] - 0.5*width_code : center[2] + 0.5*width_code]



        if not args.noslices:
            if args.all or args.resolution:
                make_resolution_slice(ds, prefix, center=refine_box_center, box=refine_box, \
                                width=refine_width)

            if args.all or args.physical or args.density or args.slices:
                make_slice_plot(ds, prefix, "density", \
                                density_slc_min, density_slc_max, density_color_map, \
                                ision=False, axis=axis, center=center, box=refine_box, \
                                width=refine_width, appendix="_refine")

            if args.all or args.physical or args.metals or args.slices:
                make_slice_plot(ds, prefix, "metallicity", \
                                metal_min, metal_max, metal_color_map, \
                                ision=False, axis=axis, center=center, box=refine_box, \
                                width=refine_width, appendix="_refine")

            if args.all or args.physical or args.slices:
                make_slice_plot(ds, prefix, "temperature", \
                                temperature_min, temperature_max, temperature_color_map, \
                                ision=False, axis=axis, center=center, box=refine_box, \
                                width=refine_width, appendix="_refine")
                make_slice_plot(ds, prefix, "entropy", \
                                entropy_min, entropy_max, entropy_color_map, \
                                ision=False, axis=axis, center=center, box=refine_box, \
                                width=refine_width, appendix="_refine")
                make_slice_plot(ds, prefix, "pressure", \
                                pressure_min, pressure_max, pressure_color_map, \
                                ision=False, axis=axis, center=center, box=refine_box, \
                                width=refine_width, appendix="_refine")


        if args.all or args.physical or args.density:
            print(width, refine_width, default_width)
            make_projection_plot(ds, prefix, "density",  \
                            density_proj_min, density_proj_max, density_color_map, \
                            ision=False, center=center, axis=axis, box=refine_box, \
                            width=refine_width, appendix="_refine")
            if args.box:
                make_projection_plot(ds, prefix, "density",  \
                            density_proj_min, density_proj_max, density_color_map, \
                            ision=False, center=center, axis=axis, box=box, \
                            width=width, appendix="_box")

        if args.all or args.physical:
            make_projection_plot(ds, prefix, "temperature",  \
                            temperature_min, temperature_max, temperature_color_map, \
                            ision=False, center=center, axis=axis, box=refine_box, \
                            width=refine_width, appendix="_refine")
            if args.box:
                make_projection_plot(ds, prefix, "temperature",  \
                            temperature_min, temperature_max, temperature_color_map, \
                            ision=False, center=center, axis=axis, box=box, \
                            width=width, appendix="_box")

        if args.all or args.physical or args.metals:
            # make_projection_plot(ds, prefix, "metal_density",  \
            #                 metal_density_min, metal_density_max, metal_color_map, \
            #                 ision=False, center=center, axis=axis, box=refine_box, \
            #                 width=refine_width, appendix="_refine")
            # make_projection_plot(ds, prefix, "metal_density",  \
            #                 metal_density_min, metal_density_max, metal_color_map, \
            #                 ision=False, center=center, axis=axis, box=box, \
            #                 width=width, appendix="_box")

            make_projection_plot(ds, prefix, "metallicity",  \
                            metal_min, metal_max, metal_color_map, \
                            ision=False, center=center, axis=axis, box=refine_box, \
                            width=refine_width, appendix="_refine")
            if args.box:
                make_projection_plot(ds, prefix, "metallicity",  \
                            metal_min, metal_max, metal_color_map, \
                            ision=False, center=center, axis=axis, box=box, \
                            width=width, appendix="_box")
        if args.hi:
            make_slice_plot(ds, prefix, "HI", \
                                h1_slc_min, h1_slc_max, h1_color_map,\
                                ision=True, axis=axis, center=center, box=refine_box, \
                                width=refine_width, appendix="_refine")

        if args.all or args.ions or args.hi:
            make_projection_plot(ds, prefix, "HI",  \
                            h1_proj_min, h1_proj_max, h1_color_map, \
                            ision=True, center=center, axis=axis, box=refine_box, \
                            width=refine_width, appendix="_refine")
            if args.box:
                make_projection_plot(ds, prefix, "HI",  \
                            h1_proj_min, h1_proj_max, h1_color_map, \
                            ision=True, center=center, axis=axis, box=box, \
                            width=width, appendix="_box")

        if args.all or args.ions or args.ovi:
            make_projection_plot(ds, prefix, "OVI",  \
                            o6_min, o6_max, o6_color_map, \
                            ision=True, center=center, axis=axis, box=refine_box, \
                            width=refine_width, appendix="_refine")
            if args.box:
                make_projection_plot(ds, prefix, "OVI",  \
                            o6_min, o6_max, o6_color_map, \
                            ision=True, center=center, axis=axis, box=box, \
                            width=width, appendix="_box")

        if args.all or args.ions or args.mgii:
            make_projection_plot(ds, prefix, "MgII",  \
                            mg2_min, mg2_max, mg2_color_map, \
                            ision=True, center=center, axis=axis, box=refine_box, \
                            width=refine_width, appendix="_refine")
            if args.box:
                make_projection_plot(ds, prefix, "MgII",  \
                            mg2_min, mg2_max, mg2_color_map, \
                            ision=True, center=center, axis=axis, box=box, \
                            width=width, appendix="_box")

        if args.all or args.ions or args.civ:
            make_projection_plot(ds, prefix, "CIV",  \
                            c4_min, c4_max, c4_color_map, \
                            ision=True, center=center, axis=axis, box=refine_box, \
                            width=refine_width, appendix="_refine")
            if args.box:
                make_projection_plot(ds, prefix, "CIV",  \
                            c4_min, c4_max, c4_color_map, \
                            ision=True, center=center, axis=axis, box=box, \
                            width=width, appendix="_box")

        if args.all or args.ions or args.silicon:
            make_projection_plot(ds, prefix, "SiIV",  \
                            si4_min, si4_max, si4_color_map, \
                            ision=True, center=center, axis=axis, box=refine_box, \
                            width=refine_width, appendix="_refine")
            if args.box:
                make_projection_plot(ds, prefix, "SiIV",  \
                            si4_min, si4_max, si4_color_map, \
                            ision=True, center=center, axis=axis, box=box, \
                            width=width, appendix="_box")
            make_projection_plot(ds, prefix, "SiIII",  \
                            si3_min, si3_max, si3_color_map, \
                            ision=True, center=center, axis=axis, box=refine_box, \
                            width=refine_width, appendix="_refine")
            if args.box:
                make_projection_plot(ds, prefix, "SiIII",  \
                            si3_min, si3_max, si3_color_map, \
                            ision=True, center=center, axis=axis, box=box, \
                            width=width, appendix="_box")
            make_projection_plot(ds, prefix, "SiII",  \
                            si2_min, si2_max, si2_color_map, \
                            ision=True, center=center, axis=axis, box=refine_box, \
                            width=refine_width, appendix="_refine")
            if args.box:
                make_projection_plot(ds, prefix, "SiII",  \
                            si2_min, si2_max, si2_color_map, \
                            ision=True, center=center, axis=axis, box=box, \
                            width=width, appendix="_box")

        if args.all or args.ions or args.neviii:
            make_projection_plot(ds, prefix, "NeVIII",  \
                            ne8_min, ne8_max, ne8_color_map, \
                            ision=True, center=center, axis=axis, box=refine_box, \
                            width=refine_width, appendix="_refine")
            if args.box:
                make_projection_plot(ds, prefix, "NeVIII",  \
                            ne8_min, ne8_max, ne8_color_map, \
                            ision=True, center=center, axis=axis, box=box, \
                            width=width, appendix="_box")

    return "yay plots! all done!"


#-----------------------------------------------------------------------------------------------------

if __name__ == "__main__":
    args = parse_args()

    if not args.clobber:
        print("NO-CLOBBER IS NOT ACTUALLY IMPLEMENTED SO I'M GOING TO CLOBBER AWAY clobber clobber clobber")

    if args.system == "oak":
        foggie_dir = "/astro/simulations/FOGGIE/"
        output_path = "/Users/molly/Dropbox/foggie-collab/"
    elif args.system == "dhumuha" or args.system == "palmetto":
        foggie_dir = "/Users/molly/foggie/"
        output_path = "/Users/molly/Dropbox/foggie-collab/"
    elif args.system == "harddrive":
        foggie_dir = "/Volumes/foggie/"
        output_path = "/Users/molly/Dropbox/foggie-collab/"
    elif args.system == "nmearl":
        foggie_dir = "/Users/nearl/data/"
        output_path = "/Users/nearl/Desktop/"
    elif args.system == "pleiades":
        foggie_dir = "/nobackup/mpeeples/"
        output_path = "/nobackup/mpeeples/"

    if args.run == "natural":
        run_loc = "nref11n/natural/"
        trackname = "halo_track"
        haloname = "halo008508_nref11n"
    elif args.run == "nref10f":
        run_loc = "nref11n/nref11n_nref10f_refine200kpc/"
        trackname = "halo_008508/nref11n/nref11n_nref10f_refine200kpc/halo_track"
        haloname = "halo008508_nref11n_nref10f"
    elif args.run == "nref11n_selfshield":
        run_loc = "nref11n/nref11n_selfshield/"
        trackname = "halo_008508/nref11n/nref11n_selfshield/halo_track"
        haloname = "halo008508_nref11n_selfshield"
        if args.system == "pleiades":
            trackname = "halo_008508/nref11f_refine200kpc/halo_track"
            run_loc = "nref11n_selfshield/"
    elif args.run == "nref11n_startest_selfshield":
        run_loc = "nref11n/nref11n_startest_selfshield/"
        trackname = "halo_008508/nref11n/nref11n_selfshield/halo_track"
        haloname = "halo008508_nref11n_startest_selfshield"
        if args.system == "pleiades":
            trackname = "halo_008508/nref11f_refine200kpc/halo_track"
            run_loc = "nref11n_selfshield/"
    elif args.run == "nref10n_nref8f_selfshield":
        run_loc = "nref10n/nref10n_nref8f_selfshield/"
        trackname = "halo_008508/nref10n/nref10n_nref8f_selfshield/halo_track"
        haloname = "halo008508_nref10n_nref8f_selfshield"
        if args.system == "pleiades":
            trackname = "halo_008508/nref10n_nref8f_selfshield/halo_track"
            run_loc = "nref10n_nref8f_selfshield/"
    elif args.run == "nref11n_nref9f_startest":
        run_loc = "nref11n/nref11n_nref9f_startest/"
        trackname = "halo_008508/nref11n_nref9f_startest/halo_track"
        haloname = "halo008508_nref11n_nref9f_startest"
    elif args.run == "nref10n_nref8f_startest_selfshield":
        run_loc = "nref10n/nref10n_nref8f_startest_selfshield/"
        trackname = "halo_008508/nref10n/nref10n_nref8f_startest_selfshield/halo_track"
        haloname = "halo008508_nref10n_nref8f_startest_selfshield"
        if args.system == "pleiades":
            trackname = "halo_008508/nref10n_nref8f_startest_selfshield/halo_track"
            run_loc = "nref10n_nref8f_startest_selfshield/"
    elif args.run == "nref10n_nref8f_startest10000_selfshield":
        run_loc = "nref10n/nref10n_nref8f_startest10000_selfshield/"
        trackname = "halo_008508/nref10n/nref10n_nref8f_startest10000_selfshield/halo_track"
        haloname = "halo008508_nref10n_nref8f_startest10000_selfshield"
        if args.system == "pleiades":
            trackname = "halo_008508/nref10n_nref8f_startest10000_selfshield/halo_track"
            run_loc = "nref10n_nref8f_startest10000_selfshield/"
    elif args.run == "nref10n_nref8f_startest5000_selfshield":
        run_loc = "nref10n/nref10n_nref8f_startest5000_selfshield/"
        trackname = "halo_008508/nref10n/nref10n_nref8f_startest5000_selfshield/halo_track"
        haloname = "halo008508_nref10n_nref8f_startest5000_selfshield"
        if args.system == "pleiades":
            trackname = "halo_008508/nref10n_nref8f_startest5000_selfshield/halo_track"
            run_loc = "nref10n_nref8f_startest5000_selfshield/"
    elif args.run == "nref10n_nref8f_startest_selfshield":
        run_loc = "nref10n/nref10n_nref8f_startest_selfshield/"
        trackname = "halo_008508/nref10n/nref10n_nref8f_startest_selfshield/halo_track"
        haloname = "halo008508_nref10n_nref8f_startest_selfshield"
        if args.system == "pleiades":
            trackname = "halo_008508/nref10n_nref8f_startest_selfshield/halo_track"
            run_loc = "nref10n_nref8f_startest_selfshield/"
    elif args.run == "nref11n_selfshield_z15":
        run_loc = "nref11n_selfshield_z15/natural/"
        trackname = "halo_008508/nref11n_selfshield_z15/nref11n_nref10f_selfshield_z6/halo_track"
        haloname = "halo008508_nref11n_selfshield_z15"
        if args.system == "pleiades":
            trackname = "halo_008508/nref11n_nref10f_selfshield_z6/halo_track"
            run_loc = "nref11n_selfshield_z15/"
    elif args.run == "nref11n_nref10f_selfshield_z6":
        run_loc = "nref11n_selfshield_z15/nref11n_nref10f_selfshield_z6/"
        trackname = "halo_008508/nref11n_selfshield_z15/nref11n_nref10f_selfshield_z6/halo_track"
        haloname = "halo008508_nref11n_nref10f_selfshield_z6"
    elif args.run == "nref11c_nref9f":
        run_loc = "nref11n_selfshield_z15/nref11c_nref9f_selfshield_z6/"
        trackname = "halo_008508/nref11n_selfshield_z15/nref11c_nref9f_selfshield_z6/halo_track"
        haloname = "halo008508_nref11c_nref9f_selfshield_z6"
    elif args.run == "nref11f":
        run_loc = "nref11n/nref11f_refine200kpc/"
        trackname =  "halo_008508/nref11n/nref11f_refine200kpc/halo_track"
        haloname = "halo008508_nref11f"
        if args.system == "pleiades":
            trackname = "halo_008508/nref11f_refine200kpc/halo_track"
            run_loc = "nref11f_refine200kpc_z4to2/"

    print("for now I am assuming you are using the Tempest halo even if you passed in something different")

    if args.output == "all" or args.output == "RD":
        message = plot_script(args.halo, foggie_dir, output_path, run_loc, "all", outs=args.output)
    else:
        message = plot_script(args.halo, foggie_dir, output_path, run_loc, "all", outs=[args.output + "/" + args.output])

    sys.exit(message)
