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

from paper_consistency import *
from get_refine_box import get_refine_box
from get_halo_center import get_halo_center
from get_proper_box_size import get_proper_box_size
from get_run_loc_etc import get_run_loc_etc

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
    basename = prefix + ds.basename + '_' + args.run + appendix
    for ax in axis:
        if ision:
            print("field = ", species_dict[field])
            p = yt.ProjectionPlot(ds, ax, species_dict[field], center=center, data_source=box, width=(width, 'kpc'))
            p.set_zlim(species_dict[field], zmin, zmax)
            p.set_cmap(field=species_dict[field], cmap=cmap)
            p.set_background_color(species_dict[field], color=cmap(0))
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

def make_projection_plot_no_labels(ds, prefix, field, zmin, zmax, cmap, **kwargs):
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
        p.hide_colorbar()
        p.hide_axes()
        p.save(basename + '_Projection_' + ax + '_' + field + '.png')
        p.save(basename + '_Projection_' + ax + '_' + field + '.pdf')

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
            s = yt.SlicePlot(ds, ax, species_dict[field], center=center,  width=(1.5*width, 'kpc'))
            s.set_zlim(species_dict[field], zmin, zmax)
            s.set_cmap(field=species_dict[field], cmap=cmap)
        else:
            s = yt.SlicePlot(ds, ax, field, center=center,  width=(1.5*width, 'kpc'))
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
    if not (os.path.exists(prefix + 'physical/' )):
        os.system("mkdir " + prefix + 'physical' )
    # s = yt.SlicePlot(ds, "x", 'dx', center=center, width=(1.5*width, 'kpc'))
    # s.set_cmap('dx', discrete_cmap)
    # s.set_unit('dx','kpc')
    # s.annotate_timestamp(corner='upper_left', redshift=True, draw_inset_box=True)
    # plot = s.plots['dx']
    # s._setup_plots()
    # colorbar = plot.cb
    # colorbar.set_label('cell size (kpc)')
    # s.save(ds.basename + '_Slice_x_cellsize_dx.png')
    # s.save(basename + '_Slice_x_cellsize_dx.png')

    s = yt.SlicePlot(ds, "y", 'dy', center=center, width=(1.5*width, 'kpc'))
    s.set_cmap('dy', discrete_cmap)
    s.set_unit('dy','kpc')
    s.annotate_timestamp(corner='upper_left', redshift=True, draw_inset_box=True)
    plot = s.plots['dy']
    s._setup_plots()
    colorbar = plot.cb
    colorbar.set_label('cell size (kpc)')
    s.save(ds.basename + '_Slice_y_cellsize_dy.png')
    s.save(basename + '_Slice_y_cellsize_dy.png')

    # s = yt.SlicePlot(ds, "z", 'dz', center=center, width=(1.5*width, 'kpc'))
    # s.set_cmap('dz', discrete_cmap)
    # s.set_unit('dz','kpc')
    # s.annotate_timestamp(corner='upper_left', redshift=True, draw_inset_box=True)
    # plot = s.plots['dz']
    # s._setup_plots()
    # colorbar = plot.cb
    # colorbar.set_label('cell size (kpc)')
    # s.save(ds.basename + '_Slice_z_cellsize_dz.png')
    # s.save(basename + '_Slice_z_cellsize_dz.png')


#-----------------------------------------------------------------------------------------------------

def plot_script(halo, foggie_dir, output_dir, run, axis, **kwargs):
    outs = kwargs.get("outs", "all")
    trackname = kwargs.get("trackname", "halo_track")
    if axis == "all":
        axis = ['x','y','z']

    print(foggie_dir)
    # track_name = foggie_dir + 'halo_00' + str(halo) + '/' + run + '/' + trackname
    #if args.system == "pleiades" and args.run == 'nref11c_400kpc':
    #    track_name = foggie_dir + "halo_008508/nref11c_nref8f_400kpc/halo_track"
    #### track_name = '/astro/simulations/FOGGIE/halo_008508/nref11n/nref11n_nref10f_refine200kpc/halo_track'

    print("opening track: " + trackname)
    track = Table.read(trackname, format='ascii')
    track.sort('col1')

    ## default is do allll the snaps in the directory
    ## want to add flag for if just one
    # run_dir = foggie_dir + 'halo_00' + str(halo) + '/' + run
    run_dir = foggie_dir
    prefix = output_dir ## + '/' + run
    print('prefix = ', prefix)
    print('run_dir = ', run_dir)
    # if halo == "8508":
    #     prefix = output_dir + 'plots_halo_008508/' + run
    # else:
    #    prefix = output_dir + 'other_halo_plots/' + str(halo) + '/' + run + '/'
    if not (os.path.exists(prefix)):
        os.system("mkdir " + prefix)
    if args.system == 'pleiades' and args.run == 'nref11c_400kpc':
        run_dir = '/nobackup/mpeeples/halo_008508/nref11c_nref8f_400kpc/'
        prefix = './'

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
        trident.add_ion_fields(ds, ions=['C IV', 'O VI', 'Mg II', 'Si II', 'C II', 'Si III', 'Si IV', 'Ne VIII'])

        ## add metal density
        # ds.add_field(("gas", "metal_density"), function=_metal_density, units="g/cm**2")


        zsnap = ds.get_parameter('CosmologyCurrentRedshift')
        proper_box_size = get_proper_box_size(ds)

        refine_box, refine_box_center, refine_width = get_refine_box(ds, zsnap, track)
        refine_width = refine_width * proper_box_size

        # center is trying to be the center of the halo
        search_radius = 10.
        this_search_radius = search_radius / (1+ds.get_parameter('CosmologyCurrentRedshift'))
        center, velocity = get_halo_center(ds, refine_box_center, radius=this_search_radius)

        print('halo center = ', center, ' and refine_box_center = ', refine_box_center)

        ## if want to add to the edges, need to loop over axes so unrefined
        ## region not in foreground / background
        width = default_width
        width_code = width / proper_box_size ## needs to be in code units
        box = ds.r[refine_box_center[0] - 0.55*width_code : refine_box_center[0] + 0.55*width_code, \
          refine_box_center[1] - 0.5*width_code : refine_box_center[1] + 0.5*width_code, \
          refine_box_center[2] - 0.05*width_code : refine_box_center[2] + 0.35*width_code]

        axis = 'y'

        make_projection_plot(ds, prefix, "HI",  \
                        h1_proj_min, h1_proj_max, h1_color_map, \
                        ision=True, center=box.center, axis=axis, box=box, \
                        width=width, appendix="_righthand")
        make_projection_plot(ds, prefix, "SiII",  \
                        si2_min, si2_max, si2_color_map, \
                        ision=True, center=box.center, axis=axis, box=box, \
                        width=refine_width, appendix="_righthand")
        make_projection_plot(ds, prefix, "OVI",  \
                        o6_min, o6_max, o6_color_map, \
                        ision=True, center=box.center, axis=axis, box=box, \
                        width=refine_width, appendix="_righthand")
        make_projection_plot(ds, prefix, "CIV",  \
                        c4_min, c4_max, c4_color_map, \
                        ision=True, center=box.center, axis=axis, box=box, \
                        width=refine_width, appendix="_righthand")
        make_projection_plot(ds, prefix, "SiIV",  \
                        si4_min, si4_max, si4_color_map, \
                        ision=True, center=box.center, axis=axis, box=box, \
                        width=refine_width, appendix="_righthand")
        make_projection_plot(ds, prefix, "SiIII",  \
                        si3_min, si3_max, si3_color_map, \
                        ision=True, center=box.center, axis=axis, box=box, \
                        width=refine_width, appendix="_righthand")
        make_projection_plot(ds, prefix, "NeVIII",  \
                        ne8_min, ne8_max, ne8_color_map, \
                        ision=True, center=box.center, axis=axis, box=box, \
                        width=refine_width, appendix="_righthand")

    return "yay plots! all done!"


#-----------------------------------------------------------------------------------------------------

if __name__ == "__main__":
    args = parse_args()

    if not args.clobber:
        print("NO-CLOBBER IS NOT ACTUALLY IMPLEMENTED SO I'M GOING TO CLOBBER AWAY clobber clobber clobber")

    foggie_dir, output_dir, run_loc, trackname, haloname, spectra_dir = get_run_loc_etc(args)
    output_dir = '/Users/molly/Dropbox/foggie-collab/papers/absorption_peeples/Figures/projections/'

    run_dir = foggie_dir +  run_loc
    if args.output == "all" or args.output == "RD":
        message = plot_script(args.halo, run_dir, output_dir, run_loc,  "all", trackname=trackname, outs=args.output)
    else:
        message = plot_script(args.halo, run_dir, output_dir, run_loc,  "all", trackname=trackname, outs=[args.output + "/" + args.output])

    sys.exit(message)
