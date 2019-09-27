from __future__ import print_function

import numpy as np

import yt
yt.enable_parallelism()

import trident

import argparse
import os
import glob
import sys
import multiprocessing as mp

import cPickle

from astropy.table import Table

from consistency import *
from get_halo_center import get_halo_center
from get_proper_box_size import get_proper_box_size
from get_refine_box import get_refine_box

from modular_plots import make_projection_plot, make_slice_plot

import seaborn as sns
sns.set_style("whitegrid", {'axes.grid' : False})
import matplotlib as mpl
mpl.rcParams['font.family'] = 'stixgeneral'

# foggie_dir = "/astro/simulations/FOGGIE/"
foggie_dir = "/Users/molly/foggie/"  ## where the simulations live
output_dir = "/Users/molly/Dropbox/foggie-collab/"  ## outputs go here

## lou
# foggie_dir = "/u/mpeeples/"  ## where the simulations live
# output_dir = "/u/mpeeples/plots/"  ## outputs go here


def parse_args():
    '''
    Parse command line arguments.  Returns args object.
    '''
    parser = argparse.ArgumentParser(description="makes a bunch of plots")

    ## optional arguments
    parser.add_argument('--halo', metavar='halo', type=str, action='store',
                        help='which halo? default is 8508')
    parser.set_defaults(halo="8508")

    ## clobber?
    parser.add_argument('--clobber', dest='clobber', action='store_true')
    parser.add_argument('--no-clobber', dest='clobber', action='store_false', help="default is no clobber")
    parser.set_defaults(clobber=False)

    ## plot groups
    parser.add_argument('--all', dest='all', action='store_true',
                        help='make all plots?, default if not')
    parser.set_defaults(all=False)

    parser.add_argument('--ions', dest='ions', action='store_true',
                        help='make plots of ions?, default if not')
    parser.set_defaults(ions=False)

    parser.add_argument('--physical', dest='physical', action='store_true',
                        help='make plots of physical properties?, default if not')
    parser.set_defaults(physical=False)

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
def _metal_density(field, data):
    return data["gas", "density"] * data["gas", "metallicity"] / 0.02  ## idk if this is the solar metallicity in enzo
#-----------------------------------------------------------------------------------------------------


def make_projection_plot_dict(args_dict):
    ds = args_dict['ds']
    axis = args_dict['axis']
    box = args_dict['box']
    center = args_dict['center']
    appendix = args_dict['appendix']
    width = args_dict['width']
    ision  = args_dict['ision']
    prefix = args_dict['prefix']
    field = args_dict['field']
    zmin = args_dict['zmin']
    zmax = args_dict['zmax']
    cmap = args_dict['cmap']
    resolution = (1048,1048)

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
            p = yt.ProjectionPlot(ds, ax, field, center=center, data_source=box, width=(width, 'kpc'))
            p.set_zlim(field, zmin, zmax)
            p.set_cmap(field=field, cmap=cmap)
        p.annotate_timestamp(corner='upper_left', redshift=True, draw_inset_box=True)
        if field == "density" or field == "metal_density":
            p.set_unit(('gas','density'),'Msun/pc**2')
        p.annotate_scale(size_bar_args={'color':'white'})
        p.hide_axes()
        p.save(basename + '_Projection_' + ax + '_' + field + '.png')
        p.save(basename + '_Projection_' + ax + '_' + field + '.pdf')
        frb = p.data_source.to_frb(width, resolution, center=center)
        if ision:
            cPickle.dump(frb[species_dict[field]], open(basename + '_Projection_' + ax + '_' + field + '.cpkl','wb'), protocol=-1)
        else:
            cPickle.dump(frb[field], open(basename + '_Projection_' + ax + '_' + field + '.cpkl','wb'), protocol=-1)

#-----------------------------------------------------------------------------------------------------

def plot_script(halo, run, axis, **kwargs):
    outs = kwargs.get("outs", "all")
    trackname = kwargs.get("trackname", "halo_track")
    if axis == "all":
        axis = ['x','y','z']

    track_name = foggie_dir + 'halo_00' + str(halo) + '/' + run + '/' + trackname
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
            trident.add_ion_fields(ds, ions=['C IV', 'O VI', 'H I', 'Si II', 'C II', 'Si III', 'Si IV'])
        #if args.silicon:
        trident.add_ion_fields(ds, ions=['Si II', 'Si III', 'Si IV'])

        # box = ds.r[ center[0]-wide/143886:center[0]+wide/143886, center[1]-wide/143886.:center[1]+wide/143886., center[2]-wide/143886.:center[2]+wide/143886.]

        #### this was for the off-center box
        # center = [centerx, centery+20. / 143886., centerz]
        # box = ds.r[ center[0]-wide/143886:center[0]+wide/143886, center[1]-wide/143886.:center[1]+wide/143886., center[2]-wide/143886.:center[2]+wide/143886.]

        zsnap = ds.get_parameter('CosmologyCurrentRedshift')
        proper_box_size = get_proper_box_size(ds)

        refine_box, refine_box_center, refine_width = get_refine_box(ds, zsnap, track)
        refine_width = refine_width * proper_box_size

        # center is trying to be the center of the halo
        center = get_halo_center(ds, refine_box_center)
        width = refine_width + 10. ## add a little on the edges
        width_code = width / proper_box_size ## needs to be in code units
        box = ds.r[center[0] - 0.5*width_code : center[0] + 0.5*width_code, \
                  center[1] - 0.5*width_code : center[1] + 0.5*width_code, \
                  center[2] - 0.5*width_code : center[2] + 0.5*width_code]

        args_dict = {'ds' : ds,
                     'prefix' : prefix,
                     'field' : 'SiII',
                     'zmin' : si2_min,
                     'zmax' : si2_max,
                     'cmap' : si2_color_map,
                     'ision' : True,
                     'axis' : ['x','y','z'],
                     'center' : center,
                     'axis' : axis,
                     'box' : refine_box,
                     'width' : refine_width-10.,
                     'appendix' : '_refine'}
#        make_projection_plot_dict(args_dict)

        make_projection_plot(ds, prefix, "SiII",  \
                            si2_min, si2_max, si2_color_map, \
                            ision=True, center=center, axis=axis, box=refine_box, \
                            width=(refine_width-10.), appendix="_refine")

        ### trying yt parallel

        # if not args.noslices:
        #     if args.all or args.physical or args.density or args.slices:
        #         make_density_slice_plot(ds, prefix, axis=axis, center=center, box=refine_box, \
        #                            width=(refine_width-10.), appendix="_refine")
        #
        #     if args.all or args.physical or args.metals or args.slices:
        #         make_metal_slice_plot(ds, prefix, axis=axis, center=center, box=refine_box, \
        #                               width=(refine_width-10.), appendix="_refine")
        #
        #     if args.all or args.physical or args.slices:
        #         make_temperature_slice_plot(ds, prefix, axis=axis, center=center, box=refine_box, \
        #                               width=(refine_width-10.), appendix="_refine")
        #         make_entropy_slice_plot(ds, prefix, axis=axis, center=center, box=box, width=width, appendix="_box")
        #         make_entropy_slice_plot(ds, prefix, axis=axis, center=refine_box_center, \
        #                               box=refine_box, width=refine_width, appendix="_refine")
        #
        #
        # if args.all or args.physical or args.density:
        #     print(width, refine_width, default_width)
        #     make_density_projection_plot(ds, prefix, axis=axis, center=center, box=box, \
        #                        width=width, appendix="_box")
        #     make_density_projection_plot(ds, prefix, axis=axis, center=refine_box_center, box=refine_box, \
        #                        width=refine_width, appendix="_refine")
        #
        # if args.all or args.physical:
        #     make_temperature_projection_plot(ds, prefix, axis=axis, center=refine_box_center,\
        #                      box=refine_box, width=refine_width, appendix="_refine")
        #     make_temperature_projection_plot(ds, prefix, axis=axis, center=center, box=box, width=width, appendix="_box")
        #
        # if args.all or args.physical or args.metals:
        #     make_metal_projection_plot(ds, prefix, axis=axis, center=refine_box_center,\
        #                      box=refine_box, width=refine_width, appendix="_refine")
        #     make_metal_projection_plot(ds, prefix, axis=axis, center=center, box=box, width=width, appendix="_box")
        #
        # if args.all or args.ions or args.hi:
        #     make_hi_plots(ds, prefix,  center=refine_box_center, \
        #                   box=refine_box, width=refine_width, appendix="_refine")
        #     make_hi_plots(ds, prefix, axis=axis, center=center, box=box, width=width, appendix="_box")
        #
        # if args.all or args.ions or args.ovi:
        #     make_o6_plots(ds, prefix, axis=axis, center=refine_box_center, \
        #                   box=refine_box, width=refine_width, appendix="_refine")
        #     make_o6_plots(ds, prefix, axis=axis, center=center, box=box, width=width, appendix="_box")
        #
        # if args.all or args.ions or args.civ:
        #     make_c4_plots(ds, prefix, axis=axis, center=refine_box_center, \
        #                   box=refine_box, width=refine_width, appendix="_refine")
        #     make_c4_plots(ds, prefix, axis=axis, center=center, box=box, width=width, appendix="_box")
        #
        # if args.all or args.ions or args.silicon:
        #     make_si2_plots(ds, prefix, axis=axis, center=refine_box_center, box=refine_box, width=refine_width, appendix="_refine")
        #     make_si2_plots(ds, prefix, axis=axis, center=center, box=box, width=width, appendix="_box")
        #     make_si3_plots(ds, prefix, axis=axis, center=refine_box_center, box=refine_box, width=refine_width, appendix="_refine")
        #     make_si3_plots(ds, prefix, axis=axis, center=center, box=box, width=width, appendix="_box")

    return "yay plots! all done!"


#-----------------------------------------------------------------------------------------------------

if __name__ == "__main__":
    args = parse_args()

    if not args.clobber:
        print("NO-CLOBBER IS NOT ACTUALLY IMPLEMENTED SO I'M GOING TO CLOBBER AWAY clobber clobber clobber")

    # message = plot_script(args.halo, "symmetric_box_tracking/nref11f_50kpc", "x")
    # message = plot_script(args.halo, "nref11n/nref11n_nref10f_refine200kpc_z4to2", "all", outs=["RD0015/RD0015"])
    # message = plot_script(args.halo, "nref11n/nref11f_refine200kpc_z4to2", "all")
    # message = plot_script(args.halo, "nref11n/natural", "all", outs=["RD0015/RD0015"])
    message = plot_script(args.halo, "nref11n/natural", "all")
    sys.exit(message)
