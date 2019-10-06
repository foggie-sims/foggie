'''
Written by Molly Peeples

makes all the slices and projection plots you could possibly want and probably some others

will put output physical plots in a hopefully correct "physical/" directory and the ion
plots in a "ions/" directory in what should be a well-organzied plots directory for that halo
'''


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

from foggie.utils.consistency import *
from foggie.utils.get_refine_box import get_refine_box
from foggie.utils.get_halo_center import get_halo_center
from foggie.utils.get_proper_box_size import get_proper_box_size
from foggie.utils.get_run_loc_etc import get_run_loc_etc

import seaborn as sns
sns.set_style("whitegrid", {'axes.grid' : False})
import matplotlib as mpl
mpl.rcParams['font.family'] = 'stixgeneral'

h1_proj_min = 5.e15
h1_proj_max = 1.e23

o6_min = 5.e12
o6_max = 1.e15

c4_min = 1.e11
c4_max = 5.e15

ne8_min = 3.e12
ne8_max = 8.e14


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
    parser.add_argument('--pwd', dest='pwd', action='store_true',
                        help='just use the pwd?, default is no')
    parser.set_defaults(pwd=False)

    parser.add_argument('--run', metavar='run', type=str, action='store',
                        help='which run? default is nref11c')
    parser.set_defaults(run="nref11c")

    parser.add_argument('--output', metavar='output', type=str, action='store',
                        help='which output? default is RD0018')
    parser.set_defaults(output="RD0018")

    parser.add_argument('--system', metavar='system', type=str, action='store',
                        help='which system are you on? default is palmetto')
    parser.set_defaults(system="palmetto")

    ## plot groups
    parser.add_argument('--all', dest='all', action='store_true',
                        help='make all plots?, default if not')
    parser.set_defaults(all=False)

    parser.add_argument('--ions', dest='ions', action='store_true',
                        help='make plots of ions?, default if not')
    parser.set_defaults(ions=False)

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
    basename = './' +  ds.basename + appendix
    for ax in axis:
        if ision:
            print("field = ", species_dict[field])
            p = yt.ProjectionPlot(ds, ax, species_dict[field], center=center, data_source=box, width=(width, 'kpc'), window_size=40.)
            p.set_zlim(species_dict[field], zmin, zmax)
            p.set_cmap(field=species_dict[field], cmap='inferno')
        else:
            if field == "density" or field == "metal_density":
                p = yt.ProjectionPlot(ds, ax, field, center=center, data_source=box, width=(width, 'kpc'))
                p.set_unit(('gas','density'),'Msun/pc**2')
                p.set_unit(('gas','metal_density'),'Msun/pc**2')
            else:
                p = yt.ProjectionPlot(ds, ax, field, center=center, data_source=box, weight_field=("gas","density"), width=(width, 'kpc'), window_size=40.)
            p.set_zlim(field, zmin, zmax)
            p.set_cmap(field=field, cmap='cmap')
        p.hide_colorbar()
        p.hide_axes()
        p.set_buff_size(4000)
        p.save(basename + '_Projection_' + ax + '_' + field + '.png')

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
        if args.all or args.ions:
            trident.add_ion_fields(ds, ions=['O VI', 'Mg II', 'Si II', 'C II', 'Si III', 'Si IV', 'Ne VIII'])
        if args.ovi:
            trident.add_ion_fields(ds, ions=['O VI'])
        if args.mgii or args.ions:
            trident.add_ion_fields(ds, ions=['Mg II'])
        if args.civ or args.ions:
            trident.add_ion_fields(ds, ions=['C IV'])
        if args.neviii or args.ions:
            trident.add_ion_fields(ds, ions=['Ne VIII'])
        if args.silicon:
            trident.add_ion_fields(ds, ions=['Si II', 'Si III', 'Si IV'])

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
        box = ds.r[center[0] - 0.5*width_code : center[0] + 0.5*width_code, \
                  center[1] - 0.5*width_code : center[1] + 0.5*width_code, \
                  center[2] - 0.5*width_code : center[2] + 0.5*width_code]




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

    return "yay plots! all done!"


#-----------------------------------------------------------------------------------------------------

if __name__ == "__main__":
    args = parse_args()

    if not args.clobber:
        print("NO-CLOBBER IS NOT ACTUALLY IMPLEMENTED SO I'M GOING TO CLOBBER AWAY clobber clobber clobber")

    foggie_dir, output_dir, run_loc, trackname, haloname, spectra_dir = get_run_loc_etc(args)

    if args.pwd:
        run_dir = '.'
    else:
        run_dir = foggie_dir + run_loc
    message = plot_script(args.halo, run_dir, output_dir, run_loc,  "y", trackname=trackname, outs=[args.output + "/" + args.output])

    sys.exit(message)
