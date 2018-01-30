from __future__ import print_function

import numpy as np

import yt
import trident

import argparse
import os
import glob
import sys

from astropy.table import Table

from consistency import *
from get_halo_center import get_halo_center
from get_proper_box_size import get_proper_box_size

import seaborn as sns
sns.set_style("whitegrid", {'axes.grid' : False})
import matplotlib as mpl
mpl.rcParams['font.family'] = 'stixgeneral'

foggie_dir = "/astro/simulations/FOGGIE/"
# foggie_dir = "/Users/molly/foggie/"  ## where the simulations live
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
    parser.set_defaults(metals=False)


    args = parser.parse_args()
    return args

#-----------------------------------------------------------------------------------------------------


def make_density_projection_plot(ds, prefix, **kwargs):
    axis = kwargs.get("axis", ['x','y','z']) # if axis not set, do all
    box = kwargs.get("box", "")
    center = kwargs.get("center", "")
    appendix = kwargs.get("appendix", "")
    width = kwargs.get("width", default_width)
    if not (os.path.exists(prefix + 'physical/' )):
        os.system("mkdir " + prefix + 'physical' )
    for ax in axis:
        if not (os.path.exists(prefix + 'physical')):
            os.system("mkdir " + prefix + 'physical')
        p = yt.ProjectionPlot(ds, ax, 'density', center=center, data_source=box, width=(width, 'kpc'))
        p.annotate_timestamp(corner='upper_left', redshift=True, draw_inset_box=True)
        p.set_cmap(field="density", cmap=density_color_map)
        p.set_unit(('gas','density'),'Msun/pc**2')
        p.set_zlim("density", density_proj_min, density_proj_max)
        p.annotate_scale(size_bar_args={'color':'white'})
        p.hide_axes()
        p.save(prefix + 'physical/' + ds.basename + appendix)

def make_density_slice_plot(ds, prefix, **kwargs):
    axis = kwargs.get("axis", ['x','y','z']) # if axis not set, do all
    box = kwargs.get("box", "")
    center = kwargs.get("center", "")
    appendix = kwargs.get("appendix", "")
    width = kwargs.get("width", default_width)
    if not (os.path.exists(prefix + 'physical/' )):
        os.system("mkdir " + prefix + 'physical' )
    for ax in axis:
        if not (os.path.exists(prefix + 'physical')):
            os.system("mkdir " + prefix + 'physical')
        p = yt.SlicePlot(ds, ax, 'density', center=center, data_source=box, width=(width, 'kpc'))
        p.annotate_timestamp(corner='upper_left', redshift=True, draw_inset_box=True)
        p.set_cmap(field="density", cmap=density_color_map)
        p.set_unit(('gas','density'),'Msun/pc**3')
        p.set_zlim("density", density_slc_min, density_slc_max)
        p.annotate_scale(size_bar_args={'color':'white'})
        p.hide_axes()
        p.save(prefix + 'physical/' + ds.basename + appendix)

def make_metal_slice_plot(ds, prefix, **kwargs):
    axis = kwargs.get("axis", ['x','y','z']) # if axis not set, do all
    box = kwargs.get("box", "")
    center = kwargs.get("center", "")
    appendix = kwargs.get("appendix", "")
    width = kwargs.get("width", default_width)
    if not (os.path.exists(prefix + 'physical/' )):
        os.system("mkdir " + prefix + 'physical/' )
    for ax in axis:
        if not (os.path.exists(prefix + 'physical/')):
            os.system("mkdir " + prefix + 'physical/')
        p = yt.SlicePlot(ds, ax, 'metallicity', center=center, data_source=box, width=(width, 'kpc'))
        p.annotate_timestamp(corner='upper_left', redshift=True, draw_inset_box=True)
        p.set_cmap(field="metallicity", cmap=metal_color_map)
        p.set_zlim("metallicity", metal_min, metal_max)
        p.annotate_scale(size_bar_args={'color':'white'})
        p.hide_axes()
        p.save(prefix + 'physical/' + ds.basename + appendix)

def make_metal_projection_plot(ds, prefix, **kwargs):
    axis = kwargs.get("axis", ['x','y','z']) # if axis not set, do all
    box = kwargs.get("box", "")
    center = kwargs.get("center", "")
    appendix = kwargs.get("appendix", "")
    width = kwargs.get("width", default_width)
    if not (os.path.exists(prefix + 'physical/' )):
        os.system("mkdir " + prefix + 'physical/' )
    for ax in axis:
        if not (os.path.exists(prefix + 'physical/')):
            os.system("mkdir " + prefix + 'physical/')
        p = yt.ProjectionPlot(ds, ax,('gas','metallicity'),weight_field=("gas","density"),\
                            center=center, data_source=box, width=(width, 'kpc'))
        p.annotate_timestamp(corner='upper_left', redshift=True, draw_inset_box=True)
        p.set_cmap(field="metallicity", cmap=metal_color_map)
        p.set_zlim("metallicity", metal_min, metal_max)
        p.annotate_scale(size_bar_args={'color':'white'})
        p.hide_axes()
        p.save(prefix + 'physical/' + ds.basename + appendix)

def make_temperature_slice_plot(ds, prefix, **kwargs):
    axis = kwargs.get("axis", ['x','y','z']) # if axis not set, do all
    box = kwargs.get("box", "")
    center = kwargs.get("center", "")
    appendix = kwargs.get("appendix", "")
    width = kwargs.get("width", default_width)
    if not (os.path.exists(prefix + 'physical/' )):
        os.system("mkdir " + prefix + 'physical/' )
    for ax in axis:
        if not (os.path.exists(prefix + 'physical/')):
            os.system("mkdir " + prefix + 'physical/')
        p = yt.SlicePlot(ds, ax, 'temperature', center=center, data_source=box, width=(width, 'kpc'))
        p.annotate_timestamp(corner='upper_left', redshift=True, draw_inset_box=True)
        p.set_cmap(field="temperature", cmap=temperature_color_map)
        p.set_zlim("temperature", temperature_min, temperature_max)
        p.annotate_scale(size_bar_args={'color':'white'})
        p.hide_axes()
        p.save(prefix + 'physical/' + ds.basename + appendix)

def make_temperature_projection_plot(ds, prefix, **kwargs):
    axis = kwargs.get("axis", ['x','y','z']) # if axis not set, do all
    box = kwargs.get("box", "")
    center = kwargs.get("center", "")
    appendix = kwargs.get("appendix", "")
    width = kwargs.get("width", default_width)
    if not (os.path.exists(prefix + 'physical/' )):
        os.system("mkdir " + prefix + 'physical/' )
    for ax in axis:
        if not (os.path.exists(prefix + 'physical/')):
            os.system("mkdir " + prefix + 'physical/')
        p = yt.ProjectionPlot(ds, ax,('gas','temperature'), weight_field=("gas","density"),\
                              center=center, data_source=box, width=(width, 'kpc'))
        p.annotate_timestamp(corner='upper_left', redshift=True, draw_inset_box=True)
        p.set_cmap(field="temperature", cmap=temperature_color_map)
        p.set_zlim("temperature", temperature_min, temperature_max)
        p.annotate_scale(size_bar_args={'color':'white'})
        p.hide_axes()
        p.save(prefix + 'physical/' + ds.basename + appendix)

def make_entropy_plots(ds, prefix, **kwargs):
    axis = kwargs.get("axis", ['x','y','z']) # if axis not set, do all
    box = kwargs.get("box", "")
    center = kwargs.get("center", "")
    appendix = kwargs.get("appendix", "")
    width = kwargs.get("width", default_width)
    if not (os.path.exists(prefix + 'physical/' )):
        os.system("mkdir " + prefix + 'physical/' )
    for ax in axis:
        if not (os.path.exists(prefix + 'physical/' )):
            os.system("mkdir " + prefix + 'physical/')
        p = yt.SlicePlot(ds, ax, 'entropy', center=center, data_source=box, width=(width, 'kpc'))
        p.annotate_timestamp(corner='upper_left', redshift=True, draw_inset_box=True)
        p.set_cmap(field="entropy", cmap=entropy_color_map)
        p.set_zlim("entropy", entropy_min, entropy_max)
        p.annotate_scale(size_bar_args={'color':'white'})
        p.hide_axes()
        p.save(prefix + 'physical/' + ds.basename + appendix)

def make_hi_plots(ds, prefix, **kwargs):
    axis = kwargs.get("axis", ['x','y','z']) # if axis not set, do all
    box = kwargs.get("box", "")
    center = kwargs.get("center", "")
    appendix = kwargs.get("appendix", "")
    width = kwargs.get("width", default_width)
    if not (os.path.exists(prefix + 'ions/' )):
        os.system("mkdir " + prefix + 'ions/' )
    for ax in axis:
        if not (os.path.exists(prefix + 'ions/' )):
            os.system("mkdir " + prefix + 'ions/' )
        p = yt.ProjectionPlot(ds, ax, 'H_p0_number_density', center=center, data_source=box, width=(width,'kpc'))
        p.annotate_timestamp(corner='upper_left', redshift=True, draw_inset_box=True)
        p.set_cmap(field="H_p0_number_density", cmap=h1_color_map)
        p.set_zlim("H_p0_number_density",h1_proj_min, h1_proj_max)
        p.annotate_scale(size_bar_args={'color':'white'})
        p.hide_axes()
        pp = p.plots['H_p0_number_density']
        colorbar = pp.cb
        p._setup_plots()
        colorbar.set_ticks([1e13,1e15,1e17,1e19,1e21,1e23])
        colorbar.set_ticklabels(['13','15','17','19','21','23'])
        colorbar.ax.tick_params(labelsize=20)
        p.save(prefix + 'ions/' + ds.basename + '_HI' + appendix)

def make_o6_plots(ds, prefix, **kwargs):
    axis = kwargs.get("axis", ['x','y','z']) # if axis not set, do all
    box = kwargs.get("box", "")
    center = kwargs.get("center", "")
    appendix = kwargs.get("appendix", "")
    width = kwargs.get("width", default_width)
    if not (os.path.exists(prefix + 'ions/' )):
        os.system("mkdir " + prefix + 'ions/' )
    for ax in axis:
        if not (os.path.exists(prefix + 'ions/' )):
            os.system("mkdir " + prefix + 'ions/' )
        p = yt.ProjectionPlot(ds, ax, 'O_p5_number_density', center=center, data_source=box, width=(width,'kpc'))
        p.annotate_timestamp(corner='upper_left', redshift=True, draw_inset_box=True)
        p.set_cmap(field="O_p5_number_density", cmap=o6_color_map)
        p.set_zlim("O_p5_number_density", o6_min, o6_max)
        p.annotate_scale(size_bar_args={'color':'white'})
        p.hide_axes()
        p.save(prefix + 'ions/' + ds.basename + '_OVI' + appendix)

def make_c4_plots(ds, prefix, **kwargs):
    axis = kwargs.get("axis", ['x','y','z']) # if axis not set, do all
    box = kwargs.get("box", "")
    center = kwargs.get("center", "")
    appendix = kwargs.get("appendix", "")
    width = kwargs.get("width", default_width)
    if not (os.path.exists(prefix + 'ions/' )):
        os.system("mkdir " + prefix + 'ions/' )
    for ax in axis:
        if not (os.path.exists(prefix + 'ions/' )):
            os.system("mkdir " + prefix + 'ions/' )
        p = yt.ProjectionPlot(ds, ax, 'C_p3_number_density', center=center, data_source=box, width=(width,'kpc'))
        p.annotate_timestamp(corner='upper_left', redshift=True, draw_inset_box=True)
        p.set_cmap(field="C_p3_number_density", cmap=c4_color_map)
        p.set_zlim("C_p3_number_density", c4_min, c4_max)
        p.annotate_scale(size_bar_args={'color':'white'})
        p.hide_axes()
        p.save(prefix + 'ions/' + ds.basename + '_CIV' + appendix)

def make_si2_plots(ds, prefix, **kwargs):
    axis = kwargs.get("axis", ['x','y','z']) # if axis not set, do all
    box = kwargs.get("box", "")
    center = kwargs.get("center", "")
    appendix = kwargs.get("appendix", "")
    width = kwargs.get("width", default_width)
    if not (os.path.exists(prefix + 'ions/' )):
        os.system("mkdir " + prefix + 'ions/' )
    for ax in axis:
        if not (os.path.exists(prefix + 'ions/' )):
            os.system("mkdir " + prefix + 'ions/' )
        p = yt.ProjectionPlot(ds, ax, 'Si_p1_number_density', center=center, data_source=box, width=(width,'kpc'))
        p.annotate_timestamp(corner='upper_left', redshift=True, draw_inset_box=True)
        p.set_cmap(field="Si_p1_number_density", cmap=c4_color_map)
        p.set_zlim("Si_p1_number_density", c4_min, c4_max)
        p.annotate_scale(size_bar_args={'color':'white'})
        p.hide_axes()
        p.save(prefix + 'ions/' + ds.basename + '_SiII' + appendix)

def make_si3_plots(ds, prefix, **kwargs):
    axis = kwargs.get("axis", ['x','y','z']) # if axis not set, do all
    box = kwargs.get("box", "")
    center = kwargs.get("center", "")
    appendix = kwargs.get("appendix", "")
    width = kwargs.get("width", default_width)
    if not (os.path.exists(prefix + 'ions/' )):
        os.system("mkdir " + prefix + 'ions/' )
    for ax in axis:
        if not (os.path.exists(prefix + 'ions/' )):
            os.system("mkdir " + prefix + 'ions/' )
        p = yt.ProjectionPlot(ds, ax, 'Si_p2_number_density', center=center, data_source=box, width=(width,'kpc'))
        p.annotate_timestamp(corner='upper_left', redshift=True, draw_inset_box=True)
        p.set_cmap(field="Si_p2_number_density", cmap=si3_color_map)
        p.set_zlim("Si_p2_number_density", si3_min, si3_max)
        p.annotate_scale(size_bar_args={'color':'white'})
        p.hide_axes()
        p.save(prefix + 'ions/' + ds.basename + '_SiIII' + appendix)

def get_refine_box(ds, zsnap, track):
    ## find closest output, modulo not updating before printout
    diff = track['col1'] - zsnap
    this_loc = track[np.where(diff == np.min(diff[np.where(diff > 1.e-6)]))]
    print("using this loc:", this_loc)
    x_left = this_loc['col2'][0]
    y_left = this_loc['col3'][0]
    z_left = this_loc['col4'][0]
    x_right = this_loc['col5'][0]
    y_right = this_loc['col6'][0]
    z_right = this_loc['col7'][0]

    refine_box_center = [0.5*(x_left+x_right), 0.5*(y_left+y_right), 0.5*(z_left+z_right)]
    refine_box = ds.r[x_left:x_right, y_left:y_right, z_left:z_right]
    refine_width = np.abs(x_right - x_left)

    return refine_box, refine_box_center, refine_width

#-----------------------------------------------------------------------------------------------------

def plot_script(halo, run, axis, **kwargs):
    outs = kwargs.get("outs", "all")
    trackname = kwargs.get("trackname", "halo_track")
    width = kwargs.get("width", default_width) ## kpc
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
            trident.add_ion_fields(ds, ions=['C IV', 'O VI', 'H I', 'Si II', 'C II', 'Si III'])
        if args.hi:
            trident.add_ion_fields(ds, ions=['H I'])
        if args.silicon:
            trident.add_ion_fields(ds, ions=['Si II', 'Si III'])

        zsnap = ds.get_parameter('CosmologyCurrentRedshift')
        proper_box_size = get_proper_box_size(ds)

        refine_box, refine_box_center, refine_width = get_refine_box(ds, zsnap, track)
        refine_width = refine_width * proper_box_size

        # center is trying to be the center of the halo
        center = get_halo_center(ds, refine_box_center)
        width_code = width / proper_box_size ## needs to be in code units
        box = ds.r[center[0] - 0.5*width_code : center[0] + 0.5*width_code, \
                   center[1] - 0.5*width_code : center[1] + 0.5*width_code, \
                   center[2] - 0.5*width_code : center[2] + 0.5*width_code]
        # box = ds.r[ center[0]-wide/143886:center[0]+wide/143886, center[1]-wide/143886.:center[1]+wide/143886., center[2]-wide/143886.:center[2]+wide/143886.]

        #### this was for the off-center box
        # center = [centerx, centery+20. / 143886., centerz]
        # box = ds.r[ center[0]-wide/143886:center[0]+wide/143886, center[1]-wide/143886.:center[1]+wide/143886., center[2]-wide/143886.:center[2]+wide/143886.]

        if args.all or args.physical or args.density or args.slices:
            make_density_slice_plot(ds, prefix, axis=axis, center=center, box=refine_box, \
                               width=(refine_width-10.), appendix="_refine")

        if args.all or args.physical or args.metals or args.slices:
            make_metal_slice_plot(ds, prefix, axis=axis, center=center, box=refine_box, \
                                  width=(refine_width-10.), appendix="_refine")

        if args.all or args.physical or args.slices:
            make_temperature_slice_plot(ds, prefix, axis=axis, center=center, box=refine_box, \
                                  width=(refine_width-10.), appendix="_refine")

        if args.all or args.physical or args.density:
            print(width, refine_width, default_width)
            make_density_projection_plot(ds, prefix, axis=axis, center=center, box=box, \
                               width=width, appendix="_box")
            make_density_projection_plot(ds, prefix, axis=axis, center=refine_box_center, box=refine_box, \
                               width=refine_width, appendix="_refine")

        if args.all or args.physical:
            make_temperature_projection_plot(ds, prefix, axis=axis, center=refine_box_center,\
                             box=refine_box, width=refine_width, appendix="_refine")
            make_temperature_projection_plot(ds, prefix, axis=axis, center=center, box=box, width=width, appendix="_box")

            make_entropy_plots(ds, prefix, axis=axis, center=center, box=box, width=width, appendix="_box")
            make_entropy_plots(ds, prefix, axis=axis, center=refine_box_center, \
                               box=refine_box, width=refine_width, appendix="_refine")

        if args.all or args.physical or args.metals:
            make_metal_projection_plot(ds, prefix, axis=axis, center=refine_box_center,\
                             box=refine_box, width=refine_width, appendix="_refine")
            make_metal_projection_plot(ds, prefix, axis=axis, center=center, box=box, width=width, appendix="_box")

        if args.all or args.ions or args.hi:
            make_hi_plots(ds, prefix,  center=refine_box_center, \
                          box=refine_box, width=refine_width, appendix="_refine")
            make_hi_plots(ds, prefix, axis=axis, center=center, box=box, width=width, appendix="_box")

        if args.all or args.ions or args.ovi:
            make_o6_plots(ds, prefix, axis=axis, center=refine_box_center, \
                          box=refine_box, width=refine_width, appendix="_refine")
            make_o6_plots(ds, prefix, axis=axis, center=center, box=box, width=width, appendix="_box")

        if args.all or args.ions or args.civ:
            make_c4_plots(ds, prefix, axis=axis, center=refine_box_center, \
                          box=refine_box, width=refine_width, appendix="_refine")
            make_c4_plots(ds, prefix, axis=axis, center=center, box=box, width=width, appendix="_box")

        if args.all or args.ions or args.silicon:
            make_si2_plots(ds, prefix, axis=axis, center=refine_box_center, box=refine_box, width=refine_width, appendix="_refine")
            make_si2_plots(ds, prefix, axis=axis, center=center, box=box, width=width, appendix="_box")
            make_si3_plots(ds, prefix, axis=axis, center=refine_box_center, box=refine_box, width=refine_width, appendix="_refine")
            make_si3_plots(ds, prefix, axis=axis, center=center, box=box, width=width, appendix="_box")

    return "yay plots! all done!"


#-----------------------------------------------------------------------------------------------------

if __name__ == "__main__":
    args = parse_args()

    if not args.clobber:
        print("NO-CLOBBER IS NOT ACTUALLY IMPLEMENTED SO I'M GOING TO CLOBBER AWAY clobber clobber clobber")

    # message = plot_script(args.halo, "symmetric_box_tracking/nref11f_50kpc", "x")
    message = plot_script(args.halo, "nref11n/nref11n_nref10f_refine200kpc_z4to2", "all", outs=["RD0015/RD0015"])
    # message = plot_script(args.halo, "nref11n/nref11f_refine200kpc_z4to2", "all")
    sys.exit(message)
