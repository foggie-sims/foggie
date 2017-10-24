from astropy.table import Table
import numpy as np

import yt
import trident

import argparse
import os
import glob
import sys

from consistency import *

foggie_dir = "/astro/simulations/FOGGIE/"
# foggie_dir = "/Users/molly/foggie/"  ## where the simulations live
dropbox_dir = "/Users/molly/Dropbox/foggie-collab/"  ## outputs go here


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

    parser.add_argument('--density', dest='density', action='store_true',
                        help='make density plots?, default if not')
    parser.set_defaults(density=False)

    parser.add_argument('--metals', dest='metals', action='store_true',
                        help='make metallicity plots?, default if not')
    parser.set_defaults(metals=False)

    args = parser.parse_args()
    return args

#-----------------------------------------------------------------------------------------------------


def make_density_plots(ds, prefix, **kwargs):
    axis = kwargs.get("axis", ['x','y','z']) # if axis not set, do all
    box = kwargs.get("box", "")
    center = kwargs.get("center", "")
    appendix = kwargs.get("appendix", "")
    width = kwargs.get("width", default_width)
    if not (os.path.exists(prefix + 'plots/density_projection_map/' )):
        os.system("mkdir " + prefix + 'plots/density_projection_map/' )
    for ax in axis:
        if not (os.path.exists(prefix + 'plots/density_projection_map/' + axis)):
            os.system("mkdir " + prefix + 'plots/density_projection_map/' + axis)
        p = yt.ProjectionPlot(ds, ax, 'density', center=center, data_source=box, width=(width, 'kpc'))
        p.annotate_timestamp(corner='upper_left', redshift=True, draw_inset_box=True)
        p.set_cmap(field="density", cmap=density_color_map)
        p.set_zlim("density", density_min, density_max)
        p.save(prefix + 'plots/density_projection_map/' + axis + '/' + ds.basename + appendix)

def make_metal_plots(ds, prefix, **kwargs):
    axis = kwargs.get("axis", ['x','y','z']) # if axis not set, do all
    box = kwargs.get("box", "")
    center = kwargs.get("center", "")
    appendix = kwargs.get("appendix", "")
    width = kwargs.get("width", default_width)
    if not (os.path.exists(prefix + 'plots/metallicity_slice_map/' )):
        os.system("mkdir " + prefix + 'plots/metallicity_slice_map/' )
    for ax in axis:
        if not (os.path.exists(prefix + 'plots/metallicity_slice_map/' + axis)):
            os.system("mkdir " + prefix + 'plots/metallicity_slice_map/' + axis)
        p = yt.SlicePlot(ds, axis, 'metallicity', center=center, data_source=box, width=(width, 'kpc'))
        p.annotate_timestamp(corner='upper_left', redshift=True, draw_inset_box=True)
        p.set_cmap(field="metallicity", cmap=metal_color_map)
        p.set_zlim("metallicity", metal_min, metal_max)
        p.save(prefix + 'plots/metallicity_slice_map/' + axis + '/' + ds.basename + appendix)

def make_temperature_plots(ds, prefix, **kwargs):
    axis = kwargs.get("axis", ['x','y','z']) # if axis not set, do all
    box = kwargs.get("box", "")
    center = kwargs.get("center", "")
    appendix = kwargs.get("appendix", "")
    width = kwargs.get("width", default_width)
    if not (os.path.exists(prefix + 'plots/temperature_map/' )):
        os.system("mkdir " + prefix + 'plots/temperature_map/' )
    for ax in axis:
        if not (os.path.exists(prefix + 'plots/temperature_map/' + axis)):
            os.system("mkdir " + prefix + 'plots/temperature_map/' + axis)
        p = yt.SlicePlot(ds, axis, 'temperature', center=center, data_source=box, width=(width, 'kpc'))
        p.annotate_timestamp(corner='upper_left', redshift=True, draw_inset_box=True)
        p.set_cmap(field="temperature", cmap=temperature_color_map)
        p.set_zlim("temperature", temperature_min, temperature_max)
        p.save(prefix + 'plots/temperature_map/' + axis + '/' + ds.basename + appendix)

def make_entropy_plots(ds, prefix, **kwargs):
    axis = kwargs.get("axis", ['x','y','z']) # if axis not set, do all
    box = kwargs.get("box", "")
    center = kwargs.get("center", "")
    appendix = kwargs.get("appendix", "")
    width = kwargs.get("width", default_width)
    if not (os.path.exists(prefix + 'plots/entropy_map/' )):
        os.system("mkdir " + prefix + 'plots/entropy_map/' )
    for ax in axis:
        if not (os.path.exists(prefix + 'plots/entropy_map/' + axis )):
            os.system("mkdir " + prefix + 'plots/entropy_map/' + axis)
        p = yt.SlicePlot(ds, axis, 'entropy', center=center, data_source=box, width=(width, 'kpc'))
        p.annotate_timestamp(corner='upper_left', redshift=True, draw_inset_box=True)
        p.set_cmap(field="entropy", cmap=entropy_color_map)
        p.set_zlim("entropy", entropy_min, entropy_max)
        p.save(prefix + 'plots/entropy_map/' + axis + '/' + ds.basename + appendix)

def make_hi_plots(ds, prefix, **kwargs):
    axis = kwargs.get("axis", ['x','y','z']) # if axis not set, do all
    box = kwargs.get("box", "")
    center = kwargs.get("center", "")
    appendix = kwargs.get("appendix", "")
    width = kwargs.get("width", default_width)
    if not (os.path.exists(prefix + 'plots/HI_projection_map/' )):
        os.system("mkdir " + prefix + 'plots/HI_projection_map/' )
    for ax in axis:
        if not (os.path.exists(prefix + 'plots/HI_projection_map/' + axis )):
            os.system("mkdir " + prefix + 'plots/HI_projection_map/' + axis )
        p = yt.ProjectionPlot(ds, axis, 'H_p0_number_density', center=center, data_source=box, width=(width,'kpc'))
        p.annotate_timestamp(corner='upper_left', redshift=True, draw_inset_box=True)
        p.set_cmap(field="H_p0_number_density", cmap=h1_color_map)
        p.set_zlim("H_p0_number_density",h1_min, h1_max)
        p.save(prefix + 'plots/HI_projection_map/'  +axis+ '/' + ds.basename + '_HI' + appendix)

def make_o6_plots(ds, prefix, **kwargs):
    axis = kwargs.get("axis", ['x','y','z']) # if axis not set, do all
    box = kwargs.get("box", "")
    center = kwargs.get("center", "")
    appendix = kwargs.get("appendix", "")
    width = kwargs.get("width", default_width)
    if not (os.path.exists(prefix + 'plots/OVI_projection_map/' )):
        os.system("mkdir " + prefix + 'plots/OVI_projection_map/' )
    for ax in axis:
        if not (os.path.exists(prefix + 'plots/OVI_projection_map/' + axis )):
            os.system("mkdir " + prefix + 'plots/OVI_projection_map/' + axis )
        p = yt.ProjectionPlot(ds, axis, 'O_p5_number_density', center=center, data_source=box, width=(width,'kpc'))
        p.annotate_timestamp(corner='upper_left', redshift=True, draw_inset_box=True)
        p.set_cmap(field="O_p5_number_density", cmap=o6_color_map)
        p.set_zlim("O_p5_number_density", o6_min, o6_max)
        p.save(prefix + 'plots/OVI_projection_map/'  +axis+ '/' + ds.basename + '_OVI' + appendix)

def make_c4_plots(ds, prefix, **kwargs):
    axis = kwargs.get("axis", ['x','y','z']) # if axis not set, do all
    box = kwargs.get("box", "")
    center = kwargs.get("center", "")
    appendix = kwargs.get("appendix", "")
    width = kwargs.get("width", default_width)
    if not (os.path.exists(prefix + 'plots/CIV_projection_map/' )):
        os.system("mkdir " + prefix + 'plots/CIV_projection_map/' )
    for ax in axis:
        if not (os.path.exists(prefix + 'plots/CIV_projection_map/' + axis )):
            os.system("mkdir " + prefix + 'plots/CIV_projection_map/' + axis )
        p = yt.ProjectionPlot(ds, axis, 'C_p3_number_density', center=center, data_source=box, width=(width,'kpc'))
        p.annotate_timestamp(corner='upper_left', redshift=True, draw_inset_box=True)
        p.set_cmap(field="C_p3_number_density", cmap=c4_color_map)
        p.set_zlim("C_p3_number_density", c4_min, c4_max)
        p.save(prefix + 'plots/CIV_projection_map/'  +axis+ '/' + ds.basename + '_CIV' + appendix)

def make_si3_plots(ds, prefix, **kwargs):
    axis = kwargs.get("axis", ['x','y','z']) # if axis not set, do all
    box = kwargs.get("box", "")
    center = kwargs.get("center", "")
    appendix = kwargs.get("appendix", "")
    width = kwargs.get("width", default_width)
    if not (os.path.exists(prefix + 'plots/SiIII_projection_map/' )):
        os.system("mkdir " + prefix + 'plots/SiIII_projection_map/' )
    for ax in axis:
        if not (os.path.exists(prefix + 'plots/SiIII_projection_map/' + axis )):
            os.system("mkdir " + prefix + 'plots/SiIII_projection_map/' + axis )
        p = yt.ProjectionPlot(ds, axis, 'Si_p2_number_density', center=center, data_source=box, width=(width,'kpc'))
        p.annotate_timestamp(corner='upper_left', redshift=True, draw_inset_box=True)
        p.set_cmap(field="Si_p2_number_density", cmap=c4_color_map)
        p.set_zlim("Si_p2_number_density", c4_min, c4_max)
        p.save(prefix + 'plots/SiIII_projection_map/'  +axis+ '/' + ds.basename + '_SiIII' + appendix)

def get_refine_box(ds, zsnap, track):
    x_left = np.interp(zsnap, track['col1'], track['col2'])
    y_left = np.interp(zsnap, track['col1'], track['col3'])
    z_left = np.interp(zsnap, track['col1'], track['col4'])
    x_right = np.interp(zsnap, track['col1'], track['col5'])
    y_right = np.interp(zsnap, track['col1'], track['col6'])
    z_right = np.interp(zsnap, track['col1'], track['col7'])

    refine_box_center = [0.5*(x_left+x_right), 0.5*(y_left+y_right), 0.5*(z_left+z_right)]
    refine_box = ds.r[x_left:x_right, y_left:y_right, z_left:z_right]

    return refine_box, refine_box_center

#-----------------------------------------------------------------------------------------------------

def plot_script(halo, run, axis, **kwargs):
    outs = kwargs.get("outs", "all")
    trackname = kwargs.get("trackname", "halo_track")
    wide = default_width

    track_name = foggie_dir + 'halo_00' + str(halo) + '/' + run + '/' + trackname
    print("opening track: " + track_name)
    track = Table.read(track_name, format='ascii')
    track.sort('col1')

    ## default is do allll the snaps in the directory
    ## want to add flag for if just one
    run_dir = foggie_dir + 'halo_00' + str(halo) + '/' + run
    prefix = dropbox_dir + 'plots/halo_00' + str(halo) + '/' + run + '/'
    if not (os.path.exists(prefix)):
        os.system("mkdir " + prefix)
    if not (os.path.exists(prefix + "/plots")):
        os.system("mkdir " + prefix + "/plots")

    if outs == "all":
        print "looking for outputs in ", run_dir
        outs = glob.glob(os.path.join(run_dir, 'DD0???/?D0???'))

    print "making plots for ", axis, " axis in ", outs

    for snap in outs:
        # load the snapshot
        print('opening snapshot '+ snap)
        ds = yt.load(snap)
        trident.add_ion_fields(ds, ions=['C IV', 'O VI', 'H I', 'Si II', 'C II', 'Si III'])
        zsnap = ds.get_parameter('CosmologyCurrentRedshift')

        def _msun_density(field, data):
            return data["density"]*1.0

        ds.add_field(("gas","Msun_density"),function=_msun_density, units="Msun/pc**3")

        # interpolate the center from the track
        centerx = 0.5 * ( np.interp(zsnap, track['col1'], track['col2']) + np.interp(zsnap, track['col1'], track['col5']))
        ### np.interp(zsnap, track['col1'], track['col2'])
        centery = 0.5 * ( np.interp(zsnap, track['col1'], track['col3']) + np.interp(zsnap, track['col1'], track['col6']))
        #### np.interp(zsnap, track['col1'], track['col3'])
        centerz = 0.5 * ( np.interp(zsnap, track['col1'], track['col4']) + np.interp(zsnap, track['col1'], track['col7']))

        center = [centerx, centery, centerz]
        box = ds.r[ center[0]-wide/143886:center[0]+wide/143886, center[1]-wide/143886.:center[1]+wide/143886., center[2]-wide/143886.:center[2]+wide/143886.]

        #### this was for the off-center box
        # center = [centerx, centery+20. / 143886., centerz]
        # box = ds.r[ center[0]-wide/143886:center[0]+wide/143886, center[1]-wide/143886.:center[1]+wide/143886., center[2]-wide/143886.:center[2]+wide/143886.]

        refine_box, refine_box_center = get_refine_box(ds, zsnap, track)

        if args.all or args.physical or args.density:
            make_density_plots(ds, prefix, axis=axis, center=center, box=box, appendix="_box")
            make_density_plots(ds, prefix, axis=axis, center=refine_box_center, box=refine_box, appendix="_refine")

        if args.all or args.physical:
            make_temperature_plots(ds, prefix, axis=axis, center=center, box=box, appendix="_box")
            make_temperature_plots(ds, prefix, axis=axis, center=refine_box_center, box=refine_box, appendix="_refine")

            make_entropy_plots(ds, prefix, axis=axis, center=center, box=box, appendix="_box")
            make_entropy_plots(ds, prefix, axis=axis, center=refine_box_center, box=refine_box, appendix="_refine")

        if args.all or args.physical or args.metals:
            make_metal_plots(ds, prefix, axis=axis, center=center, box=box, appendix="_box")
            make_metal_plots(ds, prefix, axis=axis, center=refine_box_center, box=refine_box, appendix="_refine")

        if args.all or args.ions or args.hi:
            make_hi_plots(ds, prefix, axis=axis, center=refine_box_center, box=refine_box, appendix="_refine")
            make_hi_plots(ds, prefix, axis=axis, center=center, box=box, appendix="_box")

        if args.all or args.ions or args.ovi:
            make_o6_plots(ds, prefix, axis=axis, center=refine_box_center, box=refine_box, appendix="_refine")
            make_o6_plots(ds, prefix, axis=axis, center=center, box=box, appendix="_box")

        if args.all or args.ions or args.civ:
            make_c4_plots(ds, prefix, axis=axis, center=refine_box_center, box=refine_box, appendix="_refine")
            make_c4_plots(ds, prefix, axis=axis, center=center, box=box, appendix="_box")

        if args.all or args.ions:
            make_si3_plots(ds, prefix, axis=axis, center=refine_box_center, box=refine_box, appendix="_refine")
            make_si3_plots(ds, prefix, axis=axis, center=center, box=box, appendix="_box")

    return "yay plots! all done!"


#-----------------------------------------------------------------------------------------------------

if __name__ == "__main__":
    args = parse_args()

    if not args.clobber:
        print "NO-CLOBBER IS NOT ACTUALLY IMPLEMENTED SO I'M GOING TO CLOBBER AWAY"

    message = plot_script(args.halo, "symmetric_box_tracking/nref11f_50kpc", "x")
    sys.exit(message)
