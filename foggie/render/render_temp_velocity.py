# render_temp_velocity.py
# Author: Cassi
# Makes volume renders of temperature and radial velocity.

# Import everything as needed
from __future__ import print_function

import numpy as np
import yt
import unyt
from yt.units import *
from yt import YTArray
import argparse
import os
import glob
import sys
from astropy.table import Table
from astropy.io import ascii
from astropy.cosmology import Planck15 as cosmo
import multiprocessing as multi
import scipy.ndimage as ndimage
from scipy.ndimage import gaussian_filter
from scipy.ndimage import uniform_filter1d
from scipy.ndimage import uniform_filter
from scipy import interpolate
from skimage.measure import regionprops
from scipy.spatial import cKDTree
from datetime import timedelta
import time
from scipy.interpolate import InterpolatedUnivariateSpline as IUS
from scipy.interpolate import NearestNDInterpolator
import shutil
import ast
import trident
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm
import healpy
import cmasher as cmr
from matplotlib.patches import Ellipse
import copy
import random

# These imports are FOGGIE-specific files
from foggie.utils.consistency import *
from foggie.utils.get_run_loc_etc import get_run_loc_etc
from foggie.utils.yt_fields import *
from foggie.utils.foggie_load import *
from foggie.utils.analysis_utils import *

# These imports for datashader plots
import datashader as dshader
from datashader.utils import export_image
import datashader.transfer_functions as tf
import pandas as pd
import matplotlib as mpl

def parse_args():
    '''Parse command line arguments. Returns args object.
    NOTE: Need to move command-line argument parsing to separate file.'''

    parser = argparse.ArgumentParser(description='Calculates and saves to file a bunch of fluxes.')

    # Optional arguments:
    parser.add_argument('--halo', metavar='halo', type=str, action='store', \
                        help='Which halo? Default is 8508 (Tempest)')
    parser.set_defaults(halo='8508')

    parser.add_argument('--run', metavar='run', type=str, action='store', \
                        help='Which run? Default is nref11c_nref9f')
    parser.set_defaults(run='nref11c_nref9f')

    parser.add_argument('--output', metavar='output', type=str, action='store', \
                        help='Which output(s)? Options: Specify a single output (this is default' \
                        + ' and the default output is RD0036) or specify a range of outputs ' + \
                        'using commas to list individual outputs and dashes for ranges of outputs ' + \
                        '(e.g. "RD0020-RD0025" or "DD1341,DD1353,DD1600-DD1700", no spaces!)')
    parser.set_defaults(output='RD0034')

    parser.add_argument('--output_step', metavar='output_step', type=int, action='store', \
                        help='If you want to do every Nth output, this specifies N. Default: 1 (every output in specified range)')
    parser.set_defaults(output_step=1)

    parser.add_argument('--system', metavar='system', type=str, action='store', \
                        help='Which system are you on? Default is cassiopeia')
    parser.set_defaults(system='cassiopeia')

    parser.add_argument('--pwd', dest='pwd', action='store_true',
                        help='Just use the working directory?, Default is no')
    parser.set_defaults(pwd=False)

    parser.add_argument('--save_suffix', metavar='save_suffix', type=str, action='store', \
                        help='Do you want to append a string onto the names of the saved files? Default is no.')
    parser.set_defaults(save_suffix="")

    parser.add_argument('--field_to_render', metavar='field_to_render', type=str, action='store', \
                        help='What field do you want to render? Options are:\n' + \
                        'temperature\n' + \
                        'velocity\n' + \
                        'shock_mach     -  Mach number from the Enzo shock finder output\n' + \
                        'Default is temperature.')
    parser.set_defaults(field_to_render='temperature')

    parser.add_argument('--rotation_render', dest='rotation_render', action='store_true', \
                        help='Specify this if you want the camera to rotate around the galaxy at each\n' + \
                        'snapshot.')
    parser.set_defaults(rotation_render=False)

    parser.add_argument('--time_render', dest='time_render', action='store_true', \
                        help='Specify this if you want the camera to be fixed as the galaxy evolves.')
    parser.set_defaults(time_render=False)

    parser.add_argument('--width', metavar='width', type=float, action='store', \
                        help='What width for the volume render do you want, in kpc?')
    parser.set_defaults(width=6000.)

    parser.add_argument('--nproc', metavar='nproc', type=int, action='store', \
                        help='How many processes do you want? Default is 1 ' + \
                        '(no parallelization), if multiple outputs and multiple processors are' + \
                        ' specified, code will run one output per processor.\n' + \
                        'This option only has meaning for running things on multiple snapshots.')
    parser.set_defaults(nproc=1)

    parser.add_argument('--copy_to_tmp', dest='copy_to_tmp', action='store_true', \
                        help='If running on pleiades, do you want to copy the simulation output into' + \
                        "the node's /tmp directory before analysis? Default is no.")
    parser.set_defaults(copy_to_tmp=False)

    args = parser.parse_args()
    return args

def render_snapshot(snap):

    # The shock finder's Mach number field lives in its own separate Enzo output tree, so we need
    # to point at that directory instead of the regular snapshot when rendering it.
    snap_subdir = 'shock_finding/' if (args.field_to_render=='shock_mach') else ''

    # Load simulation output
    if (args.system=='pleiades_cassi'):
        print('Copying directory to /tmp')
        if (args.copy_to_tmp):
            snap_dir = '/tmp/' + args.halo + '/' + args.run + '/' + target_dir + '/' + snap
            shutil.copytree(foggie_dir + run_dir + snap_subdir + snap, snap_dir)
            snap_name = snap_dir + '/' + snap
        else:
            # Make a dummy directory with the snap name so the script later knows the process running
            # this snapshot failed if the directory is still there
            snap_dir = '/nobackup/clochhaa/tmp/' + args.halo + '/' + args.run + '/' + target_dir + '/' + snap
            os.makedirs(snap_dir)
            snap_name = foggie_dir + run_dir + snap_subdir + snap + '/' + snap
    else:
        snap_name = foggie_dir + run_dir + snap_subdir + snap + '/' + snap
    ds, refine_box = foggie_load(snap_name, trackfile_name=trackname, do_filter_particles=False, halo_c_v_name=halo_c_v_name)
    zsnap = ds.get_parameter('CosmologyCurrentRedshift')

    box = ds.box(left_edge=ds.halo_center_kpc-ds.arr([args.width/2.,args.width/2.,args.width/2], 'kpc'), right_edge=ds.halo_center_kpc+ds.arr([args.width/2.,args.width/2.,args.width/2], 'kpc'))

    n_rotation_frames = 40     # number of frames in a full rotation for --rotation_render

    # Set up the field, bounds, colormap, and exact contour placement for the requested quantity.
    # contour_values are in the field's physical units (K for temperature, km/s for velocity);
    # contour_widths are the width of each contour in that same space (dex for temperature, since
    # it's rendered in log space; km/s for velocity); contour_alphas set each contour's opacity.
    # continuous_cmap fields (e.g. shock_mach) skip discrete contours entirely and instead map the
    # whole bounds range onto the colormap, since the field is already zero/floor everywhere except
    # at the thin structures of interest - no need to fake that thinness with narrow value bins.
    continuous_cmap = False
    if (args.field_to_render=='temperature'):
        field = ('gas', 'temperature')
        field_file = 'temperature'
        bounds = (10**3., 10**7.)
        use_log = True
        cmap_colors = sns.blend_palette(('salmon', "#984ea3", "#4daf4a", "#ffe34d"), as_cmap=True)
        cmap = 'foggie_temperature_cmap'
        contour_values = [10**4.5,10**5.5,10**6.5]      # K
        contour_widths = [0.001, 0.001, 0.001]        # dex
        contour_alphas = [0.5,0.75,1.0]
    elif (args.field_to_render=='velocity'):
        field = ('gas', 'radial_velocity_corrected')
        field_file = 'velocity'
        bounds = (-300, 300)
        use_log = False
        cmap_colors = sns.blend_palette(('#d73027', 'white', '#4575b4'), as_cmap=True)
        cmap = 'foggie_velocity_cmap'
        contour_values = [-300, -100, 0, 100, 300]      # km/s
        contour_widths = [5., 5., 5., 5., 5.]        # km/s
        contour_alphas = [1.0, 0.7, 0.4, 0.7, 1.0]
    elif (args.field_to_render=='shock_mach'):
        field = ('enzo', 'Mach')
        field_file = 'shock_mach'
        mach_cut = 2.0            # below this, Enzo's shock finder considers a cell unshocked
        mach_max = 10.
        bounds = (mach_cut, mach_max)
        use_log = False
        cmap_colors = mpl.colormaps['viridis']
        cmap = 'foggie_shock_mach_cmap'
        continuous_cmap = True
    else:
        sys.exit("field_to_render must be 'temperature', 'velocity', or 'shock_mach'.")

    # yt's transfer function API looks colormaps up by name (mpl.colormaps[name]), so the
    # Colormap object built above must be registered with matplotlib before yt can use it.
    if (cmap not in mpl.colormaps):
        mpl.colormaps.register(cmap_colors, name=cmap)

    # Build the scene and the transfer function: either discrete contour layers (temperature,
    # velocity), or a single colormap spanning the bounds with alpha increasing away from the
    # "no shock" floor (shock_mach), so nothing below bounds[0] contributes at all.
    sc = yt.create_scene(box, field=field)
    source = sc[0]
    source.set_field(field)
    source.set_log(use_log)
    tf_bounds = np.log10(bounds) if use_log else bounds
    tf = yt.ColorTransferFunction(tf_bounds)
    if (continuous_cmap):
        tf.map_to_colormap(tf_bounds[0], tf_bounds[1], scale=1.0, colormap=cmap, \
                            scale_func=lambda vals, minval, maxval: (vals-minval)/(maxval-minval))
    else:
        for v, w, a in zip(contour_values, contour_widths, contour_alphas):
            v_tf = np.log10(v) if use_log else v
            tf.sample_colormap(v_tf, w, alpha=a, colormap=cmap, col_bounds=tf_bounds)
    source.tfh.tf = tf
    source.tfh.bounds = bounds
    source.tfh.grey_opacity = True
    source.tfh.plot(prefix + snap + '_' + field_file + '_transfer_function' + save_suffix + '.png', profile_field=field)

    # Set up the camera, centered on the halo and framing the box. For a plane-parallel lens, rays
    # are parallel (no vanishing point), so cam.width directly sets the physical extent shown
    # (horizontal, vertical) and the integration depth along the line of sight, centered on focus -
    # matching box's extent exactly gives the same footprint as a ProjectionPlot(ds, 'x', ...,
    # width=args.width). cam.position only sets the viewing direction now; its distance from focus
    # no longer affects the field of view the way it did for the perspective lens.
    cam = sc.add_camera(box, lens_type='plane-parallel')
    cam.position = ds.halo_center_kpc - ds.arr([args.width, 0, 0], 'kpc')
    cam.focus = ds.halo_center_kpc
    cam.north_vector = [0, 0, 1]
    cam.width = args.width * yt.units.kpc
    cam.switch_view()

    # Render snapshot according to command line options
    if (args.rotation_render):
        sc.render()
        sc.save(prefix + snap + '_' + field_file + '_render_0000' + save_suffix + '.png', sigma_clip=4)
        frame = 1
        for _ in cam.iter_rotate(2.*np.pi*(n_rotation_frames-frame)/n_rotation_frames, n_rotation_frames-frame, rot_center=ds.halo_center_kpc):
            sc.save(prefix + snap + '_' + field_file + '_render_%04i' % frame + save_suffix + '.png', sigma_clip=4)
            frame += 1
    else:
        sc.render()
        sc.save(prefix + snap + '_' + field_file + '_render' + save_suffix + '.png', sigma_clip=4)


if __name__ == "__main__":

    start = time.perf_counter()

    gtoMsun = 1.989e33
    cmtopc = 3.086e18
    stoyr = 3.155e7
    light_c = 2.998e10
    G = 6.673e-8
    kB = 1.38e-16
    mu = 0.6
    mp = 1.67e-24
    dt = 5.38e6

    args = parse_args()
    print(args.halo)
    print(args.run)
    print(args.system)
    foggie_dir, output_dir, run_dir, code_path, trackname, haloname, spectra_dir, infofile = get_run_loc_etc(args)
    #foggie_dir = '/Volumes/Data/Simulation_Data/'

    if ('feedback' in args.run) and ('track' in args.run):
        if (args.system=='pleiades_cassi'):
            foggie_dir = '/nobackupnfs1/jtumlins/halo_008508/feedback-track/'
        else:
            foggie_dir = '/Users/clochhaas/Documents/Research/FOGGIE/Simulation_Data/halo_008508/'
        run_dir = args.run + '/'

    # Set directory for output location, making it if necessary
    prefix = output_dir + 'render_halo_00' + args.halo + '/' + args.run + '/'
    if not (os.path.exists(prefix)): os.system('mkdir -p ' + prefix)

    print('foggie_dir: ', foggie_dir)
    catalog_dir = code_path + 'halo_infos/00' + args.halo + '/' + args.run + '/'
    halo_c_v_name = catalog_dir + 'halo_c_v'
    smooth_AM_name = catalog_dir + 'AM_direction_smoothed'

    if (args.save_suffix!=''):
        save_suffix = '_' + args.save_suffix
    else:
        save_suffix = ''

    outs = make_output_list(args.output, output_step=args.output_step)

    target_dir = 'render'
    if (args.nproc==1):
        for snap in outs:
            render_snapshot(snap)
    else:
        skipped_outs = outs
        while (len(skipped_outs)>0):
            skipped_outs = []
            # Split into a number of groupings equal to the number of processors
            # and run one process per processor
            for i in range(len(outs)//args.nproc):
                threads = []
                snaps = []
                for j in range(args.nproc):
                    snap = outs[args.nproc*i+j]
                    snaps.append(snap)
                    threads.append(multi.Process(target=render_snapshot, args=[snap]))
                for t in threads:
                    t.start()
                for t in threads:
                    t.join()
                # Delete leftover outputs from failed processes from tmp directory if on pleiades
                if (args.system=='pleiades_cassi'):
                    if (args.copy_to_tmp):
                        snap_dir = '/tmp/' + args.halo + '/' + args.run + '/' + target_dir + '/'
                    else:
                        snap_dir = '/nobackup/clochhaa/tmp/' + args.halo + '/' + args.run + '/' + target_dir + '/'
                    for s in range(len(snaps)):
                        if (os.path.exists(snap_dir + snaps[s])):
                            print('Deleting failed %s from /tmp' % (snaps[s]))
                            skipped_outs.append(snaps[s])
                            shutil.rmtree(snap_dir + snaps[s])
            # For any leftover snapshots, run one per processor
            threads = []
            snaps = []
            for j in range(len(outs)%args.nproc):
                snap = outs[-(j+1)]
                snaps.append(snap)
                threads.append(multi.Process(target=render_snapshot, args=[snap]))
            for t in threads:
                t.start()
            for t in threads:
                t.join()
            # Delete leftover outputs from failed processes from tmp directory if on pleiades
            if (args.system=='pleiades_cassi'):
                if (args.copy_to_tmp):
                    snap_dir = '/tmp/' + args.halo + '/' + args.run + '/' + target_dir + '/'
                else:
                    snap_dir = '/nobackup/clochhaa/tmp/' + args.halo + '/' + args.run + '/' + target_dir + '/'
                for s in range(len(snaps)):
                    if (os.path.exists(snap_dir + snaps[s])):
                        print('Deleting failed %s from /tmp' % (snaps[s]))
                        skipped_outs.append(snaps[s])
                        shutil.rmtree(snap_dir + snaps[s])
            outs = skipped_outs

    end = time.perf_counter()
    elapsed = end - start
    duration = timedelta(seconds=elapsed)
    print("All snapshots finished!")
    print(f"Elapsed time: {duration}")