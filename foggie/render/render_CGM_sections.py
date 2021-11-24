# render_CGM_sections.py
# Author: Cassi
# Makes volume renders of the CGM "sections" by putting contours only at the edges of the extreme
# hot and cold regions, high and low metallicity regions, or fast inflow and outflow regions leaving
# all intermediate gas out. Heavy borrowing from Ayan's volume_rendering_movie.py.

# Import everything as needed
from __future__ import print_function

import numpy as np
import yt
from yt.units import *
from yt import YTArray
import argparse
import os
import glob
import sys
from astropy.table import Table
from astropy.io import ascii
import multiprocessing as multi
import datetime
from scipy.interpolate import InterpolatedUnivariateSpline as IUS
from scipy.interpolate import RegularGridInterpolator
import shutil
import ast
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter
from scipy.ndimage import uniform_filter1d
from scipy.ndimage import rotate
import copy
import matplotlib.colors as colors

# These imports are FOGGIE-specific files
from foggie.utils.consistency import *
from foggie.utils.get_refine_box import get_refine_box
from foggie.utils.get_halo_center import get_halo_center
from foggie.utils.get_proper_box_size import get_proper_box_size
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
                        'temperature        -  contours are at > 10^6 K and < 10^5 K\n' + \
                        'metallicity        -  contours are at > 0.8Zsun and < 0.05Zsun\n' + \
                        'velocity           -  contours are at < -75 km/s and > 200 km/s\n' + \
                        'Default is temperature.')
    parser.set_defaults(field_to_render='temperature')

    parser.add_argument('--rotation_render', dest='rotation_render', action='store_true', \
                        help='Specify this if you want the camera to rotate around the galaxy at each\n' + \
                        'snapshot.')
    parser.set_defaults(rotation_render=False)

    parser.add_argument('--time_render', dest='time_render', action='store_true', \
                        help='Specify this if you want the camera to be fixed as the galaxy evolves.')
    parser.set_defaults(time_render=False)

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

def rendering_rotation(snap):
    '''Loads an output and makes several images of a slowly rotating volume render.'''

    if (args.system=='pleiades_cassi') and (foggie_dir!='/nobackupp18/mpeeples/') and (args.copy_to_tmp):
        print('Copying directory to /tmp')
        snap_dir = '/tmp/' + snap
        shutil.copytree(foggie_dir + run_dir + snap, snap_dir)
        snap_name = snap_dir + '/' + snap
    else:
        snap_name = foggie_dir + run_dir + snap + '/' + snap
    ds, refine_box = foggie_load(snap_name, trackname, do_filter_particles=False, halo_c_v_name=halo_c_v_name)
    zsnap = ds.get_parameter('CosmologyCurrentRedshift')

    if (args.field_to_render=='temperature'):
        bounds = (1e0, 1e8)
        cmap = []
        cmap.append(sns.blend_palette(('salmon','salmon'), as_cmap=True))
        cmap.append(sns.blend_palette(('salmon',"#984ea3"), as_cmap=True))
        cmap.append(sns.blend_palette(('#ffe34d', 'darkorange'), as_cmap=True))
        cmap.append(sns.blend_palette(('darkorange', 'darkorange'), as_cmap=True))
        cmap_bounds = []
        cmap_bounds.append([0, 4])
        cmap_bounds.append([4, 4.8])
        cmap_bounds.append([6.3, 7])
        cmap_bounds.append([7, 8])
        field_file = 'temperature'
    elif (args.field_to_render=='metallicity'):
        bounds = (1e-5, 10.)
        cmap = []
        cmap.append(sns.blend_palette(('#4575b4','#4575b4'), as_cmap=True))
        cmap.append(sns.blend_palette(('darkorange', '#ffe34d'), as_cmap=True))
        cmap_bounds = []
        cmap_bounds.append([-5, -2])
        cmap_bounds.append([0, 1])
        field_file = 'metallicity'
    elif (args.field_to_render=='velocity'):
        args.field_to_render = 'radial_velocity_corrected'
        bounds = (-500,500)
        cmap = []
        cmap.append(sns.blend_palette(('red','red'), as_cmap=True))
        cmap.append(sns.blend_palette(('blue','blue'), as_cmap=True))
        cmap_bounds = []
        cmap_bounds.append([-500, -75])
        cmap_bounds.append([75, 500])
        field_file = 'velocity'

    sc = yt.create_scene(refine_box, field=('gas',args.field_to_render))
    source = sc[0]
    source.set_field(("gas", args.field_to_render))
    if (args.field_to_render!='radial_velocity_corrected'):
        source.set_log(True)
        tf = yt.ColorTransferFunction(np.log10(bounds))
    else:
        source.set_log(False)
        tf = yt.ColorTransferFunction(bounds)
    for i in range(len(cmap)):
        tf.map_to_colormap(cmap_bounds[i][0], cmap_bounds[i][1], scale=1.0, colormap=cmap[i])
    source.tfh.tf = tf
    source.tfh.bounds = bounds
    source.tfh.grey_opacity = False
    source.tfh.plot(save_dir + snap + '_' + field_file + "_transfer_function" + save_suffix + ".png", profile_field=("gas", 'cell_volume'))

    cam = sc.add_camera(refine_box, lens_type='perspective')
    cam.position = ds.halo_center_kpc - ds.arr([ds.refine_width, 0, 0], 'kpc')
    cam.focus = ds.halo_center_kpc
    cam.north_vector = [0, 0, 1]
    cam.width = 2.*ds.refine_width * yt.units.kpc
    cam.switch_view()

    # Use this code block if restarting from partway through the rotation sequence
    #frame = 20
    #cam.rotate(frame/40. * np.pi, rot_center = ds.halo_center_kpc)
    #sc.render()
    #sc.save(save_dir + snap + '_' + field_file + "_render_00" + str(frame) + save_suffix + ".png", sigma_clip=2)

    # Use this code block if doing the whole rotation sequence
    sc.render()
    sc.save(save_dir + snap + '_' + field_file + "_render_0000" + save_suffix + ".png", sigma_clip=2)
    frame = 1

    for _ in cam.iter_rotate((40-frame)/40.*np.pi, 40-frame, rot_center = ds.halo_center_kpc):
        sc.save(save_dir + snap + '_' + field_file + "_render_%04i" % frame + save_suffix  + '.png', sigma_clip=2)
        frame += 1

def rendering_time(snap):
    '''Makes a volume render of the snapshot in 'snap' of the field 'field_to_render'.'''

    snap_name = foggie_dir + run_dir + snap + '/' + snap
    ds, refine_box = foggie_load(snap_name, trackname, do_filter_particles=False, halo_c_v_name=halo_c_v_name)
    zsnap = ds.get_parameter('CosmologyCurrentRedshift')

    if (args.field_to_render=='temperature'):
        bounds = (1e0, 1e8)
        cmap = []
        cmap.append(sns.blend_palette(('salmon','salmon'), as_cmap=True))
        cmap.append(sns.blend_palette(('salmon',"#984ea3"), as_cmap=True))
        cmap.append(sns.blend_palette(('#ffe34d', 'darkorange'), as_cmap=True))
        cmap.append(sns.blend_palette(('darkorange', 'darkorange'), as_cmap=True))
        cmap_bounds = []
        cmap_bounds.append([0, 4])
        cmap_bounds.append([4, 4.8])
        cmap_bounds.append([6.3, 7])
        cmap_bounds.append([7, 8])
        field_file = 'temperature'
    elif (args.field_to_render=='metallicity'):
        bounds = (1e-5, 10.)
        cmap = []
        cmap.append(sns.blend_palette(('#4575b4','#4575b4'), as_cmap=True))
        cmap.append(sns.blend_palette(('darkorange', '#ffe34d'), as_cmap=True))
        cmap_bounds = []
        cmap_bounds.append([-5, -2])
        cmap_bounds.append([0, 1])
        field_file = 'metallicity'
    elif (args.field_to_render=='velocity'):
        args.field_to_render = 'radial_velocity_corrected'
        bounds = (-500,500)
        cmap = []
        cmap.append(sns.blend_palette(('red','red'), as_cmap=True))
        cmap.append(sns.blend_palette(('blue','blue'), as_cmap=True))
        cmap_bounds = []
        cmap_bounds.append([-500, -75])
        cmap_bounds.append([75, 500])
        field_file = 'velocity'

    sc = yt.create_scene(refine_box, field=('gas',args.field_to_render))
    source = sc[0]
    source.set_field(("gas", args.field_to_render))
    if (args.field_to_render!='radial_velocity_corrected'):
        source.set_log(True)
        tf = yt.ColorTransferFunction(np.log10(bounds))
    else:
        source.set_log(False)
        tf = yt.ColorTransferFunction(bounds)
    for i in range(len(cmap)):
        tf.map_to_colormap(cmap_bounds[i][0], cmap_bounds[i][1], scale=1.0, colormap=cmap[i])
    source.tfh.tf = tf
    source.tfh.bounds = bounds
    source.tfh.grey_opacity = False
    source.tfh.plot(save_dir + snap + '_' + field_file + "_transfer_function" + save_suffix + ".png", profile_field=("gas", 'cell_volume'))

    cam = sc.add_camera(refine_box, lens_type='perspective')
    cam.position = ds.halo_center_kpc - ds.arr([ds.refine_width, 0, 0], 'kpc')
    cam.focus = ds.halo_center_kpc
    cam.north_vector = [0, 0, 1]
    cam.width = 2.*ds.refine_width * yt.units.kpc
    cam.switch_view()
    sc.render()
    sc.save(save_dir + snap + '_' + field_file + "_render" + save_suffix + ".png", sigma_clip=2)

if __name__ == "__main__":

    args = parse_args()
    print(args.halo)
    print(args.run)
    print(args.system)
    foggie_dir, output_dir, run_dir, code_path, trackname, haloname, spectra_dir, infofile = get_run_loc_etc(args)
    #foggie_dir = '/nobackupp18/mpeeples/'

    # Set directory for output location, making it if necessary
    save_dir = output_dir + 'render_halo_00' + args.halo + '/' + args.run + '/'
    if not (os.path.exists(save_dir)): os.system('mkdir -p ' + save_dir)

    print('foggie_dir: ', foggie_dir)
    halo_c_v_name = code_path + 'halo_infos/00' + args.halo + '/' + args.run + '/halo_c_v'

    outs = make_output_list(args.output, output_step=args.output_step)

    if (args.save_suffix): save_suffix = '_' + args.save_suffix
    else: save_suffix = ''

    if (not args.rotation_render) and (not args.time_render):
        sys.exit('You must specify either --rotation_render or --time_render.')

    if (args.rotation_render):
        if (args.nproc==1):
            for i in range(len(outs)):
                rendering_rotation(outs[i])
        else:
            target = rendering_rotation
    elif (args.time_render):
        if (args.nproc==1):
            for i in range(len(outs)):
                rendering_time(outs[i])
        else:
            target = rendering_time

    if (args.nproc!=1):
        # Split into a number of groupings equal to the number of processors
        # and run one process per processor
        for i in range(len(outs)//args.nproc):
            threads = []
            for j in range(args.nproc):
                snap = outs[args.nproc*i+j]
                threads.append(multi.Process(target=target, args=[snap]))
            for t in threads:
                t.start()
            for t in threads:
                t.join()
        # For any leftover snapshots, run one per processor
        threads = []
        for j in range(len(outs)%args.nproc):
            snap = outs[-(j+1)]
            threads.append(multi.Process(target=target, args=[snap]))
        for t in threads:
            t.start()
        for t in threads:
            t.join()

    print(str(datetime.datetime.now()))
    print("All snapshots finished!")
