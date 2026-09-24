"""
Filename: disk_edge_flux.py
Author: Cassi
First made: 9/24/26
Date last modified: 9/24/26

This script calculates fluxes through the disk-halo interface, using Cameron's disk
finder and binary dilation as the way to define the disk-halo interface. It then
calculates fluxes as simply the sum of mass * radial velocity / thickness of dilation region.
"""

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
from foggie.clumps.clump_finder import *

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
                        + ' and the default output is DD2427) or specify a range of outputs ' + \
                        'using commas to list individual outputs and dashes for ranges of outputs ' + \
                        '(e.g. "RD0020-RD0025" or "DD1341,DD1353,DD1600-DD1700", no spaces!)')
    parser.set_defaults(output='DD2427')

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

    parser.add_argument('--nproc', metavar='nproc', type=int, action='store', \
                        help='How many processes do you want? Default is 1 ' + \
                        '(no parallelization), if multiple outputs and multiple processors are' + \
                        ' specified, code will run one output per processor')
    parser.set_defaults(nproc=1)

    args = parser.parse_args()
    return args

def calculate_disk_halo_flux(args, snap, queue):
    '''Uses the clump finder to find the disk, dilate it, then calculate mass fluxes
    through the dilated region.'''

    # Load simulation output
    if (args.system=='pleiades_cassi'):
        print('Copying directory to /tmp')
        # Make a dummy directory with the snap name so the script later knows the process running
        # this snapshot failed if the directory is still there
        snap_dir = '/nobackup/clochhaa/tmp/' + args.halo + '/' + args.run + '/' + args.target_dir + '/' + snap
        os.makedirs(snap_dir)
        snap_name = args.foggie_dir + args.run_dir + snap + '/' + snap
    else:
        snap_name = args.foggie_dir + args.run_dir + snap + '/' + snap

    ds, refine_box = foggie_load(snap_name, trackfile_name=args.trackname, halo_c_v_name=args.halo_c_v_name, disk_relative=True)
    zsnap = ds.get_parameter('CosmologyCurrentRedshift')

    # Calculate how many cells to dilate by for a 2 kpc shell around disk
    min_dx = np.min(refine_box[('index','dx')].in_units('kpc').v)
    ncells = int(2./min_dx)

    # Set up and run disk finder
    disk_args = get_default_args()
    disk_args.output = args.output_dir + "disk_clumps/halo_00" + args.halo + '/' + args.run + '/' + snap
    if not (os.path.exists(args.output_dir + "disk_clumps/halo_00" + args.halo + '/' + args.run + '/')): os.system('mkdir -p ' + args.output_dir + "disk_clumps/halo_00" + args.halo + '/' + args.run + '/')
    disk_args.identify_disk = True #Run as a disk finder
    disk_args.n_dilation_iterations = ncells 
    disk_args.n_cells_per_dilation = 1
    disk = clump_finder(disk_args, ds, refine_box)

    # Load and combine each dilation shell into one cut region
    shell_filebase = disk_args.output + "_DiskDilationShell_n"
    for i in range(ncells):
        shell_cut = load_clump(ds, shell_filebase + str(int(i)) + ".h5")
        if i==0:
            combined_shell_cuts = shell_cut
        else:
            combined_shell_cuts = combined_shell_cuts + shell_cut

    # Calculate flux through shell
    shell_mass = combined_shell_cuts[('gas','mass')].in_units('Msun').v
    shell_rv = combined_shell_cuts[('gas','radial_velocity_corrected')].in_units('kpc/yr').v
    shell_rv_kms = combined_shell_cuts[('gas','radial_velocity_corrected')].in_units('km/s').v
    out_shell = shell_rv > 0.
    in_shell = shell_rv < 0.
    out_shell_20 = shell_rv_kms > 20.
    in_shell_20 = shell_rv_kms < -20.
    out_shell_50 = shell_rv_kms > 50.
    in_shell_50 = shell_rv_kms < -50.
    out_shell_100 = shell_rv_kms > 100.
    in_shell_100 = shell_rv_kms < -100.
    out_shell_200 = shell_rv_kms > 200.
    in_shell_200 = shell_rv_kms < -200.
    flux_out = np.sum(shell_mass[out_shell]*shell_rv[out_shell]/(ncells*min_dx))
    flux_in = np.sum(shell_mass[in_shell]*shell_rv[in_shell]/(ncells*min_dx))
    flux_net = flux_out - flux_in
    flux_out_20 = np.sum(shell_mass[out_shell_20]*shell_rv[out_shell_20]/(ncells*min_dx))
    flux_in_20 = np.sum(shell_mass[in_shell_20]*shell_rv[in_shell_20]/(ncells*min_dx))
    flux_out_50 = np.sum(shell_mass[out_shell_50]*shell_rv[out_shell_50]/(ncells*min_dx))
    flux_in_50 = np.sum(shell_mass[in_shell_50]*shell_rv[in_shell_50]/(ncells*min_dx))
    flux_out_100 = np.sum(shell_mass[out_shell_100]*shell_rv[out_shell_100]/(ncells*min_dx))
    flux_in_100 = np.sum(shell_mass[in_shell_100]*shell_rv[in_shell_100]/(ncells*min_dx))
    flux_out_200 = np.sum(shell_mass[out_shell_200]*shell_rv[out_shell_200]/(ncells*min_dx))
    flux_in_200 = np.sum(shell_mass[in_shell_200]*shell_rv[in_shell_200]/(ncells*min_dx))

    # Save fluxes
    tsnap = ds.current_time.in_units('Myr').v
    row = [snap, tsnap, zsnap, (ncells*min_dx), flux_out, flux_in, flux_net, flux_out_20, flux_in_20, flux_out_50, flux_in_50, flux_out_100, flux_in_100, flux_out_200, flux_in_200]
    queue.put(row)

    print('Snapshot', snap, 'complete!')
    ds.index.clear_all_data()
    # Delete output from temp directory if on pleiades
    if (args.system=='pleiades_cassi'):
        print('Deleting directory from /tmp')
        shutil.rmtree(snap_dir)

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
    prefix = output_dir + 'fluxes_halo_00' + args.halo + '/' + args.run + '/'
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

    if (not os.path.exists(prefix + 'disk_edge_fluxes.dat')):
        flux_table = Table(dtype=('S6', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8'),
                 names=('snap', 'time', 'redshift', 'shell_width', 'flux_out', 'flux_in', 'net_flux', 'flux_out_20', 'flux_in_20', 'flux_out_50', 'flux_in_50', 'flux_out_100', 'flux_in_100', 'flux_out_200', 'flux_in_200'))
        flux_table.write(prefix + 'disk_edge_fluxes.dat', format='ascii.ecsv', overwrite=True)

    if (save_suffix != ''):
        target_dir = 'disk_fluxes_' + save_suffix
    else:
        target_dir = 'disk_fluxes'

    args.foggie_dir = foggie_dir
    args.run_dir = run_dir
    args.output_dir = output_dir
    args.trackname = trackname
    args.halo_c_v_name = halo_c_v_name
    args.target_dir = target_dir

    skipped_outs = outs
    while (len(skipped_outs)>0):
        skipped_outs = []
        # Split into a number of groupings equal to the number of processors
        # and run one process per processor
        for i in range(len(outs)//args.nproc):
            flux_table = Table.read(prefix + 'disk_edge_fluxes' + save_suffix + '.dat', format='ascii.ecsv')
            threads = []
            snaps = []
            rows = []
            queue = multi.Queue()
            for j in range(args.nproc):
                snap = outs[args.nproc*i+j]
                snaps.append(snap)
                threads.append(multi.Process(target=calculate_disk_halo_flux, args=[args, snap, queue]))
            for t in threads:
                t.start()
            for t in threads:
                row = queue.get()
                rows.append(row)
            for t in threads:
                t.join()
            for row in rows:
                flux_table.add_row(row)
            flux_table.sort('time')
            flux_table.write(prefix + 'disk_edge_fluxes' + save_suffix + '.dat', format='ascii.ecsv', overwrite=True)
            # Delete leftover outputs from failed processes from tmp directory if on pleiades
            if (args.system=='pleiades_cassi'):
                snap_dir = '/nobackup/clochhaa/tmp/' + args.halo + '/' + args.run + '/' + target_dir + '/'
                for s in range(len(snaps)):
                    if (os.path.exists(snap_dir + snaps[s])):
                        print('Deleting failed %s from /tmp' % (snaps[s]))
                        skipped_outs.append(snaps[s])
                        shutil.rmtree(snap_dir + snaps[s])
        # For any leftover snapshots, run one per processor
        threads = []
        snaps = []
        rows = []
        queue = multi.Queue()
        flux_table = Table.read(prefix + 'disk_edge_fluxes' + save_suffix + '.dat', format='ascii.ecsv')
        for j in range(len(outs)%args.nproc):
            snap = outs[-(j+1)]
            snaps.append(snap)
            threads.append(multi.Process(target=calculate_disk_halo_flux, args=[args, snap, queue]))
        for t in threads:
            t.start()
        for t in threads:
            row = queue.get()
            rows.append(row)
        for t in threads:
            t.join()
        for row in rows:
            flux_table.add_row(row)
        flux_table.sort('time')
        flux_table.write(prefix + 'disk_edge_fluxes' + save_suffix + '.dat', format='ascii.ecsv', overwrite=True)
        # Delete leftover outputs from failed processes from tmp directory if on pleiades
        if (args.system=='pleiades_cassi'):
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