"""
File name: save_satellite_positions.py
Author: Cassi
Created: 1/8/20
Last updated: 1/8/20
This program saves to file locations of satellites at each snapshot.
"""

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
from photutils.segmentation import detect_sources
from scipy.interpolate import InterpolatedUnivariateSpline as IUS
import shutil

# These imports are FOGGIE-specific files
from foggie.utils.consistency import *
from foggie.utils.get_refine_box import get_refine_box
from foggie.utils.get_halo_center import get_halo_center
from foggie.utils.get_proper_box_size import get_proper_box_size
from foggie.utils.get_run_loc_etc import get_run_loc_etc
from foggie.utils.yt_fields import *
from foggie.utils.foggie_load import *

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
                        '(e.g. "RD0020,RD0025" or "DD1340,DD2029").')
    parser.set_defaults(output='RD0036')

    parser.add_argument('--system', metavar='system', type=str, action='store', \
                        help='Which system are you on? Default is cassiopeia')
    parser.set_defaults(system='cassiopeia')

    parser.add_argument('--pwd', dest='pwd', action='store_true',
                        help='Just use the working directory?, Default is no')
    parser.set_defaults(pwd=False)

    parser.add_argument('--local', dest='local', action='store_true',
                        help='Are the simulation files stored locally? Default is no')
    parser.set_defaults(local=False)

    parser.add_argument('--surface', metavar='surface', type=str, action='store', \
                        help='What surface type for computing the flux? Default is sphere' + \
                        ' and nothing else is implemented right now')
    parser.set_defaults(surface='sphere')

    parser.add_argument('--quadrants', dest='quadrants', type=bool, action='store', \
                         help='Do you want to compute in quadrants? Default is False,' + \
                         ' which computes for whole domain')
    parser.set_defaults(quadrants=False)

    parser.add_argument('--nproc', metavar='nproc', type=int, action='store', \
                        help='How many processes do you want? Default is 1 ' + \
                        '(no parallelization), if multiple outputs and multiple processors are' + \
                        ' specified, code will run one output per processor')
    parser.set_defaults(nproc=1)


    args = parser.parse_args()
    return args

def identify_satellites(snap, sat_file, halo_center_kpc, region, width, i_orients = np.array([(0, 'x'), (1, 'y'), (2, 'z')]), selection_props = [(0.5, 5.e5), (1.0, 1.e6)]):

    satellites = []
    sat_count = 0

    print ('loading star particle data...')
    particle_ids = region['stars', 'particle_index']
    particle_ages = region['stars', 'age'].in_units('Gyr')
    mass_stars = region['stars', 'particle_mass'].to('Msun')
    x_stars = region['stars', 'particle_position_x'].to('kpc')
    y_stars = region['stars', 'particle_position_y'].to('kpc')
    z_stars = region['stars', 'particle_position_z'].to('kpc')
    all_stars = [x_stars, y_stars, z_stars]
    ortho_orients = [[1,2], [0,2], [0,1]]
    print('loaded')

    print('finding satellites')
    for (bin_size, mass_limit) in selection_props:
        xbin = np.arange(halo_center_kpc[0] - width/2., halo_center_kpc[0] + width/2. + bin_size, bin_size)
        ybin = np.arange(halo_center_kpc[1] - width/2., halo_center_kpc[1] + width/2. + bin_size, bin_size)
        zbin = np.arange(halo_center_kpc[2] - width/2., halo_center_kpc[2] + width/2. + bin_size, bin_size)
        p = np.histogramdd((x_stars, y_stars, z_stars), \
                            weights = mass_stars, \
                            bins = (xbin, ybin, zbin))
        pp = p[0]
        pp[p[0] < mass_limit] = np.nan

        for (i, orient) in i_orients:
            i = int(i)
            sm_im = np.log10(np.nansum(pp, axis = i))
            seg_im = detect_sources(sm_im, threshold = 0, npixels = 1, connectivity = 8)

            for label in seg_im.labels:
                edges1 = p[1][ortho_orients[i][0]]
                edges2 = p[1][ortho_orients[i][1]]
                gd = np.where(seg_im.data == label)[0:10]
                all_ids = np.array([])
                for gd1, gd2 in zip(gd[0], gd[1]):
                    coord1_min, coord1_max = edges1[gd1], edges1[gd1+1]
                    coord2_min, coord2_max = edges2[gd2], edges2[gd2+1]

                    coords1 = all_stars[ortho_orients[i][0]]
                    coords2 = all_stars[ortho_orients[i][1]]
                    gd_ids = np.where((coords1 > coord1_min) & (coords1 < coord1_max) &\
                                   (coords2 > coord2_min) & (coords2 < coord2_max))[0]
                    all_ids = np.concatenate((all_ids, gd_ids), axis = None)
                mn_x = np.median(x_stars[all_ids.astype('int')])
                mn_y = np.median(y_stars[all_ids.astype('int')])
                mn_z = np.median(z_stars[all_ids.astype('int')])
                already_in_catalog = False
                for sat in satellites:
                    diff = np.sqrt((mn_x - sat[1])**2. + (mn_y - sat[2])**2. + (mn_z - sat[3])**2.)
                    if diff.value < 1.:
                        already_in_catalog = True
                        break
                if not already_in_catalog:
                    sat_count+=1
                    satellites.append([sat_count, mn_x, mn_y, mn_z])

    f = open(sat_file, 'w')
    for i in range(len(satellites)):
        f.write('%d %.3f %.3f %.3f\n' % (satellites[i][0], satellites[i][1], satellites[i][2], satellites[i][3]))
    f.close()

    return 'Satellites found for snap ' + snap + '!'

def load_and_calculate(system, foggie_dir, run_dir, track, halo_c_v_name, snap, out_dir):
    '''This function loads a specified snapshot 'snap' located in the 'run_dir' within the
    'foggie_dir', the halo track 'track', the name of the table to output, and a boolean
    'quadrants' that specifies whether or not to compute in quadrants vs. the whole domain, then
    does the calculation on the loaded snapshot.'''

    snap_name = foggie_dir + run_dir + snap + '/' + snap
    if (system=='pleiades_cassi'):
        print('Copying directory to /tmp')
        snap_dir = '/tmp/' + snap
        shutil.copytree(foggie_dir + run_dir + snap, snap_dir)
        snap_name = snap_dir + '/' + snap
    ds, refine_box, refine_box_center, refine_width = load(snap_name, track, use_halo_c_v=True, halo_c_v_name=halo_c_v_name, filter_particles=False)
    refine_width_kpc = ds.quan(refine_width, 'kpc')

    # Make the region for finding satellites
    region = ds.sphere(ds.halo_center_kpc, refine_width_kpc*5.)

    # Define the output file name
    sat_file = out_dir + snap + '_satellites.dat'

    # Do the actual calculation
    message = identify_satellites(snap, sat_file, ds.halo_center_kpc.v, region, refine_width_kpc.v*5.)
    if (system=='pleiades_cassi'):
        print('Deleting directory from /tmp')
        shutil.rmtree(snap_dir)
    print(message)
    print(str(datetime.datetime.now()))


if __name__ == "__main__":
    args = parse_args()
    print(args.halo)
    print(args.run)
    print(args.system)
    foggie_dir, output_dir, run_dir, trackname, haloname, spectra_dir, infofile = get_run_loc_etc(args)
    if (args.system=='pleiades_cassi'): code_path = '/home5/clochhaa/FOGGIE/foggie/foggie/'
    elif (args.system=='cassiopeia'):
        code_path = '/Users/clochhaas/Documents/Research/FOGGIE/Analysis_Code/foggie/foggie/'
        if (args.local):
            foggie_dir = '/Users/clochhaas/Documents/Research/FOGGIE/Simulation_Data/'
    track_dir = code_path + 'halo_infos/00' + args.halo + '/' + args.run + '/'

    # Build output list
    if (',' in args.output):
        ind = args.output.find(',')
        first = args.output[2:ind]
        last = args.output[ind+3:]
        output_type = args.output[:2]
        outs = []
        for i in range(int(first), int(last)+1):
            if (i < 10):
                pad = '000'
            elif (i >= 10) and (i < 100):
                pad = '00'
            elif (i >= 100) and (i < 1000):
                pad = '0'
            elif (i >= 1000):
                pad = ''
            outs.append(output_type + pad + str(i))
    else: outs = [args.output]

    # Set directory for output location, making it if necessary
    prefix = output_dir + 'satellites_halo_00' + args.halo + '/' + args.run + '/'
    if not (os.path.exists(prefix)): os.system('mkdir -p ' + prefix)

    print('foggie_dir: ', foggie_dir)
    halo_c_v_name = track_dir + 'halo_c_v'

    # Loop over outputs, for either single-processor or parallel processor computing
    if (args.nproc==1):
        for i in range(len(outs)):
            snap = outs[i]
            # Make the output table name for this snapshot
            tablename = prefix + snap + '_fluxes'
            # Do the actual calculation
            load_and_calculate(args.system, foggie_dir, run_dir, trackname, halo_c_v_name, snap, prefix)
    else:
        # Split into a number of groupings equal to the number of processors
        # and run one process per processor
        for i in range(len(outs)//args.nproc):
            threads = []
            for j in range(args.nproc):
                snap = outs[args.nproc*i+j]
                tablename = prefix + snap + '_fluxes'
                threads.append(multi.Process(target=load_and_calculate, \
			       args=(args.system, foggie_dir, run_dir, trackname, halo_c_v_name, snap, prefix)))
            for t in threads:
                t.start()
            for t in threads:
                t.join()
        # For any leftover snapshots, run one per processor
        threads = []
        for j in range(len(outs)%args.nproc):
            snap = outs[-(j+1)]
            tablename = prefix + snap + '_fluxes'
            threads.append(multi.Process(target=load_and_calculate, \
			   args=(args.system, foggie_dir, run_dir, trackname, halo_c_v_name, snap, prefix)))
        for t in threads:
            t.start()
        for t in threads:
            t.join()

    print(str(datetime.datetime.now()))
    print("All snapshots finished!")
