# Filename: save_halo_v_corrected.py
# NOTE: THIS FILE DOES NOT NEED TO BE USED ANYMORE BECAUSE GET_HALO_C_V_PARALLEL.PY HAS BEEN UPDATED
# TO DO THE SAME THING AS THIS FILE.

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
import shutil
import ast
import trident

# These imports are FOGGIE-specific files
from foggie.utils.consistency import *
from foggie.utils.get_refine_box import get_refine_box
from foggie.utils.get_halo_center import get_halo_center
from foggie.utils.get_proper_box_size import get_proper_box_size
from foggie.utils.get_run_loc_etc import get_run_loc_etc
from foggie.utils.yt_fields import *
from foggie.utils.foggie_load import *
from foggie.utils.analysis_utils import *

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
    parser.set_defaults(output='RD0036')

    parser.add_argument('--system', metavar='system', type=str, action='store', \
                        help='Which system are you on? Default is cassiopeia')
    parser.set_defaults(system='cassiopeia')

    parser.add_argument('--pwd', dest='pwd', action='store_true',
                        help='Just use the working directory?, Default is no')
    parser.set_defaults(pwd=False)

    parser.add_argument('--nproc', metavar='nproc', type=int, action='store', \
                        help='How many processes do you want? Default is 1 ' + \
                        '(no parallelization), if multiple outputs and multiple processors are' + \
                        ' specified, code will run one output per processor')
    parser.set_defaults(nproc=1)

    args = parser.parse_args()
    return args

def find_bulk_velocity(snap_name, t):
    ds, refine_box = foggie_load(snap_name, trackfile_name=trackname, do_filter_particles=False, halo_c_v_name=halo_c_v_name, correct_bulk_velocity=True)
    print('foggie_load found ds.halo_velocity_kms to be', ds.halo_velocity_kms)
    row = [snap_name[-6:], ds.get_parameter('CosmologyCurrentRedshift'), ds.current_time.in_units('Myr').v, ds.halo_velocity_kms[0], ds.halo_velocity_kms[1], ds.halo_velocity_kms[2]]
    t.put(row)

if __name__ == "__main__":
    args = parse_args()
    print(args.halo)
    print(args.run)
    print(args.system)
    foggie_dir, output_dir, run_dir, code_path, trackname, haloname, spectra_dir, infofile = get_run_loc_etc(args)

    if ('feedback' in args.run) and ('track' in args.run):
        foggie_dir = '/nobackup/jtumlins/halo_008508/feedback-track/'
        run_dir = args.run + '/'
        
    print(foggie_dir, output_dir, run_dir)

    outs = make_output_list(args.output)

    prefix = output_dir + 'halo_centers/' + 'halo_00' + args.halo + '/' + args.run + '/'
    halo_c_v_name = code_path + 'halo_infos/00' + args.halo + '/' + args.run + '/halo_c_v'
    if not (os.path.exists(prefix)): os.system('mkdir -p ' + prefix)
    v_table = Table(names=('snap','redshift','time','vx','vy','vz'), dtype=('S6','f8','f8','f8','f8','f8'))
    # Split into a number of groupings equal to the number of processors
    # and run one process per processor
    rows = []
    for i in range(len(outs)//args.nproc):
        threads = []
        queue = multi.Queue()
        for j in range(args.nproc):
            snap = foggie_dir + run_dir + outs[args.nproc*i+j] + '/' + outs[args.nproc*i+j]
            thr = multi.Process(target=find_bulk_velocity, args=(snap, queue))
            threads.append(thr)
            thr.start()
        for thr in threads:
            row = queue.get()
            rows.append(row)
        for thr in threads:
            thr.join()
    # For any leftover snapshots, run one per processor
    threads = []
    queue = multi.Queue()
    for j in range(len(outs)%args.nproc):
        snap = foggie_dir + run_dir + outs[-(j+1)] + '/' + outs[-(j+1)]
        thr = multi.Process(target=find_bulk_velocity, args=(snap, queue))
        threads.append(thr)
        thr.start()
    for thr in threads:
        row = queue.get()
        rows.append(row)
    for thr in threads:
        thr.join()

    for row in rows:
        v_table.add_row(row)

    v_table.sort('time')
    v_table['snap'].unit = 'dimensionless'
    v_table['redshift'].unit = 'dimensionless'
    v_table['time'].unit = 'Myr'
    v_table['vx'].unit = 'km/s'
    v_table['vy'].unit = 'km/s'
    v_table['vz'].unit = 'km/s'
    ascii.write(v_table, prefix + 'bulk-v_table.dat', format='fixed_width', overwrite=True)
