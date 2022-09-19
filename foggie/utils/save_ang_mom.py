# Filename: save_ang_mom.py

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

def find_angmom(snap_name, t):
    ds, refine_box = foggie_load(snap_name, trackname, do_filter_particles=True, halo_c_v_name=halo_c_v_name, disk_relative=True, correct_bulk_velocity=True, particle_type_for_angmom='young_stars8')
    row = [snap_name[-6:], ds.get_parameter('CosmologyCurrentRedshift'), ds.current_time.in_units('Myr').v, ds.z_unit_disk[0], ds.z_unit_disk[1], ds.z_unit_disk[2]]
    t.put(row)

if __name__ == "__main__":
    args = parse_args()
    print(args.halo)
    print(args.run)
    print(args.system)
    foggie_dir, output_dir, run_dir, code_path, trackname, haloname, spectra_dir, infofile = get_run_loc_etc(args)
    print(foggie_dir, output_dir, run_dir)

    outs = make_output_list(args.output)

    prefix = output_dir + 'halo_centers/' + 'halo_00' + args.halo + '/' + args.run + '/'
    halo_c_v_name = code_path + 'halo_infos/00' + args.halo + '/' + args.run + '/halo_c_v'
    if not (os.path.exists(prefix)): os.system('mkdir -p ' + prefix)
    angmom_table = Table(names=('snap','redshift','time','Lx','Ly','Lz'), dtype=('S6','f8','f8','f8','f8','f8'))
    # Split into a number of groupings equal to the number of processors
    # and run one process per processor
    rows = []
    for i in range(len(outs)//args.nproc):
        threads = []
        queue = multi.Queue()
        for j in range(args.nproc):
            snap = foggie_dir + run_dir + outs[args.nproc*i+j] + '/' + outs[args.nproc*i+j]
            thr = multi.Process(target=find_angmom, args=(snap, queue))
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
        thr = multi.Process(target=find_angmom, args=(snap, queue))
        threads.append(thr)
        thr.start()
    for thr in threads:
        row = queue.get()
        rows.append(row)
    for thr in threads:
        thr.join()

    for row in rows:
        angmom_table.add_row(row)

    angmom_table.sort('time')
    angmom_table['snap'].unit = 'dimensionless'
    angmom_table['redshift'].unit = 'dimensionless'
    angmom_table['time'].unit = 'Myr'
    angmom_table['Lx'].unit = 'dimensionless'
    angmom_table['Ly'].unit = 'dimensionless'
    angmom_table['Lz'].unit = 'dimensionless'
    angmom_table.write(prefix + 'angmom_table.hdf5', path='all_data', serialize_meta=True, overwrite=True)
