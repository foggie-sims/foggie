'''
Filename: get_mass_profile.py
Author: Cassi
Created: 10-21-19
Last modified: 10-21-19

This file calculates and outputs to file radial profiles of enclosed stellar mass, dark matter mass,
gas mass, and total mass. The files it creates are located in halo_infos.
'''

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

    parser.add_argument('--nproc', metavar='nproc', type=int, action='store', \
                        help='How many processes do you want? Default is 1 ' + \
                        '(no parallelization), if multiple outputs and multiple processors are' + \
                        ' specified, code will run one output per processor')
    parser.set_defaults(nproc=1)


    args = parser.parse_args()
    return args

def set_table_units(table):
    '''Sets the units for the table. Note this needs to be updated whenever something is added to
    the table. Returns the table.'''

    table_units = {'redshift':None,'snapshot':None,'radius':'kpc','total_mass':'Msun', \
             'dm_mass':'Msun', 'stars_mass':'Msun','gas_mass':'Msun'}
    for key in table.keys():
        table[key].unit = table_units[key]
    return table

def calc_masses(ds, snap, zsnap, refine_width_kpc, tablename):
    """Computes the mass enclosed in spheres centered on the halo center.
    Takes the dataset for the snapshot 'ds', the name of the snapshot 'snap', the redshfit of the
    snapshot 'zsnap', and the width of the refine box in kpc 'refine_width_kpc'
    and does the calculation, then writes a hdf5 table out to 'tablename'.
    """

    halo_center_kpc = ds.halo_center_kpc

    # Set up table of everything we want
    # NOTE: Make sure table units are updated when things are added to this table!
    data = Table(names=('redshift', 'snapshot', 'radius', 'total_mass', 'dm_mass', \
                        'stars_mass', 'gas_mass'), \
                 dtype=('f8', 'S6', 'f8', 'f8', 'f8', 'f8', 'f8'))

    # Define the radii of the spheres where we want to calculate mass enclosed
    radii = refine_width_kpc * np.logspace(-2,.7,250)

    # Initialize first sphere
    print('Beginning calculation for snapshot', snap)
    print('Loading field arrays')
    sphere = ds.sphere(halo_center_kpc, radii[-1])

    gas_mass = sphere['gas','cell_mass'].in_units('Msun').v
    dm_mass = sphere['dm','particle_mass'].in_units('Msun').v
    stars_mass = sphere['stars','particle_mass'].in_units('Msun').v
    gas_radius = sphere['gas','radius_corrected'].in_units('kpc').v
    dm_radius = sphere['dm','radius_corrected'].in_units('kpc').v
    stars_radius = sphere['stars','radius_corrected'].in_units('kpc').v

    # Loop over radii
    for i in range(len(radii)):

        if (i%10==0): print("Computing radius " + str(i) + "/" + str(len(radii)-1) + \
                            " for snapshot " + snap)

        # Cut the data interior to this radius
        gas_mass_enc = np.sum(gas_mass[gas_radius <= radii[i]])
        dm_mass_enc = np.sum(dm_mass[dm_radius <= radii[i]])
        stars_mass_enc = np.sum(stars_mass[stars_radius <= radii[i]])
        total_mass_enc = gas_mass_enc + dm_mass_enc + stars_mass_enc

        # Add everything to the table
        data.add_row([zsnap, snap, radii[i], total_mass_enc, dm_mass_enc, stars_mass_enc, gas_mass_enc])

    # Save to file
    data = set_table_units(data)
    data.write(tablename + '.hdf5', path='all_data', serialize_meta=True, overwrite=True)

    return "Masses have been calculated for snapshot" + snap + "!"

def load_and_calculate(system, foggie_dir, run_dir, track, halo_c_v_name, snap, tablename):
    '''This function loads a specified snapshot 'snap' located in the 'run_dir' within the
    'foggie_dir', the halo track 'track', the halo center file 'halo_c_v_name', and the name
    of the table to output 'tablename', then does the calculation on the loaded snapshot.'''

    snap_name = foggie_dir + run_dir + snap + '/' + snap
    if (system=='pleiades_cassi'):
        print('Copying directory to /tmp')
        snap_dir = '/tmp/' + snap
        shutil.copytree(foggie_dir + run_dir + snap, snap_dir)
        snap_name = snap_dir + '/' + snap
    ds, refine_box, refine_box_center, refine_width = load(snap_name, track, use_halo_c_v=True, halo_c_v_name=halo_c_v_name)
    refine_width_kpc = YTArray([refine_width], 'kpc')
    zsnap = ds.get_parameter('CosmologyCurrentRedshift')

    # Do the actual calculation
    message = calc_masses(ds, snap, zsnap, refine_width_kpc, tablename)
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
    #foggie_dir = '/nobackup/mpeeples/'
    #output_dir = '/home5/clochhaa/FOGGIE/Outputs/'
    #run_dir = 'halo_008508/nref11n_nref10f/'
    #trackname = code_path + 'halo_tracks/008508/nref11n_selfshield_15/halo_track_200kpc_nref10'
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
    prefix = output_dir + 'masses_halo_00' + args.halo + '/' + args.run + '/'
    if not (os.path.exists(prefix)): os.system('mkdir -p ' + prefix)

    print('foggie_dir: ', foggie_dir)
    halo_c_v_name = track_dir + 'halo_c_v'
    #halo_c_v_name = 'halo_c_v'

    # Loop over outputs, for either single-processor or parallel processor computing
    if (args.nproc==1):
        for i in range(len(outs)):
            snap = outs[i]
            # Make the output table name for this snapshot
            tablename = prefix + snap + '_masses'
            # Do the actual calculation
            load_and_calculate(args.system, foggie_dir, run_dir, trackname, halo_c_v_name, snap, tablename)
    else:
        # Split into a number of groupings equal to the number of processors
        # and run one process per processor
        for i in range(len(outs)//args.nproc):
            threads = []
            for j in range(args.nproc):
                snap = outs[args.nproc*i+j]
                tablename = prefix + snap + '_masses'
                threads.append(multi.Process(target=load_and_calculate, \
			       args=(args.system, foggie_dir, run_dir, trackname, halo_c_v_name, snap, tablename)))
            for t in threads:
                t.start()
            for t in threads:
                t.join()
        # For any leftover snapshots, run one per processor
        threads = []
        for j in range(len(outs)%args.nproc):
            snap = outs[-(j+1)]
            tablename = prefix + snap + '_masses'
            threads.append(multi.Process(target=load_and_calculate, \
			   args=(args.system, foggie_dir, run_dir, trackname, halo_c_v_name, snap, tablename)))
        for t in threads:
            t.start()
        for t in threads:
            t.join()

    print(str(datetime.datetime.now()))
    print("All snapshots finished!")
