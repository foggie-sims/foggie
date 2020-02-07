'''
Filename: make_smaller_ds.py
Author: Cassi
This file loads in the fields needed from a region of a dataset and saves them as arrays
to a separate file so that the full dataset does not need to be loaded in each time any of these
arrays are needed.

Dependencies:
utils/get_refine_box.py
utils/get_halo_center.py
utils/get_proper_box_size.py
utils/get_run_loc_etc.py
utils/yt_fields.py
utils/foggie_load.py
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

    parser.add_argument('--local', dest='local', action='store_true',
                        help='If using system cassiopeia: Are the simulation files stored locally? Default is no')
    parser.set_defaults(local=False)

    parser.add_argument('--nproc', metavar='nproc', type=int, action='store', \
                        help='How many processes do you want? Default is 1 ' + \
                        '(no parallelization), if multiple outputs and multiple processors are' + \
                        ' specified, code will run one output per processor')
    parser.set_defaults(nproc=1)

    parser.add_argument('--add_fields', dest='new_fields', type=str, action='store', \
                        help='What new fields do you want to add? Specify them as a list of strings, ' + \
                        'like so:\n"[\'cell_mass\', (\'gas\',\'angular_momentum_magnitude\')]" (don\'t forget outer quotes!)\n' + \
                        'This is only necessary (and doable) once a file already exists.')
    parser.set_defaults(new_fields="['none']")

    parser.add_argument('--add_units', dest='new_units', type=str, action='store', \
                        help='What units do you want for your new fields? Specify them as a list of strings, ' + \
                        'like so:\n"[\'Msun\', \'g*cm/s\']" (don\'t forget outer quotes!)')
    parser.set_defaults(new_units="['none']")

    args = parser.parse_args()
    return args

def add_to_files(system, foggie_dir, run_dir, track, halo_c_v_name, snap, tablename, new_fields, new_units):
    snap_name = foggie_dir + run_dir + snap + '/' + snap
    if (system=='pleiades_cassi'):
        print('Copying directory to /tmp')
        snap_dir = '/tmp/' + snap
        shutil.copytree(foggie_dir + run_dir + snap, snap_dir)
        snap_name = snap_dir + '/' + snap
    ds, refine_box, refine_box_center, refine_width = load(snap_name, track, use_halo_c_v=True, \
      halo_c_v_name=halo_c_v_name, filter_particles=False)
    refine_width_kpc = ds.quan(refine_width, 'kpc')
    sphere = ds.sphere(ds.halo_center_kpc, 5.*refine_width_kpc)
    trident.add_ion_fields(ds, ions='all', ftype='gas')

    field_table = Table.read(tablename + '.hdf5', path='all_data')
    for i in range(len(new_fields)):
        if (type(new_fields[i])==tuple):
            field_name = new_fields[i][1]
        else:
            field_name = new_fields[i]
        field_table.add_column(sphere[new_fields[i]].in_units(new_units[i]).v, name=field_name)
        field_table[field_name].unit = new_units[i]

    field_table.write(tablename + '.hdf5', path='all_data', serialize_meta=True, overwrite=True)

    if (system=='pleiades_cassi'):
        print('Deleting directory from /tmp')
        shutil.rmtree(snap_dir)

def save_to_files(system, foggie_dir, run_dir, track, halo_c_v_name, snap, tablename, fields, units):
    snap_name = foggie_dir + run_dir + snap + '/' + snap
    if (system=='pleiades_cassi'):
        print('Copying directory to /tmp')
        snap_dir = '/tmp/' + snap
        shutil.copytree(foggie_dir + run_dir + snap, snap_dir)
        snap_name = snap_dir + '/' + snap
    ds, refine_box, refine_box_center, refine_width = load(snap_name, track, use_halo_c_v=True, \
      halo_c_v_name=halo_c_v_name, filter_particles=False)
    refine_width_kpc = ds.quan(refine_width, 'kpc')
    zsnap = ds.get_parameter('CosmologyCurrentRedshift')
    sphere = ds.sphere(ds.halo_center_kpc, 2.*refine_width_kpc)
    trident.add_ion_fields(ds, ions='all', ftype='gas')

    print('Loading fields')
    field_table = Table()
    for i in range(len(fields)):
        if (type(fields[i])==tuple):
            field_name = fields[i][1]
        else:
            field_name = fields[i]
        if (field_name=='x'):
            field_table.add_column(sphere['gas','x'].in_units(units[i]).v - ds.halo_center_kpc[0].in_units(units[i]).v, \
              name=field_name)
        elif (field_name=='y'):
            field_table.add_column(sphere['gas','y'].in_units(units[i]).v - ds.halo_center_kpc[1].in_units(units[i]).v, \
              name=field_name)
        elif (field_name=='z'):
            field_table.add_column(sphere['gas','z'].in_units(units[i]).v - ds.halo_center_kpc[2].in_units(units[i]).v, \
              name=field_name)
        elif (field_name=='Grav_Potential'):
            field_table.add_column((sphere['gas','cell_mass'] * \
              ds.arr(sphere['enzo','Grav_Potential'].v, 'code_length**2/code_time**2')).in_units(units[i]).v, name=field_name)
        elif (field_name=='thermal_energy'):
            field_table.add_column((sphere['gas','cell_mass']*sphere['gas','thermal_energy']).in_units(units[i]).v, name=field_name)
        else:
            field_table.add_column(sphere[fields[i]].in_units(units[i]).v, name=field_name)
            field_table[field_name].unit = units[i]
        print('Field %d/%d loaded' % (i+1, len(fields)))
    field_table.add_column(zsnap + np.zeros(len(sphere['gas','x'])), name='redshift', index=0)
    field_table['redshift'].unit = 'dimensionless'
    field_table.write(tablename + '.hdf5', path='all_data', serialize_meta=True, overwrite=True)

    if (system=='pleiades_cassi'):
        print('Deleting directory from /tmp')
        shutil.rmtree(snap_dir)

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

    # Define list of fields to start with
    fields = [('gas','x'), ('gas','y'), ('gas','z'), \
      ('gas','vx_corrected'), ('gas','vy_corrected'), ('gas','vz_corrected'), \
      ('gas','radius_corrected'), ('gas','radial_velocity_corrected'), \
      ('gas','temperature'), ('gas','cell_mass'), ('gas','metal_mass'), \
      ('gas','thermal_energy'), ('gas','kinetic_energy_corrected'), ('Enzo','Grav_Potential'), \
      ('gas','cooling_time'), ('gas','entropy'), \
      'O_p0_ion_fraction', 'O_p0_mass', 'O_p1_ion_fraction', 'O_p1_mass', 'O_p2_ion_fraction', 'O_p2_mass', \
      'O_p3_ion_fraction', 'O_p3_mass', 'O_p4_ion_fraction', 'O_p4_mass', 'O_p5_ion_fraction', 'O_p5_mass', \
      'O_p6_ion_fraction', 'O_p6_mass', 'O_p7_ion_fraction', 'O_p7_mass', 'O_p8_ion_fraction', 'O_p8_mass']
    units = ['kpc', 'kpc', 'kpc', \
      'km/s', 'km/s', 'km/s', \
      'kpc', 'km/s', \
      'K', 'Msun', 'Msun', \
      'erg', 'erg', 'erg', \
      'yr', 'keV*cm**2', \
      'dimensionless', 'Msun', 'dimensionless', 'Msun', 'dimensionless', 'Msun', \
      'dimensionless', 'Msun', 'dimensionless', 'Msun', 'dimensionless', 'Msun', \
      'dimensionless', 'Msun', 'dimensionless', 'Msun', 'dimensionless', 'Msun']

    # Build new fields list
    new_fields = ast.literal_eval(args.new_fields)
    new_units = ast.literal_eval(args.new_units)
    new_fields_fixed = []
    new_units_fixed = []
    if (not 'none' in new_fields):
        for i in range(len(new_fields)):
            if (new_fields[i] not in fields):
                new_fields_fixed.append(new_fields[i])
                new_units_fixed.append(new_units[i])
    new_fields = new_fields_fixed
    new_units = new_units_fixed

    # Build output list
    if (',' in args.output):
        outs = args.output.split(',')
        for i in range(len(outs)):
            if ('-' in outs[i]):
                ind = outs[i].find('-')
                first = outs[i][2:ind]
                last = outs[i][ind+3:]
                output_type = outs[i][:2]
                outs_sub = []
                for j in range(int(first), int(last)+1):
                    if (j < 10):
                        pad = '000'
                    elif (j >= 10) and (j < 100):
                        pad = '00'
                    elif (j >= 100) and (j < 1000):
                        pad = '0'
                    elif (j >= 1000):
                        pad = ''
                    outs_sub.append(output_type + pad + str(j))
                outs[i] = outs_sub
        flat_outs = []
        for i in outs:
            if (type(i)==list):
                for j in i:
                    flat_outs.append(j)
            else:
                flat_outs.append(i)
        outs = flat_outs
    elif ('-' in args.output):
        ind = args.output.find('-')
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
    prefix = output_dir + 'fields_halo_00' + args.halo + '/' + args.run + '/'
    if not (os.path.exists(prefix)): os.system('mkdir -p ' + prefix)

    print('foggie_dir: ', foggie_dir)
    halo_c_v_name = track_dir + 'halo_c_v'

    # Loop over outputs, for either single-processor or parallel processor computing
    if (args.nproc==1):
        for i in range(len(outs)):
            snap = outs[i]
            # Make the output table name for this snapshot
            tablename = prefix + snap + '_fields'
            # Do the actual calculation
            if (len(new_fields)!=0):
                add_to_files(args.system, foggie_dir, run_dir, trackname, halo_c_v_name, snap, \
                  tablename, new_fields, new_units)
            else:
                save_to_files(args.system, foggie_dir, run_dir, trackname, halo_c_v_name, snap, \
                  tablename, fields, units)
    else:
        # Split into a number of groupings equal to the number of processors
        # and run one process per processor
        for i in range(len(outs)//args.nproc):
            threads = []
            for j in range(args.nproc):
                snap = outs[args.nproc*i+j]
                tablename = prefix + snap + '_fields'
                if (len(new_fields)!=0):
                    threads.append(multi.Process(target=add_to_files, \
    			       args=(args.system, foggie_dir, run_dir, trackname, halo_c_v_name, snap, \
                         tablename, new_fields, new_units)))
                else:
                    threads.append(multi.Process(target=save_to_files, \
    			       args=(args.system, foggie_dir, run_dir, trackname, halo_c_v_name, snap, \
                         tablename, fields, units)))
            for t in threads:
                t.start()
            for t in threads:
                t.join()
        # For any leftover snapshots, run one per processor
        threads = []
        for j in range(len(outs)%args.nproc):
            snap = outs[-(j+1)]
            tablename = prefix + snap + '_fields'
            if (len(new_fields)!=0):
                threads.append(multi.Process(target=add_to_files, \
                   args=(args.system, foggie_dir, run_dir, trackname, halo_c_v_name, snap, \
                     tablename, new_fields, new_units)))
            else:
                threads.append(multi.Process(target=save_to_files, \
                   args=(args.system, foggie_dir, run_dir, trackname, halo_c_v_name, snap, \
                     tablename, fields, units)))
        for t in threads:
            t.start()
        for t in threads:
            t.join()

    print(str(datetime.datetime.now()))
    print("All snapshots finished!")
