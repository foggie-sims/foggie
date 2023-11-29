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

    parser.add_argument('--output_step', metavar='output_step', type=int, action='store', \
                        help='If you want to do every Nth output, this specifies N. Default: 1 (every output in specified range)')
    parser.set_defaults(output_step=1)

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

    parser.add_argument('--simple', dest='simple', action='store_true', \
                        help='Use this option to skip computing the ion gas masses.')
    parser.set_defaults(simple=False)


    args = parser.parse_args()
    return args

def set_table_units(table):
    '''Sets the units for the table. Note this needs to be updated whenever something is added to
    the table. Returns the table.'''

    table_units = {'redshift':None,'snapshot':None,'radius':'kpc','total_mass':'Msun', \
             'dm_mass':'Msun', 'stars_mass':'Msun', 'young_stars_mass':'Msun', 'old_stars_mass':'Msun', \
             'sfr':'Msun/yr', 'gas_mass':'Msun', 'gas_metal_mass':'Msun', 'gas_H_mass':'Msun', 'gas_HI_mass':'Msun', \
             'gas_HII_mass':'Msun', 'gas_CII_mass':'Msun', 'gas_CIII_mass':'Msun', 'gas_CIV_mass':'Msun', \
             'gas_OVI_mass':'Msun', 'gas_OVII_mass':'Msun', 'gas_MgII_mass':'Msun', 'gas_SiII_mass':'Msun', \
             'gas_SiIII_mass':'Msun', 'gas_SiIV_mass':'Msun', 'gas_NeVIII_mass':'Msun'}
    for key in table.keys():
        table[key].unit = table_units[key]
    return table

def calc_masses(ds, snap, zsnap, refine_width_kpc, tablename, ions=True):
    """Computes the mass enclosed in spheres centered on the halo center.
    Takes the dataset for the snapshot 'ds', the name of the snapshot 'snap', the redshfit of the
    snapshot 'zsnap', and the width of the refine box in kpc 'refine_width_kpc'
    and does the calculation, then writes a hdf5 table out to 'tablename'. If 'ions' is True then it
    computes the enclosed mass for various gas-phase ions.
    """

    halo_center_kpc = ds.halo_center_kpc

    # Set up table of everything we want
    # NOTE: Make sure table units are updated when things are added to this table!
    if (ions):
        data = Table(names=('redshift', 'snapshot', 'radius', 'total_mass', 'dm_mass', \
                            'stars_mass', 'young_stars_mass', 'old_stars_mass', 'sfr', 'gas_mass', \
                            'gas_metal_mass', 'gas_H_mass', 'gas_HI_mass', 'gas_HII_mass', 'gas_CII_mass', \
                            'gas_CIII_mass', 'gas_CIV_mass', 'gas_OVI_mass', 'gas_OVII_mass', 'gas_MgII_mass', \
                            'gas_SiII_mass', 'gas_SiIII_mass', 'gas_SiIV_mass', 'gas_NeVIII_mass'), \
                     dtype=('f8', 'S6', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', \
                            'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8'))
    else:
        data = Table(names=('redshift', 'snapshot', 'radius', 'total_mass', 'dm_mass', \
                            'stars_mass', 'young_stars_mass', 'old_stars_mass', 'sfr', 'gas_mass', \
                            'gas_metal_mass'), \
                     dtype=('f8', 'S6', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8'))

    # Define the radii of the spheres where we want to calculate mass enclosed
    radii = refine_width_kpc * np.logspace(-2,.7,250)

    # Initialize first sphere
    print('Beginning calculation for snapshot', snap)
    print('Loading field arrays')
    sphere = ds.sphere(halo_center_kpc, radii[-1])

    gas_mass = sphere['gas','cell_mass'].in_units('Msun').v
    gas_metal_mass = sphere['gas','metal_mass'].in_units('Msun').v
    if (ions):
        gas_H_mass = sphere['gas','H_mass'].in_units('Msun').v
        gas_HI_mass = sphere['gas','H_p0_mass'].in_units('Msun').v
        gas_HII_mass = sphere['gas','H_p1_mass'].in_units('Msun').v
        gas_CII_mass = sphere['gas','C_p1_mass'].in_units('Msun').v
        gas_CIII_mass = sphere['gas','C_p2_mass'].in_units('Msun').v
        gas_CIV_mass = sphere['gas','C_p3_mass'].in_units('Msun').v
        gas_OVI_mass = sphere['gas','O_p5_mass'].in_units('Msun').v
        gas_OVII_mass = sphere['gas','O_p6_mass'].in_units('Msun').v
        gas_MgII_mass = sphere['gas','Mg_p1_mass'].in_units('Msun').v
        gas_SiII_mass = sphere['gas','Si_p1_mass'].in_units('Msun').v
        gas_SiIII_mass = sphere['gas','Si_p2_mass'].in_units('Msun').v
        gas_SiIV_mass = sphere['gas','Si_p3_mass'].in_units('Msun').v
        gas_NeVIII_mass = sphere['gas','Ne_p7_mass'].in_units('Msun').v
    dm_mass = sphere['dm','particle_mass'].in_units('Msun').v
    stars_mass = sphere['stars','particle_mass'].in_units('Msun').v
    young_stars_mass = sphere['young_stars','particle_mass'].in_units('Msun').v
    old_stars_mass = sphere['old_stars','particle_mass'].in_units('Msun').v
    gas_radius = sphere['gas','radius_corrected'].in_units('kpc').v
    dm_radius = sphere['dm','radius_corrected'].in_units('kpc').v
    stars_radius = sphere['stars','radius_corrected'].in_units('kpc').v
    young_stars_radius = sphere['young_stars','radius_corrected'].in_units('kpc').v
    old_stars_radius = sphere['old_stars','radius_corrected'].in_units('kpc').v

    # Loop over radii
    for i in range(len(radii)):

        if (i%10==0): print("Computing radius " + str(i) + "/" + str(len(radii)-1) + \
                            " for snapshot " + snap)

        # Cut the data interior to this radius
        gas_mass_enc = np.sum(gas_mass[gas_radius <= radii[i]])
        gas_metal_mass_enc = np.sum(gas_metal_mass[gas_radius <= radii[i]])
        if (ions):
            gas_H_mass_enc = np.sum(gas_H_mass[gas_radius <= radii[i]])
            gas_HI_mass_enc = np.sum(gas_HI_mass[gas_radius <= radii[i]])
            gas_HII_mass_enc = np.sum(gas_HII_mass[gas_radius <= radii[i]])
            gas_CII_mass_enc = np.sum(gas_CII_mass[gas_radius <= radii[i]])
            gas_CIII_mass_enc = np.sum(gas_CIII_mass[gas_radius <= radii[i]])
            gas_CIV_mass_enc = np.sum(gas_CIV_mass[gas_radius <= radii[i]])
            gas_OVI_mass_enc = np.sum(gas_OVI_mass[gas_radius <= radii[i]])
            gas_OVII_mass_enc = np.sum(gas_OVII_mass[gas_radius <= radii[i]])
            gas_MgII_mass_enc = np.sum(gas_MgII_mass[gas_radius <= radii[i]])
            gas_SiII_mass_enc = np.sum(gas_SiII_mass[gas_radius <= radii[i]])
            gas_SiIII_mass_enc = np.sum(gas_SiIII_mass[gas_radius <= radii[i]])
            gas_SiIV_mass_enc = np.sum(gas_SiIV_mass[gas_radius <= radii[i]])
            gas_NeVIII_mass_enc = np.sum(gas_NeVIII_mass[gas_radius <= radii[i]])
        dm_mass_enc = np.sum(dm_mass[dm_radius <= radii[i]])
        stars_mass_enc = np.sum(stars_mass[stars_radius <= radii[i]])
        young_stars_mass_enc = np.sum(young_stars_mass[young_stars_radius <= radii[i]])
        old_stars_mass_enc = np.sum(old_stars_mass[old_stars_radius <= radii[i]])
        sfr_enc = young_stars_mass_enc/1.e7
        total_mass_enc = gas_mass_enc + dm_mass_enc + stars_mass_enc

        # Add everything to the table
        if (ions):
            data.add_row([zsnap, snap, radii[i], total_mass_enc, dm_mass_enc, stars_mass_enc, \
                        young_stars_mass_enc, old_stars_mass_enc, sfr_enc, gas_mass_enc, gas_metal_mass_enc, \
                        gas_H_mass_enc, gas_HI_mass_enc, gas_HII_mass_enc, gas_CII_mass_enc, \
                        gas_CIII_mass_enc, gas_CIV_mass_enc, gas_OVI_mass_enc, gas_OVII_mass_enc, \
                        gas_MgII_mass_enc, gas_SiII_mass_enc, gas_SiIII_mass_enc, gas_SiIV_mass_enc, \
                        gas_NeVIII_mass_enc])
        else:
            data.add_row([zsnap, snap, radii[i], total_mass_enc, dm_mass_enc, stars_mass_enc, \
                        young_stars_mass_enc, old_stars_mass_enc, sfr_enc, gas_mass_enc, gas_metal_mass_enc])

    # Save to file
    data = set_table_units(data)
    data.write(tablename + '.hdf5', path='all_data', serialize_meta=True, overwrite=True)

    return "Masses have been calculated for snapshot" + snap + "!"

def load_and_calculate(system, foggie_dir, run_dir, track, halo_c_v_name, snap, tablename, ions=True):
    '''This function loads a specified snapshot 'snap' located in the 'run_dir' within the
    'foggie_dir', the halo track 'track', the halo center file 'halo_c_v_name', and the name
    of the table to output 'tablename', then does the calculation on the loaded snapshot.
    If 'ions' is True then it computes the enclosed mass of various gas-phase ions.'''

    snap_name = foggie_dir + run_dir + snap + '/' + snap
    if (system=='pleiades_cassi'):
        print('Copying directory to /tmp')
        snap_dir = '/nobackup/clochhaa/tmp/' + args.halo + '/' + args.run + '/masses/' + snap
        os.makedirs(snap_dir)
        snap_name = foggie_dir + run_dir + snap + '/' + snap
    ds, refine_box = foggie_load(snap_name, track, halo_c_v_name=halo_c_v_name)
    refine_width_kpc = ds.quan(ds.refine_width, 'kpc')
    zsnap = ds.get_parameter('CosmologyCurrentRedshift')
    if (ions):
        trident.add_ion_fields(ds, ions=['O VI', 'O VII', 'Mg II', 'Si II', 'C II', 'C III', 'C IV',  'Si III', 'Si IV', 'Ne VIII'], ftype='gas')


    # Do the actual calculation
    message = calc_masses(ds, snap, zsnap, refine_width_kpc, tablename, ions=ions)
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
    foggie_dir, output_dir, run_dir, code_path, trackname, haloname, spectra_dir, infofile = get_run_loc_etc(args)

    if ('feedback' in args.run) and ('track' in args.run):
        foggie_dir = '/nobackup/jtumlins/halo_008508/feedback-track/'
        run_dir = args.run + '/'

    # Set directory for output location, making it if necessary
    prefix = output_dir + 'masses_halo_00' + args.halo + '/' + args.run + '/'
    if not (os.path.exists(prefix)): os.system('mkdir -p ' + prefix)

    print('foggie_dir: ', foggie_dir)
    halo_c_v_name = code_path + 'halo_infos/00' + args.halo + '/' + args.run + '/halo_c_v'

    # Build output list
    outs = make_output_list(args.output, output_step=args.output_step)

    if (args.simple): ions = False
    else: ions = True

    # Loop over outputs, for either single-processor or parallel processor computing
    # Split into a number of groupings equal to the number of processors
    # and run one process per processor
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
                tablename = prefix + snap + '_masses'
                snaps.append(snap)
                threads.append(multi.Process(target=load_and_calculate, \
		          args=(args.system, foggie_dir, run_dir, trackname, halo_c_v_name, snap, tablename, ions)))
            for t in threads:
                t.start()
            for t in threads:
                t.join()
            # Delete leftover outputs from failed processes from tmp directory if on pleiades
            if (args.system=='pleiades_cassi'):
                snap_dir = '/nobackup/clochhaa/tmp/' + args.halo + '/' + args.run + '/masses/'
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
            tablename = prefix + snap + '_masses'
            snaps.append(snap)
            threads.append(multi.Process(target=load_and_calculate, \
		      args=(args.system, foggie_dir, run_dir, trackname, halo_c_v_name, snap, tablename, ions)))
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        # Delete leftover outputs from failed processes from tmp directory if on pleiades
        if (args.system=='pleiades_cassi'):
            snap_dir = '/nobackup/clochhaa/tmp/' + args.halo + '/' + args.run + '/masses/'
            for s in range(len(snaps)):
                if (os.path.exists(snap_dir + snaps[s])):
                    print('Deleting failed %s from /tmp' % (snaps[s]))
                    skipped_outs.append(snaps[s])
                    shutil.rmtree(snap_dir + snaps[s])
        outs = skipped_outs

    print(str(datetime.datetime.now()))
    print("All snapshots finished!")
