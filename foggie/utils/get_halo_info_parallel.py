"""
Filename: get_halo_info_parallel.py
Author: Cassi (with copy-pasting from Molly's get_halo_info.py)
Made: 10/2/19
Last modified: 10/2/19
This program produces an ascii file of the location and velocity of the center of the halo,
the virial mass and virial radius of the halo, the stellar and ISM masses of the galaxy, and the
star formation rate, for all snapshots input by the user. Its purpose is to eliminate having to
re-find information about the halo every time new analysis code is run.

This code is very similar to Cassi's get_halo_c_v_parallel.py, except it finds more information
about the halo than just its center and velocity.

NOTE: I haven't done a test-run of this code because it was taking too long and I'm impatient.
However, given its similarity to get_halo_c_v_parallel.py, which IS tested and DOES work, this one
PROBABLY works too.
"""

import yt
from yt.units import *
from yt import YTArray
from astropy.table import Table
from astropy.io import ascii
from yt.analysis_modules.star_analysis.api import StarFormationRate
import multiprocessing as mp
import argparse

from foggie.utils.get_refine_box import get_refine_box
from foggie.utils.get_halo_center import get_halo_center
from foggie.utils.get_proper_box_size import get_proper_box_size
from foggie.utils.get_run_loc_etc import get_run_loc_etc
import numpy as np
import glob
import os
from enzoGalaxyProps import find_rvirial

import warnings

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
                        + ' and the default output is RD0036), specify a range of outputs ' + \
                        '(e.g. "RD0020,RD0025" or "DD1340,DD2029"), or "all".')
    parser.set_defaults(output='RD0036')

    parser.add_argument('--system', metavar='system', type=str, action='store', \
                        help='Which system are you on? Default is cassiopeia')
    parser.set_defaults(system='cassiopeia')

    parser.add_argument('--pwd', dest='pwd', action='store_true',
                        help='Just use the working directory?, Default is no')
    parser.set_defaults(pwd=False)

    parser.add_argument('--nproc', metavar='nproc', type=int, action='store', \
                        help='How many processes do you want? Default is 4, ' + \
                        'code will run one output per processor')
    parser.set_defaults(nproc=4)


    args = parser.parse_args()
    return args

def _stars(pfilter, data):
    return data[(pfilter.filtered_type, "particle_type")] == 2

#this gets dark matter particles in zoom region only
def _darkmatter(pfilter, data):
    return data[(pfilter.filtered_type, "particle_type")] == 4

yt.add_particle_filter("stars",function=_stars, filtered_type='all',requires=["particle_type"])
yt.add_particle_filter("darkmatter",function=_darkmatter, filtered_type='all',requires=["particle_type"])

def loop_over_halos(nproc, run_dir, trackname, output_dir, outs):
    '''
    This sets up the parallel processing for finding the halo centers of all datasets in 'outs'.
    It also takes the number of processors to use, 'nproc', the directory where the snapshots
    can be found, 'run_dir', the file name of the halo track file, 'trackname', and the directory where
    the new halo_v_c file should be placed, 'output_dir'.
    '''
    print("opening track: " + trackname)
    track = Table.read(trackname, format='ascii')
    track.sort('col1')

    t = Table(dtype=('f8','S6', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8',
                    'f8', 'f8', 'f8','f8', 'f8'),
            names=('redshift', 'name', 'xc', 'yc', 'zc', 'xv', 'yv', 'zv',
                    'Rvir', 'Mvir', 'Mstar', 'Mism', 'SFR'))

    print('Computing centers and velocities for ' + str(len(outs)) + ' snaps ' + \
          'from ' + outs[0] + ' to ' + outs[-1])
    # Split into a number of groupings equal to the number of processors
    # and run one process per processor
    rows = []
    for i in range(len(outs)//nproc):
        threads = []
        queue = mp.Queue()
        for j in range(nproc):
            snap = run_dir + outs[nproc*i+j] + '/' + outs[nproc*i+j]
            thr = mp.Process(target=get_halo_info, \
               args=(snap, track, queue))
            threads.append(thr)
            thr.start()
        for thr in threads:
            row = queue.get()
            rows.append(row)
        for thr in threads:
            thr.join()
    # For any leftover snapshots, run one per processor
    threads = []
    queue = mp.Queue()
    for j in range(len(outs)%nproc):
        snap = run_dir + outs[-(j+1)] + '/' + outs[-(j+1)]
        thr = mp.Process(target=get_halo_info, \
           args=(snap, track, queue))
        threads.append(thr)
        thr.start()
    for thr in threads:
        row = queue.get()
        rows.append(row)
    for thr in threads:
        thr.join()

    for row in rows:
        t.add_row(row)

    t.sort('redshift')
    t.reverse()
    ascii.write(t, output_dir + 'halo_info' + outs[0] + '_' + outs[-1], format='fixed_width', overwrite=True)

def get_halo_info(snap, track, t):
    '''
    Given a snapshot name 'snap' and the trackfile 'track', finds the halo center and velocity, R200, M200, and galaxy masses and SFRs and stores it in the multiprocessing queue 't'.
    '''
    print('Loading ' + snap[-6:])
    ds = yt.load(snap)
    ds.add_particle_filter('stars')
    ds.add_particle_filter('darkmatter')

    zsnap = ds.get_parameter('CosmologyCurrentRedshift')
    comoving_box_size = ds.get_parameter('CosmologyComovingBoxSize')
    proper_box_size = get_proper_box_size(ds)
    print('Comoving Box Size:', comoving_box_size)
    refine_box, refine_box_center, refine_width = get_refine_box(ds, zsnap, track)
    center, velocity = get_halo_center(ds, refine_box_center)
    halo_center_kpc = YTArray(np.array(center)*proper_box_size, 'kpc')
    halo_velocity_kms = YTArray(velocity).in_units('km/s')

    ## halo information
    rvir = find_rvirial(refine_box, ds, center)
    # Mvir is mass within rvir
    vir_sphere = ds.sphere(center,rvir)
    vir_dm_mass = vir_sphere[('darkmatter', 'particle_mass')].in_units('Msun')
    Mvir = vir_dm_mass.sum()

    ## define where the central galaxy is
    fgal = 0.07 ## fraction of the virial radius we are saying is the galaxy radius; totes made up
    print('rvir  = ', rvir, ' and rgal = ',fgal*rvir)
    gal_sphere = ds.sphere(center, fgal*rvir)
    # Mstar is sum of stellar mass; Mism is sum of gas mass; SFR is sum of SFR
    gal_stars_mass = gal_sphere[('stars', 'particle_mass')].in_units('Msun')
    Mstar = gal_stars_mass.sum()

    gal_ism_mass = gal_sphere['cell_mass'].in_units('Msun')
    Mism = gal_ism_mass.sum()

    sfr = StarFormationRate(ds, data_source=gal_sphere)

    row = [zsnap, ds.parameter_filename[-6:],
            halo_center_kpc[0], halo_center_kpc[1], halo_center_kpc[2],
            halo_velocity_kms[0], halo_velocity_kms[1], halo_velocity_kms[2],
            rvir, Mvir, Mstar, Mism, sfr.Msol_yr[-1]]
    print(snap[-6:] + ' done')
    t.put(row)


if __name__ == "__main__":

    args = parse_args()

    warnings.filterwarnings('ignore', category=FutureWarning)
    warnings.filterwarnings('ignore', category=DeprecationWarning)

    foggie_dir, output_dir, run_dir, trackname, haloname, spectra_dir = get_run_loc_etc(args)
    output_dir = output_dir + 'halo_centers/' + 'halo_00' + args.halo + '/' + args.run + '/'
    if not (os.path.exists(output_dir)): os.system('mkdir -p ' + output_dir)
    if ('/astro/simulations/' in foggie_dir):
        run_dir = 'halo_00' + args.halo + '/nref11n/' + args.run + '/'
    run_dir = foggie_dir + run_dir

    # Build output list
    if (args.output=='all'):
        outs = glob.glob(os.path.join('?D????'))
        outs_new = []
        for i in range(len(outs)):
            outs_new.append(outs[i][-6:])
        outs = outs_new
    elif (',' in args.output):
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

    loop_over_halos(args.nproc, run_dir, trackname, output_dir, outs)

    warnings.filterwarnings('default', category=FutureWarning)
    warnings.filterwarnings('default', category=DeprecationWarning)
