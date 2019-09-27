"""
Filename: flux_tracking.py
Author: Cassi
Date created: 9-27-19
Date last modified: 9-27-19
This file takes command line arguments and computes fluxes of things through
spherical shells.

Dependencies:
utils/consistency.py
utils/get_refine_box.py
utils/get_halo_center.py
utils/get_proper_box_size.py
utils/get_run_loc_etc.py
utils/yt_fields.py
utils/yt_fields.py
"""

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
from multiprocessing import Pool
import datetime

# These imports are FOGGIE-specific files
from foggie.utils.consistency import *
from foggie.utils.get_refine_box import get_refine_box
from foggie.utils.get_halo_center import get_halo_center
from foggie.utils.get_proper_box_size import get_proper_box_size
from foggie.utils.get_run_loc_etc import get_run_loc_etc
from foggie.utils.yt_fields import *

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
                        help='Which output? Default is RD0036')
    parser.set_defaults(output='RD0036')

    parser.add_argument('--system', metavar='system', type=str, action='store', \
                        help='Which system are you on? Default is cassiopeia')
    parser.set_defaults(system='cassiopeia')

    parser.add_argument('--pwd', dest='pwd', action='store_true',
                        help='Just use the working directory?, Default is no')
    parser.set_defaults(pwd=False)

    parser.add_argument('--surface', metavar='surface', type=str, action='store', \
                        help='What surface type for computing the flux? Default is sphere' + \
                        ' and nothing else is implemented right now')
    parser.set_defaults(surface='sphere')


    args = parser.parse_args()
    return args

def set_table_units(table):
    '''Sets the units for the table. Note this needs to be updated whenever something is added to
    the table. Returns the table.'''

    table_units = {'redshift':None,'quadrant':None,'radius':'kpc','net_mass_flux':'Msun/yr', \
             'net_metal_flux':'Msun/yr', 'mass_flux_in'  :'Msun/yr','mass_flux_out':'Msun/yr', \
             'metal_flux_in' :'Msun/yr', 'metal_flux_out':'Msun/yr',\
             'net_cold_mass_flux':'Msun/yr', 'cold_mass_flux_in':'Msun/yr', 'cold_mass_flux_out':'Msun/yr', \
             'net_cool_mass_flux':'Msun/yr', 'cool_mass_flux_in':'Msun/yr', 'cool_mass_flux_out':'Msun/yr', \
             'net_warm_mass_flux':'Msun/yr', 'warm_mass_flux_in':'Msun/yr', 'warm_mass_flux_out':'Msun/yr', \
             'net_hot_mass_flux' :'Msun/yr', 'hot_mass_flux_in' :'Msun/yr', 'hot_mass_flux_out' :'Msun/yr'}
    for key in table.keys():
        table[key].unit = table_units[key]
    return table

def calc_fluxes(ds, snap, halo_center, refine_width_kpc, tablename, **kwargs):
    """Computes the flux through spherical shells centered on the halo center.
    Takes the dataset for the snapshot 'ds', the name of the snapshot 'snap',
    the halo center 'halo_center', and the width of the
    refine box in kpc 'refine_width_kpc' and does the calculation,
    then writes a hdf5 table out to 'tablename'.

    Optional arguments:
    quadrants = True will calculate the flux shells within quadrants rather than the whole domain,
        default is False. If this is selected, a second table will be written with '_q' appended
        to 'tablename'.
    """

    outs = kwargs.get('quadrants', False)

    # Set up table of everything we want
    # NOTE: Make sure table units are updated when things are added to this table!
    data = Table(names=('redshift', 'quadrant', 'radius', 'net_mass_flux', 'net_metal_flux', \
                        'mass_flux_in', 'mass_flux_out', 'metal_flux_in', 'metal_flux_out', \
                        'net_cold_mass_flux', 'cold_mass_flux_in', 'cold_mass_flux_out', \
                        'net_cool_mass_flux', 'cool_mass_flux_in', 'cool_mass_flux_out', \
                        'net_warm_mass_flux', 'warm_mass_flux_in', 'warm_mass_flux_out', \
                        'net_hot_mass_flux', 'hot_mass_flux_in', 'hot_mass_flux_out'), \
                 dtype=('f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', \
                        'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8'))

    if (quadrants):
        data_q = Table(names=('redshift', 'quadrant', 'radius', 'net_mass_flux', 'net_metal_flux', \
                            'mass_flux_in', 'mass_flux_out', 'metal_flux_in', 'metal_flux_out', \
                            'net_cold_mass_flux', 'cold_mass_flux_in', 'cold_mass_flux_out', \
                            'net_cool_mass_flux', 'cool_mass_flux_in', 'cool_mass_flux_out', \
                            'net_warm_mass_flux', 'warm_mass_flux_in', 'warm_mass_flux_out', \
                            'net_hot_mass_flux', 'hot_mass_flux_in', 'hot_mass_flux_out'), \
                     dtype=('f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', \
                            'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8'))

    # Define the radii of the spherical shells where we want to calculate fluxes
    radii = 0.5*refine_width_kpc * np.arange(0.1, 0.9, 0.01)

    # Loop over radii
    for i in range(len(radii)-1):
        r_low = radii[i]
        r_high = radii[i+1]
        dr = r_high - r_low
        r = (r_low + r_high)/2.

        if (i%10==0): print("Computing radius " + str(i) + "/" + str(len(radii)-1))

        # Make the spheres and shell for computing
        inner_sphere = ds.sphere(halo_center, r_low)
        outer_sphere = ds.sphere(halo_center, r_high)
        shell = outer_sphere - inner_sphere

        # Cut the shell on radial velocity for in and out fluxes
        shell_in = shell.cut_region("obj['radial_velocity'] < 0")
        shell_out = shell.cut_region("obj['radial_velocity'] > 0")

        # Cut the shell on temperature for cold, cool, warm, and hot gas
        shell_cold = shell.cut_region("obj['temperature'] <= 10**4")
        shell_cool = shell.cut_region("(obj['temperature'] > 10**4) &" + \
                                      " (obj['temperature'] <= 10**5)")
        shell_warm = shell.cut_region("(obj['temperature'] > 10**5) &" + \
                                      " (obj['temperature'] <= 10**6)")
        shell_hot = shell.cut_region("obj['temperature'] > 10**6")

        # Cut the shell on both temperature and radial velocity
        shell_in_cold = shell_in.cut_region("obj['temperature'] <= 10**4")
        shell_in_cool = shell_in.cut_region("(obj['temperature'] > 10**4) &" + \
                                            " (obj['temperature'] <= 10**5)")
        shell_in_warm = shell_in.cut_region("(obj['temperature'] > 10**5) &" + \
                                            " (obj['temperature'] <= 10**6)")
        shell_in_hot = shell_in.cut_region("obj['temperature'] > 10**6")
        shell_out_cold = shell_out.cut_region("obj['temperature'] <= 10**4")
        shell_out_cool = shell_out.cut_region("(obj['temperature'] > 10**4) &" + \
                                              " (obj['temperature'] <= 10**5)")
        shell_out_warm = shell_out.cut_region("(obj['temperature'] > 10**5) &" + \
                                              " (obj['temperature'] <= 10**6)")
        shell_out_hot = shell_out.cut_region("obj['temperature'] > 10**6")

        # Compute fluxes
        net_mass_flux = (np.sum(shell['cell_mass']*shell['radial_velocity']) \
                         /dr).in_units('Msun/yr')
        mass_flux_in = (np.sum(shell_in['cell_mass']*shell_in['radial_velocity']) \
                        /dr).in_units('Msun/yr')
        mass_flux_out = (np.sum(shell_out['cell_mass']*shell_out['radial_velocity']) \
                         /dr).in_units('Msun/yr')

        net_metal_flux = (np.sum(shell['metal_mass']*shell['radial_velocity']) \
                          /dr).in_units('Msun/yr')
        metal_flux_in = (np.sum(shell_in['metal_mass']*shell_in['radial_velocity']) \
                         /dr).in_units('Msun/yr')
        metal_flux_out = (np.sum(shell_out['metal_mass']*shell_out['radial_velocity']) \
                          /dr).in_units('Msun/yr')

        net_cold_mass_flux = (np.sum(shell_cold['cell_mass']*shell_cold['radial_velocity']) \
                              /dr).in_units('Msun/yr')
        cold_mass_flux_in = (np.sum(shell_in_cold['cell_mass']*shell_in_cold['radial_velocity']) \
                             /dr).in_units('Msun/yr')
        cold_mass_flux_out = (np.sum(shell_out_cold['cell_mass']*shell_out_cold['radial_velocity']) \
                              /dr).in_units('Msun/yr')

        net_cool_mass_flux = (np.sum(shell_cool['cell_mass']*shell_cool['radial_velocity']) \
                              /dr).in_units('Msun/yr')
        cool_mass_flux_in = (np.sum(shell_in_cool['cell_mass']*shell_in_cool['radial_velocity']) \
                             /dr).in_units('Msun/yr')
        cool_mass_flux_out = (np.sum(shell_out_cool['cell_mass']*shell_out_cool['radial_velocity']) \
                              /dr).in_units('Msun/yr')

        net_warm_mass_flux = (np.sum(shell_warm['cell_mass']*shell_warm['radial_velocity']) \
                              /dr).in_units('Msun/yr')
        warm_mass_flux_in = (np.sum(shell_in_warm['cell_mass']*shell_in_warm['radial_velocity']) \
                             /dr).in_units('Msun/yr')
        warm_mass_flux_out = (np.sum(shell_out_warm['cell_mass']*shell_out_warm['radial_velocity']) \
                              /dr).in_units('Msun/yr')

        net_hot_mass_flux = (np.sum(shell_hot['cell_mass']*shell_hot['radial_velocity']) \
                             /dr).in_units('Msun/yr')
        hot_mass_flux_in = (np.sum(shell_in_hot['cell_mass']*shell_in_hot['radial_velocity']) \
                            /dr).in_units('Msun/yr')
        hot_mass_flux_out = (np.sum(shell_out_hot['cell_mass']*shell_out_hot['radial_velocity']) \
                             /dr).in_units('Msun/yr')

        if (quadrants):
            # Loop over quadrants
            for q in range(8):
                if (q%2==0):
                    theta_low = 0.
                    theta_up = np.pi/2.
                else:
                    theta_low = np.pi/2.
                    theta_up = np.pi
                if (q==0) or (q==1):
                    phi_low = -np.pi
                    phi_up = -np.pi/2.
                elif (q==2) or (q==3):
                    phi_low = -np.pi/2.
                    phi_up = 0.
                elif (q==4) or (q==5):
                    phi_low = 0.
                    phi_up = np.pi/2.
                elif (q==6) or (q==7):
                    phi_low = np.pi/2.
                    phi_up = np.pi

                shell_q = shell.cut_region("(obj['theta_pos'] >= " + str(theta_low) + ") & " + \
                                       "(obj['theta_pos'] < " + str(theta_up) + ") & " + \
                                       "(obj['phi_pos'] >= " + str(phi_low) + ") & " + \
                                       "(obj['phi_pos'] < " + str(phi_up) + ")")

                # Cut the shell on radial velocity for in and out fluxes
                shell_in_q = shell_q.cut_region("obj['radial_velocity'] < 0")
                shell_out_q = shell_q.cut_region("obj['radial_velocity'] > 0")

                # Cut the shell on temperature for cold, cool, warm, and hot gas
                shell_cold_q = shell_q.cut_region("obj['temperature'] <= 10**4")
                shell_cool_q = shell_q.cut_region("(obj['temperature'] > 10**4) &" + \
                                              " (obj['temperature'] <= 10**5)")
                shell_warm_q = shell_q.cut_region("(obj['temperature'] > 10**5) &" + \
                                              " (obj['temperature'] <= 10**6)")
                shell_hot_q = shell_q.cut_region("obj['temperature'] > 10**6")

                # Cut the shell on both temperature and radial velocity
                shell_in_cold_q = shell_in_q.cut_region("obj['temperature'] <= 10**4")
                shell_in_cool_q = shell_in_q.cut_region("(obj['temperature'] > 10**4) &" + \
                                                    " (obj['temperature'] <= 10**5)")
                shell_in_warm_q = shell_in_q.cut_region("(obj['temperature'] > 10**5) &" + \
                                                    " (obj['temperature'] <= 10**6)")
                shell_in_hot_q = shell_in_q.cut_region("obj['temperature'] > 10**6")
                shell_out_cold_q = shell_out_q.cut_region("obj['temperature'] <= 10**4")
                shell_out_cool_q = shell_out_q.cut_region("(obj['temperature'] > 10**4) &" + \
                                                      " (obj['temperature'] <= 10**5)")
                shell_out_warm_q = shell_out_q.cut_region("(obj['temperature'] > 10**5) &" + \
                                                      " (obj['temperature'] <= 10**6)")
                shell_out_hot_q = shell_out_q.cut_region("obj['temperature'] > 10**6")

                # Compute fluxes
                net_mass_flux_q = (np.sum(shell_q['cell_mass']*shell_q['radial_velocity']) \
                                 /dr).in_units('Msun/yr')
                mass_flux_in_q = (np.sum(shell_in_q['cell_mass']*shell_in_q['radial_velocity']) \
                                /dr).in_units('Msun/yr')
                mass_flux_out_q = (np.sum(shell_out_q['cell_mass']*shell_out_q['radial_velocity']) \
                                 /dr).in_units('Msun/yr')

                net_metal_flux_q = (np.sum(shell_q['metal_mass']*shell_q['radial_velocity']) \
                                  /dr).in_units('Msun/yr')
                metal_flux_in_q = (np.sum(shell_in_q['metal_mass']*shell_in_q['radial_velocity']) \
                                 /dr).in_units('Msun/yr')
                metal_flux_out_q = (np.sum(shell_out_q['metal_mass']*shell_out_q['radial_velocity']) \
                                  /dr).in_units('Msun/yr')

                net_cold_mass_flux_q = (np.sum(shell_cold_q['cell_mass']*shell_cold_q['radial_velocity']) \
                                      /dr).in_units('Msun/yr')
                cold_mass_flux_in_q = (np.sum(shell_in_cold_q['cell_mass']*shell_in_cold_q['radial_velocity']) \
                                     /dr).in_units('Msun/yr')
                cold_mass_flux_out_q = (np.sum(shell_out_cold_q['cell_mass']*shell_out_cold_q['radial_velocity']) \
                                      /dr).in_units('Msun/yr')

                net_cool_mass_flux_q = (np.sum(shell_cool_q['cell_mass']*shell_cool_q['radial_velocity']) \
                                      /dr).in_units('Msun/yr')
                cool_mass_flux_in_q = (np.sum(shell_in_cool_q['cell_mass']*shell_in_cool_q['radial_velocity']) \
                                     /dr).in_units('Msun/yr')
                cool_mass_flux_out_q = (np.sum(shell_out_cool_q['cell_mass']*shell_out_cool_q['radial_velocity']) \
                                      /dr).in_units('Msun/yr')

                net_warm_mass_flux_q = (np.sum(shell_warm_q['cell_mass']*shell_warm_q['radial_velocity']) \
                                      /dr).in_units('Msun/yr')
                warm_mass_flux_in_q = (np.sum(shell_in_warm_q['cell_mass']*shell_in_warm_q['radial_velocity']) \
                                     /dr).in_units('Msun/yr')
                warm_mass_flux_out_q = (np.sum(shell_out_warm_q['cell_mass']*shell_out_warm_q['radial_velocity']) \
                                      /dr).in_units('Msun/yr')

                net_hot_mass_flux_q = (np.sum(shell_hot_q['cell_mass']*shell_hot_q['radial_velocity']) \
                                     /dr).in_units('Msun/yr')
                hot_mass_flux_in_q = (np.sum(shell_in_hot_q['cell_mass']*shell_in_hot_q['radial_velocity']) \
                                    /dr).in_units('Msun/yr')
                hot_mass_flux_out_q = (np.sum(shell_out_hot_q['cell_mass']*shell_out_hot_q['radial_velocity']) \
                                     /dr).in_units('Msun/yr')

                # Add everything to the table
                data_q.add_row([zsnap, q+1, r, net_mass_flux, net_metal_flux, mass_flux_in, mass_flux_out, \
                              metal_flux_in, metal_flux_out, net_cold_mass_flux, cold_mass_flux_in, \
                              cold_mass_flux_out, net_cool_mass_flux, cool_mass_flux_in, \
                              cool_mass_flux_out, net_warm_mass_flux, warm_mass_flux_in, \
                              warm_mass_flux_out, net_hot_mass_flux, hot_mass_flux_in, \
                              hot_mass_flux_out])
        # Add everything to the table
        data.add_row([zsnap, 0, r, net_mass_flux, net_metal_flux, mass_flux_in, mass_flux_out, \
                      metal_flux_in, metal_flux_out, net_cold_mass_flux, cold_mass_flux_in, \
                      cold_mass_flux_out, net_cool_mass_flux, cool_mass_flux_in, \
                      cool_mass_flux_out, net_warm_mass_flux, warm_mass_flux_in, \
                      warm_mass_flux_out, net_hot_mass_flux, hot_mass_flux_in, \
                      hot_mass_flux_out])

    data = set_table_units(data)
    data.write(tablename, path='all_data', serialize_meta=True, overwrite=True)

    if (quadrants):
        data_q = set_table_units(data_q)
        data_q.write(tablename + '_q', path='all_data', serialize_meta=True, overwrite=True)

    return "Fluxes have been calculated for snapshot" + snap + "!"

if __name__ == "__main__":
    args = parse_args()
    print(args.halo)
    print(args.run)
    print(args.system)
    foggie_dir, output_dir, run_dir, trackname, haloname, spectra_dir = get_run_loc_etc(args)

    # Make the list of snapshots to do this for
    print("Starting...")
    print(str(datetime.datetime.now()))
    if (args.output == 'all'): outs = args.output
    else: outs = [args.output]

    # Set directory for output location, making it if necessary
    prefix = output_dir + 'fluxes_halo_00' + str(halo) + '/' + run + '/'
    if not (os.path.exists(prefix)): os.system('mkdir -p ' + prefix)

    for i in range(len(outs)):
        snap = outs[i]

        # Load halo track
        print('foggie_dir: ', foggie_dir)
        print('Opening track: ' + trackname)
        track = Table.read(trackname, format='ascii')
        track.sort('col1')

        # Define where the table will be saved to
        tablename = prefix + snap + '_fluxes'

        # Load snapshot
        print ('Opening snapshot ' + snap)
        ds = yt.load(run_dir + snap + '/' + snap)

        # Get the refined box in physical units
        zsnap = ds.get_parameter('CosmologyCurrentRedshift')
        proper_box_size = get_proper_box_size(ds)
        refine_box, refine_box_center, refine_width_code = get_refine_box(ds, zsnap, track)
        refine_width = refine_width_code * proper_box_size
        refine_width_kpc = YTArray([refine_width], 'kpc')

        # Get halo center
        halo_center, halo_velocity = get_halo_center(ds, refine_box_center)

        # Define the halo center in kpc and the halo velocity in km/s
        halo_center_kpc = YTArray(np.array(halo_center)*proper_box_size, 'kpc')
        halo_velocity_kms = YTArray(halo_velocity, 'km/s')
        # Add the fields we want
        ds.add_field(('gas','vx_corrected'), function=_vx_corrected, units='km/s', take_log=False)
        ds.add_field(('gas', 'vy_corrected'), function=_vy_corrected, units='km/s', take_log=False)
        ds.add_field(('gas', 'vz_corrected'), function=_vz_corrected, units='km/s', take_log=False)
        ds.add_field(('gas', 'radius'), function=_radius, units='kpc', take_log=False, force_override=True)
        ds.add_field(('gas', 'theta_pos'), function=_theta_pos, units=None, take_log=False)
        ds.add_field(('gas', 'phi_pos'), function=_phi_pos, units=None, take_log=False)
        ds.add_field(('gas', 'radial_velocity'), function=_radial_velocity, units='km/s', take_log=False, \
                     force_override=True)

        # Do the actual calculation
        message = calc_fluxes(ds, snap, halo_center, refine_width_kpc, tablename)
        print(message)
        print(str(datetime.datetime.now()))


    print(str(datetime.datetime.now()))
    sys.exit("All snapshots finished!")
