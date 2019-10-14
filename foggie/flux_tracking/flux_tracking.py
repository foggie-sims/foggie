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
import multiprocessing as mp
import datetime

# These imports are FOGGIE-specific files
from foggie.utils.consistency import *
from foggie.utils.get_refine_box import get_refine_box
from foggie.utils.get_halo_center import get_halo_center
from foggie.utils.get_proper_box_size import get_proper_box_size
from foggie.utils.get_run_loc_etc import get_run_loc_etc
from foggie.utils.yt_fields import *

def vel_time(field, data):
    return np.abs(data['radial_velocity_corrected'] * data.ds.dt)

def dist(field, data):
    return np.abs(data['radius_corrected'] - data.ds.surface)

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

def set_table_units(table):
    '''Sets the units for the table. Note this needs to be updated whenever something is added to
    the table. Returns the table.'''

    table_units = {'redshift':None,'quadrant':None,'radius':'kpc','net_mass_flux':'Msun/yr', \
             'net_metal_flux':'Msun/yr', 'mass_flux_in'  :'Msun/yr','mass_flux_out':'Msun/yr', \
             'metal_flux_in' :'Msun/yr', 'metal_flux_out':'Msun/yr',\
             'net_cold_mass_flux':'Msun/yr', 'cold_mass_flux_in':'Msun/yr', \
             'cold_mass_flux_out':'Msun/yr', 'net_cool_mass_flux':'Msun/yr', \
             'cool_mass_flux_in':'Msun/yr', 'cool_mass_flux_out':'Msun/yr', \
             'net_warm_mass_flux':'Msun/yr', 'warm_mass_flux_in':'Msun/yr', \
             'warm_mass_flux_out':'Msun/yr', 'net_hot_mass_flux' :'Msun/yr', \
             'hot_mass_flux_in' :'Msun/yr', 'hot_mass_flux_out' :'Msun/yr', \
             'net_cold_metal_flux':'Msun/yr', 'cold_metal_flux_in':'Msun/yr', \
             'cold_metal_flux_out':'Msun/yr', 'net_cool_metal_flux':'Msun/yr', \
             'cool_metal_flux_in':'Msun/yr', 'cool_metal_flux_out':'Msun/yr', \
             'net_warm_metal_flux':'Msun/yr', 'warm_metal_flux_in':'Msun/yr', \
             'warm_metal_flux_out':'Msun/yr', 'net_hot_metal_flux' :'Msun/yr', \
             'hot_metal_flux_in' :'Msun/yr', 'hot_metal_flux_out' :'Msun/yr', \
             'net_kinetic_energy_flux':'erg/yr', 'net_thermal_energy_flux':'erg/yr', \
             'net_entropy_flux':'cm**2*keV/yr', 'kinetic_energy_flux_in':'erg/yr', \
             'kinetic_energy_flux_out':'erg/yr', 'thermal_energy_flux_in':'erg/yr', \
             'thermal_energy_flux_out':'erg/yr', 'entropy_flux_in':'cm**2*keV/yr', \
             'entropy_flux_out':'cm**2*keV/yr', 'net_cold_kinetic_energy_flux':'erg/yr', \
             'cold_kinetic_energy_flux_in':'erg/yr', 'cold_kinetic_energy_flux_out':'erg/yr', \
             'net_cool_kinetic_energy_flux':'erg/yr', 'cool_kinetic_energy_flux_in':'erg/yr', \
             'cool_kinetic_energy_flux_out':'erg/yr', 'net_warm_kinetic_energy_flux':'erg/yr', \
             'warm_kinetic_energy_flux_in':'erg/yr', 'warm_kinetic_energy_flux_out':'erg/yr', \
             'net_hot_kinetic_energy_flux':'erg/yr', 'hot_kinetic_energy_flux_in':'erg/yr', \
             'hot_kinetic_energy_flux_out':'erg/yr', 'net_cold_thermal_energy_flux':'erg/yr', \
             'cold_thermal_energy_flux_in':'erg/yr', 'cold_thermal_energy_flux_out':'erg/yr', \
             'net_cool_thermal_energy_flux':'erg/yr', 'cool_thermal_energy_flux_in':'erg/yr', \
             'cool_thermal_energy_flux_out':'erg/yr', 'net_warm_thermal_energy_flux':'erg/yr', \
             'warm_thermal_energy_flux_in':'erg/yr', 'warm_thermal_energy_flux_out':'erg/yr', \
             'net_hot_thermal_energy_flux':'erg/yr', 'hot_thermal_energy_flux_in':'erg/yr', \
             'hot_thermal_energy_flux_out':'erg/yr', 'net_cold_entropy_flux':'cm**2*keV/yr', \
             'cold_entropy_flux_in':'cm**2*keV/yr', 'cold_entropy_flux_out':'cm**2*keV/yr', \
             'net_cool_entropy_flux':'cm**2*keV/yr', 'cool_entropy_flux_in':'cm**2*keV/yr', \
             'cool_entropy_flux_out':'cm**2*keV/yr', 'net_warm_entropy_flux':'cm**2*keV/yr', \
             'warm_entropy_flux_in':'cm**2*keV/yr', 'warm_entropy_flux_out':'cm**2*keV/yr', \
             'net_hot_entropy_flux':'cm**2*keV/yr', 'hot_entropy_flux_in':'cm**2*keV/yr', \
             'hot_entropy_flux_out':'cm**2*keV/yr'}
    for key in table.keys():
        table[key].unit = table_units[key]
    return table

def calc_fluxes(ds, snap, zsnap, refine_width_kpc, tablename, **kwargs):
    """Computes the flux through spherical shells centered on the halo center.
    Takes the dataset for the snapshot 'ds', the name of the snapshot 'snap', the redshfit of the
    snapshot 'zsnap', and the width of the refine box in kpc 'refine_width_kpc'
    and does the calculation, then writes a hdf5 table out to 'tablename'.

    Optional arguments:
    quadrants = True will calculate the flux shells within quadrants rather than the whole domain,
        default is False. If this is selected, a second table will be written with '_q' appended
        to 'tablename'.
    """

    quadrants = kwargs.get('quadrants', False)

    # Set up table of everything we want
    # NOTE: Make sure table units are updated when things are added to this table!
    data = Table(names=('redshift', 'quadrant', 'radius', 'net_mass_flux', 'net_metal_flux', \
                        'mass_flux_in', 'mass_flux_out', 'metal_flux_in', 'metal_flux_out', \
                        'net_cold_mass_flux', 'cold_mass_flux_in', 'cold_mass_flux_out', \
                        'net_cool_mass_flux', 'cool_mass_flux_in', 'cool_mass_flux_out', \
                        'net_warm_mass_flux', 'warm_mass_flux_in', 'warm_mass_flux_out', \
                        'net_hot_mass_flux', 'hot_mass_flux_in', 'hot_mass_flux_out', \
                        'net_cold_metal_flux', 'cold_metal_flux_in', 'cold_metal_flux_out', \
                        'net_cool_metal_flux', 'cool_metal_flux_in', 'cool_metal_flux_out', \
                        'net_warm_metal_flux', 'warm_metal_flux_in', 'warm_metal_flux_out', \
                        'net_hot_metal_flux', 'hot_metal_flux_in', 'hot_metal_flux_out', \
                        'net_kinetic_energy_flux', 'net_thermal_energy_flux', 'net_entropy_flux', \
                        'kinetic_energy_flux_in', 'kinetic_energy_flux_out', \
                        'thermal_energy_flux_in', 'thermal_energy_flux_out', \
                        'entropy_flux_in', 'entropy_flux_out', 'net_cold_kinetic_energy_flux', \
                        'cold_kinetic_energy_flux_in', 'cold_kinetic_energy_flux_out', \
                        'net_cool_kinetic_energy_flux', 'cool_kinetic_energy_flux_in', \
                        'cool_kinetic_energy_flux_out', 'net_warm_kinetic_energy_flux', \
                        'warm_kinetic_energy_flux_in', 'warm_kinetic_energy_flux_out', \
                        'net_hot_kinetic_energy_flux', 'hot_kinetic_energy_flux_in', \
                        'hot_kinetic_energy_flux_out', 'net_cold_thermal_energy_flux', \
                        'cold_thermal_energy_flux_in', 'cold_thermal_energy_flux_out', \
                        'net_cool_thermal_energy_flux', 'cool_thermal_energy_flux_in', \
                        'cool_thermal_energy_flux_out', 'net_warm_thermal_energy_flux', \
                        'warm_thermal_energy_flux_in', 'warm_thermal_energy_flux_out', \
                        'net_hot_thermal_energy_flux', 'hot_thermal_energy_flux_in', \
                        'hot_thermal_energy_flux_out', 'net_cold_entropy_flux', \
                        'cold_entropy_flux_in', 'cold_entropy_flux_out', \
                        'net_cool_entropy_flux', 'cool_entropy_flux_in', \
                        'cool_entropy_flux_out', 'net_warm_entropy_flux', \
                        'warm_entropy_flux_in', 'warm_entropy_flux_out', \
                        'net_hot_entropy_flux', 'hot_entropy_flux_in', \
                        'hot_entropy_flux_out'), \
                 dtype=('f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', \
                        'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', \
                        'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', \
                        'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', \
                        'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', \
                        'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', \
                        'f8', 'f8', 'f8', 'f8', 'f8', 'f8'))

    if (quadrants):
        data_q = Table(names=('redshift', 'quadrant', 'radius', 'net_mass_flux', 'net_metal_flux', \
                            'mass_flux_in', 'mass_flux_out', 'metal_flux_in', 'metal_flux_out', \
                            'net_cold_mass_flux', 'cold_mass_flux_in', 'cold_mass_flux_out', \
                            'net_cool_mass_flux', 'cool_mass_flux_in', 'cool_mass_flux_out', \
                            'net_warm_mass_flux', 'warm_mass_flux_in', 'warm_mass_flux_out', \
                            'net_hot_mass_flux', 'hot_mass_flux_in', 'hot_mass_flux_out', \
                            'net_cold_metal_flux', 'cold_metal_flux_in', 'cold_metal_flux_out', \
                            'net_cool_metal_flux', 'cool_metal_flux_in', 'cool_metal_flux_out', \
                            'net_warm_metal_flux', 'warm_metal_flux_in', 'warm_metal_flux_out', \
                            'net_hot_metal_flux', 'hot_metal_flux_in', 'hot_metal_flux_out', \
                            'net_kinetic_energy_flux', 'net_thermal_energy_flux', 'net_entropy_flux', \
                            'kinetic_energy_flux_in', 'kinetic_energy_flux_out', \
                            'thermal_energy_flux_in', 'thermal_energy_flux_out', \
                            'entropy_flux_in', 'entropy_flux_out', 'net_cold_kinetic_energy_flux', \
                            'cold_kinetic_energy_flux_in', 'cold_kinetic_energy_flux_out', \
                            'net_cool_kinetic_energy_flux', 'cool_kinetic_energy_flux_in', \
                            'cool_kinetic_energy_flux_out', 'net_warm_kinetic_energy_flux', \
                            'warm_kinetic_energy_flux_in', 'warm_kinetic_energy_flux_out', \
                            'net_hot_kinetic_energy_flux', 'hot_kinetic_energy_flux_in', \
                            'hot_kinetic_energy_flux_out', 'net_cold_thermal_energy_flux', \
                            'cold_thermal_energy_flux_in', 'cold_thermal_energy_flux_out', \
                            'net_cool_thermal_energy_flux', 'cool_thermal_energy_flux_in', \
                            'cool_thermal_energy_flux_out', 'net_warm_thermal_energy_flux', \
                            'warm_thermal_energy_flux_in', 'warm_thermal_energy_flux_out', \
                            'net_hot_thermal_energy_flux', 'hot_thermal_energy_flux_in', \
                            'hot_thermal_energy_flux_out', 'net_cold_entropy_flux', \
                            'cold_entropy_flux_in', 'cold_entropy_flux_out', \
                            'net_cool_entropy_flux', 'cool_entropy_flux_in', \
                            'cool_entropy_flux_out', 'net_warm_entropy_flux', \
                            'warm_entropy_flux_in', 'warm_entropy_flux_out', \
                            'net_hot_entropy_flux', 'hot_entropy_flux_in', \
                            'hot_entropy_flux_out'), \
                     dtype=('f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', \
                            'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', \
                            'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', \
                            'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', \
                            'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', \
                            'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', \
                            'f8', 'f8', 'f8', 'f8', 'f8', 'f8'))

    # Define the radii of the spherical shells where we want to calculate fluxes
    radii = 0.5*refine_width_kpc * np.arange(0.1, 0.9, 0.01)

    # Loop over radii
    for i in range(len(radii)-1):
        r_low = radii[i]
        r_high = radii[i+1]
        dr = r_high - r_low
        r = (r_low + r_high)/2.

        if (i%10==0): print("Computing radius " + str(i) + "/" + str(len(radii)-1) + \
                            " for snapshot " + snap)

        # Make the spheres and shell for computing
        inner_sphere = ds.sphere(ds.halo_center_kpc, r_low)
        outer_sphere = ds.sphere(ds.halo_center_kpc, r_high)
        shell = outer_sphere - inner_sphere

        # Cut the shell on radial velocity for in and out fluxes
        shell_in = shell.cut_region("obj['gas','radial_velocity_corrected'] < 0")
        shell_out = shell.cut_region("obj['gas','radial_velocity_corrected'] > 0")

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
        net_mass_flux = (np.sum(shell['cell_mass'] * \
                         shell['gas','radial_velocity_corrected'])/dr).in_units('Msun/yr')
        mass_flux_in = (np.sum(shell_in['cell_mass'] * \
                        shell_in['gas','radial_velocity_corrected'])/dr).in_units('Msun/yr')
        mass_flux_out = (np.sum(shell_out['cell_mass'] * \
                         shell_out['gas','radial_velocity_corrected'])/dr).in_units('Msun/yr')

        net_metal_flux = (np.sum(shell['metal_mass'] * \
                          shell['gas','radial_velocity_corrected'])/dr).in_units('Msun/yr')
        metal_flux_in = (np.sum(shell_in['metal_mass'] * \
                         shell_in['gas','radial_velocity_corrected'])/dr).in_units('Msun/yr')
        metal_flux_out = (np.sum(shell_out['metal_mass'] * \
                          shell_out['gas','radial_velocity_corrected'])/dr).in_units('Msun/yr')

        net_cold_mass_flux = (np.sum(shell_cold['cell_mass'] * \
                              shell_cold['gas','radial_velocity_corrected'])/dr).in_units('Msun/yr')
        cold_mass_flux_in = (np.sum(shell_in_cold['cell_mass'] * \
                             shell_in_cold['gas','radial_velocity_corrected']) \
                             /dr).in_units('Msun/yr')
        cold_mass_flux_out = (np.sum(shell_out_cold['cell_mass'] * \
                              shell_out_cold['gas','radial_velocity_corrected']) \
                              /dr).in_units('Msun/yr')

        net_cool_mass_flux = (np.sum(shell_cool['cell_mass'] * \
                              shell_cool['gas','radial_velocity_corrected'])/dr).in_units('Msun/yr')
        cool_mass_flux_in = (np.sum(shell_in_cool['cell_mass'] * \
                             shell_in_cool['gas','radial_velocity_corrected']) \
                             /dr).in_units('Msun/yr')
        cool_mass_flux_out = (np.sum(shell_out_cool['cell_mass'] * \
                              shell_out_cool['gas','radial_velocity_corrected']) \
                              /dr).in_units('Msun/yr')

        net_warm_mass_flux = (np.sum(shell_warm['cell_mass'] * \
                              shell_warm['gas','radial_velocity_corrected']) \
                              /dr).in_units('Msun/yr')
        warm_mass_flux_in = (np.sum(shell_in_warm['cell_mass'] * \
                             shell_in_warm['gas','radial_velocity_corrected']) \
                             /dr).in_units('Msun/yr')
        warm_mass_flux_out = (np.sum(shell_out_warm['cell_mass'] * \
                              shell_out_warm['gas','radial_velocity_corrected']) \
                              /dr).in_units('Msun/yr')

        net_hot_mass_flux = (np.sum(shell_hot['cell_mass'] * \
                             shell_hot['gas','radial_velocity_corrected'])/dr).in_units('Msun/yr')
        hot_mass_flux_in = (np.sum(shell_in_hot['cell_mass'] * \
                            shell_in_hot['gas','radial_velocity_corrected']) \
                            /dr).in_units('Msun/yr')
        hot_mass_flux_out = (np.sum(shell_out_hot['cell_mass'] * \
                             shell_out_hot['gas','radial_velocity_corrected']) \
                             /dr).in_units('Msun/yr')

        net_cold_metal_flux = (np.sum(shell_cold['metal_mass'] * \
                              shell_cold['gas','radial_velocity_corrected'])/dr).in_units('Msun/yr')
        cold_metal_flux_in = (np.sum(shell_in_cold['metal_mass'] * \
                             shell_in_cold['gas','radial_velocity_corrected']) \
                             /dr).in_units('Msun/yr')
        cold_metal_flux_out = (np.sum(shell_out_cold['metal_mass'] * \
                              shell_out_cold['gas','radial_velocity_corrected']) \
                              /dr).in_units('Msun/yr')

        net_cool_metal_flux = (np.sum(shell_cool['metal_mass'] * \
                              shell_cool['gas','radial_velocity_corrected'])/dr).in_units('Msun/yr')
        cool_metal_flux_in = (np.sum(shell_in_cool['metal_mass'] * \
                             shell_in_cool['gas','radial_velocity_corrected']) \
                             /dr).in_units('Msun/yr')
        cool_metal_flux_out = (np.sum(shell_out_cool['metal_mass'] * \
                              shell_out_cool['gas','radial_velocity_corrected']) \
                              /dr).in_units('Msun/yr')

        net_warm_metal_flux = (np.sum(shell_warm['metal_mass'] * \
                              shell_warm['gas','radial_velocity_corrected']) \
                              /dr).in_units('Msun/yr')
        warm_metal_flux_in = (np.sum(shell_in_warm['metal_mass'] * \
                             shell_in_warm['gas','radial_velocity_corrected']) \
                             /dr).in_units('Msun/yr')
        warm_metal_flux_out = (np.sum(shell_out_warm['metal_mass'] * \
                              shell_out_warm['gas','radial_velocity_corrected']) \
                              /dr).in_units('Msun/yr')

        net_hot_metal_flux = (np.sum(shell_hot['metal_mass'] * \
                             shell_hot['gas','radial_velocity_corrected'])/dr).in_units('Msun/yr')
        hot_metal_flux_in = (np.sum(shell_in_hot['metal_mass'] * \
                            shell_in_hot['gas','radial_velocity_corrected']) \
                            /dr).in_units('Msun/yr')
        hot_metal_flux_out = (np.sum(shell_out_hot['metal_mass'] * \
                             shell_out_hot['gas','radial_velocity_corrected']) \
                             /dr).in_units('Msun/yr')

        net_kinetic_energy_flux = (np.sum(shell['kinetic_energy_corrected'] * \
                         shell['gas','radial_velocity_corrected'])/dr).in_units('erg/yr')
        kinetic_energy_flux_in = (np.sum(shell_in['kinetic_energy_corrected'] * \
                        shell_in['gas','radial_velocity_corrected'])/dr).in_units('erg/yr')
        kinetic_energy_flux_out = (np.sum(shell_out['kinetic_energy_corrected'] * \
                         shell_out['gas','radial_velocity_corrected'])/dr).in_units('erg/yr')

        net_thermal_energy_flux = (np.sum(shell['thermal_energy'] * shell['cell_mass'] * \
                          shell['gas','radial_velocity_corrected'])/dr).in_units('erg/yr')
        thermal_energy_flux_in = (np.sum(shell_in['thermal_energy'] * shell_in['cell_mass'] * \
                         shell_in['gas','radial_velocity_corrected'])/dr).in_units('erg/yr')
        thermal_energy_flux_out = (np.sum(shell_out['thermal_energy'] * shell_out['cell_mass'] * \
                          shell_out['gas','radial_velocity_corrected'])/dr).in_units('erg/yr')

        net_entropy_flux = (np.sum(shell['entropy'] * \
                          shell['gas','radial_velocity_corrected'])/dr).in_units('keV*cm**2/yr')
        entropy_flux_in = (np.sum(shell_in['entropy'] * \
                         shell_in['gas','radial_velocity_corrected'])/dr).in_units('keV*cm**2/yr')
        entropy_flux_out = (np.sum(shell_out['entropy'] * \
                          shell_out['gas','radial_velocity_corrected'])/dr).in_units('keV*cm**2/yr')

        net_cold_kinetic_energy_flux = (np.sum(shell_cold['kinetic_energy_corrected'] * \
                              shell_cold['gas','radial_velocity_corrected'])/dr).in_units('erg/yr')
        cold_kinetic_energy_flux_in = (np.sum(shell_in_cold['kinetic_energy_corrected'] * \
                             shell_in_cold['gas','radial_velocity_corrected']) \
                             /dr).in_units('erg/yr')
        cold_kinetic_energy_flux_out = (np.sum(shell_out_cold['kinetic_energy_corrected'] * \
                              shell_out_cold['gas','radial_velocity_corrected']) \
                              /dr).in_units('erg/yr')

        net_cool_kinetic_energy_flux = (np.sum(shell_cool['kinetic_energy_corrected'] * \
                              shell_cool['gas','radial_velocity_corrected'])/dr).in_units('erg/yr')
        cool_kinetic_energy_flux_in = (np.sum(shell_in_cool['kinetic_energy_corrected'] * \
                             shell_in_cool['gas','radial_velocity_corrected']) \
                             /dr).in_units('erg/yr')
        cool_kinetic_energy_flux_out = (np.sum(shell_out_cool['kinetic_energy_corrected'] * \
                              shell_out_cool['gas','radial_velocity_corrected']) \
                              /dr).in_units('erg/yr')

        net_warm_kinetic_energy_flux = (np.sum(shell_warm['kinetic_energy_corrected'] * \
                              shell_warm['gas','radial_velocity_corrected']) \
                              /dr).in_units('erg/yr')
        warm_kinetic_energy_flux_in = (np.sum(shell_in_warm['kinetic_energy_corrected'] * \
                             shell_in_warm['gas','radial_velocity_corrected']) \
                             /dr).in_units('erg/yr')
        warm_kinetic_energy_flux_out = (np.sum(shell_out_warm['kinetic_energy_corrected'] * \
                              shell_out_warm['gas','radial_velocity_corrected']) \
                              /dr).in_units('erg/yr')

        net_hot_kinetic_energy_flux = (np.sum(shell_hot['kinetic_energy_corrected'] * \
                             shell_hot['gas','radial_velocity_corrected'])/dr).in_units('erg/yr')
        hot_kinetic_energy_flux_in = (np.sum(shell_in_hot['kinetic_energy_corrected'] * \
                            shell_in_hot['gas','radial_velocity_corrected']) \
                            /dr).in_units('erg/yr')
        hot_kinetic_energy_flux_out = (np.sum(shell_out_hot['kinetic_energy_corrected'] * \
                             shell_out_hot['gas','radial_velocity_corrected']) \
                             /dr).in_units('erg/yr')

        net_cold_thermal_energy_flux = (np.sum(shell_cold['thermal_energy'] * \
                              shell_cold['cell_mass'] * \
                              shell_cold['gas','radial_velocity_corrected'])/dr).in_units('erg/yr')
        cold_thermal_energy_flux_in = (np.sum(shell_in_cold['thermal_energy'] * \
                             shell_in_cold['cell_mass'] * \
                             shell_in_cold['gas','radial_velocity_corrected']) \
                             /dr).in_units('erg/yr')
        cold_thermal_energy_flux_out = (np.sum(shell_out_cold['thermal_energy'] * \
                              shell_out_cold['cell_mass'] * \
                              shell_out_cold['gas','radial_velocity_corrected']) \
                              /dr).in_units('erg/yr')

        net_cool_thermal_energy_flux = (np.sum(shell_cool['thermal_energy'] * \
                              shell_cool['cell_mass'] * \
                              shell_cool['gas','radial_velocity_corrected'])/dr).in_units('erg/yr')
        cool_thermal_energy_flux_in = (np.sum(shell_in_cool['thermal_energy'] * \
                             shell_in_cool['cell_mass'] * \
                             shell_in_cool['gas','radial_velocity_corrected']) \
                             /dr).in_units('erg/yr')
        cool_thermal_energy_flux_out = (np.sum(shell_out_cool['thermal_energy'] * \
                              shell_out_cool['cell_mass'] * \
                              shell_out_cool['gas','radial_velocity_corrected']) \
                              /dr).in_units('erg/yr')

        net_warm_thermal_energy_flux = (np.sum(shell_warm['thermal_energy'] * \
                              shell_warm['cell_mass'] * \
                              shell_warm['gas','radial_velocity_corrected']) \
                              /dr).in_units('erg/yr')
        warm_thermal_energy_flux_in = (np.sum(shell_in_warm['thermal_energy'] * \
                             shell_in_warm['cell_mass'] * \
                             shell_in_warm['gas','radial_velocity_corrected']) \
                             /dr).in_units('erg/yr')
        warm_thermal_energy_flux_out = (np.sum(shell_out_warm['thermal_energy'] * \
                              shell_out_warm['cell_mass'] * \
                              shell_out_warm['gas','radial_velocity_corrected']) \
                              /dr).in_units('erg/yr')

        net_hot_thermal_energy_flux = (np.sum(shell_hot['thermal_energy'] * \
                             shell_hot['cell_mass'] * \
                             shell_hot['gas','radial_velocity_corrected'])/dr).in_units('erg/yr')
        hot_thermal_energy_flux_in = (np.sum(shell_in_hot['thermal_energy'] * \
                            shell_in_hot['cell_mass'] * \
                            shell_in_hot['gas','radial_velocity_corrected']) \
                            /dr).in_units('erg/yr')
        hot_thermal_energy_flux_out = (np.sum(shell_out_hot['thermal_energy'] * \
                             shell_out_hot['cell_mass'] * \
                             shell_out_hot['gas','radial_velocity_corrected']) \
                             /dr).in_units('erg/yr')

        net_cold_entropy_flux = (np.sum(shell_cold['entropy'] * \
                              shell_cold['gas','radial_velocity_corrected'])/dr).in_units('keV*cm**2/yr')
        cold_entropy_flux_in = (np.sum(shell_in_cold['entropy'] * \
                             shell_in_cold['gas','radial_velocity_corrected']) \
                             /dr).in_units('keV*cm**2/yr')
        cold_entropy_flux_out = (np.sum(shell_out_cold['entropy'] * \
                              shell_out_cold['gas','radial_velocity_corrected']) \
                              /dr).in_units('keV*cm**2/yr')

        net_cool_entropy_flux = (np.sum(shell_cool['entropy'] * \
                              shell_cool['gas','radial_velocity_corrected'])/dr).in_units('keV*cm**2/yr')
        cool_entropy_flux_in = (np.sum(shell_in_cool['entropy'] * \
                             shell_in_cool['gas','radial_velocity_corrected']) \
                             /dr).in_units('keV*cm**2/yr')
        cool_entropy_flux_out = (np.sum(shell_out_cool['entropy'] * \
                              shell_out_cool['gas','radial_velocity_corrected']) \
                              /dr).in_units('keV*cm**2/yr')

        net_warm_entropy_flux = (np.sum(shell_warm['entropy'] * \
                              shell_warm['gas','radial_velocity_corrected']) \
                              /dr).in_units('keV*cm**2/yr')
        warm_entropy_flux_in = (np.sum(shell_in_warm['entropy'] * \
                             shell_in_warm['gas','radial_velocity_corrected']) \
                             /dr).in_units('keV*cm**2/yr')
        warm_entropy_flux_out = (np.sum(shell_out_warm['entropy'] * \
                              shell_out_warm['gas','radial_velocity_corrected']) \
                              /dr).in_units('keV*cm**2/yr')

        net_hot_entropy_flux = (np.sum(shell_hot['entropy'] * \
                             shell_hot['gas','radial_velocity_corrected'])/dr).in_units('keV*cm**2/yr')
        hot_entropy_flux_in = (np.sum(shell_in_hot['entropy'] * \
                            shell_in_hot['gas','radial_velocity_corrected']) \
                            /dr).in_units('keV*cm**2/yr')
        hot_entropy_flux_out = (np.sum(shell_out_hot['entropy'] * \
                             shell_out_hot['gas','radial_velocity_corrected']) \
                             /dr).in_units('keV*cm**2/yr')

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
                shell_in_q = shell_q.cut_region("obj['gas','radial_velocity_corrected'] < 0")
                shell_out_q = shell_q.cut_region("obj['gas','radial_velocity_corrected'] > 0")

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
                net_mass_flux_q = (np.sum(shell_q['cell_mass'] * \
                                   shell_q['gas','radial_velocity_corrected']) \
                                   /dr).in_units('Msun/yr')
                mass_flux_in_q = (np.sum(shell_in_q['cell_mass'] * \
                                  shell_in_q['gas','radial_velocity_corrected']) \
                                  /dr).in_units('Msun/yr')
                mass_flux_out_q = (np.sum(shell_out_q['cell_mass'] * \
                                   shell_out_q['gas','radial_velocity_corrected']) \
                                   /dr).in_units('Msun/yr')

                net_metal_flux_q = (np.sum(shell_q['metal_mass'] * \
                                    shell_q['gas','radial_velocity_corrected']) \
                                    /dr).in_units('Msun/yr')
                metal_flux_in_q = (np.sum(shell_in_q['metal_mass'] * \
                                   shell_in_q['gas','radial_velocity_corrected']) \
                                   /dr).in_units('Msun/yr')
                metal_flux_out_q = (np.sum(shell_out_q['metal_mass'] * \
                                    shell_out_q['gas','radial_velocity_corrected']) \
                                    /dr).in_units('Msun/yr')

                net_cold_mass_flux_q = (np.sum(shell_cold_q['cell_mass'] * \
                                        shell_cold_q['gas','radial_velocity_corrected']) \
                                        /dr).in_units('Msun/yr')
                cold_mass_flux_in_q = (np.sum(shell_in_cold_q['cell_mass'] * \
                                       shell_in_cold_q['gas','radial_velocity_corrected']) \
                                       /dr).in_units('Msun/yr')
                cold_mass_flux_out_q = (np.sum(shell_out_cold_q['cell_mass'] * \
                                        shell_out_cold_q['gas','radial_velocity_corrected']) \
                                        /dr).in_units('Msun/yr')

                net_cool_mass_flux_q = (np.sum(shell_cool_q['cell_mass'] * \
                                        shell_cool_q['gas','radial_velocity_corrected']) \
                                        /dr).in_units('Msun/yr')
                cool_mass_flux_in_q = (np.sum(shell_in_cool_q['cell_mass'] * \
                                       shell_in_cool_q['gas','radial_velocity_corrected']) \
                                       /dr).in_units('Msun/yr')
                cool_mass_flux_out_q = (np.sum(shell_out_cool_q['cell_mass'] * \
                                        shell_out_cool_q['gas','radial_velocity_corrected']) \
                                        /dr).in_units('Msun/yr')

                net_warm_mass_flux_q = (np.sum(shell_warm_q['cell_mass'] * \
                                        shell_warm_q['gas','radial_velocity_corrected']) \
                                        /dr).in_units('Msun/yr')
                warm_mass_flux_in_q = (np.sum(shell_in_warm_q['cell_mass'] * \
                                       shell_in_warm_q['gas','radial_velocity_corrected']) \
                                       /dr).in_units('Msun/yr')
                warm_mass_flux_out_q = (np.sum(shell_out_warm_q['cell_mass'] * \
                                        shell_out_warm_q['gas','radial_velocity_corrected']) \
                                        /dr).in_units('Msun/yr')

                net_hot_mass_flux_q = (np.sum(shell_hot_q['cell_mass'] * \
                                       shell_hot_q['gas','radial_velocity_corrected']) \
                                       /dr).in_units('Msun/yr')
                hot_mass_flux_in_q = (np.sum(shell_in_hot_q['cell_mass'] * \
                                      shell_in_hot_q['gas','radial_velocity_corrected']) \
                                      /dr).in_units('Msun/yr')
                hot_mass_flux_out_q = (np.sum(shell_out_hot_q['cell_mass'] * \
                                       shell_out_hot_q['gas','radial_velocity_corrected']) \
                                       /dr).in_units('Msun/yr')

                net_cold_metal_flux_q = (np.sum(shell_cold_q['metal_mass'] * \
                                        shell_cold_q['gas','radial_velocity_corrected']) \
                                        /dr).in_units('Msun/yr')
                cold_metal_flux_in_q = (np.sum(shell_in_cold_q['metal_mass'] * \
                                       shell_in_cold_q['gas','radial_velocity_corrected']) \
                                       /dr).in_units('Msun/yr')
                cold_metal_flux_out_q = (np.sum(shell_out_cold_q['metal_mass'] * \
                                        shell_out_cold_q['gas','radial_velocity_corrected']) \
                                        /dr).in_units('Msun/yr')

                net_cool_metal_flux_q = (np.sum(shell_cool_q['metal_mass'] * \
                                        shell_cool_q['gas','radial_velocity_corrected']) \
                                        /dr).in_units('Msun/yr')
                cool_metal_flux_in_q = (np.sum(shell_in_cool_q['metal_mass'] * \
                                       shell_in_cool_q['gas','radial_velocity_corrected']) \
                                       /dr).in_units('Msun/yr')
                cool_metal_flux_out_q = (np.sum(shell_out_cool_q['metal_mass'] * \
                                        shell_out_cool_q['gas','radial_velocity_corrected']) \
                                        /dr).in_units('Msun/yr')

                net_warm_metal_flux_q = (np.sum(shell_warm_q['metal_mass'] * \
                                        shell_warm_q['gas','radial_velocity_corrected']) \
                                        /dr).in_units('Msun/yr')
                warm_metal_flux_in_q = (np.sum(shell_in_warm_q['metal_mass'] * \
                                       shell_in_warm_q['gas','radial_velocity_corrected']) \
                                       /dr).in_units('Msun/yr')
                warm_metal_flux_out_q = (np.sum(shell_out_warm_q['metal_mass'] * \
                                        shell_out_warm_q['gas','radial_velocity_corrected']) \
                                        /dr).in_units('Msun/yr')

                net_hot_metal_flux_q = (np.sum(shell_hot_q['metal_mass'] * \
                                       shell_hot_q['gas','radial_velocity_corrected']) \
                                       /dr).in_units('Msun/yr')
                hot_metal_flux_in_q = (np.sum(shell_in_hot_q['metal_mass'] * \
                                      shell_in_hot_q['gas','radial_velocity_corrected']) \
                                      /dr).in_units('Msun/yr')
                hot_metal_flux_out_q = (np.sum(shell_out_hot_q['metal_mass'] * \
                                       shell_out_hot_q['gas','radial_velocity_corrected']) \
                                       /dr).in_units('Msun/yr')

                net_kinetic_energy_flux_q = (np.sum(shell_q['kinetic_energy_corrected'] * \
                                 shell_q['gas','radial_velocity_corrected'])/dr).in_units('erg/yr')
                kinetic_energy_flux_in_q = (np.sum(shell_in_q['kinetic_energy_corrected'] * \
                                shell_in_q['gas','radial_velocity_corrected'])/dr).in_units('erg/yr')
                kinetic_energy_flux_out_q = (np.sum(shell_out_q['kinetic_energy_corrected'] * \
                                 shell_out_q['gas','radial_velocity_corrected'])/dr).in_units('erg/yr')

                net_thermal_energy_flux_q = (np.sum(shell_q['thermal_energy'] * shell_q['cell_mass'] * \
                                  shell_q['gas','radial_velocity_corrected'])/dr).in_units('erg/yr')
                thermal_energy_flux_in_q = (np.sum(shell_in_q['thermal_energy'] * shell_in_q['cell_mass'] * \
                                 shell_in_q['gas','radial_velocity_corrected'])/dr).in_units('erg/yr')
                thermal_energy_flux_out_q = (np.sum(shell_out_q['thermal_energy'] * shell_out_q['cell_mass'] * \
                                  shell_out_q['gas','radial_velocity_corrected'])/dr).in_units('erg/yr')

                net_entropy_flux_q = (np.sum(shell_q['entropy'] * \
                                  shell_q['gas','radial_velocity_corrected'])/dr).in_units('keV*cm**2/yr')
                entropy_flux_in_q = (np.sum(shell_in_q['entropy'] * \
                                 shell_in_q['gas','radial_velocity_corrected'])/dr).in_units('keV*cm**2/yr')
                entropy_flux_out_q = (np.sum(shell_out_q['entropy'] * \
                                  shell_out_q['gas','radial_velocity_corrected'])/dr).in_units('keV*cm**2/yr')

                net_cold_kinetic_energy_flux_q = (np.sum(shell_cold_q['kinetic_energy_corrected'] * \
                                      shell_cold_q['gas','radial_velocity_corrected'])/dr).in_units('erg/yr')
                cold_kinetic_energy_flux_in_q = (np.sum(shell_in_cold_q['kinetic_energy_corrected'] * \
                                     shell_in_cold_q['gas','radial_velocity_corrected']) \
                                     /dr).in_units('erg/yr')
                cold_kinetic_energy_flux_out_q = (np.sum(shell_out_cold_q['kinetic_energy_corrected'] * \
                                      shell_out_cold_q['gas','radial_velocity_corrected']) \
                                      /dr).in_units('erg/yr')

                net_cool_kinetic_energy_flux_q = (np.sum(shell_cool_q['kinetic_energy_corrected'] * \
                                      shell_cool_q['gas','radial_velocity_corrected'])/dr).in_units('erg/yr')
                cool_kinetic_energy_flux_in_q = (np.sum(shell_in_cool_q['kinetic_energy_corrected'] * \
                                     shell_in_cool_q['gas','radial_velocity_corrected']) \
                                     /dr).in_units('erg/yr')
                cool_kinetic_energy_flux_out_q = (np.sum(shell_out_cool_q['kinetic_energy_corrected'] * \
                                      shell_out_cool_q['gas','radial_velocity_corrected']) \
                                      /dr).in_units('erg/yr')

                net_warm_kinetic_energy_flux_q = (np.sum(shell_warm_q['kinetic_energy_corrected'] * \
                                      shell_warm_q['gas','radial_velocity_corrected']) \
                                      /dr).in_units('erg/yr')
                warm_kinetic_energy_flux_in_q = (np.sum(shell_in_warm_q['kinetic_energy_corrected'] * \
                                     shell_in_warm_q['gas','radial_velocity_corrected']) \
                                     /dr).in_units('erg/yr')
                warm_kinetic_energy_flux_out_q = (np.sum(shell_out_warm_q['kinetic_energy_corrected'] * \
                                      shell_out_warm_q['gas','radial_velocity_corrected']) \
                                      /dr).in_units('erg/yr')

                net_hot_kinetic_energy_flux_q = (np.sum(shell_hot_q['kinetic_energy_corrected'] * \
                                     shell_hot_q['gas','radial_velocity_corrected'])/dr).in_units('erg/yr')
                hot_kinetic_energy_flux_in_q = (np.sum(shell_in_hot_q['kinetic_energy_corrected'] * \
                                    shell_in_hot_q['gas','radial_velocity_corrected']) \
                                    /dr).in_units('erg/yr')
                hot_kinetic_energy_flux_out_q = (np.sum(shell_out_hot_q['kinetic_energy_corrected'] * \
                                     shell_out_hot_q['gas','radial_velocity_corrected']) \
                                     /dr).in_units('erg/yr')

                net_cold_thermal_energy_flux_q = (np.sum(shell_cold_q['thermal_energy'] * \
                                      shell_cold_q['cell_mass'] * \
                                      shell_cold_q['gas','radial_velocity_corrected'])/dr).in_units('erg/yr')
                cold_thermal_energy_flux_in_q = (np.sum(shell_in_cold_q['thermal_energy'] * \
                                     shell_in_cold_q['cell_mass'] * \
                                     shell_in_cold_q['gas','radial_velocity_corrected']) \
                                     /dr).in_units('erg/yr')
                cold_thermal_energy_flux_out_q = (np.sum(shell_out_cold_q['thermal_energy'] * \
                                      shell_out_cold_q['cell_mass'] * \
                                      shell_out_cold_q['gas','radial_velocity_corrected']) \
                                      /dr).in_units('erg/yr')

                net_cool_thermal_energy_flux_q = (np.sum(shell_cool_q['thermal_energy'] * \
                                      shell_cool_q['cell_mass'] * \
                                      shell_cool_q['gas','radial_velocity_corrected'])/dr).in_units('erg/yr')
                cool_thermal_energy_flux_in_q = (np.sum(shell_in_cool_q['thermal_energy'] * \
                                     shell_in_cool_q['cell_mass'] * \
                                     shell_in_cool_q['gas','radial_velocity_corrected']) \
                                     /dr).in_units('erg/yr')
                cool_thermal_energy_flux_out_q = (np.sum(shell_out_cool_q['thermal_energy'] * \
                                      shell_out_cool_q['cell_mass'] * \
                                      shell_out_cool_q['gas','radial_velocity_corrected']) \
                                      /dr).in_units('erg/yr')

                net_warm_thermal_energy_flux_q = (np.sum(shell_warm_q['thermal_energy'] * \
                                      shell_warm_q['cell_mass'] * \
                                      shell_warm_q['gas','radial_velocity_corrected']) \
                                      /dr).in_units('erg/yr')
                warm_thermal_energy_flux_in_q = (np.sum(shell_in_warm_q['thermal_energy'] * \
                                     shell_in_warm_q['cell_mass'] * \
                                     shell_in_warm_q['gas','radial_velocity_corrected']) \
                                     /dr).in_units('erg/yr')
                warm_thermal_energy_flux_out_q = (np.sum(shell_out_warm_q['thermal_energy'] * \
                                      shell_out_warm_q['cell_mass'] * \
                                      shell_out_warm_q['gas','radial_velocity_corrected']) \
                                      /dr).in_units('erg/yr')

                net_hot_thermal_energy_flux_q = (np.sum(shell_hot_q['thermal_energy'] * \
                                     shell_hot_q['cell_mass'] * \
                                     shell_hot_q['gas','radial_velocity_corrected'])/dr).in_units('erg/yr')
                hot_thermal_energy_flux_in_q = (np.sum(shell_in_hot_q['thermal_energy'] * \
                                    shell_in_hot_q['cell_mass'] * \
                                    shell_in_hot_q['gas','radial_velocity_corrected']) \
                                    /dr).in_units('erg/yr')
                hot_thermal_energy_flux_out_q = (np.sum(shell_out_hot_q['thermal_energy'] * \
                                     shell_out_hot_q['cell_mass'] * \
                                     shell_out_hot_q['gas','radial_velocity_corrected']) \
                                     /dr).in_units('erg/yr')

                net_cold_entropy_flux_q = (np.sum(shell_cold_q['entropy'] * \
                                      shell_cold_q['gas','radial_velocity_corrected'])/dr).in_units('keV*cm**2/yr')
                cold_entropy_flux_in_q = (np.sum(shell_in_cold_q['entropy'] * \
                                     shell_in_cold_q['gas','radial_velocity_corrected']) \
                                     /dr).in_units('keV*cm**2/yr')
                cold_entropy_flux_out_q = (np.sum(shell_out_cold_q['entropy'] * \
                                      shell_out_cold_q['gas','radial_velocity_corrected']) \
                                      /dr).in_units('keV*cm**2/yr')

                net_cool_entropy_flux_q = (np.sum(shell_cool_q['entropy'] * \
                                      shell_cool_q['gas','radial_velocity_corrected'])/dr).in_units('keV*cm**2/yr')
                cool_entropy_flux_in_q = (np.sum(shell_in_cool_q['entropy'] * \
                                     shell_in_cool_q['gas','radial_velocity_corrected']) \
                                     /dr).in_units('keV*cm**2/yr')
                cool_entropy_flux_out_q = (np.sum(shell_out_cool_q['entropy'] * \
                                      shell_out_cool_q['gas','radial_velocity_corrected']) \
                                      /dr).in_units('keV*cm**2/yr')

                net_warm_entropy_flux_q = (np.sum(shell_warm_q['entropy'] * \
                                      shell_warm_q['gas','radial_velocity_corrected']) \
                                      /dr).in_units('keV*cm**2/yr')
                warm_entropy_flux_in_q = (np.sum(shell_in_warm_q['entropy'] * \
                                     shell_in_warm_q['gas','radial_velocity_corrected']) \
                                     /dr).in_units('keV*cm**2/yr')
                warm_entropy_flux_out_q = (np.sum(shell_out_warm_q['entropy'] * \
                                      shell_out_warm_q['gas','radial_velocity_corrected']) \
                                      /dr).in_units('keV*cm**2/yr')

                net_hot_entropy_flux_q = (np.sum(shell_hot_q['entropy'] * \
                                     shell_hot_q['gas','radial_velocity_corrected'])/dr).in_units('keV*cm**2/yr')
                hot_entropy_flux_in_q = (np.sum(shell_in_hot_q['entropy'] * \
                                    shell_in_hot_q['gas','radial_velocity_corrected']) \
                                    /dr).in_units('keV*cm**2/yr')
                hot_entropy_flux_out_q = (np.sum(shell_out_hot_q['entropy'] * \
                                     shell_out_hot_q['gas','radial_velocity_corrected']) \
                                     /dr).in_units('keV*cm**2/yr')

                # Add everything to the table
                data_q.add_row([zsnap, q+1, r, net_mass_flux_q, net_metal_flux_q, mass_flux_in_q, mass_flux_out_q, \
                              metal_flux_in_q, metal_flux_out_q, net_cold_mass_flux_q, cold_mass_flux_in_q, \
                              cold_mass_flux_out_q, net_cool_mass_flux_q, cool_mass_flux_in_q, \
                              cool_mass_flux_out_q, net_warm_mass_flux_q, warm_mass_flux_in_q, \
                              warm_mass_flux_out_q, net_hot_mass_flux_q, hot_mass_flux_in_q, \
                              hot_mass_flux_out_q, net_cold_metal_flux_q, cold_metal_flux_in_q, \
                              cold_metal_flux_out_q, net_cool_metal_flux_q, cool_metal_flux_in_q, \
                              cool_metal_flux_out_q, net_warm_metal_flux_q, warm_metal_flux_in_q, \
                              warm_metal_flux_out_q, net_hot_metal_flux_q, hot_metal_flux_in_q, \
                              hot_metal_flux_out_q, net_kinetic_energy_flux_q, \
                              net_thermal_energy_flux_q, net_entropy_flux_q, \
                              kinetic_energy_flux_in_q, kinetic_energy_flux_out_q, \
                              thermal_energy_flux_in_q, thermal_energy_flux_out_q, \
                              entropy_flux_in_q, entropy_flux_out_q, net_cold_kinetic_energy_flux_q, \
                              cold_kinetic_energy_flux_in_q, cold_kinetic_energy_flux_out_q, \
                              net_cool_kinetic_energy_flux_q, cool_kinetic_energy_flux_in_q, \
                              cool_kinetic_energy_flux_out_q, net_warm_kinetic_energy_flux_q, \
                              warm_kinetic_energy_flux_in_q, warm_kinetic_energy_flux_out_q, \
                              net_hot_kinetic_energy_flux_q, hot_kinetic_energy_flux_in_q, \
                              hot_kinetic_energy_flux_out_q, net_cold_thermal_energy_flux_q, \
                              cold_thermal_energy_flux_in_q, cold_thermal_energy_flux_out_q, \
                              net_cool_thermal_energy_flux_q, cool_thermal_energy_flux_in_q, \
                              cool_thermal_energy_flux_out_q, net_warm_thermal_energy_flux_q, \
                              warm_thermal_energy_flux_in_q, warm_thermal_energy_flux_out_q, \
                              net_hot_thermal_energy_flux_q, hot_thermal_energy_flux_in_q, \
                              hot_thermal_energy_flux_out_q, net_cold_entropy_flux_q, \
                              cold_entropy_flux_in_q, cold_entropy_flux_out_q, \
                              net_cool_entropy_flux_q, cool_entropy_flux_in_q, \
                              cool_entropy_flux_out_q, net_warm_entropy_flux_q, \
                              warm_entropy_flux_in_q, warm_entropy_flux_out_q, \
                              net_hot_entropy_flux_q, hot_entropy_flux_in_q, \
                              hot_entropy_flux_out_q])
        # Add everything to the table
        data.add_row([zsnap, 0, r, net_mass_flux, net_metal_flux, mass_flux_in, \
                      mass_flux_out, metal_flux_in, metal_flux_out, net_cold_mass_flux, \
                      cold_mass_flux_in, cold_mass_flux_out, net_cool_mass_flux, \
                      cool_mass_flux_in, cool_mass_flux_out, net_warm_mass_flux, \
                      warm_mass_flux_in, warm_mass_flux_out, net_hot_mass_flux, \
                      hot_mass_flux_in, hot_mass_flux_out, net_cold_metal_flux, \
                      cold_metal_flux_in, cold_metal_flux_out, net_cool_metal_flux, \
                      cool_metal_flux_in, cool_metal_flux_out, net_warm_metal_flux, \
                      warm_metal_flux_in, warm_metal_flux_out, net_hot_metal_flux, \
                      hot_metal_flux_in, hot_metal_flux_out, net_kinetic_energy_flux, \
                      net_thermal_energy_flux, net_entropy_flux, \
                      kinetic_energy_flux_in, kinetic_energy_flux_out, \
                      thermal_energy_flux_in, thermal_energy_flux_out, \
                      entropy_flux_in, entropy_flux_out, net_cold_kinetic_energy_flux, \
                      cold_kinetic_energy_flux_in, cold_kinetic_energy_flux_out, \
                      net_cool_kinetic_energy_flux, cool_kinetic_energy_flux_in, \
                      cool_kinetic_energy_flux_out, net_warm_kinetic_energy_flux, \
                      warm_kinetic_energy_flux_in, warm_kinetic_energy_flux_out, \
                      net_hot_kinetic_energy_flux, hot_kinetic_energy_flux_in, \
                      hot_kinetic_energy_flux_out, net_cold_thermal_energy_flux, \
                      cold_thermal_energy_flux_in, cold_thermal_energy_flux_out, \
                      net_cool_thermal_energy_flux, cool_thermal_energy_flux_in, \
                      cool_thermal_energy_flux_out, net_warm_thermal_energy_flux, \
                      warm_thermal_energy_flux_in, warm_thermal_energy_flux_out, \
                      net_hot_thermal_energy_flux, hot_thermal_energy_flux_in, \
                      hot_thermal_energy_flux_out, net_cold_entropy_flux, \
                      cold_entropy_flux_in, cold_entropy_flux_out, \
                      net_cool_entropy_flux, cool_entropy_flux_in, \
                      cool_entropy_flux_out, net_warm_entropy_flux, \
                      warm_entropy_flux_in, warm_entropy_flux_out, \
                      net_hot_entropy_flux, hot_entropy_flux_in, \
                      hot_entropy_flux_out])

    # Save to file
    data = set_table_units(data)
    data.write(tablename + '.hdf5', path='all_data', serialize_meta=True, overwrite=True)

    if (quadrants):
        data_q = set_table_units(data_q)
        data_q.write(tablename + '_q.hdf5', path='all_data', serialize_meta=True, overwrite=True)

    return "Fluxes have been calculated for snapshot" + snap + "!"

def calc_fluxes_new(ds, snap, zsnap, refine_width_kpc, tablename, dt=YTArray([5.38e6], 'yr')[0]):
    '''This function calculates the fluxes through spherical surfaces at a variety of radii
    using the dataset stored in 'ds', which is from time snapshot 'snap', has redshift
    'zsnap', and has width of the refine box in kpc 'refine_width_kpc', and stores the fluxes
    in 'tablename'. 'dt' is the time between snapshots and defaults to that for the DD outputs.

    This function differs from the other one because it calculates the flux as the sum
    of all cells whose velocity and distance from the surface of interest indicate that the gas
    contained in that cell will be displaced across the surface of interest by the next timestep.
    That is, the properties of a cell contribute to the flux if it is no further from the surface of
    interest than v*dt where v is the cell's velocity normal to the surface and dt is the time
    between snapshots, which is dt = 5.38e6 yrs for the DD outputs.'''

    # Set up table of everything we want
    # NOTE: Make sure table units are updated when things are added to this table!
    data_short = Table(names=('redshift', 'quadrant', 'radius', 'net_mass_flux', 'net_metal_flux', \
                        'mass_flux_in', 'mass_flux_out', 'metal_flux_in', 'metal_flux_out', \
                        'net_cold_mass_flux', 'cold_mass_flux_in', 'cold_mass_flux_out', \
                        'net_cool_mass_flux', 'cool_mass_flux_in', 'cool_mass_flux_out', \
                        'net_warm_mass_flux', 'warm_mass_flux_in', 'warm_mass_flux_out', \
                        'net_hot_mass_flux', 'hot_mass_flux_in', 'hot_mass_flux_out', \
                        'net_cold_metal_flux', 'cold_metal_flux_in', 'cold_metal_flux_out', \
                        'net_cool_metal_flux', 'cool_metal_flux_in', 'cool_metal_flux_out', \
                        'net_warm_metal_flux', 'warm_metal_flux_in', 'warm_metal_flux_out', \
                        'net_hot_metal_flux', 'hot_metal_flux_in', 'hot_metal_flux_out'), \
                 dtype=('f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', \
                        'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', \
                        'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8'))

    # Define the radii of the spherical shells where we want to calculate fluxes
    radii = 0.5*refine_width_kpc * np.arange(0.1, 0.9, 0.01)

    # Initialize fields we need
    ds.dt = dt
    ds.surface = radii[0]
    buffer = YTArray([10.], 'kpc')[0]
    ds.add_field(('gas', 'vel_time'), function=vel_time, units='kpc', \
                 take_log=False, force_override=True, sampling_type='cell')
    ds.add_field(('gas', 'dist'), function=dist, units='kpc', \
                 take_log=False, force_override=True, sampling_type='cell')

    # Loop over radii
    for i in range(len(radii)-1):
        r = radii[i]
        ds.surface = r

        if (i%10==0): print("Computing radius " + str(i) + "/" + str(len(radii)-1) + \
                            " for snapshot " + snap)

        # Make the shells for computing
        if (r >= buffer):
            inner_shell = ds.sphere(ds.halo_center_kpc, r) - ds.sphere(ds.halo_center_kpc, r-buffer)
        else:
            inner_shell = ds.sphere(ds.halo_center_kpc, r)
        outer_shell = ds.sphere(ds.halo_center_kpc, r+buffer) - ds.sphere(ds.halo_center_kpc, r)

        # Filter the shells on radial velocity, distance, and time to find cells that actually
        # contribute to fluxes
        inner_cont = inner_shell.cut_region("(obj['radial_velocity_corrected'] > 0)" + \
                                            " & (obj['vel_time'] >= obj['dist'])")
        outer_cont = outer_shell.cut_region("(obj['radial_velocity_corrected'] < 0)" + \
                                            " & (obj['vel_time'] >= obj['dist'])")

        # Filter on temperature
        inner_cont_cold = inner_cont.cut_region("obj['temperature'] <= 10**4")
        outer_cont_cold = outer_cont.cut_region("obj['temperature'] <= 10**4")
        inner_cont_cool = inner_cont.cut_region("(obj['temperature'] > 10**4) &" + \
                                                " (obj['temperature'] <= 10**5)")
        outer_cont_cool = outer_cont.cut_region("(obj['temperature'] > 10**4) &" + \
                                                " (obj['temperature'] <= 10**5)")
        inner_cont_warm = inner_cont.cut_region("(obj['temperature'] > 10**5) &" + \
                                                " (obj['temperature'] <= 10**6)")
        outer_cont_warm = outer_cont.cut_region("(obj['temperature'] > 10**5) &" + \
                                                " (obj['temperature'] <= 10**6)")
        inner_cont_hot = inner_cont.cut_region("obj['temperature'] > 10**6")
        outer_cont_hot = outer_cont.cut_region("obj['temperature'] > 10**6")

        # Sum over the contributing cells and divide by dt to get flux
        mass_flux_in = (np.sum(outer_cont['cell_mass'])/ds.dt).in_units('Msun/yr')
        mass_flux_out = (np.sum(inner_cont['cell_mass'])/ds.dt).in_units('Msun/yr')
        net_mass_flux = mass_flux_out - mass_flux_in

        metal_flux_in = (np.sum(outer_cont['metal_mass'])/ds.dt).in_units('Msun/yr')
        metal_flux_out = (np.sum(inner_cont['metal_mass'])/ds.dt).in_units('Msun/yr')
        net_metal_flux = metal_flux_out - metal_flux_in

        cold_mass_flux_in = (np.sum(outer_cont_cold['cell_mass'])/ds.dt).in_units('Msun/yr')
        cold_mass_flux_out = (np.sum(inner_cont_cold['cell_mass'])/ds.dt).in_units('Msun/yr')
        net_cold_mass_flux = cold_mass_flux_out - cold_mass_flux_in

        cool_mass_flux_in = (np.sum(outer_cont_cool['cell_mass'])/ds.dt).in_units('Msun/yr')
        cool_mass_flux_out = (np.sum(inner_cont_cool['cell_mass'])/ds.dt).in_units('Msun/yr')
        net_cool_mass_flux = cool_mass_flux_out - cool_mass_flux_in

        warm_mass_flux_in = (np.sum(outer_cont_warm['cell_mass'])/ds.dt).in_units('Msun/yr')
        warm_mass_flux_out = (np.sum(inner_cont_warm['cell_mass'])/ds.dt).in_units('Msun/yr')
        net_warm_mass_flux = warm_mass_flux_out - warm_mass_flux_in

        hot_mass_flux_in = (np.sum(outer_cont_hot['cell_mass'])/ds.dt).in_units('Msun/yr')
        hot_mass_flux_out = (np.sum(inner_cont_hot['cell_mass'])/ds.dt).in_units('Msun/yr')
        net_hot_mass_flux = hot_mass_flux_out - hot_mass_flux_in

        cold_metal_flux_in = (np.sum(outer_cont_cold['metal_mass'])/ds.dt).in_units('Msun/yr')
        cold_metal_flux_out = (np.sum(inner_cont_cold['metal_mass'])/ds.dt).in_units('Msun/yr')
        net_cold_metal_flux = cold_metal_flux_out - cold_metal_flux_in

        cool_metal_flux_in = (np.sum(outer_cont_cool['metal_mass'])/ds.dt).in_units('Msun/yr')
        cool_metal_flux_out = (np.sum(inner_cont_cool['metal_mass'])/ds.dt).in_units('Msun/yr')
        net_cool_metal_flux = cool_metal_flux_out - cool_metal_flux_in

        warm_metal_flux_in = (np.sum(outer_cont_warm['metal_mass'])/ds.dt).in_units('Msun/yr')
        warm_metal_flux_out = (np.sum(inner_cont_warm['metal_mass'])/ds.dt).in_units('Msun/yr')
        net_warm_metal_flux = warm_metal_flux_out - warm_metal_flux_in

        hot_metal_flux_in = (np.sum(outer_cont_hot['metal_mass'])/ds.dt).in_units('Msun/yr')
        hot_metal_flux_out = (np.sum(inner_cont_hot['metal_mass'])/ds.dt).in_units('Msun/yr')
        net_hot_metal_flux = hot_metal_flux_out - hot_metal_flux_in

        # Add everything to the table
        data_short.add_row([zsnap, 0, r, net_mass_flux, net_metal_flux, mass_flux_in, \
                      mass_flux_out, metal_flux_in, metal_flux_out, net_cold_mass_flux, \
                      cold_mass_flux_in, cold_mass_flux_out, net_cool_mass_flux, \
                      cool_mass_flux_in, cool_mass_flux_out, net_warm_mass_flux, \
                      warm_mass_flux_in, warm_mass_flux_out, net_hot_mass_flux, \
                      hot_mass_flux_in, hot_mass_flux_out, net_cold_metal_flux, \
                      cold_metal_flux_in, cold_metal_flux_out, net_cool_metal_flux, \
                      cool_metal_flux_in, cool_metal_flux_out, net_warm_metal_flux, \
                      warm_metal_flux_in, warm_metal_flux_out, net_hot_metal_flux, \
                      hot_metal_flux_in, hot_metal_flux_out])

    table_units_short = {'redshift':None,'quadrant':None,'radius':'kpc','net_mass_flux':'Msun/yr', \
             'net_metal_flux':'Msun/yr', 'mass_flux_in'  :'Msun/yr','mass_flux_out':'Msun/yr', \
             'metal_flux_in' :'Msun/yr', 'metal_flux_out':'Msun/yr',\
             'net_cold_mass_flux':'Msun/yr', 'cold_mass_flux_in':'Msun/yr', \
             'cold_mass_flux_out':'Msun/yr', 'net_cool_mass_flux':'Msun/yr', \
             'cool_mass_flux_in':'Msun/yr', 'cool_mass_flux_out':'Msun/yr', \
             'net_warm_mass_flux':'Msun/yr', 'warm_mass_flux_in':'Msun/yr', \
             'warm_mass_flux_out':'Msun/yr', 'net_hot_mass_flux' :'Msun/yr', \
             'hot_mass_flux_in' :'Msun/yr', 'hot_mass_flux_out' :'Msun/yr', \
             'net_cold_metal_flux':'Msun/yr', 'cold_metal_flux_in':'Msun/yr', \
             'cold_metal_flux_out':'Msun/yr', 'net_cool_metal_flux':'Msun/yr', \
             'cool_metal_flux_in':'Msun/yr', 'cool_metal_flux_out':'Msun/yr', \
             'net_warm_metal_flux':'Msun/yr', 'warm_metal_flux_in':'Msun/yr', \
             'warm_metal_flux_out':'Msun/yr', 'net_hot_metal_flux' :'Msun/yr', \
             'hot_metal_flux_in' :'Msun/yr', 'hot_metal_flux_out' :'Msun/yr'}
    for key in data_short.keys():
        data_short[key].unit = table_units_short[key]

    # Save to file
    data_short.write(tablename + '_new_short.hdf5', path='all_data', serialize_meta=True, overwrite=True)

    return "Fluxes have been calculated for snapshot " + snap + "!"


def load_and_calculate(foggie_dir, run_dir, track, halo_c_v, snap, tablename, quadrants):
    '''This function loads a specified snapshot 'snap' located in the 'run_dir' within the
    'foggie_dir', the halo track 'track', the name of the table to output, and a boolean
    'quadrants' that specifies whether or not to compute in quadrants vs. the whole domain, then
    does the calculation on the loaded snapshot.'''

    # Load snapshot
    print ('Opening snapshot ' + snap)
    ds = yt.load(foggie_dir + run_dir + snap + '/' + snap)

    # Get the refined box in physical units
    zsnap = ds.get_parameter('CosmologyCurrentRedshift')
    proper_box_size = get_proper_box_size(ds)
    refine_box, refine_box_center, refine_width_code = get_refine_box(ds, zsnap, track)
    refine_width = refine_width_code * proper_box_size
    refine_width_kpc = YTArray([refine_width], 'kpc')

    # This code is for the previous way of computing the halo center and velocity, before
    # the files of this info were made
    '''
    # Get halo center
    halo_center, halo_velocity = get_halo_center(ds, refine_box_center)

    # Define the halo center in kpc and the halo velocity in km/s
    halo_center_kpc = YTArray(np.array(halo_center)*proper_box_size, 'kpc')
    halo_velocity_kms = YTArray(halo_velocity).in_units('km/s')
    '''

    # Here's the new way to get halo center and velocity now that the data files are made
    halo_ind = np.where(halo_c_v['col3']==snap)[0][0]
    halo_center_kpc = YTArray([float(halo_c_v['col4'][halo_ind]), \
                              float(halo_c_v['col5'][halo_ind]), \
                              float(halo_c_v['col6'][halo_ind])], 'kpc')
    halo_velocity_kms = YTArray([float(halo_c_v['col7'][halo_ind]), \
                                float(halo_c_v['col8'][halo_ind]), \
                                float(halo_c_v['col9'][halo_ind])], 'km/s')
    sp = ds.sphere(halo_center_kpc, 0.05*refine_width_kpc[0])
    bulk_velocity = sp.quantities['BulkVelocity']().in_units('km/s')
    ds.halo_center_kpc = halo_center_kpc
    ds.halo_velocity_kms = bulk_velocity

    # Add the fields we want
    ds.add_field(('gas','vx_corrected'), function=vx_corrected, units='km/s', take_log=False, \
                 sampling_type='cell')
    ds.add_field(('gas', 'vy_corrected'), function=vy_corrected, units='km/s', take_log=False, \
                 sampling_type='cell')
    ds.add_field(('gas', 'vz_corrected'), function=vz_corrected, units='km/s', take_log=False, \
                 sampling_type='cell')
    ds.add_field(('gas', 'radius_corrected'), function=radius_corrected, units='kpc', \
                 take_log=False, force_override=True, sampling_type='cell')
    ds.add_field(('gas', 'theta_pos'), function=theta_pos, units=None, take_log=False, \
                 sampling_type='cell')
    ds.add_field(('gas', 'phi_pos'), function=phi_pos, units=None, take_log=False, \
                 sampling_type='cell')
    ds.add_field(('gas', 'radial_velocity_corrected'), function=radial_velocity_corrected, \
                 units='km/s', take_log=False, force_override=True, sampling_type='cell')
    ds.add_field(('gas', 'kinetic_energy_corrected'), function=kinetic_energy_corrected, \
                 units='erg', take_log=True, force_override=True, sampling_type='cell')

    # Do the actual calculation
    message = calc_fluxes_new(ds, snap, zsnap, refine_width_kpc, tablename)
    print(message)
    print(str(datetime.datetime.now()))


if __name__ == "__main__":
    args = parse_args()
    print(args.halo)
    print(args.run)
    print(args.system)
    foggie_dir, output_dir, run_dir, trackname, haloname, spectra_dir = get_run_loc_etc(args)
    if (args.system=='pleiades_cassi'): code_path = '/home5/clochhaa/FOGGIE/foggie/foggie/'
    elif (args.system=='cassiopeia'):
        code_path = '/Users/clochhaas/Documents/Research/FOGGIE/Analysis_Code/foggie/foggie/'
    track_dir = code_path + 'halo_infos/00' + args.halo + '/' + args.run + '/'
    if ('/astro/simulations/' in foggie_dir):
        run_dir = 'halo_00' + args.halo + '/nref11n/' + args.run + '/'

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
    prefix = output_dir + 'fluxes_halo_00' + args.halo + '/' + args.run + '/'
    if not (os.path.exists(prefix)): os.system('mkdir -p ' + prefix)

    # Load halo track
    print('foggie_dir: ', foggie_dir)
    print('Opening track: ' + trackname)
    track = Table.read(trackname, format='ascii')
    track.sort('col1')

    # Here's the new way of finding the halo
    # Load halo center and velocity
    #halo_c_v = Table.read(track_dir + '00' args.halo + '/' + args.run + '/halo_c_v', format='ascii')
    halo_c_v = Table.read(track_dir + 'halo_c_v', format='ascii')

    # Loop over outputs, for either single-processor or parallel processor computing
    if (args.nproc==1):
        for i in range(len(outs)):
            snap = outs[i]
            # Make the output table name for this snapshot
            tablename = prefix + snap + '_fluxes'
            # Do the actual calculation
            load_and_calculate(foggie_dir, run_dir, track, halo_c_v, snap, tablename, args.quadrants)
    else:
        # Split into a number of groupings equal to the number of processors
        # and run one process per processor
        for i in range(len(outs)//args.nproc):
            threads = []
            for j in range(args.nproc):
                snap = outs[args.nproc*i+j]
                tablename = prefix + snap + '_fluxes'
                threads.append(mp.Process(target=load_and_calculate, \
			       args=(foggie_dir, run_dir, track, halo_c_v, snap, tablename, args.quadrants)))
            for t in threads:
                t.start()
            for t in threads:
                t.join()
        # For any leftover snapshots, run one per processor
        threads = []
        for j in range(len(outs)%args.nproc):
            snap = outs[-(j+1)]
            tablename = prefix + snap + '_fluxes'
            threads.append(mp.Process(target=load_and_calculate, \
			   args=(foggie_dir, run_dir, track, halo_c_v, snap, tablename, args.quadrants)))
        for t in threads:
            t.start()
        for t in threads:
            t.join()

    print(str(datetime.datetime.now()))
    sys.exit("All snapshots finished!")
