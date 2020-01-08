"""
Filename: flux_tracking.py
Author: Cassi
Date created: 9-27-19
Date last modified: 12-27-19
This file takes command line arguments and computes fluxes of things through
spherical shells.

Dependencies:
utils/consistency.py
utils/get_refine_box.py
utils/get_halo_center.py
utils/get_proper_box_size.py
utils/get_run_loc_etc.py
utils/yt_fields.py
utils/foggie_load.py
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
import multiprocessing as multi
import datetime
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

def set_table_units(table):
    '''Sets the units for the table. Note this needs to be updated whenever something is added to
    the table. Returns the table.'''

    table_units = {'redshift':None,'quadrant':None,'radius':'kpc','inner_radius':'kpc','outer_radius':'kpc', \
             'net_mass_flux':'Msun/yr', 'net_metal_flux':'Msun/yr', \
             'mass_flux_in':'Msun/yr', 'mass_flux_out':'Msun/yr', \
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
             'net_potential_energy_flux':'erg/yr', 'net_radiative_energy_flux':'erg/yr', \
             'net_entropy_flux':'cm**2*keV/yr', \
             'kinetic_energy_flux_in':'erg/yr', 'kinetic_energy_flux_out':'erg/yr', \
             'thermal_energy_flux_in':'erg/yr', 'thermal_energy_flux_out':'erg/yr', \
             'potential_energy_flux_in':'erg/yr', 'potential_energy_flux_out':'erg/yr', \
             'radiative_energy_flux_in':'erg/yr', 'radiative_energy_flux_out':'erg/yr', \
             'entropy_flux_in':'cm**2*keV/yr', 'entropy_flux_out':'cm**2*keV/yr', \
             'net_cold_kinetic_energy_flux':'erg/yr', 'cold_kinetic_energy_flux_in':'erg/yr', 'cold_kinetic_energy_flux_out':'erg/yr', \
             'net_cool_kinetic_energy_flux':'erg/yr', 'cool_kinetic_energy_flux_in':'erg/yr', 'cool_kinetic_energy_flux_out':'erg/yr', \
             'net_warm_kinetic_energy_flux':'erg/yr', 'warm_kinetic_energy_flux_in':'erg/yr', 'warm_kinetic_energy_flux_out':'erg/yr', \
             'net_hot_kinetic_energy_flux':'erg/yr', 'hot_kinetic_energy_flux_in':'erg/yr', 'hot_kinetic_energy_flux_out':'erg/yr', \
             'net_cold_thermal_energy_flux':'erg/yr', 'cold_thermal_energy_flux_in':'erg/yr', 'cold_thermal_energy_flux_out':'erg/yr', \
             'net_cool_thermal_energy_flux':'erg/yr', 'cool_thermal_energy_flux_in':'erg/yr', 'cool_thermal_energy_flux_out':'erg/yr', \
             'net_warm_thermal_energy_flux':'erg/yr', 'warm_thermal_energy_flux_in':'erg/yr', 'warm_thermal_energy_flux_out':'erg/yr', \
             'net_hot_thermal_energy_flux':'erg/yr', 'hot_thermal_energy_flux_in':'erg/yr', 'hot_thermal_energy_flux_out':'erg/yr', \
             'net_cold_potential_energy_flux':'erg/yr', 'cold_potential_energy_flux_in':'erg/yr', 'cold_potential_energy_flux_out':'erg/yr', \
             'net_cool_potential_energy_flux':'erg/yr', 'cool_potential_energy_flux_in':'erg/yr', 'cool_potential_energy_flux_out':'erg/yr', \
             'net_warm_potential_energy_flux':'erg/yr', 'warm_potential_energy_flux_in':'erg/yr', 'warm_potential_energy_flux_out':'erg/yr', \
             'net_hot_potential_energy_flux':'erg/yr', 'hot_potential_energy_flux_in':'erg/yr', 'hot_potential_energy_flux_out':'erg/yr', \
             'net_cold_radiative_energy_flux':'erg/yr', 'cold_radiative_energy_flux_in':'erg/yr', 'cold_radiative_energy_flux_out':'erg/yr', \
             'net_cool_radiative_energy_flux':'erg/yr', 'cool_radiative_energy_flux_in':'erg/yr', 'cool_radiative_energy_flux_out':'erg/yr', \
             'net_warm_radiative_energy_flux':'erg/yr', 'warm_radiative_energy_flux_in':'erg/yr', 'warm_radiative_energy_flux_out':'erg/yr', \
             'net_hot_radiative_energy_flux':'erg/yr', 'hot_radiative_energy_flux_in':'erg/yr', 'hot_radiative_energy_flux_out':'erg/yr', \
             'net_cold_entropy_flux':'cm**2*keV/yr', 'cold_entropy_flux_in':'cm**2*keV/yr', 'cold_entropy_flux_out':'cm**2*keV/yr', \
             'net_cool_entropy_flux':'cm**2*keV/yr', 'cool_entropy_flux_in':'cm**2*keV/yr', 'cool_entropy_flux_out':'cm**2*keV/yr', \
             'net_warm_entropy_flux':'cm**2*keV/yr', 'warm_entropy_flux_in':'cm**2*keV/yr', 'warm_entropy_flux_out':'cm**2*keV/yr', \
             'net_hot_entropy_flux':'cm**2*keV/yr', 'hot_entropy_flux_in':'cm**2*keV/yr', 'hot_entropy_flux_out':'cm**2*keV/yr'}
    for key in table.keys():
        table[key].unit = table_units[key]
    return table

def calc_fluxes(ds, snap, zsnap, refine_width_kpc, tablename, **kwargs):
    """Computes the flux through spherical shells centered on the halo center.
    Takes the dataset for the snapshot 'ds', the name of the snapshot 'snap', the redshift of the
    snapshot 'zsnap', and the width of the refine box in kpc 'refine_width_kpc'
    and does the calculation, then writes a hdf5 table out to 'tablename'.

    Optional arguments:
    quadrants = True will calculate the flux shells within quadrants rather than the whole domain,
        default is False. If this is selected, a second table will be written with '_q' appended
        to 'tablename'. Note this functionality hasn't been updated in a while.
    """

    quadrants = kwargs.get('quadrants', False)
    Menc_func = kwargs.get('Menc_func', False)

    G = ds.arr(6.673e-8, 'cm**3/s**2/g')
    halo_center_kpc = ds.halo_center_kpc

    cmtopc = 3.086e18
    stoyr = 3.154e7
    gtoMsun = 1.989e33

    # Set up table of everything we want
    # NOTE: Make sure table units are updated when things are added to this table!
    data = Table(names=('redshift', 'quadrant', 'radius', \
                        'net_mass_flux', 'net_metal_flux', \
                        'mass_flux_in', 'mass_flux_out', 'metal_flux_in', 'metal_flux_out', \
                        'net_cold_mass_flux', 'cold_mass_flux_in', 'cold_mass_flux_out', \
                        'net_cool_mass_flux', 'cool_mass_flux_in', 'cool_mass_flux_out', \
                        'net_warm_mass_flux', 'warm_mass_flux_in', 'warm_mass_flux_out', \
                        'net_hot_mass_flux', 'hot_mass_flux_in', 'hot_mass_flux_out', \
                        'net_cold_metal_flux', 'cold_metal_flux_in', 'cold_metal_flux_out', \
                        'net_cool_metal_flux', 'cool_metal_flux_in', 'cool_metal_flux_out', \
                        'net_warm_metal_flux', 'warm_metal_flux_in', 'warm_metal_flux_out', \
                        'net_hot_metal_flux', 'hot_metal_flux_in', 'hot_metal_flux_out', \
                        'net_kinetic_energy_flux', 'net_thermal_energy_flux', \
                        'net_potential_energy_flux', 'net_radiative_energy_flux', \
                        'net_entropy_flux', \
                        'kinetic_energy_flux_in', 'kinetic_energy_flux_out', \
                        'thermal_energy_flux_in', 'thermal_energy_flux_out', \
                        'potential_energy_flux_in', 'potential_energy_flux_out', \
                        'radiative_energy_flux_in', 'radiative_energy_flux_out', \
                        'entropy_flux_in', 'entropy_flux_out', \
                        'net_cold_kinetic_energy_flux', 'cold_kinetic_energy_flux_in', 'cold_kinetic_energy_flux_out', \
                        'net_cool_kinetic_energy_flux', 'cool_kinetic_energy_flux_in', 'cool_kinetic_energy_flux_out', \
                        'net_warm_kinetic_energy_flux', 'warm_kinetic_energy_flux_in', 'warm_kinetic_energy_flux_out', \
                        'net_hot_kinetic_energy_flux', 'hot_kinetic_energy_flux_in', 'hot_kinetic_energy_flux_out', \
                        'net_cold_thermal_energy_flux', 'cold_thermal_energy_flux_in', 'cold_thermal_energy_flux_out', \
                        'net_cool_thermal_energy_flux', 'cool_thermal_energy_flux_in', 'cool_thermal_energy_flux_out', \
                        'net_warm_thermal_energy_flux', 'warm_thermal_energy_flux_in', 'warm_thermal_energy_flux_out', \
                        'net_hot_thermal_energy_flux', 'hot_thermal_energy_flux_in', 'hot_thermal_energy_flux_out', \
                        'net_cold_potential_energy_flux', 'cold_potential_energy_flux_in', 'cold_potential_energy_flux_out', \
                        'net_cool_potential_energy_flux', 'cool_potential_energy_flux_in', 'cool_potential_energy_flux_out', \
                        'net_warm_potential_energy_flux', 'warm_potential_energy_flux_in', 'warm_potential_energy_flux_out', \
                        'net_hot_potential_energy_flux', 'hot_potential_energy_flux_in', 'hot_potential_energy_flux_out', \
                        'net_cold_radiative_energy_flux', 'cold_radiative_energy_flux_in', 'cold_radiative_energy_flux_out', \
                        'net_cool_radiative_energy_flux', 'cool_radiative_energy_flux_in', 'cool_radiative_energy_flux_out', \
                        'net_warm_radiative_energy_flux', 'warm_radiative_energy_flux_in', 'warm_radiative_energy_flux_out', \
                        'net_hot_radiative_energy_flux', 'hot_radiative_energy_flux_in', 'hot_radiative_energy_flux_out', \
                        'net_cold_entropy_flux', 'cold_entropy_flux_in', 'cold_entropy_flux_out', \
                        'net_cool_entropy_flux', 'cool_entropy_flux_in', 'cool_entropy_flux_out', \
                        'net_warm_entropy_flux', 'warm_entropy_flux_in', 'warm_entropy_flux_out', \
                        'net_hot_entropy_flux', 'hot_entropy_flux_in', 'hot_entropy_flux_out'), \
                 dtype=('f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', \
                        'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', \
                        'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', \
                        'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', \
                        'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', \
                        'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', \
                        'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', \
                        'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', \
                        'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8'))

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

    # Load arrays of all fields we need
    print('Loading field arrays')
    sphere = ds.sphere(halo_center_kpc, radii[-1])

    mass = sphere['gas','cell_mass'].in_units('Msun').v
    rad_vel = sphere['gas','radial_velocity_corrected'].in_units('km/s').v
    radius = sphere['gas','radius_corrected'].in_units('kpc').v
    metal_mass = sphere['gas','metal_mass'].in_units('Msun').v
    temperature = sphere['gas','temperature'].in_units('K').v
    kinetic_energy = sphere['gas','kinetic_energy_corrected'].in_units('erg').v
    thermal_energy = sphere['gas','thermal_energy'].in_units('erg/g').v
    cooling_time = sphere['gas','cooling_time'].in_units('yr').v
    entropy = sphere['gas','entropy'].in_units('keV*cm**2').v

    # Loop over radii
    for i in range(len(radii)-1):
        r_low = radii[i]        # kpc
        r_high = radii[i+1]     # kpc
        dr = r_high - r_low     # kpc
        r = (r_low + r_high)/2. # kpc

        if (i%10==0): print("Computing radius " + str(i) + "/" + str(len(radii)-1) + \
                            " for snapshot " + snap)

        # Cut the data for this shell
        mass_shell = mass[(radius >= r_low.v) & (radius < r_high.v)]
        rad_vel_shell = rad_vel[(radius >= r_low.v) & (radius < r_high.v)]
        radius_shell = radius[(radius >= r_low.v) & (radius < r_high.v)]
        metal_mass_shell = metal_mass[(radius >= r_low.v) & (radius < r_high.v)]
        temperature_shell = temperature[(radius >= r_low.v) & (radius < r_high.v)]
        kinetic_energy_shell = kinetic_energy[(radius >= r_low.v) & (radius < r_high.v)]
        thermal_energy_shell = thermal_energy[(radius >= r_low.v) & (radius < r_high.v)]
        cooling_time_shell = cooling_time[(radius >= r_low.v) & (radius < r_high.v)]
        entropy_shell = entropy[(radius >= r_low.v) & (radius < r_high.v)]

        # Cut the data on temperature and radial velocity for in and out fluxes
        # For each field, it is a nested list where the top index is 0 through 4 for temperature phases
        # (all, cold, cool, warm, hot) and the second index is 0 through 2 for radial velocity (all, in, out)
        mass_cut = []
        rad_vel_cut = []
        radius_cut = []
        metal_mass_cut = []
        temperature_cut = []
        kinetic_energy_cut = []
        thermal_energy_cut = []
        cooling_time_cut = []
        entropy_cut = []
        for j in range(5):
            mass_cut.append([])
            rad_vel_cut.append([])
            radius_cut.append([])
            metal_mass_cut.append([])
            temperature_cut.append([])
            kinetic_energy_cut.append([])
            thermal_energy_cut.append([])
            cooling_time_cut.append([])
            entropy_cut.append([])
            if (j==0):
                t_low = 0.
                t_high = 10**12.
            if (j==1):
                t_low = 0.
                t_high = 10**4.
            if (j==2):
                t_low = 10**4.
                t_high = 10**5.
            if (j==3):
                t_low = 10**5.
                t_high = 10**6.
            if (j==4):
                t_low = 10**6.
                t_high = 10**12.
            mass_cut[j].append(mass_shell[(temperature_shell > t_low) & (temperature_shell < t_high)])
            rad_vel_cut[j].append(rad_vel_shell[(temperature_shell > t_low) & (temperature_shell < t_high)])
            radius_cut[j].append(radius_shell[(temperature_shell > t_low) & (temperature_shell < t_high)])
            metal_mass_cut[j].append(metal_mass_shell[(temperature_shell > t_low) & (temperature_shell < t_high)])
            temperature_cut[j].append(temperature_shell[(temperature_shell > t_low) & (temperature_shell < t_high)])
            kinetic_energy_cut[j].append(kinetic_energy_shell[(temperature_shell > t_low) & (temperature_shell < t_high)])
            thermal_energy_cut[j].append(thermal_energy_shell[(temperature_shell > t_low) & (temperature_shell < t_high)])
            cooling_time_cut[j].append(cooling_time_shell[(temperature_shell > t_low) & (temperature_shell < t_high)])
            entropy_cut[j].append(entropy_shell[(temperature_shell > t_low) & (temperature_shell < t_high)])
            for k in range(2):
                if (k==0): fac = -1.
                if (k==1): fac = 1.
                mass_cut[j].append(mass_cut[j][0][(fac*rad_vel_cut[j][0] > 0.)])
                rad_vel_cut[j].append(rad_vel_cut[j][0][(fac*rad_vel_cut[j][0] > 0.)])
                radius_cut[j].append(radius_cut[j][0][(fac*rad_vel_cut[j][0] > 0.)])
                metal_mass_cut[j].append(metal_mass_cut[j][0][(fac*rad_vel_cut[j][0] > 0.)])
                temperature_cut[j].append(temperature_cut[j][0][(fac*rad_vel_cut[j][0] > 0.)])
                kinetic_energy_cut[j].append(kinetic_energy_cut[j][0][(fac*rad_vel_cut[j][0] > 0.)])
                thermal_energy_cut[j].append(thermal_energy_cut[j][0][(fac*rad_vel_cut[j][0] > 0.)])
                cooling_time_cut[j].append(cooling_time_cut[j][0][(fac*rad_vel_cut[j][0] > 0.)])
                entropy_cut[j].append(entropy_cut[j][0][(fac*rad_vel_cut[j][0] > 0.)])

        # Compute fluxes
        # For each type of flux, it is a nested list where the top index goes from 0 to 4 and is
        # the phase of gas (all, cold, cool, warm, hot) and the second index goes from 0 to 2 and is
        # the radial velocity (all, in, out).
        # Ex. the net flux is flux[0][0], the net inward flux is flux[0][1], the outward warm flux is flux[3][2]
        mass_flux = []
        metal_flux = []
        kinetic_energy_flux = []
        thermal_energy_flux = []
        potential_energy_flux = []
        radiative_energy_flux = []
        entropy_flux = []

        for j in range(5):
            mass_flux.append([])
            metal_flux.append([])
            kinetic_energy_flux.append([])
            thermal_energy_flux.append([])
            potential_energy_flux.append([])
            radiative_energy_flux.append([])
            entropy_flux.append([])
            for k in range(3):
                mass_flux[j].append(np.sum(mass_cut[j][k]*rad_vel_cut[j][k]*1.e5*stoyr)/(dr*1000.*cmtopc))
                metal_flux[j].append(np.sum(metal_mass_cut[j][k]*rad_vel_cut[j][k]*1.e5*stoyr)/(dr*1000.*cmtopc))
                kinetic_energy_flux[j].append(np.sum(kinetic_energy_cut[j][k]*rad_vel_cut[j][k]*1.e5*stoyr)/(dr*1000.*cmtopc))
                thermal_energy_flux[j].append(np.sum(thermal_energy_cut[j][k] * mass_cut[j][k]*gtoMsun * \
                                              rad_vel_cut[j][k]*1.e5*stoyr)/(dr*1000.*cmtopc))
                potential_energy_flux[j].append(np.sum(G * mass_cut[j][k]*gtoMsun * Menc_func(radius_cut[j][k])*gtoMsun / \
                                               (radius_cut[j][k]*1000.*cmtopc) * rad_vel_cut[j][k]*1.e5*stoyr)/(dr*1000.*cmtopc))
                radiative_energy_flux[j].append(np.sum(thermal_energy_cut[j][k] * mass_cut[j][k]*gtoMsun / \
                                                cooling_time_cut[j][k]))
                entropy_flux[j].append(np.sum(entropy_cut[j][k] * rad_vel_cut[j][k]*1.e5*stoyr)/(dr*1000.*cmtopc))

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

                # Calculate potential energy using mass enclosed profile


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
        data.add_row([zsnap, 0, r, mass_flux[0][0], metal_flux[0][0], \
                      mass_flux[0][1], mass_flux[0][2], metal_flux[0][1], metal_flux[0][2], \
                      mass_flux[1][0], mass_flux[1][1], mass_flux[1][2], \
                      mass_flux[2][0], mass_flux[2][1], mass_flux[2][2], \
                      mass_flux[3][0], mass_flux[3][1], mass_flux[3][2], \
                      mass_flux[4][0], mass_flux[4][1], mass_flux[4][2], \
                      metal_flux[1][0], metal_flux[1][1], metal_flux[1][2], \
                      metal_flux[2][0], metal_flux[2][1], metal_flux[2][2], \
                      metal_flux[3][0], metal_flux[3][1], metal_flux[3][2], \
                      metal_flux[4][0], metal_flux[4][1], metal_flux[4][2], \
                      kinetic_energy_flux[0][0], thermal_energy_flux[0][0], \
                      potential_energy_flux[0][0], radiative_energy_flux[0][0], \
                      entropy_flux[0][0], \
                      kinetic_energy_flux[0][1], kinetic_energy_flux[0][2], \
                      thermal_energy_flux[0][1], thermal_energy_flux[0][2], \
                      potential_energy_flux[0][1], potential_energy_flux[0][2], \
                      radiative_energy_flux[0][1], radiative_energy_flux[0][2], \
                      entropy_flux[0][1], entropy_flux[0][2], \
                      kinetic_energy_flux[1][0], kinetic_energy_flux[1][1], kinetic_energy_flux[1][2], \
                      kinetic_energy_flux[2][0], kinetic_energy_flux[2][1], kinetic_energy_flux[2][2], \
                      kinetic_energy_flux[3][0], kinetic_energy_flux[3][1], kinetic_energy_flux[3][2], \
                      kinetic_energy_flux[4][0], kinetic_energy_flux[4][1], kinetic_energy_flux[4][2], \
                      thermal_energy_flux[1][0], thermal_energy_flux[1][1], thermal_energy_flux[1][2], \
                      thermal_energy_flux[2][0], thermal_energy_flux[2][1], thermal_energy_flux[2][2], \
                      thermal_energy_flux[3][0], thermal_energy_flux[3][1], thermal_energy_flux[3][2], \
                      thermal_energy_flux[4][0], thermal_energy_flux[4][1], thermal_energy_flux[4][2], \
                      potential_energy_flux[1][0], potential_energy_flux[1][1], potential_energy_flux[1][2], \
                      potential_energy_flux[2][0], potential_energy_flux[2][1], potential_energy_flux[2][2], \
                      potential_energy_flux[3][0], potential_energy_flux[3][1], potential_energy_flux[3][2], \
                      potential_energy_flux[4][0], potential_energy_flux[4][1], potential_energy_flux[4][2], \
                      radiative_energy_flux[1][0], radiative_energy_flux[1][1], radiative_energy_flux[1][2], \
                      radiative_energy_flux[2][0], radiative_energy_flux[2][1], radiative_energy_flux[2][2], \
                      radiative_energy_flux[3][0], radiative_energy_flux[3][1], radiative_energy_flux[3][2], \
                      radiative_energy_flux[4][0], radiative_energy_flux[4][1], radiative_energy_flux[4][2], \
                      entropy_flux[1][0], entropy_flux[1][1], entropy_flux[1][2], \
                      entropy_flux[2][0], entropy_flux[2][1], entropy_flux[2][2], \
                      entropy_flux[3][0], entropy_flux[3][1], entropy_flux[3][2], \
                      entropy_flux[4][0], entropy_flux[4][1], entropy_flux[4][2]])

    # Save to file
    data = set_table_units(data)
    data.write(tablename + '.hdf5', path='all_data', serialize_meta=True, overwrite=True)

    if (quadrants):
        data_q = set_table_units(data_q)
        data_q.write(tablename + '_q.hdf5', path='all_data', serialize_meta=True, overwrite=True)

    return "Fluxes have been calculated for snapshot" + snap + "!"

def calc_fluxes_region(ds, snap, zsnap, dt, refine_width_kpc, tablename, **kwargs):
    '''This function calculates the fluxes into and out of spherical shells, with satellites removed,
    at a variety of radii and in quadrant wedges (if quadrants=True) aligned with the disk. It
    uses the dataset stored in 'ds', which is from time snapshot 'snap', has redshift
    'zsnap', and has width of the refine box in kpc 'refine_width_kpc', the time step between outputs
    'dt', and stores the fluxes in 'tablename'.

    This function differs from the other one because it calculates the flux as the sum
    of all cells whose velocity and distance from the surface of interest indicate that the gas
    contained in that cell will be displaced across the surface of interest by the next timestep.
    That is, the properties of a cell contribute to the flux if it is no further from the surface of
    interest than v*dt where v is the cell's velocity normal to the surface and dt is the time
    between snapshots, which is dt = 5.38e6 yrs for the DD outputs. It is necessary to compute the
    flux this way if satellites are to be removed because they become 'holes' in the dataset
    and fluxes into/out of those holes need to be accounted for. Additionally, this allows for tracking
    of fluxes across the 'sides' of the quadrant wedges, not just radially outward/inward.'''

    quadrants = kwargs.get('quadrants', False)
    Menc_func = kwargs.get('Menc_func', False)
    sat_file = kwargs.get('sat_file', 'sat_file')
    sat_file2 = kwargs.get('sat_file2', 'sat_file2')

    G = ds.quan(6.673e-8, 'cm**3/s**2/g').v
    halo_center_kpc = ds.halo_center_kpc

    cmtopc = 3.086e18
    stoyr = 3.154e7
    gtoMsun = 1.989e33

    # Set up table of everything we want
    # NOTE: Make sure table units are updated when things are added to this table!
    fluxes = Table(names=('redshift', 'radius', \
                        'net_mass_flux', 'net_metal_flux', \
                        'mass_flux_in', 'mass_flux_out', 'metal_flux_in', 'metal_flux_out', \
                        'net_cold_mass_flux', 'cold_mass_flux_in', 'cold_mass_flux_out', \
                        'net_cool_mass_flux', 'cool_mass_flux_in', 'cool_mass_flux_out', \
                        'net_warm_mass_flux', 'warm_mass_flux_in', 'warm_mass_flux_out', \
                        'net_hot_mass_flux', 'hot_mass_flux_in', 'hot_mass_flux_out', \
                        'net_cold_metal_flux', 'cold_metal_flux_in', 'cold_metal_flux_out', \
                        'net_cool_metal_flux', 'cool_metal_flux_in', 'cool_metal_flux_out', \
                        'net_warm_metal_flux', 'warm_metal_flux_in', 'warm_metal_flux_out', \
                        'net_hot_metal_flux', 'hot_metal_flux_in', 'hot_metal_flux_out', \
                        'net_kinetic_energy_flux', 'net_thermal_energy_flux', \
                        'net_potential_energy_flux', 'net_radiative_energy_flux', \
                        'net_entropy_flux', \
                        'kinetic_energy_flux_in', 'kinetic_energy_flux_out', \
                        'thermal_energy_flux_in', 'thermal_energy_flux_out', \
                        'potential_energy_flux_in', 'potential_energy_flux_out', \
                        'radiative_energy_flux_in', 'radiative_energy_flux_out', \
                        'entropy_flux_in', 'entropy_flux_out', \
                        'net_cold_kinetic_energy_flux', 'cold_kinetic_energy_flux_in', 'cold_kinetic_energy_flux_out', \
                        'net_cool_kinetic_energy_flux', 'cool_kinetic_energy_flux_in', 'cool_kinetic_energy_flux_out', \
                        'net_warm_kinetic_energy_flux', 'warm_kinetic_energy_flux_in', 'warm_kinetic_energy_flux_out', \
                        'net_hot_kinetic_energy_flux', 'hot_kinetic_energy_flux_in', 'hot_kinetic_energy_flux_out', \
                        'net_cold_thermal_energy_flux', 'cold_thermal_energy_flux_in', 'cold_thermal_energy_flux_out', \
                        'net_cool_thermal_energy_flux', 'cool_thermal_energy_flux_in', 'cool_thermal_energy_flux_out', \
                        'net_warm_thermal_energy_flux', 'warm_thermal_energy_flux_in', 'warm_thermal_energy_flux_out', \
                        'net_hot_thermal_energy_flux', 'hot_thermal_energy_flux_in', 'hot_thermal_energy_flux_out', \
                        'net_cold_potential_energy_flux', 'cold_potential_energy_flux_in', 'cold_potential_energy_flux_out', \
                        'net_cool_potential_energy_flux', 'cool_potential_energy_flux_in', 'cool_potential_energy_flux_out', \
                        'net_warm_potential_energy_flux', 'warm_potential_energy_flux_in', 'warm_potential_energy_flux_out', \
                        'net_hot_potential_energy_flux', 'hot_potential_energy_flux_in', 'hot_potential_energy_flux_out', \
                        'net_cold_radiative_energy_flux', 'cold_radiative_energy_flux_in', 'cold_radiative_energy_flux_out', \
                        'net_cool_radiative_energy_flux', 'cool_radiative_energy_flux_in', 'cool_radiative_energy_flux_out', \
                        'net_warm_radiative_energy_flux', 'warm_radiative_energy_flux_in', 'warm_radiative_energy_flux_out', \
                        'net_hot_radiative_energy_flux', 'hot_radiative_energy_flux_in', 'hot_radiative_energy_flux_out', \
                        'net_cold_entropy_flux', 'cold_entropy_flux_in', 'cold_entropy_flux_out', \
                        'net_cool_entropy_flux', 'cool_entropy_flux_in', 'cool_entropy_flux_out', \
                        'net_warm_entropy_flux', 'warm_entropy_flux_in', 'warm_entropy_flux_out', \
                        'net_hot_entropy_flux', 'hot_entropy_flux_in', 'hot_entropy_flux_out'), \
                 dtype=('f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', \
                        'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', \
                        'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', \
                        'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', \
                        'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', \
                        'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', \
                        'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', \
                        'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', \
                        'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8'))

    fluxes_sat = Table(names=('redshift', 'inner_radius', 'outer_radius', \
                        'net_mass_flux', 'net_metal_flux', \
                        'mass_flux_in', 'mass_flux_out', 'metal_flux_in', 'metal_flux_out', \
                        'net_cold_mass_flux', 'cold_mass_flux_in', 'cold_mass_flux_out', \
                        'net_cool_mass_flux', 'cool_mass_flux_in', 'cool_mass_flux_out', \
                        'net_warm_mass_flux', 'warm_mass_flux_in', 'warm_mass_flux_out', \
                        'net_hot_mass_flux', 'hot_mass_flux_in', 'hot_mass_flux_out', \
                        'net_cold_metal_flux', 'cold_metal_flux_in', 'cold_metal_flux_out', \
                        'net_cool_metal_flux', 'cool_metal_flux_in', 'cool_metal_flux_out', \
                        'net_warm_metal_flux', 'warm_metal_flux_in', 'warm_metal_flux_out', \
                        'net_hot_metal_flux', 'hot_metal_flux_in', 'hot_metal_flux_out', \
                        'net_kinetic_energy_flux', 'net_thermal_energy_flux', \
                        'net_potential_energy_flux', 'net_entropy_flux', \
                        'kinetic_energy_flux_in', 'kinetic_energy_flux_out', \
                        'thermal_energy_flux_in', 'thermal_energy_flux_out', \
                        'potential_energy_flux_in', 'potential_energy_flux_out', \
                        'entropy_flux_in', 'entropy_flux_out', \
                        'net_cold_kinetic_energy_flux', \
                        'cold_kinetic_energy_flux_in', 'cold_kinetic_energy_flux_out', \
                        'net_cool_kinetic_energy_flux', \
                        'cool_kinetic_energy_flux_in', 'cool_kinetic_energy_flux_out', \
                        'net_warm_kinetic_energy_flux', \
                        'warm_kinetic_energy_flux_in', 'warm_kinetic_energy_flux_out', \
                        'net_hot_kinetic_energy_flux', \
                        'hot_kinetic_energy_flux_in', 'hot_kinetic_energy_flux_out', \
                        'net_cold_thermal_energy_flux', \
                        'cold_thermal_energy_flux_in', 'cold_thermal_energy_flux_out', \
                        'net_cool_thermal_energy_flux', \
                        'cool_thermal_energy_flux_in', 'cool_thermal_energy_flux_out', \
                        'net_warm_thermal_energy_flux', \
                        'warm_thermal_energy_flux_in', 'warm_thermal_energy_flux_out', \
                        'net_hot_thermal_energy_flux', \
                        'hot_thermal_energy_flux_in', 'hot_thermal_energy_flux_out', \
                        'net_cold_potential_energy_flux', \
                        'cold_potential_energy_flux_in', 'cold_potential_energy_flux_out', \
                        'net_cool_potential_energy_flux', \
                        'cool_potential_energy_flux_in', 'cool_potential_energy_flux_out', \
                        'net_warm_potential_energy_flux', \
                        'warm_potential_energy_flux_in', 'warm_potential_energy_flux_out', \
                        'net_hot_potential_energy_flux', \
                        'hot_potential_energy_flux_in', 'hot_potential_energy_flux_out', \
                        'net_cold_entropy_flux', 'cold_entropy_flux_in', 'cold_entropy_flux_out', \
                        'net_cool_entropy_flux', 'cool_entropy_flux_in', 'cool_entropy_flux_out', \
                        'net_warm_entropy_flux', 'warm_entropy_flux_in', 'warm_entropy_flux_out', \
                        'net_hot_entropy_flux', 'hot_entropy_flux_in', 'hot_entropy_flux_out'), \
                 dtype=('f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', \
                        'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', \
                        'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', \
                        'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', \
                        'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', \
                        'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', \
                        'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', \
                        'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8'))

    # Define the radii of the spherical shells where we want to calculate fluxes
    radii = 0.5*refine_width_kpc * np.arange(0.1, 0.9, 0.01)

    # Load arrays of all fields we need
    print('Loading field arrays')
    sphere = ds.sphere(halo_center_kpc, radii[-1])

    radius = sphere['gas','radius_corrected'].in_units('kpc').v
    x = sphere['gas','x'].in_units('kpc').v
    y = sphere['gas','y'].in_units('kpc').v
    z = sphere['gas','z'].in_units('kpc').v
    vx = sphere['gas','vx_corrected'].in_units('km/s').v
    vy = sphere['gas','vy_corrected'].in_units('km/s').v
    vz = sphere['gas','vz_corrected'].in_units('km/s').v
    rad_vel = sphere['gas','radial_velocity_corrected'].in_units('km/s').v
    new_x = x + vx*dt*(100./cmtopc*stoyr)
    new_y = y + vy*dt*(100./cmtopc*stoyr)
    new_z = z + vz*dt*(100./cmtopc*stoyr)
    new_radius = np.sqrt((new_x-halo_center_kpc[0].v)**2. + (new_y-halo_center_kpc[1].v)**2. + (new_z-halo_center_kpc[2].v)**2.)
    mass = sphere['gas','cell_mass'].in_units('Msun').v
    metal_mass = sphere['gas','metal_mass'].in_units('Msun').v
    temperature = sphere['gas','temperature'].in_units('K').v
    kinetic_energy = sphere['gas','kinetic_energy_corrected'].in_units('erg').v
    thermal_energy = sphere['gas','thermal_energy'].in_units('erg/g').v
    potential_energy = G * mass * gtoMsun * Menc_func(radius) * gtoMsun / (radius*1000.*cmtopc)
    cooling_time = sphere['gas','cooling_time'].in_units('yr').v
    entropy = sphere['gas','entropy'].in_units('keV*cm**2').v

    # Load list of satellite positions
    print('Loading satellite positions')
    sat_x, sat_y, sat_z = np.loadtxt(sat_file, unpack=True, usecols=[1,2,3])
    sat_list = []
    for i in range(len(sat_x)):
        if not ((np.abs(sat_x[i] - halo_center_kpc[0].v) <= 1.) & \
                (np.abs(sat_y[i] - halo_center_kpc[1].v) <= 1.) & \
                (np.abs(sat_z[i] - halo_center_kpc[2].v) <= 1.)):
            sat_list.append([sat_x[i], sat_y[i], sat_z[i]])
    sat_list = np.array(sat_list)
    sat_x2, sat_y2, sat_z2 = np.loadtxt(sat_file2, unpack=True, usecols=[1,2,3])
    sat_list2 = []
    for i in range(len(sat_x2)):
        if not ((np.abs(sat_x2[i] - halo_center_kpc[0].v) <= 1.) & \
                (np.abs(sat_y2[i] - halo_center_kpc[1].v) <= 1.) & \
                (np.abs(sat_z2[i] - halo_center_kpc[2].v) <= 1.)):
            sat_list2.append([sat_x2[i], sat_y2[i], sat_z2[i]])
    sat_list2 = np.array(sat_list2)

    # Cut data to remove anything within satellites and to things that cross into and out of satellites
    print('Cutting data to remove satellites')
    sat_radius = 5.         # kpc
    bool_nosat = []
    bool_tosat = []
    bool_fromsat = []
    for i in range(len(x)):
        pos_in_sat = []
        inside_sat1 = []
        inside_sat2 = []
        outside_sat1 = []
        outside_sat2 = []
        for s in range(len(sat_list)):
            sat_x = sat_list[s][0]
            sat_y = sat_list[s][1]
            sat_z = sat_list[s][2]
            dist_fromsat = np.sqrt((x[i]-sat_x)**2. + (y[i]-sat_y)**2. + (z[i]-sat_z)**2.)
            for s2 in range(len(sat_list2)):
                sat_x2 = sat_list2[s2][0]
                sat_y2 = sat_list2[s2][1]
                sat_z2 = sat_list2[s2][2]
                new_dist_fromsat = np.sqrt((new_x[i]-sat_x2)**2. + (new_y[i]-sat_y2)**2. + (new_z[i]-sat_z2)**2.)
                if (dist_fromsat > sat_radius) and (new_dist_fromsat > sat_radius):
                    pos_in_sat.append(False)
                else: pos_in_sat.append(True)
                if (dist_fromsat < sat_radius): inside_sat1.append(True)
                else: inside_sat1.append(False)
                if (new_dist_fromsat < sat_radius): inside_sat2.append(True)
                else: inside_sat2.append(False)
        if (True in inside_sat1) and not (True in inside_sat2):
            bool_fromsat.append(True)
        else: bool_fromsat.append(False)
        if (True in inside_sat2) and not (True in inside_sat1):
            bool_tosat.append(True)
        else: bool_tosat.append(False)
        if not (True in pos_in_sat):
            bool_nosat.append(True)
        else: bool_nosat.append(False)
    bool_nosat = np.array(bool_nosat)
    bool_fromsat = np.array(bool_fromsat)
    bool_tosat = np.array(bool_tosat)

    radius_nosat = radius[bool_nosat]
    newradius_nosat = new_radius[bool_nosat]
    x_nosat = x[bool_nosat]
    y_nosat = y[bool_nosat]
    z_nosat = z[bool_nosat]
    newx_nosat = new_x[bool_nosat]
    newy_nosat = new_y[bool_nosat]
    newz_nosat = new_z[bool_nosat]
    vx_nosat = vx[bool_nosat]
    vy_nosat = vy[bool_nosat]
    vz_nosat = vz[bool_nosat]
    rad_vel_nosat = rad_vel[bool_nosat]
    temperature_nosat = temperature[bool_nosat]
    mass_nosat = mass[bool_nosat]
    metal_mass_nosat = metal_mass[bool_nosat]
    kinetic_energy_nosat = kinetic_energy[bool_nosat]
    thermal_energy_nosat = thermal_energy[bool_nosat]
    potential_energy_nosat = potential_energy[bool_nosat]
    cooling_time_nosat = cooling_time[bool_nosat]
    entropy_nosat = entropy[bool_nosat]

    # Cut satellite-removed data on temperature
    # These are lists of lists where the index goes from 0 to 4 for [all gas, cold, cool, warm, hot]
    print('Cutting satellite-removed data on temperature')
    radius_nosat_Tcut = []
    x_nosat_Tcut = []
    y_nosat_Tcut = []
    z_nosat_Tcut = []
    vx_nosat_Tcut = []
    vy_nosat_Tcut = []
    vz_nosat_Tcut = []
    rad_vel_nosat_Tcut = []
    newx_nosat_Tcut = []
    newy_nosat_Tcut = []
    newz_nosat_Tcut = []
    newradius_nosat_Tcut = []
    mass_nosat_Tcut = []
    metal_mass_nosat_Tcut = []
    kinetic_energy_nosat_Tcut = []
    thermal_energy_nosat_Tcut = []
    potential_energy_nosat_Tcut = []
    cooling_time_nosat_Tcut = []
    entropy_nosat_Tcut = []
    for j in range(5):
        if (j==0):
            t_low = 0.
            t_high = 10**12.
        if (j==1):
            t_low = 0.
            t_high = 10**4.
        if (j==2):
            t_low = 10**4.
            t_high = 10**5.
        if (j==3):
            t_low = 10**5.
            t_high = 10**6.
        if (j==4):
            t_low = 10**6.
            t_high = 10**12.
        radius_nosat_Tcut.append(radius_nosat[(temperature_nosat < t_high) & (temperature_nosat > t_low)])
        x_nosat_Tcut.append(x_nosat[(temperature_nosat < t_high) & (temperature_nosat > t_low)])
        y_nosat_Tcut.append(y_nosat[(temperature_nosat < t_high) & (temperature_nosat > t_low)])
        z_nosat_Tcut.append(z_nosat[(temperature_nosat < t_high) & (temperature_nosat > t_low)])
        vx_nosat_Tcut.append(vx_nosat[(temperature_nosat < t_high) & (temperature_nosat > t_low)])
        vy_nosat_Tcut.append(vy_nosat[(temperature_nosat < t_high) & (temperature_nosat > t_low)])
        vz_nosat_Tcut.append(vz_nosat[(temperature_nosat < t_high) & (temperature_nosat > t_low)])
        rad_vel_nosat_Tcut.append(rad_vel_nosat[(temperature_nosat < t_high) & (temperature_nosat > t_low)])
        newx_nosat_Tcut.append(newx_nosat[(temperature_nosat < t_high) & (temperature_nosat > t_low)])
        newy_nosat_Tcut.append(newy_nosat[(temperature_nosat < t_high) & (temperature_nosat > t_low)])
        newz_nosat_Tcut.append(newz_nosat[(temperature_nosat < t_high) & (temperature_nosat > t_low)])
        newradius_nosat_Tcut.append(newradius_nosat[(temperature_nosat < t_high) & (temperature_nosat > t_low)])
        mass_nosat_Tcut.append(mass_nosat[(temperature_nosat < t_high) & (temperature_nosat > t_low)])
        metal_mass_nosat_Tcut.append(metal_mass_nosat[(temperature_nosat < t_high) & (temperature_nosat > t_low)])
        kinetic_energy_nosat_Tcut.append(kinetic_energy_nosat[(temperature_nosat < t_high) & (temperature_nosat > t_low)])
        thermal_energy_nosat_Tcut.append(thermal_energy_nosat[(temperature_nosat < t_high) & (temperature_nosat > t_low)])
        potential_energy_nosat_Tcut.append(potential_energy_nosat[(temperature_nosat < t_high) & (temperature_nosat > t_low)])
        cooling_time_nosat_Tcut.append(cooling_time_nosat[(temperature_nosat < t_high) & (temperature_nosat > t_low)])
        entropy_nosat_Tcut.append(entropy_nosat[(temperature_nosat < t_high) & (temperature_nosat > t_low)])

    # Cut data to things that cross into or out of satellites
    # These are lists of lists where the index goes from 0 to 1 for [from satellites, to satellites]
    print('Cutting data to satellite fluxes')
    radius_sat = []
    newradius_sat = []
    temperature_sat = []
    mass_sat = []
    metal_mass_sat = []
    kinetic_energy_sat = []
    thermal_energy_sat = []
    potential_energy_sat = []
    entropy_sat = []
    for j in range(2):
        if (j==0):
            radius_sat.append(radius[bool_fromsat])
            newradius_sat.append(new_radius[bool_fromsat])
            mass_sat.append(mass[bool_fromsat])
            temperature_sat.append(temperature[bool_fromsat])
            metal_mass_sat.append(metal_mass[bool_fromsat])
            kinetic_energy_sat.append(kinetic_energy[bool_fromsat])
            thermal_energy_sat.append(thermal_energy[bool_fromsat])
            potential_energy_sat.append(potential_energy[bool_fromsat])
            entropy_sat.append(entropy[bool_fromsat])
        if (j==1):
            radius_sat.append(radius[bool_tosat])
            newradius_sat.append(new_radius[bool_tosat])
            mass_sat.append(mass[bool_tosat])
            temperature_sat.append(temperature[bool_tosat])
            metal_mass_sat.append(metal_mass[bool_tosat])
            kinetic_energy_sat.append(kinetic_energy[bool_tosat])
            thermal_energy_sat.append(thermal_energy[bool_tosat])
            potential_energy_sat.append(potential_energy[bool_tosat])
            entropy_sat.append(entropy[bool_tosat])

    # Cut stuff going into/out of satellites on temperature
    # These are nested lists where the first index goes from 0 to 1 for [from satellites, to satellites]
    # and the second index goes from 0 to 4 for [all gas, cold, cool, warm, hot]
    print('Cutting satellite fluxes on temperature')
    radius_sat_Tcut = []
    newradius_sat_Tcut = []
    temperature_sat_Tcut = []
    mass_sat_Tcut = []
    metal_mass_sat_Tcut = []
    kinetic_energy_sat_Tcut = []
    thermal_energy_sat_Tcut = []
    potential_energy_sat_Tcut = []
    entropy_sat_Tcut = []
    for i in range(2):
        radius_sat_Tcut.append([])
        newradius_sat_Tcut.append([])
        temperature_sat_Tcut.append([])
        mass_sat_Tcut.append([])
        metal_mass_sat_Tcut.append([])
        kinetic_energy_sat_Tcut.append([])
        thermal_energy_sat_Tcut.append([])
        potential_energy_sat_Tcut.append([])
        entropy_sat_Tcut.append([])
        for j in range(5):
            if (j==0):
                t_low = 0.
                t_high = 10**12.
            if (j==1):
                t_low = 0.
                t_high = 10**4.
            if (j==2):
                t_low = 10**4.
                t_high = 10**5.
            if (j==3):
                t_low = 10**5.
                t_high = 10**6.
            if (j==4):
                t_low = 10**6.
                t_high = 10**12.
            radius_sat_Tcut[i].append(radius_sat[i][(temperature_sat[i] > t_low) & (temperature_sat[i] < t_high)])
            newradius_sat_Tcut[i].append(newradius_sat[i][(temperature_sat[i] > t_low) & (temperature_sat[i] < t_high)])
            mass_sat_Tcut[i].append(mass_sat[i][(temperature_sat[i] > t_low) & (temperature_sat[i] < t_high)])
            metal_mass_sat_Tcut[i].append(metal_mass_sat[i][(temperature_sat[i] > t_low) & (temperature_sat[i] < t_high)])
            kinetic_energy_sat_Tcut[i].append(kinetic_energy_sat[i][(temperature_sat[i] > t_low) & (temperature_sat[i] < t_high)])
            thermal_energy_sat_Tcut[i].append(thermal_energy_sat[i][(temperature_sat[i] > t_low) & (temperature_sat[i] < t_high)])
            potential_energy_sat_Tcut[i].append(potential_energy_sat[i][(temperature_sat[i] > t_low) & (temperature_sat[i] < t_high)])
            entropy_sat_Tcut[i].append(entropy_sat[i][(temperature_sat[i] > t_low) & (temperature_sat[i] < t_high)])

    # Loop over radii
    for i in range(len(radii)):
        inner_r = radii[i].v
        if (i < len(radii) - 1): outer_r = radii[i+1].v

        if (i%10==0): print("Computing radius " + str(i) + "/" + str(len(radii)) + \
                            " for snapshot " + snap)

        # Compute net, in, and out fluxes with satellites removed
        # These are nested lists where the first index goes from 0 to 2 for [net, in, out]
        # and the second index goes from 0 to 4 for [all, cold, cool, warm, hot]
        mass_flux_nosat = []
        metal_flux_nosat = []
        kinetic_energy_flux_nosat = []
        thermal_energy_flux_nosat = []
        potential_energy_flux_nosat = []
        entropy_flux_nosat = []
        for j in range(3):
            mass_flux_nosat.append([])
            metal_flux_nosat.append([])
            kinetic_energy_flux_nosat.append([])
            thermal_energy_flux_nosat.append([])
            potential_energy_flux_nosat.append([])
            entropy_flux_nosat.append([])
            for k in range(5):
                if (j==0):
                    mass_flux_nosat[j].append((np.sum( \
                    mass_nosat_Tcut[k][(radius_nosat_Tcut[k] < inner_r) & (newradius_nosat_Tcut[k] > inner_r)]) - \
                    np.sum(mass_nosat_Tcut[k][(radius_nosat_Tcut[k] > inner_r) & (newradius_nosat_Tcut[k] < inner_r)]))/dt)
                    metal_flux_nosat[j].append((np.sum( \
                    metal_mass_nosat_Tcut[k][(radius_nosat_Tcut[k] < inner_r) & (newradius_nosat_Tcut[k] > inner_r)]) - \
                    np.sum(metal_mass_nosat_Tcut[k][(radius_nosat_Tcut[k] > inner_r) & (newradius_nosat_Tcut[k] < inner_r)]))/dt)
                    kinetic_energy_flux_nosat[j].append((np.sum( \
                    kinetic_energy_nosat_Tcut[k][(radius_nosat_Tcut[k] < inner_r) & (newradius_nosat_Tcut[k] > inner_r)]) - \
                    np.sum(kinetic_energy_nosat_Tcut[k][(radius_nosat_Tcut[k] > inner_r) & (newradius_nosat_Tcut[k] < inner_r)]))/dt)
                    thermal_energy_flux_nosat[j].append((np.sum( \
                    thermal_energy_nosat_Tcut[k][(radius_nosat_Tcut[k] < inner_r) & (newradius_nosat_Tcut[k] > inner_r)]) - \
                    np.sum(thermal_energy_nosat_Tcut[k][(radius_nosat_Tcut[k] > inner_r) & (newradius_nosat_Tcut[k] < inner_r)]))/dt)
                    potential_energy_flux_nosat[j].append((np.sum( \
                    potential_energy_nosat_Tcut[k][(radius_nosat_Tcut[k] < inner_r) & (newradius_nosat_Tcut[k] > inner_r)]) - \
                    np.sum(potential_energy_nosat_Tcut[k][(radius_nosat_Tcut[k] > inner_r) & (newradius_nosat_Tcut[k] < inner_r)]))/dt)
                    entropy_flux_nosat[j].append((np.sum( \
                    entropy_nosat_Tcut[k][(radius_nosat_Tcut[k] < inner_r) & (newradius_nosat_Tcut[k] > inner_r)]) - \
                    np.sum(entropy_nosat_Tcut[k][(radius_nosat_Tcut[k] > inner_r) & (newradius_nosat_Tcut[k] < inner_r)]))/dt)
                if (j==1):
                    mass_flux_nosat[j].append(-np.sum( \
                    mass_nosat_Tcut[k][(radius_nosat_Tcut[k] > inner_r) & (newradius_nosat_Tcut[k] < inner_r)])/dt)
                    metal_flux_nosat[j].append(-np.sum( \
                    metal_mass_nosat_Tcut[k][(radius_nosat_Tcut[k] > inner_r) & (newradius_nosat_Tcut[k] < inner_r)])/dt)
                    kinetic_energy_flux_nosat[j].append(-np.sum( \
                    kinetic_energy_nosat_Tcut[k][(radius_nosat_Tcut[k] > inner_r) & (newradius_nosat_Tcut[k] < inner_r)])/dt)
                    thermal_energy_flux_nosat[j].append(-np.sum( \
                    thermal_energy_nosat_Tcut[k][(radius_nosat_Tcut[k] > inner_r) & (newradius_nosat_Tcut[k] < inner_r)])/dt)
                    potential_energy_flux_nosat[j].append(-np.sum( \
                    potential_energy_nosat_Tcut[k][(radius_nosat_Tcut[k] > inner_r) & (newradius_nosat_Tcut[k] < inner_r)])/dt)
                    entropy_flux_nosat[j].append(-np.sum( \
                    entropy_nosat_Tcut[k][(radius_nosat_Tcut[k] > inner_r) & (newradius_nosat_Tcut[k] < inner_r)])/dt)
                if (j==2):
                    mass_flux_nosat[j].append(np.sum( \
                    mass_nosat_Tcut[k][(radius_nosat_Tcut[k] < inner_r) & (newradius_nosat_Tcut[k] > inner_r)])/dt)
                    metal_flux_nosat[j].append(np.sum( \
                    metal_mass_nosat_Tcut[k][(radius_nosat_Tcut[k] < inner_r) & (newradius_nosat_Tcut[k] > inner_r)])/dt)
                    kinetic_energy_flux_nosat[j].append(np.sum( \
                    kinetic_energy_nosat_Tcut[k][(radius_nosat_Tcut[k] < inner_r) & (newradius_nosat_Tcut[k] > inner_r)])/dt)
                    thermal_energy_flux_nosat[j].append(np.sum( \
                    thermal_energy_nosat_Tcut[k][(radius_nosat_Tcut[k] < inner_r) & (newradius_nosat_Tcut[k] > inner_r)])/dt)
                    potential_energy_flux_nosat[j].append(np.sum( \
                    potential_energy_nosat_Tcut[k][(radius_nosat_Tcut[k] < inner_r) & (newradius_nosat_Tcut[k] > inner_r)])/dt)
                    entropy_flux_nosat[j].append(np.sum( \
                    entropy_nosat_Tcut[k][(radius_nosat_Tcut[k] < inner_r) & (newradius_nosat_Tcut[k] > inner_r)])/dt)

        # Compute fluxes from and to satellites (and net) between inner_r and outer_r
        # These are nested lists where the first index goes from 0 to 2 for [net, from, to]
        # and the second index goes from 0 to 4 for [all, cold, cool, warm, hot]
        if (i < len(radii)-1):
            mass_flux_sat = []
            metal_flux_sat = []
            kinetic_energy_flux_sat = []
            thermal_energy_flux_sat = []
            potential_energy_flux_sat = []
            entropy_flux_sat = []
            radiative_energy_flux_nosat = []
            for j in range(3):
                mass_flux_sat.append([])
                metal_flux_sat.append([])
                kinetic_energy_flux_sat.append([])
                thermal_energy_flux_sat.append([])
                potential_energy_flux_sat.append([])
                entropy_flux_sat.append([])
                radiative_energy_flux_nosat.append([])
                for k in range(5):
                    if (j==0):
                        mass_flux_sat[j].append((np.sum(mass_sat_Tcut[0][k][(newradius_sat_Tcut[0][k]>inner_r) & \
                                                                            (newradius_sat_Tcut[0][k]<outer_r)]) - \
                                                np.sum(mass_sat_Tcut[1][k][(radius_sat_Tcut[1][k]>inner_r) & \
                                                                           (radius_sat_Tcut[1][k]<outer_r)]))/dt)
                        metal_flux_sat[j].append((np.sum(metal_mass_sat_Tcut[0][k][(newradius_sat_Tcut[0][k]>inner_r) & \
                                                                              (newradius_sat_Tcut[0][k]<outer_r)]) - \
                                                np.sum(metal_mass_sat_Tcut[1][k][(radius_sat_Tcut[1][k]>inner_r) & \
                                                                            (radius_sat_Tcut[1][k]<outer_r)]))/dt)
                        kinetic_energy_flux_sat[j].append((np.sum(kinetic_energy_sat_Tcut[0][k][(newradius_sat_Tcut[0][k]>inner_r) & \
                                                                                                (newradius_sat_Tcut[0][k]<outer_r)]) - \
                                                          np.sum(kinetic_energy_sat_Tcut[1][k][(radius_sat_Tcut[1][k]>inner_r) & \
                                                                                               (radius_sat_Tcut[1][k]<outer_r)]))/dt)
                        thermal_energy_flux_sat[j].append((np.sum(thermal_energy_sat_Tcut[0][k][(newradius_sat_Tcut[0][k]>inner_r) & \
                                                                                                (newradius_sat_Tcut[0][k]<outer_r)]) - \
                                                          np.sum(thermal_energy_sat_Tcut[1][k][(radius_sat_Tcut[1][k]>inner_r) & \
                                                                                               (radius_sat_Tcut[1][k]<outer_r)]))/dt)
                        potential_energy_flux_sat[j].append((np.sum(potential_energy_sat_Tcut[0][k][(newradius_sat_Tcut[0][k]>inner_r) & \
                                                                                                    (newradius_sat_Tcut[0][k]<outer_r)]) - \
                                                            np.sum(potential_energy_sat_Tcut[1][k][(radius_sat_Tcut[1][k]>inner_r) & \
                                                                                                   (radius_sat_Tcut[1][k]<outer_r)]))/dt)
                        entropy_flux_sat[j].append((np.sum(entropy_sat_Tcut[0][k][(newradius_sat_Tcut[0][k]>inner_r) & \
                                                                                 (newradius_sat_Tcut[0][k]<outer_r)]) - \
                                                    np.sum(entropy_sat_Tcut[1][k][(radius_sat_Tcut[1][k]>inner_r) & \
                                                                                  (radius_sat_Tcut[1][k]<outer_r)]))/dt)
                        radiative_energy_flux_nosat[j].append(np.sum( \
                          thermal_energy_nosat_Tcut[k][(radius_nosat_Tcut[k]>inner_r) & (radius_nosat_Tcut[k]<outer_r)] * \
                          mass_nosat_Tcut[k][(radius_nosat_Tcut[k]>inner_r) & (radius_nosat_Tcut[k]<outer_r)]*gtoMsun / \
                          cooling_time_nosat_Tcut[k][(radius_nosat_Tcut[k]>inner_r) & (radius_nosat_Tcut[k]<outer_r)]))
                    if (j==1):
                        mass_flux_sat[j].append(np.sum(mass_sat_Tcut[0][k][(newradius_sat_Tcut[0][k]>inner_r) & \
                                                                           (newradius_sat_Tcut[0][k]<outer_r)])/dt)
                        metal_flux_sat[j].append(np.sum(metal_mass_sat_Tcut[0][k][(newradius_sat_Tcut[0][k]>inner_r) & \
                                                                             (newradius_sat_Tcut[0][k]<outer_r)])/dt)
                        kinetic_energy_flux_sat[j].append(np.sum(kinetic_energy_sat_Tcut[0][k][(newradius_sat_Tcut[0][k]>inner_r) & \
                                                                                               (newradius_sat_Tcut[0][k]<outer_r)])/dt)
                        thermal_energy_flux_sat[j].append(np.sum(thermal_energy_sat_Tcut[0][k][(newradius_sat_Tcut[0][k]>inner_r) & \
                                                                                               (newradius_sat_Tcut[0][k]<outer_r)])/dt)
                        potential_energy_flux_sat[j].append(np.sum(potential_energy_sat_Tcut[0][k][(newradius_sat_Tcut[0][k]>inner_r) & \
                                                                                                   (newradius_sat_Tcut[0][k]<outer_r)])/dt)
                        entropy_flux_sat[j].append(np.sum(entropy_sat_Tcut[0][k][(newradius_sat_Tcut[0][k]>inner_r) & \
                                                                                 (newradius_sat_Tcut[0][k]<outer_r)])/dt)
                        radiative_energy_flux_nosat[j].append(np.sum( \
                          thermal_energy_nosat_Tcut[k][(radius_nosat_Tcut[k]>inner_r) & (radius_nosat_Tcut[k]<outer_r) & (rad_vel_nosat_Tcut[k]<0.)] * \
                          mass_nosat_Tcut[k][(radius_nosat_Tcut[k]>inner_r) & (radius_nosat_Tcut[k]<outer_r) & (rad_vel_nosat_Tcut[k]<0.)]*gtoMsun / \
                          cooling_time_nosat_Tcut[k][(radius_nosat_Tcut[k]>inner_r) & (radius_nosat_Tcut[k]<outer_r) & (rad_vel_nosat_Tcut[k]<0.)]))
                    if (j==2):
                        mass_flux_sat[j].append(-np.sum(mass_sat_Tcut[1][k][(radius_sat_Tcut[1][k]>inner_r) & \
                                                                           (radius_sat_Tcut[1][k]<outer_r)])/dt)
                        metal_flux_sat[j].append(-np.sum(metal_mass_sat_Tcut[1][k][(radius_sat_Tcut[1][k]>inner_r) & \
                                                                            (radius_sat_Tcut[1][k]<outer_r)])/dt)
                        kinetic_energy_flux_sat[j].append(-np.sum(kinetic_energy_sat_Tcut[1][k][(radius_sat_Tcut[1][k]>inner_r) & \
                                                                                               (radius_sat_Tcut[1][k]<outer_r)])/dt)
                        thermal_energy_flux_sat[j].append(-np.sum(thermal_energy_sat_Tcut[1][k][(radius_sat_Tcut[1][k]>inner_r) & \
                                                                                               (radius_sat_Tcut[1][k]<outer_r)])/dt)
                        potential_energy_flux_sat[j].append(-np.sum(potential_energy_sat_Tcut[1][k][(radius_sat_Tcut[1][k]>inner_r) & \
                                                                                                   (radius_sat_Tcut[1][k]<outer_r)])/dt)
                        entropy_flux_sat[j].append(-np.sum(entropy_sat_Tcut[1][k][(radius_sat_Tcut[1][k]>inner_r) & \
                                                                                  (radius_sat_Tcut[1][k]<outer_r)])/dt)
                        radiative_energy_flux_nosat[j].append(np.sum( \
                          thermal_energy_nosat_Tcut[k][(radius_nosat_Tcut[k]>inner_r) & (radius_nosat_Tcut[k]<outer_r) & (rad_vel_nosat_Tcut[k]>0.)] * \
                          mass_nosat_Tcut[k][(radius_nosat_Tcut[k]>inner_r) & (radius_nosat_Tcut[k]<outer_r) & (rad_vel_nosat_Tcut[k]>0.)]*gtoMsun / \
                          cooling_time_nosat_Tcut[k][(radius_nosat_Tcut[k]>inner_r) & (radius_nosat_Tcut[k]<outer_r) & (rad_vel_nosat_Tcut[k]>0.)]))

        # Add everything to the table
        fluxes.add_row([zsnap, inner_r, \
                        mass_flux_nosat[0][0], metal_flux_nosat[0][0], \
                        mass_flux_nosat[1][0], mass_flux_nosat[2][0], metal_flux_nosat[1][0], metal_flux_nosat[2][0], \
                        mass_flux_nosat[0][1], mass_flux_nosat[1][1], mass_flux_nosat[2][1], \
                        mass_flux_nosat[0][2], mass_flux_nosat[1][2], mass_flux_nosat[2][2], \
                        mass_flux_nosat[0][3], mass_flux_nosat[1][3], mass_flux_nosat[2][3], \
                        mass_flux_nosat[0][4], mass_flux_nosat[1][4], mass_flux_nosat[2][4], \
                        metal_flux_nosat[0][1], metal_flux_nosat[1][1], metal_flux_nosat[2][1], \
                        metal_flux_nosat[0][2], metal_flux_nosat[1][2], metal_flux_nosat[2][2], \
                        metal_flux_nosat[0][3], metal_flux_nosat[1][3], metal_flux_nosat[2][3], \
                        metal_flux_nosat[0][4], metal_flux_nosat[1][4], metal_flux_nosat[2][4], \
                        kinetic_energy_flux_nosat[0][0], thermal_energy_flux_nosat[0][0], \
                        potential_energy_flux_nosat[0][0], radiative_energy_flux_nosat[0][0], \
                        entropy_flux_nosat[0][0], \
                        kinetic_energy_flux_nosat[1][0], kinetic_energy_flux_nosat[2][0], \
                        thermal_energy_flux_nosat[1][0], thermal_energy_flux_nosat[2][0], \
                        potential_energy_flux_nosat[1][0], potential_energy_flux_nosat[2][0], \
                        radiative_energy_flux_nosat[1][0], radiative_energy_flux_nosat[2][0], \
                        entropy_flux_nosat[1][0], entropy_flux_nosat[2][0], \
                        kinetic_energy_flux_nosat[0][1], kinetic_energy_flux_nosat[1][1], kinetic_energy_flux_nosat[2][1], \
                        kinetic_energy_flux_nosat[0][2], kinetic_energy_flux_nosat[1][2], kinetic_energy_flux_nosat[2][2], \
                        kinetic_energy_flux_nosat[0][3], kinetic_energy_flux_nosat[1][3], kinetic_energy_flux_nosat[2][3], \
                        kinetic_energy_flux_nosat[0][4], kinetic_energy_flux_nosat[1][4], kinetic_energy_flux_nosat[2][4], \
                        thermal_energy_flux_nosat[0][1], thermal_energy_flux_nosat[1][1], thermal_energy_flux_nosat[2][1], \
                        thermal_energy_flux_nosat[0][2], thermal_energy_flux_nosat[1][2], thermal_energy_flux_nosat[2][2], \
                        thermal_energy_flux_nosat[0][3], thermal_energy_flux_nosat[1][3], thermal_energy_flux_nosat[2][3], \
                        thermal_energy_flux_nosat[0][4], thermal_energy_flux_nosat[1][4], thermal_energy_flux_nosat[2][4], \
                        potential_energy_flux_nosat[0][1], potential_energy_flux_nosat[1][1], potential_energy_flux_nosat[2][1], \
                        potential_energy_flux_nosat[0][2], potential_energy_flux_nosat[1][2], potential_energy_flux_nosat[2][2], \
                        potential_energy_flux_nosat[0][3], potential_energy_flux_nosat[1][3], potential_energy_flux_nosat[2][3], \
                        potential_energy_flux_nosat[0][4], potential_energy_flux_nosat[1][4], potential_energy_flux_nosat[2][4], \
                        radiative_energy_flux_nosat[0][1], radiative_energy_flux_nosat[1][1], radiative_energy_flux_nosat[2][1], \
                        radiative_energy_flux_nosat[0][2], radiative_energy_flux_nosat[1][2], radiative_energy_flux_nosat[2][2], \
                        radiative_energy_flux_nosat[0][3], radiative_energy_flux_nosat[1][3], radiative_energy_flux_nosat[2][3], \
                        radiative_energy_flux_nosat[0][4], radiative_energy_flux_nosat[1][4], radiative_energy_flux_nosat[2][4], \
                        entropy_flux_nosat[0][1], entropy_flux_nosat[1][1], entropy_flux_nosat[2][1], \
                        entropy_flux_nosat[0][2], entropy_flux_nosat[1][2], entropy_flux_nosat[2][2], \
                        entropy_flux_nosat[0][3], entropy_flux_nosat[1][3], entropy_flux_nosat[2][3], \
                        entropy_flux_nosat[0][4], entropy_flux_nosat[1][4], entropy_flux_nosat[2][4]])

        fluxes_sat.add_row([zsnap, inner_r, outer_r, \
                            mass_flux_sat[0][0], metal_flux_sat[0][0], \
                            mass_flux_sat[1][0], mass_flux_sat[2][0], metal_flux_sat[1][0], metal_flux_sat[2][0], \
                            mass_flux_sat[0][1], mass_flux_sat[1][1], mass_flux_sat[2][1], \
                            mass_flux_sat[0][2], mass_flux_sat[1][2], mass_flux_sat[2][2], \
                            mass_flux_sat[0][3], mass_flux_sat[1][3], mass_flux_sat[2][3], \
                            mass_flux_sat[0][4], mass_flux_sat[1][4], mass_flux_sat[2][4], \
                            metal_flux_sat[0][1], metal_flux_sat[1][1], metal_flux_sat[2][1], \
                            metal_flux_sat[0][2], metal_flux_sat[1][2], metal_flux_sat[2][2], \
                            metal_flux_sat[0][3], metal_flux_sat[1][3], metal_flux_sat[2][3], \
                            metal_flux_sat[0][4], metal_flux_sat[1][4], metal_flux_sat[2][4], \
                            kinetic_energy_flux_sat[0][0], thermal_energy_flux_sat[0][0], \
                            potential_energy_flux_sat[0][0], entropy_flux_sat[0][0], \
                            kinetic_energy_flux_sat[1][0], kinetic_energy_flux_sat[2][0], \
                            thermal_energy_flux_sat[1][0], thermal_energy_flux_sat[2][0], \
                            potential_energy_flux_sat[1][0], potential_energy_flux_sat[2][0], \
                            entropy_flux_sat[1][0], entropy_flux_sat[2][0], \
                            kinetic_energy_flux_sat[0][1], kinetic_energy_flux_sat[1][1], kinetic_energy_flux_sat[2][1], \
                            kinetic_energy_flux_sat[0][2], kinetic_energy_flux_sat[1][2], kinetic_energy_flux_sat[2][2], \
                            kinetic_energy_flux_sat[0][3], kinetic_energy_flux_sat[1][3], kinetic_energy_flux_sat[2][3], \
                            kinetic_energy_flux_sat[0][4], kinetic_energy_flux_sat[1][4], kinetic_energy_flux_sat[2][4], \
                            thermal_energy_flux_sat[0][1], thermal_energy_flux_sat[1][1], thermal_energy_flux_sat[2][1], \
                            thermal_energy_flux_sat[0][2], thermal_energy_flux_sat[1][2], thermal_energy_flux_sat[2][2], \
                            thermal_energy_flux_sat[0][3], thermal_energy_flux_sat[1][3], thermal_energy_flux_sat[2][3], \
                            thermal_energy_flux_sat[0][4], thermal_energy_flux_sat[1][4], thermal_energy_flux_sat[2][4], \
                            potential_energy_flux_sat[0][1], potential_energy_flux_sat[1][1], potential_energy_flux_sat[2][1], \
                            potential_energy_flux_sat[0][2], potential_energy_flux_sat[1][2], potential_energy_flux_sat[2][2], \
                            potential_energy_flux_sat[0][3], potential_energy_flux_sat[1][3], potential_energy_flux_sat[2][3], \
                            potential_energy_flux_sat[0][4], potential_energy_flux_sat[1][4], potential_energy_flux_sat[2][4], \
                            entropy_flux_sat[0][1], entropy_flux_sat[1][1], entropy_flux_sat[2][1], \
                            entropy_flux_sat[0][2], entropy_flux_sat[1][2], entropy_flux_sat[2][2], \
                            entropy_flux_sat[0][3], entropy_flux_sat[1][3], entropy_flux_sat[2][3], \
                            entropy_flux_sat[0][4], entropy_flux_sat[1][4], entropy_flux_sat[2][4]])

    fluxes = set_table_units(fluxes)
    fluxes_sat = set_table_units(fluxes_sat)

    # Save to file
    fluxes.write(tablename + '_nosat.hdf5', path='all_data', serialize_meta=True, overwrite=True)
    fluxes_sat.write(tablename + '_sat.hdf5', path='all_data', serialize_meta=True, overwrite=True)

    return "Fluxes have been calculated for snapshot " + snap + "!"


def load_and_calculate(system, foggie_dir, run_dir, track, halo_c_v_name, snap, tablename, Menc_table, quadrants, sat_dir):
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
    refine_width_kpc = YTArray([refine_width], 'kpc')
    zsnap = ds.get_parameter('CosmologyCurrentRedshift')
    dt = 5.38e6

    # Make interpolated Menc_func using the table at this snapshot
    Menc_func = IUS(Menc_table['radius'][Menc_table['snapshot']==snap], \
      Menc_table['total_mass'][Menc_table['snapshot']==snap])

    # Specify the file where the list of satellites at this snapshot is saved
    sat_file = sat_dir + snap + '_satellites.dat'
    snap2 = int(snap[-4:])+1
    if (snap2 < 10): snap2 = '000' + str(snap2)
    elif (snap2 < 100): snap2 = '00' + str(snap2)
    elif (snap2 < 1000): snap2 = '0' + str(snap2)
    else: snap2 = str(snap2)
    sat_file2 = sat_dir + snap2 + '_satellites.dat'

    # Do the actual calculation
    #message = calc_fluxes(ds, snap, zsnap, refine_width_kpc, tablename, Menc_func=Menc_func)
    message = calc_fluxes_region(ds, snap, zsnap, dt, refine_width_kpc, tablename, Menc_func=Menc_func, sat_file=sat_file, sat_file2=sat_file2)
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
    prefix = output_dir + 'fluxes_halo_00' + args.halo + '/' + args.run + '/'
    if not (os.path.exists(prefix)): os.system('mkdir -p ' + prefix)

    print('foggie_dir: ', foggie_dir)
    halo_c_v_name = track_dir + 'halo_c_v'

    # Load the mass enclosed profile
    Menc_table = Table.read(code_path + 'halo_infos/00' + args.halo + '/' + args.run + '/masses.hdf5', \
      path='all_data')

    # Specify where satellite files are saved
    sat_dir = output_dir + 'satellites_halo_00' + args.halo + '/' + args.run + '/'

    # Loop over outputs, for either single-processor or parallel processor computing
    if (args.nproc==1):
        for i in range(len(outs)):
            snap = outs[i]
            # Make the output table name for this snapshot
            tablename = prefix + snap + '_fluxes'
            # Do the actual calculation
            load_and_calculate(args.system, foggie_dir, run_dir, trackname, halo_c_v_name, snap, tablename, Menc_table, args.quadrants, sat_dir)
    else:
        # Split into a number of groupings equal to the number of processors
        # and run one process per processor
        for i in range(len(outs)//args.nproc):
            threads = []
            for j in range(args.nproc):
                snap = outs[args.nproc*i+j]
                tablename = prefix + snap + '_fluxes'
                threads.append(multi.Process(target=load_and_calculate, \
			       args=(args.system, foggie_dir, run_dir, trackname, halo_c_v_name, snap, tablename, Menc_table, args.quadrants, sat_dir)))
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
			   args=(args.system, foggie_dir, run_dir, trackname, halo_c_v_name, snap, tablename, Menc_table, args.quadrants, sat_dir)))
        for t in threads:
            t.start()
        for t in threads:
            t.join()

    print(str(datetime.datetime.now()))
    print("All snapshots finished!")
