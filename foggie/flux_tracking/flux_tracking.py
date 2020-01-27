"""
Filename: flux_tracking.py
Author: Cassi
Date created: 9-27-19
Date last modified: 1-22-20
This file takes command line arguments and computes fluxes of things through surfaces.

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
import ast

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

    parser.add_argument('--surface', metavar='surface', type=str, action='store', \
                        help='What surface type for computing the flux? Default is sphere' + \
                        ' and the other option is "frustum".\nNote that all surfaces will be centered on halo center.\n' + \
                        'To specify the shape, size, and orientation of the surface you want, ' + \
                        'input a list as follows (don\'t forget the outer quotes):\nIf you want a sphere, give:\n' + \
                        '"[\'sphere\', inner_radius, outer_radius, num_radii]"\n' + \
                        'where inner_radius is the inner boundary as a fraction of refine_width, outer_radius is the outer ' + \
                        'boundary as a fraction (or multiple) of refine_width,\nand num_radii is the number of radii where you want the flux to be ' + \
                        'calculated between inner_radius and outer_radius\n' + \
                        '(inner_radius and outer_radius are automatically included).\n' + \
                        'If you want a frustum, give:\n' + \
                        '"[\'frustum\', axis, inner_radius, outer_radius, num_radii, opening_angle]"\n' + \
                        'where axis is a number 1 through 4 that specifies what axis to align the frustum with:\n' + \
                        '1) x axis 2) y axis 3) z axis 4) minor axis of galaxy disk.\n' + \
                        'If axis is given as a negative number, it will compute a frustum pointing the other way.\n' + \
                        'inner_radius, outer_radius, and num_radii are the same as for the sphere\n' + \
                        'and opening_angle gives the angle in degrees of the opening angle of the cone, measured from axis.')
    parser.set_defaults(surface="['sphere', 0.05, 2., 200]")

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

    table_units = {'redshift':None,'radius':'kpc','inner_radius':'kpc','outer_radius':'kpc', \
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

def calc_fluxes_sphere(ds, snap, zsnap, dt, refine_width_kpc, tablename, surface_args, **kwargs):
    '''This function calculates the fluxes into and out of spherical shells, with satellites removed,
    at a variety of radii. It
    uses the dataset stored in 'ds', which is from time snapshot 'snap', has redshift
    'zsnap', and has width of the refine box in kpc 'refine_width_kpc', the time step between outputs
    'dt', and stores the fluxes in 'tablename'. 'surface_args' gives the properties of the spheres.

    This function calculates the flux as the sum
    of all cells whose velocity and distance from the surface of interest indicate that the gas
    contained in that cell will be displaced across the surface of interest by the next timestep.
    That is, the properties of a cell contribute to the flux if it is no further from the surface of
    interest than v*dt where v is the cell's velocity normal to the surface and dt is the time
    between snapshots, which is dt = 5.38e6 yrs for the DD outputs. It is necessary to compute the
    flux this way if satellites are to be removed because they become 'holes' in the dataset
    and fluxes into/out of those holes need to be accounted for.'''

    Menc_func = kwargs.get('Menc_func', False)
    sat = kwargs.get('sat')
    halo_center_kpc2 = kwargs.get('halo_center_kpc2', ds.halo_center_kpc)

    G = ds.quan(6.673e-8, 'cm**3/s**2/g').v
    halo_center_kpc = ds.halo_center_kpc

    cmtopc = 3.086e18
    stoyr = 3.154e7
    gtoMsun = 1.989e33

    inner_radius = surface_args[1]
    outer_radius = surface_args[2]
    dr = (outer_radius - inner_radius)/surface_args[3]

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
    radii = refine_width_kpc * np.arange(inner_radius, outer_radius+dr, dr)

    # Load arrays of all fields we need
    print('Loading field arrays')
    sphere = ds.sphere(halo_center_kpc, radii[-1])

    radius = sphere['gas','radius_corrected'].in_units('kpc').v
    x = sphere['gas','x'].in_units('kpc').v - halo_center_kpc[0].v
    y = sphere['gas','y'].in_units('kpc').v - halo_center_kpc[1].v
    z = sphere['gas','z'].in_units('kpc').v - halo_center_kpc[2].v
    vx = sphere['gas','vx_corrected'].in_units('km/s').v
    vy = sphere['gas','vy_corrected'].in_units('km/s').v
    vz = sphere['gas','vz_corrected'].in_units('km/s').v
    rad_vel = sphere['gas','radial_velocity_corrected'].in_units('km/s').v
    new_x = x + vx*dt*(100./cmtopc*stoyr)
    new_y = y + vy*dt*(100./cmtopc*stoyr)
    new_z = z + vz*dt*(100./cmtopc*stoyr)
    new_radius = np.sqrt(new_x**2. + new_y**2. + new_z**2.)
    mass = sphere['gas','cell_mass'].in_units('Msun').v
    metal_mass = sphere['gas','metal_mass'].in_units('Msun').v
    temperature = sphere['gas','temperature'].in_units('K').v
    kinetic_energy = sphere['gas','kinetic_energy_corrected'].in_units('erg').v
    thermal_energy = sphere['gas','thermal_energy'].in_units('erg/g').v
    potential_energy = (sphere['gas','cell_mass'] * \
      ds.arr(sphere['enzo','Grav_Potential'].v, 'code_length**2/code_time**2')).in_units('erg').v
    cooling_time = sphere['gas','cooling_time'].in_units('yr').v
    entropy = sphere['gas','entropy'].in_units('keV*cm**2').v

    # Load list of satellite positions
    print('Loading satellite positions')
    sat_x = sat['sat_x'][sat['snap']==snap]
    sat_y = sat['sat_y'][sat['snap']==snap]
    sat_z = sat['sat_z'][sat['snap']==snap]
    sat_list = []
    for i in range(len(sat_x)):
        if not ((np.abs(sat_x[i] - halo_center_kpc[0].v) <= 1.) & \
                (np.abs(sat_y[i] - halo_center_kpc[1].v) <= 1.) & \
                (np.abs(sat_z[i] - halo_center_kpc[2].v) <= 1.)):
            sat_list.append([sat_x[i] - halo_center_kpc[0].v, sat_y[i] - halo_center_kpc[1].v, sat_z[i] - halo_center_kpc[2].v])
    sat_list = np.array(sat_list)
    snap2 = int(snap[-4:])+1
    snap_type = snap[-6:-4]
    if (snap2 < 10): snap2 = snap_type + '000' + str(snap2)
    elif (snap2 < 100): snap2 = snap_type + '00' + str(snap2)
    elif (snap2 < 1000): snap2 = snap_type + '0' + str(snap2)
    else: snap2 = snap_type + str(snap2)
    sat_x2 = sat['sat_x'][sat['snap']==snap2]
    sat_y2 = sat['sat_y'][sat['snap']==snap2]
    sat_z2 = sat['sat_z'][sat['snap']==snap2]
    sat_list2 = []
    for i in range(len(sat_x2)):
        if not ((np.abs(sat_x2[i] - halo_center_kpc2[0].v) <= 1.) & \
                (np.abs(sat_y2[i] - halo_center_kpc2[1].v) <= 1.) & \
                (np.abs(sat_z2[i] - halo_center_kpc2[2].v) <= 1.)):
            sat_list2.append([sat_x2[i] - halo_center_kpc2[0].v, sat_y2[i] - halo_center_kpc2[1].v, sat_z2[i] - halo_center_kpc2[2].v])
    sat_list2 = np.array(sat_list2)

    # Cut data to remove anything within satellites and to things that cross into and out of satellites
    print('Cutting data to remove satellites')
    sat_radius = 10.         # kpc
    sat_radius_sq = sat_radius**2.
    # An attempt to remove satellites faster:
    # Holy cow this is so much faster, do it this way
    bool_inside_sat1 = []
    bool_inside_sat2 = []
    for s in range(len(sat_list)):
        sat_x = sat_list[s][0]
        sat_y = sat_list[s][1]
        sat_z = sat_list[s][2]
        dist_from_sat1_sq = (x-sat_x)**2. + (y-sat_y)**2. + (z-sat_z)**2.
        bool_inside_sat1.append((dist_from_sat1_sq < sat_radius_sq))
    bool_inside_sat1 = np.array(bool_inside_sat1)
    for s2 in range(len(sat_list2)):
        sat_x2 = sat_list2[s2][0]
        sat_y2 = sat_list2[s2][1]
        sat_z2 = sat_list2[s2][2]
        dist_from_sat2_sq = (new_x-sat_x2)**2. + (new_y-sat_y2)**2. + (new_z-sat_z2)**2.
        bool_inside_sat2.append((dist_from_sat2_sq < sat_radius_sq))
    bool_inside_sat2 = np.array(bool_inside_sat2)
    inside_sat1 = np.count_nonzero(bool_inside_sat1, axis=0)
    inside_sat2 = np.count_nonzero(bool_inside_sat2, axis=0)
    # inside_sat1 and inside_sat2 should now both be arrays of length = # of pixels where the value is an
    # integer. If the value is zero, that pixel is not inside any satellites. If the value is > 0,
    # that pixel is in a satellite.
    bool_nosat = (inside_sat1 == 0) & (inside_sat2 == 0)
    bool_fromsat = (inside_sat1 > 0) & (inside_sat2 == 0)
    bool_tosat = (inside_sat1 == 0) & (inside_sat2 > 0)

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
        bool_temp = (temperature_nosat < t_high) & (temperature_nosat > t_low)
        radius_nosat_Tcut.append(radius_nosat[bool_temp])
        rad_vel_nosat_Tcut.append(rad_vel_nosat[bool_temp])
        newradius_nosat_Tcut.append(newradius_nosat[bool_temp])
        mass_nosat_Tcut.append(mass_nosat[bool_temp])
        metal_mass_nosat_Tcut.append(metal_mass_nosat[bool_temp])
        kinetic_energy_nosat_Tcut.append(kinetic_energy_nosat[bool_temp])
        thermal_energy_nosat_Tcut.append(thermal_energy_nosat[bool_temp])
        potential_energy_nosat_Tcut.append(potential_energy_nosat[bool_temp])
        cooling_time_nosat_Tcut.append(cooling_time_nosat[bool_temp])
        entropy_nosat_Tcut.append(entropy_nosat[bool_temp])

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
            bool_temp = (temperature_sat[i] > t_low) & (temperature_sat[i] < t_high)
            radius_sat_Tcut[i].append(radius_sat[i][bool_temp])
            newradius_sat_Tcut[i].append(newradius_sat[i][bool_temp])
            mass_sat_Tcut[i].append(mass_sat[i][bool_temp])
            metal_mass_sat_Tcut[i].append(metal_mass_sat[i][bool_temp])
            kinetic_energy_sat_Tcut[i].append(kinetic_energy_sat[i][bool_temp])
            thermal_energy_sat_Tcut[i].append(thermal_energy_sat[i][bool_temp])
            potential_energy_sat_Tcut[i].append(potential_energy_sat[i][bool_temp])
            entropy_sat_Tcut[i].append(entropy_sat[i][bool_temp])

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
                bool_in = (radius_nosat_Tcut[k] > inner_r) & (newradius_nosat_Tcut[k] < inner_r)
                bool_out = (radius_nosat_Tcut[k] < inner_r) & (newradius_nosat_Tcut[k] > inner_r)
                if (j==0):
                    mass_flux_nosat[j].append((np.sum(mass_nosat_Tcut[k][bool_out]) - \
                      np.sum(mass_nosat_Tcut[k][bool_in]))/dt)
                    metal_flux_nosat[j].append((np.sum(metal_mass_nosat_Tcut[k][bool_out]) - \
                      np.sum(metal_mass_nosat_Tcut[k][bool_in]))/dt)
                    kinetic_energy_flux_nosat[j].append((np.sum(kinetic_energy_nosat_Tcut[k][bool_out]) - \
                      np.sum(kinetic_energy_nosat_Tcut[k][bool_in]))/dt)
                    thermal_energy_flux_nosat[j].append((np.sum(thermal_energy_nosat_Tcut[k][bool_out]) - \
                      np.sum(thermal_energy_nosat_Tcut[k][bool_in]))/dt)
                    potential_energy_flux_nosat[j].append((np.sum(potential_energy_nosat_Tcut[k][bool_out]) - \
                      np.sum(potential_energy_nosat_Tcut[k][bool_in]))/dt)
                    entropy_flux_nosat[j].append((np.sum(entropy_nosat_Tcut[k][bool_out]) - \
                      np.sum(entropy_nosat_Tcut[k][bool_in]))/dt)
                if (j==1):
                    mass_flux_nosat[j].append(-np.sum(mass_nosat_Tcut[k][bool_in])/dt)
                    metal_flux_nosat[j].append(-np.sum(metal_mass_nosat_Tcut[k][bool_in])/dt)
                    kinetic_energy_flux_nosat[j].append(-np.sum(kinetic_energy_nosat_Tcut[k][bool_in])/dt)
                    thermal_energy_flux_nosat[j].append(-np.sum(thermal_energy_nosat_Tcut[k][bool_in])/dt)
                    potential_energy_flux_nosat[j].append(-np.sum(potential_energy_nosat_Tcut[k][bool_in])/dt)
                    entropy_flux_nosat[j].append(-np.sum(entropy_nosat_Tcut[k][bool_in])/dt)
                if (j==2):
                    mass_flux_nosat[j].append(np.sum(mass_nosat_Tcut[k][bool_out])/dt)
                    metal_flux_nosat[j].append(np.sum(metal_mass_nosat_Tcut[k][bool_out])/dt)
                    kinetic_energy_flux_nosat[j].append(np.sum(kinetic_energy_nosat_Tcut[k][bool_out])/dt)
                    thermal_energy_flux_nosat[j].append(np.sum(thermal_energy_nosat_Tcut[k][bool_out])/dt)
                    potential_energy_flux_nosat[j].append(np.sum(potential_energy_nosat_Tcut[k][bool_out])/dt)
                    entropy_flux_nosat[j].append(np.sum(entropy_nosat_Tcut[k][bool_out])/dt)

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
                    bool_in = (newradius_sat_Tcut[0][k]>inner_r) & (newradius_sat_Tcut[0][k]<outer_r)
                    bool_out = (radius_sat_Tcut[1][k]>inner_r) & (radius_sat_Tcut[1][k]<outer_r)
                    bool_shell = (radius_nosat_Tcut[k]>inner_r) & (radius_nosat_Tcut[k]<outer_r)
                    if (j==0):
                        mass_flux_sat[j].append((np.sum(mass_sat_Tcut[0][k][bool_in]) - \
                                                np.sum(mass_sat_Tcut[1][k][bool_out]))/dt)
                        metal_flux_sat[j].append((np.sum(metal_mass_sat_Tcut[0][k][bool_in]) - \
                                                np.sum(metal_mass_sat_Tcut[1][k][bool_out]))/dt)
                        kinetic_energy_flux_sat[j].append((np.sum(kinetic_energy_sat_Tcut[0][k][bool_in]) - \
                                                          np.sum(kinetic_energy_sat_Tcut[1][k][bool_out]))/dt)
                        thermal_energy_flux_sat[j].append((np.sum(thermal_energy_sat_Tcut[0][k][bool_in]) - \
                                                          np.sum(thermal_energy_sat_Tcut[1][k][bool_out]))/dt)
                        potential_energy_flux_sat[j].append((np.sum(potential_energy_sat_Tcut[0][k][bool_in]) - \
                                                            np.sum(potential_energy_sat_Tcut[1][k][bool_out]))/dt)
                        entropy_flux_sat[j].append((np.sum(entropy_sat_Tcut[0][k][bool_in]) - \
                                                    np.sum(entropy_sat_Tcut[1][k][bool_out]))/dt)
                        radiative_energy_flux_nosat[j].append(np.sum(thermal_energy_nosat_Tcut[k][bool_shell] * \
                          mass_nosat_Tcut[k][bool_shell]*gtoMsun / cooling_time_nosat_Tcut[k][bool_shell]))
                    if (j==1):
                        mass_flux_sat[j].append(np.sum(mass_sat_Tcut[0][k][bool_in])/dt)
                        metal_flux_sat[j].append(np.sum(metal_mass_sat_Tcut[0][k][bool_in])/dt)
                        kinetic_energy_flux_sat[j].append(np.sum(kinetic_energy_sat_Tcut[0][k][bool_in])/dt)
                        thermal_energy_flux_sat[j].append(np.sum(thermal_energy_sat_Tcut[0][k][bool_in])/dt)
                        potential_energy_flux_sat[j].append(np.sum(potential_energy_sat_Tcut[0][k][bool_in])/dt)
                        entropy_flux_sat[j].append(np.sum(entropy_sat_Tcut[0][k][bool_in])/dt)
                        radiative_energy_flux_nosat[j].append(np.sum( \
                          thermal_energy_nosat_Tcut[k][bool_shell & (rad_vel_nosat_Tcut[k]<0.)] * \
                          mass_nosat_Tcut[k][bool_shell & (rad_vel_nosat_Tcut[k]<0.)]*gtoMsun / \
                          cooling_time_nosat_Tcut[k][bool_shell & (rad_vel_nosat_Tcut[k]<0.)]))
                    if (j==2):
                        mass_flux_sat[j].append(-np.sum(mass_sat_Tcut[1][k][bool_out])/dt)
                        metal_flux_sat[j].append(-np.sum(metal_mass_sat_Tcut[1][k][bool_out])/dt)
                        kinetic_energy_flux_sat[j].append(-np.sum(kinetic_energy_sat_Tcut[1][k][bool_out])/dt)
                        thermal_energy_flux_sat[j].append(-np.sum(thermal_energy_sat_Tcut[1][k][bool_out])/dt)
                        potential_energy_flux_sat[j].append(-np.sum(potential_energy_sat_Tcut[1][k][bool_out])/dt)
                        entropy_flux_sat[j].append(-np.sum(entropy_sat_Tcut[1][k][bool_out])/dt)
                        radiative_energy_flux_nosat[j].append(np.sum( \
                          thermal_energy_nosat_Tcut[k][bool_shell & (rad_vel_nosat_Tcut[k]>0.)] * \
                          mass_nosat_Tcut[k][bool_shell & (rad_vel_nosat_Tcut[k]>0.)]*gtoMsun / \
                          cooling_time_nosat_Tcut[k][bool_shell & (rad_vel_nosat_Tcut[k]>0.)]))

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
    fluxes.write(tablename + '_nosat_sphere.hdf5', path='all_data', serialize_meta=True, overwrite=True)
    fluxes_sat.write(tablename + '_sat_sphere.hdf5', path='all_data', serialize_meta=True, overwrite=True)

    return "Fluxes have been calculated for snapshot " + snap + "!"

def calc_fluxes_frustum(ds, snap, zsnap, dt, refine_width_kpc, tablename, surface_args, **kwargs):
    '''This function calculates the fluxes into and out of radial surfaces within a frustum,
    with satellites removed, at a variety of radii. It
    uses the dataset stored in 'ds', which is from time snapshot 'snap', has redshift
    'zsnap', and has width of the refine box in kpc 'refine_width_kpc', the time step between outputs
    'dt', and stores the fluxes in 'tablename'. 'surface_args' gives the properties of the frustum.

    This function calculates the flux as the sum
    of all cells whose velocity and distance from the surface of interest indicate that the gas
    contained in that cell will be displaced across the surface of interest by the next timestep.
    That is, the properties of a cell contribute to the flux if it is no further from the surface of
    interest than v*dt where v is the cell's velocity normal to the surface and dt is the time
    between snapshots, which is dt = 5.38e6 yrs for the DD outputs. It is necessary to compute the
    flux this way if satellites are to be removed because they become 'holes' in the dataset
    and fluxes into/out of those holes need to be accounted for.'''

    Menc_func = kwargs.get('Menc_func', False)
    sat = kwargs.get('sat')
    halo_center_kpc2 = kwargs.get('halo_center_kpc2', ds.halo_center_kpc)

    G = ds.quan(6.673e-8, 'cm**3/s**2/g').v
    halo_center_kpc = ds.halo_center_kpc

    cmtopc = 3.086e18
    stoyr = 3.154e7
    gtoMsun = 1.989e33

    inner_radius = surface_args[3]
    outer_radius = surface_args[4]
    dr = (outer_radius - inner_radius)/surface_args[5]
    op_angle = surface_args[6]
    axis = surface_args[1]
    flip = surface_args[2]

    # Set up table of everything we want
    # NOTE: Make sure table units are updated when things are added to this table!
    fluxes_radial = Table(names=('redshift', 'radius', \
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

    fluxes_edges = Table(names=('redshift', 'inner_radius', 'outer_radius', \
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

    # Define the radii of the surfaces where we want to calculate fluxes
    radii = refine_width_kpc * np.arange(inner_radius, outer_radius+dr, dr)

    # Load arrays of all fields we need
    print('Loading field arrays')
    sphere = ds.sphere(halo_center_kpc, radii[-1])

    radius = sphere['gas','radius_corrected'].in_units('kpc').v
    x = sphere['gas','x'].in_units('kpc').v - halo_center_kpc[0].v
    y = sphere['gas','y'].in_units('kpc').v - halo_center_kpc[1].v
    z = sphere['gas','z'].in_units('kpc').v - halo_center_kpc[2].v
    vx = sphere['gas','vx_corrected'].in_units('km/s').v
    vy = sphere['gas','vy_corrected'].in_units('km/s').v
    vz = sphere['gas','vz_corrected'].in_units('km/s').v
    new_x = x + vx*dt*(100./cmtopc*stoyr)
    new_y = y + vy*dt*(100./cmtopc*stoyr)
    new_z = z + vz*dt*(100./cmtopc*stoyr)
    new_radius = np.sqrt(new_x**2. + new_y**2. + new_z**2.)
    rad_vel = sphere['gas','radial_velocity_corrected'].in_units('km/s').v
    mass = sphere['gas','cell_mass'].in_units('Msun').v
    metal_mass = sphere['gas','metal_mass'].in_units('Msun').v
    temperature = sphere['gas','temperature'].in_units('K').v
    kinetic_energy = sphere['gas','kinetic_energy_corrected'].in_units('erg').v
    thermal_energy = sphere['gas','thermal_energy'].in_units('erg/g').v
    potential_energy = (sphere['gas','cell_mass'] * \
      ds.arr(sphere['enzo','Grav_Potential'].v, 'code_length**2/code_time**2')).in_units('erg').v
    cooling_time = sphere['gas','cooling_time'].in_units('yr').v
    entropy = sphere['gas','entropy'].in_units('keV*cm**2').v

    # Cut data to only the frustum considered here, stuff that leaves through edges of frustum,
    # and stuff that comes in through edges of frustum
    if (flip):
        min_theta = op_angle*np.pi/180.
        max_theta = np.pi
        frus_filename = '-'
    else:
        min_theta = 0.
        max_theta = op_angle*np.pi/180.
        frus_filename = ''
    if (axis=='z'):
        theta = np.arccos(z/radius)
        new_theta = np.arccos(new_z/new_radius)
        phi = np.arctan2(y, x)
        frus_filename += 'z_' + str(op_angle)
    if (axis=='x'):
        theta = np.arccos(x/radius)
        new_theta = np.arccos(new_x/new_radius)
        phi = np.arctan2(z, y)
        frus_filename += 'x_' + str(op_angle)
    if (axis=='y'):
        theta = np.arccos(y/radius)
        new_theta = np.arccos(new_y/new_radius)
        phi = np.arctan2(x, z)
        frus_filename += 'y_' + str(op_angle)
    if (axis=='disk minor axis'):
        x_disk = sphere['gas','x_disk'].in_units('kpc').v
        y_disk = sphere['gas','y_disk'].in_units('kpc').v
        z_disk = sphere['gas','z_disk'].in_units('kpc').v
        vx_disk = sphere['gas','vx_disk'].in_units('km/s').v
        vy_disk = sphere['gas','vy_disk'].in_units('km/s').v
        vz_disk = sphere['gas','vz_disk'].in_units('km/s').v
        new_x_disk = x_disk + vx_disk*dt*(100./cmtopc*stoyr)
        new_y_disk = y_disk + vy_disk*dt*(100./cmtopc*stoyr)
        new_z_disk = z_disk + vz_disk*dt*(100./cmtopc*stoyr)
        theta = np.arccos(z_disk/radius)
        new_theta = np.arccos(new_z_disk/new_radius)
        phi = np.arctan2(y_disk, x_disk)
        frus_filename += 'disk_' + str(op_angle)

    # Load list of satellite positions
    print('Loading satellite positions')
    sat_x = sat['sat_x'][sat['snap']==snap]
    sat_y = sat['sat_y'][sat['snap']==snap]
    sat_z = sat['sat_z'][sat['snap']==snap]
    sat_list = []
    for i in range(len(sat_x)):
        if not ((np.abs(sat_x[i] - halo_center_kpc[0].v) <= 1.) & \
                (np.abs(sat_y[i] - halo_center_kpc[1].v) <= 1.) & \
                (np.abs(sat_z[i] - halo_center_kpc[2].v) <= 1.)):
            sat_list.append([sat_x[i] - halo_center_kpc[0].v, sat_y[i] - halo_center_kpc[1].v, sat_z[i] - halo_center_kpc[2].v])
    sat_list = np.array(sat_list)
    snap2 = int(snap[-4:])+1
    snap_type = snap[-6:-4]
    if (snap2 < 10): snap2 = snap_type + '000' + str(snap2)
    elif (snap2 < 100): snap2 = snap_type + '00' + str(snap2)
    elif (snap2 < 1000): snap2 = snap_type + '0' + str(snap2)
    else: snap2 = snap_type + str(snap2)
    sat_x2 = sat['sat_x'][sat['snap']==snap2]
    sat_y2 = sat['sat_y'][sat['snap']==snap2]
    sat_z2 = sat['sat_z'][sat['snap']==snap2]
    sat_list2 = []
    for i in range(len(sat_x2)):
        if not ((np.abs(sat_x2[i] - halo_center_kpc2[0].v) <= 1.) & \
                (np.abs(sat_y2[i] - halo_center_kpc2[1].v) <= 1.) & \
                (np.abs(sat_z2[i] - halo_center_kpc2[2].v) <= 1.)):
            sat_list2.append([sat_x2[i] - halo_center_kpc2[0].v, sat_y2[i] - halo_center_kpc2[1].v, sat_z2[i] - halo_center_kpc2[2].v])
    sat_list2 = np.array(sat_list2)

    # Cut data to remove anything within satellites and to things that cross into and out of satellites
    # Restrict to only things that start or end within the frustum
    print('Cutting data to remove satellites')
    sat_radius = 10.         # kpc
    sat_radius_sq = sat_radius**2.
    # An attempt to remove satellites faster:
    # Holy cow this is so much faster, do it this way
    bool_inside_sat1 = []
    bool_inside_sat2 = []
    for s in range(len(sat_list)):
        sat_x = sat_list[s][0]
        sat_y = sat_list[s][1]
        sat_z = sat_list[s][2]
        dist_from_sat1_sq = (x-sat_x)**2. + (y-sat_y)**2. + (z-sat_z)**2.
        bool_inside_sat1.append((dist_from_sat1_sq < sat_radius_sq))
    bool_inside_sat1 = np.array(bool_inside_sat1)
    for s2 in range(len(sat_list2)):
        sat_x2 = sat_list2[s2][0]
        sat_y2 = sat_list2[s2][1]
        sat_z2 = sat_list2[s2][2]
        dist_from_sat2_sq = (new_x-sat_x2)**2. + (new_y-sat_y2)**2. + (new_z-sat_z2)**2.
        bool_inside_sat2.append((dist_from_sat2_sq < sat_radius_sq))
    bool_inside_sat2 = np.array(bool_inside_sat2)
    inside_sat1 = np.count_nonzero(bool_inside_sat1, axis=0)
    inside_sat2 = np.count_nonzero(bool_inside_sat2, axis=0)
    # inside_sat1 and inside_sat2 should now both be arrays of length = # of pixels where the value is an
    # integer. If the value is zero, that pixel is not inside any satellites. If the value is > 0,
    # that pixel is in a satellite.
    bool_nosat = (inside_sat1 == 0) & (inside_sat2 == 0)
    bool_fromsat = (inside_sat1 > 0) & (inside_sat2 == 0) & (new_theta >= min_theta) & (new_theta <= max_theta)
    bool_tosat = (inside_sat1 == 0) & (inside_sat2 > 0) & (theta >= min_theta) & (theta <= max_theta)

    radius_nosat = radius[bool_nosat]
    newradius_nosat = new_radius[bool_nosat]
    theta_nosat = theta[bool_nosat]
    newtheta_nosat = new_theta[bool_nosat]
    rad_vel_nosat = rad_vel[bool_nosat]
    temperature_nosat = temperature[bool_nosat]
    mass_nosat = mass[bool_nosat]
    metal_mass_nosat = metal_mass[bool_nosat]
    kinetic_energy_nosat = kinetic_energy[bool_nosat]
    thermal_energy_nosat = thermal_energy[bool_nosat]
    potential_energy_nosat = potential_energy[bool_nosat]
    cooling_time_nosat = cooling_time[bool_nosat]
    entropy_nosat = entropy[bool_nosat]

    # Cut satellite-removed data to frustum of interest
    # These are nested lists where the index goes from 0 to 2 for [within frustum, enterting frustum, leaving frustum]
    bool_frus = (theta_nosat >= min_theta) & (theta_nosat <= max_theta) & (newtheta_nosat >= min_theta) & (newtheta_nosat <= max_theta)
    bool_infrus = ((theta_nosat < min_theta) | (theta_nosat > max_theta)) & ((newtheta_nosat >= min_theta) & (newtheta_nosat <= max_theta))
    bool_outfrus = ((theta_nosat >= min_theta) & (theta_nosat <= max_theta)) & ((newtheta_nosat < min_theta) | (newtheta_nosat > max_theta))

    radius_nosat_frus = []
    newradius_nosat_frus = []
    rad_vel_nosat_frus = []
    mass_nosat_frus = []
    metal_mass_nosat_frus = []
    temperature_nosat_frus = []
    kinetic_energy_nosat_frus = []
    thermal_energy_nosat_frus = []
    potential_energy_nosat_frus = []
    cooling_time_nosat_frus = []
    entropy_nosat_frus = []
    for j in range(3):
        if (j==0):
            radius_nosat_frus.append(radius_nosat[bool_frus])
            newradius_nosat_frus.append(newradius_nosat[bool_frus])
            rad_vel_nosat_frus.append(rad_vel_nosat[bool_frus])
            mass_nosat_frus.append(mass_nosat[bool_frus])
            metal_mass_nosat_frus.append(metal_mass_nosat[bool_frus])
            temperature_nosat_frus.append(temperature_nosat[bool_frus])
            kinetic_energy_nosat_frus.append(kinetic_energy_nosat[bool_frus])
            thermal_energy_nosat_frus.append(thermal_energy_nosat[bool_frus])
            potential_energy_nosat_frus.append(potential_energy_nosat[bool_frus])
            cooling_time_nosat_frus.append(cooling_time_nosat[bool_frus])
            entropy_nosat_frus.append(entropy_nosat[bool_frus])
        if (j==1):
            radius_nosat_frus.append(radius_nosat[bool_infrus])
            newradius_nosat_frus.append(newradius_nosat[bool_infrus])
            rad_vel_nosat_frus.append(rad_vel_nosat[bool_infrus])
            mass_nosat_frus.append(mass_nosat[bool_infrus])
            metal_mass_nosat_frus.append(metal_mass_nosat[bool_infrus])
            temperature_nosat_frus.append(temperature_nosat[bool_infrus])
            kinetic_energy_nosat_frus.append(kinetic_energy_nosat[bool_infrus])
            thermal_energy_nosat_frus.append(thermal_energy_nosat[bool_infrus])
            potential_energy_nosat_frus.append(potential_energy_nosat[bool_infrus])
            cooling_time_nosat_frus.append(cooling_time_nosat[bool_infrus])
            entropy_nosat_frus.append(entropy_nosat[bool_infrus])
        if (j==2):
            radius_nosat_frus.append(radius_nosat[bool_outfrus])
            newradius_nosat_frus.append(newradius_nosat[bool_outfrus])
            rad_vel_nosat_frus.append(rad_vel_nosat[bool_outfrus])
            mass_nosat_frus.append(mass_nosat[bool_outfrus])
            metal_mass_nosat_frus.append(metal_mass_nosat[bool_outfrus])
            temperature_nosat_frus.append(temperature_nosat[bool_outfrus])
            kinetic_energy_nosat_frus.append(kinetic_energy_nosat[bool_outfrus])
            thermal_energy_nosat_frus.append(thermal_energy_nosat[bool_outfrus])
            potential_energy_nosat_frus.append(potential_energy_nosat[bool_outfrus])
            cooling_time_nosat_frus.append(cooling_time_nosat[bool_outfrus])
            entropy_nosat_frus.append(entropy_nosat[bool_outfrus])

    # Cut satellite-removed frustum data on temperature
    # These are lists of lists where the first index goes from 0 to 2 for
    # [within frustum, entering frustum, leaving frustum] and the second index goes from 0 to 4 for
    # [all gas, cold, cool, warm, hot]
    print('Cutting satellite-removed data on temperature')
    radius_nosat_frus_Tcut = []
    rad_vel_nosat_frus_Tcut = []
    newradius_nosat_frus_Tcut = []
    mass_nosat_frus_Tcut = []
    metal_mass_nosat_frus_Tcut = []
    kinetic_energy_nosat_frus_Tcut = []
    thermal_energy_nosat_frus_Tcut = []
    potential_energy_nosat_frus_Tcut = []
    cooling_time_nosat_frus_Tcut = []
    entropy_nosat_frus_Tcut = []
    for i in range(3):
        radius_nosat_frus_Tcut.append([])
        rad_vel_nosat_frus_Tcut.append([])
        newradius_nosat_frus_Tcut.append([])
        mass_nosat_frus_Tcut.append([])
        metal_mass_nosat_frus_Tcut.append([])
        kinetic_energy_nosat_frus_Tcut.append([])
        thermal_energy_nosat_frus_Tcut.append([])
        potential_energy_nosat_frus_Tcut.append([])
        cooling_time_nosat_frus_Tcut.append([])
        entropy_nosat_frus_Tcut.append([])
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
            bool_temp_nosat_frus = (temperature_nosat_frus[i] < t_high) & (temperature_nosat_frus[i] > t_low)
            radius_nosat_frus_Tcut[i].append(radius_nosat_frus[i][bool_temp_nosat_frus])
            rad_vel_nosat_frus_Tcut[i].append(rad_vel_nosat_frus[i][bool_temp_nosat_frus])
            newradius_nosat_frus_Tcut[i].append(newradius_nosat_frus[i][bool_temp_nosat_frus])
            mass_nosat_frus_Tcut[i].append(mass_nosat_frus[i][bool_temp_nosat_frus])
            metal_mass_nosat_frus_Tcut[i].append(metal_mass_nosat_frus[i][bool_temp_nosat_frus])
            kinetic_energy_nosat_frus_Tcut[i].append(kinetic_energy_nosat_frus[i][bool_temp_nosat_frus])
            thermal_energy_nosat_frus_Tcut[i].append(thermal_energy_nosat_frus[i][bool_temp_nosat_frus])
            potential_energy_nosat_frus_Tcut[i].append(potential_energy_nosat_frus[i][bool_temp_nosat_frus])
            cooling_time_nosat_frus_Tcut[i].append(cooling_time_nosat_frus[i][bool_temp_nosat_frus])
            entropy_nosat_frus_Tcut[i].append(entropy_nosat_frus[i][bool_temp_nosat_frus])

    # Cut data to things that cross into or out of satellites in the frustum
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

    # Cut stuff going into/out of satellites in the frustum on temperature
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
            bool_temp_sat = (temperature_sat[i] > t_low) & (temperature_sat[i] < t_high)
            radius_sat_Tcut[i].append(radius_sat[i][bool_temp_sat])
            newradius_sat_Tcut[i].append(newradius_sat[i][bool_temp_sat])
            mass_sat_Tcut[i].append(mass_sat[i][bool_temp_sat])
            metal_mass_sat_Tcut[i].append(metal_mass_sat[i][bool_temp_sat])
            kinetic_energy_sat_Tcut[i].append(kinetic_energy_sat[i][bool_temp_sat])
            thermal_energy_sat_Tcut[i].append(thermal_energy_sat[i][bool_temp_sat])
            potential_energy_sat_Tcut[i].append(potential_energy_sat[i][bool_temp_sat])
            entropy_sat_Tcut[i].append(entropy_sat[i][bool_temp_sat])

    # Loop over radii
    for i in range(len(radii)):
        inner_r = radii[i].v
        if (i < len(radii) - 1): outer_r = radii[i+1].v

        if (i%10==0): print("Computing radius " + str(i) + "/" + str(len(radii)) + \
                            " for snapshot " + snap)

        # Compute net, in, and out fluxes within the frustum with satellites removed
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
                bool_in_r = (radius_nosat_frus_Tcut[0][k] > inner_r) & (newradius_nosat_frus_Tcut[0][k] < inner_r)
                bool_out_r = (radius_nosat_frus_Tcut[0][k] < inner_r) & (newradius_nosat_frus_Tcut[0][k] > inner_r)
                if (j==0):
                    mass_flux_nosat[j].append((np.sum(mass_nosat_frus_Tcut[0][k][bool_out_r]) - \
                      np.sum(mass_nosat_frus_Tcut[0][k][bool_in_r]))/dt)
                    metal_flux_nosat[j].append((np.sum(metal_mass_nosat_frus_Tcut[0][k][bool_out_r]) - \
                      np.sum(metal_mass_nosat_frus_Tcut[0][k][bool_in_r]))/dt)
                    kinetic_energy_flux_nosat[j].append((np.sum(kinetic_energy_nosat_frus_Tcut[0][k][bool_out_r]) - \
                      np.sum(kinetic_energy_nosat_frus_Tcut[0][k][bool_in_r]))/dt)
                    thermal_energy_flux_nosat[j].append((np.sum(thermal_energy_nosat_frus_Tcut[0][k][bool_out_r]) - \
                      np.sum(thermal_energy_nosat_frus_Tcut[0][k][bool_in_r]))/dt)
                    potential_energy_flux_nosat[j].append((np.sum(potential_energy_nosat_frus_Tcut[0][k][bool_out_r]) - \
                      np.sum(potential_energy_nosat_frus_Tcut[0][k][bool_in_r]))/dt)
                    entropy_flux_nosat[j].append((np.sum(entropy_nosat_frus_Tcut[0][k][bool_out_r]) - \
                    np.sum(entropy_nosat_frus_Tcut[0][k][bool_in_r]))/dt)
                if (j==1):
                    mass_flux_nosat[j].append(-np.sum(mass_nosat_frus_Tcut[0][k][bool_in_r])/dt)
                    metal_flux_nosat[j].append(-np.sum(metal_mass_nosat_frus_Tcut[0][k][bool_in_r])/dt)
                    kinetic_energy_flux_nosat[j].append(-np.sum(kinetic_energy_nosat_frus_Tcut[0][k][bool_in_r])/dt)
                    thermal_energy_flux_nosat[j].append(-np.sum(thermal_energy_nosat_frus_Tcut[0][k][bool_in_r])/dt)
                    potential_energy_flux_nosat[j].append(-np.sum(potential_energy_nosat_frus_Tcut[0][k][bool_in_r])/dt)
                    entropy_flux_nosat[j].append(-np.sum(entropy_nosat_frus_Tcut[0][k][bool_in_r])/dt)
                if (j==2):
                    mass_flux_nosat[j].append(np.sum(mass_nosat_frus_Tcut[0][k][bool_out_r])/dt)
                    metal_flux_nosat[j].append(np.sum(metal_mass_nosat_frus_Tcut[0][k][bool_out_r])/dt)
                    kinetic_energy_flux_nosat[j].append(np.sum(kinetic_energy_nosat_frus_Tcut[0][k][bool_out_r])/dt)
                    thermal_energy_flux_nosat[j].append(np.sum(thermal_energy_nosat_frus_Tcut[0][k][bool_out_r])/dt)
                    potential_energy_flux_nosat[j].append(np.sum(potential_energy_nosat_frus_Tcut[0][k][bool_out_r])/dt)
                    entropy_flux_nosat[j].append(np.sum(entropy_nosat_frus_Tcut[0][k][bool_out_r])/dt)

        # Compute fluxes from and to satellites (and net) within the frustum between inner_r and outer_r
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
                    bool_from = (newradius_sat_Tcut[0][k]>inner_r) & (newradius_sat_Tcut[0][k]<outer_r)
                    bool_to = (radius_sat_Tcut[1][k]>inner_r) & (radius_sat_Tcut[1][k]<outer_r)
                    bool_r = (radius_nosat_frus_Tcut[0][k]>inner_r) & (radius_nosat_frus_Tcut[0][k]<outer_r)
                    if (j==0):
                        mass_flux_sat[j].append((np.sum(mass_sat_Tcut[0][k][bool_from]) - \
                                                np.sum(mass_sat_Tcut[1][k][bool_to]))/dt)
                        metal_flux_sat[j].append((np.sum(metal_mass_sat_Tcut[0][k][bool_from]) - \
                                                np.sum(metal_mass_sat_Tcut[1][k][bool_to]))/dt)
                        kinetic_energy_flux_sat[j].append((np.sum(kinetic_energy_sat_Tcut[0][k][bool_from]) - \
                                                          np.sum(kinetic_energy_sat_Tcut[1][k][bool_to]))/dt)
                        thermal_energy_flux_sat[j].append((np.sum(thermal_energy_sat_Tcut[0][k][bool_from]) - \
                                                          np.sum(thermal_energy_sat_Tcut[1][k][bool_to]))/dt)
                        potential_energy_flux_sat[j].append((np.sum(potential_energy_sat_Tcut[0][k][bool_from]) - \
                                                            np.sum(potential_energy_sat_Tcut[1][k][bool_to]))/dt)
                        entropy_flux_sat[j].append((np.sum(entropy_sat_Tcut[0][k][bool_from]) - \
                                                    np.sum(entropy_sat_Tcut[1][k][bool_to]))/dt)
                        radiative_energy_flux_nosat[j].append(np.sum(thermal_energy_nosat_frus_Tcut[0][k][bool_r] * \
                          mass_nosat_frus_Tcut[0][k][bool_r]*gtoMsun /cooling_time_nosat_frus_Tcut[0][k][bool_r]))
                    if (j==1):
                        mass_flux_sat[j].append(np.sum(mass_sat_Tcut[0][k][bool_from])/dt)
                        metal_flux_sat[j].append(np.sum(metal_mass_sat_Tcut[0][k][bool_from])/dt)
                        kinetic_energy_flux_sat[j].append(np.sum(kinetic_energy_sat_Tcut[0][k][bool_from])/dt)
                        thermal_energy_flux_sat[j].append(np.sum(thermal_energy_sat_Tcut[0][k][bool_from])/dt)
                        potential_energy_flux_sat[j].append(np.sum(potential_energy_sat_Tcut[0][k][bool_from])/dt)
                        entropy_flux_sat[j].append(np.sum(entropy_sat_Tcut[0][k][bool_from])/dt)
                        radiative_energy_flux_nosat[j].append(np.sum( \
                          thermal_energy_nosat_frus_Tcut[0][k][bool_r & (rad_vel_nosat_frus_Tcut[0][k]<0.)] * \
                          mass_nosat_frus_Tcut[0][k][bool_r & (rad_vel_nosat_frus_Tcut[0][k]<0.)]*gtoMsun / \
                          cooling_time_nosat_frus_Tcut[0][k][bool_r & (rad_vel_nosat_frus_Tcut[0][k]<0.)]))
                    if (j==2):
                        mass_flux_sat[j].append(-np.sum(mass_sat_Tcut[1][k][bool_to])/dt)
                        metal_flux_sat[j].append(-np.sum(metal_mass_sat_Tcut[1][k][bool_to])/dt)
                        kinetic_energy_flux_sat[j].append(-np.sum(kinetic_energy_sat_Tcut[1][k][bool_to])/dt)
                        thermal_energy_flux_sat[j].append(-np.sum(thermal_energy_sat_Tcut[1][k][bool_to])/dt)
                        potential_energy_flux_sat[j].append(-np.sum(potential_energy_sat_Tcut[1][k][bool_to])/dt)
                        entropy_flux_sat[j].append(-np.sum(entropy_sat_Tcut[1][k][bool_to])/dt)
                        radiative_energy_flux_nosat[j].append(np.sum( \
                          thermal_energy_nosat_frus_Tcut[0][k][bool_r & (rad_vel_nosat_frus_Tcut[0][k]>0.)] * \
                          mass_nosat_frus_Tcut[0][k][bool_r & (rad_vel_nosat_frus_Tcut[0][k]>0.)]*gtoMsun / \
                          cooling_time_nosat_frus_Tcut[0][k][bool_r & (rad_vel_nosat_frus_Tcut[0][k]>0.)]))

        # Compute fluxes through edges of frustum between inner_r and outer_r
        # These are nested lists where the first index goes from 0 to 2 for [net, in, out]
        # and the second index goes from 0 to 4 for [all, cold, cool, warm, hot]
        if (i < len(radii)-1):
            mass_flux_edge = []
            metal_flux_edge = []
            kinetic_energy_flux_edge = []
            thermal_energy_flux_edge = []
            potential_energy_flux_edge = []
            entropy_flux_edge = []
            for j in range(3):
                mass_flux_edge.append([])
                metal_flux_edge.append([])
                kinetic_energy_flux_edge.append([])
                thermal_energy_flux_edge.append([])
                potential_energy_flux_edge.append([])
                entropy_flux_edge.append([])
                for k in range(5):
                    bool_in = (newradius_nosat_frus_Tcut[1][k]>inner_r) & (newradius_nosat_frus_Tcut[1][k]<outer_r)
                    bool_out = (radius_nosat_frus_Tcut[2][k]>inner_r) & (radius_nosat_frus_Tcut[2][k]<outer_r)
                    if (j==0):
                        mass_flux_edge[j].append((np.sum(mass_nosat_frus_Tcut[1][k][bool_in]) - \
                                                np.sum(mass_nosat_frus_Tcut[2][k][bool_out]))/dt)
                        metal_flux_edge[j].append((np.sum(metal_mass_nosat_frus_Tcut[1][k][bool_in]) - \
                                                np.sum(metal_mass_nosat_frus_Tcut[2][k][bool_out]))/dt)
                        kinetic_energy_flux_edge[j].append((np.sum(kinetic_energy_nosat_frus_Tcut[1][k][bool_in]) - \
                                                          np.sum(kinetic_energy_nosat_frus_Tcut[2][k][bool_out]))/dt)
                        thermal_energy_flux_edge[j].append((np.sum(thermal_energy_nosat_frus_Tcut[1][k][bool_in]) - \
                                                          np.sum(thermal_energy_nosat_frus_Tcut[2][k][bool_out]))/dt)
                        potential_energy_flux_edge[j].append((np.sum(potential_energy_nosat_frus_Tcut[1][k][bool_in]) - \
                                                            np.sum(potential_energy_nosat_frus_Tcut[2][k][bool_out]))/dt)
                        entropy_flux_edge[j].append((np.sum(entropy_nosat_frus_Tcut[1][k][bool_in]) - \
                                                    np.sum(entropy_nosat_frus_Tcut[2][k][bool_out]))/dt)
                    if (j==1):
                        mass_flux_edge[j].append(np.sum(mass_nosat_frus_Tcut[1][k][bool_in])/dt)
                        metal_flux_edge[j].append(np.sum(metal_mass_nosat_frus_Tcut[1][k][bool_in])/dt)
                        kinetic_energy_flux_edge[j].append(np.sum(kinetic_energy_nosat_frus_Tcut[1][k][bool_in])/dt)
                        thermal_energy_flux_edge[j].append(np.sum(thermal_energy_nosat_frus_Tcut[1][k][bool_in])/dt)
                        potential_energy_flux_edge[j].append(np.sum(potential_energy_nosat_frus_Tcut[1][k][bool_in])/dt)
                        entropy_flux_edge[j].append(np.sum(entropy_nosat_frus_Tcut[1][k][bool_in])/dt)
                    if (j==2):
                        mass_flux_edge[j].append(-np.sum(mass_nosat_frus_Tcut[2][k][bool_out])/dt)
                        metal_flux_edge[j].append(-np.sum(metal_mass_nosat_frus_Tcut[2][k][bool_out])/dt)
                        kinetic_energy_flux_edge[j].append(-np.sum(kinetic_energy_nosat_frus_Tcut[2][k][bool_out])/dt)
                        thermal_energy_flux_edge[j].append(-np.sum(thermal_energy_nosat_frus_Tcut[2][k][bool_out])/dt)
                        potential_energy_flux_edge[j].append(-np.sum(potential_energy_nosat_frus_Tcut[2][k][bool_out])/dt)
                        entropy_flux_edge[j].append(-np.sum(entropy_nosat_frus_Tcut[2][k][bool_out])/dt)

        # Add everything to the tables
        fluxes_radial.add_row([zsnap, inner_r, \
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

        fluxes_edges.add_row([zsnap, inner_r, outer_r, \
                            mass_flux_edge[0][0], metal_flux_edge[0][0], \
                            mass_flux_edge[1][0], mass_flux_edge[2][0], metal_flux_edge[1][0], metal_flux_edge[2][0], \
                            mass_flux_edge[0][1], mass_flux_edge[1][1], mass_flux_edge[2][1], \
                            mass_flux_edge[0][2], mass_flux_edge[1][2], mass_flux_edge[2][2], \
                            mass_flux_edge[0][3], mass_flux_edge[1][3], mass_flux_edge[2][3], \
                            mass_flux_edge[0][4], mass_flux_edge[1][4], mass_flux_edge[2][4], \
                            metal_flux_edge[0][1], metal_flux_edge[1][1], metal_flux_edge[2][1], \
                            metal_flux_edge[0][2], metal_flux_edge[1][2], metal_flux_edge[2][2], \
                            metal_flux_edge[0][3], metal_flux_edge[1][3], metal_flux_edge[2][3], \
                            metal_flux_edge[0][4], metal_flux_edge[1][4], metal_flux_edge[2][4], \
                            kinetic_energy_flux_edge[0][0], thermal_energy_flux_edge[0][0], \
                            potential_energy_flux_edge[0][0], entropy_flux_edge[0][0], \
                            kinetic_energy_flux_edge[1][0], kinetic_energy_flux_edge[2][0], \
                            thermal_energy_flux_edge[1][0], thermal_energy_flux_edge[2][0], \
                            potential_energy_flux_edge[1][0], potential_energy_flux_edge[2][0], \
                            entropy_flux_edge[1][0], entropy_flux_edge[2][0], \
                            kinetic_energy_flux_edge[0][1], kinetic_energy_flux_edge[1][1], kinetic_energy_flux_edge[2][1], \
                            kinetic_energy_flux_edge[0][2], kinetic_energy_flux_edge[1][2], kinetic_energy_flux_edge[2][2], \
                            kinetic_energy_flux_edge[0][3], kinetic_energy_flux_edge[1][3], kinetic_energy_flux_edge[2][3], \
                            kinetic_energy_flux_edge[0][4], kinetic_energy_flux_edge[1][4], kinetic_energy_flux_edge[2][4], \
                            thermal_energy_flux_edge[0][1], thermal_energy_flux_edge[1][1], thermal_energy_flux_edge[2][1], \
                            thermal_energy_flux_edge[0][2], thermal_energy_flux_edge[1][2], thermal_energy_flux_edge[2][2], \
                            thermal_energy_flux_edge[0][3], thermal_energy_flux_edge[1][3], thermal_energy_flux_edge[2][3], \
                            thermal_energy_flux_edge[0][4], thermal_energy_flux_edge[1][4], thermal_energy_flux_edge[2][4], \
                            potential_energy_flux_edge[0][1], potential_energy_flux_edge[1][1], potential_energy_flux_edge[2][1], \
                            potential_energy_flux_edge[0][2], potential_energy_flux_edge[1][2], potential_energy_flux_edge[2][2], \
                            potential_energy_flux_edge[0][3], potential_energy_flux_edge[1][3], potential_energy_flux_edge[2][3], \
                            potential_energy_flux_edge[0][4], potential_energy_flux_edge[1][4], potential_energy_flux_edge[2][4], \
                            entropy_flux_edge[0][1], entropy_flux_edge[1][1], entropy_flux_edge[2][1], \
                            entropy_flux_edge[0][2], entropy_flux_edge[1][2], entropy_flux_edge[2][2], \
                            entropy_flux_edge[0][3], entropy_flux_edge[1][3], entropy_flux_edge[2][3], \
                            entropy_flux_edge[0][4], entropy_flux_edge[1][4], entropy_flux_edge[2][4]])

    fluxes_radial = set_table_units(fluxes_radial)
    fluxes_sat = set_table_units(fluxes_sat)
    fluxes_edges = set_table_units(fluxes_edges)

    # Save to file
    fluxes_radial.write(tablename + '_nosat_frustum_' + frus_filename + '.hdf5', path='all_data', serialize_meta=True, overwrite=True)
    fluxes_sat.write(tablename + '_sat_frustum_' + frus_filename + '.hdf5', path='all_data', serialize_meta=True, overwrite=True)
    fluxes_edges.write(tablename + '_edges_frustum_' + frus_filename + '.hdf5', path='all_data', serialize_meta=True, overwrite=True)

    return "Fluxes have been calculated for snapshot " + snap + "!"

def load_and_calculate(system, foggie_dir, run_dir, track, halo_c_v_name, snap, tablename, Menc_table, surface_args, sat_dir):
    '''This function loads a specified snapshot 'snap' located in the 'run_dir' within the
    'foggie_dir', the halo track 'track', the name of the halo_c_v file, the name of the snapshot,
    the name of the table to output, the mass enclosed table, the list of surface arguments, and
    the directory where the satellites file is saved, then
    does the calculation on the loaded snapshot.'''

    snap_name = foggie_dir + run_dir + snap + '/' + snap
    if (system=='pleiades_cassi'):
        print('Copying directory to /tmp')
        snap_dir = '/tmp/' + snap
        shutil.copytree(foggie_dir + run_dir + snap, snap_dir)
        snap_name = snap_dir + '/' + snap
    if (surface_args[0]=='frustum') and (surface_args[1]=='disk minor axis'):
        ds, refine_box, refine_box_center, refine_width = load(snap_name, track, use_halo_c_v=True, \
          halo_c_v_name=halo_c_v_name, disk_relative=True)
    else:
        ds, refine_box, refine_box_center, refine_width = load(snap_name, track, use_halo_c_v=True, \
          halo_c_v_name=halo_c_v_name, filter_particles=False)
    refine_width_kpc = YTArray([refine_width], 'kpc')
    zsnap = ds.get_parameter('CosmologyCurrentRedshift')
    dt = 5.38e6

    # Make interpolated Menc_func using the table at this snapshot
    Menc_func = IUS(Menc_table['radius'][Menc_table['snapshot']==snap], \
      Menc_table['total_mass'][Menc_table['snapshot']==snap])

    # Specify the file where the list of satellites is saved
    sat_file = sat_dir + 'satellites.hdf5'
    sat = Table.read(sat_file, path='all_data')
    # Load halo center for second snapshot
    snap2 = int(snap[-4:])+1
    snap_type = snap[-6:-4]
    if (snap2 < 10): snap2 = snap_type + '000' + str(snap2)
    elif (snap2 < 100): snap2 = snap_type + '00' + str(snap2)
    elif (snap2 < 1000): snap2 = snap_type + '0' + str(snap2)
    else: snap2 = snap_type + str(snap2)
    halo_c_v = Table.read(halo_c_v_name, format='ascii')
    halo_ind = np.where(halo_c_v['col3']==snap2)[0][0]
    halo_center_kpc2 = ds.arr([float(halo_c_v['col4'][halo_ind]), \
                              float(halo_c_v['col5'][halo_ind]), \
                              float(halo_c_v['col6'][halo_ind])], 'kpc')

    # Do the actual calculation
    #message = calc_fluxes(ds, snap, zsnap, refine_width_kpc, tablename, Menc_func=Menc_func)
    if (surface_args[0]=='sphere'):
        message = calc_fluxes_sphere(ds, snap, zsnap, dt, refine_width_kpc, tablename, surface_args, \
          Menc_func=Menc_func, sat=sat, halo_center_kpc2=halo_center_kpc2)
    if (surface_args[0]=='frustum'):
        message = calc_fluxes_frustum(ds, snap, zsnap, dt, refine_width_kpc, tablename, surface_args, \
          Menc_func=Menc_func, sat=sat, halo_center_kpc2=halo_center_kpc2)
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

    surface_args = ast.literal_eval(args.surface)
    if (surface_args[0]=='sphere'):
        print('Sphere arguments: inner_radius - %.3f outer_radius - %.3f num_radius - %d' % \
          (surface_args[1], surface_args[2], surface_args[3]))
    elif (surface_args[0]=='frustum'):
        if (surface_args[1]==1):
            axis = 'x'
            flip = False
        elif (surface_args[1]==2):
            axis = 'y'
            flip = False
        elif (surface_args[1]==3):
            axis = 'z'
            flip = False
        elif (surface_args[1]==4):
            axis = 'disk minor axis'
            flip = False
        elif (surface_args[1]==-1):
            axis = 'x'
            flip = True
        elif (surface_args[1]==-2):
            axis = 'y'
            flip = True
        elif (surface_args[1]==-3):
            axis = 'z'
            flip = True
        elif (surface_args[1]==-4):
            axis = 'disk minor axis'
            flip = True
        else: sys.exit("I don't understand what axis you want.")
        surface_args = [surface_args[0], axis, flip, surface_args[2], surface_args[3], surface_args[4], surface_args[5]]
        if (flip):
            print('Frustum arguments: axis - flipped %s inner_radius - %.3f outer_radius - %.3f num_radius - %d opening_angle - %d' % \
              (axis, surface_args[3], surface_args[4], surface_args[5], surface_args[6]))
        else:
            print('Frustum arguments: axis - %s inner_radius - %.3f outer_radius - %.3f num_radius - %d opening_angle - %d' % \
              (axis, surface_args[3], surface_args[4], surface_args[5], surface_args[6]))
    else:
        sys.exit("That surface has not been implemented. Ask Cassi to add it.")

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
    prefix = output_dir + 'fluxes_halo_00' + args.halo + '/' + args.run + '/'
    if not (os.path.exists(prefix)): os.system('mkdir -p ' + prefix)

    print('foggie_dir: ', foggie_dir)
    halo_c_v_name = track_dir + 'halo_c_v'

    # Load the mass enclosed profile
    Menc_table = Table.read(code_path + 'halo_infos/00' + args.halo + '/' + args.run + '/masses.hdf5', \
      path='all_data')

    # Specify where satellite files are saved
    sat_dir = code_path + 'halo_infos/00' + args.halo + '/' + args.run + '/'

    # Loop over outputs, for either single-processor or parallel processor computing
    if (args.nproc==1):
        for i in range(len(outs)):
            snap = outs[i]
            # Make the output table name for this snapshot
            tablename = prefix + snap + '_fluxes'
            # Do the actual calculation
            load_and_calculate(args.system, foggie_dir, run_dir, trackname, halo_c_v_name, snap, tablename, Menc_table, surface_args, sat_dir)
    else:
        # Split into a number of groupings equal to the number of processors
        # and run one process per processor
        for i in range(len(outs)//args.nproc):
            threads = []
            for j in range(args.nproc):
                snap = outs[args.nproc*i+j]
                tablename = prefix + snap + '_fluxes'
                threads.append(multi.Process(target=load_and_calculate, \
			       args=(args.system, foggie_dir, run_dir, trackname, halo_c_v_name, snap, tablename, Menc_table, surface_args, sat_dir)))
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
			   args=(args.system, foggie_dir, run_dir, trackname, halo_c_v_name, snap, tablename, Menc_table, surface_args, sat_dir)))
        for t in threads:
            t.start()
        for t in threads:
            t.join()

    print(str(datetime.datetime.now()))
    print("All snapshots finished!")
