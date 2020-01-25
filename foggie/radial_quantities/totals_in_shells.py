"""
Filename: flux_tracking.py
Author: Cassi
Date created: 9-27-19
Date last modified: 1-22-20
This file takes command line arguments and computes totals of gas properties in volumes.

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
                        help='Are the simulation files stored locally? Default is no')
    parser.set_defaults(local=False)

    parser.add_argument('--surface', metavar='surface', type=str, action='store', \
                        help='What surface type for computing the totals? Default is sphere' + \
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

    table_units = {'redshift':None,'inner_radius':'kpc','outer_radius':'kpc', \
             'net_mass':'Msun', 'net_metals':'Msun', \
             'mass_in':'Msun', 'mass_out':'Msun', \
             'metals_in' :'Msun', 'metals_out':'Msun',\
             'net_cold_mass':'Msun', 'cold_mass_in':'Msun', \
             'cold_mass_out':'Msun', 'net_cool_mass':'Msun', \
             'cool_mass_in':'Msun', 'cool_mass_out':'Msun', \
             'net_warm_mass':'Msun', 'warm_mass_in':'Msun', \
             'warm_mass_out':'Msun', 'net_hot_mass' :'Msun', \
             'hot_mass_in' :'Msun', 'hot_mass_out' :'Msun', \
             'net_cold_metals':'Msun', 'cold_metals_in':'Msun', \
             'cold_metals_out':'Msun', 'net_cool_metals':'Msun', \
             'cool_metals_in':'Msun', 'cool_metals_out':'Msun', \
             'net_warm_metals':'Msun', 'warm_metals_in':'Msun', \
             'warm_metals_out':'Msun', 'net_hot_metals' :'Msun', \
             'hot_metals_in' :'Msun', 'hot_metals_out' :'Msun', \
             'net_kinetic_energy':'erg', 'net_thermal_energy':'erg', \
             'net_potential_energy':'erg', 'net_entropy':'cm**2*keV', \
             'kinetic_energy_in':'erg', 'kinetic_energy_out':'erg', \
             'thermal_energy_in':'erg', 'thermal_energy_out':'erg', \
             'potential_energy_in':'erg', 'potential_energy_out':'erg', \
             'entropy_in':'cm**2*keV', 'entropy_out':'cm**2*keV', \
             'net_cold_kinetic_energy':'erg', 'cold_kinetic_energy_in':'erg', 'cold_kinetic_energy_out':'erg', \
             'net_cool_kinetic_energy':'erg', 'cool_kinetic_energy_in':'erg', 'cool_kinetic_energy_out':'erg', \
             'net_warm_kinetic_energy':'erg', 'warm_kinetic_energy_in':'erg', 'warm_kinetic_energy_out':'erg', \
             'net_hot_kinetic_energy':'erg', 'hot_kinetic_energy_in':'erg', 'hot_kinetic_energy_out':'erg', \
             'net_cold_thermal_energy':'erg', 'cold_thermal_energy_in':'erg', 'cold_thermal_energy_out':'erg', \
             'net_cool_thermal_energy':'erg', 'cool_thermal_energy_in':'erg', 'cool_thermal_energy_out':'erg', \
             'net_warm_thermal_energy':'erg', 'warm_thermal_energy_in':'erg', 'warm_thermal_energy_out':'erg', \
             'net_hot_thermal_energy':'erg', 'hot_thermal_energy_in':'erg', 'hot_thermal_energy_out':'erg', \
             'net_cold_potential_energy':'erg', 'cold_potential_energy_in':'erg', 'cold_potential_energy_out':'erg', \
             'net_cool_potential_energy':'erg', 'cool_potential_energy_in':'erg', 'cool_potential_energy_out':'erg', \
             'net_warm_potential_energy':'erg', 'warm_potential_energy_in':'erg', 'warm_potential_energy_out':'erg', \
             'net_hot_potential_energy':'erg', 'hot_potential_energy_in':'erg', 'hot_potential_energy_out':'erg', \
             'net_cold_entropy':'cm**2*keV', 'cold_entropy_in':'cm**2*keV', 'cold_entropy_out':'cm**2*keV', \
             'net_cool_entropy':'cm**2*keV', 'cool_entropy_in':'cm**2*keV', 'cool_entropy_out':'cm**2*keV', \
             'net_warm_entropy':'cm**2*keV', 'warm_entropy_in':'cm**2*keV', 'warm_entropy_out':'cm**2*keV', \
             'net_hot_entropy':'cm**2*keV', 'hot_entropy_in':'cm**2*keV', 'hot_entropy_out':'cm**2*keV'}
    for key in table.keys():
        table[key].unit = table_units[key]
    return table

def calc_totals(ds, snap, zsnap, refine_width_kpc, tablename, **kwargs):
    """Computes the total in spherical shells of various things centered on the halo center.
    Takes the dataset for the snapshot 'ds', the name of the snapshot 'snap', the redshfit of the
    snapshot 'zsnap', and the width of the refine box in kpc 'refine_width_kpc'
    and does the calculation, then writes a hdf5 table out to 'tablename'.

    Optional arguments:
    quadrants = True will calculate the totals in shells within quadrants rather than the whole domain,
        default is False. If this is selected, a second table will be written with '_q' appended
        to 'tablename'. This functionality hasn't been updated in a while, may not work.
    """

    quadrants = kwargs.get('quadrants', False)
    Menc_func = kwargs.get('Menc_func', False)

    G = ds.quan(6.673e-8, 'cm**3/s**2/g')
    halo_center_kpc = ds.halo_center_kpc

    cmtopc = 3.086e18
    stoyr = 3.154e7
    gtoMsun = 1.989e33

    # Set up table of everything we want
    # NOTE: Make sure table units are updated when things are added to this table!
    data = Table(names=('redshift', 'quadrant', 'radius', \
                        'net_mass', 'net_metals', \
                        'mass_in', 'mass_out', 'metals_in', 'metals_out', \
                        'net_cold_mass', 'cold_mass_in', 'cold_mass_out', \
                        'net_cool_mass', 'cool_mass_in', 'cool_mass_out', \
                        'net_warm_mass', 'warm_mass_in', 'warm_mass_out', \
                        'net_hot_mass', 'hot_mass_in', 'hot_mass_out', \
                        'net_cold_metals', 'cold_metals_in', 'cold_metals_out', \
                        'net_cool_metals', 'cool_metals_in', 'cool_metals_out', \
                        'net_warm_metals', 'warm_metals_in', 'warm_metals_out', \
                        'net_hot_metals', 'hot_metals_in', 'hot_metals_out', \
                        'net_kinetic_energy', 'net_thermal_energy', 'net_potential_energy', \
                        'net_entropy', \
                        'kinetic_energy_in', 'kinetic_energy_out', \
                        'thermal_energy_in', 'thermal_energy_out', \
                        'potential_energy_in', 'potential_energy_out', \
                        'entropy_in', 'entropy_out', \
                        'net_cold_kinetic_energy', 'cold_kinetic_energy_in', 'cold_kinetic_energy_out', \
                        'net_cool_kinetic_energy', 'cool_kinetic_energy_in', 'cool_kinetic_energy_out', \
                        'net_warm_kinetic_energy', 'warm_kinetic_energy_in', 'warm_kinetic_energy_out', \
                        'net_hot_kinetic_energy', 'hot_kinetic_energy_in', 'hot_kinetic_energy_out', \
                        'net_cold_thermal_energy', 'cold_thermal_energy_in', 'cold_thermal_energy_out', \
                        'net_cool_thermal_energy', 'cool_thermal_energy_in', 'cool_thermal_energy_out', \
                        'net_warm_thermal_energy','warm_thermal_energy_in', 'warm_thermal_energy_out', \
                        'net_hot_thermal_energy', 'hot_thermal_energy_in', 'hot_thermal_energy_out', \
                        'net_cold_potential_energy', 'cold_potential_energy_in', 'cold_potential_energy_out', \
                        'net_cool_potential_energy', 'cool_potential_energy_in', 'cool_potential_energy_out', \
                        'net_warm_potential_energy','warm_potential_energy_in', 'warm_potential_energy_out', \
                        'net_hot_potential_energy', 'hot_potential_energy_in', 'hot_potential_energy_out', \
                        'net_cold_entropy', 'cold_entropy_in', 'cold_entropy_out', \
                        'net_cool_entropy', 'cool_entropy_in', 'cool_entropy_out', \
                        'net_warm_entropy', 'warm_entropy_in', 'warm_entropy_out', \
                        'net_hot_entropy', 'hot_entropy_in', 'hot_entropy_out'), \
                 dtype=('f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', \
                        'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', \
                        'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', \
                        'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', \
                        'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', \
                        'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', \
                        'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', \
                        'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8'))

    if (quadrants):
        data_q = Table(names=('redshift', 'quadrant', 'radius', 'net_mass', 'net_metals', \
                            'mass_in', 'mass_out', 'metals_in', 'metals_out', \
                            'net_cold_mass', 'cold_mass_in', 'cold_mass_out', \
                            'net_cool_mass', 'cool_mass_in', 'cool_mass_out', \
                            'net_warm_mass', 'warm_mass_in', 'warm_mass_out', \
                            'net_hot_mass', 'hot_mass_in', 'hot_mass_out', \
                            'net_cold_metals', 'cold_metals_in', 'cold_metals_out', \
                            'net_cool_metals', 'cool_metals_in', 'cool_metals_out', \
                            'net_warm_metals', 'warm_metals_in', 'warm_metals_out', \
                            'net_hot_metals', 'hot_metals_in', 'hot_metals_out', \
                            'net_kinetic_energy', 'net_thermal_energy', 'net_entropy', \
                            'kinetic_energy_in', 'kinetic_energy_out', \
                            'thermal_energy_in', 'thermal_energy_out', \
                            'entropy_in', 'entropy_out', 'net_cold_kinetic_energy', \
                            'cold_kinetic_energy_in', 'cold_kinetic_energy_out', \
                            'net_cool_kinetic_energy', 'cool_kinetic_energy_in', \
                            'cool_kinetic_energy_out', 'net_warm_kinetic_energy', \
                            'warm_kinetic_energy_in', 'warm_kinetic_energy_out', \
                            'net_hot_kinetic_energy', 'hot_kinetic_energy_in', \
                            'hot_kinetic_energy_out', 'net_cold_thermal_energy', \
                            'cold_thermal_energy_in', 'cold_thermal_energy_out', \
                            'net_cool_thermal_energy', 'cool_thermal_energy_in', \
                            'cool_thermal_energy_out', 'net_warm_thermal_energy', \
                            'warm_thermal_energy_in', 'warm_thermal_energy_out', \
                            'net_hot_thermal_energy', 'hot_thermal_energy_in', \
                            'hot_thermal_energy_out', 'net_cold_entropy', \
                            'cold_entropy_in', 'cold_entropy_out', \
                            'net_cool_entropy', 'cool_entropy_in', \
                            'cool_entropy_out', 'net_warm_entropy', \
                            'warm_entropy_in', 'warm_entropy_out', \
                            'net_hot_entropy', 'hot_entropy_in', \
                            'hot_entropy_out'), \
                     dtype=('f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', \
                            'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', \
                            'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', \
                            'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', \
                            'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', \
                            'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', \
                            'f8', 'f8', 'f8', 'f8', 'f8', 'f8'))

    # Define the radii of the spherical shells where we want to calculate totals
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
        r_low = radii[i]
        r_high = radii[i+1]
        dr = r_high - r_low
        r = (r_low + r_high)/2.

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

        # Cut the data on temperature and radial velocity for in and out totals
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

        # Compute totals
        # For each parameter total, it is a nested list where the top index goes from 0 to 4 and is
        # the phase of gas (all, cold, cool, warm, hot) and the second index goes from 0 to 2 and is
        # the radial velocity (all, in, out).
        # Ex. the total of a parameter is total[0][0], the total of a parameter with negative radial velocity is total[0][1], the total warm gas of a parameter with positive radial velocity is total[3][2]
        mass_total = []
        metal_total = []
        kinetic_energy_total = []
        thermal_energy_total = []
        potential_energy_total = []
        entropy_total = []

        for j in range(5):
            mass_total.append([])
            metal_total.append([])
            kinetic_energy_total.append([])
            thermal_energy_total.append([])
            potential_energy_total.append([])
            entropy_total.append([])
            for k in range(3):
                mass_total[j].append(np.sum(mass_cut[j][k]))
                metal_total[j].append(np.sum(metal_mass_cut[j][k]))
                kinetic_energy_total[j].append(np.sum(kinetic_energy_cut[j][k]))
                thermal_energy_total[j].append(np.sum(thermal_energy_cut[j][k] * mass_cut[j][k]*gtoMsun))
                potential_energy_total[j].append(np.sum(G * mass_cut[j][k]*gtoMsun * Menc_func(radius_cut[j][k])*gtoMsun / \
                                               (radius_cut[j][k]*1000.*cmtopc)))
                entropy_total[j].append(np.sum(entropy_cut[j][k]))

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
                net_mass_q = (np.sum(shell_q['cell_mass'])).in_units('Msun')
                mass_in_q = (np.sum(shell_in_q['cell_mass'])).in_units('Msun')
                mass_out_q = (np.sum(shell_out_q['cell_mass'])).in_units('Msun')

                net_metals_q = (np.sum(shell_q['metal_mass'])).in_units('Msun')
                metals_in_q = (np.sum(shell_in_q['metal_mass'])).in_units('Msun')
                metals_out_q = (np.sum(shell_out_q['metal_mass'])).in_units('Msun')

                net_cold_mass_q = (np.sum(shell_cold_q['cell_mass'])).in_units('Msun')
                cold_mass_in_q = (np.sum(shell_in_cold_q['cell_mass'])).in_units('Msun')
                cold_mass_out_q = (np.sum(shell_out_cold_q['cell_mass'])).in_units('Msun')

                net_cool_mass_q = (np.sum(shell_cool_q['cell_mass'])).in_units('Msun')
                cool_mass_in_q = (np.sum(shell_in_cool_q['cell_mass'])).in_units('Msun')
                cool_mass_out_q = (np.sum(shell_out_cool_q['cell_mass'])).in_units('Msun')

                net_warm_mass_q = (np.sum(shell_warm_q['cell_mass'])).in_units('Msun')
                warm_mass_in_q = (np.sum(shell_in_warm_q['cell_mass'])).in_units('Msun')
                warm_mass_out_q = (np.sum(shell_out_warm_q['cell_mass'])).in_units('Msun')

                net_hot_mass_q = (np.sum(shell_hot_q['cell_mass'])).in_units('Msun')
                hot_mass_in_q = (np.sum(shell_in_hot_q['cell_mass'])).in_units('Msun')
                hot_mass_out_q = (np.sum(shell_out_hot_q['cell_mass'])).in_units('Msun')

                net_cold_metals_q = (np.sum(shell_cold_q['metal_mass'])).in_units('Msun')
                cold_metals_in_q = (np.sum(shell_in_cold_q['metal_mass'])).in_units('Msun')
                cold_metals_out_q = (np.sum(shell_out_cold_q['metal_mass'])).in_units('Msun')

                net_cool_metals_q = (np.sum(shell_cool_q['metal_mass'])).in_units('Msun')
                cool_metals_in_q = (np.sum(shell_in_cool_q['metal_mass'])).in_units('Msun')
                cool_metals_out_q = (np.sum(shell_out_cool_q['metal_mass'])).in_units('Msun')

                net_warm_metals_q = (np.sum(shell_warm_q['metal_mass'])).in_units('Msun')
                warm_metals_in_q = (np.sum(shell_in_warm_q['metal_mass'])).in_units('Msun')
                warm_metals_out_q = (np.sum(shell_out_warm_q['metal_mass'])).in_units('Msun')

                net_hot_metals_q = (np.sum(shell_hot_q['metal_mass'])).in_units('Msun')
                hot_metals_in_q = (np.sum(shell_in_hot_q['metal_mass'])).in_units('Msun')
                hot_metals_out_q = (np.sum(shell_out_hot_q['metal_mass'])).in_units('Msun')

                net_kinetic_energy_q = (np.sum(shell_q['kinetic_energy_corrected'])).in_units('erg')
                kinetic_energy_in_q = (np.sum(shell_in_q['kinetic_energy_corrected'])).in_units('erg')
                kinetic_energy_out_q = (np.sum(shell_out_q['kinetic_energy_corrected'])).in_units('erg')

                net_thermal_energy_q = (np.sum(shell_q['thermal_energy'] * \
                                        shell_q['cell_mass'])).in_units('erg')
                thermal_energy_in_q = (np.sum(shell_in_q['thermal_energy'] * \
                                       shell_in_q['cell_mass'])).in_units('erg')
                thermal_energy_out_q = (np.sum(shell_out_q['thermal_energy'] * \
                                        shell_out_q['cell_mass'])).in_units('erg')

                net_entropy_q = (np.sum(shell_q['entropy'])).in_units('keV*cm**2')
                entropy_in_q = (np.sum(shell_in_q['entropy'])).in_units('keV*cm**2')
                entropy_out_q = (np.sum(shell_out_q['entropy'])).in_units('keV*cm**2')

                net_cold_kinetic_energy_q = (np.sum(shell_cold_q['kinetic_energy_corrected'])).in_units('erg')
                cold_kinetic_energy_in_q = (np.sum(shell_in_cold_q['kinetic_energy_corrected'])).in_units('erg')
                cold_kinetic_energy_out_q = (np.sum(shell_out_cold_q['kinetic_energy_corrected'])).in_units('erg')

                net_cool_kinetic_energy_q = (np.sum(shell_cool_q['kinetic_energy_corrected'])).in_units('erg')
                cool_kinetic_energy_in_q = (np.sum(shell_in_cool_q['kinetic_energy_corrected'])).in_units('erg')
                cool_kinetic_energy_out_q = (np.sum(shell_out_cool_q['kinetic_energy_corrected'])).in_units('erg')

                net_warm_kinetic_energy_q = (np.sum(shell_warm_q['kinetic_energy_corrected'])).in_units('erg')
                warm_kinetic_energy_in_q = (np.sum(shell_in_warm_q['kinetic_energy_corrected'])).in_units('erg')
                warm_kinetic_energy_out_q = (np.sum(shell_out_warm_q['kinetic_energy_corrected'])).in_units('erg')

                net_hot_kinetic_energy_q = (np.sum(shell_hot_q['kinetic_energy_corrected'])).in_units('erg')
                hot_kinetic_energy_in_q = (np.sum(shell_in_hot_q['kinetic_energy_corrected'])).in_units('erg')
                hot_kinetic_energy_out_q = (np.sum(shell_out_hot_q['kinetic_energy_corrected'])).in_units('erg')

                net_cold_thermal_energy_q = (np.sum(shell_cold_q['thermal_energy'] * \
                                      shell_cold_q['cell_mass'])).in_units('erg')
                cold_thermal_energy_in_q = (np.sum(shell_in_cold_q['thermal_energy'] * \
                                     shell_in_cold_q['cell_mass'])).in_units('erg')
                cold_thermal_energy_out_q = (np.sum(shell_out_cold_q['thermal_energy'] * \
                                      shell_out_cold_q['cell_mass'])).in_units('erg')

                net_cool_thermal_energy_q = (np.sum(shell_cool_q['thermal_energy'] * \
                                      shell_cool_q['cell_mass'])).in_units('erg')
                cool_thermal_energy_in_q = (np.sum(shell_in_cool_q['thermal_energy'] * \
                                     shell_in_cool_q['cell_mass'])).in_units('erg')
                cool_thermal_energy_out_q = (np.sum(shell_out_cool_q['thermal_energy'] * \
                                      shell_out_cool_q['cell_mass'])).in_units('erg')

                net_warm_thermal_energy_q = (np.sum(shell_warm_q['thermal_energy'] * \
                                      shell_warm_q['cell_mass'])).in_units('erg')
                warm_thermal_energy_in_q = (np.sum(shell_in_warm_q['thermal_energy'] * \
                                     shell_in_warm_q['cell_mass'])).in_units('erg')
                warm_thermal_energy_out_q = (np.sum(shell_out_warm_q['thermal_energy'] * \
                                      shell_out_warm_q['cell_mass'])).in_units('erg')

                net_hot_thermal_energy_q = (np.sum(shell_hot_q['thermal_energy'] * \
                                     shell_hot_q['cell_mass'])).in_units('erg')
                hot_thermal_energy_in_q = (np.sum(shell_in_hot_q['thermal_energy'] * \
                                    shell_in_hot_q['cell_mass'])).in_units('erg')
                hot_thermal_energy_out_q = (np.sum(shell_out_hot_q['thermal_energy'] * \
                                     shell_out_hot_q['cell_mass'])).in_units('erg')

                net_cold_entropy_q = (np.sum(shell_cold_q['entropy'])).in_units('keV*cm**2')
                cold_entropy_in_q = (np.sum(shell_in_cold_q['entropy'])).in_units('keV*cm**2')
                cold_entropy_out_q = (np.sum(shell_out_cold_q['entropy'])).in_units('keV*cm**2')

                net_cool_entropy_q = (np.sum(shell_cool_q['entropy'])).in_units('keV*cm**2')
                cool_entropy_in_q = (np.sum(shell_in_cool_q['entropy'])).in_units('keV*cm**2')
                cool_entropy_out_q = (np.sum(shell_out_cool_q['entropy'])).in_units('keV*cm**2')

                net_warm_entropy_q = (np.sum(shell_warm_q['entropy'])).in_units('keV*cm**2')
                warm_entropy_in_q = (np.sum(shell_in_warm_q['entropy'])).in_units('keV*cm**2')
                warm_entropy_out_q = (np.sum(shell_out_warm_q['entropy'])).in_units('keV*cm**2')

                net_hot_entropy_q = (np.sum(shell_hot_q['entropy'])).in_units('keV*cm**2')
                hot_entropy_in_q = (np.sum(shell_in_hot_q['entropy'])).in_units('keV*cm**2')
                hot_entropy_out_q = (np.sum(shell_out_hot_q['entropy'])).in_units('keV*cm**2')

                # Add everything to the table
                data_q.add_row([zsnap, q+1, r, net_mass_q, net_metals_q, mass_in_q, mass_out_q, \
                              metals_in_q, metals_out_q, net_cold_mass_q, cold_mass_in_q, \
                              cold_mass_out_q, net_cool_mass_q, cool_mass_in_q, \
                              cool_mass_out_q, net_warm_mass_q, warm_mass_in_q, \
                              warm_mass_out_q, net_hot_mass_q, hot_mass_in_q, \
                              hot_mass_out_q, net_cold_metals_q, cold_metals_in_q, \
                              cold_metals_out_q, net_cool_metals_q, cool_metals_in_q, \
                              cool_metals_out_q, net_warm_metals_q, warm_metals_in_q, \
                              warm_metals_out_q, net_hot_metals_q, hot_metals_in_q, \
                              hot_metals_out_q, net_kinetic_energy_q, \
                              net_thermal_energy_q, net_entropy_q, \
                              kinetic_energy_in_q, kinetic_energy_out_q, \
                              thermal_energy_in_q, thermal_energy_out_q, \
                              entropy_in_q, entropy_out_q, net_cold_kinetic_energy_q, \
                              cold_kinetic_energy_in_q, cold_kinetic_energy_out_q, \
                              net_cool_kinetic_energy_q, cool_kinetic_energy_in_q, \
                              cool_kinetic_energy_out_q, net_warm_kinetic_energy_q, \
                              warm_kinetic_energy_in_q, warm_kinetic_energy_out_q, \
                              net_hot_kinetic_energy_q, hot_kinetic_energy_in_q, \
                              hot_kinetic_energy_out_q, net_cold_thermal_energy_q, \
                              cold_thermal_energy_in_q, cold_thermal_energy_out_q, \
                              net_cool_thermal_energy_q, cool_thermal_energy_in_q, \
                              cool_thermal_energy_out_q, net_warm_thermal_energy_q, \
                              warm_thermal_energy_in_q, warm_thermal_energy_out_q, \
                              net_hot_thermal_energy_q, hot_thermal_energy_in_q, \
                              hot_thermal_energy_out_q, net_cold_entropy_q, \
                              cold_entropy_in_q, cold_entropy_out_q, \
                              net_cool_entropy_q, cool_entropy_in_q, \
                              cool_entropy_out_q, net_warm_entropy_q, \
                              warm_entropy_in_q, warm_entropy_out_q, \
                              net_hot_entropy_q, hot_entropy_in_q, \
                              hot_entropy_out_q])
        # Add everything to the table
        data.add_row([zsnap, 0, r, mass_total[0][0], metal_total[0][0], \
                      mass_total[0][1], mass_total[0][2], metal_total[0][1], metal_total[0][2], \
                      mass_total[1][0], mass_total[1][1], mass_total[1][2], \
                      mass_total[2][0], mass_total[2][1], mass_total[2][2], \
                      mass_total[3][0], mass_total[3][1], mass_total[3][2], \
                      mass_total[4][0], mass_total[4][1], mass_total[4][2], \
                      metal_total[1][0], metal_total[1][1], metal_total[1][2], \
                      metal_total[2][0], metal_total[2][1], metal_total[2][2], \
                      metal_total[3][0], metal_total[3][1], metal_total[3][2], \
                      metal_total[4][0], metal_total[4][1], metal_total[4][2], \
                      kinetic_energy_total[0][0], thermal_energy_total[0][0], \
                      potential_energy_total[0][0], \
                      entropy_total[0][0], \
                      kinetic_energy_total[0][1], kinetic_energy_total[0][2], \
                      thermal_energy_total[0][1], thermal_energy_total[0][2], \
                      potential_energy_total[0][1], potential_energy_total[0][2], \
                      entropy_total[0][1], entropy_total[0][2], \
                      kinetic_energy_total[1][0], kinetic_energy_total[1][1], kinetic_energy_total[1][2], \
                      kinetic_energy_total[2][0], kinetic_energy_total[2][1], kinetic_energy_total[2][2], \
                      kinetic_energy_total[3][0], kinetic_energy_total[3][1], kinetic_energy_total[3][2], \
                      kinetic_energy_total[4][0], kinetic_energy_total[4][1], kinetic_energy_total[4][2], \
                      thermal_energy_total[1][0], thermal_energy_total[1][1], thermal_energy_total[1][2], \
                      thermal_energy_total[2][0], thermal_energy_total[2][1], thermal_energy_total[2][2], \
                      thermal_energy_total[3][0], thermal_energy_total[3][1], thermal_energy_total[3][2], \
                      thermal_energy_total[4][0], thermal_energy_total[4][1], thermal_energy_total[4][2], \
                      potential_energy_total[1][0], potential_energy_total[1][1], potential_energy_total[1][2], \
                      potential_energy_total[2][0], potential_energy_total[2][1], potential_energy_total[2][2], \
                      potential_energy_total[3][0], potential_energy_total[3][1], potential_energy_total[3][2], \
                      potential_energy_total[4][0], potential_energy_total[4][1], potential_energy_total[4][2], \
                      entropy_total[1][0], entropy_total[1][1], entropy_total[1][2], \
                      entropy_total[2][0], entropy_total[2][1], entropy_total[2][2], \
                      entropy_total[3][0], entropy_total[3][1], entropy_total[3][2], \
                      entropy_total[4][0], entropy_total[4][1], entropy_total[4][2]])

    # Save to file
    data = set_table_units(data)
    data.write(tablename + '.hdf5', path='all_data', serialize_meta=True, overwrite=True)

    if (quadrants):
        data_q = set_table_units(data_q)
        data_q.write(tablename + '_q.hdf5', path='all_data', serialize_meta=True, overwrite=True)

    return "Totals have been calculated for snapshot" + snap + "!"

def calc_totals_sphere(ds, snap, zsnap, refine_width_kpc, tablename, surface_args, **kwargs):
    '''This function calculates the total of each gas property in spherical shells, with satellites removed,
    at a variety of radii. It uses the dataset stored in 'ds', which is from time snapshot 'snap', has redshift
    'zsnap', and has width of the refine box in kpc 'refine_width_kpc' and stores the totals in 'tablename'.
    'surface_args' gives the properties of the spheres.'''

    Menc_func = kwargs.get('Menc_func', False)
    sat = kwargs.get('sat')

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
    totals = Table(names=('redshift', 'inner_radius', 'outer_radius', \
                        'net_mass', 'net_metals', \
                        'mass_in', 'mass_out', 'metals_in', 'metals_out', \
                        'net_cold_mass', 'cold_mass_in', 'cold_mass_out', \
                        'net_cool_mass', 'cool_mass_in', 'cool_mass_out', \
                        'net_warm_mass', 'warm_mass_in', 'warm_mass_out', \
                        'net_hot_mass', 'hot_mass_in', 'hot_mass_out', \
                        'net_cold_metals', 'cold_metals_in', 'cold_metals_out', \
                        'net_cool_metals', 'cool_metals_in', 'cool_metals_out', \
                        'net_warm_metals', 'warm_metals_in', 'warm_metals_out', \
                        'net_hot_metals', 'hot_metals_in', 'hot_metals_out', \
                        'net_kinetic_energy', 'net_thermal_energy', \
                        'net_potential_energy', 'net_entropy', \
                        'kinetic_energy_in', 'kinetic_energy_out', \
                        'thermal_energy_in', 'thermal_energy_out', \
                        'potential_energy_in', 'potential_energy_out', \
                        'entropy_in', 'entropy_out', \
                        'net_cold_kinetic_energy', 'cold_kinetic_energy_in', 'cold_kinetic_energy_out', \
                        'net_cool_kinetic_energy', 'cool_kinetic_energy_in', 'cool_kinetic_energy_out', \
                        'net_warm_kinetic_energy', 'warm_kinetic_energy_in', 'warm_kinetic_energy_out', \
                        'net_hot_kinetic_energy', 'hot_kinetic_energy_in', 'hot_kinetic_energy_out', \
                        'net_cold_thermal_energy', 'cold_thermal_energy_in', 'cold_thermal_energy_out', \
                        'net_cool_thermal_energy', 'cool_thermal_energy_in', 'cool_thermal_energy_out', \
                        'net_warm_thermal_energy', 'warm_thermal_energy_in', 'warm_thermal_energy_out', \
                        'net_hot_thermal_energy', 'hot_thermal_energy_in', 'hot_thermal_energy_out', \
                        'net_cold_potential_energy', 'cold_potential_energy_in', 'cold_potential_energy_out', \
                        'net_cool_potential_energy', 'cool_potential_energy_in', 'cool_potential_energy_out', \
                        'net_warm_potential_energy', 'warm_potential_energy_in', 'warm_potential_energy_out', \
                        'net_hot_potential_energy', 'hot_potential_energy_in', 'hot_potential_energy_out', \
                        'net_cold_entropy', 'cold_entropy_in', 'cold_entropy_out', \
                        'net_cool_entropy', 'cool_entropy_in', 'cool_entropy_out', \
                        'net_warm_entropy', 'warm_entropy_in', 'warm_entropy_out', \
                        'net_hot_entropy', 'hot_entropy_in', 'hot_entropy_out'), \
                 dtype=('f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', \
                        'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', \
                        'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', \
                        'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', \
                        'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', \
                        'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', \
                        'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', \
                        'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8'))

    # Define the radii of the spherical shells where we want to calculate totals
    radii = refine_width_kpc * np.arange(inner_radius, outer_radius+dr, dr)

    # Load arrays of all fields we need
    print('Loading field arrays')
    sphere = ds.sphere(halo_center_kpc, radii[-1])

    radius = sphere['gas','radius_corrected'].in_units('kpc').v
    x = sphere['gas','x'].in_units('kpc').v - halo_center_kpc[0].v
    y = sphere['gas','y'].in_units('kpc').v - halo_center_kpc[1].v
    z = sphere['gas','z'].in_units('kpc').v - halo_center_kpc[2].v
    rad_vel = sphere['gas','radial_velocity_corrected'].in_units('km/s').v
    mass = sphere['gas','cell_mass'].in_units('Msun').v
    metal_mass = sphere['gas','metal_mass'].in_units('Msun').v
    temperature = sphere['gas','temperature'].in_units('K').v
    kinetic_energy = sphere['gas','kinetic_energy_corrected'].in_units('erg').v
    thermal_energy = sphere['gas','thermal_energy'].in_units('erg/g').v
    potential_energy = (sphere['gas','cell_mass'] * \
      ds.arr(sphere['enzo','Grav_Potential'].v, 'code_length**2/code_time**2')).in_units('erg').v
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

    # Cut data to remove anything within satellites and to things that cross into and out of satellites
    print('Cutting data to remove satellites')
    sat_radius = 5.         # kpc
    sat_radius_sq = sat_radius**2.
    # An attempt to remove satellites faster:
    # Holy cow this is so much faster, do it this way
    bool_inside_sat = []
    for s in range(len(sat_list)):
        sat_x = sat_list[s][0]
        sat_y = sat_list[s][1]
        sat_z = sat_list[s][2]
        dist_from_sat_sq = (x-sat_x)**2. + (y-sat_y)**2. + (z-sat_z)**2.
        bool_inside_sat.append((dist_from_sat_sq < sat_radius_sq))
    bool_inside_sat = np.array(bool_inside_sat)
    inside_sat = np.count_nonzero(bool_inside_sat, axis=0)
    # inside_sat should now be an array of length = # of pixels where the value is an
    # integer. If the value is zero, that pixel is not inside any satellites. If the value is > 0,
    # that pixel is in a satellite.
    bool_nosat = (inside_sat == 0)

    radius_nosat = radius[bool_nosat]
    rad_vel_nosat = rad_vel[bool_nosat]
    temperature_nosat = temperature[bool_nosat]
    mass_nosat = mass[bool_nosat]
    metal_mass_nosat = metal_mass[bool_nosat]
    kinetic_energy_nosat = kinetic_energy[bool_nosat]
    thermal_energy_nosat = thermal_energy[bool_nosat]
    potential_energy_nosat = potential_energy[bool_nosat]
    entropy_nosat = entropy[bool_nosat]

    # Cut satellite-removed data on temperature
    # These are lists of lists where the index goes from 0 to 4 for [all gas, cold, cool, warm, hot]
    print('Cutting satellite-removed data on temperature')
    radius_nosat_Tcut = []
    rad_vel_nosat_Tcut = []
    mass_nosat_Tcut = []
    metal_mass_nosat_Tcut = []
    kinetic_energy_nosat_Tcut = []
    thermal_energy_nosat_Tcut = []
    potential_energy_nosat_Tcut = []
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
        mass_nosat_Tcut.append(mass_nosat[bool_temp])
        metal_mass_nosat_Tcut.append(metal_mass_nosat[bool_temp])
        kinetic_energy_nosat_Tcut.append(kinetic_energy_nosat[bool_temp])
        thermal_energy_nosat_Tcut.append(thermal_energy_nosat[bool_temp])
        potential_energy_nosat_Tcut.append(potential_energy_nosat[bool_temp])
        entropy_nosat_Tcut.append(entropy_nosat[bool_temp])

    # Loop over radii
    for i in range(len(radii)-1):
        inner_r = radii[i].v
        outer_r = radii[i+1].v

        if (i%10==0): print("Computing radius " + str(i) + "/" + str(len(radii)) + \
                            " for snapshot " + snap)

        # Compute net, in, and out totals with satellites removed
        # These are nested lists where the first index goes from 0 to 2 for [net, in, out]
        # and the second index goes from 0 to 4 for [all, cold, cool, warm, hot]
        mass_total_nosat = []
        metals_total_nosat = []
        kinetic_energy_total_nosat = []
        thermal_energy_total_nosat = []
        potential_energy_total_nosat = []
        entropy_total_nosat = []
        for j in range(3):
            mass_total_nosat.append([])
            metals_total_nosat.append([])
            kinetic_energy_total_nosat.append([])
            thermal_energy_total_nosat.append([])
            potential_energy_total_nosat.append([])
            entropy_total_nosat.append([])
            for k in range(5):
                bool_in = (radius_nosat_Tcut[k] > inner_r) & (radius_nosat_Tcut[k] < outer_r) & \
                  (rad_vel_nosat_Tcut[k] < 0.)
                bool_out = (radius_nosat_Tcut[k] > inner_r) & (radius_nosat_Tcut[k] < outer_r) & \
                  (rad_vel_nosat_Tcut[k] > 0.)
                if (j==0):
                    mass_total_nosat[j].append((np.sum(mass_nosat_Tcut[k][bool_out]) + \
                      np.sum(mass_nosat_Tcut[k][bool_in])))
                    metals_total_nosat[j].append((np.sum(metal_mass_nosat_Tcut[k][bool_out]) + \
                      np.sum(metal_mass_nosat_Tcut[k][bool_in])))
                    kinetic_energy_total_nosat[j].append((np.sum(kinetic_energy_nosat_Tcut[k][bool_out]) + \
                      np.sum(kinetic_energy_nosat_Tcut[k][bool_in])))
                    thermal_energy_total_nosat[j].append((np.sum(thermal_energy_nosat_Tcut[k][bool_out]) + \
                      np.sum(thermal_energy_nosat_Tcut[k][bool_in])))
                    potential_energy_total_nosat[j].append((np.sum(potential_energy_nosat_Tcut[k][bool_out]) + \
                      np.sum(potential_energy_nosat_Tcut[k][bool_in])))
                    entropy_total_nosat[j].append((np.sum(entropy_nosat_Tcut[k][bool_out]) + \
                      np.sum(entropy_nosat_Tcut[k][bool_in])))
                if (j==1):
                    mass_total_nosat[j].append(np.sum(mass_nosat_Tcut[k][bool_in]))
                    metals_total_nosat[j].append(np.sum(metal_mass_nosat_Tcut[k][bool_in]))
                    kinetic_energy_total_nosat[j].append(np.sum(kinetic_energy_nosat_Tcut[k][bool_in]))
                    thermal_energy_total_nosat[j].append(np.sum(thermal_energy_nosat_Tcut[k][bool_in]))
                    potential_energy_total_nosat[j].append(np.sum(potential_energy_nosat_Tcut[k][bool_in]))
                    entropy_total_nosat[j].append(np.sum(entropy_nosat_Tcut[k][bool_in]))
                if (j==2):
                    mass_total_nosat[j].append(np.sum(mass_nosat_Tcut[k][bool_out]))
                    metals_total_nosat[j].append(np.sum(metal_mass_nosat_Tcut[k][bool_out]))
                    kinetic_energy_total_nosat[j].append(np.sum(kinetic_energy_nosat_Tcut[k][bool_out]))
                    thermal_energy_total_nosat[j].append(np.sum(thermal_energy_nosat_Tcut[k][bool_out]))
                    potential_energy_total_nosat[j].append(np.sum(potential_energy_nosat_Tcut[k][bool_out]))
                    entropy_total_nosat[j].append(np.sum(entropy_nosat_Tcut[k][bool_out]))

        # Add everything to the table
        totals.add_row([zsnap, inner_r, outer_r, \
                        mass_total_nosat[0][0], metals_total_nosat[0][0], \
                        mass_total_nosat[1][0], mass_total_nosat[2][0], metals_total_nosat[1][0], metals_total_nosat[2][0], \
                        mass_total_nosat[0][1], mass_total_nosat[1][1], mass_total_nosat[2][1], \
                        mass_total_nosat[0][2], mass_total_nosat[1][2], mass_total_nosat[2][2], \
                        mass_total_nosat[0][3], mass_total_nosat[1][3], mass_total_nosat[2][3], \
                        mass_total_nosat[0][4], mass_total_nosat[1][4], mass_total_nosat[2][4], \
                        metals_total_nosat[0][1], metals_total_nosat[1][1], metals_total_nosat[2][1], \
                        metals_total_nosat[0][2], metals_total_nosat[1][2], metals_total_nosat[2][2], \
                        metals_total_nosat[0][3], metals_total_nosat[1][3], metals_total_nosat[2][3], \
                        metals_total_nosat[0][4], metals_total_nosat[1][4], metals_total_nosat[2][4], \
                        kinetic_energy_total_nosat[0][0], thermal_energy_total_nosat[0][0], \
                        potential_energy_total_nosat[0][0], entropy_total_nosat[0][0], \
                        kinetic_energy_total_nosat[1][0], kinetic_energy_total_nosat[2][0], \
                        thermal_energy_total_nosat[1][0], thermal_energy_total_nosat[2][0], \
                        potential_energy_total_nosat[1][0], potential_energy_total_nosat[2][0], \
                        entropy_total_nosat[1][0], entropy_total_nosat[2][0], \
                        kinetic_energy_total_nosat[0][1], kinetic_energy_total_nosat[1][1], kinetic_energy_total_nosat[2][1], \
                        kinetic_energy_total_nosat[0][2], kinetic_energy_total_nosat[1][2], kinetic_energy_total_nosat[2][2], \
                        kinetic_energy_total_nosat[0][3], kinetic_energy_total_nosat[1][3], kinetic_energy_total_nosat[2][3], \
                        kinetic_energy_total_nosat[0][4], kinetic_energy_total_nosat[1][4], kinetic_energy_total_nosat[2][4], \
                        thermal_energy_total_nosat[0][1], thermal_energy_total_nosat[1][1], thermal_energy_total_nosat[2][1], \
                        thermal_energy_total_nosat[0][2], thermal_energy_total_nosat[1][2], thermal_energy_total_nosat[2][2], \
                        thermal_energy_total_nosat[0][3], thermal_energy_total_nosat[1][3], thermal_energy_total_nosat[2][3], \
                        thermal_energy_total_nosat[0][4], thermal_energy_total_nosat[1][4], thermal_energy_total_nosat[2][4], \
                        potential_energy_total_nosat[0][1], potential_energy_total_nosat[1][1], potential_energy_total_nosat[2][1], \
                        potential_energy_total_nosat[0][2], potential_energy_total_nosat[1][2], potential_energy_total_nosat[2][2], \
                        potential_energy_total_nosat[0][3], potential_energy_total_nosat[1][3], potential_energy_total_nosat[2][3], \
                        potential_energy_total_nosat[0][4], potential_energy_total_nosat[1][4], potential_energy_total_nosat[2][4], \
                        entropy_total_nosat[0][1], entropy_total_nosat[1][1], entropy_total_nosat[2][1], \
                        entropy_total_nosat[0][2], entropy_total_nosat[1][2], entropy_total_nosat[2][2], \
                        entropy_total_nosat[0][3], entropy_total_nosat[1][3], entropy_total_nosat[2][3], \
                        entropy_total_nosat[0][4], entropy_total_nosat[1][4], entropy_total_nosat[2][4]])

    totals = set_table_units(totals)

    # Save to file
    totals.write(tablename + '_nosat_sphere.hdf5', path='all_data', serialize_meta=True, overwrite=True)

    return "Totals have been calculated for snapshot " + snap + "!"

def calc_totals_frustum(ds, snap, zsnap, refine_width_kpc, tablename, surface_args, **kwargs):
    '''This function calculates the totals of gas properties between radial surfaces within a frustum,
    with satellites removed, at a variety of radii. It
    uses the dataset stored in 'ds', which is from time snapshot 'snap', has redshift
    'zsnap', and has width of the refine box in kpc 'refine_width_kpc', and stores the totals in
    'tablename'. 'surface_args' gives the properties of the frustum.'''

    Menc_func = kwargs.get('Menc_func', False)
    sat = kwargs.get('sat')

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
    totals = Table(names=('redshift', 'inner_radius', 'outer_radius', \
                        'net_mass', 'net_metals', \
                        'mass_in', 'mass_out', 'metals_in', 'metals_out', \
                        'net_cold_mass', 'cold_mass_in', 'cold_mass_out', \
                        'net_cool_mass', 'cool_mass_in', 'cool_mass_out', \
                        'net_warm_mass', 'warm_mass_in', 'warm_mass_out', \
                        'net_hot_mass', 'hot_mass_in', 'hot_mass_out', \
                        'net_cold_metals', 'cold_metals_in', 'cold_metals_out', \
                        'net_cool_metals', 'cool_metals_in', 'cool_metals_out', \
                        'net_warm_metals', 'warm_metals_in', 'warm_metals_out', \
                        'net_hot_metals', 'hot_metals_in', 'hot_metals_out', \
                        'net_kinetic_energy', 'net_thermal_energy', \
                        'net_potential_energy', 'net_entropy', \
                        'kinetic_energy_in', 'kinetic_energy_out', \
                        'thermal_energy_in', 'thermal_energy_out', \
                        'potential_energy_in', 'potential_energy_out', \
                        'entropy_in', 'entropy_out', \
                        'net_cold_kinetic_energy', 'cold_kinetic_energy_in', 'cold_kinetic_energy_out', \
                        'net_cool_kinetic_energy', 'cool_kinetic_energy_in', 'cool_kinetic_energy_out', \
                        'net_warm_kinetic_energy', 'warm_kinetic_energy_in', 'warm_kinetic_energy_out', \
                        'net_hot_kinetic_energy', 'hot_kinetic_energy_in', 'hot_kinetic_energy_out', \
                        'net_cold_thermal_energy', 'cold_thermal_energy_in', 'cold_thermal_energy_out', \
                        'net_cool_thermal_energy', 'cool_thermal_energy_in', 'cool_thermal_energy_out', \
                        'net_warm_thermal_energy', 'warm_thermal_energy_in', 'warm_thermal_energy_out', \
                        'net_hot_thermal_energy', 'hot_thermal_energy_in', 'hot_thermal_energy_out', \
                        'net_cold_potential_energy', 'cold_potential_energy_in', 'cold_potential_energy_out', \
                        'net_cool_potential_energy', 'cool_potential_energy_in', 'cool_potential_energy_out', \
                        'net_warm_potential_energy', 'warm_potential_energy_in', 'warm_potential_energy_out', \
                        'net_hot_potential_energy', 'hot_potential_energy_in', 'hot_potential_energy_out', \
                        'net_cold_entropy', 'cold_entropy_in', 'cold_entropy_out', \
                        'net_cool_entropy', 'cool_entropy_in', 'cool_entropy_out', \
                        'net_warm_entropy', 'warm_entropy_in', 'warm_entropy_out', \
                        'net_hot_entropy', 'hot_entropy_in', 'hot_entropy_out'), \
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
    rad_vel = sphere['gas','radial_velocity_corrected'].in_units('km/s').v
    mass = sphere['gas','cell_mass'].in_units('Msun').v
    metal_mass = sphere['gas','metal_mass'].in_units('Msun').v
    temperature = sphere['gas','temperature'].in_units('K').v
    kinetic_energy = sphere['gas','kinetic_energy_corrected'].in_units('erg').v
    thermal_energy = sphere['gas','thermal_energy'].in_units('erg/g').v
    potential_energy = (sphere['gas','cell_mass'] * \
      ds.arr(sphere['enzo','Grav_Potential'].v, 'code_length**2/code_time**2')).in_units('erg').v
    entropy = sphere['gas','entropy'].in_units('keV*cm**2').v

    # Cut data to only the frustum considered here
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
        phi = np.arctan2(y, x)
        frus_filename += 'z_' + str(op_angle)
    if (axis=='x'):
        theta = np.arccos(x/radius)
        phi = np.arctan2(z, y)
        frus_filename += 'x_' + str(op_angle)
    if (axis=='y'):
        theta = np.arccos(y/radius)
        phi = np.arctan2(x, z)
        frus_filename += 'y_' + str(op_angle)
    if (axis=='disk minor axis'):
        x_disk = sphere['gas','x_disk'].in_units('kpc').v
        y_disk = sphere['gas','y_disk'].in_units('kpc').v
        z_disk = sphere['gas','z_disk'].in_units('kpc').v
        theta = np.arccos(z_disk/radius)
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

    # Cut data to remove anything within satellites and to things that cross into and out of satellites
    # Restrict to only things that start or end within the frustum
    print('Cutting data to remove satellites')
    sat_radius = 5.         # kpc
    sat_radius_sq = sat_radius**2.
    # An attempt to remove satellites faster:
    # Holy cow this is so much faster, do it this way
    bool_inside_sat = []
    for s in range(len(sat_list)):
        sat_x = sat_list[s][0]
        sat_y = sat_list[s][1]
        sat_z = sat_list[s][2]
        dist_from_sat_sq = (x-sat_x)**2. + (y-sat_y)**2. + (z-sat_z)**2.
        bool_inside_sat.append((dist_from_sat_sq < sat_radius_sq))
    bool_inside_sat = np.array(bool_inside_sat)
    inside_sat = np.count_nonzero(bool_inside_sat, axis=0)
    # inside_sat should now both be an array of length = # of pixels where the value is an
    # integer. If the value is zero, that pixel is not inside any satellites. If the value is > 0,
    # that pixel is in a satellite.
    bool_nosat = (inside_sat == 0)

    radius_nosat = radius[bool_nosat]
    theta_nosat = theta[bool_nosat]
    rad_vel_nosat = rad_vel[bool_nosat]
    temperature_nosat = temperature[bool_nosat]
    mass_nosat = mass[bool_nosat]
    metal_mass_nosat = metal_mass[bool_nosat]
    kinetic_energy_nosat = kinetic_energy[bool_nosat]
    thermal_energy_nosat = thermal_energy[bool_nosat]
    potential_energy_nosat = potential_energy[bool_nosat]
    entropy_nosat = entropy[bool_nosat]

    # Cut satellite-removed data to frustum of interest
    bool_frus = (theta_nosat >= min_theta) & (theta_nosat <= max_theta)

    radius_nosat_frus = radius_nosat[bool_frus]
    newradius_nosat_frus = newradius_nosat[bool_frus]
    rad_vel_nosat_frus = rad_vel_nosat[bool_frus]
    mass_nosat_frus = mass_nosat[bool_frus]
    metal_mass_nosat_frus = metal_mass_nosat[bool_frus]
    temperature_nosat_frus = temperature_nosat[bool_frus]
    kinetic_energy_nosat_frus = kinetic_energy_nosat[bool_frus]
    thermal_energy_nosat_frus = thermal_energy_nosat[bool_frus]
    potential_energy_nosat_frus = potential_energy_nosat[bool_frus]
    cooling_time_nosat_frus = cooling_time_nosat[bool_frus]
    entropy_nosat_frus = entropy_nosat[bool_frus]

    # Cut satellite-removed frustum data on temperature
    # These are lists of lists where the first index goes from 0 to 4 for
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
        bool_temp_nosat_frus = (temperature_nosat_frus < t_high) & (temperature_nosat_frus > t_low)
        radius_nosat_frus_Tcut.append(radius_nosat_frus[bool_temp_nosat_frus])
        rad_vel_nosat_frus_Tcut.append(rad_vel_nosat_frus[bool_temp_nosat_frus])
        newradius_nosat_frus_Tcut.append(newradius_nosat_frus[bool_temp_nosat_frus])
        mass_nosat_frus_Tcut.append(mass_nosat_frus[bool_temp_nosat_frus])
        metal_mass_nosat_frus_Tcut.append(metal_mass_nosat_frus[bool_temp_nosat_frus])
        kinetic_energy_nosat_frus_Tcut.append(kinetic_energy_nosat_frus[bool_temp_nosat_frus])
        thermal_energy_nosat_frus_Tcut.append(thermal_energy_nosat_frus[bool_temp_nosat_frus])
        potential_energy_nosat_frus_Tcut.append(potential_energy_nosat_frus[bool_temp_nosat_frus])
        cooling_time_nosat_frus_Tcut.append(cooling_time_nosat_frus[bool_temp_nosat_frus])
        entropy_nosat_frus_Tcut.append(entropy_nosat_frus[bool_temp_nosat_frus])

    # Loop over radii
    for i in range(len(radii)-1):
        inner_r = radii[i].v
        outer_r = radii[i+1].v

        if (i%10==0): print("Computing radius " + str(i) + "/" + str(len(radii)) + \
                            " for snapshot " + snap)

        # Compute net, in, and out totals within the frustum with satellites removed
        # These are nested lists where the first index goes from 0 to 2 for [net, in, out]
        # and the second index goes from 0 to 4 for [all, cold, cool, warm, hot]
        mass_total_nosat = []
        metals_total_nosat = []
        kinetic_energy_total_nosat = []
        thermal_energy_total_nosat = []
        potential_energy_total_nosat = []
        entropy_total_nosat = []
        for j in range(3):
            mass_total_nosat.append([])
            metals_total_nosat.append([])
            kinetic_energy_total_nosat.append([])
            thermal_energy_total_nosat.append([])
            potential_energy_total_nosat.append([])
            entropy_total_nosat.append([])
            for k in range(5):
                bool_in_r = (radius_nosat_frus_Tcut[k] > inner_r) & (radius_nosat_frus_Tcut[k] < outer_r) & \
                  (rad_vel_nosat_frus_Tcut[k] < 0.)
                bool_out_r = (radius_nosat_frus_Tcut[k] > inner_r) & (radius_nosat_frus_Tcut[k] < outer_r) & \
                  (rad_vel_nosat_frus_Tcut[k] > 0.)
                if (j==0):
                    mass_total_nosat[j].append((np.sum(mass_nosat_frus_Tcut[k][bool_out_r]) + \
                      np.sum(mass_nosat_frus_Tcut[k][bool_in_r])))
                    metals_total_nosat[j].append((np.sum(metal_mass_nosat_frus_Tcut[k][bool_out_r]) + \
                      np.sum(metal_mass_nosat_frus_Tcut[k][bool_in_r])))
                    kinetic_energy_total_nosat[j].append((np.sum(kinetic_energy_nosat_frus_Tcut[k][bool_out_r]) + \
                      np.sum(kinetic_energy_nosat_frus_Tcut[k][bool_in_r])))
                    thermal_energy_total_nosat[j].append((np.sum(thermal_energy_nosat_frus_Tcut[k][bool_out_r]) + \
                      np.sum(thermal_energy_nosat_frus_Tcut[k][bool_in_r])))
                    potential_energy_total_nosat[j].append((np.sum(potential_energy_nosat_frus_Tcut[k][bool_out_r]) + \
                      np.sum(potential_energy_nosat_frus_Tcut[k][bool_in_r])))
                    entropy_total_nosat[j].append((np.sum(entropy_nosat_frus_Tcut[k][bool_out_r]) + \
                    np.sum(entropy_nosat_frus_Tcut[k][bool_in_r])))
                if (j==1):
                    mass_total_nosat[j].append(np.sum(mass_nosat_frus_Tcut[k][bool_in_r]))
                    metals_total_nosat[j].append(np.sum(metal_mass_nosat_frus_Tcut[k][bool_in_r]))
                    kinetic_energy_total_nosat[j].append(np.sum(kinetic_energy_nosat_frus_Tcut[k][bool_in_r]))
                    thermal_energy_total_nosat[j].append(np.sum(thermal_energy_nosat_frus_Tcut[k][bool_in_r]))
                    potential_energy_total_nosat[j].append(np.sum(potential_energy_nosat_frus_Tcut[k][bool_in_r]))
                    entropy_total_nosat[j].append(np.sum(entropy_nosat_frus_Tcut[k][bool_in_r]))
                if (j==2):
                    mass_total_nosat[j].append(np.sum(mass_nosat_frus_Tcut[k][bool_out_r]))
                    metals_total_nosat[j].append(np.sum(metal_mass_nosat_frus_Tcut[k][bool_out_r]))
                    kinetic_energy_total_nosat[j].append(np.sum(kinetic_energy_nosat_frus_Tcut[k][bool_out_r]))
                    thermal_energy_total_nosat[j].append(np.sum(thermal_energy_nosat_frus_Tcut[k][bool_out_r]))
                    potential_energy_total_nosat[j].append(np.sum(potential_energy_nosat_frus_Tcut[k][bool_out_r]))
                    entropy_total_nosat[j].append(np.sum(entropy_nosat_frus_Tcut[k][bool_out_r]))

        # Add everything to the tables
        totals.add_row([zsnap, inner_r, outer_r, \
                        mass_total_nosat[0][0], metals_total_nosat[0][0], \
                        mass_total_nosat[1][0], mass_total_nosat[2][0], metals_total_nosat[1][0], metals_total_nosat[2][0], \
                        mass_total_nosat[0][1], mass_total_nosat[1][1], mass_total_nosat[2][1], \
                        mass_total_nosat[0][2], mass_total_nosat[1][2], mass_total_nosat[2][2], \
                        mass_total_nosat[0][3], mass_total_nosat[1][3], mass_total_nosat[2][3], \
                        mass_total_nosat[0][4], mass_total_nosat[1][4], mass_total_nosat[2][4], \
                        metals_total_nosat[0][1], metals_total_nosat[1][1], metals_total_nosat[2][1], \
                        metals_total_nosat[0][2], metals_total_nosat[1][2], metals_total_nosat[2][2], \
                        metals_total_nosat[0][3], metals_total_nosat[1][3], metals_total_nosat[2][3], \
                        metals_total_nosat[0][4], metals_total_nosat[1][4], metals_total_nosat[2][4], \
                        kinetic_energy_total_nosat[0][0], thermal_energy_total_nosat[0][0], \
                        potential_energy_total_nosat[0][0], entropy_total_nosat[0][0], \
                        kinetic_energy_total_nosat[1][0], kinetic_energy_total_nosat[2][0], \
                        thermal_energy_total_nosat[1][0], thermal_energy_total_nosat[2][0], \
                        potential_energy_total_nosat[1][0], potential_energy_total_nosat[2][0], \
                        entropy_total_nosat[1][0], entropy_total_nosat[2][0], \
                        kinetic_energy_total_nosat[0][1], kinetic_energy_total_nosat[1][1], kinetic_energy_total_nosat[2][1], \
                        kinetic_energy_total_nosat[0][2], kinetic_energy_total_nosat[1][2], kinetic_energy_total_nosat[2][2], \
                        kinetic_energy_total_nosat[0][3], kinetic_energy_total_nosat[1][3], kinetic_energy_total_nosat[2][3], \
                        kinetic_energy_total_nosat[0][4], kinetic_energy_total_nosat[1][4], kinetic_energy_total_nosat[2][4], \
                        thermal_energy_total_nosat[0][1], thermal_energy_total_nosat[1][1], thermal_energy_total_nosat[2][1], \
                        thermal_energy_total_nosat[0][2], thermal_energy_total_nosat[1][2], thermal_energy_total_nosat[2][2], \
                        thermal_energy_total_nosat[0][3], thermal_energy_total_nosat[1][3], thermal_energy_total_nosat[2][3], \
                        thermal_energy_total_nosat[0][4], thermal_energy_total_nosat[1][4], thermal_energy_total_nosat[2][4], \
                        potential_energy_total_nosat[0][1], potential_energy_total_nosat[1][1], potential_energy_total_nosat[2][1], \
                        potential_energy_total_nosat[0][2], potential_energy_total_nosat[1][2], potential_energy_total_nosat[2][2], \
                        potential_energy_total_nosat[0][3], potential_energy_total_nosat[1][3], potential_energy_total_nosat[2][3], \
                        potential_energy_total_nosat[0][4], potential_energy_total_nosat[1][4], potential_energy_total_nosat[2][4], \
                        entropy_total_nosat[0][1], entropy_total_nosat[1][1], entropy_total_nosat[2][1], \
                        entropy_total_nosat[0][2], entropy_total_nosat[1][2], entropy_total_nosat[2][2], \
                        entropy_total_nosat[0][3], entropy_total_nosat[1][3], entropy_total_nosat[2][3], \
                        entropy_total_nosat[0][4], entropy_total_nosat[1][4], entropy_total_nosat[2][4]])

    totals = set_table_units(totals)

    # Save to file
    totals.write(tablename + '_nosat_frustum_' + frus_filename + '.hdf5', path='all_data', serialize_meta=True, overwrite=True)

    return "Totals have been calculated for snapshot " + snap + "!"

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

    # Make interpolated Menc_func using the table at this snapshot
    Menc_func = IUS(Menc_table['radius'][Menc_table['snapshot']==snap], \
      Menc_table['total_mass'][Menc_table['snapshot']==snap])

    # Specify the file where the list of satellites is saved
    sat_file = sat_dir + 'satellites.hdf5'
    sat = Table.read(sat_file, path='all_data')

    # Do the actual calculation
    if (surface_args[0]=='sphere'):
        message = calc_totals_sphere(ds, snap, zsnap, refine_width_kpc, tablename, surface_args, \
          Menc_func=Menc_func, sat=sat)
    if (surface_args[0]=='frustum'):
        message = calc_totals_frustum(ds, snap, zsnap, refine_width_kpc, tablename, surface_args, \
          Menc_func=Menc_func, sat=sat)
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
    prefix = output_dir + 'totals_halo_00' + args.halo + '/' + args.run + '/'
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
            tablename = prefix + snap + '_totals'
            # Do the actual calculation
            load_and_calculate(args.system, foggie_dir, run_dir, trackname, halo_c_v_name, snap, tablename, Menc_table, surface_args, sat_dir)
    else:
        # Split into a number of groupings equal to the number of processors
        # and run one process per processor
        for i in range(len(outs)//args.nproc):
            threads = []
            for j in range(args.nproc):
                snap = outs[args.nproc*i+j]
                tablename = prefix + snap + '_totals'
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
            tablename = prefix + snap + '_totals'
            threads.append(multi.Process(target=load_and_calculate, \
			   args=(args.system, foggie_dir, run_dir, trackname, halo_c_v_name, snap, tablename, Menc_table, surface_args, sat_dir)))
        for t in threads:
            t.start()
        for t in threads:
            t.join()

    print(str(datetime.datetime.now()))
    print("All snapshots finished!")
