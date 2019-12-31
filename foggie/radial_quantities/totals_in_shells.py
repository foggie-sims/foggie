"""
Filename: flux_tracking.py
Author: Cassi
Date created: 9-27-19
Date last modified: 12-22-19
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
import multiprocessing as multi
import datetime
from scipy.interpolate import InterpolatedUnivariateSpline as IUS

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

    table_units = {'redshift':None,'quadrant':None,'radius':'kpc', \
             'net_mass':'Msun', 'net_metals':'Msun', \
             'mass_in':'Msun', 'mass_out':'Msun', \
             'metals_in':'Msun', 'metals_out':'Msun',\
             'net_cold_mass':'Msun', 'cold_mass_in':'Msun', 'cold_mass_out':'Msun', \
             'net_cool_mass':'Msun', 'cool_mass_in':'Msun', 'cool_mass_out':'Msun', \
             'net_warm_mass':'Msun', 'warm_mass_in':'Msun', 'warm_mass_out':'Msun', \
             'net_hot_mass':'Msun', 'hot_mass_in':'Msun', 'hot_mass_out':'Msun', \
             'net_cold_metals':'Msun', 'cold_metals_in':'Msun', 'cold_metals_out':'Msun', \
             'net_cool_metals':'Msun', 'cool_metals_in':'Msun', 'cool_metals_out':'Msun', \
             'net_warm_metals':'Msun', 'warm_metals_in':'Msun', 'warm_metals_out':'Msun', \
             'net_hot_metals':'Msun', 'hot_metals_in':'Msun', 'hot_metals_out':'Msun', \
             'net_kinetic_energy':'erg', 'net_thermal_energy':'erg', 'net_potential_energy':'erg', \
             'net_entropy':'cm**2*keV', \
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

def load_and_calculate(system, foggie_dir, run_dir, track, halo_c_v_name, snap, tablename, Menc_table, quadrants):
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

    # Make interpolated Menc_func using the table at this snapshot
    Menc_func = IUS(Menc_table['radius'][Menc_table['snapshot']==snap], \
      Menc_table['total_mass'][Menc_table['snapshot']==snap])

    # Do the actual calculation
    message = calc_totals(ds, snap, zsnap, refine_width_kpc, tablename, Menc_func=Menc_func)
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
    prefix = output_dir + 'totals_halo_00' + args.halo + '/' + args.run + '/'
    if not (os.path.exists(prefix)): os.system('mkdir -p ' + prefix)

    print('foggie_dir: ', foggie_dir)
    halo_c_v_name = track_dir + 'halo_c_v'

    # Load the mass enclosed profile
    Menc_table = Table.read(code_path + 'halo_infos/00' + args.halo + '/' + args.run + '/DD1202_DD1547_masses.hdf5', \
      path='all_data')

    # Loop over outputs, for either single-processor or parallel processor computing
    if (args.nproc==1):
        for i in range(len(outs)):
            snap = outs[i]
            # Make the output table name for this snapshot
            tablename = prefix + snap + '_totals'
            # Do the actual calculation
            load_and_calculate(args.system, foggie_dir, run_dir, trackname, halo_c_v_name, snap, tablename, Menc_table, args.quadrants)
    else:
        # Split into a number of groupings equal to the number of processors
        # and run one process per processor
        for i in range(len(outs)//args.nproc):
            threads = []
            for j in range(args.nproc):
                snap = outs[args.nproc*i+j]
                tablename = prefix + snap + '_totals'
                threads.append(multi.Process(target=load_and_calculate, \
			       args=(args.system, foggie_dir, run_dir, trackname, halo_c_v_name, snap, tablename, Menc_table, args.quadrants)))
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
			   args=(args.system, foggie_dir, run_dir, trackname, halo_c_v_name, snap, tablename, Menc_table, args.quadrants)))
        for t in threads:
            t.start()
        for t in threads:
            t.join()

    print(str(datetime.datetime.now()))
    print("All snapshots finished!")
