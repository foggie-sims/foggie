"""
Filename: pdf_averages_in_shells.py
Author: Cassi
Date created: 10-25-19
Date last modified: 10-25-19
This file takes command line arguments and computes averages and distributions of
properties in shells.

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

    parser.add_argument('--nproc', metavar='nproc', type=int, action='store', \
                        help='How many processes do you want? Default is 1 ' + \
                        '(no parallelization), if multiple outputs and multiple processors are' + \
                        ' specified, code will run one output per processor')
    parser.set_defaults(nproc=1)

    parser.add_argument('--local', dest='local', action='store_true', \
                        help='Use local simulation files? Default is no')
    parser.set_defaults(local=False)


    args = parser.parse_args()
    return args

def set_table_units(table_avg, table_pdf):
    '''Sets the units for the table_avg and the table_pdf. Note this needs to be updated whenever something is added to
    the tables. Returns the table.'''

    table_avg_units = {'redshift':None, 'radius':'kpc', \
                        'temperature':'K', 'density':'g/cm**3', 'pressure':'erg/cm**3', 'entropy':'keV*cm**2', \
                        'radial_velocity':'km/s', 'theta_velocity':'km/s', 'phi_velocity':'km/s', 'tangential_velocity':'km/s', \
                        'temperature_in':'K', 'temperature_out':'K', 'density_in':'g/cm**3', 'density_out':'g/cm**3', \
                        'pressure_in':'erg/cm**3', 'pressure_out':'erg/cm**3', 'entropy_in':'keV*cm**2', 'entropy_out':'keV*cm**2', \
                        'radial_velocity_in':'km/s', 'radial_velocity_out':'km/s', 'theta_velocity_in':'km/s', 'theta_velocity_out':'km/s', \
                        'phi_velocity_in':'km/s', 'phi_velocity_out':'km/s', 'tangential_velocity_in':'km/s', 'tangential_velocity_out':'km/s', \
                        'cold_temperature':'K', 'cold_temperature_in':'K', 'cold_temperature_out':'K', \
                        'cool_temperature':'K', 'cool_temperature_in':'K', 'cool_temperature_out':'K', \
                        'warm_temperature':'K', 'warm_temperature_in':'K', 'warm_temperature_out':'K', \
                        'hot_temperature':'K', 'hot_temperature_in':'K', 'hot_temperature_out':'K', \
                        'cold_density':'g/cm**3', 'cold_density_in':'g/cm**3', 'cold_density_out':'g/cm**3', \
                        'cool_density':'g/cm**3', 'cool_density_in':'g/cm**3', 'cool_density_out':'g/cm**3', \
                        'warm_density':'g/cm**3', 'warm_density_in':'g/cm**3', 'warm_density_out':'g/cm**3', \
                        'hot_density':'g/cm**3', 'hot_density_in':'g/cm**3', 'hot_density_out':'g/cm**3', \
                        'cold_pressure':'erg/cm**3', 'cold_pressure_in':'erg/cm**3', 'cold_pressure_out':'erg/cm**3', \
                        'cool_pressure':'erg/cm**3', 'cool_pressure_in':'erg/cm**3', 'cool_pressure_out':'erg/cm**3', \
                        'warm_pressure':'erg/cm**3', 'warm_pressure_in':'erg/cm**3', 'warm_pressure_out':'erg/cm**3', \
                        'hot_pressure':'erg/cm**3', 'hot_pressure_in':'erg/cm**3', 'hot_pressure_out':'erg/cm**3', \
                        'cold_entropy':'keV*cm**2', 'cold_entropy_in':'keV*cm**2', 'cold_entropy_out':'keV*cm**2', \
                        'cool_entropy':'keV*cm**2', 'cool_entropy_in':'keV*cm**2', 'cool_entropy_out':'keV*cm**2', \
                        'warm_entropy':'keV*cm**2', 'warm_entropy_in':'keV*cm**2', 'warm_entropy_out':'keV*cm**2', \
                        'hot_entropy':'keV*cm**2', 'hot_entropy_in':'keV*cm**2', 'hot_entropy_out':'keV*cm**2', \
                        'cold_radial_velocity':'km/s', 'cold_radial_velocity_in':'km/s', 'cold_radial_velocity_out':'km/s', \
                        'cool_radial_velocity':'km/s', 'cool_radial_velocity_in':'km/s', 'cool_radial_velocity_out':'km/s', \
                        'warm_radial_velocity':'km/s', 'warm_radial_velocity_in':'km/s', 'warm_radial_velocity_out':'km/s', \
                        'hot_radial_velocity':'km/s', 'hot_radial_velocity_in':'km/s', 'hot_radial_velocity_out':'km/s', \
                        'cold_theta_velocity':'km/s', 'cold_theta_velocity_in':'km/s', 'cold_theta_velocity_out':'km/s', \
                        'cool_theta_velocity':'km/s', 'cool_theta_velocity_in':'km/s', 'cool_theta_velocity_out':'km/s', \
                        'warm_theta_velocity':'km/s', 'warm_theta_velocity_in':'km/s', 'warm_theta_velocity_out':'km/s', \
                        'hot_theta_velocity':'km/s', 'hot_theta_velocity_in':'km/s', 'hot_theta_velocity_out':'km/s', \
                        'cold_phi_velocity':'km/s', 'cold_phi_velocity_in':'km/s', 'cold_phi_velocity_out':'km/s', \
                        'cool_phi_velocity':'km/s', 'cool_phi_velocity_in':'km/s', 'cool_phi_velocity_out':'km/s', \
                        'warm_phi_velocity':'km/s', 'warm_phi_velocity_in':'km/s', 'warm_phi_velocity_out':'km/s', \
                        'hot_phi_velocity':'km/s', 'hot_phi_velocity_in':'km/s', 'hot_phi_velocity_out':'km/s', \
                        'cold_tangential_velocity':'km/s', 'cold_tangential_velocity_in':'km/s', 'cold_tangential_velocity_out':'km/s', \
                        'cool_tangential_velocity':'km/s', 'cool_tangential_velocity_in':'km/s', 'cool_tangential_velocity_out':'km/s', \
                        'warm_tangential_velocity':'km/s', 'warm_tangential_velocity_in':'km/s', 'warm_tangential_velocity_out':'km/s', \
                        'hot_tangential_velocity':'km/s', 'hot_tangential_velocity_in':'km/s', 'hot_tangential_velocity_out':'km/s'}
    for key in table_avg.keys():
        table_avg[key].unit = table_avg_units[key]

    table_pdf_units = {'redshift':None, 'radius':'kpc', \
                        'temperature':'K', 'temperature_pdf':None, 'density':'g/cm**3', 'density_pdf':None, \
                        'pressure':'erg/cm**3', 'pressure_pdf':None, 'entropy':'keV*cm**2', 'entropy_pdf':None, \
                        'radial_velocity':'km/s', 'radial_velocity_pdf':None, 'theta_velocity':'km/s', 'theta_velocity_pdf':None, \
                        'phi_velocity':'km/s', 'phi_velocity_pdf':None, 'tangential_velocity':'km/s', 'tangential_velocity_pdf':None, \
                        'cold_temperature':'K', 'cold_temperature_pdf':None, 'cool_temperature':'K', 'cool_temperature_pdf':None, \
                        'warm_temperature':'K', 'warm_temperature_pdf':None, 'hot_temperature':'K', 'hot_temperature_pdf':None, \
                        'cold_density':'g/cm**3', 'cold_density_pdf':None, 'cool_density':'g/cm**3', 'cool_density_pdf':None, \
                        'warm_density':'g/cm**3', 'warm_density_pdf':None, 'hot_density':'g/cm**3', 'hot_density_pdf':None, \
                        'cold_pressure':'erg/cm**3', 'cold_pressure_pdf':None, 'cool_pressure':'erg/cm**3', 'cool_pressure_pdf':None, \
                        'warm_pressure':'erg/cm**3', 'warm_pressure_pdf':None, 'hot_pressure':'erg/cm**3', 'hot_pressure_pdf':None, \
                        'cold_entropy':'keV*cm**2', 'cold_entropy_pdf':None, 'cool_entropy':'keV*cm**2', 'cool_entropy_pdf':None, \
                        'warm_entropy':'keV*cm**2', 'warm_entropy_pdf':None, 'hot_entropy':'keV*cm**2', 'hot_entropy_pdf':None, \
                        'cold_radial_velocity':'km/s', 'cold_radial_velocity_pdf':None, 'cool_radial_velocity':'km/s', 'cool_radial_velocity_pdf':None, \
                        'warm_radial_velocity':'km/s', 'warm_radial_velocity_pdf':None, 'hot_radial_velocity':'km/s', 'hot_radial_velocity_pdf':None, \
                        'cold_theta_velocity':'km/s', 'cold_theta_velocity_pdf':None, 'cool_theta_velocity':'km/s', 'cool_theta_velocity_pdf':None, \
                        'warm_theta_velocity':'km/s', 'warm_theta_velocity_pdf':None, 'hot_theta_velocity':'km/s', 'hot_theta_velocity_pdf':None, \
                        'cold_phi_velocity':'km/s', 'cold_phi_velocity_pdf':None, 'cool_phi_velocity':'km/s', 'cool_phi_velocity_pdf':None, \
                        'warm_phi_velocity':'km/s', 'warm_phi_velocity_pdf':None, 'hot_phi_velocity':'km/s', 'hot_phi_velocity_pdf':None, \
                        'cold_tangential_velocity':'km/s', 'cold_tangential_velocity_pdf':None, 'cool_tangential_velocity':'km/s', 'cool_tangential_velocity_pdf':None, \
                        'warm_tangential_velocity':'km/s', 'warm_tangential_velocity_pdf':None, 'hot_tangential_velocity':'km/s', 'hot_tangential_velocity_pdf':None}
    for key in table_pdf.keys():
        table_pdf[key].unit = table_pdf_units[key]

    return table_avg, table_pdf

def calc_shells(ds, snap, zsnap, refine_width_kpc, tablename, disk_rel=True):
    """Computes various average quantities and pdfs in spherical shells centered on the halo center.
    Takes the dataset for the snapshot 'ds', the name of the snapshot 'snap', the redshift of the
    snapshot 'zsnap', and the width of the refine box in kpc 'refine_width_kpc'
    and does the calculation, then writes a hdf5 table out to 'tablename'.
    """

    halo_center_kpc = ds.halo_center_kpc

    cmtopc = 3.086e18
    stoyr = 3.154e7
    gtoMsun = 1.989e33

    # Set up table of everything we want
    # NOTE: Make sure table units are updated when things are added to this table!
    data_avg = Table(names=('redshift', 'radius', \
                        'temperature', 'density', 'pressure', 'entropy', \
                        'radial_velocity', 'theta_velocity', 'phi_velocity', 'tangential_velocity', \
                        'temperature_in', 'temperature_out', 'density_in', 'density_out', \
                        'pressure_in', 'pressure_out', 'entropy_in', 'entropy_out', \
                        'radial_velocity_in', 'radial_velocity_out', 'theta_velocity_in', 'theta_velocity_out', \
                        'phi_velocity_in', 'phi_velocity_out', 'tangential_velocity_in', 'tangential_velocity_out', \
                        'cold_temperature', 'cold_temperature_in', 'cold_temperature_out', \
                        'cool_temperature', 'cool_temperature_in', 'cool_temperature_out', \
                        'warm_temperature', 'warm_temperature_in', 'warm_temperature_out', \
                        'hot_temperature', 'hot_temperature_in', 'hot_temperature_out', \
                        'cold_density', 'cold_density_in', 'cold_density_out', \
                        'cool_density', 'cool_density_in', 'cool_density_out', \
                        'warm_density', 'warm_density_in', 'warm_density_out', \
                        'hot_density', 'hot_density_in', 'hot_density_out', \
                        'cold_pressure', 'cold_pressure_in', 'cold_pressure_out', \
                        'cool_pressure', 'cool_pressure_in', 'cool_pressure_out', \
                        'warm_pressure', 'warm_pressure_in', 'warm_pressure_out', \
                        'hot_pressure', 'hot_pressure_in', 'hot_pressure_out', \
                        'cold_entropy', 'cold_entropy_in', 'cold_entropy_out', \
                        'cool_entropy', 'cool_entropy_in', 'cool_entropy_out', \
                        'warm_entropy', 'warm_entropy_in', 'warm_entropy_out', \
                        'hot_entropy', 'hot_entropy_in', 'hot_entropy_out', \
                        'cold_radial_velocity', 'cold_radial_velocity_in', 'cold_radial_velocity_out', \
                        'cool_radial_velocity', 'cool_radial_velocity_in', 'cool_radial_velocity_out', \
                        'warm_radial_velocity', 'warm_radial_velocity_in', 'warm_radial_velocity_out', \
                        'hot_radial_velocity', 'hot_radial_velocity_in', 'hot_radial_velocity_out', \
                        'cold_theta_velocity', 'cold_theta_velocity_in', 'cold_theta_velocity_out', \
                        'cool_theta_velocity', 'cool_theta_velocity_in', 'cool_theta_velocity_out', \
                        'warm_theta_velocity', 'warm_theta_velocity_in', 'warm_theta_velocity_out', \
                        'hot_theta_velocity', 'hot_theta_velocity_in', 'hot_theta_velocity_out', \
                        'cold_phi_velocity', 'cold_phi_velocity_in', 'cold_phi_velocity_out', \
                        'cool_phi_velocity', 'cool_phi_velocity_in', 'cool_phi_velocity_out', \
                        'warm_phi_velocity', 'warm_phi_velocity_in', 'warm_phi_velocity_out', \
                        'hot_phi_velocity', 'hot_phi_velocity_in', 'hot_phi_velocity_out', \
                        'cold_tangential_velocity', 'cold_tangential_velocity_in', 'cold_tangential_velocity_out', \
                        'cool_tangential_velocity', 'cool_tangential_velocity_in', 'cool_tangential_velocity_out', \
                        'warm_tangential_velocity', 'warm_tangential_velocity_in', 'warm_tangential_velocity_out', \
                        'hot_tangential_velocity', 'hot_tangential_velocity_in', 'hot_tangential_velocity_out'),
                 dtype=('f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', \
                        'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', \
                        'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', \
                        'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', \
                        'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', \
                        'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', \
                        'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', \
                        'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', \
                        'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', \
                        'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', \
                        'f8', 'f8'))

    data_pdf = Table(names=('redshift', 'radius', \
                        'temperature', 'temperature_pdf', 'density', 'density_pdf', \
                        'pressure', 'pressure_pdf', 'entropy', 'entropy_pdf', \
                        'radial_velocity', 'radial_velocity_pdf', 'theta_velocity', 'theta_velocity_pdf', \
                        'phi_velocity', 'phi_velocity_pdf', 'tangential_velocity', 'tangential_velocity_pdf', \
                        'cold_temperature', 'cold_temperature_pdf', 'cool_temperature', 'cool_temperature_pdf', \
                        'warm_temperature', 'warm_temperature_pdf', 'hot_temperature', 'hot_temperature_pdf', \
                        'cold_density', 'cold_density_pdf', 'cool_density', 'cool_density_pdf', \
                        'warm_density', 'warm_density_pdf', 'hot_density', 'hot_density_pdf', \
                        'cold_pressure', 'cold_pressure_pdf', 'cool_pressure', 'cool_pressure_pdf', \
                        'warm_pressure', 'warm_pressure_pdf', 'hot_pressure', 'hot_pressure_pdf', \
                        'cold_entropy', 'cold_entropy_pdf', 'cool_entropy', 'cool_entropy_pdf', \
                        'warm_entropy', 'warm_entropy_pdf', 'hot_entropy', 'hot_entropy_pdf', \
                        'cold_radial_velocity', 'cold_radial_velocity_pdf', 'cool_radial_velocity', 'cool_radial_velocity_pdf', \
                        'warm_radial_velocity', 'warm_radial_velocity_pdf', 'hot_radial_velocity', 'hot_radial_velocity_pdf', \
                        'cold_theta_velocity', 'cold_theta_velocity_pdf', 'cool_theta_velocity', 'cool_theta_velocity_pdf', \
                        'warm_theta_velocity', 'warm_theta_velocity_pdf', 'hot_theta_velocity', 'hot_theta_velocity_pdf', \
                        'cold_phi_velocity', 'cold_phi_velocity_pdf', 'cool_phi_velocity', 'cool_phi_velocity_pdf', \
                        'warm_phi_velocity', 'warm_phi_velocity_pdf', 'hot_phi_velocity', 'hot_phi_velocity_pdf', \
                        'cold_tangential_velocity', 'cold_tangential_velocity_pdf', 'cool_tangential_velocity', 'cool_tangential_velocity_pdf', \
                        'warm_tangential_velocity', 'warm_tangential_velocity_pdf', 'hot_tangential_velocity', 'hot_tangential_velocity_pdf'),
                 dtype=('f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', \
                        'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', \
                        'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', \
                        'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', \
                        'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', \
                        'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', \
                        'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8'))

    # Define the radii of the spherical shells where we want to calculate fluxes
    radii = 0.5*refine_width_kpc * np.arange(0.1, 0.9, 0.01)

    # Load arrays of all fields we need
    print('Loading field arrays')
    sphere = ds.sphere(halo_center_kpc, radii[-1])

    rad_vel = sphere['gas','radial_velocity_corrected'].in_units('km/s').v
    radius = sphere['gas','radius_corrected'].in_units('kpc').v
    temperature = sphere['gas','temperature'].in_units('K').v
    density = sphere['gas','density'].in_units('g/cm**3').v
    pressure = sphere['gas','pressure'].in_units('erg/cm**3').v
    if (disk_rel):
        theta_vel = sphere['gas','vtheta_disk'].in_units('km/s').v
        phi_vel = sphere['gas','vphi_disk'].in_units('km/s').v
        tan_vel = sphere['gas','vtan_disk'].in_units('km/s').v
    else:
        theta_vel = sphere['gas','theta_velocity_corrected'].in_units('km/s').v
        phi_vel = sphere['gas','phi_velocity_corrected'].in_units('km/s').v
        tan_vel = sphere['gas','tangential_velocity_corrected'].in_units('km/s').v
    entropy = sphere['gas','entropy'].in_units('keV*cm**2').v
    cell_mass = sphere['gas','cell_mass'].in_units('Msun').v
    cell_volume = sphere['gas','cell_volume'].in_units('kpc**3').v

    # Loop over radii
    for i in range(len(radii)-1):
        r_low = radii[i]
        r_high = radii[i+1]
        dr = r_high - r_low
        r = (r_low + r_high)/2.

        if (i%10==0): print("Computing radius " + str(i) + "/" + str(len(radii)-1) + \
                            " for snapshot " + snap)

        # Cut the data for this shell
        rad_vel_shell = rad_vel[(radius >= r_low.v) & (radius < r_high.v)]
        radius_shell = radius[(radius >= r_low.v) & (radius < r_high.v)]
        temperature_shell = temperature[(radius >= r_low.v) & (radius < r_high.v)]
        density_shell = density[(radius >= r_low.v) & (radius < r_high.v)]
        pressure_shell = pressure[(radius >= r_low.v) & (radius < r_high.v)]
        theta_vel_shell = theta_vel[(radius >= r_low.v) & (radius < r_high.v)]
        phi_vel_shell = phi_vel[(radius >= r_low.v) & (radius < r_high.v)]
        tan_vel_shell = tan_vel[(radius >= r_low.v) & (radius < r_high.v)]
        entropy_shell = entropy[(radius >= r_low.v) & (radius < r_high.v)]
        cell_mass_shell = cell_mass[(radius >= r_low.v) & (radius < r_high.v)]
        cell_volume_shell = cell_volume[(radius >= r_low.v) & (radius < r_high.v)]

        # Cut the data on temperature and radial velocity for in and out averages
        # For each field, it is a nested list where the top index is 0 through 4 for temperature phases
        # (all, cold, cool, warm, hot) and the second index is 0 through 2 for radial velocity (all, in, out)
        rad_vel_cut = []
        radius_cut = []
        temperature_cut = []
        density_cut = []
        pressure_cut = []
        theta_vel_cut = []
        phi_vel_cut = []
        tan_vel_cut = []
        entropy_cut = []
        cell_mass_cut = []
        cell_volume_cut = []
        for j in range(5):
            rad_vel_cut.append([])
            radius_cut.append([])
            temperature_cut.append([])
            density_cut.append([])
            pressure_cut.append([])
            theta_vel_cut.append([])
            phi_vel_cut.append([])
            tan_vel_cut.append([])
            entropy_cut.append([])
            cell_mass_cut.append([])
            cell_volume_cut.append([])
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
            rad_vel_cut[j].append(rad_vel_shell[(temperature_shell > t_low) & (temperature_shell < t_high)])
            radius_cut[j].append(radius_shell[(temperature_shell > t_low) & (temperature_shell < t_high)])
            temperature_cut[j].append(temperature_shell[(temperature_shell > t_low) & (temperature_shell < t_high)])
            density_cut[j].append(density_shell[(temperature_shell > t_low) & (temperature_shell < t_high)])
            pressure_cut[j].append(pressure_shell[(temperature_shell > t_low) & (temperature_shell < t_high)])
            theta_vel_cut[j].append(theta_vel_shell[(temperature_shell > t_low) & (temperature_shell < t_high)])
            phi_vel_cut[j].append(phi_vel_shell[(temperature_shell > t_low) & (temperature_shell < t_high)])
            tan_vel_cut[j].append(tan_vel_shell[(temperature_shell > t_low) & (temperature_shell < t_high)])
            entropy_cut[j].append(entropy_shell[(temperature_shell > t_low) & (temperature_shell < t_high)])
            cell_mass_cut[j].append(cell_mass_shell[(temperature_shell > t_low) & (temperature_shell < t_high)])
            cell_volume_cut[j].append(cell_volume_shell[(temperature_shell > t_low) & (temperature_shell < t_high)])
            for k in range(2):
                if (k==0): fac = -1.
                if (k==1): fac = 1.
                rad_vel_cut[j].append(rad_vel_cut[j][0][(fac*rad_vel_cut[j][0] > 0.)])
                radius_cut[j].append(radius_cut[j][0][(fac*rad_vel_cut[j][0] > 0.)])
                temperature_cut[j].append(temperature_cut[j][0][(fac*rad_vel_cut[j][0] > 0.)])
                density_cut[j].append(density_cut[j][0][(fac*rad_vel_cut[j][0] > 0.)])
                pressure_cut[j].append(pressure_cut[j][0][(fac*rad_vel_cut[j][0] > 0.)])
                theta_vel_cut[j].append(theta_vel_cut[j][0][(fac*rad_vel_cut[j][0] > 0.)])
                phi_vel_cut[j].append(phi_vel_cut[j][0][(fac*rad_vel_cut[j][0] > 0.)])
                tan_vel_cut[j].append(tan_vel_cut[j][0][(fac*rad_vel_cut[j][0] > 0.)])
                entropy_cut[j].append(entropy_cut[j][0][(fac*rad_vel_cut[j][0] > 0.)])
                cell_mass_cut[j].append(cell_mass_cut[j][0][(fac*rad_vel_cut[j][0] > 0.)])
                cell_volume_cut[j].append(cell_volume_cut[j][0][(fac*rad_vel_cut[j][0] > 0.)])

        # Compute averages and histograms
        # For each type parameter, the averages are a nested list where the top index goes from 0 to 4 and is
        # the phase of gas (all, cold, cool, warm, hot) and the second index goes from 0 to 2 and is
        # the radial velocity (all, in, out).
        # Ex. the average of a parameter is avg[0][0], the average of a parameter with negative radial velocities is avg[0][1], the warm phase average of a parameter with positive radial velocity is avg[3][2]
        # For the distribution and histograms, they do not have the second index
        rad_vel_avg = []
        density_avg = []
        temperature_avg = []
        pressure_avg = []
        theta_vel_avg = []
        phi_vel_avg = []
        tan_vel_avg = []
        entropy_avg = []

        rad_vel_dist = []
        rad_vel_hist = []
        density_dist = []
        density_hist = []
        temperature_dist = []
        temperature_hist = []
        pressure_dist = []
        pressure_hist = []
        entropy_dist = []
        entropy_hist = []
        theta_vel_dist = []
        theta_vel_hist = []
        phi_vel_dist = []
        phi_vel_hist = []
        tan_vel_dist = []
        tan_vel_hist = []

        for j in range(5):
            rad_vel_avg.append([])
            density_avg.append([])
            temperature_avg.append([])
            pressure_avg.append([])
            theta_vel_avg.append([])
            phi_vel_avg.append([])
            tan_vel_avg.append([])
            entropy_avg.append([])
            for k in range(3):
                if (np.sum(cell_mass_cut[j][k])!=0):
                    rad_vel_avg[j].append(np.average(rad_vel_cut[j][k], weights=cell_mass_cut[j][k]))
                    theta_vel_avg[j].append(np.average(theta_vel_cut[j][k], weights=cell_mass_cut[j][k]))
                    phi_vel_avg[j].append(np.average(phi_vel_cut[j][k], weights=cell_mass_cut[j][k]))
                    tan_vel_avg[j].append(np.average(tan_vel_cut[j][k], weights=cell_mass_cut[j][k]))
                else:
                    rad_vel_avg[j].append(0.)
                    theta_vel_avg[j].append(0.)
                    phi_vel_avg[j].append(0.)
                    tan_vel_avg[j].append(0.)
                if (np.sum(cell_volume_cut[j][k])!=0):
                    density_avg[j].append(np.average(density_cut[j][k], weights=cell_volume_cut[j][k]))
                    temperature_avg[j].append(np.average(temperature_cut[j][k], weights=cell_volume_cut[j][k]))
                    pressure_avg[j].append(np.average(pressure_cut[j][k], weights=cell_volume_cut[j][k]))
                    entropy_avg[j].append(np.average(entropy_cut[j][k], weights=cell_volume_cut[j][k]))
                else:
                    density_avg[j].append(0.)
                    temperature_avg[j].append(0.)
                    pressure_avg[j].append(0.)
                    entropy_avg[j].append(0.)

            if (np.sum(cell_mass_cut[j][0])!=0):
                hist, dist = np.histogram(rad_vel_cut[j][0], bins=128, weights=cell_mass_cut[j][0], density=True)
                dist_cen = []
                for b in range(len(dist)-1):
                    dist_cen.append(0.5*(dist[b]+dist[b+1]))
                rad_vel_dist.append(dist_cen)
                rad_vel_hist.append(hist)
                hist, dist = np.histogram(theta_vel_cut[j][0], bins=128, weights=cell_mass_cut[j][0], density=True)
                dist_cen = []
                for b in range(len(dist)-1):
                    dist_cen.append(0.5*(dist[b]+dist[b+1]))
                theta_vel_dist.append(dist_cen)
                theta_vel_hist.append(hist)
                hist, dist = np.histogram(phi_vel_cut[j][0], bins=128, weights=cell_mass_cut[j][0], density=True)
                dist_cen = []
                for b in range(len(dist)-1):
                    dist_cen.append(0.5*(dist[b]+dist[b+1]))
                phi_vel_dist.append(dist_cen)
                phi_vel_hist.append(hist)
                hist, dist = np.histogram(tan_vel_cut[j][0], bins=128, weights=cell_mass_cut[j][0], density=True)
                dist_cen = []
                for b in range(len(dist)-1):
                    dist_cen.append(0.5*(dist[b]+dist[b+1]))
                tan_vel_dist.append(dist_cen)
                tan_vel_hist.append(hist)
            else:
                rad_vel_dist.append(np.zeros(128))
                rad_vel_hist.append(np.zeros(128))
                theta_vel_dist.append(np.zeros(128))
                theta_vel_hist.append(np.zeros(128))
                phi_vel_dist.append(np.zeros(128))
                phi_vel_hist.append(np.zeros(128))
                tan_vel_dist.append(np.zeros(128))
                tan_vel_hist.append(np.zeros(128))
            if (np.sum(cell_volume_cut[j][0])!=0):
                hist, dist = np.histogram(np.log10(density_cut[j][0]), bins=128, weights=cell_volume_cut[j][0], density=True)
                dist_cen = []
                for b in range(len(dist)-1):
                    dist_cen.append(0.5*(dist[b]+dist[b+1]))
                density_dist.append(10**np.array(dist_cen))
                density_hist.append(hist)
                hist, dist = np.histogram(np.log10(temperature_cut[j][0]), bins=128, weights=cell_volume_cut[j][0], density=True)
                dist_cen = []
                for b in range(len(dist)-1):
                    dist_cen.append(0.5*(dist[b]+dist[b+1]))
                temperature_dist.append(10**np.array(dist_cen))
                temperature_hist.append(hist)
                hist, dist = np.histogram(np.log10(pressure_cut[j][0]), bins=128, weights=cell_volume_cut[j][0], density=True)
                dist_cen = []
                for b in range(len(dist)-1):
                    dist_cen.append(0.5*(dist[b]+dist[b+1]))
                pressure_dist.append(10**np.array(dist_cen))
                pressure_hist.append(hist)
                hist, dist = np.histogram(np.log10(entropy_cut[j][0]), bins=128, weights=cell_volume_cut[j][0], density=True)
                dist_cen = []
                for b in range(len(dist)-1):
                    dist_cen.append(0.5*(dist[b]+dist[b+1]))
                entropy_dist.append(np.array(dist_cen))
                entropy_hist.append(hist)
            else:
                density_dist.append(np.zeros(128))
                density_hist.append(np.zeros(128))
                temperature_dist.append(np.zeros(128))
                temperature_hist.append(np.zeros(128))
                pressure_dist.append(np.zeros(128))
                pressure_hist.append(np.zeros(128))
                entropy_dist.append(np.zeros(128))
                entropy_hist.append(np.zeros(128))

        # Add averages to the table
        data_avg.add_row([zsnap, r, \
                        temperature_avg[0][0], density_avg[0][0], pressure_avg[0][0], entropy_avg[0][0], \
                        rad_vel_avg[0][0], theta_vel_avg[0][0], phi_vel_avg[0][0], tan_vel_avg[0][0], \
                        temperature_avg[0][1], temperature_avg[0][2], density_avg[0][1], density_avg[0][2], \
                        pressure_avg[0][1], pressure_avg[0][2], entropy_avg[0][1], entropy_avg[0][2], \
                        rad_vel_avg[0][1], rad_vel_avg[0][2], theta_vel_avg[0][1], theta_vel_avg[0][2], \
                        phi_vel_avg[0][1], phi_vel_avg[0][2], tan_vel_avg[0][1], tan_vel_avg[0][2], \
                        temperature_avg[1][0], temperature_avg[1][1], temperature_avg[1][2], \
                        temperature_avg[2][0], temperature_avg[2][1], temperature_avg[2][2], \
                        temperature_avg[3][0], temperature_avg[3][1], temperature_avg[3][2], \
                        temperature_avg[4][0], temperature_avg[4][1], temperature_avg[4][2], \
                        density_avg[1][0], density_avg[1][1], density_avg[1][2], \
                        density_avg[2][0], density_avg[2][1], density_avg[2][2], \
                        density_avg[3][0], density_avg[3][1], density_avg[3][2], \
                        density_avg[4][0], density_avg[4][1], density_avg[4][2], \
                        pressure_avg[1][0], pressure_avg[1][1], pressure_avg[1][2], \
                        pressure_avg[2][0], pressure_avg[2][1], pressure_avg[2][2], \
                        pressure_avg[3][0], pressure_avg[3][1], pressure_avg[3][2], \
                        pressure_avg[4][0], pressure_avg[4][1], pressure_avg[4][2], \
                        entropy_avg[1][0], entropy_avg[1][1], entropy_avg[1][2], \
                        entropy_avg[2][0], entropy_avg[2][1], entropy_avg[2][2], \
                        entropy_avg[3][0], entropy_avg[3][1], entropy_avg[3][2], \
                        entropy_avg[4][0], entropy_avg[4][1], entropy_avg[4][2], \
                        rad_vel_avg[1][0], rad_vel_avg[1][1], rad_vel_avg[1][2], \
                        rad_vel_avg[2][0], rad_vel_avg[2][1], rad_vel_avg[2][2], \
                        rad_vel_avg[3][0], rad_vel_avg[3][1], rad_vel_avg[3][2], \
                        rad_vel_avg[4][0], rad_vel_avg[4][1], rad_vel_avg[4][2], \
                        theta_vel_avg[1][0], theta_vel_avg[1][1], theta_vel_avg[1][2], \
                        theta_vel_avg[2][0], theta_vel_avg[2][1], theta_vel_avg[2][2], \
                        theta_vel_avg[3][0], theta_vel_avg[3][1], theta_vel_avg[3][2], \
                        theta_vel_avg[4][0], theta_vel_avg[4][1], theta_vel_avg[4][2], \
                        phi_vel_avg[1][0], phi_vel_avg[1][1], phi_vel_avg[1][2], \
                        phi_vel_avg[2][0], phi_vel_avg[2][1], phi_vel_avg[2][2], \
                        phi_vel_avg[3][0], phi_vel_avg[3][1], phi_vel_avg[3][2], \
                        phi_vel_avg[4][0], phi_vel_avg[4][1], phi_vel_avg[4][2], \
                        tan_vel_avg[1][0], tan_vel_avg[1][1], tan_vel_avg[1][2], \
                        tan_vel_avg[2][0], tan_vel_avg[2][1], tan_vel_avg[2][2], \
                        tan_vel_avg[3][0], tan_vel_avg[3][1], tan_vel_avg[3][2], \
                        tan_vel_avg[4][0], tan_vel_avg[4][1], tan_vel_avg[4][2]])

        # Add PDFs to table
        for b in range(128):
            data_pdf.add_row([zsnap, r, \
                            temperature_dist[0][b], temperature_hist[0][b], density_dist[0][b], density_hist[0][b], \
                            pressure_dist[0][b], pressure_hist[0][b], entropy_dist[0][b], entropy_hist[0][b], \
                            rad_vel_dist[0][b], rad_vel_hist[0][b], theta_vel_dist[0][b], theta_vel_hist[0][b], \
                            phi_vel_dist[0][b], phi_vel_hist[0][b], tan_vel_dist[0][b], tan_vel_hist[0][b], \
                            temperature_dist[1][b], temperature_hist[1][b], temperature_dist[2][b], temperature_hist[2][b], \
                            temperature_dist[3][b], temperature_hist[3][b], temperature_dist[4][b], temperature_hist[4][b], \
                            density_dist[1][b], density_hist[1][b], density_dist[2][b], density_hist[2][b], \
                            density_dist[3][b], density_hist[3][b], density_dist[4][b], density_hist[4][b], \
                            pressure_dist[1][b], pressure_hist[1][b], pressure_dist[2][b], pressure_hist[2][b], \
                            pressure_dist[3][b], pressure_hist[3][b], pressure_dist[4][b], pressure_hist[4][b], \
                            entropy_dist[1][b], entropy_hist[1][b], entropy_dist[2][b], entropy_hist[2][b], \
                            entropy_dist[3][b], entropy_hist[3][b], entropy_dist[4][b], entropy_hist[4][b], \
                            rad_vel_dist[1][b], rad_vel_hist[1][b], rad_vel_dist[2][b], rad_vel_hist[2][b], \
                            rad_vel_dist[3][b], rad_vel_hist[3][b], rad_vel_dist[4][b], rad_vel_hist[4][b], \
                            theta_vel_dist[1][b], theta_vel_hist[1][b], theta_vel_dist[2][b], theta_vel_hist[2][b], \
                            theta_vel_dist[3][b], theta_vel_hist[3][b], theta_vel_dist[4][b], theta_vel_hist[4][b], \
                            phi_vel_dist[1][b], phi_vel_hist[1][b], phi_vel_dist[2][b], phi_vel_hist[2][b], \
                            phi_vel_dist[3][b], phi_vel_hist[3][b], phi_vel_dist[4][b], phi_vel_hist[4][b], \
                            tan_vel_dist[1][b], tan_vel_hist[1][b], tan_vel_dist[2][b], tan_vel_hist[2][b], \
                            tan_vel_dist[3][b], tan_vel_hist[3][b], tan_vel_dist[4][b], tan_vel_hist[4][b]])


    # Save to file
    data_avg, data_pdf = set_table_units(data_avg, data_pdf)
    data_avg.write(tablename + '_avg.hdf5', path='all_data', serialize_meta=True, overwrite=True)
    data_pdf.write(tablename + '_pdf.hdf5', path='all_data', serialize_meta=True, overwrite=True)

    return "Averages and PDFs have been calculated for snapshot" + snap + "!"

def load_and_calculate(system, foggie_dir, run_dir, track, halo_c_v_name, snap, tablename):
    '''This function loads a specified snapshot 'snap' located in the 'run_dir' within the
    'foggie_dir', the halo track 'track', the name of the table to output, and a boolean
    'quadrants' that specifies whether or not to compute in quadrants vs. the whole domain, then
    does the calculation on the loaded snapshot.'''

    if (halo_c_v_name=='halo_c_v'):
        use_halo_c_v=False
    else:
        use_halo_c_v=True

    snap_name = foggie_dir + run_dir + snap + '/' + snap
    if (system=='pleiades_cassi'):
        print('Copying directory to /tmp')
        snap_dir = '/tmp/' + snap
        shutil.copytree(foggie_dir + run_dir + snap, snap_dir)
        snap_name = snap_dir + '/' + snap
    ds, refine_box, refine_box_center, refine_width = load(snap_name, track, disk_relative=True, \
                                                           use_halo_c_v=use_halo_c_v, halo_c_v_name=halo_c_v_name)
    refine_width_kpc = YTArray([refine_width], 'kpc')
    zsnap = ds.get_parameter('CosmologyCurrentRedshift')

    # Do the actual calculation
    message = calc_shells(ds, snap, zsnap, refine_width_kpc, tablename, disk_rel=True)
    if (system=='pleiades_cassi'):
        print('Deleting directory from /tmp')
        shutil.rmtree(snap_dir)
    print(message)
    print(str(datetime.datetime.now()))


if __name__ == "__main__":
    args = parse_args()
    print('Halo:', args.halo)
    print('Run:', args.run)
    print('System:', args.system)
    print('Local?', args.local)
    foggie_dir, output_dir, run_dir, trackname, haloname, spectra_dir, infofile = get_run_loc_etc(args)
    if (args.system=='pleiades_cassi'):
        code_path = '/home5/clochhaa/FOGGIE/foggie/foggie/'
    elif (args.system=='cassiopeia'):
        code_path = '/Users/clochhaas/Documents/Research/FOGGIE/Analysis_Code/foggie/foggie/'
        if (args.local):
            foggie_dir = '/Users/clochhaas/Documents/Research/FOGGIE/Simulation_Data/'
    if (args.run=='nref11c_nref9f'):
        track_dir = code_path + 'halo_infos/00' + args.halo + '/' + args.run + '/'
    else:
        track_dir = ''

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
    prefix = output_dir + 'shells_halo_00' + args.halo + '/' + args.run + '/'
    if not (os.path.exists(prefix)): os.system('mkdir -p ' + prefix)

    print('foggie_dir: ', foggie_dir)
    halo_c_v_name = track_dir + 'halo_c_v'

    # Loop over outputs, for either single-processor or parallel processor computing
    if (args.nproc==1):
        for i in range(len(outs)):
            snap = outs[i]
            # Make the output table name for this snapshot
            tablename = prefix + snap + '_shells'
            # Do the actual calculation
            load_and_calculate(args.system, foggie_dir, run_dir, trackname, halo_c_v_name, snap, tablename)
    else:
        # Split into a number of groupings equal to the number of processors
        # and run one process per processor
        for i in range(len(outs)//args.nproc):
            threads = []
            for j in range(args.nproc):
                snap = outs[args.nproc*i+j]
                tablename = prefix + snap + '_shells'
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
            tablename = prefix + snap + '_shells'
            threads.append(multi.Process(target=load_and_calculate, \
			   args=(args.system, foggie_dir, run_dir, trackname, halo_c_v_name, snap, tablename)))
        for t in threads:
            t.start()
        for t in threads:
            t.join()

    print(str(datetime.datetime.now()))
    print("All snapshots finished!")
