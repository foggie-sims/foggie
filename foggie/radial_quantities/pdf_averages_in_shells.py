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


    args = parser.parse_args()
    return args

def set_table_units(table_avg, table_pdf):
    '''Sets the units for the table_avg and the table_pdf. Note this needs to be updated whenever something is added to
    the tables. Returns the table.'''

    table_avg_units = {'redshift':None,'quadrant':None,'radius':'kpc',
                        'temperature':'K', 'density':'g/cm**3',
                        'radial_velocity':'km/s', 'radial_velocity_in':'km/s', 'radial_velocity_out':'km/s',
                        'tangential_velocity':'km/s', 'theta_velocity':'km/s', 'phi_velocity':'km/s', 'pressure':'erg/cm**3',
                        'entropy':'keV*cm**2', 'cold_temperature':'K', 'cold_density':'g/cm**3',
                        'cold_radial_velocity':'km/s', 'cold_radial_velocity_in':'km/s', 'cold_radial_velocity_out':'km/s',
                        'cold_tangential_velocity':'km/s', 'cold_theta_velocity':'km/s', 'cold_phi_velocity':'km/s',
                        'cold_pressure':'erg/cm**3', 'cold_entropy':'keV*cm**2', 'cool_temperature':'K', 'cool_density':'g/cm**3',
                        'cool_radial_velocity':'km/s', 'cool_radial_velocity_in':'km/s', 'cool_radial_velocity_out':'km/s',
                        'cool_tangential_velocity':'km/s', 'cool_theta_velocity':'km/s', 'cool_phi_velocity':'km/s',
                        'cool_pressure':'erg/cm**3', 'cool_entropy':'keV*cm**2', 'warm_temperature':'K', 'warm_density':'g/cm**3',
                        'warm_radial_velocity':'km/s', 'warm_radial_velocity_in':'km/s', 'warm_radial_velocity_out':'km/s',
                        'warm_tangential_velocity':'km/s', 'warm_theta_velocity':'km/s', 'warm_phi_velocity':'km/s',
                        'warm_pressure':'erg/cm**3', 'warm_entropy':'keV*cm**2', 'hot_temperature':'K', 'hot_density':'g/cm**3',
                        'hot_radial_velocity':'km/s', 'hot_radial_velocity_in':'km/s', 'hot_radial_velocity_out':'km/s',
                        'hot_tangential_velocity':'km/s', 'hot_theta_velocity':'km/s', 'hot_phi_velocity':'km/s',
                        'hot_pressure':'erg/cm**3', 'hot_entropy':'keV*cm**2'}
    for key in table_avg.keys():
        table_avg[key].unit = table_avg_units[key]

    table_pdf_units = {'redshift':None, 'quadrant':None, 'radius':'kpc',
                        'temperature':'K', 'temperature_pdf':None,
                        'density':'g/cm**3', 'density_pdf':None, 'radial_velocity':'km/s', 'radial_velocity_pdf':None,
                        'theta_velocity':'km/s', 'theta_velocity_pdf':None, 'phi_velocity':'km/s', 'phi_velocity_pdf':None,
                        'pressure':'erg/cm**3', 'pressure_pdf':None, 'cold_temperature':'K', 'cold_temperature_pdf':None,
                        'cold_density':'g/cm**3', 'cold_density_pdf':None, 'cold_radial_velocity':'km/s', 'cold_radial_velocity_pdf':None,
                        'cold_theta_velocity':'km/s', 'cold_theta_velocity_pdf':None, 'cold_phi_velocity':'km/s', 'cold_phi_velocity_pdf':None,
                        'cold_pressure':'erg/cm**3', 'cold_pressure_pdf':None,'cool_temperature':'K', 'cool_temperature_pdf':None,
                        'cool_density':'g/cm**3', 'cool_density_pdf':None, 'cool_radial_velocity':'km/s', 'cool_radial_velocity_pdf':None,
                        'cool_theta_velocity':'km/s', 'cool_theta_velocity_pdf':None, 'cool_phi_velocity':'km/s', 'cool_phi_velocity_pdf':None,
                        'cool_pressure':'erg/cm**3', 'cool_pressure_pdf':None, 'warm_temperature':'K', 'warm_temperature_pdf':None,
                        'warm_density':'g/cm**3', 'warm_density_pdf':None, 'warm_radial_velocity':'km/s', 'warm_radial_velocity_pdf':None,
                        'warm_theta_velocity':'km/s', 'warm_theta_velocity_pdf':None, 'warm_phi_velocity':'km/s', 'warm_phi_velocity_pdf':None,
                        'warm_pressure':'erg/cm**3', 'warm_pressure_pdf':None,'hot_temperature':'K', 'hot_temperature_pdf':None,
                        'hot_density':'g/cm**3', 'hot_density_pdf':None, 'hot_radial_velocity':'km/s', 'hot_radial_velocity_pdf':None,
                        'hot_theta_velocity':'km/s', 'hot_theta_velocity_pdf':None, 'hot_phi_velocity':'km/s', 'hot_phi_velocity_pdf':None,
                        'hot_pressure':'erg/cm**3', 'hot_pressure_pdf':None}
    for key in table_pdf.keys():
        table_pdf[key].unit = table_pdf_units[key]

    return table_avg, table_pdf

def calc_shells(ds, snap, zsnap, refine_width_kpc, tablename):
    """Computes various average quantities and pdfs in spherical shells centered on the halo center.
    Takes the dataset for the snapshot 'ds', the name of the snapshot 'snap', the redshift of the
    snapshot 'zsnap', and the width of the refine box in kpc 'refine_width_kpc'
    and does the calculation, then writes a hdf5 table out to 'tablename'.
    """

    # Set up table of everything we want
    # NOTE: Make sure table units are updated when things are added to this table!
    data_avg = Table(names=('redshift', 'quadrant', 'radius', 'temperature', 'density',
                        'radial_velocity', 'radial_velocity_in', 'radial_velocity_out',
                        'tangential_velocity', 'theta_velocity', 'phi_velocity', 'pressure',
                        'entropy', 'cold_temperature', 'cold_density',
                        'cold_radial_velocity', 'cold_radial_velocity_in', 'cold_radial_velocity_out',
                        'cold_tangential_velocity', 'cold_theta_velocity', 'cold_phi_velocity',
                        'cold_pressure', 'cold_entropy', 'cool_temperature', 'cool_density',
                        'cool_radial_velocity', 'cool_radial_velocity_in', 'cool_radial_velocity_out',
                        'cool_tangential_velocity', 'cool_theta_velocity', 'cool_phi_velocity',
                        'cool_pressure', 'cool_entropy', 'warm_temperature', 'warm_density',
                        'warm_radial_velocity', 'warm_radial_velocity_in', 'warm_radial_velocity_out',
                        'warm_tangential_velocity', 'warm_theta_velocity', 'warm_phi_velocity',
                        'warm_pressure', 'warm_entropy', 'hot_temperature', 'hot_density',
                        'hot_radial_velocity', 'hot_radial_velocity_in', 'hot_radial_velocity_out',
                        'hot_tangential_velocity', 'hot_theta_velocity', 'hot_phi_velocity',
                        'hot_pressure', 'hot_entropy'),
                 dtype=('f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', \
                        'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', \
                        'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', \
                        'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', \
                        'f8', 'f8', 'f8', 'f8', 'f8'))

    data_pdf = Table(names=('redshift', 'quadrant', 'radius', 'temperature', 'temperature_pdf',
                        'density', 'density_pdf', 'radial_velocity', 'radial_velocity_pdf',
                        'theta_velocity', 'theta_velocity_pdf', 'phi_velocity', 'phi_velocity_pdf',
                        'pressure', 'pressure_pdf', 'cold_temperature', 'cold_temperature_pdf',
                        'cold_density', 'cold_density_pdf', 'cold_radial_velocity', 'cold_radial_velocity_pdf',
                        'cold_theta_velocity', 'cold_theta_velocity_pdf', 'cold_phi_velocity', 'cold_phi_velocity_pdf',
                        'cold_pressure', 'cold_pressure_pdf','cool_temperature', 'cool_temperature_pdf',
                        'cool_density', 'cool_density_pdf', 'cool_radial_velocity', 'cool_radial_velocity_pdf',
                        'cool_theta_velocity', 'cool_theta_velocity_pdf', 'cool_phi_velocity', 'cool_phi_velocity_pdf',
                        'cool_pressure', 'cool_pressure_pdf','warm_temperature', 'warm_temperature_pdf',
                        'warm_density', 'warm_density_pdf', 'warm_radial_velocity', 'warm_radial_velocity_pdf',
                        'warm_theta_velocity', 'warm_theta_velocity_pdf', 'warm_phi_velocity', 'warm_phi_velocity_pdf',
                        'warm_pressure', 'warm_pressure_pdf','hot_temperature', 'hot_temperature_pdf',
                        'hot_density', 'hot_density_pdf', 'hot_radial_velocity', 'hot_radial_velocity_pdf',
                        'hot_theta_velocity', 'hot_theta_velocity_pdf', 'hot_phi_velocity', 'hot_phi_velocity_pdf',
                        'hot_pressure', 'hot_pressure_pdf'),
                 dtype=('f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', \
                        'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', \
                        'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', \
                        'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', \
                        'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', \
                        'f8', 'f8', 'f8'))

    # Define the radii of the spherical shells where we want to calculate fluxes
    radii = 0.5*refine_width_kpc * np.arange(0.1, 1.0, 0.05)

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

        # Compute averages
        temperature = shell.quantities.weighted_average_quantity(('gas','temperature'), 'cell_volume')
        density = shell.quantities.weighted_average_quantity(('gas','density'), 'cell_volume')
        radial_velocity = shell.quantities.weighted_average_quantity(('gas','radial_velocity_corrected'), 'cell_mass')
        radial_velocity_in = shell_in.quantities.weighted_average_quantity(('gas','radial_velocity_corrected'), 'cell_mass')
        radial_velocity_out = shell_out.quantities.weighted_average_quantity(('gas','radial_velocity_corrected'), 'cell_mass')
        tangential_velocity = shell.quantities.weighted_average_quantity(('gas','vtan_disk'), 'cell_mass')
        theta_velocity = shell.quantities.weighted_average_quantity(('gas','vtheta_disk'), 'cell_mass')
        phi_velocity = shell.quantities.weighted_average_quantity(('gas','vphi_disk'), 'cell_mass')
        pressure = shell.quantities.weighted_average_quantity(('gas','pressure'), 'cell_volume')
        entropy = shell.quantities.weighted_average_quantity(('gas','entropy'), 'cell_volume')

        cold_temperature = shell_cold.quantities.weighted_average_quantity(('gas','temperature'), 'cell_volume')
        cold_density = shell_cold.quantities.weighted_average_quantity(('gas','density'), 'cell_volume')
        cold_radial_velocity = shell_cold.quantities.weighted_average_quantity(('gas','radial_velocity_corrected'), 'cell_mass')
        cold_radial_velocity_in = shell_in_cold.quantities.weighted_average_quantity(('gas','radial_velocity_corrected'), 'cell_mass')
        cold_radial_velocity_out = shell_out_cold.quantities.weighted_average_quantity(('gas','radial_velocity_corrected'), 'cell_mass')
        cold_tangential_velocity = shell_cold.quantities.weighted_average_quantity(('gas','vtan_disk'), 'cell_mass')
        cold_theta_velocity = shell_cold.quantities.weighted_average_quantity(('gas','vtheta_disk'), 'cell_mass')
        cold_phi_velocity = shell_cold.quantities.weighted_average_quantity(('gas','vphi_disk'), 'cell_mass')
        cold_pressure = shell_cold.quantities.weighted_average_quantity(('gas','pressure'), 'cell_volume')
        cold_entropy = shell_cold.quantities.weighted_average_quantity(('gas','entropy'), 'cell_volume')

        cool_temperature = shell_cool.quantities.weighted_average_quantity(('gas','temperature'), 'cell_volume')
        cool_density = shell_cool.quantities.weighted_average_quantity(('gas','density'), 'cell_volume')
        cool_radial_velocity = shell_cool.quantities.weighted_average_quantity(('gas','radial_velocity_corrected'), 'cell_mass')
        cool_radial_velocity_in = shell_in_cool.quantities.weighted_average_quantity(('gas','radial_velocity_corrected'), 'cell_mass')
        cool_radial_velocity_out = shell_out_cool.quantities.weighted_average_quantity(('gas','radial_velocity_corrected'), 'cell_mass')
        cool_tangential_velocity = shell_cool.quantities.weighted_average_quantity(('gas','vtan_disk'), 'cell_mass')
        cool_theta_velocity = shell_cool.quantities.weighted_average_quantity(('gas','vtheta_disk'), 'cell_mass')
        cool_phi_velocity = shell_cool.quantities.weighted_average_quantity(('gas','vphi_disk'), 'cell_mass')
        cool_pressure = shell_cool.quantities.weighted_average_quantity(('gas','pressure'), 'cell_volume')
        cool_entropy = shell_cool.quantities.weighted_average_quantity(('gas','entropy'), 'cell_volume')

        warm_temperature = shell_warm.quantities.weighted_average_quantity(('gas','temperature'), 'cell_volume')
        warm_density = shell_warm.quantities.weighted_average_quantity(('gas','density'), 'cell_volume')
        warm_radial_velocity = shell_warm.quantities.weighted_average_quantity(('gas','radial_velocity_corrected'), 'cell_mass')
        warm_radial_velocity_in = shell_in_warm.quantities.weighted_average_quantity(('gas','radial_velocity_corrected'), 'cell_mass')
        warm_radial_velocity_out = shell_out_warm.quantities.weighted_average_quantity(('gas','radial_velocity_corrected'), 'cell_mass')
        warm_tangential_velocity = shell_warm.quantities.weighted_average_quantity(('gas','vtan_disk'), 'cell_mass')
        warm_theta_velocity = shell_warm.quantities.weighted_average_quantity(('gas','vtheta_disk'), 'cell_mass')
        warm_phi_velocity = shell_warm.quantities.weighted_average_quantity(('gas','vphi_disk'), 'cell_mass')
        warm_pressure = shell_warm.quantities.weighted_average_quantity(('gas','pressure'), 'cell_volume')
        warm_entropy = shell_warm.quantities.weighted_average_quantity(('gas','entropy'), 'cell_volume')

        hot_temperature = shell_hot.quantities.weighted_average_quantity(('gas','temperature'), 'cell_volume')
        hot_density = shell_hot.quantities.weighted_average_quantity(('gas','density'), 'cell_volume')
        hot_radial_velocity = shell_hot.quantities.weighted_average_quantity(('gas','radial_velocity_corrected'), 'cell_mass')
        hot_radial_velocity_in = shell_in_hot.quantities.weighted_average_quantity(('gas','radial_velocity_corrected'), 'cell_mass')
        hot_radial_velocity_out = shell_out_hot.quantities.weighted_average_quantity(('gas','radial_velocity_corrected'), 'cell_mass')
        hot_tangential_velocity = shell_hot.quantities.weighted_average_quantity(('gas','vtan_disk'), 'cell_mass')
        hot_theta_velocity = shell_hot.quantities.weighted_average_quantity(('gas','vtheta_disk'), 'cell_mass')
        hot_phi_velocity = shell_hot.quantities.weighted_average_quantity(('gas','vphi_disk'), 'cell_mass')
        hot_pressure = shell_hot.quantities.weighted_average_quantity(('gas','pressure'), 'cell_volume')
        hot_entropy = shell_hot.quantities.weighted_average_quantity(('gas','entropy'), 'cell_volume')

        # Add everything to the table
        data_avg.add_row([zsnap, 0, r, temperature, density,
                            radial_velocity, radial_velocity_in, radial_velocity_out,
                            tangential_velocity, theta_velocity, phi_velocity, pressure,
                            entropy, cold_temperature, cold_density,
                            cold_radial_velocity, cold_radial_velocity_in, cold_radial_velocity_out,
                            cold_tangential_velocity, cold_theta_velocity, cold_phi_velocity,
                            cold_pressure, cold_entropy, cool_temperature, cool_density,
                            cool_radial_velocity, cool_radial_velocity_in, cool_radial_velocity_out,
                            cool_tangential_velocity, cool_theta_velocity, cool_phi_velocity,
                            cool_pressure, cool_entropy, warm_temperature, warm_density,
                            warm_radial_velocity, warm_radial_velocity_in, warm_radial_velocity_out,
                            warm_tangential_velocity, warm_theta_velocity, warm_phi_velocity,
                            warm_pressure, warm_entropy, hot_temperature, hot_density,
                            hot_radial_velocity, hot_radial_velocity_in, hot_radial_velocity_out,
                            hot_tangential_velocity, hot_theta_velocity, hot_phi_velocity,
                            hot_pressure, hot_entropy])

        # Compute PDFs
        plot = yt.ProfilePlot(shell, ('gas','temperature'), ['cell_volume'], weight_field=None, n_bins=128)
        profile = plot.profiles[0]
        temperature_dist = profile.x
        temperature_hist = profile['cell_volume']/shell.sum('cell_volume')
        plot = yt.ProfilePlot(shell, ('gas','density'), ['cell_volume'], weight_field=None, n_bins=128)
        profile = plot.profiles[0]
        density_dist = profile.x
        density_hist = profile['cell_volume']/shell.sum('cell_volume')
        plot = yt.ProfilePlot(shell, ('gas','radial_velocity_corrected'), ['cell_mass'], weight_field=None, n_bins=128)
        profile = plot.profiles[0]
        rv_dist = profile.x
        rv_hist = profile['cell_mass']/shell.sum('cell_mass')
        plot = yt.ProfilePlot(shell, ('gas','vtheta_disk'), ['cell_mass'], weight_field=None, n_bins=128)
        profile = plot.profiles[0]
        vtheta_dist = profile.x
        vtheta_hist = profile['cell_mass']/shell.sum('cell_mass')
        plot = yt.ProfilePlot(shell, ('gas','vphi_disk'), ['cell_mass'], weight_field=None, n_bins=128)
        profile = plot.profiles[0]
        vphi_dist = profile.x
        vphi_hist = profile['cell_mass']/shell.sum('cell_mass')
        plot = yt.ProfilePlot(shell, ('gas','pressure'), ['cell_volume'], weight_field=None, n_bins=128)
        profile = plot.profiles[0]
        pressure_dist = profile.x
        pressure_hist = profile['cell_volume']/shell.sum('cell_volume')

        plot = yt.ProfilePlot(shell_cold, ('gas','temperature'), ['cell_volume'], weight_field=None, n_bins=128)
        profile = plot.profiles[0]
        cold_temperature_dist = profile.x
        cold_temperature_hist = profile['cell_volume']/shell.sum('cell_volume')
        plot = yt.ProfilePlot(shell_cold, ('gas','density'), ['cell_volume'], weight_field=None, n_bins=128)
        profile = plot.profiles[0]
        cold_density_dist = profile.x
        cold_density_hist = profile['cell_volume']/shell.sum('cell_volume')
        plot = yt.ProfilePlot(shell_cold, ('gas','radial_velocity_corrected'), ['cell_mass'], weight_field=None, n_bins=128)
        profile = plot.profiles[0]
        cold_rv_dist = profile.x
        cold_rv_hist = profile['cell_mass']/shell.sum('cell_mass')
        plot = yt.ProfilePlot(shell_cold, ('gas','vtheta_disk'), ['cell_mass'], weight_field=None, n_bins=128)
        profile = plot.profiles[0]
        cold_vtheta_dist = profile.x
        cold_vtheta_hist = profile['cell_mass']/shell.sum('cell_mass')
        plot = yt.ProfilePlot(shell_cold, ('gas','vphi_disk'), ['cell_mass'], weight_field=None, n_bins=128)
        profile = plot.profiles[0]
        cold_vphi_dist = profile.x
        cold_vphi_hist = profile['cell_mass']/shell.sum('cell_mass')
        plot = yt.ProfilePlot(shell_cold, ('gas','pressure'), ['cell_volume'], weight_field=None, n_bins=128)
        profile = plot.profiles[0]
        cold_pressure_dist = profile.x
        cold_pressure_hist = profile['cell_volume']/shell.sum('cell_volume')

        plot = yt.ProfilePlot(shell_cool, ('gas','temperature'), ['cell_volume'], weight_field=None, n_bins=128)
        profile = plot.profiles[0]
        cool_temperature_dist = profile.x
        cool_temperature_hist = profile['cell_volume']/shell.sum('cell_volume')
        plot = yt.ProfilePlot(shell_cool, ('gas','density'), ['cell_volume'], weight_field=None, n_bins=128)
        profile = plot.profiles[0]
        cool_density_dist = profile.x
        cool_density_hist = profile['cell_volume']/shell.sum('cell_volume')
        plot = yt.ProfilePlot(shell_cool, ('gas','radial_velocity_corrected'), ['cell_mass'], weight_field=None, n_bins=128)
        profile = plot.profiles[0]
        cool_rv_dist = profile.x
        cool_rv_hist = profile['cell_mass']/shell.sum('cell_mass')
        plot = yt.ProfilePlot(shell_cool, ('gas','vtheta_disk'), ['cell_mass'], weight_field=None, n_bins=128)
        profile = plot.profiles[0]
        cool_vtheta_dist = profile.x
        cool_vtheta_hist = profile['cell_mass']/shell.sum('cell_mass')
        plot = yt.ProfilePlot(shell_cool, ('gas','vphi_disk'), ['cell_mass'], weight_field=None, n_bins=128)
        profile = plot.profiles[0]
        cool_vphi_dist = profile.x
        cool_vphi_hist = profile['cell_mass']/shell.sum('cell_mass')
        plot = yt.ProfilePlot(shell_cool, ('gas','pressure'), ['cell_volume'], weight_field=None, n_bins=128)
        profile = plot.profiles[0]
        cool_pressure_dist = profile.x
        cool_pressure_hist = profile['cell_volume']/shell.sum('cell_volume')

        plot = yt.ProfilePlot(shell_warm, ('gas','temperature'), ['cell_volume'], weight_field=None, n_bins=128)
        profile = plot.profiles[0]
        warm_temperature_dist = profile.x
        warm_temperature_hist = profile['cell_volume']/shell.sum('cell_volume')
        plot = yt.ProfilePlot(shell_warm, ('gas','density'), ['cell_volume'], weight_field=None, n_bins=128)
        profile = plot.profiles[0]
        warm_density_dist = profile.x
        warm_density_hist = profile['cell_volume']/shell.sum('cell_volume')
        plot = yt.ProfilePlot(shell_warm, ('gas','radial_velocity_corrected'), ['cell_mass'], weight_field=None, n_bins=128)
        profile = plot.profiles[0]
        warm_rv_dist = profile.x
        warm_rv_hist = profile['cell_mass']/shell.sum('cell_mass')
        plot = yt.ProfilePlot(shell_warm, ('gas','vtheta_disk'), ['cell_mass'], weight_field=None, n_bins=128)
        profile = plot.profiles[0]
        warm_vtheta_dist = profile.x
        warm_vtheta_hist = profile['cell_mass']/shell.sum('cell_mass')
        plot = yt.ProfilePlot(shell_warm, ('gas','vphi_disk'), ['cell_mass'], weight_field=None, n_bins=128)
        profile = plot.profiles[0]
        warm_vphi_dist = profile.x
        warm_vphi_hist = profile['cell_mass']/shell.sum('cell_mass')
        plot = yt.ProfilePlot(shell_warm, ('gas','pressure'), ['cell_volume'], weight_field=None, n_bins=128)
        profile = plot.profiles[0]
        warm_pressure_dist = profile.x
        warm_pressure_hist = profile['cell_volume']/shell.sum('cell_volume')

        plot = yt.ProfilePlot(shell_hot, ('gas','temperature'), ['cell_volume'], weight_field=None, n_bins=128)
        profile = plot.profiles[0]
        hot_temperature_dist = profile.x
        hot_temperature_hist = profile['cell_volume']/shell.sum('cell_volume')
        plot = yt.ProfilePlot(shell_hot, ('gas','density'), ['cell_volume'], weight_field=None, n_bins=128)
        profile = plot.profiles[0]
        hot_density_dist = profile.x
        hot_density_hist = profile['cell_volume']/shell.sum('cell_volume')
        plot = yt.ProfilePlot(shell_hot, ('gas','radial_velocity_corrected'), ['cell_mass'], weight_field=None, n_bins=128)
        profile = plot.profiles[0]
        hot_rv_dist = profile.x
        hot_rv_hist = profile['cell_mass']/shell.sum('cell_mass')
        plot = yt.ProfilePlot(shell_hot, ('gas','vtheta_disk'), ['cell_mass'], weight_field=None, n_bins=128)
        profile = plot.profiles[0]
        hot_vtheta_dist = profile.x
        hot_vtheta_hist = profile['cell_mass']/shell.sum('cell_mass')
        plot = yt.ProfilePlot(shell_hot, ('gas','vphi_disk'), ['cell_mass'], weight_field=None, n_bins=128)
        profile = plot.profiles[0]
        hot_vphi_dist = profile.x
        hot_vphi_hist = profile['cell_mass']/shell.sum('cell_mass')
        plot = yt.ProfilePlot(shell_hot, ('gas','pressure'), ['cell_volume'], weight_field=None, n_bins=128)
        profile = plot.profiles[0]
        hot_pressure_dist = profile.x
        hot_pressure_hist = profile['cell_volume']/shell.sum('cell_volume')

        # Add PDFs to table
        for b in range(128):
            data_pdf.add_row([zsnap, 0, r, temperature_dist[b], temperature_hist[b],
                            density_dist[b], density_hist[b], rv_dist[b], rv_hist[b],
                            vtheta_dist[b], vtheta_hist[b], vphi_dist[b], vphi_hist[b],
                            pressure_dist[b], pressure_hist[b],
                            cold_temperature_dist[b], cold_temperature_hist[b],
                            cold_density_dist[b], cold_density_hist[b], cold_rv_dist[b], cold_rv_hist[b],
                            cold_vtheta_dist[b], cold_vtheta_hist[b], cold_vphi_dist[b], cold_vphi_hist[b],
                            cold_pressure_dist[b], cold_pressure_hist[b],
                            cool_temperature_dist[b], cool_temperature_hist[b],
                            cool_density_dist[b], cool_density_hist[b], cool_rv_dist[b], cool_rv_hist[b],
                            cool_vtheta_dist[b], cool_vtheta_hist[b], cool_vphi_dist[b], cool_vphi_hist[b],
                            cool_pressure_dist[b], cool_pressure_hist[b],
                            warm_temperature_dist[b], warm_temperature_hist[b],
                            warm_density_dist[b], warm_density_hist[b], warm_rv_dist[b], warm_rv_hist[b],
                            warm_vtheta_dist[b], warm_vtheta_hist[b], warm_vphi_dist[b], warm_vphi_hist[b],
                            warm_pressure_dist[b], warm_pressure_hist[b],
                            hot_temperature_dist[b], hot_temperature_hist[b],
                            hot_density_dist[b], hot_density_hist[b], hot_rv_dist[b], hot_rv_hist[b],
                            hot_vtheta_dist[b], hot_vtheta_hist[b], hot_vphi_dist[b], hot_vphi_hist[b],
                            hot_pressure_dist[b], hot_pressure_hist[b]])


    # Save to file
    data_avg, data_pdf = set_table_units(data_avg, data_pdf)
    data_avg.write(tablename + '_avg.hdf5', path='all_data', serialize_meta=True, overwrite=True)
    data_pdf.write(tablename + '_pdf.hdf5', path='all_data', serialize_meta=True, overwrite=True)

    return "Averages and PDFs have been calculated for snapshot" + snap + "!"

def load_and_calculate(foggie_dir, run_dir, track, halo_c_v_name, snap, tablename):
    '''This function loads a specified snapshot 'snap' located in the 'run_dir' within the
    'foggie_dir', the halo track 'track', the name of the table to output, and a boolean
    'quadrants' that specifies whether or not to compute in quadrants vs. the whole domain, then
    does the calculation on the loaded snapshot.'''

    snap_name = foggie_dir + run_dir + snap + '/' + snap
    ds, refine_box, refine_box_center, refine_width = load(snap_name, track, use_halo_c_v=True, \
                                                      halo_c_v_name=halo_c_v_name, disk_relative=True)
    refine_width_kpc = YTArray([refine_width], 'kpc')
    zsnap = ds.get_parameter('CosmologyCurrentRedshift')

    # Do the actual calculation
    message = calc_shells(ds, snap, zsnap, refine_width_kpc, tablename)
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
        foggie_dir = '/Users/clochhaas/Documents/Research/FOGGIE/Simulation_Data/'
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
            load_and_calculate(foggie_dir, run_dir, trackname, halo_c_v_name, snap, tablename)
    else:
        # Split into a number of groupings equal to the number of processors
        # and run one process per processor
        for i in range(len(outs)//args.nproc):
            threads = []
            for j in range(args.nproc):
                snap = outs[args.nproc*i+j]
                tablename = prefix + snap + '_shells'
                threads.append(multi.Process(target=load_and_calculate, \
			       args=(foggie_dir, run_dir, trackname, halo_c_v_name, snap, tablename)))
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
			   args=(foggie_dir, run_dir, trackname, halo_c_v_name, snap, tablename)))
        for t in threads:
            t.start()
        for t in threads:
            t.join()

    print(str(datetime.datetime.now()))
    print("All snapshots finished!")
