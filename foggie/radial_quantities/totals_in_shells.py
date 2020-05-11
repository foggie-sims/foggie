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
import trident

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
    parser.set_defaults(output='RD0034')

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

    parser.add_argument('--remove_sats', dest='remove_sats', action='store_true',
                        help='Do you want to remove satellites when calculating fluxes? ' + \
                        "This requires a satellites.hdf5 file to exist for the halo/run you're using." + \
                        ' Default is no.')
    parser.set_defaults(remove_sats=False)

    parser.add_argument('--sat_radius', metavar='sat_radius', type=float, action='store', \
                        help='What radius (in kpc) do you want to excise around satellites? Default is 10.')
    parser.set_defaults(sat_radius=10.)

    parser.add_argument('--flux_type', metavar='flux_type', type=str, action='store', \
                        help='What fluxes do you want to compute? Currently, the options are "mass" (includes metal masses)' + \
                        ' "energy" "entropy" and "O_ion_mass". You can compute all of them by inputting "mass,energy,entropy,O_ion_mass" (no spaces!) ' + \
                        'and the default is to do all.')
    parser.set_defaults(flux_type="mass,energy,entropy,O_ion_mass")

    parser.add_argument('--surface', metavar='surface', type=str, action='store', \
                        help='What surface type for computing the totals? Default is sphere' + \
                        ' and the other options are "frustum" or "cylinder".\nNote that all surfaces will be centered on halo center.\n' + \
                        'To specify the shape, size, and orientation of the surface you want, ' + \
                        'input a list as follows (don\'t forget the outer quotes):\nIf you want a sphere, give:\n' + \
                        '"[\'sphere\', inner_radius, outer_radius, num_radii]"\n' + \
                        'where inner_radius is the inner boundary as a fraction of refine_width, outer_radius is the outer ' + \
                        'boundary as a fraction (or multiple) of refine_width,\nand num_radii is the number of radii where you want the flux to be ' + \
                        'calculated between inner_radius and outer_radius\n' + \
                        '(inner_radius and outer_radius are automatically included).\n' + \
                        'If you want a frustum, give:\n' + \
                        '"[\'frustum\', axis, inner_radius, outer_radius, num_radii, opening_angle]"\n' + \
                        'where axis specifies what axis to align the frustum with and can be one of the following:\n' + \
                        "'x'\n'y'\n'z'\n'minor' (aligns with disk minor axis)\n(x,y,z) (a tuple giving a 3D vector for an arbitrary axis).\n" + \
                        'For all axis definitions other than the arbitrary vector, if the axis string starts with a \'-\', it will compute a frustum pointing in the opposite direction.\n' + \
                        'inner_radius, outer_radius, and num_radii are the same as for the sphere\n' + \
                        'and opening_angle gives the angle in degrees of the opening angle of the cone, measured from axis.\n' + \
                        'If you want a cylinder, give:\n' + \
                        '"[\'cylinder\', axis, bottom_edge, top_edge, radius, step_direction, num_steps]"\n' + \
                        'where axis specifies what axis to align the length of the cylinder with and can be one of the following:\n' + \
                        "'x'\n'y'\n'z'\n'minor' (aligns with disk minor axis)\n(x,y,z) (a tuple giving a 3D vector for an arbitrary axis).\n" + \
                        'For all axis definitions other than the arbitrary vector, if the axis string starts with a \'-\', it will compute a cylinder pointing in the opposite direction.\n' + \
                        'bottom_edge, top_edge, and radius give the dimensions of the cylinder,\n' + \
                        'by default in units of refine_width (unless the --kpc option is specified), where bottom_ and top_edge are' + \
                        ' distance from halo center.\n' + \
                        "step_direction can be 'height', which will compute fluxes across circular planes in the cylinder parallel to the flat sides, or 'radius', which\n" + \
                        "will compute fluxes across different radii within the cylinder perpendicular to the cylinder's flat sides.\n" + \
                        "'num_steps' gives the number of places (either heights or radii) within the cylinder where to calculate fluxes.")
    parser.set_defaults(surface="['sphere', 0.05, 2., 200]")

    parser.add_argument('--kpc', dest='kpc', action='store_true',
                        help='Do you want to give inner_radius and outer_radius in the surface arguments ' + \
                        'in kpc rather than the default of fraction of refine_width? Default is no.\n' + \
                        'Note that if you want to track fluxes over time, using kpc instead of fractions ' + \
                        'of refine_width will lead to larger errors.')
    parser.set_defaults(kpc=False)

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
             'bottom_edge':'kpc', 'top_edge':'kpc', \
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
             'net_potential_energy':'erg', 'net_total_energy':'erg', 'net_entropy':'cm**2*keV', \
             'kinetic_energy_in':'erg', 'kinetic_energy_out':'erg', \
             'thermal_energy_in':'erg', 'thermal_energy_out':'erg', \
             'potential_energy_in':'erg', 'potential_energy_out':'erg', \
             'total_energy_in':'erg', 'total_energy_out':'erg', \
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
             'net_cold_total_energy':'erg', 'cold_total_energy_in':'erg', 'cold_total_energy_out':'erg', \
             'net_cool_total_energy':'erg', 'cool_total_energy_in':'erg', 'cool_total_energy_out':'erg', \
             'net_warm_total_energy':'erg', 'warm_total_energy_in':'erg', 'warm_total_energy_out':'erg', \
             'net_hot_total_energy':'erg', 'hot_total_energy_in':'erg', 'hot_total_energy_out':'erg', \
             'net_cold_entropy':'cm**2*keV', 'cold_entropy_in':'cm**2*keV', 'cold_entropy_out':'cm**2*keV', \
             'net_cool_entropy':'cm**2*keV', 'cool_entropy_in':'cm**2*keV', 'cool_entropy_out':'cm**2*keV', \
             'net_warm_entropy':'cm**2*keV', 'warm_entropy_in':'cm**2*keV', 'warm_entropy_out':'cm**2*keV', \
             'net_hot_entropy':'cm**2*keV', 'hot_entropy_in':'cm**2*keV', 'hot_entropy_out':'cm**2*keV', \
             'net_O_mass':'Msun', 'O_mass_in':'Msun', 'O_mass_out':'Msun', \
             'net_cold_O_mass':'Msun', 'cold_O_mass_in':'Msun', 'cold_O_mass_out':'Msun', \
             'net_cool_O_mass':'Msun', 'cool_O_mass_in':'Msun', 'cool_O_mass_out':'Msun', \
             'net_warm_O_mass':'Msun', 'warm_O_mass_in':'Msun', 'warm_O_mass_out':'Msun', \
             'net_hot_O_mass':'Msun', 'hot_O_mass_in':'Msun', 'hot_O_mass_out':'Msun', \
             'net_OI_mass':'Msun', 'OI_mass_in':'Msun', 'OI_mass_out':'Msun', \
             'net_cold_OI_mass':'Msun', 'cold_OI_mass_in':'Msun', 'cold_OI_mass_out':'Msun', \
             'net_cool_OI_mass':'Msun', 'cool_OI_mass_in':'Msun', 'cool_OI_mass_out':'Msun', \
             'net_warm_OI_mass':'Msun', 'warm_OI_mass_in':'Msun', 'warm_OI_mass_out':'Msun', \
             'net_hot_OI_mass':'Msun', 'hot_OI_mass_in':'Msun', 'hot_OI_mass_out':'Msun', \
             'net_OII_mass':'Msun', 'OII_mass_in':'Msun', 'OII_mass_out':'Msun', \
             'net_cold_OII_mass':'Msun', 'cold_OII_mass_in':'Msun', 'cold_OII_mass_out':'Msun', \
             'net_cool_OII_mass':'Msun', 'cool_OII_mass_in':'Msun', 'cool_OII_mass_out':'Msun', \
             'net_warm_OII_mass':'Msun', 'warm_OII_mass_in':'Msun', 'warm_OII_mass_out':'Msun', \
             'net_hot_OII_mass':'Msun', 'hot_OII_mass_in':'Msun', 'hot_OII_mass_out':'Msun', \
             'net_OIII_mass':'Msun', 'OIII_mass_in':'Msun', 'OIII_mass_out':'Msun', \
             'net_cold_OIII_mass':'Msun', 'cold_OIII_mass_in':'Msun', 'cold_OIII_mass_out':'Msun', \
             'net_cool_OIII_mass':'Msun', 'cool_OIII_mass_in':'Msun', 'cool_OIII_mass_out':'Msun', \
             'net_warm_OIII_mass':'Msun', 'warm_OIII_mass_in':'Msun', 'warm_OIII_mass_out':'Msun', \
             'net_hot_OIII_mass':'Msun', 'hot_OIII_mass_in':'Msun', 'hot_OIII_mass_out':'Msun', \
             'net_OIV_mass':'Msun', 'OIV_mass_in':'Msun', 'OIV_mass_out':'Msun', \
             'net_cold_OIV_mass':'Msun', 'cold_OIV_mass_in':'Msun', 'cold_OIV_mass_out':'Msun', \
             'net_cool_OIV_mass':'Msun', 'cool_OIV_mass_in':'Msun', 'cool_OIV_mass_out':'Msun', \
             'net_warm_OIV_mass':'Msun', 'warm_OIV_mass_in':'Msun', 'warm_OIV_mass_out':'Msun', \
             'net_hot_OIV_mass':'Msun', 'hot_OIV_mass_in':'Msun', 'hot_OIV_mass_out':'Msun', \
             'net_OV_mass':'Msun', 'OV_mass_in':'Msun', 'OV_mass_out':'Msun', \
             'net_cold_OV_mass':'Msun', 'cold_OV_mass_in':'Msun', 'cold_OV_mass_out':'Msun', \
             'net_cool_OV_mass':'Msun', 'cool_OV_mass_in':'Msun', 'cool_OV_mass_out':'Msun', \
             'net_warm_OV_mass':'Msun', 'warm_OV_mass_in':'Msun', 'warm_OV_mass_out':'Msun', \
             'net_hot_OV_mass':'Msun', 'hot_OV_mass_in':'Msun', 'hot_OV_mass_out':'Msun', \
             'net_OVI_mass':'Msun', 'OVI_mass_in':'Msun', 'OVI_mass_out':'Msun', \
             'net_cold_OVI_mass':'Msun', 'cold_OVI_mass_in':'Msun', 'cold_OVI_mass_out':'Msun', \
             'net_cool_OVI_mass':'Msun', 'cool_OVI_mass_in':'Msun', 'cool_OVI_mass_out':'Msun', \
             'net_warm_OVI_mass':'Msun', 'warm_OVI_mass_in':'Msun', 'warm_OVI_mass_out':'Msun', \
             'net_hot_OVI_mass':'Msun', 'hot_OVI_mass_in':'Msun', 'hot_OVI_mass_out':'Msun', \
             'net_OVII_mass':'Msun', 'OVII_mass_in':'Msun', 'OVII_mass_out':'Msun', \
             'net_cold_OVII_mass':'Msun', 'cold_OVII_mass_in':'Msun', 'cold_OVII_mass_out':'Msun', \
             'net_cool_OVII_mass':'Msun', 'cool_OVII_mass_in':'Msun', 'cool_OVII_mass_out':'Msun', \
             'net_warm_OVII_mass':'Msun', 'warm_OVII_mass_in':'Msun', 'warm_OVII_mass_out':'Msun', \
             'net_hot_OVII_mass':'Msun', 'hot_OVII_mass_in':'Msun', 'hot_OVII_mass_out':'Msun', \
             'net_OVIII_mass':'Msun', 'OVIII_mass_in':'Msun', 'OVIII_mass_out':'Msun', \
             'net_cold_OVIII_mass':'Msun', 'cold_OVIII_mass_in':'Msun', 'cold_OVIII_mass_out':'Msun', \
             'net_cool_OVIII_mass':'Msun', 'cool_OVIII_mass_in':'Msun', 'cool_OVIII_mass_out':'Msun', \
             'net_warm_OVIII_mass':'Msun', 'warm_OVIII_mass_in':'Msun', 'warm_OVIII_mass_out':'Msun', \
             'net_hot_OVIII_mass':'Msun', 'hot_OVIII_mass_in':'Msun', 'hot_OVIII_mass_out':'Msun', \
             'net_OIX_mass':'Msun', 'OIX_mass_in':'Msun', 'OIX_mass_out':'Msun', \
             'net_cold_OIX_mass':'Msun', 'cold_OIX_mass_in':'Msun', 'cold_OIX_mass_out':'Msun', \
             'net_cool_OIX_mass':'Msun', 'cool_OIX_mass_in':'Msun', 'cool_OIX_mass_out':'Msun', \
             'net_warm_OIX_mass':'Msun', 'warm_OIX_mass_in':'Msun', 'warm_OIX_mass_out':'Msun', \
             'net_hot_OIX_mass':'Msun', 'hot_OIX_mass_in':'Msun', 'hot_OIX_mass_out':'Msun'}
    for key in table.keys():
        table[key].unit = table_units[key]
    return table

def calc_totals_sphere(ds, snap, zsnap, refine_width_kpc, tablename, surface_args, flux_types, **kwargs):
    '''This function calculates the total of each gas property in spherical shells, with satellites removed,
    at a variety of radii. It uses the dataset stored in 'ds', which is from time snapshot 'snap', has redshift
    'zsnap', and has width of the refine box in kpc 'refine_width_kpc' and stores the totals in 'tablename'.
    'surface_args' gives the properties of the spheres.'''

    sat = kwargs.get('sat')
    sat_radius = kwargs.get('sat_radius', 0.)

    halo_center_kpc = ds.halo_center_kpc

    cmtopc = 3.086e18
    stoyr = 3.154e7
    gtoMsun = 1.989e33

    inner_radius = surface_args[1]
    outer_radius = surface_args[2]
    dr = (outer_radius - inner_radius)/surface_args[3]
    units_kpc = surface_args[4]

    # Set up table of everything we want
    # NOTE: Make sure table units are updated when things are added to this table!
    names_list = ('redshift', 'inner_radius', 'outer_radius')
    types_list = ('f8', 'f8', 'f8')
    if ('mass' in flux_types):
        new_names = ('net_mass', 'net_metals', \
        'mass_in', 'mass_out', 'metals_in', 'metals_out', \
        'net_cold_mass', 'cold_mass_in', 'cold_mass_out', \
        'net_cool_mass', 'cool_mass_in', 'cool_mass_out', \
        'net_warm_mass', 'warm_mass_in', 'warm_mass_out', \
        'net_hot_mass', 'hot_mass_in', 'hot_mass_out', \
        'net_cold_metals', 'cold_metals_in', 'cold_metals_out', \
        'net_cool_metals', 'cool_metals_in', 'cool_metals_out', \
        'net_warm_metals', 'warm_metals_in', 'warm_metals_out', \
        'net_hot_metals', 'hot_metals_in', 'hot_metals_out')
        new_types = ('f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', \
        'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', \
        'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8')
        names_list += new_names
        types_list += new_types
    if ('energy' in flux_types):
        new_names = ('net_kinetic_energy', 'net_thermal_energy', 'net_potential_energy', 'net_total_energy', \
        'kinetic_energy_in', 'kinetic_energy_out', \
        'thermal_energy_in', 'thermal_energy_out', \
        'potential_energy_in', 'potential_energy_out', \
        'total_energy_in', 'total_energy_out', \
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
        'net_cold_total_energy', 'cold_total_energy_in', 'cold_total_energy_out', \
        'net_cool_total_energy', 'cool_total_energy_in', 'cool_total_energy_out', \
        'net_warm_total_energy', 'warm_total_energy_in', 'warm_total_energy_out', \
        'net_hot_total_energy', 'hot_total_energy_in', 'hot_total_energy_out')
        new_types = ('f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', \
        'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', \
        'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', \
        'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', \
        'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', \
        'f8', 'f8', 'f8')
        names_list += new_names
        types_list += new_types
    if ('entropy' in flux_types):
        new_names = ('net_entropy', \
        'entropy_in', 'entropy_out', \
        'net_cold_entropy', 'cold_entropy_in', 'cold_entropy_out', \
        'net_cool_entropy', 'cool_entropy_in', 'cool_entropy_out', \
        'net_warm_entropy', 'warm_entropy_in', 'warm_entropy_out', \
        'net_hot_entropy', 'hot_entropy_in', 'hot_entropy_out')
        new_types = ('f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', \
        'f8', 'f8', 'f8', 'f8', 'f8', 'f8')
        names_list += new_names
        types_list += new_types
    if ('O_ion_mass' in flux_types):
        new_names = ('net_O_mass', 'O_mass_in', 'O_mass_out', \
        'net_cold_O_mass', 'cold_O_mass_in', 'cold_O_mass_out', \
        'net_cool_O_mass', 'cool_O_mass_in', 'cool_O_mass_out', \
        'net_warm_O_mass', 'warm_O_mass_in', 'warm_O_mass_out', \
        'net_hot_O_mass', 'hot_O_mass_in', 'hot_O_mass_out', \
        'net_OI_mass', 'OI_mass_in', 'OI_mass_out', \
        'net_cold_OI_mass', 'cold_OI_mass_in', 'cold_OI_mass_out', \
        'net_cool_OI_mass', 'cool_OI_mass_in', 'cool_OI_mass_out', \
        'net_warm_OI_mass', 'warm_OI_mass_in', 'warm_OI_mass_out', \
        'net_hot_OI_mass', 'hot_OI_mass_in', 'hot_OI_mass_out', \
        'net_OII_mass', 'OII_mass_in', 'OII_mass_out', \
        'net_cold_OII_mass', 'cold_OII_mass_in', 'cold_OII_mass_out', \
        'net_cool_OII_mass', 'cool_OII_mass_in', 'cool_OII_mass_out', \
        'net_warm_OII_mass', 'warm_OII_mass_in', 'warm_OII_mass_out', \
        'net_hot_OII_mass', 'hot_OII_mass_in', 'hot_OII_mass_out', \
        'net_OIII_mass', 'OIII_mass_in', 'OIII_mass_out', \
        'net_cold_OIII_mass', 'cold_OIII_mass_in', 'cold_OIII_mass_out', \
        'net_cool_OIII_mass', 'cool_OIII_mass_in', 'cool_OIII_mass_out', \
        'net_warm_OIII_mass', 'warm_OIII_mass_in', 'warm_OIII_mass_out', \
        'net_hot_OIII_mass', 'hot_OIII_mass_in', 'hot_OIII_mass_out', \
        'net_OIV_mass', 'OIV_mass_in', 'OIV_mass_out', \
        'net_cold_OIV_mass', 'cold_OIV_mass_in', 'cold_OIV_mass_out', \
        'net_cool_OIV_mass', 'cool_OIV_mass_in', 'cool_OIV_mass_out', \
        'net_warm_OIV_mass', 'warm_OIV_mass_in', 'warm_OIV_mass_out', \
        'net_hot_OIV_mass', 'hot_OIV_mass_in', 'hot_OIV_mass_out', \
        'net_OV_mass', 'OV_mass_in', 'OV_mass_out', \
        'net_cold_OV_mass', 'cold_OV_mass_in', 'cold_OV_mass_out', \
        'net_cool_OV_mass', 'cool_OV_mass_in', 'cool_OV_mass_out', \
        'net_warm_OV_mass', 'warm_OV_mass_in', 'warm_OV_mass_out', \
        'net_hot_OV_mass', 'hot_OV_mass_in', 'hot_OV_mass_out', \
        'net_OVI_mass', 'OVI_mass_in', 'OVI_mass_out', \
        'net_cold_OVI_mass', 'cold_OVI_mass_in', 'cold_OVI_mass_out', \
        'net_cool_OVI_mass', 'cool_OVI_mass_in', 'cool_OVI_mass_out', \
        'net_warm_OVI_mass', 'warm_OVI_mass_in', 'warm_OVI_mass_out', \
        'net_hot_OVI_mass', 'hot_OVI_mass_in', 'hot_OVI_mass_out', \
        'net_OVII_mass', 'OVII_mass_in', 'OVII_mass_out', \
        'net_cold_OVII_mass', 'cold_OVII_mass_in', 'cold_OVII_mass_out', \
        'net_cool_OVII_mass', 'cool_OVII_mass_in', 'cool_OVII_mass_out', \
        'net_warm_OVII_mass', 'warm_OVII_mass_in', 'warm_OVII_mass_out', \
        'net_hot_OVII_mass', 'hot_OVII_mass_in', 'hot_OVII_mass_out', \
        'net_OVIII_mass', 'OVIII_mass_in', 'OVIII_mass_out', \
        'net_cold_OVIII_mass', 'cold_OVIII_mass_in', 'cold_OVIII_mass_out', \
        'net_cool_OVIII_mass', 'cool_OVIII_mass_in', 'cool_OVIII_mass_out', \
        'net_warm_OVIII_mass', 'warm_OVIII_mass_in', 'warm_OVIII_mass_out', \
        'net_hot_OVIII_mass', 'hot_OVIII_mass_in', 'hot_OVIII_mass_out', \
        'net_OIX_mass', 'OIX_mass_in', 'OIX_mass_out', \
        'net_cold_OIX_mass', 'cold_OIX_mass_in', 'cold_OIX_mass_out', \
        'net_cool_OIX_mass', 'cool_OIX_mass_in', 'cool_OIX_mass_out', \
        'net_warm_OIX_mass', 'warm_OIX_mass_in', 'warm_OIX_mass_out', \
        'net_hot_OIX_mass', 'hot_OIX_mass_in', 'hot_OIX_mass_out')
        new_types = ('f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', \
        'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', \
        'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', \
        'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', \
        'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', \
        'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', \
        'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', \
        'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', \
        'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', \
        'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', \
        'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', \
        'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', \
        'f8', 'f8', 'f8','f8', 'f8', 'f8')
        names_list += new_names
        types_list += new_types
    totals = Table(names=names_list, dtype=types_list)

    # Define the radii of the spherical shells where we want to calculate totals
    if (units_kpc):
        radii = ds.arr(np.arange(inner_radius, outer_radius+dr, dr), 'kpc')
    else:
        radii = refine_width_kpc * np.arange(inner_radius, outer_radius+dr, dr)

    # Load arrays of all fields we need
    print('Loading field arrays')
    sphere = ds.sphere(halo_center_kpc, radii[-1])

    radius = sphere['gas','radius_corrected'].in_units('kpc').v
    x = sphere['gas','x'].in_units('kpc').v - halo_center_kpc[0].v
    y = sphere['gas','y'].in_units('kpc').v - halo_center_kpc[1].v
    z = sphere['gas','z'].in_units('kpc').v - halo_center_kpc[2].v
    rad_vel = sphere['gas','radial_velocity_corrected'].in_units('km/s').v
    temperature = sphere['gas','temperature'].in_units('K').v
    if ('mass' in flux_types):
        mass = sphere['gas','cell_mass'].in_units('Msun').v
        metal_mass = sphere['gas','metal_mass'].in_units('Msun').v
    if ('energy' in flux_types):
        kinetic_energy = sphere['gas','kinetic_energy_corrected'].in_units('erg').v
        thermal_energy = (sphere['cell_mass']*sphere['gas','thermal_energy']).in_units('erg').v
        potential_energy = (sphere['gas','cell_mass'] * \
          ds.arr(sphere['enzo','Grav_Potential'].v, 'code_length**2/code_time**2')).in_units('erg').v
        total_energy = kinetic_energy + thermal_energy + potential_energy
    if ('entropy' in flux_types):
        entropy = sphere['gas','entropy'].in_units('keV*cm**2').v
    if ('O_ion_mass' in flux_types):
        trident.add_ion_fields(ds, ions='all', ftype='gas')
        abundances = trident.ion_balance.solar_abundance
        OI_frac = sphere['O_p0_ion_fraction'].v
        OII_frac = sphere['O_p1_ion_fraction'].v
        OIII_frac = sphere['O_p2_ion_fraction'].v
        OIV_frac = sphere['O_p3_ion_fraction'].v
        OV_frac = sphere['O_p4_ion_fraction'].v
        OVI_frac = sphere['O_p5_ion_fraction'].v
        OVII_frac = sphere['O_p6_ion_fraction'].v
        OVIII_frac = sphere['O_p7_ion_fraction'].v
        OIX_frac = sphere['O_p8_ion_fraction'].v
        renorm = OI_frac + OII_frac + OIII_frac + OIV_frac + OV_frac + \
          OVI_frac + OVII_frac + OVIII_frac + OIX_frac
        O_frac = abundances['O']/(sum(abundances.values()) - abundances['H'] - abundances['He'])
        O_mass = sphere['metal_mass'].in_units('Msun').v*O_frac
        OI_mass = OI_frac/renorm*O_mass
        OII_mass = OII_frac/renorm*O_mass
        OIII_mass = OIII_frac/renorm*O_mass
        OIV_mass = OIV_frac/renorm*O_mass
        OV_mass = OV_frac/renorm*O_mass
        OVI_mass = OVI_frac/renorm*O_mass
        OVII_mass = OVII_frac/renorm*O_mass
        OVIII_mass = OVIII_frac/renorm*O_mass
        OIX_mass = OIX_frac/renorm*O_mass

    # Load list of satellite positions
    if (sat_radius!=0):
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
        sat_radius = 10.         # kpc
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
        if ('mass' in flux_types):
            mass_nosat = mass[bool_nosat]
            metal_mass_nosat = metal_mass[bool_nosat]
        if ('energy' in flux_types):
            kinetic_energy_nosat = kinetic_energy[bool_nosat]
            thermal_energy_nosat = thermal_energy[bool_nosat]
            potential_energy_nosat = potential_energy[bool_nosat]
            total_energy_nosat = total_energy[bool_nosat]
        if ('entropy' in flux_types):
            entropy_nosat = entropy[bool_nosat]
        if ('O_ion_mass' in flux_types):
            O_mass_nosat = O_mass[bool_nosat]
            OI_mass_nosat = OI_mass[bool_nosat]
            OII_mass_nosat = OII_mass[bool_nosat]
            OIII_mass_nosat = OIII_mass[bool_nosat]
            OIV_mass_nosat = OIV_mass[bool_nosat]
            OV_mass_nosat = OV_mass[bool_nosat]
            OVI_mass_nosat = OVI_mass[bool_nosat]
            OVII_mass_nosat = OVII_mass[bool_nosat]
            OVIII_mass_nosat = OVIII_mass[bool_nosat]
            OIX_mass_nosat = OIX_mass[bool_nosat]
    else:
        radius_nosat = radius
        rad_vel_nosat = rad_vel
        temperature_nosat = temperature
        if ('mass' in flux_types):
            mass_nosat = mass
            metal_mass_nosat = metal_mass
        if ('energy' in flux_types):
            kinetic_energy_nosat = kinetic_energy
            thermal_energy_nosat = thermal_energy
            potential_energy_nosat = potential_energy
            total_energy_nosat = total_energy
            cooling_time_nosat = cooling_time
        if ('entropy' in flux_types):
            entropy_nosat = entropy
        if ('O_ion_mass' in flux_types):
            O_mass_nosat = O_mass
            OI_mass_nosat = OI_mass
            OII_mass_nosat = OII_mass
            OIII_mass_nosat = OIII_mass
            OIV_mass_nosat = OIV_mass
            OV_mass_nosat = OV_mass
            OVI_mass_nosat = OVI_mass
            OVII_mass_nosat = OVII_mass
            OVIII_mass_nosat = OVIII_mass
            OIX_mass_nosat = OIX_mass

    # Cut satellite-removed data on temperature
    # These are lists of lists where the index goes from 0 to 4 for [all gas, cold, cool, warm, hot]
    if (sat_radius!=0):
        print('Cutting satellite-removed data on temperature')
    else:
        print('Cutting data on temperature')
    radius_nosat_Tcut = []
    rad_vel_nosat_Tcut = []
    if ('mass' in flux_types):
        mass_nosat_Tcut = []
        metal_mass_nosat_Tcut = []
    if ('energy' in flux_types):
        kinetic_energy_nosat_Tcut = []
        thermal_energy_nosat_Tcut = []
        potential_energy_nosat_Tcut = []
        total_energy_nosat_Tcut = []
    if ('entropy' in flux_types):
        entropy_nosat_Tcut = []
    if ('O_ion_mass' in flux_types):
        O_mass_nosat_Tcut = []
        OI_mass_nosat_Tcut = []
        OII_mass_nosat_Tcut = []
        OIII_mass_nosat_Tcut = []
        OIV_mass_nosat_Tcut = []
        OV_mass_nosat_Tcut = []
        OVI_mass_nosat_Tcut = []
        OVII_mass_nosat_Tcut = []
        OVIII_mass_nosat_Tcut = []
        OIX_mass_nosat_Tcut = []
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
        if ('mass' in flux_types):
            mass_nosat_Tcut.append(mass_nosat[bool_temp])
            metal_mass_nosat_Tcut.append(metal_mass_nosat[bool_temp])
        if ('energy' in flux_types):
            kinetic_energy_nosat_Tcut.append(kinetic_energy_nosat[bool_temp])
            thermal_energy_nosat_Tcut.append(thermal_energy_nosat[bool_temp])
            potential_energy_nosat_Tcut.append(potential_energy_nosat[bool_temp])
            total_energy_nosat_Tcut.append(total_energy_nosat[bool_temp])
        if ('entropy' in flux_types):
            entropy_nosat_Tcut.append(entropy_nosat[bool_temp])
        if ('O_ion_mass' in flux_types):
            O_mass_nosat_Tcut.append(O_mass_nosat[bool_temp])
            OI_mass_nosat_Tcut.append(OI_mass_nosat[bool_temp])
            OII_mass_nosat_Tcut.append(OII_mass_nosat[bool_temp])
            OIII_mass_nosat_Tcut.append(OIII_mass_nosat[bool_temp])
            OIV_mass_nosat_Tcut.append(OIV_mass_nosat[bool_temp])
            OV_mass_nosat_Tcut.append(OV_mass_nosat[bool_temp])
            OVI_mass_nosat_Tcut.append(OVI_mass_nosat[bool_temp])
            OVII_mass_nosat_Tcut.append(OVII_mass_nosat[bool_temp])
            OVIII_mass_nosat_Tcut.append(OVIII_mass_nosat[bool_temp])
            OIX_mass_nosat_Tcut.append(OIX_mass_nosat[bool_temp])

    # Loop over radii
    for i in range(len(radii)-1):
        inner_r = radii[i].v
        outer_r = radii[i+1].v

        if (i%10==0): print("Computing radius " + str(i) + "/" + str(len(radii)) + \
                            " for snapshot " + snap)

        # Compute net, in, and out totals with satellites removed
        # These are nested lists where the first index goes from 0 to 2 for [net, in, out]
        # and the second index goes from 0 to 4 for [all, cold, cool, warm, hot]
        if ('mass' in flux_types):
            mass_total_nosat = []
            metals_total_nosat = []
        if ('energy' in flux_types):
            kinetic_energy_total_nosat = []
            thermal_energy_total_nosat = []
            potential_energy_total_nosat = []
            total_energy_total_nosat = []
        if ('entropy' in flux_types):
            entropy_total_nosat = []
        if ('O_ion_mass' in flux_types):
            O_total_nosat = []
            OI_total_nosat = []
            OII_total_nosat = []
            OIII_total_nosat = []
            OIV_total_nosat = []
            OV_total_nosat = []
            OVI_total_nosat = []
            OVII_total_nosat = []
            OVIII_total_nosat = []
            OIX_total_nosat = []
        for j in range(3):
            if ('mass' in flux_types):
                mass_total_nosat.append([])
                metals_total_nosat.append([])
            if ('energy' in flux_types):
                kinetic_energy_total_nosat.append([])
                thermal_energy_total_nosat.append([])
                potential_energy_total_nosat.append([])
                total_energy_total_nosat.append([])
            if ('entropy' in flux_types):
                entropy_total_nosat.append([])
            if ('O_ion_mass' in flux_types):
                O_total_nosat.append([])
                OI_total_nosat.append([])
                OII_total_nosat.append([])
                OIII_total_nosat.append([])
                OIV_total_nosat.append([])
                OV_total_nosat.append([])
                OVI_total_nosat.append([])
                OVII_total_nosat.append([])
                OVIII_total_nosat.append([])
                OIX_total_nosat.append([])
            for k in range(5):
                bool_in = (radius_nosat_Tcut[k] > inner_r) & (radius_nosat_Tcut[k] < outer_r) & \
                  (rad_vel_nosat_Tcut[k] < 0.)
                bool_out = (radius_nosat_Tcut[k] > inner_r) & (radius_nosat_Tcut[k] < outer_r) & \
                  (rad_vel_nosat_Tcut[k] > 0.)
                if (j==0):
                    if ('mass' in flux_types):
                        mass_total_nosat[j].append((np.sum(mass_nosat_Tcut[k][bool_out]) + \
                          np.sum(mass_nosat_Tcut[k][bool_in])))
                        metals_total_nosat[j].append((np.sum(metal_mass_nosat_Tcut[k][bool_out]) + \
                          np.sum(metal_mass_nosat_Tcut[k][bool_in])))
                    if ('energy' in flux_types):
                        kinetic_energy_total_nosat[j].append((np.sum(kinetic_energy_nosat_Tcut[k][bool_out]) + \
                          np.sum(kinetic_energy_nosat_Tcut[k][bool_in])))
                        thermal_energy_total_nosat[j].append((np.sum(thermal_energy_nosat_Tcut[k][bool_out]) + \
                          np.sum(thermal_energy_nosat_Tcut[k][bool_in])))
                        potential_energy_total_nosat[j].append((np.sum(potential_energy_nosat_Tcut[k][bool_out]) + \
                          np.sum(potential_energy_nosat_Tcut[k][bool_in])))
                        total_energy_total_nosat[j].append((np.sum(total_energy_nosat_Tcut[k][bool_out]) + \
                          np.sum(total_energy_nosat_Tcut[k][bool_in])))
                    if ('entropy' in flux_types):
                        entropy_total_nosat[j].append((np.sum(entropy_nosat_Tcut[k][bool_out]) + \
                          np.sum(entropy_nosat_Tcut[k][bool_in])))
                    if ('O_ion_mass' in flux_types):
                        O_total_nosat[j].append((np.sum(O_mass_nosat_Tcut[k][bool_out]) + \
                          np.sum(O_mass_nosat_Tcut[k][bool_in])))
                        OI_total_nosat[j].append((np.sum(OI_mass_nosat_Tcut[k][bool_out]) + \
                          np.sum(OI_mass_nosat_Tcut[k][bool_in])))
                        OII_total_nosat[j].append((np.sum(OII_mass_nosat_Tcut[k][bool_out]) + \
                          np.sum(OII_mass_nosat_Tcut[k][bool_in])))
                        OIII_total_nosat[j].append((np.sum(OIII_mass_nosat_Tcut[k][bool_out]) + \
                          np.sum(OIII_mass_nosat_Tcut[k][bool_in])))
                        OIV_total_nosat[j].append((np.sum(OIV_mass_nosat_Tcut[k][bool_out]) + \
                          np.sum(OIV_mass_nosat_Tcut[k][bool_in])))
                        OV_total_nosat[j].append((np.sum(OV_mass_nosat_Tcut[k][bool_out]) + \
                          np.sum(OV_mass_nosat_Tcut[k][bool_in])))
                        OVI_total_nosat[j].append((np.sum(OVI_mass_nosat_Tcut[k][bool_out]) + \
                          np.sum(OVI_mass_nosat_Tcut[k][bool_in])))
                        OVII_total_nosat[j].append((np.sum(OVII_mass_nosat_Tcut[k][bool_out]) + \
                          np.sum(OVII_mass_nosat_Tcut[k][bool_in])))
                        OVIII_total_nosat[j].append((np.sum(OVIII_mass_nosat_Tcut[k][bool_out]) + \
                          np.sum(OVIII_mass_nosat_Tcut[k][bool_in])))
                        OIX_total_nosat[j].append((np.sum(OIX_mass_nosat_Tcut[k][bool_out]) + \
                          np.sum(OIX_mass_nosat_Tcut[k][bool_in])))
                if (j==1):
                    if ('mass' in flux_types):
                        mass_total_nosat[j].append(np.sum(mass_nosat_Tcut[k][bool_in]))
                        metals_total_nosat[j].append(np.sum(metal_mass_nosat_Tcut[k][bool_in]))
                    if ('energy' in flux_types):
                        kinetic_energy_total_nosat[j].append(np.sum(kinetic_energy_nosat_Tcut[k][bool_in]))
                        thermal_energy_total_nosat[j].append(np.sum(thermal_energy_nosat_Tcut[k][bool_in]))
                        potential_energy_total_nosat[j].append(np.sum(potential_energy_nosat_Tcut[k][bool_in]))
                        total_energy_total_nosat[j].append(np.sum(total_energy_nosat_Tcut[k][bool_in]))
                    if ('entropy' in flux_types):
                        entropy_total_nosat[j].append(np.sum(entropy_nosat_Tcut[k][bool_in]))
                    if ('O_ion_mass' in flux_types):
                        O_total_nosat[j].append(np.sum(O_mass_nosat_Tcut[k][bool_in]))
                        OI_total_nosat[j].append(np.sum(OI_mass_nosat_Tcut[k][bool_in]))
                        OII_total_nosat[j].append(np.sum(OII_mass_nosat_Tcut[k][bool_in]))
                        OIII_total_nosat[j].append(np.sum(OIII_mass_nosat_Tcut[k][bool_in]))
                        OIV_total_nosat[j].append(np.sum(OIV_mass_nosat_Tcut[k][bool_in]))
                        OV_total_nosat[j].append(np.sum(OV_mass_nosat_Tcut[k][bool_in]))
                        OVI_total_nosat[j].append(np.sum(OVI_mass_nosat_Tcut[k][bool_in]))
                        OVII_total_nosat[j].append(np.sum(OVII_mass_nosat_Tcut[k][bool_in]))
                        OVIII_total_nosat[j].append(np.sum(OVIII_mass_nosat_Tcut[k][bool_in]))
                        OIX_total_nosat[j].append(np.sum(OIX_mass_nosat_Tcut[k][bool_in]))
                if (j==2):
                    if ('mass' in flux_types):
                        mass_total_nosat[j].append(np.sum(mass_nosat_Tcut[k][bool_out]))
                        metals_total_nosat[j].append(np.sum(metal_mass_nosat_Tcut[k][bool_out]))
                    if ('energy' in flux_types):
                        kinetic_energy_total_nosat[j].append(np.sum(kinetic_energy_nosat_Tcut[k][bool_out]))
                        thermal_energy_total_nosat[j].append(np.sum(thermal_energy_nosat_Tcut[k][bool_out]))
                        potential_energy_total_nosat[j].append(np.sum(potential_energy_nosat_Tcut[k][bool_out]))
                        total_energy_total_nosat[j].append(np.sum(total_energy_nosat_Tcut[k][bool_out]))
                    if ('entropy' in flux_types):
                        entropy_total_nosat[j].append(np.sum(entropy_nosat_Tcut[k][bool_out]))
                    if ('O_ion_mass' in flux_types):
                        O_total_nosat[j].append(np.sum(O_mass_nosat_Tcut[k][bool_out]))
                        OI_total_nosat[j].append(np.sum(OI_mass_nosat_Tcut[k][bool_out]))
                        OII_total_nosat[j].append(np.sum(OII_mass_nosat_Tcut[k][bool_out]))
                        OIII_total_nosat[j].append(np.sum(OIII_mass_nosat_Tcut[k][bool_out]))
                        OIV_total_nosat[j].append(np.sum(OIV_mass_nosat_Tcut[k][bool_out]))
                        OV_total_nosat[j].append(np.sum(OV_mass_nosat_Tcut[k][bool_out]))
                        OVI_total_nosat[j].append(np.sum(OVI_mass_nosat_Tcut[k][bool_out]))
                        OVII_total_nosat[j].append(np.sum(OVII_mass_nosat_Tcut[k][bool_out]))
                        OVIII_total_nosat[j].append(np.sum(OVIII_mass_nosat_Tcut[k][bool_out]))
                        OIX_total_nosat[j].append(np.sum(OIX_mass_nosat_Tcut[k][bool_out]))

        # Add everything to the table
        new_row = [zsnap, inner_r, outer_r]
        if ('mass' in flux_types):
            new_row += [mass_total_nosat[0][0], metals_total_nosat[0][0], \
            mass_total_nosat[1][0], mass_total_nosat[2][0], metals_total_nosat[1][0], metals_total_nosat[2][0], \
            mass_total_nosat[0][1], mass_total_nosat[1][1], mass_total_nosat[2][1], \
            mass_total_nosat[0][2], mass_total_nosat[1][2], mass_total_nosat[2][2], \
            mass_total_nosat[0][3], mass_total_nosat[1][3], mass_total_nosat[2][3], \
            mass_total_nosat[0][4], mass_total_nosat[1][4], mass_total_nosat[2][4], \
            metals_total_nosat[0][1], metals_total_nosat[1][1], metals_total_nosat[2][1], \
            metals_total_nosat[0][2], metals_total_nosat[1][2], metals_total_nosat[2][2], \
            metals_total_nosat[0][3], metals_total_nosat[1][3], metals_total_nosat[2][3], \
            metals_total_nosat[0][4], metals_total_nosat[1][4], metals_total_nosat[2][4]]
        if ('energy' in flux_types):
            new_row += [kinetic_energy_total_nosat[0][0], thermal_energy_total_nosat[0][0], \
            potential_energy_total_nosat[0][0], total_energy_total_nosat[0][0], \
            kinetic_energy_total_nosat[1][0], kinetic_energy_total_nosat[2][0], \
            thermal_energy_total_nosat[1][0], thermal_energy_total_nosat[2][0], \
            potential_energy_total_nosat[1][0], potential_energy_total_nosat[2][0], \
            total_energy_total_nosat[1][0], total_energy_total_nosat[2][0], \
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
            total_energy_total_nosat[0][1], total_energy_total_nosat[1][1], total_energy_total_nosat[2][1], \
            total_energy_total_nosat[0][2], total_energy_total_nosat[1][2], total_energy_total_nosat[2][2], \
            total_energy_total_nosat[0][3], total_energy_total_nosat[1][3], total_energy_total_nosat[2][3], \
            total_energy_total_nosat[0][4], total_energy_total_nosat[1][4], total_energy_total_nosat[2][4]]
        if ('entropy' in flux_types):
            new_row += [entropy_total_nosat[0][0], \
            entropy_total_nosat[1][0], entropy_total_nosat[2][0], \
            entropy_total_nosat[0][1], entropy_total_nosat[1][1], entropy_total_nosat[2][1], \
            entropy_total_nosat[0][2], entropy_total_nosat[1][2], entropy_total_nosat[2][2], \
            entropy_total_nosat[0][3], entropy_total_nosat[1][3], entropy_total_nosat[2][3], \
            entropy_total_nosat[0][4], entropy_total_nosat[1][4], entropy_total_nosat[2][4]]
        if ('O_ion_mass' in flux_types):
            new_row += [
            O_total_nosat[0][0], O_total_nosat[1][0], O_total_nosat[2][0], \
            O_total_nosat[0][1], O_total_nosat[1][1], O_total_nosat[2][1], \
            O_total_nosat[0][2], O_total_nosat[1][2], O_total_nosat[2][2], \
            O_total_nosat[0][3], O_total_nosat[1][3], O_total_nosat[2][3], \
            O_total_nosat[0][4], O_total_nosat[1][4], O_total_nosat[2][4], \
            OI_total_nosat[0][0], OI_total_nosat[1][0], OI_total_nosat[2][0], \
            OI_total_nosat[0][1], OI_total_nosat[1][1], OI_total_nosat[2][1], \
            OI_total_nosat[0][2], OI_total_nosat[1][2], OI_total_nosat[2][2], \
            OI_total_nosat[0][3], OI_total_nosat[1][3], OI_total_nosat[2][3], \
            OI_total_nosat[0][4], OI_total_nosat[1][4], OI_total_nosat[2][4], \
            OII_total_nosat[0][0], OII_total_nosat[1][0], OII_total_nosat[2][0], \
            OII_total_nosat[0][1], OII_total_nosat[1][1], OII_total_nosat[2][1], \
            OII_total_nosat[0][2], OII_total_nosat[1][2], OII_total_nosat[2][2], \
            OII_total_nosat[0][3], OII_total_nosat[1][3], OII_total_nosat[2][3], \
            OII_total_nosat[0][4], OII_total_nosat[1][4], OII_total_nosat[2][4], \
            OIII_total_nosat[0][0], OIII_total_nosat[1][0], OIII_total_nosat[2][0], \
            OIII_total_nosat[0][1], OIII_total_nosat[1][1], OIII_total_nosat[2][1], \
            OIII_total_nosat[0][2], OIII_total_nosat[1][2], OIII_total_nosat[2][2], \
            OIII_total_nosat[0][3], OIII_total_nosat[1][3], OIII_total_nosat[2][3], \
            OIII_total_nosat[0][4], OIII_total_nosat[1][4], OIII_total_nosat[2][4], \
            OIV_total_nosat[0][0], OIV_total_nosat[1][0], OIV_total_nosat[2][0], \
            OIV_total_nosat[0][1], OIV_total_nosat[1][1], OIV_total_nosat[2][1], \
            OIV_total_nosat[0][2], OIV_total_nosat[1][2], OIV_total_nosat[2][2], \
            OIV_total_nosat[0][3], OIV_total_nosat[1][3], OIV_total_nosat[2][3], \
            OIV_total_nosat[0][4], OIV_total_nosat[1][4], OIV_total_nosat[2][4], \
            OV_total_nosat[0][0], OV_total_nosat[1][0], OV_total_nosat[2][0], \
            OV_total_nosat[0][1], OV_total_nosat[1][1], OV_total_nosat[2][1], \
            OV_total_nosat[0][2], OV_total_nosat[1][2], OV_total_nosat[2][2], \
            OV_total_nosat[0][3], OV_total_nosat[1][3], OV_total_nosat[2][3], \
            OV_total_nosat[0][4], OV_total_nosat[1][4], OV_total_nosat[2][4], \
            OVI_total_nosat[0][0], OVI_total_nosat[1][0], OVI_total_nosat[2][0], \
            OVI_total_nosat[0][1], OVI_total_nosat[1][1], OVI_total_nosat[2][1], \
            OVI_total_nosat[0][2], OVI_total_nosat[1][2], OVI_total_nosat[2][2], \
            OVI_total_nosat[0][3], OVI_total_nosat[1][3], OVI_total_nosat[2][3], \
            OVI_total_nosat[0][4], OVI_total_nosat[1][4], OVI_total_nosat[2][4], \
            OVII_total_nosat[0][0], OVII_total_nosat[1][0], OVII_total_nosat[2][0], \
            OVII_total_nosat[0][1], OVII_total_nosat[1][1], OVII_total_nosat[2][1], \
            OVII_total_nosat[0][2], OVII_total_nosat[1][2], OVII_total_nosat[2][2], \
            OVII_total_nosat[0][3], OVII_total_nosat[1][3], OVII_total_nosat[2][3], \
            OVII_total_nosat[0][4], OVII_total_nosat[1][4], OVII_total_nosat[2][4], \
            OVIII_total_nosat[0][0], OVIII_total_nosat[1][0], OVIII_total_nosat[2][0], \
            OVIII_total_nosat[0][1], OVIII_total_nosat[1][1], OVIII_total_nosat[2][1], \
            OVIII_total_nosat[0][2], OVIII_total_nosat[1][2], OVIII_total_nosat[2][2], \
            OVIII_total_nosat[0][3], OVIII_total_nosat[1][3], OVIII_total_nosat[2][3], \
            OVIII_total_nosat[0][4], OVIII_total_nosat[1][4], OVIII_total_nosat[2][4], \
            OIX_total_nosat[0][0], OIX_total_nosat[1][0], OIX_total_nosat[2][0], \
            OIX_total_nosat[0][1], OIX_total_nosat[1][1], OIX_total_nosat[2][1], \
            OIX_total_nosat[0][2], OIX_total_nosat[1][2], OIX_total_nosat[2][2], \
            OIX_total_nosat[0][3], OIX_total_nosat[1][3], OIX_total_nosat[2][3], \
            OIX_total_nosat[0][4], OIX_total_nosat[1][4], OIX_total_nosat[2][4]]
        totals.add_row(new_row)

    totals = set_table_units(totals)

    fluxtype_filename = ''
    if ('mass' in flux_types):
        fluxtype_filename += '_mass'
    if ('energy' in flux_types):
        fluxtype_filename += '_energy'
    if ('entropy' in flux_types):
        fluxtype_filename += '_entropy'
    if ('O_ion_mass' in flux_types):
        fluxtype_filename += '_Oions'

    # Save to file
    if (sat_radius!=0.):
        totals.write(tablename + '_nosat_sphere' + fluxtype_filename + '.hdf5', path='all_data', serialize_meta=True, overwrite=True)
    else:
        totals.write(tablename + '_sphere' + fluxtype_filename + '.hdf5', path='all_data', serialize_meta=True, overwrite=True)

    return "Totals have been calculated for snapshot " + snap + "!"

def calc_totals_frustum(ds, snap, zsnap, refine_width_kpc, tablename, surface_args, flux_types, **kwargs):
    '''This function calculates the totals of gas properties between radial surfaces within a frustum,
    with satellites removed, at a variety of radii. It
    uses the dataset stored in 'ds', which is from time snapshot 'snap', has redshift
    'zsnap', and has width of the refine box in kpc 'refine_width_kpc', and stores the totals in
    'tablename'. 'surface_args' gives the properties of the frustum.'''

    sat = kwargs.get('sat')
    sat_radius = kwargs.get('sat_radius', 0.)

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
    units_kpc = surface_args[7]

    # Set up table of everything we want
    # NOTE: Make sure table units are updated when things are added to this table!
    names_list = ('redshift', 'inner_radius', 'outer_radius')
    types_list = ('f8', 'f8', 'f8')
    if ('mass' in flux_types):
        new_names = ('net_mass', 'net_metals', \
        'mass_in', 'mass_out', 'metals_in', 'metals_out', \
        'net_cold_mass', 'cold_mass_in', 'cold_mass_out', \
        'net_cool_mass', 'cool_mass_in', 'cool_mass_out', \
        'net_warm_mass', 'warm_mass_in', 'warm_mass_out', \
        'net_hot_mass', 'hot_mass_in', 'hot_mass_out', \
        'net_cold_metals', 'cold_metals_in', 'cold_metals_out', \
        'net_cool_metals', 'cool_metals_in', 'cool_metals_out', \
        'net_warm_metals', 'warm_metals_in', 'warm_metals_out', \
        'net_hot_metals', 'hot_metals_in', 'hot_metals_out')
        new_types = ('f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', \
        'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', \
        'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8')
        names_list += new_names
        types_list += new_types
    if ('energy' in flux_types):
        new_names = ('net_kinetic_energy', 'net_thermal_energy', 'net_potential_energy', 'net_total_energy', \
        'kinetic_energy_in', 'kinetic_energy_out', \
        'thermal_energy_in', 'thermal_energy_out', \
        'potential_energy_in', 'potential_energy_out', \
        'total_energy_in', 'total_energy_out', \
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
        'net_cold_total_energy', 'cold_total_energy_in', 'cold_total_energy_out', \
        'net_cool_total_energy', 'cool_total_energy_in', 'cool_total_energy_out', \
        'net_warm_total_energy', 'warm_total_energy_in', 'warm_total_energy_out', \
        'net_hot_total_energy', 'hot_total_energy_in', 'hot_total_energy_out')
        new_types = ('f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', \
        'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', \
        'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', \
        'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', \
        'f8', 'f8', 'f8', 'f8', 'f8', 'f8')
        names_list += new_names
        types_list += new_types
    if ('entropy' in flux_types):
        new_names = ('net_entropy', 'entropy_in', 'entropy_out', \
        'net_cold_entropy', 'cold_entropy_in', 'cold_entropy_out', \
        'net_cool_entropy', 'cool_entropy_in', 'cool_entropy_out', \
        'net_warm_entropy', 'warm_entropy_in', 'warm_entropy_out', \
        'net_hot_entropy', 'hot_entropy_in', 'hot_entropy_out')
        new_types = ('f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', \
        'f8', 'f8', 'f8', 'f8', 'f8', 'f8')
        names_list += new_names
        types_list += new_types
    if ('O_ion_mass' in flux_types):
        new_names = ('net_O_mass', 'O_mass_in', 'O_mass_out', \
        'net_cold_O_mass', 'cold_O_mass_in', 'cold_O_mass_out', \
        'net_cool_O_mass', 'cool_O_mass_in', 'cool_O_mass_out', \
        'net_warm_O_mass', 'warm_O_mass_in', 'warm_O_mass_out', \
        'net_hot_O_mass', 'hot_O_mass_in', 'hot_O_mass_out', \
        'net_OI_mass', 'OI_mass_in', 'OI_mass_out', \
        'net_cold_OI_mass', 'cold_OI_mass_in', 'cold_OI_mass_out', \
        'net_cool_OI_mass', 'cool_OI_mass_in', 'cool_OI_mass_out', \
        'net_warm_OI_mass', 'warm_OI_mass_in', 'warm_OI_mass_out', \
        'net_hot_OI_mass', 'hot_OI_mass_in', 'hot_OI_mass_out', \
        'net_OII_mass', 'OII_mass_in', 'OII_mass_out', \
        'net_cold_OII_mass', 'cold_OII_mass_in', 'cold_OII_mass_out', \
        'net_cool_OII_mass', 'cool_OII_mass_in', 'cool_OII_mass_out', \
        'net_warm_OII_mass', 'warm_OII_mass_in', 'warm_OII_mass_out', \
        'net_hot_OII_mass', 'hot_OII_mass_in', 'hot_OII_mass_out', \
        'net_OIII_mass', 'OIII_mass_in', 'OIII_mass_out', \
        'net_cold_OIII_mass', 'cold_OIII_mass_in', 'cold_OIII_mass_out', \
        'net_cool_OIII_mass', 'cool_OIII_mass_in', 'cool_OIII_mass_out', \
        'net_warm_OIII_mass', 'warm_OIII_mass_in', 'warm_OIII_mass_out', \
        'net_hot_OIII_mass', 'hot_OIII_mass_in', 'hot_OIII_mass_out', \
        'net_OIV_mass', 'OIV_mass_in', 'OIV_mass_out', \
        'net_cold_OIV_mass', 'cold_OIV_mass_in', 'cold_OIV_mass_out', \
        'net_cool_OIV_mass', 'cool_OIV_mass_in', 'cool_OIV_mass_out', \
        'net_warm_OIV_mass', 'warm_OIV_mass_in', 'warm_OIV_mass_out', \
        'net_hot_OIV_mass', 'hot_OIV_mass_in', 'hot_OIV_mass_out', \
        'net_OV_mass', 'OV_mass_in', 'OV_mass_out', \
        'net_cold_OV_mass', 'cold_OV_mass_in', 'cold_OV_mass_out', \
        'net_cool_OV_mass', 'cool_OV_mass_in', 'cool_OV_mass_out', \
        'net_warm_OV_mass', 'warm_OV_mass_in', 'warm_OV_mass_out', \
        'net_hot_OV_mass', 'hot_OV_mass_in', 'hot_OV_mass_out', \
        'net_OVI_mass', 'OVI_mass_in', 'OVI_mass_out', \
        'net_cold_OVI_mass', 'cold_OVI_mass_in', 'cold_OVI_mass_out', \
        'net_cool_OVI_mass', 'cool_OVI_mass_in', 'cool_OVI_mass_out', \
        'net_warm_OVI_mass', 'warm_OVI_mass_in', 'warm_OVI_mass_out', \
        'net_hot_OVI_mass', 'hot_OVI_mass_in', 'hot_OVI_mass_out', \
        'net_OVII_mass', 'OVII_mass_in', 'OVII_mass_out', \
        'net_cold_OVII_mass', 'cold_OVII_mass_in', 'cold_OVII_mass_out', \
        'net_cool_OVII_mass', 'cool_OVII_mass_in', 'cool_OVII_mass_out', \
        'net_warm_OVII_mass', 'warm_OVII_mass_in', 'warm_OVII_mass_out', \
        'net_hot_OVII_mass', 'hot_OVII_mass_in', 'hot_OVII_mass_out', \
        'net_OVIII_mass', 'OVIII_mass_in', 'OVIII_mass_out', \
        'net_cold_OVIII_mass', 'cold_OVIII_mass_in', 'cold_OVIII_mass_out', \
        'net_cool_OVIII_mass', 'cool_OVIII_mass_in', 'cool_OVIII_mass_out', \
        'net_warm_OVIII_mass', 'warm_OVIII_mass_in', 'warm_OVIII_mass_out', \
        'net_hot_OVIII_mass', 'hot_OVIII_mass_in', 'hot_OVIII_mass_out', \
        'net_OIX_mass', 'OIX_mass_in', 'OIX_mass_out', \
        'net_cold_OIX_mass', 'cold_OIX_mass_in', 'cold_OIX_mass_out', \
        'net_cool_OIX_mass', 'cool_OIX_mass_in', 'cool_OIX_mass_out', \
        'net_warm_OIX_mass', 'warm_OIX_mass_in', 'warm_OIX_mass_out', \
        'net_hot_OIX_mass', 'hot_OIX_mass_in', 'hot_OIX_mass_out')
        new_types = ('f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', \
        'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', \
        'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', \
        'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', \
        'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', \
        'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', \
        'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', \
        'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', \
        'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', \
        'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', \
        'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', \
        'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', \
        'f8', 'f8', 'f8','f8', 'f8', 'f8')
        names_list += new_names
        types_list += new_types
    totals = Table(names=names_list, dtype=types_list)

    # Define the radii of the surfaces where we want to calculate fluxes
    if (units_kpc):
        radii = ds.arr(np.arange(inner_radius, outer_radius+dr, dr), 'kpc')
    else:
        radii = refine_width_kpc * np.arange(inner_radius, outer_radius+dr, dr)

    # Load arrays of all fields we need
    print('Loading field arrays')
    sphere = ds.sphere(halo_center_kpc, radii[-1])

    radius = sphere['gas','radius_corrected'].in_units('kpc').v
    x = sphere['gas','x'].in_units('kpc').v - halo_center_kpc[0].v
    y = sphere['gas','y'].in_units('kpc').v - halo_center_kpc[1].v
    z = sphere['gas','z'].in_units('kpc').v - halo_center_kpc[2].v
    rad_vel = sphere['gas','radial_velocity_corrected'].in_units('km/s').v
    temperature = sphere['gas','temperature'].in_units('K').v
    if ('mass' in flux_types):
        mass = sphere['gas','cell_mass'].in_units('Msun').v
        metal_mass = sphere['gas','metal_mass'].in_units('Msun').v
    if ('energy' in flux_types):
        kinetic_energy = sphere['gas','kinetic_energy_corrected'].in_units('erg').v
        thermal_energy = (sphere['gas','cell_mass']*sphere['gas','thermal_energy']).in_units('erg').v
        potential_energy = (sphere['gas','cell_mass'] * \
          ds.arr(sphere['enzo','Grav_Potential'].v, 'code_length**2/code_time**2')).in_units('erg').v
        total_energy = kinetic_energy + thermal_energy + potential_energy
    if ('entropy' in flux_types):
        entropy = sphere['gas','entropy'].in_units('keV*cm**2').v
    if ('O_ion_mass' in flux_types):
        trident.add_ion_fields(ds, ions='all', ftype='gas')
        abundances = trident.ion_balance.solar_abundance
        OI_frac = sphere['O_p0_ion_fraction'].v
        OII_frac = sphere['O_p1_ion_fraction'].v
        OIII_frac = sphere['O_p2_ion_fraction'].v
        OIV_frac = sphere['O_p3_ion_fraction'].v
        OV_frac = sphere['O_p4_ion_fraction'].v
        OVI_frac = sphere['O_p5_ion_fraction'].v
        OVII_frac = sphere['O_p6_ion_fraction'].v
        OVIII_frac = sphere['O_p7_ion_fraction'].v
        OIX_frac = sphere['O_p8_ion_fraction'].v
        renorm = OI_frac + OII_frac + OIII_frac + OIV_frac + OV_frac + \
          OVI_frac + OVII_frac + OVIII_frac + OIX_frac
        O_frac = abundances['O']/(sum(abundances.values()) - abundances['H'] - abundances['He'])
        O_mass = sphere['metal_mass'].in_units('Msun').v*O_frac
        OI_mass = OI_frac/renorm*O_mass
        OII_mass = OII_frac/renorm*O_mass
        OIII_mass = OIII_frac/renorm*O_mass
        OIV_mass = OIV_frac/renorm*O_mass
        OV_mass = OV_frac/renorm*O_mass
        OVI_mass = OVI_frac/renorm*O_mass
        OVII_mass = OVII_frac/renorm*O_mass
        OVIII_mass = OVIII_frac/renorm*O_mass
        OIX_mass = OIX_frac/renorm*O_mass

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
        frus_filename += 'z_op' + str(op_angle)
    if (axis=='x'):
        theta = np.arccos(x/radius)
        phi = np.arctan2(z, y)
        frus_filename += 'x_op' + str(op_angle)
    if (axis=='y'):
        theta = np.arccos(y/radius)
        phi = np.arctan2(x, z)
        frus_filename += 'y_op' + str(op_angle)
    if (axis=='disk minor axis'):
        x_disk = sphere['gas','x_disk'].in_units('kpc').v
        y_disk = sphere['gas','y_disk'].in_units('kpc').v
        z_disk = sphere['gas','z_disk'].in_units('kpc').v
        theta = np.arccos(z_disk/radius)
        phi = np.arctan2(y_disk, x_disk)
        frus_filename += 'disk_op' + str(op_angle)
    if (type(axis)==tuple) or (type(axis)==list):
        axis = np.array(axis)
        norm_axis = axis / np.sqrt((axis**2.).sum())
        # Define other unit vectors orthagonal to the angular momentum vector
        np.random.seed(99)
        x_axis = np.random.randn(3)            # take a random vector
        x_axis -= x_axis.dot(norm_axis) * norm_axis       # make it orthogonal to L
        x_axis /= np.linalg.norm(x_axis)            # normalize it
        y_axis = np.cross(norm_axis, x_axis)           # cross product with L
        x_vec = ds.arr(x_axis)
        y_vec = ds.arr(y_axis)
        z_vec = ds.arr(norm_axis)
        # Calculate the rotation matrix for converting from original coordinate system
        # into this new basis
        xhat = np.array([1,0,0])
        yhat = np.array([0,1,0])
        zhat = np.array([0,0,1])
        transArr0 = np.array([[xhat.dot(x_vec), xhat.dot(y_vec), xhat.dot(z_vec)],
                             [yhat.dot(x_vec), yhat.dot(y_vec), yhat.dot(z_vec)],
                             [zhat.dot(x_vec), zhat.dot(y_vec), zhat.dot(z_vec)]])
        rotationArr = np.linalg.inv(transArr0)
        x_rot = rotationArr[0][0]*x + rotationArr[0][1]*y + rotationArr[0][2]*z
        y_rot = rotationArr[1][0]*x + rotationArr[1][1]*y + rotationArr[1][2]*z
        z_rot = rotationArr[2][0]*x + rotationArr[2][1]*y + rotationArr[2][2]*z
        theta = np.arccos(z_rot/radius)
        frus_filename += 'axis_' + str(axis[0]) + '_' + str(axis[1]) + '_' + str(axis[2]) + '_op' + str(op_angle)

    # Load list of satellite positions
    if (sat_radius!=0):
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
        if ('mass' in flux_types):
            mass_nosat = mass[bool_nosat]
            metal_mass_nosat = metal_mass[bool_nosat]
        if ('energy' in flux_types):
            kinetic_energy_nosat = kinetic_energy[bool_nosat]
            thermal_energy_nosat = thermal_energy[bool_nosat]
            potential_energy_nosat = potential_energy[bool_nosat]
            total_energy_nosat = total_energy[bool_nosat]
        if ('entropy' in flux_types):
            entropy_nosat = entropy[bool_nosat]
        if ('O_ion_mass' in flux_types):
            O_mass_nosat = O_mass[bool_nosat]
            OI_mass_nosat = OI_mass[bool_nosat]
            OII_mass_nosat = OII_mass[bool_nosat]
            OIII_mass_nosat = OIII_mass[bool_nosat]
            OIV_mass_nosat = OIV_mass[bool_nosat]
            OV_mass_nosat = OV_mass[bool_nosat]
            OVI_mass_nosat = OVI_mass[bool_nosat]
            OVII_mass_nosat = OVII_mass[bool_nosat]
            OVIII_mass_nosat = OVIII_mass[bool_nosat]
            OIX_mass_nosat = OIX_mass[bool_nosat]
    else:
        radius_nosat = radius
        theta_nosat = theta
        rad_vel_nosat = rad_vel
        temperature_nosat = temperature
        if ('mass' in flux_types):
            mass_nosat = mass
            metal_mass_nosat = metal_mass
        if ('energy' in flux_types):
            kinetic_energy_nosat = kinetic_energy
            thermal_energy_nosat = thermal_energy
            potential_energy_nosat = potential_energy
            total_energy_nosat = total_energy
        if ('entropy' in flux_types):
            entropy_nosat = entropy
        if ('O_ion_mass' in flux_types):
            O_mass_nosat = O_mass
            OI_mass_nosat = OI_mass
            OII_mass_nosat = OII_mass
            OIII_mass_nosat = OIII_mass
            OIV_mass_nosat = OIV_mass
            OV_mass_nosat = OV_mass
            OVI_mass_nosat = OVI_mass
            OVII_mass_nosat = OVII_mass
            OVIII_mass_nosat = OVIII_mass
            OIX_mass_nosat = OIX_mass

    # Cut satellite-removed data to frustum of interest
    bool_frus = (theta_nosat >= min_theta) & (theta_nosat <= max_theta)

    radius_nosat_frus = radius_nosat[bool_frus]
    rad_vel_nosat_frus = rad_vel_nosat[bool_frus]
    temperature_nosat_frus = temperature_nosat[bool_frus]
    if ('mass' in flux_types):
        mass_nosat_frus = mass_nosat[bool_frus]
        metal_mass_nosat_frus = metal_mass_nosat[bool_frus]
    if ('energy' in flux_types):
        kinetic_energy_nosat_frus = kinetic_energy_nosat[bool_frus]
        thermal_energy_nosat_frus = thermal_energy_nosat[bool_frus]
        potential_energy_nosat_frus = potential_energy_nosat[bool_frus]
        total_energy_nosat_frus = total_energy_nosat[bool_frus]
    if ('entropy' in flux_types):
        entropy_nosat_frus = entropy_nosat[bool_frus]
    if ('O_ion_mass' in flux_types):
        O_mass_nosat_frus = O_mass_nosat[bool_frus]
        OI_mass_nosat_frus = OI_mass_nosat[bool_frus]
        OII_mass_nosat_frus = OII_mass_nosat[bool_frus]
        OIII_mass_nosat_frus = OIII_mass_nosat[bool_frus]
        OIV_mass_nosat_frus = OIV_mass_nosat[bool_frus]
        OV_mass_nosat_frus = OV_mass_nosat[bool_frus]
        OVI_mass_nosat_frus = OVI_mass_nosat[bool_frus]
        OVII_mass_nosat_frus = OVII_mass_nosat[bool_frus]
        OVIII_mass_nosat_frus = OVIII_mass_nosat[bool_frus]
        OIX_mass_nosat_frus = OIX_mass_nosat[bool_frus]

    # Cut satellite-removed frustum data on temperature
    # These are lists of lists where the first index goes from 0 to 4 for
    # [all gas, cold, cool, warm, hot]
    if (sat_radius!=0):
        print('Cutting satellite-removed data on temperature')
    else:
        print('Cutting data on temperature')
    radius_nosat_frus_Tcut = []
    rad_vel_nosat_frus_Tcut = []
    if ('mass' in flux_types):
        mass_nosat_frus_Tcut = []
        metal_mass_nosat_frus_Tcut = []
    if ('energy' in flux_types):
        kinetic_energy_nosat_frus_Tcut = []
        thermal_energy_nosat_frus_Tcut = []
        potential_energy_nosat_frus_Tcut = []
        total_energy_nosat_frus_Tcut = []
    if ('entropy' in flux_types):
        entropy_nosat_frus_Tcut = []
    if ('O_ion_mass' in flux_types):
        O_mass_nosat_frus_Tcut = []
        OI_mass_nosat_frus_Tcut = []
        OII_mass_nosat_frus_Tcut = []
        OIII_mass_nosat_frus_Tcut = []
        OIV_mass_nosat_frus_Tcut = []
        OV_mass_nosat_frus_Tcut = []
        OVI_mass_nosat_frus_Tcut = []
        OVII_mass_nosat_frus_Tcut = []
        OVIII_mass_nosat_frus_Tcut = []
        OIX_mass_nosat_frus_Tcut = []
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
        if ('mass' in flux_types):
            mass_nosat_frus_Tcut.append(mass_nosat_frus[bool_temp_nosat_frus])
            metal_mass_nosat_frus_Tcut.append(metal_mass_nosat_frus[bool_temp_nosat_frus])
        if ('energy' in flux_types):
            kinetic_energy_nosat_frus_Tcut.append(kinetic_energy_nosat_frus[bool_temp_nosat_frus])
            thermal_energy_nosat_frus_Tcut.append(thermal_energy_nosat_frus[bool_temp_nosat_frus])
            potential_energy_nosat_frus_Tcut.append(potential_energy_nosat_frus[bool_temp_nosat_frus])
            total_energy_nosat_frus_Tcut.append(total_energy_nosat_frus[bool_temp_nosat_frus])
        if ('entropy' in flux_types):
            entropy_nosat_frus_Tcut.append(entropy_nosat_frus[bool_temp_nosat_frus])
        if ('O_ion_mass' in flux_types):
            O_mass_nosat_frus_Tcut.append(O_mass_nosat_frus[bool_temp_nosat_frus])
            OI_mass_nosat_frus_Tcut.append(OI_mass_nosat_frus[bool_temp_nosat_frus])
            OII_mass_nosat_frus_Tcut.append(OII_mass_nosat_frus[bool_temp_nosat_frus])
            OIII_mass_nosat_frus_Tcut.append(OIII_mass_nosat_frus[bool_temp_nosat_frus])
            OIV_mass_nosat_frus_Tcut.append(OIV_mass_nosat_frus[bool_temp_nosat_frus])
            OV_mass_nosat_frus_Tcut.append(OV_mass_nosat_frus[bool_temp_nosat_frus])
            OVI_mass_nosat_frus_Tcut.append(OVI_mass_nosat_frus[bool_temp_nosat_frus])
            OVII_mass_nosat_frus_Tcut.append(OVII_mass_nosat_frus[bool_temp_nosat_frus])
            OVIII_mass_nosat_frus_Tcut.append(OVIII_mass_nosat_frus[bool_temp_nosat_frus])
            OIX_mass_nosat_frus_Tcut.append(OIX_mass_nosat_frus[bool_temp_nosat_frus])

    # Loop over radii
    for i in range(len(radii)-1):
        inner_r = radii[i].v
        outer_r = radii[i+1].v

        if (i%10==0): print("Computing radius " + str(i) + "/" + str(len(radii)) + \
                            " for snapshot " + snap)

        # Compute net, in, and out totals within the frustum with satellites removed
        # These are nested lists where the first index goes from 0 to 2 for [net, in, out]
        # and the second index goes from 0 to 4 for [all, cold, cool, warm, hot]
        if ('mass' in flux_types):
            mass_total_nosat = []
            metals_total_nosat = []
        if ('energy' in flux_types):
            kinetic_energy_total_nosat = []
            thermal_energy_total_nosat = []
            potential_energy_total_nosat = []
            total_energy_total_nosat = []
        if ('entropy' in flux_types):
            entropy_total_nosat = []
        if ('O_ion_mass' in flux_types):
            O_total_nosat = []
            OI_total_nosat = []
            OII_total_nosat = []
            OIII_total_nosat = []
            OIV_total_nosat = []
            OV_total_nosat = []
            OVI_total_nosat = []
            OVII_total_nosat = []
            OVIII_total_nosat = []
            OIX_total_nosat = []
        for j in range(3):
            if ('mass' in flux_types):
                mass_total_nosat.append([])
                metals_total_nosat.append([])
            if ('energy' in flux_types):
                kinetic_energy_total_nosat.append([])
                thermal_energy_total_nosat.append([])
                potential_energy_total_nosat.append([])
                total_energy_total_nosat.append([])
            if ('entropy' in flux_types):
                entropy_total_nosat.append([])
            if ('O_ion_mass' in flux_types):
                O_total_nosat.append([])
                OI_total_nosat.append([])
                OII_total_nosat.append([])
                OIII_total_nosat.append([])
                OIV_total_nosat.append([])
                OV_total_nosat.append([])
                OVI_total_nosat.append([])
                OVII_total_nosat.append([])
                OVIII_total_nosat.append([])
                OIX_total_nosat.append([])
            for k in range(5):
                bool_in_r = (radius_nosat_frus_Tcut[k] > inner_r) & (radius_nosat_frus_Tcut[k] < outer_r) & \
                  (rad_vel_nosat_frus_Tcut[k] < 0.)
                bool_out_r = (radius_nosat_frus_Tcut[k] > inner_r) & (radius_nosat_frus_Tcut[k] < outer_r) & \
                  (rad_vel_nosat_frus_Tcut[k] > 0.)
                if (j==0):
                    if ('mass' in flux_types):
                        mass_total_nosat[j].append((np.sum(mass_nosat_frus_Tcut[k][bool_out_r]) + \
                          np.sum(mass_nosat_frus_Tcut[k][bool_in_r])))
                        metals_total_nosat[j].append((np.sum(metal_mass_nosat_frus_Tcut[k][bool_out_r]) + \
                          np.sum(metal_mass_nosat_frus_Tcut[k][bool_in_r])))
                    if ('energy' in flux_types):
                        kinetic_energy_total_nosat[j].append((np.sum(kinetic_energy_nosat_frus_Tcut[k][bool_out_r]) + \
                          np.sum(kinetic_energy_nosat_frus_Tcut[k][bool_in_r])))
                        thermal_energy_total_nosat[j].append((np.sum(thermal_energy_nosat_frus_Tcut[k][bool_out_r]) + \
                          np.sum(thermal_energy_nosat_frus_Tcut[k][bool_in_r])))
                        potential_energy_total_nosat[j].append((np.sum(potential_energy_nosat_frus_Tcut[k][bool_out_r]) + \
                          np.sum(potential_energy_nosat_frus_Tcut[k][bool_in_r])))
                        total_energy_total_nosat[j].append((np.sum(total_energy_nosat_frus_Tcut[k][bool_out_r]) + \
                          np.sum(total_energy_nosat_frus_Tcut[k][bool_in_r])))
                    if ('entropy' in flux_types):
                        entropy_total_nosat[j].append((np.sum(entropy_nosat_frus_Tcut[k][bool_out_r]) + \
                        np.sum(entropy_nosat_frus_Tcut[k][bool_in_r])))
                    if ('O_ion_mass' in flux_types):
                        O_total_nosat[j].append((np.sum(O_mass_nosat_frus_Tcut[k][bool_out_r]) + \
                          np.sum(O_mass_nosat_frus_Tcut[k][bool_in_r])))
                        OI_total_nosat[j].append((np.sum(OI_mass_nosat_frus_Tcut[k][bool_out_r]) + \
                          np.sum(OI_mass_nosat_frus_Tcut[k][bool_in_r])))
                        OII_total_nosat[j].append((np.sum(OII_mass_nosat_frus_Tcut[k][bool_out_r]) + \
                          np.sum(OII_mass_nosat_frus_Tcut[k][bool_in_r])))
                        OIII_total_nosat[j].append((np.sum(OIII_mass_nosat_frus_Tcut[k][bool_out_r]) + \
                          np.sum(OIII_mass_nosat_frus_Tcut[k][bool_in_r])))
                        OIV_total_nosat[j].append((np.sum(OIV_mass_nosat_frus_Tcut[k][bool_out_r]) + \
                          np.sum(OIV_mass_nosat_frus_Tcut[k][bool_in_r])))
                        OV_total_nosat[j].append((np.sum(OV_mass_nosat_frus_Tcut[k][bool_out_r]) + \
                          np.sum(OV_mass_nosat_frus_Tcut[k][bool_in_r])))
                        OVI_total_nosat[j].append((np.sum(OVI_mass_nosat_frus_Tcut[k][bool_out_r]) + \
                          np.sum(OVI_mass_nosat_frus_Tcut[k][bool_in_r])))
                        OVII_total_nosat[j].append((np.sum(OVII_mass_nosat_frus_Tcut[k][bool_out_r]) + \
                          np.sum(OVII_mass_nosat_frus_Tcut[k][bool_in_r])))
                        OVIII_total_nosat[j].append((np.sum(OVIII_mass_nosat_frus_Tcut[k][bool_out_r]) + \
                          np.sum(OVIII_mass_nosat_frus_Tcut[k][bool_in_r])))
                        OIX_total_nosat[j].append((np.sum(OIX_mass_nosat_frus_Tcut[k][bool_out_r]) + \
                          np.sum(OIX_mass_nosat_frus_Tcut[k][bool_in_r])))
                if (j==1):
                    if ('mass' in flux_types):
                        mass_total_nosat[j].append(np.sum(mass_nosat_frus_Tcut[k][bool_in_r]))
                        metals_total_nosat[j].append(np.sum(metal_mass_nosat_frus_Tcut[k][bool_in_r]))
                    if ('energy' in flux_types):
                        kinetic_energy_total_nosat[j].append(np.sum(kinetic_energy_nosat_frus_Tcut[k][bool_in_r]))
                        thermal_energy_total_nosat[j].append(np.sum(thermal_energy_nosat_frus_Tcut[k][bool_in_r]))
                        potential_energy_total_nosat[j].append(np.sum(potential_energy_nosat_frus_Tcut[k][bool_in_r]))
                        total_energy_total_nosat[j].append(np.sum(total_energy_nosat_frus_Tcut[k][bool_in_r]))
                    if ('entropy' in flux_types):
                        entropy_total_nosat[j].append(np.sum(entropy_nosat_frus_Tcut[k][bool_in_r]))
                    if ('O_ion_mass' in flux_types):
                        O_total_nosat[j].append(np.sum(O_mass_nosat_frus_Tcut[k][bool_in_r]))
                        OI_total_nosat[j].append(np.sum(OI_mass_nosat_frus_Tcut[k][bool_in_r]))
                        OII_total_nosat[j].append(np.sum(OII_mass_nosat_frus_Tcut[k][bool_in_r]))
                        OIII_total_nosat[j].append(np.sum(OIII_mass_nosat_frus_Tcut[k][bool_in_r]))
                        OIV_total_nosat[j].append(np.sum(OIV_mass_nosat_frus_Tcut[k][bool_in_r]))
                        OV_total_nosat[j].append(np.sum(OV_mass_nosat_frus_Tcut[k][bool_in_r]))
                        OVI_total_nosat[j].append(np.sum(OVI_mass_nosat_frus_Tcut[k][bool_in_r]))
                        OVII_total_nosat[j].append(np.sum(OVII_mass_nosat_frus_Tcut[k][bool_in_r]))
                        OVIII_total_nosat[j].append(np.sum(OVIII_mass_nosat_frus_Tcut[k][bool_in_r]))
                        OIX_total_nosat[j].append(np.sum(OIX_mass_nosat_frus_Tcut[k][bool_in_r]))
                if (j==2):
                    if ('mass' in flux_types):
                        mass_total_nosat[j].append(np.sum(mass_nosat_frus_Tcut[k][bool_out_r]))
                        metals_total_nosat[j].append(np.sum(metal_mass_nosat_frus_Tcut[k][bool_out_r]))
                    if ('energy' in flux_types):
                        kinetic_energy_total_nosat[j].append(np.sum(kinetic_energy_nosat_frus_Tcut[k][bool_out_r]))
                        thermal_energy_total_nosat[j].append(np.sum(thermal_energy_nosat_frus_Tcut[k][bool_out_r]))
                        potential_energy_total_nosat[j].append(np.sum(potential_energy_nosat_frus_Tcut[k][bool_out_r]))
                        total_energy_total_nosat[j].append(np.sum(total_energy_nosat_frus_Tcut[k][bool_out_r]))
                    if ('entropy' in flux_types):
                        entropy_total_nosat[j].append(np.sum(entropy_nosat_frus_Tcut[k][bool_out_r]))
                    if ('O_ion_mass' in flux_types):
                        O_total_nosat[j].append(np.sum(O_mass_nosat_frus_Tcut[k][bool_out_r]))
                        OI_total_nosat[j].append(np.sum(OI_mass_nosat_frus_Tcut[k][bool_out_r]))
                        OII_total_nosat[j].append(np.sum(OII_mass_nosat_frus_Tcut[k][bool_out_r]))
                        OIII_total_nosat[j].append(np.sum(OIII_mass_nosat_frus_Tcut[k][bool_out_r]))
                        OIV_total_nosat[j].append(np.sum(OIV_mass_nosat_frus_Tcut[k][bool_out_r]))
                        OV_total_nosat[j].append(np.sum(OV_mass_nosat_frus_Tcut[k][bool_out_r]))
                        OVI_total_nosat[j].append(np.sum(OVI_mass_nosat_frus_Tcut[k][bool_out_r]))
                        OVII_total_nosat[j].append(np.sum(OVII_mass_nosat_frus_Tcut[k][bool_out_r]))
                        OVIII_total_nosat[j].append(np.sum(OVIII_mass_nosat_frus_Tcut[k][bool_out_r]))
                        OIX_total_nosat[j].append(np.sum(OIX_mass_nosat_frus_Tcut[k][bool_out_r]))

        # Add everything to the tables
        new_row = [zsnap, inner_r, outer_r]
        if ('mass' in flux_types):
            new_row += [mass_total_nosat[0][0], metals_total_nosat[0][0], \
            mass_total_nosat[1][0], mass_total_nosat[2][0], metals_total_nosat[1][0], metals_total_nosat[2][0], \
            mass_total_nosat[0][1], mass_total_nosat[1][1], mass_total_nosat[2][1], \
            mass_total_nosat[0][2], mass_total_nosat[1][2], mass_total_nosat[2][2], \
            mass_total_nosat[0][3], mass_total_nosat[1][3], mass_total_nosat[2][3], \
            mass_total_nosat[0][4], mass_total_nosat[1][4], mass_total_nosat[2][4], \
            metals_total_nosat[0][1], metals_total_nosat[1][1], metals_total_nosat[2][1], \
            metals_total_nosat[0][2], metals_total_nosat[1][2], metals_total_nosat[2][2], \
            metals_total_nosat[0][3], metals_total_nosat[1][3], metals_total_nosat[2][3], \
            metals_total_nosat[0][4], metals_total_nosat[1][4], metals_total_nosat[2][4]]
        if ('energy' in flux_types):
            new_row += [kinetic_energy_total_nosat[0][0], thermal_energy_total_nosat[0][0], \
            potential_energy_total_nosat[0][0], total_energy_total_nosat[0][0], \
            kinetic_energy_total_nosat[1][0], kinetic_energy_total_nosat[2][0], \
            thermal_energy_total_nosat[1][0], thermal_energy_total_nosat[2][0], \
            potential_energy_total_nosat[1][0], potential_energy_total_nosat[2][0], \
            total_energy_total_nosat[1][0], total_energy_total_nosat[2][0], \
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
            total_energy_total_nosat[0][1], total_energy_total_nosat[1][1], total_energy_total_nosat[2][1], \
            total_energy_total_nosat[0][2], total_energy_total_nosat[1][2], total_energy_total_nosat[2][2], \
            total_energy_total_nosat[0][3], total_energy_total_nosat[1][3], total_energy_total_nosat[2][3], \
            total_energy_total_nosat[0][4], total_energy_total_nosat[1][4], total_energy_total_nosat[2][4]]
        if ('entropy' in flux_types):
            new_row += [entropy_total_nosat[0][0], \
            entropy_total_nosat[1][0], entropy_total_nosat[2][0], \
            entropy_total_nosat[0][1], entropy_total_nosat[1][1], entropy_total_nosat[2][1], \
            entropy_total_nosat[0][2], entropy_total_nosat[1][2], entropy_total_nosat[2][2], \
            entropy_total_nosat[0][3], entropy_total_nosat[1][3], entropy_total_nosat[2][3], \
            entropy_total_nosat[0][4], entropy_total_nosat[1][4], entropy_total_nosat[2][4]]
        if ('O_ion_mass' in flux_types):
            new_row += [O_total_nosat[0][0], O_total_nosat[1][0], O_total_nosat[2][0], \
            O_total_nosat[0][1], O_total_nosat[1][1], O_total_nosat[2][1], \
            O_total_nosat[0][2], O_total_nosat[1][2], O_total_nosat[2][2], \
            O_total_nosat[0][3], O_total_nosat[1][3], O_total_nosat[2][3], \
            O_total_nosat[0][4], O_total_nosat[1][4], O_total_nosat[2][4], \
            OI_total_nosat[0][0], OI_total_nosat[1][0], OI_total_nosat[2][0], \
            OI_total_nosat[0][1], OI_total_nosat[1][1], OI_total_nosat[2][1], \
            OI_total_nosat[0][2], OI_total_nosat[1][2], OI_total_nosat[2][2], \
            OI_total_nosat[0][3], OI_total_nosat[1][3], OI_total_nosat[2][3], \
            OI_total_nosat[0][4], OI_total_nosat[1][4], OI_total_nosat[2][4], \
            OII_total_nosat[0][0], OII_total_nosat[1][0], OII_total_nosat[2][0], \
            OII_total_nosat[0][1], OII_total_nosat[1][1], OII_total_nosat[2][1], \
            OII_total_nosat[0][2], OII_total_nosat[1][2], OII_total_nosat[2][2], \
            OII_total_nosat[0][3], OII_total_nosat[1][3], OII_total_nosat[2][3], \
            OII_total_nosat[0][4], OII_total_nosat[1][4], OII_total_nosat[2][4], \
            OIII_total_nosat[0][0], OIII_total_nosat[1][0], OIII_total_nosat[2][0], \
            OIII_total_nosat[0][1], OIII_total_nosat[1][1], OIII_total_nosat[2][1], \
            OIII_total_nosat[0][2], OIII_total_nosat[1][2], OIII_total_nosat[2][2], \
            OIII_total_nosat[0][3], OIII_total_nosat[1][3], OIII_total_nosat[2][3], \
            OIII_total_nosat[0][4], OIII_total_nosat[1][4], OIII_total_nosat[2][4], \
            OIV_total_nosat[0][0], OIV_total_nosat[1][0], OIV_total_nosat[2][0], \
            OIV_total_nosat[0][1], OIV_total_nosat[1][1], OIV_total_nosat[2][1], \
            OIV_total_nosat[0][2], OIV_total_nosat[1][2], OIV_total_nosat[2][2], \
            OIV_total_nosat[0][3], OIV_total_nosat[1][3], OIV_total_nosat[2][3], \
            OIV_total_nosat[0][4], OIV_total_nosat[1][4], OIV_total_nosat[2][4], \
            OV_total_nosat[0][0], OV_total_nosat[1][0], OV_total_nosat[2][0], \
            OV_total_nosat[0][1], OV_total_nosat[1][1], OV_total_nosat[2][1], \
            OV_total_nosat[0][2], OV_total_nosat[1][2], OV_total_nosat[2][2], \
            OV_total_nosat[0][3], OV_total_nosat[1][3], OV_total_nosat[2][3], \
            OV_total_nosat[0][4], OV_total_nosat[1][4], OV_total_nosat[2][4], \
            OVI_total_nosat[0][0], OVI_total_nosat[1][0], OVI_total_nosat[2][0], \
            OVI_total_nosat[0][1], OVI_total_nosat[1][1], OVI_total_nosat[2][1], \
            OVI_total_nosat[0][2], OVI_total_nosat[1][2], OVI_total_nosat[2][2], \
            OVI_total_nosat[0][3], OVI_total_nosat[1][3], OVI_total_nosat[2][3], \
            OVI_total_nosat[0][4], OVI_total_nosat[1][4], OVI_total_nosat[2][4], \
            OVII_total_nosat[0][0], OVII_total_nosat[1][0], OVII_total_nosat[2][0], \
            OVII_total_nosat[0][1], OVII_total_nosat[1][1], OVII_total_nosat[2][1], \
            OVII_total_nosat[0][2], OVII_total_nosat[1][2], OVII_total_nosat[2][2], \
            OVII_total_nosat[0][3], OVII_total_nosat[1][3], OVII_total_nosat[2][3], \
            OVII_total_nosat[0][4], OVII_total_nosat[1][4], OVII_total_nosat[2][4], \
            OVIII_total_nosat[0][0], OVIII_total_nosat[1][0], OVIII_total_nosat[2][0], \
            OVIII_total_nosat[0][1], OVIII_total_nosat[1][1], OVIII_total_nosat[2][1], \
            OVIII_total_nosat[0][2], OVIII_total_nosat[1][2], OVIII_total_nosat[2][2], \
            OVIII_total_nosat[0][3], OVIII_total_nosat[1][3], OVIII_total_nosat[2][3], \
            OVIII_total_nosat[0][4], OVIII_total_nosat[1][4], OVIII_total_nosat[2][4], \
            OIX_total_nosat[0][0], OIX_total_nosat[1][0], OIX_total_nosat[2][0], \
            OIX_total_nosat[0][1], OIX_total_nosat[1][1], OIX_total_nosat[2][1], \
            OIX_total_nosat[0][2], OIX_total_nosat[1][2], OIX_total_nosat[2][2], \
            OIX_total_nosat[0][3], OIX_total_nosat[1][3], OIX_total_nosat[2][3], \
            OIX_total_nosat[0][4], OIX_total_nosat[1][4], OIX_total_nosat[2][4]]
        totals.add_row(new_row)

    totals = set_table_units(totals)

    fluxtype_filename = ''
    if ('mass' in flux_types):
        fluxtype_filename += '_mass'
    if ('energy' in flux_types):
        fluxtype_filename += '_energy'
    if ('entropy' in flux_types):
        fluxtype_filename += '_entropy'
    if ('O_ion_mass' in flux_types):
        fluxtype_filename += '_Oions'

    # Save to file
    if (sat_radius!=0.):
        totals.write(tablename + '_nosat_frustum_' + frus_filename + fluxtype_filename + '.hdf5', path='all_data', serialize_meta=True, overwrite=True)
    else:
        totals.write(tablename + '_frustum_' + frus_filename + fluxtype_filename + '.hdf5', path='all_data', serialize_meta=True, overwrite=True)

    return "Totals have been calculated for snapshot " + snap + "!"

def calc_totals_cylinder(ds, snap, zsnap, refine_width_kpc, tablename, surface_args, flux_types, **kwargs):
    '''This function calculates the totals of gas properties between surfaces within a cylinder,
    with satellites removed, at a variety of heights or radii. It
    uses the dataset stored in 'ds', which is from time snapshot 'snap', has redshift
    'zsnap', and has width of the refine box in kpc 'refine_width_kpc', and stores the totals in
    'tablename'. 'surface_args' gives the properties of the cylinder.'''

    sat = kwargs.get('sat')
    sat_radius = kwargs.get('sat_radius', 0.)

    halo_center_kpc = ds.halo_center_kpc

    cmtopc = 3.086e18
    stoyr = 3.154e7
    gtoMsun = 1.989e33

    units_kpc = surface_args[8]
    if (units_kpc):
        bottom_edge = ds.quan(surface_args[3], 'kpc')
        top_edge = ds.quan(surface_args[4], 'kpc')
        cyl_radius = ds.quan(surface_args[5], 'kpc')
    else:
        bottom_edge = surface_args[3]*refine_width_kpc
        top_edge = surface_args[4]*refine_width_kpc
        cyl_radius = surface_args[5]*refine_width_kpc
    if (surface_args[6]=='height'):
        dz = (top_edge - bottom_edge)/surface_args[7]
    elif (surface_args[6]=='radius'):
        dz = cyl_radius/surface_args[7]
    axis = surface_args[1]
    flip = surface_args[2]

    # Set up table of everything we want
    # NOTE: Make sure table units are updated when things are added to this table!
    if (surface_args[6]=='height'):
        names_list = ('redshift', 'bottom_edge', 'top_edge')
    elif (surface_args[6]=='radius'):
        names_list = ('redshift', 'inner_radius', 'outer_radius')
    types_list = ('f8', 'f8', 'f8')
    if ('mass' in flux_types):
        new_names = ('net_mass', 'net_metals', \
        'mass_in', 'mass_out', 'metals_in', 'metals_out', \
        'net_cold_mass', 'cold_mass_in', 'cold_mass_out', \
        'net_cool_mass', 'cool_mass_in', 'cool_mass_out', \
        'net_warm_mass', 'warm_mass_in', 'warm_mass_out', \
        'net_hot_mass', 'hot_mass_in', 'hot_mass_out', \
        'net_cold_metals', 'cold_metals_in', 'cold_metals_out', \
        'net_cool_metals', 'cool_metals_in', 'cool_metals_out', \
        'net_warm_metals', 'warm_metals_in', 'warm_metals_out', \
        'net_hot_metals', 'hot_metals_in', 'hot_metals_out')
        new_types = ('f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', \
        'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', \
        'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8')
        names_list += new_names
        types_list += new_types
    if ('energy' in flux_types):
        new_names = ('net_kinetic_energy', 'net_thermal_energy', 'net_potential_energy', 'net_total_energy', \
        'kinetic_energy_in', 'kinetic_energy_out', \
        'thermal_energy_in', 'thermal_energy_out', \
        'potential_energy_in', 'potential_energy_out', \
        'total_energy_in', 'total_energy_out', \
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
        'net_cold_total_energy', 'cold_total_energy_in', 'cold_total_energy_out', \
        'net_cool_total_energy', 'cool_total_energy_in', 'cool_total_energy_out', \
        'net_warm_total_energy', 'warm_total_energy_in', 'warm_total_energy_out', \
        'net_hot_total_energy', 'hot_total_energy_in', 'hot_total_energy_out')
        new_types = ('f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', \
        'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', \
        'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', \
        'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', \
        'f8', 'f8', 'f8', 'f8', 'f8', 'f8')
        names_list += new_names
        types_list += new_types
    if ('entropy' in flux_types):
        new_names = ('net_entropy', 'entropy_in', 'entropy_out', \
        'net_cold_entropy', 'cold_entropy_in', 'cold_entropy_out', \
        'net_cool_entropy', 'cool_entropy_in', 'cool_entropy_out', \
        'net_warm_entropy', 'warm_entropy_in', 'warm_entropy_out', \
        'net_hot_entropy', 'hot_entropy_in', 'hot_entropy_out')
        new_types = ('f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', \
        'f8', 'f8', 'f8', 'f8', 'f8', 'f8')
        names_list += new_names
        types_list += new_types
    if ('O_ion_mass' in flux_types):
        new_names = ('net_O_mass', 'O_mass_in', 'O_mass_out', \
        'net_cold_O_mass', 'cold_O_mass_in', 'cold_O_mass_out', \
        'net_cool_O_mass', 'cool_O_mass_in', 'cool_O_mass_out', \
        'net_warm_O_mass', 'warm_O_mass_in', 'warm_O_mass_out', \
        'net_hot_O_mass', 'hot_O_mass_in', 'hot_O_mass_out', \
        'net_OI_mass', 'OI_mass_in', 'OI_mass_out', \
        'net_cold_OI_mass', 'cold_OI_mass_in', 'cold_OI_mass_out', \
        'net_cool_OI_mass', 'cool_OI_mass_in', 'cool_OI_mass_out', \
        'net_warm_OI_mass', 'warm_OI_mass_in', 'warm_OI_mass_out', \
        'net_hot_OI_mass', 'hot_OI_mass_in', 'hot_OI_mass_out', \
        'net_OII_mass', 'OII_mass_in', 'OII_mass_out', \
        'net_cold_OII_mass', 'cold_OII_mass_in', 'cold_OII_mass_out', \
        'net_cool_OII_mass', 'cool_OII_mass_in', 'cool_OII_mass_out', \
        'net_warm_OII_mass', 'warm_OII_mass_in', 'warm_OII_mass_out', \
        'net_hot_OII_mass', 'hot_OII_mass_in', 'hot_OII_mass_out', \
        'net_OIII_mass', 'OIII_mass_in', 'OIII_mass_out', \
        'net_cold_OIII_mass', 'cold_OIII_mass_in', 'cold_OIII_mass_out', \
        'net_cool_OIII_mass', 'cool_OIII_mass_in', 'cool_OIII_mass_out', \
        'net_warm_OIII_mass', 'warm_OIII_mass_in', 'warm_OIII_mass_out', \
        'net_hot_OIII_mass', 'hot_OIII_mass_in', 'hot_OIII_mass_out', \
        'net_OIV_mass', 'OIV_mass_in', 'OIV_mass_out', \
        'net_cold_OIV_mass', 'cold_OIV_mass_in', 'cold_OIV_mass_out', \
        'net_cool_OIV_mass', 'cool_OIV_mass_in', 'cool_OIV_mass_out', \
        'net_warm_OIV_mass', 'warm_OIV_mass_in', 'warm_OIV_mass_out', \
        'net_hot_OIV_mass', 'hot_OIV_mass_in', 'hot_OIV_mass_out', \
        'net_OV_mass', 'OV_mass_in', 'OV_mass_out', \
        'net_cold_OV_mass', 'cold_OV_mass_in', 'cold_OV_mass_out', \
        'net_cool_OV_mass', 'cool_OV_mass_in', 'cool_OV_mass_out', \
        'net_warm_OV_mass', 'warm_OV_mass_in', 'warm_OV_mass_out', \
        'net_hot_OV_mass', 'hot_OV_mass_in', 'hot_OV_mass_out', \
        'net_OVI_mass', 'OVI_mass_in', 'OVI_mass_out', \
        'net_cold_OVI_mass', 'cold_OVI_mass_in', 'cold_OVI_mass_out', \
        'net_cool_OVI_mass', 'cool_OVI_mass_in', 'cool_OVI_mass_out', \
        'net_warm_OVI_mass', 'warm_OVI_mass_in', 'warm_OVI_mass_out', \
        'net_hot_OVI_mass', 'hot_OVI_mass_in', 'hot_OVI_mass_out', \
        'net_OVII_mass', 'OVII_mass_in', 'OVII_mass_out', \
        'net_cold_OVII_mass', 'cold_OVII_mass_in', 'cold_OVII_mass_out', \
        'net_cool_OVII_mass', 'cool_OVII_mass_in', 'cool_OVII_mass_out', \
        'net_warm_OVII_mass', 'warm_OVII_mass_in', 'warm_OVII_mass_out', \
        'net_hot_OVII_mass', 'hot_OVII_mass_in', 'hot_OVII_mass_out', \
        'net_OVIII_mass', 'OVIII_mass_in', 'OVIII_mass_out', \
        'net_cold_OVIII_mass', 'cold_OVIII_mass_in', 'cold_OVIII_mass_out', \
        'net_cool_OVIII_mass', 'cool_OVIII_mass_in', 'cool_OVIII_mass_out', \
        'net_warm_OVIII_mass', 'warm_OVIII_mass_in', 'warm_OVIII_mass_out', \
        'net_hot_OVIII_mass', 'hot_OVIII_mass_in', 'hot_OVIII_mass_out', \
        'net_OIX_mass', 'OIX_mass_in', 'OIX_mass_out', \
        'net_cold_OIX_mass', 'cold_OIX_mass_in', 'cold_OIX_mass_out', \
        'net_cool_OIX_mass', 'cool_OIX_mass_in', 'cool_OIX_mass_out', \
        'net_warm_OIX_mass', 'warm_OIX_mass_in', 'warm_OIX_mass_out', \
        'net_hot_OIX_mass', 'hot_OIX_mass_in', 'hot_OIX_mass_out')
        new_types = ('f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', \
        'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', \
        'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', \
        'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', \
        'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', \
        'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', \
        'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', \
        'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', \
        'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', \
        'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', \
        'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', \
        'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', \
        'f8', 'f8', 'f8','f8', 'f8', 'f8')
        names_list += new_names
        types_list += new_types
    totals = Table(names=names_list, dtype=types_list)

    # Define the surfaces where we want to calculate fluxes
    if (surface_args[6]=='height'):
        surfaces = ds.arr(np.arange(bottom_edge, top_edge+dz, dz), 'kpc')
    elif (surface_args[6]=='radius'):
        surfaces = ds.arr(np.arange(0., cyl_radius+dz, dz), 'kpc')

    # Load arrays of all fields we need
    print('Loading field arrays')
    sphere = ds.sphere(halo_center_kpc, max([bottom_edge, top_edge+dz, cyl_radius+dz]))

    x = sphere['gas','x'].in_units('kpc').v - halo_center_kpc[0].v
    y = sphere['gas','y'].in_units('kpc').v - halo_center_kpc[1].v
    z = sphere['gas','z'].in_units('kpc').v - halo_center_kpc[2].v
    vx = sphere['gas','vx_corrected'].in_units('km/s').v
    vy = sphere['gas','vy_corrected'].in_units('km/s').v
    vz = sphere['gas','vz_corrected'].in_units('km/s').v
    temperature = sphere['gas','temperature'].in_units('K').v
    if ('mass' in flux_types):
        mass = sphere['gas','cell_mass'].in_units('Msun').v
        metal_mass = sphere['gas','metal_mass'].in_units('Msun').v
    if ('energy' in flux_types):
        kinetic_energy = sphere['gas','kinetic_energy_corrected'].in_units('erg').v
        thermal_energy = (sphere['gas','cell_mass']*sphere['gas','thermal_energy']).in_units('erg').v
        potential_energy = (sphere['gas','cell_mass'] * \
          ds.arr(sphere['enzo','Grav_Potential'].v, 'code_length**2/code_time**2')).in_units('erg').v
        total_energy = kinetic_energy + thermal_energy + potential_energy
    if ('entropy' in flux_types):
        entropy = sphere['gas','entropy'].in_units('keV*cm**2').v
    if ('O_ion_mass' in flux_types):
        trident.add_ion_fields(ds, ions='all', ftype='gas')
        abundances = trident.ion_balance.solar_abundance
        OI_frac = sphere['O_p0_ion_fraction'].v
        OII_frac = sphere['O_p1_ion_fraction'].v
        OIII_frac = sphere['O_p2_ion_fraction'].v
        OIV_frac = sphere['O_p3_ion_fraction'].v
        OV_frac = sphere['O_p4_ion_fraction'].v
        OVI_frac = sphere['O_p5_ion_fraction'].v
        OVII_frac = sphere['O_p6_ion_fraction'].v
        OVIII_frac = sphere['O_p7_ion_fraction'].v
        OIX_frac = sphere['O_p8_ion_fraction'].v
        renorm = OI_frac + OII_frac + OIII_frac + OIV_frac + OV_frac + \
          OVI_frac + OVII_frac + OVIII_frac + OIX_frac
        O_frac = abundances['O']/(sum(abundances.values()) - abundances['H'] - abundances['He'])
        O_mass = sphere['metal_mass'].in_units('Msun').v*O_frac
        OI_mass = OI_frac/renorm*O_mass
        OII_mass = OII_frac/renorm*O_mass
        OIII_mass = OIII_frac/renorm*O_mass
        OIV_mass = OIV_frac/renorm*O_mass
        OV_mass = OV_frac/renorm*O_mass
        OVI_mass = OVI_frac/renorm*O_mass
        OVII_mass = OVII_frac/renorm*O_mass
        OVIII_mass = OVIII_frac/renorm*O_mass
        OIX_mass = OIX_frac/renorm*O_mass

    # Cut data to only the cylinder considered here
    if (flip):
        cyl_filename = '-'
    else:
        cyl_filename = ''
    if (axis=='z'):
        norm_coord = z
        rad_coord = np.sqrt(x**2. + y**2.)
        norm_v = vz
        rad_v = vx*x/rad_coord + vy*y/rad_coord
        cyl_filename += 'z'
    if (axis=='x'):
        norm_coord = x
        rad_coord = np.sqrt(y**2. + z**2.)
        norm_v = vx
        rad_v = vz*z/rad_coord + vy*y/rad_coord
        cyl_filename += 'x'
    if (axis=='y'):
        norm_coord = y
        rad_coord = np.sqrt(x**2. + z**2.)
        norm_v = vy
        rad_v = vz*z/rad_coord + vx*x/rad_coord
        cyl_filename += 'y'
    if (axis=='disk minor axis'):
        x_disk = sphere['gas','x_disk'].in_units('kpc').v
        y_disk = sphere['gas','y_disk'].in_units('kpc').v
        z_disk = sphere['gas','z_disk'].in_units('kpc').v
        vx_disk = sphere['gas','vx_disk'].in_units('km/s').v
        vy_disk = sphere['gas','vy_disk'].in_units('km/s').v
        vz_disk = sphere['gas','vz_disk'].in_units('km/s').v
        norm_coord = z_disk
        rad_coord = np.sqrt(x_disk**2. + y_disk**2.)
        norm_v = vz_disk
        rad_v = vx_disk*x_disk/rad_coord + vy_disk*y_disk/rad_coord
        cyl_filename += 'disk'
    if (type(axis)==tuple) or (type(axis)==list):
        axis = np.array(axis)
        norm_axis = axis / np.sqrt((axis**2.).sum())
        # Define other unit vectors orthagonal to the angular momentum vector
        np.random.seed(99)
        x_axis = np.random.randn(3)            # take a random vector
        x_axis -= x_axis.dot(norm_axis) * norm_axis       # make it orthogonal to L
        x_axis /= np.linalg.norm(x_axis)            # normalize it
        y_axis = np.cross(norm_axis, x_axis)           # cross product with L
        x_vec = ds.arr(x_axis)
        y_vec = ds.arr(y_axis)
        z_vec = ds.arr(norm_axis)
        # Calculate the rotation matrix for converting from original coordinate system
        # into this new basis
        xhat = np.array([1,0,0])
        yhat = np.array([0,1,0])
        zhat = np.array([0,0,1])
        transArr0 = np.array([[xhat.dot(x_vec), xhat.dot(y_vec), xhat.dot(z_vec)],
                             [yhat.dot(x_vec), yhat.dot(y_vec), yhat.dot(z_vec)],
                             [zhat.dot(x_vec), zhat.dot(y_vec), zhat.dot(z_vec)]])
        rotationArr = np.linalg.inv(transArr0)
        x_rot = rotationArr[0][0]*x + rotationArr[0][1]*y + rotationArr[0][2]*z
        y_rot = rotationArr[1][0]*x + rotationArr[1][1]*y + rotationArr[1][2]*z
        z_rot = rotationArr[2][0]*x + rotationArr[2][1]*y + rotationArr[2][2]*z
        vx_rot = rotationArr[0][0]*vx + rotationArr[0][1]*vy + rotationArr[0][2]*vz
        vy_rot = rotationArr[1][0]*vx + rotationArr[1][1]*vy + rotationArr[1][2]*vz
        vz_rot = rotationArr[2][0]*vx + rotationArr[2][1]*vy + rotationArr[2][2]*vz
        norm_coord = z_rot
        rad_coord = np.sqrt(x_rot**2. + y_rot**2.)
        norm_v = vz_rot
        rad_v = vx_rot*x_rot/rad_coord + vy_rot*y_rot/rad_coord
        cyl_filename += 'axis_' + str(axis[0]) + '_' + str(axis[1]) + '_' + str(axis[2])
    if (surface_args[6]=='height'): cyl_filename += '_r' + str(surface_args[5]) + '_' + surface_args[6]
    elif (surface_args[6]=='radius'): cyl_filename += '_h' + str(np.abs(surface_args[4]-surface_args[3])) + '_' + surface_args[6]

    # Load list of satellite positions
    if (sat_radius!=0):
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

        # Cut data to remove anything within satellites
        # Restrict to only things that are within the cylinder
        print('Cutting data to remove satellites')
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

        norm_coord_nosat = norm_coord[bool_nosat]
        rad_coord_nosat = rad_coord[bool_nosat]
        norm_v_nosat = norm_v[bool_nosat]
        rad_v_nosat = rad_v[bool_nosat]
        temperature_nosat = temperature[bool_nosat]
        if ('mass' in flux_types):
            mass_nosat = mass[bool_nosat]
            metal_mass_nosat = metal_mass[bool_nosat]
        if ('energy' in flux_types):
            kinetic_energy_nosat = kinetic_energy[bool_nosat]
            thermal_energy_nosat = thermal_energy[bool_nosat]
            potential_energy_nosat = potential_energy[bool_nosat]
            total_energy_nosat = total_energy[bool_nosat]
        if ('entropy' in flux_types):
            entropy_nosat = entropy[bool_nosat]
        if ('O_ion_mass' in flux_types):
            O_mass_nosat = O_mass[bool_nosat]
            OI_mass_nosat = OI_mass[bool_nosat]
            OII_mass_nosat = OII_mass[bool_nosat]
            OIII_mass_nosat = OIII_mass[bool_nosat]
            OIV_mass_nosat = OIV_mass[bool_nosat]
            OV_mass_nosat = OV_mass[bool_nosat]
            OVI_mass_nosat = OVI_mass[bool_nosat]
            OVII_mass_nosat = OVII_mass[bool_nosat]
            OVIII_mass_nosat = OVIII_mass[bool_nosat]
            OIX_mass_nosat = OIX_mass[bool_nosat]
    else:
        norm_coord_nosat = norm_coord
        rad_coord_nosat = rad_coord
        norm_v_nosat = norm_v
        rad_v_nosat = rad_v
        temperature_nosat = temperature
        if ('mass' in flux_types):
            mass_nosat = mass
            metal_mass_nosat = metal_mass
        if ('energy' in flux_types):
            kinetic_energy_nosat = kinetic_energy
            thermal_energy_nosat = thermal_energy
            potential_energy_nosat = potential_energy
            total_energy_nosat = total_energy
        if ('entropy' in flux_types):
            entropy_nosat = entropy
        if ('O_ion_mass' in flux_types):
            O_mass_nosat = O_mass
            OI_mass_nosat = OI_mass
            OII_mass_nosat = OII_mass
            OIII_mass_nosat = OIII_mass
            OIV_mass_nosat = OIV_mass
            OV_mass_nosat = OV_mass
            OVI_mass_nosat = OVI_mass
            OVII_mass_nosat = OVII_mass
            OVIII_mass_nosat = OVIII_mass
            OIX_mass_nosat = OIX_mass

    # Cut satellite-removed data to frustum of interest
    bool_cyl = (norm_coord_nosat >= bottom_edge) & (norm_coord_nosat <= top_edge) & (rad_coord_nosat <= cyl_radius)

    norm_nosat_cyl = norm_coord_nosat[bool_cyl]
    rad_nosat_cyl = rad_coord_nosat[bool_cyl]
    norm_v_nosat_cyl = norm_v_nosat[bool_cyl]
    rad_v_nosat_cyl = rad_v_nosat[bool_cyl]
    temperature_nosat_cyl = temperature_nosat[bool_cyl]
    if ('mass' in flux_types):
        mass_nosat_cyl = mass_nosat[bool_cyl]
        metal_mass_nosat_cyl = metal_mass_nosat[bool_cyl]
    if ('energy' in flux_types):
        kinetic_energy_nosat_cyl = kinetic_energy_nosat[bool_cyl]
        thermal_energy_nosat_cyl = thermal_energy_nosat[bool_cyl]
        potential_energy_nosat_cyl = potential_energy_nosat[bool_cyl]
        total_energy_nosat_cyl = total_energy_nosat[bool_cyl]
    if ('entropy' in flux_types):
        entropy_nosat_cyl = entropy_nosat[bool_cyl]
    if ('O_ion_mass' in flux_types):
        O_mass_nosat_cyl = O_mass_nosat[bool_cyl]
        OI_mass_nosat_cyl = OI_mass_nosat[bool_cyl]
        OII_mass_nosat_cyl = OII_mass_nosat[bool_cyl]
        OIII_mass_nosat_cyl = OIII_mass_nosat[bool_cyl]
        OIV_mass_nosat_cyl = OIV_mass_nosat[bool_cyl]
        OV_mass_nosat_cyl = OV_mass_nosat[bool_cyl]
        OVI_mass_nosat_cyl = OVI_mass_nosat[bool_cyl]
        OVII_mass_nosat_cyl = OVII_mass_nosat[bool_cyl]
        OVIII_mass_nosat_cyl = OVIII_mass_nosat[bool_cyl]
        OIX_mass_nosat_cyl = OIX_mass_nosat[bool_cyl]

    # Cut satellite-removed frustum data on temperature
    # These are lists of lists where the first index goes from 0 to 4 for
    # [all gas, cold, cool, warm, hot]
    if (sat_radius!=0):
        print('Cutting satellite-removed data on temperature')
    else:
        print('Cutting data on temperature')
    norm_nosat_cyl_Tcut = []
    rad_nosat_cyl_Tcut = []
    norm_v_nosat_cyl_Tcut = []
    rad_v_nosat_cyl_Tcut = []
    if ('mass' in flux_types):
        mass_nosat_cyl_Tcut = []
        metal_mass_nosat_cyl_Tcut = []
    if ('energy' in flux_types):
        kinetic_energy_nosat_cyl_Tcut = []
        thermal_energy_nosat_cyl_Tcut = []
        potential_energy_nosat_cyl_Tcut = []
        total_energy_nosat_cyl_Tcut = []
    if ('entropy' in flux_types):
        entropy_nosat_cyl_Tcut = []
    if ('O_ion_mass' in flux_types):
        O_mass_nosat_cyl_Tcut = []
        OI_mass_nosat_cyl_Tcut = []
        OII_mass_nosat_cyl_Tcut = []
        OIII_mass_nosat_cyl_Tcut = []
        OIV_mass_nosat_cyl_Tcut = []
        OV_mass_nosat_cyl_Tcut = []
        OVI_mass_nosat_cyl_Tcut = []
        OVII_mass_nosat_cyl_Tcut = []
        OVIII_mass_nosat_cyl_Tcut = []
        OIX_mass_nosat_cyl_Tcut = []
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
        bool_temp_nosat_cyl = (temperature_nosat_cyl < t_high) & (temperature_nosat_cyl > t_low)
        norm_nosat_cyl_Tcut.append(norm_nosat_cyl[bool_temp_nosat_cyl])
        rad_nosat_cyl_Tcut.append(rad_nosat_cyl[bool_temp_nosat_cyl])
        norm_v_nosat_cyl_Tcut.append(norm_v_nosat_cyl[bool_temp_nosat_cyl])
        rad_v_nosat_cyl_Tcut.append(rad_v_nosat_cyl[bool_temp_nosat_cyl])
        if ('mass' in flux_types):
            mass_nosat_cyl_Tcut.append(mass_nosat_cyl[bool_temp_nosat_cyl])
            metal_mass_nosat_cyl_Tcut.append(metal_mass_nosat_cyl[bool_temp_nosat_cyl])
        if ('energy' in flux_types):
            kinetic_energy_nosat_cyl_Tcut.append(kinetic_energy_nosat_cyl[bool_temp_nosat_cyl])
            thermal_energy_nosat_cyl_Tcut.append(thermal_energy_nosat_cyl[bool_temp_nosat_cyl])
            potential_energy_nosat_cyl_Tcut.append(potential_energy_nosat_cyl[bool_temp_nosat_cyl])
            total_energy_nosat_cyl_Tcut.append(total_energy_nosat_cyl[bool_temp_nosat_cyl])
        if ('entropy' in flux_types):
            entropy_nosat_cyl_Tcut.append(entropy_nosat_cyl[bool_temp_nosat_cyl])
        if ('O_ion_mass' in flux_types):
            O_mass_nosat_cyl_Tcut.append(O_mass_nosat_cyl[bool_temp_nosat_cyl])
            OI_mass_nosat_cyl_Tcut.append(OI_mass_nosat_cyl[bool_temp_nosat_cyl])
            OII_mass_nosat_cyl_Tcut.append(OII_mass_nosat_cyl[bool_temp_nosat_cyl])
            OIII_mass_nosat_cyl_Tcut.append(OIII_mass_nosat_cyl[bool_temp_nosat_cyl])
            OIV_mass_nosat_cyl_Tcut.append(OIV_mass_nosat_cyl[bool_temp_nosat_cyl])
            OV_mass_nosat_cyl_Tcut.append(OV_mass_nosat_cyl[bool_temp_nosat_cyl])
            OVI_mass_nosat_cyl_Tcut.append(OVI_mass_nosat_cyl[bool_temp_nosat_cyl])
            OVII_mass_nosat_cyl_Tcut.append(OVII_mass_nosat_cyl[bool_temp_nosat_cyl])
            OVIII_mass_nosat_cyl_Tcut.append(OVIII_mass_nosat_cyl[bool_temp_nosat_cyl])
            OIX_mass_nosat_cyl_Tcut.append(OIX_mass_nosat_cyl[bool_temp_nosat_cyl])

    # Loop over steps
    for i in range(len(surfaces)):
        inner_surface = surfaces[i].v
        if (i < len(surfaces) - 1): outer_surface = surfaces[i+1].v

        if (surface_args[6]=='radius'):
            if (i%10==0): print("Computing radius " + str(i) + "/" + str(len(surfaces)) + \
                                " for snapshot " + snap)
        elif (surface_args[6]=='height'):
            if (i%10==0): print("Computing height " + str(i) + "/" + str(len(surfaces)) + \
                                " for snapshot " + snap)

        # Compute net, in, and out totals within the cylinder with satellites removed
        # These are nested lists where the first index goes from 0 to 2 for [net, in, out]
        # and the second index goes from 0 to 4 for [all, cold, cool, warm, hot]
        if ('mass' in flux_types):
            mass_total_nosat = []
            metals_total_nosat = []
        if ('energy' in flux_types):
            kinetic_energy_total_nosat = []
            thermal_energy_total_nosat = []
            potential_energy_total_nosat = []
            total_energy_total_nosat = []
        if ('entropy' in flux_types):
            entropy_total_nosat = []
        if ('O_ion_mass' in flux_types):
            O_total_nosat = []
            OI_total_nosat = []
            OII_total_nosat = []
            OIII_total_nosat = []
            OIV_total_nosat = []
            OV_total_nosat = []
            OVI_total_nosat = []
            OVII_total_nosat = []
            OVIII_total_nosat = []
            OIX_total_nosat = []
        for j in range(3):
            if ('mass' in flux_types):
                mass_total_nosat.append([])
                metals_total_nosat.append([])
            if ('energy' in flux_types):
                kinetic_energy_total_nosat.append([])
                thermal_energy_total_nosat.append([])
                potential_energy_total_nosat.append([])
                total_energy_total_nosat.append([])
            if ('entropy' in flux_types):
                entropy_total_nosat.append([])
            if ('O_ion_mass' in flux_types):
                O_total_nosat.append([])
                OI_total_nosat.append([])
                OII_total_nosat.append([])
                OIII_total_nosat.append([])
                OIV_total_nosat.append([])
                OV_total_nosat.append([])
                OVI_total_nosat.append([])
                OVII_total_nosat.append([])
                OVIII_total_nosat.append([])
                OIX_total_nosat.append([])
            for k in range(5):
                if (surface_args[6]=='radius'):
                    bool_in_s = (rad_nosat_cyl_Tcut[k] > inner_surface) & (rad_nosat_cyl_Tcut[k] < outer_surface) & (rad_v_nosat_cyl_Tcut[k] < 0.)
                    bool_out_s = (rad_nosat_cyl_Tcut[k] > inner_surface) & (rad_nosat_cyl_Tcut[k] < outer_surface) & (rad_v_nosat_cyl_Tcut[k] > 0.)
                elif (surface_args[6]=='height'):
                    bool_in_s = (norm_nosat_cyl_Tcut[k] > inner_surface) & (norm_nosat_cyl_Tcut[k] < outer_surface) & (norm_v_nosat_cyl_Tcut[k] < 0.)
                    bool_out_s = (norm_nosat_cyl_Tcut[k] > inner_surface) & (norm_nosat_cyl_Tcut[k] < outer_surface) & (norm_v_nosat_cyl_Tcut[k] > 0.)
                if (j==0):
                    if ('mass' in flux_types):
                        mass_total_nosat[j].append((np.sum(mass_nosat_cyl_Tcut[k][bool_out_s]) + \
                          np.sum(mass_nosat_cyl_Tcut[k][bool_in_s])))
                        metals_total_nosat[j].append((np.sum(metal_mass_nosat_cyl_Tcut[k][bool_out_s]) + \
                          np.sum(metal_mass_nosat_cyl_Tcut[k][bool_in_s])))
                    if ('energy' in flux_types):
                        kinetic_energy_total_nosat[j].append((np.sum(kinetic_energy_nosat_cyl_Tcut[k][bool_out_s]) + \
                          np.sum(kinetic_energy_nosat_cyl_Tcut[k][bool_in_s])))
                        thermal_energy_total_nosat[j].append((np.sum(thermal_energy_nosat_cyl_Tcut[k][bool_out_s]) + \
                          np.sum(thermal_energy_nosat_cyl_Tcut[k][bool_in_s])))
                        potential_energy_total_nosat[j].append((np.sum(potential_energy_nosat_cyl_Tcut[k][bool_out_s]) + \
                          np.sum(potential_energy_nosat_cyl_Tcut[k][bool_in_s])))
                        total_energy_total_nosat[j].append((np.sum(total_energy_nosat_cyl_Tcut[k][bool_out_s]) + \
                          np.sum(total_energy_nosat_cyl_Tcut[k][bool_in_s])))
                    if ('entropy' in flux_types):
                        entropy_total_nosat[j].append((np.sum(entropy_nosat_cyl_Tcut[k][bool_out_s]) + \
                        np.sum(entropy_nosat_cyl_Tcut[k][bool_in_s])))
                    if ('O_ion_mass' in flux_types):
                        O_total_nosat[j].append((np.sum(O_mass_nosat_cyl_Tcut[k][bool_out_s]) + \
                          np.sum(O_mass_nosat_cyl_Tcut[k][bool_in_s])))
                        OI_total_nosat[j].append((np.sum(OI_mass_nosat_cyl_Tcut[k][bool_out_s]) + \
                          np.sum(OI_mass_nosat_cyl_Tcut[k][bool_in_s])))
                        OII_total_nosat[j].append((np.sum(OII_mass_nosat_cyl_Tcut[k][bool_out_s]) + \
                          np.sum(OII_mass_nosat_cyl_Tcut[k][bool_in_s])))
                        OIII_total_nosat[j].append((np.sum(OIII_mass_nosat_cyl_Tcut[k][bool_out_s]) + \
                          np.sum(OIII_mass_nosat_cyl_Tcut[k][bool_in_s])))
                        OIV_total_nosat[j].append((np.sum(OIV_mass_nosat_cyl_Tcut[k][bool_out_s]) + \
                          np.sum(OIV_mass_nosat_cyl_Tcut[k][bool_in_s])))
                        OV_total_nosat[j].append((np.sum(OV_mass_nosat_cyl_Tcut[k][bool_out_s]) + \
                          np.sum(OV_mass_nosat_cyl_Tcut[k][bool_in_s])))
                        OVI_total_nosat[j].append((np.sum(OVI_mass_nosat_cyl_Tcut[k][bool_out_s]) + \
                          np.sum(OVI_mass_nosat_cyl_Tcut[k][bool_in_s])))
                        OVII_total_nosat[j].append((np.sum(OVII_mass_nosat_cyl_Tcut[k][bool_out_s]) + \
                          np.sum(OVII_mass_nosat_cyl_Tcut[k][bool_in_s])))
                        OVIII_total_nosat[j].append((np.sum(OVIII_mass_nosat_cyl_Tcut[k][bool_out_s]) + \
                          np.sum(OVIII_mass_nosat_cyl_Tcut[k][bool_in_s])))
                        OIX_total_nosat[j].append((np.sum(OIX_mass_nosat_cyl_Tcut[k][bool_out_s]) + \
                          np.sum(OIX_mass_nosat_cyl_Tcut[k][bool_in_s])))
                if (j==1):
                    if ('mass' in flux_types):
                        mass_total_nosat[j].append(np.sum(mass_nosat_cyl_Tcut[k][bool_in_s]))
                        metals_total_nosat[j].append(np.sum(metal_mass_nosat_cyl_Tcut[k][bool_in_s]))
                    if ('energy' in flux_types):
                        kinetic_energy_total_nosat[j].append(np.sum(kinetic_energy_nosat_cyl_Tcut[k][bool_in_s]))
                        thermal_energy_total_nosat[j].append(np.sum(thermal_energy_nosat_cyl_Tcut[k][bool_in_s]))
                        potential_energy_total_nosat[j].append(np.sum(potential_energy_nosat_cyl_Tcut[k][bool_in_s]))
                        total_energy_total_nosat[j].append(np.sum(total_energy_nosat_cyl_Tcut[k][bool_in_s]))
                    if ('entropy' in flux_types):
                        entropy_total_nosat[j].append(np.sum(entropy_nosat_cyl_Tcut[k][bool_in_s]))
                    if ('O_ion_mass' in flux_types):
                        O_total_nosat[j].append(np.sum(O_mass_nosat_cyl_Tcut[k][bool_in_s]))
                        OI_total_nosat[j].append(np.sum(OI_mass_nosat_cyl_Tcut[k][bool_in_s]))
                        OII_total_nosat[j].append(np.sum(OII_mass_nosat_cyl_Tcut[k][bool_in_s]))
                        OIII_total_nosat[j].append(np.sum(OIII_mass_nosat_cyl_Tcut[k][bool_in_s]))
                        OIV_total_nosat[j].append(np.sum(OIV_mass_nosat_cyl_Tcut[k][bool_in_s]))
                        OV_total_nosat[j].append(np.sum(OV_mass_nosat_cyl_Tcut[k][bool_in_s]))
                        OVI_total_nosat[j].append(np.sum(OVI_mass_nosat_cyl_Tcut[k][bool_in_s]))
                        OVII_total_nosat[j].append(np.sum(OVII_mass_nosat_cyl_Tcut[k][bool_in_s]))
                        OVIII_total_nosat[j].append(np.sum(OVIII_mass_nosat_cyl_Tcut[k][bool_in_s]))
                        OIX_total_nosat[j].append(np.sum(OIX_mass_nosat_cyl_Tcut[k][bool_in_s]))
                if (j==2):
                    if ('mass' in flux_types):
                        mass_total_nosat[j].append(np.sum(mass_nosat_cyl_Tcut[k][bool_out_s]))
                        metals_total_nosat[j].append(np.sum(metal_mass_nosat_cyl_Tcut[k][bool_out_s]))
                    if ('energy' in flux_types):
                        kinetic_energy_total_nosat[j].append(np.sum(kinetic_energy_nosat_cyl_Tcut[k][bool_out_s]))
                        thermal_energy_total_nosat[j].append(np.sum(thermal_energy_nosat_cyl_Tcut[k][bool_out_s]))
                        potential_energy_total_nosat[j].append(np.sum(potential_energy_nosat_cyl_Tcut[k][bool_out_s]))
                        total_energy_total_nosat[j].append(np.sum(total_energy_nosat_cyl_Tcut[k][bool_out_s]))
                    if ('entropy' in flux_types):
                        entropy_total_nosat[j].append(np.sum(entropy_nosat_cyl_Tcut[k][bool_out_s]))
                    if ('O_ion_mass' in flux_types):
                        O_total_nosat[j].append(np.sum(O_mass_nosat_cyl_Tcut[k][bool_out_s]))
                        OI_total_nosat[j].append(np.sum(OI_mass_nosat_cyl_Tcut[k][bool_out_s]))
                        OII_total_nosat[j].append(np.sum(OII_mass_nosat_cyl_Tcut[k][bool_out_s]))
                        OIII_total_nosat[j].append(np.sum(OIII_mass_nosat_cyl_Tcut[k][bool_out_s]))
                        OIV_total_nosat[j].append(np.sum(OIV_mass_nosat_cyl_Tcut[k][bool_out_s]))
                        OV_total_nosat[j].append(np.sum(OV_mass_nosat_cyl_Tcut[k][bool_out_s]))
                        OVI_total_nosat[j].append(np.sum(OVI_mass_nosat_cyl_Tcut[k][bool_out_s]))
                        OVII_total_nosat[j].append(np.sum(OVII_mass_nosat_cyl_Tcut[k][bool_out_s]))
                        OVIII_total_nosat[j].append(np.sum(OVIII_mass_nosat_cyl_Tcut[k][bool_out_s]))
                        OIX_total_nosat[j].append(np.sum(OIX_mass_nosat_cyl_Tcut[k][bool_out_s]))

        # Add everything to the tables
        new_row = [zsnap, inner_surface, outer_surface]
        if ('mass' in flux_types):
            new_row += [mass_total_nosat[0][0], metals_total_nosat[0][0], \
            mass_total_nosat[1][0], mass_total_nosat[2][0], metals_total_nosat[1][0], metals_total_nosat[2][0], \
            mass_total_nosat[0][1], mass_total_nosat[1][1], mass_total_nosat[2][1], \
            mass_total_nosat[0][2], mass_total_nosat[1][2], mass_total_nosat[2][2], \
            mass_total_nosat[0][3], mass_total_nosat[1][3], mass_total_nosat[2][3], \
            mass_total_nosat[0][4], mass_total_nosat[1][4], mass_total_nosat[2][4], \
            metals_total_nosat[0][1], metals_total_nosat[1][1], metals_total_nosat[2][1], \
            metals_total_nosat[0][2], metals_total_nosat[1][2], metals_total_nosat[2][2], \
            metals_total_nosat[0][3], metals_total_nosat[1][3], metals_total_nosat[2][3], \
            metals_total_nosat[0][4], metals_total_nosat[1][4], metals_total_nosat[2][4]]
        if ('energy' in flux_types):
            new_row += [kinetic_energy_total_nosat[0][0], thermal_energy_total_nosat[0][0], \
            potential_energy_total_nosat[0][0], total_energy_total_nosat[0][0], \
            kinetic_energy_total_nosat[1][0], kinetic_energy_total_nosat[2][0], \
            thermal_energy_total_nosat[1][0], thermal_energy_total_nosat[2][0], \
            potential_energy_total_nosat[1][0], potential_energy_total_nosat[2][0], \
            total_energy_total_nosat[1][0], total_energy_total_nosat[2][0], \
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
            total_energy_total_nosat[0][1], total_energy_total_nosat[1][1], total_energy_total_nosat[2][1], \
            total_energy_total_nosat[0][2], total_energy_total_nosat[1][2], total_energy_total_nosat[2][2], \
            total_energy_total_nosat[0][3], total_energy_total_nosat[1][3], total_energy_total_nosat[2][3], \
            total_energy_total_nosat[0][4], total_energy_total_nosat[1][4], total_energy_total_nosat[2][4]]
        if ('entropy' in flux_types):
            new_row += [entropy_total_nosat[0][0], \
            entropy_total_nosat[1][0], entropy_total_nosat[2][0], \
            entropy_total_nosat[0][1], entropy_total_nosat[1][1], entropy_total_nosat[2][1], \
            entropy_total_nosat[0][2], entropy_total_nosat[1][2], entropy_total_nosat[2][2], \
            entropy_total_nosat[0][3], entropy_total_nosat[1][3], entropy_total_nosat[2][3], \
            entropy_total_nosat[0][4], entropy_total_nosat[1][4], entropy_total_nosat[2][4]]
        if ('O_ion_mass' in flux_types):
            new_row += [O_total_nosat[0][0], O_total_nosat[1][0], O_total_nosat[2][0], \
            O_total_nosat[0][1], O_total_nosat[1][1], O_total_nosat[2][1], \
            O_total_nosat[0][2], O_total_nosat[1][2], O_total_nosat[2][2], \
            O_total_nosat[0][3], O_total_nosat[1][3], O_total_nosat[2][3], \
            O_total_nosat[0][4], O_total_nosat[1][4], O_total_nosat[2][4], \
            OI_total_nosat[0][0], OI_total_nosat[1][0], OI_total_nosat[2][0], \
            OI_total_nosat[0][1], OI_total_nosat[1][1], OI_total_nosat[2][1], \
            OI_total_nosat[0][2], OI_total_nosat[1][2], OI_total_nosat[2][2], \
            OI_total_nosat[0][3], OI_total_nosat[1][3], OI_total_nosat[2][3], \
            OI_total_nosat[0][4], OI_total_nosat[1][4], OI_total_nosat[2][4], \
            OII_total_nosat[0][0], OII_total_nosat[1][0], OII_total_nosat[2][0], \
            OII_total_nosat[0][1], OII_total_nosat[1][1], OII_total_nosat[2][1], \
            OII_total_nosat[0][2], OII_total_nosat[1][2], OII_total_nosat[2][2], \
            OII_total_nosat[0][3], OII_total_nosat[1][3], OII_total_nosat[2][3], \
            OII_total_nosat[0][4], OII_total_nosat[1][4], OII_total_nosat[2][4], \
            OIII_total_nosat[0][0], OIII_total_nosat[1][0], OIII_total_nosat[2][0], \
            OIII_total_nosat[0][1], OIII_total_nosat[1][1], OIII_total_nosat[2][1], \
            OIII_total_nosat[0][2], OIII_total_nosat[1][2], OIII_total_nosat[2][2], \
            OIII_total_nosat[0][3], OIII_total_nosat[1][3], OIII_total_nosat[2][3], \
            OIII_total_nosat[0][4], OIII_total_nosat[1][4], OIII_total_nosat[2][4], \
            OIV_total_nosat[0][0], OIV_total_nosat[1][0], OIV_total_nosat[2][0], \
            OIV_total_nosat[0][1], OIV_total_nosat[1][1], OIV_total_nosat[2][1], \
            OIV_total_nosat[0][2], OIV_total_nosat[1][2], OIV_total_nosat[2][2], \
            OIV_total_nosat[0][3], OIV_total_nosat[1][3], OIV_total_nosat[2][3], \
            OIV_total_nosat[0][4], OIV_total_nosat[1][4], OIV_total_nosat[2][4], \
            OV_total_nosat[0][0], OV_total_nosat[1][0], OV_total_nosat[2][0], \
            OV_total_nosat[0][1], OV_total_nosat[1][1], OV_total_nosat[2][1], \
            OV_total_nosat[0][2], OV_total_nosat[1][2], OV_total_nosat[2][2], \
            OV_total_nosat[0][3], OV_total_nosat[1][3], OV_total_nosat[2][3], \
            OV_total_nosat[0][4], OV_total_nosat[1][4], OV_total_nosat[2][4], \
            OVI_total_nosat[0][0], OVI_total_nosat[1][0], OVI_total_nosat[2][0], \
            OVI_total_nosat[0][1], OVI_total_nosat[1][1], OVI_total_nosat[2][1], \
            OVI_total_nosat[0][2], OVI_total_nosat[1][2], OVI_total_nosat[2][2], \
            OVI_total_nosat[0][3], OVI_total_nosat[1][3], OVI_total_nosat[2][3], \
            OVI_total_nosat[0][4], OVI_total_nosat[1][4], OVI_total_nosat[2][4], \
            OVII_total_nosat[0][0], OVII_total_nosat[1][0], OVII_total_nosat[2][0], \
            OVII_total_nosat[0][1], OVII_total_nosat[1][1], OVII_total_nosat[2][1], \
            OVII_total_nosat[0][2], OVII_total_nosat[1][2], OVII_total_nosat[2][2], \
            OVII_total_nosat[0][3], OVII_total_nosat[1][3], OVII_total_nosat[2][3], \
            OVII_total_nosat[0][4], OVII_total_nosat[1][4], OVII_total_nosat[2][4], \
            OVIII_total_nosat[0][0], OVIII_total_nosat[1][0], OVIII_total_nosat[2][0], \
            OVIII_total_nosat[0][1], OVIII_total_nosat[1][1], OVIII_total_nosat[2][1], \
            OVIII_total_nosat[0][2], OVIII_total_nosat[1][2], OVIII_total_nosat[2][2], \
            OVIII_total_nosat[0][3], OVIII_total_nosat[1][3], OVIII_total_nosat[2][3], \
            OVIII_total_nosat[0][4], OVIII_total_nosat[1][4], OVIII_total_nosat[2][4], \
            OIX_total_nosat[0][0], OIX_total_nosat[1][0], OIX_total_nosat[2][0], \
            OIX_total_nosat[0][1], OIX_total_nosat[1][1], OIX_total_nosat[2][1], \
            OIX_total_nosat[0][2], OIX_total_nosat[1][2], OIX_total_nosat[2][2], \
            OIX_total_nosat[0][3], OIX_total_nosat[1][3], OIX_total_nosat[2][3], \
            OIX_total_nosat[0][4], OIX_total_nosat[1][4], OIX_total_nosat[2][4]]
        totals.add_row(new_row)

    totals = set_table_units(totals)

    fluxtype_filename = ''
    if ('mass' in flux_types):
        fluxtype_filename += '_mass'
    if ('energy' in flux_types):
        fluxtype_filename += '_energy'
    if ('entropy' in flux_types):
        fluxtype_filename += '_entropy'
    if ('O_ion_mass' in flux_types):
        fluxtype_filename += '_Oions'

    # Save to file
    if (sat_radius!=0.):
        totals.write(tablename + '_nosat_cylinder_' + cyl_filename + fluxtype_filename + '.hdf5', path='all_data', serialize_meta=True, overwrite=True)
    else:
        totals.write(tablename + '_cylinder_' + cyl_filename + fluxtype_filename + '.hdf5', path='all_data', serialize_meta=True, overwrite=True)

    return "Totals have been calculated for snapshot " + snap + "!"

def load_and_calculate(system, foggie_dir, run_dir, track, halo_c_v_name, snap, tablename, surface_args, flux_types, sat_dir, sat_radius):
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
    if ((surface_args[0]=='frustum') or (surface_args[0]=='cylinder')) and (surface_args[1]=='disk minor axis'):
        ds, refine_box = foggie_load(snap_name, track, halo_c_v_name=halo_c_v_name, disk_relative=True)
    else:
        ds, refine_box = foggie_load(snap_name, track, halo_c_v_name=halo_c_v_name, do_filter_particles=False)
    refine_width_kpc = ds.quan(ds.refine_width, 'kpc')
    zsnap = ds.get_parameter('CosmologyCurrentRedshift')

    # Specify the file where the list of satellites is saved
    if (sat_radius!=0.):
        sat_file = sat_dir + 'satellites.hdf5'
        sat = Table.read(sat_file, path='all_data')

    # Do the actual calculation
    if (surface_args[0]=='sphere'):
        if (sat_radius!=0.):
            message = calc_totals_sphere(ds, snap, zsnap, refine_width_kpc, tablename, surface_args, \
              flux_types, sat=sat, sat_radius=sat_radius)
        else:
            message = calc_totals_sphere(ds, snap, zsnap, refine_width_kpc, tablename, surface_args, \
              flux_types)
    if (surface_args[0]=='frustum'):
        if (sat_radius!=0.):
            message = calc_totals_frustum(ds, snap, zsnap, refine_width_kpc, tablename, surface_args, \
              flux_types, sat=sat, sat_radius=sat_radius)
        else:
            message = calc_totals_frustum(ds, snap, zsnap, refine_width_kpc, tablename, surface_args, \
              flux_types)
    if (surface_args[0]=='cylinder'):
        if (sat_radius!=0.):
            message = calc_totals_cylinder(ds, snap, zsnap, refine_width_kpc, tablename, surface_args, \
              flux_types, sat=sat, sat_radius=sat_radius)
        else:
            message = calc_totals_cylinder(ds, snap, zsnap, refine_width_kpc, tablename, surface_args, \
              flux_types)
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

    # Set directory for output location, making it if necessary
    prefix = output_dir + 'totals_halo_00' + args.halo + '/' + args.run + '/'
    if not (os.path.exists(prefix)): os.system('mkdir -p ' + prefix)

    print('foggie_dir: ', foggie_dir)
    halo_c_v_name = code_path + 'halo_infos/00' + args.halo + '/' + args.run + '/halo_c_v'

    # Specify where satellite files are saved
    if (args.remove_sats):
        sat_dir = code_path + 'halo_infos/00' + args.halo + '/' + args.run + '/'
        sat_radius = args.sat_radius
    else:
        sat_dir = 'sat_dir'
        sat_radius = 0.

    try:
        surface_args = ast.literal_eval(args.surface)
    except ValueError:
        sys.exit("Something's wrong with your surface arguments. Make sure to include both the outer " + \
        "quotes and the inner quotes around the surface type, like so:\n" + \
        '"[\'sphere\', 0.05, 2., 200.]"')
    if (surface_args[0]=='sphere'):
        print('Sphere arguments: inner_radius - %.3f outer_radius - %.3f num_radius - %d' % \
          (surface_args[1], surface_args[2], surface_args[3]))
    elif (surface_args[0]=='frustum'):
        if (surface_args[1]=='x'):
            axis = 'x'
            flip = False
        elif (surface_args[1]=='y'):
            axis = 'y'
            flip = False
        elif (surface_args[1]=='z'):
            axis = 'z'
            flip = False
        elif (surface_args[1]=='minor'):
            axis = 'disk minor axis'
            flip = False
        elif (surface_args[1]=='-x'):
            axis = 'x'
            flip = True
        elif (surface_args[1]=='-y'):
            axis = 'y'
            flip = True
        elif (surface_args[1]=='-z'):
            axis = 'z'
            flip = True
        elif (surface_args[1]=='-minor'):
            axis = 'disk minor axis'
            flip = True
        elif (type(surface_args[1])==tuple) or (type(surface_args[1])==list):
            axis = surface_args[1]
            flip = False
        else: sys.exit("I don't understand what axis you want.")
        surface_args = [surface_args[0], axis, flip, surface_args[2], surface_args[3], surface_args[4], surface_args[5]]
        if (flip):
            print('Frustum arguments: axis - flipped %s inner_radius - %.3f outer_radius - %.3f num_radius - %d opening_angle - %d' % \
              (axis, surface_args[3], surface_args[4], surface_args[5], surface_args[6]))
        else:
            print('Frustum arguments: axis - %s inner_radius - %.3f outer_radius - %.3f num_radius - %d opening_angle - %d' % \
              (str(axis), surface_args[3], surface_args[4], surface_args[5], surface_args[6]))
    elif (surface_args[0]=='cylinder'):
        if (surface_args[1]=='x'):
            axis = 'x'
            flip = False
        elif (surface_args[1]=='y'):
            axis = 'y'
            flip = False
        elif (surface_args[1]=='z'):
            axis = 'z'
            flip = False
        elif (surface_args[1]=='minor'):
            axis = 'disk minor axis'
            flip = False
        elif (surface_args[1]=='-x'):
            axis = 'x'
            flip = True
        elif (surface_args[1]=='-y'):
            axis = 'y'
            flip = True
        elif (surface_args[1]=='-z'):
            axis = 'z'
            flip = True
        elif (surface_args[1]=='-minor'):
            axis = 'disk minor axis'
            flip = True
        elif (type(surface_args[1])==tuple) or (type(surface_args[1])==list):
            axis = surface_args[1]
            flip = False
        else: sys.exit("I don't understand what axis you want.")
        if (surface_args[5]!='height') and (surface_args[5]!='radius'):
            sys.exit("I don't understand which way you want to calculate fluxes. Specify 'height' or 'radius'.")
        surface_args = [surface_args[0], axis, flip, surface_args[2], surface_args[3], surface_args[4], surface_args[5], surface_args[6]]
        if (flip):
            print('Cylinder arguments: axis - flipped %s bottom_edge - %.3f top_edge - %.3f radius - %.3f step_direction - %s num_steps - %d' % \
              (axis, surface_args[3], surface_args[4], surface_args[5], surface_args[6], surface_args[7]))
        else:
            print('Cylinder arguments: axis - %s bottom_edge - %.3f top_edge - %.3f radius - %.3f step_direction - %s num_steps - %d' % \
              (str(axis), surface_args[3], surface_args[4], surface_args[5], surface_args[6], surface_args[7]))
    else:
        sys.exit("That surface has not been implemented. Ask Cassi to add it.")

    if (args.kpc):
        print('Surface arguments are in units of kpc.')
        surface_args.append(True)
    else:
        print('Surface arguments are fractions of refine_width.')
        surface_args.append(False)

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
                for j in range(int(first), int(last)+1, args.output_step):
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
        for i in range(int(first), int(last)+1, args.output_step):
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

    # Build flux type list
    if (',' in args.flux_type):
        flux_types = args.flux_type.split(',')
    else:
        flux_types = [args.flux_type]
    for i in range(len(flux_types)):
        if (flux_types[i]!='mass') and (flux_types[i]!='energy') and (flux_types[i]!='entropy') and (flux_types[i]!='O_ion_mass'):
            print('The flux type   %s   has not been implemented. Ask Cassi to add it.' % (flux_types[i]))
            sys.exit()

    # Loop over outputs, for either single-processor or parallel processor computing
    if (args.nproc==1):
        for i in range(len(outs)):
            snap = outs[i]
            # Make the output table name for this snapshot
            tablename = prefix + snap + '_totals'
            # Do the actual calculation
            load_and_calculate(args.system, foggie_dir, run_dir, trackname, halo_c_v_name, snap, \
            tablename, surface_args, flux_types, sat_dir, sat_radius)
    else:
        # Split into a number of groupings equal to the number of processors
        # and run one process per processor
        for i in range(len(outs)//args.nproc):
            threads = []
            for j in range(args.nproc):
                snap = outs[args.nproc*i+j]
                tablename = prefix + snap + '_totals'
                threads.append(multi.Process(target=load_and_calculate, \
			       args=(args.system, foggie_dir, run_dir, trackname, halo_c_v_name, snap, \
                   tablename, surface_args, flux_types, sat_dir, sat_radius)))
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
			   args=(args.system, foggie_dir, run_dir, trackname, halo_c_v_name, snap, \
               tablename, surface_args, flux_types, sat_dir, sat_radius)))
        for t in threads:
            t.start()
        for t in threads:
            t.join()

    print(str(datetime.datetime.now()))
    print("All snapshots finished!")
