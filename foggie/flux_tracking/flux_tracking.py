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
                        help='If using system cassiopeia: Are the simulation files stored locally? Default is no')
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
                        help='What surface type for computing the flux? Default is sphere' + \
                        ' and the other options are "frustum" or "cylinder".\nNote that all surfaces will be centered on halo center.\n' + \
                        'To specify the shape, size, and orientation of the surface you want, ' + \
                        'input a list as follows (don\'t forget the outer quotes, and put the shape in a different quote type!):\n' + \
                        'If you want a sphere, give:\n' + \
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
                        help='Do you want to give inner_radius and outer_radius (sphere, frustum) or bottom_edge and top_edge (cylinder) in the surface arguments ' + \
                        'in kpc rather than the default of fraction of refine_width? Default is no.\n' + \
                        'Note that if you want to track fluxes over time, using kpc instead of fractions ' + \
                        'of refine_width will be inaccurate because refine_width is comoving and kpc are not.')
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

    table_units = {'redshift':None,'radius':'kpc','inner_radius':'kpc','outer_radius':'kpc', \
             'height':'kpc','bottom_edge':'kpc','top_edge':'kpc', \
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
             'net_total_energy_flux':'erg/yr', 'net_entropy_flux':'cm**2*keV/yr', \
             'kinetic_energy_flux_in':'erg/yr', 'kinetic_energy_flux_out':'erg/yr', \
             'thermal_energy_flux_in':'erg/yr', 'thermal_energy_flux_out':'erg/yr', \
             'potential_energy_flux_in':'erg/yr', 'potential_energy_flux_out':'erg/yr', \
             'radiative_energy_flux_in':'erg/yr', 'radiative_energy_flux_out':'erg/yr', \
             'total_energy_flux_in':'erg/yr', 'total_energy_flux_out':'erg/yr', \
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
             'net_cold_total_energy_flux':'erg/yr', 'cold_total_energy_flux_in':'erg/yr', 'cold_total_energy_flux_out':'erg/yr', \
             'net_cool_total_energy_flux':'erg/yr', 'cool_total_energy_flux_in':'erg/yr', 'cool_total_energy_flux_out':'erg/yr', \
             'net_warm_total_energy_flux':'erg/yr', 'warm_total_energy_flux_in':'erg/yr', 'warm_total_energy_flux_out':'erg/yr', \
             'net_hot_total_energy_flux':'erg/yr', 'hot_total_energy_flux_in':'erg/yr', 'hot_total_energy_flux_out':'erg/yr', \
             'net_cold_entropy_flux':'cm**2*keV/yr', 'cold_entropy_flux_in':'cm**2*keV/yr', 'cold_entropy_flux_out':'cm**2*keV/yr', \
             'net_cool_entropy_flux':'cm**2*keV/yr', 'cool_entropy_flux_in':'cm**2*keV/yr', 'cool_entropy_flux_out':'cm**2*keV/yr', \
             'net_warm_entropy_flux':'cm**2*keV/yr', 'warm_entropy_flux_in':'cm**2*keV/yr', 'warm_entropy_flux_out':'cm**2*keV/yr', \
             'net_hot_entropy_flux':'cm**2*keV/yr', 'hot_entropy_flux_in':'cm**2*keV/yr', 'hot_entropy_flux_out':'cm**2*keV/yr', \
             'net_O_flux':'Msun/yr', 'O_flux_in':'Msun/yr', 'O_flux_out':'Msun/yr', \
             'net_cold_O_flux':'Msun/yr', 'cold_O_flux_in':'Msun/yr', 'cold_O_flux_out':'Msun/yr', \
             'net_cool_O_flux':'Msun/yr', 'cool_O_flux_in':'Msun/yr', 'cool_O_flux_out':'Msun/yr', \
             'net_warm_O_flux':'Msun/yr', 'warm_O_flux_in':'Msun/yr', 'warm_O_flux_out':'Msun/yr', \
             'net_hot_O_flux':'Msun/yr', 'hot_O_flux_in':'Msun/yr', 'hot_O_flux_out':'Msun/yr', \
             'net_OI_flux':'Msun/yr', 'OI_flux_in':'Msun/yr', 'OI_flux_out':'Msun/yr', \
             'net_cold_OI_flux':'Msun/yr', 'cold_OI_flux_in':'Msun/yr', 'cold_OI_flux_out':'Msun/yr', \
             'net_cool_OI_flux':'Msun/yr', 'cool_OI_flux_in':'Msun/yr', 'cool_OI_flux_out':'Msun/yr', \
             'net_warm_OI_flux':'Msun/yr', 'warm_OI_flux_in':'Msun/yr', 'warm_OI_flux_out':'Msun/yr', \
             'net_hot_OI_flux':'Msun/yr', 'hot_OI_flux_in':'Msun/yr', 'hot_OI_flux_out':'Msun/yr', \
             'net_OII_flux':'Msun/yr', 'OII_flux_in':'Msun/yr', 'OII_flux_out':'Msun/yr', \
             'net_cold_OII_flux':'Msun/yr', 'cold_OII_flux_in':'Msun/yr', 'cold_OII_flux_out':'Msun/yr', \
             'net_cool_OII_flux':'Msun/yr', 'cool_OII_flux_in':'Msun/yr', 'cool_OII_flux_out':'Msun/yr', \
             'net_warm_OII_flux':'Msun/yr', 'warm_OII_flux_in':'Msun/yr', 'warm_OII_flux_out':'Msun/yr', \
             'net_hot_OII_flux':'Msun/yr', 'hot_OII_flux_in':'Msun/yr', 'hot_OII_flux_out':'Msun/yr', \
             'net_OIII_flux':'Msun/yr', 'OIII_flux_in':'Msun/yr', 'OIII_flux_out':'Msun/yr', \
             'net_cold_OIII_flux':'Msun/yr', 'cold_OIII_flux_in':'Msun/yr', 'cold_OIII_flux_out':'Msun/yr', \
             'net_cool_OIII_flux':'Msun/yr', 'cool_OIII_flux_in':'Msun/yr', 'cool_OIII_flux_out':'Msun/yr', \
             'net_warm_OIII_flux':'Msun/yr', 'warm_OIII_flux_in':'Msun/yr', 'warm_OIII_flux_out':'Msun/yr', \
             'net_hot_OIII_flux':'Msun/yr', 'hot_OIII_flux_in':'Msun/yr', 'hot_OIII_flux_out':'Msun/yr', \
             'net_OIV_flux':'Msun/yr', 'OIV_flux_in':'Msun/yr', 'OIV_flux_out':'Msun/yr', \
             'net_cold_OIV_flux':'Msun/yr', 'cold_OIV_flux_in':'Msun/yr', 'cold_OIV_flux_out':'Msun/yr', \
             'net_cool_OIV_flux':'Msun/yr', 'cool_OIV_flux_in':'Msun/yr', 'cool_OIV_flux_out':'Msun/yr', \
             'net_warm_OIV_flux':'Msun/yr', 'warm_OIV_flux_in':'Msun/yr', 'warm_OIV_flux_out':'Msun/yr', \
             'net_hot_OIV_flux':'Msun/yr', 'hot_OIV_flux_in':'Msun/yr', 'hot_OIV_flux_out':'Msun/yr', \
             'net_OV_flux':'Msun/yr', 'OV_flux_in':'Msun/yr', 'OV_flux_out':'Msun/yr', \
             'net_cold_OV_flux':'Msun/yr', 'cold_OV_flux_in':'Msun/yr', 'cold_OV_flux_out':'Msun/yr', \
             'net_cool_OV_flux':'Msun/yr', 'cool_OV_flux_in':'Msun/yr', 'cool_OV_flux_out':'Msun/yr', \
             'net_warm_OV_flux':'Msun/yr', 'warm_OV_flux_in':'Msun/yr', 'warm_OV_flux_out':'Msun/yr', \
             'net_hot_OV_flux':'Msun/yr', 'hot_OV_flux_in':'Msun/yr', 'hot_OV_flux_out':'Msun/yr', \
             'net_OVI_flux':'Msun/yr', 'OVI_flux_in':'Msun/yr', 'OVI_flux_out':'Msun/yr', \
             'net_cold_OVI_flux':'Msun/yr', 'cold_OVI_flux_in':'Msun/yr', 'cold_OVI_flux_out':'Msun/yr', \
             'net_cool_OVI_flux':'Msun/yr', 'cool_OVI_flux_in':'Msun/yr', 'cool_OVI_flux_out':'Msun/yr', \
             'net_warm_OVI_flux':'Msun/yr', 'warm_OVI_flux_in':'Msun/yr', 'warm_OVI_flux_out':'Msun/yr', \
             'net_hot_OVI_flux':'Msun/yr', 'hot_OVI_flux_in':'Msun/yr', 'hot_OVI_flux_out':'Msun/yr', \
             'net_OVII_flux':'Msun/yr', 'OVII_flux_in':'Msun/yr', 'OVII_flux_out':'Msun/yr', \
             'net_cold_OVII_flux':'Msun/yr', 'cold_OVII_flux_in':'Msun/yr', 'cold_OVII_flux_out':'Msun/yr', \
             'net_cool_OVII_flux':'Msun/yr', 'cool_OVII_flux_in':'Msun/yr', 'cool_OVII_flux_out':'Msun/yr', \
             'net_warm_OVII_flux':'Msun/yr', 'warm_OVII_flux_in':'Msun/yr', 'warm_OVII_flux_out':'Msun/yr', \
             'net_hot_OVII_flux':'Msun/yr', 'hot_OVII_flux_in':'Msun/yr', 'hot_OVII_flux_out':'Msun/yr', \
             'net_OVIII_flux':'Msun/yr', 'OVIII_flux_in':'Msun/yr', 'OVIII_flux_out':'Msun/yr', \
             'net_cold_OVIII_flux':'Msun/yr', 'cold_OVIII_flux_in':'Msun/yr', 'cold_OVIII_flux_out':'Msun/yr', \
             'net_cool_OVIII_flux':'Msun/yr', 'cool_OVIII_flux_in':'Msun/yr', 'cool_OVIII_flux_out':'Msun/yr', \
             'net_warm_OVIII_flux':'Msun/yr', 'warm_OVIII_flux_in':'Msun/yr', 'warm_OVIII_flux_out':'Msun/yr', \
             'net_hot_OVIII_flux':'Msun/yr', 'hot_OVIII_flux_in':'Msun/yr', 'hot_OVIII_flux_out':'Msun/yr', \
             'net_OIX_flux':'Msun/yr', 'OIX_flux_in':'Msun/yr', 'OIX_flux_out':'Msun/yr', \
             'net_cold_OIX_flux':'Msun/yr', 'cold_OIX_flux_in':'Msun/yr', 'cold_OIX_flux_out':'Msun/yr', \
             'net_cool_OIX_flux':'Msun/yr', 'cool_OIX_flux_in':'Msun/yr', 'cool_OIX_flux_out':'Msun/yr', \
             'net_warm_OIX_flux':'Msun/yr', 'warm_OIX_flux_in':'Msun/yr', 'warm_OIX_flux_out':'Msun/yr', \
             'net_hot_OIX_flux':'Msun/yr', 'hot_OIX_flux_in':'Msun/yr', 'hot_OIX_flux_out':'Msun/yr'}
    for key in table.keys():
        table[key].unit = table_units[key]
    return table

def calc_fluxes_sphere(ds, snap, zsnap, dt, refine_width_kpc, tablename, surface_args, flux_types, **kwargs):
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

    sat = kwargs.get('sat')
    sat_radius = kwargs.get('sat_radius', 0.)
    halo_center_kpc2 = kwargs.get('halo_center_kpc2', ds.halo_center_kpc)

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
    if (sat_radius!=0):
        names_list_sat = ('redshift', 'inner_radius', 'outer_radius')
        types_list_sat = ('f8', 'f8', 'f8')
    names_list = ('redshift', 'radius')
    types_list = ('f8', 'f8')
    if ('mass' in flux_types):
        new_names = ('net_mass_flux', 'net_metal_flux', \
        'mass_flux_in', 'mass_flux_out', 'metal_flux_in', 'metal_flux_out', \
        'net_cold_mass_flux', 'cold_mass_flux_in', 'cold_mass_flux_out', \
        'net_cool_mass_flux', 'cool_mass_flux_in', 'cool_mass_flux_out', \
        'net_warm_mass_flux', 'warm_mass_flux_in', 'warm_mass_flux_out', \
        'net_hot_mass_flux', 'hot_mass_flux_in', 'hot_mass_flux_out', \
        'net_cold_metal_flux', 'cold_metal_flux_in', 'cold_metal_flux_out', \
        'net_cool_metal_flux', 'cool_metal_flux_in', 'cool_metal_flux_out', \
        'net_warm_metal_flux', 'warm_metal_flux_in', 'warm_metal_flux_out', \
        'net_hot_metal_flux', 'hot_metal_flux_in', 'hot_metal_flux_out')
        new_types = ('f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', \
        'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', \
        'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8')
        names_list += new_names
        types_list += new_types
        if (sat_radius!=0):
            names_list_sat += new_names
            types_list_sat += new_types
    if ('energy' in flux_types):
        new_names = ('net_kinetic_energy_flux', 'net_thermal_energy_flux', \
        'net_potential_energy_flux', 'net_radiative_energy_flux', 'net_total_energy_flux',\
        'kinetic_energy_flux_in', 'kinetic_energy_flux_out', \
        'thermal_energy_flux_in', 'thermal_energy_flux_out', \
        'potential_energy_flux_in', 'potential_energy_flux_out', \
        'radiative_energy_flux_in', 'radiative_energy_flux_out', \
        'total_energy_flux_in', 'total_energy_flux_out', \
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
        'net_cold_total_energy_flux', 'cold_total_energy_flux_in', 'cold_total_energy_flux_out', \
        'net_cool_total_energy_flux', 'cool_total_energy_flux_in', 'cool_total_energy_flux_out', \
        'net_warm_total_energy_flux', 'warm_total_energy_flux_in', 'warm_total_energy_flux_out', \
        'net_hot_total_energy_flux', 'hot_total_energy_flux_in', 'hot_total_energy_flux_out')
        new_types = ('f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', \
        'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', \
        'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', \
        'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', \
        'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', \
        'f8', 'f8', 'f8', 'f8', 'f8', 'f8')
        names_list += new_names
        types_list += new_types
        if (sat_radius!=0):
            new_names_sat = ('net_kinetic_energy_flux', 'net_thermal_energy_flux', \
            'net_potential_energy_flux', 'net_total_energy_flux',\
            'kinetic_energy_flux_in', 'kinetic_energy_flux_out', \
            'thermal_energy_flux_in', 'thermal_energy_flux_out', \
            'potential_energy_flux_in', 'potential_energy_flux_out', \
            'total_energy_flux_in', 'total_energy_flux_out', \
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
            'net_cold_total_energy_flux', 'cold_total_energy_flux_in', 'cold_total_energy_flux_out', \
            'net_cool_total_energy_flux', 'cool_total_energy_flux_in', 'cool_total_energy_flux_out', \
            'net_warm_total_energy_flux', 'warm_total_energy_flux_in', 'warm_total_energy_flux_out', \
            'net_hot_total_energy_flux', 'hot_total_energy_flux_in', 'hot_total_energy_flux_out')
            new_types_sat = ('f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', \
            'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', \
            'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', \
            'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', \
            'f8', 'f8', 'f8', 'f8', 'f8', 'f8')
            names_list_sat += new_names_sat
            types_list_sat += new_types_sat
    if ('entropy' in flux_types):
        new_names = ('net_entropy_flux', 'entropy_flux_in', 'entropy_flux_out', \
        'net_cold_entropy_flux', 'cold_entropy_flux_in', 'cold_entropy_flux_out', \
        'net_cool_entropy_flux', 'cool_entropy_flux_in', 'cool_entropy_flux_out', \
        'net_warm_entropy_flux', 'warm_entropy_flux_in', 'warm_entropy_flux_out', \
        'net_hot_entropy_flux', 'hot_entropy_flux_in', 'hot_entropy_flux_out')
        new_types = ('f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', \
        'f8', 'f8', 'f8', 'f8', 'f8', 'f8')
        names_list += new_names
        types_list += new_types
        if (sat_radius!=0):
            names_list_sat += new_names
            types_list_sat += new_types
    if ('O_ion_mass' in flux_types):
        new_names = ('net_O_flux', 'O_flux_in', 'O_flux_out', \
        'net_cold_O_flux', 'cold_O_flux_in', 'cold_O_flux_out', \
        'net_cool_O_flux', 'cool_O_flux_in', 'cool_O_flux_out', \
        'net_warm_O_flux', 'warm_O_flux_in', 'warm_O_flux_out', \
        'net_hot_O_flux', 'hot_O_flux_in', 'hot_O_flux_out', \
        'net_OI_flux', 'OI_flux_in', 'OI_flux_out', \
        'net_cold_OI_flux', 'cold_OI_flux_in', 'cold_OI_flux_out', \
        'net_cool_OI_flux', 'cool_OI_flux_in', 'cool_OI_flux_out', \
        'net_warm_OI_flux', 'warm_OI_flux_in', 'warm_OI_flux_out', \
        'net_hot_OI_flux', 'hot_OI_flux_in', 'hot_OI_flux_out', \
        'net_OII_flux', 'OII_flux_in', 'OII_flux_out', \
        'net_cold_OII_flux', 'cold_OII_flux_in', 'cold_OII_flux_out', \
        'net_cool_OII_flux', 'cool_OII_flux_in', 'cool_OII_flux_out', \
        'net_warm_OII_flux', 'warm_OII_flux_in', 'warm_OII_flux_out', \
        'net_hot_OII_flux', 'hot_OII_flux_in', 'hot_OII_flux_out', \
        'net_OIII_flux', 'OIII_flux_in', 'OIII_flux_out', \
        'net_cold_OIII_flux', 'cold_OIII_flux_in', 'cold_OIII_flux_out', \
        'net_cool_OIII_flux', 'cool_OIII_flux_in', 'cool_OIII_flux_out', \
        'net_warm_OIII_flux', 'warm_OIII_flux_in', 'warm_OIII_flux_out', \
        'net_hot_OIII_flux', 'hot_OIII_flux_in', 'hot_OIII_flux_out', \
        'net_OIV_flux', 'OIV_flux_in', 'OIV_flux_out', \
        'net_cold_OIV_flux', 'cold_OIV_flux_in', 'cold_OIV_flux_out', \
        'net_cool_OIV_flux', 'cool_OIV_flux_in', 'cool_OIV_flux_out', \
        'net_warm_OIV_flux', 'warm_OIV_flux_in', 'warm_OIV_flux_out', \
        'net_hot_OIV_flux', 'hot_OIV_flux_in', 'hot_OIV_flux_out', \
        'net_OV_flux', 'OV_flux_in', 'OV_flux_out', \
        'net_cold_OV_flux', 'cold_OV_flux_in', 'cold_OV_flux_out', \
        'net_cool_OV_flux', 'cool_OV_flux_in', 'cool_OV_flux_out', \
        'net_warm_OV_flux', 'warm_OV_flux_in', 'warm_OV_flux_out', \
        'net_hot_OV_flux', 'hot_OV_flux_in', 'hot_OV_flux_out', \
        'net_OVI_flux', 'OVI_flux_in', 'OVI_flux_out', \
        'net_cold_OVI_flux', 'cold_OVI_flux_in', 'cold_OVI_flux_out', \
        'net_cool_OVI_flux', 'cool_OVI_flux_in', 'cool_OVI_flux_out', \
        'net_warm_OVI_flux', 'warm_OVI_flux_in', 'warm_OVI_flux_out', \
        'net_hot_OVI_flux', 'hot_OVI_flux_in', 'hot_OVI_flux_out', \
        'net_OVII_flux', 'OVII_flux_in', 'OVII_flux_out', \
        'net_cold_OVII_flux', 'cold_OVII_flux_in', 'cold_OVII_flux_out', \
        'net_cool_OVII_flux', 'cool_OVII_flux_in', 'cool_OVII_flux_out', \
        'net_warm_OVII_flux', 'warm_OVII_flux_in', 'warm_OVII_flux_out', \
        'net_hot_OVII_flux', 'hot_OVII_flux_in', 'hot_OVII_flux_out', \
        'net_OVIII_flux', 'OVIII_flux_in', 'OVIII_flux_out', \
        'net_cold_OVIII_flux', 'cold_OVIII_flux_in', 'cold_OVIII_flux_out', \
        'net_cool_OVIII_flux', 'cool_OVIII_flux_in', 'cool_OVIII_flux_out', \
        'net_warm_OVIII_flux', 'warm_OVIII_flux_in', 'warm_OVIII_flux_out', \
        'net_hot_OVIII_flux', 'hot_OVIII_flux_in', 'hot_OVIII_flux_out', \
        'net_OIX_flux', 'OIX_flux_in', 'OIX_flux_out', \
        'net_cold_OIX_flux', 'cold_OIX_flux_in', 'cold_OIX_flux_out', \
        'net_cool_OIX_flux', 'cool_OIX_flux_in', 'cool_OIX_flux_out', \
        'net_warm_OIX_flux', 'warm_OIX_flux_in', 'warm_OIX_flux_out', \
        'net_hot_OIX_flux', 'hot_OIX_flux_in', 'hot_OIX_flux_out')
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
        'f8', 'f8', 'f8', 'f8', 'f8', 'f8')
        names_list += new_names
        types_list += new_types
        if (sat_radius!=0):
            names_list_sat += new_names
            types_list_sat += new_types
    fluxes = Table(names=names_list, dtype=types_list)
    if (sat_radius!=0):
        fluxes_sat = Table(names=names_list_sat, dtype=types_list_sat)

    # Define the radii of the spherical shells where we want to calculate fluxes
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
    vx = sphere['gas','vx_corrected'].in_units('km/s').v
    vy = sphere['gas','vy_corrected'].in_units('km/s').v
    vz = sphere['gas','vz_corrected'].in_units('km/s').v
    rad_vel = sphere['gas','radial_velocity_corrected'].in_units('km/s').v
    new_x = x + vx*dt*(100./cmtopc*stoyr)
    new_y = y + vy*dt*(100./cmtopc*stoyr)
    new_z = z + vz*dt*(100./cmtopc*stoyr)
    new_radius = np.sqrt(new_x**2. + new_y**2. + new_z**2.)
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
        cooling_time = sphere['gas','cooling_time'].in_units('yr').v
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

    '''sphere = Table.read('/Users/clochhaas/Documents/Research/FOGGIE/Outputs/fields_halo_008508/nref11c_nref9f/DD1201_fields.hdf5', path='all_data')
    radius = sphere['radius_corrected']
    x = sphere['x']
    y = sphere['y']
    z = sphere['z']
    vx = sphere['vx_corrected']
    vy = sphere['vy_corrected']
    vz = sphere['vz_corrected']
    rad_vel = sphere['radial_velocity_corrected']
    new_x = x + vx*dt*(100./cmtopc*stoyr)
    new_y = y + vy*dt*(100./cmtopc*stoyr)
    new_z = z + vz*dt*(100./cmtopc*stoyr)
    new_radius = np.sqrt(new_x**2. + new_y**2. + new_z**2.)
    temperature = sphere['temperature']
    if ('mass' in flux_types):
        mass = sphere['cell_mass']
        metal_mass = sphere['metal_mass']
    if ('energy' in flux_types):
        kinetic_energy = sphere['kinetic_energy_corrected']
        thermal_energy = sphere['thermal_energy']
        potential_energy = sphere['Grav_Potential']
        cooling_time = sphere['cooling_time']
    if ('entropy' in flux_types):
        entropy = sphere['entropy']
    if ('O_ion_mass' in flux_types):
        OI_frac = sphere['O_p0_ion_fraction']
        OII_frac = sphere['O_p1_ion_fraction']
        OIII_frac = sphere['O_p2_ion_fraction']
        OIV_frac = sphere['O_p3_ion_fraction']
        OV_frac = sphere['O_p4_ion_fraction']
        OVI_frac = sphere['O_p5_ion_fraction']
        OVII_frac = sphere['O_p6_ion_fraction']
        OVIII_frac = sphere['O_p7_ion_fraction']
        OIX_frac = sphere['O_p8_ion_fraction']
        renorm = OI_frac + OII_frac + OIII_frac + OIV_frac + OV_frac + \
          OVI_frac + OVII_frac + OVIII_frac + OIX_frac
        OI_mass = sphere['O_p0_mass']/renorm
        OII_mass = sphere['O_p1_mass']/renorm
        OIII_mass = sphere['O_p2_mass']/renorm
        OIV_mass = sphere['O_p3_mass']/renorm
        OV_mass = sphere['O_p4_mass']/renorm
        OVI_mass = sphere['O_p5_mass']/renorm
        OVII_mass = sphere['O_p6_mass']/renorm
        OVIII_mass = sphere['O_p7_mass']/renorm
        OIX_mass = sphere['O_p8_mass']/renorm'''

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
        snap_type = snap[-6:-4]
        if (snap_type=='RD'):
            # If using an RD output, calculate satellite fluxes as if satellites don't move in snap2, i.e. snap1 = snap2
            snap2 = int(snap[-4:])
        else:
            snap2 = int(snap[-4:])+1
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
        if ('mass' in flux_types):
            mass_nosat = mass[bool_nosat]
            metal_mass_nosat = metal_mass[bool_nosat]
        if ('energy' in flux_types):
            kinetic_energy_nosat = kinetic_energy[bool_nosat]
            thermal_energy_nosat = thermal_energy[bool_nosat]
            potential_energy_nosat = potential_energy[bool_nosat]
            total_energy_nosat = total_energy[bool_nosat]
            cooling_time_nosat = cooling_time[bool_nosat]
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
        newradius_nosat = new_radius
        x_nosat = x
        y_nosat = y
        z_nosat = z
        newx_nosat = new_x
        newy_nosat = new_y
        newz_nosat = new_z
        vx_nosat = vx
        vy_nosat = vy
        vz_nosat = vz
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
    newradius_nosat_Tcut = []
    if ('mass' in flux_types):
        mass_nosat_Tcut = []
        metal_mass_nosat_Tcut = []
    if ('energy' in flux_types):
        kinetic_energy_nosat_Tcut = []
        thermal_energy_nosat_Tcut = []
        potential_energy_nosat_Tcut = []
        total_energy_nosat_Tcut = []
        cooling_time_nosat_Tcut = []
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
        newradius_nosat_Tcut.append(newradius_nosat[bool_temp])
        if ('mass' in flux_types):
            mass_nosat_Tcut.append(mass_nosat[bool_temp])
            metal_mass_nosat_Tcut.append(metal_mass_nosat[bool_temp])
        if ('energy' in flux_types):
            kinetic_energy_nosat_Tcut.append(kinetic_energy_nosat[bool_temp])
            thermal_energy_nosat_Tcut.append(thermal_energy_nosat[bool_temp])
            potential_energy_nosat_Tcut.append(potential_energy_nosat[bool_temp])
            total_energy_nosat_Tcut.append(total_energy_nosat[bool_temp])
            cooling_time_nosat_Tcut.append(cooling_time_nosat[bool_temp])
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

    if (sat_radius!=0):
        # Cut data to things that cross into or out of satellites
        # These are lists of lists where the index goes from 0 to 1 for [from satellites, to satellites]
        print('Cutting data to satellite fluxes')
        radius_sat = []
        newradius_sat = []
        temperature_sat = []
        if ('mass' in flux_types):
            mass_sat = []
            metal_mass_sat = []
        if ('energy' in flux_types):
            kinetic_energy_sat = []
            thermal_energy_sat = []
            potential_energy_sat = []
            total_energy_sat = []
        if ('entropy' in flux_types):
            entropy_sat = []
        if ('O_ion_mass' in flux_types):
            O_mass_sat = []
            OI_mass_sat = []
            OII_mass_sat = []
            OIII_mass_sat = []
            OIV_mass_sat = []
            OV_mass_sat = []
            OVI_mass_sat = []
            OVII_mass_sat = []
            OVIII_mass_sat = []
            OIX_mass_sat = []
        for j in range(2):
            if (j==0):
                radius_sat.append(radius[bool_fromsat])
                newradius_sat.append(new_radius[bool_fromsat])
                temperature_sat.append(temperature[bool_fromsat])
                if ('mass' in flux_types):
                    mass_sat.append(mass[bool_fromsat])
                    metal_mass_sat.append(metal_mass[bool_fromsat])
                if ('energy' in flux_types):
                    kinetic_energy_sat.append(kinetic_energy[bool_fromsat])
                    thermal_energy_sat.append(thermal_energy[bool_fromsat])
                    potential_energy_sat.append(potential_energy[bool_fromsat])
                    total_energy_sat.append(total_energy[bool_fromsat])
                if ('entropy' in flux_types):
                    entropy_sat.append(entropy[bool_fromsat])
                if ('O_ion_mass' in flux_types):
                    O_mass_sat.append(O_mass[bool_fromsat])
                    OI_mass_sat.append(OI_mass[bool_fromsat])
                    OII_mass_sat.append(OII_mass[bool_fromsat])
                    OIII_mass_sat.append(OIII_mass[bool_fromsat])
                    OIV_mass_sat.append(OIV_mass[bool_fromsat])
                    OV_mass_sat.append(OV_mass[bool_fromsat])
                    OVI_mass_sat.append(OVI_mass[bool_fromsat])
                    OVII_mass_sat.append(OVII_mass[bool_fromsat])
                    OVIII_mass_sat.append(OVIII_mass[bool_fromsat])
                    OIX_mass_sat.append(OIX_mass[bool_fromsat])
            if (j==1):
                radius_sat.append(radius[bool_tosat])
                newradius_sat.append(new_radius[bool_tosat])
                temperature_sat.append(temperature[bool_tosat])
                if ('mass' in flux_types):
                    mass_sat.append(mass[bool_tosat])
                    metal_mass_sat.append(metal_mass[bool_tosat])
                if ('energy' in flux_types):
                    kinetic_energy_sat.append(kinetic_energy[bool_tosat])
                    thermal_energy_sat.append(thermal_energy[bool_tosat])
                    potential_energy_sat.append(potential_energy[bool_tosat])
                    total_energy_sat.append(total_energy[bool_tosat])
                if ('entropy' in flux_types):
                    entropy_sat.append(entropy[bool_tosat])
                if ('O_ion_mass' in flux_types):
                    O_mass_sat.append(O_mass[bool_tosat])
                    OI_mass_sat.append(OI_mass[bool_tosat])
                    OII_mass_sat.append(OII_mass[bool_tosat])
                    OIII_mass_sat.append(OIII_mass[bool_tosat])
                    OIV_mass_sat.append(OIV_mass[bool_tosat])
                    OV_mass_sat.append(OV_mass[bool_tosat])
                    OVI_mass_sat.append(OVI_mass[bool_tosat])
                    OVII_mass_sat.append(OVII_mass[bool_tosat])
                    OVIII_mass_sat.append(OVIII_mass[bool_tosat])
                    OIX_mass_sat.append(OIX_mass[bool_tosat])

        # Cut stuff going into/out of satellites on temperature
        # These are nested lists where the first index goes from 0 to 1 for [from satellites, to satellites]
        # and the second index goes from 0 to 4 for [all gas, cold, cool, warm, hot]
        print('Cutting satellite fluxes on temperature')
        radius_sat_Tcut = []
        newradius_sat_Tcut = []
        if ('mass' in flux_types):
            mass_sat_Tcut = []
            metal_mass_sat_Tcut = []
        if ('energy' in flux_types):
            kinetic_energy_sat_Tcut = []
            thermal_energy_sat_Tcut = []
            potential_energy_sat_Tcut = []
            total_energy_sat_Tcut = []
        if ('entropy' in flux_types):
            entropy_sat_Tcut = []
        if ('O_ion_mass' in flux_types):
            O_mass_sat_Tcut = []
            OI_mass_sat_Tcut = []
            OII_mass_sat_Tcut = []
            OIII_mass_sat_Tcut = []
            OIV_mass_sat_Tcut = []
            OV_mass_sat_Tcut = []
            OVI_mass_sat_Tcut = []
            OVII_mass_sat_Tcut = []
            OVIII_mass_sat_Tcut = []
            OIX_mass_sat_Tcut = []
        for i in range(2):
            radius_sat_Tcut.append([])
            newradius_sat_Tcut.append([])
            if ('mass' in flux_types):
                mass_sat_Tcut.append([])
                metal_mass_sat_Tcut.append([])
            if ('energy' in flux_types):
                kinetic_energy_sat_Tcut.append([])
                thermal_energy_sat_Tcut.append([])
                potential_energy_sat_Tcut.append([])
                total_energy_sat_Tcut.append([])
            if ('entropy' in flux_types):
                entropy_sat_Tcut.append([])
            if ('O_ion_mass' in flux_types):
                O_mass_sat_Tcut.append([])
                OI_mass_sat_Tcut.append([])
                OII_mass_sat_Tcut.append([])
                OIII_mass_sat_Tcut.append([])
                OIV_mass_sat_Tcut.append([])
                OV_mass_sat_Tcut.append([])
                OVI_mass_sat_Tcut.append([])
                OVII_mass_sat_Tcut.append([])
                OVIII_mass_sat_Tcut.append([])
                OIX_mass_sat_Tcut.append([])
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
                if ('mass' in flux_types):
                    mass_sat_Tcut[i].append(mass_sat[i][bool_temp])
                    metal_mass_sat_Tcut[i].append(metal_mass_sat[i][bool_temp])
                if ('energy' in flux_types):
                    kinetic_energy_sat_Tcut[i].append(kinetic_energy_sat[i][bool_temp])
                    thermal_energy_sat_Tcut[i].append(thermal_energy_sat[i][bool_temp])
                    potential_energy_sat_Tcut[i].append(potential_energy_sat[i][bool_temp])
                    total_energy_sat_Tcut[i].append(total_energy_sat[i][bool_temp])
                if ('entropy' in flux_types):
                    entropy_sat_Tcut[i].append(entropy_sat[i][bool_temp])
                if ('O_ion_mass' in flux_types):
                    O_mass_sat_Tcut[i].append(O_mass_sat[i][bool_temp])
                    OI_mass_sat_Tcut[i].append(OI_mass_sat[i][bool_temp])
                    OII_mass_sat_Tcut[i].append(OII_mass_sat[i][bool_temp])
                    OIII_mass_sat_Tcut[i].append(OIII_mass_sat[i][bool_temp])
                    OIV_mass_sat_Tcut[i].append(OIV_mass_sat[i][bool_temp])
                    OV_mass_sat_Tcut[i].append(OV_mass_sat[i][bool_temp])
                    OVI_mass_sat_Tcut[i].append(OVI_mass_sat[i][bool_temp])
                    OVII_mass_sat_Tcut[i].append(OVII_mass_sat[i][bool_temp])
                    OVIII_mass_sat_Tcut[i].append(OVIII_mass_sat[i][bool_temp])
                    OIX_mass_sat_Tcut[i].append(OIX_mass_sat[i][bool_temp])

    # Loop over radii
    for i in range(len(radii)):
        inner_r = radii[i].v
        if (i < len(radii) - 1): outer_r = radii[i+1].v

        if (i%10==0): print("Computing radius " + str(i) + "/" + str(len(radii)) + \
                            " for snapshot " + snap)

        # Compute net, in, and out fluxes with satellites removed
        # These are nested lists where the first index goes from 0 to 2 for [net, in, out]
        # and the second index goes from 0 to 4 for [all, cold, cool, warm, hot]
        if ('mass' in flux_types):
            mass_flux_nosat = []
            metal_flux_nosat = []
        if ('energy' in flux_types):
            kinetic_energy_flux_nosat = []
            thermal_energy_flux_nosat = []
            potential_energy_flux_nosat = []
            total_energy_flux_nosat = []
        if ('entropy' in flux_types):
            entropy_flux_nosat = []
        if ('O_ion_mass' in flux_types):
            O_flux_nosat = []
            OI_flux_nosat = []
            OII_flux_nosat = []
            OIII_flux_nosat = []
            OIV_flux_nosat = []
            OV_flux_nosat = []
            OVI_flux_nosat = []
            OVII_flux_nosat = []
            OVIII_flux_nosat = []
            OIX_flux_nosat = []
        for j in range(3):
            if ('mass' in flux_types):
                mass_flux_nosat.append([])
                metal_flux_nosat.append([])
            if ('energy' in flux_types):
                kinetic_energy_flux_nosat.append([])
                thermal_energy_flux_nosat.append([])
                potential_energy_flux_nosat.append([])
                total_energy_flux_nosat.append([])
            if ('entropy' in flux_types):
                entropy_flux_nosat.append([])
            if ('O_ion_mass' in flux_types):
                O_flux_nosat.append([])
                OI_flux_nosat.append([])
                OII_flux_nosat.append([])
                OIII_flux_nosat.append([])
                OIV_flux_nosat.append([])
                OV_flux_nosat.append([])
                OVI_flux_nosat.append([])
                OVII_flux_nosat.append([])
                OVIII_flux_nosat.append([])
                OIX_flux_nosat.append([])
            for k in range(5):
                bool_in = (radius_nosat_Tcut[k] > inner_r) & (newradius_nosat_Tcut[k] < inner_r)
                bool_out = (radius_nosat_Tcut[k] < inner_r) & (newradius_nosat_Tcut[k] > inner_r)
                if (j==0):
                    if ('mass' in flux_types):
                        mass_flux_nosat[j].append((np.sum(mass_nosat_Tcut[k][bool_out]) - \
                          np.sum(mass_nosat_Tcut[k][bool_in]))/dt)
                        metal_flux_nosat[j].append((np.sum(metal_mass_nosat_Tcut[k][bool_out]) - \
                          np.sum(metal_mass_nosat_Tcut[k][bool_in]))/dt)
                    if ('energy' in flux_types):
                        kinetic_energy_flux_nosat[j].append((np.sum(kinetic_energy_nosat_Tcut[k][bool_out]) - \
                          np.sum(kinetic_energy_nosat_Tcut[k][bool_in]))/dt)
                        thermal_energy_flux_nosat[j].append((np.sum(thermal_energy_nosat_Tcut[k][bool_out]) - \
                          np.sum(thermal_energy_nosat_Tcut[k][bool_in]))/dt)
                        potential_energy_flux_nosat[j].append((np.sum(potential_energy_nosat_Tcut[k][bool_out]) - \
                          np.sum(potential_energy_nosat_Tcut[k][bool_in]))/dt)
                        total_energy_flux_nosat[j].append((np.sum(total_energy_nosat_Tcut[k][bool_out]) - \
                          np.sum(total_energy_nosat_Tcut[k][bool_in]))/dt)
                    if ('entropy' in flux_types):
                        entropy_flux_nosat[j].append((np.sum(entropy_nosat_Tcut[k][bool_out]) - \
                          np.sum(entropy_nosat_Tcut[k][bool_in]))/dt)
                    if ('O_ion_mass' in flux_types):
                        O_flux_nosat[j].append((np.sum(O_mass_nosat_Tcut[k][bool_out]) - \
                          np.sum(O_mass_nosat_Tcut[k][bool_in]))/dt)
                        OI_flux_nosat[j].append((np.sum(OI_mass_nosat_Tcut[k][bool_out]) - \
                          np.sum(OI_mass_nosat_Tcut[k][bool_in]))/dt)
                        OII_flux_nosat[j].append((np.sum(OII_mass_nosat_Tcut[k][bool_out]) - \
                          np.sum(OII_mass_nosat_Tcut[k][bool_in]))/dt)
                        OIII_flux_nosat[j].append((np.sum(OIII_mass_nosat_Tcut[k][bool_out]) - \
                          np.sum(OIII_mass_nosat_Tcut[k][bool_in]))/dt)
                        OIV_flux_nosat[j].append((np.sum(OIV_mass_nosat_Tcut[k][bool_out]) - \
                          np.sum(OIV_mass_nosat_Tcut[k][bool_in]))/dt)
                        OV_flux_nosat[j].append((np.sum(OV_mass_nosat_Tcut[k][bool_out]) - \
                          np.sum(OV_mass_nosat_Tcut[k][bool_in]))/dt)
                        OVI_flux_nosat[j].append((np.sum(OVI_mass_nosat_Tcut[k][bool_out]) - \
                          np.sum(OVI_mass_nosat_Tcut[k][bool_in]))/dt)
                        OVII_flux_nosat[j].append((np.sum(OVII_mass_nosat_Tcut[k][bool_out]) - \
                          np.sum(OVII_mass_nosat_Tcut[k][bool_in]))/dt)
                        OVIII_flux_nosat[j].append((np.sum(OVIII_mass_nosat_Tcut[k][bool_out]) - \
                          np.sum(OVIII_mass_nosat_Tcut[k][bool_in]))/dt)
                        OIX_flux_nosat[j].append((np.sum(OIX_mass_nosat_Tcut[k][bool_out]) - \
                          np.sum(OIX_mass_nosat_Tcut[k][bool_in]))/dt)
                if (j==1):
                    if ('mass' in flux_types):
                        mass_flux_nosat[j].append(-np.sum(mass_nosat_Tcut[k][bool_in])/dt)
                        metal_flux_nosat[j].append(-np.sum(metal_mass_nosat_Tcut[k][bool_in])/dt)
                    if ('energy' in flux_types):
                        kinetic_energy_flux_nosat[j].append(-np.sum(kinetic_energy_nosat_Tcut[k][bool_in])/dt)
                        thermal_energy_flux_nosat[j].append(-np.sum(thermal_energy_nosat_Tcut[k][bool_in])/dt)
                        potential_energy_flux_nosat[j].append(-np.sum(potential_energy_nosat_Tcut[k][bool_in])/dt)
                        total_energy_flux_nosat[j].append(-np.sum(total_energy_nosat_Tcut[k][bool_in])/dt)
                    if ('entropy' in flux_types):
                        entropy_flux_nosat[j].append(-np.sum(entropy_nosat_Tcut[k][bool_in])/dt)
                    if ('O_ion_mass' in flux_types):
                        O_flux_nosat[j].append(-np.sum(O_mass_nosat_Tcut[k][bool_in])/dt)
                        OI_flux_nosat[j].append(-np.sum(OI_mass_nosat_Tcut[k][bool_in])/dt)
                        OII_flux_nosat[j].append(-np.sum(OII_mass_nosat_Tcut[k][bool_in])/dt)
                        OIII_flux_nosat[j].append(-np.sum(OIII_mass_nosat_Tcut[k][bool_in])/dt)
                        OIV_flux_nosat[j].append(-np.sum(OIV_mass_nosat_Tcut[k][bool_in])/dt)
                        OV_flux_nosat[j].append(-np.sum(OV_mass_nosat_Tcut[k][bool_in])/dt)
                        OVI_flux_nosat[j].append(-np.sum(OVI_mass_nosat_Tcut[k][bool_in])/dt)
                        OVII_flux_nosat[j].append(-np.sum(OVII_mass_nosat_Tcut[k][bool_in])/dt)
                        OVIII_flux_nosat[j].append(-np.sum(OVIII_mass_nosat_Tcut[k][bool_in])/dt)
                        OIX_flux_nosat[j].append(-np.sum(OIX_mass_nosat_Tcut[k][bool_in])/dt)
                if (j==2):
                    if ('mass' in flux_types):
                        mass_flux_nosat[j].append(np.sum(mass_nosat_Tcut[k][bool_out])/dt)
                        metal_flux_nosat[j].append(np.sum(metal_mass_nosat_Tcut[k][bool_out])/dt)
                    if ('energy' in flux_types):
                        kinetic_energy_flux_nosat[j].append(np.sum(kinetic_energy_nosat_Tcut[k][bool_out])/dt)
                        thermal_energy_flux_nosat[j].append(np.sum(thermal_energy_nosat_Tcut[k][bool_out])/dt)
                        potential_energy_flux_nosat[j].append(np.sum(potential_energy_nosat_Tcut[k][bool_out])/dt)
                        total_energy_flux_nosat[j].append(np.sum(total_energy_nosat_Tcut[k][bool_out])/dt)
                    if ('entropy' in flux_types):
                        entropy_flux_nosat[j].append(np.sum(entropy_nosat_Tcut[k][bool_out])/dt)
                    if ('O_ion_mass' in flux_types):
                        O_flux_nosat[j].append(np.sum(O_mass_nosat_Tcut[k][bool_out])/dt)
                        OI_flux_nosat[j].append(np.sum(OI_mass_nosat_Tcut[k][bool_out])/dt)
                        OII_flux_nosat[j].append(np.sum(OII_mass_nosat_Tcut[k][bool_out])/dt)
                        OIII_flux_nosat[j].append(np.sum(OIII_mass_nosat_Tcut[k][bool_out])/dt)
                        OIV_flux_nosat[j].append(np.sum(OIV_mass_nosat_Tcut[k][bool_out])/dt)
                        OV_flux_nosat[j].append(np.sum(OV_mass_nosat_Tcut[k][bool_out])/dt)
                        OVI_flux_nosat[j].append(np.sum(OVI_mass_nosat_Tcut[k][bool_out])/dt)
                        OVII_flux_nosat[j].append(np.sum(OVII_mass_nosat_Tcut[k][bool_out])/dt)
                        OVIII_flux_nosat[j].append(np.sum(OVIII_mass_nosat_Tcut[k][bool_out])/dt)
                        OIX_flux_nosat[j].append(np.sum(OIX_mass_nosat_Tcut[k][bool_out])/dt)

        # Compute fluxes from and to satellites (and net) between inner_r and outer_r
        # These are nested lists where the first index goes from 0 to 2 for [net, from, to]
        # and the second index goes from 0 to 4 for [all, cold, cool, warm, hot]
        if (i < len(radii)-1):
            if (sat_radius!=0) and ('mass' in flux_types):
                mass_flux_sat = []
                metal_flux_sat = []
            if (sat_radius!=0) and ('energy' in flux_types):
                kinetic_energy_flux_sat = []
                thermal_energy_flux_sat = []
                potential_energy_flux_sat = []
                total_energy_flux_sat = []
            if (sat_radius!=0) and ('entropy' in flux_types):
                entropy_flux_sat = []
            if ('energy' in flux_types):
                radiative_energy_flux_nosat = []
            if (sat_radius!=0) and ('O_ion_mass' in flux_types):
                O_flux_sat = []
                OI_flux_sat = []
                OII_flux_sat = []
                OIII_flux_sat = []
                OIV_flux_sat = []
                OV_flux_sat = []
                OVI_flux_sat = []
                OVII_flux_sat = []
                OVIII_flux_sat = []
                OIX_flux_sat = []
            for j in range(3):
                if (sat_radius!=0) and ('mass' in flux_types):
                    mass_flux_sat.append([])
                    metal_flux_sat.append([])
                if (sat_radius!=0) and ('energy' in flux_types):
                    kinetic_energy_flux_sat.append([])
                    thermal_energy_flux_sat.append([])
                    potential_energy_flux_sat.append([])
                    total_energy_flux_sat.append([])
                if (sat_radius!=0) and ('entropy' in flux_types):
                    entropy_flux_sat.append([])
                if ('energy' in flux_types):
                    radiative_energy_flux_nosat.append([])
                if (sat_radius!=0) and ('O_ion_mass' in flux_types):
                    O_flux_sat.append([])
                    OI_flux_sat.append([])
                    OII_flux_sat.append([])
                    OIII_flux_sat.append([])
                    OIV_flux_sat.append([])
                    OV_flux_sat.append([])
                    OVI_flux_sat.append([])
                    OVII_flux_sat.append([])
                    OVIII_flux_sat.append([])
                    OIX_flux_sat.append([])
                for k in range(5):
                    if (sat_radius!=0):
                        bool_in = (newradius_sat_Tcut[0][k]>inner_r) & (newradius_sat_Tcut[0][k]<outer_r)
                        bool_out = (radius_sat_Tcut[1][k]>inner_r) & (radius_sat_Tcut[1][k]<outer_r)
                    if ('energy' in flux_types):
                        bool_shell = (radius_nosat_Tcut[k]>inner_r) & (radius_nosat_Tcut[k]<outer_r)
                    if (j==0):
                        if (sat_radius!=0) and ('mass' in flux_types):
                            mass_flux_sat[j].append((np.sum(mass_sat_Tcut[0][k][bool_in]) - \
                                                    np.sum(mass_sat_Tcut[1][k][bool_out]))/dt)
                            metal_flux_sat[j].append((np.sum(metal_mass_sat_Tcut[0][k][bool_in]) - \
                                                    np.sum(metal_mass_sat_Tcut[1][k][bool_out]))/dt)
                        if (sat_radius!=0) and ('energy' in flux_types):
                            kinetic_energy_flux_sat[j].append((np.sum(kinetic_energy_sat_Tcut[0][k][bool_in]) - \
                                                              np.sum(kinetic_energy_sat_Tcut[1][k][bool_out]))/dt)
                            thermal_energy_flux_sat[j].append((np.sum(thermal_energy_sat_Tcut[0][k][bool_in]) - \
                                                              np.sum(thermal_energy_sat_Tcut[1][k][bool_out]))/dt)
                            potential_energy_flux_sat[j].append((np.sum(potential_energy_sat_Tcut[0][k][bool_in]) - \
                                                                np.sum(potential_energy_sat_Tcut[1][k][bool_out]))/dt)
                            total_energy_flux_sat[j].append((np.sum(total_energy_sat_Tcut[0][k][bool_in]) - \
                                                                np.sum(total_energy_sat_Tcut[1][k][bool_out]))/dt)
                        if (sat_radius!=0) and ('entropy' in flux_types):
                            entropy_flux_sat[j].append((np.sum(entropy_sat_Tcut[0][k][bool_in]) - \
                                                        np.sum(entropy_sat_Tcut[1][k][bool_out]))/dt)
                        if ('energy' in flux_types):
                            radiative_energy_flux_nosat[j].append(np.sum(thermal_energy_nosat_Tcut[k][bool_shell] * \
                              mass_nosat_Tcut[k][bool_shell]*gtoMsun / cooling_time_nosat_Tcut[k][bool_shell]))
                        if (sat_radius!=0) and ('O_ion_mass' in flux_types):
                            O_flux_sat[j].append((np.sum(O_mass_sat_Tcut[0][k][bool_in]) - \
                              np.sum(O_mass_sat_Tcut[1][k][bool_out]))/dt)
                            OI_flux_sat[j].append((np.sum(OI_mass_sat_Tcut[0][k][bool_in]) - \
                              np.sum(OI_mass_sat_Tcut[1][k][bool_out]))/dt)
                            OII_flux_sat[j].append((np.sum(OII_mass_sat_Tcut[0][k][bool_in]) - \
                              np.sum(OII_mass_sat_Tcut[1][k][bool_out]))/dt)
                            OIII_flux_sat[j].append((np.sum(OIII_mass_sat_Tcut[0][k][bool_in]) - \
                              np.sum(OIII_mass_sat_Tcut[1][k][bool_out]))/dt)
                            OIV_flux_sat[j].append((np.sum(OIV_mass_sat_Tcut[0][k][bool_in]) - \
                              np.sum(OIV_mass_sat_Tcut[1][k][bool_out]))/dt)
                            OV_flux_sat[j].append((np.sum(OV_mass_sat_Tcut[0][k][bool_in]) - \
                              np.sum(OV_mass_sat_Tcut[1][k][bool_out]))/dt)
                            OVI_flux_sat[j].append((np.sum(OVI_mass_sat_Tcut[0][k][bool_in]) - \
                              np.sum(OVI_mass_sat_Tcut[1][k][bool_out]))/dt)
                            OVII_flux_sat[j].append((np.sum(OVII_mass_sat_Tcut[0][k][bool_in]) - \
                              np.sum(OVII_mass_sat_Tcut[1][k][bool_out]))/dt)
                            OVIII_flux_sat[j].append((np.sum(OVIII_mass_sat_Tcut[0][k][bool_in]) - \
                              np.sum(OVIII_mass_sat_Tcut[1][k][bool_out]))/dt)
                            OIX_flux_sat[j].append((np.sum(OIX_mass_sat_Tcut[0][k][bool_in]) - \
                              np.sum(OIX_mass_sat_Tcut[1][k][bool_out]))/dt)
                    if (j==1):
                        if (sat_radius!=0) and ('mass' in flux_types):
                            mass_flux_sat[j].append(np.sum(mass_sat_Tcut[0][k][bool_in])/dt)
                            metal_flux_sat[j].append(np.sum(metal_mass_sat_Tcut[0][k][bool_in])/dt)
                        if (sat_radius!=0) and ('energy' in flux_types):
                            kinetic_energy_flux_sat[j].append(np.sum(kinetic_energy_sat_Tcut[0][k][bool_in])/dt)
                            thermal_energy_flux_sat[j].append(np.sum(thermal_energy_sat_Tcut[0][k][bool_in])/dt)
                            potential_energy_flux_sat[j].append(np.sum(potential_energy_sat_Tcut[0][k][bool_in])/dt)
                            total_energy_flux_sat[j].append(np.sum(total_energy_sat_Tcut[0][k][bool_in])/dt)
                        if (sat_radius!=0) and ('entropy' in flux_types):
                            entropy_flux_sat[j].append(np.sum(entropy_sat_Tcut[0][k][bool_in])/dt)
                        if ('energy' in flux_types):
                            radiative_energy_flux_nosat[j].append(np.sum( \
                              thermal_energy_nosat_Tcut[k][bool_shell & (rad_vel_nosat_Tcut[k]<0.)] * \
                              mass_nosat_Tcut[k][bool_shell & (rad_vel_nosat_Tcut[k]<0.)]*gtoMsun / \
                              cooling_time_nosat_Tcut[k][bool_shell & (rad_vel_nosat_Tcut[k]<0.)]))
                        if (sat_radius!=0) and ('O_ion_mass' in flux_types):
                            O_flux_sat[j].append(np.sum(O_mass_sat_Tcut[0][k][bool_in])/dt)
                            OI_flux_sat[j].append(np.sum(OI_mass_sat_Tcut[0][k][bool_in])/dt)
                            OII_flux_sat[j].append(np.sum(OII_mass_sat_Tcut[0][k][bool_in])/dt)
                            OIII_flux_sat[j].append(np.sum(OIII_mass_sat_Tcut[0][k][bool_in])/dt)
                            OIV_flux_sat[j].append(np.sum(OIV_mass_sat_Tcut[0][k][bool_in])/dt)
                            OV_flux_sat[j].append(np.sum(OV_mass_sat_Tcut[0][k][bool_in])/dt)
                            OVI_flux_sat[j].append(np.sum(OVI_mass_sat_Tcut[0][k][bool_in])/dt)
                            OVII_flux_sat[j].append(np.sum(OVII_mass_sat_Tcut[0][k][bool_in])/dt)
                            OVIII_flux_sat[j].append(np.sum(OVIII_mass_sat_Tcut[0][k][bool_in])/dt)
                            OIX_flux_sat[j].append(np.sum(OIX_mass_sat_Tcut[0][k][bool_in])/dt)
                    if (j==2):
                        if (sat_radius!=0) and ('mass' in flux_types):
                            mass_flux_sat[j].append(-np.sum(mass_sat_Tcut[1][k][bool_out])/dt)
                            metal_flux_sat[j].append(-np.sum(metal_mass_sat_Tcut[1][k][bool_out])/dt)
                        if (sat_radius!=0) and ('energy' in flux_types):
                            kinetic_energy_flux_sat[j].append(-np.sum(kinetic_energy_sat_Tcut[1][k][bool_out])/dt)
                            thermal_energy_flux_sat[j].append(-np.sum(thermal_energy_sat_Tcut[1][k][bool_out])/dt)
                            potential_energy_flux_sat[j].append(-np.sum(potential_energy_sat_Tcut[1][k][bool_out])/dt)
                            total_energy_flux_sat[j].append(-np.sum(total_energy_sat_Tcut[1][k][bool_out])/dt)
                        if (sat_radius!=0) and ('entropy' in flux_types):
                            entropy_flux_sat[j].append(-np.sum(entropy_sat_Tcut[1][k][bool_out])/dt)
                        if ('energy' in flux_types):
                            radiative_energy_flux_nosat[j].append(np.sum( \
                              thermal_energy_nosat_Tcut[k][bool_shell & (rad_vel_nosat_Tcut[k]>0.)] * \
                              mass_nosat_Tcut[k][bool_shell & (rad_vel_nosat_Tcut[k]>0.)]*gtoMsun / \
                              cooling_time_nosat_Tcut[k][bool_shell & (rad_vel_nosat_Tcut[k]>0.)]))
                        if (sat_radius!=0) and ('O_ion_mass' in flux_types):
                            O_flux_sat[j].append(-np.sum(O_mass_sat_Tcut[1][k][bool_out])/dt)
                            OI_flux_sat[j].append(-np.sum(OI_mass_sat_Tcut[1][k][bool_out])/dt)
                            OII_flux_sat[j].append(-np.sum(OII_mass_sat_Tcut[1][k][bool_out])/dt)
                            OIII_flux_sat[j].append(-np.sum(OIII_mass_sat_Tcut[1][k][bool_out])/dt)
                            OIV_flux_sat[j].append(-np.sum(OIV_mass_sat_Tcut[1][k][bool_out])/dt)
                            OV_flux_sat[j].append(-np.sum(OV_mass_sat_Tcut[1][k][bool_out])/dt)
                            OVI_flux_sat[j].append(-np.sum(OVI_mass_sat_Tcut[1][k][bool_out])/dt)
                            OVII_flux_sat[j].append(-np.sum(OVII_mass_sat_Tcut[1][k][bool_out])/dt)
                            OVIII_flux_sat[j].append(-np.sum(OVIII_mass_sat_Tcut[1][k][bool_out])/dt)
                            OIX_flux_sat[j].append(-np.sum(OIX_mass_sat_Tcut[1][k][bool_out])/dt)

        # Add everything to the table
        new_row = [zsnap, inner_r]
        if ('mass' in flux_types):
            new_row += [mass_flux_nosat[0][0], metal_flux_nosat[0][0], \
            mass_flux_nosat[1][0], mass_flux_nosat[2][0], metal_flux_nosat[1][0], metal_flux_nosat[2][0], \
            mass_flux_nosat[0][1], mass_flux_nosat[1][1], mass_flux_nosat[2][1], \
            mass_flux_nosat[0][2], mass_flux_nosat[1][2], mass_flux_nosat[2][2], \
            mass_flux_nosat[0][3], mass_flux_nosat[1][3], mass_flux_nosat[2][3], \
            mass_flux_nosat[0][4], mass_flux_nosat[1][4], mass_flux_nosat[2][4], \
            metal_flux_nosat[0][1], metal_flux_nosat[1][1], metal_flux_nosat[2][1], \
            metal_flux_nosat[0][2], metal_flux_nosat[1][2], metal_flux_nosat[2][2], \
            metal_flux_nosat[0][3], metal_flux_nosat[1][3], metal_flux_nosat[2][3], \
            metal_flux_nosat[0][4], metal_flux_nosat[1][4], metal_flux_nosat[2][4]]
        if ('energy' in flux_types):
            new_row += [kinetic_energy_flux_nosat[0][0], thermal_energy_flux_nosat[0][0], \
            potential_energy_flux_nosat[0][0], radiative_energy_flux_nosat[0][0], \
            total_energy_flux_nosat[0][0], \
            kinetic_energy_flux_nosat[1][0], kinetic_energy_flux_nosat[2][0], \
            thermal_energy_flux_nosat[1][0], thermal_energy_flux_nosat[2][0], \
            potential_energy_flux_nosat[1][0], potential_energy_flux_nosat[2][0], \
            radiative_energy_flux_nosat[1][0], radiative_energy_flux_nosat[2][0], \
            total_energy_flux_nosat[1][0], total_energy_flux_nosat[2][0], \
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
            total_energy_flux_nosat[0][1], total_energy_flux_nosat[1][1], total_energy_flux_nosat[2][1], \
            total_energy_flux_nosat[0][2], total_energy_flux_nosat[1][2], total_energy_flux_nosat[2][2], \
            total_energy_flux_nosat[0][3], total_energy_flux_nosat[1][3], total_energy_flux_nosat[2][3], \
            total_energy_flux_nosat[0][4], total_energy_flux_nosat[1][4], total_energy_flux_nosat[2][4]]
        if ('entropy' in flux_types):
            new_row += [entropy_flux_nosat[0][0], entropy_flux_nosat[1][0], entropy_flux_nosat[2][0], \
                        entropy_flux_nosat[0][1], entropy_flux_nosat[1][1], entropy_flux_nosat[2][1], \
                        entropy_flux_nosat[0][2], entropy_flux_nosat[1][2], entropy_flux_nosat[2][2], \
                        entropy_flux_nosat[0][3], entropy_flux_nosat[1][3], entropy_flux_nosat[2][3], \
                        entropy_flux_nosat[0][4], entropy_flux_nosat[1][4], entropy_flux_nosat[2][4]]
        if ('O_ion_mass' in flux_types):
            new_row += [O_flux_nosat[0][0], O_flux_nosat[1][0], O_flux_nosat[2][0], \
            O_flux_nosat[0][1], O_flux_nosat[1][1], O_flux_nosat[2][1], \
            O_flux_nosat[0][2], O_flux_nosat[1][2], O_flux_nosat[2][2], \
            O_flux_nosat[0][3], O_flux_nosat[1][3], O_flux_nosat[2][3], \
            O_flux_nosat[0][4], O_flux_nosat[1][4], O_flux_nosat[2][4], \
            OI_flux_nosat[0][0], OI_flux_nosat[1][0], OI_flux_nosat[2][0], \
            OI_flux_nosat[0][1], OI_flux_nosat[1][1], OI_flux_nosat[2][1], \
            OI_flux_nosat[0][2], OI_flux_nosat[1][2], OI_flux_nosat[2][2], \
            OI_flux_nosat[0][3], OI_flux_nosat[1][3], OI_flux_nosat[2][3], \
            OI_flux_nosat[0][4], OI_flux_nosat[1][4], OI_flux_nosat[2][4], \
            OII_flux_nosat[0][0], OII_flux_nosat[1][0], OII_flux_nosat[2][0], \
            OII_flux_nosat[0][1], OII_flux_nosat[1][1], OII_flux_nosat[2][1], \
            OII_flux_nosat[0][2], OII_flux_nosat[1][2], OII_flux_nosat[2][2], \
            OII_flux_nosat[0][3], OII_flux_nosat[1][3], OII_flux_nosat[2][3], \
            OII_flux_nosat[0][4], OII_flux_nosat[1][4], OII_flux_nosat[2][4], \
            OIII_flux_nosat[0][0], OIII_flux_nosat[1][0], OIII_flux_nosat[2][0], \
            OIII_flux_nosat[0][1], OIII_flux_nosat[1][1], OIII_flux_nosat[2][1], \
            OIII_flux_nosat[0][2], OIII_flux_nosat[1][2], OIII_flux_nosat[2][2], \
            OIII_flux_nosat[0][3], OIII_flux_nosat[1][3], OIII_flux_nosat[2][3], \
            OIII_flux_nosat[0][4], OIII_flux_nosat[1][4], OIII_flux_nosat[2][4], \
            OIV_flux_nosat[0][0], OIV_flux_nosat[1][0], OIV_flux_nosat[2][0], \
            OIV_flux_nosat[0][1], OIV_flux_nosat[1][1], OIV_flux_nosat[2][1], \
            OIV_flux_nosat[0][2], OIV_flux_nosat[1][2], OIV_flux_nosat[2][2], \
            OIV_flux_nosat[0][3], OIV_flux_nosat[1][3], OIV_flux_nosat[2][3], \
            OIV_flux_nosat[0][4], OIV_flux_nosat[1][4], OIV_flux_nosat[2][4], \
            OV_flux_nosat[0][0], OV_flux_nosat[1][0], OV_flux_nosat[2][0], \
            OV_flux_nosat[0][1], OV_flux_nosat[1][1], OV_flux_nosat[2][1], \
            OV_flux_nosat[0][2], OV_flux_nosat[1][2], OV_flux_nosat[2][2], \
            OV_flux_nosat[0][3], OV_flux_nosat[1][3], OV_flux_nosat[2][3], \
            OV_flux_nosat[0][4], OV_flux_nosat[1][4], OV_flux_nosat[2][4], \
            OVI_flux_nosat[0][0], OVI_flux_nosat[1][0], OVI_flux_nosat[2][0], \
            OVI_flux_nosat[0][1], OVI_flux_nosat[1][1], OVI_flux_nosat[2][1], \
            OVI_flux_nosat[0][2], OVI_flux_nosat[1][2], OVI_flux_nosat[2][2], \
            OVI_flux_nosat[0][3], OVI_flux_nosat[1][3], OVI_flux_nosat[2][3], \
            OVI_flux_nosat[0][4], OVI_flux_nosat[1][4], OVI_flux_nosat[2][4], \
            OVII_flux_nosat[0][0], OVII_flux_nosat[1][0], OVII_flux_nosat[2][0], \
            OVII_flux_nosat[0][1], OVII_flux_nosat[1][1], OVII_flux_nosat[2][1], \
            OVII_flux_nosat[0][2], OVII_flux_nosat[1][2], OVII_flux_nosat[2][2], \
            OVII_flux_nosat[0][3], OVII_flux_nosat[1][3], OVII_flux_nosat[2][3], \
            OVII_flux_nosat[0][4], OVII_flux_nosat[1][4], OVII_flux_nosat[2][4], \
            OVIII_flux_nosat[0][0], OVIII_flux_nosat[1][0], OVIII_flux_nosat[2][0], \
            OVIII_flux_nosat[0][1], OVIII_flux_nosat[1][1], OVIII_flux_nosat[2][1], \
            OVIII_flux_nosat[0][2], OVIII_flux_nosat[1][2], OVIII_flux_nosat[2][2], \
            OVIII_flux_nosat[0][3], OVIII_flux_nosat[1][3], OVIII_flux_nosat[2][3], \
            OVIII_flux_nosat[0][4], OVIII_flux_nosat[1][4], OVIII_flux_nosat[2][4], \
            OIX_flux_nosat[0][0], OIX_flux_nosat[1][0], OIX_flux_nosat[2][0], \
            OIX_flux_nosat[0][1], OIX_flux_nosat[1][1], OIX_flux_nosat[2][1], \
            OIX_flux_nosat[0][2], OIX_flux_nosat[1][2], OIX_flux_nosat[2][2], \
            OIX_flux_nosat[0][3], OIX_flux_nosat[1][3], OIX_flux_nosat[2][3], \
            OIX_flux_nosat[0][4], OIX_flux_nosat[1][4], OIX_flux_nosat[2][4]]
        fluxes.add_row(new_row)

        if (sat_radius!=0):
            new_row = [zsnap, inner_r, outer_r]
            if ('mass' in flux_types):
                new_row += [mass_flux_sat[0][0], metal_flux_sat[0][0], \
                mass_flux_sat[1][0], mass_flux_sat[2][0], metal_flux_sat[1][0], metal_flux_sat[2][0], \
                mass_flux_sat[0][1], mass_flux_sat[1][1], mass_flux_sat[2][1], \
                mass_flux_sat[0][2], mass_flux_sat[1][2], mass_flux_sat[2][2], \
                mass_flux_sat[0][3], mass_flux_sat[1][3], mass_flux_sat[2][3], \
                mass_flux_sat[0][4], mass_flux_sat[1][4], mass_flux_sat[2][4], \
                metal_flux_sat[0][1], metal_flux_sat[1][1], metal_flux_sat[2][1], \
                metal_flux_sat[0][2], metal_flux_sat[1][2], metal_flux_sat[2][2], \
                metal_flux_sat[0][3], metal_flux_sat[1][3], metal_flux_sat[2][3], \
                metal_flux_sat[0][4], metal_flux_sat[1][4], metal_flux_sat[2][4]]
            if ('energy' in flux_types):
                new_row += [kinetic_energy_flux_sat[0][0], thermal_energy_flux_sat[0][0], potential_energy_flux_sat[0][0], total_energy_flux_sat[0][0], \
                kinetic_energy_flux_sat[1][0], kinetic_energy_flux_sat[2][0], \
                thermal_energy_flux_sat[1][0], thermal_energy_flux_sat[2][0], \
                potential_energy_flux_sat[1][0], potential_energy_flux_sat[2][0], \
                total_energy_flux_sat[1][0], total_energy_flux_sat[2][0], \
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
                total_energy_flux_sat[0][1], total_energy_flux_sat[1][1], total_energy_flux_sat[2][1], \
                total_energy_flux_sat[0][2], total_energy_flux_sat[1][2], total_energy_flux_sat[2][2], \
                total_energy_flux_sat[0][3], total_energy_flux_sat[1][3], total_energy_flux_sat[2][3], \
                total_energy_flux_sat[0][4], total_energy_flux_sat[1][4], total_energy_flux_sat[2][4]]
            if ('entropy' in flux_types):
                new_row += [entropy_flux_sat[0][0], entropy_flux_sat[1][0], entropy_flux_sat[2][0], \
                entropy_flux_sat[0][1], entropy_flux_sat[1][1], entropy_flux_sat[2][1], \
                entropy_flux_sat[0][2], entropy_flux_sat[1][2], entropy_flux_sat[2][2], \
                entropy_flux_sat[0][3], entropy_flux_sat[1][3], entropy_flux_sat[2][3], \
                entropy_flux_sat[0][4], entropy_flux_sat[1][4], entropy_flux_sat[2][4]]
            if ('O_ion_mass' in flux_types):
                new_row += [O_flux_sat[0][0], O_flux_sat[1][0], O_flux_sat[2][0], \
                O_flux_sat[0][1], O_flux_sat[1][1], O_flux_sat[2][1], \
                O_flux_sat[0][2], O_flux_sat[1][2], O_flux_sat[2][2], \
                O_flux_sat[0][3], O_flux_sat[1][3], O_flux_sat[2][3], \
                O_flux_sat[0][4], O_flux_sat[1][4], O_flux_sat[2][4], \
                OI_flux_sat[0][0], OI_flux_sat[1][0], OI_flux_sat[2][0], \
                OI_flux_sat[0][1], OI_flux_sat[1][1], OI_flux_sat[2][1], \
                OI_flux_sat[0][2], OI_flux_sat[1][2], OI_flux_sat[2][2], \
                OI_flux_sat[0][3], OI_flux_sat[1][3], OI_flux_sat[2][3], \
                OI_flux_sat[0][4], OI_flux_sat[1][4], OI_flux_sat[2][4], \
                OII_flux_sat[0][0], OII_flux_sat[1][0], OII_flux_sat[2][0], \
                OII_flux_sat[0][1], OII_flux_sat[1][1], OII_flux_sat[2][1], \
                OII_flux_sat[0][2], OII_flux_sat[1][2], OII_flux_sat[2][2], \
                OII_flux_sat[0][3], OII_flux_sat[1][3], OII_flux_sat[2][3], \
                OII_flux_sat[0][4], OII_flux_sat[1][4], OII_flux_sat[2][4], \
                OIII_flux_sat[0][0], OIII_flux_sat[1][0], OIII_flux_sat[2][0], \
                OIII_flux_sat[0][1], OIII_flux_sat[1][1], OIII_flux_sat[2][1], \
                OIII_flux_sat[0][2], OIII_flux_sat[1][2], OIII_flux_sat[2][2], \
                OIII_flux_sat[0][3], OIII_flux_sat[1][3], OIII_flux_sat[2][3], \
                OIII_flux_sat[0][4], OIII_flux_sat[1][4], OIII_flux_sat[2][4], \
                OIV_flux_sat[0][0], OIV_flux_sat[1][0], OIV_flux_sat[2][0], \
                OIV_flux_sat[0][1], OIV_flux_sat[1][1], OIV_flux_sat[2][1], \
                OIV_flux_sat[0][2], OIV_flux_sat[1][2], OIV_flux_sat[2][2], \
                OIV_flux_sat[0][3], OIV_flux_sat[1][3], OIV_flux_sat[2][3], \
                OIV_flux_sat[0][4], OIV_flux_sat[1][4], OIV_flux_sat[2][4], \
                OV_flux_sat[0][0], OV_flux_sat[1][0], OV_flux_sat[2][0], \
                OV_flux_sat[0][1], OV_flux_sat[1][1], OV_flux_sat[2][1], \
                OV_flux_sat[0][2], OV_flux_sat[1][2], OV_flux_sat[2][2], \
                OV_flux_sat[0][3], OV_flux_sat[1][3], OV_flux_sat[2][3], \
                OV_flux_sat[0][4], OV_flux_sat[1][4], OV_flux_sat[2][4], \
                OVI_flux_sat[0][0], OVI_flux_sat[1][0], OVI_flux_sat[2][0], \
                OVI_flux_sat[0][1], OVI_flux_sat[1][1], OVI_flux_sat[2][1], \
                OVI_flux_sat[0][2], OVI_flux_sat[1][2], OVI_flux_sat[2][2], \
                OVI_flux_sat[0][3], OVI_flux_sat[1][3], OVI_flux_sat[2][3], \
                OVI_flux_sat[0][4], OVI_flux_sat[1][4], OVI_flux_sat[2][4], \
                OVII_flux_sat[0][0], OVII_flux_sat[1][0], OVII_flux_sat[2][0], \
                OVII_flux_sat[0][1], OVII_flux_sat[1][1], OVII_flux_sat[2][1], \
                OVII_flux_sat[0][2], OVII_flux_sat[1][2], OVII_flux_sat[2][2], \
                OVII_flux_sat[0][3], OVII_flux_sat[1][3], OVII_flux_sat[2][3], \
                OVII_flux_sat[0][4], OVII_flux_sat[1][4], OVII_flux_sat[2][4], \
                OVIII_flux_sat[0][0], OVIII_flux_sat[1][0], OVIII_flux_sat[2][0], \
                OVIII_flux_sat[0][1], OVIII_flux_sat[1][1], OVIII_flux_sat[2][1], \
                OVIII_flux_sat[0][2], OVIII_flux_sat[1][2], OVIII_flux_sat[2][2], \
                OVIII_flux_sat[0][3], OVIII_flux_sat[1][3], OVIII_flux_sat[2][3], \
                OVIII_flux_sat[0][4], OVIII_flux_sat[1][4], OVIII_flux_sat[2][4], \
                OIX_flux_sat[0][0], OIX_flux_sat[1][0], OIX_flux_sat[2][0], \
                OIX_flux_sat[0][1], OIX_flux_sat[1][1], OIX_flux_sat[2][1], \
                OIX_flux_sat[0][2], OIX_flux_sat[1][2], OIX_flux_sat[2][2], \
                OIX_flux_sat[0][3], OIX_flux_sat[1][3], OIX_flux_sat[2][3], \
                OIX_flux_sat[0][4], OIX_flux_sat[1][4], OIX_flux_sat[2][4]]
            fluxes_sat.add_row(new_row)

    fluxes = set_table_units(fluxes)
    if (sat_radius!=0):
        fluxes_sat = set_table_units(fluxes_sat)

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
    if (sat_radius!=0):
        fluxes.write(tablename + '_nosat_sphere' + fluxtype_filename + '.hdf5', path='all_data', serialize_meta=True, overwrite=True)
        fluxes_sat.write(tablename + '_sat_sphere' + fluxtype_filename + '.hdf5', path='all_data', serialize_meta=True, overwrite=True)
    else:
        fluxes.write(tablename + '_sphere' + fluxtype_filename + '.hdf5', path='all_data', serialize_meta=True, overwrite=True)

    return "Fluxes have been calculated for snapshot " + snap + "!"

def calc_fluxes_frustum(ds, snap, zsnap, dt, refine_width_kpc, tablename, surface_args, flux_types, **kwargs):
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

    sat = kwargs.get('sat')
    sat_radius = kwargs.get('sat_radius', 0.)
    halo_center_kpc2 = kwargs.get('halo_center_kpc2', ds.halo_center_kpc)

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
    names_rad = ('redshift', 'radius')
    type_list_rad = ('f8','f8')
    names_edge = ('redshift', 'inner_radius', 'outer_radius')
    type_list_edge = ('f8','f8','f8')
    if ('mass' in flux_types):
        new_names = ('net_mass_flux', 'net_metal_flux', \
        'mass_flux_in', 'mass_flux_out', 'metal_flux_in', 'metal_flux_out', \
        'net_cold_mass_flux', 'cold_mass_flux_in', 'cold_mass_flux_out', \
        'net_cool_mass_flux', 'cool_mass_flux_in', 'cool_mass_flux_out', \
        'net_warm_mass_flux', 'warm_mass_flux_in', 'warm_mass_flux_out', \
        'net_hot_mass_flux', 'hot_mass_flux_in', 'hot_mass_flux_out', \
        'net_cold_metal_flux', 'cold_metal_flux_in', 'cold_metal_flux_out', \
        'net_cool_metal_flux', 'cool_metal_flux_in', 'cool_metal_flux_out', \
        'net_warm_metal_flux', 'warm_metal_flux_in', 'warm_metal_flux_out', \
        'net_hot_metal_flux', 'hot_metal_flux_in', 'hot_metal_flux_out')
        new_types = ('f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', \
        'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', \
        'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8')
        names_rad += new_names
        names_edge += new_names
        type_list_rad += new_types
        type_list_edge += new_types
    if ('energy' in flux_types):
        new_names = ('net_kinetic_energy_flux', 'net_thermal_energy_flux', \
        'net_potential_energy_flux', 'net_radiative_energy_flux', 'net_total_energy_flux',\
        'kinetic_energy_flux_in', 'kinetic_energy_flux_out', \
        'thermal_energy_flux_in', 'thermal_energy_flux_out', \
        'potential_energy_flux_in', 'potential_energy_flux_out', \
        'radiative_energy_flux_in', 'radiative_energy_flux_out', \
        'total_energy_flux_in', 'total_energy_flux_out', \
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
        'net_cold_total_energy_flux', 'cold_total_energy_flux_in', 'cold_total_energy_flux_out', \
        'net_cool_total_energy_flux', 'cool_total_energy_flux_in', 'cool_total_energy_flux_out', \
        'net_warm_total_energy_flux', 'warm_total_energy_flux_in', 'warm_total_energy_flux_out', \
        'net_hot_total_energy_flux', 'hot_total_energy_flux_in', 'hot_total_energy_flux_out')
        new_types = ('f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', \
        'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', \
        'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', \
        'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', \
        'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', \
        'f8', 'f8', 'f8', 'f8', 'f8', 'f8')
        new_names_edge = ('net_kinetic_energy_flux', 'net_thermal_energy_flux', \
        'net_potential_energy_flux', 'net_total_energy_flux', \
        'kinetic_energy_flux_in', 'kinetic_energy_flux_out', \
        'thermal_energy_flux_in', 'thermal_energy_flux_out', \
        'potential_energy_flux_in', 'potential_energy_flux_out', \
        'total_energy_flux_in', 'total_energy_flux_out', \
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
        'net_cold_total_energy_flux', 'cold_total_energy_flux_in', 'cold_total_energy_flux_out', \
        'net_cool_total_energy_flux', 'cool_total_energy_flux_in', 'cool_total_energy_flux_out', \
        'net_warm_total_energy_flux', 'warm_total_energy_flux_in', 'warm_total_energy_flux_out', \
        'net_hot_total_energy_flux', 'hot_total_energy_flux_in', 'hot_total_energy_flux_out')
        new_types_edge = ('f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', \
        'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', \
        'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', \
        'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', \
        'f8', 'f8', 'f8', 'f8', 'f8', 'f8')
        names_rad += new_names
        names_edge += new_names_edge
        type_list_rad += new_types
        type_list_edge += new_types_edge
    if ('entropy' in flux_types):
        new_names = ('net_entropy_flux', 'entropy_flux_in', 'entropy_flux_out', \
        'net_cold_entropy_flux', 'cold_entropy_flux_in', 'cold_entropy_flux_out', \
        'net_cool_entropy_flux', 'cool_entropy_flux_in', 'cool_entropy_flux_out', \
        'net_warm_entropy_flux', 'warm_entropy_flux_in', 'warm_entropy_flux_out', \
        'net_hot_entropy_flux', 'hot_entropy_flux_in', 'hot_entropy_flux_out')
        new_types = ('f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', \
        'f8', 'f8', 'f8', 'f8', 'f8', 'f8')
        names_rad += new_names
        names_edge += new_names
        type_list_rad += new_types
        type_list_edge += new_types
    if ('O_ion_mass' in flux_types):
        new_names = ('net_O_flux', 'O_flux_in', 'O_flux_out', \
        'net_cold_O_flux', 'cold_O_flux_in', 'cold_O_flux_out', \
        'net_cool_O_flux', 'cool_O_flux_in', 'cool_O_flux_out', \
        'net_warm_O_flux', 'warm_O_flux_in', 'warm_O_flux_out', \
        'net_hot_O_flux', 'hot_O_flux_in', 'hot_O_flux_out', \
        'net_OI_flux', 'OI_flux_in', 'OI_flux_out', \
        'net_cold_OI_flux', 'cold_OI_flux_in', 'cold_OI_flux_out', \
        'net_cool_OI_flux', 'cool_OI_flux_in', 'cool_OI_flux_out', \
        'net_warm_OI_flux', 'warm_OI_flux_in', 'warm_OI_flux_out', \
        'net_hot_OI_flux', 'hot_OI_flux_in', 'hot_OI_flux_out', \
        'net_OII_flux', 'OII_flux_in', 'OII_flux_out', \
        'net_cold_OII_flux', 'cold_OII_flux_in', 'cold_OII_flux_out', \
        'net_cool_OII_flux', 'cool_OII_flux_in', 'cool_OII_flux_out', \
        'net_warm_OII_flux', 'warm_OII_flux_in', 'warm_OII_flux_out', \
        'net_hot_OII_flux', 'hot_OII_flux_in', 'hot_OII_flux_out', \
        'net_OIII_flux', 'OIII_flux_in', 'OIII_flux_out', \
        'net_cold_OIII_flux', 'cold_OIII_flux_in', 'cold_OIII_flux_out', \
        'net_cool_OIII_flux', 'cool_OIII_flux_in', 'cool_OIII_flux_out', \
        'net_warm_OIII_flux', 'warm_OIII_flux_in', 'warm_OIII_flux_out', \
        'net_hot_OIII_flux', 'hot_OIII_flux_in', 'hot_OIII_flux_out', \
        'net_OIV_flux', 'OIV_flux_in', 'OIV_flux_out', \
        'net_cold_OIV_flux', 'cold_OIV_flux_in', 'cold_OIV_flux_out', \
        'net_cool_OIV_flux', 'cool_OIV_flux_in', 'cool_OIV_flux_out', \
        'net_warm_OIV_flux', 'warm_OIV_flux_in', 'warm_OIV_flux_out', \
        'net_hot_OIV_flux', 'hot_OIV_flux_in', 'hot_OIV_flux_out', \
        'net_OV_flux', 'OV_flux_in', 'OV_flux_out', \
        'net_cold_OV_flux', 'cold_OV_flux_in', 'cold_OV_flux_out', \
        'net_cool_OV_flux', 'cool_OV_flux_in', 'cool_OV_flux_out', \
        'net_warm_OV_flux', 'warm_OV_flux_in', 'warm_OV_flux_out', \
        'net_hot_OV_flux', 'hot_OV_flux_in', 'hot_OV_flux_out', \
        'net_OVI_flux', 'OVI_flux_in', 'OVI_flux_out', \
        'net_cold_OVI_flux', 'cold_OVI_flux_in', 'cold_OVI_flux_out', \
        'net_cool_OVI_flux', 'cool_OVI_flux_in', 'cool_OVI_flux_out', \
        'net_warm_OVI_flux', 'warm_OVI_flux_in', 'warm_OVI_flux_out', \
        'net_hot_OVI_flux', 'hot_OVI_flux_in', 'hot_OVI_flux_out', \
        'net_OVII_flux', 'OVII_flux_in', 'OVII_flux_out', \
        'net_cold_OVII_flux', 'cold_OVII_flux_in', 'cold_OVII_flux_out', \
        'net_cool_OVII_flux', 'cool_OVII_flux_in', 'cool_OVII_flux_out', \
        'net_warm_OVII_flux', 'warm_OVII_flux_in', 'warm_OVII_flux_out', \
        'net_hot_OVII_flux', 'hot_OVII_flux_in', 'hot_OVII_flux_out', \
        'net_OVIII_flux', 'OVIII_flux_in', 'OVIII_flux_out', \
        'net_cold_OVIII_flux', 'cold_OVIII_flux_in', 'cold_OVIII_flux_out', \
        'net_cool_OVIII_flux', 'cool_OVIII_flux_in', 'cool_OVIII_flux_out', \
        'net_warm_OVIII_flux', 'warm_OVIII_flux_in', 'warm_OVIII_flux_out', \
        'net_hot_OVIII_flux', 'hot_OVIII_flux_in', 'hot_OVIII_flux_out', \
        'net_OIX_flux', 'OIX_flux_in', 'OIX_flux_out', \
        'net_cold_OIX_flux', 'cold_OIX_flux_in', 'cold_OIX_flux_out', \
        'net_cool_OIX_flux', 'cool_OIX_flux_in', 'cool_OIX_flux_out', \
        'net_warm_OIX_flux', 'warm_OIX_flux_in', 'warm_OIX_flux_out', \
        'net_hot_OIX_flux', 'hot_OIX_flux_in', 'hot_OIX_flux_out')
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
        names_rad += new_names
        names_edge += new_names
        type_list_rad += new_types
        type_list_edge += new_types
    fluxes_radial = Table(names=names_rad, dtype=type_list_rad)
    fluxes_edges = Table(names=names_edge, dtype=type_list_edge)
    if (sat_radius!=0):
        fluxes_sat = Table(names=names_edge, dtype=type_list_edge)

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
    vx = sphere['gas','vx_corrected'].in_units('km/s').v
    vy = sphere['gas','vy_corrected'].in_units('km/s').v
    vz = sphere['gas','vz_corrected'].in_units('km/s').v
    new_x = x + vx*dt*(100./cmtopc*stoyr)
    new_y = y + vy*dt*(100./cmtopc*stoyr)
    new_z = z + vz*dt*(100./cmtopc*stoyr)
    new_radius = np.sqrt(new_x**2. + new_y**2. + new_z**2.)
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
        cooling_time = sphere['gas','cooling_time'].in_units('yr').v
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

    # Cut data to only the frustum considered here, stuff that leaves through edges of frustum,
    # and stuff that comes in through edges of frustum
    if (flip):
        min_theta = np.pi-op_angle*np.pi/180.
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
        frus_filename += 'z_op' + str(op_angle)
    if (axis=='x'):
        theta = np.arccos(x/radius)
        new_theta = np.arccos(new_x/new_radius)
        phi = np.arctan2(z, y)
        frus_filename += 'x_op' + str(op_angle)
    if (axis=='y'):
        theta = np.arccos(y/radius)
        new_theta = np.arccos(new_y/new_radius)
        phi = np.arctan2(x, z)
        frus_filename += 'y_op' + str(op_angle)
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
        new_x_rot = rotationArr[0][0]*new_x + rotationArr[0][1]*new_y + rotationArr[0][2]*new_z
        new_y_rot = rotationArr[1][0]*new_x + rotationArr[1][1]*new_y + rotationArr[1][2]*new_z
        new_z_rot = rotationArr[2][0]*new_x + rotationArr[2][1]*new_y + rotationArr[2][2]*new_z
        theta = np.arccos(z_rot/radius)
        new_theta = np.arccos(new_z_rot/new_radius)
        frus_filename += 'axis_' + str(axis[0]) + '_' + str(axis[1]) + '_' + str(axis[2]) + '_op' + str(op_angle)

    fluxtype_filename = ''
    if ('mass' in flux_types):
        fluxtype_filename += '_mass'
    if ('energy' in flux_types):
        fluxtype_filename += '_energy'
    if ('entropy' in flux_types):
        fluxtype_filename += '_entropy'
    if ('O_ion_mass' in flux_types):
        fluxtype_filename += '_Oions'

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
        snap_type = snap[-6:-4]
        if (snap_type=='RD'):
            # If using an RD output, calculate satellite fluxes as if satellites don't move in snap2, i.e. snap1 = snap2
            snap2 = int(snap[-4:])
        else:
            snap2 = int(snap[-4:])+1
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
        if ('mass' in flux_types):
            mass_nosat = mass[bool_nosat]
            metal_mass_nosat = metal_mass[bool_nosat]
        if ('energy' in flux_types):
            kinetic_energy_nosat = kinetic_energy[bool_nosat]
            thermal_energy_nosat = thermal_energy[bool_nosat]
            potential_energy_nosat = potential_energy[bool_nosat]
            total_energy_nosat = total_energy[bool_nosat]
            cooling_time_nosat = cooling_time[bool_nosat]
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
        newradius_nosat = new_radius
        theta_nosat = theta
        newtheta_nosat = new_theta
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

    # Cut satellite-removed data to frustum of interest
    # These are nested lists where the index goes from 0 to 2 for [within frustum, enterting frustum, leaving frustum]
    bool_frus = (theta_nosat >= min_theta) & (theta_nosat <= max_theta) & (newtheta_nosat >= min_theta) & (newtheta_nosat <= max_theta)
    bool_infrus = ((theta_nosat < min_theta) | (theta_nosat > max_theta)) & ((newtheta_nosat >= min_theta) & (newtheta_nosat <= max_theta))
    bool_outfrus = ((theta_nosat >= min_theta) & (theta_nosat <= max_theta)) & ((newtheta_nosat < min_theta) | (newtheta_nosat > max_theta))

    radius_nosat_frus = []
    newradius_nosat_frus = []
    rad_vel_nosat_frus = []
    temperature_nosat_frus = []
    if ('mass' in flux_types):
        mass_nosat_frus = []
        metal_mass_nosat_frus = []
    if ('energy' in flux_types):
        kinetic_energy_nosat_frus = []
        thermal_energy_nosat_frus = []
        potential_energy_nosat_frus = []
        total_energy_nosat_frus = []
        cooling_time_nosat_frus = []
    if ('entropy' in flux_types):
        entropy_nosat_frus = []
    if ('O_ion_mass' in flux_types):
        O_mass_nosat_frus = []
        OI_mass_nosat_frus = []
        OII_mass_nosat_frus = []
        OIII_mass_nosat_frus = []
        OIV_mass_nosat_frus = []
        OV_mass_nosat_frus = []
        OVI_mass_nosat_frus = []
        OVII_mass_nosat_frus = []
        OVIII_mass_nosat_frus = []
        OIX_mass_nosat_frus = []
    for j in range(3):
        if (j==0):
            radius_nosat_frus.append(radius_nosat[bool_frus])
            newradius_nosat_frus.append(newradius_nosat[bool_frus])
            rad_vel_nosat_frus.append(rad_vel_nosat[bool_frus])
            temperature_nosat_frus.append(temperature_nosat[bool_frus])
            if ('mass' in flux_types):
                mass_nosat_frus.append(mass_nosat[bool_frus])
                metal_mass_nosat_frus.append(metal_mass_nosat[bool_frus])
            if ('energy' in flux_types):
                kinetic_energy_nosat_frus.append(kinetic_energy_nosat[bool_frus])
                thermal_energy_nosat_frus.append(thermal_energy_nosat[bool_frus])
                potential_energy_nosat_frus.append(potential_energy_nosat[bool_frus])
                total_energy_nosat_frus.append(total_energy_nosat[bool_frus])
                cooling_time_nosat_frus.append(cooling_time_nosat[bool_frus])
            if ('entropy' in flux_types):
                entropy_nosat_frus.append(entropy_nosat[bool_frus])
            if ('O_ion_mass' in flux_types):
                O_mass_nosat_frus.append(O_mass_nosat[bool_frus])
                OI_mass_nosat_frus.append(OI_mass_nosat[bool_frus])
                OII_mass_nosat_frus.append(OII_mass_nosat[bool_frus])
                OIII_mass_nosat_frus.append(OIII_mass_nosat[bool_frus])
                OIV_mass_nosat_frus.append(OIV_mass_nosat[bool_frus])
                OV_mass_nosat_frus.append(OV_mass_nosat[bool_frus])
                OVI_mass_nosat_frus.append(OVI_mass_nosat[bool_frus])
                OVII_mass_nosat_frus.append(OVII_mass_nosat[bool_frus])
                OVIII_mass_nosat_frus.append(OVIII_mass_nosat[bool_frus])
                OIX_mass_nosat_frus.append(OIX_mass_nosat[bool_frus])
        if (j==1):
            radius_nosat_frus.append(radius_nosat[bool_infrus])
            newradius_nosat_frus.append(newradius_nosat[bool_infrus])
            rad_vel_nosat_frus.append(rad_vel_nosat[bool_infrus])
            temperature_nosat_frus.append(temperature_nosat[bool_infrus])
            if ('mass' in flux_types):
                mass_nosat_frus.append(mass_nosat[bool_infrus])
                metal_mass_nosat_frus.append(metal_mass_nosat[bool_infrus])
            if ('energy' in flux_types):
                kinetic_energy_nosat_frus.append(kinetic_energy_nosat[bool_infrus])
                thermal_energy_nosat_frus.append(thermal_energy_nosat[bool_infrus])
                potential_energy_nosat_frus.append(potential_energy_nosat[bool_infrus])
                total_energy_nosat_frus.append(total_energy_nosat[bool_infrus])
                cooling_time_nosat_frus.append(cooling_time_nosat[bool_infrus])
            if ('entropy' in flux_types):
                entropy_nosat_frus.append(entropy_nosat[bool_infrus])
            if ('O_ion_mass' in flux_types):
                O_mass_nosat_frus.append(O_mass_nosat[bool_infrus])
                OI_mass_nosat_frus.append(OI_mass_nosat[bool_infrus])
                OII_mass_nosat_frus.append(OII_mass_nosat[bool_infrus])
                OIII_mass_nosat_frus.append(OIII_mass_nosat[bool_infrus])
                OIV_mass_nosat_frus.append(OIV_mass_nosat[bool_infrus])
                OV_mass_nosat_frus.append(OV_mass_nosat[bool_infrus])
                OVI_mass_nosat_frus.append(OVI_mass_nosat[bool_infrus])
                OVII_mass_nosat_frus.append(OVII_mass_nosat[bool_infrus])
                OVIII_mass_nosat_frus.append(OVIII_mass_nosat[bool_infrus])
                OIX_mass_nosat_frus.append(OIX_mass_nosat[bool_infrus])
        if (j==2):
            radius_nosat_frus.append(radius_nosat[bool_outfrus])
            newradius_nosat_frus.append(newradius_nosat[bool_outfrus])
            rad_vel_nosat_frus.append(rad_vel_nosat[bool_outfrus])
            temperature_nosat_frus.append(temperature_nosat[bool_outfrus])
            if ('mass' in flux_types):
                mass_nosat_frus.append(mass_nosat[bool_outfrus])
                metal_mass_nosat_frus.append(metal_mass_nosat[bool_outfrus])
            if ('energy' in flux_types):
                kinetic_energy_nosat_frus.append(kinetic_energy_nosat[bool_outfrus])
                thermal_energy_nosat_frus.append(thermal_energy_nosat[bool_outfrus])
                potential_energy_nosat_frus.append(potential_energy_nosat[bool_outfrus])
                total_energy_nosat_frus.append(total_energy_nosat[bool_outfrus])
                cooling_time_nosat_frus.append(cooling_time_nosat[bool_outfrus])
            if ('entropy' in flux_types):
                entropy_nosat_frus.append(entropy_nosat[bool_outfrus])
            if ('O_ion_mass' in flux_types):
                O_mass_nosat_frus.append(O_mass_nosat[bool_outfrus])
                OI_mass_nosat_frus.append(OI_mass_nosat[bool_outfrus])
                OII_mass_nosat_frus.append(OII_mass_nosat[bool_outfrus])
                OIII_mass_nosat_frus.append(OIII_mass_nosat[bool_outfrus])
                OIV_mass_nosat_frus.append(OIV_mass_nosat[bool_outfrus])
                OV_mass_nosat_frus.append(OV_mass_nosat[bool_outfrus])
                OVI_mass_nosat_frus.append(OVI_mass_nosat[bool_outfrus])
                OVII_mass_nosat_frus.append(OVII_mass_nosat[bool_outfrus])
                OVIII_mass_nosat_frus.append(OVIII_mass_nosat[bool_outfrus])
                OIX_mass_nosat_frus.append(OIX_mass_nosat[bool_outfrus])

    # Cut satellite-removed frustum data on temperature
    # These are lists of lists where the first index goes from 0 to 2 for
    # [within frustum, entering frustum, leaving frustum] and the second index goes from 0 to 4 for
    # [all gas, cold, cool, warm, hot]
    if (sat_radius!=0):
        print('Cutting satellite-removed data on temperature')
    else:
        print('Cutting data on temperature')
    radius_nosat_frus_Tcut = []
    rad_vel_nosat_frus_Tcut = []
    newradius_nosat_frus_Tcut = []
    if ('mass' in flux_types):
        mass_nosat_frus_Tcut = []
        metal_mass_nosat_frus_Tcut = []
    if ('energy' in flux_types):
        kinetic_energy_nosat_frus_Tcut = []
        thermal_energy_nosat_frus_Tcut = []
        potential_energy_nosat_frus_Tcut = []
        total_energy_nosat_frus_Tcut = []
        cooling_time_nosat_frus_Tcut = []
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
    for i in range(3):
        radius_nosat_frus_Tcut.append([])
        rad_vel_nosat_frus_Tcut.append([])
        newradius_nosat_frus_Tcut.append([])
        if ('mass' in flux_types):
            mass_nosat_frus_Tcut.append([])
            metal_mass_nosat_frus_Tcut.append([])
        if ('energy' in flux_types):
            kinetic_energy_nosat_frus_Tcut.append([])
            thermal_energy_nosat_frus_Tcut.append([])
            potential_energy_nosat_frus_Tcut.append([])
            total_energy_nosat_frus_Tcut.append([])
            cooling_time_nosat_frus_Tcut.append([])
        if ('entropy' in flux_types):
            entropy_nosat_frus_Tcut.append([])
        if ('O_ion_mass' in flux_types):
            O_mass_nosat_frus_Tcut.append([])
            OI_mass_nosat_frus_Tcut.append([])
            OII_mass_nosat_frus_Tcut.append([])
            OIII_mass_nosat_frus_Tcut.append([])
            OIV_mass_nosat_frus_Tcut.append([])
            OV_mass_nosat_frus_Tcut.append([])
            OVI_mass_nosat_frus_Tcut.append([])
            OVII_mass_nosat_frus_Tcut.append([])
            OVIII_mass_nosat_frus_Tcut.append([])
            OIX_mass_nosat_frus_Tcut.append([])
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
            if ('mass' in flux_types):
                mass_nosat_frus_Tcut[i].append(mass_nosat_frus[i][bool_temp_nosat_frus])
                metal_mass_nosat_frus_Tcut[i].append(metal_mass_nosat_frus[i][bool_temp_nosat_frus])
            if ('energy' in flux_types):
                kinetic_energy_nosat_frus_Tcut[i].append(kinetic_energy_nosat_frus[i][bool_temp_nosat_frus])
                thermal_energy_nosat_frus_Tcut[i].append(thermal_energy_nosat_frus[i][bool_temp_nosat_frus])
                potential_energy_nosat_frus_Tcut[i].append(potential_energy_nosat_frus[i][bool_temp_nosat_frus])
                total_energy_nosat_frus_Tcut[i].append(total_energy_nosat_frus[i][bool_temp_nosat_frus])
                cooling_time_nosat_frus_Tcut[i].append(cooling_time_nosat_frus[i][bool_temp_nosat_frus])
            if ('entropy' in flux_types):
                entropy_nosat_frus_Tcut[i].append(entropy_nosat_frus[i][bool_temp_nosat_frus])
            if ('O_ion_mass' in flux_types):
                O_mass_nosat_frus_Tcut[i].append(O_mass_nosat_frus[i][bool_temp_nosat_frus])
                OI_mass_nosat_frus_Tcut[i].append(OI_mass_nosat_frus[i][bool_temp_nosat_frus])
                OII_mass_nosat_frus_Tcut[i].append(OII_mass_nosat_frus[i][bool_temp_nosat_frus])
                OIII_mass_nosat_frus_Tcut[i].append(OIII_mass_nosat_frus[i][bool_temp_nosat_frus])
                OIV_mass_nosat_frus_Tcut[i].append(OIV_mass_nosat_frus[i][bool_temp_nosat_frus])
                OV_mass_nosat_frus_Tcut[i].append(OV_mass_nosat_frus[i][bool_temp_nosat_frus])
                OVI_mass_nosat_frus_Tcut[i].append(OVI_mass_nosat_frus[i][bool_temp_nosat_frus])
                OVII_mass_nosat_frus_Tcut[i].append(OVII_mass_nosat_frus[i][bool_temp_nosat_frus])
                OVIII_mass_nosat_frus_Tcut[i].append(OVIII_mass_nosat_frus[i][bool_temp_nosat_frus])
                OIX_mass_nosat_frus_Tcut[i].append(OIX_mass_nosat_frus[i][bool_temp_nosat_frus])

    # Cut data to things that cross into or out of satellites in the frustum
    # These are lists of lists where the index goes from 0 to 1 for [from satellites, to satellites]
    if (sat_radius!=0):
        print('Cutting data to satellite fluxes')
        radius_sat = []
        newradius_sat = []
        temperature_sat = []
        if ('mass' in flux_types):
            mass_sat = []
            metal_mass_sat = []
        if ('energy' in flux_types):
            kinetic_energy_sat = []
            thermal_energy_sat = []
            potential_energy_sat = []
            total_energy_sat = []
        if ('entropy' in flux_types):
            entropy_sat = []
        if ('O_ion_mass' in flux_types):
            O_mass_sat = []
            OI_mass_sat = []
            OII_mass_sat = []
            OIII_mass_sat = []
            OIV_mass_sat = []
            OV_mass_sat = []
            OVI_mass_sat = []
            OVII_mass_sat = []
            OVIII_mass_sat = []
            OIX_mass_sat = []
        for j in range(2):
            if (j==0):
                radius_sat.append(radius[bool_fromsat])
                newradius_sat.append(new_radius[bool_fromsat])
                temperature_sat.append(temperature[bool_fromsat])
                if ('mass' in flux_types):
                    mass_sat.append(mass[bool_fromsat])
                    metal_mass_sat.append(metal_mass[bool_fromsat])
                if ('energy' in flux_types):
                    kinetic_energy_sat.append(kinetic_energy[bool_fromsat])
                    thermal_energy_sat.append(thermal_energy[bool_fromsat])
                    potential_energy_sat.append(potential_energy[bool_fromsat])
                    total_energy_sat.append(total_energy[bool_fromsat])
                if ('entropy' in flux_types):
                    entropy_sat.append(entropy[bool_fromsat])
                if ('O_ion_mass' in flux_types):
                    O_mass_sat.append(O_mass[bool_fromsat])
                    OI_mass_sat.append(OI_mass[bool_fromsat])
                    OII_mass_sat.append(OII_mass[bool_fromsat])
                    OIII_mass_sat.append(OIII_mass[bool_fromsat])
                    OIV_mass_sat.append(OIV_mass[bool_fromsat])
                    OV_mass_sat.append(OV_mass[bool_fromsat])
                    OVI_mass_sat.append(OVI_mass[bool_fromsat])
                    OVII_mass_sat.append(OVII_mass[bool_fromsat])
                    OVIII_mass_sat.append(OVIII_mass[bool_fromsat])
                    OIX_mass_sat.append(OIX_mass[bool_fromsat])
            if (j==1):
                radius_sat.append(radius[bool_tosat])
                newradius_sat.append(new_radius[bool_tosat])
                temperature_sat.append(temperature[bool_tosat])
                if ('mass' in flux_types):
                    mass_sat.append(mass[bool_tosat])
                    metal_mass_sat.append(metal_mass[bool_tosat])
                if ('energy' in flux_types):
                    kinetic_energy_sat.append(kinetic_energy[bool_tosat])
                    thermal_energy_sat.append(thermal_energy[bool_tosat])
                    potential_energy_sat.append(potential_energy[bool_tosat])
                    total_energy_sat.append(total_energy[bool_tosat])
                if ('entropy' in flux_types):
                    entropy_sat.append(entropy[bool_tosat])
                if ('O_ion_mass' in flux_types):
                    O_mass_sat.append(O_mass[bool_tosat])
                    OI_mass_sat.append(OI_mass[bool_tosat])
                    OII_mass_sat.append(OII_mass[bool_tosat])
                    OIII_mass_sat.append(OIII_mass[bool_tosat])
                    OIV_mass_sat.append(OIV_mass[bool_tosat])
                    OV_mass_sat.append(OV_mass[bool_tosat])
                    OVI_mass_sat.append(OVI_mass[bool_tosat])
                    OVII_mass_sat.append(OVII_mass[bool_tosat])
                    OVIII_mass_sat.append(OVIII_mass[bool_tosat])
                    OIX_mass_sat.append(OIX_mass[bool_tosat])

        # Cut stuff going into/out of satellites in the frustum on temperature
        # These are nested lists where the first index goes from 0 to 1 for [from satellites, to satellites]
        # and the second index goes from 0 to 4 for [all gas, cold, cool, warm, hot]
        print('Cutting satellite fluxes on temperature')
        radius_sat_Tcut = []
        newradius_sat_Tcut = []
        if ('mass' in flux_types):
            mass_sat_Tcut = []
            metal_mass_sat_Tcut = []
        if ('energy' in flux_types):
            kinetic_energy_sat_Tcut = []
            thermal_energy_sat_Tcut = []
            potential_energy_sat_Tcut = []
            total_energy_sat_Tcut = []
        if ('entropy' in flux_types):
            entropy_sat_Tcut = []
        if ('O_ion_mass' in flux_types):
            O_mass_sat_Tcut = []
            OI_mass_sat_Tcut = []
            OII_mass_sat_Tcut = []
            OIII_mass_sat_Tcut = []
            OIV_mass_sat_Tcut = []
            OV_mass_sat_Tcut = []
            OVI_mass_sat_Tcut = []
            OVII_mass_sat_Tcut = []
            OVIII_mass_sat_Tcut = []
            OIX_mass_sat_Tcut = []
        for i in range(2):
            radius_sat_Tcut.append([])
            newradius_sat_Tcut.append([])
            if ('mass' in flux_types):
                mass_sat_Tcut.append([])
                metal_mass_sat_Tcut.append([])
            if ('energy' in flux_types):
                kinetic_energy_sat_Tcut.append([])
                thermal_energy_sat_Tcut.append([])
                potential_energy_sat_Tcut.append([])
                total_energy_sat_Tcut.append([])
            if ('entropy' in flux_types):
                entropy_sat_Tcut.append([])
            if ('O_ion_mass' in flux_types):
                O_mass_sat_Tcut.append([])
                OI_mass_sat_Tcut.append([])
                OII_mass_sat_Tcut.append([])
                OIII_mass_sat_Tcut.append([])
                OIV_mass_sat_Tcut.append([])
                OV_mass_sat_Tcut.append([])
                OVI_mass_sat_Tcut.append([])
                OVII_mass_sat_Tcut.append([])
                OVIII_mass_sat_Tcut.append([])
                OIX_mass_sat_Tcut.append([])
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
                if ('mass' in flux_types):
                    mass_sat_Tcut[i].append(mass_sat[i][bool_temp_sat])
                    metal_mass_sat_Tcut[i].append(metal_mass_sat[i][bool_temp_sat])
                if ('energy' in flux_types):
                    kinetic_energy_sat_Tcut[i].append(kinetic_energy_sat[i][bool_temp_sat])
                    thermal_energy_sat_Tcut[i].append(thermal_energy_sat[i][bool_temp_sat])
                    potential_energy_sat_Tcut[i].append(potential_energy_sat[i][bool_temp_sat])
                    total_energy_sat_Tcut[i].append(total_energy_sat[i][bool_temp_sat])
                if ('entropy' in flux_types):
                    entropy_sat_Tcut[i].append(entropy_sat[i][bool_temp_sat])
                if ('O_ion_mass' in flux_types):
                    O_mass_sat_Tcut[i].append(O_mass_sat[i][bool_temp_sat])
                    OI_mass_sat_Tcut[i].append(OI_mass_sat[i][bool_temp_sat])
                    OII_mass_sat_Tcut[i].append(OII_mass_sat[i][bool_temp_sat])
                    OIII_mass_sat_Tcut[i].append(OIII_mass_sat[i][bool_temp_sat])
                    OIV_mass_sat_Tcut[i].append(OIV_mass_sat[i][bool_temp_sat])
                    OV_mass_sat_Tcut[i].append(OV_mass_sat[i][bool_temp_sat])
                    OVI_mass_sat_Tcut[i].append(OVI_mass_sat[i][bool_temp_sat])
                    OVII_mass_sat_Tcut[i].append(OVII_mass_sat[i][bool_temp_sat])
                    OVIII_mass_sat_Tcut[i].append(OVIII_mass_sat[i][bool_temp_sat])
                    OIX_mass_sat_Tcut[i].append(OIX_mass_sat[i][bool_temp_sat])

    # Loop over radii
    for i in range(len(radii)):
        inner_r = radii[i].v
        if (i < len(radii) - 1): outer_r = radii[i+1].v

        if (i%10==0): print("Computing radius " + str(i) + "/" + str(len(radii)) + \
                            " for snapshot " + snap)

        # Compute net, in, and out fluxes within the frustum with satellites removed
        # These are nested lists where the first index goes from 0 to 2 for [net, in, out]
        # and the second index goes from 0 to 4 for [all, cold, cool, warm, hot]
        if ('mass' in flux_types):
            mass_flux_nosat = []
            metal_flux_nosat = []
        if ('energy' in flux_types):
            kinetic_energy_flux_nosat = []
            thermal_energy_flux_nosat = []
            potential_energy_flux_nosat = []
            total_energy_flux_nosat = []
        if ('entropy' in flux_types):
            entropy_flux_nosat = []
        if ('O_ion_mass' in flux_types):
            O_flux_nosat = []
            OI_flux_nosat = []
            OII_flux_nosat = []
            OIII_flux_nosat = []
            OIV_flux_nosat = []
            OV_flux_nosat = []
            OVI_flux_nosat = []
            OVII_flux_nosat = []
            OVIII_flux_nosat = []
            OIX_flux_nosat = []
        for j in range(3):
            if ('mass' in flux_types):
                mass_flux_nosat.append([])
                metal_flux_nosat.append([])
            if ('energy' in flux_types):
                kinetic_energy_flux_nosat.append([])
                thermal_energy_flux_nosat.append([])
                potential_energy_flux_nosat.append([])
                total_energy_flux_nosat.append([])
            if ('entropy' in flux_types):
                entropy_flux_nosat.append([])
            if ('O_ion_mass' in flux_types):
                O_flux_nosat.append([])
                OI_flux_nosat.append([])
                OII_flux_nosat.append([])
                OIII_flux_nosat.append([])
                OIV_flux_nosat.append([])
                OV_flux_nosat.append([])
                OVI_flux_nosat.append([])
                OVII_flux_nosat.append([])
                OVIII_flux_nosat.append([])
                OIX_flux_nosat.append([])
            for k in range(5):
                bool_in_r = (radius_nosat_frus_Tcut[0][k] > inner_r) & (newradius_nosat_frus_Tcut[0][k] < inner_r)
                bool_out_r = (radius_nosat_frus_Tcut[0][k] < inner_r) & (newradius_nosat_frus_Tcut[0][k] > inner_r)
                if (j==0):
                    if ('mass' in flux_types):
                        mass_flux_nosat[j].append((np.sum(mass_nosat_frus_Tcut[0][k][bool_out_r]) - \
                          np.sum(mass_nosat_frus_Tcut[0][k][bool_in_r]))/dt)
                        metal_flux_nosat[j].append((np.sum(metal_mass_nosat_frus_Tcut[0][k][bool_out_r]) - \
                          np.sum(metal_mass_nosat_frus_Tcut[0][k][bool_in_r]))/dt)
                    if ('energy' in flux_types):
                        kinetic_energy_flux_nosat[j].append((np.sum(kinetic_energy_nosat_frus_Tcut[0][k][bool_out_r]) - \
                          np.sum(kinetic_energy_nosat_frus_Tcut[0][k][bool_in_r]))/dt)
                        thermal_energy_flux_nosat[j].append((np.sum(thermal_energy_nosat_frus_Tcut[0][k][bool_out_r]) - \
                          np.sum(thermal_energy_nosat_frus_Tcut[0][k][bool_in_r]))/dt)
                        potential_energy_flux_nosat[j].append((np.sum(potential_energy_nosat_frus_Tcut[0][k][bool_out_r]) - \
                          np.sum(potential_energy_nosat_frus_Tcut[0][k][bool_in_r]))/dt)
                        total_energy_flux_nosat[j].append((np.sum(total_energy_nosat_frus_Tcut[0][k][bool_out_r]) - \
                          np.sum(total_energy_nosat_frus_Tcut[0][k][bool_in_r]))/dt)
                    if ('entropy' in flux_types):
                        entropy_flux_nosat[j].append((np.sum(entropy_nosat_frus_Tcut[0][k][bool_out_r]) - \
                          np.sum(entropy_nosat_frus_Tcut[0][k][bool_in_r]))/dt)
                    if ('O_ion_mass' in flux_types):
                        O_flux_nosat[j].append((np.sum(O_mass_nosat_frus_Tcut[0][k][bool_out_r]) - \
                          np.sum(O_mass_nosat_frus_Tcut[0][k][bool_in_r]))/dt)
                        OI_flux_nosat[j].append((np.sum(OI_mass_nosat_frus_Tcut[0][k][bool_out_r]) - \
                          np.sum(OI_mass_nosat_frus_Tcut[0][k][bool_in_r]))/dt)
                        OII_flux_nosat[j].append((np.sum(OII_mass_nosat_frus_Tcut[0][k][bool_out_r]) - \
                          np.sum(OII_mass_nosat_frus_Tcut[0][k][bool_in_r]))/dt)
                        OIII_flux_nosat[j].append((np.sum(OIII_mass_nosat_frus_Tcut[0][k][bool_out_r]) - \
                          np.sum(OIII_mass_nosat_frus_Tcut[0][k][bool_in_r]))/dt)
                        OIV_flux_nosat[j].append((np.sum(OIV_mass_nosat_frus_Tcut[0][k][bool_out_r]) - \
                          np.sum(OIV_mass_nosat_frus_Tcut[0][k][bool_in_r]))/dt)
                        OV_flux_nosat[j].append((np.sum(OV_mass_nosat_frus_Tcut[0][k][bool_out_r]) - \
                          np.sum(OV_mass_nosat_frus_Tcut[0][k][bool_in_r]))/dt)
                        OVI_flux_nosat[j].append((np.sum(OVI_mass_nosat_frus_Tcut[0][k][bool_out_r]) - \
                          np.sum(OVI_mass_nosat_frus_Tcut[0][k][bool_in_r]))/dt)
                        OVII_flux_nosat[j].append((np.sum(OVII_mass_nosat_frus_Tcut[0][k][bool_out_r]) - \
                          np.sum(OVII_mass_nosat_frus_Tcut[0][k][bool_in_r]))/dt)
                        OVIII_flux_nosat[j].append((np.sum(OVIII_mass_nosat_frus_Tcut[0][k][bool_out_r]) - \
                          np.sum(OVIII_mass_nosat_frus_Tcut[0][k][bool_in_r]))/dt)
                        OIX_flux_nosat[j].append((np.sum(OIX_mass_nosat_frus_Tcut[0][k][bool_out_r]) - \
                          np.sum(OIX_mass_nosat_frus_Tcut[0][k][bool_in_r]))/dt)
                if (j==1):
                    if ('mass' in flux_types):
                        mass_flux_nosat[j].append(-np.sum(mass_nosat_frus_Tcut[0][k][bool_in_r])/dt)
                        metal_flux_nosat[j].append(-np.sum(metal_mass_nosat_frus_Tcut[0][k][bool_in_r])/dt)
                    if ('energy' in flux_types):
                        kinetic_energy_flux_nosat[j].append(-np.sum(kinetic_energy_nosat_frus_Tcut[0][k][bool_in_r])/dt)
                        thermal_energy_flux_nosat[j].append(-np.sum(thermal_energy_nosat_frus_Tcut[0][k][bool_in_r])/dt)
                        potential_energy_flux_nosat[j].append(-np.sum(potential_energy_nosat_frus_Tcut[0][k][bool_in_r])/dt)
                        total_energy_flux_nosat[j].append(-np.sum(total_energy_nosat_frus_Tcut[0][k][bool_in_r])/dt)
                    if ('entropy' in flux_types):
                        entropy_flux_nosat[j].append(-np.sum(entropy_nosat_frus_Tcut[0][k][bool_in_r])/dt)
                    if ('O_ion_mass' in flux_types):
                        O_flux_nosat[j].append(-np.sum(O_mass_nosat_frus_Tcut[0][k][bool_in_r])/dt)
                        OI_flux_nosat[j].append(-np.sum(OI_mass_nosat_frus_Tcut[0][k][bool_in_r])/dt)
                        OII_flux_nosat[j].append(-np.sum(OII_mass_nosat_frus_Tcut[0][k][bool_in_r])/dt)
                        OIII_flux_nosat[j].append(-np.sum(OIII_mass_nosat_frus_Tcut[0][k][bool_in_r])/dt)
                        OIV_flux_nosat[j].append(-np.sum(OIV_mass_nosat_frus_Tcut[0][k][bool_in_r])/dt)
                        OV_flux_nosat[j].append(-np.sum(OV_mass_nosat_frus_Tcut[0][k][bool_in_r])/dt)
                        OVI_flux_nosat[j].append(-np.sum(OVI_mass_nosat_frus_Tcut[0][k][bool_in_r])/dt)
                        OVII_flux_nosat[j].append(-np.sum(OVII_mass_nosat_frus_Tcut[0][k][bool_in_r])/dt)
                        OVIII_flux_nosat[j].append(-np.sum(OVIII_mass_nosat_frus_Tcut[0][k][bool_in_r])/dt)
                        OIX_flux_nosat[j].append(-np.sum(OIX_mass_nosat_frus_Tcut[0][k][bool_in_r])/dt)
                if (j==2):
                    if ('mass' in flux_types):
                        mass_flux_nosat[j].append(np.sum(mass_nosat_frus_Tcut[0][k][bool_out_r])/dt)
                        metal_flux_nosat[j].append(np.sum(metal_mass_nosat_frus_Tcut[0][k][bool_out_r])/dt)
                    if ('energy' in flux_types):
                        kinetic_energy_flux_nosat[j].append(np.sum(kinetic_energy_nosat_frus_Tcut[0][k][bool_out_r])/dt)
                        thermal_energy_flux_nosat[j].append(np.sum(thermal_energy_nosat_frus_Tcut[0][k][bool_out_r])/dt)
                        potential_energy_flux_nosat[j].append(np.sum(potential_energy_nosat_frus_Tcut[0][k][bool_out_r])/dt)
                        total_energy_flux_nosat[j].append(np.sum(total_energy_nosat_frus_Tcut[0][k][bool_out_r])/dt)
                    if ('entropy' in flux_types):
                        entropy_flux_nosat[j].append(np.sum(entropy_nosat_frus_Tcut[0][k][bool_out_r])/dt)
                    if ('O_ion_mass' in flux_types):
                        O_flux_nosat[j].append(np.sum(O_mass_nosat_frus_Tcut[0][k][bool_out_r])/dt)
                        OI_flux_nosat[j].append(np.sum(OI_mass_nosat_frus_Tcut[0][k][bool_out_r])/dt)
                        OII_flux_nosat[j].append(np.sum(OII_mass_nosat_frus_Tcut[0][k][bool_out_r])/dt)
                        OIII_flux_nosat[j].append(np.sum(OIII_mass_nosat_frus_Tcut[0][k][bool_out_r])/dt)
                        OIV_flux_nosat[j].append(np.sum(OIV_mass_nosat_frus_Tcut[0][k][bool_out_r])/dt)
                        OV_flux_nosat[j].append(np.sum(OV_mass_nosat_frus_Tcut[0][k][bool_out_r])/dt)
                        OVI_flux_nosat[j].append(np.sum(OVI_mass_nosat_frus_Tcut[0][k][bool_out_r])/dt)
                        OVII_flux_nosat[j].append(np.sum(OVII_mass_nosat_frus_Tcut[0][k][bool_out_r])/dt)
                        OVIII_flux_nosat[j].append(np.sum(OVIII_mass_nosat_frus_Tcut[0][k][bool_out_r])/dt)
                        OIX_flux_nosat[j].append(np.sum(OIX_mass_nosat_frus_Tcut[0][k][bool_out_r])/dt)

        # Compute fluxes from and to satellites (and net) within the frustum between inner_r and outer_r
        # These are nested lists where the first index goes from 0 to 2 for [net, from, to]
        # and the second index goes from 0 to 4 for [all, cold, cool, warm, hot]
        if (i < len(radii)-1):
            if (sat_radius!=0) and ('mass' in flux_types):
                mass_flux_sat = []
                metal_flux_sat = []
            if (sat_radius!=0) and ('energy' in flux_types):
                kinetic_energy_flux_sat = []
                thermal_energy_flux_sat = []
                potential_energy_flux_sat = []
                total_energy_flux_sat = []
            if (sat_radius!=0) and ('entropy' in flux_types):
                entropy_flux_sat = []
            if ('energy' in flux_types):
                radiative_energy_flux_nosat = []
            if (sat_radius!=0) and ('O_ion_mass' in flux_types):
                O_flux_sat = []
                OI_flux_sat = []
                OII_flux_sat = []
                OIII_flux_sat = []
                OIV_flux_sat = []
                OV_flux_sat = []
                OVI_flux_sat = []
                OVII_flux_sat = []
                OVIII_flux_sat = []
                OIX_flux_sat = []
            for j in range(3):
                if (sat_radius!=0) and ('mass' in flux_types):
                    mass_flux_sat.append([])
                    metal_flux_sat.append([])
                if (sat_radius!=0) and ('energy' in flux_types):
                    kinetic_energy_flux_sat.append([])
                    thermal_energy_flux_sat.append([])
                    potential_energy_flux_sat.append([])
                    total_energy_flux_sat.append([])
                if (sat_radius!=0) and ('entropy' in flux_types):
                    entropy_flux_sat.append([])
                if ('energy' in flux_types):
                    radiative_energy_flux_nosat.append([])
                if (sat_radius!=0) and ('O_ion_mass' in flux_types):
                    O_flux_sat.append([])
                    OI_flux_sat.append([])
                    OII_flux_sat.append([])
                    OIII_flux_sat.append([])
                    OIV_flux_sat.append([])
                    OV_flux_sat.append([])
                    OVI_flux_sat.append([])
                    OVII_flux_sat.append([])
                    OVIII_flux_sat.append([])
                    OIX_flux_sat.append([])
                for k in range(5):
                    if (sat_radius!=0):
                        bool_from = (newradius_sat_Tcut[0][k]>inner_r) & (newradius_sat_Tcut[0][k]<outer_r)
                        bool_to = (radius_sat_Tcut[1][k]>inner_r) & (radius_sat_Tcut[1][k]<outer_r)
                    if ('energy' in flux_types):
                        bool_r = (radius_nosat_frus_Tcut[0][k]>inner_r) & (radius_nosat_frus_Tcut[0][k]<outer_r)
                    if (j==0):
                        if (sat_radius!=0) and ('mass' in flux_types):
                            mass_flux_sat[j].append((np.sum(mass_sat_Tcut[0][k][bool_from]) - \
                                                    np.sum(mass_sat_Tcut[1][k][bool_to]))/dt)
                            metal_flux_sat[j].append((np.sum(metal_mass_sat_Tcut[0][k][bool_from]) - \
                                                    np.sum(metal_mass_sat_Tcut[1][k][bool_to]))/dt)
                        if (sat_radius!=0) and ('energy' in flux_types):
                            kinetic_energy_flux_sat[j].append((np.sum(kinetic_energy_sat_Tcut[0][k][bool_from]) - \
                                                              np.sum(kinetic_energy_sat_Tcut[1][k][bool_to]))/dt)
                            thermal_energy_flux_sat[j].append((np.sum(thermal_energy_sat_Tcut[0][k][bool_from]) - \
                                                              np.sum(thermal_energy_sat_Tcut[1][k][bool_to]))/dt)
                            potential_energy_flux_sat[j].append((np.sum(potential_energy_sat_Tcut[0][k][bool_from]) - \
                                                                np.sum(potential_energy_sat_Tcut[1][k][bool_to]))/dt)
                            total_energy_flux_sat[j].append((np.sum(total_energy_sat_Tcut[0][k][bool_from]) - \
                                                                np.sum(total_energy_sat_Tcut[1][k][bool_to]))/dt)
                        if (sat_radius!=0) and ('entropy' in flux_types):
                            entropy_flux_sat[j].append((np.sum(entropy_sat_Tcut[0][k][bool_from]) - \
                                                        np.sum(entropy_sat_Tcut[1][k][bool_to]))/dt)
                        if ('energy' in flux_types):
                            radiative_energy_flux_nosat[j].append(np.sum(thermal_energy_nosat_frus_Tcut[0][k][bool_r] * \
                              mass_nosat_frus_Tcut[0][k][bool_r]*gtoMsun /cooling_time_nosat_frus_Tcut[0][k][bool_r]))
                        if (sat_radius!=0) and ('O_ion_mass' in flux_types):
                            O_flux_sat[j].append((np.sum(O_mass_sat_Tcut[0][k][bool_from]) - \
                              np.sum(O_mass_sat_Tcut[1][k][bool_to]))/dt)
                            OI_flux_sat[j].append((np.sum(OI_mass_sat_Tcut[0][k][bool_from]) - \
                              np.sum(OI_mass_sat_Tcut[1][k][bool_to]))/dt)
                            OII_flux_sat[j].append((np.sum(OII_mass_sat_Tcut[0][k][bool_from]) - \
                              np.sum(OII_mass_sat_Tcut[1][k][bool_to]))/dt)
                            OIII_flux_sat[j].append((np.sum(OIII_mass_sat_Tcut[0][k][bool_from]) - \
                              np.sum(OIII_mass_sat_Tcut[1][k][bool_to]))/dt)
                            OIV_flux_sat[j].append((np.sum(OIV_mass_sat_Tcut[0][k][bool_from]) - \
                              np.sum(OIV_mass_sat_Tcut[1][k][bool_to]))/dt)
                            OV_flux_sat[j].append((np.sum(OV_mass_sat_Tcut[0][k][bool_from]) - \
                              np.sum(OV_mass_sat_Tcut[1][k][bool_to]))/dt)
                            OVI_flux_sat[j].append((np.sum(OVI_mass_sat_Tcut[0][k][bool_from]) - \
                              np.sum(OVI_mass_sat_Tcut[1][k][bool_to]))/dt)
                            OVII_flux_sat[j].append((np.sum(OVII_mass_sat_Tcut[0][k][bool_from]) - \
                              np.sum(OVII_mass_sat_Tcut[1][k][bool_to]))/dt)
                            OVIII_flux_sat[j].append((np.sum(OVIII_mass_sat_Tcut[0][k][bool_from]) - \
                              np.sum(OVIII_mass_sat_Tcut[1][k][bool_to]))/dt)
                            OIX_flux_sat[j].append((np.sum(OIX_mass_sat_Tcut[0][k][bool_from]) - \
                              np.sum(OIX_mass_sat_Tcut[1][k][bool_to]))/dt)
                    if (j==1):
                        if (sat_radius!=0) and ('mass' in flux_types):
                            mass_flux_sat[j].append(np.sum(mass_sat_Tcut[0][k][bool_from])/dt)
                            metal_flux_sat[j].append(np.sum(metal_mass_sat_Tcut[0][k][bool_from])/dt)
                        if (sat_radius!=0) and ('energy' in flux_types):
                            kinetic_energy_flux_sat[j].append(np.sum(kinetic_energy_sat_Tcut[0][k][bool_from])/dt)
                            thermal_energy_flux_sat[j].append(np.sum(thermal_energy_sat_Tcut[0][k][bool_from])/dt)
                            potential_energy_flux_sat[j].append(np.sum(potential_energy_sat_Tcut[0][k][bool_from])/dt)
                            total_energy_flux_sat[j].append(np.sum(total_energy_sat_Tcut[0][k][bool_from])/dt)
                        if (sat_radius!=0) and ('entropy' in flux_types):
                            entropy_flux_sat[j].append(np.sum(entropy_sat_Tcut[0][k][bool_from])/dt)
                        if ('energy' in flux_types):
                            radiative_energy_flux_nosat[j].append(np.sum( \
                              thermal_energy_nosat_frus_Tcut[0][k][bool_r & (rad_vel_nosat_frus_Tcut[0][k]<0.)] * \
                              mass_nosat_frus_Tcut[0][k][bool_r & (rad_vel_nosat_frus_Tcut[0][k]<0.)]*gtoMsun / \
                              cooling_time_nosat_frus_Tcut[0][k][bool_r & (rad_vel_nosat_frus_Tcut[0][k]<0.)]))
                        if (sat_radius!=0) and ('O_ion_mass' in flux_types):
                            O_flux_sat[j].append(np.sum(O_mass_sat_Tcut[0][k][bool_from])/dt)
                            OI_flux_sat[j].append(np.sum(OI_mass_sat_Tcut[0][k][bool_from])/dt)
                            OII_flux_sat[j].append(np.sum(OII_mass_sat_Tcut[0][k][bool_from])/dt)
                            OIII_flux_sat[j].append(np.sum(OIII_mass_sat_Tcut[0][k][bool_from])/dt)
                            OIV_flux_sat[j].append(np.sum(OIV_mass_sat_Tcut[0][k][bool_from])/dt)
                            OV_flux_sat[j].append(np.sum(OV_mass_sat_Tcut[0][k][bool_from])/dt)
                            OVI_flux_sat[j].append(np.sum(OVI_mass_sat_Tcut[0][k][bool_from])/dt)
                            OVII_flux_sat[j].append(np.sum(OVII_mass_sat_Tcut[0][k][bool_from])/dt)
                            OVIII_flux_sat[j].append(np.sum(OVIII_mass_sat_Tcut[0][k][bool_from])/dt)
                            OIX_flux_sat[j].append(np.sum(OIX_mass_sat_Tcut[0][k][bool_from])/dt)
                    if (j==2):
                        if (sat_radius!=0) and ('mass' in flux_types):
                            mass_flux_sat[j].append(-np.sum(mass_sat_Tcut[1][k][bool_to])/dt)
                            metal_flux_sat[j].append(-np.sum(metal_mass_sat_Tcut[1][k][bool_to])/dt)
                        if (sat_radius!=0) and ('energy' in flux_types):
                            kinetic_energy_flux_sat[j].append(-np.sum(kinetic_energy_sat_Tcut[1][k][bool_to])/dt)
                            thermal_energy_flux_sat[j].append(-np.sum(thermal_energy_sat_Tcut[1][k][bool_to])/dt)
                            potential_energy_flux_sat[j].append(-np.sum(potential_energy_sat_Tcut[1][k][bool_to])/dt)
                            total_energy_flux_sat[j].append(-np.sum(total_energy_sat_Tcut[1][k][bool_to])/dt)
                        if (sat_radius!=0) and ('entropy' in flux_types):
                            entropy_flux_sat[j].append(-np.sum(entropy_sat_Tcut[1][k][bool_to])/dt)
                        if ('energy' in flux_types):
                            radiative_energy_flux_nosat[j].append(np.sum( \
                              thermal_energy_nosat_frus_Tcut[0][k][bool_r & (rad_vel_nosat_frus_Tcut[0][k]>0.)] * \
                              mass_nosat_frus_Tcut[0][k][bool_r & (rad_vel_nosat_frus_Tcut[0][k]>0.)]*gtoMsun / \
                              cooling_time_nosat_frus_Tcut[0][k][bool_r & (rad_vel_nosat_frus_Tcut[0][k]>0.)]))
                        if (sat_radius!=0) and ('O_ion_mass' in flux_types):
                            O_flux_sat[j].append(-np.sum(O_mass_sat_Tcut[1][k][bool_to])/dt)
                            OI_flux_sat[j].append(-np.sum(OI_mass_sat_Tcut[1][k][bool_to])/dt)
                            OII_flux_sat[j].append(-np.sum(OII_mass_sat_Tcut[1][k][bool_to])/dt)
                            OIII_flux_sat[j].append(-np.sum(OIII_mass_sat_Tcut[1][k][bool_to])/dt)
                            OIV_flux_sat[j].append(-np.sum(OIV_mass_sat_Tcut[1][k][bool_to])/dt)
                            OV_flux_sat[j].append(-np.sum(OV_mass_sat_Tcut[1][k][bool_to])/dt)
                            OVI_flux_sat[j].append(-np.sum(OVI_mass_sat_Tcut[1][k][bool_to])/dt)
                            OVII_flux_sat[j].append(-np.sum(OVII_mass_sat_Tcut[1][k][bool_to])/dt)
                            OVIII_flux_sat[j].append(-np.sum(OVIII_mass_sat_Tcut[1][k][bool_to])/dt)
                            OIX_flux_sat[j].append(-np.sum(OIX_mass_sat_Tcut[1][k][bool_to])/dt)

        # Compute fluxes through edges of frustum between inner_r and outer_r
        # These are nested lists where the first index goes from 0 to 2 for [net, in, out]
        # and the second index goes from 0 to 4 for [all, cold, cool, warm, hot]
        if (i < len(radii)-1):
            if ('mass' in flux_types):
                mass_flux_edge = []
                metal_flux_edge = []
            if ('energy' in flux_types):
                kinetic_energy_flux_edge = []
                thermal_energy_flux_edge = []
                potential_energy_flux_edge = []
                total_energy_flux_edge = []
            if ('entropy' in flux_types):
                entropy_flux_edge = []
            if ('O_ion_mass' in flux_types):
                O_flux_edge = []
                OI_flux_edge = []
                OII_flux_edge = []
                OIII_flux_edge = []
                OIV_flux_edge = []
                OV_flux_edge = []
                OVI_flux_edge = []
                OVII_flux_edge = []
                OVIII_flux_edge = []
                OIX_flux_edge = []
            for j in range(3):
                if ('mass' in flux_types):
                    mass_flux_edge.append([])
                    metal_flux_edge.append([])
                if ('energy' in flux_types):
                    kinetic_energy_flux_edge.append([])
                    thermal_energy_flux_edge.append([])
                    potential_energy_flux_edge.append([])
                    total_energy_flux_edge.append([])
                if ('entropy' in flux_types):
                    entropy_flux_edge.append([])
                if ('O_ion_mass' in flux_types):
                    O_flux_edge.append([])
                    OI_flux_edge.append([])
                    OII_flux_edge.append([])
                    OIII_flux_edge.append([])
                    OIV_flux_edge.append([])
                    OV_flux_edge.append([])
                    OVI_flux_edge.append([])
                    OVII_flux_edge.append([])
                    OVIII_flux_edge.append([])
                    OIX_flux_edge.append([])
                for k in range(5):
                    bool_in = (newradius_nosat_frus_Tcut[1][k]>inner_r) & (newradius_nosat_frus_Tcut[1][k]<outer_r)
                    bool_out = (radius_nosat_frus_Tcut[2][k]>inner_r) & (radius_nosat_frus_Tcut[2][k]<outer_r)
                    if (j==0):
                        if ('mass' in flux_types):
                            mass_flux_edge[j].append((np.sum(mass_nosat_frus_Tcut[1][k][bool_in]) - \
                                                    np.sum(mass_nosat_frus_Tcut[2][k][bool_out]))/dt)
                            metal_flux_edge[j].append((np.sum(metal_mass_nosat_frus_Tcut[1][k][bool_in]) - \
                                                    np.sum(metal_mass_nosat_frus_Tcut[2][k][bool_out]))/dt)
                        if ('energy' in flux_types):
                            kinetic_energy_flux_edge[j].append((np.sum(kinetic_energy_nosat_frus_Tcut[1][k][bool_in]) - \
                                                              np.sum(kinetic_energy_nosat_frus_Tcut[2][k][bool_out]))/dt)
                            thermal_energy_flux_edge[j].append((np.sum(thermal_energy_nosat_frus_Tcut[1][k][bool_in]) - \
                                                              np.sum(thermal_energy_nosat_frus_Tcut[2][k][bool_out]))/dt)
                            potential_energy_flux_edge[j].append((np.sum(potential_energy_nosat_frus_Tcut[1][k][bool_in]) - \
                                                                np.sum(potential_energy_nosat_frus_Tcut[2][k][bool_out]))/dt)
                            total_energy_flux_edge[j].append((np.sum(total_energy_nosat_frus_Tcut[1][k][bool_in]) - \
                                                                np.sum(total_energy_nosat_frus_Tcut[2][k][bool_out]))/dt)
                        if ('entropy' in flux_types):
                            entropy_flux_edge[j].append((np.sum(entropy_nosat_frus_Tcut[1][k][bool_in]) - \
                                                        np.sum(entropy_nosat_frus_Tcut[2][k][bool_out]))/dt)
                        if ('O_ion_mass' in flux_types):
                            O_flux_edge[j].append((np.sum(O_mass_nosat_frus_Tcut[1][k][bool_in]) - \
                              np.sum(O_mass_nosat_frus_Tcut[2][k][bool_out]))/dt)
                            OI_flux_edge[j].append((np.sum(OI_mass_nosat_frus_Tcut[1][k][bool_in]) - \
                              np.sum(OI_mass_nosat_frus_Tcut[2][k][bool_out]))/dt)
                            OII_flux_edge[j].append((np.sum(OII_mass_nosat_frus_Tcut[1][k][bool_in]) - \
                              np.sum(OII_mass_nosat_frus_Tcut[2][k][bool_out]))/dt)
                            OIII_flux_edge[j].append((np.sum(OIII_mass_nosat_frus_Tcut[1][k][bool_in]) - \
                              np.sum(OIII_mass_nosat_frus_Tcut[2][k][bool_out]))/dt)
                            OIV_flux_edge[j].append((np.sum(OIV_mass_nosat_frus_Tcut[1][k][bool_in]) - \
                              np.sum(OIV_mass_nosat_frus_Tcut[2][k][bool_out]))/dt)
                            OV_flux_edge[j].append((np.sum(OV_mass_nosat_frus_Tcut[1][k][bool_in]) - \
                              np.sum(OV_mass_nosat_frus_Tcut[2][k][bool_out]))/dt)
                            OVI_flux_edge[j].append((np.sum(OVI_mass_nosat_frus_Tcut[1][k][bool_in]) - \
                              np.sum(OVI_mass_nosat_frus_Tcut[2][k][bool_out]))/dt)
                            OVII_flux_edge[j].append((np.sum(OVII_mass_nosat_frus_Tcut[1][k][bool_in]) - \
                              np.sum(OVII_mass_nosat_frus_Tcut[2][k][bool_out]))/dt)
                            OVIII_flux_edge[j].append((np.sum(OVIII_mass_nosat_frus_Tcut[1][k][bool_in]) - \
                              np.sum(OVIII_mass_nosat_frus_Tcut[2][k][bool_out]))/dt)
                            OIX_flux_edge[j].append((np.sum(OIX_mass_nosat_frus_Tcut[1][k][bool_in]) - \
                              np.sum(OIX_mass_nosat_frus_Tcut[2][k][bool_out]))/dt)
                    if (j==1):
                        if ('mass' in flux_types):
                            mass_flux_edge[j].append(np.sum(mass_nosat_frus_Tcut[1][k][bool_in])/dt)
                            metal_flux_edge[j].append(np.sum(metal_mass_nosat_frus_Tcut[1][k][bool_in])/dt)
                        if ('energy' in flux_types):
                            kinetic_energy_flux_edge[j].append(np.sum(kinetic_energy_nosat_frus_Tcut[1][k][bool_in])/dt)
                            thermal_energy_flux_edge[j].append(np.sum(thermal_energy_nosat_frus_Tcut[1][k][bool_in])/dt)
                            potential_energy_flux_edge[j].append(np.sum(potential_energy_nosat_frus_Tcut[1][k][bool_in])/dt)
                            total_energy_flux_edge[j].append(np.sum(total_energy_nosat_frus_Tcut[1][k][bool_in])/dt)
                        if ('entropy' in flux_types):
                            entropy_flux_edge[j].append(np.sum(entropy_nosat_frus_Tcut[1][k][bool_in])/dt)
                        if ('O_ion_mass' in flux_types):
                            O_flux_edge[j].append(np.sum(O_mass_nosat_frus_Tcut[1][k][bool_in])/dt)
                            OI_flux_edge[j].append(np.sum(OI_mass_nosat_frus_Tcut[1][k][bool_in])/dt)
                            OII_flux_edge[j].append(np.sum(OII_mass_nosat_frus_Tcut[1][k][bool_in])/dt)
                            OIII_flux_edge[j].append(np.sum(OIII_mass_nosat_frus_Tcut[1][k][bool_in])/dt)
                            OIV_flux_edge[j].append(np.sum(OIV_mass_nosat_frus_Tcut[1][k][bool_in])/dt)
                            OV_flux_edge[j].append(np.sum(OV_mass_nosat_frus_Tcut[1][k][bool_in])/dt)
                            OVI_flux_edge[j].append(np.sum(OVI_mass_nosat_frus_Tcut[1][k][bool_in])/dt)
                            OVII_flux_edge[j].append(np.sum(OVII_mass_nosat_frus_Tcut[1][k][bool_in])/dt)
                            OVIII_flux_edge[j].append(np.sum(OVIII_mass_nosat_frus_Tcut[1][k][bool_in])/dt)
                            OIX_flux_edge[j].append(np.sum(OIX_mass_nosat_frus_Tcut[1][k][bool_in])/dt)
                    if (j==2):
                        if ('mass' in flux_types):
                            mass_flux_edge[j].append(-np.sum(mass_nosat_frus_Tcut[2][k][bool_out])/dt)
                            metal_flux_edge[j].append(-np.sum(metal_mass_nosat_frus_Tcut[2][k][bool_out])/dt)
                        if ('energy' in flux_types):
                            kinetic_energy_flux_edge[j].append(-np.sum(kinetic_energy_nosat_frus_Tcut[2][k][bool_out])/dt)
                            thermal_energy_flux_edge[j].append(-np.sum(thermal_energy_nosat_frus_Tcut[2][k][bool_out])/dt)
                            potential_energy_flux_edge[j].append(-np.sum(potential_energy_nosat_frus_Tcut[2][k][bool_out])/dt)
                            total_energy_flux_edge[j].append(-np.sum(total_energy_nosat_frus_Tcut[2][k][bool_out])/dt)
                        if ('entropy' in flux_types):
                            entropy_flux_edge[j].append(-np.sum(entropy_nosat_frus_Tcut[2][k][bool_out])/dt)
                        if ('O_ion_mass' in flux_types):
                            O_flux_edge[j].append(-np.sum(O_mass_nosat_frus_Tcut[2][k][bool_out])/dt)
                            OI_flux_edge[j].append(-np.sum(OI_mass_nosat_frus_Tcut[2][k][bool_out])/dt)
                            OII_flux_edge[j].append(-np.sum(OII_mass_nosat_frus_Tcut[2][k][bool_out])/dt)
                            OIII_flux_edge[j].append(-np.sum(OIII_mass_nosat_frus_Tcut[2][k][bool_out])/dt)
                            OIV_flux_edge[j].append(-np.sum(OIV_mass_nosat_frus_Tcut[2][k][bool_out])/dt)
                            OV_flux_edge[j].append(-np.sum(OV_mass_nosat_frus_Tcut[2][k][bool_out])/dt)
                            OVI_flux_edge[j].append(-np.sum(OVI_mass_nosat_frus_Tcut[2][k][bool_out])/dt)
                            OVII_flux_edge[j].append(-np.sum(OVII_mass_nosat_frus_Tcut[2][k][bool_out])/dt)
                            OVIII_flux_edge[j].append(-np.sum(OVIII_mass_nosat_frus_Tcut[2][k][bool_out])/dt)
                            OIX_flux_edge[j].append(-np.sum(OIX_mass_nosat_frus_Tcut[2][k][bool_out])/dt)

        # Add everything to the tables
        new_row_rad = [zsnap, inner_r]
        new_row_edge = [zsnap, inner_r, outer_r]
        if ('mass' in flux_types):
            new_row_rad += [mass_flux_nosat[0][0], metal_flux_nosat[0][0], \
            mass_flux_nosat[1][0], mass_flux_nosat[2][0], metal_flux_nosat[1][0], metal_flux_nosat[2][0], \
            mass_flux_nosat[0][1], mass_flux_nosat[1][1], mass_flux_nosat[2][1], \
            mass_flux_nosat[0][2], mass_flux_nosat[1][2], mass_flux_nosat[2][2], \
            mass_flux_nosat[0][3], mass_flux_nosat[1][3], mass_flux_nosat[2][3], \
            mass_flux_nosat[0][4], mass_flux_nosat[1][4], mass_flux_nosat[2][4], \
            metal_flux_nosat[0][1], metal_flux_nosat[1][1], metal_flux_nosat[2][1], \
            metal_flux_nosat[0][2], metal_flux_nosat[1][2], metal_flux_nosat[2][2], \
            metal_flux_nosat[0][3], metal_flux_nosat[1][3], metal_flux_nosat[2][3], \
            metal_flux_nosat[0][4], metal_flux_nosat[1][4], metal_flux_nosat[2][4]]
            new_row_edge += [mass_flux_edge[0][0], metal_flux_edge[0][0], \
            mass_flux_edge[1][0], mass_flux_edge[2][0], metal_flux_edge[1][0], metal_flux_edge[2][0], \
            mass_flux_edge[0][1], mass_flux_edge[1][1], mass_flux_edge[2][1], \
            mass_flux_edge[0][2], mass_flux_edge[1][2], mass_flux_edge[2][2], \
            mass_flux_edge[0][3], mass_flux_edge[1][3], mass_flux_edge[2][3], \
            mass_flux_edge[0][4], mass_flux_edge[1][4], mass_flux_edge[2][4], \
            metal_flux_edge[0][1], metal_flux_edge[1][1], metal_flux_edge[2][1], \
            metal_flux_edge[0][2], metal_flux_edge[1][2], metal_flux_edge[2][2], \
            metal_flux_edge[0][3], metal_flux_edge[1][3], metal_flux_edge[2][3], \
            metal_flux_edge[0][4], metal_flux_edge[1][4], metal_flux_edge[2][4]]
        if ('energy' in flux_types):
            new_row_rad += [kinetic_energy_flux_nosat[0][0], thermal_energy_flux_nosat[0][0], \
            potential_energy_flux_nosat[0][0], radiative_energy_flux_nosat[0][0], \
            total_energy_flux_nosat[0][0], \
            kinetic_energy_flux_nosat[1][0], kinetic_energy_flux_nosat[2][0], \
            thermal_energy_flux_nosat[1][0], thermal_energy_flux_nosat[2][0], \
            potential_energy_flux_nosat[1][0], potential_energy_flux_nosat[2][0], \
            total_energy_flux_nosat[1][0], total_energy_flux_nosat[2][0], \
            radiative_energy_flux_nosat[1][0], radiative_energy_flux_nosat[2][0], \
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
            total_energy_flux_nosat[0][1], total_energy_flux_nosat[1][1], total_energy_flux_nosat[2][1], \
            total_energy_flux_nosat[0][2], total_energy_flux_nosat[1][2], total_energy_flux_nosat[2][2], \
            total_energy_flux_nosat[0][3], total_energy_flux_nosat[1][3], total_energy_flux_nosat[2][3], \
            total_energy_flux_nosat[0][4], total_energy_flux_nosat[1][4], total_energy_flux_nosat[2][4]]
            new_row_edge += [kinetic_energy_flux_edge[0][0], thermal_energy_flux_edge[0][0], \
            potential_energy_flux_edge[0][0], total_energy_flux_edge[0][0], \
            kinetic_energy_flux_edge[1][0], kinetic_energy_flux_edge[2][0], \
            thermal_energy_flux_edge[1][0], thermal_energy_flux_edge[2][0], \
            potential_energy_flux_edge[1][0], potential_energy_flux_edge[2][0], \
            total_energy_flux_edge[1][0], total_energy_flux_edge[2][0], \
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
            total_energy_flux_edge[0][1], total_energy_flux_edge[1][1], total_energy_flux_edge[2][1], \
            total_energy_flux_edge[0][2], total_energy_flux_edge[1][2], total_energy_flux_edge[2][2], \
            total_energy_flux_edge[0][3], total_energy_flux_edge[1][3], total_energy_flux_edge[2][3], \
            total_energy_flux_edge[0][4], total_energy_flux_edge[1][4], total_energy_flux_edge[2][4]]
        if ('entropy' in flux_types):
            new_row_rad += [entropy_flux_nosat[0][0], entropy_flux_nosat[1][0], entropy_flux_nosat[2][0], \
            entropy_flux_nosat[0][1], entropy_flux_nosat[1][1], entropy_flux_nosat[2][1], \
            entropy_flux_nosat[0][2], entropy_flux_nosat[1][2], entropy_flux_nosat[2][2], \
            entropy_flux_nosat[0][3], entropy_flux_nosat[1][3], entropy_flux_nosat[2][3], \
            entropy_flux_nosat[0][4], entropy_flux_nosat[1][4], entropy_flux_nosat[2][4]]
            new_row_edge += [entropy_flux_edge[0][0], entropy_flux_edge[1][0], entropy_flux_edge[2][0], \
            entropy_flux_edge[0][1], entropy_flux_edge[1][1], entropy_flux_edge[2][1], \
            entropy_flux_edge[0][2], entropy_flux_edge[1][2], entropy_flux_edge[2][2], \
            entropy_flux_edge[0][3], entropy_flux_edge[1][3], entropy_flux_edge[2][3], \
            entropy_flux_edge[0][4], entropy_flux_edge[1][4], entropy_flux_edge[2][4]]
        if ('O_ion_mass' in flux_types):
            new_row_rad += [O_flux_nosat[0][0], O_flux_nosat[1][0], O_flux_nosat[2][0], \
            O_flux_nosat[0][1], O_flux_nosat[1][1], O_flux_nosat[2][1], \
            O_flux_nosat[0][2], O_flux_nosat[1][2], O_flux_nosat[2][2], \
            O_flux_nosat[0][3], O_flux_nosat[1][3], O_flux_nosat[2][3], \
            O_flux_nosat[0][4], O_flux_nosat[1][4], O_flux_nosat[2][4], \
            OI_flux_nosat[0][0], OI_flux_nosat[1][0], OI_flux_nosat[2][0], \
            OI_flux_nosat[0][1], OI_flux_nosat[1][1], OI_flux_nosat[2][1], \
            OI_flux_nosat[0][2], OI_flux_nosat[1][2], OI_flux_nosat[2][2], \
            OI_flux_nosat[0][3], OI_flux_nosat[1][3], OI_flux_nosat[2][3], \
            OI_flux_nosat[0][4], OI_flux_nosat[1][4], OI_flux_nosat[2][4], \
            OII_flux_nosat[0][0], OII_flux_nosat[1][0], OII_flux_nosat[2][0], \
            OII_flux_nosat[0][1], OII_flux_nosat[1][1], OII_flux_nosat[2][1], \
            OII_flux_nosat[0][2], OII_flux_nosat[1][2], OII_flux_nosat[2][2], \
            OII_flux_nosat[0][3], OII_flux_nosat[1][3], OII_flux_nosat[2][3], \
            OII_flux_nosat[0][4], OII_flux_nosat[1][4], OII_flux_nosat[2][4], \
            OIII_flux_nosat[0][0], OIII_flux_nosat[1][0], OIII_flux_nosat[2][0], \
            OIII_flux_nosat[0][1], OIII_flux_nosat[1][1], OIII_flux_nosat[2][1], \
            OIII_flux_nosat[0][2], OIII_flux_nosat[1][2], OIII_flux_nosat[2][2], \
            OIII_flux_nosat[0][3], OIII_flux_nosat[1][3], OIII_flux_nosat[2][3], \
            OIII_flux_nosat[0][4], OIII_flux_nosat[1][4], OIII_flux_nosat[2][4], \
            OIV_flux_nosat[0][0], OIV_flux_nosat[1][0], OIV_flux_nosat[2][0], \
            OIV_flux_nosat[0][1], OIV_flux_nosat[1][1], OIV_flux_nosat[2][1], \
            OIV_flux_nosat[0][2], OIV_flux_nosat[1][2], OIV_flux_nosat[2][2], \
            OIV_flux_nosat[0][3], OIV_flux_nosat[1][3], OIV_flux_nosat[2][3], \
            OIV_flux_nosat[0][4], OIV_flux_nosat[1][4], OIV_flux_nosat[2][4], \
            OV_flux_nosat[0][0], OV_flux_nosat[1][0], OV_flux_nosat[2][0], \
            OV_flux_nosat[0][1], OV_flux_nosat[1][1], OV_flux_nosat[2][1], \
            OV_flux_nosat[0][2], OV_flux_nosat[1][2], OV_flux_nosat[2][2], \
            OV_flux_nosat[0][3], OV_flux_nosat[1][3], OV_flux_nosat[2][3], \
            OV_flux_nosat[0][4], OV_flux_nosat[1][4], OV_flux_nosat[2][4], \
            OVI_flux_nosat[0][0], OVI_flux_nosat[1][0], OVI_flux_nosat[2][0], \
            OVI_flux_nosat[0][1], OVI_flux_nosat[1][1], OVI_flux_nosat[2][1], \
            OVI_flux_nosat[0][2], OVI_flux_nosat[1][2], OVI_flux_nosat[2][2], \
            OVI_flux_nosat[0][3], OVI_flux_nosat[1][3], OVI_flux_nosat[2][3], \
            OVI_flux_nosat[0][4], OVI_flux_nosat[1][4], OVI_flux_nosat[2][4], \
            OVII_flux_nosat[0][0], OVII_flux_nosat[1][0], OVII_flux_nosat[2][0], \
            OVII_flux_nosat[0][1], OVII_flux_nosat[1][1], OVII_flux_nosat[2][1], \
            OVII_flux_nosat[0][2], OVII_flux_nosat[1][2], OVII_flux_nosat[2][2], \
            OVII_flux_nosat[0][3], OVII_flux_nosat[1][3], OVII_flux_nosat[2][3], \
            OVII_flux_nosat[0][4], OVII_flux_nosat[1][4], OVII_flux_nosat[2][4], \
            OVIII_flux_nosat[0][0], OVIII_flux_nosat[1][0], OVIII_flux_nosat[2][0], \
            OVIII_flux_nosat[0][1], OVIII_flux_nosat[1][1], OVIII_flux_nosat[2][1], \
            OVIII_flux_nosat[0][2], OVIII_flux_nosat[1][2], OVIII_flux_nosat[2][2], \
            OVIII_flux_nosat[0][3], OVIII_flux_nosat[1][3], OVIII_flux_nosat[2][3], \
            OVIII_flux_nosat[0][4], OVIII_flux_nosat[1][4], OVIII_flux_nosat[2][4], \
            OIX_flux_nosat[0][0], OIX_flux_nosat[1][0], OIX_flux_nosat[2][0], \
            OIX_flux_nosat[0][1], OIX_flux_nosat[1][1], OIX_flux_nosat[2][1], \
            OIX_flux_nosat[0][2], OIX_flux_nosat[1][2], OIX_flux_nosat[2][2], \
            OIX_flux_nosat[0][3], OIX_flux_nosat[1][3], OIX_flux_nosat[2][3], \
            OIX_flux_nosat[0][4], OIX_flux_nosat[1][4], OIX_flux_nosat[2][4]]
            new_row_edge += [O_flux_edge[0][0], O_flux_edge[1][0], O_flux_edge[2][0], \
            O_flux_edge[0][1], O_flux_edge[1][1], O_flux_edge[2][1], \
            O_flux_edge[0][2], O_flux_edge[1][2], O_flux_edge[2][2], \
            O_flux_edge[0][3], O_flux_edge[1][3], O_flux_edge[2][3], \
            O_flux_edge[0][4], O_flux_edge[1][4], O_flux_edge[2][4], \
            OI_flux_edge[0][0], OI_flux_edge[1][0], OI_flux_edge[2][0], \
            OI_flux_edge[0][1], OI_flux_edge[1][1], OI_flux_edge[2][1], \
            OI_flux_edge[0][2], OI_flux_edge[1][2], OI_flux_edge[2][2], \
            OI_flux_edge[0][3], OI_flux_edge[1][3], OI_flux_edge[2][3], \
            OI_flux_edge[0][4], OI_flux_edge[1][4], OI_flux_edge[2][4], \
            OII_flux_edge[0][0], OII_flux_edge[1][0], OII_flux_edge[2][0], \
            OII_flux_edge[0][1], OII_flux_edge[1][1], OII_flux_edge[2][1], \
            OII_flux_edge[0][2], OII_flux_edge[1][2], OII_flux_edge[2][2], \
            OII_flux_edge[0][3], OII_flux_edge[1][3], OII_flux_edge[2][3], \
            OII_flux_edge[0][4], OII_flux_edge[1][4], OII_flux_edge[2][4], \
            OIII_flux_edge[0][0], OIII_flux_edge[1][0], OIII_flux_edge[2][0], \
            OIII_flux_edge[0][1], OIII_flux_edge[1][1], OIII_flux_edge[2][1], \
            OIII_flux_edge[0][2], OIII_flux_edge[1][2], OIII_flux_edge[2][2], \
            OIII_flux_edge[0][3], OIII_flux_edge[1][3], OIII_flux_edge[2][3], \
            OIII_flux_edge[0][4], OIII_flux_edge[1][4], OIII_flux_edge[2][4], \
            OIV_flux_edge[0][0], OIV_flux_edge[1][0], OIV_flux_edge[2][0], \
            OIV_flux_edge[0][1], OIV_flux_edge[1][1], OIV_flux_edge[2][1], \
            OIV_flux_edge[0][2], OIV_flux_edge[1][2], OIV_flux_edge[2][2], \
            OIV_flux_edge[0][3], OIV_flux_edge[1][3], OIV_flux_edge[2][3], \
            OIV_flux_edge[0][4], OIV_flux_edge[1][4], OIV_flux_edge[2][4], \
            OV_flux_edge[0][0], OV_flux_edge[1][0], OV_flux_edge[2][0], \
            OV_flux_edge[0][1], OV_flux_edge[1][1], OV_flux_edge[2][1], \
            OV_flux_edge[0][2], OV_flux_edge[1][2], OV_flux_edge[2][2], \
            OV_flux_edge[0][3], OV_flux_edge[1][3], OV_flux_edge[2][3], \
            OV_flux_edge[0][4], OV_flux_edge[1][4], OV_flux_edge[2][4], \
            OVI_flux_edge[0][0], OVI_flux_edge[1][0], OVI_flux_edge[2][0], \
            OVI_flux_edge[0][1], OVI_flux_edge[1][1], OVI_flux_edge[2][1], \
            OVI_flux_edge[0][2], OVI_flux_edge[1][2], OVI_flux_edge[2][2], \
            OVI_flux_edge[0][3], OVI_flux_edge[1][3], OVI_flux_edge[2][3], \
            OVI_flux_edge[0][4], OVI_flux_edge[1][4], OVI_flux_edge[2][4], \
            OVII_flux_edge[0][0], OVII_flux_edge[1][0], OVII_flux_edge[2][0], \
            OVII_flux_edge[0][1], OVII_flux_edge[1][1], OVII_flux_edge[2][1], \
            OVII_flux_edge[0][2], OVII_flux_edge[1][2], OVII_flux_edge[2][2], \
            OVII_flux_edge[0][3], OVII_flux_edge[1][3], OVII_flux_edge[2][3], \
            OVII_flux_edge[0][4], OVII_flux_edge[1][4], OVII_flux_edge[2][4], \
            OVIII_flux_edge[0][0], OVIII_flux_edge[1][0], OVIII_flux_edge[2][0], \
            OVIII_flux_edge[0][1], OVIII_flux_edge[1][1], OVIII_flux_edge[2][1], \
            OVIII_flux_edge[0][2], OVIII_flux_edge[1][2], OVIII_flux_edge[2][2], \
            OVIII_flux_edge[0][3], OVIII_flux_edge[1][3], OVIII_flux_edge[2][3], \
            OVIII_flux_edge[0][4], OVIII_flux_edge[1][4], OVIII_flux_edge[2][4], \
            OIX_flux_edge[0][0], OIX_flux_edge[1][0], OIX_flux_edge[2][0], \
            OIX_flux_edge[0][1], OIX_flux_edge[1][1], OIX_flux_edge[2][1], \
            OIX_flux_edge[0][2], OIX_flux_edge[1][2], OIX_flux_edge[2][2], \
            OIX_flux_edge[0][3], OIX_flux_edge[1][3], OIX_flux_edge[2][3], \
            OIX_flux_edge[0][4], OIX_flux_edge[1][4], OIX_flux_edge[2][4]]
        fluxes_radial.add_row(new_row_rad)
        fluxes_edges.add_row(new_row_edge)
        if (sat_radius!=0):
            new_row_sat = [zsnap, inner_r, outer_r]
            if ('mass' in flux_types):
                new_row_sat += [mass_flux_sat[0][0], metal_flux_sat[0][0], \
                mass_flux_sat[1][0], mass_flux_sat[2][0], metal_flux_sat[1][0], metal_flux_sat[2][0], \
                mass_flux_sat[0][1], mass_flux_sat[1][1], mass_flux_sat[2][1], \
                mass_flux_sat[0][2], mass_flux_sat[1][2], mass_flux_sat[2][2], \
                mass_flux_sat[0][3], mass_flux_sat[1][3], mass_flux_sat[2][3], \
                mass_flux_sat[0][4], mass_flux_sat[1][4], mass_flux_sat[2][4], \
                metal_flux_sat[0][1], metal_flux_sat[1][1], metal_flux_sat[2][1], \
                metal_flux_sat[0][2], metal_flux_sat[1][2], metal_flux_sat[2][2], \
                metal_flux_sat[0][3], metal_flux_sat[1][3], metal_flux_sat[2][3], \
                metal_flux_sat[0][4], metal_flux_sat[1][4], metal_flux_sat[2][4]]
            if ('energy' in flux_types):
                new_row_sat += [kinetic_energy_flux_sat[0][0], thermal_energy_flux_sat[0][0], \
                potential_energy_flux_sat[0][0], total_energy_flux_sat[0][0],  kinetic_energy_flux_sat[1][0], kinetic_energy_flux_sat[2][0], \
                thermal_energy_flux_sat[1][0], thermal_energy_flux_sat[2][0], \
                potential_energy_flux_sat[1][0], potential_energy_flux_sat[2][0], \
                total_energy_flux_sat[1][0], total_energy_flux_sat[2][0], \
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
                total_energy_flux_sat[0][1], total_energy_flux_sat[1][1], total_energy_flux_sat[2][1], \
                total_energy_flux_sat[0][2], total_energy_flux_sat[1][2], total_energy_flux_sat[2][2], \
                total_energy_flux_sat[0][3], total_energy_flux_sat[1][3], total_energy_flux_sat[2][3], \
                total_energy_flux_sat[0][4], total_energy_flux_sat[1][4], total_energy_flux_sat[2][4]]
            if ('entropy' in flux_types):
                new_row_sat += [entropy_flux_sat[0][0], entropy_flux_sat[1][0], entropy_flux_sat[2][0], \
                entropy_flux_sat[0][1], entropy_flux_sat[1][1], entropy_flux_sat[2][1], \
                entropy_flux_sat[0][2], entropy_flux_sat[1][2], entropy_flux_sat[2][2], \
                entropy_flux_sat[0][3], entropy_flux_sat[1][3], entropy_flux_sat[2][3], \
                entropy_flux_sat[0][4], entropy_flux_sat[1][4], entropy_flux_sat[2][4]]
            if ('O_ion_mass' in flux_types):
                new_row_sat += [O_flux_sat[0][0], O_flux_sat[1][0], O_flux_sat[2][0], \
                O_flux_sat[0][1], O_flux_sat[1][1], O_flux_sat[2][1], \
                O_flux_sat[0][2], O_flux_sat[1][2], O_flux_sat[2][2], \
                O_flux_sat[0][3], O_flux_sat[1][3], O_flux_sat[2][3], \
                O_flux_sat[0][4], O_flux_sat[1][4], O_flux_sat[2][4], \
                OI_flux_sat[0][0], OI_flux_sat[1][0], OI_flux_sat[2][0], \
                OI_flux_sat[0][1], OI_flux_sat[1][1], OI_flux_sat[2][1], \
                OI_flux_sat[0][2], OI_flux_sat[1][2], OI_flux_sat[2][2], \
                OI_flux_sat[0][3], OI_flux_sat[1][3], OI_flux_sat[2][3], \
                OI_flux_sat[0][4], OI_flux_sat[1][4], OI_flux_sat[2][4], \
                OII_flux_sat[0][0], OII_flux_sat[1][0], OII_flux_sat[2][0], \
                OII_flux_sat[0][1], OII_flux_sat[1][1], OII_flux_sat[2][1], \
                OII_flux_sat[0][2], OII_flux_sat[1][2], OII_flux_sat[2][2], \
                OII_flux_sat[0][3], OII_flux_sat[1][3], OII_flux_sat[2][3], \
                OII_flux_sat[0][4], OII_flux_sat[1][4], OII_flux_sat[2][4], \
                OIII_flux_sat[0][0], OIII_flux_sat[1][0], OIII_flux_sat[2][0], \
                OIII_flux_sat[0][1], OIII_flux_sat[1][1], OIII_flux_sat[2][1], \
                OIII_flux_sat[0][2], OIII_flux_sat[1][2], OIII_flux_sat[2][2], \
                OIII_flux_sat[0][3], OIII_flux_sat[1][3], OIII_flux_sat[2][3], \
                OIII_flux_sat[0][4], OIII_flux_sat[1][4], OIII_flux_sat[2][4], \
                OIV_flux_sat[0][0], OIV_flux_sat[1][0], OIV_flux_sat[2][0], \
                OIV_flux_sat[0][1], OIV_flux_sat[1][1], OIV_flux_sat[2][1], \
                OIV_flux_sat[0][2], OIV_flux_sat[1][2], OIV_flux_sat[2][2], \
                OIV_flux_sat[0][3], OIV_flux_sat[1][3], OIV_flux_sat[2][3], \
                OIV_flux_sat[0][4], OIV_flux_sat[1][4], OIV_flux_sat[2][4], \
                OV_flux_sat[0][0], OV_flux_sat[1][0], OV_flux_sat[2][0], \
                OV_flux_sat[0][1], OV_flux_sat[1][1], OV_flux_sat[2][1], \
                OV_flux_sat[0][2], OV_flux_sat[1][2], OV_flux_sat[2][2], \
                OV_flux_sat[0][3], OV_flux_sat[1][3], OV_flux_sat[2][3], \
                OV_flux_sat[0][4], OV_flux_sat[1][4], OV_flux_sat[2][4], \
                OVI_flux_sat[0][0], OVI_flux_sat[1][0], OVI_flux_sat[2][0], \
                OVI_flux_sat[0][1], OVI_flux_sat[1][1], OVI_flux_sat[2][1], \
                OVI_flux_sat[0][2], OVI_flux_sat[1][2], OVI_flux_sat[2][2], \
                OVI_flux_sat[0][3], OVI_flux_sat[1][3], OVI_flux_sat[2][3], \
                OVI_flux_sat[0][4], OVI_flux_sat[1][4], OVI_flux_sat[2][4], \
                OVII_flux_sat[0][0], OVII_flux_sat[1][0], OVII_flux_sat[2][0], \
                OVII_flux_sat[0][1], OVII_flux_sat[1][1], OVII_flux_sat[2][1], \
                OVII_flux_sat[0][2], OVII_flux_sat[1][2], OVII_flux_sat[2][2], \
                OVII_flux_sat[0][3], OVII_flux_sat[1][3], OVII_flux_sat[2][3], \
                OVII_flux_sat[0][4], OVII_flux_sat[1][4], OVII_flux_sat[2][4], \
                OVIII_flux_sat[0][0], OVIII_flux_sat[1][0], OVIII_flux_sat[2][0], \
                OVIII_flux_sat[0][1], OVIII_flux_sat[1][1], OVIII_flux_sat[2][1], \
                OVIII_flux_sat[0][2], OVIII_flux_sat[1][2], OVIII_flux_sat[2][2], \
                OVIII_flux_sat[0][3], OVIII_flux_sat[1][3], OVIII_flux_sat[2][3], \
                OVIII_flux_sat[0][4], OVIII_flux_sat[1][4], OVIII_flux_sat[2][4], \
                OIX_flux_sat[0][0], OIX_flux_sat[1][0], OIX_flux_sat[2][0], \
                OIX_flux_sat[0][1], OIX_flux_sat[1][1], OIX_flux_sat[2][1], \
                OIX_flux_sat[0][2], OIX_flux_sat[1][2], OIX_flux_sat[2][2], \
                OIX_flux_sat[0][3], OIX_flux_sat[1][3], OIX_flux_sat[2][3], \
                OIX_flux_sat[0][4], OIX_flux_sat[1][4], OIX_flux_sat[2][4]]
            fluxes_sat.add_row(new_row_sat)
            fluxes_sat = set_table_units(fluxes_sat)
            fluxes_sat.write(tablename + '_sat_frustum_' + frus_filename + fluxtype_filename + '.hdf5', path='all_data', serialize_meta=True, overwrite=True)

    fluxes_radial = set_table_units(fluxes_radial)
    fluxes_edges = set_table_units(fluxes_edges)

    # Save to file
    if (sat_radius!=0):
        fluxes_radial.write(tablename + '_nosat_frustum_' + frus_filename + fluxtype_filename + '.hdf5', path='all_data', serialize_meta=True, overwrite=True)
    else:
        fluxes_radial.write(tablename + '_frustum_' + frus_filename + fluxtype_filename + '.hdf5', path='all_data', serialize_meta=True, overwrite=True)
    fluxes_edges.write(tablename + '_edges_frustum_' + frus_filename + fluxtype_filename + '.hdf5', path='all_data', serialize_meta=True, overwrite=True)

    return "Fluxes have been calculated for snapshot " + snap + "!"

def calc_fluxes_cylinder(ds, snap, zsnap, dt, refine_width_kpc, tablename, surface_args, flux_types, **kwargs):
    '''This function calculates the fluxes across surfaces within a cylinder,
    with satellites removed, at a variety of heights or radii within the cylinder. It
    uses the dataset stored in 'ds', which is from time snapshot 'snap', has redshift
    'zsnap', and has width of the refine box in kpc 'refine_width_kpc', the time step between outputs
    'dt', and stores the fluxes in 'tablename'. 'surface_args' gives the properties of the cylinder.

    This function calculates the flux as the sum
    of all cells whose velocity and distance from the surface of interest indicate that the gas
    contained in that cell will be displaced across the surface of interest by the next timestep.
    That is, the properties of a cell contribute to the flux if it is no further from the surface of
    interest than v*dt where v is the cell's velocity normal to the surface and dt is the time
    between snapshots, which is dt = 5.38e6 yrs for the DD outputs. It is necessary to compute the
    flux this way if satellites are to be removed because they become 'holes' in the dataset
    and fluxes into/out of those holes need to be accounted for.'''

    sat = kwargs.get('sat')
    sat_radius = kwargs.get('sat_radius', 0.)
    halo_center_kpc2 = kwargs.get('halo_center_kpc2', ds.halo_center_kpc)

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
        names_dir = ('redshift', 'height')
        names_edge = ('redshift', 'bottom_edge', 'top_edge')
    elif (surface_args[6]=='radius'):
        names_dir = ('redshift', 'radius')
        names_edge = ('redshift', 'inner_radius', 'outer_radius')
    type_list_dir = ('f8','f8')
    type_list_edge = ('f8','f8','f8')
    if ('mass' in flux_types):
        new_names = ('net_mass_flux', 'net_metal_flux', \
        'mass_flux_in', 'mass_flux_out', 'metal_flux_in', 'metal_flux_out', \
        'net_cold_mass_flux', 'cold_mass_flux_in', 'cold_mass_flux_out', \
        'net_cool_mass_flux', 'cool_mass_flux_in', 'cool_mass_flux_out', \
        'net_warm_mass_flux', 'warm_mass_flux_in', 'warm_mass_flux_out', \
        'net_hot_mass_flux', 'hot_mass_flux_in', 'hot_mass_flux_out', \
        'net_cold_metal_flux', 'cold_metal_flux_in', 'cold_metal_flux_out', \
        'net_cool_metal_flux', 'cool_metal_flux_in', 'cool_metal_flux_out', \
        'net_warm_metal_flux', 'warm_metal_flux_in', 'warm_metal_flux_out', \
        'net_hot_metal_flux', 'hot_metal_flux_in', 'hot_metal_flux_out')
        new_types = ('f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', \
        'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', \
        'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8')
        names_dir += new_names
        names_edge += new_names
        type_list_dir += new_types
        type_list_edge += new_types
    if ('energy' in flux_types):
        new_names = ('net_kinetic_energy_flux', 'net_thermal_energy_flux', \
        'net_potential_energy_flux', 'net_radiative_energy_flux', 'net_total_energy_flux',\
        'kinetic_energy_flux_in', 'kinetic_energy_flux_out', \
        'thermal_energy_flux_in', 'thermal_energy_flux_out', \
        'potential_energy_flux_in', 'potential_energy_flux_out', \
        'radiative_energy_flux_in', 'radiative_energy_flux_out', \
        'total_energy_flux_in', 'total_energy_flux_out', \
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
        'net_cold_total_energy_flux', 'cold_total_energy_flux_in', 'cold_total_energy_flux_out', \
        'net_cool_total_energy_flux', 'cool_total_energy_flux_in', 'cool_total_energy_flux_out', \
        'net_warm_total_energy_flux', 'warm_total_energy_flux_in', 'warm_total_energy_flux_out', \
        'net_hot_total_energy_flux', 'hot_total_energy_flux_in', 'hot_total_energy_flux_out')
        new_types = ('f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', \
        'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', \
        'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', \
        'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', \
        'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', \
        'f8', 'f8', 'f8', 'f8', 'f8', 'f8')
        new_names_edge = ('net_kinetic_energy_flux', 'net_thermal_energy_flux', \
        'net_potential_energy_flux', 'net_total_energy_flux', \
        'kinetic_energy_flux_in', 'kinetic_energy_flux_out', \
        'thermal_energy_flux_in', 'thermal_energy_flux_out', \
        'potential_energy_flux_in', 'potential_energy_flux_out', \
        'total_energy_flux_in', 'total_energy_flux_out', \
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
        'net_cold_total_energy_flux', 'cold_total_energy_flux_in', 'cold_total_energy_flux_out', \
        'net_cool_total_energy_flux', 'cool_total_energy_flux_in', 'cool_total_energy_flux_out', \
        'net_warm_total_energy_flux', 'warm_total_energy_flux_in', 'warm_total_energy_flux_out', \
        'net_hot_total_energy_flux', 'hot_total_energy_flux_in', 'hot_total_energy_flux_out')
        new_types_edge = ('f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', \
        'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', \
        'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', \
        'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', \
        'f8', 'f8', 'f8', 'f8', 'f8', 'f8')
        names_dir += new_names
        names_edge += new_names_edge
        type_list_dir += new_types
        type_list_edge += new_types_edge
    if ('entropy' in flux_types):
        new_names = ('net_entropy_flux', 'entropy_flux_in', 'entropy_flux_out', \
        'net_cold_entropy_flux', 'cold_entropy_flux_in', 'cold_entropy_flux_out', \
        'net_cool_entropy_flux', 'cool_entropy_flux_in', 'cool_entropy_flux_out', \
        'net_warm_entropy_flux', 'warm_entropy_flux_in', 'warm_entropy_flux_out', \
        'net_hot_entropy_flux', 'hot_entropy_flux_in', 'hot_entropy_flux_out')
        new_types = ('f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', \
        'f8', 'f8', 'f8', 'f8', 'f8', 'f8')
        names_dir += new_names
        names_edge += new_names
        type_list_dir += new_types
        type_list_edge += new_types
    if ('O_ion_mass' in flux_types):
        new_names = ('net_O_flux', 'O_flux_in', 'O_flux_out', \
        'net_cold_O_flux', 'cold_O_flux_in', 'cold_O_flux_out', \
        'net_cool_O_flux', 'cool_O_flux_in', 'cool_O_flux_out', \
        'net_warm_O_flux', 'warm_O_flux_in', 'warm_O_flux_out', \
        'net_hot_O_flux', 'hot_O_flux_in', 'hot_O_flux_out', \
        'net_OI_flux', 'OI_flux_in', 'OI_flux_out', \
        'net_cold_OI_flux', 'cold_OI_flux_in', 'cold_OI_flux_out', \
        'net_cool_OI_flux', 'cool_OI_flux_in', 'cool_OI_flux_out', \
        'net_warm_OI_flux', 'warm_OI_flux_in', 'warm_OI_flux_out', \
        'net_hot_OI_flux', 'hot_OI_flux_in', 'hot_OI_flux_out', \
        'net_OII_flux', 'OII_flux_in', 'OII_flux_out', \
        'net_cold_OII_flux', 'cold_OII_flux_in', 'cold_OII_flux_out', \
        'net_cool_OII_flux', 'cool_OII_flux_in', 'cool_OII_flux_out', \
        'net_warm_OII_flux', 'warm_OII_flux_in', 'warm_OII_flux_out', \
        'net_hot_OII_flux', 'hot_OII_flux_in', 'hot_OII_flux_out', \
        'net_OIII_flux', 'OIII_flux_in', 'OIII_flux_out', \
        'net_cold_OIII_flux', 'cold_OIII_flux_in', 'cold_OIII_flux_out', \
        'net_cool_OIII_flux', 'cool_OIII_flux_in', 'cool_OIII_flux_out', \
        'net_warm_OIII_flux', 'warm_OIII_flux_in', 'warm_OIII_flux_out', \
        'net_hot_OIII_flux', 'hot_OIII_flux_in', 'hot_OIII_flux_out', \
        'net_OIV_flux', 'OIV_flux_in', 'OIV_flux_out', \
        'net_cold_OIV_flux', 'cold_OIV_flux_in', 'cold_OIV_flux_out', \
        'net_cool_OIV_flux', 'cool_OIV_flux_in', 'cool_OIV_flux_out', \
        'net_warm_OIV_flux', 'warm_OIV_flux_in', 'warm_OIV_flux_out', \
        'net_hot_OIV_flux', 'hot_OIV_flux_in', 'hot_OIV_flux_out', \
        'net_OV_flux', 'OV_flux_in', 'OV_flux_out', \
        'net_cold_OV_flux', 'cold_OV_flux_in', 'cold_OV_flux_out', \
        'net_cool_OV_flux', 'cool_OV_flux_in', 'cool_OV_flux_out', \
        'net_warm_OV_flux', 'warm_OV_flux_in', 'warm_OV_flux_out', \
        'net_hot_OV_flux', 'hot_OV_flux_in', 'hot_OV_flux_out', \
        'net_OVI_flux', 'OVI_flux_in', 'OVI_flux_out', \
        'net_cold_OVI_flux', 'cold_OVI_flux_in', 'cold_OVI_flux_out', \
        'net_cool_OVI_flux', 'cool_OVI_flux_in', 'cool_OVI_flux_out', \
        'net_warm_OVI_flux', 'warm_OVI_flux_in', 'warm_OVI_flux_out', \
        'net_hot_OVI_flux', 'hot_OVI_flux_in', 'hot_OVI_flux_out', \
        'net_OVII_flux', 'OVII_flux_in', 'OVII_flux_out', \
        'net_cold_OVII_flux', 'cold_OVII_flux_in', 'cold_OVII_flux_out', \
        'net_cool_OVII_flux', 'cool_OVII_flux_in', 'cool_OVII_flux_out', \
        'net_warm_OVII_flux', 'warm_OVII_flux_in', 'warm_OVII_flux_out', \
        'net_hot_OVII_flux', 'hot_OVII_flux_in', 'hot_OVII_flux_out', \
        'net_OVIII_flux', 'OVIII_flux_in', 'OVIII_flux_out', \
        'net_cold_OVIII_flux', 'cold_OVIII_flux_in', 'cold_OVIII_flux_out', \
        'net_cool_OVIII_flux', 'cool_OVIII_flux_in', 'cool_OVIII_flux_out', \
        'net_warm_OVIII_flux', 'warm_OVIII_flux_in', 'warm_OVIII_flux_out', \
        'net_hot_OVIII_flux', 'hot_OVIII_flux_in', 'hot_OVIII_flux_out', \
        'net_OIX_flux', 'OIX_flux_in', 'OIX_flux_out', \
        'net_cold_OIX_flux', 'cold_OIX_flux_in', 'cold_OIX_flux_out', \
        'net_cool_OIX_flux', 'cool_OIX_flux_in', 'cool_OIX_flux_out', \
        'net_warm_OIX_flux', 'warm_OIX_flux_in', 'warm_OIX_flux_out', \
        'net_hot_OIX_flux', 'hot_OIX_flux_in', 'hot_OIX_flux_out')
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
        names_dir += new_names
        names_edge += new_names
        type_list_dir += new_types
        type_list_edge += new_types
    fluxes_cylinder = Table(names=names_dir, dtype=type_list_dir)
    fluxes_edges = Table(names=names_edge, dtype=type_list_edge)
    if (sat_radius!=0):
        fluxes_sat = Table(names=names_edge, dtype=type_list_edge)

    # Define the heights or radii of the surfaces where we want to calculate fluxes
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
    new_x = x + vx*dt*(100./cmtopc*stoyr)
    new_y = y + vy*dt*(100./cmtopc*stoyr)
    new_z = z + vz*dt*(100./cmtopc*stoyr)
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
        cooling_time = sphere['gas','cooling_time'].in_units('yr').v
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

    # Cut data to only the cylinder considered here, stuff that leaves through edges of cylinder,
    # and stuff that comes in through edges of cylinder, where "edges" can be either the top and bottom
    # or the curved side, depending on which way we're calculating fluxes
    if (flip):
        cyl_filename = '-'
    else:
        cyl_filename = ''
    if (axis=='z'):
        norm_coord = z
        new_norm_coord = new_z
        rad_coord = np.sqrt(x**2. + y**2.)
        new_rad_coord = np.sqrt(new_x**2. + new_y**2.)
        norm_v = vz
        rad_v = vx*x/rad_coord + vy*y/rad_coord
        cyl_filename += 'z'
    if (axis=='x'):
        norm_coord = x
        new_norm_coord = new_x
        rad_coord = np.sqrt(y**2. + z**2.)
        new_rad_coord = np.sqrt(new_y**2. + new_z**2.)
        norm_v = vx
        rad_v = vz*z/rad_coord + vy*y/rad_coord
        cyl_filename += 'x'
    if (axis=='y'):
        norm_coord = y
        new_norm_coord = new_y
        rad_coord = np.sqrt(x**2. + z**2.)
        new_rad_coord = np.sqrt(new_x**2. + new_z**2.)
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
        new_x_disk = x_disk + vx_disk*dt*(100./cmtopc*stoyr)
        new_y_disk = y_disk + vy_disk*dt*(100./cmtopc*stoyr)
        new_z_disk = z_disk + vz_disk*dt*(100./cmtopc*stoyr)
        norm_coord = z_disk
        new_norm_coord = new_z_disk
        rad_coord = np.sqrt(x_disk**2. + y_disk**2.)
        new_rad_coord = np.sqrt(new_x_disk**2. + new_y_disk**2.)
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
        new_x_rot = rotationArr[0][0]*new_x + rotationArr[0][1]*new_y + rotationArr[0][2]*new_z
        new_y_rot = rotationArr[1][0]*new_x + rotationArr[1][1]*new_y + rotationArr[1][2]*new_z
        new_z_rot = rotationArr[2][0]*new_x + rotationArr[2][1]*new_y + rotationArr[2][2]*new_z
        vx_rot = rotationArr[0][0]*vx + rotationArr[0][1]*vy + rotationArr[0][2]*vz
        vy_rot = rotationArr[1][0]*vx + rotationArr[1][1]*vy + rotationArr[1][2]*vz
        vz_rot = rotationArr[2][0]*vx + rotationArr[2][1]*vy + rotationArr[2][2]*vz
        norm_coord = z_rot
        new_norm_coord = new_z_rot
        rad_coord = np.sqrt(x_rot**2. + y_rot**2.)
        new_rad_coord = np.sqrt(new_x_rot**2. + new_y_rot**2.)
        norm_v = vz_rot
        rad_v = vx_rot*x_rot/rad_coord + vy_rot*y_rot/rad_coord
        cyl_filename += 'axis_' + str(axis[0]) + '_' + str(axis[1]) + '_' + str(axis[2])
    if (surface_args[6]=='height'): cyl_filename += '_r' + str(surface_args[5]) + '_' + surface_args[6]
    elif (surface_args[6]=='radius'): cyl_filename += '_h' + str(np.abs(surface_args[4]-surface_args[3])) + '_' + surface_args[6]
    fluxtype_filename = ''
    if ('mass' in flux_types):
        fluxtype_filename += '_mass'
    if ('energy' in flux_types):
        fluxtype_filename += '_energy'
    if ('entropy' in flux_types):
        fluxtype_filename += '_entropy'
    if ('O_ion_mass' in flux_types):
        fluxtype_filename += '_Oions'

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
        snap_type = snap[-6:-4]
        if (snap_type=='RD'):
            # If using an RD output, calculate satellite fluxes as if satellites don't move in snap2, i.e. snap1 = snap2
            snap2 = int(snap[-4:])
        else:
            snap2 = int(snap[-4:])+1
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
        # Restrict to only things that start or end within the cylinder
        print('Cutting data to remove satellites')
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
        bool_fromsat = (inside_sat1 > 0) & (inside_sat2 == 0) & (new_norm_coord >= bottom_edge) & (new_norm_coord <= top_edge) & (new_rad_coord <= cyl_radius)
        bool_tosat = (inside_sat1 == 0) & (inside_sat2 > 0) & (norm_coord >= bottom_edge) & (norm_coord <= top_edge) & (rad_coord <= cyl_radius)

        norm_coord_nosat = norm_coord[bool_nosat]
        newnorm_nosat = new_norm_coord[bool_nosat]
        rad_coord_nosat = rad_coord[bool_nosat]
        newrad_nosat = new_rad_coord[bool_nosat]
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
            cooling_time_nosat = cooling_time[bool_nosat]
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
        newnorm_nosat = new_norm_coord
        rad_coord_nosat = rad_coord
        newrad_nosat = new_rad_coord
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

    # Cut satellite-removed data to cylinder of interest
    # These are nested lists where the index goes from 0 to 2 for [within cylinder, enterting cylinder, leaving cylinder]
    bool_cyl = (norm_coord_nosat >= bottom_edge) & (norm_coord_nosat <= top_edge) & (rad_coord_nosat <= cyl_radius) & \
               (newnorm_nosat >= bottom_edge) & (newnorm_nosat <= top_edge) & (newrad_nosat <= cyl_radius)
    if (surface_args[6]=='height'):
        bool_incyl = (rad_coord_nosat > cyl_radius) & (newrad_nosat <= cyl_radius) & (newnorm_nosat >= bottom_edge) & (newnorm_nosat <= top_edge)
        bool_outcyl = (rad_coord_nosat <= cyl_radius) & (newrad_nosat > cyl_radius) & (norm_coord_nosat >= bottom_edge) & (norm_coord_nosat <= top_edge)
    elif (surface_args[6]=='radius'):
        bool_incyl = ((norm_coord_nosat < bottom_edge) | (norm_coord_nosat > top_edge)) & ((newnorm_nosat >= bottom_edge) & (newnorm_nosat <= top_edge)) & (newrad_nosat <= cyl_radius)
        bool_outcyl = ((norm_coord_nosat >= bottom_edge) & (norm_coord_nosat <= top_edge)) & ((newnorm_nosat < bottom_edge) | (newnorm_nosat > top_edge)) & (rad_coord_nosat <= cyl_radius)

    norm_nosat_cyl = []
    newnorm_nosat_cyl = []
    rad_nosat_cyl = []
    newrad_nosat_cyl = []
    norm_v_nosat_cyl = []
    rad_v_nosat_cyl = []
    temperature_nosat_cyl = []
    if ('mass' in flux_types):
        mass_nosat_cyl = []
        metal_mass_nosat_cyl = []
    if ('energy' in flux_types):
        kinetic_energy_nosat_cyl = []
        thermal_energy_nosat_cyl = []
        potential_energy_nosat_cyl = []
        total_energy_nosat_cyl = []
        cooling_time_nosat_cyl = []
    if ('entropy' in flux_types):
        entropy_nosat_cyl = []
    if ('O_ion_mass' in flux_types):
        O_mass_nosat_cyl = []
        OI_mass_nosat_cyl = []
        OII_mass_nosat_cyl = []
        OIII_mass_nosat_cyl = []
        OIV_mass_nosat_cyl = []
        OV_mass_nosat_cyl = []
        OVI_mass_nosat_cyl = []
        OVII_mass_nosat_cyl = []
        OVIII_mass_nosat_cyl = []
        OIX_mass_nosat_cyl = []
    for j in range(3):
        if (j==0):
            norm_nosat_cyl.append(norm_coord_nosat[bool_cyl])
            newnorm_nosat_cyl.append(newnorm_nosat[bool_cyl])
            rad_nosat_cyl.append(rad_coord_nosat[bool_cyl])
            newrad_nosat_cyl.append(newrad_nosat[bool_cyl])
            norm_v_nosat_cyl.append(norm_v_nosat[bool_cyl])
            rad_v_nosat_cyl.append(rad_v_nosat[bool_cyl])
            temperature_nosat_cyl.append(temperature_nosat[bool_cyl])
            if ('mass' in flux_types):
                mass_nosat_cyl.append(mass_nosat[bool_cyl])
                metal_mass_nosat_cyl.append(metal_mass_nosat[bool_cyl])
            if ('energy' in flux_types):
                kinetic_energy_nosat_cyl.append(kinetic_energy_nosat[bool_cyl])
                thermal_energy_nosat_cyl.append(thermal_energy_nosat[bool_cyl])
                potential_energy_nosat_cyl.append(potential_energy_nosat[bool_cyl])
                total_energy_nosat_cyl.append(total_energy_nosat[bool_cyl])
                cooling_time_nosat_cyl.append(cooling_time_nosat[bool_cyl])
            if ('entropy' in flux_types):
                entropy_nosat_cyl.append(entropy_nosat[bool_cyl])
            if ('O_ion_mass' in flux_types):
                O_mass_nosat_cyl.append(O_mass_nosat[bool_cyl])
                OI_mass_nosat_cyl.append(OI_mass_nosat[bool_cyl])
                OII_mass_nosat_cyl.append(OII_mass_nosat[bool_cyl])
                OIII_mass_nosat_cyl.append(OIII_mass_nosat[bool_cyl])
                OIV_mass_nosat_cyl.append(OIV_mass_nosat[bool_cyl])
                OV_mass_nosat_cyl.append(OV_mass_nosat[bool_cyl])
                OVI_mass_nosat_cyl.append(OVI_mass_nosat[bool_cyl])
                OVII_mass_nosat_cyl.append(OVII_mass_nosat[bool_cyl])
                OVIII_mass_nosat_cyl.append(OVIII_mass_nosat[bool_cyl])
                OIX_mass_nosat_cyl.append(OIX_mass_nosat[bool_cyl])
        if (j==1):
            norm_nosat_cyl.append(norm_coord_nosat[bool_incyl])
            newnorm_nosat_cyl.append(newnorm_nosat[bool_incyl])
            rad_nosat_cyl.append(rad_coord_nosat[bool_incyl])
            newrad_nosat_cyl.append(newrad_nosat[bool_incyl])
            norm_v_nosat_cyl.append(norm_v_nosat[bool_incyl])
            rad_v_nosat_cyl.append(rad_v_nosat[bool_incyl])
            temperature_nosat_cyl.append(temperature_nosat[bool_incyl])
            if ('mass' in flux_types):
                mass_nosat_cyl.append(mass_nosat[bool_incyl])
                metal_mass_nosat_cyl.append(metal_mass_nosat[bool_incyl])
            if ('energy' in flux_types):
                kinetic_energy_nosat_cyl.append(kinetic_energy_nosat[bool_incyl])
                thermal_energy_nosat_cyl.append(thermal_energy_nosat[bool_incyl])
                potential_energy_nosat_cyl.append(potential_energy_nosat[bool_incyl])
                total_energy_nosat_cyl.append(total_energy_nosat[bool_incyl])
                cooling_time_nosat_cyl.append(cooling_time_nosat[bool_incyl])
            if ('entropy' in flux_types):
                entropy_nosat_cyl.append(entropy_nosat[bool_incyl])
            if ('O_ion_mass' in flux_types):
                O_mass_nosat_cyl.append(O_mass_nosat[bool_incyl])
                OI_mass_nosat_cyl.append(OI_mass_nosat[bool_incyl])
                OII_mass_nosat_cyl.append(OII_mass_nosat[bool_incyl])
                OIII_mass_nosat_cyl.append(OIII_mass_nosat[bool_incyl])
                OIV_mass_nosat_cyl.append(OIV_mass_nosat[bool_incyl])
                OV_mass_nosat_cyl.append(OV_mass_nosat[bool_incyl])
                OVI_mass_nosat_cyl.append(OVI_mass_nosat[bool_incyl])
                OVII_mass_nosat_cyl.append(OVII_mass_nosat[bool_incyl])
                OVIII_mass_nosat_cyl.append(OVIII_mass_nosat[bool_incyl])
                OIX_mass_nosat_cyl.append(OIX_mass_nosat[bool_incyl])
        if (j==2):
            norm_nosat_cyl.append(norm_coord_nosat[bool_outcyl])
            newnorm_nosat_cyl.append(newnorm_nosat[bool_outcyl])
            rad_nosat_cyl.append(rad_coord_nosat[bool_outcyl])
            newrad_nosat_cyl.append(newrad_nosat[bool_outcyl])
            norm_v_nosat_cyl.append(norm_v_nosat[bool_outcyl])
            rad_v_nosat_cyl.append(rad_v_nosat[bool_outcyl])
            temperature_nosat_cyl.append(temperature_nosat[bool_outcyl])
            if ('mass' in flux_types):
                mass_nosat_cyl.append(mass_nosat[bool_outcyl])
                metal_mass_nosat_cyl.append(metal_mass_nosat[bool_outcyl])
            if ('energy' in flux_types):
                kinetic_energy_nosat_cyl.append(kinetic_energy_nosat[bool_outcyl])
                thermal_energy_nosat_cyl.append(thermal_energy_nosat[bool_outcyl])
                potential_energy_nosat_cyl.append(potential_energy_nosat[bool_outcyl])
                total_energy_nosat_cyl.append(total_energy_nosat[bool_outcyl])
                cooling_time_nosat_cyl.append(cooling_time_nosat[bool_outcyl])
            if ('entropy' in flux_types):
                entropy_nosat_cyl.append(entropy_nosat[bool_outcyl])
            if ('O_ion_mass' in flux_types):
                O_mass_nosat_cyl.append(O_mass_nosat[bool_outcyl])
                OI_mass_nosat_cyl.append(OI_mass_nosat[bool_outcyl])
                OII_mass_nosat_cyl.append(OII_mass_nosat[bool_outcyl])
                OIII_mass_nosat_cyl.append(OIII_mass_nosat[bool_outcyl])
                OIV_mass_nosat_cyl.append(OIV_mass_nosat[bool_outcyl])
                OV_mass_nosat_cyl.append(OV_mass_nosat[bool_outcyl])
                OVI_mass_nosat_cyl.append(OVI_mass_nosat[bool_outcyl])
                OVII_mass_nosat_cyl.append(OVII_mass_nosat[bool_outcyl])
                OVIII_mass_nosat_cyl.append(OVIII_mass_nosat[bool_outcyl])
                OIX_mass_nosat_cyl.append(OIX_mass_nosat[bool_outcyl])

    # Cut satellite-removed cylinder data on temperature
    # These are lists of lists where the first index goes from 0 to 2 for
    # [within cylinder, entering cylinder, leaving cylinder] and the second index goes from 0 to 4 for
    # [all gas, cold, cool, warm, hot]
    if (sat_radius!=0):
        print('Cutting satellite-removed data on temperature')
    else:
        print('Cutting data on temperature')
    norm_nosat_cyl_Tcut = []
    newnorm_nosat_cyl_Tcut = []
    rad_nosat_cyl_Tcut = []
    newrad_nosat_cyl_Tcut = []
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
        cooling_time_nosat_cyl_Tcut = []
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
    for i in range(3):
        norm_nosat_cyl_Tcut.append([])
        newnorm_nosat_cyl_Tcut.append([])
        rad_nosat_cyl_Tcut.append([])
        newrad_nosat_cyl_Tcut.append([])
        norm_v_nosat_cyl_Tcut.append([])
        rad_v_nosat_cyl_Tcut.append([])
        if ('mass' in flux_types):
            mass_nosat_cyl_Tcut.append([])
            metal_mass_nosat_cyl_Tcut.append([])
        if ('energy' in flux_types):
            kinetic_energy_nosat_cyl_Tcut.append([])
            thermal_energy_nosat_cyl_Tcut.append([])
            potential_energy_nosat_cyl_Tcut.append([])
            total_energy_nosat_cyl_Tcut.append([])
            cooling_time_nosat_cyl_Tcut.append([])
        if ('entropy' in flux_types):
            entropy_nosat_cyl_Tcut.append([])
        if ('O_ion_mass' in flux_types):
            O_mass_nosat_cyl_Tcut.append([])
            OI_mass_nosat_cyl_Tcut.append([])
            OII_mass_nosat_cyl_Tcut.append([])
            OIII_mass_nosat_cyl_Tcut.append([])
            OIV_mass_nosat_cyl_Tcut.append([])
            OV_mass_nosat_cyl_Tcut.append([])
            OVI_mass_nosat_cyl_Tcut.append([])
            OVII_mass_nosat_cyl_Tcut.append([])
            OVIII_mass_nosat_cyl_Tcut.append([])
            OIX_mass_nosat_cyl_Tcut.append([])
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
            bool_temp_nosat_cyl = (temperature_nosat_cyl[i] < t_high) & (temperature_nosat_cyl[i] > t_low)
            norm_nosat_cyl_Tcut[i].append(norm_nosat_cyl[i][bool_temp_nosat_cyl])
            newnorm_nosat_cyl_Tcut[i].append(newnorm_nosat_cyl[i][bool_temp_nosat_cyl])
            rad_nosat_cyl_Tcut[i].append(rad_nosat_cyl[i][bool_temp_nosat_cyl])
            newrad_nosat_cyl_Tcut[i].append(newrad_nosat_cyl[i][bool_temp_nosat_cyl])
            norm_v_nosat_cyl_Tcut[i].append(norm_v_nosat_cyl[i][bool_temp_nosat_cyl])
            rad_v_nosat_cyl_Tcut[i].append(rad_v_nosat_cyl[i][bool_temp_nosat_cyl])
            if ('mass' in flux_types):
                mass_nosat_cyl_Tcut[i].append(mass_nosat_cyl[i][bool_temp_nosat_cyl])
                metal_mass_nosat_cyl_Tcut[i].append(metal_mass_nosat_cyl[i][bool_temp_nosat_cyl])
            if ('energy' in flux_types):
                kinetic_energy_nosat_cyl_Tcut[i].append(kinetic_energy_nosat_cyl[i][bool_temp_nosat_cyl])
                thermal_energy_nosat_cyl_Tcut[i].append(thermal_energy_nosat_cyl[i][bool_temp_nosat_cyl])
                potential_energy_nosat_cyl_Tcut[i].append(potential_energy_nosat_cyl[i][bool_temp_nosat_cyl])
                total_energy_nosat_cyl_Tcut[i].append(total_energy_nosat_cyl[i][bool_temp_nosat_cyl])
                cooling_time_nosat_cyl_Tcut[i].append(cooling_time_nosat_cyl[i][bool_temp_nosat_cyl])
            if ('entropy' in flux_types):
                entropy_nosat_cyl_Tcut[i].append(entropy_nosat_cyl[i][bool_temp_nosat_cyl])
            if ('O_ion_mass' in flux_types):
                O_mass_nosat_cyl_Tcut[i].append(O_mass_nosat_cyl[i][bool_temp_nosat_cyl])
                OI_mass_nosat_cyl_Tcut[i].append(OI_mass_nosat_cyl[i][bool_temp_nosat_cyl])
                OII_mass_nosat_cyl_Tcut[i].append(OII_mass_nosat_cyl[i][bool_temp_nosat_cyl])
                OIII_mass_nosat_cyl_Tcut[i].append(OIII_mass_nosat_cyl[i][bool_temp_nosat_cyl])
                OIV_mass_nosat_cyl_Tcut[i].append(OIV_mass_nosat_cyl[i][bool_temp_nosat_cyl])
                OV_mass_nosat_cyl_Tcut[i].append(OV_mass_nosat_cyl[i][bool_temp_nosat_cyl])
                OVI_mass_nosat_cyl_Tcut[i].append(OVI_mass_nosat_cyl[i][bool_temp_nosat_cyl])
                OVII_mass_nosat_cyl_Tcut[i].append(OVII_mass_nosat_cyl[i][bool_temp_nosat_cyl])
                OVIII_mass_nosat_cyl_Tcut[i].append(OVIII_mass_nosat_cyl[i][bool_temp_nosat_cyl])
                OIX_mass_nosat_cyl_Tcut[i].append(OIX_mass_nosat_cyl[i][bool_temp_nosat_cyl])

    # Cut data to things that cross into or out of satellites in the cylinder
    # These are lists of lists where the index goes from 0 to 1 for [from satellites, to satellites]
    if (sat_radius!=0):
        print('Cutting data to satellite fluxes')
        norm_sat = []
        newnorm_sat = []
        rad_sat = []
        newrad_sat = []
        norm_v_sat = []
        rad_v_sat = []
        temperature_sat = []
        if ('mass' in flux_types):
            mass_sat = []
            metal_mass_sat = []
        if ('energy' in flux_types):
            kinetic_energy_sat = []
            thermal_energy_sat = []
            potential_energy_sat = []
            total_energy_sat = []
        if ('entropy' in flux_types):
            entropy_sat = []
        if ('O_ion_mass' in flux_types):
            O_mass_sat = []
            OI_mass_sat = []
            OII_mass_sat = []
            OIII_mass_sat = []
            OIV_mass_sat = []
            OV_mass_sat = []
            OVI_mass_sat = []
            OVII_mass_sat = []
            OVIII_mass_sat = []
            OIX_mass_sat = []
        for j in range(2):
            if (j==0):
                norm_sat.append(norm_coord[bool_fromsat])
                newnorm_sat.append(new_norm_coord[bool_fromsat])
                rad_sat.append(rad_coord[bool_fromsat])
                newrad_sat.append(new_rad_coord[bool_fromsat])
                norm_v_sat.append(norm_v[bool_fromsat])
                rad_v_sat.append(rad_v[bool_fromsat])
                temperature_sat.append(temperature[bool_fromsat])
                if ('mass' in flux_types):
                    mass_sat.append(mass[bool_fromsat])
                    metal_mass_sat.append(metal_mass[bool_fromsat])
                if ('energy' in flux_types):
                    kinetic_energy_sat.append(kinetic_energy[bool_fromsat])
                    thermal_energy_sat.append(thermal_energy[bool_fromsat])
                    potential_energy_sat.append(potential_energy[bool_fromsat])
                    total_energy_sat.append(total_energy[bool_fromsat])
                if ('entropy' in flux_types):
                    entropy_sat.append(entropy[bool_fromsat])
                if ('O_ion_mass' in flux_types):
                    O_mass_sat.append(O_mass[bool_fromsat])
                    OI_mass_sat.append(OI_mass[bool_fromsat])
                    OII_mass_sat.append(OII_mass[bool_fromsat])
                    OIII_mass_sat.append(OIII_mass[bool_fromsat])
                    OIV_mass_sat.append(OIV_mass[bool_fromsat])
                    OV_mass_sat.append(OV_mass[bool_fromsat])
                    OVI_mass_sat.append(OVI_mass[bool_fromsat])
                    OVII_mass_sat.append(OVII_mass[bool_fromsat])
                    OVIII_mass_sat.append(OVIII_mass[bool_fromsat])
                    OIX_mass_sat.append(OIX_mass[bool_fromsat])
            if (j==1):
                norm_sat.append(norm_coord[bool_tosat])
                newnorm_sat.append(new_norm_coord[bool_tosat])
                rad_sat.append(rad_coord[bool_tosat])
                newrad_sat.append(new_rad_coord[bool_tosat])
                norm_v_sat.append(norm_v[bool_tosat])
                rad_v_sat.append(rad_v[bool_tosat])
                temperature_sat.append(temperature[bool_tosat])
                if ('mass' in flux_types):
                    mass_sat.append(mass[bool_tosat])
                    metal_mass_sat.append(metal_mass[bool_tosat])
                if ('energy' in flux_types):
                    kinetic_energy_sat.append(kinetic_energy[bool_tosat])
                    thermal_energy_sat.append(thermal_energy[bool_tosat])
                    potential_energy_sat.append(potential_energy[bool_tosat])
                    total_energy_sat.append(total_energy[bool_tosat])
                if ('entropy' in flux_types):
                    entropy_sat.append(entropy[bool_tosat])
                if ('O_ion_mass' in flux_types):
                    O_mass_sat.append(O_mass[bool_tosat])
                    OI_mass_sat.append(OI_mass[bool_tosat])
                    OII_mass_sat.append(OII_mass[bool_tosat])
                    OIII_mass_sat.append(OIII_mass[bool_tosat])
                    OIV_mass_sat.append(OIV_mass[bool_tosat])
                    OV_mass_sat.append(OV_mass[bool_tosat])
                    OVI_mass_sat.append(OVI_mass[bool_tosat])
                    OVII_mass_sat.append(OVII_mass[bool_tosat])
                    OVIII_mass_sat.append(OVIII_mass[bool_tosat])
                    OIX_mass_sat.append(OIX_mass[bool_tosat])

        # Cut stuff going into/out of satellites in the cylinder on temperature
        # These are nested lists where the first index goes from 0 to 1 for [from satellites, to satellites]
        # and the second index goes from 0 to 4 for [all gas, cold, cool, warm, hot]
        print('Cutting satellite fluxes on temperature')
        norm_sat_Tcut = []
        newnorm_sat_Tcut = []
        rad_sat_Tcut = []
        newrad_sat_Tcut = []
        norm_v_sat_Tcut = []
        rad_v_sat_Tcut = []
        if ('mass' in flux_types):
            mass_sat_Tcut = []
            metal_mass_sat_Tcut = []
        if ('energy' in flux_types):
            kinetic_energy_sat_Tcut = []
            thermal_energy_sat_Tcut = []
            potential_energy_sat_Tcut = []
            total_energy_sat_Tcut = []
        if ('entropy' in flux_types):
            entropy_sat_Tcut = []
        if ('O_ion_mass' in flux_types):
            O_mass_sat_Tcut = []
            OI_mass_sat_Tcut = []
            OII_mass_sat_Tcut = []
            OIII_mass_sat_Tcut = []
            OIV_mass_sat_Tcut = []
            OV_mass_sat_Tcut = []
            OVI_mass_sat_Tcut = []
            OVII_mass_sat_Tcut = []
            OVIII_mass_sat_Tcut = []
            OIX_mass_sat_Tcut = []
        for i in range(2):
            norm_sat_Tcut.append([])
            newnorm_sat_Tcut.append([])
            rad_sat_Tcut.append([])
            newrad_sat_Tcut.append([])
            norm_v_sat_Tcut.append([])
            rad_v_sat_Tcut.append([])
            if ('mass' in flux_types):
                mass_sat_Tcut.append([])
                metal_mass_sat_Tcut.append([])
            if ('energy' in flux_types):
                kinetic_energy_sat_Tcut.append([])
                thermal_energy_sat_Tcut.append([])
                potential_energy_sat_Tcut.append([])
                total_energy_sat_Tcut.append([])
            if ('entropy' in flux_types):
                entropy_sat_Tcut.append([])
            if ('O_ion_mass' in flux_types):
                O_mass_sat_Tcut.append([])
                OI_mass_sat_Tcut.append([])
                OII_mass_sat_Tcut.append([])
                OIII_mass_sat_Tcut.append([])
                OIV_mass_sat_Tcut.append([])
                OV_mass_sat_Tcut.append([])
                OVI_mass_sat_Tcut.append([])
                OVII_mass_sat_Tcut.append([])
                OVIII_mass_sat_Tcut.append([])
                OIX_mass_sat_Tcut.append([])
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
                norm_sat_Tcut[i].append(norm_sat[i][bool_temp_sat])
                newnorm_sat_Tcut[i].append(newnorm_sat[i][bool_temp_sat])
                rad_sat_Tcut[i].append(rad_sat[i][bool_temp_sat])
                newrad_sat_Tcut[i].append(newrad_sat[i][bool_temp_sat])
                norm_v_sat_Tcut[i].append(norm_v_sat[i][bool_temp_sat])
                rad_v_sat_Tcut[i].append(rad_v_sat[i][bool_temp_sat])
                if ('mass' in flux_types):
                    mass_sat_Tcut[i].append(mass_sat[i][bool_temp_sat])
                    metal_mass_sat_Tcut[i].append(metal_mass_sat[i][bool_temp_sat])
                if ('energy' in flux_types):
                    kinetic_energy_sat_Tcut[i].append(kinetic_energy_sat[i][bool_temp_sat])
                    thermal_energy_sat_Tcut[i].append(thermal_energy_sat[i][bool_temp_sat])
                    potential_energy_sat_Tcut[i].append(potential_energy_sat[i][bool_temp_sat])
                    total_energy_sat_Tcut[i].append(total_energy_sat[i][bool_temp_sat])
                if ('entropy' in flux_types):
                    entropy_sat_Tcut[i].append(entropy_sat[i][bool_temp_sat])
                if ('O_ion_mass' in flux_types):
                    O_mass_sat_Tcut[i].append(O_mass_sat[i][bool_temp_sat])
                    OI_mass_sat_Tcut[i].append(OI_mass_sat[i][bool_temp_sat])
                    OII_mass_sat_Tcut[i].append(OII_mass_sat[i][bool_temp_sat])
                    OIII_mass_sat_Tcut[i].append(OIII_mass_sat[i][bool_temp_sat])
                    OIV_mass_sat_Tcut[i].append(OIV_mass_sat[i][bool_temp_sat])
                    OV_mass_sat_Tcut[i].append(OV_mass_sat[i][bool_temp_sat])
                    OVI_mass_sat_Tcut[i].append(OVI_mass_sat[i][bool_temp_sat])
                    OVII_mass_sat_Tcut[i].append(OVII_mass_sat[i][bool_temp_sat])
                    OVIII_mass_sat_Tcut[i].append(OVIII_mass_sat[i][bool_temp_sat])
                    OIX_mass_sat_Tcut[i].append(OIX_mass_sat[i][bool_temp_sat])

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

        # Compute net, in, and out fluxes within the cylinder with satellites removed
        # These are nested lists where the first index goes from 0 to 2 for [net, in, out]
        # and the second index goes from 0 to 4 for [all, cold, cool, warm, hot]
        if ('mass' in flux_types):
            mass_flux_nosat = []
            metal_flux_nosat = []
        if ('energy' in flux_types):
            kinetic_energy_flux_nosat = []
            thermal_energy_flux_nosat = []
            potential_energy_flux_nosat = []
            total_energy_flux_nosat = []
        if ('entropy' in flux_types):
            entropy_flux_nosat = []
        if ('O_ion_mass' in flux_types):
            O_flux_nosat = []
            OI_flux_nosat = []
            OII_flux_nosat = []
            OIII_flux_nosat = []
            OIV_flux_nosat = []
            OV_flux_nosat = []
            OVI_flux_nosat = []
            OVII_flux_nosat = []
            OVIII_flux_nosat = []
            OIX_flux_nosat = []
        for j in range(3):
            if ('mass' in flux_types):
                mass_flux_nosat.append([])
                metal_flux_nosat.append([])
            if ('energy' in flux_types):
                kinetic_energy_flux_nosat.append([])
                thermal_energy_flux_nosat.append([])
                potential_energy_flux_nosat.append([])
                total_energy_flux_nosat.append([])
            if ('entropy' in flux_types):
                entropy_flux_nosat.append([])
            if ('O_ion_mass' in flux_types):
                O_flux_nosat.append([])
                OI_flux_nosat.append([])
                OII_flux_nosat.append([])
                OIII_flux_nosat.append([])
                OIV_flux_nosat.append([])
                OV_flux_nosat.append([])
                OVI_flux_nosat.append([])
                OVII_flux_nosat.append([])
                OVIII_flux_nosat.append([])
                OIX_flux_nosat.append([])
            for k in range(5):
                if (surface_args[6]=='radius'):
                    bool_in_s = (rad_nosat_cyl_Tcut[0][k] > inner_surface) & (newrad_nosat_cyl_Tcut[0][k] < inner_surface)
                    bool_out_s = (rad_nosat_cyl_Tcut[0][k] < inner_surface) & (newrad_nosat_cyl_Tcut[0][k] > inner_surface)
                elif (surface_args[6]=='height'):
                    bool_in_s = (norm_nosat_cyl_Tcut[0][k] > inner_surface) & (newnorm_nosat_cyl_Tcut[0][k] < inner_surface)
                    bool_out_s = (norm_nosat_cyl_Tcut[0][k] < inner_surface) & (newnorm_nosat_cyl_Tcut[0][k] > inner_surface)
                if (j==0):
                    if ('mass' in flux_types):
                        mass_flux_nosat[j].append((np.sum(mass_nosat_cyl_Tcut[0][k][bool_out_s]) - \
                          np.sum(mass_nosat_cyl_Tcut[0][k][bool_in_s]))/dt)
                        metal_flux_nosat[j].append((np.sum(metal_mass_nosat_cyl_Tcut[0][k][bool_out_s]) - \
                          np.sum(metal_mass_nosat_cyl_Tcut[0][k][bool_in_s]))/dt)
                    if ('energy' in flux_types):
                        kinetic_energy_flux_nosat[j].append((np.sum(kinetic_energy_nosat_cyl_Tcut[0][k][bool_out_s]) - \
                          np.sum(kinetic_energy_nosat_cyl_Tcut[0][k][bool_in_s]))/dt)
                        thermal_energy_flux_nosat[j].append((np.sum(thermal_energy_nosat_cyl_Tcut[0][k][bool_out_s]) - \
                          np.sum(thermal_energy_nosat_cyl_Tcut[0][k][bool_in_s]))/dt)
                        potential_energy_flux_nosat[j].append((np.sum(potential_energy_nosat_cyl_Tcut[0][k][bool_out_s]) - \
                          np.sum(potential_energy_nosat_cyl_Tcut[0][k][bool_in_s]))/dt)
                        total_energy_flux_nosat[j].append((np.sum(total_energy_nosat_cyl_Tcut[0][k][bool_out_s]) - \
                          np.sum(total_energy_nosat_cyl_Tcut[0][k][bool_in_s]))/dt)
                    if ('entropy' in flux_types):
                        entropy_flux_nosat[j].append((np.sum(entropy_nosat_cyl_Tcut[0][k][bool_out_s]) - \
                          np.sum(entropy_nosat_cyl_Tcut[0][k][bool_in_s]))/dt)
                    if ('O_ion_mass' in flux_types):
                        O_flux_nosat[j].append((np.sum(O_mass_nosat_cyl_Tcut[0][k][bool_out_s]) - \
                          np.sum(O_mass_nosat_cyl_Tcut[0][k][bool_in_s]))/dt)
                        OI_flux_nosat[j].append((np.sum(OI_mass_nosat_cyl_Tcut[0][k][bool_out_s]) - \
                          np.sum(OI_mass_nosat_cyl_Tcut[0][k][bool_in_s]))/dt)
                        OII_flux_nosat[j].append((np.sum(OII_mass_nosat_cyl_Tcut[0][k][bool_out_s]) - \
                          np.sum(OII_mass_nosat_cyl_Tcut[0][k][bool_in_s]))/dt)
                        OIII_flux_nosat[j].append((np.sum(OIII_mass_nosat_cyl_Tcut[0][k][bool_out_s]) - \
                          np.sum(OIII_mass_nosat_cyl_Tcut[0][k][bool_in_s]))/dt)
                        OIV_flux_nosat[j].append((np.sum(OIV_mass_nosat_cyl_Tcut[0][k][bool_out_s]) - \
                          np.sum(OIV_mass_nosat_cyl_Tcut[0][k][bool_in_s]))/dt)
                        OV_flux_nosat[j].append((np.sum(OV_mass_nosat_cyl_Tcut[0][k][bool_out_s]) - \
                          np.sum(OV_mass_nosat_cyl_Tcut[0][k][bool_in_s]))/dt)
                        OVI_flux_nosat[j].append((np.sum(OVI_mass_nosat_cyl_Tcut[0][k][bool_out_s]) - \
                          np.sum(OVI_mass_nosat_cyl_Tcut[0][k][bool_in_s]))/dt)
                        OVII_flux_nosat[j].append((np.sum(OVII_mass_nosat_cyl_Tcut[0][k][bool_out_s]) - \
                          np.sum(OVII_mass_nosat_cyl_Tcut[0][k][bool_in_s]))/dt)
                        OVIII_flux_nosat[j].append((np.sum(OVIII_mass_nosat_cyl_Tcut[0][k][bool_out_s]) - \
                          np.sum(OVIII_mass_nosat_cyl_Tcut[0][k][bool_in_s]))/dt)
                        OIX_flux_nosat[j].append((np.sum(OIX_mass_nosat_cyl_Tcut[0][k][bool_out_s]) - \
                          np.sum(OIX_mass_nosat_cyl_Tcut[0][k][bool_in_s]))/dt)
                if (j==1):
                    if ('mass' in flux_types):
                        mass_flux_nosat[j].append(-np.sum(mass_nosat_cyl_Tcut[0][k][bool_in_s])/dt)
                        metal_flux_nosat[j].append(-np.sum(metal_mass_nosat_cyl_Tcut[0][k][bool_in_s])/dt)
                    if ('energy' in flux_types):
                        kinetic_energy_flux_nosat[j].append(-np.sum(kinetic_energy_nosat_cyl_Tcut[0][k][bool_in_s])/dt)
                        thermal_energy_flux_nosat[j].append(-np.sum(thermal_energy_nosat_cyl_Tcut[0][k][bool_in_s])/dt)
                        potential_energy_flux_nosat[j].append(-np.sum(potential_energy_nosat_cyl_Tcut[0][k][bool_in_s])/dt)
                        total_energy_flux_nosat[j].append(-np.sum(total_energy_nosat_cyl_Tcut[0][k][bool_in_s])/dt)
                    if ('entropy' in flux_types):
                        entropy_flux_nosat[j].append(-np.sum(entropy_nosat_cyl_Tcut[0][k][bool_in_s])/dt)
                    if ('O_ion_mass' in flux_types):
                        O_flux_nosat[j].append(-np.sum(O_mass_nosat_cyl_Tcut[0][k][bool_in_s])/dt)
                        OI_flux_nosat[j].append(-np.sum(OI_mass_nosat_cyl_Tcut[0][k][bool_in_s])/dt)
                        OII_flux_nosat[j].append(-np.sum(OII_mass_nosat_cyl_Tcut[0][k][bool_in_s])/dt)
                        OIII_flux_nosat[j].append(-np.sum(OIII_mass_nosat_cyl_Tcut[0][k][bool_in_s])/dt)
                        OIV_flux_nosat[j].append(-np.sum(OIV_mass_nosat_cyl_Tcut[0][k][bool_in_s])/dt)
                        OV_flux_nosat[j].append(-np.sum(OV_mass_nosat_cyl_Tcut[0][k][bool_in_s])/dt)
                        OVI_flux_nosat[j].append(-np.sum(OVI_mass_nosat_cyl_Tcut[0][k][bool_in_s])/dt)
                        OVII_flux_nosat[j].append(-np.sum(OVII_mass_nosat_cyl_Tcut[0][k][bool_in_s])/dt)
                        OVIII_flux_nosat[j].append(-np.sum(OVIII_mass_nosat_cyl_Tcut[0][k][bool_in_s])/dt)
                        OIX_flux_nosat[j].append(-np.sum(OIX_mass_nosat_cyl_Tcut[0][k][bool_in_s])/dt)
                if (j==2):
                    if ('mass' in flux_types):
                        mass_flux_nosat[j].append(np.sum(mass_nosat_cyl_Tcut[0][k][bool_out_s])/dt)
                        metal_flux_nosat[j].append(np.sum(metal_mass_nosat_cyl_Tcut[0][k][bool_out_s])/dt)
                    if ('energy' in flux_types):
                        kinetic_energy_flux_nosat[j].append(np.sum(kinetic_energy_nosat_cyl_Tcut[0][k][bool_out_s])/dt)
                        thermal_energy_flux_nosat[j].append(np.sum(thermal_energy_nosat_cyl_Tcut[0][k][bool_out_s])/dt)
                        potential_energy_flux_nosat[j].append(np.sum(potential_energy_nosat_cyl_Tcut[0][k][bool_out_s])/dt)
                        total_energy_flux_nosat[j].append(np.sum(total_energy_nosat_cyl_Tcut[0][k][bool_out_s])/dt)
                    if ('entropy' in flux_types):
                        entropy_flux_nosat[j].append(np.sum(entropy_nosat_cyl_Tcut[0][k][bool_out_s])/dt)
                    if ('O_ion_mass' in flux_types):
                        O_flux_nosat[j].append(np.sum(O_mass_nosat_cyl_Tcut[0][k][bool_out_s])/dt)
                        OI_flux_nosat[j].append(np.sum(OI_mass_nosat_cyl_Tcut[0][k][bool_out_s])/dt)
                        OII_flux_nosat[j].append(np.sum(OII_mass_nosat_cyl_Tcut[0][k][bool_out_s])/dt)
                        OIII_flux_nosat[j].append(np.sum(OIII_mass_nosat_cyl_Tcut[0][k][bool_out_s])/dt)
                        OIV_flux_nosat[j].append(np.sum(OIV_mass_nosat_cyl_Tcut[0][k][bool_out_s])/dt)
                        OV_flux_nosat[j].append(np.sum(OV_mass_nosat_cyl_Tcut[0][k][bool_out_s])/dt)
                        OVI_flux_nosat[j].append(np.sum(OVI_mass_nosat_cyl_Tcut[0][k][bool_out_s])/dt)
                        OVII_flux_nosat[j].append(np.sum(OVII_mass_nosat_cyl_Tcut[0][k][bool_out_s])/dt)
                        OVIII_flux_nosat[j].append(np.sum(OVIII_mass_nosat_cyl_Tcut[0][k][bool_out_s])/dt)
                        OIX_flux_nosat[j].append(np.sum(OIX_mass_nosat_cyl_Tcut[0][k][bool_out_s])/dt)

        # Compute fluxes from and to satellites (and net) within the cylinder between inner_surface and outer_surface
        # These are nested lists where the first index goes from 0 to 2 for [net, from, to]
        # and the second index goes from 0 to 4 for [all, cold, cool, warm, hot]
        if (i < len(surfaces)-1):
            if (sat_radius!=0) and ('mass' in flux_types):
                mass_flux_sat = []
                metal_flux_sat = []
            if (sat_radius!=0) and ('energy' in flux_types):
                kinetic_energy_flux_sat = []
                thermal_energy_flux_sat = []
                potential_energy_flux_sat = []
                total_energy_flux_sat = []
            if (sat_radius!=0) and ('entropy' in flux_types):
                entropy_flux_sat = []
            if ('energy' in flux_types):
                radiative_energy_flux_nosat = []
            if (sat_radius!=0) and ('O_ion_mass' in flux_types):
                O_flux_sat = []
                OI_flux_sat = []
                OII_flux_sat = []
                OIII_flux_sat = []
                OIV_flux_sat = []
                OV_flux_sat = []
                OVI_flux_sat = []
                OVII_flux_sat = []
                OVIII_flux_sat = []
                OIX_flux_sat = []
            for j in range(3):
                if (sat_radius!=0) and ('mass' in flux_types):
                    mass_flux_sat.append([])
                    metal_flux_sat.append([])
                if (sat_radius!=0) and ('energy' in flux_types):
                    kinetic_energy_flux_sat.append([])
                    thermal_energy_flux_sat.append([])
                    potential_energy_flux_sat.append([])
                    total_energy_flux_sat.append([])
                if (sat_radius!=0) and ('entropy' in flux_types):
                    entropy_flux_sat.append([])
                if ('energy' in flux_types):
                    radiative_energy_flux_nosat.append([])
                if (sat_radius!=0) and ('O_ion_mass' in flux_types):
                    O_flux_sat.append([])
                    OI_flux_sat.append([])
                    OII_flux_sat.append([])
                    OIII_flux_sat.append([])
                    OIV_flux_sat.append([])
                    OV_flux_sat.append([])
                    OVI_flux_sat.append([])
                    OVII_flux_sat.append([])
                    OVIII_flux_sat.append([])
                    OIX_flux_sat.append([])
                for k in range(5):
                    if (sat_radius!=0):
                        if (surface_args[6]=='radius'):
                            bool_from = (newrad_sat_Tcut[0][k]>inner_surface) & (newrad_sat_Tcut[0][k]<outer_surface)
                            bool_to = (rad_sat_Tcut[1][k]>inner_surface) & (rad_sat_Tcut[1][k]<outer_surface)
                        elif (surface_args[6]=='height'):
                            bool_from = (newnorm_sat_Tcut[0][k]>inner_surface) & (newnorm_sat_Tcut[0][k]<outer_surface)
                            bool_to = (norm_sat_Tcut[1][k]>inner_surface) & (norm_sat_Tcut[1][k]<outer_surface)
                    if ('energy' in flux_types):
                        if (surface_args[6]=='radius'):
                            bool_s = (rad_nosat_cyl_Tcut[0][k]>inner_surface) & (rad_nosat_cyl_Tcut[0][k]<outer_surface)
                            vel_surface = rad_v_nosat_cyl_Tcut
                        elif (surface_args[6]=='height'):
                            bool_s = (norm_nosat_cyl_Tcut[0][k]>inner_surface) & (norm_nosat_cyl_Tcut[0][k]<outer_surface)
                            vel_surface = norm_v_nosat_cyl_Tcut
                    if (j==0):
                        if (sat_radius!=0) and ('mass' in flux_types):
                            mass_flux_sat[j].append((np.sum(mass_sat_Tcut[0][k][bool_from]) - \
                                                    np.sum(mass_sat_Tcut[1][k][bool_to]))/dt)
                            metal_flux_sat[j].append((np.sum(metal_mass_sat_Tcut[0][k][bool_from]) - \
                                                    np.sum(metal_mass_sat_Tcut[1][k][bool_to]))/dt)
                        if (sat_radius!=0) and ('energy' in flux_types):
                            kinetic_energy_flux_sat[j].append((np.sum(kinetic_energy_sat_Tcut[0][k][bool_from]) - \
                                                              np.sum(kinetic_energy_sat_Tcut[1][k][bool_to]))/dt)
                            thermal_energy_flux_sat[j].append((np.sum(thermal_energy_sat_Tcut[0][k][bool_from]) - \
                                                              np.sum(thermal_energy_sat_Tcut[1][k][bool_to]))/dt)
                            potential_energy_flux_sat[j].append((np.sum(potential_energy_sat_Tcut[0][k][bool_from]) - \
                                                                np.sum(potential_energy_sat_Tcut[1][k][bool_to]))/dt)
                            total_energy_flux_sat[j].append((np.sum(total_energy_sat_Tcut[0][k][bool_from]) - \
                                                                np.sum(total_energy_sat_Tcut[1][k][bool_to]))/dt)
                        if (sat_radius!=0) and ('entropy' in flux_types):
                            entropy_flux_sat[j].append((np.sum(entropy_sat_Tcut[0][k][bool_from]) - \
                                                        np.sum(entropy_sat_Tcut[1][k][bool_to]))/dt)
                        if ('energy' in flux_types):
                            radiative_energy_flux_nosat[j].append(np.sum(thermal_energy_nosat_cyl_Tcut[0][k][bool_s] * \
                              mass_nosat_cyl_Tcut[0][k][bool_s]*gtoMsun /cooling_time_nosat_cyl_Tcut[0][k][bool_s]))
                        if (sat_radius!=0) and ('O_ion_mass' in flux_types):
                            O_flux_sat[j].append((np.sum(O_mass_sat_Tcut[0][k][bool_from]) - \
                              np.sum(O_mass_sat_Tcut[1][k][bool_to]))/dt)
                            OI_flux_sat[j].append((np.sum(OI_mass_sat_Tcut[0][k][bool_from]) - \
                              np.sum(OI_mass_sat_Tcut[1][k][bool_to]))/dt)
                            OII_flux_sat[j].append((np.sum(OII_mass_sat_Tcut[0][k][bool_from]) - \
                              np.sum(OII_mass_sat_Tcut[1][k][bool_to]))/dt)
                            OIII_flux_sat[j].append((np.sum(OIII_mass_sat_Tcut[0][k][bool_from]) - \
                              np.sum(OIII_mass_sat_Tcut[1][k][bool_to]))/dt)
                            OIV_flux_sat[j].append((np.sum(OIV_mass_sat_Tcut[0][k][bool_from]) - \
                              np.sum(OIV_mass_sat_Tcut[1][k][bool_to]))/dt)
                            OV_flux_sat[j].append((np.sum(OV_mass_sat_Tcut[0][k][bool_from]) - \
                              np.sum(OV_mass_sat_Tcut[1][k][bool_to]))/dt)
                            OVI_flux_sat[j].append((np.sum(OVI_mass_sat_Tcut[0][k][bool_from]) - \
                              np.sum(OVI_mass_sat_Tcut[1][k][bool_to]))/dt)
                            OVII_flux_sat[j].append((np.sum(OVII_mass_sat_Tcut[0][k][bool_from]) - \
                              np.sum(OVII_mass_sat_Tcut[1][k][bool_to]))/dt)
                            OVIII_flux_sat[j].append((np.sum(OVIII_mass_sat_Tcut[0][k][bool_from]) - \
                              np.sum(OVIII_mass_sat_Tcut[1][k][bool_to]))/dt)
                            OIX_flux_sat[j].append((np.sum(OIX_mass_sat_Tcut[0][k][bool_from]) - \
                              np.sum(OIX_mass_sat_Tcut[1][k][bool_to]))/dt)
                    if (j==1):
                        if (sat_radius!=0) and ('mass' in flux_types):
                            mass_flux_sat[j].append(np.sum(mass_sat_Tcut[0][k][bool_from])/dt)
                            metal_flux_sat[j].append(np.sum(metal_mass_sat_Tcut[0][k][bool_from])/dt)
                        if (sat_radius!=0) and ('energy' in flux_types):
                            kinetic_energy_flux_sat[j].append(np.sum(kinetic_energy_sat_Tcut[0][k][bool_from])/dt)
                            thermal_energy_flux_sat[j].append(np.sum(thermal_energy_sat_Tcut[0][k][bool_from])/dt)
                            potential_energy_flux_sat[j].append(np.sum(potential_energy_sat_Tcut[0][k][bool_from])/dt)
                            total_energy_flux_sat[j].append(np.sum(total_energy_sat_Tcut[0][k][bool_from])/dt)
                        if (sat_radius!=0) and ('entropy' in flux_types):
                            entropy_flux_sat[j].append(np.sum(entropy_sat_Tcut[0][k][bool_from])/dt)
                        if ('energy' in flux_types):
                            radiative_energy_flux_nosat[j].append(np.sum( \
                              thermal_energy_nosat_cyl_Tcut[0][k][bool_s & (vel_surface[0][k]<0.)] * \
                              mass_nosat_cyl_Tcut[0][k][bool_s & (vel_surface[0][k]<0.)]*gtoMsun / \
                              cooling_time_nosat_cyl_Tcut[0][k][bool_s & (vel_surface[0][k]<0.)]))
                        if (sat_radius!=0) and ('O_ion_mass' in flux_types):
                            O_flux_sat[j].append(np.sum(O_mass_sat_Tcut[0][k][bool_from])/dt)
                            OI_flux_sat[j].append(np.sum(OI_mass_sat_Tcut[0][k][bool_from])/dt)
                            OII_flux_sat[j].append(np.sum(OII_mass_sat_Tcut[0][k][bool_from])/dt)
                            OIII_flux_sat[j].append(np.sum(OIII_mass_sat_Tcut[0][k][bool_from])/dt)
                            OIV_flux_sat[j].append(np.sum(OIV_mass_sat_Tcut[0][k][bool_from])/dt)
                            OV_flux_sat[j].append(np.sum(OV_mass_sat_Tcut[0][k][bool_from])/dt)
                            OVI_flux_sat[j].append(np.sum(OVI_mass_sat_Tcut[0][k][bool_from])/dt)
                            OVII_flux_sat[j].append(np.sum(OVII_mass_sat_Tcut[0][k][bool_from])/dt)
                            OVIII_flux_sat[j].append(np.sum(OVIII_mass_sat_Tcut[0][k][bool_from])/dt)
                            OIX_flux_sat[j].append(np.sum(OIX_mass_sat_Tcut[0][k][bool_from])/dt)
                    if (j==2):
                        if (sat_radius!=0) and ('mass' in flux_types):
                            mass_flux_sat[j].append(-np.sum(mass_sat_Tcut[1][k][bool_to])/dt)
                            metal_flux_sat[j].append(-np.sum(metal_mass_sat_Tcut[1][k][bool_to])/dt)
                        if (sat_radius!=0) and ('energy' in flux_types):
                            kinetic_energy_flux_sat[j].append(-np.sum(kinetic_energy_sat_Tcut[1][k][bool_to])/dt)
                            thermal_energy_flux_sat[j].append(-np.sum(thermal_energy_sat_Tcut[1][k][bool_to])/dt)
                            potential_energy_flux_sat[j].append(-np.sum(potential_energy_sat_Tcut[1][k][bool_to])/dt)
                            total_energy_flux_sat[j].append(-np.sum(total_energy_sat_Tcut[1][k][bool_to])/dt)
                        if (sat_radius!=0) and ('entropy' in flux_types):
                            entropy_flux_sat[j].append(-np.sum(entropy_sat_Tcut[1][k][bool_to])/dt)
                        if ('energy' in flux_types):
                            radiative_energy_flux_nosat[j].append(np.sum( \
                              thermal_energy_nosat_cyl_Tcut[0][k][bool_s & (vel_surface[0][k]>0.)] * \
                              mass_nosat_cyl_Tcut[0][k][bool_s & (vel_surface[0][k]>0.)]*gtoMsun / \
                              cooling_time_nosat_cyl_Tcut[0][k][bool_s & (vel_surface[0][k]>0.)]))
                        if (sat_radius!=0) and ('O_ion_mass' in flux_types):
                            O_flux_sat[j].append(-np.sum(O_mass_sat_Tcut[1][k][bool_to])/dt)
                            OI_flux_sat[j].append(-np.sum(OI_mass_sat_Tcut[1][k][bool_to])/dt)
                            OII_flux_sat[j].append(-np.sum(OII_mass_sat_Tcut[1][k][bool_to])/dt)
                            OIII_flux_sat[j].append(-np.sum(OIII_mass_sat_Tcut[1][k][bool_to])/dt)
                            OIV_flux_sat[j].append(-np.sum(OIV_mass_sat_Tcut[1][k][bool_to])/dt)
                            OV_flux_sat[j].append(-np.sum(OV_mass_sat_Tcut[1][k][bool_to])/dt)
                            OVI_flux_sat[j].append(-np.sum(OVI_mass_sat_Tcut[1][k][bool_to])/dt)
                            OVII_flux_sat[j].append(-np.sum(OVII_mass_sat_Tcut[1][k][bool_to])/dt)
                            OVIII_flux_sat[j].append(-np.sum(OVIII_mass_sat_Tcut[1][k][bool_to])/dt)
                            OIX_flux_sat[j].append(-np.sum(OIX_mass_sat_Tcut[1][k][bool_to])/dt)

        # Compute fluxes through edges of cylinder between inner_surface and outer_surface
        # These are nested lists where the first index goes from 0 to 2 for [net, in, out]
        # and the second index goes from 0 to 4 for [all, cold, cool, warm, hot]
        if (i < len(surfaces)-1):
            if ('mass' in flux_types):
                mass_flux_edge = []
                metal_flux_edge = []
            if ('energy' in flux_types):
                kinetic_energy_flux_edge = []
                thermal_energy_flux_edge = []
                potential_energy_flux_edge = []
                total_energy_flux_edge = []
            if ('entropy' in flux_types):
                entropy_flux_edge = []
            if ('O_ion_mass' in flux_types):
                O_flux_edge = []
                OI_flux_edge = []
                OII_flux_edge = []
                OIII_flux_edge = []
                OIV_flux_edge = []
                OV_flux_edge = []
                OVI_flux_edge = []
                OVII_flux_edge = []
                OVIII_flux_edge = []
                OIX_flux_edge = []
            for j in range(3):
                if ('mass' in flux_types):
                    mass_flux_edge.append([])
                    metal_flux_edge.append([])
                if ('energy' in flux_types):
                    kinetic_energy_flux_edge.append([])
                    thermal_energy_flux_edge.append([])
                    potential_energy_flux_edge.append([])
                    total_energy_flux_edge.append([])
                if ('entropy' in flux_types):
                    entropy_flux_edge.append([])
                if ('O_ion_mass' in flux_types):
                    O_flux_edge.append([])
                    OI_flux_edge.append([])
                    OII_flux_edge.append([])
                    OIII_flux_edge.append([])
                    OIV_flux_edge.append([])
                    OV_flux_edge.append([])
                    OVI_flux_edge.append([])
                    OVII_flux_edge.append([])
                    OVIII_flux_edge.append([])
                    OIX_flux_edge.append([])
                for k in range(5):
                    if (surface_args[6]=='radius'):
                        bool_in = (newrad_nosat_cyl_Tcut[1][k]>inner_surface) & (newrad_nosat_cyl_Tcut[1][k]<outer_surface)
                        bool_out = (rad_nosat_cyl_Tcut[2][k]>inner_surface) & (rad_nosat_cyl_Tcut[2][k]<outer_surface)
                    elif (surface_args[6]=='height'):
                        bool_in = (newnorm_nosat_cyl_Tcut[1][k]>inner_surface) & (newnorm_nosat_cyl_Tcut[1][k]<outer_surface)
                        bool_out = (norm_nosat_cyl_Tcut[2][k]>inner_surface) & (norm_nosat_cyl_Tcut[2][k]<outer_surface)
                    if (j==0):
                        if ('mass' in flux_types):
                            mass_flux_edge[j].append((np.sum(mass_nosat_cyl_Tcut[1][k][bool_in]) - \
                                                    np.sum(mass_nosat_cyl_Tcut[2][k][bool_out]))/dt)
                            metal_flux_edge[j].append((np.sum(metal_mass_nosat_cyl_Tcut[1][k][bool_in]) - \
                                                    np.sum(metal_mass_nosat_cyl_Tcut[2][k][bool_out]))/dt)
                        if ('energy' in flux_types):
                            kinetic_energy_flux_edge[j].append((np.sum(kinetic_energy_nosat_cyl_Tcut[1][k][bool_in]) - \
                                                              np.sum(kinetic_energy_nosat_cyl_Tcut[2][k][bool_out]))/dt)
                            thermal_energy_flux_edge[j].append((np.sum(thermal_energy_nosat_cyl_Tcut[1][k][bool_in]) - \
                                                              np.sum(thermal_energy_nosat_cyl_Tcut[2][k][bool_out]))/dt)
                            potential_energy_flux_edge[j].append((np.sum(potential_energy_nosat_cyl_Tcut[1][k][bool_in]) - \
                                                                np.sum(potential_energy_nosat_cyl_Tcut[2][k][bool_out]))/dt)
                            total_energy_flux_edge[j].append((np.sum(total_energy_nosat_cyl_Tcut[1][k][bool_in]) - \
                                                                np.sum(total_energy_nosat_cyl_Tcut[2][k][bool_out]))/dt)
                        if ('entropy' in flux_types):
                            entropy_flux_edge[j].append((np.sum(entropy_nosat_cyl_Tcut[1][k][bool_in]) - \
                                                        np.sum(entropy_nosat_cyl_Tcut[2][k][bool_out]))/dt)
                        if ('O_ion_mass' in flux_types):
                            O_flux_edge[j].append((np.sum(O_mass_nosat_cyl_Tcut[1][k][bool_in]) - \
                              np.sum(O_mass_nosat_cyl_Tcut[2][k][bool_out]))/dt)
                            OI_flux_edge[j].append((np.sum(OI_mass_nosat_cyl_Tcut[1][k][bool_in]) - \
                              np.sum(OI_mass_nosat_cyl_Tcut[2][k][bool_out]))/dt)
                            OII_flux_edge[j].append((np.sum(OII_mass_nosat_cyl_Tcut[1][k][bool_in]) - \
                              np.sum(OII_mass_nosat_cyl_Tcut[2][k][bool_out]))/dt)
                            OIII_flux_edge[j].append((np.sum(OIII_mass_nosat_cyl_Tcut[1][k][bool_in]) - \
                              np.sum(OIII_mass_nosat_cyl_Tcut[2][k][bool_out]))/dt)
                            OIV_flux_edge[j].append((np.sum(OIV_mass_nosat_cyl_Tcut[1][k][bool_in]) - \
                              np.sum(OIV_mass_nosat_cyl_Tcut[2][k][bool_out]))/dt)
                            OV_flux_edge[j].append((np.sum(OV_mass_nosat_cyl_Tcut[1][k][bool_in]) - \
                              np.sum(OV_mass_nosat_cyl_Tcut[2][k][bool_out]))/dt)
                            OVI_flux_edge[j].append((np.sum(OVI_mass_nosat_cyl_Tcut[1][k][bool_in]) - \
                              np.sum(OVI_mass_nosat_cyl_Tcut[2][k][bool_out]))/dt)
                            OVII_flux_edge[j].append((np.sum(OVII_mass_nosat_cyl_Tcut[1][k][bool_in]) - \
                              np.sum(OVII_mass_nosat_cyl_Tcut[2][k][bool_out]))/dt)
                            OVIII_flux_edge[j].append((np.sum(OVIII_mass_nosat_cyl_Tcut[1][k][bool_in]) - \
                              np.sum(OVIII_mass_nosat_cyl_Tcut[2][k][bool_out]))/dt)
                            OIX_flux_edge[j].append((np.sum(OIX_mass_nosat_cyl_Tcut[1][k][bool_in]) - \
                              np.sum(OIX_mass_nosat_cyl_Tcut[2][k][bool_out]))/dt)
                    if (j==1):
                        if ('mass' in flux_types):
                            mass_flux_edge[j].append(np.sum(mass_nosat_cyl_Tcut[1][k][bool_in])/dt)
                            metal_flux_edge[j].append(np.sum(metal_mass_nosat_cyl_Tcut[1][k][bool_in])/dt)
                        if ('energy' in flux_types):
                            kinetic_energy_flux_edge[j].append(np.sum(kinetic_energy_nosat_cyl_Tcut[1][k][bool_in])/dt)
                            thermal_energy_flux_edge[j].append(np.sum(thermal_energy_nosat_cyl_Tcut[1][k][bool_in])/dt)
                            potential_energy_flux_edge[j].append(np.sum(potential_energy_nosat_cyl_Tcut[1][k][bool_in])/dt)
                            total_energy_flux_edge[j].append(np.sum(total_energy_nosat_cyl_Tcut[1][k][bool_in])/dt)
                        if ('entropy' in flux_types):
                            entropy_flux_edge[j].append(np.sum(entropy_nosat_cyl_Tcut[1][k][bool_in])/dt)
                        if ('O_ion_mass' in flux_types):
                            O_flux_edge[j].append(np.sum(O_mass_nosat_cyl_Tcut[1][k][bool_in])/dt)
                            OI_flux_edge[j].append(np.sum(OI_mass_nosat_cyl_Tcut[1][k][bool_in])/dt)
                            OII_flux_edge[j].append(np.sum(OII_mass_nosat_cyl_Tcut[1][k][bool_in])/dt)
                            OIII_flux_edge[j].append(np.sum(OIII_mass_nosat_cyl_Tcut[1][k][bool_in])/dt)
                            OIV_flux_edge[j].append(np.sum(OIV_mass_nosat_cyl_Tcut[1][k][bool_in])/dt)
                            OV_flux_edge[j].append(np.sum(OV_mass_nosat_cyl_Tcut[1][k][bool_in])/dt)
                            OVI_flux_edge[j].append(np.sum(OVI_mass_nosat_cyl_Tcut[1][k][bool_in])/dt)
                            OVII_flux_edge[j].append(np.sum(OVII_mass_nosat_cyl_Tcut[1][k][bool_in])/dt)
                            OVIII_flux_edge[j].append(np.sum(OVIII_mass_nosat_cyl_Tcut[1][k][bool_in])/dt)
                            OIX_flux_edge[j].append(np.sum(OIX_mass_nosat_cyl_Tcut[1][k][bool_in])/dt)
                    if (j==2):
                        if ('mass' in flux_types):
                            mass_flux_edge[j].append(-np.sum(mass_nosat_cyl_Tcut[2][k][bool_out])/dt)
                            metal_flux_edge[j].append(-np.sum(metal_mass_nosat_cyl_Tcut[2][k][bool_out])/dt)
                        if ('energy' in flux_types):
                            kinetic_energy_flux_edge[j].append(-np.sum(kinetic_energy_nosat_cyl_Tcut[2][k][bool_out])/dt)
                            thermal_energy_flux_edge[j].append(-np.sum(thermal_energy_nosat_cyl_Tcut[2][k][bool_out])/dt)
                            potential_energy_flux_edge[j].append(-np.sum(potential_energy_nosat_cyl_Tcut[2][k][bool_out])/dt)
                            total_energy_flux_edge[j].append(-np.sum(total_energy_nosat_cyl_Tcut[2][k][bool_out])/dt)
                        if ('entropy' in flux_types):
                            entropy_flux_edge[j].append(-np.sum(entropy_nosat_cyl_Tcut[2][k][bool_out])/dt)
                        if ('O_ion_mass' in flux_types):
                            O_flux_edge[j].append(-np.sum(O_mass_nosat_cyl_Tcut[2][k][bool_out])/dt)
                            OI_flux_edge[j].append(-np.sum(OI_mass_nosat_cyl_Tcut[2][k][bool_out])/dt)
                            OII_flux_edge[j].append(-np.sum(OII_mass_nosat_cyl_Tcut[2][k][bool_out])/dt)
                            OIII_flux_edge[j].append(-np.sum(OIII_mass_nosat_cyl_Tcut[2][k][bool_out])/dt)
                            OIV_flux_edge[j].append(-np.sum(OIV_mass_nosat_cyl_Tcut[2][k][bool_out])/dt)
                            OV_flux_edge[j].append(-np.sum(OV_mass_nosat_cyl_Tcut[2][k][bool_out])/dt)
                            OVI_flux_edge[j].append(-np.sum(OVI_mass_nosat_cyl_Tcut[2][k][bool_out])/dt)
                            OVII_flux_edge[j].append(-np.sum(OVII_mass_nosat_cyl_Tcut[2][k][bool_out])/dt)
                            OVIII_flux_edge[j].append(-np.sum(OVIII_mass_nosat_cyl_Tcut[2][k][bool_out])/dt)
                            OIX_flux_edge[j].append(-np.sum(OIX_mass_nosat_cyl_Tcut[2][k][bool_out])/dt)

        # Add everything to the tables
        new_row_s = [zsnap, inner_surface]
        new_row_edge = [zsnap, inner_surface, outer_surface]
        if ('mass' in flux_types):
            new_row_s += [mass_flux_nosat[0][0], metal_flux_nosat[0][0], \
            mass_flux_nosat[1][0], mass_flux_nosat[2][0], metal_flux_nosat[1][0], metal_flux_nosat[2][0], \
            mass_flux_nosat[0][1], mass_flux_nosat[1][1], mass_flux_nosat[2][1], \
            mass_flux_nosat[0][2], mass_flux_nosat[1][2], mass_flux_nosat[2][2], \
            mass_flux_nosat[0][3], mass_flux_nosat[1][3], mass_flux_nosat[2][3], \
            mass_flux_nosat[0][4], mass_flux_nosat[1][4], mass_flux_nosat[2][4], \
            metal_flux_nosat[0][1], metal_flux_nosat[1][1], metal_flux_nosat[2][1], \
            metal_flux_nosat[0][2], metal_flux_nosat[1][2], metal_flux_nosat[2][2], \
            metal_flux_nosat[0][3], metal_flux_nosat[1][3], metal_flux_nosat[2][3], \
            metal_flux_nosat[0][4], metal_flux_nosat[1][4], metal_flux_nosat[2][4]]
            new_row_edge += [mass_flux_edge[0][0], metal_flux_edge[0][0], \
            mass_flux_edge[1][0], mass_flux_edge[2][0], metal_flux_edge[1][0], metal_flux_edge[2][0], \
            mass_flux_edge[0][1], mass_flux_edge[1][1], mass_flux_edge[2][1], \
            mass_flux_edge[0][2], mass_flux_edge[1][2], mass_flux_edge[2][2], \
            mass_flux_edge[0][3], mass_flux_edge[1][3], mass_flux_edge[2][3], \
            mass_flux_edge[0][4], mass_flux_edge[1][4], mass_flux_edge[2][4], \
            metal_flux_edge[0][1], metal_flux_edge[1][1], metal_flux_edge[2][1], \
            metal_flux_edge[0][2], metal_flux_edge[1][2], metal_flux_edge[2][2], \
            metal_flux_edge[0][3], metal_flux_edge[1][3], metal_flux_edge[2][3], \
            metal_flux_edge[0][4], metal_flux_edge[1][4], metal_flux_edge[2][4]]
        if ('energy' in flux_types):
            new_row_s += [kinetic_energy_flux_nosat[0][0], thermal_energy_flux_nosat[0][0], \
            potential_energy_flux_nosat[0][0], radiative_energy_flux_nosat[0][0], \
            total_energy_flux_nosat[0][0], \
            kinetic_energy_flux_nosat[1][0], kinetic_energy_flux_nosat[2][0], \
            thermal_energy_flux_nosat[1][0], thermal_energy_flux_nosat[2][0], \
            potential_energy_flux_nosat[1][0], potential_energy_flux_nosat[2][0], \
            total_energy_flux_nosat[1][0], total_energy_flux_nosat[2][0], \
            radiative_energy_flux_nosat[1][0], radiative_energy_flux_nosat[2][0], \
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
            total_energy_flux_nosat[0][1], total_energy_flux_nosat[1][1], total_energy_flux_nosat[2][1], \
            total_energy_flux_nosat[0][2], total_energy_flux_nosat[1][2], total_energy_flux_nosat[2][2], \
            total_energy_flux_nosat[0][3], total_energy_flux_nosat[1][3], total_energy_flux_nosat[2][3], \
            total_energy_flux_nosat[0][4], total_energy_flux_nosat[1][4], total_energy_flux_nosat[2][4]]
            new_row_edge += [kinetic_energy_flux_edge[0][0], thermal_energy_flux_edge[0][0], \
            potential_energy_flux_edge[0][0], total_energy_flux_edge[0][0], \
            kinetic_energy_flux_edge[1][0], kinetic_energy_flux_edge[2][0], \
            thermal_energy_flux_edge[1][0], thermal_energy_flux_edge[2][0], \
            potential_energy_flux_edge[1][0], potential_energy_flux_edge[2][0], \
            total_energy_flux_edge[1][0], total_energy_flux_edge[2][0], \
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
            total_energy_flux_edge[0][1], total_energy_flux_edge[1][1], total_energy_flux_edge[2][1], \
            total_energy_flux_edge[0][2], total_energy_flux_edge[1][2], total_energy_flux_edge[2][2], \
            total_energy_flux_edge[0][3], total_energy_flux_edge[1][3], total_energy_flux_edge[2][3], \
            total_energy_flux_edge[0][4], total_energy_flux_edge[1][4], total_energy_flux_edge[2][4]]
        if ('entropy' in flux_types):
            new_row_s += [entropy_flux_nosat[0][0], entropy_flux_nosat[1][0], entropy_flux_nosat[2][0], \
            entropy_flux_nosat[0][1], entropy_flux_nosat[1][1], entropy_flux_nosat[2][1], \
            entropy_flux_nosat[0][2], entropy_flux_nosat[1][2], entropy_flux_nosat[2][2], \
            entropy_flux_nosat[0][3], entropy_flux_nosat[1][3], entropy_flux_nosat[2][3], \
            entropy_flux_nosat[0][4], entropy_flux_nosat[1][4], entropy_flux_nosat[2][4]]
            new_row_edge += [entropy_flux_edge[0][0], entropy_flux_edge[1][0], entropy_flux_edge[2][0], \
            entropy_flux_edge[0][1], entropy_flux_edge[1][1], entropy_flux_edge[2][1], \
            entropy_flux_edge[0][2], entropy_flux_edge[1][2], entropy_flux_edge[2][2], \
            entropy_flux_edge[0][3], entropy_flux_edge[1][3], entropy_flux_edge[2][3], \
            entropy_flux_edge[0][4], entropy_flux_edge[1][4], entropy_flux_edge[2][4]]
        if ('O_ion_mass' in flux_types):
            new_row_s += [O_flux_nosat[0][0], O_flux_nosat[1][0], O_flux_nosat[2][0], \
            O_flux_nosat[0][1], O_flux_nosat[1][1], O_flux_nosat[2][1], \
            O_flux_nosat[0][2], O_flux_nosat[1][2], O_flux_nosat[2][2], \
            O_flux_nosat[0][3], O_flux_nosat[1][3], O_flux_nosat[2][3], \
            O_flux_nosat[0][4], O_flux_nosat[1][4], O_flux_nosat[2][4], \
            OI_flux_nosat[0][0], OI_flux_nosat[1][0], OI_flux_nosat[2][0], \
            OI_flux_nosat[0][1], OI_flux_nosat[1][1], OI_flux_nosat[2][1], \
            OI_flux_nosat[0][2], OI_flux_nosat[1][2], OI_flux_nosat[2][2], \
            OI_flux_nosat[0][3], OI_flux_nosat[1][3], OI_flux_nosat[2][3], \
            OI_flux_nosat[0][4], OI_flux_nosat[1][4], OI_flux_nosat[2][4], \
            OII_flux_nosat[0][0], OII_flux_nosat[1][0], OII_flux_nosat[2][0], \
            OII_flux_nosat[0][1], OII_flux_nosat[1][1], OII_flux_nosat[2][1], \
            OII_flux_nosat[0][2], OII_flux_nosat[1][2], OII_flux_nosat[2][2], \
            OII_flux_nosat[0][3], OII_flux_nosat[1][3], OII_flux_nosat[2][3], \
            OII_flux_nosat[0][4], OII_flux_nosat[1][4], OII_flux_nosat[2][4], \
            OIII_flux_nosat[0][0], OIII_flux_nosat[1][0], OIII_flux_nosat[2][0], \
            OIII_flux_nosat[0][1], OIII_flux_nosat[1][1], OIII_flux_nosat[2][1], \
            OIII_flux_nosat[0][2], OIII_flux_nosat[1][2], OIII_flux_nosat[2][2], \
            OIII_flux_nosat[0][3], OIII_flux_nosat[1][3], OIII_flux_nosat[2][3], \
            OIII_flux_nosat[0][4], OIII_flux_nosat[1][4], OIII_flux_nosat[2][4], \
            OIV_flux_nosat[0][0], OIV_flux_nosat[1][0], OIV_flux_nosat[2][0], \
            OIV_flux_nosat[0][1], OIV_flux_nosat[1][1], OIV_flux_nosat[2][1], \
            OIV_flux_nosat[0][2], OIV_flux_nosat[1][2], OIV_flux_nosat[2][2], \
            OIV_flux_nosat[0][3], OIV_flux_nosat[1][3], OIV_flux_nosat[2][3], \
            OIV_flux_nosat[0][4], OIV_flux_nosat[1][4], OIV_flux_nosat[2][4], \
            OV_flux_nosat[0][0], OV_flux_nosat[1][0], OV_flux_nosat[2][0], \
            OV_flux_nosat[0][1], OV_flux_nosat[1][1], OV_flux_nosat[2][1], \
            OV_flux_nosat[0][2], OV_flux_nosat[1][2], OV_flux_nosat[2][2], \
            OV_flux_nosat[0][3], OV_flux_nosat[1][3], OV_flux_nosat[2][3], \
            OV_flux_nosat[0][4], OV_flux_nosat[1][4], OV_flux_nosat[2][4], \
            OVI_flux_nosat[0][0], OVI_flux_nosat[1][0], OVI_flux_nosat[2][0], \
            OVI_flux_nosat[0][1], OVI_flux_nosat[1][1], OVI_flux_nosat[2][1], \
            OVI_flux_nosat[0][2], OVI_flux_nosat[1][2], OVI_flux_nosat[2][2], \
            OVI_flux_nosat[0][3], OVI_flux_nosat[1][3], OVI_flux_nosat[2][3], \
            OVI_flux_nosat[0][4], OVI_flux_nosat[1][4], OVI_flux_nosat[2][4], \
            OVII_flux_nosat[0][0], OVII_flux_nosat[1][0], OVII_flux_nosat[2][0], \
            OVII_flux_nosat[0][1], OVII_flux_nosat[1][1], OVII_flux_nosat[2][1], \
            OVII_flux_nosat[0][2], OVII_flux_nosat[1][2], OVII_flux_nosat[2][2], \
            OVII_flux_nosat[0][3], OVII_flux_nosat[1][3], OVII_flux_nosat[2][3], \
            OVII_flux_nosat[0][4], OVII_flux_nosat[1][4], OVII_flux_nosat[2][4], \
            OVIII_flux_nosat[0][0], OVIII_flux_nosat[1][0], OVIII_flux_nosat[2][0], \
            OVIII_flux_nosat[0][1], OVIII_flux_nosat[1][1], OVIII_flux_nosat[2][1], \
            OVIII_flux_nosat[0][2], OVIII_flux_nosat[1][2], OVIII_flux_nosat[2][2], \
            OVIII_flux_nosat[0][3], OVIII_flux_nosat[1][3], OVIII_flux_nosat[2][3], \
            OVIII_flux_nosat[0][4], OVIII_flux_nosat[1][4], OVIII_flux_nosat[2][4], \
            OIX_flux_nosat[0][0], OIX_flux_nosat[1][0], OIX_flux_nosat[2][0], \
            OIX_flux_nosat[0][1], OIX_flux_nosat[1][1], OIX_flux_nosat[2][1], \
            OIX_flux_nosat[0][2], OIX_flux_nosat[1][2], OIX_flux_nosat[2][2], \
            OIX_flux_nosat[0][3], OIX_flux_nosat[1][3], OIX_flux_nosat[2][3], \
            OIX_flux_nosat[0][4], OIX_flux_nosat[1][4], OIX_flux_nosat[2][4]]
            new_row_edge += [O_flux_edge[0][0], O_flux_edge[1][0], O_flux_edge[2][0], \
            O_flux_edge[0][1], O_flux_edge[1][1], O_flux_edge[2][1], \
            O_flux_edge[0][2], O_flux_edge[1][2], O_flux_edge[2][2], \
            O_flux_edge[0][3], O_flux_edge[1][3], O_flux_edge[2][3], \
            O_flux_edge[0][4], O_flux_edge[1][4], O_flux_edge[2][4], \
            OI_flux_edge[0][0], OI_flux_edge[1][0], OI_flux_edge[2][0], \
            OI_flux_edge[0][1], OI_flux_edge[1][1], OI_flux_edge[2][1], \
            OI_flux_edge[0][2], OI_flux_edge[1][2], OI_flux_edge[2][2], \
            OI_flux_edge[0][3], OI_flux_edge[1][3], OI_flux_edge[2][3], \
            OI_flux_edge[0][4], OI_flux_edge[1][4], OI_flux_edge[2][4], \
            OII_flux_edge[0][0], OII_flux_edge[1][0], OII_flux_edge[2][0], \
            OII_flux_edge[0][1], OII_flux_edge[1][1], OII_flux_edge[2][1], \
            OII_flux_edge[0][2], OII_flux_edge[1][2], OII_flux_edge[2][2], \
            OII_flux_edge[0][3], OII_flux_edge[1][3], OII_flux_edge[2][3], \
            OII_flux_edge[0][4], OII_flux_edge[1][4], OII_flux_edge[2][4], \
            OIII_flux_edge[0][0], OIII_flux_edge[1][0], OIII_flux_edge[2][0], \
            OIII_flux_edge[0][1], OIII_flux_edge[1][1], OIII_flux_edge[2][1], \
            OIII_flux_edge[0][2], OIII_flux_edge[1][2], OIII_flux_edge[2][2], \
            OIII_flux_edge[0][3], OIII_flux_edge[1][3], OIII_flux_edge[2][3], \
            OIII_flux_edge[0][4], OIII_flux_edge[1][4], OIII_flux_edge[2][4], \
            OIV_flux_edge[0][0], OIV_flux_edge[1][0], OIV_flux_edge[2][0], \
            OIV_flux_edge[0][1], OIV_flux_edge[1][1], OIV_flux_edge[2][1], \
            OIV_flux_edge[0][2], OIV_flux_edge[1][2], OIV_flux_edge[2][2], \
            OIV_flux_edge[0][3], OIV_flux_edge[1][3], OIV_flux_edge[2][3], \
            OIV_flux_edge[0][4], OIV_flux_edge[1][4], OIV_flux_edge[2][4], \
            OV_flux_edge[0][0], OV_flux_edge[1][0], OV_flux_edge[2][0], \
            OV_flux_edge[0][1], OV_flux_edge[1][1], OV_flux_edge[2][1], \
            OV_flux_edge[0][2], OV_flux_edge[1][2], OV_flux_edge[2][2], \
            OV_flux_edge[0][3], OV_flux_edge[1][3], OV_flux_edge[2][3], \
            OV_flux_edge[0][4], OV_flux_edge[1][4], OV_flux_edge[2][4], \
            OVI_flux_edge[0][0], OVI_flux_edge[1][0], OVI_flux_edge[2][0], \
            OVI_flux_edge[0][1], OVI_flux_edge[1][1], OVI_flux_edge[2][1], \
            OVI_flux_edge[0][2], OVI_flux_edge[1][2], OVI_flux_edge[2][2], \
            OVI_flux_edge[0][3], OVI_flux_edge[1][3], OVI_flux_edge[2][3], \
            OVI_flux_edge[0][4], OVI_flux_edge[1][4], OVI_flux_edge[2][4], \
            OVII_flux_edge[0][0], OVII_flux_edge[1][0], OVII_flux_edge[2][0], \
            OVII_flux_edge[0][1], OVII_flux_edge[1][1], OVII_flux_edge[2][1], \
            OVII_flux_edge[0][2], OVII_flux_edge[1][2], OVII_flux_edge[2][2], \
            OVII_flux_edge[0][3], OVII_flux_edge[1][3], OVII_flux_edge[2][3], \
            OVII_flux_edge[0][4], OVII_flux_edge[1][4], OVII_flux_edge[2][4], \
            OVIII_flux_edge[0][0], OVIII_flux_edge[1][0], OVIII_flux_edge[2][0], \
            OVIII_flux_edge[0][1], OVIII_flux_edge[1][1], OVIII_flux_edge[2][1], \
            OVIII_flux_edge[0][2], OVIII_flux_edge[1][2], OVIII_flux_edge[2][2], \
            OVIII_flux_edge[0][3], OVIII_flux_edge[1][3], OVIII_flux_edge[2][3], \
            OVIII_flux_edge[0][4], OVIII_flux_edge[1][4], OVIII_flux_edge[2][4], \
            OIX_flux_edge[0][0], OIX_flux_edge[1][0], OIX_flux_edge[2][0], \
            OIX_flux_edge[0][1], OIX_flux_edge[1][1], OIX_flux_edge[2][1], \
            OIX_flux_edge[0][2], OIX_flux_edge[1][2], OIX_flux_edge[2][2], \
            OIX_flux_edge[0][3], OIX_flux_edge[1][3], OIX_flux_edge[2][3], \
            OIX_flux_edge[0][4], OIX_flux_edge[1][4], OIX_flux_edge[2][4]]
        fluxes_cylinder.add_row(new_row_s)
        fluxes_edges.add_row(new_row_edge)
        if (sat_radius!=0):
            new_row_sat = [zsnap, inner_surface, outer_surface]
            if ('mass' in flux_types):
                new_row_sat += [mass_flux_sat[0][0], metal_flux_sat[0][0], \
                mass_flux_sat[1][0], mass_flux_sat[2][0], metal_flux_sat[1][0], metal_flux_sat[2][0], \
                mass_flux_sat[0][1], mass_flux_sat[1][1], mass_flux_sat[2][1], \
                mass_flux_sat[0][2], mass_flux_sat[1][2], mass_flux_sat[2][2], \
                mass_flux_sat[0][3], mass_flux_sat[1][3], mass_flux_sat[2][3], \
                mass_flux_sat[0][4], mass_flux_sat[1][4], mass_flux_sat[2][4], \
                metal_flux_sat[0][1], metal_flux_sat[1][1], metal_flux_sat[2][1], \
                metal_flux_sat[0][2], metal_flux_sat[1][2], metal_flux_sat[2][2], \
                metal_flux_sat[0][3], metal_flux_sat[1][3], metal_flux_sat[2][3], \
                metal_flux_sat[0][4], metal_flux_sat[1][4], metal_flux_sat[2][4]]
            if ('energy' in flux_types):
                new_row_sat += [kinetic_energy_flux_sat[0][0], thermal_energy_flux_sat[0][0], \
                potential_energy_flux_sat[0][0], total_energy_flux_sat[0][0], kinetic_energy_flux_sat[1][0], kinetic_energy_flux_sat[2][0], \
                thermal_energy_flux_sat[1][0], thermal_energy_flux_sat[2][0], \
                potential_energy_flux_sat[1][0], potential_energy_flux_sat[2][0], \
                total_energy_flux_sat[1][0], total_energy_flux_sat[2][0], \
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
                total_energy_flux_sat[0][1], total_energy_flux_sat[1][1], total_energy_flux_sat[2][1], \
                total_energy_flux_sat[0][2], total_energy_flux_sat[1][2], total_energy_flux_sat[2][2], \
                total_energy_flux_sat[0][3], total_energy_flux_sat[1][3], total_energy_flux_sat[2][3], \
                total_energy_flux_sat[0][4], total_energy_flux_sat[1][4], total_energy_flux_sat[2][4]]
            if ('entropy' in flux_types):
                new_row_sat += [entropy_flux_sat[0][0], entropy_flux_sat[1][0], entropy_flux_sat[2][0], \
                entropy_flux_sat[0][1], entropy_flux_sat[1][1], entropy_flux_sat[2][1], \
                entropy_flux_sat[0][2], entropy_flux_sat[1][2], entropy_flux_sat[2][2], \
                entropy_flux_sat[0][3], entropy_flux_sat[1][3], entropy_flux_sat[2][3], \
                entropy_flux_sat[0][4], entropy_flux_sat[1][4], entropy_flux_sat[2][4]]
            if ('O_ion_mass' in flux_types):
                new_row_sat += [O_flux_sat[0][0], O_flux_sat[1][0], O_flux_sat[2][0], \
                O_flux_sat[0][1], O_flux_sat[1][1], O_flux_sat[2][1], \
                O_flux_sat[0][2], O_flux_sat[1][2], O_flux_sat[2][2], \
                O_flux_sat[0][3], O_flux_sat[1][3], O_flux_sat[2][3], \
                O_flux_sat[0][4], O_flux_sat[1][4], O_flux_sat[2][4], \
                OI_flux_sat[0][0], OI_flux_sat[1][0], OI_flux_sat[2][0], \
                OI_flux_sat[0][1], OI_flux_sat[1][1], OI_flux_sat[2][1], \
                OI_flux_sat[0][2], OI_flux_sat[1][2], OI_flux_sat[2][2], \
                OI_flux_sat[0][3], OI_flux_sat[1][3], OI_flux_sat[2][3], \
                OI_flux_sat[0][4], OI_flux_sat[1][4], OI_flux_sat[2][4], \
                OII_flux_sat[0][0], OII_flux_sat[1][0], OII_flux_sat[2][0], \
                OII_flux_sat[0][1], OII_flux_sat[1][1], OII_flux_sat[2][1], \
                OII_flux_sat[0][2], OII_flux_sat[1][2], OII_flux_sat[2][2], \
                OII_flux_sat[0][3], OII_flux_sat[1][3], OII_flux_sat[2][3], \
                OII_flux_sat[0][4], OII_flux_sat[1][4], OII_flux_sat[2][4], \
                OIII_flux_sat[0][0], OIII_flux_sat[1][0], OIII_flux_sat[2][0], \
                OIII_flux_sat[0][1], OIII_flux_sat[1][1], OIII_flux_sat[2][1], \
                OIII_flux_sat[0][2], OIII_flux_sat[1][2], OIII_flux_sat[2][2], \
                OIII_flux_sat[0][3], OIII_flux_sat[1][3], OIII_flux_sat[2][3], \
                OIII_flux_sat[0][4], OIII_flux_sat[1][4], OIII_flux_sat[2][4], \
                OIV_flux_sat[0][0], OIV_flux_sat[1][0], OIV_flux_sat[2][0], \
                OIV_flux_sat[0][1], OIV_flux_sat[1][1], OIV_flux_sat[2][1], \
                OIV_flux_sat[0][2], OIV_flux_sat[1][2], OIV_flux_sat[2][2], \
                OIV_flux_sat[0][3], OIV_flux_sat[1][3], OIV_flux_sat[2][3], \
                OIV_flux_sat[0][4], OIV_flux_sat[1][4], OIV_flux_sat[2][4], \
                OV_flux_sat[0][0], OV_flux_sat[1][0], OV_flux_sat[2][0], \
                OV_flux_sat[0][1], OV_flux_sat[1][1], OV_flux_sat[2][1], \
                OV_flux_sat[0][2], OV_flux_sat[1][2], OV_flux_sat[2][2], \
                OV_flux_sat[0][3], OV_flux_sat[1][3], OV_flux_sat[2][3], \
                OV_flux_sat[0][4], OV_flux_sat[1][4], OV_flux_sat[2][4], \
                OVI_flux_sat[0][0], OVI_flux_sat[1][0], OVI_flux_sat[2][0], \
                OVI_flux_sat[0][1], OVI_flux_sat[1][1], OVI_flux_sat[2][1], \
                OVI_flux_sat[0][2], OVI_flux_sat[1][2], OVI_flux_sat[2][2], \
                OVI_flux_sat[0][3], OVI_flux_sat[1][3], OVI_flux_sat[2][3], \
                OVI_flux_sat[0][4], OVI_flux_sat[1][4], OVI_flux_sat[2][4], \
                OVII_flux_sat[0][0], OVII_flux_sat[1][0], OVII_flux_sat[2][0], \
                OVII_flux_sat[0][1], OVII_flux_sat[1][1], OVII_flux_sat[2][1], \
                OVII_flux_sat[0][2], OVII_flux_sat[1][2], OVII_flux_sat[2][2], \
                OVII_flux_sat[0][3], OVII_flux_sat[1][3], OVII_flux_sat[2][3], \
                OVII_flux_sat[0][4], OVII_flux_sat[1][4], OVII_flux_sat[2][4], \
                OVIII_flux_sat[0][0], OVIII_flux_sat[1][0], OVIII_flux_sat[2][0], \
                OVIII_flux_sat[0][1], OVIII_flux_sat[1][1], OVIII_flux_sat[2][1], \
                OVIII_flux_sat[0][2], OVIII_flux_sat[1][2], OVIII_flux_sat[2][2], \
                OVIII_flux_sat[0][3], OVIII_flux_sat[1][3], OVIII_flux_sat[2][3], \
                OVIII_flux_sat[0][4], OVIII_flux_sat[1][4], OVIII_flux_sat[2][4], \
                OIX_flux_sat[0][0], OIX_flux_sat[1][0], OIX_flux_sat[2][0], \
                OIX_flux_sat[0][1], OIX_flux_sat[1][1], OIX_flux_sat[2][1], \
                OIX_flux_sat[0][2], OIX_flux_sat[1][2], OIX_flux_sat[2][2], \
                OIX_flux_sat[0][3], OIX_flux_sat[1][3], OIX_flux_sat[2][3], \
                OIX_flux_sat[0][4], OIX_flux_sat[1][4], OIX_flux_sat[2][4]]
            fluxes_sat.add_row(new_row_sat)
            fluxes_sat = set_table_units(fluxes_sat)
            fluxes_sat.write(tablename + '_sat_cylinder_' + cyl_filename + fluxtype_filename + '.hdf5', path='all_data', serialize_meta=True, overwrite=True)

    fluxes_cylinder = set_table_units(fluxes_cylinder)
    fluxes_edges = set_table_units(fluxes_edges)

    # Save to file
    if (sat_radius!=0):
        fluxes_cylinder.write(tablename + '_nosat_cylinder_' + cyl_filename + fluxtype_filename + '.hdf5', path='all_data', serialize_meta=True, overwrite=True)
    else:
        fluxes_cylinder.write(tablename + '_cylinder_' + cyl_filename + fluxtype_filename + '.hdf5', path='all_data', serialize_meta=True, overwrite=True)
    fluxes_edges.write(tablename + '_edges_cylinder_' + cyl_filename + fluxtype_filename + '.hdf5', path='all_data', serialize_meta=True, overwrite=True)

    return "Fluxes have been calculated for snapshot " + snap + "!"

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
        ds, refine_box = foggie_load(snap_name, track, disk_relative=True, halo_c_v_name=halo_c_v_name)
    else:
        ds, refine_box = foggie_load(snap_name, track, do_filter_particles=False, halo_c_v_name=halo_c_v_name)
    refine_width_kpc = ds.quan(ds.refine_width, 'kpc')
    zsnap = ds.get_parameter('CosmologyCurrentRedshift')
    dt = 5.38e6

    # Specify the file where the list of satellites is saved
    if (sat_radius!=0.):
        sat_file = sat_dir + 'satellites.hdf5'
        sat = Table.read(sat_file, path='all_data')
        # Load halo center for second snapshot if first snapshot is not an RD output
        if (snap[:2]!='RD'):
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
        else:
            print("Removing satellites for an RD output is not as accurate as for DD outputs, but I'll do it anyway")
            halo_c_v = Table.read(halo_c_v_name, format='ascii')
            halo_ind = np.where(halo_c_v['col3']==snap[-6:])[0][0]
            halo_center_kpc2 = ds.arr([float(halo_c_v['col4'][halo_ind]), \
                                      float(halo_c_v['col5'][halo_ind]), \
                                      float(halo_c_v['col6'][halo_ind])], 'kpc')
    # Do the actual calculation
    #message = calc_fluxes(ds, snap, zsnap, refine_width_kpc, tablename, Menc_func=Menc_func)
    if (surface_args[0]=='sphere'):
        if (sat_radius!=0.):
            message = calc_fluxes_sphere(ds, snap, zsnap, dt, refine_width_kpc, tablename, surface_args, \
              flux_types, sat=sat, sat_radius=sat_radius, halo_center_kpc2=halo_center_kpc2)
        else:
            message = calc_fluxes_sphere(ds, snap, zsnap, dt, refine_width_kpc, tablename, surface_args, \
              flux_types)
    if (surface_args[0]=='frustum'):
        if (sat_radius!=0.):
            message = calc_fluxes_frustum(ds, snap, zsnap, dt, refine_width_kpc, tablename, surface_args, \
              flux_types, sat=sat, sat_radius=sat_radius, halo_center_kpc2=halo_center_kpc2)
        else:
            message = calc_fluxes_frustum(ds, snap, zsnap, dt, refine_width_kpc, tablename, surface_args, \
              flux_types)
    if (surface_args[0]=='cylinder'):
        if (sat_radius!=0.):
            message = calc_fluxes_cylinder(ds, snap, zsnap, dt, refine_width_kpc, tablename, surface_args, \
              flux_types, sat=sat, sat_radius=sat_radius, halo_center_kpc2=halo_center_kpc2)
        else:
            message = calc_fluxes_cylinder(ds, snap, zsnap, dt, refine_width_kpc, tablename, surface_args, \
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
    prefix = output_dir + 'fluxes_halo_00' + args.halo + '/' + args.run + '/'
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
            tablename = prefix + snap + '_fluxes'
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
                tablename = prefix + snap + '_fluxes'
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
            tablename = prefix + snap + '_fluxes'
            threads.append(multi.Process(target=load_and_calculate, \
			   args=(args.system, foggie_dir, run_dir, trackname, halo_c_v_name, snap, \
                 tablename, surface_args, flux_types, sat_dir, sat_radius)))
        for t in threads:
            t.start()
        for t in threads:
            t.join()

    print(str(datetime.datetime.now()))
    print("All snapshots finished!")
