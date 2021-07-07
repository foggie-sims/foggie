"""
Filename: flux_tracking.py
Author: Cassi
Date created: 9-27-19
Date last modified: 4-30-21
This file takes command line arguments and computes fluxes of things through surfaces.

Dependencies:
utils/consistency.py
utils/get_refine_box.py
utils/get_halo_center.py
utils/get_proper_box_size.py
utils/get_run_loc_etc.py
utils/yt_fields.py
utils/foggie_load.py
utils/analysis_utils.py
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
from foggie.utils.analysis_utils import *

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
                        ' Default is no.\n Don\'t do this if your surface is a cylinder enclosing the\n' + \
                        'central disk because it will try to remove parts of the disk.')
    parser.set_defaults(remove_sats=False)

    parser.add_argument('--sat_radius', metavar='sat_radius', type=float, action='store', \
                        help='What radius (in kpc) do you want to excise around satellites? Default is 10.')
    parser.set_defaults(sat_radius=10.)

    parser.add_argument('--flux_type', metavar='flux_type', type=str, action='store', \
                        help='What fluxes do you want to compute? Currently, the options are "mass" (includes metal masses)' + \
                        ' "energy" "entropy" "O_ion_mass" and "angular_momentum".\nYou can compute all of them by inputting ' + \
                        '"mass,energy,entropy,O_ion_mass,angular_momentum" (no spaces!) ' + \
                        'and the default is to do all.')
    parser.set_defaults(flux_type="mass,energy,entropy,O_ion_mass,angular_momentum")

    parser.add_argument('--surface', metavar='surface', type=str, action='store', \
                        help='What surface type for computing the flux? Default is sphere' + \
                        ' and the other options are "frustum" or "cylinder".\nNote that all surfaces will be centered on halo center.\n' + \
                        'Make sure the extent of your surface goes at least 15 kpc past the end point where you\n' + \
                        'actually want fluxes! Flux tracking is not accurate within 15 kpc of the ends.\n' + \
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
                        "will compute fluxes across different radii within the cylinder parallel to the cylinder's flat sides.\n" + \
                        "'num_steps' gives the number of places (either heights or radii) within the cylinder where to calculate fluxes.\n" + \
                        'If you want elliptical cone(s), give:\n' + \
                        '"[\'ellipse\', ellipse_filename, inner_radius, outer_radius, num_steps]"\n' + \
                        'where ellipse_filename is the name of the file where the ellipse(s) parameters are saved,\n' + \
                        'and inner_radius, outer_radius, and num_steps are the same as for the other shapes.\n' + \
                        'If you want multiple regions, use the same syntax but put each region list into a larger list, like:\n' + \
                        '"[[\'frustum\', \'x\', 0.05, 2, 200, 15],[\'frustum\', \'y\', 0.05, 2, 200, 30]]"\n' + \
                        'If you specify multiple shapes, they must all have the same inner_radius, outer_radius, and num_steps.\n' + \
                        'You can specify many different shapes at once, as long as none of them are cylinders.\n' + \
                        'If you want a cylinder, you can only do one at at time.')
    parser.set_defaults(surface="['sphere', 0.05, 2., 200]")

    parser.add_argument('--simple', dest='simple', action='store_true',
                        help='Specify this if you just want to compute fluxes into and out of the shape(s)\n' + \
                        "you've specified and ignoring fluxes within the shape itself. If using --simple,\n" + \
                        'simply use 0 for num_steps in the --surface argument.')
    parser.set_defaults(simple=False)

    parser.add_argument('--inverse', dest='inverse', action='store_true',
                        help='Do you want to calculate for everything *outside* of the shape(s) you\'ve specified?')
    parser.set_defaults(inverse=False)

    parser.add_argument('--kpc', dest='units_kpc', action='store_true',
                        help='Do you want to give inner_radius and outer_radius (sphere, frustum) or bottom_edge and top_edge (cylinder) in the surface arguments ' + \
                        'in kpc rather than the default of fraction of refine_width? Default is no.\n' + \
                        'Note that if you want to track fluxes over time, using kpc instead of fractions ' + \
                        'of refine_width will be inaccurate because refine_width is comoving and kpc are not.')
    parser.set_defaults(units_kpc=False)

    parser.add_argument('--Rvir', dest='units_rvir', action='store_true',
                        help='Do you want to give inner_radius and outer_radius or bottom_edge and top_edge in the surface arguments ' + \
                        'as fractions of the virial radius rather than the default of fraction of refine_width? Default is no.\n' + \
                        'Note that if you want to track things over time, using anything other than fractions ' + \
                        'of refine_width will be less accurate because refine_width is comoving and Rvir is not.')
    parser.set_defaults(units_rvir=False)

    parser.add_argument('--ang_mom_dir', metavar='ang_mom_dir', type=str, action='store', \
                        help='If computing the angular momentum flux, would you like to compute the components\n' + \
                        'of the angular momentum vector in a specific coordinate system?\n' + \
                        "Options are:\nx - realign z direction with x axis\ny - realign z direction with y axis\n" + \
                        'minor - realign z direction with disk minor axis\n(x,y,z) - realign z direction with vector given by (x,y,z) tuple\n' + \
                        "The default is to calculate angular momentum vector coordinates in the simulation box's x,y,z coordinates.")
    parser.set_defaults(ang_mom_dir='default')

    parser.add_argument('--temp_cut', dest='temp_cut', action='store_true',
                        help='Do you want to compute everything broken into cold, cool, warm, and hot gas? Default is no.')
    parser.set_defaults(temp_cut=False)

    parser.add_argument('--nproc', metavar='nproc', type=int, action='store', \
                        help='How many processes do you want? Default is 1 ' + \
                        '(no parallelization), if multiple outputs and multiple processors are' + \
                        ' specified, code will run one output per processor')
    parser.set_defaults(nproc=1)

    parser.add_argument('--save_suffix', metavar='save_suffix', type=str, action='store', \
                        help='Do you want to append a string onto the names of the saved files? Default is no.')
    parser.set_defaults(save_suffix="")


    args = parser.parse_args()
    return args

def set_table_units(table):
    '''Sets the units for the table. Note this needs to be updated whenever something is added to
    the table. Returns the table.'''

    for key in table.keys():
        if (key=='redshift'):
            table[key].unit = None
        elif ('radius' in key) or ('height' in key) or ('edge' in key):
            table[key].unit = 'kpc'
        elif ('mass' in key) or ('metal' in key):
            table[key].unit = 'Msun/yr'
        elif ('energy' in key):
            table[key].unit = 'erg/yr'
        elif ('entropy' in key):
            table[key].unit = 'cm**2*keV/yr'
        elif ('O' in key):
            table[key].unit = 'Msun/yr'
        elif ('momentum' in key):
            table[key].unit = 'g*cm**2/s/yr'
    return table

def make_table(flux_types, surface_type, edge=False):
    '''Makes the giant table that will be saved to file.
    If 'edge' is True, this table is for fluxes into/out of edges of shape or into/out of satellite holes.'''

    if (surface_type[0]=='sphere'):
        if (edge):
            names_list = ['redshift', 'inner_radius', 'outer_radius']
            types_list = ['f8', 'f8', 'f8']
        else:
            names_list = ['redshift', 'radius']
            types_list = ['f8', 'f8']
    if (surface_type[0]=='cylinder'):
        if (surface_type[1]=='radius'):
            if (edge):
                names_list = ['redshift', 'inner_radius', 'outer_radius']
                types_list = ['f8', 'f8', 'f8']
            else:
                names_list = ['redshift', 'radius']
                types_list = ['f8', 'f8']
        if (surface_type[1]=='height'):
            if (edge):
                names_list = ['redshift', 'bottom_edge', 'top_edge']
                types_list = ['f8', 'f8', 'f8']
            else:
                names_list = ['redshift', 'height']
                types_list = ['f8', 'f8']

    dir_name = ['net_', '_in', '_out']
    if (args.temp_cut): temp_name = ['', 'cold_', 'cool_', 'warm_', 'hot_']
    else: temp_name = ['']
    for i in range(len(flux_types)):
        for k in range(len(temp_name)):
            for j in range(len(dir_name)):
                if (flux_types[i]=='cooling_energy_flux') and (not edge):
                    if (j==0):
                        name = 'net_' + temp_name[k] + 'cooling_energy_flux'
                        names_list += [name]
                        types_list += ['f8']
                elif (flux_types[i]!='cooling_energy_flux'):
                    if (j==0): name = dir_name[j]
                    else: name = ''
                    name += temp_name[k]
                    name += flux_types[i]
                    if (j>0): name += dir_name[j]
                    names_list += [name]
                    types_list += ['f8']

    table = Table(names=names_list, dtype=types_list)

    return table

def make_table_simple(flux_types, surface_type):
    '''Makes the giant table that will be saved to file.'''

    if (surface_type[0]=='sphere'):
            names_list = ['redshift', 'inner_radius', 'outer_radius']
            types_list = ['f8', 'f8', 'f8']
    if (surface_type[0]=='cylinder'):
        if (surface_type[1]=='radius'):
            names_list = ['redshift', 'radius', 'bottom_edge', 'top_edge']
            types_list = ['f8', 'f8', 'f8', 'f8']

    dir_name = ['net_', '_in', '_out']
    if (args.temp_cut): temp_name = ['', 'cold_', 'cool_', 'warm_', 'hot_']
    else: temp_name = ['']
    for i in range(len(flux_types)):
        for k in range(len(temp_name)):
            for j in range(len(dir_name)):
                if (flux_types[i]=='cooling_energy_flux'):
                    if (j==0):
                        name = 'net_' + temp_name[k] + 'cooling_energy_flux'
                        names_list += [name]
                        types_list += ['f8']
                else:
                    if (j==0): name = dir_name[j]
                    else: name = ''
                    name += temp_name[k]
                    name += flux_types[i]
                    if (j>0): name += dir_name[j]
                    names_list += [name]
                    types_list += ['f8']

    table = Table(names=names_list, dtype=types_list)

    return table

def calc_fluxes_simple(ds, snap, zsnap, dt, refine_width_kpc, tablename, save_suffix, surface_args, flux_types, Menc_profile, disk=False, Rvir=100., halo_center_kpc2=[0,0,0]):
    '''This function calculates the fluxes specified by 'flux_types' into and out of the surfaces specified by 'surface_args'. It
    uses the dataset stored in 'ds', which is from time snapshot 'snap', has redshift
    'zsnap', and has width of the refine box in kpc 'refine_width_kpc', the time step between outputs
    'dt', and stores the fluxes in 'tablename' with 'save_suffix' appended. If 'disk' is True,
    then at least once surface shape requires disk-relative fields or angular momentum will be
    calculated relative to disk directions.

    This function calculates the flux as the sum of all cells whose velocity and distance from the
    surface of interest indicate that the gas contained in that cell will be displaced across the
    surface of interest by the next timestep. That is, the properties of a cell contribute to the
    flux if it is no further from the surface of interest than v*dt where v is the cell's velocity
    normal to the surface and dt is the time between snapshots, which is dt = 5.38e6 yrs for the DD
    outputs.

    This function differs from calc_fluxes below in that it returns just one flux across each surface
    specified, rather than fluxes at a number of steps within the total shape bounded by the surface.
    It only tracks things entering or leaving the closed surface and nothing else.'''

    # Set up table of everything we want
    fluxes = []
    flux_filename = ''
    if ('mass' in flux_types):
        fluxes.append('mass_flux')
        fluxes.append('metal_flux')
        flux_filename += '_mass'
    if ('energy' in flux_types):
        fluxes.append('thermal_energy_flux')
        fluxes.append('kinetic_energy_flux')
        fluxes.append('radial_kinetic_energy_flux')
        fluxes.append('tangential_kinetic_energy_flux')
        fluxes.append('potential_energy_flux')
        fluxes.append('bernoulli_energy_flux')
        fluxes.append('cooling_energy_flux')
        flux_filename += '_energy'
    if ('entropy' in flux_types):
        fluxes.append('entropy_flux')
        flux_filename += '_entropy'
    if ('O_ion_mass' in flux_types):
        fluxes.append('O_mass_flux')
        fluxes.append('OI_mass_flux')
        fluxes.append('OII_mass_flux')
        fluxes.append('OIII_mass_flux')
        fluxes.append('OIV_mass_flux')
        fluxes.append('OV_mass_flux')
        fluxes.append('OVI_mass_flux')
        fluxes.append('OVII_mass_flux')
        fluxes.append('OVIII_mass_flux')
        fluxes.append('OIX_mass_flux')
        flux_filename += '_Oion'
    if ('angular_momentum' in flux_types):
        fluxes.append('angular_momentum_x_flux')
        fluxes.append('angular_momentum_y_flux')
        fluxes.append('angular_momentum_z_flux')
        flux_filename += '_angmom'

    if (surface_args[0][0]=='cylinder'):
        table = make_table_simple(fluxes, ['cylinder', surface_args[0][7]])
        bottom_edge = surface_args[0][1]
        top_edge = surface_args[0][2]
        cyl_radius = surface_args[0][6]
        max_radius = np.sqrt(cyl_radius**2. + max(abs(bottom_edge), abs(top_edge)))
        if (args.units_kpc):
            max_radius = ds.quan(max_radius+20., 'kpc')
            row = [zsnap, cyl_radius, bottom_edge, top_edge]
        elif (args.units_rvir):
            max_radius = ds.quan(max_radius*Rvir+20., 'kpc')
            row = [zsnap, cyl_radius*Rvir, bottom_edge*Rvir, top_edge*Rvir]
        else:
            max_radius = max_radius*refine_width_kpc+20.
            row = [zsnap, cyl_radius*refine_width_kpc, bottom_edge*refine_width_kpc, top_edge*refine_width_kpc]
    else:
        table = make_table_simple(fluxes, ['sphere', 0])
        inner_radius = surface_args[0][1]
        outer_radius = surface_args[0][2]
        if (args.units_kpc):
            max_radius = ds.quan(outer_radius+20., 'kpc')
            row = [zsnap, inner_radius, outer_radius]
        elif (args.units_rvir):
            max_radius = ds.quan(outer_radius*Rvir+20., 'kpc')
            row = [zsnap, inner_radius*Rvir, outer_radius*Rvir]
        else:
            max_radius = outer_radius*refine_width_kpc+20.
            row = [zsnap, inner_radius*refine_width_kpc, outer_radius*refine_width_kpc]

    # Load arrays of all fields we need
    print('Loading field arrays')
    sphere = ds.sphere(ds.halo_center_kpc, max_radius)

    radius = sphere['gas','radius_corrected'].in_units('kpc').v
    x = sphere['gas','x'].in_units('kpc').v - ds.halo_center_kpc[0].v
    y = sphere['gas','y'].in_units('kpc').v - ds.halo_center_kpc[1].v
    z = sphere['gas','z'].in_units('kpc').v - ds.halo_center_kpc[2].v
    if (disk):
        x_disk = sphere['gas','x_disk'].in_units('kpc').v
        y_disk = sphere['gas','y_disk'].in_units('kpc').v
        z_disk = sphere['gas','z_disk'].in_units('kpc').v
        vx_disk = sphere['gas','vx_disk'].in_units('km/s').v
        vy_disk = sphere['gas','vy_disk'].in_units('km/s').v
        vz_disk = sphere['gas','vz_disk'].in_units('km/s').v
        new_x_disk = x_disk + vx_disk*dt*(100./cmtopc*stoyr)
        new_y_disk = y_disk + vy_disk*dt*(100./cmtopc*stoyr)
        new_z_disk = z_disk + vz_disk*dt*(100./cmtopc*stoyr)
    theta = sphere['gas','theta_pos'].v
    phi = sphere['gas', 'phi_pos'].v
    vx = sphere['gas','vx_corrected'].in_units('km/s').v
    vy = sphere['gas','vy_corrected'].in_units('km/s').v
    vz = sphere['gas','vz_corrected'].in_units('km/s').v
    rad_vel = sphere['gas','radial_velocity_corrected'].in_units('km/s').v
    new_x = x + vx*dt*(100./cmtopc*stoyr)
    new_y = y + vy*dt*(100./cmtopc*stoyr)
    new_z = z + vz*dt*(100./cmtopc*stoyr)
    new_radius = np.sqrt(new_x**2. + new_y**2. + new_z**2.)
    new_theta = np.arccos(new_z/new_radius)
    new_phi = np.arctan2(new_y, new_x)
    temperature = np.log10(sphere['gas','temperature'].in_units('K').v)
    fields = []
    if ('mass' in flux_types):
        mass = sphere['gas','cell_mass'].in_units('Msun').v
        metal_mass = sphere['gas','metal_mass'].in_units('Msun').v
        fields.append(mass)
        fields.append(metal_mass)
    if ('energy' in flux_types):
        kinetic_energy = sphere['gas','kinetic_energy_corrected'].in_units('erg').v
        radial_kinetic_energy = sphere['gas','radial_kinetic_energy'].in_units('erg').v
        if (disk): tangential_kinetic_energy = sphere['gas','tangential_kinetic_energy_disk'].in_units('erg').v
        else: tangential_kinetic_energy = sphere['gas','tangential_kinetic_energy'].in_units('erg').v
        thermal_energy = (sphere['gas','cell_mass']*sphere['gas','thermal_energy']).in_units('erg').v
        potential_energy = -G * Menc_profile(radius)*gtoMsun / (radius*1000.*cmtopc)*sphere['gas','cell_mass'].in_units('g').v
        bernoulli_energy = kinetic_energy + 5./3.*thermal_energy + potential_energy
        cooling_energy = thermal_energy/sphere['gas','cooling_time'].in_units('yr').v
        fields.append(thermal_energy)
        fields.append(kinetic_energy)
        fields.append(radial_kinetic_energy)
        fields.append(tangential_kinetic_energy)
        fields.append(potential_energy)
        fields.append(bernoulli_energy)
        fields.append(cooling_energy)
    if ('entropy' in flux_types):
        entropy = sphere['gas','entropy'].in_units('keV*cm**2').v
        fields.append(entropy)
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
        fields.append(O_mass)
        fields.append(OI_mass)
        fields.append(OII_mass)
        fields.append(OIII_mass)
        fields.append(OIV_mass)
        fields.append(OV_mass)
        fields.append(OVI_mass)
        fields.append(OVII_mass)
        fields.append(OVIII_mass)
        fields.append(OIX_mass)
    if ('angular_momentum' in flux_types):
        if (args.ang_mom_dir!='default') and (args.ang_mom_dir!='x') and (args.ang_mom_dir!='y') and (args.ang_mom_dir!='minor'):
            try:
                ang_mom_dir = ast.literal_eval(args.ang_mom_dir)
            except ValueError:
                sys.exit("Something's wrong with the way you are specifying your angular momentum vector.\n" + \
                        "options are: x, y, minor, or a tuple specifying the 3D coordinates of a vector.")
        if (args.ang_mom_dir=='default'):
            ang_x = x
            ang_y = y
            ang_z = z
            ang_vx = vx
            ang_vy = vy
            ang_vz = vz
        elif (args.ang_mom_dir=='x'):
            ang_x = y
            ang_y = z
            ang_z = x
            ang_vx = vy
            ang_vy = vz
            ang_vz = vx
        elif (args.ang_mom_dir=='y'):
            ang_x = z
            ang_y = x
            ang_z = y
            ang_vx = vz
            ang_vy = vx
            ang_vz = vy
        elif (args.ang_mom_dir=='minor'):
            ang_x = sphere['gas','x_disk'].in_units('kpc').v
            ang_y = sphere['gas','y_disk'].in_units('kpc').v
            ang_z = sphere['gas','z_disk'].in_units('kpc').v
            ang_vx = sphere['gas','vx_disk'].in_units('km/s').v
            ang_vy = sphere['gas','vy_disk'].in_units('km/s').v
            ang_vz = sphere['gas','vz_disk'].in_units('km/s').v
        else:
            axis = np.array(ang_mom_dir)
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
            ang_x = rotationArr[0][0]*x + rotationArr[0][1]*y + rotationArr[0][2]*z
            ang_y = rotationArr[1][0]*x + rotationArr[1][1]*y + rotationArr[1][2]*z
            ang_z = rotationArr[2][0]*x + rotationArr[2][1]*y + rotationArr[2][2]*z
            ang_vx = rotationArr[0][0]*vx + rotationArr[0][1]*vy + rotationArr[0][2]*vz
            ang_vy = rotationArr[1][0]*vx + rotationArr[1][1]*vy + rotationArr[1][2]*vz
            ang_vz = rotationArr[2][0]*vx + rotationArr[2][1]*vy + rotationArr[2][2]*vz
        ang_mom_x = mass*gtoMsun*(ang_y*ang_vz - ang_z*ang_vy)*1e5*1000*cmtopc
        ang_mom_y = mass*gtoMsun*(ang_z*ang_vx - ang_x*ang_vz)*1e5*1000*cmtopc
        ang_mom_z = mass*gtoMsun*(ang_x*ang_vy - ang_y*ang_vx)*1e5*1000*cmtopc
        fields.append(ang_mom_x)
        fields.append(ang_mom_y)
        fields.append(ang_mom_z)

    # Cut to just the shapes specified
    if (disk):
        if (surface_args[0][0]=='cylinder'):
            bool_inshapes, radius = segment_region(x, y, z, theta, phi, radius, surface_args, refine_width_kpc, \
              x_disk=x_disk, y_disk=y_disk, z_disk=z_disk, Rvir=Rvir, units_kpc=args.units_kpc, units_rvir=args.units_rvir)
            bool_inshapes_new, new_radius = segment_region(new_x, new_y, new_z, new_theta, new_phi, new_radius, surface_args, refine_width_kpc, \
              x_disk=new_x_disk, y_disk=new_y_disk, z_disk=new_z_disk, Rvir=Rvir, units_kpc=args.units_kpc, units_rvir=args.units_rvir)
        else:
            bool_inshapes = segment_region(x, y, z, theta, phi, radius, surface_args, refine_width_kpc, \
              x_disk=x_disk, y_disk=y_disk, z_disk=z_disk, Rvir=Rvir, units_kpc=args.units_kpc, units_rvir=args.units_rvir)
            bool_inshapes_new = segment_region(new_x, new_y, new_z, new_theta, new_phi, new_radius, surface_args, refine_width_kpc, \
              x_disk=new_x_disk, y_disk=new_y_disk, z_disk=new_z_disk, Rvir=Rvir, units_kpc=args.units_kpc, units_rvir=args.units_rvir)
    else:
        if (surface_args[0][0]=='cylinder'):
            bool_inshapes, radius = segment_region(x, y, z, theta, phi, radius, surface_args, refine_width_kpc, \
              Rvir=Rvir, units_kpc=args.units_kpc, units_rvir=args.units_rvir)
            bool_inshapes_new, new_radius = segment_region(new_x, new_y, new_z, new_theta, new_phi, new_radius, surface_args, refine_width_kpc, \
              Rvir=Rvir, units_kpc=args.units_kpc, units_rvir=args.units_rvir)
        else:
            bool_inshapes = segment_region(x, y, z, theta, phi, radius, surface_args, refine_width_kpc, \
              Rvir=Rvir, units_kpc=args.units_kpc, units_rvir=args.units_rvir)
            bool_inshapes_new = segment_region(new_x, new_y, new_z, new_theta, new_phi, new_radius, surface_args, refine_width_kpc, \
              Rvir=Rvir, units_kpc=args.units_kpc, units_rvir=args.units_rvir)
    bool_inshapes_entire = (bool_inshapes) & (bool_inshapes_new)
    bool_toshapes = (~bool_inshapes) & (bool_inshapes_new)
    bool_fromshapes = (bool_inshapes) & (~bool_inshapes_new)

    # Cut to entering/leaving shapes
    fields_in_shapes = []
    fields_out_shapes = []
    radius_in_shapes = radius[bool_toshapes]
    new_radius_in_shapes = new_radius[bool_toshapes]
    temperature_in_shapes = temperature[bool_toshapes]
    temperature_shapes = temperature[bool_inshapes_entire]
    radius_out_shapes = radius[bool_fromshapes]
    new_radius_out_shapes = new_radius[bool_fromshapes]
    temperature_out_shapes = temperature[bool_fromshapes]
    for i in range(len(fields)):
        field = fields[i]
        fields_in_shapes.append(field[bool_toshapes])
        fields_out_shapes.append(field[bool_fromshapes])

    if (args.temp_cut): temps = [0.,4.,5.,6.,12.]
    else: temps = [0.]

    for i in range(len(fields)):
        if (fluxes[i]=='cooling_energy_flux'):
            iter = [0]
            field = fields[i][bool_inshapes_entire]
        else:
            iter = [0,1,2]
            field_in = fields_in_shapes[i]
            field_out = fields_out_shapes[i]

        for k in range(len(temps)):
            if (k==0):
                if (fluxes[i]=='cooling_energy_flux'):
                    field_t = field
                else:
                    field_in_t = field_in
                    field_out_t = field_out
            else:
                if (fluxes[i]=='cooling_energy_flux'):
                    field_t = field[(temperature_shapes > temps[k-1]) & (temperature_shapes < temps[k])]
                else:
                    field_in_t = field_in[(temperature_in_shapes > temps[k-1]) & (temperature_in_shapes < temps[k])]
                    field_out_t = field_out[(temperature_out_shapes > temps[k-1]) & (temperature_out_shapes < temps[k])]
            for j in iter:
                if (j==0):
                    if (fluxes[i]=='cooling_energy_flux'):
                        row.append(-np.sum(field_t))
                    else:
                        row.append(np.sum(field_out_t)/dt - np.sum(field_in_t)/dt)
                if (j==1):
                    row.append(-np.sum(field_in_t)/dt)
                if (j==2):
                    row.append(np.sum(field_out_t)/dt)

    table.add_row(row)
    table = set_table_units(table)

    # Save to file
    table.write(tablename + flux_filename + save_suffix + '_simple.hdf5', path='all_data', serialize_meta=True, overwrite=True)

    return "Fluxes have been calculated for snapshot " + snap + "!"

def calc_fluxes(ds, snap, zsnap, dt, refine_width_kpc, tablename, save_suffix, surface_args, flux_types, Menc_profile, sat=False, sat_radius=0.,inverse=False, disk=False, Rvir=100., halo_center_kpc2=[0,0,0]):
    '''This function calculates the fluxes specified by 'flux_types' into and out of the surfaces specified by 'surface_args'. It
    uses the dataset stored in 'ds', which is from time snapshot 'snap', has redshift
    'zsnap', and has width of the refine box in kpc 'refine_width_kpc', the time step between outputs
    'dt', and stores the fluxes in 'tablename' with 'save_suffix' appended. 'sat' is either False if
    satellites are not removed or is the table of satellite positions if they are, and 'sat_radius'
    is the radius (in kpc) around satellites to excise. If 'inverse' is True, then calculate for everything
    *outside* the surfaces given in 'surface_args'. If 'disk' is True, then at least once surface shape
    requires disk-relative fields or angular momentum will be calculated relative to disk directions.

    This function calculates the flux as the sum
    of all cells whose velocity and distance from the surface of interest indicate that the gas
    contained in that cell will be displaced across the surface of interest by the next timestep.
    That is, the properties of a cell contribute to the flux if it is no further from the surface of
    interest than v*dt where v is the cell's velocity normal to the surface and dt is the time
    between snapshots, which is dt = 5.38e6 yrs for the DD outputs. It is necessary to compute the
    flux this way if satellites are to be removed because they become 'holes' in the dataset
    and fluxes into/out of those holes need to be accounted for.'''

    # Set up table of everything we want
    fluxes = []
    flux_filename = ''
    if ('mass' in flux_types):
        fluxes.append('mass_flux')
        fluxes.append('metal_flux')
        flux_filename += '_mass'
    if ('energy' in flux_types):
        fluxes.append('thermal_energy_flux')
        fluxes.append('kinetic_energy_flux')
        fluxes.append('radial_kinetic_energy_flux')
        fluxes.append('tangential_kinetic_energy_flux')
        fluxes.append('potential_energy_flux')
        fluxes.append('bernoulli_energy_flux')
        fluxes.append('cooling_energy_flux')
        flux_filename += '_energy'
    if ('entropy' in flux_types):
        fluxes.append('entropy_flux')
        flux_filename += '_entropy'
    if ('O_ion_mass' in flux_types):
        fluxes.append('O_mass_flux')
        fluxes.append('OI_mass_flux')
        fluxes.append('OII_mass_flux')
        fluxes.append('OIII_mass_flux')
        fluxes.append('OIV_mass_flux')
        fluxes.append('OV_mass_flux')
        fluxes.append('OVI_mass_flux')
        fluxes.append('OVII_mass_flux')
        fluxes.append('OVIII_mass_flux')
        fluxes.append('OIX_mass_flux')
        flux_filename += '_Oion'
    if ('angular_momentum' in flux_types):
        fluxes.append('angular_momentum_x_flux')
        fluxes.append('angular_momentum_y_flux')
        fluxes.append('angular_momentum_z_flux')
        flux_filename += '_angmom'

    # Define list of ways to chunk up the shape over radius or height
    edges = False
    if (surface_args[0][0]=='cylinder'):
        table = make_table(fluxes, ['cylinder', surface_args[0][7]])
        table_edge = make_table(fluxes, ['cylinder', surface_args[0][7]], edge=True)
        edges = True
        if (sat):
            table_sat = make_table(fluxes, ['cylinder', surface_args[0][7]], edge=True)
        bottom_edge = surface_args[0][1]
        top_edge = surface_args[0][2]
        cyl_radius = surface_args[0][6]
        num_steps = surface_args[0][3]
        if (surface_args[0][7]=='height'):
            if (args.units_kpc):
                dz = (top_edge-bottom_edge)/num_steps
                chunks = ds.arr(np.arange(bottom_edge,top_edge+dz,dz), 'kpc')
            elif (args.units_rvir):
                dz = (top_edge-bottom_edge)/num_steps*Rvir
                chunks = ds.arr(np.arange(bottom_edge*Rvir,top_edge*Rvir+dz,dz), 'kpc')
            else:
                dz = (top_edge-bottom_edge)/num_steps*refine_width_kpc
                chunks = np.arange(bottom_edge*refine_width_kpc,top_edge*refine_width_kpc+dz,dz)
        else:
            if (args.units_kpc):
                dr = cyl_radius/num_steps
                chunks = ds.arr(np.arange(0.,cyl_radius+dr,dr), 'kpc')
            elif (args.units_rvir):
                dr = cyl_radius/num_steps*Rvir
                chunks = ds.arr(np.arange(0.,cyl_radius*Rvir+dr,dr), 'kpc')
            else:
                dr = cyl_radius/num_steps*refine_width_kpc
                chunks = np.arange(0.,cyl_radius*refine_width_kpc+dr,dr)
    else:
        inner_radius = surface_args[0][1]
        outer_radius = surface_args[0][2]
        num_steps = surface_args[0][3]
        table = make_table(fluxes, ['sphere', 0])
        if (surface_args[0][0]=='frustum') or (surface_args[0][0]=='ellipse'):
            table_edge = make_table(fluxes, ['sphere', 0], edge=True)
            edges = True
        if (sat):
            table_sat = make_table(fluxes, ['sphere', 0], edge=True)
        if (args.units_kpc):
            dr = (outer_radius-inner_radius)/num_steps
            chunks = ds.arr(np.arange(inner_radius,outer_radius+dr,dr), 'kpc')
        elif (args.units_rvir):
            dr = (outer_radius-inner_radius)/num_steps*Rvir
            chunks = ds.arr(np.arange(inner_radius*Rvir,outer_radius*Rvir+dr,dr), 'kpc')
        else:
            dr = (outer_radius-inner_radius)/num_steps*refine_width_kpc
            chunks = np.arange(inner_radius*refine_width_kpc,outer_radius*refine_width_kpc+dr,dr)

    # Load arrays of all fields we need
    print('Loading field arrays')
    sphere = ds.sphere(ds.halo_center_kpc, chunks[-1])

    radius = sphere['gas','radius_corrected'].in_units('kpc').v
    x = sphere['gas','x'].in_units('kpc').v - ds.halo_center_kpc[0].v
    y = sphere['gas','y'].in_units('kpc').v - ds.halo_center_kpc[1].v
    z = sphere['gas','z'].in_units('kpc').v - ds.halo_center_kpc[2].v
    if (disk):
        x_disk = sphere['gas','x_disk'].in_units('kpc').v
        y_disk = sphere['gas','y_disk'].in_units('kpc').v
        z_disk = sphere['gas','z_disk'].in_units('kpc').v
        vx_disk = sphere['gas','vx_disk'].in_units('km/s').v
        vy_disk = sphere['gas','vy_disk'].in_units('km/s').v
        vz_disk = sphere['gas','vz_disk'].in_units('km/s').v
        new_x_disk = x_disk + vx_disk*dt*(100./cmtopc*stoyr)
        new_y_disk = y_disk + vy_disk*dt*(100./cmtopc*stoyr)
        new_z_disk = z_disk + vz_disk*dt*(100./cmtopc*stoyr)
    theta = sphere['gas','theta_pos'].v
    phi = sphere['gas', 'phi_pos'].v
    vx = sphere['gas','vx_corrected'].in_units('km/s').v
    vy = sphere['gas','vy_corrected'].in_units('km/s').v
    vz = sphere['gas','vz_corrected'].in_units('km/s').v
    rad_vel = sphere['gas','radial_velocity_corrected'].in_units('km/s').v
    new_x = x + vx*dt*(100./cmtopc*stoyr)
    new_y = y + vy*dt*(100./cmtopc*stoyr)
    new_z = z + vz*dt*(100./cmtopc*stoyr)
    new_radius = np.sqrt(new_x**2. + new_y**2. + new_z**2.)
    new_theta = np.arccos(new_z/new_radius)
    new_phi = np.arctan2(new_y, new_x)
    temperature = np.log10(sphere['gas','temperature'].in_units('K').v)
    fields = []
    if ('mass' in flux_types):
        mass = sphere['gas','cell_mass'].in_units('Msun').v
        metal_mass = sphere['gas','metal_mass'].in_units('Msun').v
        fields.append(mass)
        fields.append(metal_mass)
    if ('energy' in flux_types):
        kinetic_energy = sphere['gas','kinetic_energy_corrected'].in_units('erg').v
        radial_kinetic_energy = sphere['gas','radial_kinetic_energy'].in_units('erg').v
        if (disk): tangential_kinetic_energy = sphere['gas','tangential_kinetic_energy_disk'].in_units('erg').v
        else: tangential_kinetic_energy = sphere['gas','tangential_kinetic_energy'].in_units('erg').v
        thermal_energy = (sphere['gas','cell_mass']*sphere['gas','thermal_energy']).in_units('erg').v
        potential_energy = -G * Menc_profile(radius)*gtoMsun / (radius*1000.*cmtopc)*sphere['gas','cell_mass'].in_units('g').v
        bernoulli_energy = kinetic_energy + 5./3.*thermal_energy + potential_energy
        cooling_energy = thermal_energy/sphere['gas','cooling_time'].in_units('yr').v
        fields.append(thermal_energy)
        fields.append(kinetic_energy)
        fields.append(radial_kinetic_energy)
        fields.append(tangential_kinetic_energy)
        fields.append(potential_energy)
        fields.append(bernoulli_energy)
        fields.append(cooling_energy)
    if ('entropy' in flux_types):
        entropy = sphere['gas','entropy'].in_units('keV*cm**2').v
        fields.append(entropy)
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
        fields.append(O_mass)
        fields.append(OI_mass)
        fields.append(OII_mass)
        fields.append(OIII_mass)
        fields.append(OIV_mass)
        fields.append(OV_mass)
        fields.append(OVI_mass)
        fields.append(OVII_mass)
        fields.append(OVIII_mass)
        fields.append(OIX_mass)
    if ('angular_momentum' in flux_types):
        if (args.ang_mom_dir!='default') and (args.ang_mom_dir!='x') and (args.ang_mom_dir!='y') and (args.ang_mom_dir!='minor'):
            try:
                ang_mom_dir = ast.literal_eval(args.ang_mom_dir)
            except ValueError:
                sys.exit("Something's wrong with the way you are specifying your angular momentum vector.\n" + \
                        "options are: x, y, minor, or a tuple specifying the 3D coordinates of a vector.")
        if (args.ang_mom_dir=='default'):
            ang_x = x
            ang_y = y
            ang_z = z
            ang_vx = vx
            ang_vy = vy
            ang_vz = vz
        elif (args.ang_mom_dir=='x'):
            ang_x = y
            ang_y = z
            ang_z = x
            ang_vx = vy
            ang_vy = vz
            ang_vz = vx
        elif (args.ang_mom_dir=='y'):
            ang_x = z
            ang_y = x
            ang_z = y
            ang_vx = vz
            ang_vy = vx
            ang_vz = vy
        elif (args.ang_mom_dir=='minor'):
            ang_x = sphere['gas','x_disk'].in_units('kpc').v
            ang_y = sphere['gas','y_disk'].in_units('kpc').v
            ang_z = sphere['gas','z_disk'].in_units('kpc').v
            ang_vx = sphere['gas','vx_disk'].in_units('km/s').v
            ang_vy = sphere['gas','vy_disk'].in_units('km/s').v
            ang_vz = sphere['gas','vz_disk'].in_units('km/s').v
        else:
            axis = np.array(ang_mom_dir)
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
            ang_x = rotationArr[0][0]*x + rotationArr[0][1]*y + rotationArr[0][2]*z
            ang_y = rotationArr[1][0]*x + rotationArr[1][1]*y + rotationArr[1][2]*z
            ang_z = rotationArr[2][0]*x + rotationArr[2][1]*y + rotationArr[2][2]*z
            ang_vx = rotationArr[0][0]*vx + rotationArr[0][1]*vy + rotationArr[0][2]*vz
            ang_vy = rotationArr[1][0]*vx + rotationArr[1][1]*vy + rotationArr[1][2]*vz
            ang_vz = rotationArr[2][0]*vx + rotationArr[2][1]*vy + rotationArr[2][2]*vz
        ang_mom_x = mass*gtoMsun*(ang_y*ang_vz - ang_z*ang_vy)*1e5*1000*cmtopc
        ang_mom_y = mass*gtoMsun*(ang_z*ang_vx - ang_x*ang_vz)*1e5*1000*cmtopc
        ang_mom_z = mass*gtoMsun*(ang_x*ang_vy - ang_y*ang_vx)*1e5*1000*cmtopc
        fields.append(ang_mom_x)
        fields.append(ang_mom_y)
        fields.append(ang_mom_z)

    # Cut to just the shapes specified
    if (disk):
        if (surface_args[0][0]=='cylinder'):
            bool_inshapes, radius = segment_region(x, y, z, theta, phi, radius, surface_args, refine_width_kpc, \
              x_disk=x_disk, y_disk=y_disk, z_disk=z_disk, Rvir=Rvir, units_kpc=args.units_kpc, units_rvir=args.units_rvir)
            bool_inshapes_new, new_radius = segment_region(new_x, new_y, new_z, new_theta, new_phi, new_radius, surface_args, refine_width_kpc, \
              x_disk=new_x_disk, y_disk=new_y_disk, z_disk=new_z_disk, Rvir=Rvir, units_kpc=args.units_kpc, units_rvir=args.units_rvir)
        else:
            bool_inshapes = segment_region(x, y, z, theta, phi, radius, surface_args, refine_width_kpc, \
              x_disk=x_disk, y_disk=y_disk, z_disk=z_disk, Rvir=Rvir, units_kpc=args.units_kpc, units_rvir=args.units_rvir)
            bool_inshapes_new = segment_region(new_x, new_y, new_z, new_theta, new_phi, new_radius, surface_args, refine_width_kpc, \
              x_disk=new_x_disk, y_disk=new_y_disk, z_disk=new_z_disk, Rvir=Rvir, units_kpc=args.units_kpc, units_rvir=args.units_rvir)
    else:
        if (surface_args[0][0]=='cylinder'):
            bool_inshapes, radius = segment_region(x, y, z, theta, phi, radius, surface_args, refine_width_kpc, \
              Rvir=Rvir, units_kpc=args.units_kpc, units_rvir=args.units_rvir)
            bool_inshapes_new, new_radius = segment_region(new_x, new_y, new_z, new_theta, new_phi, new_radius, surface_args, refine_width_kpc, \
              Rvir=Rvir, units_kpc=args.units_kpc, units_rvir=args.units_rvir)
        else:
            bool_inshapes = segment_region(x, y, z, theta, phi, radius, surface_args, refine_width_kpc, \
              Rvir=Rvir, units_kpc=args.units_kpc, units_rvir=args.units_rvir)
            bool_inshapes_new = segment_region(new_x, new_y, new_z, new_theta, new_phi, new_radius, surface_args, refine_width_kpc, \
              Rvir=Rvir, units_kpc=args.units_kpc, units_rvir=args.units_rvir)
    bool_inshapes_entire = (bool_inshapes) & (bool_inshapes_new)
    bool_toshapes = (~bool_inshapes) & (bool_inshapes_new)
    bool_fromshapes = (bool_inshapes) & (~bool_inshapes_new)
    if (inverse):
        bool_inshapes_entire = ~bool_inshapes_entire
        bool_toshapes_2 = bool_fromshapes
        bool_fromshapes = bool_toshapes
        bool_toshapes = bool_toshapes_2

    # Load list of satellite positions
    if (sat):
        print('Loading satellite positions')
        sat_x = sat['sat_x'][sat['snap']==snap]
        sat_y = sat['sat_y'][sat['snap']==snap]
        sat_z = sat['sat_z'][sat['snap']==snap]
        sat_list = []
        for i in range(len(sat_x)):
            if not ((np.abs(sat_x[i] - ds.halo_center_kpc[0].v) <= 1.) & \
                    (np.abs(sat_y[i] - ds.halo_center_kpc[1].v) <= 1.) & \
                    (np.abs(sat_z[i] - ds.halo_center_kpc[2].v) <= 1.)):
                sat_list.append([sat_x[i] - ds.halo_center_kpc[0].v, sat_y[i] - ds.halo_center_kpc[1].v, sat_z[i] - ds.halo_center_kpc[2].v])
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
    else:
        bool_nosat = np.ones(len(x), dtype=bool)

    # Cut to within shapes, entering/leaving shapes, and entering/leaving satellites
    fields_shapes = []
    if (edges):
        fields_in_shapes = []
        fields_out_shapes = []
    if (sat):
        fields_in_sat = []
        fields_out_sat = []
    radius_shapes = radius[(bool_inshapes_entire) & (bool_nosat)]
    new_radius_shapes = new_radius[(bool_inshapes_entire) & (bool_nosat)]
    temperature_shapes = temperature[(bool_inshapes_entire) & (bool_nosat)]
    if (edges):
        radius_in_shapes = radius[(bool_toshapes) & (bool_nosat)]
        new_radius_in_shapes = new_radius[(bool_toshapes) & (bool_nosat)]
        temperature_in_shapes = temperature[(bool_toshapes) & (bool_nosat)]
        radius_out_shapes = radius[(bool_fromshapes) & (bool_nosat)]
        new_radius_out_shapes = new_radius[(bool_fromshapes) & (bool_nosat)]
        temperature_out_shapes = temperature[(bool_fromshapes) & (bool_nosat)]
    if (sat):
        radius_in_sat = radius[(bool_fromsat) & ((bool_inshapes_entire) | (bool_toshapes))]
        new_radius_in_sat = new_radius[(bool_fromsat) & ((bool_inshapes_entire) | (bool_toshapes))]
        temperature_in_sat = temperature[(bool_fromsat) & ((bool_inshapes_entire) | (bool_toshapes))]
        radius_out_sat = radius[(bool_tosat) & ((bool_inshapes_entire) | (bool_fromshapes))]
        new_radius_out_sat = new_radius[(bool_tosat) & ((bool_inshapes_entire) | (bool_fromshapes))]
        temperature_out_sat = temperature[(bool_tosat) & ((bool_inshapes_entire) | (bool_fromshapes))]
    for i in range(len(fields)):
        field = fields[i]
        fields_shapes.append(field[(bool_inshapes_entire) & (bool_nosat)])
        if (edges):
            fields_in_shapes.append(field[(bool_toshapes) & (bool_nosat)])
            fields_out_shapes.append(field[(bool_fromshapes) & (bool_nosat)])
        if (sat):
            fields_in_sat.append(field[(bool_fromsat) & ((bool_inshapes_entire) | (bool_toshapes))])
            fields_out_sat.append(field[(bool_tosat) & ((bool_inshapes_entire) | (bool_fromshapes))])

    if (args.temp_cut): temps = [0.,4.,5.,6.,12.]
    else: temps = [0.]

    # Loop over chunks and compute fluxes to add to tables
    for r in range(len(chunks)-1):
        if (r%10==0): print("Computing chunk " + str(r) + "/" + str(len(chunks)) + \
                            " for snapshot " + snap)

        inner = chunks[r]
        outer = chunks[r+1]
        row = [zsnap, inner]
        if (edges): row_edge = [zsnap, inner, outer]
        if (sat): row_sat = [zsnap, inner, outer]

        temp_up = temperature_shapes[(radius_shapes < inner) & (new_radius_shapes > inner)]
        temp_down = temperature_shapes[(radius_shapes > inner) & (new_radius_shapes < inner)]
        temp_r = temperature_shapes[(radius_shapes > inner) & (radius_shapes < outer)]
        if (edges):
            temp_in = temperature_in_shapes[(new_radius_in_shapes > inner) & (new_radius_in_shapes < outer)]
            temp_out = temperature_out_shapes[(radius_out_shapes > inner) & (radius_out_shapes < outer)]
        if (sat):
            temp_in_sat = temperature_in_sat[(new_radius_in_sat > inner) & (new_radius_in_sat < outer)]
            temp_out_sat = temperature_out_sat[(radius_out_sat > inner) & (radius_out_sat < outer)]

        for i in range(len(fields)):
            if (fluxes[i]=='cooling_energy_flux'):
                iter = [0]
                field_r = fields_shapes[i][(radius_shapes > inner) & (radius_shapes < outer)]
            else:
                iter = [0,1,2]
                field_up = fields_shapes[i][(radius_shapes < inner) & (new_radius_shapes > inner)]
                field_down = fields_shapes[i][(radius_shapes > inner) & (new_radius_shapes < inner)]
                if (edges):
                    field_in = fields_in_shapes[i][(new_radius_in_shapes > inner) & (new_radius_in_shapes < outer)]
                    field_out = fields_out_shapes[i][(radius_out_shapes > inner) & (radius_out_shapes < outer)]
                if (sat):
                    field_in_sat = fields_in_sat[i][(new_radius_in_sat > inner) & (new_radius_in_sat < outer)]
                    field_out_sat = fields_out_sat[i][(radius_out_sat > inner) & (radius_out_sat < outer)]

            for k in range(len(temps)):
                if (k==0):
                    if (fluxes[i]=='cooling_energy_flux'):
                        field_r_t = field_r
                    else:
                        field_up_t = field_up
                        field_down_t = field_down
                        if (edges):
                            field_in_t = field_in
                            field_out_t = field_out
                        if (sat):
                            field_in_sat_t = field_in_sat
                            field_out_sat_t = field_out_sat
                else:
                    if (fluxes[i]=='cooling_energy_flux'):
                        field_r_t = field_r[(temp_r > temps[k-1]) & (temp_r < temps[k])]
                    else:
                        field_up_t = field_up[(temp_up > temps[k-1]) & (temp_up < temps[k])]
                        field_down_t = field_down[(temp_down > temps[k-1]) & (temp_down < temps[k])]
                        if (edges):
                            field_in_t = field_in[(temp_in > temps[k-1]) & (temp_in < temps[k])]
                            field_out_t = field_out[(temp_out > temps[k-1]) & (temp_out < temps[k])]
                        if (sat):
                            field_in_sat_t = field_in_sat[(temp_in_sat > temps[k-1]) & (temp_in_sat < temps[k])]
                            field_out_sat_t = field_out_sat[(temp_out_sat > temps[k-1]) & (temp_out_sat < temps[k])]
                for j in iter:
                    if (j==0):
                        if (fluxes[i]=='cooling_energy_flux'):
                            row.append(-np.sum(field_r_t))
                        else:
                            row.append(np.sum(field_up_t)/dt - np.sum(field_down_t)/dt)
                            if (edges):
                                row_edge.append(np.sum(field_in_t)/dt - np.sum(field_out_t)/dt)
                            if (sat):
                                row_sat.append(np.sum(field_in_sat_t)/dt - np.sum(field_out_sat_t)/dt)
                    if (j==1):
                        row.append(-np.sum(field_down_t)/dt)
                        if (edges):
                            row_edge.append(np.sum(field_in_t)/dt)
                        if (sat):
                            row_sat.append(np.sum(field_in_sat_t)/dt)
                    if (j==2):
                        row.append(np.sum(field_up_t)/dt)
                        if (edges):
                            row_edge.append(-np.sum(field_out_t)/dt)
                        if (sat):
                            row_sat.append(-np.sum(field_out_sat_t)/dt)


        table.add_row(row)
        if (edges): table_edge.add_row(row_edge)
        if (sat): table_sat.add_row(row_sat)

    table = set_table_units(table)
    if (edges): table_edge = set_table_units(table_edge)
    if (sat): table_sat = set_table_units(table_sat)

    # Save to file
    if (sat):
        table.write(tablename + '_nosat' + flux_filename + save_suffix + '.hdf5', path='all_data', serialize_meta=True, overwrite=True)
        if (edges): table_edge.write(tablename + '_nosat_edge' + flux_filename + save_suffix + '.hdf5', path='all_data', serialize_meta=True, overwrite=True)
        table_sat.write(tablename + '_sat_edge' + flux_filename + save_suffix + '.hdf5', path='all_data', serialize_meta=True, overwrite=True)
    else:
        table.write(tablename + flux_filename + save_suffix + '.hdf5', path='all_data', serialize_meta=True, overwrite=True)
        if (edges): table_edge.write(tablename + '_edge' + flux_filename + save_suffix + '.hdf5', path='all_data', serialize_meta=True, overwrite=True)

    return "Fluxes have been calculated for snapshot " + snap + "!"

def load_and_calculate(system, foggie_dir, run_dir, track, halo_c_v_name, snap, tablename, save_suffix, surface_args, flux_types, sat_dir, sat_radius, masses_dir):
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

    # Load the snapshot depending on if disk minor axis is needed
    disk = False
    for i in range(len(surface_args)):
        if (((surface_args[i][0]=='frustum') or (surface_args[i][0]=='cylinder')) and (surface_args[i][4]=='disk minor axis')):
            disk = True
    if (disk):
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
    else:
        sat = False
        halo_center_kpc2 = [0,0,0]

    # Load the mass enclosed profile
    if (zsnap > 2.):
        masses = Table.read(masses_dir + 'masses_z-gtr-2.hdf5', path='all_data')
    else:
        masses = Table.read(masses_dir + 'masses_z-less-2.hdf5', path='all_data')
    rvir_masses = Table.read(masses_dir + 'rvir_masses.hdf5', path='all_data')
    snap_ind = masses['snapshot']==snap
    Menc_profile = IUS(masses['radius'][snap_ind], masses['total_mass'][snap_ind])
    Rvir = rvir_masses['radius'][rvir_masses['snapshot']==snap]

    # Do the actual calculation
    if (args.simple):
        message = calc_fluxes_simple(ds, snap, zsnap, dt, refine_width_kpc, tablename, save_suffix, \
          surface_args, flux_types, Menc_profile, disk=disk, Rvir=Rvir, halo_center_kpc2=halo_center_kpc2)
    else:
        message = calc_fluxes(ds, snap, zsnap, dt, refine_width_kpc, tablename, save_suffix, \
          surface_args, flux_types, Menc_profile, sat=sat, sat_radius=sat_radius,inverse=args.inverse, \
          disk=disk, Rvir=Rvir, halo_center_kpc2=halo_center_kpc2)

    # Delete output from temp directory if on pleiades
    if (system=='pleiades_cassi'):
        print('Deleting directory from /tmp')
        shutil.rmtree(snap_dir)
    print(message)
    print(str(datetime.datetime.now()))


if __name__ == "__main__":

    gtoMsun = 1.989e33
    cmtopc = 3.086e18
    stoyr = 3.155e7
    G = 6.673e-8
    kB = 1.38e-16
    mu = 0.6
    mp = 1.67e-24

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
    masses_dir = code_path + 'halo_infos/00' + args.halo + '/' + args.run + '/'

    # Specify where satellite files are saved
    if (args.remove_sats):
        sat_dir = code_path + 'halo_infos/00' + args.halo + '/' + args.run + '/'
        sat_radius = args.sat_radius
    else:
        sat_dir = 'sat_dir'
        sat_radius = 0.

    if (args.save_suffix!=''):
        save_suffix = '_' + args.save_suffix
    else:
        save_suffix = ''

    surfaces = identify_shape(args.surface, args.halo, args.run, units_kpc=args.units_kpc, units_rvir=args.units_rvir)
    outs = make_output_list(args.output, output_step=args.output_step)

    # Build flux type list
    if (',' in args.flux_type):
        flux_types = args.flux_type.split(',')
    else:
        flux_types = [args.flux_type]
    for i in range(len(flux_types)):
        if (flux_types[i]!='mass') and (flux_types[i]!='energy') and (flux_types[i]!='entropy') and \
           (flux_types[i]!='O_ion_mass') and (flux_types[i]!='angular_momentum'):
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
              tablename, save_suffix, surfaces, flux_types, sat_dir, sat_radius, masses_dir)
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
                     tablename, save_suffix, surfaces, flux_types, sat_dir, sat_radius, masses_dir)))
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
                 tablename, save_suffix, surfaces, flux_types, sat_dir, sat_radius, masses_dir)))
        for t in threads:
            t.start()
        for t in threads:
            t.join()

    print(str(datetime.datetime.now()))
    print("All snapshots finished!")
