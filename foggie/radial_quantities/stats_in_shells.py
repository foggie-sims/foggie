"""
Filename: stats_in_shells.py
Author: Cassi
Date created: 7-30-20
Date last modified: 7-31-20
This file takes command line arguments and computes statistics of fields in shells:
median, interquartile range, standard deviation, and mean, and saves these to file.

Dependencies:
utils/consistency.py
utils/get_refine_box.py
utils/get_halo_center.py
utils/get_proper_box_size.py
utils/get_run_loc_etc.py
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
from scipy.optimize import curve_fit
import scipy.special as sse
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

    parser = argparse.ArgumentParser(description='Calculates and saves to file statistics of fields in shells: median, interquartile range, standard deviation, and mean.')

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

    parser.add_argument('--remove_sats', dest='remove_sats', action='store_true',
                        help='Do you want to remove satellites before calculating averages and PDFs? ' + \
                        "This requires a satellites.hdf5 file to exist for the halo/run you're using." + \
                        ' Default is no.')
    parser.set_defaults(remove_sats=False)

    parser.add_argument('--sat_radius', metavar='sat_radius', type=float, action='store', \
                        help='What radius (in kpc) do you want to excise around satellites? Default is 10.')
    parser.set_defaults(sat_radius=10.)

    parser.add_argument('--cgm_filter', dest='cgm_filter', action='store_true',
                        help='Do you want to remove gas above a certain density threshold and below\n' + \
                        'a certain temperature threshold defined in consistency.py? This is much more\n' + \
                        'effective at removing gas associated with satellites than the --remove_sats option,\n' + \
                        "but shouldn't be used if you're calculating things changing phases/species\n" + \
                        'in tandem with flux_tracking. Default is not to do this.')
    parser.set_defaults(cgm_filter=False)

    parser.add_argument('--stat_type', metavar='stat_type', type=str, action='store', \
                        help='What fields do you want to compute stats for? Currently, the options are "temperature"' + \
                        ' "pressure" "density" "energy" "entropy" "metallicity" and "velocity".\nYou can compute all of them by inputting ' + \
                        '"temperature,pressure,density,energy,entropy,metallicity,velocity" (no spaces!) ' + \
                        'and the default is to do all.')
    parser.set_defaults(stat_type="temperature,pressure,density,energy,entropy,metallicity,velocity")

    parser.add_argument('--stat_weight', metavar='stat_weight', type=str, action='store', \
                        help='What field do you want to weight the statistics by? Options are volume and mass. Defaults is volume.')
    parser.set_defaults(stat_weight='volume')

    parser.add_argument('--pdf', dest='pdf', action='store_true', \
                        help='Do you want to save a histogram of the PDF in addition to the statistical quantities?\n' + \
                        'Note this will drastically increase file size so think carefully about what stat_type\n' + \
                        'and how many geometric bins you want. Default is not to save the PDFs.')
    parser.set_defaults(pdf=False)

    parser.add_argument('--disk', dest='disk', action='store_true',
                        help='Do you want to calculate theta and phi velocities and kinetic energies' + \
                        'relative to the disk? Default is no. If you use a shape that is aligned with the disk,' + \
                        'this will be done automatically.')
    parser.set_defaults(disk=False)

    parser.add_argument('--shape', metavar='shape', type=str, action='store', \
                        help='What shape for computing properties for a segment of the CGM? Default is sphere' + \
                        ' and the other options are "frustum", "cylinder", or "ellipse" (elliptical cones).\n' + \
                        'Note that all surfaces will be centered on halo center.\n' + \
                        'To specify the shape, size, and orientation of the segment you want, ' + \
                        'input a list as follows (don\'t forget the outer quotes, and put the shape in a different quote type!):\n' + \
                        'If you want a sphere, give:\n' + \
                        '"[\'sphere\', inner_radius, outer_radius, num_steps]"\n' + \
                        'where inner_radius is the inner boundary as a fraction of refine_width, outer_radius is the outer ' + \
                        'boundary as a fraction (or multiple) of refine_width,\nand num_radii is the number of radii where you want to chunk up the shape ' + \
                        'between inner_radius and outer_radius\n' + \
                        '(inner_radius and outer_radius are automatically included).\n' + \
                        'If you want a frustum, give:\n' + \
                        '"[\'frustum\', axis, inner_radius, outer_radius, num_steps, opening_angle]"\n' + \
                        'where axis specifies what axis to align the frustum with and can be one of the following:\n' + \
                        "'x'\n'y'\n'z'\n'minor' (aligns with disk minor axis)\n(x,y,z) (a tuple giving a 3D vector for an arbitrary axis).\n" + \
                        'For all axis definitions other than the arbitrary vector, if the axis string starts with a \'-\', it will compute a frustum pointing in the opposite direction.\n' + \
                        'inner_radius, outer_radius, and num_steps are the same as for the sphere\n' + \
                        'and opening_angle gives the angle in degrees of the opening angle of the cone, measured from axis.\n' + \
                        'If you want a cylinder, give:\n' + \
                        '"[\'cylinder\', axis, bottom_edge, top_edge, radius, step_direction, num_steps]"\n' + \
                        'where axis specifies what axis to align the length of the cylinder with and can be one of the following:\n' + \
                        "'x'\n'y'\n'z'\n'minor' (aligns with disk minor axis)\n(x,y,z) (a tuple giving a 3D vector for an arbitrary axis).\n" + \
                        'For all axis definitions other than the arbitrary vector, if the axis string starts with a \'-\', it will compute a cylinder pointing in the opposite direction.\n' + \
                        'bottom_edge, top_edge, and radius give the dimensions of the cylinder,\n' + \
                        'by default in units of refine_width (unless the --kpc option is specified), where bottom_ and top_edge are' + \
                        ' distance from halo center.\n' + \
                        "step_direction can be 'height', which will compute stats in circular slices in the cylinder parallel to the flat sides, or 'radius', which\n" + \
                        "will compute stats in cylindrical shells at different radii within the cylinder perpendicular to the cylinder's flat sides.\n" + \
                        "'num_steps' gives the number of places (either heights or radii) within the cylinder where to calculate.\n" + \
                        'If you want elliptical cone(s), give:\n' + \
                        '"[\'ellipse\', ellipse_filename, inner_radius, outer_radius, num_steps]"\n' + \
                        'where ellipse_filename is the name of the file where the ellipse(s) parameters are saved,\n' + \
                        'and inner_radius, outer_radius, and num_steps are the same as for the other shapes.\n' + \
                        'If you want multiple regions, use the same syntax but put each region list into a larger list, like:\n' + \
                        '"[[\'frustum\', \'x\', 0.05, 2, 200, 15],[\'frustum\', \'y\', 0.05, 2, 200, 30]]"\n' + \
                        'If you specify multiple shapes, they must all have the same inner_radius, outer_radius, and num_steps.\n' + \
                        'You can specify many different shapes at once, as long as none of them are cylinders.\n' + \
                        'If you want a cylinder, you can only do one at at time.')
    parser.set_defaults(shape="['sphere', 0.05, 2., 200]")

    parser.add_argument('--inverse', dest='inverse', action='store_true',
                        help='Do you want to calculate for everything *outside* of the shape(s) you\'ve specified?')
    parser.set_defaults(inverse=False)

    parser.add_argument('--kpc', dest='units_kpc', action='store_true',
                        help='Do you want to give inner_radius and outer_radius or bottom_edge and top_edge in the shape arguments ' + \
                        'in kpc rather than the default of fraction of refine_width? Default is no.\n' + \
                        'Note that if you want to track PDFs over time, using kpc instead of fractions ' + \
                        'of refine_width will be less accurate because refine_width is comoving and kpc are not.')
    parser.set_defaults(units_kpc=False)

    parser.add_argument('--Rvir', dest='units_rvir', action='store_true',
                        help='Do you want to give inner_radius and outer_radius or bottom_edge and top_edge in the shape arguments ' + \
                        'as fractions of the virial radius rather than the default of fraction of refine_width? Default is no.\n' + \
                        'Note that if you want to track PDFs over time, using anything other than fractions ' + \
                        'of refine_width will be less accurate because refine_width is comoving and Rvir is not.')
    parser.set_defaults(units_rvir=False)

    parser.add_argument('--vel_fit', dest='vel_fit', action='store_true',
                        help='Do you want to fit a Gaussian to the velocity distributions and save the best-fit parameters? Default is no.')
    parser.set_defaults(vel_fit=False)

    parser.add_argument('--vel_cut', dest='vel_cut', action='store_true',
                        help='Do you want to cut everything to remove inflowing material around\n' + \
                        'the free-fall velocity before calculating stats, pdfs, or fitting? Default is no.')
    parser.set_defaults(vel_cut=False)

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

    parser.add_argument('--refined_only', dest='refined_only', action='store_true', \
                        help='Do you want to calculate totals of only those cells refined to at least\n' + \
                        'level 9? This enforces only the refine box is used for volumes that are partially\n' + \
                        'in, partially out of the refine box. Default is not to do this.')
    parser.set_defaults(refined_only=False)


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
        elif ('pdf' in key) and (args.stat_weight=='mass'):
            table[key].unit = 'mass PDF'
        elif ('pdf' in key) and (args.stat_weight=='volume'):
            table[key].unit = 'volume PDF'
        elif ('energy' in key):
            table[key].unit = 'erg/g'
        elif ('entropy' in key):
            table[key].unit = 'log cm**2*keV'
        elif ('temperature' in key) or (key=='log_Tcassi'):
            table[key].unit = 'log K'
        elif ('density' in key):
            table[key].unit = 'log g/cm**3'
        elif ('pressure' in key):
            table[key].unit = 'log erg/cm**3'
        elif ('metallicity' in key):
            table[key].unit = 'log Zsun'
        elif ('velocity' in key):
            table[key].unit == 'km/s'
    return table

def make_table(stat_types, shape_type):
    '''Makes the giant table that will be saved to file.'''

    if (shape_type[0]=='sphere'):
        names_list = ['redshift', 'inner_radius', 'outer_radius']
        types_list = ['f8', 'f8', 'f8']
    if (shape_type[0]=='cylinder'):
        if (shape_type[1]=='radius'):
            names_list = ['redshift', 'inner_radius', 'outer_radius']
            types_list = ['f8', 'f8', 'f8']
        if (shape_type[1]=='height'):
            names_list = ['redshift', 'bottom_edge', 'top_edge']
            types_list = ['f8', 'f8', 'f8']

    dir_name = ['net_', '_in', '_out']
    if (args.temp_cut): temp_name = ['', 'cold_', 'cool_', 'warm_', 'hot_']
    else: temp_name = ['']
    stat_names = ['_med', '_iqr', '_avg', '_std']
    for i in range(len(stat_types)):
        for j in range(len(dir_name)):
            for k in range(len(temp_name)):
                for l in range(len(stat_names)):
                    if (j==0): name = dir_name[j]
                    else: name = ''
                    name += temp_name[k]
                    name += stat_types[i]
                    name += stat_names[l]
                    if (j>0): name += dir_name[j]
                    names_list += [name]
                    types_list += ['f8']
                if (args.vel_fit) and ('velocity' in stat_types[i]) and (j==0):
                    if (stat_types[i]=='radial_velocity'): fit_names = ['_mu', '_sig', '_A1', '_A2']
                    else: fit_names = ['_mu', '_sig', '_A']
                    for fit_name in fit_names:
                        name = dir_name[j]
                        name += temp_name[k]
                        name += stat_types[i]
                        name += fit_name
                        names_list += [name]
                        types_list += ['f8']

    table = Table(names=names_list, dtype=types_list)

    return table

def make_pdf_table(stat_types, shape_type):
    '''Makes the giant table for the PDFs that will be saved to file.'''

    if (shape_type[0]=='sphere'):
        names_list = ['redshift', 'inner_radius', 'outer_radius']
        types_list = ['f8', 'f8', 'f8']
    if (shape_type[0]=='cylinder'):
        if (shape_type[1]=='radius'):
            names_list = ['redshift', 'inner_radius', 'outer_radius']
            types_list = ['f8', 'f8', 'f8']
        if (shape_type[1]=='height'):
            names_list = ['redshift', 'bottom_edge', 'top_edge']
            types_list = ['f8', 'f8', 'f8']

    dir_name = ['net_', '_in', '_out']
    if (args.temp_cut): temp_name = ['', 'cold_', 'cool_', 'warm_', 'hot_']
    else: temp_name = ['']
    for i in range(len(stat_types)):
        names_list += ['lower_' + stat_types[i], 'upper_' + stat_types[i]]
        types_list += ['f8', 'f8']
        for j in range(3):
            for k in range(len(temp_name)):
                if (j==0): name = dir_name[j]
                else: name = ''
                name += temp_name[k]
                name += stat_types[i]
                if (j>0): name += dir_name[j]
                names_list += [name + '_pdf']
                types_list += ['f8']

    table = Table(names=names_list, dtype=types_list)

    return table

def gauss(x, mu, sig, A):
    return A*np.exp(-(x-mu)**2/2/sig**2)

def gauss_pos(x, mu, sig, A):
    func = A*np.exp(-(x-mu)**2/2/sig**2)
    func[x<=0.] = 0.
    return func

def double_gauss(x, mu1, sig1, A1, mu2, sig2, A2):
    func1 = A1*np.exp(-(x-mu1)**2/2/sig1**2)
    func2 = A2*np.exp(-(x-mu2)**2/2/sig2**2)
    func2[x<=0.] = 0.
    return func1 + func2

def exp_gauss(x, mu, sig, A, lam):
    func = A*sig*lam*np.sqrt(np.pi/2.)*np.exp(lam/2.*(2.*mu+lam*sig**2.-2.*x))*sse.erfc((mu+lam*sig**2.-x)/(np.sqrt(2.)*sig))
    func[x<=0.] = 0.
    return func

def weighted_quantile(values, weights, quantiles):
    """ Very close to numpy.percentile, but supports weights.
    NOTE: quantiles should be in [0, 1]!
    :param values: numpy.array with data
    :param weights: array-like of the same length as `array`
    :param quantiles: array-like with many quantiles needed
    :return: numpy.array with computed quantiles.
    """

    sorter = np.argsort(values)
    values = values[sorter]
    weights = weights[sorter]
    weighted_quantiles = np.cumsum(weights) - 0.5 * weights
    weighted_quantiles /= np.sum(weights)

    return np.interp(quantiles, weighted_quantiles, values)

def weighted_avg_and_std(values, weights):
    """
    Return the weighted average and standard deviation.
    values, weights -- Numpy ndarrays with the same shape.
    """
    average = np.average(values, weights=weights)
    variance = np.average((values-average)**2, weights=weights)
    return average, np.sqrt(variance)

def calc_stats(ds, snap, zsnap, refine_width_kpc, tablename, save_suffix, shape_args, stat_types, Menc_profile, sat=False, sat_radius=0., inverse=False, disk=False, weight_field='volume', Rvir=100.):
    '''Calculates the statistics specified by 'stat_types' in the subset of the dataset 'ds' given by
    'shape_args'. 'snap' and 'zsnap' are the snapshot name and redshift, respectively, 'refine_width_kpc'
    is the size of the refine box in kpc, 'tablename' is the name of the table where the statistics
    will be saved, 'Menc_profile' is the enclosed mass profile, 'sat' is either False if satellites
    are not removed or the table of satellite positions if they are, and 'sat_radius' is the radius
    (in kpc) around satellites to excise. If 'inverse' is True, then calculate for everything *outside*
    of the shapes given in 'shape_args'. If 'disk' is True, then at least one shape requires disk-relative fields.
    'weight_field' gives which field to weight the statistics by, and can be either 'volume' or 'mass'.'''

    cmtopc = 3.086e18
    stoyr = 3.154e7
    gtoMsun = 1.989e33
    G = 6.673e-8

    stats = []
    stat_filename = ''
    if ('temperature' in stat_types):
        stats.append('log_temperature')
        stat_filename += '_temperature'
    if ('pressure' in stat_types):
        stats.append('log_pressure')
        stat_filename += '_pressure'
    if ('density' in stat_types):
        stats.append('log_density')
        stat_filename += '_density'
    if ('energy' in stat_types):
        stats.append('thermal_energy')
        stats.append('kinetic_energy')
        stats.append('potential_energy')
        stats.append('total_energy')
        stats.append('virial_energy')
        stat_filename += '_energy'
    if ('entropy' in stat_types):
        stats.append('log_entropy')
        stat_filename += '_entropy'
    if ('metallicity' in stat_types):
        stats.append('log_metallicity')
        stat_filename += '_metallicity'
    if ('velocity' in stat_types):
        stats.append('theta_velocity')
        stats.append('phi_velocity')
        stats.append('tangential_velocity')
        stats.append('radial_velocity')
        stat_filename += '_velocity'

    # Define list of ways to chunk up the shape over radius or height
    if (shape_args[0][0]=='cylinder'):
        table = make_table(stats, ['cylinder', shape_args[0][7]])
        if (args.pdf):
            table_pdf = make_pdf_table(stats, ['cylinder', shape_args[0][7]])
        bottom_edge = shape_args[0][1]
        top_edge = shape_args[0][2]
        cyl_radius = shape_args[0][6]
        num_steps = shape_args[0][3]
        if (shape_args[0][7]=='height'):
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
        inner_radius = shape_args[0][1]
        outer_radius = shape_args[0][2]
        num_steps = shape_args[0][3]
        table = make_table(stats, ['sphere', 0])
        if (args.pdf):
            table_pdf = make_pdf_table(stats, ['sphere', 0])
        if (args.units_kpc):
            dr = (outer_radius-inner_radius)/num_steps
            chunks = ds.arr(np.arange(inner_radius,outer_radius+dr,dr), 'kpc')
        elif (args.units_rvir):
            dr = (outer_radius-inner_radius)/num_steps*Rvir
            chunks = ds.arr(np.arange(inner_radius*Rvir,outer_radius*Rvir+dr,dr), 'kpc')
        else:
            dr = (outer_radius-inner_radius)/num_steps*refine_width_kpc
            chunks = np.arange(inner_radius*refine_width_kpc,outer_radius*refine_width_kpc+dr,dr)

    print('Loading field arrays')
    sphere = ds.sphere(ds.halo_center_kpc, chunks[-1])
    if (args.cgm_filter):
        sphere = sphere.cut_region("(obj['density'] < %.2e) & (obj['temperature'] > %.2e)" % (cgm_density_max, cgm_temperature_min))
    if (args.refined_only):
        sphere = sphere.cut_region("(obj['grid_level'] > 8)")

    x = (sphere['gas','x'].in_units('kpc') - ds.halo_center_kpc[0]).v
    y = (sphere['gas','y'].in_units('kpc') - ds.halo_center_kpc[1]).v
    z = (sphere['gas','z'].in_units('kpc') - ds.halo_center_kpc[2]).v
    if (disk):
        x_disk = sphere['gas','x_disk'].in_units('kpc').v
        y_disk = sphere['gas','y_disk'].in_units('kpc').v
        z_disk = sphere['gas','z_disk'].in_units('kpc').v
    theta = sphere['gas','theta_pos'].v
    phi = sphere['gas', 'phi_pos'].v
    radius = sphere['gas','radius_corrected'].in_units('kpc').v
    rad_vel = sphere['gas','radial_velocity_corrected'].in_units('km/s').v
    temperature = np.log10(sphere['gas','temperature'].in_units('K').v)
    if (weight_field=='volume'):
        weights = sphere['gas','cell_volume'].v
    if (weight_field=='mass'):
        weights = sphere['gas','cell_mass'].v
    fields = []
    x_ranges = []
    if ('temperature' in stat_types):
        fields.append(temperature)
        x_ranges.append([3,8])
    if ('pressure' in stat_types):
        pressure = np.log10(sphere['gas', 'pressure'].in_units('erg/cm**3').v)
        fields.append(pressure)
        x_ranges.append([-18,-12])
    if ('density' in stat_types):
        density = np.log10(sphere['gas','density'].in_units('g/cm**3').v)
        fields.append(density)
        x_ranges.append([-31,-26])
    if ('energy' in stat_types):
        kinetic_energy = sphere['gas','kinetic_energy_corrected'].in_units('erg').v/sphere['gas','cell_mass'].in_units('g').v
        thermal_energy = sphere['gas','thermal_energy'].in_units('erg/g').v
        potential_energy = -G * Menc_profile(radius)*gtoMsun / (radius*1000.*cmtopc)
        total_energy = kinetic_energy + thermal_energy + potential_energy
        virial_energy = 2.*(thermal_energy + kinetic_energy) + 3./2.*potential_energy
        fields.append(thermal_energy)
        fields.append(kinetic_energy)
        fields.append(potential_energy)
        fields.append(total_energy)
        fields.append(virial_energy)
        x_ranges.append([-4e14,6e14])
        x_ranges.append([-4e14,6e14])
        x_ranges.append([-4e14,6e14])
        x_ranges.append([-4e14,6e14])
        x_ranges.append([-4e14,6e14])
    if ('entropy' in stat_types):
        entropy = np.log10(sphere['gas','entropy'].in_units('keV*cm**2').v)
        fields.append(entropy)
        x_ranges.append([-3,5])
    if ('metallicity' in stat_types):
        metallicity = np.log10(sphere['gas', 'metallicity'].in_units('Zsun').v)
        fields.append(metallicity)
        x_ranges.append([-4,1])
    if ('velocity' in stat_types):
        if (disk):
            theta_vel = sphere['gas','vtheta_disk'].in_units('km/s').v
            phi_vel = sphere['gas','vphi_disk'].in_units('km/s').v
            tan_vel = sphere['gas','vtan_disk'].in_units('km/s').v
        else:
            theta_vel = sphere['gas','theta_velocity_corrected'].in_units('km/s').v
            phi_vel = sphere['gas', 'phi_velocity_corrected'].in_units('km/s').v
            tan_vel = sphere['gas', 'tangential_velocity_corrected'].in_units('km/s').v
        fields.append(theta_vel)
        fields.append(phi_vel)
        fields.append(tan_vel)
        fields.append(rad_vel)
        x_ranges.append([-200,200])
        x_ranges.append([-200,200])
        x_ranges.append([-200,200])
        x_ranges.append([-200,500])

    # Cut to just the shapes specified
    if (disk):
        if (shape_args[0][0]=='cylinder'):
            bool_inshapes, radius = segment_region(x, y, z, theta, phi, radius, shapes, refine_width_kpc, \
              x_disk=x_disk, y_disk=y_disk, z_disk=z_disk, Rvir=Rvir, units_kpc=args.units_kpc, units_rvir=args.units_rvir)
        else:
            bool_inshapes = segment_region(x, y, z, theta, phi, radius, shapes, refine_width_kpc, \
              x_disk=x_disk, y_disk=y_disk, z_disk=z_disk, Rvir=Rvir, units_kpc=args.units_kpc, units_rvir=args.units_rvir)
    else:
        if (shape_args[0][0]=='cylinder'):
            bool_inshapes, radius = segment_region(x, y, z, theta, phi, radius, shapes, refine_width_kpc, \
              Rvir=Rvir, units_kpc=args.units_kpc, units_rvir=args.units_rvir)
        else:
            bool_inshapes = segment_region(x, y, z, theta, phi, radius, shapes, refine_width_kpc, \
              Rvir=Rvir, units_kpc=args.units_kpc, units_rvir=args.units_rvir)
    if (inverse): bool_inshapes = ~bool_inshapes
    x = x[bool_inshapes]
    y = y[bool_inshapes]
    z = z[bool_inshapes]
    theta = theta[bool_inshapes]
    phi = phi[bool_inshapes]
    radius = radius[bool_inshapes]
    rad_vel = rad_vel[bool_inshapes]
    temperature = temperature[bool_inshapes]
    weights = weights[bool_inshapes]
    for i in range(len(fields)):
        fields[i] = fields[i][bool_inshapes]

    # Load list of satellite positions
    if (sat_radius!=0):
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
    else:
        bool_nosat = np.ones(len(x), dtype=bool)

    x = x[bool_nosat]
    y = y[bool_nosat]
    z = z[bool_nosat]
    theta = theta[bool_nosat]
    phi = phi[bool_nosat]
    radius = radius[bool_nosat]
    rad_vel = rad_vel[bool_nosat]
    temperature = temperature[bool_nosat]
    weights = weights[bool_nosat]
    for i in range(len(fields)):
        fields[i] = fields[i][bool_nosat]

    # Loop over chunks and compute stats to add to table
    if (args.temp_cut): temps = [0.,4.,5.,6.,12.]
    else: temps = [0.]
    # Index r is for radial/height chunk, index i is for the property we're computing stats for,
    # index j is for net, in, or out, and index k is for temperature, net, cold, cool, warm, hot
    for r in range(len(chunks)-1):
        if (r%10==0): print("Computing chunk " + str(r) + "/" + str(len(chunks)) + \
                            " for snapshot " + snap)
        inner = chunks[r]
        outer = chunks[r+1]
        row = [zsnap, inner, outer]
        if (args.pdf):
            pdf_array = []
        bool_r = (radius > inner) & (radius < outer)
        rad_vel_r = rad_vel[bool_r]
        temp_r = temperature[bool_r]
        weights_r = weights[bool_r]
        if (args.vel_cut):
            center = 0.5*(inner + outer)
            rho = Menc_profile(center)*gtoMsun/((center*1000*cmtopc)**3.) * 3./(4.*np.pi)
            vff = -(center*1000*cmtopc)/np.sqrt(3.*np.pi/(32.*G*rho))/1e5
            vesc = np.sqrt(2.*G*Menc_profile(center)*gtoMsun/(center*1000.*cmtopc))/1e5
            temp_r = temp_r[(rad_vel_r > 0.5*vff)]
            weights_r = weights_r[(rad_vel_r > 0.5*vff)]
            rad_vel_cut_r = rad_vel_r[(rad_vel_r > 0.5*vff)]
        else:
            rad_vel_cut_r = rad_vel_r
        for i in range(len(fields)):
            field = fields[i]
            field_r = field[bool_r]
            if (args.vel_cut):
                field_r = field_r[(rad_vel_r > 0.5*vff)]
            for j in range(3):
                if (j==0): bool_vel = np.ones(len(field_r), dtype=bool)
                if (j==1): bool_vel = (rad_vel_cut_r < 0.)
                if (j==2): bool_vel = (rad_vel_cut_r > 0.)
                field_v = field_r[bool_vel]
                temp_v = temp_r[bool_vel]
                weights_v = weights_r[bool_vel]
                if (args.vel_fit) and (stats[i]=='tangential_velocity') and (j==0):
                    sig_tan = []
                for k in range(len(temps)):
                    if (k==0):
                        bool_temp = (temp_v > 0.)
                    else:
                        bool_temp = (temp_v > temps[k-1]) & (temp_v < temps[k])
                    field_t = field_v[bool_temp]
                    weights_t = weights_v[bool_temp]
                    if (args.pdf) and (j==0) and (k==0):
                        if (len(field_t)>0):
                            hist, bin_edges = np.histogram(field_t, weights=weights_t, bins=(200), range=x_ranges[i], density=True)
                            pdf_array.append(bin_edges[:-1])
                            pdf_array.append(bin_edges[1:])
                            pdf_array.append(hist)
                    if (len(field_t)==0):
                        row.append(0.)
                        row.append(0.)
                        row.append(0.)
                        row.append(0.)
                        if (args.vel_fit) and ('velocity' in stats[i]) and (j==0):
                            row.append(0.)
                            row.append(0.)
                            row.append(0.)
                            if (stats[i]=='radial_velocity'): row.append(0.)
                        if (args.pdf):
                            pdf_array.append(np.zeros(200))
                            if (j==0) and (k==0):
                                pdf_array.append(np.zeros(200))
                                pdf_array.append(np.zeros(200))
                    else:
                        quantiles = weighted_quantile(field_t, weights_t, np.array([0.25,0.5,0.75]))
                        row.append(quantiles[1])
                        row.append(quantiles[2]-quantiles[0])
                        avg, std = weighted_avg_and_std(field_t, weights_t)
                        row.append(avg)
                        row.append(std)
                        if (args.pdf) and (j+k>0):
                            hist, bins = np.histogram(field_t, weights=weights_t, bins=(200), range=x_ranges[i], density=True)
                            pdf_array.append(hist)
                        if (args.vel_fit) and ('velocity' in stats[i]) and (j==0):
                            hist_vel, bins_vel = np.histogram(field_t, weights=weights_t, bins=(200), range=x_ranges[i], density=True)
                            bin_centers = np.diff(bins_vel) + bins_vel[:-1]
                            if (stats[i]=='radial_velocity'):
                                #skew = 3./2.*(avg - quantiles[1])/std
                                #if (skew<0.): skew=0.01
                                #guesses = [avg - std*skew**(1./3.), std*np.sqrt(1. - skew**(2./3.)), np.max(hist_vel), 1./(std*skew**(1./3.))]
                                guesses = [0., sig_tan[k], hist_vel[np.where(bin_centers>=0.)[0][0]]]
                                try:
                                    #params, cov = curve_fit(gauss, bin_centers, hist_vel, p0=guesses, \
                                      #bounds=([0.,sig_tan[j][k],0.],[0.1,sig_tan[j][k]+0.1,2.*np.max(hist_vel)]))
                                    # Start with estimate of how much turbulence contributes to the radial velocity distribution
                                    fit = gauss(bin_centers, 0., sig_tan[k], hist_vel[np.where(bin_centers>=0.)[0][0]])
                                    A2 = hist_vel[np.where(bin_centers>=0.)[0][0]]
                                    # Subtract this out to get just the outflow at positive velocities
                                    leftover = hist_vel - fit
                                    leftover_pos = leftover[bin_centers>=0.]
                                    bins_pos = bin_centers[bin_centers>=0.]
                                    # Use the outflow as the initial guesses to the fit
                                    guesses = [0., sig_tan[k], A2, bins_pos[leftover_pos==np.max(leftover_pos)][0], sig_tan[k], np.max(leftover_pos)]
                                    # Fit two Gaussians, one that is forced to the properties of turbulence and one that is only at positive velocities
                                    params, cov = curve_fit(double_gauss, bin_centers, hist_vel, p0=guesses, \
                                      bounds=([0., sig_tan[k], 0.5*A2, 0., 0., 0.],[0.1,sig_tan[k]+0.1, 1.5*A2, np.inf, np.inf, np.inf]))
                                    # Save the parameters of the outflow Gaussian and the height of the turbulence Gaussian (its mu and std are saved elsewhere)
                                    row.append(params[3])
                                    row.append(params[4])
                                    row.append(params[5])
                                    row.append(params[2])
                                except RuntimeError:
                                    row.append(0.)
                                    row.append(0.)
                                    row.append(0.)
                                    row.append(0.)
                            else:
                                guesses = [avg, std, np.max(hist)]
                                try:
                                    params, cov = curve_fit(gauss, bin_centers, hist_vel, p0=guesses)
                                    if (stats[i]=='tangential_velocity'): sig_tan.append(params[1])
                                    row.append(params[0])
                                    row.append(params[1])
                                    row.append(params[2])
                                except RuntimeError:
                                    if (stats[i]=='tangential_velocity'): sig_tan.append(0.)
                                    row.append(0.)
                                    row.append(0.)
                                    row.append(0.)
        if (args.pdf):
            pdf_array = np.vstack(pdf_array)
            pdf_array = np.transpose(pdf_array)
            for p in range(len(pdf_array)):
                row_pdf = [zsnap, inner, outer]
                row_pdf += list(pdf_array[p])
                table_pdf.add_row(row_pdf)
        table.add_row(row)

    table = set_table_units(table)
    if (args.pdf): table_pdf = set_table_units(table_pdf)

    # Save to file
    if (sat_radius!=0.):
        table.write(tablename + '_nosat' + stat_filename + save_suffix + '.hdf5', path='all_data', serialize_meta=True, overwrite=True)
    else:
        table.write(tablename + stat_filename + save_suffix + '.hdf5', path='all_data', serialize_meta=True, overwrite=True)
    if (args.pdf):
        if (sat_radius!=0.):
            table_pdf.write(tablename + '_nosat' + stat_filename + '_pdf' + save_suffix + '.hdf5', path='all_data', serialize_meta=True, overwrite=True)
        else:
            table_pdf.write(tablename + stat_filename + '_pdf' + save_suffix + '.hdf5', path='all_data', serialize_meta=True, overwrite=True)

    return "Stats have been calculated for snapshot " + snap + "!"


def load_and_calculate(system, foggie_dir, run_dir, track, halo_c_v_name, snap, tablename, save_suffix, shape_args, stat_types, sat_dir, sat_radius, masses_dir):
    '''This function loads a specified snapshot 'snap' located in the 'run_dir' within the
    'foggie_dir', the halo track 'track', the name of the halo_c_v file, the name of the snapshot,
    the name of the table to output, the mass enclosed table, the list of surface arguments, and
    the directory where the satellites file is saved, then
    does the calculation on the loaded snapshot.'''

    snap_name = foggie_dir + run_dir + snap + '/' + snap
    # Copy output to temp directory if on pleiades
    if (system=='pleiades_cassi'):
        print('Copying directory to /tmp')
        snap_dir = '/tmp/' + snap
        shutil.copytree(foggie_dir + run_dir + snap, snap_dir)
        snap_name = snap_dir + '/' + snap

    # Load the snapshot depending on if disk minor axis is needed
    for i in range(len(shape_args)):
        if (((shape_args[i][0]=='frustum') or (shape_args[i][0]=='cylinder')) and (shape_args[i][4]=='disk minor axis')) or (args.disk):
            ds, refine_box = foggie_load(snap_name, track, disk_relative=True, halo_c_v_name=halo_c_v_name)
            disk = True
        else:
            ds, refine_box = foggie_load(snap_name, track, do_filter_particles=False, halo_c_v_name=halo_c_v_name)
            disk = False
    refine_width_kpc = ds.quan(ds.refine_width, 'kpc')
    zsnap = ds.get_parameter('CosmologyCurrentRedshift')

    # Specify the file where the list of satellites is saved
    if (sat_radius!=0.):
        sat_file = sat_dir + 'satellites.hdf5'
        sat = Table.read(sat_file, path='all_data')
    else:
        sat = False

    # Load the mass enclosed profile
    if (zsnap > 2.):
        masses = Table.read(masses_dir + 'masses_z-gtr-2.hdf5', path='all_data')
    else:
        masses = Table.read(masses_dir + 'masses_z-less-2.hdf5', path='all_data')
    rvir_masses = Table.read(masses_dir + 'rvir_masses.hdf5', path='all_data')
    snap_ind = masses['snapshot']==snap
    Menc_profile = IUS(np.concatenate(([0],masses['radius'][snap_ind])), np.concatenate(([0],masses['total_mass'][snap_ind])))
    Rvir = rvir_masses['radius'][rvir_masses['snapshot']==snap]

    # Do the actual calculation
    message = calc_stats(ds, snap, zsnap, refine_width_kpc, tablename, save_suffix, shape_args, \
      stat_types, sat=sat, sat_radius=sat_radius, Menc_profile=Menc_profile, inverse=args.inverse, disk=disk, weight_field=args.stat_weight, Rvir=Rvir)

    # Delete output from temp directory if on pleiades
    if (system=='pleiades_cassi'):
        print('Deleting directory from /tmp')
        shutil.rmtree(snap_dir)

    print(message)

if __name__ == "__main__":
    args = parse_args()
    print(args.halo)
    print(args.run)
    print(args.system)
    foggie_dir, output_dir, run_dir, code_path, trackname, haloname, spectra_dir, infofile = get_run_loc_etc(args)

    # Set directory for output location, making it if necessary
    prefix = output_dir + 'stats_halo_00' + args.halo + '/' + args.run + '/'
    if not (os.path.exists(prefix)): os.system('mkdir -p ' + prefix)

    print('foggie_dir: ', foggie_dir)
    halo_c_v_name = code_path + 'halo_infos/00' + args.halo + '/' + args.run + '/halo_c_v'
    masses_dir = code_path + 'halo_infos/00' + args.halo + '/' + args.run + '/'

    if (args.save_suffix!=''):
        save_suffix = '_' + args.save_suffix
    else:
        save_suffix = ''

    # Specify where satellite files are saved
    if (args.remove_sats):
        sat_dir = code_path + 'halo_infos/00' + args.halo + '/' + args.run + '/'
        sat_radius = args.sat_radius
    else:
        sat_dir = 'sat_dir'
        sat_radius = 0.

    if (args.stat_weight!='volume') and (args.stat_weight!='mass'):
        sys.exit('You must choose either mass or volume to weight the statistics by.')

    shapes = identify_shape(args.shape, args.halo, args.run, units_kpc=args.units_kpc, units_rvir=args.units_rvir)
    outs = make_output_list(args.output, output_step=args.output_step)

    # Build stat type list
    if (',' in args.stat_type):
        stat_types = args.stat_type.split(',')
    else:
        stat_types = [args.stat_type]
    for i in range(len(stat_types)):
        if (stat_types[i]!='temperature') and (stat_types[i]!='energy') and (stat_types[i]!='entropy') and \
           (stat_types[i]!='metallicity') and (stat_types[i]!='velocity') and (stat_types[i]!='density') and \
           (stat_types[i]!='pressure'):
            print('The property   %s   has not been implemented. Ask Cassi to add it.' % (stat_types[i]))
            sys.exit()

    # Loop over outputs, for either single-processor or parallel processor computing
    if (args.nproc==1):
        for i in range(len(outs)):
            snap = outs[i]
            # Make the output table name for this snapshot
            tablename = prefix + snap + '_stats'
            # Do the actual calculation
            load_and_calculate(args.system, foggie_dir, run_dir, trackname, halo_c_v_name, snap, \
              tablename, save_suffix, shapes, stat_types, sat_dir, sat_radius, masses_dir)
    else:
        # Split into a number of groupings equal to the number of processors
        # and run one process per processor
        for i in range(len(outs)//args.nproc):
            threads = []
            for j in range(args.nproc):
                snap = outs[args.nproc*i+j]
                tablename = prefix + snap + '_stats'
                threads.append(multi.Process(target=load_and_calculate, \
			       args=(args.system, foggie_dir, run_dir, trackname, halo_c_v_name, snap, \
                     tablename, save_suffix, shapes, stat_types, sat_dir, sat_radius, masses_dir)))
            for t in threads:
                t.start()
            for t in threads:
                t.join()
        # For any leftover snapshots, run one per processor
        threads = []
        for j in range(len(outs)%args.nproc):
            snap = outs[-(j+1)]
            tablename = prefix + snap + '_stats'
            threads.append(multi.Process(target=load_and_calculate, \
			   args=(args.system, foggie_dir, run_dir, trackname, halo_c_v_name, snap, \
                 tablename, save_suffix, shapes, stat_types, sat_dir, sat_radius, masses_dir)))
        for t in threads:
            t.start()
        for t in threads:
            t.join()

    print("All snapshots finished!")
