"""
Filename: accretion.py
Author: Cassi
First made: 6/21/22
Date last modified: 6/21/22

This file investigates and plots various properties of accretion (breakdown by phase, direction, and
time) at both the galaxy disk and at various places in the halo. It uses edge finding through
binary dilation and a new way of calculating fluxes into and out of arbitrary shapes on a grid.

Dependencies:
utils/consistency.py
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
import scipy.ndimage as ndimage
from scipy.ndimage import gaussian_filter
from scipy.ndimage import uniform_filter1d
import datetime
from scipy.interpolate import InterpolatedUnivariateSpline as IUS
import shutil
import ast
import trident
import matplotlib.pyplot as plt
import healpy

# These imports are FOGGIE-specific files
from foggie.utils.consistency import *
from foggie.utils.get_run_loc_etc import get_run_loc_etc
from foggie.utils.yt_fields import *
from foggie.utils.foggie_load import *
from foggie.utils.analysis_utils import *

# These imports for datashader plots
import datashader as dshader
from datashader.utils import export_image
import datashader.transfer_functions as tf
import pandas as pd
import matplotlib as mpl

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

    parser.add_argument('--surface', metavar='surface', type=str, action='store', \
                        help='What closed surface do you want to compute flux across? Options are sphere,' + \
                        ' cylinder, and disk.\n' + \
                        'To specify the shape, size, and orientation of the surface you want, ' + \
                        'input a list as follows (don\'t forget the outer quotes, and put the shape in a different quote type!):\n' + \
                        'If you want a sphere, give:\n' + \
                        '"[\'sphere\', radius]"\n' + \
                        'where radius is by default in units of kpc but can be in units of Rvir if --Rvir keyword is used.\n' + \
                        'If you want a cylinder, give:\n' + \
                        '"[\'cylinder\', radius, height, axis]"\n' + \
                        'where axis specifies what axis to align the length of the cylinder with and can be one of the following:\n' + \
                        "'x'\n'y'\n'z'\n'minor' (aligns with disk minor axis)\n(x,y,z) (a tuple giving a 3D vector for an arbitrary axis).\n" + \
                        'radius and height give the dimensions of the cylinder,\n' + \
                        'by default in units of kpc but can be in units of Rvir if --Rvir keyword is used.\n' + \
                        'If you want to use the non-standard-shaped galaxy disk as identified by a density cut, give:\n' + \
                        '"[\'disk\']"\n' + \
                        'This will calculate fluxes into and out of whatever shape the disk takes on the grid.\n' + \
                        'The default option is to calculate fluxes into and out of the disk.')
    parser.set_defaults(surface="['disk']")

    parser.add_argument('--Rvir', dest='Rvir', action='store_true',
                        help='Do you want to specify your surface dimensions in units of Rvir rather than the default of kpc?\n' + \
                        'Default is no.')
    parser.set_defaults(Rvir=False)

    parser.add_argument('--flux_type', metavar='flux_type', type=str, action='store', \
                        help='What fluxes do you want to compute? Currently, the options are "mass" (includes metal masses)' + \
                        ' and "energy".\nYou can compute all of them by inputting ' + \
                        '"mass,energy" (no spaces!) ' + \
                        'and the default is to do all.')
    parser.set_defaults(flux_type="mass,energy")

    parser.add_argument('--cgm_only', dest='cgm_only', action='store_true',
                        help='Do you want to remove gas above a certain density threshold?\n' + \
                        'Default is not to do this.')
    parser.set_defaults(cgm_only=False)

    parser.add_argument('--region_filter', metavar='region_filter', type=str, action='store', \
                        help='Do you want to calculate fluxes separately for the different CGM segments?\n' + \
                        'Options are "temperature", "metallicity", and "radial velocity".\n' + \
                        'Default is not to do this.')
    parser.set_defaults(region_filter='none')

    parser.add_argument('--direction', dest='direction', action='store_true',
                        help='Do you want to calculate fluxes separately based on the direction the gas is coming from?\n' + \
                        'This will break up the fluxes into a few bins of theta and phi directions relative to\n' + \
                        'the minor axis of the galaxy disk. Default is not to do this.')
    parser.set_defaults(direction=False)

    parser.add_argument('--spherical', dest='spherical', action='store_true',
                        help='Do you want to use exact spherical boundaries, rather than computing flux into a shape that\n' + \
                            'is approximately spherical? Default is not to do this.')
    parser.set_defaults(spherical=False)

    parser.add_argument('--level', metavar='level', type=int, action='store', \
                        help='What refinement level do you want for the grid on which fluxes are calculated?\n' + \
                        'If using whole refine box or larger, going above level 9 will consume significant memory.\n' + \
                        'Default is level 9 (forced refinement level).')
    parser.set_defaults(level=9)

    parser.add_argument('--nproc', metavar='nproc', type=int, action='store', \
                        help='How many processes do you want? Default is 1 ' + \
                        '(no parallelization), if multiple outputs and multiple processors are' + \
                        ' specified, code will run one output per processor')
    parser.set_defaults(nproc=1)

    parser.add_argument('--save_suffix', metavar='save_suffix', type=str, action='store', \
                        help='Do you want to append a string onto the names of the saved files? Default is no.')
    parser.set_defaults(save_suffix="")

    parser.add_argument('--copy_to_tmp', dest='copy_to_tmp', action='store_true', \
                        help="If running on pleiades, do you want to copy the snapshot to the node's /tmp/\n" + \
                        "directory? This may speed up analysis somewhat, but also requires a large-memory node.\n" + \
                        "Default is not to do this.")
    parser.set_defaults(copy_to_tmp=False)

    parser.add_argument('--plot', metavar='plot', type=str, action='store', \
                        help='Do you want to plot fluxes, in addition to or instead of merely calculating them?\n' + \
                        'Plot options are:\n' + \
                        'accretion_viz          - Projection plots of the shape use for calculation and cells that will accrete to it\n' + \
                        'accretion_direction    - 2D plots in theta and phi bins showing location of inflow cells, colored by mass, temperature, and metallicity\n' + \
                        'accretion_vs_time      - line plot of inward mass flux vs time and redshift\n' + \
                        'accretion_vs_radius    - line plot of various properties of accreting gas vs radius\n' + \
                        'flux_vs_radius         - line plot of accreting mass and metal fluxes vs radius\n' + \
                        'phase_plot             - 2D phase plots of various properties of accreting gas and non-accreting gas in same shell\n' + \
                        'sky_map                - column density maps of all gas and only accreting gas\n' + \
                        'streamlines            - projection plots with streamlines overplotted\n' + \
                        'Default is not to do any plotting. Specify multiple plots by listing separated with commas, no spaces.')
    parser.set_defaults(plot='none')

    parser.add_argument('--dark_matter', dest='dark_matter', action='store_true', \
                        help='Do you want to calculate fluxes and/or make plots for dark matter too?\n' + \
                        'This is very slow, so the default is not to do this.')
    parser.set_defaults(dark_matter=False)

    parser.add_argument('--load_from_file', metavar='load_from_file', type=str, action='store', \
                        help='If plotting something, do you want to read in from file rather than re-calculating\n' + \
                        "everything? Note this doesn't work for any visualization plots; those need the full simulation\n" + \
                        'output to make. Pass the name of the file after the snapshot name with this option. Default is not to do this.')
    parser.set_defaults(load_from_file='none')

    parser.add_argument('--constant_box', metavar='constant_box', type=float, action='store', \
                        help='Do you want to use a constant box size for all calculations? If so, use\n' + \
                        'this to specify the length of one side of the box, using the same units you used for --surface.\n' + \
                        'This is useful for making videos but may not allow for higher resolution if the box is unnecessarily large.\n' + \
                        'Default is not to do this and dynamically select box size from the surface.')
    parser.set_defaults(constant_box=0.)

    parser.add_argument('--radial_stepping', metavar='radial_stepping', type=int, action='store',\
                        help='If using the sphere surface type, do you want to calculate flux and make\n' + \
                        'plots for several radii within the sphere? If so, use this keyword to specify\n' + \
                        'how many steps of radius you want. Default is not to do this.')
    parser.set_defaults(radial_stepping=0)

    parser.add_argument('--calculate', metavar='calculate', type=str, action='store', \
                        help='What do you want to calculate and save to file? Options are:\n' + \
                        'fluxes (default)    -  will calculate fluxes onto shape\n' + \
                        'accretion_compare   -  will calculate statistics (mean, median, etc) of gas properties\n' + \
                        '                       comparing accreting cells to non-accreting cells in edge around shape')
    parser.set_defaults(calculate='none')

    parser.add_argument('--weight', metavar='weight', type=str, action='store', \
                        help='If calculating statistics of gas properties comparing accretion and non-accretion,\n' + \
                        'what weight do you want to use? Options are volume and mass, and default is mass.')
    parser.set_defaults(weight='mass')

    parser.add_argument('--location_compare', dest='location_compare', action='store_true', \
                        help='If plotting accretion vs time, do you want to plot the accretion at\n' + \
                        'many different locations in the halo on the same plot? Default is not to do this.')
    parser.set_defaults(location_compare=False)

    parser.add_argument('--time_avg', metavar='radius', type=float, action='store', \
                        help='If plotting anything over time and you want to time-average, how long to average over?\n' + \
                        'Give in units of Myr. Default is not to time-average.')
    parser.set_defaults(time_avg=0)

    parser.add_argument('--streamlines', dest='streamlines', action='store_true', \
                        help='Use this to specify the streamlines calculation. Default is not to do this.')
    parser.set_defaults(streamlines=False)


    args = parser.parse_args()
    return args

def make_flux_table(flux_types):
    '''Makes the giant table that will be saved to file.'''

    if (args.radial_stepping > 0):
        names_list = ['radius']
        types_list = ['f8']
    elif ('disk' in surface[0]):
        names_list = ['disk_radius']
        types_list = ['f8']
    else:
        names_list = []
        types_list = []

    if (args.direction):
        names_list += ['phi_bin']
        types_list += ['S5']

    dir_name = ['_in', '_out']
    dv_name = ['','_0-50','_50-100','_100-150','_150-200','_200-300','_300-inf']
    for i in range(len(flux_types)):
        if (args.region_filter != 'none') and ('dm' not in flux_types[i]):
            region_name = ['', 'lowest_', 'low-mid_', 'high-mid_', 'highest_']
        else: region_name = ['']
        for j in range(len(dir_name)):
            if (dir_name[j]=='_in'):
                for dv in range(len(dv_name)):
                    for k in range(len(region_name)):
                        name = ''
                        name += region_name[k]
                        name += flux_types[i]
                        name += dir_name[j]
                        name += dv_name[dv]
                        names_list += [name]
                        types_list += ['f8']
            else:
                for k in range(len(region_name)):
                    name = ''
                    name += region_name[k]
                    name += flux_types[i]
                    name += dir_name[j]
                    names_list += [name]
                    types_list += ['f8']

    table = Table(names=names_list, dtype=types_list)

    return table

def set_flux_table_units(table):
    '''Sets the units for the table. Note this needs to be updated whenever something is added to
    the table. Returns the table.'''

    for key in table.keys():
        if ('mass' in key) or ('metal' in key):
            table[key].unit = 'Msun/yr'
        elif ('radius' in key):
            table[key].unit = 'kpc'
        elif ('energy' in key):
            table[key].unit = 'erg/yr'
        else:
            table[key].unit = 'none'
    return table

def make_props_table(prop_types):
    '''Makes the giant table that will be saved to file.'''

    if (args.radial_stepping > 0):
        names_list = ['radius']
        types_list = ['f8']
    elif ('disk' in surface[0]):
        names_list = ['disk_radius']
        types_list = ['f8']
    else:
        names_list = []
        types_list = []

    if (args.direction):
        names_list += ['phi_bin']
        types_list += ['S5']

    dir_name = ['_all', '_non', '_acc', '_acc_0-50','_acc_50-100','_acc_100-150','_acc_150-200','_acc_200-300','_acc_300-inf']
    stat_names = ['_med', '_iqr', '_avg', '_std']
    for i in range(len(prop_types)):
        if (args.region_filter != 'none'):
            region_name = ['', 'low_', 'mid_', 'high_']
        else: region_name = ['']
        for k in range(len(region_name)):
            for j in range(len(dir_name)):
                for l in range(len(stat_names)):
                    name = region_name[k]
                    name += prop_types[i]
                    name += dir_name[j]
                    if ('mass' not in prop_types[i]) and ('covering' not in prop_types[i]) and ('energy' not in prop_types[i]):
                        name += stat_names[l]
                    if ('mass' in prop_types[i]) or ('energy' in prop_types[i]):
                        if (l==0):
                            names_list += [name]
                            types_list += ['f8']
                    elif ('covering' in prop_types[i]):
                        if (l==0) and (j!=0):
                            names_list += [region_name[k] + 'covering_fraction' + dir_name[j]]
                            types_list += ['f8']
                    else:
                        names_list += [name]
                        types_list += ['f8']

    table = Table(names=names_list, dtype=types_list)

    return table

def set_props_table_units(table):
    '''Sets the units for the table. Note this needs to be updated whenever something is added to
    the table. Returns the table.'''

    for key in table.keys():
        if ('mass' in key) or ('metal' in key):
            table[key].unit = 'Msun'
        elif ('radius' in key):
            table[key].unit = 'kpc'
        elif ('entropy' in key):
            table[key].unit = 'cm**2*keV'
        elif ('energy' in key):
            if ('cooling' in key):
                table[key].unit = 'erg/s'
            else:
                table[key].unit = 'erg'
        elif ('temperature' in key):
            table[key].unit = 'K'
        elif ('density' in key):
            table[key].unit = 'g/cm**3'
        elif ('pressure' in key):
            table[key].unit = 'erg/cm**3'
        elif ('metallicity' in key):
            table[key].unit = 'Zsun'
        elif ('velocity' in key) or ('speed' in key):
            table[key].unit == 'km/s'
        elif ('tcool' in key):
            table[key].unit == 'Myr'
        else:
            table[key].unit = 'none'
    return table

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

def plot_accretion_direction(theta_acc, phi_acc, temperature, metallicity, radial_velocity, cooling_time, mass, metal_mass, theta_out, phi_out, tsnap, zsnap, prefix, snap, radius, save_r, save_dv, theta_acc_dm='', phi_acc_dm='', mass_dm=''):
    '''Plots the temperature, metallicity, radial velocity, cooling time, mass, and metal mass of only those cells
    identified as accreting, while over-plotting contours showing the location of fast outflows.'''

    for c in ['temperature','metallicity','cooling_time','radial_velocity']:
        if (c=='temperature'):
            color_field = 'temperature'
            color_val = np.log10(temperature)
            cmap = sns.blend_palette(('salmon', "#984ea3", "#4daf4a", "#ffe34d", 'darkorange'), as_cmap=True)
            cmin = 4.
            cmax = 7.
            field_label = 'log Temperature [K]'
        elif (c=='metallicity'):
            color_field = 'metallicity'
            color_val = np.log10(metallicity)
            cmap = metal_color_map
            cmin = -2.
            cmax = 0.5
            field_label = 'log Metallicity [$Z_\odot$]'
        elif (c=='cooling_time'):
            color_field = 'cooling-time'
            color_val = np.log10(cooling_time)
            cmap = tcool_color_map
            cmin = np.log10(tcool_min)
            cmax = np.log10(tcool_max)
            field_label = 'log Cooling Time [Myr]'
        elif (c=='radial_velocity'):
            color_field = 'radial-velocity'
            color_val = radial_velocity
            cmap = 'coolwarm'
            cmin = -200.
            cmax = 200.
            field_label = 'Radial velocity [km/s]'

        fig1 = plt.figure(num=1, figsize=(10,6), dpi=300)
        contour_fig = plt.figure(num=2)
        nside = 32
        pixel_acc = healpy.ang2pix(nside, phi_acc, theta_acc)
        u, dup_ind = np.unique(pixel_acc, return_index=True)
        for d in dup_ind:
            val_dup = color_val[np.where(pixel_acc==pixel_acc[d])[0]]
            if (color_field=='temperature') or (color_field=='cooling_time'):
                val_to_set = np.log10(np.mean(10**val_dup))
            else:
                val_to_set = np.mean(val_dup)
            color_val[np.where(pixel_acc==pixel_acc[d])[0]] = val_to_set
        m = np.zeros(healpy.nside2npix(nside))              # make empty array of map pixels
        m[pixel_acc] = color_val           # assign pixels of map to data values
        m[m==0.] = np.nan
        # Make contours of outflow gas
        hist_out, xedges, yedges = np.histogram2d(theta_out, phi_out, bins=[100,50], range=[[0., 2.*np.pi], [0., np.pi]])
        ax = contour_fig.add_subplot(1,1,1)
        contours = ax.contour(hist_out.transpose(),extent=[xedges[0],xedges[-1],yedges[0],yedges[-1]],levels = [5,10])
        plt.close(2)
        segs = contours.allsegs
        if (len(segs)>0):
            contour1_segs = contours.allsegs[0]
        else:
            contour1_segs = []
        if (len(segs)>1):
            contour2_segs = contours.allsegs[1]
        else:
            contour2_segs = []
        if (surface[0]=='sphere') and (args.radial_stepping>0):
            title = '$r=%.2f$ kpc' % (radius)
        else:
            title = ''
        healpy.mollview(m, fig=1, cmap=cmap, min=cmin, max=cmax, title=title, unit=field_label, badcolor='white')
        healpy.graticule()
        for i in range(len(contour1_segs)):
            healpy.projplot(contour1_segs[i][:,1], contour1_segs[i][:,0], color='#d4d4d4', ls='-', lw=2)
        for i in range(len(contour2_segs)):
            healpy.projplot(contour2_segs[i][:,1], contour2_segs[i][:,0], color='#969595', ls='-', lw=2)
        plt.text(0., 0.9, '%.2f Gyr\n$z=%.2f$' % (tsnap, zsnap), fontsize=20, ha='left', va='center', transform=ax.transAxes, bbox={'fc':'white','ec':'black','boxstyle':'round','lw':2})
        plt.savefig(prefix + 'Plots/' + snap + '_accretion-direction_' + color_field + '-colored' + save_r + save_dv + save_suffix + '.png')
        plt.close()

    fig1 = plt.figure(num=1, figsize=(10,6), dpi=300)
    contour_fig = plt.figure(num=2)
    pixel_acc = healpy.ang2pix(nside, phi_acc, theta_acc)
    u, dup_ind = np.unique(pixel_acc, return_index=True)
    for d in dup_ind:
        val_dup = mass[np.where(pixel_acc==pixel_acc[d])[0]]
        val_to_set = np.sum(val_dup)
        mass[np.where(pixel_acc==pixel_acc[d])[0]] = val_to_set
    m = np.zeros(healpy.nside2npix(nside))              # make empty array of map pixels
    m[pixel_acc] = mass           # assign pixels of map to data values
    m[m==0.] = np.nan
    # Make contours of outflow gas
    hist_out, xedges, yedges = np.histogram2d(theta_out, phi_out, bins=[100,50], range=[[0., 2.*np.pi], [0., np.pi]])
    ax = contour_fig.add_subplot(1,1,1)
    contours = ax.contour(hist_out.transpose(),extent=[xedges[0],xedges[-1],yedges[0],yedges[-1]],levels = [5,10])
    plt.close(2)
    segs = contours.allsegs
    if (len(segs)>0):
        contour1_segs = contours.allsegs[0]
    else:
        contour1_segs = []
    if (len(segs)>1):
        contour2_segs = contours.allsegs[1]
    else:
        contour2_segs = []
    if (surface[0]=='sphere') and (args.radial_stepping>0):
        title = '$r=%.2f$ kpc' % (radius)
    else:
        title = ''
    healpy.mollview(m, fig=1, cmap='winter_r', title=title, unit='Mass', badcolor='white', format='')
    healpy.graticule()
    for i in range(len(contour1_segs)):
        healpy.projplot(contour1_segs[i][:,1], contour1_segs[i][:,0], color='#d4d4d4', ls='-', lw=2)
    for i in range(len(contour2_segs)):
        healpy.projplot(contour2_segs[i][:,1], contour2_segs[i][:,0], color='#969595', ls='-', lw=2)
    plt.text(0., 0.9, '%.2f Gyr\n$z=%.2f$' % (tsnap, zsnap), fontsize=20, ha='left', va='center', transform=ax.transAxes, bbox={'fc':'white','ec':'black','boxstyle':'round','lw':2})
    plt.savefig(prefix + 'Plots/' + snap + '_accretion-direction_mass-colored' + save_r + save_dv + save_suffix + '.png')
    plt.close()

    if (args.dark_matter):
        fig1 = plt.figure(num=1, figsize=(10,6), dpi=300)
        contour_fig = plt.figure(num=2)
        pixel_acc = healpy.ang2pix(nside, phi_acc_dm, theta_acc_dm)
        u, dup_ind = np.unique(pixel_acc, return_index=True)
        for d in dup_ind:
            val_dup = mass_dm[np.where(pixel_acc==pixel_acc[d])[0]]
            val_to_set = np.sum(val_dup)
            mass_dm[np.where(pixel_acc==pixel_acc[d])[0]] = val_to_set
        m = np.zeros(healpy.nside2npix(nside))              # make empty array of map pixels
        m[pixel_acc] = mass_dm           # assign pixels of map to data values
        m[m==0.] = np.nan
        # Make contours of outflow gas
        hist_out, xedges, yedges = np.histogram2d(theta_out, phi_out, bins=[100,50], range=[[0., 2.*np.pi], [0., np.pi]])
        ax = contour_fig.add_subplot(1,1,1)
        contours = ax.contour(hist_out.transpose(),extent=[xedges[0],xedges[-1],yedges[0],yedges[-1]],levels = [5,10])
        plt.close(2)
        segs = contours.allsegs
        if (len(segs)>0):
            contour1_segs = contours.allsegs[0]
        else:
            contour1_segs = []
        if (len(segs)>1):
            contour2_segs = contours.allsegs[1]
        else:
            contour2_segs = []
        if (surface[0]=='sphere') and (args.radial_stepping>0):
            title = '$r=%.2f$ kpc' % (radius)
        else:
            title = ''
        healpy.mollview(m, fig=1, cmap='winter_r', title=title, unit='Dark matter mass', badcolor='white', format='')
        healpy.graticule()
        for i in range(len(contour1_segs)):
            healpy.projplot(contour1_segs[i][:,1], contour1_segs[i][:,0], color='#d4d4d4', ls='-', lw=2)
        for i in range(len(contour2_segs)):
            healpy.projplot(contour2_segs[i][:,1], contour2_segs[i][:,0], color='#969595', ls='-', lw=2)
        plt.text(0., 0.9, '%.2f Gyr\n$z=%.2f$' % (tsnap, zsnap), fontsize=20, ha='left', va='center', transform=ax.transAxes, bbox={'fc':'white','ec':'black','boxstyle':'round','lw':2})
        plt.savefig(prefix + 'Plots/' + snap + '_accretion-direction_dm-mass-colored' + save_r + save_dv + save_suffix + '.png')
        plt.close()

    fig1 = plt.figure(num=1, figsize=(10,6), dpi=300)
    contour_fig = plt.figure(num=2)
    pixel_acc = healpy.ang2pix(nside, phi_acc, theta_acc)
    u, dup_ind = np.unique(pixel_acc, return_index=True)
    for d in dup_ind:
        val_dup = metal_mass[np.where(pixel_acc==pixel_acc[d])[0]]
        val_to_set = np.sum(val_dup)
        metal_mass[np.where(pixel_acc==pixel_acc[d])[0]] = val_to_set
    m = np.zeros(healpy.nside2npix(nside))              # make empty array of map pixels
    m[pixel_acc] = metal_mass           # assign pixels of map to data values
    m[m==0.] = np.nan
    # Make contours of outflow gas
    hist_out, xedges, yedges = np.histogram2d(theta_out, phi_out, bins=[100,50], range=[[0., 2.*np.pi], [0., np.pi]])
    ax = contour_fig.add_subplot(1,1,1)
    contours = ax.contour(hist_out.transpose(),extent=[xedges[0],xedges[-1],yedges[0],yedges[-1]],levels = [5,10])
    plt.close(2)
    segs = contours.allsegs
    if (len(segs)>0):
        contour1_segs = contours.allsegs[0]
    else:
        contour1_segs = []
    if (len(segs)>1):
        contour2_segs = contours.allsegs[1]
    else:
        contour2_segs = []
    if (surface[0]=='sphere') and (args.radial_stepping>0):
        title = '$r=%.2f$ kpc' % (radius)
    else:
        title = ''
    healpy.mollview(m, fig=1, cmap='winter_r', title=title, unit='Metal mass', badcolor='white', format='')
    healpy.graticule()
    for i in range(len(contour1_segs)):
        healpy.projplot(contour1_segs[i][:,1], contour1_segs[i][:,0], color='#d4d4d4', ls='-', lw=2)
    for i in range(len(contour2_segs)):
        healpy.projplot(contour2_segs[i][:,1], contour2_segs[i][:,0], color='#969595', ls='-', lw=2)
    plt.text(0., 0.9, '%.2f Gyr\n$z=%.2f$' % (tsnap, zsnap), fontsize=20, ha='left', va='center', transform=ax.transAxes, bbox={'fc':'white','ec':'black','boxstyle':'round','lw':2})
    plt.savefig(prefix + 'Plots/' + snap + '_accretion-direction_metal-mass-colored' + save_r + save_dv + save_suffix + '.png')
    plt.close()

def sky_map(ds, sp, snap, snap_props):
    '''Makes a sky map of column densities as viewed from the center of the galaxy for all gas and
    only accreting gas.'''

    prefix = output_dir + 'projections_halo_00' + args.halo + '/' + args.run + '/'
    Menc_profile, Mvir, Rvir = snap_props
    tsnap = ds.current_time.in_units('Gyr').v
    zsnap = ds.get_parameter('CosmologyCurrentRedshift')

    abundances = trident.ion_balance.solar_abundance
    trident.add_ion_fields(ds, ions='all', ftype='gas')

    # Load grid properties
    radius = sp['gas','radius_corrected'].in_units('cm').v
    theta = sp['gas','theta_pos_disk'].v*(180./np.pi)
    phi = sp['gas','phi_pos_disk'].v*(180./np.pi)
    mass = sp['gas','cell_mass'].in_units('g').v
    rv = sp['gas','radial_velocity_corrected'].in_units('km/s').v
    vff = sp['gas','vff'].in_units('km/s').v
    density = sp['gas','density'].in_units('g/cm**3').v
    dx = sp['gas','dx'].in_units('cm').v
    NHI = sp['gas','H_p0_number_density'].in_units('cm**-3').v

    for i in ['all','acc']:
        if (i=='acc'):
            accreting = (rv < 1.2*vff)
        else:
            accreting = np.ones(np.shape(theta), dtype=bool)

        radius_plot = radius[accreting]
        theta_plot = theta[accreting]
        phi_plot = phi[accreting]
        density_plot = density[accreting]
        dx_plot = dx[accreting]
        NHI_plot = NHI[accreting]

        fig1 = plt.figure(num=1, figsize=(10,6), dpi=300)
        nside = 2**6
        npix = healpy.nside2npix(nside)
        pix_area = healpy.nside2pixarea(nside)
        pixels = healpy.ang2pix(nside, phi_plot*(np.pi/180.), theta_plot*(np.pi/180.)+np.pi)
        uniq, dup_ind = np.unique(pixels, return_index=True)
        #col_den = density_plot*dx_plot/mp
        col_den = NHI_plot*dx_plot
        m = np.zeros(healpy.nside2npix(nside))              # make empty array of map pixels
        npix = np.zeros(healpy.nside2npix(nside))
        for u in uniq:
            pix_col_den = np.sum(col_den[(pixels==u)])
            m[u] = np.log10(pix_col_den)
            npix[u] = len(col_den[(pixels==u)])

        # If a pixel is an outlier from its neighbors, set it to the median value of the neighbors
        for pix in range(len(m)):
            neighbors = healpy.get_all_neighbours(nside, pix)
            std = np.std(10**m[neighbors])
            med = np.median(10**m[neighbors])
            if (10**m[pix]>(med+3.*std)) or (10**m[pix]<(med-3.*std)):
                m[pix] = np.log10(med)

        # Smooth over neighbors
        for pix in range(len(m)):
            neighbors = healpy.get_all_neighbours(nside, pix)
            m[pix] = np.log10(np.mean(10**m[neighbors]))

        #cmap = density_color_map
        cmap = sns.blend_palette(("black","#4575b4", "#984ea3", "#d73027","darkorange", "#ffe34d"), as_cmap=True)
        cmin = 12
        cmax = 22
        if (i=='all'):
            title = 'All gas'
        else:
            title = 'Only accreting gas'
        healpy.mollview(m, fig=1, cmap=cmap, min=cmin, max=cmax, title=title, unit='log H I Column Density [cm$^{-2}$]')
        healpy.graticule()
        plt.savefig(prefix + snap + '_col-den-map_' + i + save_suffix + '.png')
        plt.close()

def calculate_flux(ds, grid, shape, snap, snap_props):
    '''Calculates the flux into and out of the specified shape at the snapshot 'snap' and saves to file.'''

    tablename = prefix + 'Tables/' + snap + '_fluxes'
    Menc_profile, Mvir, Rvir = snap_props
    tsnap = ds.current_time.in_units('Gyr').v
    zsnap = ds.get_parameter('CosmologyCurrentRedshift')

    # Set up table of everything we want
    fluxes = []
    flux_filename = ''
    if ('mass' in flux_types):
        fluxes.append('mass_flux')
        fluxes.append('metal_flux')
        if (args.dark_matter): fluxes.append('dm_mass_flux')
        flux_filename += '_mass'
    if ('energy' in flux_types):
        fluxes.append('thermal_energy_flux')
        fluxes.append('kinetic_energy_flux')
        fluxes.append('potential_energy_flux')
        fluxes.append('bernoulli_energy_flux')
        fluxes.append('cooling_energy_flux')
        flux_filename += '_energy'
    table = make_flux_table(fluxes)

    if (args.cgm_only):
        # Define the density cut between disk and CGM to vary smoothly between 1 and 0.1 between z = 0.5 and z = 0.25,
        # with it being 1 at higher redshifts and 0.1 at lower redshifts
        current_time = ds.current_time.in_units('Myr').v
        if (current_time<=7091.48):
            density_cut_factor = 20. - 19.*current_time/7091.48
        elif (current_time<=8656.88):
            density_cut_factor = 1.
        elif (current_time<=10787.12):
            density_cut_factor = 1. - 0.9*(current_time-8656.88)/2130.24
        else:
            density_cut_factor = 0.1
        cgm_bool = (grid['gas','density'].in_units('g/cm**3').v < density_cut_factor * cgm_density_max)
    else:
        cgm_bool = (grid['gas','density'].in_units('g/cm**3').v > 0.)
    shape = shape & cgm_bool

    # Load grid properties
    x = grid['gas', 'x'].in_units('kpc').v - ds.halo_center_kpc[0].v
    y = grid['gas', 'y'].in_units('kpc').v - ds.halo_center_kpc[1].v
    z = grid['gas', 'z'].in_units('kpc').v - ds.halo_center_kpc[2].v
    xbins = x[:,0,0][:-1] - 0.5*np.diff(x[:,0,0])
    ybins = y[0,:,0][:-1] - 0.5*np.diff(y[0,:,0])
    zbins = z[0,0,:][:-1] - 0.5*np.diff(z[0,0,:])
    vx = grid['gas','vx_corrected'].in_units('kpc/yr').v
    vy = grid['gas','vy_corrected'].in_units('kpc/yr').v
    vz = grid['gas','vz_corrected'].in_units('kpc/yr').v
    radius = grid['gas','radius_corrected'].in_units('kpc').v
    rv = grid['radial_velocity_corrected'].in_units('km/s').v
    if (args.direction) or ('disk' in surface[0]) or ('accretion_direction' in plots):
        theta = grid['gas','theta_pos_disk'].v*(180./np.pi)
        phi = grid['gas','phi_pos_disk'].v*(180./np.pi)
    if ('accretion_viz' in plots) or ('accretion_direction' in plots):
        temperature = grid['gas','temperature'].in_units('K').v
        density = grid['gas','density'].in_units('g/cm**3').v
        tcool = grid['gas','cooling_time'].in_units('Myr').v
        metallicity = grid['gas', 'metallicity'].in_units('Zsun').v
    # Load dark matter velocities and positions and digitize onto grid
    if (args.dark_matter):
        left_edge = grid.LeftEdge.in_units('kpc')
        right_edge = grid.RightEdge.in_units('kpc')
        box_dm = ds.box(left_edge, right_edge)
        x_dm = box_dm['dm','particle_position_x'].in_units('kpc').v - ds.halo_center_kpc[0].v
        y_dm = box_dm['dm','particle_position_y'].in_units('kpc').v - ds.halo_center_kpc[1].v
        z_dm = box_dm['dm','particle_position_z'].in_units('kpc').v - ds.halo_center_kpc[2].v
        vx_dm = box_dm['dm','particle_velocity_x'].in_units('kpc/yr').v - ds.halo_velocity_kms[0].in_units('kpc/yr').v
        vy_dm = box_dm['dm','particle_velocity_y'].in_units('kpc/yr').v - ds.halo_velocity_kms[1].in_units('kpc/yr').v
        vz_dm = box_dm['dm','particle_velocity_z'].in_units('kpc/yr').v - ds.halo_velocity_kms[2].in_units('kpc/yr').v
        inds_x = np.digitize(x_dm, xbins)-1      # indices of x positions
        inds_y = np.digitize(y_dm, ybins)-1      # indices of y positions
        inds_z = np.digitize(z_dm, zbins)-1      # indices of z positions
        inds_dm = np.array([inds_x, inds_y, inds_z])
        in_shape_dm = shape[tuple(inds_dm)]
    properties = []
    if ('mass' in flux_types):
        mass = grid['gas', 'cell_mass'].in_units('Msun').v
        metals = grid['gas', 'metal_mass'].in_units('Msun').v
        properties.append(mass)
        properties.append(metals)
        if (args.dark_matter):
            mass_dm = box_dm[('dm','particle_mass')].in_units('Msun').v
            properties.append(mass_dm)
    if ('energy' in flux_types):
        kinetic_energy = grid['gas','kinetic_energy_corrected'].in_units('erg').v
        thermal_energy = (grid['gas','cell_mass']*grid['gas','thermal_energy']).in_units('erg').v
        potential_energy = -G * Menc_profile(radius)*gtoMsun / (radius*1000.*cmtopc)*grid['gas','cell_mass'].in_units('g').v
        bernoulli_energy = kinetic_energy + 5./3.*thermal_energy + potential_energy
        cooling_energy = thermal_energy/grid['gas','cooling_time'].in_units('yr').v
        properties.append(thermal_energy)
        properties.append(kinetic_energy)
        properties.append(potential_energy)
        properties.append(bernoulli_energy)
        properties.append(cooling_energy)

    # Calculate new positions of gas cells for a long elapsed time (necessary because digitizing onto grid can "reset" positions of slow-moving gas)
    new_x = vx*(5.*dt) + x
    new_y = vy*(5.*dt) + y
    new_z = vz*(5.*dt) + z
    displacement = np.sqrt((new_x-x)**2. + (new_y-y)**2. + (new_z-z)**2.)
    displacement_vel = displacement*1000.*cmtopc/(5.*dt*stoyr)/1e5
    inds_x = np.digitize(new_x, xbins)-1      # indices of new x positions
    inds_y = np.digitize(new_y, ybins)-1      # indices of new y positions
    inds_z = np.digitize(new_z, zbins)-1      # indices of new z positions
    new_inds = np.array([inds_x, inds_y, inds_z])


    # Calculate new positions of dark matter
    if (args.dark_matter):
        new_x_dm = vx_dm*(5.*dt) + x_dm
        new_y_dm = vy_dm*(5.*dt) + y_dm
        new_z_dm = vz_dm*(5.*dt) + z_dm
        displacement_vel_dm = np.sqrt((new_x_dm-x_dm)**2. + (new_y_dm-y_dm)**2. + (new_z_dm-z_dm)**2.)/(5.*dt)*(cmtopc*1000./stoyr/1e5)
        inds_x = np.digitize(new_x_dm, xbins)-1      # indices of new x positions
        inds_y = np.digitize(new_y_dm, ybins)-1      # indices of new y positions
        inds_z = np.digitize(new_z_dm, zbins)-1      # indices of new z positions
        new_inds_dm = np.array([inds_x, inds_y, inds_z])

    # If calculating direction of accretion, set up theta and phi and bins
    if (args.direction):
        phi_bins = ['all','major','minor']
        if (args.dark_matter):
            theta_dm = box_dm['dm','theta_pos_disk'].v*(180./np.pi)
            phi_dm = box_dm['dm','phi_pos_disk'].v*(180./np.pi)
    else:
        phi_bins = ['all']

    # If stepping through radius, set up radii list
    if (surface[0]=='sphere') and (args.radial_stepping>0):
        if (args.Rvir):
            max_R = surface[1]*Rvir
        else:
            max_R = surface[1]
        radii = np.linspace(0., max_R, args.radial_stepping+1)[1:]
    else:
        radii = [0]

    # Set up filtering
    if (args.region_filter!='none'):
        if (args.region_filter=='temperature'):
            regions = [0., 10**4., 10**5., 10**6., np.inf]
            filter = grid['gas','temperature'].in_units('K').v
        elif (args.region_filter=='metallicity'):
            regions = [0., 0.1, 0.5, 1., np.inf]
            filter = grid['gas','metallicity'].in_units('Zsun').v
        elif (args.region_filter=='velocity'):
            #regions = [-np.inf, -100., 0., 100., np.inf]
            regions = [0., -20., -50., -100., -200.]
            filter = grid['gas','radial_velocity_corrected'].in_units('km/s').v
    disp_vel_bins = [0., 50., 100., 150., 200., 300., np.inf]
    disp_vel_saves = ['_0-50','_50-100','_100-150','_150-200','_200-300','_300-inf']

    # Step through radii (if chosen) and calculate fluxes and plot things for each radius
    for r in range(len(radii)):
        # If stepping through radii, define the shape for this radius value
        if (surface[0]=='sphere') and (args.radial_stepping>0):
            shape = (radius < radii[r])
            save_r = '_r%d' % (r)
        else:
            save_r = ''
        # Define which cells are entering and leaving shape
        new_in_shape = shape[tuple(new_inds)]
        from_shape = shape & ~new_in_shape
        to_shape = ~shape & new_in_shape
        from_shape_fast = from_shape & (rv > 200.)

        # Bin cells by displacement velocity
        to_shape_dv = []
        for dv in range(len(disp_vel_bins)-1):
            low_dv = disp_vel_bins[dv]
            upp_dv = disp_vel_bins[dv+1]
            to_shape_dv.append(to_shape & (displacement_vel >= low_dv) & (displacement_vel < upp_dv))

        # Define which cells are entering and leaving shape for dark matter
        if (args.dark_matter):
            new_in_shape_dm = shape[tuple(new_inds_dm)]
            from_shape_dm = in_shape_dm & ~new_in_shape_dm
            to_shape_dm = ~in_shape_dm & new_in_shape_dm
            to_shape_dv_dm = []
            for dv in range(len(disp_vel_bins)-1):
                low_dv = disp_vel_bins[dv]
                upp_dv = disp_vel_bins[dv+1]
                to_shape_dv_dm.append(to_shape_dm & (displacement_vel_dm >= low_dv) & (displacement_vel_dm < upp_dv))

        if ('accretion_viz' in plots):
            # Set all values outside of the shapes of interest to zero
            temp_shape = np.copy(temperature)
            temp_shape[~shape] = 0.
            temp_acc = np.copy(temperature)
            temp_acc[~to_shape] = 0.
            # Load these back into yt so we can make projections
            data = dict(temperature = (temperature, "K"), temperature_shape = (temp_shape, 'K'), \
                        temperature_accreting = (temp_acc, 'K'), density = (density, 'g/cm**3'))
            bbox = np.array([[np.min(x), np.max(x)], [np.min(y), np.max(y)], [np.min(z), np.max(z)]])
            ds_viz = yt.load_uniform_grid(data, temperature.shape, length_unit="kpc", bbox=bbox)
            ad = ds_viz.all_data()
            # Make cut regions to remove the "null values" from before
            shape_region = ad.cut_region("obj['temperature_shape'] > 0")
            accreting_region = ad.cut_region("obj['temperature_accreting'] > 0")
            # Make projection plots
            proj = yt.ProjectionPlot(ds_viz, 'x', 'temperature_shape', data_source=shape_region, weight_field='density', fontsize=28)
            proj.set_log('temperature_shape', True)
            proj.set_cmap('temperature_shape', sns.blend_palette(('salmon', "#984ea3", "#4daf4a", "#ffe34d", 'darkorange'), as_cmap=True))
            proj.set_zlim('temperature_shape', 1e4,1e7)
            proj.set_colorbar_label('temperature_shape', 'Temperature [K]')
            proj.annotate_text((0.03, 0.885), '%.2f Gyr\n$z=%.2f$' % (tsnap, zsnap), coord_system="axis", text_args={'color':'black'}, \
              inset_box_args={"boxstyle":"round,pad=0.3","facecolor":"white","linewidth":2,"edgecolor":"black"})
            proj.save(prefix + 'Plots/' + snap + '_temperature-shape_x' + save_suffix + '.png')
            proj = yt.ProjectionPlot(ds_viz, 'x', 'temperature_accreting', data_source=accreting_region, weight_field='density', fontsize=28)
            proj.set_log('temperature_accreting', True)
            proj.set_cmap('temperature_accreting', sns.blend_palette(('salmon', "#984ea3", "#4daf4a", "#ffe34d", 'darkorange'), as_cmap=True))
            proj.set_zlim('temperature_accreting', 1e4,1e7)
            proj.set_colorbar_label('temperature_accreting', 'Temperature [K]')
            proj.annotate_text((0.03, 0.885), '%.2f Gyr\n$z=%.2f$' % (tsnap, zsnap), coord_system="axis", text_args={'color':'black'}, \
              inset_box_args={"boxstyle":"round,pad=0.3","facecolor":"white","linewidth":2,"edgecolor":"black"})
            proj.save(prefix + 'Plots/' + snap + '_temperature-accreting_x' + save_suffix + '.png')

        if (args.direction):
            theta_to_dv = []
            phi_to_dv = []
            if (args.dark_matter):
                theta_to_dv_dm = []
                phi_to_dv_dm = []
            for dv in range(len(disp_vel_bins)-1):
                theta_to_dv.append(theta[to_shape_dv[dv]])
                phi_to_dv.append(phi[to_shape_dv[dv]])
                if (args.dark_matter):
                    theta_to_dv_dm.append(theta_dm[to_shape_dv_dm[dv]])
                    phi_to_dv_dm.append(phi_dm[to_shape_dv_dm[dv]])
            if (args.dark_matter):
                theta_to_dm = theta_dm[to_shape_dm]
                phi_to_dm = phi_dm[to_shape_dm]
                phi_from_dm = phi_dm[from_shape_dm]
            theta_to = theta[to_shape]
            phi_to = phi[to_shape]
            phi_from = phi[from_shape]
            theta_out = theta[from_shape_fast]
            phi_out = phi[from_shape_fast]

        for p in range(len(phi_bins)):
            if (surface[0]=='sphere') and (args.radial_stepping>0):
                results = [radii[r]]
            else: results = []
            if (args.direction):
                results.append(phi_bins[p])
            if (phi_bins[p]=='all'):
                angle_bin_from = np.ones(np.count_nonzero(from_shape), dtype=bool)
                if (args.dark_matter):
                    angle_bin_from_dm = np.ones(np.count_nonzero(from_shape_dm), dtype=bool)
            elif (phi_bins[p]=='major'):
                angle_bin_from = (phi_from >= 60.) & (phi_from <= 120.)
                if (args.dark_matter):
                    angle_bin_from_dm = (phi_from_dm >= 60.) & (phi_from_dm <= 120.)
            elif (phi_bins[p]=='minor'):
                angle_bin_from = (phi_from < 60.) | (phi_from > 120.)
                if (args.dark_matter):
                    angle_bin_from_dm = (phi_from_dm < 60.) | (phi_from_dm > 120.)
            angle_bin_to_dv = []
            angle_bin_to_dv_dm = []
            if (phi_bins[p]=='all'):
                angle_bin_to = np.ones(len(phi_to), dtype=bool)
                if (args.dark_matter):
                    angle_bin_to_dm = np.ones(len(phi_to_dm), dtype=bool)
                for dv in range(len(disp_vel_bins)-1):
                    angle_bin_to_dv.append(np.ones(np.count_nonzero(to_shape_dv[dv]), dtype=bool))
                    if (args.dark_matter):
                        angle_bin_to_dv_dm.append(np.ones(np.count_nonzero(to_shape_dv_dm[dv]), dtype=bool))
            elif (phi_bins[p]=='major'):
                angle_bin_to = (phi_to >= 60.) & (phi_to <= 120.)
                if (args.dark_matter):
                    angle_bin_to_dm = (phi_to_dm >=60.) & (phi_to_dm <= 120.)
                for dv in range(len(disp_vel_bins)-1):
                    angle_bin_to_dv.append((phi_to_dv[dv] >= 60.) & (phi_to_dv[dv] <= 120.))
                    if (args.dark_matter):
                        angle_bin_to_dm.append((phi_to_dm[dv] >= 60.) & (phi_to_dm[dv] <= 120.))
            elif (phi_bins[p]=='minor'):
                angle_bin_to = (phi_to < 60.) | (phi_to > 120.)
                if (args.dark_matter):
                    angle_bin_to_dm = (phi_to_dm < 60.) | (phi_to_dm > 120.)
                for dv in range(len(disp_vel_bins)-1):
                    angle_bin_to_dv.append((phi_to_dv[dv] < 60.) | (phi_to_dv[dv] > 120.))
                    if (args.dark_matter):
                        angle_bin_to_dm.append((phi_to_dm[dv] < 60.) | (phi_to_dm[dv] > 120.))

            for i in range(len(fluxes)):
                if ('dm' in fluxes[i]):
                    prop_to_dm = properties[i][to_shape_dm[dv]][angle_bin_to_dm]
                    flux_in = np.sum(prop_to_dm)/(5.*dt)
                    results.append(flux_in)
                    for dv in range(len(disp_vel_bins)-1):
                        prop_to_dm_dv = properties[i][to_shape_dv_dm[dv]][angle_bin_to_dv_dm[dv]]
                        flux_in = np.sum(prop_to_dm_dv)/(5.*dt)
                        results.append(flux_in)
                    prop_from_dm = properties[i][from_shape_dm][angle_bin_from_dm]
                    flux_out = np.sum(prop_from_dm)/(5.*dt)
                    results.append(flux_out)
                else:
                    prop_to = properties[i][to_shape][angle_bin_to]
                    flux_in = np.sum(prop_to)/(5.*dt)
                    results.append(flux_in)
                    if (args.region_filter!='none'):
                        region_to = filter[to_shape][angle_bin_to]
                        for j in range(len(regions)-1):
                            prop_to_region = prop_to[(region_to > regions[j]) & (region_to < regions[j+1])]
                            flux_in = np.sum(prop_to_region)/(5.*dt)
                            results.append(flux_in)
                    for dv in range(len(disp_vel_bins)-1):
                        prop_to_dv = properties[i][to_shape_dv[dv]][angle_bin_to_dv[dv]]
                        flux_in = np.sum(prop_to_dv)/(5.*dt)
                        results.append(flux_in)
                        if (args.region_filter!='none'):
                            region_to = filter[to_shape_dv[dv]][angle_bin_to_dv[dv]]
                            for j in range(len(regions)-1):
                                prop_to_region = prop_to_dv[(region_to > regions[j]) & (region_to < regions[j+1])]
                                flux_in = np.sum(prop_to_region)/(5.*dt)
                                results.append(flux_in)
                    prop_from = properties[i][from_shape][angle_bin_from]
                    flux_out = np.sum(prop_from)/(5.*dt)
                    results.append(flux_out)
                    if (args.region_filter!='none'):
                        region_from = filter[from_shape][angle_bin_from]
                        for j in range(len(regions)-1):
                            prop_from_region = prop_from[(region_from > regions[j]) & (region_from < regions[j+1])]
                            flux_out = np.sum(prop_from_region)/(5.*dt)
                            results.append(flux_out)
            table.add_row(results)

        if ('accretion_direction' in plots):
            if (args.dark_matter):
                plot_accretion_direction(theta_to*(np.pi/180.)+np.pi, phi_to*(np.pi/180.), temperature[to_shape], metallicity[to_shape], rv[to_shape], tcool[to_shape], mass[to_shape], metals[to_shape], theta_out*(np.pi/180.)+np.pi, phi_out*(np.pi/180.), tsnap, zsnap, prefix, snap, radii[r], save_r, '', theta_acc_dm=theta_to_dm*(np.pi/180.)+np.pi, phi_acc_dm=phi_to_dm*(np.pi/180.), mass_dm=mass_dm[to_shape_dm])
                for dv in range(len(disp_vel_bins)-1):
                    save_dv = disp_vel_saves[dv]
                    plot_accretion_direction(theta_to*(np.pi/180.)+np.pi, phi_to*(np.pi/180.), temperature[to_shape], metallicity[to_shape], rv[to_shape], tcool[to_shape], mass[to_shape], metals[to_shape], theta_out*(np.pi/180.)+np.pi, phi_out*(np.pi/180.), tsnap, zsnap, prefix, snap, radii[r], save_r, save_dv, theta_acc_dm=theta_to_dm*(np.pi/180.)+np.pi, phi_acc_dm=phi_to_dm*(np.pi/180.), mass_dm=mass_dm[to_shape_dm])
            else:
                plot_accretion_direction(theta_to[dv]*(np.pi/180.)+np.pi, phi_to[dv]*(np.pi/180.), temperature[to_shape_dv[dv]], metallicity[to_shape_dv[dv]], rv[to_shape_dv[dv]], tcool[to_shape_dv[dv]], mass[to_shape_dv[dv]], metals[to_shape_dv[dv]], theta_out*(np.pi/180.)+np.pi, phi_out*(np.pi/180.), tsnap, zsnap, prefix, snap, radii[r], save_r, '')
                for dv in range(len(disp_vel_bins)-1):
                    save_dv = disp_vel_saves[dv]
                    plot_accretion_direction(theta_to[dv]*(np.pi/180.)+np.pi, phi_to[dv]*(np.pi/180.), temperature[to_shape_dv[dv]], metallicity[to_shape_dv[dv]], rv[to_shape_dv[dv]], tcool[to_shape_dv[dv]], mass[to_shape_dv[dv]], metals[to_shape_dv[dv]], theta_out*(np.pi/180.)+np.pi, phi_out*(np.pi/180.), tsnap, zsnap, prefix, snap, radii[r], save_r, save_dv)

    table = set_flux_table_units(table)
    table.write(tablename + flux_filename + save_suffix + '.hdf5', path='all_data', serialize_meta=True, overwrite=True)

def calculate_flux_spherical(ds, snap, snap_props):
    '''Calculates the flux into and out of the specified shape at the snapshot 'snap' and saves to file.'''

    tablename = prefix + 'Tables/' + snap + '_fluxes'
    Menc_profile, Mvir, Rvir = snap_props
    tsnap = ds.current_time.in_units('Gyr').v
    zsnap = ds.get_parameter('CosmologyCurrentRedshift')

    # Set up table of everything we want
    fluxes = []
    flux_filename = ''
    if ('mass' in flux_types):
        fluxes.append('mass_flux')
        fluxes.append('metal_flux')
        if (args.dark_matter): fluxes.append('dm_mass_flux')
        flux_filename += '_mass'
    if ('energy' in flux_types):
        fluxes.append('thermal_energy_flux')
        fluxes.append('kinetic_energy_flux')
        fluxes.append('potential_energy_flux')
        fluxes.append('bernoulli_energy_flux')
        fluxes.append('cooling_energy_flux')
        flux_filename += '_energy'
    table = make_flux_table(fluxes)

    sph = ds.sphere(center=ds.halo_center_kpc, radius=(2.*Rvir, 'kpc'))

    if (args.cgm_only):
        # Define the density cut between disk and CGM to vary smoothly between 1 and 0.1 between z = 0.5 and z = 0.25,
        # with it being 1 at higher redshifts and 0.1 at lower redshifts
        current_time = ds.current_time.in_units('Myr').v
        if (current_time<=7091.48):
            density_cut_factor = 20. - 19.*current_time/7091.48
        elif (current_time<=8656.88):
            density_cut_factor = 1.
        elif (current_time<=10787.12):
            density_cut_factor = 1. - 0.9*(current_time-8656.88)/2130.24
        else:
            density_cut_factor = 0.1
        sph_cgm = sph.cut_region("obj['density'] < %.3e" % (density_cut_factor * cgm_density_max))
    else:
        sph_cgm = sph

    # Load grid properties
    radius = sph_cgm['gas','radius_corrected'].in_units('kpc').v
    rv = sph_cgm['radial_velocity_corrected'].in_units('km/s').v
    if (args.direction) or ('disk' in surface[0]) or ('accretion_direction' in plots):
        theta = sph_cgm['gas','theta_pos_disk'].v*(180./np.pi)
        phi = sph_cgm['gas','phi_pos_disk'].v*(180./np.pi)
    if ('accretion_viz' in plots) or ('accretion_direction' in plots):
        temperature = sph_cgm['gas','temperature'].in_units('K').v
        density = sph_cgm['gas','density'].in_units('g/cm**3').v
        tcool = sph_cgm['gas','cooling_time'].in_units('Myr').v
        metallicity = sph_cgm['gas', 'metallicity'].in_units('Zsun').v
    # Load dark matter velocities and positions and digitize onto grid
    if (args.dark_matter):
        radius_dm = sph_cgm['dm','radius_corrected'].in_units('kpc').v
        rv_dm = sph_cgm['dm','radial_velocity_corrected'].in_units('km/s').v
    properties = []
    if ('mass' in flux_types):
        mass = sph_cgm['gas', 'cell_mass'].in_units('Msun').v
        metals = sph_cgm['gas', 'metal_mass'].in_units('Msun').v
        properties.append(mass)
        properties.append(metals)
        if (args.dark_matter):
            mass_dm = sph_cgm[('dm','particle_mass')].in_units('Msun').v
            properties.append(mass_dm)
    if ('energy' in flux_types):
        kinetic_energy = sph_cgm['gas','kinetic_energy_corrected'].in_units('erg').v
        thermal_energy = (sph_cgm['gas','cell_mass']*sph_cgm['gas','thermal_energy']).in_units('erg').v
        potential_energy = -G * Menc_profile(radius)*gtoMsun / (radius*1000.*cmtopc)*sph_cgm['gas','cell_mass'].in_units('g').v
        bernoulli_energy = kinetic_energy + 5./3.*thermal_energy + potential_energy
        cooling_energy = thermal_energy/sph_cgm['gas','cooling_time'].in_units('yr').v
        properties.append(thermal_energy)
        properties.append(kinetic_energy)
        properties.append(potential_energy)
        properties.append(bernoulli_energy)
        properties.append(cooling_energy)

    # Calculate new radius of gas cells
    new_radius = radius + ((rv*1e5)*(5.*dt*stoyr))/(1000.*cmtopc)

    # Calculate new radius of dark matter
    if (args.dark_matter):
        new_radius_dm = radius_dm + ((rv_dm*1e5)*(5.*dt*stoyr))/(1000.*cmtopc)

    # If calculating direction of accretion, set up theta and phi and bins
    if (args.direction):
        phi_bins = ['all','major','minor']
        if (args.dark_matter):
            theta_dm = sph_cgm['dm','theta_pos_disk'].v*(180./np.pi)
            phi_dm = sph_cgm['dm','phi_pos_disk'].v*(180./np.pi)
    else:
        phi_bins = ['all']

    # Set up radii list
    max_R = 1.5*Rvir
    R_step = float(np.min(sph_cgm[('gas','dx')].in_units('kpc')))*(2.**11.)/(2.**9.)  # at level 11
    radii = np.arange(0., max_R, R_step)[1:]

    # Set up filtering
    if (args.region_filter!='none'):
        if (args.region_filter=='temperature'):
            regions = [0., 10**4., 10**5., 10**6., np.inf]
            filter = sph_cgm['gas','temperature'].in_units('K').v
        elif (args.region_filter=='metallicity'):
            regions = [0., 0.1, 0.5, 1., np.inf]
            filter = sph_cgm['gas','metallicity'].in_units('Zsun').v
        elif (args.region_filter=='velocity'):
            regions = [-np.inf, -100., 0., 100., np.inf]
            filter = rv

    # Step through radii and calculate fluxes and plot things for each radius
    for r in range(len(radii)-1):
        save_r = '_r%d' % (r)
        # Define which cells are entering and leaving this radius
        enter_r = (radius > radii[r]) & (new_radius < radii[r])
        leave_r = (radius < radii[r]) & (new_radius > radii[r])
        # Define which cells maintain their position just outside this radius
        stay_r = ((radius > radii[r]) & (radius < radii[r+1])) & (new_radius > radii[r])

        # Define which cells are entering and leaving shape for dark matter
        if (args.dark_matter):
            enter_r_dm = (radius_dm > radii[r]) & (new_radius_dm < radii[r])
            leave_r_dm = (radius_dm < radii[r]) & (new_radius_dm > radii[r])
            stay_r_dm = ((radius_dm > radii[r]) & (radius_dm < radii[r+1])) & (new_radius_dm > radii[r])

        if (args.direction):
            if (args.dark_matter):
                theta_enter_dm = theta_dm[enter_r_dm]
                phi_enter_dm = phi_dm[enter_r_dm]
                theta_leave_dm = theta_dm[leave_r_dm]
                phi_leave_dm = phi_dm[leave_r_dm]
            theta_enter = theta[enter_r]
            phi_enter = phi[enter_r]
            theta_leave = theta[leave_r]
            phi_leave = phi[leave_r]

        for p in range(len(phi_bins)):
            results = [radii[r]]
            if (args.direction):
                results.append(phi_bins[p])
            if (phi_bins[p]=='all'):
                angle_bin_enter = np.ones(np.count_nonzero(enter_r), dtype=bool)
                angle_bin_leave = np.ones(np.count_nonzero(leave_r), dtype=bool)
                if (args.dark_matter):
                    angle_bin_enter_dm = np.ones(np.count_nonzero(enter_r_dm), dtype=bool)
                    angle_bin_leave_dm = np.ones(np.count_nonzero(leave_r_dm), dtype=bool)
            elif (phi_bins[p]=='major'):
                angle_bin_enter = (phi_enter >= 60.) & (phi_enter <= 120.)
                angle_bin_leave = (phi_leave >= 60.) & (phi_leave <= 120.)
                if (args.dark_matter):
                    angle_bin_enter_dm = (phi_enter_dm >= 60.) & (phi_enter_dm <= 120.)
                    angle_bin_leave_dm = (phi_leave_dm >= 60.) & (phi_leave_dm <= 120.)
            elif (phi_bins[p]=='minor'):
                angle_bin_enter = (phi_enter < 60.) | (phi_enter > 120.)
                angle_bin_leave = (phi_leave < 60.) | (phi_leave > 120.)
                if (args.dark_matter):
                    angle_bin_enter_dm = (phi_enter_dm < 60.) | (phi_enter_dm > 120.)
                    angle_bin_leave_dm = (phi_leave_dm < 60.) | (phi_leave_dm > 120.)
            for i in range(len(fluxes)):
                if ('dm' in fluxes[i]):
                    prop_enter_dm = properties[i][enter_r_dm][angle_bin_enter_dm]
                    prop_leave_dm = properties[i][leave_r_dm][angle_bin_leave_dm]
                    flux_in = np.sum(prop_enter_dm)/(5.*dt)
                    flux_out = np.sum(prop_leave_dm)/(5.*dt)
                    results.append(flux_in)
                    results.append(flux_out)
                else:
                    prop_enter = properties[i][enter_r][angle_bin_enter]
                    prop_leave = properties[i][leave_r][angle_bin_leave]
                    flux_in = np.sum(prop_enter)/(5.*dt)
                    flux_out = np.sum(prop_leave)/(5.*dt)
                    results.append(flux_in)
                    results.append(flux_out)
                    if (args.region_filter!='none'):
                        region_enter = filter[enter_r][angle_bin_enter]
                        region_leave = filter[leave_r][angle_bin_leave]
                        for j in range(len(regions)-1):
                            prop_enter_region = prop_enter[(region_enter > regions[j]) & (region_enter < regions[j+1])]
                            prop_leave_region = prop_leave[(region_leave > regions[j]) & (region_leave < regions[j+1])]
                            flux_in = np.sum(prop_enter_region)/(5.*dt)
                            flux_out = np.sum(prop_leave_region)/(5.*dt)
                            results.append(flux_in)
                            results.append(flux_out)
            table.add_row(results)

        if ('accretion_direction' in plots):
            if (args.dark_matter):
                plot_accretion_direction(theta_enter*(np.pi/180.)+np.pi, phi_enter*(np.pi/180.), temperature[enter_r], metallicity[enter_r], rv[enter_r], tcool[enter_r], mass[enter_r], metals[enter_r], theta_leave*(np.pi/180.)+np.pi, phi_leave*(np.pi/180.), tsnap, zsnap, prefix, snap, radii[r], save_r, theta_acc_dm=theta_enter_dm*(np.pi/180.)+np.pi, phi_acc_dm=phi_enter_dm*(np.pi/180.), mass_dm=mass_dm[enter_r_dm])
            else:
                plot_accretion_direction(theta_enter*(np.pi/180.)+np.pi, phi_enter*(np.pi/180.), temperature[enter_r], metallicity[enter_r], rv[enter_r], tcool[enter_r], mass[enter_r], metals[enter_r], theta_leave*(np.pi/180.)+np.pi, phi_leave*(np.pi/180.), tsnap, zsnap, prefix, snap, radii[r], save_r)

    table = set_flux_table_units(table)
    table.write(tablename + flux_filename + save_suffix + '.hdf5', path='all_data', serialize_meta=True, overwrite=True)

def compare_accreting_cells(ds, grid, shape, snap, snap_props):
    '''Calculates properties of cells that are identified as accreting compared to cells in a thin
    boundary layer that are not accreting.'''

    prefix = output_dir + 'stats_halo_00' + args.halo + '/' + args.run + '/'
    plot_prefix = output_dir + 'fluxes_halo_00' + args.halo + '/' + args.run + '/'
    tablename = prefix + 'Tables/' + snap + '_accretion-compare'
    Menc_profile, Mvir, Rvir = snap_props
    tsnap = ds.current_time.in_units('Gyr').v
    zsnap = ds.get_parameter('CosmologyCurrentRedshift')

    # Set up table of everything we want
    props = []
    props.append('covering_fraction')
    props.append('mass')
    props.append('metal_mass')
    props.append('thermal_energy')
    props.append('radial_kinetic_energy')
    props.append('turbulent_kinetic_energy')
    props.append('rotational_kinetic_energy')
    props.append('cooling_energy')
    props.append('temperature')
    props.append('metallicity')
    props.append('cooling_time')
    props.append('entropy')
    props.append('pressure')
    props.append('radial_velocity')
    props.append('sound_speed')
    table = make_props_table(props)

    dx = float(np.min(grid[('gas','dx')].in_units('kpc')))
    smooth_scale = (25./dx)/6.

    # Load grid properties
    x = grid['gas', 'x'].in_units('kpc').v - ds.halo_center_kpc[0].v
    y = grid['gas', 'y'].in_units('kpc').v - ds.halo_center_kpc[1].v
    z = grid['gas', 'z'].in_units('kpc').v - ds.halo_center_kpc[2].v
    xbins = x[:,0,0][:-1] - 0.5*np.diff(x[:,0,0])
    ybins = y[0,:,0][:-1] - 0.5*np.diff(y[0,:,0])
    zbins = z[0,0,:][:-1] - 0.5*np.diff(z[0,0,:])
    vx = grid['gas','vx_corrected'].in_units('kpc/yr').v
    vy = grid['gas','vy_corrected'].in_units('kpc/yr').v
    vz = grid['gas','vz_corrected'].in_units('kpc/yr').v
    radius = grid['gas','radius_corrected'].in_units('kpc').v
    theta = grid['gas','theta_pos_disk'].v*(180./np.pi)
    phi = grid['gas','phi_pos_disk'].v*(180./np.pi)
    rv = grid['radial_velocity_corrected'].in_units('km/s').v
    temperature = grid['gas','temperature'].in_units('K').v
    metallicity = grid['gas','metallicity'].in_units('Zsun').v
    tcool = grid['gas','cooling_time'].in_units('Myr').v
    entropy = grid['gas','entropy'].in_units('cm**2*keV').v
    pressure = grid['gas','pressure'].in_units('erg/cm**3').v
    mass = grid['gas', 'cell_mass'].in_units('Msun').v
    metals = grid['gas', 'metal_mass'].in_units('Msun').v
    sound_speed = grid['gas','sound_speed'].in_units('km/s').v
    thermal = grid['gas','thermal_energy'].in_units('erg/g').v*grid['gas','cell_mass'].in_units('g').v
    cooling_energy = thermal/(tcool*1e6)*(5.*dt)
    if (args.weight=='mass'): weights = np.copy(mass)
    if (args.weight=='volume'): weights = grid['gas','cell_volume'].in_units('kpc**3').v
    smooth_vx = gaussian_filter(vx, smooth_scale)
    smooth_vy = gaussian_filter(vy, smooth_scale)
    smooth_vz = gaussian_filter(vz, smooth_scale)
    smooth_vr = gaussian_filter(rv, smooth_scale)*1e5
    sig_x = (vx - smooth_vx)**2.*(cmtopc*1000/stoyr)**2.
    sig_y = (vy - smooth_vy)**2.*(cmtopc*1000/stoyr)**2.
    sig_z = (vz - smooth_vz)**2.*(cmtopc*1000/stoyr)**2.
    turbulent_kinetic = 1./2.*(sig_x + sig_y + sig_z)*grid['gas','cell_mass'].in_units('g').v
    radial_kinetic = 1./2.*smooth_vr**2.*grid['gas','cell_mass'].in_units('g').v
    rotational_kinetic = 1./2.*(smooth_vx**2. + smooth_vy**2. + smooth_vz**2.)*(cmtopc*1000/stoyr)**2.*grid['gas','cell_mass'].in_units('g').v
    # Load dark matter velocities and positions and digitize onto grid
    properties = [mass, metals, thermal, radial_kinetic, turbulent_kinetic, rotational_kinetic, cooling_energy, temperature, metallicity, tcool, entropy, pressure, rv, sound_speed]

    # Calculate new positions of gas cells
    new_x = vx*(5.*dt) + x
    new_y = vy*(5.*dt) + y
    new_z = vz*(5.*dt) + z
    displacement = np.sqrt((new_x-x)**2. + (new_y-y)**2. + (new_z-z)**2.)
    displacement_vel = displacement*1000.*cmtopc/(5.*dt*stoyr)/1e5
    inds_x = np.digitize(new_x, xbins)-1      # indices of new x positions
    inds_y = np.digitize(new_y, ybins)-1      # indices of new y positions
    inds_z = np.digitize(new_z, zbins)-1      # indices of new z positions
    new_inds = np.array([inds_x, inds_y, inds_z])


    # Define shape edge
    struct = ndimage.generate_binary_structure(3,3)
    shape_expanded = ndimage.binary_dilation(shape, structure=struct, iterations=2)
    shape_edge = shape_expanded & ~shape

    # Set up filtering
    if (args.region_filter!='none'):
        if (args.region_filter=='temperature'):
            regions = [0., 10**4.9, 10**5.5, np.inf]
            filter = np.copy(temperature)
        elif (args.region_filter=='metallicity'):
            regions = [0., 1e-2, 1e-1, np.inf]
            filter = np.copy(metallicity)

    disp_vel_bins = [0., 50., 100., 150., 200., 300., np.inf]
    disp_vel_saves = ['_0-50','_50-100','_100-150','_150-200','_200-300','_300-inf']

    # If calculating direction of accretion, set up bins
    if (args.direction):
        phi_bins = ['all','major','minor']
    else:
        phi_bins = ['all']

    # If stepping through radius, set up radii list
    if (surface[0]=='sphere') and (args.radial_stepping>0):
        if (args.Rvir):
            max_R = surface[1]*Rvir
        else:
            max_R = surface[1]
        min_R = 0.1*Rvir
        radii = np.linspace(min_R, max_R, args.radial_stepping+1)[1:]
    else:
        radii = [0]

    # Step through radii (if chosen) and calculate properties for each radius
    for r in range(len(radii)):
        # If stepping through radii, define the shape and edge for this radius value
        if (surface[0]=='sphere') and (args.radial_stepping>0):
            shape = (radius < radii[r])
            shape_expanded = ndimage.binary_dilation(shape, structure=struct, iterations=2)
            shape_edge = shape_expanded & ~shape
            save_r = '_r%d' % (r)
        else:
            save_r = ''
        # Define which cells are entering shape
        new_in_shape = shape[tuple(new_inds)]
        to_shape = ~shape & new_in_shape
        from_shape = shape & ~new_in_shape
        from_shape_fast = from_shape & (rv > 200.)
        shape_non = shape_edge & ~to_shape

        # Bin cells by displacement velocity
        to_shape_dv = []
        for dv in range(len(disp_vel_bins)-1):
            low_dv = disp_vel_bins[dv]
            upp_dv = disp_vel_bins[dv+1]
            to_shape_dv.append(to_shape & (displacement_vel >= low_dv) & (displacement_vel < upp_dv))

        theta_to = theta[to_shape]
        phi_to = phi[to_shape]
        theta_to_dv = []
        phi_to_dv = []
        for dv in range(len(disp_vel_bins)-1):
            theta_to_dv.append(theta[to_shape_dv[dv]])
            phi_to_dv.append(phi[to_shape_dv[dv]])
        theta_edge = theta[shape_edge]
        phi_edge = phi[shape_edge]
        theta_non = theta[shape_non]
        phi_non = phi[shape_non]
        # Calculate covering fraction of accretion
        nside = 32
        pix_area = healpy.nside2pixarea(nside)

        angle_bin_to_dv = []
        for p in range(len(phi_bins)):
            if (surface[0]=='sphere') and (args.radial_stepping>0):
                results = [radii[r]]
            elif ('disk' in surface[0]):
                results = [np.mean(radius[shape_edge][(phi_edge>=60.)&(phi_edge<=120.)])]
            else:
                results = []
            if (args.direction):
                results.append(phi_bins[p])
            if (phi_bins[p]=='all'):
                angle_bin_to = np.ones(len(phi_to), dtype=bool)
                angle_bin_edge = np.ones(len(phi_edge), dtype=bool)
                angle_bin_non = np.ones(len(phi_non), dtype=bool)
                for dv in range(len(disp_vel_bins)-1):
                    angle_bin_to_dv.append(np.ones(np.count_nonzero(to_shape_dv[dv]), dtype=bool))
            elif (phi_bins[p]=='major'):
                angle_bin_to = (phi_to >= 60.) & (phi_to <= 120.)
                angle_bin_edge = (phi_edge >= 60.) & (phi_edge <= 120.)
                angle_bin_non = (phi_non >= 60.) & (phi_non <= 120.)
                for dv in range(len(disp_vel_bins)-1):
                    angle_bin_to_dv.append((phi_to_dv[dv] >= 60.) & (phi_to_dv[dv] <= 120.))
            elif (phi_bins[p]=='minor'):
                angle_bin_to = (phi_to < 60.) | (phi_to > 120.)
                angle_bin_edge = (phi_edge < 60.) | (phi_edge > 120.)
                angle_bin_non = (phi_non < 60.) | (phi_non > 120.)
                for dv in range(len(disp_vel_bins)-1):
                    angle_bin_to_dv.append((phi_to_dv[dv] < 60.) | (phi_to_dv[dv] > 120.))
            pixel_acc = healpy.ang2pix(nside, phi_to[angle_bin_to]*(np.pi/180.), theta_to[angle_bin_to]*(np.pi/180.)+np.pi)
            u, dup_ind = np.unique(pixel_acc, return_index=True)
            covering_acc = len(u)*pix_area
            covering_acc_dv = []
            for dv in range(len(disp_vel_bins)-1):
                pixel_acc = healpy.ang2pix(nside, phi_to_dv[dv][angle_bin_to_dv[dv]]*(np.pi/180.), theta_to_dv[dv][angle_bin_to_dv[dv]]*(np.pi/180.)+np.pi)
                u, dup_ind = np.unique(pixel_acc, return_index=True)
                covering_acc_dv.append(len(u)*pix_area)
            pixel_non = healpy.ang2pix(nside, phi_non[angle_bin_non]*(np.pi/180.), theta_non[angle_bin_non]*(np.pi/180.)+np.pi)
            u, dup_ind = np.unique(pixel_non, return_index=True)
            covering_non = len(u)*pix_area
            pixel_edge = healpy.ang2pix(nside, phi_edge[angle_bin_edge]*(np.pi/180.), theta_edge[angle_bin_edge]*(np.pi/180.)+np.pi)
            u, dup_ind = np.unique(pixel_edge, return_index=True)
            covering_edge = len(u)*pix_area
            results.append(covering_non/covering_edge)
            results.append(covering_acc/covering_edge)
            for dv in range(len(disp_vel_bins)-1):
                results.append(covering_acc_dv[dv]/covering_edge)
            weights_to = weights[to_shape][angle_bin_to]
            weights_to_dv = []
            for dv in range(len(disp_vel_bins)-1):
                weights_to_dv.append(weights[to_shape_dv[dv]][angle_bin_to_dv[dv]])
            weights_edge = weights[shape_edge][angle_bin_edge]
            weights_non = weights[shape_non][angle_bin_non]
            if (args.region_filter!='none'):
                region_to = filter[to_shape][angle_bin_to]
                region_to_dv = []
                for dv in range(len(disp_vel_bins)-1):
                    region_to_dv.append(filter[to_shape_dv[dv]][angle_bin_to_dv[dv]])
                region_edge = filter[shape_edge][angle_bin_edge]
                region_non = filter[shape_non][angle_bin_non]
                for f in range(len(regions)-1):
                    phi_non_f = phi_non[angle_bin_non][(region_non>=regions[f]) & (region_non<regions[f+1])]
                    theta_non_f = theta_non[angle_bin_non][(region_non>=regions[f]) & (region_non<regions[f+1])]
                    pixel_f = healpy.ang2pix(nside, phi_non_f*(np.pi/180.), theta_non_f*(np.pi/180.)+np.pi)
                    u, dup_ind = np.unique(pixel_f, return_index=True)
                    covering_f = len(u)*pix_area
                    results.append(covering_f/covering_edge)
                    phi_to_f = phi_to[angle_bin_to][(region_to>=regions[f]) & (region_to<regions[f+1])]
                    theta_to_f = theta_to[angle_bin_to][(region_to>=regions[f]) & (region_to<regions[f+1])]
                    pixel_f = healpy.ang2pix(nside, phi_to_f*(np.pi/180.), theta_to_f*(np.pi/180.)+np.pi)
                    u, dup_ind = np.unique(pixel_f, return_index=True)
                    covering_f = len(u)*pix_area
                    results.append(covering_f/covering_edge)
                    for dv in range(len(disp_vel_bins)-1):
                        phi_to_f = phi_to_dv[dv][angle_bin_to_dv[dv]][(region_to_dv[dv]>=regions[f]) & (region_to_dv[dv]<regions[f+1])]
                        theta_to_f = theta_to_dv[dv][angle_bin_to_dv[dv]][(region_to_dv[dv]>=regions[f]) & (region_to_dv[dv]<regions[f+1])]
                        pixel_f = healpy.ang2pix(nside, phi_to_f*(np.pi/180.), theta_to_f*(np.pi/180.)+np.pi)
                        results.append(covering_f/covering_edge)
            for i in range(len(properties)):
                prop_to = properties[i][to_shape][angle_bin_to]
                prop_to_dv = []
                for dv in range(len(disp_vel_bins)-1):
                    prop_to_dv.append(properties[i][to_shape_dv[dv]][angle_bin_to_dv[dv]])
                prop_edge = properties[i][shape_edge][angle_bin_edge]
                prop_non = properties[i][shape_non][angle_bin_non]
                if ('mass' in props[i+1]) or ('energy' in props[i+1]):
                    results.append(np.sum(prop_edge))
                    results.append(np.sum(prop_non))
                    results.append(np.sum(prop_to))
                    for dv in range(len(disp_vel_bins)-1):
                        results.append(np.sum(prop_to_dv[dv]))
                    if (args.region_filter!='none'):
                        for f in range(len(regions)-1):
                            results.append(np.sum(prop_edge[(region_edge>=regions[f]) & (region_edge<regions[f+1])]))
                            results.append(np.sum(prop_non[(region_non>=regions[f]) & (region_non<regions[f+1])]))
                            results.append(np.sum(prop_to[(region_to>=regions[f]) & (region_to<regions[f+1])]))
                            for dv in range(len(disp_vel_bins)-1):
                                results.append(np.sum(prop_to_dv[dv][(region_to_dv[dv]>=regions[f]) & (region_to_dv[dv]<regions[f+1])]))
                else:
                    if (len(prop_edge)>0):
                        quantiles = weighted_quantile(prop_edge, weights_edge, np.array([0.25,0.5,0.75]))
                        results.append(quantiles[1])
                        results.append(quantiles[2]-quantiles[0])
                        avg, std = weighted_avg_and_std(prop_edge, weights_edge)
                        results.append(avg)
                        results.append(std)
                    else:
                        results.append(np.nan)
                        results.append(np.nan)
                        results.append(np.nan)
                        results.append(np.nan)
                    if (len(prop_non)>0):
                        quantiles = weighted_quantile(prop_non, weights_non, np.array([0.25,0.5,0.75]))
                        results.append(quantiles[1])
                        results.append(quantiles[2]-quantiles[0])
                        avg, std = weighted_avg_and_std(prop_non, weights_non)
                        results.append(avg)
                        results.append(std)
                    else:
                        results.append(np.nan)
                        results.append(np.nan)
                        results.append(np.nan)
                        results.append(np.nan)
                    if (len(prop_to)>0):
                        quantiles = weighted_quantile(prop_to, weights_to, np.array([0.25,0.5,0.75]))
                        results.append(quantiles[1])
                        results.append(quantiles[2]-quantiles[0])
                        avg, std = weighted_avg_and_std(prop_to, weights_to)
                        results.append(avg)
                        results.append(std)
                    else:
                        results.append(np.nan)
                        results.append(np.nan)
                        results.append(np.nan)
                        results.append(np.nan)
                    for dv in range(len(disp_vel_bins)-1):
                        if (len(prop_to_dv[dv])>0):
                            quantiles = weighted_quantile(prop_to_dv[dv], weights_to_dv[dv], np.array([0.25,0.5,0.75]))
                            results.append(quantiles[1])
                            results.append(quantiles[2]-quantiles[0])
                            avg, std = weighted_avg_and_std(prop_to_dv[dv], weights_to_dv[dv])
                            results.append(avg)
                            results.append(std)
                        else:
                            results.append(np.nan)
                            results.append(np.nan)
                            results.append(np.nan)
                            results.append(np.nan)
                    if (args.region_filter!='none'):
                        for f in range(len(regions)-1):
                            prop_edge_f = prop_edge[(region_edge>=regions[f]) & (region_edge<regions[f+1])]
                            weights_edge_f = weights_edge[(region_edge>=regions[f]) & (region_edge<regions[f+1])]
                            if (len(prop_edge_f)>0):
                                quantiles = weighted_quantile(prop_edge_f, weights_edge_f, np.array([0.25,0.5,0.75]))
                                results.append(quantiles[1])
                                results.append(quantiles[2]-quantiles[0])
                                avg, std = weighted_avg_and_std(prop_edge_f, weights_edge_f)
                                results.append(avg)
                                results.append(std)
                            else:
                                results.append(np.nan)
                                results.append(np.nan)
                                results.append(np.nan)
                                results.append(np.nan)
                            prop_non_f = prop_non[(region_non>=regions[f]) & (region_non<regions[f+1])]
                            weights_non_f = weights_non[(region_non>=regions[f]) & (region_non<regions[f+1])]
                            if (len(prop_non_f)>0):
                                quantiles = weighted_quantile(prop_non_f, weights_non_f, np.array([0.25,0.5,0.75]))
                                results.append(quantiles[1])
                                results.append(quantiles[2]-quantiles[0])
                                avg, std = weighted_avg_and_std(prop_non_f, weights_non_f)
                                results.append(avg)
                                results.append(std)
                            else:
                                results.append(np.nan)
                                results.append(np.nan)
                                results.append(np.nan)
                                results.append(np.nan)
                            prop_to_f = prop_to[(region_to>=regions[f]) & (region_to<regions[f+1])]
                            weights_to_f = weights_to[(region_to>=regions[f]) & (region_to<regions[f+1])]
                            if (len(prop_to_f)>0):
                                quantiles = weighted_quantile(prop_to_f, weights_to_f, np.array([0.25,0.5,0.75]))
                                results.append(quantiles[1])
                                results.append(quantiles[2]-quantiles[0])
                                avg, std = weighted_avg_and_std(prop_to_f, weights_to_f)
                                results.append(avg)
                                results.append(std)
                            else:
                                results.append(np.nan)
                                results.append(np.nan)
                                results.append(np.nan)
                                results.append(np.nan)
                            for dv in range(len(disp_vel_bins)-1):
                                prop_to_f = prop_to_dv[dv][(region_to_dv[dv]>=regions[f]) & (region_to_dv[dv]<regions[f+1])]
                                weights_to_f = weights_to_dv[dv][(region_to_dv[dv]>=regions[f]) & (region_to_dv[dv]<regions[f+1])]
                                if (len(prop_to_f)>0):
                                    quantiles = weighted_quantile(prop_to_f, weights_to_f, np.array([0.25,0.5,0.75]))
                                    results.append(quantiles[1])
                                    results.append(quantiles[2]-quantiles[0])
                                    avg, std = weighted_avg_and_std(prop_to_f, weights_to_f)
                                    results.append(avg)
                                    results.append(std)
                                else:
                                    results.append(np.nan)
                                    results.append(np.nan)
                                    results.append(np.nan)
                                    results.append(np.nan)

            table.add_row(results)

        if ('accretion_direction' in plots):
            plot_accretion_direction(theta_to*(np.pi/180.)+np.pi, phi_to*(np.pi/180.), temperature[to_shape], metallicity[to_shape], rv[to_shape], tcool[to_shape], mass[to_shape], metals[to_shape], theta[from_shape_fast]*(np.pi/180.)+np.pi, phi[from_shape_fast]*(np.pi/180.), tsnap, zsnap, plot_prefix, snap, radii[r], save_r, '')
            for dv in range(len(disp_vel_bins)-1):
                save_dv = disp_vel_saves[dv]
                plot_accretion_direction(theta_to_dv[dv]*(np.pi/180.)+np.pi, phi_to_dv[dv]*(np.pi/180.), temperature[to_shape_dv[dv]], metallicity[to_shape_dv[dv]], rv[to_shape_dv[dv]], tcool[to_shape_dv[dv]], mass[to_shape_dv[dv]], metals[to_shape_dv[dv]], theta[from_shape_fast]*(np.pi/180.)+np.pi, phi[from_shape_fast]*(np.pi/180.), tsnap, zsnap, plot_prefix, snap, radii[r], save_r, save_dv)
        if ('phase_plot' in plots):
            phase_plots(temperature[to_shape], rv[to_shape], tcool[to_shape], metallicity[to_shape], entropy[to_shape], pressure[to_shape], mass[to_shape], temperature[shape_non], rv[shape_non], tcool[shape_non], metallicity[shape_non], entropy[shape_non], pressure[shape_non], mass[shape_non], tsnap, zsnap, prefix, snap, radii[r], save_r, '')
            for dv in range(len(disp_vel_bins)-1):
                save_dv = disp_vel_saves[dv]
                phase_plots(temperature[to_shape_dv[dv]], rv[to_shape_dv[dv]], tcool[to_shape_dv[dv]], metallicity[to_shape_dv[dv]], entropy[to_shape_dv[dv]], pressure[to_shape_dv[dv]], mass[to_shape_dv[dv]], temperature[shape_non], rv[shape_non], tcool[shape_non], metallicity[shape_non], entropy[shape_non], pressure[shape_non], mass[shape_non], tsnap, zsnap, prefix, snap, radii[r], save_r, save_dv)

    table = set_props_table_units(table)
    table.write(tablename + save_suffix + '.hdf5', path='all_data', serialize_meta=True, overwrite=True)

def phase_plots(temp_acc, vel_acc, tcool_acc, met_acc, entropy_acc, pressure_acc, mass_acc, temp_else, vel_else, tcool_else, met_else, entropy_else, pressure_else, mass_else, tsnap, zsnap, prefix, snap, radius, save_r, save_dv):
    '''Makes 2D phase plots of temperature vs radial velocity, temperature vs cooling time, temperature
    vs velocity dispersion, radial velocity vs velocity dispersion, radial velocity vs cooling time,
    and cooling time vs velocity dispersion for only that gas that is accreting and for the gas that is not
    accreting in the same shell, for a given radius at a given snapshot in time.'''

    props_acc = [np.log10(temp_acc), vel_acc, np.log10(tcool_acc), np.log10(entropy_acc), np.log10(pressure_acc), np.log10(met_acc)]
    props_else = [np.log10(temp_else), vel_else, np.log10(tcool_else), np.log10(entropy_else), np.log10(pressure_else), np.log10(met_else)]
    props_labels = ['log Temperature [K]', 'Radial velocity [km/s]', 'log Cooling time [Myr]', 'log Entropy [keV cm$^2$]', 'log Pressure [erg/cm$^3$]', 'log Metallicity [$Z_\odot$]']
    props_save = ['temp', 'rv', 'tcool', 'entropy', 'pressure', 'met']
    ranges = [[3,7], [-300,300], [0,6], [0,3], [-17,-13], [-4,0.5]]
    for i in range(len(props_acc)):
        for j in range(i+1, len(props_acc)):
            fig = plt.figure(figsize=(12,5), dpi=300)
            ax_acc = fig.add_subplot(1,2,1)
            ax_else = fig.add_subplot(1,2,2)

            hist = ax_acc.hist2d(props_acc[i], props_acc[j], bins=100, range=[ranges[i],ranges[j]], weights=np.log10(mass_acc), cmap=plt.cm.BuPu)
            ax_else.hist2d(props_else[i], props_else[j], bins=100, range=[ranges[i],ranges[j]], weights=np.log10(mass_else), cmap=plt.cm.BuPu)
            ax_acc.axis([ranges[i][0],ranges[i][1],ranges[j][0],ranges[j][1]])
            ax_else.axis([ranges[i][0],ranges[i][1],ranges[j][0],ranges[j][1]])
            ax_acc.set_xlabel(props_labels[i], fontsize=18)
            ax_acc.set_ylabel(props_labels[j], fontsize=18)
            ax_else.set_xlabel(props_labels[i], fontsize=18)
            ax_else.set_ylabel(props_labels[j], fontsize=18)
            ax_acc.tick_params(axis='both', which='both', direction='in', length=8, width=2, pad=5, labelsize=14)
            ax_else.tick_params(axis='both', which='both', direction='in', length=8, width=2, pad=5, labelsize=14)
            ax_acc.text(0.05, 0.9, 'Accreting gas', ha='left', va='center', fontsize=14, transform=ax_acc.transAxes)
            ax_else.text(0.05, 0.9, 'Non-accreting gas', ha='left', va='center', fontsize=14, transform=ax_else.transAxes)
            if (props_save[j]=='met'): ax_acc.plot([ranges[i][0],ranges[i][1]], [-2, -2], 'k-', lw=1)
            cbaxes = fig.add_axes([0.65, 0.95, 0.25, 0.03])
            cbar = plt.colorbar(hist[3], cax=cbaxes, orientation='horizontal', ticks=[])
            cbaxes.text(0, -0.1, 'Less mass', fontsize=16, ha='center', va='top', transform=cbaxes.transAxes)
            cbaxes.text(1, -0.1, 'More mass', fontsize=16, ha='center', va='top', transform=cbaxes.transAxes)
            fig.subplots_adjust(left=0.09, bottom=0.12, right=0.97, wspace=0.26)
            ax_acc.text(0.95, 0.95, '%.2f Gyr\n$z=%.2f$' % (tsnap, zsnap), fontsize=14, ha='right', va='bottom', transform=ax_acc.transAxes, bbox={'fc':'white','ec':'black','boxstyle':'round','lw':2})
            if (args.radial_stepping>0):
                ax_acc.text(0.2, 1.05, '$r=%.2f$ kpc' % (radius), ha='left', va='bottom', fontsize=16, transform=ax_acc.transAxes)
            plt.savefig(prefix + 'Plots/' + snap + '_phase_' + props_save[i] + '-vs-' + props_save[j] + save_r + save_dv + save_suffix + '.png')
            plt.close()

def find_shape(ds, surface, snap_props):
    '''Defines the grid within the data set, identifies the specified shape,
    and returns both the full grid and the boolean array for the shape.'''

    Menc_profile, Mvir, Rvir = snap_props
    edge_kpc = 20.      # Edge region around shape in kpc

    if (args.constant_box!=0.):
        if (args.Rvir):
            max_extent = args.constant_box/2.*Rvir
        else:
            max_extent = args.constant_box/2.
    else:
        if (surface[0]=='disk'):
            max_extent = edge_kpc * 3.
        elif (surface[0]=='stellar_disk'):
            max_extent = edge_kpc * 2.
        elif (surface[0]=='sphere'):
            if (args.Rvir):
                max_extent = surface[1]*Rvir + 2.*edge_kpc
            else:
                max_extent = surface[1] + 2.*edge_kpc
        elif (surface[0]=='cylinder'):
            if (args.Rvir):
                radius = surface[1] * Rvir
                height = surface[2] * Rvir
            else:
                radius = surface[1]
                height = surface[2]
            max_extent = np.max([radius, height/2.])*np.sqrt(2.) + edge_kpc*2.

    data = ds.sphere(ds.halo_center_kpc, (max_extent, 'kpc'))
    pix_res = float(np.min(data[('gas','dx')].in_units('kpc')))  # at level 11
    lvl1_res = pix_res*2.**11.
    dx = lvl1_res/(2.**args.level)

    if (args.constant_box!=0.):
        left_edge = ds.halo_center_kpc - ds.arr([max_extent, max_extent, max_extent], 'kpc')
        box_width = ds.arr([int(2.*max_extent/dx), int(2.*max_extent/dx), int(2.*max_extent/dx)])
        box = ds.covering_grid(level=args.level, left_edge=left_edge, dims=box_width, num_ghost_zones=1)
    else:
        if (surface[0]=='disk'):
            # Define the density cut between disk and CGM to vary smoothly between 1 and 0.1 between z = 0.5 and z = 0.25,
            # with it being 1 at higher redshifts and 0.1 at lower redshifts
            current_time = ds.current_time.in_units('Myr').v
            if (current_time<=8656.88):
                density_cut_factor = 1.
            elif (current_time<=10787.12):
                density_cut_factor = 1. - 0.9*(current_time-8656.88)/2130.24
            else:
                density_cut_factor = 0.1
            density = data['gas','density'].in_units('g/cm**3')
            disk = data.include_above(('gas','density'), density_cut_factor * cgm_density_max)
            x = disk['gas','x'].in_units('kpc').v - ds.halo_center_kpc[0].v
            y = disk['gas','y'].in_units('kpc').v - ds.halo_center_kpc[1].v
            z = disk['gas','z'].in_units('kpc').v - ds.halo_center_kpc[2].v
            x_extent = max([np.max(x)+2.*edge_kpc,np.abs(np.min(x)-2.*edge_kpc)])
            y_extent = max([np.max(y)+2.*edge_kpc,np.abs(np.min(y)-2.*edge_kpc)])
            z_extent = max([np.max(z)+2.*edge_kpc,np.abs(np.min(z)-2.*edge_kpc)])
            left_edge = ds.halo_center_kpc - ds.arr([x_extent, y_extent, z_extent], 'kpc')
            box_width = np.array([int(2.*x_extent/dx), int(2.*y_extent/dx), int(2.*z_extent/dx)])
            box = ds.covering_grid(level=args.level, left_edge=left_edge, dims=box_width, num_ghost_zones=1)
        elif (surface[0]=='stellar_disk'):
            x_stars = data['young_stars8','x_disk'].v
            y_stars = data['young_stars8','y_disk'].v
            z_stars = data['young_stars8','z_disk'].v
            r_stars = data['young_stars8','radius_corrected'].v
            x_stars = x_stars[r_stars < 20.]
            y_stars = y_stars[r_stars < 20.]
            z_stars = z_stars[r_stars < 20.]
            x_extent = max([np.max(x_stars)+2.*edge_kpc,np.abs(np.min(x_stars)-2.*edge_kpc)])
            y_extent = max([np.max(y_stars)+2.*edge_kpc,np.abs(np.min(y_stars)-2.*edge_kpc)])
            z_extent = max([np.max(z_stars)+2.*edge_kpc,np.abs(np.min(z_stars)-2.*edge_kpc)])
            left_edge = ds.halo_center_kpc - ds.arr([x_extent, y_extent, z_extent], 'kpc')
            box_width = np.array([int(2.*x_extent/dx), int(2.*y_extent/dx), int(2.*z_extent/dx)])
            box = ds.covering_grid(level=args.level, left_edge=left_edge, dims=box_width, num_ghost_zones=1)
        elif (surface[0]=='sphere'):
            left_edge = ds.halo_center_kpc - ds.arr([max_extent, max_extent, max_extent], 'kpc')
            box_width = ds.arr([int(max_extent*2./dx), int(max_extent*2./dx), int(max_extent*2./dx)], 'kpc')
            box = ds.covering_grid(level=args.level, left_edge=left_edge, dims=box_width, num_ghost_zones=1)
        elif (surface[0]=='cylinder'):
            left_edge = ds.halo_center_kpc - ds.arr([max_extent, max_extent, max_extent], 'kpc')
            box_width = ds.arr([int(max_extent*2./dx), int(max_extent*2./dx), int(max_extent*2./dx)])
            box = ds.covering_grid(level=args.level, left_edge=left_edge, dims=box_width, num_ghost_zones=1)

    if (surface[0]=='disk'):
        # Define the density cut between disk and CGM to vary smoothly between 1 and 0.1 between z = 0.5 and z = 0.25,
        # with it being 1 at higher redshifts and 0.1 at lower redshifts
        current_time = ds.current_time.in_units('Myr').v
        if (current_time<=8656.88):
            density_cut_factor = 1.
        elif (current_time<=10787.12):
            density_cut_factor = 1. - 0.9*(current_time-8656.88)/2130.24
        else:
            density_cut_factor = 0.1
        density = box['gas','density'].in_units('g/cm**3').v
        smooth_density = gaussian_filter(density, (5./dx)/6.)
        shape = (smooth_density > density_cut_factor * cgm_density_max)
    elif (surface[0]=='stellar_disk'):
        x_stars = data['young_stars8','x_disk'].v
        y_stars = data['young_stars8','y_disk'].v
        z_stars = data['young_stars8','z_disk'].v
        r_stars = data['young_stars8','radius_corrected'].v
        x_stars = x_stars[r_stars < 20.]
        y_stars = y_stars[r_stars < 20.]
        z_stars = z_stars[r_stars < 20.]
        stars_radius = np.max(np.sqrt(x_stars**2. + y_stars**2.))
        stars_height = np.max(np.abs(z_stars))
        x = box['gas','x_disk'].in_units('kpc').v
        y = box['gas','y_disk'].in_units('kpc').v
        z = box['gas','z_disk'].in_units('kpc').v
        shape = (z >= -stars_height) & (z <= stars_height) & (np.sqrt(x**2.+y**2.) <= stars_radius)
    elif (surface[0]=='sphere'):
        if (args.Rvir):
            R = surface[1] * Rvir
        else:
            R = surface[1]
        radius = box['gas','radius_corrected'].in_units('kpc').v
        shape = (radius < R)
    elif (surface[0]=='cylinder'):
        if (args.Rvir):
            radius = surface[1] * Rvir
            height = surface[2] * Rvir
        else:
            radius = surface[1]
            height = surface[2]
        if (surface[3]=='minor'):
            x = box['gas','x_disk'].in_units('kpc').v
            y = box['gas','y_disk'].in_units('kpc').v
            z = box['gas','z_disk'].in_units('kpc').v
        else:
            x = box['gas','x'].in_units('kpc').v - ds.halo_center_kpc[0].v
            y = box['gas','y'].in_units('kpc').v - ds.halo_center_kpc[1].v
            z = box['gas','z'].in_units('kpc').v - ds.halo_center_kpc[2].v
        if (surface[3]=='z') or (surface[3]=='minor'):
            norm_coord = z
            rad_coord = np.sqrt(x**2. + y**2.)
        if (surface[3]=='x'):
            norm_coord = x
            rad_coord = np.sqrt(y**2. + z**2.)
        if (surface[3]=='y'):
            norm_coord = y
            rad_coord = np.sqrt(x**2. + z**2.)
        if (type(surface[3])==tuple) or (type(surface[3])==list):
            axis = np.array(surface[3])
            norm_axis = axis / np.sqrt((axis**2.).sum())
            # Define other unit vectors orthagonal to the angular momentum vector
            np.random.seed(99)
            x_axis = np.random.randn(3)            # take a random vector
            x_axis -= x_axis.dot(norm_axis) * norm_axis       # make it orthogonal to L
            x_axis /= np.linalg.norm(x_axis)            # normalize it
            y_axis = np.cross(norm_axis, x_axis)           # cross product with L
            x_vec = np.array(x_axis)
            y_vec = np.array(y_axis)
            z_vec = np.array(norm_axis)
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
            norm_coord = z_rot
            rad_coord = np.sqrt(x_rot**2. + y_rot**2.)

        shape = (norm_coord >= -height/2.) & (norm_coord <= height/2.) & (rad_coord <= radius)

    return box, shape

def load_and_calculate(snap, surface):
    '''Loads the simulation output given by 'snap' and calls the functions to define the surface
    and calculate flux through that surface.'''

    # Load simulation output
    if (args.system=='pleiades_cassi'):
        print('Copying directory to /tmp')
        if (args.copy_to_tmp):
            snap_dir = '/tmp/' + args.halo + '/' + args.run + '/' + target_dir + '/' + snap
            shutil.copytree(foggie_dir + run_dir + snap, snap_dir)
            snap_name = snap_dir + '/' + snap
        else:
            # Make a dummy directory with the snap name so the script later knows the process running
            # this snapshot failed if the directory is still there
            snap_dir = '/nobackup/clochhaa/tmp/' + args.halo + '/' + args.run + '/' + target_dir + '/' + snap
            os.makedirs(snap_dir)
            snap_name = foggie_dir + run_dir + snap + '/' + snap
    else:
        snap_name = foggie_dir + run_dir + snap + '/' + snap
    if ((surface[0]=='cylinder') and (surface[3]=='minor')) or (args.direction) or ('disk' in surface[0]):
        ds, refine_box = foggie_load(snap_name, trackname, do_filter_particles=True, halo_c_v_name=halo_c_v_name, gravity=True, masses_dir=catalog_dir, disk_relative=True, correct_bulk_velocity=True, smooth_AM_name=smooth_AM_name)
    else:
        ds, refine_box = foggie_load(snap_name, trackname, do_filter_particles=True, halo_c_v_name=halo_c_v_name, gravity=True, masses_dir=catalog_dir, correct_bulk_velocity=True)
    zsnap = ds.get_parameter('CosmologyCurrentRedshift')

    # Load the mass enclosed profile
    if (zsnap > 2.):
        masses = Table.read(catalog_dir + 'masses_z-gtr-2.hdf5', path='all_data')
    else:
        masses = Table.read(catalog_dir + 'masses_z-less-2.hdf5', path='all_data')
    rvir_masses = Table.read(catalog_dir + 'rvir_masses.hdf5', path='all_data')
    masses_ind = np.where(masses['snapshot']==snap)[0]
    Menc_profile = IUS(np.concatenate(([0],masses['radius'][masses_ind])), np.concatenate(([0],masses['total_mass'][masses_ind])))
    Mvir = rvir_masses['total_mass'][rvir_masses['snapshot']==snap][0]
    Rvir = rvir_masses['radius'][rvir_masses['snapshot']==snap][0]
    snap_props = [Menc_profile, Mvir, Rvir]

    if (args.plot=='sky_map'):
        sp = ds.sphere(ds.halo_center_kpc, (Rvir, 'kpc'))
        sky_map(ds, sp, snap, snap_props)
    else:
        # Find the covering grid and the shape specified by 'surface'
        grid, shape = find_shape(ds, surface, snap_props)
        # Calculate fluxes and save to file
        if ('fluxes' in args.calculate):
            calculate_flux(ds, grid, shape, snap, snap_props)
        if ('accretion_compare' in args.calculate):
            compare_accreting_cells(ds, grid, shape, snap, snap_props)

    print('Fluxes calculated for snapshot', snap)

    # Delete output from temp directory if on pleiades
    if (args.system=='pleiades_cassi'):
        print('Deleting directory from /tmp')
        shutil.rmtree(snap_dir)

def accretion_flux_vs_time(snaplist):
    '''Plots fluxes of accretion over time and redshift, broken into CGM sections if --region_filter is specified
    and broken into angle of accretion if --direction is specified.'''

    tablename_prefix = prefix + 'Tables/'
    time_table = Table.read(output_dir + 'times_halo_00' + args.halo + '/' + args.run + '/time_table.hdf5', path='all_data')

    fig = plt.figure(figsize=(13,6), dpi=200)
    ax = fig.add_subplot(1,1,1)

    if (args.region_filter=='temperature'):
        plot_colors = ['salmon', "#984ea3", "#4daf4a", 'darkorange', 'k']
        region_label = ['$<10^4$ K', '$10^4-10^5$ K', '$10^5-10^6$ K', '$>10^6$ K', 'All']
        region_name = ['lowest_', 'low-mid_', 'high-mid_', 'highest_']
    elif (args.region_filter=='metallicity'):
        plot_colors = ["#4575b4", "#984ea3", "#d73027", "darkorange", 'k']
        region_label = ['$<0.1Z_\odot$', '$0.1-0.5Z_\odot$', '$0.5-1Z_\odot$', '$>Z_\odot$', 'All']
        region_name = ['lowest_', 'low-mid_', 'high-mid_', 'highest_']
    elif (args.location_compare):
        plot_colors = ['m', 'c', 'g', 'b', 'k']
        region_label = ['Stellar disk', 'Gas disk', '$0.25R_\mathrm{vir}$', '$0.5R_\mathrm{vir}$', '$R_\mathrm{vir}$']
        filenames = ['_stellar-disk', '_gas-disk', '_0p25Rvir', '_0p5Rvir', '_Rvir']
        region_name = ['']

    if (args.direction):
        linestyles = ['-', '--']
        angle_labels = ['major axis', 'minor axis']
    else:
        linestyles = ['-']

    if (args.time_avg!=0):
        dt_step = 5.38*args.output_step
        avg_window = int(np.ceil(args.time_avg/dt_step))

    zlist = []
    timelist = []
    accretion_list = []
    for i in range(len(plot_colors)):
        accretion_list.append([])
        for j in range(len(linestyles)):
            accretion_list[i].append([])

    for i in range(len(snaplist)):
        snap = snaplist[i]
        if (args.location_compare):
            fluxes = Table.read(tablename_prefix + snap + '_fluxes_' + args.load_from_file + filenames[0] + '.hdf5', path='all_data')
        else:
            fluxes = Table.read(tablename_prefix + snap + '_fluxes' + args.load_from_file + '.hdf5', path='all_data')
        timelist.append(time_table['time'][time_table['snap']==snap][0]/1000.)
        zlist.append(time_table['redshift'][time_table['snap']==snap][0])
        for j in range(len(plot_colors)):
            if (args.location_compare):
                fluxes = Table.read(tablename_prefix + snap + '_fluxes_' + args.load_from_file + filenames[j] + '.hdf5', path='all_data')
            for k in range(len(linestyles)):
                if (args.direction):
                    if (k==0):
                        accretion_list[j][k].append(np.sum(fluxes[region_name[j] + 'mass_flux_in'][(fluxes['phi_bin']>=60.) & (fluxes['phi_bin']<120.)])-np.sum(fluxes[region_name[j] + 'mass_flux_out'][(fluxes['phi_bin']>=60.) & (fluxes['phi_bin']<120.)]))
                    if (k==1):
                        accretion_list[j][k].append(np.sum(fluxes[region_name[j] + 'mass_flux_in'][(fluxes['phi_bin']<60.) | (fluxes['phi_bin']>=120.)])-np.sum(fluxes[region_name[j] + 'mass_flux_out'][(fluxes['phi_bin']<60.) | (fluxes['phi_bin']>=120.)]))
                elif (args.location_compare):
                    accretion_list[j][k].append(fluxes['mass_flux_in'][fluxes['phi_bin']=='all'][0] -fluxes['mass_flux_out'][fluxes['phi_bin']=='all'][0])
                else:
                    accretion_list[j][k].append(fluxes[region_name[j] + 'mass_flux_in'][fluxes['phi_bin']=='all'][0]-fluxes[region_name[j] + 'mass_flux_out'][fluxes['phi_bin']=='all'][0])

    if (args.time_avg!=0):
        accretion_list_avgd = []
        for j in range(len(plot_colors)):
            accretion_list_avgd.append([])
            for k in range(len(linestyles)):
                avg = uniform_filter1d(accretion_list[j][k], size=avg_window)
                accretion_list_avgd[j].append(avg)
        accretion_list = accretion_list_avgd

    for j in range(len(plot_colors)):
        for k in range(len(linestyles)):
            if (k==0): label = region_label[j]
            else: label = '_nolegend_'
            ax.plot(timelist, accretion_list[j][k], color=plot_colors[j], ls=linestyles[k], lw=2, label=label)
            if (args.direction) and (j==len(plot_colors)-1):
                ax.plot([-100,-100], [-100,-100], color='k', ls=linestyles[k], lw=2, label=angle_labels[k])

    ax.axis([np.min(timelist), np.max(timelist), -30, 30])
    ax.set_ylabel('Accretion Rate [$M_\odot$/yr]', fontsize=18)
    #ax.set_yscale('log')

    zlist.reverse()
    timelist.reverse()
    time_func = IUS(zlist, timelist)
    timelist.reverse()
    timelist = np.array(timelist).flatten()
    zlist = np.array(zlist)

    ax2 = ax.twiny()
    ax.tick_params(axis='both', which='both', direction='in', length=8, width=2, pad=5, labelsize=16, \
      top=False, right=True)
    ax2.tick_params(axis='x', which='both', direction='in', length=8, width=2, pad=5, labelsize=16, \
      top=True)
    x0, x1 = ax.get_xlim()
    z_ticks = [2,1.5,1,.75,.5,.3,.2,.1,0]
    last_z = np.where(z_ticks >= zlist[0])[0][-1]
    first_z = np.where(z_ticks <= zlist[-1])[0][0]
    z_ticks = z_ticks[first_z:last_z+1]
    tick_pos = [z for z in time_func(z_ticks)]
    tick_labels = ['%.2f' % (z) for z in z_ticks]
    ax2.set_xlim(x0,x1)
    ax2.set_xticks(tick_pos)
    ax2.set_xticklabels(tick_labels)
    ax2.set_xlabel('Redshift', fontsize=18)
    ax.set_xlabel('Time [Gyr]', fontsize=18)

    z_sfr, sfr = np.loadtxt(code_path + 'halo_infos/00' + args.halo + '/' + args.run + '/sfr', unpack=True, usecols=[1,2], skiprows=1)
    t_sfr = time_func(z_sfr)

    '''ax3 = ax.twinx()
    ax3.plot(t_sfr, sfr, 'k:', lw=1)
    ax.plot([timelist[0],timelist[-1]], [0,0], 'k:', lw=1, label='SFR (right axis)')
    ax3.tick_params(axis='y', which='both', direction='in', length=8, width=2, pad=5, labelsize=20, right=True)
    ax3.set_ylim(-5,200)
    ax3.set_ylabel('SFR [$M_\odot$/yr]', fontsize=20)'''
    ax.plot(t_sfr, sfr, 'k--', lw=1, label='SFR')

    ax.plot([np.min(timelist), np.max(timelist)], [0,0], 'k-', lw=1)

    ax.legend(loc='upper center', fontsize=14, bbox_to_anchor=(0.5,-0.15), ncol=6)
    fig.subplots_adjust(left=0.11, bottom=0.2, right=0.98, top=0.89)
    fig.savefig(prefix + 'accretion_vs_time' + save_suffix + '.png')
    plt.close(fig)

def accretion_compare_vs_radius(snap):
    '''Plots various properties of accretion as a function of radius at the snapshot given by 'snap'.'''

    tablename_prefix = output_dir + 'stats_halo_00' + args.halo + '/' + args.run + '/Tables/'
    save_prefix = output_dir + 'stats_halo_00' + args.halo + '/' + args.run + '/Plots/'
    time_table = Table.read(output_dir + 'times_halo_00' + args.halo + '/' + args.run + '/time_table.hdf5', path='all_data')
    zsnap = time_table['redshift'][time_table['snap']==snap][0]
    tsnap = time_table['time'][time_table['snap']==snap][0]

    data = Table.read(tablename_prefix + snap + '_' + args.load_from_file + '.hdf5', path='all_data')
    if ('phi_bin' in data.columns):
        radii = data['radius'][data['phi_bin']=='all']
    else:
        radii = data['radius']

    props = ['covering_fraction','mass','metal_mass','temperature','metallicity',
            'cooling_time','entropy','pressure','radial_velocity','Mach',#'velocity_dispersion',
            'energy']
    ranges = [[0,1], [2e5,3e7], [0,0.25], [1e4,3e6], [5e-3,1],
             [1e2,1e6], [1,1e3], [1e-17,5e-12], [-300, 200], [0, 15], #[0,100],
             [5e12,5e15]]
    logs = [False, True, False, True, True, True, True, True, False, False, #False,
            True]
    ylabels = ['Accretion Covering Fraction', 'Filament Mass [$M_\odot$]', 'Accretion Metal Mass Fraction',
              'Temperature [K]', 'Metallicity [$Z_\odot$]', 'Cooling Time [Myr]', 'Entropy [keV cm$^2$]',
              'Pressure [erg/cm$^3$]', 'Radial Velocity [km/s]', 'Accretion Mach number', #'Velocity dispersion [km/s]',
              'Energy [erg/g]']

    if ('phi_bin' in data.columns):
        all = (data['phi_bin']=='all')
        major = (data['phi_bin']=='major')
        minor = (data['phi_bin']=='minor')
        directions = [all, major, minor]
        dir_labels = ['All directions', 'Major axis', 'Minor axis']
        dir_colors = ['k','b','r']
        if (not args.direction):
            directions = [all]
    else:
        all = np.ones(len(radii), dtype=bool)
        directions = [all]
        dir_labels = ['_nolegend_']
        dir_colors = ['k']

    if (args.region_filter!='none'):
        region_file = ['', 'low_', 'mid_']
        #region_file = ['low_']
        if (args.region_filter=='temperature'):
            region_labels = ['All temperatures', '$T<10^{4.9}$ K', '$10^{4.9}$ K $<T<10^{5.5}$ K', '$T>10^{5.5}$ K']
            region_colors = ['k', "#984ea3", "#4daf4a", "#ffe34d"]
        if (args.region_filter=='metallicity'):
            region_labels = ['All metallicities', '$Z<10^{-2}Z_\odot$', '$10^{-2}Z_\odot < Z < 10^{-1}Z_\odot$', '$Z>10^{-1}Z_\odot$']
            region_colors = ['k',"#4575b4", "#984ea3", "#d73027"]
            #region_labels = ['Filaments']
            #region_colors = ['k']
    else:
        region_file = ['']

    for i in range(len(props)):
        fig = plt.figure(figsize=(7.5,5), dpi=200)
        ax = fig.add_subplot(1,1,1)
        for j in range(len(directions)):
            for k in range(len(region_file)):
                if (args.direction):
                    color = dir_colors[j]
                    label = dir_labels[j]
                elif (args.region_filter!='none'):
                    color = region_colors[k]
                    label = region_labels[k]
                if (props[i]=='covering_fraction'):
                    labels = ['_nolegend_', '_nolegend_']
                    mults = [1,0.5,0.5]
                    acc_plot = mults[j]*data[region_file[k] + 'covering_fraction_acc'][directions[j]]
                elif ('mass' in props[i]):
                    #labels = ['_nolegend_', '_nolegend_']
                    label = '_nolegend_'
                    #mults = [1,0.5,0.5]
                    acc_plot = data[region_file[k] + props[i] + '_acc'][directions[j]]
                    print(np.sum(acc_plot))
                elif (props[i]=='energy'):
                    if (k==0): labels = ['Radial kinetic energy', 'Turbulent kinetic energy', 'Thermal energy']
                    else: labels = ['_nolegend_','_nolegend_','_nolegend_']
                    e_props = ['radial_kinetic_energy', 'turbulent_kinetic_energy','thermal_energy']
                    linestyles = ['-','--',':']
                    if (k>-1):
                        for l in range(len(e_props)):
                            ax.plot(radii, data[region_file[k] + e_props[l] + '_acc'][directions[j]]/data[region_file[k] + 'mass_acc'][directions[j]]/gtoMsun,
                                    color=color, ls=linestyles[l], lw=2, label=labels[l])
                            if (k==0): ax.plot(radii[1:], data[e_props[l] + '_non'][directions[j]][1:]/data['mass_non'][directions[j]][1:]/gtoMsun,
                                    color='darkorange', ls=linestyles[l], lw=2, label='_nolabel_')
                        ax.plot([-100,-100],[-100,-100], color=color, ls='-', lw=2, label=region_labels[k])
                        if (k==1): ax.plot([-100,-100],[-100,-100], color='darkorange', ls='-', lw=2, label='rest of CGM')
                elif (props[i]=='Mach'):
                    labels = ['_nolegend_', '_nolegend_']
                    acc_plot = -data[region_file[k] + 'radial_velocity_med_acc'][directions[j]]/data[region_file[k] + 'sound_speed_med_acc'][directions[j]]
                else:
                    labels = ['Accreting gas', 'Non-accreting gas']
                    acc_plot = data[region_file[k] + props[i] + '_med_acc'][directions[j]]
                if (props[i]!='energy'):
                    ax.plot(radii, acc_plot, color=color, ls='-', lw=2, label=label)
                if ('fraction' not in props[i]) and ('mass' not in props[i]) and (props[i]!='Mach') and ('energy' not in props[i]):
                    #ax.fill_between(radii, data[props[i] + '_med_acc'][directions[j]]-0.5*data[props[i] + '_iqr_acc'][directions[j]],
                                    #data[props[i] + '_med_acc'][directions[j]]+0.5*data[props[i] + '_iqr_acc'][directions[j]],
                                    #color=dir_colors[j], alpha=0.3)
                    ax.plot(radii, data[region_file[k] + props[i] + '_med_non'][directions[j]], color=color, ls=':', lw=2, label='_nolegend_')
                    #if (k==0): ax.plot(radii, data[props[i] + '_med_non'][directions[j]], color="darkorange", ls=':', lw=2, label='Rest of CGM')
                    #ax.fill_between(radii, data[props[i] + '_med_all'][directions[j]]-0.5*data[props[i] + '_iqr_all'][directions[j]],
                                    #data[props[i] + '_med_all'][directions[j]]+0.5*data[props[i] + '_iqr_all'][directions[j]],
                                    #color=dir_colors[j], alpha=0.3)
                    if (props[i]=='radial_velocity'):
                        masses = Table.read(catalog_dir + 'masses_z-less-2.hdf5', path='all_data')
                        masses_ind = np.where(masses['snapshot']==snap)[0]
                        Menc_profile = IUS(np.concatenate(([0],masses['radius'][masses_ind])), np.concatenate(([0],masses['total_mass'][masses_ind])))
                        vff = -np.sqrt((2.*G*Menc_profile(radii)*gtoMsun)/(radii*1000.*cmtopc))/1e5
                        if (j==2) or (k==2): ax.plot(radii, vff, 'k--', lw=2, label='Free fall velocity')
                    if (props[i]=='temperature'):
                        start_T = data[region_file[k] + 'temperature_med_acc'][directions[j]][-1]
                        #volume = data[region_file[k] + 'covering_fraction_acc'][directions[j]][1:]*(4.*np.pi*(radii[1:]*1000.*cmtopc)**2.)*np.diff(radii*1000*cmtopc)
                        pressure = data[region_file[k] + 'pressure_med_acc'][directions[j]]
                        #PdV = pressure[2:]*np.diff(volume)
                        #compression_T = PdV/kB + start_T
                        compression_T = start_T*(pressure/pressure[-1])**(2./5.)
                        if (j==2) or (k==0): comp_label = 'Adiabatic compression'
                        else: comp_label = '_nolegend_'
                        ax.plot(radii, compression_T, color=region_colors[k], ls='--', lw=2, label=comp_label)
                    if (props[i]=='cooling_time'):
                        ax.plot([0,250], [13.76e3,13.76e3], 'k--', lw=1)
            #if (j==len(directions)-1):
                #ax.plot([-100,-100],[-100,-100], color='k', ls='-', lw=2, label=labels[0])
                #ax.plot([-100,-100],[-100,-100], color='k', ls=':', lw=2, label=labels[1])

        ax.axis([0,250,ranges[i][0],ranges[i][1]])
        ax.set_xlabel('Galactocentric Radius [kpc]', fontsize=16)
        ax.set_ylabel(ylabels[i], fontsize=16)
        if (logs[i]): ax.set_yscale('log')
        ax.tick_params(axis='both', which='both', direction='in', length=8, width=2, pad=5, labelsize=14)
        if ('mass' in props[i]) or ('fraction' in props[i]) or (props[i]=='Mach'):
            ax.legend(frameon=False, loc=2, fontsize=14)
            ax.text(0.95, 0.9, '%.2f Gyr\n$z=%.2f$' % (tsnap/1e3, zsnap), fontsize=14, ha='right', va='center', transform=ax.transAxes, bbox={'fc':'white','ec':'black','boxstyle':'round','lw':1})
        elif ('velocity_dispersion' in props[i]) or ('metallicity' in props[i]) or \
          ('temperature' in props[i]) or ('pressure' in props[i]) or ('energy' in props[i]):
            ax.legend(frameon=False, loc=2, fontsize=14, ncol=2)
            ax.text(0.95, 0.65, '%.2f Gyr\n$z=%.2f$' % (tsnap/1e3, zsnap), fontsize=14, ha='right', va='center', transform=ax.transAxes, bbox={'fc':'white','ec':'black','boxstyle':'round','lw':1})
        else:
            ax.legend(frameon=False, loc=2, fontsize=14, ncol=2)
            ax.text(0.95, 0.1, '%.2f Gyr\n$z=%.2f$' % (tsnap/1e3, zsnap), fontsize=14, ha='right', va='center', transform=ax.transAxes, bbox={'fc':'white','ec':'black','boxstyle':'round','lw':1})

        fig.subplots_adjust(left=0.13,bottom=0.14,top=0.95,right=0.97)
        plt.savefig(save_prefix + snap + '_accretion-compare_vs_radius_' + props[i] + save_suffix + '.png')
        plt.close()

def accretion_flux_vs_radius(snap):
    '''Plots accretion flux as a function of radius at the snapshot given by 'snap'.'''

    tablename_prefix = output_dir + 'fluxes_halo_00' + args.halo + '/' + args.run + '/Tables/'
    save_prefix = output_dir + 'fluxes_halo_00' + args.halo + '/' + args.run + '/Plots/'
    time_table = Table.read(output_dir + 'times_halo_00' + args.halo + '/' + args.run + '/time_table.hdf5', path='all_data')
    zsnap = time_table['redshift'][time_table['snap']==snap][0]
    tsnap = time_table['time'][time_table['snap']==snap][0]

    data = Table.read(tablename_prefix + snap + '_' + args.load_from_file + '.hdf5', path='all_data')
    if ('phi_bin' in data.columns):
        radii = data['radius'][data['phi_bin']=='all']
    else:
        radii = data['radius']

    if (args.region_filter=='temperature'):
        plot_colors = ['salmon', "#984ea3", "#4daf4a", 'darkorange', 'k']
        region_label = ['$<10^4$ K', '$10^4-10^5$ K', '$10^5-10^6$ K', '$>10^6$ K', 'All temperatures']
        region_name = ['lowest_', 'low-mid_', 'high-mid_', 'highest_']
    elif (args.region_filter=='metallicity'):
        plot_colors = ["#4575b4", "#984ea3", "#d73027", "darkorange", 'k']
        region_label = ['$<0.1Z_\odot$', '$0.1-0.5Z_\odot$', '$0.5-1Z_\odot$', '$>Z_\odot$', 'All metallicities']
        region_name = ['lowest_', 'low-mid_', 'high-mid_', 'highest_']
    else:
        plot_colors = ['k']
        region_label = ['_nolegend_']
        region_name = ['']

    fluxes = ['mass','metal']
    ranges = [[1e-4,1e3],[1e-6,5]]
    ylabels = ['Mass Flux [$M_\odot$/yr]', 'Metal Mass Flux [$M_\odot$/yr]']

    if ('phi_bin' in data.columns):
        all = (data['phi_bin']=='all')
        major = (data['phi_bin']=='major')
        minor = (data['phi_bin']=='minor')
        directions = [all, major, minor]
        dir_labels = ['All directions', 'Major axis', 'Minor axis']
        dir_ls = ['-','--',':']
    else:
        all = np.ones(len(radii), dtype=bool)
        directions = [all]
        dir_labels = ['_nolegend_']
        dir_ls = ['-']

    for i in range(len(fluxes)):
        fig = plt.figure(figsize=(7,5), dpi=300)
        ax = fig.add_subplot(1,1,1)
        for j in range(len(directions)):
            flux_sum = np.zeros(len(radii))
            for k in range(len(region_label)):
                if (j==0): label=region_label[k]
                else: label='_nolegend_'
                if (region_label[k][:3]!='All'):
                    plot_flux = data[region_name[k]+fluxes[i]+'_flux_in'][directions[j]]
                    flux_sum += plot_flux
                else:
                    plot_flux = flux_sum
                ax.plot(radii, plot_flux, color=plot_colors[k], ls=dir_ls[j], lw=2, label=label)
            ax.plot([-100,-100],[-100,-100], color='k', ls=dir_ls[j], lw=2, label=dir_labels[j])

        ax.axis([0,250,ranges[i][0],ranges[i][1]])
        ax.set_xlabel('Galactocentric Radius [kpc]', fontsize=16)
        ax.set_ylabel(ylabels[i], fontsize=16)
        ax.set_yscale('log')
        ax.tick_params(axis='both', which='both', direction='in', length=8, width=2, pad=5, labelsize=14)
        ax.legend(frameon=False, loc=1, fontsize=14, ncol=2)
        ax.text(0.05, 0.9, '%.2f Gyr\n$z=%.2f$' % (tsnap/1e3, zsnap), fontsize=14, ha='left', va='center', transform=ax.transAxes, bbox={'fc':'white','ec':'black','boxstyle':'round','lw':1})

        fig.subplots_adjust(left=0.13,bottom=0.14,top=0.95,right=0.97)
        plt.savefig(save_prefix + snap + '_accretion-flux_vs_radius_' + fluxes[i] + '.png')

def streamlines_over_time(snaplist):
    '''Uses yt's streamlines function to integrate paths through the velocity field at each time step, using
    the previous snapshot's integration as input for the next snapshot's integration. Saves to file the positions
    and properties at each position of each streamline.'''

    from yt.visualization.api import Streamlines

    # Find starting locations of streams in first snapshot
    snap = snaplist[0]
    snap_name = foggie_dir + run_dir + snap + '/' + snap
    ds, refine_box = foggie_load(snap_name, trackname, do_filter_particles=False, halo_c_v_name=halo_c_v_name, gravity=True, masses_dir=catalog_dir, correct_bulk_velocity=True)
    # Set up covering grid
    pix_res = float(np.min(refine_box[('gas','dx')].in_units('kpc')))
    lvl1_res = pix_res*2.**11.
    level = 9
    dx = lvl1_res/(2.**level)
    box_size = 400
    refine_res = int(box_size/dx)
    box = ds.covering_grid(level=level, left_edge=ds.halo_center_kpc-ds.arr([box_size/2.,box_size/2.,box_size/2.],'kpc'), dims=[refine_res, refine_res, refine_res])
    x = box['gas', 'x'].in_units('kpc').v - ds.halo_center_kpc[0].v
    y = box['gas', 'y'].in_units('kpc').v - ds.halo_center_kpc[1].v
    z = box['gas', 'z'].in_units('kpc').v - ds.halo_center_kpc[2].v
    xbins = x[:,0,0][:-1] - 0.5*np.diff(x[:,0,0])
    ybins = y[0,:,0][:-1] - 0.5*np.diff(y[0,:,0])
    zbins = z[0,0,:][:-1] - 0.5*np.diff(z[0,0,:])
    temperature = box['gas','temperature'].in_units('K').v
    vx = box['gas','vx_corrected'].in_units('km/s').v
    vy = box['gas','vy_corrected'].in_units('km/s').v
    vz = box['gas','vz_corrected'].in_units('km/s').v
    vmag = box['gas','vel_mag_corrected'].in_units('km/s').v
    density = box['gas','density'].in_units('g/cm**3.').v
    pressure = box['gas','pressure'].in_units('erg/cm**3').v
    radius = box['gas','radius_corrected'].in_units('kpc').v
    radial_velocity = box['gas','radial_velocity_corrected'].v
    vff = box['gas','vff'].in_units('km/s').v

    Nstreams = 400
    length = ds.quan(50.,'kpc')
    dx = ds.quan(dx, 'kpc')

    vff_shell = np.mean(vff[(radius > 95.) & (radius < 100.)])
    x_shell = x[(radius > 95.) & (radius < 100.) & (radial_velocity < 0.75*vff_shell)]
    y_shell = y[(radius > 95.) & (radius < 100.) & (radial_velocity < 0.75*vff_shell)]
    z_shell = z[(radius > 95.) & (radius < 100.) & (radial_velocity < 0.75*vff_shell)]
    inds = np.random.randint(len(x_shell), size=Nstreams)
    start_pos = np.transpose(np.array([x_shell[inds], y_shell[inds], z_shell[inds]]))

    data = dict(x = (x, 'kpc'), y = (y, 'kpc'), z = (z, 'kpc'), \
                temperature = (temperature, "K"), density = (density, 'K'), pressure = (pressure, 'erg/cm**3'), \
                vx = (vx, 'km/s'), vy = (vy, 'km/s'), vz = (vz, 'km/s'), vmag = (vmag, 'km/s'))
    bbox = np.array([[np.min(x), np.max(x)], [np.min(y), np.max(y)], [np.min(z), np.max(z)]])
    new_ds = yt.load_uniform_grid(data, x.shape, length_unit="kpc", bbox=bbox)

    for s in range(len(snaplist)):
        # Make table for saving stream positions to file
        names_list = ['stream_id','elapsed_time','z_ind','y_ind','x_ind','x_pos','y_pos','z_pos','vx','vy','vz','density','temperature','pressure']
        types_list = ['f8']*14
        stream_table = Table(names=names_list, dtype=types_list)
        stream_table_units = ['none','Myr','none','none','none','kpc','kpc','kpc','km/s','km/s','km/s','g/cm**3','K','erg/cm**3']
        for i in range(len(names_list)):
            stream_table[names_list[i]].unit = stream_table_units[i]
        print('Defining streamlines for %s' % (snap), str(datetime.datetime.now()))
        streamlines = Streamlines(new_ds, start_pos, 'vx', 'vy', 'vz', length=length, dx=dx)
        print('Streamlines defined for %s' % (snap), str(datetime.datetime.now()))
        streamlines.integrate_through_volume()
        # Calculate information along each stream and save
        next_start_pos = []
        for i in range(len(start_pos)):
            stream = streamlines.path(i)
            stream_path = stream.positions.in_units('kpc').v      # stream_path[j] is (x,y,z) position in the box of jth location along the path
            stream_path_T = np.transpose(stream_path)
            stream_path_x = stream_path_T[0]
            stream_path_y = stream_path_T[1]
            stream_path_z = stream_path_T[2]
            displacements = np.sqrt((np.diff(stream_path_x))**2. + (np.diff(stream_path_y))**2. + (np.diff(stream_path_z))**2.)
            # Digitize positions along stream onto covering grid
            inds_x = np.digitize(stream_path_x, xbins)-1      # indices of new x positions
            inds_y = np.digitize(stream_path_y, ybins)-1      # indices of new y positions
            inds_z = np.digitize(stream_path_z, zbins)-1      # indices of new z positions
            stream_path_x_digi = x[inds_x,inds_y,inds_z]
            stream_path_y_digi = y[inds_x,inds_y,inds_z]
            stream_path_z_digi = z[inds_x,inds_y,inds_z]
            stream_path_vmag = vmag[inds_x,inds_y,inds_z]
            elapsed_time = np.cumsum((displacements*1000*cmtopc)/(stream_path_vmag[:-1]*1e5)/(stoyr))
            elapsed_time = np.insert(elapsed_time, 0, 0)
            end_ind = np.where(elapsed_time>=(5.*dt))[0][0]
            inds_x = inds_x[:end_ind]
            inds_y = inds_y[:end_ind]
            inds_z = inds_z[:end_ind]
            vx_path = vx[inds_x,inds_y,inds_z]
            vy_path = vy[inds_x,inds_y,inds_z]
            vz_path = vz[inds_x,inds_y,inds_z]
            den_path = density[inds_x,inds_y,inds_z]
            temp_path = temperature[inds_x,inds_y,inds_z]
            pres_path = pressure[inds_x,inds_y,inds_z]
            ids = np.zeros(len(inds_x)) + i
            track = np.array([ids, elapsed_time[:end_ind], inds_z, inds_y, inds_x, stream_path_x_digi[:end_ind], stream_path_y_digi[:end_ind], stream_path_z_digi[:end_ind], vx_path, vy_path, vz_path, den_path, temp_path, pres_path])
            track = np.transpose(track)
            for t in range(len(track)):
                stream_table.add_row(track[t])
            next_start_pos.append(np.array([stream_path_x[end_ind-1], stream_path_y[end_ind-1], stream_path_z[end_ind-1]]))

        tablename = prefix + 'Tables/' + snap + '_streams'
        stream_table.write(tablename + save_suffix + '.hdf5', path='all_data', serialize_meta=True, overwrite=True)
        print('Stream properties saved for %s' % (snap), str(datetime.datetime.now()))
        next_start_pos = np.transpose(np.array(next_start_pos))

        # Load next snapshot
        if (s<len(snaplist)-1):
            snap = snaplist[s+1]
            snap_name = foggie_dir + run_dir + snap + '/' + snap
            ds, refine_box = foggie_load(snap_name, trackname, do_filter_particles=False, halo_c_v_name=halo_c_v_name, gravity=False, masses_dir=catalog_dir, correct_bulk_velocity=True)
            # Set up covering grid
            pix_res = float(np.min(refine_box[('gas','dx')].in_units('kpc')))
            lvl1_res = pix_res*2.**11.
            level = 9
            dx = lvl1_res/(2.**level)
            box_size = 400
            refine_res = int(box_size/dx)
            box = ds.covering_grid(level=level, left_edge=ds.halo_center_kpc-ds.arr([box_size/2.,box_size/2.,box_size/2.],'kpc'), dims=[refine_res, refine_res, refine_res])
            x = box['gas', 'x'].in_units('kpc').v - ds.halo_center_kpc[0].v
            y = box['gas', 'y'].in_units('kpc').v - ds.halo_center_kpc[1].v
            z = box['gas', 'z'].in_units('kpc').v - ds.halo_center_kpc[2].v
            xbins = x[:,0,0][:-1] - 0.5*np.diff(x[:,0,0])
            ybins = y[0,:,0][:-1] - 0.5*np.diff(y[0,:,0])
            zbins = z[0,0,:][:-1] - 0.5*np.diff(z[0,0,:])
            vmag = box['gas','vel_mag_corrected'].in_units('km/s').v
            temperature = box['gas','temperature'].in_units('K').v
            vx = box['gas','vx_corrected'].in_units('km/s').v
            vy = box['gas','vy_corrected'].in_units('km/s').v
            vz = box['gas','vz_corrected'].in_units('km/s').v
            density = box['gas','density'].in_units('g/cm**3.').v
            pressure = box['gas','pressure'].in_units('erg/cm**3').v

            data = dict(x = (x, 'kpc'), y = (y, 'kpc'), z = (z, 'kpc'), \
                        temperature = (temperature, "K"), density = (density, 'K'), pressure = (pressure, 'erg/cm**3'), \
                        vx = (vx, 'km/s'), vy = (vy, 'km/s'), vz = (vz, 'km/s'), vmag = (vmag, 'km/s'))
            bbox = np.array([[np.min(x), np.max(x)], [np.min(y), np.max(y)], [np.min(z), np.max(z)]])
            new_ds = yt.load_uniform_grid(data, x.shape, length_unit="kpc", bbox=bbox)

            # Digitize starting position onto new grid
            inds_x = np.digitize(next_start_pos[0], xbins)-1      # indices of new x positions
            inds_y = np.digitize(next_start_pos[1], ybins)-1      # indices of new y positions
            inds_z = np.digitize(next_start_pos[2], zbins)-1      # indices of new z positions
            start_x = x[inds_x,inds_y,inds_z]
            start_y = y[inds_x,inds_y,inds_z]
            start_z = z[inds_x,inds_y,inds_z]
            start_pos = np.transpose(np.array([start_x, start_y, start_z]))

            length = ds.quan(50.,'kpc')
            dx = ds.quan(dx, 'kpc')

def plot_streamlines(snap):
    '''Saves x, y, and z projections of gas density with stream lines found with streamlines_over_time.
    Streamlines must have already been found and saved to file before this function can be called to plot them.'''

    from yt.units import kpc

    # Load simulation output
    if (args.system=='pleiades_cassi'):
        print('Copying directory to /tmp')
        if (args.copy_to_tmp):
            snap_dir = '/tmp/' + args.halo + '/' + args.run + '/' + target_dir + '/' + snap
            shutil.copytree(foggie_dir + run_dir + snap, snap_dir)
            snap_name = snap_dir + '/' + snap
        else:
            # Make a dummy directory with the snap name so the script later knows the process running
            # this snapshot failed if the directory is still there
            snap_dir = '/nobackup/clochhaa/tmp/' + args.halo + '/' + args.run + '/' + target_dir + '/' + snap
            os.makedirs(snap_dir)
            snap_name = foggie_dir + run_dir + snap + '/' + snap
    else:
        snap_name = foggie_dir + run_dir + snap + '/' + snap
    ds, refine_box = foggie_load(snap_name, trackname, do_filter_particles=False, halo_c_v_name=halo_c_v_name, gravity=False, masses_dir=catalog_dir, correct_bulk_velocity=True)
    sph = ds.sphere(center=ds.halo_center_kpc, radius=(200., 'kpc'))

    Nstreams = 400
    snap_ind = outs.index(snap)
    stream_snaps = []
    stream_x = [[] for x in range(Nstreams)]
    stream_y = [[] for x in range(Nstreams)]
    stream_z = [[] for x in range(Nstreams)]
    for i in range(3):
        if (snap_ind-i>=0):
            stream_snaps.append(outs[snap_ind-i])
    stream_snaps.reverse()
    for i in range(len(stream_snaps)):
        tablename = prefix + 'Tables/' + stream_snaps[i] + '_streams'
        streams = Table.read(tablename + save_suffix + '.hdf5', path='all_data')
        for s in range(Nstreams):
            for p in range(len(streams['x_pos'][streams['stream_id']==s])):
                stream_x[s].append(streams['x_pos'][streams['stream_id']==s][p] * kpc)
                stream_y[s].append(streams['y_pos'][streams['stream_id']==s][p] * kpc)
                stream_z[s].append(streams['z_pos'][streams['stream_id']==s][p] * kpc)

    for d in ['x','y','z']:
        # Make projection plot as usual
        proj = yt.ProjectionPlot(ds, d, 'density', data_source=sph, center=ds.halo_center_kpc, width=(400., 'kpc'))
        proj.set_log('density', True)
        proj.set_unit('density','Msun/kpc**2')
        proj.set_cmap('density', density_color_map)
        proj.set_zlim('density', 5e3, 2e8)
        proj.set_font_size(20)
        proj.annotate_timestamp(corner='upper_left', redshift=True, time=True, draw_inset_box=True)

        # Overplot streamlines
        for s in range(Nstreams):
            for i in range(len(stream_x[s])-1):
                if (d=='x'):
                    proj.annotate_line((stream_y[s][i],stream_z[s][i]), (stream_y[s][i+1],stream_z[s][i+1]), coord_system='plot', plot_args={'color':'white', 'linewidth':1})
                if (d=='y'):
                    proj.annotate_line((stream_z[s][i],stream_x[s][i]), (stream_z[s][i+1],stream_x[s][i+1]), coord_system='plot', plot_args={'color':'white', 'linewidth':1})
                if (d=='z'):
                    proj.annotate_line((stream_x[s][i],stream_y[s][i]), (stream_x[s][i+1],stream_y[s][i+1]), coord_system='plot', plot_args={'color':'white', 'linewidth':1})

        proj.save(prefix + 'Plots/' + snap + '_Projection_' + d + '_density_streamlines.png')

        # Make projection plot as usual
        proj = yt.ProjectionPlot(ds, d, 'temperature', data_source=sph, center=ds.halo_center_kpc, width=(400., 'kpc'), weight_field='density')
        proj.set_log('temperature', True)
        proj.set_cmap('temperature', sns.blend_palette(('salmon', "#984ea3", "#4daf4a", "#ffe34d", 'darkorange'), as_cmap=True))
        proj.set_zlim('temperature', 1e4, 1e7)
        proj.set_font_size(20)
        proj.annotate_timestamp(corner='upper_left', redshift=True, time=True, draw_inset_box=True)

        # Overplot streamlines
        for s in range(Nstreams):
            for i in range(len(stream_x[s])-1):
                if (d=='x'):
                    proj.annotate_line((stream_y[s][i],stream_z[s][i]), (stream_y[s][i+1],stream_z[s][i+1]), coord_system='plot', plot_args={'color':'white', 'linewidth':1})
                if (d=='y'):
                    proj.annotate_line((stream_z[s][i],stream_x[s][i]), (stream_z[s][i+1],stream_x[s][i+1]), coord_system='plot', plot_args={'color':'white', 'linewidth':1})
                if (d=='z'):
                    proj.annotate_line((stream_x[s][i],stream_y[s][i]), (stream_x[s][i+1],stream_y[s][i+1]), coord_system='plot', plot_args={'color':'white', 'linewidth':1})

        proj.save(prefix + 'Plots/' + snap + '_Projection_' + d + '_temperature_streamlines.png')

    # Delete output from temp directory if on pleiades
    if (args.system=='pleiades_cassi'):
        print('Deleting directory from /tmp')
        shutil.rmtree(snap_dir)


if __name__ == "__main__":

    gtoMsun = 1.989e33
    cmtopc = 3.086e18
    stoyr = 3.155e7
    G = 6.673e-8
    kB = 1.38e-16
    mu = 0.6
    mp = 1.67e-24
    dt = 5.38e6

    args = parse_args()
    print(args.halo)
    print(args.run)
    print(args.system)
    foggie_dir, output_dir, run_dir, code_path, trackname, haloname, spectra_dir, infofile = get_run_loc_etc(args)
    #foggie_dir = '/Volumes/Data/Simulation_Data/'

    # Set directory for output location, making it if necessary
    prefix = output_dir + 'fluxes_halo_00' + args.halo + '/' + args.run + '/'
    if not (os.path.exists(prefix)): os.system('mkdir -p ' + prefix)

    print('foggie_dir: ', foggie_dir)
    catalog_dir = code_path + 'halo_infos/00' + args.halo + '/' + args.run + '/'
    halo_c_v_name = catalog_dir + 'halo_c_v'
    smooth_AM_name = catalog_dir + 'AM_direction_smoothed'

    if (args.save_suffix!=''):
        save_suffix = '_' + args.save_suffix
    else:
        save_suffix = ''

    # Build flux type list
    if (',' in args.flux_type):
        flux_types = args.flux_type.split(',')
    else:
        flux_types = [args.flux_type]

    # Build plots list
    if (',' in args.plot):
        plots = args.plot.split(',')
    else:
        plots = [args.plot]

    surface = ast.literal_eval(args.surface)
    outs = make_output_list(args.output, output_step=args.output_step)

    if (args.load_from_file!='none'):
        if ('accretion_vs_time' in plots):
            accretion_flux_vs_time(outs)
        if ('accretion_vs_radius' in plots):
            for i in range(len(outs)):
                accretion_compare_vs_radius(outs[i])
        if ('flux_vs_radius' in plots):
            for i in range(len(outs)):
                accretion_flux_vs_radius(outs[i])

    else:
        if (save_suffix != ''):
            target_dir = save_suffix
        else:
            target_dir = 'fluxes'
        if (args.nproc==1):
            if (args.streamlines):
                streamlines_over_time(outs)
            elif (args.plot=='streamlines'):
                for snap in outs:
                    plot_streamlines(snap)
            else:
                for snap in outs:
                    load_and_calculate(snap, surface)
        else:
            skipped_outs = outs
            while (len(skipped_outs)>0):
                skipped_outs = []
                # Split into a number of groupings equal to the number of processors
                # and run one process per processor
                for i in range(len(outs)//args.nproc):
                    threads = []
                    snaps = []
                    for j in range(args.nproc):
                        snap = outs[args.nproc*i+j]
                        snaps.append(snap)
                        if (args.plot=='streamlines'):
                            threads.append(multi.Process(target=plot_streamlines, args=[snap]))
                        else:
                            threads.append(multi.Process(target=load_and_calculate, args=[snap, surface]))
                    for t in threads:
                        t.start()
                    for t in threads:
                        t.join()
                    # Delete leftover outputs from failed processes from tmp directory if on pleiades
                    if (args.system=='pleiades_cassi'):
                        if (args.copy_to_tmp):
                            snap_dir = '/tmp/' + args.halo + '/' + args.run + '/' + target_dir + '/'
                        else:
                            snap_dir = '/nobackup/clochhaa/tmp/' + args.halo + '/' + args.run + '/' + target_dir + '/'
                        for s in range(len(snaps)):
                            if (os.path.exists(snap_dir + snaps[s])):
                                print('Deleting failed %s from /tmp' % (snaps[s]))
                                skipped_outs.append(snaps[s])
                                shutil.rmtree(snap_dir + snaps[s])
                # For any leftover snapshots, run one per processor
                threads = []
                snaps = []
                for j in range(len(outs)%args.nproc):
                    snap = outs[-(j+1)]
                    snaps.append(snap)
                    if (args.plot=='streamlines'):
                        threads.append(multi.Process(target=plot_streamlines, args=[snap]))
                    else:
                        threads.append(multi.Process(target=load_and_calculate, args=[snap, surface]))
                for t in threads:
                    t.start()
                for t in threads:
                    t.join()
                # Delete leftover outputs from failed processes from tmp directory if on pleiades
                if (args.system=='pleiades_cassi'):
                    if (args.copy_to_tmp):
                        snap_dir = '/tmp/' + args.halo + '/' + args.run + '/' + target_dir + '/'
                    else:
                        snap_dir = '/nobackup/clochhaa/tmp/' + args.halo + '/' + args.run + '/' + target_dir + '/'
                    for s in range(len(snaps)):
                        if (os.path.exists(snap_dir + snaps[s])):
                            print('Deleting failed %s from /tmp' % (snaps[s]))
                            skipped_outs.append(snaps[s])
                            shutil.rmtree(snap_dir + snaps[s])
                outs = skipped_outs

    print(str(datetime.datetime.now()))
    print("All snapshots finished!")
