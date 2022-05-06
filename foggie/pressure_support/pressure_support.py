"""
Filename: pressure_support.py
Author: Cassi
Date created: 4-30-21
Date last modified: 5-11-21
This file takes command line arguments and plots the various types of pressure support against gravity.
It requires the files made by stats_in_shells.py.

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
from scipy.interpolate import RegularGridInterpolator
import shutil
import ast
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from scipy.ndimage import rotate
from scipy.ndimage import uniform_filter1d
import scipy.ndimage as ndimage
from scipy.interpolate import LinearNDInterpolator
from scipy.interpolate import NearestNDInterpolator
from astropy.convolution import Gaussian1DKernel
from astropy.convolution import Gaussian2DKernel
from astropy.convolution import CustomKernel
from astropy.convolution import interpolate_replace_nans
from astropy.convolution import convolve_fft
import copy
import matplotlib.colors as colors
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
                        + ' and the default output is DD2427) or specify a range of outputs ' + \
                        'using commas to list individual outputs and dashes for ranges of outputs ' + \
                        '(e.g. "RD0020-RD0025" or "DD1341,DD1353,DD1600-DD1700", no spaces!)')
    parser.set_defaults(output='DD2427')

    parser.add_argument('--output_step', metavar='output_step', type=int, action='store', \
                        help='If you want to do every Nth output, this specifies N. Default: 1 (every output in specified range)')
    parser.set_defaults(output_step=1)

    parser.add_argument('--system', metavar='system', type=str, action='store', \
                        help='Which system are you on? Default is cassiopeia')
    parser.set_defaults(system='cassiopeia')

    parser.add_argument('--pwd', dest='pwd', action='store_true',
                        help='Just use the working directory?, Default is no')
    parser.set_defaults(pwd=False)

    parser.add_argument('--filename', metavar='filename', type=str, action='store', \
                        help='What is the name of the file (after the snapshot name) to pull pressures,\n' + \
                        'densities, gravitational potential, and velocity distributions from?\n' + \
                        'There is no default for this, you must specify a filename, unless you are plotting\n' + \
                        'a datashader plot.')

    parser.add_argument('--save_suffix', metavar='save_suffix', type=str, action='store', \
                        help='Do you want to append a string onto the names of the saved files? Default is no.')
    parser.set_defaults(save_suffix="")

    parser.add_argument('--plot', metavar='plot', type=str, action='store', \
                        help='What plot do you want? Options are:\n' + \
                        'velocity_PDF           -  mass PDFs of 3D velocity distributions and best fits\n' + \
                        'metallicity_PDF        -  mass PDFs of metallicity distributions in inner and outer CGM\n' + \
                        'pressure_vs_time       -  pressures (thermal, turb, ram) at a specified radius over time\n' + \
                        'pressure_vs_radius     -  pressures (thermal, turb, ram) over radius\n' + \
                        'force_vs_radius        -  forces (thermal, turb, ram, rotation, gravity, total) over radius\n' + \
                        'force_vs_radius_time_averaged  -  forces over radius with shading showing time variation\n' + \
                        'force_vs_radius_pres   -  forces (thermal, turb, ram) over radius, calculated from gradient of median pressure\n' + \
                        'force_vs_time          -  forces (thermal, turb, ram, rotation, gravity, total) over time\n' + \
                        'force_vs_energy_output -  2D plot of forces vs energy output from galaxy and radius\n' + \
                        'work_vs_time           -  work done by different forces over time\n' + \
                        'pressure_vs_r_shaded   -  pressure over radius or radial velocity for each cell as a datashader plot\n' + \
                        '       or                 For these options, specify what type of pressure to plot with the --pressure_type keyword\n' + \
                        'pressure_vs_rv_shaded     and what you want to color-code the points by with the --shader_color keyword\n' + \
                        'force_vs_r_shaded      -  force over radius or radial velocity for each cell as a datashader plot\n' + \
                        '       or                 For these options, specify what force to plot with the --force_type keyword\n' + \
                        'force_vs_rv_shaded        and what you want to color-code the points by with the --shader_color keyword\n' + \
                        'shader_force_colored   -  Datashader plot of --shader_y vs. --shader_x colored by forces or force ratios\n' + \
                        'support_vs_time        -  pressure support (thermal, turb, ram) relative to gravity at a specified radius over time\n' + \
                        'support_vs_radius      -  pressure support (thermal, turb, ram) relative to gravity over radius\n' + \
                        'support_vs_radius_time_averaged  -  support (forces/gravity) over radius with shading showing time variation\n' + \
                        'support_vs_r_shaded    -  pressure support relative to gravity over radius or radial velocity for each cell as a datashader plot\n' + \
                        '       or                 For these options, specify what type of support to plot with the --pressure_type keyword\n' + \
                        'support_vs_rv_shaded      and what you want to color-code the points by with the --shader_color keyword\n' + \
                        'pressure_slice         -  x-slices of different types of pressure (specify with --pressure_type keyword)\n' + \
                        'force_slice            -  x-slices of different forces (specify with --force_type keyword)\n' + \
                        'tangential_force_slice -  x-slices of the tangential component of different forces\n' + \
                        'force_ratio_slice      -  x-slice of ratio of thermal to turbulent force\n' + \
                        'support_slice          -  x-slices of different types of pressure support (specify with --pressure_type keyword)\n' + \
                        'velocity_slice         -  x-slices of the three spherical components of velocity, comparing the velocity,\n' + \
                        '                          the smoothed velocity, and the difference between the velocity and the smoothed velocity\n' + \
                        'vorticity_slice        -  x-slice of the velocity vorticity magnitude\n' + \
                        'ion_slice              -  x-slice of the ion mass of the ion given with the --ion keyword\n' + \
                        'vorticity_direction    -  2D histograms of vorticity direction split by temperature and radius\n' + \
                        'force_rays             -  Line plots of forces along rays from the halo center to various points\n' + \
                        'turbulent_spectrum     -  Turbulent energy power spectrum\n' + \
                        'turbulence_compare     -  Compares different ways of computing turbulent pressure\n' + \
                        'visualization          -  3D viewer of pressure fields using napari')

    parser.add_argument('--region_filter', metavar='region_filter', type=str, action='store', \
                        help='Do you want to show pressures in different regions? Options are:\n' + \
                        '"velocity", "temperature", or "metallicity". If plotting from a stats_in_shells file,\n' + \
                        'the files must have been created with the same specified region_filter.')
    parser.set_defaults(region_filter='none')

    parser.add_argument('--nproc', metavar='nproc', type=int, action='store', \
                        help='How many processes do you want? Default is 1 ' + \
                        '(no parallelization), if multiple outputs and multiple processors are' + \
                        ' specified, code will run one output per processor.\n' + \
                        'This option only has meaning for running things on multiple snapshots.')
    parser.set_defaults(nproc=1)

    parser.add_argument('--pdf_R', metavar='pdf_R', type=str, action='store', \
                        help='If plotting velocity PDFs, what radius do you want? Default is 50 kpc.')
    parser.set_defaults(pdf_R='50.')

    parser.add_argument('--spheres', metavar='spheres', type=str, action='store',
                        help='If you want to overplot quantities calculated in spheres for comparison\n' + \
                        'to the default of shells, use this to specify a filename for the spheres file.')
    parser.set_defaults(spheres='none')

    parser.add_argument('--pressure_type', metavar='pressure_type', type=str, action='store', \
                        help='If plotting pressures_vs_r_shaded or support_vs_r_shaded, what type of pressure do you\n' + \
                        'want to plot? Options are "thermal", "turbulent", "outflow", "inflow", "rotation",\n' + \
                        '"total", or "all", which will make one datashader plot per pressure type.\n' + \
                        'Default is "thermal".')
    parser.set_defaults(pressure_type='thermal')

    parser.add_argument('--force_type', metavar='pressure_type', type=str, action='store', \
                        help='If plotting forces_vs_r_shaded or force_slice, what forces do you\n' + \
                        'want to plot? Options are "thermal", "turbulent", "outflow", "inflow", "rotation",\n' + \
                        '"gravity", "total", or "all", which will make one datashader plot per force.\n' + \
                        'Default is "thermal".')
    parser.set_defaults(force_type='thermal')

    parser.add_argument('--ion', metavar='ion', type=str, action='store', \
                        help='If plotting ion_slice, what ion do you want to plot? Give in Trident syntax, i.e. O VI is O_p5.\n' + \
                        'Default is O_p5.')
    parser.set_defaults(ion='O_p5')

    parser.add_argument('--shader_color', metavar='shader_color', type=str, action='store', \
                        help='If plotting any of the datashader options, what do you want to color by?\n' + \
                        'Default is temperature.')
    parser.set_defaults(shader_color='temperature')

    parser.add_argument('--shader_x', metavar='shader_x', type=str, action='store', \
                        help='If plotting any of the datashader options, what do you want to plot on the x axis?\n' + \
                        'Default is radius.')
    parser.set_defaults(shader_x='radius')

    parser.add_argument('--shader_y', metavar='shader_y', type=str, action='store', \
                        help='If plotting any of the datashader options, what do you want to plot on the y axis?\n' + \
                        'Default is radial velocity.')
    parser.set_defaults(shader_y='radial_velocity')

    parser.add_argument('--load_stats', dest='load_stats', action='store_true', \
                        help='If plotting pressure_vs_radius, force_vs_radius, or support_vs_radius,\n' + \
                        'do you want to load from file for plotting? This requires the files you need\n' + \
                        'to already exist. Run first without this command to make the files.')
    parser.set_defaults(load_stats=False)

    parser.add_argument('--weight', metavar='weight', type=str, action='store', \
                        help='If plotting pressure_vs_radius, force_vs_radius, or support_vs_radius,\n' + \
                        'do you want to weight statistics by mass or volume? Default is mass.')
    parser.set_defaults(weight='mass')

    parser.add_argument('--cgm_only', dest='cgm_only', action='store_true', \
                        help='Do you want to compute pressures or forces using a density and temperature\n' + \
                        'cut to isolate the CGM? Default is no.')
    parser.set_defaults(cgm_only=False)

    parser.add_argument('--feedback_diff', dest='feedback_diff', action='store_true', \
                        help='Are you making plots for the different feedback strengths for the Tempest re-runs?\n' + \
                        'Default is no.')
    parser.set_defaults(feedback_diff=False)

    parser.add_argument('--radius', metavar='radius', type=float, action='store', \
                        help='If plotting pressures or forces over time, what radius do you want to plot at?\n' + \
                        'Give as fraction of Rvir. Default is 0.5Rvir.')
    parser.set_defaults(radius=0.5)

    parser.add_argument('--time_avg', metavar='radius', type=float, action='store', \
                        help='If plotting pressures or forces over time and you want to time-average, how long to average over?\n' + \
                        'Give in units of Myr. Default is not to time-average.')
    parser.set_defaults(time_avg=0)

    parser.add_argument('--radius_range', metavar='radius', type=str, action='store', \
                        help='If plotting pressures or forces over time, give a range of radii you want to average (pressure) or sum (force) over.\n' + \
                        'Give as fraction of Rvir, like "[0.25, 0.75]" (do not forget outer quotes!). Default is not to average or sum over a range in radius.')
    parser.set_defaults(radius_range="none")

    parser.add_argument('--normalized', dest='normalized', action='store_true', \
                        help='If plotting forces vs radius or time, do you want to normalize the forces\n' + \
                        'by mass (i.e., plot accelerations)? Default is not to do this.')
    parser.set_defaults(normalized=False)

    parser.add_argument('--smoothed', dest='smoothed', action='store_true', \
                        help='If plotting force slices, do you want to smooth the force field?\n' + \
                        'Default is not to do this.')
    parser.set_defaults(smoothed=False)

    parser.add_argument('--copy_to_tmp', dest='copy_to_tmp', action='store_true', \
                        help='If running on pleiades, do you want to copy simulation outputs too the\n' + \
                        '/tmp directory on the run node before doing calculations? This may speed up\n' + \
                        'run time and reduce weight on IO file system. Default is no.')
    parser.set_defaults(copy_to_tmp=False)

    args = parser.parse_args()
    return args

def gauss(x, mu, sig, A):
    return A*np.exp(-(x-mu)**2/2/sig**2)

def gauss_pos(x, mu, sig, A):
    func = A*np.exp(-(x-mu)**2/2/sig**2)
    func[x<=0.] = 0.
    return func

def gauss_neg(x, mu, sig, A):
    func = A*np.exp(-(x-mu)**2/2/sig**2)
    func[x>0.] = 0.
    return func

def double_gauss(x, mu1, sig1, A1, mu2, sig2, A2):
    func1 = A1*np.exp(-(x-mu1)**2/2/sig1**2)
    func2 = A2*np.exp(-(x-mu2)**2/2/sig2**2)
    func2[x<=0.] = 0.
    return func1 + func2

def triple_gauss(x, mu1, sig1, A1, mu2, sig2, A2, mu3, sig3, A3):
    func1 = A1*np.exp(-(x-mu1)**2/2/sig1**2)
    func2 = A2*np.exp(-(x-mu2)**2/2/sig2**2)
    func3 = A3*np.exp(-(x-mu3)**2/2/sig3**2)
    func2[x<=0.] = 0.
    func3[x>=0.] = 0.
    return func1 + func2 + func3

def Gaussian_convolve_3D(masked_arr, smooth_scale):
    '''Performs a 3D Gaussian convolution by doing a 2D convolution along each slice in each direction.
    Very slow for large arrays, but at least faster than a 3D convolution with astropy's convolve function
    (which can handle masked values whereas scipy's cannot).

    Requires:
    from astropy.convolution import Gaussian2DKernel
    from astropy.convolution import convolve_fft
    '''

    kernel_2d = Gaussian2DKernel(smooth_scale/2./np.sqrt(2.))       # Need sqrt(2) because each axis will be convolved twice

    smooth_arr_1 = []
    for j in range(len(masked_arr[:,0,0])):
        smooth_arr_1.append(convolve_fft(masked_arr[j,:,:], kernel_2d, preserve_nan=True, allow_huge=True))
    smooth_arr_1 = np.array(smooth_arr_1)
    smooth_arr_2 = []
    for j in range(len(masked_arr[0,:,0])):
        smooth_arr_2.append(convolve_fft(smooth_arr_1[:,j,:], kernel_2d, preserve_nan=True, allow_huge=True))
    smooth_arr_2 = np.array(smooth_arr_2)
    smooth_arr_2 = np.transpose(smooth_arr_2, axes=(1,0,2))
    smooth_arr = []
    for j in range(len(masked_arr[0,0,:])):
        smooth_arr.append(convolve_fft(smooth_arr_2[:,:,j], kernel_2d, preserve_nan=True, allow_huge=True))
    smooth_arr = np.array(smooth_arr)
    smooth_arr = np.transpose(smooth_arr, axes=(1,2,0))

    return smooth_arr

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

def set_table_units(table):
    '''Sets the units for the table. Note this needs to be updated whenever something is added to
    the table. Returns the table.'''

    for key in table.keys():
        if (key=='redshift'):
            table[key].unit = None
        elif ('radius' in key):
            table[key].unit = 'kpc'
        elif ('pdf' in key) and (args.weight=='mass'):
            table[key].unit = 'mass PDF'
        elif ('pdf' in key) and (args.weight=='volume'):
            table[key].unit = 'volume PDF'
        elif ('pressure' in key):
            table[key].unit = 'log erg/cm**3'
        elif ('force' in key):
            if ('sum' in key):
                table[key].unit = 'g*cm/s**2'
            else:
                table[key].unit = 'cm/s**2'
        elif ('support' in key):
            table[key].unit = 'dimensionless'
    return table

def make_table(stat_types):
    '''Makes the giant table that will be saved to file.'''

    names_list = ['redshift', 'inner_radius', 'outer_radius']
    types_list = ['f8', 'f8', 'f8']

    if (args.region_filter!='none'):
        regions_name = ['', 'low_' + args.region_filter + '_', 'mid_' + args.region_filter + '_', \
                        'high_' + args.region_filter + '_']
    else: regions_name = ['']
    if (args.plot=='force_vs_radius'):
        stat_names = ['_med', '_iqr', '_avg', '_std', '_sum', '_weight_sum']
    else:
        stat_names = ['_med', '_iqr', '_avg', '_std']
    for i in range(len(stat_types)):
        for k in range(len(regions_name)):
            for l in range(len(stat_names)):
                name = ''
                name += regions_name[k]
                name += stat_types[i]
                name += stat_names[l]
                names_list += [name]
                types_list += ['f8']

    table = Table(names=names_list, dtype=types_list)

    return table

def make_pdf_table(stat_types):
    '''Makes the giant table for the PDFs that will be saved to file.'''

    names_list = ['redshift', 'inner_radius', 'outer_radius']
    types_list = ['f8', 'f8', 'f8']

    if (args.region_filter!='none'):
        regions_name = ['', 'low_' + args.region_filter + '_', 'mid_' + args.region_filter + '_', \
                        'high_' + args.region_filter + '_']
    else: regions_name = ['']
    for i in range(len(stat_types)):
        names_list += ['lower_' + stat_types[i], 'upper_' + stat_types[i]]
        types_list += ['f8', 'f8']
        for k in range(len(regions_name)):
            name = ''
            name += regions_name[k]
            name += stat_types[i]
            names_list += [name + '_pdf']
            types_list += ['f8']

    table = Table(names=names_list, dtype=types_list)

    return table

def create_foggie_cmap(cmin, cmax, cfunc, color_key, log=False):
    '''This function makes the image for the little colorbar that can be put on the datashader main
    image. It takes the minimum and maximum values of the field that is being turned into a colorbar,
    'cmin' and 'cmax', the name of the color-categorization function (in consistency.py), 'cfunc',
    and the name of the color key (also in consistency.py), 'color_key', and returns the color bar.'''

    x = np.random.rand(100000)
    y = np.random.rand(100000)
    if (log): rand = np.random.rand(100000) * (np.log10(cmax)-np.log10(cmin)) + np.log10(cmin)
    else: rand = np.random.rand(100000) * (cmax-cmin) + cmin

    df = pd.DataFrame({})
    df['x'] = x
    df['y'] = y
    df['rand'] = rand
    n_labels = np.size(list(color_key))
    sightline_length = np.max(df['x']) - np.min(df['x'])
    value = np.max(df['x'])

    cat = cfunc(rand)
    for index in np.flip(np.arange(n_labels), 0):
        cat[x > value - sightline_length*(1.*index+1)/n_labels] = \
          list(color_key)[index]
    df['cat'] = cat
    df.cat = df.cat.astype('category')

    cvs = dshader.Canvas(plot_width=750, plot_height=100,
                         x_range=(np.min(df['x']),
                                  np.max(df['x'])),
                         y_range=(np.min(df['y']),
                                  np.max(df['y'])))
    agg = cvs.points(df, 'x', 'y', dshader.count_cat('cat'))
    cmap = tf.spread(tf.shade(agg, color_key=color_key), px=2, shape='square')
    return cmap

def categorize_by_force_ratio(ratio):
    """ define the force ratio category strings"""
    ra = np.chararray(np.size(ratio), 5)
    ra[ratio > force_ratio_max] = force_ratio_color_labels[-1]
    for i in range(len(force_ratio_color_labels)):
        val = force_ratio_max - (force_ratio_max-force_ratio_min)/(np.size(force_ratio_color_labels)-1.)*i
        ra[ratio < val] = force_ratio_color_labels[-1 - i]
    return ra

def velocity_PDF(snap):
    '''Plots PDFs of the three velocity components in a given radius bin for a given snapshot.'''

    stats_dir = output_dir + 'stats_halo_00' + args.halo + '/' + args.run + '/'
    stats = Table.read(stats_dir + '/Tables/' + snap + '_' + args.filename.replace('_pdf', '') + '.hdf5', path='all_data')
    pdf = Table.read(stats_dir + '/Tables/' + snap + '_' + args.filename + '.hdf5', path='all_data')
    radius_list = 0.5*(stats['inner_radius'] + stats['outer_radius'])
    rvir = rvir_masses['radius'][rvir_masses['snapshot']==snap]
    if ('Rvir' in args.pdf_R):
        r_ind = np.where(radius_list>=float(args.pdf_R.split('Rvir')[0])*rvir)[0][0]
    else:
        r_ind = np.where(radius_list>=float(args.pdf_R))[0][0]
    inds = np.where(pdf['inner_radius']==stats['inner_radius'][r_ind])[0]

    if (args.spheres!='none'):
        stats_spheres = Table.read(stats_dir + '/Tables/' + snap + '_' + args.spheres.replace('_pdf', '') + '.hdf5', path='all_data')
        pdf_spheres = Table.read(stats_dir + '/Tables/' + snap + '_' + args.spheres + '.hdf5', path='all_data')
        if ('Rvir' in args.pdf_R):
            radius_list_sph = stats_spheres['center_radius'][(stats_spheres['center_radius']>=float(args.pdf_R.split('Rvir')[0])*rvir-5) & \
              (stats_spheres['center_radius']<float(args.pdf_R.split('Rvir')[0])*rvir+5)]
        else:
            radius_list_sph = stats_spheres['center_radius'][(stats_spheres['center_radius']>=float(args.pdf_R)-5) & \
              (stats_spheres['center_radius']<float(args.pdf_R)+5)]
        print('There are %d spheres within 5 kpc of the chosen radius' % (len(radius_list_sph)))
        inds_sph = []
        for i in range(len(radius_list_sph)):
            inds_sph.append(np.where(pdf_spheres['center_radius']==radius_list_sph[i])[0])

    fig = plt.figure(figsize=(16,4),dpi=500)
    components_list = ['theta_velocity','phi_velocity','radial_velocity']
    for j in range(len(components_list)):
        comp = components_list[j]
        ax = fig.add_subplot(1,3,j+1)
        data = 0.5 * (pdf['lower_' + comp][inds] + pdf['upper_' + comp][inds])
        hist_data = pdf['net_' + comp + '_pdf'][inds]
        ax.plot(data, hist_data, 'k-', lw=2, label='PDF')
        if (args.spheres!='none'):
            for i in range(len(inds_sph)):
                data_sph = 0.5 * (pdf_spheres['lower_' + comp][inds_sph[i]] + pdf_spheres['upper_' + comp][inds_sph[i]])
                hist_data_sph = pdf_spheres[comp + '_pdf'][inds_sph[i]]
                ax.plot(data_sph, hist_data_sph, 'c-', lw=1, alpha=0.2)
        ax.set_ylabel('Mass PDF', fontsize=18)
        if (j<2): ax.set_xlim(-400,400)
        else: ax.set_xlim(-400,400)
        ax.tick_params(axis='both', which='both', direction='in', length=8, width=2, pad=5, labelsize=18, \
                       top=True, right=True)
        x_range = ax.get_xlim()
        y_range = ax.get_ylim()
        if (comp=='radial_velocity'):
            vr_mu_turb = 0.
            vr_sig_turb = stats['net_tangential_velocity_sig'][r_ind]
            vr_A_turb = stats['net_radial_velocity_Aturb'][r_ind]
            vr_mu_in = stats['net_radial_velocity_muIn'][r_ind]
            vr_sig_in = stats['net_radial_velocity_sigIn'][r_ind]
            vr_A_in = stats['net_radial_velocity_Ain'][r_ind]
            vr_mu_out = stats['net_radial_velocity_muOut'][r_ind]
            vr_sig_out = stats['net_radial_velocity_sigOut'][r_ind]
            vr_A_out = stats['net_radial_velocity_AOut'][r_ind]
            fit = triple_gauss(data, vr_mu_turb, vr_sig_turb, vr_A_turb, vr_mu_in, vr_sig_in, vr_A_in, vr_mu_out, vr_sig_out, vr_A_out)
            fit1 = gauss_pos(data, vr_mu_out, vr_sig_out, vr_A_out)
            fit2 = gauss(data, vr_mu_turb, vr_sig_turb, vr_A_turb)
            fit3 = gauss_neg(data, vr_mu_in, vr_sig_in, vr_A_in)
            ax.plot(data, fit, 'b--', lw=2, label='Best fit')
            ax.plot(data, fit1, 'b:', lw=1)
            ax.plot(data, fit2, 'b:', lw=1)
            ax.plot(data, fit3, 'b:', lw=1)
            #ax.plot([vr_mu, vr_mu], [y_range[0], y_range[1]], 'b:', lw=1)
            #ax.plot([vr_mu - vr_sig, vr_mu + vr_sig], [0.1*(y_range[1]-y_range[0])+y_range[0],0.1*(y_range[1]-y_range[0])+y_range[0]], 'b:', lw=1)
            xloc = 0.97*(x_range[1]-x_range[0])+x_range[0]
            yloc = 0.9*(y_range[1]-y_range[0])+y_range[0]
            ax.text(xloc, yloc, '$\mu_\mathrm{in}=%.1f$\n$\sigma_\mathrm{in}=%.1f$\n$\mu_\mathrm{out}=%.1f$\n$\sigma_\mathrm{out}=%.1f$' % (vr_mu_in, vr_sig_in, vr_mu_out, vr_sig_out), va='top', ha='right', fontsize=18)
            xlabel = 'Radial Velocity [km/s]'
        else:
            v_mu = stats['net_' + comp + '_mu'][r_ind]
            v_sig = stats['net_' + comp + '_sig'][r_ind]
            v_A = stats['net_' + comp + '_A'][r_ind]
            fit = gauss(data, v_mu, v_sig, v_A)
            ax.plot(data, fit, 'b--', lw=2, label='Best fit')
            ax.plot([v_mu, v_mu], [y_range[0], y_range[1]], 'b:', lw=1)
            ax.plot([v_mu - v_sig, v_mu + v_sig], [0.1*(y_range[1]-y_range[0])+y_range[0],0.1*(y_range[1]-y_range[0])+y_range[0]], 'b:', lw=1)
            xloc = 0.97*(x_range[1]-x_range[0])+x_range[0]
            yloc = 0.7*(y_range[1]-y_range[0])+y_range[0]
            ax.text(xloc, yloc, '$\mu=%.1f$\n$\sigma=%.1f$' % (v_mu, v_sig), va='top', ha='right', fontsize=18)
            if (j==0):
                xlabel = '$\\theta$ Velocity [km/s]'
                xloc = 0.2*(x_range[1]-x_range[0])+x_range[0]
                yloc = 0.85*(y_range[1]-y_range[0])+y_range[0]
                ax.text(xloc, yloc, '$z=%.2f$' % (stats['redshift'][0]), fontsize=18, ha='center', va='center')
            if (j==1):
                xlabel = '$\\phi$ Velocity [km/s]'
                xloc = 0.97*(x_range[1]-x_range[0])+x_range[0]
                yloc = 0.2*(y_range[1]-y_range[0])+y_range[0]
                ax.text(xloc, yloc, '$%.1f$ kpc' % (radius_list[r_ind]), fontsize=18, ha='right', va='center')

        if (j==0): ax.legend(loc=1, frameon=False, fontsize=18)
        ax.set_xlabel(xlabel, fontsize=18)
        ax.set_ylim(y_range)

    plt.subplots_adjust(left=0.07, bottom=0.18, right=0.98, top=0.97, wspace=0.35)
    plt.savefig(save_dir + snap + '_velocity_PDFs' + save_suffix + '.pdf')
    plt.close()

    print('Plot made!')

def metallicity_PDF(snaplist):
    '''Plots PDFs of the mass-weighted metallicity in the inner and outer CGM for the snapshots in 'snaplist'.'''

    stats_dir = output_dir + 'stats_halo_00' + args.halo + '/' + args.run + '/'
    fig = plt.figure(figsize=(12,5),dpi=500)
    ax1 = fig.add_subplot(1,2,1)
    ax2 = fig.add_subplot(1,2,2)

    linecolors = plt.cm.YlGnBu(np.linspace(0.2,1,4))
    snap_z = [2.,1.,0.5,0.]

    for i in range(len(snaplist)):
        snap = snaplist[i]
        inner_pdf = Table.read(stats_dir + '/Tables/' + snap + '_stats_metallicity_pdf_sphere_mass-weighted_inner-cgm-only.hdf5', path='all_data')
        outer_pdf = Table.read(stats_dir + '/Tables/' + snap + '_stats_metallicity_pdf_sphere_mass-weighted_outer-cgm-only.hdf5', path='all_data')
        met_dist = 0.5*(inner_pdf['lower_log_metallicity'] + inner_pdf['upper_log_metallicity'])

        print(i, inner_pdf['inner_radius'][0], inner_pdf['outer_radius'][0])
        print(i, outer_pdf['inner_radius'][0], outer_pdf['outer_radius'][0])
        ax1.plot(met_dist, inner_pdf['net_log_metallicity_pdf'][:200], ls='-', lw=2, color=linecolors[i], label='$z=%.1f$' % snap_z[i])
        ax2.plot(met_dist, outer_pdf['net_log_metallicity_pdf'][:200], ls='-', lw=2, color=linecolors[i])

    ax1.plot([-2,-2],[0,1.2], 'k--', lw=1)
    ax1.plot([0,0],[0,1.2], 'k--', lw=1)
    ax2.plot([-2,-2],[0,1.2], 'k--', lw=1)
    ax2.plot([0,0],[0,1.2], 'k--', lw=1)

    ax1.set_ylabel('Mass PDF', fontsize=18)
    ax1.set_xlabel('log Metallicity [$Z_\odot$]', fontsize=18)
    ax2.set_xlabel('log Metallicity [$Z_\odot$]', fontsize=18)
    ax1.axis([-4,1,0,1.2])
    ax2.axis([-4,1,0,1.2])
    ax1.tick_params(axis='both', which='both', direction='in', length=8, width=2, pad=5, labelsize=18, \
                   top=True, right=True)
    ax2.tick_params(axis='both', which='both', direction='in', length=8, width=2, pad=5, labelsize=18, \
                   top=True, right=True, labelleft=False)
    ax1.legend(loc=2, frameon=False, fontsize=18)
    ax1.set_title('Inner CGM $0.1R_{200} < r < 0.5R_{200}$', fontsize=18)
    ax2.set_title('Outer CGM $0.5R_{200} < r < 1R_{200}$', fontsize=18)
    ax2.text(-3.8, 1.1,'Tempest', fontsize=18, ha='left', va='center')

    plt.subplots_adjust(left=0.07, bottom=0.15, right=0.98, top=0.92, wspace=0.07)
    plt.savefig(save_dir + 'metallicity_PDFs' + save_suffix + '.png')
    plt.close()

    print('Plot made!')

def pressures_vs_radius(snap):
    '''Plots different types of pressure (thermal, turbulent, bulk inflow/outflow ram)
    as functions of radius for the simulation output given by 'snap'.
    Also saves to file the statistics and pdfs of the distribution in each radial bin.
    Use --load_stats to skip the calculation step and only plot from file.'''

    tablename_prefix = output_dir + 'stats_halo_00' + args.halo + '/' + args.run + '/Tables/'
    if (args.load_stats):
        stats = Table.read(tablename_prefix + snap + '_stats_pressure-types_' + args.filename + '.hdf5', path='all_data')
    Rvir = rvir_masses['radius'][rvir_masses['snapshot']==snap][0]

    if (not args.load_stats):
        if (args.system=='pleiades_cassi'):
            print('Copying directory to /tmp')
            snap_dir = '/tmp/' + args.halo + '/' + args.run + '/' + target_dir + '/' + snap
            if (args.copy_to_tmp):
                shutil.copytree(foggie_dir + run_dir + snap, snap_dir)
                snap_name = snap_dir + '/' + snap
            else:
                # Make a dummy directory with the snap name so the script later knows the process running
                # this snapshot failed if the directory is still there
                os.makedirs(snap_dir)
                snap_name = foggie_dir + run_dir + snap + '/' + snap
        else:
            snap_name = foggie_dir + run_dir + snap + '/' + snap
        ds, refine_box = foggie_load(snap_name, trackname, do_filter_particles=False, halo_c_v_name=halo_c_v_name, gravity=True, masses_dir=masses_dir)
        zsnap = ds.get_parameter('CosmologyCurrentRedshift')

        # Define the density cut between disk and CGM to vary smoothly between 1 and 0.1 between z = 0.5 and z = 0.25,
        # with it being 1 at higher redshifts and 0.1 at lower redshifts
        current_time = ds.current_time.in_units('Myr').v
        if (current_time<=8656.88):
            density_cut_factor = 1.
        elif (current_time<=10787.12):
            density_cut_factor = 1. - 0.9*(current_time-8656.88)/2130.24
        else:
            density_cut_factor = 0.1

        pix_res = float(np.min(refine_box['gas','dx'].in_units('kpc')))  # at level 11
        lvl1_res = pix_res*2.**11.
        level = 9
        dx = lvl1_res/(2.**level)
        dx_cm = dx*1000*cmtopc
        smooth_scale = int(25./dx)/6.
        refine_res = int(3.*Rvir/dx)
        box = ds.covering_grid(level=level, left_edge=ds.halo_center_kpc-ds.arr([1.5*Rvir,1.5*Rvir,1.5*Rvir],'kpc'), dims=[refine_res, refine_res, refine_res])
        density = box['density'].in_units('g/cm**3').v
        temperature = box['temperature'].v
        radius = box['radius_corrected'].in_units('kpc').v
        x = box[('gas','x')].in_units('cm').v - ds.halo_center_kpc[0].to('cm').v
        y = box[('gas','y')].in_units('cm').v - ds.halo_center_kpc[1].to('cm').v
        z = box[('gas','z')].in_units('cm').v - ds.halo_center_kpc[2].to('cm').v
        r = box['radius_corrected'].in_units('cm').v
        x_hat = x/r
        y_hat = y/r
        z_hat = z/r

        # This next block needed for removing any ISM regions and then interpolating over the holes left behind
        if (args.cgm_only):
            # Define ISM regions to remove
            disk_mask = (density > cgm_density_max * density_cut_factor)
            # disk_mask_expanded is a binary mask of both ISM regions AND their surrounding pixels
            struct = ndimage.generate_binary_structure(3,3)
            disk_mask_expanded = ndimage.binary_dilation(disk_mask, structure=struct, iterations=3)
            disk_mask_expanded = ndimage.binary_closing(disk_mask_expanded, structure=struct, iterations=3)
            disk_mask_expanded = disk_mask_expanded | disk_mask
            # disk_edges is a binary mask of ONLY pixels surrounding ISM regions -- nothing inside ISM regions
            disk_edges = disk_mask_expanded & ~disk_mask
            x_edges = x[disk_edges].flatten()
            y_edges = y[disk_edges].flatten()
            z_edges = z[disk_edges].flatten()

            den_edges = density[disk_edges]
            den_interp_func = NearestNDInterpolator(list(zip(x_edges,y_edges,z_edges)), den_edges)
            den_masked = np.copy(density)
            den_masked[disk_mask] = den_interp_func(x[disk_mask], y[disk_mask], z[disk_mask])
        else:
            den_masked = density

        if (args.cgm_only): radius = radius[(density < cgm_density_max * density_cut_factor)]
        if (args.weight=='mass'):
            weights = box['cell_mass'].in_units('Msun').v
        if (args.weight=='volume'):
            weights = box['cell_volume'].in_units('kpc**3').v
        if (args.cgm_only): weights = weights[(density < cgm_density_max * density_cut_factor)]
        if (args.region_filter=='metallicity'):
            metallicity = box['metallicity'].in_units('Zsun').v
            if (args.cgm_only): metallicity = metallicity[(density < cgm_density_max * density_cut_factor)]
        if (args.region_filter=='velocity'):
            rv = box['radial_velocity_corrected'].in_units('km/s').v
            rv = rv[(density < cgm_density_max * density_cut_factor)]
            vff = box['vff'].in_units('km/s').v
            vesc = box['vesc'].in_units('km/s').v
            if (args.cgm_only):
                vff = vff[(density < cgm_density_max * density_cut_factor)]
                vesc = vesc[(density < cgm_density_max * density_cut_factor)]

        thermal_pressure = box['pressure'].in_units('erg/cm**3').v
        if (args.cgm_only):
            pres_edges = thermal_pressure[disk_edges]
            pres_interp_func = NearestNDInterpolator(list(zip(x_edges,y_edges,z_edges)), pres_edges)
            pres_masked = np.copy(thermal_pressure)
            pres_masked[disk_mask] = pres_interp_func(x[disk_mask], y[disk_mask], z[disk_mask])
        else:
            pres_masked = thermal_pressure
        vx = box['vx_corrected'].in_units('cm/s').v
        vy = box['vy_corrected'].in_units('cm/s').v
        vz = box['vz_corrected'].in_units('cm/s').v
        if (args.cgm_only):
            vx_edges = vx[disk_edges]
            vy_edges = vy[disk_edges]
            vz_edges = vz[disk_edges]
            vx_interp_func = NearestNDInterpolator(list(zip(x_edges,y_edges,z_edges)), vx_edges)
            vy_interp_func = NearestNDInterpolator(list(zip(x_edges,y_edges,z_edges)), vy_edges)
            vz_interp_func = NearestNDInterpolator(list(zip(x_edges,y_edges,z_edges)), vz_edges)
            vx_masked = np.copy(vx)
            vy_masked = np.copy(vy)
            vz_masked = np.copy(vz)
            vx_masked[disk_mask] = vx_interp_func(x[disk_mask], y[disk_mask], z[disk_mask])
            vy_masked[disk_mask] = vy_interp_func(x[disk_mask], y[disk_mask], z[disk_mask])
            vz_masked[disk_mask] = vz_interp_func(x[disk_mask], y[disk_mask], z[disk_mask])
        else:
            vx_masked = vx
            vy_masked = vy
            vz_masked = vz
        smooth_vx = gaussian_filter(vx_masked, smooth_scale)
        smooth_vy = gaussian_filter(vy_masked, smooth_scale)
        smooth_vz = gaussian_filter(vz_masked, smooth_scale)
        smooth_den = gaussian_filter(den_masked, smooth_scale)
        sig_x = (vx_masked - smooth_vx)**2.
        sig_y = (vy_masked - smooth_vy)**2.
        sig_z = (vz_masked - smooth_vz)**2.
        vdisp = np.sqrt((sig_x + sig_y + sig_z)/3.)
        turb_pressure = smooth_den*vdisp**2.
        vr = box['radial_velocity_corrected'].in_units('cm/s').v
        if (args.cgm_only):
            vr_edges = vr[disk_edges]
            vr_interp_func = NearestNDInterpolator(list(zip(x_edges,y_edges,z_edges)), vr_edges)
            vr_masked = np.copy(vr)
            vr_masked[disk_mask] = vr_interp_func(x[disk_mask], y[disk_mask], z[disk_mask])
        else:
            vr_masked = vr
        vr_masked = gaussian_filter(vr_masked, smooth_scale)
        dvr = np.gradient(vr_masked, dx_cm)
        delta_vr = dvr[0]*dx_cm*x_hat + dvr[1]*dx_cm*y_hat + dvr[2]*dx_cm*z_hat
        ram_pressure = smooth_den*(delta_vr)**2.

        pressures = [thermal_pressure, turb_pressure, ram_pressure]
        if (args.cgm_only):
            for i in range(len(pressures)):
                pressures[i] = pressures[i][(density < cgm_density_max * density_cut_factor)]
            new_density = density[(density < cgm_density_max * density_cut_factor)]
            temperature = temperature[(density < cgm_density_max * density_cut_factor)]
            density = new_density

        stats = ['thermal_pressure', 'turbulent_pressure', 'ram_pressure']
        table = make_table(stats)
        table_pdf = make_pdf_table(stats)

        radius_list = np.linspace(0., 1.5*Rvir, 100)
        if (args.region_filter!='none'):
            pressure_regions = []
            for i in range(len(pressures)):
                pressure_regions.append([])
            weights_regions = []
            radius_regions = []
        if (args.region_filter=='temperature'):
            regions = ['_low-T', '_mid-T', '_high-T']
            weights_regions.append(weights[temperature < 10**4.8])
            weights_regions.append(weights[(temperature > 10**4.8) & (temperature < 10**6.3)])
            weights_regions.append(weights[temperature > 10**6.3])
            radius_regions.append(radius[temperature < 10**4.8])
            radius_regions.append(radius[(temperature > 10**4.8) & (temperature < 10**6.3)])
            radius_regions.append(radius[temperature > 10**6.3])
            for i in range(len(pressures)):
                pressure_regions[i].append(pressures[i][temperature < 10**4.8])
                pressure_regions[i].append(pressures[i][(temperature > 10**4.8) & (temperature < 10**6.3)])
                pressure_regions[i].append(pressures[i][temperature > 10**6.3])
        elif (args.region_filter=='metallicity'):
            regions = ['_low-Z', '_mid-Z', '_high-Z']
            weights_regions.append(weights[metallicity < 0.01])
            weights_regions.append(weights[(metallicity > 0.01) & (metallicity < 1)])
            weights_regions.append(weights[metallicity > 1])
            radius_regions.append(radius[metallicity < 0.01])
            radius_regions.append(radius[(metallicity > 0.01) & (metallicity < 1)])
            radius_regions.append(radius[metallicity > 1])
            for i in range(len(pressures)):
                pressure_regions[i].append(pressures[i][metallicity < 0.01])
                pressure_regions[i].append(pressures[i][(metallicity > 0.01) & (metallicity < 1)])
                pressure_regions[i].append(pressures[i][metallicity > 1])
        elif (args.region_filter=='velocity'):
            regions = ['_low-v', '_mid-v', '_high-v']
            weights_regions.append(weights[rv < 0.5*vff])
            weights_regions.append(weights[(rv > 0.5*vff) & (rv < vesc)])
            weights_regions.append(weights[rv > vesc])
            radius_regions.append(radius[rv < 0.5*vff])
            radius_regions.append(radius[(rv > 0.5*vff) & (rv < vesc)])
            radius_regions.append(radius[rv > vesc])
            for i in range(len(pressures)):
                pressure_regions[i].append(pressures[i][rv < 0.5*vff])
                pressure_regions[i].append(pressures[i][(rv > 0.5*vff) & (rv < vesc)])
                pressure_regions[i].append(pressures[i][rv > vesc])
        else:
            regions = []

        for i in range(len(radius_list)-1):
            row = [zsnap, radius_list[i], radius_list[i+1]]
            pdf_array = []
            for j in range(len(pressures)):
                pressure_shell = np.log10(pressures[j][(radius >= radius_list[i]) & (radius < radius_list[i+1])])
                weights_shell = weights[(radius >= radius_list[i]) & (radius < radius_list[i+1])]
                if (len(pressure_shell)!=0.):
                    quantiles = weighted_quantile(pressure_shell, weights_shell, np.array([0.25,0.5,0.75]))
                    row.append(quantiles[1])
                    row.append(quantiles[2]-quantiles[0])
                    avg, std = weighted_avg_and_std(pressure_shell, weights_shell)
                    row.append(avg)
                    row.append(std)
                    hist, bin_edges = np.histogram(pressure_shell, weights=weights_shell, bins=(200), range=[-20, -12], density=True)
                    pdf_array.append(bin_edges[:-1])
                    pdf_array.append(bin_edges[1:])
                    pdf_array.append(hist)
                    for k in range(len(regions)):
                        pressure_shell = np.log10(pressure_regions[j][k][(radius_regions[k] >= radius_list[i]) & (radius_regions[k] < radius_list[i+1])])
                        weights_shell = weights_regions[k][(radius_regions[k] >= radius_list[i]) & (radius_regions[k] < radius_list[i+1])]
                        if (len(pressure_shell)!=0.):
                            quantiles = weighted_quantile(pressure_shell, weights_shell, np.array([0.25,0.5,0.75]))
                            row.append(quantiles[1])
                            row.append(quantiles[2]-quantiles[0])
                            avg, std = weighted_avg_and_std(pressure_shell, weights_shell)
                            row.append(avg)
                            row.append(std)
                            hist, bin_edges = np.histogram(pressure_shell, weights=weights_shell, bins=(200), range=[-20, -12], density=True)
                            pdf_array.append(hist)
                        else:
                            row.append(0.)
                            row.append(0.)
                            row.append(0.)
                            row.append(0.)
                            pdf_array.append(np.zeros(200))
                else:
                    row.append(0.)
                    row.append(0.)
                    row.append(0.)
                    row.append(0.)
                    pdf_array.append(np.zeros(200))
                    pdf_array.append(np.zeros(200))
                    pdf_array.append(np.zeros(200))
                    for k in range(len(regions)):
                        row.append(0.)
                        row.append(0.)
                        row.append(0.)
                        row.append(0.)
                        pdf_array.append(np.zeros(200))

            table.add_row(row)
            pdf_array = np.vstack(pdf_array)
            pdf_array = np.transpose(pdf_array)
            for p in range(len(pdf_array)):
                row_pdf = [zsnap, radius_list[i], radius_list[i+1]]
                row_pdf += list(pdf_array[p])
                table_pdf.add_row(row_pdf)

        table = set_table_units(table)
        table_pdf = set_table_units(table_pdf)

        # Save to file
        table.write(tablename_prefix + snap + '_stats_pressure-types' + save_suffix + '.hdf5', path='all_data', serialize_meta=True, overwrite=True)
        table_pdf.write(tablename_prefix + snap + '_stats_pressure-types_pdf' + save_suffix + '.hdf5', path='all_data', serialize_meta=True, overwrite=True)

        stats = table
        print("Stats have been calculated and saved to file for snapshot " + snap + "!")
        # Delete output from temp directory if on pleiades
        if (args.system=='pleiades_cassi'):
            print('Deleting directory from /tmp')
            shutil.rmtree(snap_dir)

    plot_colors = ['r', 'g', 'm']
    plot_labels = ['Thermal', 'Turbulent', 'Ram']
    file_labels = ['thermal_pressure', 'turbulent_pressure', 'ram_pressure']
    linestyles = ['-', '--', ':']

    radius_list = 0.5*(stats['inner_radius'] + stats['outer_radius'])
    zsnap = stats['redshift'][0]

    fig = plt.figure(figsize=(8,6), dpi=500)
    ax = fig.add_subplot(1,1,1)

    for i in range(len(plot_colors)):
        label = plot_labels[i]
        ax.plot(radius_list, stats[file_labels[i] + '_med'], ls=linestyles[i], color=plot_colors[i], \
                lw=2, label=label)

    ax.set_ylabel('log Median Pressure [erg/cm$^3$]', fontsize=18)
    ax.set_xlabel('Radius [kpc]', fontsize=18)
    ax.axis([0,250,-18,-10])
    ax.text(15, -17.5, '$z=%.2f$' % (zsnap), fontsize=18, ha='left', va='center')
    ax.text(15,-17,halo_dict[args.halo],ha='left',va='center',fontsize=18)
    ax.tick_params(axis='both', which='both', direction='in', length=8, width=2, pad=5, labelsize=18, \
      top=True, right=True)
    ax.plot([Rvir, Rvir], [ax.get_ylim()[0], ax.get_ylim()[1]], 'k--', lw=1)
    ax.text(Rvir-3., -17.5, '$R_{200}$', fontsize=18, ha='right', va='center')
    if (args.halo=='8508'): ax.legend(loc=2, frameon=False, fontsize=18)
    plt.subplots_adjust(top=0.94,bottom=0.11,right=0.95,left=0.15)
    plt.savefig(save_dir + snap + '_pressures_vs_r' + save_suffix + '.png')
    plt.close()

    if (args.region_filter!='none'):
        fig = plt.figure(figsize=(8,6), dpi=500)
        ax = fig.add_subplot(1,1,1)

        for i in range(len(plot_colors)):
            label = plot_labels[i]
            if (i==0): label_regions = ['Low ' + args.region_filter, 'Mid ' + args.region_filter, 'High ' + args.region_filter]
            else: label_regions = ['__nolegend__', '__nolegend__', '__nolegend__']
            ax.plot(radius_list, stats['low_' + args.region_filter + '_' + file_labels[i] + '_med'], ls=linestyles[i], color=plot_colors[i], \
                    lw=2, alpha=0.25, label=label_regions[0])
            ax.plot(radius_list, stats['mid_' + args.region_filter + '_' + file_labels[i] + '_med'], ls=linestyles[i], color=plot_colors[i], \
                    lw=2, alpha=0.5, label=label_regions[1])
            ax.plot(radius_list, stats['high_' + args.region_filter + '_' + file_labels[i] + '_med'], ls=linestyles[i], color=plot_colors[i], \
                    lw=2, label=label_regions[2])
            ax.plot(radius_list, stats['high_' + args.region_filter + '_' + file_labels[i] + '_med'], ls=linestyles[i], color=plot_colors[i], \
                    lw=2, label=label)

        ax.set_ylabel('log Median Pressure [erg/cm$^3$]', fontsize=18)
        ax.set_xlabel('Radius [kpc]', fontsize=18)
        ax.axis([0,250,-18,-10])
        ax.text(15, -17.5, '$z=%.2f$' % (zsnap), fontsize=18, ha='left', va='center')
        ax.text(15,-17,halo_dict[args.halo],ha='left',va='center',fontsize=18)
        ax.tick_params(axis='both', which='both', direction='in', length=8, width=2, pad=5, labelsize=18, \
          top=True, right=True)
        ax.plot([Rvir, Rvir], [ax.get_ylim()[0], ax.get_ylim()[1]], 'k--', lw=1)
        ax.text(Rvir-3., -17.5, '$R_{200}$', fontsize=18, ha='right', va='center')
        if (args.halo=='8508'): ax.legend(loc=2, frameon=False, fontsize=18)
        plt.subplots_adjust(top=0.94,bottom=0.11,right=0.95,left=0.15)
        plt.savefig(save_dir + snap + '_pressures_vs_r_regions-' + args.region_filter + save_suffix + '.png')
        plt.close()

def pressures_vs_time(snaplist):
    '''Plots different pressures at a given radius over time, for all snaps in the list snaplist.'''

    tablename_prefix = output_dir + 'stats_halo_00' + args.halo + '/' + args.run + '/Tables/'
    time_table = Table.read(output_dir + 'times_halo_00' + args.halo + '/' + args.run + '/time_table.hdf5', path='all_data')
    if (args.filename != ''): filename = '_' + args.filename
    else: filename = ''

    fig = plt.figure(figsize=(8,6), dpi=500)
    ax = fig.add_subplot(1,1,1)

    plot_colors = ['r', 'g', 'm']
    plot_labels = ['Thermal', 'Turbulent', 'Ram']
    file_labels = ['thermal_pressure', 'turbulent_pressure', 'ram_pressure']
    linestyles = ['-', '--', ':']
    alphas = [0.3, 0.6, 1.]

    zlist = []
    timelist = []
    pressures_list = []
    if (args.region_filter!='none'):
        pressures_regions = [[],[],[]]
        region_label = ['Low ' + args.region_filter, 'Mid ' + args.region_filter, 'High ' + args.region_filter]
        region_name = ['low-', 'mid-', 'high-']
    for j in range(len(plot_labels)):
        pressures_list.append([])
        if (args.region_filter!='none'):
            pressures_regions[0].append([])
            pressures_regions[1].append([])
            pressures_regions[2].append([])

    for i in range(len(snaplist)):
        snap = snaplist[i]
        stats = Table.read(tablename_prefix + snap + '_stats_pressure-types' + filename + '.hdf5', path='all_data')
        Rvir = rvir_masses['radius'][rvir_masses['snapshot']==snap][0]
        radius = args.radius*Rvir
        rad_ind = np.where(stats['inner_radius']<=radius)[0][-1]
        timelist.append(time_table['time'][time_table['snap']==snap][0]/1000.)
        zlist.append(stats['redshift'][0])
        for j in range(len(file_labels)):
            pressures_list[j].append(stats[file_labels[j] + '_med'][rad_ind])
            if (args.region_filter!='none'):
                pressures_regions[0][j].append(stats['low_' + args.region_filter + '_' + file_labels[j] + '_med'][rad_ind])
                pressures_regions[1][j].append(stats['mid_' + args.region_filter + '_' + file_labels[j] + '_med'][rad_ind])
                pressures_regions[2][j].append(stats['high_' + args.region_filter + '_' + file_labels[j] + '_med'][rad_ind])

    zlist.reverse()
    timelist.reverse()
    time_func = IUS(zlist, timelist)
    timelist.reverse()
    timelist = np.array(timelist).flatten()
    zlist = np.array(zlist)

    fig = plt.figure(figsize=(8,6), dpi=500)
    ax = fig.add_subplot(1,1,1)

    for j in range(len(plot_labels)):
        ax.plot(timelist, pressures_list[j], lw=2, ls=linestyles[j], color=plot_colors[j], label=plot_labels[j])

    ax.axis([np.min(timelist), np.max(timelist), -18,-10])
    ax.legend(loc=1, frameon=False, fontsize=18)
    ax.text(4.,-10.5, halo_dict[args.halo], ha='left', va='center', fontsize=18)
    ax.text(7,-10.5, '$r=%.2f R_{200}$' % (args.radius), ha='left', va='center', fontsize=18)

    ax2 = ax.twiny()
    ax.tick_params(axis='both', which='both', direction='in', length=8, width=2, pad=5, labelsize=18, \
      top=False, right=True)
    ax2.tick_params(axis='x', which='both', direction='in', length=8, width=2, pad=5, labelsize=18, \
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
    ax.set_ylabel('log Median Pressure [erg/cm$^3$]', fontsize=18)

    z_sfr, sfr = np.loadtxt(code_path + 'halo_infos/00' + args.halo + '/' + args.run + '/sfr', unpack=True, usecols=[1,2], skiprows=1)
    t_sfr = time_func(z_sfr)

    ax3 = ax.twinx()
    ax3.plot(t_sfr, sfr, 'k-', lw=1)
    ax.plot([timelist[0],timelist[-1]], [0,0], 'k-', lw=1, label='SFR (right axis)')
    ax3.tick_params(axis='y', which='both', direction='in', length=8, width=2, pad=5, labelsize=18, right=True)
    ax3.set_ylim(-5,200)
    ax3.set_ylabel('SFR [$M_\odot$/yr]', fontsize=18)

    fig.subplots_adjust(left=0.13, bottom=0.12, right=0.87, top=0.89)
    fig.savefig(save_dir + 'pressures_vs_time' + save_suffix + '.png')
    plt.close(fig)

    if (args.region_filter!='none'):
        fig_regions = []
        axs_regions = []
        for i in range(len(plot_colors)):
            fig_regions.append(plt.figure(figsize=(8,6), dpi=500))
            axs_regions.append(fig_regions[-1].add_subplot(1,1,1))
        for i in range(3):
            fig = plt.figure(figsize=(8,6), dpi=500)
            ax = fig.add_subplot(1,1,1)

            for j in range(len(plot_labels)):
                axs_regions[j].plot(timelist, pressures_regions[i][j], lw=2, ls=linestyles[j], color=plot_colors[j], alpha=alphas[i], label=region_label[i])
                ax.plot(timelist, pressures_regions[i][j], lw=2, ls=linestyles[j], color=plot_colors[j], label=plot_labels[j])

            ax.axis([np.min(timelist), np.max(timelist), -18,-10])
            ax.legend(loc=1, frameon=False, fontsize=18)
            ax.text(4.,-10.5, halo_dict[args.halo], ha='left', va='center', fontsize=18)
            ax.text(7,-10.5, '$r=%.2f R_{200}$' % (args.radius), ha='left', va='center', fontsize=18)
            ax.text(13,-17, region_label[i], fontsize=18, ha='right', va='center')

            ax2 = ax.twiny()
            ax.tick_params(axis='both', which='both', direction='in', length=8, width=2, pad=5, labelsize=18, \
              top=False, right=True)
            ax2.tick_params(axis='x', which='both', direction='in', length=8, width=2, pad=5, labelsize=18, \
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
            ax.set_ylabel('log Median Pressure [erg/cm$^3$]', fontsize=18)

            z_sfr, sfr = np.loadtxt(code_path + 'halo_infos/00' + args.halo + '/' + args.run + '/sfr', unpack=True, usecols=[1,2], skiprows=1)
            t_sfr = time_func(z_sfr)

            ax3 = ax.twinx()
            ax3.plot(t_sfr, sfr, 'k-', lw=1)
            ax.plot([timelist[0],timelist[-1]], [0,0], 'k-', lw=1, label='SFR (right axis)')
            ax3.tick_params(axis='y', which='both', direction='in', length=8, width=2, pad=5, labelsize=18, right=True)
            ax3.set_ylim(-5,200)
            ax3.set_ylabel('SFR [$M_\odot$/yr]', fontsize=18)

            fig.subplots_adjust(left=0.13, bottom=0.12, right=0.87, top=0.89)
            fig.savefig(save_dir + 'pressures_vs_time_region-' + region_name[i] + args.region_filter + save_suffix + '.png')
            plt.close(fig)

        for j in range(len(plot_labels)):
            axs_regions[j].axis([np.min(timelist), np.max(timelist), -18,-10])
            axs_regions[j].legend(loc=1, frameon=False, fontsize=18)
            axs_regions[j].text(4.,-10.5, halo_dict[args.halo], ha='left', va='center', fontsize=18)
            axs_regions[j].text(13,-12.75, '$r=%.2f R_{200}$' % (args.radius), ha='right', va='center', fontsize=18)
            axs_regions[j].text(13,-17, plot_labels[j], fontsize=18, ha='right', va='center')

            ax2 = axs_regions[j].twiny()
            axs_regions[j].tick_params(axis='both', which='both', direction='in', length=8, width=2, pad=5, labelsize=18, \
              top=False, right=True)
            ax2.tick_params(axis='x', which='both', direction='in', length=8, width=2, pad=5, labelsize=18, \
              top=True)
            x0, x1 = axs_regions[j].get_xlim()
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
            axs_regions[j].set_xlabel('Time [Gyr]', fontsize=18)
            axs_regions[j].set_ylabel('log Median Pressure [erg/cm$^3$]', fontsize=18)

            z_sfr, sfr = np.loadtxt(code_path + 'halo_infos/00' + args.halo + '/' + args.run + '/sfr', unpack=True, usecols=[1,2], skiprows=1)
            t_sfr = time_func(z_sfr)

            ax3 = axs_regions[j].twinx()
            ax3.plot(t_sfr, sfr, 'k-', lw=1)
            axs_regions[j].plot([timelist[0],timelist[-1]], [0,0], 'k-', lw=1, label='SFR (right axis)')
            ax3.tick_params(axis='y', which='both', direction='in', length=8, width=2, pad=5, labelsize=18, right=True)
            ax3.set_ylim(-5,200)
            ax3.set_ylabel('SFR [$M_\odot$/yr]', fontsize=18)

            fig_regions[j].subplots_adjust(left=0.13, bottom=0.12, right=0.87, top=0.89)
            fig_regions[j].savefig(save_dir + file_labels[j] + '_vs_time_regions-' + args.region_filter + save_suffix + '.png')
            plt.close(fig_regions[j])

def forces_vs_radius(snap):
    '''Plots different forces (thermal pressure, turbulent pressure, bulk inflow/outflow ram pressure, gravity, rotation, total)
    as functions of radius for the simulation output given by 'snap'.
    Also saves to file the statistics and pdfs of the distribution in each radial bin.
    Use --load_stats to skip the calculation step and only plot from file.'''

    tablename_prefix = output_dir + 'stats_halo_00' + args.halo + '/' + args.run + '/Tables/'
    if (args.load_stats):
        if (args.filename == ''):
            filename = ''
        else:
            filename = '_' + args.filename
        stats = Table.read(tablename_prefix + snap + '_stats_force-types' + args.filename + '.hdf5', path='all_data')
    masses_ind = np.where(masses['snapshot']==snap)[0]
    Menc_profile = IUS(np.concatenate(([0],masses['radius'][masses_ind])), np.concatenate(([0],masses['total_mass'][masses_ind])))
    Mvir = rvir_masses['total_mass'][rvir_masses['snapshot']==snap]
    Rvir = rvir_masses['radius'][rvir_masses['snapshot']==snap][0]

    if (not args.load_stats):
        if (args.system=='pleiades_cassi'):
            print('Copying directory to /tmp')
            snap_dir = '/tmp/' + args.halo + '/' + args.run + '/' + target_dir + '/' + snap
            if (args.copy_to_tmp):
                shutil.copytree(foggie_dir + run_dir + snap, snap_dir)
                snap_name = snap_dir + '/' + snap
            else:
                snap_dir = '/nobackup/clochhaa/tmp/' + args.halo + '/' + args.run + '/' + target_dir + '/' + snap
                # Make a dummy directory with the snap name so the script later knows the process running
                # this snapshot failed if the directory is still there
                os.makedirs(snap_dir)
                snap_name = foggie_dir + run_dir + snap + '/' + snap
        else:
            snap_name = foggie_dir + run_dir + snap + '/' + snap
        ds, refine_box = foggie_load(snap_name, trackname, do_filter_particles=False, halo_c_v_name=halo_c_v_name, gravity=True, masses_dir=masses_dir)
        zsnap = ds.get_parameter('CosmologyCurrentRedshift')

        # Define the density cut between disk and CGM to vary smoothly between 1 and 0.1 between z = 0.5 and z = 0.25,
        # with it being 1 at higher redshifts and 0.1 at lower redshifts
        current_time = ds.current_time.in_units('Myr').v
        if (current_time<=8656.88):
            density_cut_factor = 1.
        elif (current_time<=10787.12):
            density_cut_factor = 1. - 0.9*(current_time-8656.88)/2130.24
        else:
            density_cut_factor = 0.1

        pix_res = float(np.min(refine_box[('gas','dx')].in_units('kpc')))  # at level 11
        lvl1_res = pix_res*2.**11.
        level = 9
        dx = lvl1_res/(2.**level)
        smooth_scale = (25./dx)/6.
        dx_cm = dx*1000*cmtopc
        refine_res = int(3.*Rvir/dx)
        box = ds.covering_grid(level=level, left_edge=ds.halo_center_kpc-ds.arr([1.5*Rvir,1.5*Rvir,1.5*Rvir],'kpc'), dims=[refine_res, refine_res, refine_res])
        density = box['density'].in_units('g/cm**3').v
        temperature = box['temperature'].v
        x = box[('gas','x')].in_units('cm').v - ds.halo_center_kpc[0].to('cm').v
        y = box[('gas','y')].in_units('cm').v - ds.halo_center_kpc[1].to('cm').v
        z = box[('gas','z')].in_units('cm').v - ds.halo_center_kpc[2].to('cm').v
        r = box['radius_corrected'].in_units('cm').v
        radius = box['radius_corrected'].in_units('kpc').v
        x_hat = x/r
        y_hat = y/r
        z_hat = z/r

        # This next block needed for removing any ISM regions and then interpolating over the holes left behind
        if (args.cgm_only):
            # Define ISM regions to remove
            disk_mask = (density > cgm_density_max * density_cut_factor)
            # disk_mask_expanded is a binary mask of both ISM regions AND their surrounding pixels
            struct = ndimage.generate_binary_structure(3,3)
            disk_mask_expanded = ndimage.binary_dilation(disk_mask, structure=struct, iterations=3)
            disk_mask_expanded = ndimage.binary_closing(disk_mask_expanded, structure=struct, iterations=3)
            disk_mask_expanded = disk_mask_expanded | disk_mask
            # disk_edges is a binary mask of ONLY pixels surrounding ISM regions -- nothing inside ISM regions
            disk_edges = disk_mask_expanded & ~disk_mask
            x_edges = x[disk_edges].flatten()
            y_edges = y[disk_edges].flatten()
            z_edges = z[disk_edges].flatten()

            den_edges = density[disk_edges]
            den_interp_func = NearestNDInterpolator(list(zip(x_edges,y_edges,z_edges)), den_edges)
            den_masked = np.copy(density)
            den_masked[disk_mask] = den_interp_func(x[disk_mask], y[disk_mask], z[disk_mask])
        else:
            den_masked = density

        thermal_pressure = box['pressure'].in_units('erg/cm**3').v
        if (args.cgm_only):
            pres_edges = thermal_pressure[disk_edges]
            pres_interp_func = NearestNDInterpolator(list(zip(x_edges,y_edges,z_edges)), pres_edges)
            pres_masked = np.copy(thermal_pressure)
            pres_masked[disk_mask] = pres_interp_func(x[disk_mask], y[disk_mask], z[disk_mask])
        else:
            pres_masked = thermal_pressure
        pres_grad = np.gradient(pres_masked, dx_cm)
        dPdr = pres_grad[0]*x_hat + pres_grad[1]*y_hat + pres_grad[2]*z_hat
        thermal_force = -1./den_masked * dPdr
        vx = box['vx_corrected'].in_units('cm/s').v
        vy = box['vy_corrected'].in_units('cm/s').v
        vz = box['vz_corrected'].in_units('cm/s').v
        if (args.cgm_only):
            vx_edges = vx[disk_edges]
            vy_edges = vy[disk_edges]
            vz_edges = vz[disk_edges]
            vx_interp_func = NearestNDInterpolator(list(zip(x_edges,y_edges,z_edges)), vx_edges)
            vy_interp_func = NearestNDInterpolator(list(zip(x_edges,y_edges,z_edges)), vy_edges)
            vz_interp_func = NearestNDInterpolator(list(zip(x_edges,y_edges,z_edges)), vz_edges)
            vx_masked = np.copy(vx)
            vy_masked = np.copy(vy)
            vz_masked = np.copy(vz)
            vx_masked[disk_mask] = vx_interp_func(x[disk_mask], y[disk_mask], z[disk_mask])
            vy_masked[disk_mask] = vy_interp_func(x[disk_mask], y[disk_mask], z[disk_mask])
            vz_masked[disk_mask] = vz_interp_func(x[disk_mask], y[disk_mask], z[disk_mask])
        else:
            vx_masked = vx
            vy_masked = vy
            vz_masked = vz
        smooth_vx = gaussian_filter(vx_masked, smooth_scale)
        smooth_vy = gaussian_filter(vy_masked, smooth_scale)
        smooth_vz = gaussian_filter(vz_masked, smooth_scale)
        smooth_den = gaussian_filter(den_masked, smooth_scale)
        sig_x = (vx_masked - smooth_vx)**2.
        sig_y = (vy_masked - smooth_vy)**2.
        sig_z = (vz_masked - smooth_vz)**2.
        vdisp = np.sqrt((sig_x + sig_y + sig_z)/3.)
        turb_pressure = smooth_den*vdisp**2.
        pres_grad = np.gradient(turb_pressure, dx_cm)
        dPdr = pres_grad[0]*x_hat + pres_grad[1]*y_hat + pres_grad[2]*z_hat
        turb_force = -1./den_masked * dPdr
        vr = box['radial_velocity_corrected'].in_units('cm/s').v
        if (args.cgm_only):
            vr_edges = vr[disk_edges]
            vr_interp_func = NearestNDInterpolator(list(zip(x_edges,y_edges,z_edges)), vr_edges)
            vr_masked = np.copy(vr)
            vr_masked[disk_mask] = vr_interp_func(x[disk_mask], y[disk_mask], z[disk_mask])
        else:
            vr_masked = vr
        vr_masked = gaussian_filter(vr_masked, smooth_scale)
        dvr = np.gradient(vr_masked, dx_cm)
        delta_vr = dvr[0]*dx_cm*x_hat + dvr[1]*dx_cm*y_hat + dvr[2]*dx_cm*z_hat
        ram_pressure = smooth_den*(delta_vr)**2.
        pres_grad = np.gradient(ram_pressure, dx_cm)
        dPdr = pres_grad[0]*x_hat + pres_grad[1]*y_hat + pres_grad[2]*z_hat
        ram_force = -1./den_masked * dPdr
        vtheta = box['theta_velocity_corrected'].in_units('cm/s').v
        vphi = box['phi_velocity_corrected'].in_units('cm/s').v
        if (args.cgm_only):
            vtheta_edges = vtheta[disk_edges]
            vphi_edges = vphi[disk_edges]
            vtheta_interp_func = NearestNDInterpolator(list(zip(x_edges,y_edges,z_edges)), vtheta_edges)
            vtheta_masked = np.copy(vtheta)
            vphi_interp_func = NearestNDInterpolator(list(zip(x_edges,y_edges,z_edges)), vphi_edges)
            vphi_masked = np.copy(vphi)
            vtheta_masked[disk_mask] = vtheta_interp_func(x[disk_mask], y[disk_mask], z[disk_mask])
            vphi_masked[disk_mask] = vphi_interp_func(x[disk_mask], y[disk_mask], z[disk_mask])
        else:
            vtheta_masked = vtheta
            vphi_masked = vphi
        smooth_vtheta = gaussian_filter(vtheta_masked, smooth_scale)
        smooth_vphi = gaussian_filter(vphi_masked, smooth_scale)
        rot_force = (smooth_vtheta**2. + smooth_vphi**2.)/r
        grav_force = -G*Menc_profile(r/(1000*cmtopc))*gtoMsun/r**2.
        tot_force = thermal_force + turb_force + rot_force + ram_force + grav_force
        forces = [thermal_force, turb_force, ram_force, rot_force, grav_force, tot_force]

        density = density.flatten()
        temperature = temperature.flatten()
        radius = radius.flatten()
        for i in range(len(forces)):
            forces[i] = forces[i].flatten()

        if (args.cgm_only): radius = radius[(density < cgm_density_max * density_cut_factor)]
        if (args.weight=='mass'):
            weights = box['cell_mass'].in_units('g').v.flatten()
        if (args.weight=='volume'):
            weights = box['cell_volume'].in_units('kpc**3').v.flatten()
        if (args.cgm_only): weights = weights[(density < cgm_density_max * density_cut_factor)]
        if (args.region_filter=='metallicity'):
            metallicity = box['metallicity'].in_units('Zsun').v.flatten()
            if (args.cgm_only): metallicity = metallicity[(density < cgm_density_max * density_cut_factor)]
        if (args.region_filter=='velocity'):
            rv = box['radial_velocity_corrected'].in_units('km/s').v.flatten()
            vff = box['vff'].in_units('km/s').v.flatten()
            vesc = box['vesc'].in_units('km/s').v.flatten()
            if (args.cgm_only):
                rv = rv[(density < cgm_density_max * density_cut_factor)]
                vff = vff[(density < cgm_density_max * density_cut_factor)]
                vesc = vesc[(density < cgm_density_max * density_cut_factor)]

        if (args.cgm_only):
            for i in range(len(forces)):
                forces[i] = forces[i][(density < cgm_density_max * density_cut_factor)]
            new_density = density[(density < cgm_density_max * density_cut_factor)]
            new_temp = temperature[(density < cgm_density_max * density_cut_factor)]
            density = new_density
            temperature = new_temp

        stats = ['thermal_force', 'turbulent_force', 'ram_force', 'rotation_force', 'gravity_force', 'total_force']
        table = make_table(stats)
        table_pdf = make_pdf_table(stats)

        radius_list = np.linspace(0., 1.5*Rvir, 100)
        if (args.region_filter!='none'):
            force_regions = []
            for i in range(len(forces)):
                force_regions.append([])
            weights_regions = []
            radius_regions = []
        if (args.region_filter=='temperature'):
            regions = ['_low-T', '_mid-T', '_high-T']
            weights_regions.append(weights[temperature < 10**4.8])
            weights_regions.append(weights[(temperature > 10**4.8) & (temperature < 10**6.3)])
            weights_regions.append(weights[temperature > 10**6.3])
            radius_regions.append(radius[temperature < 10**4.8])
            radius_regions.append(radius[(temperature > 10**4.8) & (temperature < 10**6.3)])
            radius_regions.append(radius[temperature > 10**6.3])
            for i in range(len(forces)):
                force_regions[i].append(forces[i][temperature < 10**4.8])
                force_regions[i].append(forces[i][(temperature > 10**4.8) & (temperature < 10**6.3)])
                force_regions[i].append(forces[i][temperature > 10**6.3])
        elif (args.region_filter=='metallicity'):
            regions = ['_low-Z', '_mid-Z', '_high-Z']
            weights_regions.append(weights[metallicity < 0.01])
            weights_regions.append(weights[(metallicity > 0.01) & (metallicity < 1)])
            weights_regions.append(weights[metallicity > 1])
            radius_regions.append(radius[metallicity < 0.01])
            radius_regions.append(radius[(metallicity > 0.01) & (metallicity < 1)])
            radius_regions.append(radius[metallicity > 1])
            for i in range(len(forces)):
                force_regions[i].append(forces[i][metallicity < 0.01])
                force_regions[i].append(forces[i][(metallicity > 0.01) & (metallicity < 1)])
                force_regions[i].append(forces[i][metallicity > 1])
        elif (args.region_filter=='velocity'):
            regions = ['_low-v', '_mid-v', '_high-v']
            weights_regions.append(weights[rv < 0.5*vff])
            weights_regions.append(weights[(rv > 0.5*vff) & (rv < vesc)])
            weights_regions.append(weights[rv > vesc])
            radius_regions.append(radius[rv < 0.5*vff])
            radius_regions.append(radius[(rv > 0.5*vff) & (rv < vesc)])
            radius_regions.append(radius[rv > vesc])
            for i in range(len(forces)):
                force_regions[i].append(forces[i][rv < 0.5*vff])
                force_regions[i].append(forces[i][(rv > 0.5*vff) & (rv < vesc)])
                force_regions[i].append(forces[i][rv > vesc])
        else:
            regions = []

        for i in range(len(radius_list)-1):
            row = [zsnap, radius_list[i], radius_list[i+1]]
            pdf_array = []
            for j in range(len(forces)):
                force_shell = forces[j][(radius >= radius_list[i]) & (radius < radius_list[i+1])]
                weights_shell = weights[(radius >= radius_list[i]) & (radius < radius_list[i+1])]
                if (len(force_shell)!=0.):
                    quantiles = weighted_quantile(force_shell, weights_shell, np.array([0.25,0.5,0.75]))
                    row.append(quantiles[1])
                    row.append(quantiles[2]-quantiles[0])
                    avg, std = weighted_avg_and_std(force_shell, weights_shell)
                    row.append(avg)
                    row.append(std)
                    row.append(np.sum(force_shell*weights_shell))
                    row.append(np.sum(weights_shell))
                    hist_pos, bin_edges_pos = np.histogram(np.log10(force_shell[force_shell>=1e-9]), \
                      weights=weights_shell[force_shell>=1e-9], bins=(100), range=[-9, -5])
                    hist_mid, bin_edges_mid = np.histogram(force_shell[(force_shell>=-1e-9) & (force_shell<=1e-9)], \
                      weights=weights_shell[(force_shell>=-1e-9) & (force_shell<=1e-9)], bins=(100), range=[-1e-9, 1e-9])
                    hist_neg, bin_edges_neg = np.histogram(np.log10(-force_shell[force_shell<=-1e-9]), \
                      weights=weights_shell[force_shell<=-1e-9], bins=(100), range=[-9, -5])
                    norm = np.sum([hist_neg, hist_mid, hist_pos])
                    bin_edges_neg = np.flip(bin_edges_neg)
                    hist_neg = np.flip(hist_neg)
                    bin_edges = np.hstack([10**bin_edges_neg[:-1], bin_edges_mid, 10**bin_edges_pos[1:]])
                    hist = np.hstack([hist_neg, hist_mid, hist_pos])/norm
                    pdf_array.append(bin_edges[:-1])
                    pdf_array.append(bin_edges[1:])
                    pdf_array.append(hist)
                    for k in range(len(regions)):
                        force_shell = force_regions[j][k][(radius_regions[k] >= radius_list[i]) & (radius_regions[k] < radius_list[i+1])]
                        weights_shell = weights_regions[k][(radius_regions[k] >= radius_list[i]) & (radius_regions[k] < radius_list[i+1])]
                        if (len(force_shell)!=0.):
                            quantiles = weighted_quantile(force_shell, weights_shell, np.array([0.25,0.5,0.75]))
                            row.append(quantiles[1])
                            row.append(quantiles[2]-quantiles[0])
                            avg, std = weighted_avg_and_std(force_shell, weights_shell)
                            row.append(avg)
                            row.append(std)
                            row.append(np.sum(force_shell*weights_shell))
                            row.append(np.sum(weights_shell))
                            hist_pos, bin_edges_pos = np.histogram(np.log10(force_shell[force_shell>=1e-9]), \
                              weights=weights_shell[force_shell>=1e-9], bins=(100), range=[-9, -5])
                            hist_mid, bin_edges_mid = np.histogram(force_shell[(force_shell>=-1e-9) & (force_shell<=1e-9)], \
                              weights=weights_shell[(force_shell>=-1e-9) & (force_shell<=1e-9)], bins=(100), range=[-1e-9, 1e-9])
                            hist_neg, bin_edges_neg = np.histogram(np.log10(-force_shell[force_shell<=-1e-9]), \
                              weights=weights_shell[force_shell<=-1e-9], bins=(100), range=[-9, -5])
                            norm = np.sum([hist_neg, hist_mid, hist_pos])
                            hist_neg = np.flip(hist_neg)
                            hist = np.hstack([hist_neg, hist_mid, hist_pos])/norm
                            pdf_array.append(hist)
                        else:
                            row.append(0.)
                            row.append(0.)
                            row.append(0.)
                            row.append(0.)
                            row.append(0.)
                            row.append(0.)
                            pdf_array.append(np.zeros(300))
                else:
                    row.append(0.)
                    row.append(0.)
                    row.append(0.)
                    row.append(0.)
                    row.append(0.)
                    row.append(0.)
                    pdf_array.append(np.zeros(300))
                    pdf_array.append(np.zeros(300))
                    pdf_array.append(np.zeros(300))
                    for k in range(len(regions)):
                        row.append(0.)
                        row.append(0.)
                        row.append(0.)
                        row.append(0.)
                        row.append(0.)
                        row.append(0.)
                        pdf_array.append(np.zeros(300))

            table.add_row(row)
            pdf_array = np.vstack(pdf_array)
            pdf_array = np.transpose(pdf_array)
            for p in range(len(pdf_array)):
                row_pdf = [zsnap, radius_list[i], radius_list[i+1]]
                row_pdf += list(pdf_array[p])
                table_pdf.add_row(row_pdf)

        table = set_table_units(table)
        table_pdf = set_table_units(table_pdf)

        # Save to file
        table.write(tablename_prefix + snap + '_stats_force-types' + save_suffix + '.hdf5', path='all_data', serialize_meta=True, overwrite=True)
        table_pdf.write(tablename_prefix + snap + '_stats_force-types_pdf' + save_suffix + '.hdf5', path='all_data', serialize_meta=True, overwrite=True)

        stats = table
        print("Stats have been calculated and saved to file for snapshot " + snap + "!")
        # Delete output or dummy directory from temp directory if on pleiades
        if (args.system=='pleiades_cassi'):
            print('Deleting directory from /tmp')
            shutil.rmtree(snap_dir)


    plot_colors = ['r', 'g', 'm', 'b', 'gold', 'k']
    plot_labels = ['Thermal', 'Turbulent', 'Ram', 'Rotation', 'Gravity', 'Total']
    file_labels = ['thermal_force', 'turbulent_force', 'ram_force', 'rotation_force', 'gravity_force', 'total_force']
    linestyles = ['-', '--', ':', '-.', '--', '-']

    radius_list = 0.5*(stats['inner_radius'] + stats['outer_radius'])
    zsnap = stats['redshift'][0]

    fig = plt.figure(figsize=(8,6), dpi=500)
    ax = fig.add_subplot(1,1,1)

    for i in range(len(plot_colors)):
        label = plot_labels[i]
        if (args.normalized):
            ax.plot(radius_list, stats[file_labels[i] + '_sum']/stats[file_labels[i] + '_weight_sum'], ls=linestyles[i], color=plot_colors[i], \
                    lw=2, label=label)
        else:
            ax.plot(radius_list, stats[file_labels[i] + '_sum'], ls=linestyles[i], color=plot_colors[i], \
                    lw=2, label=label)

    if (args.normalized):
        ax.set_ylabel('Force on CGM gas [cm/s$^2$]', fontsize=20)
        ax.axis([0,250,-1e-5,1e-5])
        ax.set_yscale('symlog', linthreshy=1e-9)
        ax.text(15, -3e-6, '$z=%.2f$' % (zsnap), fontsize=20, ha='left', va='center')
        ax.text(15,-3e-7,halo_dict[args.halo],ha='left',va='center',fontsize=20)
        ax.text(Rvir-3., -3e-6, '$R_{200}$', fontsize=20, ha='right', va='center')
    else:
        ax.set_ylabel('Force on CGM gas [$M_\odot$ cm/s$^2$]', fontsize=20)
        ax.axis([0,250,-10,100])
        ax.set_yscale('symlog', linthreshy=1e-2)
        ax.text(15, -3, '$z=%.2f$' % (zsnap), fontsize=20, ha='left', va='center')
        ax.text(15,3,halo_dict[args.halo],ha='left',va='center',fontsize=20)
        ax.text(Rvir-3., -3, '$R_{200}$', fontsize=20, ha='right', va='center')
    ax.set_xlabel('Radius [kpc]', fontsize=20)
    ax.tick_params(axis='both', which='both', direction='in', length=8, width=2, pad=5, labelsize=20, \
      top=True, right=True)
    ax.plot([ax.get_xlim()[0], ax.get_xlim()[1]], [0,0], 'k-', lw=1)
    ax.plot([Rvir, Rvir], [ax.get_ylim()[0], ax.get_ylim()[1]], 'k--', lw=1)
    ax.legend(loc=1, frameon=False, fontsize=20, ncol=2)
    fig.subplots_adjust(top=0.96,bottom=0.12,right=0.96,left=0.18)
    fig.savefig(save_dir + snap + '_forces_vs_r' + save_suffix + '.png')
    plt.close(fig)

    if (args.region_filter!='none'):
        regions = ['low_', 'mid_', 'high_']
        alphas = [0.25,0.6,1.]
        # Fig1 is a plot of all force types for all regions on one plot
        fig1 = plt.figure(figsize=(8,6), dpi=500)
        ax1 = fig1.add_subplot(1,1,1)
        # Fig2 is plots of all force types for each region, one per region
        figs2 = []
        axs2 = []
        for r in range(len(regions)):
            figs2.append(plt.figure(figsize=(8,6), dpi=500))
            axs2.append(figs2[-1].add_subplot(1,1,1))

        for i in range(len(plot_colors)):
            # Fig3 is plots of all regions for each force type, one per force type
            fig3 = plt.figure(figsize=(8,6), dpi=500)
            ax3 = fig3.add_subplot(1,1,1)
            label = plot_labels[i]
            label_regions = ['Low ' + args.region_filter, 'Mid ' + args.region_filter, '$0.1\\times$ High ' + args.region_filter]
            mult_regions = [1., 1., 0.1]
            if (i==0):
                label_regions_bigplot = ['Low ' + args.region_filter, 'Mid ' + args.region_filter, '$0.1\\times$ High ' + args.region_filter]
            else:
                label_regions_bigplot = ['__nolegend__', '__nolegend__', '__nolegend__']
            for j in range(len(regions)):
                if (args.normalized):
                    ax1.plot(radius_list, mult_regions[j]*stats[regions[j] + args.region_filter + '_' + file_labels[i] + '_sum']/ \
                      stats[regions[j] + args.region_filter + '_' + file_labels[i] + '_weight_sum'], \
                      ls=linestyles[i], color=plot_colors[i], lw=2, alpha=alphas[j], label=label_regions_bigplot[j])
                    axs2[j].plot(radius_list, mult_regions[j]*stats[regions[j] + args.region_filter + '_' + file_labels[i] + '_sum']/ \
                      stats[regions[j] + args.region_filter + '_' + file_labels[i] + '_weight_sum'], \
                      ls=linestyles[i], color=plot_colors[i], lw=2, label=label)
                    ax3.plot(radius_list, mult_regions[j]*stats[regions[j] + args.region_filter + '_' + file_labels[i] + '_sum']/ \
                      stats[regions[j] + args.region_filter + '_' + file_labels[i] + '_weight_sum'], \
                      ls=linestyles[i], color=plot_colors[i], lw=2, alpha=alphas[j], label=label_regions[j])
                else:
                    ax1.plot(radius_list, mult_regions[j]*stats[regions[j] + args.region_filter + '_' + file_labels[i] + '_sum'], \
                      ls=linestyles[i], color=plot_colors[i], lw=2, alpha=alphas[j], label=label_regions_bigplot[j])
                    axs2[j].plot(radius_list, mult_regions[j]*stats[regions[j] + args.region_filter + '_' + file_labels[i] + '_sum'], \
                      ls=linestyles[i], color=plot_colors[i], lw=2, label=label)
                    ax3.plot(radius_list, mult_regions[j]*stats[regions[j] + args.region_filter + '_' + file_labels[i] + '_sum'], \
                      ls=linestyles[i], color=plot_colors[i], lw=2, alpha=alphas[j], label=label_regions[j])
            if (args.normalized):
                ax1.plot(radius_list, mult_regions[2]*stats['high_' + args.region_filter + '_' + file_labels[i] + '_sum']/ \
                  stats['high_' + args.region_filter + '_' + file_labels[i] + '_weight_sum'], \
                  ls=linestyles[i], color=plot_colors[i], lw=2, label=label)
                ax3.plot(radius_list, mult_regions[2]*stats['high_' + args.region_filter + '_' + file_labels[i] + '_sum']/ \
                  stats['high_' + args.region_filter + '_' + file_labels[i] + '_weight_sum'], \
                  ls=linestyles[i], color=plot_colors[i], lw=2, label=label)
            else:
                ax1.plot(radius_list, mult_regions[2]*stats['high_' + args.region_filter + '_' + file_labels[i] + '_sum'], \
                  ls=linestyles[i], color=plot_colors[i], lw=2, label=label)
                ax3.plot(radius_list, mult_regions[2]*stats['high_' + args.region_filter + '_' + file_labels[i] + '_sum'], \
                  ls=linestyles[i], color=plot_colors[i], lw=2, label=label)

            if (args.normalized):
                ax3.set_ylabel('Force on CGM gas [cm/s$^2$]', fontsize=20)
                ax3.axis([0,250,-1e-5,1e-5])
                ax3.set_yscale('symlog', linthreshy=1e-9)
                ax3.text(Rvir-3., -3e-6, '$R_{200}$', fontsize=20, ha='right', va='center')
                ax3.text(15,-3e-7,halo_dict[args.halo],ha='left',va='center',fontsize=20)
                ax3.text(15, -3e-6, '$z=%.2f$' % (zsnap), fontsize=20, ha='left', va='center')
            else:
                ax3.set_ylabel('Force on CGM gas [$M_\odot$ cm/s$^2$]', fontsize=20)
                ax3.axis([0,250,-10,100])
                ax3.set_yscale('symlog', linthreshy=1e-2)
                ax3.text(Rvir-3., -3, '$R_{200}$', fontsize=20, ha='right', va='center')
                ax3.text(15,3,halo_dict[args.halo],ha='left',va='center',fontsize=20)
                ax3.text(15, -3, '$z=%.2f$' % (zsnap), fontsize=20, ha='left', va='center')
            ax3.set_xlabel('Radius [kpc]', fontsize=20)
            ax3.tick_params(axis='both', which='both', direction='in', length=8, width=2, pad=5, labelsize=20, \
              top=True, right=True)
            ax3.plot([Rvir, Rvir], [ax.get_ylim()[0], ax.get_ylim()[1]], 'k--', lw=1)
            ax3.plot([ax.get_xlim()[0], ax.get_xlim()[1]], [0,0], 'k-', lw=1)
            ax3.legend(loc=1, frameon=False, fontsize=20)
            fig3.subplots_adjust(top=0.96,bottom=0.12,right=0.96,left=0.18)
            fig3.savefig(save_dir + snap + '_' + file_labels[i] + '_vs_r_regions-' + args.region_filter + save_suffix + '.png')
            plt.close(fig3)

        for r in range(len(regions)):
            if (args.normalized):
                axs2[r].set_ylabel('Force on CGM gas [cm/s$^2$]', fontsize=20)
                axs2[r].axis([0,250,-1e-5,1e-5])
                axs2[r].set_yscale('symlog', linthreshy=1e-9)
                axs2[r].text(Rvir-3., -3e-6, '$R_{200}$', fontsize=20, ha='right', va='center')
                axs2[r].text(15,-3e-7,halo_dict[args.halo],ha='left',va='center',fontsize=20)
                axs2[r].text(15, -3e-6, '$z=%.2f$' % (zsnap), fontsize=20, ha='left', va='center')
                axs2[r].text(240, 5e-10, label_regions[r], fontsize=20, ha='right', va='center')
            else:
                axs2[r].set_ylabel('Force on CGM gas [$M_\odot$ cm/s$^2$]', fontsize=20)
                axs2[r].axis([0,250,-10,100])
                axs2[r].set_yscale('symlog', linthreshy=1e-2)
                axs2[r].text(Rvir-3., -3, '$R_{200}$', fontsize=20, ha='right', va='center')
                axs2[r].text(15,3,halo_dict[args.halo],ha='left',va='center',fontsize=20)
                axs2[r].text(15, -3, '$z=%.2f$' % (zsnap), fontsize=20, ha='left', va='center')
                axs2[r].text(240, 5e-3, label_regions[r], fontsize=20, ha='right', va='center')
            axs2[r].set_xlabel('Radius [kpc]', fontsize=20)
            axs2[r].tick_params(axis='both', which='both', direction='in', length=8, width=2, pad=5, labelsize=20, \
              top=True, right=True)
            axs2[r].plot([Rvir, Rvir], [ax.get_ylim()[0], ax.get_ylim()[1]], 'k--', lw=1)
            axs2[r].plot([ax.get_xlim()[0], ax.get_xlim()[1]], [0,0], 'k-', lw=1)
            axs2[r].legend(loc=1, frameon=False, fontsize=20, ncol=2)
            figs2[r].subplots_adjust(top=0.96,bottom=0.12,right=0.96,left=0.18)
            figs2[r].savefig(save_dir + snap + '_forces_vs_r_' + regions[r] + args.region_filter + save_suffix + '.png')
            plt.close(figs2[r])

        if (args.normalized):
            ax1.set_ylabel('Force on CGM gas [cm/s$^2$]', fontsize=20)
            ax1.axis([0,250,-1e-5,1e-5])
            ax1.set_yscale('symlog', linthreshy=1e-9)
            ax1.text(15, -3e-6, '$z=%.2f$' % (zsnap), fontsize=20, ha='left', va='center')
            ax1.text(15,-3e-7,halo_dict[args.halo],ha='left',va='center',fontsize=20)
            ax1.text(Rvir-3., -3e-6, '$R_{200}$', fontsize=20, ha='right', va='center')
        else:
            ax1.set_ylabel('Force on CGM gas [$M_\odot$ cm/s$^2$]', fontsize=20)
            ax1.axis([0,250,-10,100])
            ax1.set_yscale('symlog', linthreshy=1e-2)
            ax1.text(15, -3, '$z=%.2f$' % (zsnap), fontsize=20, ha='left', va='center')
            ax1.text(15,3,halo_dict[args.halo],ha='left',va='center',fontsize=20)
            ax1.text(Rvir-3., -3, '$R_{200}$', fontsize=20, ha='right', va='center')
        ax1.set_xlabel('Radius [kpc]', fontsize=20)
        ax1.tick_params(axis='both', which='both', direction='in', length=8, width=2, pad=5, labelsize=20, \
          top=True, right=True)
        ax1.plot([Rvir, Rvir], [ax.get_ylim()[0], ax.get_ylim()[1]], 'k--', lw=1)
        ax1.plot([ax.get_xlim()[0], ax.get_xlim()[1]], [0,0], 'k-', lw=1)
        ax1.legend(loc=1, frameon=False, fontsize=14, ncol=2)
        fig1.subplots_adjust(top=0.96,bottom=0.12,right=0.96,left=0.18)
        fig1.savefig(save_dir + snap + '_all_forces_vs_r_regions-' + args.region_filter + save_suffix + '.png')
        plt.close(fig1)

def forces_vs_radius_time_averaged(snaplist):
    '''Plots time-averaged forces as a function of radius, with shading showing the time variation of
    the forces.'''

    tablename_prefix = output_dir + 'stats_halo_00' + args.halo + '/' + args.run + '/Tables/'
    if (args.filename == ''):
        filename = ''
    else:
        filename = '_' + args.filename

    plot_colors = ['r', 'g', 'm', 'b', 'gold', 'k']
    plot_labels = ['Thermal', 'Turbulent', 'Ram', 'Rotation', 'Gravity', 'Total']
    file_labels = ['thermal_force', 'turbulent_force', 'ram_force', 'rotation_force', 'gravity_force', 'total_force']
    linestyles = ['-', '--', ':', '-.', '--', '-']

    fig = plt.figure(figsize=(8,6), dpi=500)
    ax = fig.add_subplot(1,1,1)

    zlist = []
    timelist = []
    forces_list = []
    if (args.region_filter!='none'):
        forces_regions = [[],[],[]]
        avg_forces_regions = [[],[],[]]
        std_forces_regions = [[],[],[]]
        region_label = ['Low ' + args.region_filter, 'Mid ' + args.region_filter, 'High ' + args.region_filter]
        region_name = ['low-', 'mid-', 'high-']
    for j in range(len(plot_labels)):
        forces_list.append([])
        if (args.region_filter!='none'):
            forces_regions[0].append([])
            forces_regions[1].append([])
            forces_regions[2].append([])

    for i in range(len(snaplist)):
        snap = snaplist[i]
        stats = Table.read(tablename_prefix + snap + '_stats_force-types' + filename + '.hdf5', path='all_data')
        Rvir = rvir_masses['radius'][rvir_masses['snapshot']==snap][0]
        radius_list = 0.5*(stats['inner_radius'] + stats['outer_radius'])/Rvir

        for j in range(len(file_labels)):
            if (args.normalized):
                forces_list[j].append(stats[file_labels[j] + '_sum']/stats[file_labels[j] + '_weight_sum'])
            else:
                forces_list[j].append(stats[file_labels[j] + '_sum']/gtoMsun)
            if (args.region_filter!='none'):
                if (args.normalized):
                    forces_regions[0][j].append(stats['low_' + args.region_filter + '_' + file_labels[j] + '_sum'] / \
                      stats['low_' + args.region_filter + '_' + file_labels[j] + '_weight_sum'])
                    forces_regions[1][j].append(stats['mid_' + args.region_filter + '_' + file_labels[j] + '_sum'] / \
                      stats['mid_' + args.region_filter + '_' + file_labels[j] + '_weight_sum'])
                    forces_regions[2][j].append(stats['high_' + args.region_filter + '_' + file_labels[j] + '_sum'] / \
                      stats['high_' + args.region_filter + '_' + file_labels[j] + '_weight_sum'])
                else:
                    forces_regions[0][j].append(stats['low_' + args.region_filter + '_' + file_labels[j] + '_sum']/gtoMsun)
                    forces_regions[1][j].append(stats['mid_' + args.region_filter + '_' + file_labels[j] + '_sum']/gtoMsun)
                    forces_regions[2][j].append(stats['high_' + args.region_filter + '_' + file_labels[j] + '_sum']/gtoMsun)

    avg_forces_list = np.nanmean(forces_list, axis=1)
    std_forces_list = np.nanstd(forces_list, axis=1)
    if (args.region_filter!='none'):
        avg_forces_regions[0] = np.nanmean(forces_regions[0], axis=1)
        avg_forces_regions[1] = np.nanmean(forces_regions[1], axis=1)
        avg_forces_regions[2] = np.nanmean(forces_regions[2], axis=1)
        std_forces_regions[0] = np.nanmean(forces_regions[0], axis=1)
        std_forces_regions[1] = np.nanmean(forces_regions[1], axis=1)
        std_forces_regions[2] = np.nanmean(forces_regions[2], axis=1)

    for i in range(len(plot_colors)):
        label = plot_labels[i]
        ax.plot(radius_list, avg_forces_list[i], ls=linestyles[i], color=plot_colors[i], \
                lw=2, label=label)
        ax.fill_between(radius_list, avg_forces_list[i]-std_forces_list[i], \
                        avg_forces_list[i]+std_forces_list[i], alpha=0.2, color=plot_colors[i])

    if (args.normalized):
        ax.set_ylabel('Force on CGM gas [cm/s$^2$]', fontsize=14)
        ax.axis([0,1.5,-1e-5,1e-5])
        ax.set_yscale('symlog', linthreshy=1e-9)
        ax.text(1.4, -1e-6, halo_dict[args.halo], ha='right', va='center', fontsize=14)
    else:
        ax.set_ylabel('Force on CGM gas [$M_\odot$ cm/s$^2$]', fontsize=14)
        ax.axis([0,1.5,-10,100])
        ax.set_yscale('symlog', linthreshy=1e-2)
        ax.text(1.4, -9, halo_dict[args.halo], ha='right', va='center', fontsize=14)
    ax.set_xlabel('Radius [$R_{200}$]', fontsize=14)
    ax.tick_params(axis='both', which='both', direction='in', length=8, width=2, pad=5, labelsize=14, \
      top=True, right=True)
    ax.plot([ax.get_xlim()[0], ax.get_xlim()[1]], [0,0], 'k-', lw=1)
    ax.legend(loc=1, frameon=False, fontsize=14)
    fig.subplots_adjust(top=0.96,bottom=0.09,right=0.96,left=0.13)
    fig.savefig(save_dir + snap + '_forces_vs_r_time-avg' + save_suffix + '.png')
    plt.close(fig)

    if (args.region_filter!='none'):
        regions = ['low_', 'mid_', 'high_']
        label_regions = ['Low ' + args.region_filter, 'Mid ' + args.region_filter, 'High ' + args.region_filter]
        # Fig2 is plots of all force types for each region, one per region
        figs2 = []
        axs2 = []
        for r in range(len(regions)):
            figs2.append(plt.figure(figsize=(8,6), dpi=500))
            axs2.append(figs2[-1].add_subplot(1,1,1))

        for i in range(len(plot_colors)):
            label = plot_labels[i]
            for j in range(len(regions)):
                axs2[j].plot(radius_list, avg_forces_regions[j][i], \
                  ls=linestyles[i], color=plot_colors[i], lw=2, label=label)
                axs2[j].fill_between(radius_list, avg_forces_regions[j][i]-std_forces_regions[j][i], \
                                     avg_forces_regions[j][i]+std_forces_regions[j][i], color=plot_colors[i], alpha=0.2)

        for r in range(len(regions)):
            if (args.normalized):
                axs2[r].set_ylabel('Net Force on Shell [cm/s$^2$]', fontsize=14)
                axs2[r].axis([0,1.5,-1e-5,1e-5])
                axs2[r].text(1.4, -1e-7, label_regions[r], ha='right', va='center', fontsize=14)
                axs2[r].set_yscale('symlog', linthreshy=1e-9)
                axs2[r].text(1.4, -1e-6, halo_dict[args.halo], ha='right', va='center', fontsize=14)
            else:
                axs2[r].set_ylabel('Net Force on Shell [$M_\odot$ cm/s$^2$]', fontsize=14)
                axs2[r].axis([0,1.5,-10,100])
                axs2[r].set_yscale('symlog', linthreshy=1e-2)
                axs2[r].text(1.4, -7, label_regions[r], ha='right', va='center', fontsize=14)
                axs2[r].text(1.4, -9, halo_dict[args.halo], ha='right', va='center', fontsize=14)
            axs2[r].set_xlabel('Radius [$R_{200}$]', fontsize=14)
            axs2[r].tick_params(axis='both', which='both', direction='in', length=8, width=2, pad=5, labelsize=14, \
              top=True, right=True)
            axs2[r].legend(loc=1, frameon=False, fontsize=14)
            figs2[r].subplots_adjust(top=0.96,bottom=0.09,right=0.96,left=0.13)
            figs2[r].savefig(save_dir + snap + '_forces_vs_r_time-avg_' + regions[r] + args.region_filter + save_suffix + '.png')
            plt.close(figs2[r])

def forces_vs_radius_from_med_pressures(snap):
    '''Plots various forces as function of radius, but rather than computing forces cell-by-cell and
    summing within radial shells, computes forces from the gradient of the median pressure profile.
    Requires pressures to have already been calculated and saved to file, passed in with --filename.'''

    tablename_prefix = output_dir + 'stats_halo_00' + args.halo + '/' + args.run + '/Tables/'
    if (args.filename==''):
        filename = ''
    else:
        filename = '_' + args.filename
    stats = Table.read(tablename_prefix + snap + '_stats_pressure-types' + args.filename + '.hdf5', path='all_data')
    den_stats = Table.read(tablename_prefix + snap + '_stats_pressure_density_sphere_mass-weighted_vcut-p5vff_cgm-filtered.hdf5', path='all_data')
    Rvir = rvir_masses['radius'][rvir_masses['snapshot']==snap][0]

    den_profile = IUS(0.5*(den_stats['inner_radius']+den_stats['outer_radius']), 10**den_stats['net_log_density_med'])

    plot_colors = ['r', 'g', 'm']
    plot_labels = ['Thermal', 'Turbulent', 'Ram']
    file_labels = ['thermal_pressure', 'turbulent_pressure', 'ram_pressure']
    linestyles = ['-', '--', ':']

    radius_list = 0.5*(stats['inner_radius'] + stats['outer_radius'])
    radius_list_cm = radius_list*cmtopc*1000
    zsnap = stats['redshift'][0]

    fig = plt.figure(figsize=(8,6), dpi=500)
    ax = fig.add_subplot(1,1,1)

    for i in range(len(plot_colors)):
        label = plot_labels[i]
        pressure = 10**stats[file_labels[i] + '_med']
        dPdr = np.diff(pressure)/np.diff(radius_list)
        force = -1.*dPdr
        ax.plot(radius_list[:-1], force, ls=linestyles[i], color=plot_colors[i], \
                lw=2, label=label)

    ax.set_ylabel('$-\\nabla P$ [erg/cm$^3$/kpc]', fontsize=18)
    ax.axis([0,250,-1e-15,1e-13])
    ax.set_yscale('symlog', linthreshy=1e-16)
    #ax.text(15, -3e-6, '$z=%.2f$' % (zsnap), fontsize=18, ha='left', va='center')
    #ax.text(Rvir-3., -3e-6, '$R_{200}$', fontsize=18, ha='right', va='center')
    ax.set_xlabel('Radius [kpc]', fontsize=18)
    ax.tick_params(axis='both', which='both', direction='in', length=8, width=2, pad=5, labelsize=18, \
      top=True, right=True)
    ax.plot([ax.get_xlim()[0], ax.get_xlim()[1]], [0,0], 'k-', lw=1)
    ax.plot([Rvir, Rvir], [ax.get_ylim()[0], ax.get_ylim()[1]], 'k--', lw=1)
    ax.legend(loc=1, frameon=False, fontsize=14)
    fig.subplots_adjust(top=0.94,bottom=0.11,right=0.95,left=0.17)
    plt.savefig(save_dir + snap + '_grad_med_pressure_vs_r' + save_suffix + '.png')
    plt.close()

    if (args.region_filter!='none'):
        fig = plt.figure(figsize=(8,6), dpi=500)
        ax = fig.add_subplot(1,1,1)

        for i in range(len(plot_colors)):
            label = plot_labels[i]
            if (i==0): label_regions = ['Low ' + args.region_filter, 'Mid ' + args.region_filter, 'High ' + args.region_filter]
            else: label_regions = ['__nolegend__', '__nolegend__', '__nolegend__']
            pressure = 10**stats['low_' + args.region_filter + '_' + file_labels[i] + '_med']
            dPdr = np.diff(pressure)/np.diff(radius_list)
            force = -1.*dPdr
            ax.plot(radius_list[:-1], force, ls=linestyles[i], color=plot_colors[i], \
                    lw=2, alpha=0.25, label=label_regions[0])
            pressure = 10**stats['mid_' + args.region_filter + '_' + file_labels[i] + '_med']
            dPdr = np.diff(pressure)/np.diff(radius_list)
            force = -1.*dPdr
            ax.plot(radius_list[:-1], force, ls=linestyles[i], color=plot_colors[i], \
                    lw=2, alpha=0.5, label=label_regions[1])
            pressure = 10**stats['high_' + args.region_filter + '_' + file_labels[i] + '_med']
            dPdr = np.diff(pressure)/np.diff(radius_list)
            force = -1.*dPdr
            ax.plot(radius_list[:-1], force, ls=linestyles[i], color=plot_colors[i], \
                    lw=2, label=label_regions[2])
            ax.plot(radius_list[:-1], force, ls=linestyles[i], color=plot_colors[i], \
                    lw=2, label=label)

        #ax.set_ylabel('Force on CGM gas [cm/s$^2$]', fontsize=18)
        #ax.axis([0,250,-1e-5,1e-5])
        #ax.set_yscale('symlog', linthreshy=1e-9)
        ax.set_ylabel('$-\\nabla P$ [erg/cm$^3$/kpc]', fontsize=18)
        ax.axis([0,250,-1e-15,1e-13])
        ax.set_yscale('symlog', linthreshy=1e-16)
        #ax.text(15, -3e-6, '$z=%.2f$' % (zsnap), fontsize=18, ha='left', va='center')
        #ax.text(Rvir-3., -3e-6, '$R_{200}$', fontsize=18, ha='right', va='center')
        ax.set_xlabel('Radius [kpc]', fontsize=18)
        ax.tick_params(axis='both', which='both', direction='in', length=8, width=2, pad=5, labelsize=18, \
          top=True, right=True)
        ax.plot([ax.get_xlim()[0], ax.get_xlim()[1]], [0,0], 'k-', lw=1)
        ax.plot([Rvir, Rvir], [ax.get_ylim()[0], ax.get_ylim()[1]], 'k--', lw=1)
        ax.legend(loc=1, frameon=False, fontsize=14)
        fig.subplots_adjust(top=0.94,bottom=0.11,right=0.95,left=0.17)
        plt.savefig(save_dir + snap + '_grad_med_pressure_vs_r_regions-' + args.region_filter + save_suffix + '.png')
        plt.close()

def forces_vs_time(snaplist):
    '''Plots different forces at a given radius or averaged over a range of radii over time, for all
    snaps in the list snaplist.'''

    tablename_prefix = output_dir + 'stats_halo_00' + args.halo + '/' + args.run + '/Tables/'
    time_table = Table.read(output_dir + 'times_halo_00' + args.halo + '/' + args.run + '/time_table.hdf5', path='all_data')
    if (args.filename != ''): filename = '_' + args.filename
    else: filename = ''

    fig = plt.figure(figsize=(8,6), dpi=500)
    ax = fig.add_subplot(1,1,1)

    plot_colors = ['r', 'g', 'm', 'b', 'gold', 'k']
    plot_labels = ['Thermal', 'Turbulent', 'Ram', 'Rotation', 'Gravity', 'Total']
    file_labels = ['thermal_force', 'turbulent_force', 'ram_force', 'rotation_force', 'gravity_force', 'total_force']
    linestyles = ['-', '--', ':', '-.', '--', '-']
    alphas = [0.3, 0.6, 1.]

    if (args.time_avg!=0):
        dt = 5.38*args.output_step
        avg_window = int(np.ceil(args.time_avg/dt))

    if (args.radius_range!='none'):
        radius_range = ast.literal_eval(args.radius_range)

    zlist = []
    timelist = []
    forces_list = []
    if (args.region_filter!='none'):
        forces_regions = [[],[],[]]
        region_label = ['Low ' + args.region_filter, 'Mid ' + args.region_filter, 'High ' + args.region_filter]
        region_name = ['low-', 'mid-', 'high-']
    for j in range(len(plot_labels)):
        forces_list.append([])
        if (args.region_filter!='none'):
            forces_regions[0].append([])
            forces_regions[1].append([])
            forces_regions[2].append([])

    for i in range(len(snaplist)):
        snap = snaplist[i]
        stats = Table.read(tablename_prefix + snap + '_stats_force-types' + filename + '.hdf5', path='all_data')
        Rvir = rvir_masses['radius'][rvir_masses['snapshot']==snap][0]
        if (args.radius_range!='none'):
            radius_in = radius_range[0]*Rvir
            radius_out = radius_range[1]*Rvir
            rad_in = np.where(stats['inner_radius']<=radius_in)[0][-1]
            rad_out = np.where(stats['outer_radius']>=radius_out)[0][0]
        else:
            radius = args.radius*Rvir
            rad_ind = np.where(stats['inner_radius']<=radius)[0][-1]
        timelist.append(time_table['time'][time_table['snap']==snap][0]/1000.)
        zlist.append(stats['redshift'][0])
        for j in range(len(file_labels)):
            if (args.radius_range!='none'):
                if (args.normalized):
                    forces_list[j].append(np.sum(stats[file_labels[j] + '_sum'][rad_in:rad_out])/np.sum(stats[file_labels[j] + '_weight_sum'][rad_in:rad_out]))
                else:
                    forces_list[j].append(np.sum(stats[file_labels[j] + '_sum'][rad_in:rad_out])/gtoMsun)
            else:
                if (args.normalized):
                    forces_list[j].append(stats[file_labels[j] + '_sum'][rad_ind]/stats[file_labels[j] + '_weight_sum'][rad_ind])
                else:
                    forces_list[j].append(stats[file_labels[j] + '_sum'][rad_ind]/gtoMsun)
            if (args.region_filter!='none'):
                if (args.radius_range!='none'):
                    if (args.normalized):
                        forces_regions[0][j].append(np.sum(stats['low_' + args.region_filter + '_' + file_labels[j] + '_sum'][rad_in:rad_out]) / \
                          np.sum(stats['low_' + args.region_filter + '_' + file_labels[j] + '_weight_sum'][rad_in:rad_out]))
                        forces_regions[1][j].append(np.sum(stats['mid_' + args.region_filter + '_' + file_labels[j] + '_sum'][rad_in:rad_out]) / \
                          np.sum(stats['mid_' + args.region_filter + '_' + file_labels[j] + '_weight_sum'][rad_in:rad_out]))
                        forces_regions[2][j].append(np.sum(stats['high_' + args.region_filter + '_' + file_labels[j] + '_sum'][rad_in:rad_out]) / \
                          np.sum(stats['high_' + args.region_filter + '_' + file_labels[j] + '_weight_sum'][rad_in:rad_out]))
                    else:
                        forces_regions[0][j].append(np.sum(stats['low_' + args.region_filter + '_' + file_labels[j] + '_sum'][rad_in:rad_out])/gtoMsun)
                        forces_regions[1][j].append(np.sum(stats['mid_' + args.region_filter + '_' + file_labels[j] + '_sum'][rad_in:rad_out])/gtoMsun)
                        forces_regions[2][j].append(np.sum(stats['high_' + args.region_filter + '_' + file_labels[j] + '_sum'][rad_in:rad_out])/gtoMsun)
                else:
                    if (args.normalized):
                        forces_regions[0][j].append(stats['low_' + args.region_filter + '_' + file_labels[j] + '_sum'][rad_ind] / \
                          stats['low_' + args.region_filter + '_' + file_labels[j] + '_weight_sum'][rad_ind])
                        forces_regions[1][j].append(stats['mid_' + args.region_filter + '_' + file_labels[j] + '_sum'][rad_ind] / \
                          stats['mid_' + args.region_filter + '_' + file_labels[j] + '_weight_sum'][rad_ind])
                        forces_regions[2][j].append(stats['high_' + args.region_filter + '_' + file_labels[j] + '_sum'][rad_ind] / \
                          stats['high_' + args.region_filter + '_' + file_labels[j] + '_weight_sum'][rad_ind])
                    else:
                        forces_regions[0][j].append(stats['low_' + args.region_filter + '_' + file_labels[j] + '_sum'][rad_ind]/gtoMsun)
                        forces_regions[1][j].append(stats['mid_' + args.region_filter + '_' + file_labels[j] + '_sum'][rad_ind]/gtoMsun)
                        forces_regions[2][j].append(stats['high_' + args.region_filter + '_' + file_labels[j] + '_sum'][rad_ind]/gtoMsun)

    if (args.time_avg!=0):
        forces_list_avgd = []
        if (args.region_filter!='none'):
            forces_regions_avgd = [[], [], []]
        for j in range(len(plot_labels)):
            forces_list_avgd.append(uniform_filter1d(forces_list[j], size=avg_window))
            if (args.region_filter!='none'):
                for k in range(3):
                    forces_regions_avgd[k].append(uniform_filter1d(forces_regions[k][j], size=avg_window))
        forces_list = forces_list_avgd
        if (args.region_filter!='none'):
            forces_regions = forces_regions_avgd

    zlist.reverse()
    timelist.reverse()
    time_func = IUS(zlist, timelist)
    timelist.reverse()
    timelist = np.array(timelist).flatten()
    zlist = np.array(zlist)

    fig = plt.figure(figsize=(8,6), dpi=500)
    ax = fig.add_subplot(1,1,1)

    for j in range(len(plot_labels)):
        ax.plot(timelist, forces_list[j], lw=2, ls=linestyles[j], color=plot_colors[j], label=plot_labels[j])

    if (args.normalized):
        ax.axis([np.min(timelist), np.max(timelist), -1e-6,1e-6])
        ax.set_yscale('symlog', linthresh=1e-9)
        ax.text(13,-2e-8, halo_dict[args.halo], ha='right', va='center', fontsize=14)
        ax.set_ylabel('log Net Force [cm/s$^2$]', fontsize=14)
        if (args.radius_range!='none'):
            ax.text(13,-1e-7, '$r=%.2f-%.2f R_{200}$' % (radius_range[0], radius_range[1]), ha='right', va='center', fontsize=14)
        else:
            ax.text(13,-1e-7, '$r=%.2f R_{200}$' % (args.radius), ha='right', va='center', fontsize=14)
    else:
        ax.axis([np.min(timelist), np.max(timelist), -1e5,1e5])
        ax.set_yscale('symlog', linthresh=1e1)
        ax.text(4,2e4, halo_dict[args.halo], ha='left', va='center', fontsize=14)
        ax.set_ylabel('log Net Force [$M_\odot$ cm/s$^2$]', fontsize=14)
        if (args.radius_range!='none'):
            ax.text(13,-5e3, '$r=%.2f-%.2f R_{200}$' % (radius_range[0], radius_range[1]), ha='right', va='center', fontsize=14)
        else:
            ax.text(13,-5e3, '$r=%.2f R_{200}$' % (args.radius), ha='right', va='center', fontsize=14)
    ax.legend(loc=1, frameon=False, fontsize=14, ncol=2)

    ax2 = ax.twiny()
    ax.tick_params(axis='both', which='both', direction='in', length=8, width=2, pad=5, labelsize=14, \
      top=False, right=True)
    ax2.tick_params(axis='x', which='both', direction='in', length=8, width=2, pad=5, labelsize=14, \
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
    ax2.set_xlabel('Redshift', fontsize=14)
    ax.set_xlabel('Time [Gyr]', fontsize=14)

    z_sfr, sfr = np.loadtxt(code_path + 'halo_infos/00' + args.halo + '/' + args.run + '/sfr', unpack=True, usecols=[1,2], skiprows=1)
    t_sfr = time_func(z_sfr)

    ax3 = ax.twinx()
    ax3.plot(t_sfr, sfr, 'k-', lw=1)
    ax.plot([timelist[0],timelist[-1]], [0,0], 'k-', lw=1, label='SFR (right axis)')
    ax3.tick_params(axis='y', which='both', direction='in', length=8, width=2, pad=5, labelsize=14, right=True)
    ax3.set_ylim(-5,200)
    ax3.set_ylabel('SFR [$M_\odot$/yr]', fontsize=14)

    fig.subplots_adjust(left=0.14, bottom=0.1, right=0.9, top=0.92)
    fig.savefig(save_dir + 'forces_vs_time' + save_suffix + '.png')
    plt.close(fig)

    if (args.region_filter!='none'):
        fig_regions = []
        axs_regions = []
        for i in range(len(plot_colors)):
            fig_regions.append(plt.figure(figsize=(8,6), dpi=500))
            axs_regions.append(fig_regions[-1].add_subplot(1,1,1))
        for i in range(3):
            fig = plt.figure(figsize=(8,6), dpi=500)
            ax = fig.add_subplot(1,1,1)

            for j in range(len(plot_labels)):
                axs_regions[j].plot(timelist, forces_regions[i][j], lw=2, ls=linestyles[j], color=plot_colors[j], alpha=alphas[i], label=region_label[i])
                ax.plot(timelist, forces_regions[i][j], lw=2, ls=linestyles[j], color=plot_colors[j], label=plot_labels[j])

            if (args.normalized):
                ax.axis([np.min(timelist), np.max(timelist), -1e-6,1e-6])
                ax.set_yscale('symlog', linthresh=1e-9)
                ax.set_ylabel('log Net Force [cm/s$^2$]', fontsize=14)
                if (args.radius_range!='none'):
                    ax.text(13,-1e-7, '$r=%.2f-%.2f R_{200}$' % (radius_range[0], radius_range[1]), ha='right', va='center', fontsize=14)
                else:
                    ax.text(13,-1e-7, '$r=%.2f R_{200}$' % (args.radius), ha='right', va='center', fontsize=14)
                ax.text(13,-2e-8, region_label[i], fontsize=14, ha='right', va='center')
            else:
                ax.axis([np.min(timelist), np.max(timelist), -1e5,1e5])
                ax.set_yscale('symlog', linthresh=1e1)
                ax.set_ylabel('log Net Force [$M_\odot$ cm/s$^2$]', fontsize=14)
                if (args.radius_range!='none'):
                    ax.text(13,-5e3, '$r=%.2f-%.2f R_{200}$' % (radius_range[0], radius_range[1]), ha='right', va='center', fontsize=14)
                else:
                    ax.text(13,-5e3, '$r=%.2f R_{200}$' % (args.radius), ha='right', va='center', fontsize=14)
                ax.text(4,2e4, region_label[i], fontsize=14, ha='left', va='center')

            ax.legend(loc=1, frameon=False, fontsize=14, ncol=2)
            ax2 = ax.twiny()
            ax.tick_params(axis='both', which='both', direction='in', length=8, width=2, pad=5, labelsize=14, \
              top=False, right=True)
            ax2.tick_params(axis='x', which='both', direction='in', length=8, width=2, pad=5, labelsize=14, \
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
            ax2.set_xlabel('Redshift', fontsize=14)
            ax.set_xlabel('Time [Gyr]', fontsize=14)

            z_sfr, sfr = np.loadtxt(code_path + 'halo_infos/00' + args.halo + '/' + args.run + '/sfr', unpack=True, usecols=[1,2], skiprows=1)
            t_sfr = time_func(z_sfr)

            ax3 = ax.twinx()
            ax3.plot(t_sfr, sfr, 'k-', lw=1)
            ax.plot([timelist[0],timelist[-1]], [0,0], 'k-', lw=1, label='SFR (right axis)')
            ax3.tick_params(axis='y', which='both', direction='in', length=8, width=2, pad=5, labelsize=14, right=True)
            ax3.set_ylim(-5,200)
            ax3.set_ylabel('SFR [$M_\odot$/yr]', fontsize=14)

            fig.subplots_adjust(left=0.14, bottom=0.1, right=0.9, top=0.92)
            fig.savefig(save_dir + 'forces_vs_time_region-' + region_name[i] + args.region_filter + save_suffix + '.png')
            plt.close(fig)

        for j in range(len(plot_labels)):
            if (args.normalized):
                axs_regions[j].axis([np.min(timelist), np.max(timelist), -1e-6,1e-6])
                axs_regions[j].set_yscale('symlog', linthresh=1e-9)
                axs_regions[j].set_ylabel('log Net Force [cm/s$^2$]', fontsize=14)
                if (args.radius_range!='none'):
                    axs_regions[j].text(13,-1e-7, '$r=%.2f-%.2f R_{200}$' % (radius_range[0], radius_range[1]), ha='right', va='center', fontsize=14)
                else:
                    axs_regions[j].text(13,-1e-7, '$r=%.2f R_{200}$' % (args.radius), ha='right', va='center', fontsize=14)
                axs_regions[j].text(13,-2e-8, plot_labels[j], fontsize=14, ha='right', va='center')
            else:
                axs_regions[j].axis([np.min(timelist), np.max(timelist), -1e5,1e5])
                axs_regions[j].set_yscale('symlog', linthresh=1e1)
                axs_regions[j].set_ylabel('log Net Force [$M_\odot$ cm/s$^2$]', fontsize=14)
                if (args.radius_range!='none'):
                    axs_regions[j].text(13,-5e3, '$r=%.2f-%.2f R_{200}$' % (radius_range[0], radius_range[1]), ha='right', va='center', fontsize=14)
                else:
                    axs_regions[j].text(13,-5e3, '$r=%.2f R_{200}$' % (args.radius), ha='right', va='center', fontsize=14)
                axs_regions[j].text(4,2e4, plot_labels[j] + ' Force', fontsize=14, ha='left', va='center')

            axs_regions[j].legend(loc=1, frameon=False, fontsize=14)
            ax2 = axs_regions[j].twiny()
            axs_regions[j].tick_params(axis='both', which='both', direction='in', length=8, width=2, pad=5, labelsize=14, \
              top=False, right=True)
            ax2.tick_params(axis='x', which='both', direction='in', length=8, width=2, pad=5, labelsize=14, \
              top=True)
            x0, x1 = axs_regions[j].get_xlim()
            z_ticks = [2,1.5,1,.75,.5,.3,.2,.1,0]
            last_z = np.where(z_ticks >= zlist[0])[0][-1]
            first_z = np.where(z_ticks <= zlist[-1])[0][0]
            z_ticks = z_ticks[first_z:last_z+1]
            tick_pos = [z for z in time_func(z_ticks)]
            tick_labels = ['%.2f' % (z) for z in z_ticks]
            ax2.set_xlim(x0,x1)
            ax2.set_xticks(tick_pos)
            ax2.set_xticklabels(tick_labels)
            ax2.set_xlabel('Redshift', fontsize=14)
            axs_regions[j].set_xlabel('Time [Gyr]', fontsize=14)

            z_sfr, sfr = np.loadtxt(code_path + 'halo_infos/00' + args.halo + '/' + args.run + '/sfr', unpack=True, usecols=[1,2], skiprows=1)
            t_sfr = time_func(z_sfr)

            ax3 = axs_regions[j].twinx()
            ax3.plot(t_sfr, sfr, 'k-', lw=1)
            axs_regions[j].plot([timelist[0],timelist[-1]], [0,0], 'k-', lw=1, label='SFR (right axis)')
            ax3.tick_params(axis='y', which='both', direction='in', length=8, width=2, pad=5, labelsize=14, right=True)
            ax3.set_ylim(-5,200)
            ax3.set_ylabel('SFR [$M_\odot$/yr]', fontsize=14)

            fig_regions[j].subplots_adjust(left=0.14, bottom=0.1, right=0.9, top=0.92)
            fig_regions[j].savefig(save_dir + file_labels[j] + '_vs_time_regions-' + args.region_filter + save_suffix + '.png')
            plt.close(fig_regions[j])

def forces_vs_energy_output(snaplist):
    '''Makes a 2D plot of each force type vs. energy output from the central 0.1Rvir of the halo and
    galactocentric radius. Requires saved files of both forces vs. radius and energy fluxes for each
    snapshot in 'snaplist'.'''

    tablename_prefix = output_dir + 'stats_halo_00' + args.halo + '/' + args.run + '/Tables/'
    if (args.filename == ''):
        filename = ''
    else:
        filename = '_' + args.filename

    thermal_energy_fluxes = []
    kinetic_energy_fluxes = []
    tot_energy_fluxes = []
    thermal_support = []
    turbulent_support = []
    rotation_support = []
    total_support = []
    for i in range(len(snaplist)):
        snap = snaplist[i]
        Rvir = rvir_masses['radius'][rvir_masses['snapshot']==snap][0]
        forces = Table.read(tablename_prefix + snap + '_stats_force-types' + filename + '.hdf5', path='all_data')
        radius_list = 0.5*(forces['inner_radius'] + forces['outer_radius'])/Rvir
        fluxes = Table.read(output_dir + 'fluxes_halo_00' + args.halo + '/' + args.run + '/Tables/' + \
                            snap + '_fluxes_mass_energy_cgm-only.hdf5', path='all_data')
        thermal_energy_fluxes.append(np.log10(fluxes['net_thermal_energy_flux'][np.where(fluxes['radius']<=0.1*Rvir)[0][-1]]))
        kinetic_energy_fluxes.append(np.log10(fluxes['net_kinetic_energy_flux'][np.where(fluxes['radius']<=0.1*Rvir)[0][-1]]))
        tot_energy_fluxes.append(np.log10(fluxes['net_thermal_energy_flux'][np.where(fluxes['radius']<=0.1*Rvir)[0][-1]] + \
          fluxes['net_kinetic_energy_flux'][np.where(fluxes['radius']<=0.1*Rvir)[0][-1]]))
        thermal_support.append(forces['thermal_force_sum']/-forces['gravity_force_sum'])
        turbulent_support.append(forces['turbulent_force_sum']/-forces['gravity_force_sum'])
        rotation_support.append(forces['rotation_force_sum']/-forces['gravity_force_sum'])
        total_support.append((forces['thermal_force_sum']+forces['turbulent_force_sum']+forces['rotation_force_sum']+forces['ram_force_sum'])/-forces['gravity_force_sum'])
    thermal_energy_fluxes = np.array(thermal_energy_fluxes)
    kinetic_energy_fluxes = np.array(kinetic_energy_fluxes)
    tot_energy_fluxes = np.array(tot_energy_fluxes)
    thermal_support = np.array(thermal_support)
    turbulent_support = np.array(turbulent_support)
    rotation_support = np.array(rotation_support)
    total_support = np.array(total_support)

    thermal_energy_bins = np.linspace(46., 49., 7)
    thermal_bin_indices = np.digitize(thermal_energy_fluxes, thermal_energy_bins)
    thermal_support_thermal_bins = []
    turbulent_support_thermal_bins = []
    rotation_support_thermal_bins = []
    total_support_thermal_bins = []
    for i in range(1,len(thermal_energy_bins)):
        values = thermal_support[np.where(thermal_bin_indices==i)[0]]
        thermal_support_thermal_bins.append(np.nanmean(values, axis=0))
        values = turbulent_support[np.where(thermal_bin_indices==i)[0]]
        turbulent_support_thermal_bins.append(np.nanmean(values, axis=0))
        values = rotation_support[np.where(thermal_bin_indices==i)[0]]
        rotation_support_thermal_bins.append(np.nanmean(values, axis=0))
        values = total_support[np.where(thermal_bin_indices==i)[0]]
        total_support_thermal_bins.append(np.nanmean(values, axis=0))
    thermal_support_thermal_bins = np.array(thermal_support_thermal_bins)
    turbulent_support_thermal_bins = np.array(turbulent_support_thermal_bins)
    rotation_support_thermal_bins = np.array(rotation_support_thermal_bins)
    total_support_thermal_bins = np.array(total_support_thermal_bins)

    kinetic_energy_bins = np.linspace(46., 49., 7)
    kinetic_bin_indices = np.digitize(kinetic_energy_fluxes, kinetic_energy_bins)
    thermal_support_kinetic_bins = []
    turbulent_support_kinetic_bins = []
    rotation_support_kinetic_bins = []
    total_support_kinetic_bins = []
    for i in range(1,len(kinetic_energy_bins)):
        values = thermal_support[np.where(kinetic_bin_indices==i)[0]]
        thermal_support_kinetic_bins.append(np.nanmean(values, axis=0))
        values = turbulent_support[np.where(kinetic_bin_indices==i)[0]]
        turbulent_support_kinetic_bins.append(np.nanmean(values, axis=0))
        values = rotation_support[np.where(kinetic_bin_indices==i)[0]]
        rotation_support_kinetic_bins.append(np.nanmean(values, axis=0))
        values = total_support[np.where(kinetic_bin_indices==i)[0]]
        total_support_kinetic_bins.append(np.nanmean(values, axis=0))
    thermal_support_kinetic_bins = np.array(thermal_support_kinetic_bins)
    turbulent_support_kinetic_bins = np.array(turbulent_support_kinetic_bins)
    rotation_support_kinetic_bins = np.array(rotation_support_kinetic_bins)
    total_support_kinetic_bins = np.array(total_support_kinetic_bins)

    tot_energy_bins = np.linspace(46.5, 49.5, 7)
    tot_bin_indices = np.digitize(tot_energy_fluxes, tot_energy_bins)
    thermal_support_tot_bins = []
    turbulent_support_tot_bins = []
    rotation_support_tot_bins = []
    total_support_tot_bins = []
    for i in range(1,len(tot_energy_bins)):
        values = thermal_support[np.where(tot_bin_indices==i)[0]]
        thermal_support_tot_bins.append(np.nanmean(values, axis=0))
        values = turbulent_support[np.where(tot_bin_indices==i)[0]]
        turbulent_support_tot_bins.append(np.nanmean(values, axis=0))
        values = rotation_support[np.where(tot_bin_indices==i)[0]]
        rotation_support_tot_bins.append(np.nanmean(values, axis=0))
        values = total_support[np.where(tot_bin_indices==i)[0]]
        total_support_tot_bins.append(np.nanmean(values, axis=0))
    thermal_support_tot_bins = np.array(thermal_support_tot_bins)
    turbulent_support_tot_bins = np.array(turbulent_support_tot_bins)
    rotation_support_tot_bins = np.array(rotation_support_tot_bins)
    total_support_tot_bins = np.array(total_support_tot_bins)

    cmap = sns.blend_palette(('#ffffff', '#b482ff', "#6600ff", "#8cf4ff", "#8cf4ff", "#8cffb4", "#8cffb4", '#c7ff8c', '#c7ff8c'), as_cmap=True)

    fig = plt.figure(figsize=(15,12), dpi=500)
    ax1 = fig.add_subplot(2,2,1)
    ax2 = fig.add_subplot(2,2,2)
    ax3 = fig.add_subplot(2,2,3)
    ax4 = fig.add_subplot(2,2,4)

    im = ax1.pcolormesh(thermal_support_thermal_bins, cmap=cmap, vmin=0, vmax=4)
    ax2.pcolormesh(turbulent_support_thermal_bins, cmap=cmap, vmin=0, vmax=4)
    ax3.pcolormesh(rotation_support_thermal_bins, cmap=cmap, vmin=0, vmax=4)
    ax4.pcolormesh(total_support_thermal_bins, cmap=cmap, vmin=0, vmax=4)

    ax1.set_xticks([0,16.667,33.333,50.,66.667,83.333,100.])
    ax1.set_xticklabels([0,0.25,0.5,0.75,1.,1.25,1.5])
    ax1.set_yticks([0,1,2,3,4,5,6])
    ax1.set_yticklabels(thermal_energy_bins)
    ax2.set_xticks([0,16.667,33.333,50.,66.667,83.333,100.])
    ax2.set_xticklabels([0,0.25,0.5,0.75,1.,1.25,1.5])
    ax2.set_yticks([0,1,2,3,4,5,6])
    ax2.set_yticklabels(thermal_energy_bins)
    ax3.set_xticks([0,16.667,33.333,50.,66.667,83.333,100.])
    ax3.set_xticklabels([0,0.25,0.5,0.75,1.,1.25,1.5])
    ax3.set_yticks([0,1,2,3,4,5,6])
    ax3.set_yticklabels(thermal_energy_bins)
    ax4.set_xticks([0,16.667,33.333,50.,66.667,83.333,100.])
    ax4.set_xticklabels([0,0.25,0.5,0.75,1.,1.25,1.5])
    ax4.set_yticks([0,1,2,3,4,5,6])
    ax4.set_yticklabels(thermal_energy_bins)
    ax1.tick_params(axis='both', which='both', direction='in', length=8, width=2, pad=5, labelsize=18, \
      top=True, right=True)
    ax2.tick_params(axis='both', which='both', direction='in', length=8, width=2, pad=5, labelsize=18, \
      top=True, right=True)
    ax3.tick_params(axis='both', which='both', direction='in', length=8, width=2, pad=5, labelsize=18, \
      top=True, right=True)
    ax4.tick_params(axis='both', which='both', direction='in', length=8, width=2, pad=5, labelsize=18, \
      top=True, right=True)

    ax1.set_xlabel('Radius [$R_{200}$]', fontsize=20)
    ax1.set_ylabel('$\dot{E}_\mathrm{therm}$ [erg/s]', fontsize=20)
    ax2.set_xlabel('Radius [$R_{200}$]', fontsize=20)
    ax2.set_ylabel('$\dot{E}_\mathrm{therm}$ [erg/s]', fontsize=20)
    ax3.set_xlabel('Radius [$R_{200}$]', fontsize=20)
    ax3.set_ylabel('$\dot{E}_\mathrm{therm}$ [erg/s]', fontsize=20)
    ax4.set_xlabel('Radius [$R_{200}$]', fontsize=20)
    ax4.set_ylabel('$\dot{E}_\mathrm{therm}$ [erg/s]', fontsize=20)

    ax1.set_title('Thermal Support', fontsize=20)
    ax2.set_title('Turbulent Support', fontsize=20)
    ax3.set_title('Rotation Support', fontsize=20)
    ax4.set_title('Sum Support', fontsize=20)

    cax = fig.add_axes([0.9, 0.07, 0.03, 0.9])
    cax.tick_params(axis='both', which='both', direction='in', length=8, width=2, pad=5, labelsize=18, \
      top=True, right=True)
    fig.colorbar(im, cax=cax, orientation='vertical')
    cax.text(2.75, 0.5, 'Support', fontsize=20, rotation='vertical', ha='center', va='center', transform=cax.transAxes)
    fig.subplots_adjust(left=0.08,right=0.88, top=0.97, bottom=0.07, wspace=0.25, hspace=0.25)
    fig.savefig(save_dir + 'support_vs_radius_thermal-energy-output' + save_suffix + '.png')


    fig = plt.figure(figsize=(15,12), dpi=500)
    ax1 = fig.add_subplot(2,2,1)
    ax2 = fig.add_subplot(2,2,2)
    ax3 = fig.add_subplot(2,2,3)
    ax4 = fig.add_subplot(2,2,4)

    im = ax1.pcolormesh(thermal_support_kinetic_bins, cmap=cmap, vmin=0, vmax=4)
    ax2.pcolormesh(turbulent_support_kinetic_bins, cmap=cmap, vmin=0, vmax=4)
    ax3.pcolormesh(rotation_support_kinetic_bins, cmap=cmap, vmin=0, vmax=4)
    ax4.pcolormesh(total_support_kinetic_bins, cmap=cmap, vmin=0, vmax=4)

    ax1.set_xticks([0,16.667,33.333,50.,66.667,83.333,100.])
    ax1.set_xticklabels([0,0.25,0.5,0.75,1.,1.25,1.5])
    ax1.set_yticks([0,1,2,3,4,5,6])
    ax1.set_yticklabels(kinetic_energy_bins)
    ax2.set_xticks([0,16.667,33.333,50.,66.667,83.333,100.])
    ax2.set_xticklabels([0,0.25,0.5,0.75,1.,1.25,1.5])
    ax2.set_yticks([0,1,2,3,4,5,6])
    ax2.set_yticklabels(kinetic_energy_bins)
    ax3.set_xticks([0,16.667,33.333,50.,66.667,83.333,100.])
    ax3.set_xticklabels([0,0.25,0.5,0.75,1.,1.25,1.5])
    ax3.set_yticks([0,1,2,3,4,5,6])
    ax3.set_yticklabels(kinetic_energy_bins)
    ax4.set_xticks([0,16.667,33.333,50.,66.667,83.333,100.])
    ax4.set_xticklabels([0,0.25,0.5,0.75,1.,1.25,1.5])
    ax4.set_yticks([0,1,2,3,4,5,6])
    ax4.set_yticklabels(kinetic_energy_bins)
    ax1.tick_params(axis='both', which='both', direction='in', length=8, width=2, pad=5, labelsize=18, \
      top=True, right=True)
    ax2.tick_params(axis='both', which='both', direction='in', length=8, width=2, pad=5, labelsize=18, \
      top=True, right=True)
    ax3.tick_params(axis='both', which='both', direction='in', length=8, width=2, pad=5, labelsize=18, \
      top=True, right=True)
    ax4.tick_params(axis='both', which='both', direction='in', length=8, width=2, pad=5, labelsize=18, \
      top=True, right=True)

    ax1.set_xlabel('Radius [$R_{200}$]', fontsize=20)
    ax1.set_ylabel('$\dot{E}_\mathrm{kin}$ [erg/s]', fontsize=20)
    ax2.set_xlabel('Radius [$R_{200}$]', fontsize=20)
    ax2.set_ylabel('$\dot{E}_\mathrm{kin}$ [erg/s]', fontsize=20)
    ax3.set_xlabel('Radius [$R_{200}$]', fontsize=20)
    ax3.set_ylabel('$\dot{E}_\mathrm{kin}$ [erg/s]', fontsize=20)
    ax4.set_xlabel('Radius [$R_{200}$]', fontsize=20)
    ax4.set_ylabel('$\dot{E}_\mathrm{kin}$ [erg/s]', fontsize=20)

    ax1.set_title('Thermal Support', fontsize=20)
    ax2.set_title('Turbulent Support', fontsize=20)
    ax3.set_title('Rotation Support', fontsize=20)
    ax4.set_title('Sum Support', fontsize=20)

    cax = fig.add_axes([0.9, 0.07, 0.03, 0.9])
    cax.tick_params(axis='both', which='both', direction='in', length=8, width=2, pad=5, labelsize=18, \
      top=True, right=True)
    fig.colorbar(im, cax=cax, orientation='vertical')
    cax.text(2.75, 0.5, 'Support', fontsize=20, rotation='vertical', ha='center', va='center', transform=cax.transAxes)

    fig.subplots_adjust(left=0.08,right=0.88, top=0.97, bottom=0.07, wspace=0.25, hspace=0.25)
    fig.savefig(save_dir + 'support_vs_radius_kinetic-energy-output' + save_suffix + '.png')

    fig = plt.figure(figsize=(15,12), dpi=500)
    ax1 = fig.add_subplot(2,2,1)
    ax2 = fig.add_subplot(2,2,2)
    ax3 = fig.add_subplot(2,2,3)
    ax4 = fig.add_subplot(2,2,4)

    im = ax1.pcolormesh(thermal_support_tot_bins, cmap=cmap, vmin=0, vmax=4)
    ax2.pcolormesh(turbulent_support_tot_bins, cmap=cmap, vmin=0, vmax=4)
    ax3.pcolormesh(rotation_support_tot_bins, cmap=cmap, vmin=0, vmax=4)
    ax4.pcolormesh(total_support_tot_bins, cmap=cmap, vmin=0, vmax=4)

    ax1.set_xticks([0,16.667,33.333,50.,66.667,83.333,100.])
    ax1.set_xticklabels([0,0.25,0.5,0.75,1.,1.25,1.5])
    ax1.set_yticks([0,1,2,3,4,5,6])
    ax1.set_yticklabels(tot_energy_bins)
    ax2.set_xticks([0,16.667,33.333,50.,66.667,83.333,100.])
    ax2.set_xticklabels([0,0.25,0.5,0.75,1.,1.25,1.5])
    ax2.set_yticks([0,1,2,3,4,5,6])
    ax2.set_yticklabels(tot_energy_bins)
    ax3.set_xticks([0,16.667,33.333,50.,66.667,83.333,100.])
    ax3.set_xticklabels([0,0.25,0.5,0.75,1.,1.25,1.5])
    ax3.set_yticks([0,1,2,3,4,5,6])
    ax3.set_yticklabels(tot_energy_bins)
    ax4.set_xticks([0,16.667,33.333,50.,66.667,83.333,100.])
    ax4.set_xticklabels([0,0.25,0.5,0.75,1.,1.25,1.5])
    ax4.set_yticks([0,1,2,3,4,5,6])
    ax4.set_yticklabels(tot_energy_bins)
    ax1.tick_params(axis='both', which='both', direction='in', length=8, width=2, pad=5, labelsize=18, \
      top=True, right=True)
    ax2.tick_params(axis='both', which='both', direction='in', length=8, width=2, pad=5, labelsize=18, \
      top=True, right=True)
    ax3.tick_params(axis='both', which='both', direction='in', length=8, width=2, pad=5, labelsize=18, \
      top=True, right=True)
    ax4.tick_params(axis='both', which='both', direction='in', length=8, width=2, pad=5, labelsize=18, \
      top=True, right=True)

    ax1.set_xlabel('Radius [$R_{200}$]', fontsize=20)
    ax1.set_ylabel('$\dot{E}_\mathrm{out}$ [erg/s]', fontsize=20)
    ax2.set_xlabel('Radius [$R_{200}$]', fontsize=20)
    ax2.set_ylabel('$\dot{E}_\mathrm{out}$ [erg/s]', fontsize=20)
    ax3.set_xlabel('Radius [$R_{200}$]', fontsize=20)
    ax3.set_ylabel('$\dot{E}_\mathrm{out}$ [erg/s]', fontsize=20)
    ax4.set_xlabel('Radius [$R_{200}$]', fontsize=20)
    ax4.set_ylabel('$\dot{E}_\mathrm{out}$ [erg/s]', fontsize=20)

    ax1.set_title('Thermal Support', fontsize=20)
    ax2.set_title('Turbulent Support', fontsize=20)
    ax3.set_title('Rotation Support', fontsize=20)
    ax4.set_title('Sum Support', fontsize=20)

    cax = fig.add_axes([0.9, 0.07, 0.03, 0.9])
    cax.tick_params(axis='both', which='both', direction='in', length=8, width=2, pad=5, labelsize=18, \
      top=True, right=True)
    fig.colorbar(im, cax=cax, orientation='vertical')
    cax.text(2.75, 0.5, 'Support', fontsize=20, rotation='vertical', ha='center', va='center', transform=cax.transAxes)

    fig.subplots_adjust(left=0.08,right=0.88, top=0.97, bottom=0.07, wspace=0.25, hspace=0.25)
    fig.savefig(save_dir + 'support_vs_radius_energy-output' + save_suffix + '.png')

def work_vs_time(snaplist):
    '''Plots the work done by different forces in the halo over time. Takes same parameters and
    set up the same way as forces_vs_time.'''

    tablename_prefix = output_dir + 'stats_halo_00' + args.halo + '/' + args.run + '/Tables/'
    time_table = Table.read(output_dir + 'times_halo_00' + args.halo + '/' + args.run + '/time_table.hdf5', path='all_data')
    if (args.filename != ''): filename = '_' + args.filename
    else: filename = ''

    fig = plt.figure(figsize=(8,6), dpi=500)
    ax = fig.add_subplot(1,1,1)

    plot_colors = ['r', 'g', 'm', 'b', 'gold', 'k']
    plot_labels = ['Thermal', 'Turbulent', 'Ram', 'Rotation', 'Gravity', 'Total']
    file_labels = ['thermal_force', 'turbulent_force', 'ram_force', 'rotation_force', 'gravity_force', 'total_force']
    output_labels = ['thermal', 'turbulent', 'ram', 'rotation', 'gravity', 'total']
    linestyles = ['-', '--', ':', '-.', '--', '-']
    alphas = [0.3, 0.6, 1.]

    if (args.time_avg!=0):
        dt = 5.38*args.output_step
        avg_window = int(np.ceil(args.time_avg/dt))

    if (args.radius_range!='none'):
        radius_range = ast.literal_eval(args.radius_range)
    else:
        radius_range = [0., 1.]

    zlist = []
    timelist = []
    forces_list = []
    if (args.region_filter!='none'):
        forces_regions = [[],[],[]]
        region_label = ['Low ' + args.region_filter, 'Mid ' + args.region_filter, 'High ' + args.region_filter]
        region_name = ['low-', 'mid-', 'high-']
    for j in range(len(plot_labels)):
        forces_list.append([])
        if (args.region_filter!='none'):
            forces_regions[0].append([])
            forces_regions[1].append([])
            forces_regions[2].append([])

    for i in range(len(snaplist)):
        snap = snaplist[i]
        stats = Table.read(tablename_prefix + snap + '_stats_force-types' + filename + '.hdf5', path='all_data')
        Rvir = rvir_masses['radius'][rvir_masses['snapshot']==snap][0]
        radius_in = radius_range[0]*Rvir
        radius_out = radius_range[1]*Rvir
        rad_in = np.where(stats['inner_radius']<=radius_in)[0][-1]
        rad_out = np.where(stats['outer_radius']>=radius_out)[0][0]
        timelist.append(time_table['time'][time_table['snap']==snap][0]/1000.)
        zlist.append(stats['redshift'][0])
        for j in range(len(file_labels)):
            if (args.normalized):
                forces_list[j].append(np.sum(stats[file_labels[j] + '_sum'][rad_in:rad_out] * \
                                    (stats['outer_radius'][rad_in:rad_out]-stats['inner_radius'][rad_in:rad_out])*1000*cmtopc/stats[file_labels[j] + '_weight_sum'][rad_in:rad_out]))
            else:
                forces_list[j].append(np.sum(stats[file_labels[j] + '_sum'][rad_in:rad_out] * \
                                    (stats['outer_radius'][rad_in:rad_out]-stats['inner_radius'][rad_in:rad_out])*1000*cmtopc))
            if (args.region_filter!='none'):
                if (args.normalized):
                    forces_regions[0][j].append(np.sum(stats['low_' + args.region_filter + '_' + file_labels[j] + '_sum'][rad_in:rad_out] * \
                                                (stats['outer_radius'][rad_in:rad_out]-stats['inner_radius'][rad_in:rad_out])*1000*cmtopc/stats[file_labels[j] + '_weight_sum'][rad_in:rad_out]))
                    forces_regions[1][j].append(np.sum(stats['mid_' + args.region_filter + '_' + file_labels[j] + '_sum'][rad_in:rad_out] * \
                                                (stats['outer_radius'][rad_in:rad_out]-stats['inner_radius'][rad_in:rad_out])*1000*cmtopc/stats[file_labels[j] + '_weight_sum'][rad_in:rad_out]))
                    forces_regions[2][j].append(np.sum(stats['high_' + args.region_filter + '_' + file_labels[j] + '_sum'][rad_in:rad_out] * \
                                                (stats['outer_radius'][rad_in:rad_out]-stats['inner_radius'][rad_in:rad_out])*1000*cmtopc/stats[file_labels[j] + '_weight_sum'][rad_in:rad_out]))
                else:
                    forces_regions[0][j].append(np.sum(stats['low_' + args.region_filter + '_' + file_labels[j] + '_sum'][rad_in:rad_out] * \
                                                (stats['outer_radius'][rad_in:rad_out]-stats['inner_radius'][rad_in:rad_out])*1000*cmtopc))
                    forces_regions[1][j].append(np.sum(stats['mid_' + args.region_filter + '_' + file_labels[j] + '_sum'][rad_in:rad_out] * \
                                                (stats['outer_radius'][rad_in:rad_out]-stats['inner_radius'][rad_in:rad_out])*1000*cmtopc))
                    forces_regions[2][j].append(np.sum(stats['high_' + args.region_filter + '_' + file_labels[j] + '_sum'][rad_in:rad_out] * \
                                                (stats['outer_radius'][rad_in:rad_out]-stats['inner_radius'][rad_in:rad_out])*1000*cmtopc))

    if (args.time_avg!=0):
        forces_list_avgd = []
        if (args.region_filter!='none'):
            forces_regions_avgd = [[], [], []]
        for j in range(len(plot_labels)):
            forces_list_avgd.append(uniform_filter1d(forces_list[j], size=avg_window))
            if (args.region_filter!='none'):
                for k in range(3):
                    forces_regions_avgd[k].append(uniform_filter1d(forces_regions[k][j], size=avg_window))
        forces_list = forces_list_avgd
        if (args.region_filter!='none'):
            forces_regions = forces_regions_avgd

    zlist.reverse()
    timelist.reverse()
    time_func = IUS(zlist, timelist)
    timelist.reverse()
    timelist = np.array(timelist).flatten()
    zlist = np.array(zlist)

    fig = plt.figure(figsize=(8,6), dpi=500)
    ax = fig.add_subplot(1,1,1)

    for j in range(len(plot_labels)):
        ax.plot(timelist, forces_list[j], lw=2, ls=linestyles[j], color=plot_colors[j], label=plot_labels[j])

    if (args.normalized):
        ax.axis([np.min(timelist), np.max(timelist), -1e16,1e16])
        ax.set_yscale('symlog', linthresh=1e14)
        ax.set_ylabel('Work done per mass [ergs/g]', fontsize=14)
    else:
        ax.axis([np.min(timelist), np.max(timelist), -1e59,1e59])
        ax.set_yscale('symlog', linthresh=1e56)
        ax.set_ylabel('Work done per mass [ergs/g]', fontsize=14)
    #ax.text(4,2e4, halo_dict[args.halo], ha='left', va='center', fontsize=14)
    #ax.text(13,-5e3, '$r=%.2f-%.2f R_{200}$' % (radius_range[0], radius_range[1]), ha='right', va='center', fontsize=14)
    ax.legend(loc=1, frameon=False, fontsize=14, ncol=2)

    ax2 = ax.twiny()
    ax.tick_params(axis='both', which='both', direction='in', length=8, width=2, pad=5, labelsize=14, \
      top=False, right=True)
    ax2.tick_params(axis='x', which='both', direction='in', length=8, width=2, pad=5, labelsize=14, \
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
    ax2.set_xlabel('Redshift', fontsize=14)
    ax.set_xlabel('Time [Gyr]', fontsize=14)

    z_sfr, sfr = np.loadtxt(code_path + 'halo_infos/00' + args.halo + '/' + args.run + '/sfr', unpack=True, usecols=[1,2], skiprows=1)
    t_sfr = time_func(z_sfr)

    ax3 = ax.twinx()
    ax3.plot(t_sfr, sfr, 'k-', lw=1)
    ax.plot([timelist[0],timelist[-1]], [0,0], 'k-', lw=1, label='SFR (right axis)')
    ax3.tick_params(axis='y', which='both', direction='in', length=8, width=2, pad=5, labelsize=14, right=True)
    ax3.set_ylim(-5,200)
    ax3.set_ylabel('SFR [$M_\odot$/yr]', fontsize=14)

    fig.subplots_adjust(left=0.14, bottom=0.1, right=0.9, top=0.92)
    fig.savefig(save_dir + 'work-done_vs_time' + save_suffix + '.png')
    plt.close(fig)

    if (args.region_filter!='none'):
        fig_regions = []
        axs_regions = []
        for i in range(len(plot_colors)):
            fig_regions.append(plt.figure(figsize=(8,6), dpi=500))
            axs_regions.append(fig_regions[-1].add_subplot(1,1,1))
        for i in range(3):
            fig = plt.figure(figsize=(8,6), dpi=500)
            ax = fig.add_subplot(1,1,1)

            for j in range(len(plot_labels)):
                axs_regions[j].plot(timelist, forces_regions[i][j], lw=2, ls=linestyles[j], color=plot_colors[j], alpha=alphas[i], label=region_label[i])
                ax.plot(timelist, forces_regions[i][j], lw=2, ls=linestyles[j], color=plot_colors[j], label=plot_labels[j])

            if (args.normalized):
                ax.axis([np.min(timelist), np.max(timelist), -1e16,1e16])
                ax.set_yscale('symlog', linthresh=1e14)
                ax.set_ylabel('Work done per mass [ergs/g]', fontsize=14)
            else:
                ax.axis([np.min(timelist), np.max(timelist), -1e59,1e59])
                ax.set_yscale('symlog', linthresh=1e56)
                ax.set_ylabel('Work done [ergs]', fontsize=14)
            #ax.text(13,-5e3, '$r=%.2f-%.2f R_{200}$' % (radius_range[0], radius_range[1]), ha='right', va='center', fontsize=14)
            #ax.text(4,2e4, region_label[i], fontsize=14, ha='left', va='center')

            ax.legend(loc=1, frameon=False, fontsize=14, ncol=2)
            ax2 = ax.twiny()
            ax.tick_params(axis='both', which='both', direction='in', length=8, width=2, pad=5, labelsize=14, \
              top=False, right=True)
            ax2.tick_params(axis='x', which='both', direction='in', length=8, width=2, pad=5, labelsize=14, \
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
            ax2.set_xlabel('Redshift', fontsize=14)
            ax.set_xlabel('Time [Gyr]', fontsize=14)

            z_sfr, sfr = np.loadtxt(code_path + 'halo_infos/00' + args.halo + '/' + args.run + '/sfr', unpack=True, usecols=[1,2], skiprows=1)
            t_sfr = time_func(z_sfr)

            ax3 = ax.twinx()
            ax3.plot(t_sfr, sfr, 'k-', lw=1)
            ax.plot([timelist[0],timelist[-1]], [0,0], 'k-', lw=1, label='SFR (right axis)')
            ax3.tick_params(axis='y', which='both', direction='in', length=8, width=2, pad=5, labelsize=14, right=True)
            ax3.set_ylim(-5,200)
            ax3.set_ylabel('SFR [$M_\odot$/yr]', fontsize=14)

            fig.subplots_adjust(left=0.14, bottom=0.1, right=0.9, top=0.92)
            fig.savefig(save_dir + 'work-done_vs_time_region-' + region_name[i] + args.region_filter + save_suffix + '.png')
            plt.close(fig)

        for j in range(len(plot_labels)):
            if (args.normalized):
                axs_regions[j].axis([np.min(timelist), np.max(timelist), -1e16,1e16])
                axs_regions[j].set_yscale('symlog', linthresh=1e14)
                axs_regions[j].set_ylabel('Work done per mass [ergs/g]', fontsize=14)
            else:
                axs_regions[j].axis([np.min(timelist), np.max(timelist), -1e59,1e59])
                axs_regions[j].set_yscale('symlog', linthresh=1e56)
                axs_regions[j].set_ylabel('Work done [ergs]', fontsize=14)
            #axs_regions[j].text(13,-5e3, '$r=%.2f-%.2f R_{200}$' % (radius_range[0], radius_range[1]), ha='right', va='center', fontsize=14)
            #axs_regions[j].text(4,2e4, plot_labels[j] + ' Force', fontsize=14, ha='left', va='center')

            axs_regions[j].legend(loc=1, frameon=False, fontsize=14)
            ax2 = axs_regions[j].twiny()
            axs_regions[j].tick_params(axis='both', which='both', direction='in', length=8, width=2, pad=5, labelsize=14, \
              top=False, right=True)
            ax2.tick_params(axis='x', which='both', direction='in', length=8, width=2, pad=5, labelsize=14, \
              top=True)
            x0, x1 = axs_regions[j].get_xlim()
            z_ticks = [2,1.5,1,.75,.5,.3,.2,.1,0]
            last_z = np.where(z_ticks >= zlist[0])[0][-1]
            first_z = np.where(z_ticks <= zlist[-1])[0][0]
            z_ticks = z_ticks[first_z:last_z+1]
            tick_pos = [z for z in time_func(z_ticks)]
            tick_labels = ['%.2f' % (z) for z in z_ticks]
            ax2.set_xlim(x0,x1)
            ax2.set_xticks(tick_pos)
            ax2.set_xticklabels(tick_labels)
            ax2.set_xlabel('Redshift', fontsize=14)
            axs_regions[j].set_xlabel('Time [Gyr]', fontsize=14)

            z_sfr, sfr = np.loadtxt(code_path + 'halo_infos/00' + args.halo + '/' + args.run + '/sfr', unpack=True, usecols=[1,2], skiprows=1)
            t_sfr = time_func(z_sfr)

            ax3 = axs_regions[j].twinx()
            ax3.plot(t_sfr, sfr, 'k-', lw=1)
            axs_regions[j].plot([timelist[0],timelist[-1]], [0,0], 'k-', lw=1, label='SFR (right axis)')
            ax3.tick_params(axis='y', which='both', direction='in', length=8, width=2, pad=5, labelsize=14, right=True)
            ax3.set_ylim(-5,200)
            ax3.set_ylabel('SFR [$M_\odot$/yr]', fontsize=14)

            fig_regions[j].subplots_adjust(left=0.14, bottom=0.1, right=0.9, top=0.92)
            fig_regions[j].savefig(save_dir + output_labels[j] + '_work-done_vs_time_regions-' + args.region_filter + save_suffix + '.png')
            plt.close(fig_regions[j])

def force_rays(snap):
    '''Makes plots of different forces (thermal, turbulent, ram, rotation, gravity, total) along
    rays originating at the center of the galaxy and traveling outward in different directions.'''

    masses_ind = np.where(masses['snapshot']==snap)[0]
    Menc_profile = IUS(np.concatenate(([0],masses['radius'][masses_ind])), np.concatenate(([0],masses['total_mass'][masses_ind])))
    Mvir = rvir_masses['total_mass'][rvir_masses['snapshot']==snap]
    Rvir = rvir_masses['radius'][rvir_masses['snapshot']==snap][0]

    if (args.system=='pleiades_cassi'):
        print('Copying directory to /tmp')
        snap_dir = '/tmp/' + snap
        if (args.copy_to_tmp):
            shutil.copytree(foggie_dir + run_dir + snap, snap_dir)
            snap_name = snap_dir + '/' + snap
        else:
            # Make a dummy directory with the snap name so the script later knows the process running
            # this snapshot failed if the directory is still there
            os.makedirs(snap_dir)
            snap_name = foggie_dir + run_dir + snap + '/' + snap
    else:
        snap_name = foggie_dir + run_dir + snap + '/' + snap
    ds, refine_box = foggie_load(snap_name, trackname, do_filter_particles=False, halo_c_v_name=halo_c_v_name, gravity=True, masses_dir=masses_dir)
    zsnap = ds.get_parameter('CosmologyCurrentRedshift')

    # Define the density cut between disk and CGM to vary smoothly between 1 and 0.1 between z = 0.5 and z = 0.25,
    # with it being 1 at higher redshifts and 0.1 at lower redshifts
    current_time = ds.current_time.in_units('Myr').v
    if (current_time<=8656.88):
        density_cut_factor = 1.
    elif (current_time<=10787.12):
        density_cut_factor = 1. - 0.9*(current_time-8656.88)/2130.24
    else:
        density_cut_factor = 0.1

    pix_res = float(np.min(refine_box[('gas','dx')].in_units('kpc')))  # at level 11
    lvl1_res = pix_res*2.**11.
    level = 9
    dx = lvl1_res/(2.**level)
    smooth_scale = (25./dx)/6.
    dx_cm = dx*1000*cmtopc
    refine_res = int(ds.refine_width/dx)
    box = ds.covering_grid(level=level, left_edge=ds.halo_center_kpc-ds.arr([0.5*ds.refine_width,0.5*ds.refine_width,0.5*ds.refine_width],'kpc'), dims=[refine_res, refine_res, refine_res])
    density = box['density'].in_units('g/cm**3').v
    temperature = box['temperature'].v
    mass = box['cell_mass'].in_units('g').v
    x = box[('gas','x')].in_units('cm') - ds.halo_center_kpc[0].to('cm')
    y = box[('gas','y')].in_units('cm') - ds.halo_center_kpc[1].to('cm')
    z = box[('gas','z')].in_units('cm') - ds.halo_center_kpc[2].to('cm')
    coords = (x[:,0,0].to('kpc').v,y[0,:,0].to('kpc').v,z[0,0,:].to('kpc').v)
    x = x.v
    y = y.v
    z = z.v
    radius = box['radius_corrected'].in_units('kpc').v
    r = box['radius_corrected'].in_units('cm').v
    x_hat = x/r
    y_hat = y/r
    z_hat = z/r

    # This next block needed for removing any ISM regions and then interpolating over the holes left behind
    # Define ISM regions to remove
    disk_mask = (density > cgm_density_max * density_cut_factor)
    # disk_mask_expanded is a binary mask of both ISM regions AND their surrounding pixels
    struct = ndimage.generate_binary_structure(3,3)
    disk_mask_expanded = ndimage.binary_dilation(disk_mask, structure=struct, iterations=3)
    disk_mask_expanded = ndimage.binary_closing(disk_mask_expanded, structure=struct, iterations=3)
    disk_mask_expanded = disk_mask_expanded | disk_mask
    # disk_edges is a binary mask of ONLY pixels surrounding ISM regions -- nothing inside ISM regions
    disk_edges = disk_mask_expanded & ~disk_mask
    x_edges = x[disk_edges].flatten()
    y_edges = y[disk_edges].flatten()
    z_edges = z[disk_edges].flatten()

    density_edges = density[disk_edges]
    density_interp_func = NearestNDInterpolator(list(zip(x_edges,y_edges,z_edges)), density_edges)
    den_masked = np.copy(density)
    den_masked[disk_mask] = density_interp_func(x[disk_mask], y[disk_mask], z[disk_mask])
    temperature_edges = temperature[disk_edges]
    temperature_interp_func = NearestNDInterpolator(list(zip(x_edges,y_edges,z_edges)), temperature_edges)
    temp_masked = np.copy(temperature)
    temp_masked[disk_mask] = temperature_interp_func(x[disk_mask], y[disk_mask], z[disk_mask])

    thermal_pressure = box['pressure'].in_units('erg/cm**3').v
    pres_edges = thermal_pressure[disk_edges]
    pres_interp_func = NearestNDInterpolator(list(zip(x_edges,y_edges,z_edges)), pres_edges)
    pres_masked = np.copy(thermal_pressure)
    pres_masked[disk_mask] = pres_interp_func(x[disk_mask], y[disk_mask], z[disk_mask])
    pres_grad = np.gradient(pres_masked, dx_cm)
    dPdr = pres_grad[0]*x_hat + pres_grad[1]*y_hat + pres_grad[2]*z_hat
    thermal_force = -1./den_masked * dPdr
    vx = box['vx_corrected'].in_units('cm/s').v
    vy = box['vy_corrected'].in_units('cm/s').v
    vz = box['vz_corrected'].in_units('cm/s').v
    vx_edges = vx[disk_edges]
    vy_edges = vy[disk_edges]
    vz_edges = vz[disk_edges]
    vx_interp_func = NearestNDInterpolator(list(zip(x_edges,y_edges,z_edges)), vx_edges)
    vy_interp_func = NearestNDInterpolator(list(zip(x_edges,y_edges,z_edges)), vy_edges)
    vz_interp_func = NearestNDInterpolator(list(zip(x_edges,y_edges,z_edges)), vz_edges)
    vx_masked = np.copy(vx)
    vy_masked = np.copy(vy)
    vz_masked = np.copy(vz)
    vx_masked[disk_mask] = vx_interp_func(x[disk_mask], y[disk_mask], z[disk_mask])
    vy_masked[disk_mask] = vy_interp_func(x[disk_mask], y[disk_mask], z[disk_mask])
    vz_masked[disk_mask] = vz_interp_func(x[disk_mask], y[disk_mask], z[disk_mask])
    smooth_vx = gaussian_filter(vx_masked, smooth_scale)
    smooth_vy = gaussian_filter(vy_masked, smooth_scale)
    smooth_vz = gaussian_filter(vz_masked, smooth_scale)
    smooth_den = gaussian_filter(den_masked, smooth_scale)
    sig_x = (vx_masked - smooth_vx)**2.
    sig_y = (vy_masked - smooth_vy)**2.
    sig_z = (vz_masked - smooth_vz)**2.
    vdisp = np.sqrt((sig_x + sig_y + sig_z)/3.)
    turb_pressure = smooth_den*vdisp**2.
    pres_grad = np.gradient(turb_pressure, dx_cm)
    dPdr = pres_grad[0]*x_hat + pres_grad[1]*y_hat + pres_grad[2]*z_hat
    turb_force = -1./den_masked * dPdr
    vr = box['radial_velocity_corrected'].in_units('cm/s').v
    vr_edges = vr[disk_edges]
    vr_interp_func = NearestNDInterpolator(list(zip(x_edges,y_edges,z_edges)), vr_edges)
    vr_masked = np.copy(vr)
    vr_masked[disk_mask] = vr_interp_func(x[disk_mask], y[disk_mask], z[disk_mask])
    vr_masked = gaussian_filter(vr_masked, smooth_scale)
    dvr = np.gradient(vr_masked, dx_cm)
    delta_vr = dvr[0]*dx_cm*x_hat + dvr[1]*dx_cm*y_hat + dvr[2]*dx_cm*z_hat
    ram_pressure = smooth_den*(delta_vr)**2.
    pres_grad = np.gradient(ram_pressure, dx_cm)
    dPdr = pres_grad[0]*x_hat + pres_grad[1]*y_hat + pres_grad[2]*z_hat
    ram_force = -1./den_masked * dPdr
    vtheta = box['theta_velocity_corrected'].in_units('cm/s').v
    vphi = box['phi_velocity_corrected'].in_units('cm/s').v
    vtheta_edges = vtheta[disk_edges]
    vphi_edges = vphi[disk_edges]
    vtheta_interp_func = NearestNDInterpolator(list(zip(x_edges,y_edges,z_edges)), vtheta_edges)
    vtheta_masked = np.copy(vtheta)
    vphi_interp_func = NearestNDInterpolator(list(zip(x_edges,y_edges,z_edges)), vphi_edges)
    vphi_masked = np.copy(vphi)
    vtheta_masked[disk_mask] = vtheta_interp_func(x[disk_mask], y[disk_mask], z[disk_mask])
    vphi_masked[disk_mask] = vphi_interp_func(x[disk_mask], y[disk_mask], z[disk_mask])
    smooth_vtheta = gaussian_filter(vtheta_masked, smooth_scale)
    smooth_vphi = gaussian_filter(vphi_masked, smooth_scale)
    rot_force = (smooth_vtheta**2. + smooth_vphi**2.)/r
    grav_force = -G*Menc_profile(r/(1000*cmtopc))*gtoMsun/r**2.
    tot_force = thermal_force + turb_force + rot_force + ram_force + grav_force
    forces = [thermal_force, turb_force, ram_force, rot_force, grav_force, tot_force]

    radius_func = RegularGridInterpolator(coords, radius)
    density_func = RegularGridInterpolator(coords, den_masked)
    temperature_func = RegularGridInterpolator(coords, temp_masked)
    radial_velocity_func = RegularGridInterpolator(coords, vr)
    ray_start = [0, 0, 0]
    # List end points of rays
    # Want: 3 within 30 deg opening angle of minor axis on both sides, 3 within 30 deg opening angle
    # of major axis on both sides, 2 close to halfway between major and minor axis in all 4 quadrants
    # = 6 + 6 + 8 = 20 total rays
    #ray_ends = [[0, -100, -150], [0, 150, -150], [0, 100, 150], [0, -150, 150]]
    ray_ends = [[0, -75, -100], [0, -50, 100]]
    rays = ray_ends

    plot_colors = ['r', 'g', 'm', 'b', 'gold', 'k']
    force_linestyles = ['-', '--', ':', '-.', '--', '-']
    linestyles = ['-', '--', ':', '-.']
    labels = ['Thermal', 'Turbulent', 'Ram', 'Rotation', 'Gravity', 'Total']
    ftypes = ['thermal', 'turbulent', 'ram', 'rotation', 'gravity', 'total']

    fig2 = plt.figure(figsize=(12,10), dpi=500)
    ax2 = fig2.add_subplot(1,1,1)
    im = ax2.imshow(rotate(temp_masked[len(temp_masked)//2,:,:], 90), cmap=sns.blend_palette(('salmon', "#984ea3", "#4daf4a", "#ffe34d", 'darkorange'), as_cmap=True), norm=colors.LogNorm(vmin=10**4, vmax=10**7), \
          extent=[np.min(coords[1]),np.max(coords[1]),np.min(coords[2]),np.max(coords[2])])
    for r in range(len(rays)):
        ray_end = rays[r]
        ax2.plot([ray_start[1], ray_end[1]], [ray_start[2], ray_end[2]], color='k', ls=linestyles[r], lw=2)
    ax2.axis([-150,150,-150,150])
    ax2.set_xlabel('y [kpc]', fontsize=20)
    ax2.set_ylabel('z [kpc]', fontsize=20)
    ax2.tick_params(axis='both', which='both', direction='in', length=8, width=2, pad=5, labelsize=20, \
      top=True, right=True)
    cax = fig2.add_axes([0.85, 0.08, 0.03, 0.9])
    cax.tick_params(axis='both', which='both', direction='in', length=8, width=2, pad=5, labelsize=20, \
      top=True, right=True)
    fig2.colorbar(im, cax=cax, orientation='vertical')
    ax2.text(1.15, 0.5, 'Temperature [K]', fontsize=20, rotation='vertical', ha='center', va='center', transform=ax2.transAxes)
    fig2.subplots_adjust(bottom=0.08, top=0.98, left=0.1, right=0.85)
    fig2.savefig(save_dir + snap + '_temperature_slice_x_rays' + save_suffix + '.png')
    plt.close(fig2)
    #plt.show()

    ray_figs = []
    ray_axs = []
    for r in range(len(rays)):
        ray_figs.append(plt.figure(figsize=(8,6), dpi=500))
        ray_axs.append(ray_figs[-1].add_subplot(1,1,1))

    for i in range(len(forces)):
        # Fig 2 is a slice plot of each force for all rays, one per force
        fig2 = plt.figure(figsize=(12,10),dpi=500)
        ax2 = fig2.add_subplot(1,1,1)
        # Fig. 3 is plot of each force for all rays, one per force
        fig3 = plt.figure(figsize=(8,6), dpi=500)
        ax3 = fig3.add_subplot(1,1,1)
        force = forces[i]
        im = ax2.imshow(rotate(force[len(force)//2,:,:], 90), cmap='BrBG', norm=colors.SymLogNorm(vmin=-1e-5, vmax=1e-5, linthresh=1e-9, base=10), \
              extent=[np.min(coords[1]),np.max(coords[1]),np.min(coords[2]),np.max(coords[2])])
        for r in range(len(rays)):
            # Fig 1 is a plot of all forces for each ray, one per ray
            fig1 = ray_figs[r]
            ax1 = ray_axs[r]
            ray_end = rays[r]
            xpoints = np.linspace(ray_start[0], ray_end[0], 100)
            ypoints = np.linspace(ray_start[1], ray_end[1], 100)
            zpoints = np.linspace(ray_start[2], ray_end[2], 100)
            points = np.array([xpoints, ypoints, zpoints]).transpose()
            radius_ray = radius_func(points)
            density_ray = density_func(points)
            temperature_ray = temperature_func(points)
            rv_ray = radial_velocity_func(points)
            den_ray = density_func(points)
            force_func = RegularGridInterpolator(coords, force)
            force_ray = force_func(points)
            work_done = np.sum(force_ray[:-1]*np.diff(radius_ray)*1000*cmtopc*density_ray[:-1])
            print('Ray', r, labels[i], work_done)
            radius_ray = radius_ray[(density_ray < cgm_density_max * density_cut_factor)]
            rv_ray = rv_ray[(density_ray < cgm_density_max * density_cut_factor)]
            force_ray = force_ray[(density_ray < cgm_density_max * density_cut_factor)]
            density_ray = density_ray[(density_ray < cgm_density_max * density_cut_factor)]
            ax1.plot(radius_ray, force_ray, color=plot_colors[i], ls=force_linestyles[i], lw=2, label=labels[i])
            ax3.plot(radius_ray, force_ray, color=plot_colors[i], ls=linestyles[r], lw=2)
            ax2.plot([ray_start[1], ray_end[1]], [ray_start[2], ray_end[2]], color='k', ls='-', lw=2)

            if (i==len(forces)-1):
                ax1.plot([0,150],[0,0], 'k-', lw=1)
                ax1.set_xlabel('Distance along ray from galaxy center [kpc]', fontsize=18)
                ax1.set_ylabel('Force on CGM gas [cm/s$^2$]', fontsize=18)
                ax1.axis([0,150,-1e-5,1e-5])
                ax1.set_yscale('symlog', linthreshy=1e-9)
                ax1.tick_params(axis='both', which='both', direction='in', length=8, width=2, pad=5, labelsize=18, \
                  top=True, right=True)
                if (r==1): ax1.legend(loc=1, frameon=False, fontsize=14)
                fig1.subplots_adjust(top=0.94,bottom=0.11,right=0.95,left=0.17)
                fig1.savefig(save_dir + snap + '_forces_along_ray-' + str(r) + save_suffix + '.png')
                plt.close(fig1)

                # Fig 4 is a plot of radial velocity and total force along each ray, one per ray
                fig4 = plt.figure(figsize=(8,6), dpi=500)
                ax4 = fig4.add_subplot(1,1,1)
                ax4.plot(radius_ray, rv_ray, color='k', ls='--', lw=2, label='Radial velocity')
                ax4.axis([0,150,-1500*1e5,1500*1e5])
                ax4.set_ylabel('Radial velocity [cm/s]', fontsize=18)
                ax4.tick_params(axis='both', which='both', direction='in', length=8, width=2, pad=5, labelsize=18, right=False)
                ax4.set_xlabel('Distance along ray from galaxy center [kpc]', fontsize=18)
                ax4a = ax4.twinx()
                ax4a.plot(radius_ray, force_ray, 'k-', lw=2)
                ax4.plot([0,150], [0,0], 'k-', lw=1, label='Total force (right axis)')
                ax4a.tick_params(axis='y', which='both', direction='in', length=8, width=2, pad=5, labelsize=18, right=True)
                ax4a.set_ylim(-1e-5,1e-5)
                ax4a.set_yscale('symlog', linthreshy=1e-9)
                ax4a.set_ylabel('Force [cm/s$^2$]', fontsize=18)
                ax4.legend(loc=1, frameon=False, fontsize=14)
                fig4.subplots_adjust(top=0.94,bottom=0.11,right=0.85,left=0.15)
                fig4.savefig(save_dir + snap + '_rv-and-force_along_ray-' + str(r) + save_suffix + '.png')
                plt.close(fig4)

                # Fig 5 is a plot of density and total force along each ray, one per ray
                fig5 = plt.figure(figsize=(8,6), dpi=500)
                ax5 = fig5.add_subplot(1,1,1)
                ax5.plot(radius_ray, np.log10(density_ray), color='k', ls='--', lw=2, label='Density')
                ax5.axis([0,150,-33,-20])
                ax5.set_ylabel('log Density [g/cm$^3$]', fontsize=18)
                ax5.tick_params(axis='both', which='both', direction='in', length=8, width=2, pad=5, labelsize=18, right=False)
                ax5.set_xlabel('Distance along ray from galaxy center [kpc]', fontsize=18)
                ax5a = ax5.twinx()
                ax5a.plot(radius_ray, force_ray, 'k-', lw=2)
                ax5.plot([0,150], [0,0], 'k-', lw=1, label='Total force (right axis)')
                ax5a.tick_params(axis='y', which='both', direction='in', length=8, width=2, pad=5, labelsize=18, right=True)
                ax5a.set_ylim(-1e-5,1e-5)
                ax5a.set_yscale('symlog', linthreshy=1e-9)
                ax5a.set_ylabel('Force [cm/s$^2$]', fontsize=18)
                ax5.legend(loc=1, frameon=False, fontsize=14)
                fig5.subplots_adjust(top=0.94,bottom=0.11,right=0.85,left=0.15)
                fig5.savefig(save_dir + snap + '_den-and-force_along_ray-' + str(r) + save_suffix + '.png')
                plt.close(fig5)


        ax2.axis([-150,150,-150,150])
        ax2.set_xlabel('y [kpc]', fontsize=20)
        ax2.set_ylabel('z [kpc]', fontsize=20)
        if (i<len(ftypes)-1): ax2.text(-125,125,labels[i],fontsize=20,ha='left',va='center')
        else: ax2.text(-125,125,'$F_\mathrm{net}$',fontsize=20,ha='left',va='center')
        ax2.tick_params(axis='both', which='both', direction='in', length=8, width=2, pad=5, labelsize=20, \
          top=True, right=True)
        cax = fig2.add_axes([0.85, 0.08, 0.03, 0.9])
        cax.tick_params(axis='both', which='both', direction='in', length=8, width=2, pad=5, labelsize=20, \
          top=True, right=True)
        fig2.colorbar(im, cax=cax, orientation='vertical')
        ax2.text(1.17, 0.5, labels[i] + ' Force [cm/s$^2$]', fontsize=20, rotation='vertical', ha='center', va='center', transform=ax2.transAxes)
        fig2.subplots_adjust(bottom=0.08, top=0.98, left=0.1, right=0.85)
        fig2.savefig(save_dir + snap + '_' + ftypes[i] + '_force_slice_x_ray' + save_suffix + '.png')
        plt.close(fig2)

        ax3.set_xlabel('Distance along ray from galaxy center [kpc]', fontsize=18)
        ax3.set_ylabel(labels[i] + ' Force on CGM gas [cm/s$^2$]', fontsize=18)
        ax3.axis([0,150,-1e-5,1e-5])
        ax3.plot([0,150],[0,0], 'k-', lw=1)
        ax3.set_yscale('symlog', linthreshy=1e-9)
        ax3.tick_params(axis='both', which='both', direction='in', length=8, width=2, pad=5, labelsize=18, \
          top=True, right=True)
        ax3.legend(loc=1, frameon=False, fontsize=14)
        fig3.subplots_adjust(top=0.94,bottom=0.11,right=0.95,left=0.17)
        fig3.savefig(save_dir + snap + '_' + ftypes[i] + '_force_along_rays' + save_suffix + '.png')
        plt.close(fig3)

    # Delete output or dummy directory from temp directory if on pleiades
    if (args.system=='pleiades_cassi'):
        print('Deleting directory from /tmp')
        shutil.rmtree(snap_dir)

def support_vs_radius(snap):
    '''Plots the ratio of different types of force (thermal, turbulent, rotational, ram)
    to gravity as functions of radius for the simulation output given by 'snap'. Requires a file
    already created by force_vs_radius to exist for the snapshot desired.'''

    tablename_prefix = output_dir + 'stats_halo_00' + args.halo + '/' + args.run + '/Tables/'
    stats = Table.read(tablename_prefix + snap + '_stats_force-types' + args.filename + '.hdf5', path='all_data')
    Rvir = rvir_masses['radius'][rvir_masses['snapshot']==snap]

    plot_colors = ['r', 'g', 'm', 'b', 'k']
    #plot_colors = ['r', 'g', 'm', 'b']
    plot_labels = ['Thermal', 'Turbulent', 'Ram', 'Rotation', 'Sum']
    file_labels = ['thermal_force', 'turbulent_force', 'ram_force', 'rotation_force']
    save_labels = ['thermal', 'turbulent', 'ram', 'rotation', 'sum']
    linestyles = ['-', '--', ':', '-.', '-']

    radius_list = 0.5*(stats['inner_radius'] + stats['outer_radius'])
    zsnap = stats['redshift'][0]

    grav_force = -stats['gravity_force_sum']

    fig = plt.figure(figsize=(8,6), dpi=500)
    ax = fig.add_subplot(1,1,1)

    sum_list = np.zeros(len(radius_list))

    for i in range(len(plot_colors)):
        label = plot_labels[i]
        if (plot_labels[i]!='Sum'):
            sum_list += stats[file_labels[i] + '_sum']
            ax.plot(radius_list, stats[file_labels[i] + '_sum']/grav_force, ls=linestyles[i], color=plot_colors[i], lw=2, label=label)
        else:
            ax.plot(radius_list, sum_list/grav_force, ls=linestyles[i], color=plot_colors[i], lw=2, label=label)

    ax.set_ylabel('Force / Gravitational Force', fontsize=22)
    ax.set_xlabel('Radius [kpc]', fontsize=22)
    ax.axis([0,250,-2,4])
    ax.text(15, -1., '$z=%.2f$' % (zsnap), fontsize=22, ha='left', va='center')
    ax.text(15,-1.5,halo_dict[args.halo],ha='left',va='center',fontsize=22)
    ax.tick_params(axis='both', which='both', direction='in', length=8, width=2, pad=5, labelsize=22, \
      top=True, right=True)
    ax.plot([Rvir, Rvir], [ax.get_ylim()[0], ax.get_ylim()[1]], 'k--', lw=1)
    ax.text(Rvir-3., -1.5, '$R_{200}$', fontsize=22, ha='right', va='center')
    ax.plot([ax.get_xlim()[0], ax.get_xlim()[1]], [0,0], 'k-', lw=1)
    ax.plot([ax.get_xlim()[0], ax.get_xlim()[1]], [1,1], 'k--', lw=1)
    if (halo_dict[args.halo]=='Tempest'): ax.legend(loc=1, frameon=False, fontsize=22, ncol=2)
    plt.subplots_adjust(top=0.96,bottom=0.13,right=0.95,left=0.12)
    plt.savefig(save_dir + snap + '_support_vs_r' + save_suffix + '.png')
    plt.close(fig)

    if (args.region_filter!='none'):
        regions = ['low_', 'mid_', 'high_']
        alphas = [0.25,0.6,1.]
        # Fig1 is a plot of all force types for all regions on one plot
        fig1 = plt.figure(figsize=(8,6), dpi=500)
        ax1 = fig1.add_subplot(1,1,1)
        # Fig2 is plots of all force types for each region, one per region
        figs2 = []
        axs2 = []
        for r in range(len(regions)):
            figs2.append(plt.figure(figsize=(8,6), dpi=500))
            axs2.append(figs2[-1].add_subplot(1,1,1))

        sums_list_regions = []
        for j in range(len(regions)):
            sums_list_regions.append(np.zeros(len(radius_list)))

        for i in range(len(plot_colors)):
            # Fig3 is plots of all regions for each force type, one per force type
            fig3 = plt.figure(figsize=(8,6), dpi=500)
            ax3 = fig3.add_subplot(1,1,1)
            label = plot_labels[i]
            label_regions = ['Low ' + args.region_filter, 'Mid ' + args.region_filter, '$0.1\\times$ High ' + args.region_filter]
            mult_regions = [1., 1., 0.1]
            if (i==0):
                label_regions_bigplot = ['Low ' + args.region_filter, 'Mid ' + args.region_filter, '$0.1\\times$ High ' + args.region_filter]
            else:
                label_regions_bigplot = ['__nolegend__', '__nolegend__', '__nolegend__']
            for j in range(len(regions)):
                if (plot_labels[i]!='Sum'):
                    ax1.plot(radius_list, mult_regions[j]*stats[regions[j] + args.region_filter + '_' + file_labels[i] + '_sum'] / \
                      -stats[regions[j] + args.region_filter + '_gravity_force_sum'], \
                      ls=linestyles[i], color=plot_colors[i], lw=2, alpha=alphas[j], label=label_regions_bigplot[j])
                    axs2[j].plot(radius_list, mult_regions[j]*stats[regions[j] + args.region_filter + '_' + file_labels[i] + '_sum']/ \
                      -stats[regions[j] + args.region_filter + '_gravity_force_sum'], \
                      ls=linestyles[i], color=plot_colors[i], lw=2, label=label)
                    ax3.plot(radius_list, mult_regions[j]*stats[regions[j] + args.region_filter + '_' + file_labels[i] + '_sum']/ \
                      -stats[regions[j] + args.region_filter + '_gravity_force_sum'], \
                      ls=linestyles[i], color=plot_colors[i], lw=2, alpha=alphas[j], label=label_regions[j])
                    sums_list_regions[j] += stats[regions[j] + args.region_filter + '_' + file_labels[i] + '_sum']
                else:
                    ax1.plot(radius_list, mult_regions[j]*sums_list_regions[j]/ \
                      -stats[regions[j] + args.region_filter + '_gravity_force_sum'], \
                      ls=linestyles[i], color=plot_colors[i], lw=2, alpha=alphas[j], label=label_regions_bigplot[j])
                    axs2[j].plot(radius_list, mult_regions[j]*sums_list_regions[j]/ \
                      -stats[regions[j] + args.region_filter + '_gravity_force_sum'], \
                      ls=linestyles[i], color=plot_colors[i], lw=2, label=label)
                    ax3.plot(radius_list, mult_regions[j]*sums_list_regions[j]/ \
                      -stats[regions[j] + args.region_filter + '_gravity_force_sum'], \
                      ls=linestyles[i], color=plot_colors[i], lw=2, alpha=alphas[j], label=label_regions[j])
            if (plot_labels[i]!='Sum'):
                ax1.plot(radius_list, mult_regions[2]*stats['high_' + args.region_filter + '_' + file_labels[i] + '_sum']/ \
                  -stats['high_' + args.region_filter + '_gravity_force_sum'], \
                  ls=linestyles[i], color=plot_colors[i], lw=2, label=label)
                ax3.plot(radius_list, mult_regions[2]*stats['high_' + args.region_filter + '_' + file_labels[i] + '_sum']/ \
                  -stats['high_' + args.region_filter + '_gravity_force_sum'], \
                  ls=linestyles[i], color=plot_colors[i], lw=2, label=label)
            else:
                ax1.plot(radius_list, mult_regions[j]*sums_list_regions[j]/ \
                  -stats[regions[j] + args.region_filter + '_gravity_force_sum'], \
                  ls=linestyles[i], color=plot_colors[i], lw=2, label=label)
                ax3.plot(radius_list, mult_regions[j]*sums_list_regions[j]/ \
                  -stats[regions[j] + args.region_filter + '_gravity_force_sum'], \
                  ls=linestyles[i], color=plot_colors[i], lw=2, label=label)

            ax3.set_ylabel('Force / Gravitational Force', fontsize=22)
            ax3.axis([0,250,-2,4])
            ax3.set_xlabel('Radius [kpc]', fontsize=22)
            ax3.tick_params(axis='both', which='both', direction='in', length=8, width=2, pad=5, labelsize=22, \
              top=True, right=True)
            ax3.legend(loc=1, frameon=False, fontsize=22)
            ax3.text(15, -1., '$z=%.2f$' % (zsnap), fontsize=22, ha='left', va='center')
            ax3.text(15,-1.5,halo_dict[args.halo],ha='left',va='center',fontsize=22)
            ax3.plot([Rvir, Rvir], [ax.get_ylim()[0], ax.get_ylim()[1]], 'k--', lw=1)
            ax3.text(Rvir-3., -1.5, '$R_{200}$', fontsize=22, ha='right', va='center')
            ax3.plot([ax.get_xlim()[0], ax.get_xlim()[1]], [0,0], 'k-', lw=1)
            ax3.plot([ax.get_xlim()[0], ax.get_xlim()[1]], [1,1], 'k--', lw=1)
            fig3.subplots_adjust(top=0.96,bottom=0.13,right=0.95,left=0.12)
            fig3.savefig(save_dir + snap + '_' + save_labels[i] + '_support_vs_r_regions-' + args.region_filter + save_suffix + '.png')
            plt.close(fig3)

        for r in range(len(regions)):
            axs2[r].set_ylabel('Force / Gravitational Force', fontsize=22)
            axs2[r].axis([0,250,-2,4])
            axs2[r].set_xlabel('Radius [kpc]', fontsize=22)
            axs2[r].tick_params(axis='both', which='both', direction='in', length=8, width=2, pad=5, labelsize=22, \
              top=True, right=True)
            axs2[r].text(15, 3.5, label_regions[r], fontsize=22, ha='left', va='center')
            axs2[r].text(15, -1., '$z=%.2f$' % (zsnap), fontsize=22, ha='left', va='center')
            axs2[r].text(15,-1.5,halo_dict[args.halo],ha='left',va='center',fontsize=22)
            axs2[r].plot([Rvir, Rvir], [ax.get_ylim()[0], ax.get_ylim()[1]], 'k--', lw=1)
            axs2[r].text(Rvir-3., -1.5, '$R_{200}$', fontsize=22, ha='right', va='center')
            axs2[r].plot([ax.get_xlim()[0], ax.get_xlim()[1]], [0,0], 'k-', lw=1)
            axs2[r].plot([ax.get_xlim()[0], ax.get_xlim()[1]], [1,1], 'k--', lw=1)
            axs2[r].legend(loc=1, frameon=False, fontsize=22)
            figs2[r].subplots_adjust(top=0.96,bottom=0.13,right=0.95,left=0.12)
            figs2[r].savefig(save_dir + snap + '_support_vs_r_' + regions[r] + args.region_filter + save_suffix + '.png')
            plt.close(figs2[r])

        ax1.set_ylabel('Force / Gravitational Force', fontsize=22)
        ax1.axis([0,250,-2,4])
        ax1.set_xlabel('Radius [kpc]', fontsize=22)
        ax1.tick_params(axis='both', which='both', direction='in', length=8, width=2, pad=5, labelsize=22, \
          top=True, right=True)
        ax1.text(15, -1., '$z=%.2f$' % (zsnap), fontsize=22, ha='left', va='center')
        ax1.text(15,-1.5,halo_dict[args.halo],ha='left',va='center',fontsize=22)
        ax1.plot([Rvir, Rvir], [ax.get_ylim()[0], ax.get_ylim()[1]], 'k--', lw=1)
        ax1.text(Rvir-3., -1.5, '$R_{200}$', fontsize=22, ha='right', va='center')
        ax1.plot([ax.get_xlim()[0], ax.get_xlim()[1]], [0,0], 'k-', lw=1)
        ax1.plot([ax.get_xlim()[0], ax.get_xlim()[1]], [1,1], 'k--', lw=1)
        ax1.legend(loc=1, frameon=False, fontsize=14, ncol=2)
        fig1.subplots_adjust(top=0.96,bottom=0.13,right=0.95,left=0.12)
        fig1.savefig(save_dir + snap + '_all_support_vs_r_regions-' + args.region_filter + save_suffix + '.png')
        plt.close(fig1)

def support_vs_radius_time_averaged(snaplist):
    '''Plots time-averaged support (forces / gravity) as a function of radius, with shading showing
    the time variation of support.'''

    tablename_prefix = output_dir + 'stats_halo_00' + args.halo + '/' + args.run + '/Tables/'
    if (args.filename == ''):
        filename = ''
    else:
        filename = '_' + args.filename

    plot_colors = ['r', 'g', 'm', 'b', 'k']
    plot_labels = ['Thermal', 'Turbulent', 'Ram', 'Rotation', 'Sum']
    file_labels = ['thermal_force', 'turbulent_force', 'ram_force', 'rotation_force']
    save_labels = ['thermal', 'turbulent', 'ram', 'rotation', 'sum']
    linestyles = ['-', '--', ':', '-.', '-']

    fig = plt.figure(figsize=(8,6), dpi=500)
    ax = fig.add_subplot(1,1,1)

    zlist = []
    timelist = []
    forces_list = []
    if (args.region_filter!='none'):
        forces_regions = [[],[],[]]
        avg_forces_regions = [[],[],[]]
        std_forces_regions = [[],[],[]]
        region_label = ['Low ' + args.region_filter, 'Mid ' + args.region_filter, 'High ' + args.region_filter]
        region_name = ['low-', 'mid-', 'high-']
    for j in range(len(plot_labels)):
        forces_list.append([])
        if (args.region_filter!='none'):
            forces_regions[0].append([])
            forces_regions[1].append([])
            forces_regions[2].append([])

    for i in range(len(snaplist)):
        snap = snaplist[i]
        stats = Table.read(tablename_prefix + snap + '_stats_force-types' + filename + '.hdf5', path='all_data')
        Rvir = rvir_masses['radius'][rvir_masses['snapshot']==snap][0]
        radius_list = 0.5*(stats['inner_radius'] + stats['outer_radius'])/Rvir
        sum_list = np.zeros(len(radius_list))
        if (args.region_filter!='none'):
            sum_list_regions = [np.zeros(len(radius_list)), np.zeros(len(radius_list)), np.zeros(len(radius_list))]

        for j in range(len(plot_labels)):
            if (plot_labels[j]!='Sum'):
                forces_list[j].append(stats[file_labels[j] + '_sum']/-stats['gravity_force_sum'])
                sum_list += stats[file_labels[j] + '_sum']
            else:
                forces_list[j].append(sum_list/-stats['gravity_force_sum'])
            if (args.region_filter!='none'):
                if (plot_labels[j]!='Sum'):
                    forces_regions[0][j].append(stats['low_' + args.region_filter + '_' + file_labels[j] + '_sum'] / \
                      -stats['low_' + args.region_filter + '_gravity_force_sum'])
                    forces_regions[1][j].append(stats['mid_' + args.region_filter + '_' + file_labels[j] + '_sum'] / \
                      -stats['mid_' + args.region_filter + '_gravity_force_sum'])
                    forces_regions[2][j].append(0.1*stats['high_' + args.region_filter + '_' + file_labels[j] + '_sum'] / \
                      -stats['high_' + args.region_filter + '_gravity_force_sum'])
                    sum_list_regions[0] += stats['low_' + args.region_filter + '_' + file_labels[j] + '_sum']
                    sum_list_regions[1] += stats['mid_' + args.region_filter + '_' + file_labels[j] + '_sum']
                    sum_list_regions[2] += stats['high_' + args.region_filter + '_' + file_labels[j] + '_sum']
                else:
                    forces_regions[0][j].append(sum_list_regions[0]/-stats['low_' + args.region_filter + '_gravity_force_sum'])
                    forces_regions[1][j].append(sum_list_regions[1]/-stats['mid_' + args.region_filter + '_gravity_force_sum'])
                    forces_regions[2][j].append(0.1*sum_list_regions[2]/-stats['high_' + args.region_filter + '_gravity_force_sum'])

    avg_forces_list = np.nanmean(forces_list, axis=1)
    std_forces_list = np.nanstd(forces_list, axis=1)
    if (args.region_filter!='none'):
        avg_forces_regions[0] = np.nanmean(forces_regions[0], axis=1)
        avg_forces_regions[1] = np.nanmean(forces_regions[1], axis=1)
        avg_forces_regions[2] = np.nanmean(forces_regions[2], axis=1)
        std_forces_regions[0] = np.nanstd(forces_regions[0], axis=1)
        std_forces_regions[1] = np.nanstd(forces_regions[1], axis=1)
        std_forces_regions[2] = np.nanstd(forces_regions[2], axis=1)

    for i in range(len(plot_colors)):
        label = plot_labels[i]
        ax.plot(radius_list, avg_forces_list[i], ls=linestyles[i], color=plot_colors[i], \
                lw=2, label=label)
        ax.fill_between(radius_list, avg_forces_list[i]-std_forces_list[i], \
                        avg_forces_list[i]+std_forces_list[i], alpha=0.1, color=plot_colors[i])

    ax.set_ylabel('Force / Gravitational Force ', fontsize=20)
    ax.axis([0,1.5,-2,4])
    ax.text(1.4, -1.5, halo_dict[args.halo], ha='right', va='center', fontsize=20)
    ax.set_xlabel('Radius [$R_{200}$]', fontsize=20)
    ax.tick_params(axis='both', which='both', direction='in', length=8, width=2, pad=5, labelsize=20, \
      top=True, right=True)
    ax.plot([ax.get_xlim()[0], ax.get_xlim()[1]], [0,0], 'k-', lw=1)
    ax.plot([ax.get_xlim()[0], ax.get_xlim()[1]], [1,1], 'k--', lw=1)
    if (halo_dict[args.halo]=='Tempest'): ax.legend(loc=1, frameon=False, fontsize=20, ncol=2)
    fig.subplots_adjust(top=0.96,bottom=0.12,right=0.98,left=0.11)
    fig.savefig(save_dir + 'support_vs_r_time-avg' + save_suffix + '.png')
    plt.close(fig)

    if (args.region_filter!='none'):
        regions = ['low_', 'mid_', 'high_']
        label_regions = ['Low ' + args.region_filter, 'Mid ' + args.region_filter, 'High ' + args.region_filter]
        # Fig2 is plots of all force types for each region, one per region
        figs2 = []
        axs2 = []
        for r in range(len(regions)):
            figs2.append(plt.figure(figsize=(8,6), dpi=500))
            axs2.append(figs2[-1].add_subplot(1,1,1))

        for i in range(len(plot_colors)):
            label = plot_labels[i]
            for j in range(len(regions)):
                axs2[j].plot(radius_list, avg_forces_regions[j][i], \
                  ls=linestyles[i], color=plot_colors[i], lw=2, label=label)
                axs2[j].fill_between(radius_list, avg_forces_regions[j][i]-std_forces_regions[j][i], \
                                     avg_forces_regions[j][i]+std_forces_regions[j][i], color=plot_colors[i], alpha=0.1)

        for r in range(len(regions)):
            axs2[r].set_ylabel('Force / Gravitational Force ', fontsize=24)
            if (r<2): axs2[r].axis([0,1.5,-2,4])
            else: axs2[r].axis([0,1.5,-10,15])
            if (halo_dict[args.halo]=='Tempest'): axs2[r].text(0.5, 0.96, label_regions[r], fontsize=36, ha='center', va='center', transform=ax.transAxes)
            if (r==0): axs2[r].text(0.9,0.1,halo_dict[args.halo],ha='right',va='center',fontsize=24, transform=ax.transAxes)
            axs2[r].plot([ax.get_xlim()[0], ax.get_xlim()[1]], [0,0], 'k-', lw=1)
            axs2[r].plot([ax.get_xlim()[0], ax.get_xlim()[1]], [1,1], 'k--', lw=1)
            axs2[r].set_xlabel('Radius [$R_{200}$]', fontsize=24)
            axs2[r].tick_params(axis='both', which='both', direction='in', length=8, width=2, pad=5, labelsize=24, \
              top=True, right=True)
            if (halo_dict[args.halo]=='Tempest') and (r==0): axs2[r].legend(loc=1, frameon=False, fontsize=24, ncol=2)
            if (halo_dict[args.halo]=='Tempest'):
                if (r<2): figs2[r].subplots_adjust(top=0.87,bottom=0.14,right=0.95,left=0.13)
                else: figs2[r].subplots_adjust(top=0.87,bottom=0.14,right=0.95,left=0.17)
            else:
                if (r<2): figs2[r].subplots_adjust(top=0.96,bottom=0.14,right=0.95,left=0.13)
                else: figs2[r].subplots_adjust(top=0.96,bottom=0.14,right=0.95,left=0.17)
            figs2[r].savefig(save_dir + 'support_vs_r_time-avg_' + regions[r] + args.region_filter + save_suffix + '.png')
            plt.close(figs2[r])

def support_vs_time(snaplist):
    '''Plots ratio of different forces to gravity at a given radius or averaged over a range of radii
    over time, for all snaps in the list snaplist.'''

    tablename_prefix = output_dir + 'stats_halo_00' + args.halo + '/' + args.run + '/Tables/'
    time_table = Table.read(output_dir + 'times_halo_00' + args.halo + '/' + args.run + '/time_table.hdf5', path='all_data')
    if (args.run != 'feedback_return'):
        masses_dir = code_path + 'halo_infos/00' + args.halo + '/' + args.run + '/'
        rvir_masses = Table.read(masses_dir + 'rvir_masses.hdf5', path='all_data')

    fig = plt.figure(figsize=(8,6), dpi=500)
    ax = fig.add_subplot(1,1,1)

    plot_colors = ['r', 'g', 'm', 'b', 'k']
    plot_labels = ['Thermal', 'Turbulent', 'Ram', 'Rotation', 'Sum']
    file_labels = ['thermal_force', 'turbulent_force', 'ram_force', 'rotation_force']
    save_labels = ['thermal', 'turbulent', 'ram', 'rotation', 'sum']
    linestyles = ['-', '--', ':', '-.', '-']
    alphas = [0.3, 0.6, 1.]

    if (args.time_avg!=0):
        dt = 5.38*args.output_step
        avg_window = int(np.ceil(args.time_avg/dt))

    if (args.radius_range!='none'):
        radius_range = ast.literal_eval(args.radius_range)

    zlist = []
    timelist = []
    forces_list = []
    if (args.region_filter!='none'):
        forces_regions = [[],[],[]]
        region_label = ['Low ' + args.region_filter, 'Mid ' + args.region_filter, 'High ' + args.region_filter]
        region_name = ['low-', 'mid-', 'high-']
    for j in range(len(plot_labels)):
        forces_list.append([])
        if (args.region_filter!='none'):
            forces_regions[0].append([])
            forces_regions[1].append([])
            forces_regions[2].append([])

    for i in range(len(snaplist)):
        snap = snaplist[i]
        if (args.run=='feedback_return'):
            if (int(snap[2:])<1737):
                tablename_prefix = output_dir + 'stats_halo_00' + args.halo + '/nref11c_nref9f/Tables/'
                masses_dir = code_path + 'halo_infos/00' + args.halo + '/nref11c_nref9f/'
                rvir_masses = Table.read(masses_dir + 'rvir_masses.hdf5', path='all_data')
            elif (int(snap[2:])<1747):
                tablename_prefix = output_dir + 'stats_halo_00' + args.halo + '/high_feedback_restart/Tables/'
                masses_dir = code_path + 'halo_infos/00' + args.halo + '/high_feedback_restart/'
                rvir_masses = Table.read(masses_dir + 'rvir_masses.hdf5', path='all_data')
            else:
                tablename_prefix = output_dir + 'stats_halo_00' + args.halo + '/feedback_return/Tables/'
                masses_dir = code_path + 'halo_infos/00' + args.halo + '/feedback_return/'
                rvir_masses = Table.read(masses_dir + 'rvir_masses.hdf5', path='all_data')
        stats = Table.read(tablename_prefix + snap + '_stats_force-types' + args.filename + '.hdf5', path='all_data')
        radius_list = 0.5*(stats['inner_radius'] + stats['outer_radius'])
        Rvir = rvir_masses['radius'][rvir_masses['snapshot']==snap][0]
        if (args.radius_range!='none'):
            radius_in = radius_range[0]*Rvir
            radius_out = radius_range[1]*Rvir
            rad_in = np.where(stats['inner_radius']<=radius_in)[0][-1]
            rad_out = np.where(stats['outer_radius']>=radius_out)[0][0]
        else:
            radius = args.radius*Rvir
            rad_ind = np.where(stats['inner_radius']<=radius)[0][-1]
        timelist.append(time_table['time'][time_table['snap']==snap][0]/1000.)
        zlist.append(stats['redshift'][0])
        grav_list = -stats['gravity_force_sum']
        sum_list = np.zeros(len(radius_list))
        if (args.region_filter!='none'):
            sum_list_low = np.zeros(len(radius_list))
            sum_list_mid = np.zeros(len(radius_list))
            sum_list_high = np.zeros(len(radius_list))
        for j in range(len(plot_labels)):
            if (plot_labels[j]!='Sum'):
                sum_list += stats[file_labels[j] + '_sum']
                if (args.radius_range!='none'):
                    forces_list[j].append(np.mean(stats[file_labels[j] + '_sum'][rad_in:rad_out]/grav_list[rad_in:rad_out]))
                else:
                    forces_list[j].append(stats[file_labels[j] + '_sum'][rad_ind]/grav_list[rad_ind])
            else:
                if (args.radius_range!='none'):
                    forces_list[j].append(np.mean(sum_list[rad_in:rad_out]/grav_list[rad_in:rad_out]))
                else:
                    forces_list[j].append(sum_list[rad_ind]/grav_list[rad_ind])
            if (args.region_filter!='none'):
                grav_list_low = -stats['low_' + args.region_filter + '_gravity_force_sum']
                grav_list_mid = -stats['mid_' + args.region_filter + '_gravity_force_sum']
                grav_list_high = -stats['high_' + args.region_filter + '_gravity_force_sum']
                if (plot_labels[j]!='Sum'):
                    sum_list_low += stats['low_' + args.region_filter + '_' + file_labels[j] + '_sum']
                    sum_list_mid += stats['mid_' + args.region_filter + '_' + file_labels[j] + '_sum']
                    sum_list_high += stats['high_' + args.region_filter + '_' + file_labels[j] + '_sum']
                    if (args.radius_range!='none'):
                        forces_regions[0][j].append(np.mean(stats['low_' + args.region_filter + '_' + file_labels[j] + '_sum'][rad_in:rad_out]/grav_list_low[rad_in:rad_out]))
                        forces_regions[1][j].append(np.mean(stats['mid_' + args.region_filter + '_' + file_labels[j] + '_sum'][rad_in:rad_out]/grav_list_mid[rad_in:rad_out]))
                        forces_regions[2][j].append(np.mean(stats['high_' + args.region_filter + '_' + file_labels[j] + '_sum'][rad_in:rad_out]/grav_list_high[rad_in:rad_out]))
                    else:
                        forces_regions[0][j].append(stats['low_' + args.region_filter + '_' + file_labels[j] + '_sum'][rad_ind]/grav_list_low[rad_ind])
                        forces_regions[1][j].append(stats['mid_' + args.region_filter + '_' + file_labels[j] + '_sum'][rad_ind]/grav_list_mid[rad_ind])
                        forces_regions[2][j].append(stats['high_' + args.region_filter + '_' + file_labels[j] + '_sum'][rad_ind]/grav_list_high[rad_ind])
                else:
                    if (args.radius_range!='none'):
                        forces_regions[0][j].append(np.mean(sum_list_low[rad_in:rad_out]/grav_list_low[rad_in:rad_out]))
                        forces_regions[1][j].append(np.mean(sum_list_mid[rad_in:rad_out]/grav_list_mid[rad_in:rad_out]))
                        forces_regions[2][j].append(np.mean(sum_list_high[rad_in:rad_out]/grav_list_high[rad_in:rad_out]))
                    else:
                        forces_regions[0][j].append(sum_list_low[rad_ind]/grav_list_low[rad_ind])
                        forces_regions[1][j].append(sum_list_mid[rad_ind]/grav_list_mid[rad_ind])
                        forces_regions[2][j].append(sum_list_high[rad_ind]/grav_list_high[rad_ind])

    if (args.time_avg!=0):
        forces_list_avgd = []
        if (args.region_filter!='none'):
            forces_regions_avgd = [[], [], []]
        for j in range(len(plot_labels)):
            forces_list_avgd.append(uniform_filter1d(forces_list[j], size=avg_window))
            if (args.region_filter!='none'):
                for k in range(3):
                    forces_regions_avgd[k].append(uniform_filter1d(forces_regions[k][j], size=avg_window))
        forces_list = forces_list_avgd
        if (args.region_filter!='none'):
            forces_regions = forces_regions_avgd

    zlist.reverse()
    timelist.reverse()
    time_func = IUS(zlist, timelist)
    timelist.reverse()
    timelist = np.array(timelist).flatten()
    zlist = np.array(zlist)

    fig = plt.figure(figsize=(8,6), dpi=500)
    ax = fig.add_subplot(1,1,1)

    for j in range(len(plot_labels)):
        ax.plot(timelist, forces_list[j], lw=2, ls=linestyles[j], color=plot_colors[j], label=plot_labels[j])

    ax.plot([np.min(timelist), np.max(timelist)], [0,0], 'k-', lw=1)
    ax.plot([np.min(timelist), np.max(timelist)], [1,1], 'k--', lw=1)
    ax.axis([np.min(timelist), np.max(timelist), -2,4])
    if (args.feedback_diff):
        if (args.run=='nref11c_nref9f'): ax.text(13,-0.4,'Fiducial',ha='right',va='center',fontsize=20)
        if (args.run=='feedback_return'): ax.text(12.5,-0.4,'Strong burst',ha='right',va='center',fontsize=20)
        if (args.run=='low_feedback_06'): ax.text(13,-0.4,'Weak feedback',ha='right',va='center',fontsize=20)
    else: ax.text(13,-0.4, halo_dict[args.halo], ha='right', va='center', fontsize=20)
    ax.set_ylabel('Force / Gravitational Force', fontsize=20)
    if (args.radius_range!='none'):
        if (halo_dict[args.halo]=='Tempest'):
            if (not args.feedback_diff) or ((args.feedback_diff) and (args.run=='nref11c_nref9f')):
                ax.text(0.5,1.2, '$r=%.2f-%.2f R_{200}$' % (radius_range[0], radius_range[1]), ha='center', va='center', fontsize=24, transform=ax.transAxes)
    else:
        if (halo_dict[args.halo]=='Tempest'):
            if (not args.feedback_diff) or ((args.feedback_diff) and (args.run=='nref11c_nref9f')):
                ax.text(0.5,1.2, '$r=%.2f R_{200}$' % (args.radius), ha='right', va='center', fontsize=24, transform=ax.transAxes)
    if (halo_dict[args.halo]=='Tempest'):
        if (not args.feedback_diff) or ((args.feedback_diff) and (args.run=='nref11c_nref9f')):
            ax.legend(loc=1, frameon=False, fontsize=20, ncol=2)

    ax2 = ax.twiny()
    ax.tick_params(axis='both', which='both', direction='in', length=8, width=2, pad=5, labelsize=20, \
      top=False, right=True)
    ax2.tick_params(axis='x', which='both', direction='in', length=8, width=2, pad=5, labelsize=20, \
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
    ax2.set_xlabel('Redshift', fontsize=20)
    ax.set_xlabel('Time [Gyr]', fontsize=20)

    z_sfr, sfr = np.loadtxt(code_path + 'halo_infos/00' + args.halo + '/' + args.run + '/sfr', unpack=True, usecols=[1,2], skiprows=1)
    t_sfr = time_func(z_sfr)

    ax3 = ax.twinx()
    ax3.plot(t_sfr, sfr, 'k-', lw=1)
    ax.plot([timelist[0],timelist[-1]], [0,0], 'k-', lw=1, label='SFR (right axis)')
    ax3.tick_params(axis='y', which='both', direction='in', length=8, width=2, pad=5, labelsize=20, right=True)
    if (args.feedback_diff): ax3.set_ylim(-5,50)
    else: ax3.set_ylim(-5,200)
    ax3.set_ylabel('SFR [$M_\odot$/yr]', fontsize=20)

    if (halo_dict[args.halo]=='Tempest'):
        if (not args.feedback_diff) or ((args.feedback_diff) and (args.run=='nref11c_nref9f')):
            fig.subplots_adjust(left=0.12, bottom=0.12, right=0.87, top=0.82)
        else:
            fig.subplots_adjust(left=0.12, bottom=0.12, right=0.87, top=0.89)
    else: fig.subplots_adjust(left=0.12, bottom=0.12, right=0.87, top=0.89)
    fig.savefig(save_dir + 'support_vs_time' + save_suffix + '.png')
    plt.close(fig)

    if (args.region_filter!='none'):
        fig_regions = []
        axs_regions = []
        for i in range(len(plot_colors)):
            fig_regions.append(plt.figure(figsize=(8,6), dpi=500))
            axs_regions.append(fig_regions[-1].add_subplot(1,1,1))
        for i in range(3):
            fig = plt.figure(figsize=(8,6), dpi=500)
            ax = fig.add_subplot(1,1,1)

            for j in range(len(plot_labels)):
                axs_regions[j].plot(timelist, forces_regions[i][j], lw=2, ls=linestyles[j], color=plot_colors[j], alpha=alphas[i], label=region_label[i])
                ax.plot(timelist, forces_regions[i][j], lw=2, ls=linestyles[j], color=plot_colors[j], label=plot_labels[j])

            ax.plot([np.min(timelist), np.max(timelist)], [0,0], 'k-', lw=1)
            ax.plot([np.min(timelist), np.max(timelist)], [1,1], 'k--', lw=1)
            ax.axis([np.min(timelist), np.max(timelist), -2,4])
            ax.set_ylabel('Force / Gravitational Force]', fontsize=14)
            if (args.radius_range!='none'):
                ax.text(13,-1.5, '$r=%.2f-%.2f R_{200}$' % (radius_range[0], radius_range[1]), ha='right', va='center', fontsize=14)
            else:
                ax.text(13,-1.5, '$r=%.2f R_{200}$' % (args.radius), ha='right', va='center', fontsize=14)
            ax.text(4,3.6, region_label[i], fontsize=14, ha='left', va='center')

            ax.legend(loc=1, frameon=False, fontsize=14, ncol=2)
            ax2 = ax.twiny()
            ax.tick_params(axis='both', which='both', direction='in', length=8, width=2, pad=5, labelsize=14, \
              top=False, right=True)
            ax2.tick_params(axis='x', which='both', direction='in', length=8, width=2, pad=5, labelsize=14, \
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
            ax2.set_xlabel('Redshift', fontsize=14)
            ax.set_xlabel('Time [Gyr]', fontsize=14)

            z_sfr, sfr = np.loadtxt(code_path + 'halo_infos/00' + args.halo + '/' + args.run + '/sfr', unpack=True, usecols=[1,2], skiprows=1)
            t_sfr = time_func(z_sfr)

            ax3 = ax.twinx()
            ax3.plot(t_sfr, sfr, 'k-', lw=1)
            ax.plot([timelist[0],timelist[-1]], [0,0], 'k-', lw=1, label='SFR (right axis)')
            ax3.tick_params(axis='y', which='both', direction='in', length=8, width=2, pad=5, labelsize=14, right=True)
            ax3.set_ylim(-5,200)
            ax3.set_ylabel('SFR [$M_\odot$/yr]', fontsize=14)

            fig.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.92)
            fig.savefig(save_dir + 'support_vs_time_region-' + region_name[i] + args.region_filter + save_suffix + '.png')
            plt.close(fig)

        for j in range(len(plot_labels)):
            axs_regions[j].axis([np.min(timelist), np.max(timelist), -2,4])
            axs_regions[j].set_ylabel('Force / Gravitational Force', fontsize=14)
            if (args.radius_range!='none'):
                axs_regions[j].text(13,-1.5, '$r=%.2f-%.2f R_{200}$' % (radius_range[0], radius_range[1]), ha='right', va='center', fontsize=14)
            else:
                axs_regions[j].text(13,-1.5, '$r=%.2f R_{200}$' % (args.radius), ha='right', va='center', fontsize=14)
            axs_regions[j].text(4,3.6, plot_labels[j] + ' Force', fontsize=14, ha='left', va='center')

            axs_regions[j].plot([np.min(timelist), np.max(timelist)], [0,0], 'k-', lw=1)
            axs_regions[j].plot([np.min(timelist), np.max(timelist)], [1,1], 'k--', lw=1)
            axs_regions[j].legend(loc=1, frameon=False, fontsize=14)
            ax2 = axs_regions[j].twiny()
            axs_regions[j].tick_params(axis='both', which='both', direction='in', length=8, width=2, pad=5, labelsize=14, \
              top=False, right=True)
            ax2.tick_params(axis='x', which='both', direction='in', length=8, width=2, pad=5, labelsize=14, \
              top=True)
            x0, x1 = axs_regions[j].get_xlim()
            z_ticks = [2,1.5,1,.75,.5,.3,.2,.1,0]
            last_z = np.where(z_ticks >= zlist[0])[0][-1]
            first_z = np.where(z_ticks <= zlist[-1])[0][0]
            z_ticks = z_ticks[first_z:last_z+1]
            tick_pos = [z for z in time_func(z_ticks)]
            tick_labels = ['%.2f' % (z) for z in z_ticks]
            ax2.set_xlim(x0,x1)
            ax2.set_xticks(tick_pos)
            ax2.set_xticklabels(tick_labels)
            ax2.set_xlabel('Redshift', fontsize=14)
            axs_regions[j].set_xlabel('Time [Gyr]', fontsize=14)

            z_sfr, sfr = np.loadtxt(code_path + 'halo_infos/00' + args.halo + '/' + args.run + '/sfr', unpack=True, usecols=[1,2], skiprows=1)
            t_sfr = time_func(z_sfr)

            ax3 = axs_regions[j].twinx()
            ax3.plot(t_sfr, sfr, 'k-', lw=1)
            axs_regions[j].plot([timelist[0],timelist[-1]], [0,0], 'k-', lw=1, label='SFR (right axis)')
            ax3.tick_params(axis='y', which='both', direction='in', length=8, width=2, pad=5, labelsize=14, right=True)
            ax3.set_ylim(-5,200)
            ax3.set_ylabel('SFR [$M_\odot$/yr]', fontsize=14)

            fig_regions[j].subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.92)
            fig_regions[j].savefig(save_dir + save_labels[j] + '_support_vs_time_regions-' + args.region_filter + save_suffix + '.png')
            plt.close(fig_regions[j])

def pressure_vs_r_rv_shaded(snap):
    '''Plots a datashader plot of pressure vs radius or radial velocity, color-coded by the field specified
    by the --shader_color parameter. The --pressure_type parameter determines which type of pressure
    is plotted.'''

    masses_ind = np.where(masses['snapshot']==snap)[0]
    Menc_profile = IUS(np.concatenate(([0],masses['radius'][masses_ind])), np.concatenate(([0],masses['total_mass'][masses_ind])))
    Mvir = rvir_masses['total_mass'][rvir_masses['snapshot']==snap]
    Rvir = rvir_masses['radius'][rvir_masses['snapshot']==snap][0]

    # Copy output to temp directory if on pleiades
    if (args.system=='pleiades_cassi'):
        print('Copying directory to /tmp')
        snap_dir = '/tmp/' + snap
        shutil.copytree(foggie_dir + run_dir + snap, snap_dir)
        snap_name = snap_dir + '/' + snap
    else:
        snap_name = foggie_dir + run_dir + snap + '/' + snap
    ds, refine_box = foggie_load(snap_name, trackname, do_filter_particles=False, halo_c_v_name=halo_c_v_name, gravity=True, masses_dir=masses_dir)

    if (args.pressure_type=='all'):
        ptypes = ['thermal', 'turbulent', 'outflow', 'inflow']
    elif (',' in args.pressure_type):
        ptypes = args.pressure_type.split(',')
    else:
        ptypes = [args.pressure_type]

    if (args.region_filter=='temperature'):
        regions = ['_low-T', '_mid-T', '_high-T']
    elif (args.region_filter=='metallicity'):
        regions = ['_low-Z', '_mid-Z', '_high-Z']
    elif (args.region_filter=='velocity'):
        regions = ['_low-v', '_mid-v', '_high-v']
    else:
        regions = ['']

    colorparam = args.shader_color
    data_frame = pd.DataFrame({})
    pix_res = float(np.min(refine_box['dx'].in_units('kpc')))  # at level 11
    lvl1_res = pix_res*2.**11.
    level = 9
    dx = lvl1_res/(2.**level)
    refine_res = int(3.*Rvir/dx)
    box = ds.covering_grid(level=level, left_edge=ds.halo_center_kpc-ds.arr([1.5*Rvir,1.5*Rvir,1.5*Rvir],'kpc'), dims=[refine_res, refine_res, refine_res])
    density = box['density'].in_units('g/cm**3').v
    temperature = box['temperature'].v
    if (args.plot=='pressure_vs_r_shaded'):
        radius = box['radius_corrected'].in_units('kpc').v
        data_frame['radius'] = radius.flatten()
    if (args.plot=='pressure_vs_rv_shaded') or (args.region_filter=='velocity'):
        rv = box['radial_velocity_corrected'].in_units('km/s').v
        data_frame['rv'] = rv.flatten()
        if (args.region_filter=='velocity'):
            vff = box['vff'].in_units('km/s').v
            vesc = box['vesc'].in_units('km/s').v
    if (args.shader_color=='temperature'):
        data_frame['temperature'] = np.log10(temperature).flatten()
    if (args.shader_color=='metallicity') or (args.region_filter=='metallicity'):
        metallicity = box['metallicity'].in_units('Zsun').v
        data_frame['metallicity'] = metallicity.flatten()
    for i in range(len(ptypes)):
        if (ptypes[i]=='thermal'):
            thermal_pressure = box['pressure'].in_units('erg/cm**3').v
            pressure = thermal_pressure
            pressure_label = 'Thermal'
        if (ptypes[i]=='turbulent'):
            vx = box['vx_corrected'].in_units('cm/s').v
            vy = box['vy_corrected'].in_units('cm/s').v
            vz = box['vz_corrected'].in_units('cm/s').v
            smooth_vx = uniform_filter(vx, size=20)
            smooth_vy = uniform_filter(vy, size=20)
            smooth_vz = uniform_filter(vz, size=20)
            sig_x = (vx - smooth_vx)**2.
            sig_y = (vy - smooth_vy)**2.
            sig_z = (vz - smooth_vz)**2.
            vdisp = np.sqrt((sig_x + sig_y + sig_z)/3.)
            turb_pressure = density*vdisp**2.
            pressure = turb_pressure
            pressure_label = 'Turbulent'
        if (ptypes[i]=='inflow') or (ptypes[i]=='outflow'):
            vr = box['radial_velocity_corrected'].in_units('cm/s').v
            smooth_vr = uniform_filter(vr, size=20)
            if (ptypes[i]=='inflow'):
                vr_in = 1.0*smooth_vr
                vr_in[smooth_vr > 0.] = 0.
                in_pressure = density*vr_in**2.
                pressure = in_pressure
                pressure_label = 'Inflow Ram'
            if (ptypes[i]=='outflow'):
                vr_out = 1.0*smooth_vr
                vr_out[smooth_vr < 0.] = 0.
                out_pressure = density*vr_out**2.
                pressure = out_pressure
                pressure_label = 'Outflow Ram'
        if (args.shader_color=='temperature'):
            data_frame['temp_cat'] = categorize_by_temp(data_frame['temperature'])
            data_frame.temp_cat = data_frame.temp_cat.astype('category')
            color_key = new_phase_color_key
            cat = 'temp_cat'
        if (args.shader_color=='metallicity'):
            data_frame['met_cat'] = categorize_by_metals(data_frame['metallicity'])
            data_frame.met_cat = data_frame.met_cat.astype('category')
            color_key = new_metals_color_key
            cat = 'met_cat'

        pressure_filtered = 1.0*pressure
        pressure_filtered[(density > cgm_density_max) & (temperature < cgm_temperature_min)] = 1e-10
        for j in range(len(regions)):
            if (regions[j]=='_low-T'):
                pressure_masked = 1.0*pressure_filtered
                pressure_masked[(temperature > 10**5)] = 1e-10
            if (regions[j]=='_mid-T'):
                pressure_masked = 1.0*pressure_filtered
                pressure_masked[(temperature > 10**6) | (temperature < 10**5)] = 1e-10
            if (regions[j]=='_high-T'):
                pressure_masked = 1.0*pressure_filtered
                pressure_masked[(temperature < 10**6)] = 1e-10
            if (regions[j]=='_low-Z'):
                pressure_masked = 1.0*pressure_filtered
                pressure_masked[(metallicity > 0.01)] = 1e-10
            if (regions[j]=='_mid-Z'):
                pressure_masked = 1.0*pressure_filtered
                pressure_masked[(metallicity < 0.01) | (metallicity > 1)] = 1e-10
            if (regions[j]=='_high-Z'):
                pressure_masked = 1.0*pressure_filtered
                pressure_masked[(metallicity < 1)] = 1e-10
            if (regions[j]=='_low-v'):
                pressure_masked = 1.0*pressure_filtered
                pressure_masked[(rv > 0.5*vff)] = 1e-10
            if (regions[j]=='_mid-v'):
                pressure_masked = 1.0*pressure_filtered
                pressure_masked[(rv < 0.5*vff) | (rv > vesc)] = 1e-10
            if (regions[j]=='_high-v'):
                pressure_masked = 1.0*pressure_filtered
                pressure_masked[(rv < vesc)] = 1e-10
            if (regions[j]==''): pressure_masked = pressure_filtered
            pressure_masked[pressure_masked <= 0.] = 1e-10
            data_frame['pressure'] = np.log10(pressure_masked).flatten()
            if (args.plot=='pressure_vs_r_shaded'): x_range = [0., 250.]
            if (args.plot=='pressure_vs_rv_shaded'): x_range = [-500,1000]
            y_range = [-18, -12]
            cvs = dshader.Canvas(plot_width=1000, plot_height=800, x_range=x_range, y_range=y_range)
            if (args.plot=='pressure_vs_r_shaded'):
                agg = cvs.points(data_frame, 'radius', 'pressure', dshader.count_cat(cat))
                file_xaxis = 'r'
            if (args.plot=='pressure_vs_rv_shaded'):
                agg = cvs.points(data_frame, 'rv', 'pressure', dshader.count_cat(cat))
                file_xaxis = 'rv'
            img = tf.spread(tf.shade(agg, color_key=color_key, how='eq_hist',min_alpha=40), shape='square', px=0)
            export_image(img, save_dir + snap + '_' + ptypes[i] + '_pressure_vs_' + file_xaxis + '_' + args.shader_color + '-colored' + regions[j] + save_suffix + '_intermediate')
            fig = plt.figure(figsize=(10,8),dpi=500)
            ax = fig.add_subplot(1,1,1)
            image = plt.imread(save_dir + snap + '_' + ptypes[i] + '_pressure_vs_' + file_xaxis + '_' + args.shader_color + '-colored' + regions[j] + save_suffix + '_intermediate.png')
            ax.imshow(image, extent=[x_range[0],x_range[1],y_range[0],y_range[1]])
            ax.set_aspect(8*abs(x_range[1]-x_range[0])/(10*abs(y_range[1]-y_range[0])))
            if (args.plot=='pressure_vs_r_shaded'): ax.set_xlabel('Radius [kpc]', fontsize=20)
            if (args.plot=='pressure_vs_rv_shaded'): ax.set_xlabel('Radial velocity [km/s]', fontsize=20)
            ax.set_ylabel('log ' + pressure_label + ' Pressure [erg/cm$^3$]', fontsize=20)
            ax.tick_params(axis='both', which='both', direction='in', length=8, width=2, pad=5, labelsize=20, \
              top=True, right=True)
            ax2 = fig.add_axes([0.7, 0.93, 0.25, 0.06])
            if (args.shader_color=='temperature'):
                cmap = create_foggie_cmap(temperature_min_datashader, temperature_max_datashader, categorize_by_temp, new_phase_color_key, log=True)
            if (args.shader_color=='metallicity'):
                cmap = create_foggie_cmap(metal_min, metal_max, categorize_by_metals, new_metals_color_key, log=False)
            ax2.imshow(np.flip(cmap.to_pil(), 1))
            if (args.shader_color=='temperature'):
                ax2.set_xticks([50,300,550])
                ax2.set_xticklabels(['4','5','6'],fontsize=16)
                ax2.text(400, 150, 'log T [K]',fontsize=20, ha='center', va='center')
            if (args.shader_color=='metallicity'):
                rng = (np.log10(metal_max)-np.log10(metal_min))/750.
                start = np.log10(metal_min)
                ax2.set_xticks([(np.log10(0.01)-start)/rng,(np.log10(0.1)-start)/rng,(np.log10(0.5)-start)/rng,(np.log10(1.)-start)/rng,(np.log10(2.)-start)/rng])
                ax2.set_xticklabels(['0.01','0.1','0.5','1','2'],fontsize=16)
                ax2.text(400, 150, '$Z$ [$Z_\odot$]',fontsize=20, ha='center', va='center')
            ax2.spines["top"].set_color('white')
            ax2.spines["bottom"].set_color('white')
            ax2.spines["left"].set_color('white')
            ax2.spines["right"].set_color('white')
            ax2.set_ylim(60, 180)
            ax2.set_xlim(-10, 750)
            ax2.set_yticklabels([])
            ax2.set_yticks([])
            plt.savefig(save_dir + snap + '_' + ptypes[i] + '_pressure_vs_' + file_xaxis + '_' + args.shader_color + '-colored' + regions[j] + save_suffix + '.png')
            os.system('rm ' + save_dir + snap + '_' + ptypes[i] + '_pressure_vs_' + file_xaxis + '_' + args.shader_color + '-colored' + regions[j] + save_suffix + '_intermediate.png')
            plt.close()

    # Delete output from temp directory if on pleiades
    if (args.system=='pleiades_cassi'):
        print('Deleting directory from /tmp')
        shutil.rmtree(snap_dir)

def force_vs_r_rv_shaded(snap):
    '''Plots a datashader plot of radial forces vs radius or radial velocity, color-coded by the field specified
    by the --shader_color parameter. The --force_type parameter determines which force
    is plotted.'''

    masses_ind = np.where(masses['snapshot']==snap)[0]
    Menc_profile = IUS(np.concatenate(([0],masses['radius'][masses_ind])), np.concatenate(([0],masses['total_mass'][masses_ind])))
    Mvir = rvir_masses['total_mass'][rvir_masses['snapshot']==snap]
    Rvir = rvir_masses['radius'][rvir_masses['snapshot']==snap][0]

    # Copy output to temp directory if on pleiades
    if (args.system=='pleiades_cassi'):
        print('Copying directory to /tmp')
        snap_dir = '/tmp/' + snap
        shutil.copytree(foggie_dir + run_dir + snap, snap_dir)
        snap_name = snap_dir + '/' + snap
    else:
        snap_name = foggie_dir + run_dir + snap + '/' + snap
    ds, refine_box = foggie_load(snap_name, trackname, do_filter_particles=False, halo_c_v_name=halo_c_v_name, gravity=True, masses_dir=masses_dir)

    if (args.force_type=='all'):
        ftypes = ['thermal', 'turbulent', 'outflow', 'inflow', 'rotation', 'gravity', 'total']
    elif (',' in args.force_type):
        ftypes = args.force_type.split(',')
    else:
        ftypes = [args.force_type]

    if (args.region_filter=='temperature'):
        regions = ['_low-T', '_mid-T', '_high-T']
    elif (args.region_filter=='metallicity'):
        regions = ['_low-Z', '_mid-Z', '_high-Z']
    elif (args.region_filter=='velocity'):
        regions = ['_low-v', '_mid-v', '_high-v']
    else:
        regions = ['']

    colorparam = args.shader_color
    data_frame = pd.DataFrame({})
    pix_res = float(np.min(refine_box['dx'].in_units('kpc')))  # at level 11
    lvl1_res = pix_res*2.**11.
    level = 9
    dx = lvl1_res/(2.**level)
    dx_cm = lvl1_res/(2.**level)*1000*cmtopc
    refine_res = int(3.*Rvir/dx)
    box = ds.covering_grid(level=level, left_edge=ds.halo_center_kpc-ds.arr([1.5*Rvir,1.5*Rvir,1.5*Rvir],'kpc'), dims=[refine_res, refine_res, refine_res])
    density = box['density'].in_units('g/cm**3').v
    temperature = box['temperature'].v
    x_hat = box['x'].in_units('cm').v - ds.halo_center_kpc[0].to('cm').v
    y_hat = box['y'].in_units('cm').v - ds.halo_center_kpc[1].to('cm').v
    z_hat = box['z'].in_units('cm').v - ds.halo_center_kpc[2].to('cm').v
    r = box['radius_corrected'].in_units('cm').v
    x_hat /= r
    y_hat /= r
    z_hat /= r
    if (args.plot=='force_vs_r_shaded'):
        radius = box['radius_corrected'].in_units('kpc').v
        data_frame['radius'] = radius.flatten()
    if (args.plot=='force_vs_rv_shaded') or (args.region_filter=='velocity'):
        rv = box['radial_velocity_corrected'].in_units('km/s').v
        data_frame['rv'] = rv.flatten()
        if (args.region_filter=='velocity'):
            vff = box['vff'].in_units('km/s').v
            vesc = box['vesc'].in_units('km/s').v
    if (args.shader_color=='temperature'):
        data_frame['temperature'] = np.log10(temperature).flatten()
    if (args.shader_color=='metallicity') or (args.region_filter=='metallicity'):
        metallicity = box['metallicity'].in_units('Zsun').v
        data_frame['metallicity'] = metallicity.flatten()
    for i in range(len(ftypes)):
        if (ftypes[i]=='thermal') or ('total' in ftypes):
            thermal_pressure = box['pressure'].in_units('erg/cm**3').v
            pres_grad = np.gradient(thermal_pressure, dx_cm)
            dPdr = pres_grad[0]*x_hat + pres_grad[1]*y_hat + pres_grad[2]*z_hat
            thermal_force = 1./density * dPdr
            force = thermal_force
            force_label = 'Thermal Pressure'
        if (ftypes[i]=='turbulent') or ('total' in ftypes):
            vx = box['vx_corrected'].in_units('cm/s').v
            vy = box['vy_corrected'].in_units('cm/s').v
            vz = box['vz_corrected'].in_units('cm/s').v
            smooth_vx = uniform_filter(vx, size=20)
            smooth_vy = uniform_filter(vy, size=20)
            smooth_vz = uniform_filter(vz, size=20)
            sig_x = (vx - smooth_vx)**2.
            sig_y = (vy - smooth_vy)**2.
            sig_z = (vz - smooth_vz)**2.
            vdisp = np.sqrt((sig_x + sig_y + sig_z)/3.)
            turb_pressure = density*vdisp**2.
            pres_grad = np.gradient(turb_pressure, dx_cm)
            dPdr = pres_grad[0]*x_hat + pres_grad[1]*y_hat + pres_grad[2]*z_hat
            turb_force = 1./density * dPdr
            force = turb_force
            force_label = 'Turbulent Pressure'
        if (ftypes[i]=='inflow') or (ftypes[i]=='outflow') or ('total' in ftypes):
            vr = box['radial_velocity_corrected'].in_units('cm/s').v
            smooth_vr = uniform_filter(vr, size=20)
            if (ftypes[i]=='inflow') or ('total' in ftypes):
                vr_in = 1.0*smooth_vr
                vr_in[smooth_vr > 0.] = 0.
                in_pressure = density*vr_in**2.
                pres_grad = np.gradient(in_pressure, dx_cm)
                dPdr = pres_grad[0]*x_hat + pres_grad[1]*y_hat + pres_grad[2]*z_hat
                in_force = 1./density * dPdr
                force = in_force
                force_label = 'Inflow Ram Pressure'
            if (ftypes[i]=='outflow') or ('total' in ftypes):
                vr_out = 1.0*smooth_vr
                vr_out[smooth_vr < 0.] = 0.
                out_pressure = density*vr_out**2.
                pres_grad = np.gradient(out_pressure, dx_cm)
                dPdr = pres_grad[0]*x_hat + pres_grad[1]*y_hat + pres_grad[2]*z_hat
                out_force = 1./density * dPdr
                force = out_force
                force_label = 'Outflow Ram Pressure'
        if (ftypes[i]=='rotation') or ('total' in ftypes):
            vtheta = box['theta_velocity_corrected'].in_units('cm/s').v
            vphi = box['phi_velocity_corrected'].in_units('cm/s').v
            smooth_vtheta = uniform_filter(vtheta, size=20)
            smooth_vphi = uniform_filter(vphi, size=20)
            rot_force = (smooth_vtheta**2. + smooth_vphi**2.)/r
            force = rot_force
            force_label = 'Rotation'
        if (ftypes[i]=='gravity') or ('total' in ftypes):
            grav_force = -G*Menc_profile(r/(1000*cmtopc))*gtoMsun/r**2.
            force = grav_force
            force_label = 'Gravity'
        if (ftypes[i]=='total'):
            tot_force = thermal_force + turb_force + rot_force + out_force + in_force + grav_force
            force = tot_force
            force_label = 'Total'
        if (args.shader_color=='temperature'):
            data_frame['temp_cat'] = categorize_by_temp(data_frame['temperature'])
            data_frame.temp_cat = data_frame.temp_cat.astype('category')
            color_key = new_phase_color_key
            cat = 'temp_cat'
        if (args.shader_color=='metallicity'):
            data_frame['met_cat'] = categorize_by_metals(data_frame['metallicity'])
            data_frame.met_cat = data_frame.met_cat.astype('category')
            color_key = new_metals_color_key
            cat = 'met_cat'

        force_filtered = 1.0*force
        force_filtered[(density > cgm_density_max) & (temperature < cgm_temperature_min)] = 1
        for j in range(len(regions)):
            if (regions[j]=='_low-T'):
                force_masked = 1.0*force_filtered
                force_masked[(temperature > 10**5)] = 1
            if (regions[j]=='_mid-T'):
                force_masked = 1.0*force_filtered
                force_masked[(temperature > 10**6) | (temperature < 10**5)] = 1
            if (regions[j]=='_high-T'):
                force_masked = 1.0*force_filtered
                force_masked[(temperature < 10**6)] = 1
            if (regions[j]=='_low-Z'):
                force_masked = 1.0*force_filtered
                force_masked[(metallicity > 0.01)] = 1
            if (regions[j]=='_mid-Z'):
                force_masked = 1.0*force_filtered
                force_masked[(metallicity < 0.01) | (metallicity > 1)] = 1
            if (regions[j]=='_high-Z'):
                force_masked = 1.0*force_filtered
                force_masked[(metallicity < 1)] = 1
            if (regions[j]=='_low-v'):
                force_masked = 1.0*force_filtered
                force_masked[(rv > 0.5*vff)] = 1
            if (regions[j]=='_mid-v'):
                force_masked = 1.0*force_filtered
                force_masked[(rv < 0.5*vff) | (rv > vesc)] = 1
            if (regions[j]=='_high-v'):
                force_masked = 1.0*force_filtered
                force_masked[(rv < vesc)] = 1
            if (regions[j]==''): force_masked = force_filtered
            force_pos = 1.0*force_masked
            force_pos[force_masked <= 1e-10] = 1
            force_neg = 1.0*force_masked
            force_neg[force_masked >= -1e-10] = -1
            force_neg = -1.*force_neg
            force_mid = 1.0*force_masked
            force_mid[(force_masked > 1e-10) | (force_masked < -1e-10)] = 1
            data_frame['force_pos'] = np.log10(force_pos).flatten()
            data_frame['force_neg'] = np.log10(force_neg).flatten()
            data_frame['force_mid'] = force_mid.flatten()
            if (args.plot=='force_vs_r_shaded'): x_range = [0., 250.]
            if (args.plot=='force_vs_rv_shaded'): x_range = [-500,1000]
            y_range = [-10, -6]
            y_range_mid = [-1e-10, 1e-10]
            print('Making %s force plot' % (ftypes[i]))
            cvs_pos = dshader.Canvas(plot_width=1000, plot_height=400, x_range=x_range, y_range=y_range)
            cvs_neg = dshader.Canvas(plot_width=1000, plot_height=400, x_range=x_range, y_range=y_range)
            cvs_mid = dshader.Canvas(plot_width=1000, plot_height=100, x_range=x_range, y_range=y_range_mid)
            if (args.plot=='force_vs_r_shaded'):
                agg_pos = cvs_pos.points(data_frame, 'radius', 'force_pos', dshader.count_cat(cat))
                agg_neg = cvs_neg.points(data_frame, 'radius', 'force_neg', dshader.count_cat(cat))
                agg_mid = cvs_mid.points(data_frame, 'radius', 'force_mid', dshader.count_cat(cat))
                file_xaxis = 'r'
            if (args.plot=='force_vs_rv_shaded'):
                agg_pos = cvs_pos.points(data_frame, 'rv', 'force_pos', dshader.count_cat(cat))
                agg_neg = cvs_neg.points(data_frame, 'rv', 'force_neg', dshader.count_cat(cat))
                agg_mid = cvs_mid.points(data_frame, 'rv', 'force_mid', dshader.count_cat(cat))
                file_xaxis = 'rv'
            img_pos = tf.spread(tf.shade(agg_pos, color_key=color_key, how='eq_hist',min_alpha=40), shape='square', px=0)
            img_neg = tf.spread(tf.shade(agg_neg, color_key=color_key, how='eq_hist',min_alpha=40), shape='square', px=0)
            img_mid = tf.spread(tf.shade(agg_mid, color_key=color_key, how='eq_hist',min_alpha=40), shape='square', px=0)
            export_image(img_pos, save_dir + snap + '_' + ftypes[i] + '_force_pos_vs_' + file_xaxis + '_' + args.shader_color + '-colored' + regions[j] + save_suffix)
            export_image(img_neg, save_dir + snap + '_' + ftypes[i] + '_force_neg_vs_' + file_xaxis + '_' + args.shader_color + '-colored' + regions[j] + save_suffix)
            export_image(img_mid, save_dir + snap + '_' + ftypes[i] + '_force_mid_vs_' + file_xaxis + '_' + args.shader_color + '-colored' + regions[j] + save_suffix)
            fig = plt.figure(figsize=(10,8),dpi=500)
            ax = fig.add_subplot(1,1,1)
            image_pos = plt.imread(save_dir + snap + '_' + ftypes[i] + '_force_pos_vs_' + file_xaxis + '_' + args.shader_color + '-colored' + regions[j] + save_suffix + '.png')
            image_neg = plt.imread(save_dir + snap + '_' + ftypes[i] + '_force_neg_vs_' + file_xaxis + '_' + args.shader_color + '-colored' + regions[j] + save_suffix + '.png')
            image_mid = plt.imread(save_dir + snap + '_' + ftypes[i] + '_force_mid_vs_' + file_xaxis + '_' + args.shader_color + '-colored' + regions[j] + save_suffix + '.png')
            image = np.vstack((image_pos,image_mid,np.flipud(image_neg)))
            ax.imshow(image, extent=[x_range[0],x_range[1],-4.5,4.5])
            ax.set_aspect(9*abs(x_range[1]-x_range[0])/(10*abs(4.5-(-4.5))))
            ax.set_yticks([-4.5,-3.5,-2.5,-1.5,-0.5,0,0.5,1.5,2.5,3.5,4.5])
            ax.set_yticklabels(['$-10^{-6}$','$-10^{-7}$','$-10^{-8}$','$-10^{-9}$','$-10^{-10}$','$0$','$10^{-10}$','$10^{-9}$','$10^{-8}$','$10^{-7}$','$10^{-6}$'],fontsize=20)
            if (args.plot=='force_vs_r_shaded'): ax.set_xlabel('Radius [kpc]', fontsize=20)
            if (args.plot=='force_vs_rv_shaded'): ax.set_xlabel('Radial velocity [km/s]', fontsize=20)
            ax.set_ylabel(force_label + ' Force [cm/s$^2$]', fontsize=20)
            ax.tick_params(axis='both', which='both', direction='in', length=8, width=2, pad=5, labelsize=20, \
              top=True, right=True)
            ax2 = fig.add_axes([0.7, 0.93, 0.25, 0.06])
            if (args.shader_color=='temperature'):
                cmap = create_foggie_cmap(temperature_min_datashader, temperature_max_datashader, categorize_by_temp, new_phase_color_key, log=True)
            if (args.shader_color=='metallicity'):
                cmap = create_foggie_cmap(metal_min, metal_max, categorize_by_metals, new_metals_color_key, log=False)
            ax2.imshow(np.flip(cmap.to_pil(), 1))
            if (args.shader_color=='temperature'):
                ax2.set_xticks([50,300,550])
                ax2.set_xticklabels(['4','5','6'],fontsize=16)
                ax2.text(400, 150, 'log T [K]',fontsize=20, ha='center', va='center')
            if (args.shader_color=='metallicity'):
                rng = (np.log10(metal_max)-np.log10(metal_min))/750.
                start = np.log10(metal_min)
                ax2.set_xticks([(np.log10(0.01)-start)/rng,(np.log10(0.1)-start)/rng,(np.log10(0.5)-start)/rng,(np.log10(1.)-start)/rng,(np.log10(2.)-start)/rng])
                ax2.set_xticklabels(['0.01','0.1','0.5','1','2'],fontsize=16)
                ax2.text(400, 150, '$Z$ [$Z_\odot$]',fontsize=20, ha='center', va='center')
            ax2.spines["top"].set_color('white')
            ax2.spines["bottom"].set_color('white')
            ax2.spines["left"].set_color('white')
            ax2.spines["right"].set_color('white')
            ax2.set_ylim(60, 180)
            ax2.set_xlim(-10, 750)
            ax2.set_yticklabels([])
            ax2.set_yticks([])
            plt.savefig(save_dir + snap + '_' + ftypes[i] + '_force_vs_' + file_xaxis + '_' + args.shader_color + '-colored' + regions[j] + save_suffix + '.png')
            os.system('rm ' + save_dir + snap + '_' + ftypes[i] + '_force_pos_vs_' + file_xaxis + '_' + args.shader_color + '-colored' + regions[j] + save_suffix + '.png')
            os.system('rm ' + save_dir + snap + '_' + ftypes[i] + '_force_neg_vs_' + file_xaxis + '_' + args.shader_color + '-colored' + regions[j] + save_suffix + '.png')
            os.system('rm ' + save_dir + snap + '_' + ftypes[i] + '_force_mid_vs_' + file_xaxis + '_' + args.shader_color + '-colored' + regions[j] + save_suffix + '.png')
            plt.close()

    # Delete output from temp directory if on pleiades
    if (args.system=='pleiades_cassi'):
        print('Deleting directory from /tmp')
        shutil.rmtree(snap_dir)

def support_vs_r_rv_shaded(snap):
    '''Plots a datashader plot of pressure support vs radius, color-coded by the field specified
    by the --shader_color parameter. The --pressure_type parameter determines which type of pressure
    support is plotted.'''

    masses_ind = np.where(masses['snapshot']==snap)[0]
    Menc_profile = IUS(np.concatenate(([0],masses['radius'][masses_ind])), np.concatenate(([0],masses['total_mass'][masses_ind])))
    Mvir = rvir_masses['total_mass'][rvir_masses['snapshot']==snap]
    Rvir = rvir_masses['radius'][rvir_masses['snapshot']==snap][0]

    if (args.system=='pleiades_cassi'):
        print('Copying directory to /tmp')
        snap_dir = '/tmp/' + snap
        shutil.copytree(foggie_dir + run_dir + snap, snap_dir)
        snap_name = snap_dir + '/' + snap
    else:
        snap_name = foggie_dir + run_dir + snap + '/' + snap
    ds, refine_box = foggie_load(snap_name, trackname, do_filter_particles=False, halo_c_v_name=halo_c_v_name, gravity=True, masses_dir=masses_dir)

    if (args.force_type=='all'):
        ftypes = ['thermal', 'turbulent', 'outflow', 'inflow', 'rotation']
    elif (',' in args.force_type):
        ftypes = args.force_type.split(',')
    else:
        ftypes = [args.force_type]

    if (args.region_filter=='temperature'):
        regions = ['_low-T', '_mid-T', '_high-T']
    elif (args.region_filter=='metallicity'):
        regions = ['_low-Z', '_mid-Z', '_high-Z']
    elif (args.region_filter=='velocity'):
        regions = ['_low-v', '_mid-v', '_high-v']
    else:
        regions = ['']

    colorparam = args.shader_color
    data_frame = pd.DataFrame({})
    pix_res = float(np.min(refine_box['dx'].in_units('kpc')))  # at level 11
    lvl1_res = pix_res*2.**11.
    level = 9
    dx = lvl1_res/(2.**level)
    dx_cm = lvl1_res/(2.**level)*1000*cmtopc
    refine_res = int(3.*Rvir/dx)
    box = ds.covering_grid(level=level, left_edge=ds.halo_center_kpc-ds.arr([1.5*Rvir,1.5*Rvir,1.5*Rvir],'kpc'), dims=[refine_res, refine_res, refine_res])
    density = box['density'].in_units('g/cm**3').v
    temperature = box['temperature'].v
    x_hat = box['x'].in_units('cm').v - ds.halo_center_kpc[0].to('cm').v
    y_hat = box['y'].in_units('cm').v - ds.halo_center_kpc[1].to('cm').v
    z_hat = box['z'].in_units('cm').v - ds.halo_center_kpc[2].to('cm').v
    r = box['radius_corrected'].in_units('cm').v
    x_hat /= r
    y_hat /= r
    z_hat /= r

    grav_force = -G*Menc_profile(r/(1000*cmtopc))*gtoMsun/r**2.
    thermal_pressure = box['pressure'].in_units('erg/cm**3').v
    pres_grad = np.gradient(thermal_pressure, dx_cm)
    dPdr = pres_grad[0]*x_hat + pres_grad[1]*y_hat + pres_grad[2]*z_hat
    thermal_force = 1./density * dPdr
    vx = box['vx_corrected'].in_units('cm/s').v
    vy = box['vy_corrected'].in_units('cm/s').v
    vz = box['vz_corrected'].in_units('cm/s').v
    smooth_vx = uniform_filter(vx, size=20)
    smooth_vy = uniform_filter(vy, size=20)
    smooth_vz = uniform_filter(vz, size=20)
    sig_x = (vx - smooth_vx)**2.
    sig_y = (vy - smooth_vy)**2.
    sig_z = (vz - smooth_vz)**2.
    vdisp = np.sqrt((sig_x + sig_y + sig_z)/3.)
    turb_pressure = density*vdisp**2.
    pres_grad = np.gradient(turb_pressure, dx_cm)
    dPdr = pres_grad[0]*x_hat + pres_grad[1]*y_hat + pres_grad[2]*z_hat
    turb_force = 1./density * dPdr
    vr = box['radial_velocity_corrected'].in_units('cm/s').v
    smooth_vr = uniform_filter(vr, size=20)
    vr_in = 1.0*smooth_vr
    vr_in[smooth_vr > 0.] = 0.
    in_pressure = density*vr_in**2.
    pres_grad = np.gradient(in_pressure, dx_cm)
    dPdr = pres_grad[0]*x_hat + pres_grad[1]*y_hat + pres_grad[2]*z_hat
    in_force = 1./density * dPdr
    vr_out = 1.0*smooth_vr
    vr_out[smooth_vr < 0.] = 0.
    out_pressure = density*vr_out**2.
    pres_grad = np.gradient(out_pressure, dx_cm)
    dPdr = pres_grad[0]*x_hat + pres_grad[1]*y_hat + pres_grad[2]*z_hat
    out_force = 1./density * dPdr
    vtheta = box['theta_velocity_corrected'].in_units('cm/s').v
    vphi = box['phi_velocity_corrected'].in_units('cm/s').v
    smooth_vtheta = uniform_filter(vtheta, size=20)
    smooth_vphi = uniform_filter(vphi, size=20)
    rot_force = (smooth_vtheta**2. + smooth_vphi**2.)/r
    tot_force = grav_force + thermal_force + turb_force + rot_force + in_force + out_force

    if (args.shader_color=='temperature'):
        data_frame['temperature'] = np.log10(temperature).flatten()
    if (args.shader_color=='metallicity') or (args.region_filter=='metallicity'):
        metallicity = box['metallicity'].in_units('Zsun').v
        data_frame['metallicity'] = metallicity.flatten()
    if (args.plot=='support_vs_r_shaded'):
        radius = box['radius_corrected'].in_units('kpc').v
        data_frame['radius'] = radius.flatten()
    if (args.plot=='support_vs_rv_shaded') or (args.region_filter=='velocity'):
        rv = box['radial_velocity_corrected'].in_units('km/s').v
        data_frame['rv'] = rv.flatten()
    for i in range(len(ftypes)):
        if (ftypes[i]=='thermal'):
            support = thermal_force/-grav_force
            support_label = 'Thermal'
        if (ftypes[i]=='turbulent'):
            support = turb_force/-grav_force
            support_label = 'Turbulent'
        if (ftypes[i]=='inflow'):
            support = in_force/-grav_force
            support_label = 'Inflow'
        if (ftypes[i]=='outflow'):
            support = out_force/-grav_force
            support_label = 'Outflow'
        if (ftypes[i]=='rotation'):
            support = rot_force/-grav_force
            support_label = 'Rotation'

        support_filtered = support.flatten()
        #plt.hist(support_filtered, range=[-1e-9,1e-9])
        #plt.show()
        #print(np.max(support_filtered), np.min(support_filtered), np.mean(support_filtered), np.std(support_filtered))
        support_filtered[(density.flatten() > cgm_density_max) & (temperature.flatten() < cgm_temperature_min)] = -10
        #plt.hist(support_filtered, range=[-1e-9,1e-9])
        #plt.show()
        #print(np.max(support_filtered), np.min(support_filtered), np.mean(support_filtered), np.std(support_filtered))
        support_filtered[(tot_force.flatten() > 1e-9) | (tot_force.flatten() < -1e-9)] = -10
        #plt.hist(support_filtered, range=[-1e-9,1e-9])
        #plt.show()
        #print(np.max(tot_force), np.min(tot_force), np.mean(tot_force), np.std(tot_force))
        #print(np.max(support_filtered), np.min(support_filtered), np.mean(support_filtered), np.std(support_filtered))
        for j in range(len(regions)):
            if (regions[j]=='_low-T'):
                support_masked = 1.0*support_filtered
                support_masked[(temperature.flatten() > 10**5)] = -10
            if (regions[j]=='_mid-T'):
                support_masked = 1.0*support_filtered
                support_masked[(temperature.flatten() > 10**6) | (temperature.flatten() < 10**5)] = -10
            if (regions[j]=='_high-T'):
                support_masked = 1.0*support_filtered
                support_masked[(temperature.flatten() < 10**6)] = -10
            if (regions[j]=='_low-Z'):
                support_masked = 1.0*support_filtered
                support_masked[(metallicity.flatten() > 0.01)] = -10
            if (regions[j]=='_mid-Z'):
                support_masked = 1.0*support_filtered
                support_masked[(metallicity.flatten() < 0.01) | (metallicity.flatten() > 1)] = -10
            if (regions[j]=='_high-Z'):
                support_masked = 1.0*support_filtered
                support_masked[(metallicity.flatten() < 1)] = -10
            if (regions[j]=='_low-v'):
                support_masked = 1.0*support_filtered
                support_masked[(rv.flatten() > 0.5*vff.flatten())] = -10
            if (regions[j]=='_mid-v'):
                support_masked = 1.0*support_filtered
                support_masked[(rv.flatten() < 0.5*vff.flatten()) | (rv.flatten() > vesc.flatten())] = -10
            if (regions[j]=='_high-v'):
                support_masked = 1.0*support_filtered
                support_masked[(rv.flatten() < vesc.flatten())] = -10
            if (regions[j]==''): support_masked = 1.0*support_filtered
            data_frame['support'] = support_masked
            #print(np.max(data_frame['support']), np.min(data_frame['support']), np.mean(data_frame['support']), np.std(data_frame['support']))
            #plt.hist(support_masked, range=[-1e-9, 1e-9])
            #plt.show()
            if (args.shader_color=='temperature'):
                data_frame['temp_cat'] = categorize_by_temp(data_frame['temperature'])
                data_frame.temp_cat = data_frame.temp_cat.astype('category')
                color_key = new_phase_color_key
                cat = 'temp_cat'
            if (args.shader_color=='metallicity'):
                data_frame['met_cat'] = categorize_by_metals(data_frame['metallicity'])
                data_frame.met_cat = data_frame.met_cat.astype('category')
                color_key = new_metals_color_key
                cat = 'met_cat'
            if (args.plot=='support_vs_r_shaded'): x_range = [0., 250.]
            if (args.plot=='support_vs_rv_shaded'): x_range = [-500,1000]
            y_range = [-9, 11]
            cvs = dshader.Canvas(plot_width=1000, plot_height=800, x_range=x_range, y_range=y_range)
            if (args.plot=='support_vs_r_shaded'):
                agg = cvs.points(data_frame, 'radius', 'support', dshader.count_cat(cat))
                file_xaxis = 'r'
            if (args.plot=='support_vs_rv_shaded'):
                agg = cvs.points(data_frame, 'rv', 'support', dshader.count_cat(cat))
                file_xaxis = 'rv'
            img = tf.spread(tf.shade(agg, color_key=color_key, how='eq_hist',min_alpha=40), shape='square', px=0)
            export_image(img, save_dir + snap + '_' + ftypes[i] + '_support_vs_' + file_xaxis + '_' + args.shader_color + '-colored' + regions[j] + save_suffix + '_intermediate')
            fig = plt.figure(figsize=(10,8),dpi=500)
            ax = fig.add_subplot(1,1,1)
            image = plt.imread(save_dir + snap + '_' + ftypes[i] + '_support_vs_' + file_xaxis + '_' + args.shader_color + '-colored' + regions[j] + save_suffix + '_intermediate.png')
            ax.imshow(image, extent=[x_range[0],x_range[1],y_range[0],y_range[1]])
            ax.set_aspect(8*abs(x_range[1]-x_range[0])/(10*abs(y_range[1]-y_range[0])))
            if (args.plot=='support_vs_r_shaded'): ax.set_xlabel('Radius [kpc]', fontsize=20)
            if (args.plot=='support_vs_rv_shaded'): ax.set_xlabel('Radial velocity [km/s]', fontsize=20)
            ax.set_ylabel(support_label + ' Force / $F_\mathrm{grav}$', fontsize=20)
            ax.plot(x_range, [1,1], 'k-', lw=2)
            ax.tick_params(axis='both', which='both', direction='in', length=8, width=2, pad=5, labelsize=20, \
              top=True, right=True)
            ax2 = fig.add_axes([0.7, 0.93, 0.25, 0.06])
            if (args.shader_color=='temperature'):
                cmap = create_foggie_cmap(temperature_min_datashader, temperature_max_datashader, categorize_by_temp, new_phase_color_key, log=True)
            if (args.shader_color=='metallicity'):
                cmap = create_foggie_cmap(metal_min, metal_max, categorize_by_metals, new_metals_color_key, log=False)
            ax2.imshow(np.flip(cmap.to_pil(), 1))
            if (args.shader_color=='temperature'):
                ax2.set_xticks([50,300,550])
                ax2.set_xticklabels(['4','5','6'],fontsize=16)
                ax2.text(400, 150, 'log T [K]',fontsize=20, ha='center', va='center')
            if (args.shader_color=='metallicity'):
                rng = (np.log10(metal_max)-np.log10(metal_min))/750.
                start = np.log10(metal_min)
                ax2.set_xticks([(np.log10(0.01)-start)/rng,(np.log10(0.1)-start)/rng,(np.log10(0.5)-start)/rng,(np.log10(1.)-start)/rng,(np.log10(2.)-start)/rng])
                ax2.set_xticklabels(['0.01','0.1','0.5','1','2'],fontsize=16)
                ax2.text(400, 150, '$Z$ [$Z_\odot$]',fontsize=20, ha='center', va='center')
            ax2.spines["top"].set_color('white')
            ax2.spines["bottom"].set_color('white')
            ax2.spines["left"].set_color('white')
            ax2.spines["right"].set_color('white')
            ax2.set_ylim(60, 180)
            ax2.set_xlim(-10, 750)
            ax2.set_yticklabels([])
            ax2.set_yticks([])
            plt.savefig(save_dir + snap + '_' + ftypes[i] + '_support_vs_' + file_xaxis + '_' + args.shader_color + '-colored' + regions[j] + save_suffix + '.png')
            os.system('rm ' + save_dir + snap + '_' + ftypes[i] + '_support_vs_' + file_xaxis + '_' + args.shader_color + '-colored' + regions[j] + save_suffix + '_intermediate.png')
            plt.close()

    # Delete output from temp directory if on pleiades
    if (args.system=='pleiades_cassi'):
        print('Deleting directory from /tmp')
        shutil.rmtree(snap_dir)

def shader_force_colored(snap):
    '''Plots a datashader plot of properties as functions of radius or radial velocity color-coded
    by forces or ratios of forces.'''

    masses_ind = np.where(masses['snapshot']==snap)[0]
    Menc_profile = IUS(np.concatenate(([0],masses['radius'][masses_ind])), np.concatenate(([0],masses['total_mass'][masses_ind])))
    Mvir = rvir_masses['total_mass'][rvir_masses['snapshot']==snap]
    Rvir = rvir_masses['radius'][rvir_masses['snapshot']==snap][0]

    if (args.system=='pleiades_cassi'):
        print('Copying directory to /tmp')
        snap_dir = '/tmp/' + args.halo + '/' + args.run + '/' + target_dir + '/' + snap
        if (args.copy_to_tmp):
            shutil.copytree(foggie_dir + run_dir + snap, snap_dir)
            snap_name = snap_dir + '/' + snap
        else:
            # Make a dummy directory with the snap name so the script later knows the process running
            # this snapshot failed if the directory is still there
            os.makedirs(snap_dir)
            snap_name = foggie_dir + run_dir + snap + '/' + snap
    else:
        snap_name = foggie_dir + run_dir + snap + '/' + snap
    ds, refine_box = foggie_load(snap_name, trackname, do_filter_particles=False, halo_c_v_name=halo_c_v_name, gravity=True, masses_dir=masses_dir)

    # Define the density cut between disk and CGM to vary smoothly between 1 and 0.1 between z = 0.5 and z = 0.25,
    # with it being 1 at higher redshifts and 0.1 at lower redshifts
    current_time = ds.current_time.in_units('Myr').v
    if (current_time<=8656.88):
        density_cut_factor = 1.
    elif (current_time<=10787.12):
        density_cut_factor = 1. - 0.9*(current_time-8656.88)/2130.24
    else:
        density_cut_factor = 0.1

    pix_res = float(np.min(refine_box[('gas','dx')].in_units('kpc')))  # at level 11
    lvl1_res = pix_res*2.**11.
    level = 9
    dx = lvl1_res/(2.**level)
    smooth_scale = (25./dx)/6.
    dx_cm = dx*1000*cmtopc
    refine_res = int(3.*Rvir/dx)
    box = ds.covering_grid(level=level, left_edge=ds.halo_center_kpc-ds.arr([1.5*Rvir,1.5*Rvir,1.5*Rvir],'kpc'), dims=[refine_res, refine_res, refine_res])
    density = box['density'].in_units('g/cm**3').v
    temperature = box['temperature'].v
    mass = box['cell_mass'].in_units('g').v
    x = box[('gas','x')].in_units('cm').v - ds.halo_center_kpc[0].to('cm').v
    y = box[('gas','y')].in_units('cm').v - ds.halo_center_kpc[1].to('cm').v
    z = box[('gas','z')].in_units('cm').v - ds.halo_center_kpc[2].to('cm').v
    r = box['radius_corrected'].in_units('cm').v
    x_hat = x/r
    y_hat = y/r
    z_hat = z/r

    # This next block needed for removing any ISM regions and then interpolating over the holes left behind
    # Define ISM regions to remove
    disk_mask = (density > cgm_density_max * density_cut_factor)
    # disk_mask_expanded is a binary mask of both ISM regions AND their surrounding pixels
    struct = ndimage.generate_binary_structure(3,3)
    disk_mask_expanded = ndimage.binary_dilation(disk_mask, structure=struct, iterations=3)
    disk_mask_expanded = ndimage.binary_closing(disk_mask_expanded, structure=struct, iterations=3)
    disk_mask_expanded = disk_mask_expanded | disk_mask
    # disk_edges is a binary mask of ONLY pixels surrounding ISM regions -- nothing inside ISM regions
    disk_edges = disk_mask_expanded & ~disk_mask
    x_edges = x[disk_edges].flatten()
    y_edges = y[disk_edges].flatten()
    z_edges = z[disk_edges].flatten()

    den_edges = density[disk_edges]
    den_interp_func = NearestNDInterpolator(list(zip(x_edges,y_edges,z_edges)), den_edges)
    den_masked = np.copy(density)
    den_masked[disk_mask] = den_interp_func(x[disk_mask], y[disk_mask], z[disk_mask])
    smooth_den = gaussian_filter(den_masked, smooth_scale)

    thermal_pressure = box['pressure'].in_units('erg/cm**3').v
    if (args.cgm_only):
        pres_edges = thermal_pressure[disk_edges]
        pres_interp_func = NearestNDInterpolator(list(zip(x_edges,y_edges,z_edges)), pres_edges)
        pres_masked = np.copy(thermal_pressure)
        pres_masked[disk_mask] = pres_interp_func(x[disk_mask], y[disk_mask], z[disk_mask])
    else:
        pres_masked = thermal_pressure
    pres_grad = np.gradient(pres_masked, dx_cm)
    dPdr = pres_grad[0]*x_hat + pres_grad[1]*y_hat + pres_grad[2]*z_hat
    thermal_force = -1./den_masked * dPdr
    vx = box['vx_corrected'].in_units('cm/s').v
    vy = box['vy_corrected'].in_units('cm/s').v
    vz = box['vz_corrected'].in_units('cm/s').v
    vx_edges = vx[disk_edges]
    vy_edges = vy[disk_edges]
    vz_edges = vz[disk_edges]
    vx_interp_func = NearestNDInterpolator(list(zip(x_edges,y_edges,z_edges)), vx_edges)
    vy_interp_func = NearestNDInterpolator(list(zip(x_edges,y_edges,z_edges)), vy_edges)
    vz_interp_func = NearestNDInterpolator(list(zip(x_edges,y_edges,z_edges)), vz_edges)
    vx_masked = np.copy(vx)
    vy_masked = np.copy(vy)
    vz_masked = np.copy(vz)
    vx_masked[disk_mask] = vx_interp_func(x[disk_mask], y[disk_mask], z[disk_mask])
    vy_masked[disk_mask] = vy_interp_func(x[disk_mask], y[disk_mask], z[disk_mask])
    vz_masked[disk_mask] = vz_interp_func(x[disk_mask], y[disk_mask], z[disk_mask])
    smooth_vx = gaussian_filter(vx_masked, smooth_scale)
    smooth_vy = gaussian_filter(vy_masked, smooth_scale)
    smooth_vz = gaussian_filter(vz_masked, smooth_scale)
    smooth_den = gaussian_filter(den_masked, smooth_scale)
    sig_x = (vx_masked - smooth_vx)**2.
    sig_y = (vy_masked - smooth_vy)**2.
    sig_z = (vz_masked - smooth_vz)**2.
    vdisp = np.sqrt((sig_x + sig_y + sig_z)/3.)
    turb_pressure = smooth_den*vdisp**2.
    pres_grad = np.gradient(turb_pressure, dx_cm)
    dPdr = pres_grad[0]*x_hat + pres_grad[1]*y_hat + pres_grad[2]*z_hat
    turb_force = -1./den_masked * dPdr
    vr = box['radial_velocity_corrected'].in_units('cm/s').v
    vr_edges = vr[disk_edges]
    vr_interp_func = NearestNDInterpolator(list(zip(x_edges,y_edges,z_edges)), vr_edges)
    vr_masked = np.copy(vr)
    vr_masked[disk_mask] = vr_interp_func(x[disk_mask], y[disk_mask], z[disk_mask])
    vr_masked = gaussian_filter(vr_masked, smooth_scale)
    dvr = np.gradient(vr_masked, dx_cm)
    delta_vr = dvr[0]*dx_cm*x_hat + dvr[1]*dx_cm*y_hat + dvr[2]*dx_cm*z_hat
    ram_pressure = smooth_den*(delta_vr)**2.
    pres_grad = np.gradient(ram_pressure, dx_cm)
    dPdr = pres_grad[0]*x_hat + pres_grad[1]*y_hat + pres_grad[2]*z_hat
    ram_force = -1./den_masked * dPdr
    vtheta = box['theta_velocity_corrected'].in_units('cm/s').v
    vphi = box['phi_velocity_corrected'].in_units('cm/s').v
    vtheta_edges = vtheta[disk_edges]
    vphi_edges = vphi[disk_edges]
    vtheta_interp_func = NearestNDInterpolator(list(zip(x_edges,y_edges,z_edges)), vtheta_edges)
    vtheta_masked = np.copy(vtheta)
    vphi_interp_func = NearestNDInterpolator(list(zip(x_edges,y_edges,z_edges)), vphi_edges)
    vphi_masked = np.copy(vphi)
    vtheta_masked[disk_mask] = vtheta_interp_func(x[disk_mask], y[disk_mask], z[disk_mask])
    vphi_masked[disk_mask] = vphi_interp_func(x[disk_mask], y[disk_mask], z[disk_mask])
    smooth_vtheta = gaussian_filter(vtheta_masked, smooth_scale)
    smooth_vphi = gaussian_filter(vphi_masked, smooth_scale)
    rot_force = (smooth_vtheta**2. + smooth_vphi**2.)/r
    grav_force = -G*Menc_profile(r/(1000*cmtopc))*gtoMsun/r**2.
    tot_force = thermal_force + turb_force + rot_force + ram_force + grav_force

    if (args.shader_x == 'radius'):
        x = box['radius_corrected'].in_units('kpc').v
        x_range = [0, 250]
        xlabel = 'Radius [kpc]'
    elif (args.shader_x == 'radial_velocity'):
        x = box['radial_velocity_corrected'].in_units('km/s').v
        x_range = [-500,1000]
        xlabel = 'Radial velocity [km/s]'

    if (args.shader_y == 'radial_velocity'):
        y = box['radial_velocity_corrected'].in_units('km/s').v
        y_range = [-500,1500]
        ylabel = 'Radial velocity [km/s]'
    elif (args.shader_y == 'temperature'):
        y = np.log10(box['temperature'].v)
        y_range = [4,7]
        ylabel = 'log Temperature [K]'

    if (args.shader_color == 'thermal_to_turbulent'):
        color_field = 'thermal-to-turbulent-force-ratio'
        color_val = thermal_force/turb_force
        color_func = categorize_by_force_ratio
        color_key = force_ratio_color_key
        cmin = force_ratio_min
        cmax = force_ratio_max
        step = 750./np.size(list(color_key))
        color_ticks = [step*1,step*3.,step*5.,step*7.,step*9.]
        color_ticklabels = ['-4','-2','0','2','4']
        field_label = 'Ratio of Thermal to Turbulent Forces'
        color_log = False

    data_frame = pd.DataFrame({})
    data_frame['x'] = x.flatten()
    data_frame['y'] = y.flatten()
    data_frame[color_field] = color_val.flatten()
    data_frame['color'] = color_func(data_frame[color_field])
    data_frame.color = data_frame.color.astype('category')
    cvs = dshader.Canvas(plot_width=800, plot_height=600, x_range=x_range, y_range=y_range)
    agg = cvs.points(data_frame, 'x', 'y', dshader.count_cat('color'))
    img = tf.spread(tf.shade(agg, color_key=color_key, how='eq_hist',min_alpha=40), shape='square', px=0)
    export_image(img, save_dir + args.shader_y + '_vs_' + args.shader_x + '_' + color_field + '-colored' + save_suffix)
    fig = plt.figure(figsize=(8,6),dpi=500)
    ax = fig.add_subplot(1,1,1)
    image = plt.imread(save_dir + args.shader_y + '_vs_' + args.shader_x + '_' + color_field + '-colored' + save_suffix + '.png')
    ax.imshow(image, extent=[x_range[0],x_range[1],y_range[0],y_range[1]])
    ax.set_aspect(6*abs(x_range[1]-x_range[0])/(8*abs(y_range[1]-y_range[0])))
    ax.set_xlabel(xlabel, fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)
    ax.tick_params(axis='both', which='both', direction='in', length=8, width=2, pad=5, labelsize=14, \
      top=True, right=True)
    ax2 = fig.add_axes([0.7, 0.93, 0.25, 0.06])
    cmap = create_foggie_cmap(cmin, cmax, color_func, color_key, color_log)
    ax2.imshow(np.flip(cmap.to_pil(), 1))
    ax2.set_xticks(color_ticks)
    ax2.set_xticklabels(color_ticklabels, fontsize=14)
    ax2.text(400, 150, field_label, fontsize=14, ha='center', va='center')
    ax2.spines["top"].set_color('white')
    ax2.spines["bottom"].set_color('white')
    ax2.spines["left"].set_color('white')
    ax2.spines["right"].set_color('white')
    ax2.set_ylim(60, 180)
    ax2.set_xlim(-10, 750)
    ax2.set_yticklabels([])
    ax2.set_yticks([])
    plt.subplots_adjust(left=0.15, bottom=0.05)
    plt.savefig(save_dir + args.shader_y + '_vs_' + args.shader_x + '_' + color_field + '-colored' + save_suffix + '.png')
    plt.close()
    print('Plot made.')

def pressure_slice(snap):
    '''Plots a slice of pressure through the center of the halo. The option --pressure_type indicates
    what type of pressure to plot.'''

    masses_ind = np.where(masses['snapshot']==snap)[0]
    Menc_profile = IUS(np.concatenate(([0],masses['radius'][masses_ind])), np.concatenate(([0],masses['total_mass'][masses_ind])))
    Mvir = rvir_masses['total_mass'][rvir_masses['snapshot']==snap]
    Rvir = rvir_masses['radius'][rvir_masses['snapshot']==snap][0]

    if (args.system=='pleiades_cassi'):
        print('Copying directory to /tmp')
        snap_dir = '/tmp/' + args.halo + '/' + args.run + '/' + target_dir + '/' + snap
        if (args.copy_to_tmp):
            shutil.copytree(foggie_dir + run_dir + snap, snap_dir)
            snap_name = snap_dir + '/' + snap
        else:
            snap_dir = '/nobackup/clochhaa/tmp/' + args.halo + '/' + args.run + '/' + target_dir + '/' + snap
            # Make a dummy directory with the snap name so the script later knows the process running
            # this snapshot failed if the directory is still there
            os.makedirs(snap_dir)
            snap_name = foggie_dir + run_dir + snap + '/' + snap
    else:
        snap_name = foggie_dir + run_dir + snap + '/' + snap
    ds, refine_box = foggie_load(snap_name, trackname, do_filter_particles=False, halo_c_v_name=halo_c_v_name, gravity=True, masses_dir=masses_dir)

    if (args.pressure_type=='all'):
        ptypes = ['thermal', 'turbulent', 'ram']
    elif (',' in args.pressure_type):
        ptypes = args.pressure_type.split(',')
    else:
        ptypes = [args.pressure_type]

    # Define the density cut between disk and CGM to vary smoothly between 1 and 0.1 between z = 0.5 and z = 0.25,
    # with it being 1 at higher redshifts and 0.1 at lower redshifts
    current_time = ds.current_time.in_units('Myr').v
    if (current_time<=8656.88):
        density_cut_factor = 1.
    elif (current_time<=10787.12):
        density_cut_factor = 1. - 0.9*(current_time-8656.88)/2130.24
    else:
        density_cut_factor = 0.1

    pix_res = float(np.min(refine_box['gas','dx'].in_units('kpc')))  # at level 11
    lvl1_res = pix_res*2.**11.
    level = 9
    dx = lvl1_res/(2.**level)
    smooth_scale = (25./dx)/6.
    dx_cm = lvl1_res/(2.**level)*1000*cmtopc
    refine_res = int(3.*Rvir/dx)
    box = ds.covering_grid(level=level, left_edge=ds.halo_center_kpc-ds.arr([1.5*Rvir,1.5*Rvir,1.5*Rvir],'kpc'), dims=[refine_res, refine_res, refine_res])
    density = box['density'].in_units('g/cm**3').v
    temperature = box['temperature'].v

    # This next block needed for removing any ISM regions and then interpolating over the holes left behind
    x = box[('gas','x')].in_units('cm').v - ds.halo_center_kpc[0].to('cm').v
    y = box[('gas','y')].in_units('cm').v - ds.halo_center_kpc[1].to('cm').v
    z = box[('gas','z')].in_units('cm').v - ds.halo_center_kpc[2].to('cm').v
    # Define ISM regions to remove
    disk_mask = (density > cgm_density_max * density_cut_factor)
    # disk_mask_expanded is a binary mask of both ISM regions AND their surrounding pixels
    struct = ndimage.generate_binary_structure(3,3)
    disk_mask_expanded = ndimage.binary_dilation(disk_mask, structure=struct, iterations=3)
    disk_mask_expanded = ndimage.binary_closing(disk_mask_expanded, structure=struct, iterations=3)
    disk_mask_expanded = disk_mask_expanded | disk_mask
    # disk_edges is a binary mask of ONLY pixels surrounding ISM regions -- nothing inside ISM regions
    disk_edges = disk_mask_expanded & ~disk_mask
    x_edges = x[disk_edges].flatten()
    y_edges = y[disk_edges].flatten()
    z_edges = z[disk_edges].flatten()

    den_edges = density[disk_edges]
    den_interp_func = NearestNDInterpolator(list(zip(x_edges,y_edges,z_edges)), den_edges)
    den_masked = np.copy(density)
    den_masked[disk_mask] = den_interp_func(x[disk_mask], y[disk_mask], z[disk_mask])
    smooth_den = gaussian_filter(den_masked, smooth_scale)

    for i in range(len(ptypes)):
        if (ptypes[i]=='thermal'):
            thermal_pressure = box['pressure'].in_units('erg/cm**3').v
            # Cut to only those values closest to removed ISM regions
            pres_edges = thermal_pressure[disk_edges]
            # Interpolate across removed ISM regions
            pres_interp_func = NearestNDInterpolator(list(zip(x_edges,y_edges,z_edges)), pres_edges)
            pres_masked = np.copy(thermal_pressure)
            # Replace removed ISM regions with interpolated values
            interp_values = pres_interp_func(x[disk_mask], y[disk_mask], z[disk_mask])
            pres_masked[disk_mask] = interp_values
            pressure = pres_masked
            pressure_label = 'Thermal'
        if (ptypes[i]=='turbulent'):
            vx = box['vx_corrected'].in_units('cm/s').v
            vy = box['vy_corrected'].in_units('cm/s').v
            vz = box['vz_corrected'].in_units('cm/s').v
            # Cut to only those values closest to removed ISM regions
            vx_edges = vx[disk_edges]
            vy_edges = vy[disk_edges]
            vz_edges = vz[disk_edges]
            # Interpolate across removed ISM regions
            vx_interp_func = NearestNDInterpolator(list(zip(x_edges,y_edges,z_edges)), vx_edges)
            vy_interp_func = NearestNDInterpolator(list(zip(x_edges,y_edges,z_edges)), vy_edges)
            vz_interp_func = NearestNDInterpolator(list(zip(x_edges,y_edges,z_edges)), vz_edges)
            vx_masked = np.copy(vx)
            vy_masked = np.copy(vy)
            vz_masked = np.copy(vz)
            # Replace removed ISM regions with interpolated values
            vx_masked[disk_mask] = vx_interp_func(x[disk_mask], y[disk_mask], z[disk_mask])
            vy_masked[disk_mask] = vy_interp_func(x[disk_mask], y[disk_mask], z[disk_mask])
            vz_masked[disk_mask] = vz_interp_func(x[disk_mask], y[disk_mask], z[disk_mask])
            # Smooth resulting velocity field -- without contamination from ISM regions
            smooth_vx = gaussian_filter(vx_masked, smooth_scale)
            smooth_vy = gaussian_filter(vy_masked, smooth_scale)
            smooth_vz = gaussian_filter(vz_masked, smooth_scale)
            sig_x = (vx - smooth_vx)**2.
            sig_y = (vy - smooth_vy)**2.
            sig_z = (vz - smooth_vz)**2.
            vdisp = np.sqrt((sig_x + sig_y + sig_z)/3.)
            smooth_vdisp = gaussian_filter(vdisp, smooth_scale)
            turb_pressure = smooth_den*smooth_vdisp**2.
            pressure = turb_pressure
            pressure_label = 'Turbulent'
        if (ptypes[i]=='ram'):
            r = box['radius_corrected'].in_units('cm').v
            x_hat = x/r
            y_hat = y/r
            z_hat = z/r
            vr = box['radial_velocity_corrected'].in_units('cm/s').v
            # Cut to only those values closest to removed ISM regions
            vr_edges = vr[disk_edges]
            # Interpolate across removed ISM regions
            vr_interp_func = NearestNDInterpolator(list(zip(x_edges,y_edges,z_edges)), vr_edges)
            vr_masked = np.copy(vr)
            # Replace removed ISM regions with interpolated values
            vr_masked[disk_mask] = vr_interp_func(x[disk_mask], y[disk_mask], z[disk_mask])
            dvr = np.gradient(vr_masked, dx_cm)
            delta_vr = dvr[0]*dx_cm*x_hat + dvr[1]*dx_cm*y_hat + dvr[2]*dx_cm*z_hat
            smooth_delta_vr = gaussian_filter(delta_vr, smooth_scale)
            ram_pressure = smooth_den*(smooth_delta_vr)**2.
            pressure = ram_pressure
            pressure_label = 'Ram'

        pressure = np.ma.masked_where((density > cgm_density_max * density_cut_factor), pressure)
        fig = plt.figure(figsize=(13,10),dpi=500)
        ax = fig.add_subplot(1,1,1)
        p_cmap = copy.copy(mpl.cm.get_cmap(pressure_color_map))
        p_cmap.set_over(color='w', alpha=1.)
        # Need to rotate to match up with how yt plots it
        im = ax.imshow(rotate(np.log10(pressure[len(pressure)//2,:,:]),90), cmap=p_cmap, norm=colors.Normalize(vmin=-18, vmax=-12), \
                  extent=[-1.5*Rvir,1.5*Rvir,-1.5*Rvir,1.5*Rvir])
        ax.axis([-250,250,-250,250])
        ax.set_xlabel('y [kpc]', fontsize=28)
        ax.set_ylabel('z [kpc]', fontsize=28)
        ax.text(-200,200,pressure_label, fontsize=28, ha='left', va='center', color='w')
        ax.tick_params(axis='both', which='both', direction='in', length=8, width=2, pad=5, labelsize=28, \
          top=True, right=True)
        cax = fig.add_axes([0.808, 0.1, 0.03, 0.88])
        cax.tick_params(axis='both', which='both', direction='in', length=8, width=2, pad=5, labelsize=28, \
          top=True, right=True)
        fig.colorbar(im, cax=cax, orientation='vertical')
        ax.text(1.2, 0.5, 'log ' + pressure_label + ' Pressure [erg/cm$^3$]', fontsize=28, rotation='vertical', ha='center', va='center', transform=ax.transAxes)
        plt.subplots_adjust(bottom=0.1, top=0.98, left=0.08, right=0.86)
        plt.savefig(save_dir + snap + '_' + ptypes[i] + '_pressure_slice_x' + save_suffix + '.png')

    # Delete output from temp directory if on pleiades
    if (args.system=='pleiades_cassi'):
        print('Deleting directory from /tmp')
        shutil.rmtree(snap_dir)

def force_slice(snap):
    '''Plots a slice of different force terms through the center of the halo. The option --force_type indicates
    what type of force to plot.'''

    masses_ind = np.where(masses['snapshot']==snap)[0]
    Menc_profile = IUS(np.concatenate(([0],masses['radius'][masses_ind])), np.concatenate(([0],masses['total_mass'][masses_ind])))
    Mvir = rvir_masses['total_mass'][rvir_masses['snapshot']==snap]
    Rvir = rvir_masses['radius'][rvir_masses['snapshot']==snap][0]

    if (args.system=='pleiades_cassi'):
        print('Copying directory to /tmp')
        snap_dir = '/tmp/' + args.halo + '/' + args.run + '/' + target_dir + '/' + snap
        if (args.copy_to_tmp):
            shutil.copytree(foggie_dir + run_dir + snap, snap_dir)
            snap_name = snap_dir + '/' + snap
        else:
            snap_dir = '/nobackup/clochhaa/tmp/' + args.halo + '/' + args.run + '/' + target_dir + '/' + snap
            # Make a dummy directory with the snap name so the script later knows the process running
            # this snapshot failed if the directory is still there
            os.makedirs(snap_dir)
            snap_name = foggie_dir + run_dir + snap + '/' + snap
    else:
        snap_name = foggie_dir + run_dir + snap + '/' + snap
    ds, refine_box = foggie_load(snap_name, trackname, do_filter_particles=False, halo_c_v_name=halo_c_v_name, gravity=True, masses_dir=masses_dir)

    # Define the density cut between disk and CGM to vary smoothly between 1 and 0.1 between z = 0.5 and z = 0.25,
    # with it being 1 at higher redshifts and 0.1 at lower redshifts
    current_time = ds.current_time.in_units('Myr').v
    if (current_time<=8656.88):
        density_cut_factor = 1.
    elif (current_time<=10787.12):
        density_cut_factor = 1. - 0.9*(current_time-8656.88)/2130.24
    else:
        density_cut_factor = 0.1

    if (args.force_type=='all'):
        ftypes = ['thermal', 'turbulent', 'ram', 'rotation', 'gravity', 'total']
    elif (',' in args.force_type):
        ftypes = args.force_type.split(',')
    else:
        ftypes = [args.force_type]

    pix_res = float(np.min(refine_box[('gas','dx')].in_units('kpc')))  # at level 11
    lvl1_res = pix_res*2.**11.
    level = 9
    dx = lvl1_res/(2.**level)
    smooth_scale = (25./dx)/6.
    smooth_scale1 = (5./dx)/6.
    smooth_scale2 = (15./dx)/6.
    smooth_scale3 = (50./dx)/6.
    smooth_scale4 = (100./dx)/6.
    smooth_scale5 = (200./dx)/6.
    dx_cm = dx*1000*cmtopc
    refine_res = int(3.*Rvir/dx)
    box = ds.covering_grid(level=level, left_edge=ds.halo_center_kpc-ds.arr([1.5*Rvir,1.5*Rvir,1.5*Rvir],'kpc'), dims=[refine_res, refine_res, refine_res])
    density = box['density'].in_units('g/cm**3').v
    temperature = box['temperature'].v
    mass = box['cell_mass'].in_units('g').v
    x = box[('gas','x')].in_units('cm').v - ds.halo_center_kpc[0].to('cm').v
    y = box[('gas','y')].in_units('cm').v - ds.halo_center_kpc[1].to('cm').v
    z = box[('gas','z')].in_units('cm').v - ds.halo_center_kpc[2].to('cm').v
    r = box['radius_corrected'].in_units('cm').v
    x_hat = x/r
    y_hat = y/r
    z_hat = z/r

    # This next block needed for removing any ISM regions and then interpolating over the holes left behind
    # Define ISM regions to remove
    disk_mask = (density > cgm_density_max * density_cut_factor)
    # disk_mask_expanded is a binary mask of both ISM regions AND their surrounding pixels
    struct = ndimage.generate_binary_structure(3,3)
    disk_mask_expanded = ndimage.binary_dilation(disk_mask, structure=struct, iterations=3)
    disk_mask_expanded = ndimage.binary_closing(disk_mask_expanded, structure=struct, iterations=3)
    disk_mask_expanded = disk_mask_expanded | disk_mask
    # disk_edges is a binary mask of ONLY pixels surrounding ISM regions -- nothing inside ISM regions
    disk_edges = disk_mask_expanded & ~disk_mask
    x_edges = x[disk_edges].flatten()
    y_edges = y[disk_edges].flatten()
    z_edges = z[disk_edges].flatten()

    den_edges = density[disk_edges]
    den_interp_func = NearestNDInterpolator(list(zip(x_edges,y_edges,z_edges)), den_edges)
    den_masked = np.copy(density)
    den_masked[disk_mask] = den_interp_func(x[disk_mask], y[disk_mask], z[disk_mask])
    smooth_den = gaussian_filter(den_masked, smooth_scale)

    for i in range(len(ftypes)):
        if (ftypes[i]=='thermal') or ((ftypes[i]=='total') and (args.force_type!='all')):
            thermal_pressure = box['pressure'].in_units('erg/cm**3').v
            # Cut to only those values closest to removed ISM regions
            pres_edges = thermal_pressure[disk_edges]
            # Interpolate across removed ISM regions
            pres_interp_func = NearestNDInterpolator(list(zip(x_edges,y_edges,z_edges)), pres_edges)
            pres_masked = np.copy(thermal_pressure)
            # Replace removed ISM regions with interpolated values
            pres_masked[disk_mask] = pres_interp_func(x[disk_mask], y[disk_mask], z[disk_mask])
            pres_grad = np.gradient(thermal_pressure, dx_cm)
            dPdr = pres_grad[0]*x_hat + pres_grad[1]*y_hat + pres_grad[2]*z_hat
            thermal_force = -1./den_masked * dPdr
            #thermal_force = -dPdr
            if (ftypes[i]=='thermal'):
                if (args.smoothed):
                    force = gaussian_filter(thermal_force, smooth_scale5)
                else:
                    force = thermal_force
                force_label = 'Thermal Pressure'
                slice_label = 'Thermal'
        if (ftypes[i]=='turbulent') or ((ftypes[i]=='total') and (args.force_type!='all')):
            vx = box['vx_corrected'].in_units('cm/s').v
            vy = box['vy_corrected'].in_units('cm/s').v
            vz = box['vz_corrected'].in_units('cm/s').v
            # Cut to only those values closest to removed ISM regions
            vx_edges = vx[disk_edges]
            vy_edges = vy[disk_edges]
            vz_edges = vz[disk_edges]
            # Interpolate across removed ISM regions
            vx_interp_func = NearestNDInterpolator(list(zip(x_edges,y_edges,z_edges)), vx_edges)
            vy_interp_func = NearestNDInterpolator(list(zip(x_edges,y_edges,z_edges)), vy_edges)
            vz_interp_func = NearestNDInterpolator(list(zip(x_edges,y_edges,z_edges)), vz_edges)
            vx_masked = np.copy(vx)
            vy_masked = np.copy(vy)
            vz_masked = np.copy(vz)
            # Replace removed ISM regions with interpolated values
            vx_masked[disk_mask] = vx_interp_func(x[disk_mask], y[disk_mask], z[disk_mask])
            vy_masked[disk_mask] = vy_interp_func(x[disk_mask], y[disk_mask], z[disk_mask])
            vz_masked[disk_mask] = vz_interp_func(x[disk_mask], y[disk_mask], z[disk_mask])
            # Smooth resulting velocity field -- without contamination from ISM regions
            smooth_vx = gaussian_filter(vx_masked, smooth_scale)
            smooth_vy = gaussian_filter(vy_masked, smooth_scale)
            smooth_vz = gaussian_filter(vz_masked, smooth_scale)
            sig_x = (vx - smooth_vx)**2.
            sig_y = (vy - smooth_vy)**2.
            sig_z = (vz - smooth_vz)**2.
            vdisp = np.sqrt((sig_x + sig_y + sig_z)/3.)
            turb_pressure = smooth_den*vdisp**2.
            pres_grad = np.gradient(turb_pressure, dx_cm)
            dPdr = pres_grad[0]*x_hat + pres_grad[1]*y_hat + pres_grad[2]*z_hat
            turb_force = -1./den_masked * dPdr
            #turb_force = -dPdr
            if (ftypes[i]=='turbulent'):
                if (args.smoothed):
                    force = gaussian_filter(turb_force, smooth_scale5)
                else:
                    force = turb_force
                force_label = 'Turbulent Pressure'
                slice_label = 'Turbulent'
        if (ftypes[i]=='ram') or ((ftypes[i]=='total') and (args.force_type!='all')):
            vr = box['radial_velocity_corrected'].in_units('cm/s').v
            # Cut to only those values closest to removed ISM regions
            vr_edges = vr[disk_edges]
            # Interpolate across removed ISM regions
            vr_interp_func = NearestNDInterpolator(list(zip(x_edges,y_edges,z_edges)), vr_edges)
            vr_masked = np.copy(vr)
            # Replace removed ISM regions with interpolated values
            vr_masked[disk_mask] = vr_interp_func(x[disk_mask], y[disk_mask], z[disk_mask])
            dvr = np.gradient(vr_masked, dx_cm)
            delta_vr = dvr[0]*dx_cm*x_hat + dvr[1]*dx_cm*y_hat + dvr[2]*dx_cm*z_hat
            smooth_delta_vr = gaussian_filter(delta_vr, smooth_scale)
            ram_pressure = smooth_den*(smooth_delta_vr)**2.
            pres_grad = np.gradient(ram_pressure, dx_cm)
            dPdr = pres_grad[0]*x_hat + pres_grad[1]*y_hat + pres_grad[2]*z_hat
            ram_force = -1./den_masked * dPdr
            #ram_force = -dPdr
            if (ftypes[i]=='ram'):
                if (args.smoothed):
                    force = gaussian_filter(ram_force, smooth_scale5)
                else:
                    force = ram_force
                force_label = 'Ram Pressure'
                slice_label = 'Ram'
        if (ftypes[i]=='rotation') or ((ftypes[i]=='total') and (args.force_type!='all')):
            vtheta = box['theta_velocity_corrected'].in_units('cm/s').v
            vphi = box['phi_velocity_corrected'].in_units('cm/s').v
            # Cut to only those values closest to removed ISM regions
            vtheta_edges = vtheta[disk_edges]
            vphi_edges = vphi[disk_edges]
            # Interpolate across removed ISM regions
            vtheta_interp_func = NearestNDInterpolator(list(zip(x_edges,y_edges,z_edges)), vtheta_edges)
            vtheta_masked = np.copy(vtheta)
            vphi_interp_func = NearestNDInterpolator(list(zip(x_edges,y_edges,z_edges)), vphi_edges)
            vphi_masked = np.copy(vphi)
            # Replace removed ISM regions with interpolated values
            vtheta_masked[disk_mask] = vtheta_interp_func(x[disk_mask], y[disk_mask], z[disk_mask])
            vphi_masked[disk_mask] = vphi_interp_func(x[disk_mask], y[disk_mask], z[disk_mask])
            smooth_vtheta = gaussian_filter(vtheta_masked, smooth_scale)
            smooth_vphi = gaussian_filter(vphi_masked, smooth_scale)
            rot_force = (smooth_vtheta**2. + smooth_vphi**2.)/r
            if (ftypes[i]=='rotation'):
                if (args.smoothed):
                    force = gaussian_filter(rot_force, smooth_scale5)
                else:
                    force = rot_force
                force_label = 'Rotation'
                slice_label = 'Rotation'
        if (ftypes[i]=='gravity') or ((ftypes[i]=='total') and (args.force_type!='all')):
            grav_force = -G*Menc_profile(r/(1000*cmtopc))*gtoMsun/r**2.
            if (ftypes[i]=='gravity'):
                if (args.smoothed):
                    force = gaussian_filter(grav_force, smooth_scale5)
                else:
                    force = grav_force
                force_label = 'Gravity'
                slice_label = 'Gravity'
        if (ftypes[i]=='total'):
            tot_force = thermal_force + turb_force + rot_force + ram_force + grav_force
            if (args.smoothed):
                force = gaussian_filter(tot_force, smooth_scale5)
            else:
                force = tot_force
            force_label = 'Total'
            slice_label = '$F_\mathrm{net}$'

        force = np.ma.masked_where((density > cgm_density_max * density_cut_factor), force)
        fig = plt.figure(figsize=(12,10),dpi=500)
        ax = fig.add_subplot(1,1,1)
        f_cmap = copy.copy(mpl.cm.get_cmap('BrBG'))
        f_cmap.set_over(color='w', alpha=1.)
        # Need to rotate to match up with how yt plots it
        im = ax.imshow(rotate(force[len(force)//2,:,:],90), cmap=f_cmap, norm=colors.SymLogNorm(vmin=-1e-5, vmax=1e-5, linthresh=1e-9, base=10), \
                  extent=[-1.5*Rvir,1.5*Rvir,-1.5*Rvir,1.5*Rvir])
        #im = ax.imshow(rotate(force[len(force)//2,:,:],90), cmap=f_cmap, norm=colors.SymLogNorm(vmin=-1e-33, vmax=1e-33, linthresh=1e-37, base=10), \
                  #extent=[-1.5*Rvir,1.5*Rvir,-1.5*Rvir,1.5*Rvir])
        ax.axis([-250,250,-250,250])
        ax.set_xlabel('y [kpc]', fontsize=24)
        ax.set_ylabel('z [kpc]', fontsize=24)
        ax.text(-200,200,slice_label, fontsize=24, ha='left', va='center')
        ax.tick_params(axis='both', which='both', direction='in', length=8, width=2, pad=5, labelsize=24, \
          top=True, right=True)
        cax = fig.add_axes([0.82, 0.11, 0.03, 0.84])
        cax.tick_params(axis='both', which='both', direction='in', length=8, width=2, pad=5, labelsize=24, \
          top=True, right=True)
        fig.colorbar(im, cax=cax, orientation='vertical')
        ax.text(1.2, 0.5, force_label + ' Force [cm/s$^2$]', fontsize=24, rotation='vertical', ha='center', va='center', transform=ax.transAxes)
        #ax.text(1.2, 0.5, force_label + ' Pressure Gradient [erg/cm$^4$]', fontsize=20, rotation='vertical', ha='center', va='center', transform=ax.transAxes)
        plt.subplots_adjust(bottom=0.08, top=0.98, left=0.12, right=0.82)
        plt.savefig(save_dir + snap + '_' + ftypes[i] + '_force_slice_x' + save_suffix + '.png')

    # Delete output from temp directory if on pleiades
    if (args.system=='pleiades_cassi'):
        print('Deleting directory from /tmp')
        shutil.rmtree(snap_dir)

def tangential_force_slice(snap):
    '''Plots a slice of the tangential directions of different force terms through the center of the
    halo. The option --force_type indicates what type of force to plot.'''

    masses_ind = np.where(masses['snapshot']==snap)[0]
    Menc_profile = IUS(np.concatenate(([0],masses['radius'][masses_ind])), np.concatenate(([0],masses['total_mass'][masses_ind])))
    Mvir = rvir_masses['total_mass'][rvir_masses['snapshot']==snap]
    Rvir = rvir_masses['radius'][rvir_masses['snapshot']==snap][0]

    if (args.system=='pleiades_cassi'):
        print('Copying directory to /tmp')
        snap_dir = '/tmp/' + args.halo + '/' + args.run + '/' + target_dir + '/' + snap
        if (args.copy_to_tmp):
            shutil.copytree(foggie_dir + run_dir + snap, snap_dir)
            snap_name = snap_dir + '/' + snap
        else:
            # Make a dummy directory with the snap name so the script later knows the process running
            # this snapshot failed if the directory is still there
            os.makedirs(snap_dir)
            snap_name = foggie_dir + run_dir + snap + '/' + snap
    else:
        snap_name = foggie_dir + run_dir + snap + '/' + snap
    ds, refine_box = foggie_load(snap_name, trackname, do_filter_particles=False, halo_c_v_name=halo_c_v_name, gravity=True, masses_dir=masses_dir)

    # Define the density cut between disk and CGM to vary smoothly between 1 and 0.1 between z = 0.5 and z = 0.25,
    # with it being 1 at higher redshifts and 0.1 at lower redshifts
    current_time = ds.current_time.in_units('Myr').v
    if (current_time<=8656.88):
        density_cut_factor = 1.
    elif (current_time<=10787.12):
        density_cut_factor = 1. - 0.9*(current_time-8656.88)/2130.24
    else:
        density_cut_factor = 0.1

    if (args.force_type=='all'):
        ftypes = ['thermal', 'turbulent', 'ram']
    elif (',' in args.force_type):
        ftypes = args.force_type.split(',')
    else:
        ftypes = [args.force_type]

    pix_res = float(np.min(refine_box[('gas','dx')].in_units('kpc')))  # at level 11
    lvl1_res = pix_res*2.**11.
    level = 9
    dx = lvl1_res/(2.**level)
    smooth_scale = (25./dx)/6.
    dx_cm = dx*1000*cmtopc
    refine_res = int(3.*Rvir/dx)
    box = ds.covering_grid(level=level, left_edge=ds.halo_center_kpc-ds.arr([1.5*Rvir,1.5*Rvir,1.5*Rvir],'kpc'), dims=[refine_res, refine_res, refine_res])
    density = box['density'].in_units('g/cm**3').v
    temperature = box['temperature'].v
    mass = box['cell_mass'].in_units('g').v
    x = box[('gas','x')].in_units('cm').v - ds.halo_center_kpc[0].to('cm').v
    y = box[('gas','y')].in_units('cm').v - ds.halo_center_kpc[1].to('cm').v
    z = box[('gas','z')].in_units('cm').v - ds.halo_center_kpc[2].to('cm').v
    r = box['radius_corrected'].in_units('cm').v
    x_hat = x/r
    y_hat = y/r
    z_hat = z/r

    # This next block needed for removing any ISM regions and then interpolating over the holes left behind
    # Define ISM regions to remove
    disk_mask = (density > cgm_density_max * density_cut_factor)
    # disk_mask_expanded is a binary mask of both ISM regions AND their surrounding pixels
    struct = ndimage.generate_binary_structure(3,3)
    disk_mask_expanded = ndimage.binary_dilation(disk_mask, structure=struct, iterations=3)
    disk_mask_expanded = ndimage.binary_closing(disk_mask_expanded, structure=struct, iterations=3)
    disk_mask_expanded = disk_mask_expanded | disk_mask
    # disk_edges is a binary mask of ONLY pixels surrounding ISM regions -- nothing inside ISM regions
    disk_edges = disk_mask_expanded & ~disk_mask
    x_edges = x[disk_edges].flatten()
    y_edges = y[disk_edges].flatten()
    z_edges = z[disk_edges].flatten()

    den_edges = density[disk_edges]
    den_interp_func = NearestNDInterpolator(list(zip(x_edges,y_edges,z_edges)), den_edges)
    den_masked = np.copy(density)
    den_masked[disk_mask] = den_interp_func(x[disk_mask], y[disk_mask], z[disk_mask])
    smooth_den = gaussian_filter(den_masked, smooth_scale)

    for i in range(len(ftypes)):
        if (ftypes[i]=='thermal') or ((ftypes[i]=='total') and (args.force_type!='all')):
            thermal_pressure = box['pressure'].in_units('erg/cm**3').v
            # Cut to only those values closest to removed ISM regions
            pres_edges = thermal_pressure[disk_edges]
            # Interpolate across removed ISM regions
            pres_interp_func = NearestNDInterpolator(list(zip(x_edges,y_edges,z_edges)), pres_edges)
            pres_masked = np.copy(thermal_pressure)
            # Replace removed ISM regions with interpolated values
            pres_masked[disk_mask] = pres_interp_func(x[disk_mask], y[disk_mask], z[disk_mask])
            pres_grad = np.gradient(thermal_pressure, dx_cm)
            #dPdr = pres_grad[0]*x_hat + pres_grad[1]*y_hat + pres_grad[2]*z_hat
            #dPdtheta = (y*pres_grad[0] - x*pres_grad[1])/(np.sqrt(x*x+y*y))
            dPdphi = (z*x*pres_grad[0] + z*y*pres_grad[1] - (x*x+y*y)*pres_grad[2])/(r*np.sqrt(x*x+y*y))
            thermal_force = -1./den_masked * dPdphi
            if (ftypes[i]=='thermal'):
                force = thermal_force
                force_label = 'Thermal Pressure'
        if (ftypes[i]=='turbulent') or ((ftypes[i]=='total') and (args.force_type!='all')):
            vx = box['vx_corrected'].in_units('cm/s').v
            vy = box['vy_corrected'].in_units('cm/s').v
            vz = box['vz_corrected'].in_units('cm/s').v
            # Cut to only those values closest to removed ISM regions
            vx_edges = vx[disk_edges]
            vy_edges = vy[disk_edges]
            vz_edges = vz[disk_edges]
            # Interpolate across removed ISM regions
            vx_interp_func = NearestNDInterpolator(list(zip(x_edges,y_edges,z_edges)), vx_edges)
            vy_interp_func = NearestNDInterpolator(list(zip(x_edges,y_edges,z_edges)), vy_edges)
            vz_interp_func = NearestNDInterpolator(list(zip(x_edges,y_edges,z_edges)), vz_edges)
            vx_masked = np.copy(vx)
            vy_masked = np.copy(vy)
            vz_masked = np.copy(vz)
            # Replace removed ISM regions with interpolated values
            vx_masked[disk_mask] = vx_interp_func(x[disk_mask], y[disk_mask], z[disk_mask])
            vy_masked[disk_mask] = vy_interp_func(x[disk_mask], y[disk_mask], z[disk_mask])
            vz_masked[disk_mask] = vz_interp_func(x[disk_mask], y[disk_mask], z[disk_mask])
            # Smooth resulting velocity field -- without contamination from ISM regions
            smooth_vx = gaussian_filter(vx_masked, smooth_scale)
            smooth_vy = gaussian_filter(vy_masked, smooth_scale)
            smooth_vz = gaussian_filter(vz_masked, smooth_scale)
            sig_x = (vx - smooth_vx)**2.
            sig_y = (vy - smooth_vy)**2.
            sig_z = (vz - smooth_vz)**2.
            vdisp = np.sqrt((sig_x + sig_y + sig_z)/3.)
            turb_pressure = smooth_den*vdisp**2.
            pres_grad = np.gradient(turb_pressure, dx_cm)
            #dPdr = pres_grad[0]*x_hat + pres_grad[1]*y_hat + pres_grad[2]*z_hat
            #dPdtheta = (y*pres_grad[0] - x*pres_grad[1])/(np.sqrt(x*x+y*y))
            dPdphi = (z*x*pres_grad[0] + z*y*pres_grad[1] - (x*x+y*y)*pres_grad[2])/(r*np.sqrt(x*x+y*y))
            turb_force = -1./den_masked * dPdphi
            if (ftypes[i]=='turbulent'):
                force = turb_force
                force_label = 'Turbulent Pressure'
        if (ftypes[i]=='ram') or ((ftypes[i]=='total') and (args.force_type!='all')):
            vr = box['radial_velocity_corrected'].in_units('cm/s').v
            # Cut to only those values closest to removed ISM regions
            vr_edges = vr[disk_edges]
            # Interpolate across removed ISM regions
            vr_interp_func = NearestNDInterpolator(list(zip(x_edges,y_edges,z_edges)), vr_edges)
            vr_masked = np.copy(vr)
            # Replace removed ISM regions with interpolated values
            vr_masked[disk_mask] = vr_interp_func(x[disk_mask], y[disk_mask], z[disk_mask])
            dvr = np.gradient(vr_masked, dx_cm)
            delta_vr = dvr[0]*dx_cm*x_hat + dvr[1]*dx_cm*y_hat + dvr[2]*dx_cm*z_hat
            smooth_delta_vr = gaussian_filter(delta_vr, smooth_scale)
            ram_pressure = smooth_den*(smooth_delta_vr)**2.
            pres_grad = np.gradient(ram_pressure, dx_cm)
            #dPdr = pres_grad[0]*x_hat + pres_grad[1]*y_hat + pres_grad[2]*z_hat
            #dPdtheta = (y*pres_grad[0] - x*pres_grad[1])/(np.sqrt(x*x+y*y))
            dPdphi = (z*x*pres_grad[0] + z*y*pres_grad[1] - (x*x+y*y)*pres_grad[2])/(r*np.sqrt(x*x+y*y))
            ram_force = -1./den_masked * dPdphi
            if (ftypes[i]=='ram'):
                force = ram_force
                force_label = 'Ram Pressure'

        force = np.ma.masked_where((density > cgm_density_max * density_cut_factor), force)
        fig = plt.figure(figsize=(12,10),dpi=500)
        ax = fig.add_subplot(1,1,1)
        f_cmap = copy.copy(mpl.cm.get_cmap('BrBG'))
        f_cmap.set_over(color='w', alpha=1.)
        # Need to rotate to match up with how yt plots it
        im = ax.imshow(rotate(force[len(force)//2,:,:],90), cmap=f_cmap, norm=colors.SymLogNorm(vmin=-1e-5, vmax=1e-5, linthresh=1e-9, base=10), \
                  extent=[-1.5*Rvir,1.5*Rvir,-1.5*Rvir,1.5*Rvir])
        ax.axis([-250,250,-250,250])
        ax.set_xlabel('y [kpc]', fontsize=20)
        ax.set_ylabel('z [kpc]', fontsize=20)
        ax.tick_params(axis='both', which='both', direction='in', length=8, width=2, pad=5, labelsize=20, \
          top=True, right=True)
        cax = fig.add_axes([0.82, 0.11, 0.03, 0.84])
        cax.tick_params(axis='both', which='both', direction='in', length=8, width=2, pad=5, labelsize=20, \
          top=True, right=True)
        fig.colorbar(im, cax=cax, orientation='vertical')
        ax.text(1.2, 0.5, force_label + ' Force [cm/s$^2$]', fontsize=20, rotation='vertical', ha='center', va='center', transform=ax.transAxes)
        plt.subplots_adjust(bottom=0.08, top=0.98, left=0.12, right=0.82)
        plt.savefig(save_dir + snap + '_' + ftypes[i] + '_phi-force_slice_x' + save_suffix + '.png')

    # Delete output from temp directory if on pleiades
    if (args.system=='pleiades_cassi'):
        print('Deleting directory from /tmp')
        shutil.rmtree(snap_dir)

def force_ratio_slice(snap):
    '''Plots a slice of different force terms through the center of the halo. The option --force_type indicates
    what type of force to plot.'''

    masses_ind = np.where(masses['snapshot']==snap)[0]
    Menc_profile = IUS(np.concatenate(([0],masses['radius'][masses_ind])), np.concatenate(([0],masses['total_mass'][masses_ind])))
    Mvir = rvir_masses['total_mass'][rvir_masses['snapshot']==snap]
    Rvir = rvir_masses['radius'][rvir_masses['snapshot']==snap][0]

    if (args.system=='pleiades_cassi'):
        print('Copying directory to /tmp')
        snap_dir = '/tmp/' + args.halo + '/' + args.run + '/' + target_dir + '/' + snap
        if (args.copy_to_tmp):
            shutil.copytree(foggie_dir + run_dir + snap, snap_dir)
            snap_name = snap_dir + '/' + snap
        else:
            # Make a dummy directory with the snap name so the script later knows the process running
            # this snapshot failed if the directory is still there
            os.makedirs(snap_dir)
            snap_name = foggie_dir + run_dir + snap + '/' + snap
    else:
        snap_name = foggie_dir + run_dir + snap + '/' + snap
    ds, refine_box = foggie_load(snap_name, trackname, do_filter_particles=False, halo_c_v_name=halo_c_v_name, gravity=True, masses_dir=masses_dir)

    # Define the density cut between disk and CGM to vary smoothly between 1 and 0.1 between z = 0.5 and z = 0.25,
    # with it being 1 at higher redshifts and 0.1 at lower redshifts
    current_time = ds.current_time.in_units('Myr').v
    if (current_time<=8656.88):
        density_cut_factor = 1.
    elif (current_time<=10787.12):
        density_cut_factor = 1. - 0.9*(current_time-8656.88)/2130.24
    else:
        density_cut_factor = 0.1

    pix_res = float(np.min(refine_box[('gas','dx')].in_units('kpc')))  # at level 11
    lvl1_res = pix_res*2.**11.
    level = 9
    dx = lvl1_res/(2.**level)
    smooth_scale = (25./dx)/6.
    smooth_scale3 = (50./dx)/6.
    smooth_scale4 = (75./dx)/6.
    smooth_scale5 = (100./dx)/6.
    dx_cm = dx*1000*cmtopc
    refine_res = int(3.*Rvir/dx)
    box = ds.covering_grid(level=level, left_edge=ds.halo_center_kpc-ds.arr([1.5*Rvir,1.5*Rvir,1.5*Rvir],'kpc'), dims=[refine_res, refine_res, refine_res])
    density = box['density'].in_units('g/cm**3').v
    temperature = box['temperature'].v
    mass = box['cell_mass'].in_units('g').v
    x = box[('gas','x')].in_units('cm').v - ds.halo_center_kpc[0].to('cm').v
    y = box[('gas','y')].in_units('cm').v - ds.halo_center_kpc[1].to('cm').v
    z = box[('gas','z')].in_units('cm').v - ds.halo_center_kpc[2].to('cm').v
    r = box['radius_corrected'].in_units('cm').v
    x_hat = x/r
    y_hat = y/r
    z_hat = z/r

    # This next block needed for removing any ISM regions and then interpolating over the holes left behind
    # Define ISM regions to remove
    disk_mask = (density > cgm_density_max * density_cut_factor)
    # disk_mask_expanded is a binary mask of both ISM regions AND their surrounding pixels
    struct = ndimage.generate_binary_structure(3,3)
    disk_mask_expanded = ndimage.binary_dilation(disk_mask, structure=struct, iterations=3)
    disk_mask_expanded = ndimage.binary_closing(disk_mask_expanded, structure=struct, iterations=3)
    disk_mask_expanded = disk_mask_expanded | disk_mask
    # disk_edges is a binary mask of ONLY pixels surrounding ISM regions -- nothing inside ISM regions
    disk_edges = disk_mask_expanded & ~disk_mask
    x_edges = x[disk_edges].flatten()
    y_edges = y[disk_edges].flatten()
    z_edges = z[disk_edges].flatten()

    den_edges = density[disk_edges]
    den_interp_func = NearestNDInterpolator(list(zip(x_edges,y_edges,z_edges)), den_edges)
    den_masked = np.copy(density)
    den_masked[disk_mask] = den_interp_func(x[disk_mask], y[disk_mask], z[disk_mask])
    smooth_den = gaussian_filter(den_masked, smooth_scale)

    thermal_pressure = box['pressure'].in_units('erg/cm**3').v
    # Cut to only those values closest to removed ISM regions
    pres_edges = thermal_pressure[disk_edges]
    # Interpolate across removed ISM regions
    pres_interp_func = NearestNDInterpolator(list(zip(x_edges,y_edges,z_edges)), pres_edges)
    pres_masked = np.copy(thermal_pressure)
    # Replace removed ISM regions with interpolated values
    pres_masked[disk_mask] = pres_interp_func(x[disk_mask], y[disk_mask], z[disk_mask])
    pres_grad = np.gradient(thermal_pressure, dx_cm)
    dPdr = pres_grad[0]*x_hat + pres_grad[1]*y_hat + pres_grad[2]*z_hat
    thermal_force = -1./den_masked * dPdr
    if (args.smoothed):
        thermal_force = gaussian_filter(thermal_force, smooth_scale4)
    vx = box['vx_corrected'].in_units('cm/s').v
    vy = box['vy_corrected'].in_units('cm/s').v
    vz = box['vz_corrected'].in_units('cm/s').v
    # Cut to only those values closest to removed ISM regions
    vx_edges = vx[disk_edges]
    vy_edges = vy[disk_edges]
    vz_edges = vz[disk_edges]
    # Interpolate across removed ISM regions
    vx_interp_func = NearestNDInterpolator(list(zip(x_edges,y_edges,z_edges)), vx_edges)
    vy_interp_func = NearestNDInterpolator(list(zip(x_edges,y_edges,z_edges)), vy_edges)
    vz_interp_func = NearestNDInterpolator(list(zip(x_edges,y_edges,z_edges)), vz_edges)
    vx_masked = np.copy(vx)
    vy_masked = np.copy(vy)
    vz_masked = np.copy(vz)
    # Replace removed ISM regions with interpolated values
    vx_masked[disk_mask] = vx_interp_func(x[disk_mask], y[disk_mask], z[disk_mask])
    vy_masked[disk_mask] = vy_interp_func(x[disk_mask], y[disk_mask], z[disk_mask])
    vz_masked[disk_mask] = vz_interp_func(x[disk_mask], y[disk_mask], z[disk_mask])
    # Smooth resulting velocity field -- without contamination from ISM regions
    smooth_vx = gaussian_filter(vx_masked, smooth_scale)
    smooth_vy = gaussian_filter(vy_masked, smooth_scale)
    smooth_vz = gaussian_filter(vz_masked, smooth_scale)
    sig_x = (vx - smooth_vx)**2.
    sig_y = (vy - smooth_vy)**2.
    sig_z = (vz - smooth_vz)**2.
    vdisp = np.sqrt((sig_x + sig_y + sig_z)/3.)
    turb_pressure = smooth_den*vdisp**2.
    pres_grad = np.gradient(turb_pressure, dx_cm)
    dPdr = pres_grad[0]*x_hat + pres_grad[1]*y_hat + pres_grad[2]*z_hat
    turb_force = -1./den_masked * dPdr
    if (args.smoothed):
        turb_force = gaussian_filter(turb_force, smooth_scale4)

    force_ratio = thermal_force/turb_force
    force_ratio = np.ma.masked_where((density > cgm_density_max * density_cut_factor), force_ratio)
    fig = plt.figure(figsize=(12,10),dpi=500)
    ax = fig.add_subplot(1,1,1)
    f_cmap = copy.copy(mpl.cm.get_cmap('Spectral'))
    # Need to rotate to match up with how yt plots it
    im = ax.imshow(rotate(force_ratio[len(force_ratio)//2,:,:],90), cmap=f_cmap, norm=colors.Normalize(vmin=-2, vmax=2), \
              extent=[-1.5*Rvir,1.5*Rvir,-1.5*Rvir,1.5*Rvir])
    ax.axis([-250,250,-250,250])
    ax.set_xlabel('y [kpc]', fontsize=20)
    ax.set_ylabel('z [kpc]', fontsize=20)
    ax.tick_params(axis='both', which='both', direction='in', length=8, width=2, pad=5, labelsize=20, \
      top=True, right=True)
    cax = fig.add_axes([0.82, 0.11, 0.03, 0.84])
    cax.tick_params(axis='both', which='both', direction='in', length=8, width=2, pad=5, labelsize=20, \
      top=True, right=True)
    fig.colorbar(im, cax=cax, orientation='vertical')
    ax.text(1.2, 0.5, 'Thermal to Turbulent Force Ratio', fontsize=20, rotation='vertical', ha='center', va='center', transform=ax.transAxes)
    plt.subplots_adjust(bottom=0.08, top=0.98, left=0.12, right=0.82)
    plt.savefig(save_dir + snap + '_thermal-turbulent-force-ratio_slice_x' + save_suffix + '.png')

    # Delete output from temp directory if on pleiades
    if (args.system=='pleiades_cassi'):
        print('Deleting directory from /tmp')
        shutil.rmtree(snap_dir)

def ion_slice(snap):
    '''Plots a slice of an ion mass given by --ion.'''

    Rvir = rvir_masses['radius'][rvir_masses['snapshot']==snap][0]

    if (args.system=='pleiades_cassi'):
        print('Copying directory to /tmp')
        snap_dir = '/tmp/' + args.halo + '/' + args.run + '/' + target_dir + '/' + snap
        if (args.copy_to_tmp):
            shutil.copytree(foggie_dir + run_dir + snap, snap_dir)
            snap_name = snap_dir + '/' + snap
        else:
            # Make a dummy directory with the snap name so the script later knows the process running
            # this snapshot failed if the directory is still there
            os.makedirs(snap_dir)
            snap_name = foggie_dir + run_dir + snap + '/' + snap
    else:
        snap_name = foggie_dir + run_dir + snap + '/' + snap
    ds, refine_box = foggie_load(snap_name, trackname, do_filter_particles=False, halo_c_v_name=halo_c_v_name)

    abundances = trident.ion_balance.solar_abundance
    trident.add_ion_fields(ds, ions='all', ftype='gas')

    pix_res = float(np.min(refine_box[('gas','dx')].in_units('kpc')))  # at level 11
    lvl1_res = pix_res*2.**11.
    level = 9
    dx = lvl1_res/(2.**level)
    smooth_scale = (25./dx)/6.
    dx_cm = dx*1000*cmtopc
    refine_res = int(3.*Rvir/dx)
    box = ds.covering_grid(level=level, left_edge=ds.halo_center_kpc-ds.arr([1.5*Rvir,1.5*Rvir,1.5*Rvir],'kpc'), dims=[refine_res, refine_res, refine_res])
    ion_mass = box[args.ion + '_mass'].in_units('g').v

    fig = plt.figure(figsize=(12,10),dpi=500)
    ax = fig.add_subplot(1,1,1)
    if (args.ion=='O_p5'):
        f_cmap = copy.copy(mpl.cm.get_cmap('magma'))
        vmin = 1e21
        vmax = 1e33
    if (args.ion=='C_p4'):
        f_cmap = copy.copy(mpl.cm.get_cmap('inferno'))
    if (args.ion=='H_p0'):
        f_cmap = sns.blend_palette(("white", "#ababab", "#565656", "black",
                                      "#4575b4", "#984ea3", "#d73027",
                                      "darkorange", "#ffe34d"), as_cmap=True)
        vmin = 1e27
        vmax = 1e39
    #f_cmap.set_over(color='w', alpha=1.)
    # Need to rotate to match up with how yt plots it
    im = ax.imshow(rotate(ion_mass[len(ion_mass)//2,:,:],90), cmap=f_cmap, norm=colors.LogNorm(vmin=vmin, vmax=vmax), \
              extent=[-1.5*Rvir,1.5*Rvir,-1.5*Rvir,1.5*Rvir])
    ax.axis([-250,250,-250,250])
    ax.set_xlabel('y [kpc]', fontsize=20)
    ax.set_ylabel('z [kpc]', fontsize=20)
    ax.tick_params(axis='both', which='both', direction='in', length=8, width=2, pad=5, labelsize=20, \
      top=True, right=True)
    cax = fig.add_axes([0.82, 0.11, 0.03, 0.84])
    cax.tick_params(axis='both', which='both', direction='in', length=8, width=2, pad=5, labelsize=20, \
      top=True, right=True)
    fig.colorbar(im, cax=cax, orientation='vertical')
    if (args.ion=='O_p5'):
        ax.text(1.2, 0.5, 'O VI Mass [g]', fontsize=20, rotation='vertical', ha='center', va='center', transform=ax.transAxes)
    if (args.ion=='C_p4'):
        ax.text(1.2, 0.5, 'C IV Mass [g]', fontsize=20, rotation='vertical', ha='center', va='center', transform=ax.transAxes)
    if (args.ion=='H_p0'):
        ax.text(1.2, 0.5, 'H I Mass [g]', fontsize=20, rotation='vertical', ha='center', va='center', transform=ax.transAxes)
    plt.subplots_adjust(bottom=0.08, top=0.98, left=0.12, right=0.82)
    plt.savefig(save_dir + snap + '_' + args.ion + '_mass_slice_x' + save_suffix + '.png')

    '''slc = yt.SlicePlot(ds, 'x', 'O_p5_mass', center=ds.halo_center_kpc, width=ds.quan(3.*Rvir, 'kpc'))
    #slc.set_zlim('O_p5_mass', o6_min, o6_max)
    slc.set_cmap('O_p5_mass', o6_color_map)
    #slc.set_unit('O_p5_mass', 'Msun')
    #slc.set_log('O_p5_mass', True)
    slc.save(save_dir + snap + '_OVI_mass_slice_x' + save_suffix + '.png')'''

    # Delete output or dummy directory from temp directory if on pleiades
    if (args.system=='pleiades_cassi'):
        print('Deleting directory from /tmp')
        shutil.rmtree(snap_dir)

def support_slice(snap):
    '''Plots a slice of the ratio of supporting forces to gravity through the center of the halo,
    for only those cells with Fnet ~ 0.
    The option --force_type indicates what type of support to plot.'''

    masses_ind = np.where(masses['snapshot']==snap)[0]
    Menc_profile = IUS(np.concatenate(([0],masses['radius'][masses_ind])), np.concatenate(([0],masses['total_mass'][masses_ind])))
    Mvir = rvir_masses['total_mass'][rvir_masses['snapshot']==snap]
    Rvir = rvir_masses['radius'][rvir_masses['snapshot']==snap][0]

    if (args.system=='pleiades_cassi'):
        print('Copying directory to /tmp')
        snap_dir = '/tmp/' + snap
        shutil.copytree(foggie_dir + run_dir + snap, snap_dir)
        snap_name = snap_dir + '/' + snap
    else:
        snap_name = foggie_dir + run_dir + snap + '/' + snap
    ds, refine_box = foggie_load(snap_name, trackname, do_filter_particles=False, halo_c_v_name=halo_c_v_name, gravity=True, masses_dir=masses_dir)
    ds.add_gradient_fields(('gas','pressure'))

    if (args.force_type=='all'):
        ftypes = ['thermal', 'turbulent', 'outflow', 'inflow', 'rotation']
    elif (',' in args.force_type):
        ftypes = args.force_type.split(',')
    else:
        ftypes = [args.force_type]

    pix_res = float(np.min(refine_box['dx'].in_units('kpc')))  # at level 11
    lvl1_res = pix_res*2.**11.
    level = 9
    dx = lvl1_res/(2.**level)
    dx_cm = dx*1000.*cmtopc
    refine_res = int(3.*Rvir/dx)
    box = ds.covering_grid(level=level, left_edge=ds.halo_center_kpc-ds.arr([1.5*Rvir,1.5*Rvir,1.5*Rvir],'kpc'), dims=[refine_res, refine_res, refine_res])
    density = box['density'].in_units('g/cm**3').v
    temperature = box['temperature'].v
    x_hat = box['x'].in_units('cm').v - ds.halo_center_kpc[0].to('cm').v
    y_hat = box['y'].in_units('cm').v - ds.halo_center_kpc[1].to('cm').v
    z_hat = box['z'].in_units('cm').v - ds.halo_center_kpc[2].to('cm').v
    r = box['radius_corrected'].in_units('cm').v
    x_hat /= r
    y_hat /= r
    z_hat /= r

    grav_force = -G*Menc_profile(r/(1000*cmtopc))*gtoMsun/r**2.
    thermal_pressure = box['pressure'].in_units('erg/cm**3').v
    pres_grad = np.gradient(thermal_pressure, dx_cm)
    dPdr = pres_grad[0]*x_hat + pres_grad[1]*y_hat + pres_grad[2]*z_hat
    thermal_force = 1./density * dPdr
    vx = box['vx_corrected'].in_units('cm/s').v
    vy = box['vy_corrected'].in_units('cm/s').v
    vz = box['vz_corrected'].in_units('cm/s').v
    smooth_vx = uniform_filter(vx, size=20)
    smooth_vy = uniform_filter(vy, size=20)
    smooth_vz = uniform_filter(vz, size=20)
    sig_x = (vx - smooth_vx)**2.
    sig_y = (vy - smooth_vy)**2.
    sig_z = (vz - smooth_vz)**2.
    vdisp = np.sqrt((sig_x + sig_y + sig_z)/3.)
    turb_pressure = density*vdisp**2.
    pres_grad = np.gradient(turb_pressure, dx_cm)
    dPdr = pres_grad[0]*x_hat + pres_grad[1]*y_hat + pres_grad[2]*z_hat
    turb_force = 1./density * dPdr
    vr = box['radial_velocity_corrected'].in_units('cm/s').v
    smooth_vr = uniform_filter(vr, size=20)
    vr_in = 1.0*smooth_vr
    vr_in[smooth_vr > 0.] = 0.
    in_pressure = density*vr_in**2.
    pres_grad = np.gradient(in_pressure, dx_cm)
    dPdr = pres_grad[0]*x_hat + pres_grad[1]*y_hat + pres_grad[2]*z_hat
    in_force = 1./density * dPdr
    vr_out = 1.0*smooth_vr
    vr_out[smooth_vr < 0.] = 0.
    out_pressure = density*vr_out**2.
    pres_grad = np.gradient(out_pressure, dx_cm)
    dPdr = pres_grad[0]*x_hat + pres_grad[1]*y_hat + pres_grad[2]*z_hat
    out_force = 1./density * dPdr
    vtheta = box['theta_velocity_corrected'].in_units('cm/s').v
    vphi = box['phi_velocity_corrected'].in_units('cm/s').v
    smooth_vtheta = uniform_filter(vtheta, size=20)
    smooth_vphi = uniform_filter(vphi, size=20)
    rot_force = (smooth_vtheta**2. + smooth_vphi**2.)/r
    tot_force = grav_force + thermal_force + turb_force + rot_force + in_force + out_force

    for i in range(len(ftypes)):
        if (ftypes[i]=='thermal'):
            support = thermal_force/-grav_force
            support_label = 'Thermal'
        if (ftypes[i]=='turbulent'):
            support = turb_force/-grav_force
            support_label = 'Turbulent'
        if (ftypes[i]=='inflow'):
            support = in_force/-grav_force
            support_label = 'Inflow'
        if (ftypes[i]=='outflow'):
            support = out_force/-grav_force
            support_label = 'Outflow'
        if (ftypes[i]=='rotation'):
            support = rot_force/-grav_force
            support_label = 'Rotation'

        support[(density > cgm_density_max) & (temperature < cgm_temperature_min)] = 6.
        support[(tot_force > 1e-9) | (tot_force < -1e-9)] = 6.
        fig = plt.figure(figsize=(12,10),dpi=500)
        ax = fig.add_subplot(1,1,1)
        cmap = copy.copy(mpl.cm.get_cmap("BrBG"))
        cmap.set_over('lightgray')
        # Need to rotate to match up with how yt plots it
        im = ax.imshow(rotate(support[len(support)//2,:,:],90), cmap=cmap, norm=colors.Normalize(vmin=0, vmax=2), \
                  extent=[-1.5*Rvir,1.5*Rvir,-1.5*Rvir,1.5*Rvir])
        ax.axis([-250,250,-250,250])
        ax.set_xlabel('y [kpc]', fontsize=20)
        ax.set_ylabel('z [kpc]', fontsize=20)
        ax.tick_params(axis='both', which='both', direction='in', length=8, width=2, pad=5, labelsize=20, \
          top=True, right=True)
        cax = fig.add_axes([0.855, 0.08, 0.03, 0.9])
        cax.tick_params(axis='both', which='both', direction='in', length=8, width=2, pad=5, labelsize=20, \
          top=True, right=True)
        fig.colorbar(im, cax=cax, orientation='vertical')
        ax.text(1.15, 0.5, support_label + ' Force / $F_\mathrm{grav}$', fontsize=20, rotation='vertical', ha='center', va='center', transform=ax.transAxes)
        plt.subplots_adjust(bottom=0.08, top=0.98, left=0.08, right=0.88)
        plt.savefig(save_dir + snap + '_' + ftypes[i] + '_support_slice_x' + save_suffix + '.png')

    # Delete output from temp directory if on pleiades
    if (args.system=='pleiades_cassi'):
        print('Deleting directory from /tmp')
        shutil.rmtree(snap_dir)

def velocity_slice(snap):
    '''Plots slices of radial, theta, and phi velocity fields through the center of the halo. The field,
    the smoothed field, and the difference between the field and the smoothed field are all plotted.'''

    masses_ind = np.where(masses['snapshot']==snap)[0]
    Menc_profile = IUS(np.concatenate(([0],masses['radius'][masses_ind])), np.concatenate(([0],masses['total_mass'][masses_ind])))
    Mvir = rvir_masses['total_mass'][rvir_masses['snapshot']==snap]
    Rvir = rvir_masses['radius'][rvir_masses['snapshot']==snap][0]

    if (args.system=='pleiades_cassi'):
        print('Copying directory to /tmp')
        snap_dir = '/tmp/' + args.halo + '/' + args.run + '/' + target_dir + '/' + snap
        if (args.copy_to_tmp):
            shutil.copytree(foggie_dir + run_dir + snap, snap_dir)
            snap_name = snap_dir + '/' + snap
        else:
            snap_dir = '/nobackup/clochhaa/tmp/' + args.halo + '/' + args.run + '/' + target_dir + '/' + snap
            # Make a dummy directory with the snap name so the script later knows the process running
            # this snapshot failed if the directory is still there
            os.makedirs(snap_dir)
            snap_name = foggie_dir + run_dir + snap + '/' + snap
    else:
        snap_name = foggie_dir + run_dir + snap + '/' + snap
    ds, refine_box = foggie_load(snap_name, trackname, do_filter_particles=False, halo_c_v_name=halo_c_v_name, gravity=True, masses_dir=masses_dir)

    # Define the density cut between disk and CGM to vary smoothly between 1 and 0.1 between z = 0.5 and z = 0.25,
    # with it being 1 at higher redshifts and 0.1 at lower redshifts
    current_time = ds.current_time.in_units('Myr').v
    if (current_time<=8656.88):
        density_cut_factor = 1.
    elif (current_time<=10787.12):
        density_cut_factor = 1. - 0.9*(current_time-8656.88)/2130.24
    else:
        density_cut_factor = 0.1

    vtypes = [['vx', 'vy', 'vz'], ['radial_velocity', 'theta_velocity', 'phi_velocity']]
    vlabels = [['$x$', '$y$', '$z$'], ['Radial', '$\\theta$', '$\phi$']]
    vmins = [[-500, -500, -500], [-500, -200, -200]]
    vmaxes = [[500, 500, 500], [500, 200, 200]]
    vfile = ['linear', 'angular']

    pix_res = float(np.min(refine_box[('gas','dx')].in_units('kpc')))  # at level 11
    lvl1_res = pix_res*2.**11.
    level = 9
    dx = lvl1_res/(2.**level)
    smooth_scale = int(25./dx)
    dx_cm = dx*1000*cmtopc
    refine_res = int(3.*Rvir/dx)
    box = ds.covering_grid(level=level, left_edge=ds.halo_center_kpc-ds.arr([1.5*Rvir,1.5*Rvir,1.5*Rvir],'kpc'), dims=[refine_res, refine_res, refine_res])
    density = box['density'].in_units('g/cm**3').v
    temperature = box['temperature'].v

    # This next block needed for removing any ISM regions and then interpolating over the holes left behind
    x = box[('gas','x')].in_units('cm').v - ds.halo_center_kpc[0].to('cm').v
    y = box[('gas','y')].in_units('cm').v - ds.halo_center_kpc[1].to('cm').v
    z = box[('gas','z')].in_units('cm').v - ds.halo_center_kpc[2].to('cm').v
    # Define ISM regions to remove
    disk_mask = (density > cgm_density_max * density_cut_factor)
    # disk_mask_expanded is a binary mask of both ISM regions AND their surrounding pixels
    struct = ndimage.generate_binary_structure(3,3)
    disk_mask_expanded = ndimage.binary_dilation(disk_mask, structure=struct, iterations=3)
    # disk_edges is a binary mask of ONLY pixels surrounding ISM regions -- nothing inside ISM regions
    disk_edges = disk_mask_expanded & ~disk_mask
    x_edges = x[disk_edges].flatten()
    y_edges = y[disk_edges].flatten()
    z_edges = z[disk_edges].flatten()

    for j in range(len(vtypes)):
        fig = plt.figure(num=j+1,figsize=(24,18),dpi=500)
        for i in range(len(vtypes[j])):
            v = box[vtypes[j][i] + '_corrected'].in_units('km/s').v
            # Cut to only those values closest to removed ISM regions
            v_edges = v[disk_edges]
            # Interpolate across removed ISM regions
            v_interp_func = NearestNDInterpolator(list(zip(x_edges,y_edges,z_edges)), v_edges)
            v_masked = np.copy(v)
            # Replace removed ISM regions with interpolated values
            v_masked[disk_mask] = v_interp_func(x[disk_mask], y[disk_mask], z[disk_mask])
            # Smooth resulting velocity field -- without contamination from ISM regions
            smooth_v = gaussian_filter(v_masked, smooth_scale/6.)
            sig_v = v_masked - smooth_v
            ax1 = fig.add_subplot(3,3,3*i+1)
            ax2 = fig.add_subplot(3,3,3*i+2)
            ax3 = fig.add_subplot(3,3,3*i+3)
            v_cmap = mpl.cm.get_cmap('RdBu')
            # Need to rotate to match up with how yt plots it
            im1 = ax1.imshow(rotate(v_masked[len(v)//2,:,:],90), cmap=v_cmap, norm=colors.Normalize(vmin=vmins[j][i], vmax=vmaxes[j][i]), \
                      extent=[-1.5*Rvir,1.5*Rvir,-1.5*Rvir,1.5*Rvir])
            ax1.set_xlabel('y [kpc]', fontsize=20)
            ax1.set_ylabel('z [kpc]', fontsize=20)
            ax1.text(-200,200,'$v_x$',fontsize=20,ha='left',va='center')
            ax1.tick_params(axis='both', which='both', direction='in', length=8, width=2, pad=5, labelsize=20, \
              top=True, right=True)
            cb = fig.colorbar(im1, ax=ax1, orientation='vertical', pad=0)
            cb.ax.tick_params(labelsize=16)
            cb.ax.set_ylabel(vlabels[j][i] + ' Velocity [km/s]', fontsize=16)
            im2 = ax2.imshow(rotate(smooth_v[len(smooth_v)//2,:,:],90), cmap=v_cmap, norm=colors.Normalize(vmin=vmins[j][i], vmax=vmaxes[j][i]), \
                      extent=[-1.5*Rvir,1.5*Rvir,-1.5*Rvir,1.5*Rvir])
            ax2.set_xlabel('y [kpc]', fontsize=20)
            ax2.set_ylabel('z [kpc]', fontsize=20)
            ax2.text(-200,200,'$v_{x,\mathrm{sm}}$',fontsize=20,ha='left',va='center')
            ax2.tick_params(axis='both', which='both', direction='in', length=8, width=2, pad=5, labelsize=20, \
              top=True, right=True)
            cb = fig.colorbar(im2, ax=ax2, orientation='vertical', pad=0)
            cb.ax.tick_params(labelsize=16)
            cb.ax.set_ylabel('Smoothed ' + vlabels[j][i] + ' Velocity [km/s]', fontsize=16)
            im3 = ax3.imshow(rotate(sig_v[len(sig_v)//2,:,:],90), cmap=v_cmap, norm=colors.Normalize(vmin=-200, vmax=200), \
                      extent=[-1.5*Rvir,1.5*Rvir,-1.5*Rvir,1.5*Rvir])
            ax3.set_xlabel('y [kpc]', fontsize=20)
            ax3.set_ylabel('z [kpc]', fontsize=20)
            ax3.text(-200,200,'$v_x - v_{x,\mathrm{sm}}$',fontsize=20,ha='left',va='center')
            ax3.tick_params(axis='both', which='both', direction='in', length=8, width=2, pad=5, labelsize=20, \
              top=True, right=True)
            cb = fig.colorbar(im3, ax=ax3, orientation='vertical', pad=0)
            cb.ax.tick_params(labelsize=16)
            cb.ax.set_ylabel(vlabels[j][i] + ' Velocity - Smoothed Velocity [km/s]', fontsize=16)
        plt.subplots_adjust(bottom=0.05, top=0.97, left=0.04, right=0.97, wspace=0.15, hspace=0.15)
        plt.savefig(save_dir + snap + '_' + vfile[j] + '_velocities_slice_x' + save_suffix + '.png')
        plt.close(fig)

    # Delete output from temp directory if on pleiades
    if (args.system=='pleiades_cassi'):
        print('Deleting directory from /tmp')
        shutil.rmtree(snap_dir)

def vorticity_slice(snap):
    '''Plots a slice of velocity vorticity through the center of the halo.'''

    Rvir = rvir_masses['radius'][rvir_masses['snapshot']==snap][0]

    snap_name = foggie_dir + run_dir + snap + '/' + snap
    ds, refine_box = foggie_load(snap_name, trackname, do_filter_particles=False, halo_c_v_name=halo_c_v_name, gravity=True, masses_dir=masses_dir)

    slc = yt.SlicePlot(ds, 'x', 'vorticity_magnitude', center=ds.halo_center_kpc, width=(Rvir*2, 'kpc'))
    slc.set_zlim('vorticity_magnitude', 1e-17, 1e-13)
    slc.set_cmap('vorticity_magnitude', 'BuGn')
    slc.save(save_dir + snap + '_vorticity_slice_x' + save_suffix + '.png')

def vorticity_direction(snap):
    '''Plots a mass-weighted histogram of vorticity direction in theta-phi space split into cold,
    cool, warm, and hot gas for each thin spherical shell in the dataset given by snapshot 'snap'.'''

    masses_ind = np.where(masses['snapshot']==snap)[0]
    Menc_profile = IUS(np.concatenate(([0],masses['radius'][masses_ind])), np.concatenate(([0],masses['total_mass'][masses_ind])))
    Mvir = rvir_masses['total_mass'][rvir_masses['snapshot']==snap]
    Rvir = rvir_masses['radius'][rvir_masses['snapshot']==snap][0]

    # Copy output to temp directory if on pleiades
    if (args.system=='pleiades_cassi'):
        print('Copying directory to /tmp')
        snap_dir = '/tmp/' + snap
        shutil.copytree(foggie_dir + run_dir + snap, snap_dir)
        snap_name = snap_dir + '/' + snap
    else:
        snap_name = foggie_dir + run_dir + snap + '/' + snap
    ds, refine_box = foggie_load(snap_name, trackname, do_filter_particles=False, halo_c_v_name=halo_c_v_name, gravity=True, masses_dir=masses_dir)

    sphere = ds.sphere(ds.halo_center_kpc, (100., 'kpc'))
    mass = sphere['cell_mass'].in_units('Msun').v
    radius = sphere['radius_corrected'].in_units('kpc').v
    temperature = sphere['temperature'].v
    vort_x = sphere['vorticity_x'].v
    vort_y = sphere['vorticity_y'].v
    vort_z = sphere['vorticity_z'].v
    r = np.sqrt(vort_x*vort_x + vort_y*vort_y + vort_z*vort_z)
    vort_theta = np.arccos(vort_z/r)
    vort_phi = np.arctan2(vort_y, vort_x)
    print('Fields loaded')

    Tcut = [0.,10**4.5,10**5.5,10**6.5,10**9]
    Tlabels = ['Cold gas', 'Cool gas', 'Warm gas', 'Hot gas']
    shells = np.arange(5.,101.,1)
    x_range = [0, np.pi]
    y_range = [-np.pi, np.pi]

    for i in range(len(shells)-1):
        if (i % 10 == 0): print('%d/%d' % (i, len(shells)-1))
        fig = plt.figure(figsize=(8,8))
        vort_theta_shell = vort_theta[(radius >= shells[i]) & (radius < shells[i+1])]
        vort_phi_shell = vort_phi[(radius >= shells[i]) & (radius < shells[i+1])]
        temp_shell = temperature[(radius >= shells[i]) & (radius < shells[i+1])]
        mass_shell = mass[(radius >= shells[i]) & (radius < shells[i+1])]
        for j in range(len(Tcut)-1):
            vort_theta_T = vort_theta_shell[(temp_shell >= Tcut[j]) & (temp_shell < Tcut[j+1])]
            vort_phi_T = vort_phi_shell[(temp_shell >= Tcut[j]) & (temp_shell < Tcut[j+1])]
            mass_T = mass_shell[(temp_shell >= Tcut[j]) & (temp_shell < Tcut[j+1])]
            ax = fig.add_subplot(2,2,j+1)
            hist = ax.hist2d(vort_theta_T, vort_phi_T, weights=mass_T, bins=100, range=[x_range, y_range])
            ax.set_xlabel('$\\theta$', fontsize=18)
            ax.set_ylabel('$\\phi$', fontsize=18)
            ax.set_title(Tlabels[j], fontsize=20)
            ax.tick_params(axis='both', which='both', direction='in', length=8, width=2, pad=5, labelsize=14, \
              top=True, right=True)
            if (j==1): ax.text(np.pi, np.pi+1., '$%.1f \\mathrm{kpc} < r < %.1f \\mathrm{kpc}$' % (shells[i], shells[i+1]), \
                               fontsize=20, ha='right', va='center')
        plt.subplots_adjust(left=0.095, bottom=0.067, right=0.979, top=0.917, wspace=0.248, hspace=0.286)
        plt.savefig(save_dir + '/' + snap + '_vorticity_direction_r%.1f-%.1fkpc' % (shells[i], shells[i+1]) + save_suffix + '.png')

def Pk_turbulence(snap):
    '''Plots a turbulent energy power spectrum for the output given in 'snap'.'''

    if (args.system=='pleiades_cassi'):
        print('Copying directory to /tmp')
        snap_dir = '/tmp/' + snap
        shutil.copytree(foggie_dir + run_dir + snap, snap_dir)
        snap_name = snap_dir + '/' + snap
    else:
        snap_name = foggie_dir + run_dir + snap + '/' + snap
    ds, refine_box = foggie_load(snap_name, trackname, do_filter_particles=False, halo_c_v_name=halo_c_v_name)
    zsnap = ds.get_parameter('CosmologyCurrentRedshift')
    Rvir = rvir_masses['radius'][rvir_masses['snapshot']==snap][0]

    pix_res = float(np.min(refine_box['dx'].in_units('kpc')))  # at level 11
    lvl1_res = pix_res*2.**11.
    level = 9
    dx = lvl1_res/(2.**level)
    refine_res = int(3.*Rvir/dx)
    left_edge = ds.halo_center_kpc - ds.arr([1.5*Rvir,1.5*Rvir,1.5*Rvir],'kpc')
    right_edge = ds.halo_center_kpc + ds.arr([1.5*Rvir,1.5*Rvir,1.5*Rvir],'kpc')
    dims = np.array([refine_res, refine_res, refine_res])
    box = ds.covering_grid(level=level, left_edge=left_edge, dims=dims)

    nx, ny, nz = dims
    nindex_rho = 1./3.
    Kk = np.zeros((nx//2+1, ny//2+1, nz//2+1))

    for vel in ["vx_corrected", "vy_corrected", "vz_corrected"]:
        rho = box['density'].v
        u = box[vel].in_units('cm/s').v

        # do the FFTs -- note that since our data is real, there will be
        # too much information here.  fftn puts the positive freq terms in
        # the first half of the axes -- that's what we keep.  Our
        # normalization has an '8' to account for this clipping to one
        # octant.
        ru = np.fft.fftn(rho**nindex_rho * u)[0:nx//2+1, 0:ny//2+1, 0:nz//2+1]
        ru = 8.0*ru/(nx * ny * nz)
        k_fft = np.abs(ru)**2
        Kk += 0.5 * k_fft

    # wavenumbers
    L = (right_edge - left_edge).v
    kx = np.fft.rfftfreq(nx) * nx/L[0]
    ky = np.fft.rfftfreq(ny) * ny/L[1]
    kz = np.fft.rfftfreq(nz) * nz/L[2]

    # physical limits to the wavenumbers
    kmin = np.min(1./L)
    kmax = np.min(0.5*dims/L)

    kbins = np.arange(kmin, kmax, kmin)
    N = len(kbins)

    # bin the Fourier KE into radial kbins
    kx3d, ky3d, kz3d = np.meshgrid(kx, ky, kz, indexing="ij")
    k = np.sqrt(kx3d**2 + ky3d**2 + kz3d**2)
    whichbin = np.digitize(k.flat, kbins)
    ncount = np.bincount(whichbin)
    E_spectrum = np.zeros(len(ncount) - 1)

    for n in range(1, len(ncount)):
        E_spectrum[n - 1] = np.sum(Kk.flat[whichbin == n])

    k = 0.5 * (kbins[0:N-1] + kbins[1:N])
    l = 1./k
    E_spectrum = E_spectrum[1:N]

    index = np.argmax(E_spectrum)
    kmax = k[index]
    Emax = E_spectrum[index]

    plt.loglog(l, E_spectrum)
    plt.loglog(l, Emax * (k / kmax) ** (-5.0 / 3.0), ls=":", color="0.5")

    #plt.xlabel(r"$k$")
    plt.xlabel('$l$ [kpc]')
    plt.ylabel(r"$E(k)dk$")

    plt.savefig(save_dir + snap + '_turbulent_energy_spectrum' + save_suffix + '.pdf')

def turbulence_compare(snap):
    '''Computes the turbulent pressure several different ways and compares them in a single plot.'''

    if (args.system=='pleiades_cassi'):
        print('Copying directory to /tmp')
        snap_dir = '/tmp/' + snap
        if (args.copy_to_tmp):
            shutil.copytree(foggie_dir + run_dir + snap, snap_dir)
            snap_name = snap_dir + '/' + snap
        else:
            # Make a dummy directory with the snap name so the script later knows the process running
            # this snapshot failed if the directory is still there
            os.makedirs(snap_dir)
            snap_name = foggie_dir + run_dir + snap + '/' + snap
    else:
        snap_name = foggie_dir + run_dir + snap + '/' + snap
    ds, refine_box = foggie_load(snap_name, trackname, do_filter_particles=False, halo_c_v_name=halo_c_v_name, gravity=True, masses_dir=masses_dir)
    zsnap = ds.get_parameter('CosmologyCurrentRedshift')

    masses_ind = np.where(masses['snapshot']==snap)[0]
    Menc_profile = IUS(np.concatenate(([0],masses['radius'][masses_ind])), np.concatenate(([0],masses['total_mass'][masses_ind])))
    Mvir = rvir_masses['total_mass'][rvir_masses['snapshot']==snap]
    Rvir = rvir_masses['radius'][rvir_masses['snapshot']==snap][0]

    radius_list = np.linspace(0., 1.5*Rvir, 100)
    radius_centers = radius_list[:-1] + np.diff(radius_list)

    pix_res = float(np.min(refine_box[('gas','dx')].in_units('kpc')))  # at level 11
    lvl1_res = pix_res*2.**11.
    level = 9
    dx = lvl1_res/(2.**level)
    smooth_scale = (25./dx)/6.
    dx_cm = dx*1000*cmtopc
    refine_res = int(3.*Rvir/dx)
    box = ds.covering_grid(level=level, left_edge=ds.halo_center_kpc-ds.arr([1.5*Rvir,1.5*Rvir,1.5*Rvir],'kpc'), dims=[refine_res, refine_res, refine_res])
    density = box['density'].in_units('g/cm**3').v
    cs = box['sound_speed'].in_units('km/s').v
    thermal = box['pressure'].in_units('erg/cm**3').v
    mass = box['cell_mass'].in_units('Msun').v
    x = box[('gas','x')].in_units('cm').v - ds.halo_center_kpc[0].to('cm').v
    y = box[('gas','y')].in_units('cm').v - ds.halo_center_kpc[1].to('cm').v
    z = box[('gas','z')].in_units('cm').v - ds.halo_center_kpc[2].to('cm').v
    r = box['radius_corrected'].in_units('cm').v
    radius = box['radius_corrected'].in_units('kpc').v
    x_hat = x/r
    y_hat = y/r
    z_hat = z/r

    # This next block needed for removing any ISM regions and then interpolating over the holes left behind
    if (args.cgm_only):
        # Define ISM regions to remove
        disk_mask = (density > cgm_density_max/20.)
        # disk_mask_expanded is a binary mask of both ISM regions AND their surrounding pixels
        struct = ndimage.generate_binary_structure(3,3)
        disk_mask_expanded = ndimage.binary_dilation(disk_mask, structure=struct, iterations=3)
        disk_mask_expanded = ndimage.binary_closing(disk_mask_expanded, structure=struct, iterations=3)
        disk_mask_expanded = disk_mask_expanded | disk_mask
        # disk_edges is a binary mask of ONLY pixels surrounding ISM regions -- nothing inside ISM regions
        disk_edges = disk_mask_expanded & ~disk_mask
        x_edges = x[disk_edges].flatten()
        y_edges = y[disk_edges].flatten()
        z_edges = z[disk_edges].flatten()

    vx = box['vx_corrected'].in_units('cm/s').v
    vy = box['vy_corrected'].in_units('cm/s').v
    vz = box['vz_corrected'].in_units('cm/s').v
    if (args.cgm_only):
        vx_edges = vx[disk_edges]
        vy_edges = vy[disk_edges]
        vz_edges = vz[disk_edges]
        vx_interp_func = LinearNDInterpolator(list(zip(x_edges,y_edges,z_edges)), vx_edges, fill_value=0)
        vy_interp_func = LinearNDInterpolator(list(zip(x_edges,y_edges,z_edges)), vy_edges, fill_value=0)
        vz_interp_func = LinearNDInterpolator(list(zip(x_edges,y_edges,z_edges)), vz_edges, fill_value=0)
        vx_masked = np.copy(vx)
        vy_masked = np.copy(vy)
        vz_masked = np.copy(vz)
        vx_masked[disk_mask] = vx_interp_func(x[disk_mask], y[disk_mask], z[disk_mask])
        vy_masked[disk_mask] = vy_interp_func(x[disk_mask], y[disk_mask], z[disk_mask])
        vz_masked[disk_mask] = vz_interp_func(x[disk_mask], y[disk_mask], z[disk_mask])
        density_edges = density[disk_edges]
        density_interp_func = LinearNDInterpolator(list(zip(x_edges,y_edges,z_edges)), density_edges, fill_value=np.min(density))
        density_masked = np.copy(density)
        density_masked[disk_mask] = density_interp_func(x[disk_mask], y[disk_mask], z[disk_mask])
    else:
        vx_masked = vx
        vy_masked = vy
        vz_masked = vz
        density_masked = density
    smooth_vx = gaussian_filter(vx_masked, smooth_scale)
    smooth_vy = gaussian_filter(vy_masked, smooth_scale)
    smooth_vz = gaussian_filter(vz_masked, smooth_scale)
    smooth_den = gaussian_filter(density_masked, smooth_scale)
    sig_x = (vx_masked - smooth_vx)**2.
    sig_y = (vy_masked - smooth_vy)**2.
    sig_z = (vz_masked - smooth_vz)**2.
    turb_pressure = smooth_den*(sig_x + sig_y + sig_z)/3.
    if (args.cgm_only):
        sig_x = sig_x[(density < cgm_density_max/20.)]
        sig_y = sig_y[(density < cgm_density_max/20.)]
        sig_z = sig_z[(density < cgm_density_max/20.)]
        radius = radius[(density < cgm_density_max/20.)]
        mass = mass[(density < cgm_density_max/20.)]
        cs = cs[(density < cgm_density_max/20.)]
        thermal = thermal[(density < cgm_density_max/20.)]
        turb_pressure = turb_pressure[(density < cgm_density_max/20.)]
        #rv = rv[(density < cgm_density_max/20.)]
        vx = vx[(density < cgm_density_max/20.)]
        vy = vy[(density < cgm_density_max/20.)]
        vz = vz[(density < cgm_density_max/20.)]
        smooth_vx = smooth_vx[(density < cgm_density_max/20.)]
        smooth_vy = smooth_vy[(density < cgm_density_max/20.)]
        smooth_vz = smooth_vz[(density < cgm_density_max/20.)]
        density_cgm = density[(density < cgm_density_max/20.)]
        density = density_cgm

    '''turb_pressure_smoothing = turb_pressure_smoothing[(rv < 200)]
    radius = radius[(rv < 200)]
    mass = mass[(rv < 200)]
    vx = vx[(rv < 200)]
    vy = vy[(rv < 200)]
    vz = vz[(rv < 200)]
    density = density[(rv < 200)]'''

    #turb_pressure_smoothing_med = []
    turb_pressure_smoothing_avg = []
    turb_pressure_shells = []
    avg_v_shells = []
    avg_v_shells_smoothed = []
    avg_cs = []
    avg_thermal_pressure = []
    for i in range(len(radius_list)-1):
        sig_x_shell = sig_x[(radius >= radius_list[i]) & (radius < radius_list[i+1])]
        sig_y_shell = sig_y[(radius >= radius_list[i]) & (radius < radius_list[i+1])]
        sig_z_shell = sig_z[(radius >= radius_list[i]) & (radius < radius_list[i+1])]
        smooth_vx_shell = smooth_vx[(radius >= radius_list[i]) & (radius < radius_list[i+1])]
        smooth_vy_shell = smooth_vy[(radius >= radius_list[i]) & (radius < radius_list[i+1])]
        smooth_vz_shell = smooth_vz[(radius >= radius_list[i]) & (radius < radius_list[i+1])]
        weights_shell = mass[(radius >= radius_list[i]) & (radius < radius_list[i+1])]
        density_shell = density[(radius >= radius_list[i]) & (radius < radius_list[i+1])]
        cs_shell = cs[(radius >= radius_list[i]) & (radius < radius_list[i+1])]
        thermal_shell = thermal[(radius >= radius_list[i]) & (radius < radius_list[i+1])]
        turb_pressure_shell = turb_pressure[(radius >= radius_list[i]) & (radius < radius_list[i+1])]
        if (len(weights_shell)!=0):
            density_avg, density_std = weighted_avg_and_std(density_shell, weights_shell)
            #quantiles = weighted_quantile(turb_shell, weights_shell, np.array([0.25,0.5,0.75]))
            #turb_pressure_smoothing_med.append(quantiles[1])
            avg_x, std_x = weighted_avg_and_std(sig_x_shell, weights_shell)
            avg_y, std_y = weighted_avg_and_std(sig_y_shell, weights_shell)
            avg_z, std_z = weighted_avg_and_std(sig_z_shell, weights_shell)
            avg_turb, std_turb = weighted_avg_and_std(turb_pressure_shell, weights_shell)
            vdisp = np.sqrt((avg_x + avg_y + avg_z)/3.)
            turb_pressure_smoothing_avg.append(avg_turb)
            avg_v_shells_smoothed.append(np.sqrt(weighted_avg_and_std(smooth_vx_shell, weights_shell)[0]**2. + weighted_avg_and_std(smooth_vy_shell, weights_shell)[0]**2. + weighted_avg_and_std(smooth_vz_shell, weights_shell)[0]**2.))
            avg_cs.append(weighted_avg_and_std(cs_shell, weights_shell)[0])
            avg_thermal_pressure.append(weighted_avg_and_std(thermal_shell, weights_shell)[0])
        else:
            turb_pressure_smoothing_avg.append(0.)
            avg_cs.append(0.)
            avg_thermal_pressure.append(0.)
            #turb_pressure_smoothing_med.append(0.)

        vx_shell = vx[(radius >= radius_list[i]) & (radius < radius_list[i+1])]
        vy_shell = vy[(radius >= radius_list[i]) & (radius < radius_list[i+1])]
        vz_shell = vz[(radius >= radius_list[i]) & (radius < radius_list[i+1])]
        if (len(weights_shell)!=0):
            vx_avg, vx_disp = weighted_avg_and_std(vx_shell, weights_shell)
            vy_avg, vy_disp = weighted_avg_and_std(vy_shell, weights_shell)
            vz_avg, vz_disp = weighted_avg_and_std(vz_shell, weights_shell)
            v_avg = np.sqrt((vx_avg**2. + vy_avg**2. + vz_avg**2.))
            vdisp = np.sqrt((vx_disp**2. + vy_disp**2. + vz_disp**2.)/3.)
            turb_pressure_shells.append(density_avg*vdisp**2.)
            avg_v_shells.append(v_avg)
        else:
            turb_pressure_shells.append(0.)

    #turb_pressure_smoothing_med = np.array(turb_pressure_smoothing_med)
    turb_pressure_smoothing_avg = np.array(turb_pressure_smoothing_avg)
    turb_pressure_shells = np.array(turb_pressure_shells)
    avg_cs = np.array(avg_cs)
    avg_thermal_pressure = np.array(avg_thermal_pressure)
    avg_v_shells = np.array(avg_v_shells)
    avg_v_shells_smoothed = np.array(avg_v_shells_smoothed)

    fig = plt.figure(figsize=(8,6), dpi=500)
    ax = fig.add_subplot(1,1,1)

    ax.plot(radius_centers, np.log10(turb_pressure_smoothing_avg), ls='-', color='k', \
            lw=2, label='Cell-centered smoothing')
    ax.plot(radius_centers, np.log10(turb_pressure_shells), ls='--', color='b', \
            lw=2, label='Shells')
    #ax.plot(radius_centers, avg_cs, ls=':', color='g', \
            #lw=2, label='Sound speed')
    ax.plot(radius_centers, np.log10(avg_thermal_pressure), ls=':', color='r', \
            lw=2, label='Thermal pressure')

    ax.set_ylabel('log Turbulent Pressure [erg/cm$^3$]', fontsize=18)
    #ax.set_ylabel('Velocity Dispersion [km/s]', fontsize=18)
    ax.set_xlabel('Radius [kpc]', fontsize=18)
    ax.axis([0,250,-18,-8])
    ax.text(240, -8.5, '$z=%.2f$' % (zsnap), fontsize=18, ha='right', va='center')
    ax.text(240,-9,halo_dict[args.halo],ha='right',va='center',fontsize=18)
    #ax.text(Rvir-3., -8.5, '$R_{200}$', fontsize=18, ha='right', va='center')
    #ax.axis([0,250,0,200])
    #ax.text(240, 190, '$z=%.2f$' % (zsnap), fontsize=18, ha='right', va='center')
    #ax.text(240,180,halo_dict[args.halo],ha='right',va='center',fontsize=18)
    #ax.text(Rvir-3., 190, '$R_{200}$', fontsize=18, ha='right', va='center')
    ax.tick_params(axis='both', which='both', direction='in', length=8, width=2, pad=5, labelsize=18, \
      top=True, right=True)
    #ax.plot([Rvir, Rvir], [ax.get_ylim()[0], ax.get_ylim()[1]], 'k--', lw=1)
    if (args.halo=='8508'): ax.legend(loc=9, frameon=False, fontsize=14)
    plt.subplots_adjust(top=0.94,bottom=0.11,right=0.95,left=0.15)
    plt.savefig(save_dir + snap + '_turbulent-pressure-compare' + save_suffix + '.png')
    plt.close()

    # Delete output or dummy directory from temp directory if on pleiades
    if (args.system=='pleiades_cassi'):
        print('Deleting directory from /tmp')
        shutil.rmtree(snap_dir)

def turbulence_visualization(snap):
    '''Opens a napari viewer to view turbulence in 3D.'''

    snap_name = foggie_dir + run_dir + snap + '/' + snap
    ds, refine_box = foggie_load(snap_name, trackname, do_filter_particles=False, halo_c_v_name=halo_c_v_name, gravity=True, masses_dir=masses_dir)
    zsnap = ds.get_parameter('CosmologyCurrentRedshift')

    masses_ind = np.where(masses['snapshot']==snap)[0]
    Menc_profile = IUS(np.concatenate(([0],masses['radius'][masses_ind])), np.concatenate(([0],masses['total_mass'][masses_ind])))
    Mvir = rvir_masses['total_mass'][rvir_masses['snapshot']==snap]
    Rvir = rvir_masses['radius'][rvir_masses['snapshot']==snap][0]
    # Define the density cut between disk and CGM to vary smoothly between 1 and 0.1 between z = 0.5 and z = 0.25,
    # with it being 1 at higher redshifts and 0.1 at lower redshifts
    current_time = ds.current_time.in_units('Myr').v
    if (current_time<=8656.88):
        density_cut_factor = 1.
    elif (current_time<=10787.12):
        density_cut_factor = 1. - 0.9*(current_time-8656.88)/2130.24
    else:
        density_cut_factor = 0.1
    print(current_time, density_cut_factor)

    print('Making covering grid', str(datetime.datetime.now()))
    pix_res = float(np.min(refine_box[('gas','dx')].in_units('kpc')))  # at level 11
    lvl1_res = pix_res*2.**11.
    level = 9
    dx = lvl1_res/(2.**level)
    smooth_scale= (25./dx)/6.
    dx_cm = dx*1000*cmtopc
    refine_res = int(3.*Rvir/dx)
    box = ds.covering_grid(level=level, left_edge=ds.halo_center_kpc-ds.arr([1.5*Rvir,1.5*Rvir,1.5*Rvir],'kpc'), dims=[refine_res, refine_res, refine_res])
    print('Loading arrays in grid', str(datetime.datetime.now()))
    density = box['density'].in_units('g/cm**3').v
    thermal = box['pressure'].in_units('erg/cm**3').v
    x = box[('gas','x')].in_units('cm').v - ds.halo_center_kpc[0].to('cm').v
    y = box[('gas','y')].in_units('cm').v - ds.halo_center_kpc[1].to('cm').v
    z = box[('gas','z')].in_units('cm').v - ds.halo_center_kpc[2].to('cm').v
    r = box['radius_corrected'].in_units('cm').v
    x_hat = x/r
    y_hat = y/r
    z_hat = z/r

    # This next block needed for removing any ISM regions and then interpolating over the holes left behind
    # Define ISM regions to remove
    print('Making mask', str(datetime.datetime.now()))
    disk_mask = (density > density_cut_factor * cgm_density_max)
    # disk_mask_expanded is a binary mask of both ISM regions AND their surrounding pixels
    print('Generating binary structure', str(datetime.datetime.now()))
    struct = ndimage.generate_binary_structure(3,3)
    print('Expanding mask', str(datetime.datetime.now()))
    disk_mask_expanded = ndimage.binary_dilation(disk_mask, structure=struct, iterations=3)
    print('Filling holes', str(datetime.datetime.now()))
    disk_mask_expanded = ndimage.binary_closing(disk_mask_expanded, structure=struct, iterations=3)
    print('Adding in disk', str(datetime.datetime.now()))
    disk_mask_expanded = disk_mask_expanded | disk_mask
    # disk_edges is a binary mask of ONLY pixels surrounding ISM regions -- nothing inside ISM regions
    print('Defining edges', str(datetime.datetime.now()))
    disk_edges = disk_mask_expanded & ~disk_mask
    print('Saving x,y,z coords of edges', str(datetime.datetime.now()))
    x_edges = x[disk_edges].flatten()
    y_edges = y[disk_edges].flatten()
    z_edges = z[disk_edges].flatten()

    '''print('Loading vx,vy,vz arrays', str(datetime.datetime.now()))
    vx = box['vx_corrected'].in_units('cm/s').v
    vy = box['vy_corrected'].in_units('cm/s').v
    vz = box['vz_corrected'].in_units('cm/s').v
    print('Cutting to edges', str(datetime.datetime.now()))
    vx_edges = vx[disk_edges]
    vy_edges = vy[disk_edges]
    vz_edges = vz[disk_edges]'''
    density_edges = density[disk_edges]
    #thermal_edges = thermal[disk_edges]
    print('Making interpolators', str(datetime.datetime.now()))
    '''vx_interp_func = LinearNDInterpolator(list(zip(x_edges,y_edges,z_edges)), vx_edges, fill_value=0)
    vy_interp_func = LinearNDInterpolator(list(zip(x_edges,y_edges,z_edges)), vy_edges, fill_value=0)
    vz_interp_func = LinearNDInterpolator(list(zip(x_edges,y_edges,z_edges)), vz_edges, fill_value=0)'''
    density_interp_func = NearestNDInterpolator(list(zip(x_edges,y_edges,z_edges)), density_edges)
    #thermal_interp_func = LinearNDInterpolator(list(zip(x_edges,y_edges,z_edges)), thermal_edges, fill_value=np.min(thermal))
    print('Copying arrays', str(datetime.datetime.now()))
    '''vx_masked = np.copy(vx)
    vy_masked = np.copy(vy)
    vz_masked = np.copy(vz)'''
    density_masked = np.copy(density)
    #thermal_masked = np.copy(thermal)
    print('Using interpolator to fill in holes', str(datetime.datetime.now()))
    '''vx_masked[disk_mask] = vx_interp_func(x[disk_mask], y[disk_mask], z[disk_mask])
    vy_masked[disk_mask] = vy_interp_func(x[disk_mask], y[disk_mask], z[disk_mask])
    vz_masked[disk_mask] = vz_interp_func(x[disk_mask], y[disk_mask], z[disk_mask])'''
    density_masked[disk_mask] = density_interp_func(x[disk_mask], y[disk_mask], z[disk_mask])
    '''thermal_masked[disk_mask] = thermal_interp_func(x[disk_mask], y[disk_mask], z[disk_mask])
    pres_grad = np.gradient(thermal_masked, dx_cm)
    dPdr = pres_grad[0]*x_hat + pres_grad[1]*y_hat + pres_grad[2]*z_hat
    thermal_force = -1./density_masked * dPdr
    thermal_pos = thermal_force[thermal_force > 0.]
    thermal_pos = np.log10(thermal_pos) + 9
    thermal_pos[thermal_pos < 0.] = 0.
    thermal_neg = thermal_force[thermal_force < 0.]
    thermal_neg = np.log10(-thermal_neg) + 9
    thermal_neg[thermal_neg < 0.] = 0.
    thermal_neg = -thermal_neg
    thermal_force[thermal_force > 0.] = thermal_pos
    thermal_force[thermal_force < 0.] = thermal_neg'''

    print('Smoothing fields', str(datetime.datetime.now()))
    '''smooth_vx = gaussian_filter(vx_masked, smooth_scale)
    smooth_vy = gaussian_filter(vy_masked, smooth_scale)
    smooth_vz = gaussian_filter(vz_masked, smooth_scale)'''
    smooth_den = gaussian_filter(density_masked, smooth_scale)
    '''print('Calculating velocity dispersions', str(datetime.datetime.now()))
    sig_x_masked = (vx_masked - smooth_vx)**2.
    sig_y_masked = (vy_masked - smooth_vy)**2.
    sig_z_masked = (vz_masked - smooth_vz)**2.
    vdisp_masked = np.sqrt((sig_x_masked + sig_y_masked + sig_z_masked)/3.)
    turb_pressure_masked = smooth_den*vdisp_masked**2.
    pres_grad = np.gradient(turb_pressure_masked, dx_cm)
    dPdr = pres_grad[0]*x_hat + pres_grad[1]*y_hat + pres_grad[2]*z_hat
    turb_force = -1./density_masked * dPdr
    turb_pos = turb_force[turb_force > 0.]
    turb_pos = np.log10(turb_pos) + 9
    turb_pos[turb_pos < 0.] = 0.
    turb_neg = turb_force[turb_force < 0.]
    turb_neg = np.log10(-turb_neg) + 9
    turb_neg[turb_neg < 0.] = 0.
    turb_neg = -turb_neg
    turb_force[turb_force > 0.] = turb_pos
    turb_force[turb_force < 0.] = turb_neg'''

    from vispy.color.colormap import MatplotlibColormap
    force_cmap = MatplotlibColormap('BrBG')

    import napari
    print('Starting up viewer', str(datetime.datetime.now()))
    #viewer = napari.view_image(vdisp/1e5, name='velocity dispersion', colormap='viridis', contrast_limits=[0,400])
    #viewer = napari.view_image(thermal_force, name='thermal force', colormap=force_cmap, contrast_limits=[-5,5])
    #turb_force_layer = viewer.add_image(turb_force, name='turbulent force', colormap=force_cmap, contrast_limits=[-5,5])
    #thermal_pressure_layer = viewer.add_image(np.log10(thermal), name='thermal pressure', colormap='magma', contrast_limits=[-18,-10])
    #thermal_pressure_masked_layer = viewer.add_image(np.log10(thermal_masked), name='masked thermal pressure', colormap='magma', contrast_limits=[-18,-10])
    viewer = napari.view_image(np.log10(density), name='density', colormap='viridis', contrast_limits=[-30,-20])
    density_masked_layer = viewer.add_image(np.log10(density_masked), name='masked density', colormap='viridis', contrast_limits=[-30,-20])
    density_masked_layer = viewer.add_image(np.log10(smooth_den), name='masked and smoothed density', colormap='viridis', contrast_limits=[-30,-20])
    napari.run()


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
    if ('feedback' in args.run):
        foggie_dir = '/nobackup/clochhaa/'
        run_dir = args.run + '/'

    # Set directory for output location, making it if necessary
    save_dir = output_dir + 'pressures_halo_00' + args.halo + '/' + args.run + '/'
    if not (os.path.exists(save_dir)): os.system('mkdir -p ' + save_dir)

    print('foggie_dir: ', foggie_dir)
    halo_c_v_name = code_path + 'halo_infos/00' + args.halo + '/' + args.run + '/halo_c_v'
    masses_dir = code_path + 'halo_infos/00' + args.halo + '/' + args.run + '/'
    masses = Table.read(masses_dir + 'masses_z-less-2.hdf5', path='all_data')
    rvir_masses = Table.read(masses_dir + 'rvir_masses.hdf5', path='all_data')

    outs = make_output_list(args.output, output_step=args.output_step)

    if ('shade' in args.plot):
        force_ratio_discrete_cmap = mpl.cm.get_cmap('Spectral', 9)
        force_ratio_color_key = collections.OrderedDict()
        force_ratio_min = -4
        force_ratio_max = 4
        force_ratio_color_labels = [b'low1', b'low2', b'med', b'med1', b'med2',
                              b'high1', b'high2', b'high3', b'vhi1']
        for i in np.arange(np.size(force_ratio_color_labels)):
            force_ratio_color_key[force_ratio_color_labels[i]] = to_hex(force_ratio_discrete_cmap(i))

    if (not args.filename) and ((args.plot=='pressures_vs_radius') or (args.plot=='support_vs_radius')):
        sys.exit("You must specify a filename where the data you want to plot is saved.")

    if (args.save_suffix): save_suffix = '_' + args.save_suffix
    else: save_suffix = ''

    if (len(outs)>1) and ('time' not in args.plot) and ('energy' not in args.plot):
        save_dir += 'Movie_frames/'

    if (args.plot=='pressure_vs_radius'):
        target_dir = 'pressures_vs_radius'
        if (args.nproc==1):
            for i in range(len(outs)):
                pressures_vs_radius(outs[i])
        else:
            target = pressures_vs_radius
    elif (args.plot=='pressure_vs_time'):
        pressures_vs_time(outs)
    elif (args.plot=='force_vs_radius'):
        target_dir = 'forces_vs_radius'
        if (args.nproc==1):
            for i in range(len(outs)):
                forces_vs_radius(outs[i])
        else:
            target = forces_vs_radius
    elif (args.plot=='force_vs_radius_time_averaged'):
        forces_vs_radius_time_averaged(outs)
    elif (args.plot=='force_vs_radius_pres'):
        if (args.nproc==1):
            for i in range(len(outs)):
                forces_vs_radius_from_med_pressures(outs[i])
        else:
            target = forces_vs_radius_from_med_pressures
    elif (args.plot=='force_vs_time'):
        forces_vs_time(outs)
    elif (args.plot=='force_vs_energy_output'):
        forces_vs_energy_output(outs)
    elif (args.plot=='support_vs_time'):
        support_vs_time(outs)
    elif (args.plot=='work_vs_time'):
        work_vs_time(outs)
    elif (args.plot=='support_vs_radius'):
        if (args.nproc==1):
            for i in range(len(outs)):
                support_vs_radius(outs[i])
        else:
            target = support_vs_radius
    elif (args.plot=='support_vs_radius_time_averaged'):
        support_vs_radius_time_averaged(outs)
    elif (args.plot=='velocity_PDF'):
        if (args.nproc==1):
            for i in range(len(outs)):
                velocity_PDF(outs[i])
        else:
            target = velocity_PDF
    elif (args.plot=='metallicity_PDF'):
        metallicity_PDF(outs)
    elif (args.plot=='pressure_vs_r_shaded') or (args.plot=='pressure_vs_rv_shaded'):
        if (args.nproc==1):
            for i in range(len(outs)):
                pressure_vs_r_rv_shaded(outs[i])
        else:
            target = pressure_vs_r_rv_shaded
    elif (args.plot=='support_vs_r_shaded') or (args.plot=='support_vs_rv_shaded'):
        if (args.nproc==1):
            for i in range(len(outs)):
                support_vs_r_rv_shaded(outs[i])
        else:
            target = support_vs_r_rv_shaded
    elif (args.plot=='force_vs_r_shaded') or (args.plot=='force_vs_rv_shaded'):
        if (args.nproc==1):
            for i in range(len(outs)):
                force_vs_r_rv_shaded(outs[i])
        else:
            target = force_vs_r_rv_shaded
    elif (args.plot=='shader_force_colored'):
        if (args.nproc==1):
            for i in range(len(outs)):
                shader_force_colored(outs[i])
        else:
            target = shader_force_colored
    elif (args.plot=='pressure_slice'):
        target_dir = 'pressure_slice'
        if (args.nproc==1):
            for i in range(len(outs)):
                pressure_slice(outs[i])
        else:
            target = pressure_slice
    elif (args.plot=='support_slice'):
        if (args.nproc==1):
            for i in range(len(outs)):
                support_slice(outs[i])
        else:
            target = support_slice
    elif (args.plot=='velocity_slice'):
        target_dir = 'velocity_slice'
        if (args.nproc==1):
            for i in range(len(outs)):
                velocity_slice(outs[i])
        else:
            target = velocity_slice
    elif (args.plot=='vorticity_slice'):
        if (args.nproc==1):
            for i in range(len(outs)):
                vorticity_slice(outs[i])
        else:
            target = vorticity_slice
    elif (args.plot=='force_slice'):
        target_dir = 'force_slice'
        if (args.nproc==1):
            for i in range(len(outs)):
                force_slice(outs[i])
        else:
            target = force_slice
    elif (args.plot=='tangential_force_slice'):
        target_dir = 'tangential_force_slice'
        if (args.nproc==1):
            for i in range(len(outs)):
                tangential_force_slice(outs[i])
        else:
            target = tangential_force_slice
    elif (args.plot=='force_ratio_slice'):
        target_dir = 'force_ratio_slice'
        if (args.nproc==1):
            for i in range(len(outs)):
                force_ratio_slice(outs[i])
        else:
            target = force_ratio_slice
    elif (args.plot=='ion_slice'):
        target_dir = 'ion_slice'
        if (args.nproc==1):
            for i in range(len(outs)):
                ion_slice(outs[i])
        else:
            target = ion_slice
    elif (args.plot=='vorticity_direction'):
        if (args.nproc==1):
            for i in range(len(outs)):
                vorticity_direction(outs[i])
        else:
            target = vorticity_direction
    elif (args.plot=='force_rays'):
        if (args.nproc==1):
            for i in range(len(outs)):
                force_rays(outs[i])
        else:
            target = force_rays
    elif (args.plot=='turbulent_spectrum'):
        if (args.nproc==1):
            for i in range(len(outs)):
                Pk_turbulence(outs[i])
        else:
            target = Pk_turbulence
    elif (args.plot=='turbulence_compare'):
        if (args.nproc==1):
            for i in range(len(outs)):
                turbulence_compare(outs[i])
        else:
            target = turbulence_compare
    elif (args.plot=='visualization'):
        for i in range(len(outs)):
            turbulence_visualization(outs[i])
    else:
        sys.exit("That plot type hasn't been implemented!")

    if (args.nproc!=1):
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
                    threads.append(multi.Process(target=target, args=[snap]))
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
                threads.append(multi.Process(target=target, args=[snap]))
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
