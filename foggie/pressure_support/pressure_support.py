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
import scipy.ndimage as ndimage
from scipy.interpolate import LinearNDInterpolator
from astropy.convolution import Gaussian1DKernel
from astropy.convolution import Gaussian2DKernel
from astropy.convolution import CustomKernel
from astropy.convolution import interpolate_replace_nans
from astropy.convolution import convolve_fft
import copy
import matplotlib.colors as colors

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
                        'pressure_vs_time       -  pressures (thermal, turb, ram) at a specified radius over time\n' + \
                        'pressure_vs_radius     -  pressures (thermal, turb, ram) over radius\n' + \
                        'force_vs_radius        -  forces (thermal, turb, ram, rotation, gravity, total) over radius\n' + \
                        'force_vs_radius_pres   -  forces (thermal, turb, ram) over radius, calculated from gradient of median pressure\n' + \
                        'force_vs_time          -  forces (thermal, turb, ram, rotation, gravity, total) over time\n' + \
                        'pressure_vs_r_shaded   -  pressure over radius or radial velocity for each cell as a datashader plot\n' + \
                        '       or                 For these options, specify what type of pressure to plot with the --pressure_type keyword\n' + \
                        'pressure_vs_rv_shaded     and what you want to color-code the points by with the --shader_color keyword\n' + \
                        'force_vs_r_shaded      -  force over radius or radial velocity for each cell as a datashader plot\n' + \
                        '       or                 For these options, specify what force to plot with the --force_type keyword\n' + \
                        'force_vs_rv_shaded        and what you want to color-code the points by with the --shader_color keyword\n' + \
                        'support_vs_time        -  pressure support (thermal, turb, ram) relative to gravity at a specified radius over time\n' + \
                        'support_vs_radius      -  pressure support (thermal, turb, ram) relative to gravity over radius\n' + \
                        'support_vs_r_shaded    -  pressure support relative to gravity over radius or radial velocity for each cell as a datashader plot\n' + \
                        '       or                 For these options, specify what type of support to plot with the --pressure_type keyword\n' + \
                        'support_vs_rv_shaded      and what you want to color-code the points by with the --shader_color keyword\n' + \
                        'pressure_slice         -  x-slices of different types of pressure (specify with --pressure_type keyword)\n' + \
                        'force_slice            -  x-slices of different forces (specify with --force_type keyword)\n' + \
                        'support_slice          -  x-slices of different types of pressure support (specify with --pressure_type keyword)\n' + \
                        'velocity_slice         -  x-slices of the three spherical components of velocity, comparing the velocity,\n' + \
                        '                          the smoothed velocity, and the difference between the velocity and the smoothed velocity\n' + \
                        'vorticity_slice        -  x-slice of the velocity vorticity magnitude\n' + \
                        'vorticity_direction    -  2D histograms of vorticity direction split by temperature and radius\n' + \
                        'force_rays             -  Line plots of forces along rays from the halo center to various points' + \
                        'turbulent_spectrum    -  Turbulent energy power spectrum')

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

    parser.add_argument('--shader_color', metavar='shader_color', type=str, action='store', \
                        help='If plotting support_vs_r_shaded, what field do you want to color-code the points by?\n' + \
                        'Options are "temperature" and "metallicity" and the default is "temperature".')
    parser.set_defaults(shader_color='temperature')

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
        if (args.system=='pleiades_cassi') and (foggie_dir!='/nobackupp18/mpeeples/'):
            print('Copying directory to /tmp')
            snap_dir = '/tmp/' + snap
            shutil.copytree(foggie_dir + run_dir + snap, snap_dir)
            snap_name = snap_dir + '/' + snap
        else:
            snap_name = foggie_dir + run_dir + snap + '/' + snap
        ds, refine_box = foggie_load(snap_name, trackname, do_filter_particles=False, halo_c_v_name=halo_c_v_name, gravity=True, masses_dir=masses_dir)
        zsnap = ds.get_parameter('CosmologyCurrentRedshift')

        pix_res = float(np.min(refine_box['gas','dx'].in_units('kpc')))  # at level 11
        lvl1_res = pix_res*2.**11.
        level = 9
        dx = lvl1_res/(2.**level)
        smooth_scale = int(25./dx)
        refine_res = int(3.*Rvir/dx)
        box = ds.covering_grid(level=level, left_edge=ds.halo_center_kpc-ds.arr([1.5*Rvir,1.5*Rvir,1.5*Rvir],'kpc'), dims=[refine_res, refine_res, refine_res])
        density = box['density'].in_units('g/cm**3').v.flatten()
        temperature = box['temperature'].v.flatten()
        radius = box['radius_corrected'].in_units('kpc').v.flatten()
        if (args.cgm_only): radius = radius[(density < cgm_density_max) & (temperature > cgm_temperature_min)]
        if (args.weight=='mass'):
            weights = box['cell_mass'].in_units('Msun').v.flatten()
        if (args.weight=='volume'):
            weights = box['cell_volume'].in_units('kpc**3').v.flatten()
        if (args.cgm_only): weights = weights[(density < cgm_density_max) & (temperature > cgm_temperature_min)]
        if (args.region_filter=='metallicity'):
            metallicity = box['metallicity'].in_units('Zsun').v.flatten()
            if (args.cgm_only): metallicity = metallicity[(density < cgm_density_max) & (temperature > cgm_temperature_min)]
        if (args.region_filter=='velocity'):
            rv = box['radial_velocity_corrected'].in_units('km/s').v.flatten()
            rv = rv[(density < cgm_density_max) & (temperature > cgm_temperature_min)]
            vff = box['vff'].in_units('km/s').v.flatten()
            vesc = box['vesc'].in_units('km/s').v.flatten()
            if (args.cgm_only):
                vff = vff[(density < cgm_density_max) & (temperature > cgm_temperature_min)]
                vesc = vesc[(density < cgm_density_max) & (temperature > cgm_temperature_min)]

        thermal_pressure = box['pressure'].in_units('erg/cm**3').v.flatten()
        vx = box['vx_corrected'].in_units('cm/s').v
        vy = box['vy_corrected'].in_units('cm/s').v
        vz = box['vz_corrected'].in_units('cm/s').v
        smooth_vx = uniform_filter(vx, size=smooth_scale)
        smooth_vy = uniform_filter(vy, size=smooth_scale)
        smooth_vz = uniform_filter(vz, size=smooth_scale)
        sig_x = (vx - smooth_vx)**2.
        sig_y = (vy - smooth_vy)**2.
        sig_z = (vz - smooth_vz)**2.
        vdisp = np.sqrt((sig_x + sig_y + sig_z)/3.).flatten()
        turb_pressure = (density*vdisp**2.)
        vr = box['radial_velocity_corrected'].in_units('cm/s').v
        smooth_vr = uniform_filter(vr, size=smooth_scale).flatten()
        ram_pressure = (density*smooth_vr**2.)

        pressures = [thermal_pressure, turb_pressure, ram_pressure]
        if (args.cgm_only):
            for i in range(len(pressures)):
                pressures[i] = pressures[i][(density < cgm_density_max) & (temperature > cgm_temperature_min)]
            new_density = density[(density < cgm_density_max) & (temperature > cgm_temperature_min)]
            new_temp = temperature[(density < cgm_density_max) & (temperature > cgm_temperature_min)]
            density = new_density
            temperature = new_temp

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
            weights_regions.append(weights[temperature < 10**5])
            weights_regions.append(weights[(temperature > 10**5) & (temperature < 10**6)])
            weights_regions.append(weights[temperature > 10**6])
            radius_regions.append(radius[temperature < 10**5])
            radius_regions.append(radius[(temperature > 10**5) & (temperature < 10**6)])
            radius_regions.append(radius[temperature > 10**6])
            for i in range(len(pressures)):
                pressure_regions[i].append(pressures[i][temperature < 10**5])
                pressure_regions[i].append(pressures[i][(temperature > 10**5) & (temperature < 10**6)])
                pressure_regions[i].append(pressures[i][temperature > 10**6])
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
        if (args.system=='pleiades_cassi') and (foggie_dir!='/nobackupp18/mpeeples/'):
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
        if (args.system=='pleiades_cassi') and (foggie_dir!='/nobackupp18/mpeeples/') and (args.copy_to_tmp):
            print('Copying directory to /tmp')
            snap_dir = '/tmp/' + snap
            shutil.copytree(foggie_dir + run_dir + snap, snap_dir)
            snap_name = snap_dir + '/' + snap
        else:
            snap_name = foggie_dir + run_dir + snap + '/' + snap
        ds, refine_box = foggie_load(snap_name, trackname, do_filter_particles=False, halo_c_v_name=halo_c_v_name, gravity=True, masses_dir=masses_dir)
        zsnap = ds.get_parameter('CosmologyCurrentRedshift')

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
            disk_mask = (density > cgm_density_max) & (temperature < cgm_temperature_min)
            # disk_mask_expanded is a binary mask of both ISM regions AND their surrounding pixels
            struct = ndimage.generate_binary_structure(3,3)
            disk_mask_expanded = ndimage.binary_dilation(disk_mask, structure=struct, iterations=3)
            disk_mask_expanded = ndimage.binary_closing(disk_mask_expanded, structure=struct, iterations=3)
            # disk_edges is a binary mask of ONLY pixels surrounding ISM regions -- nothing inside ISM regions
            disk_edges = disk_mask_expanded & ~disk_mask
            x_edges = x[disk_edges].flatten()
            y_edges = y[disk_edges].flatten()
            z_edges = z[disk_edges].flatten()

        thermal_pressure = box['pressure'].in_units('erg/cm**3').v
        if (args.cgm_only):
            pres_edges = thermal_pressure[disk_edges]
            pres_interp_func = LinearNDInterpolator(list(zip(x_edges,y_edges,z_edges)), pres_edges)
            pres_masked = np.copy(thermal_pressure)
            pres_masked[disk_mask] = pres_interp_func(x[disk_mask], y[disk_mask], z[disk_mask])
        else:
            pres_masked = thermal_pressure
        pres_grad = np.gradient(pres_masked, dx_cm)
        dPdr = pres_grad[0]*x_hat + pres_grad[1]*y_hat + pres_grad[2]*z_hat
        thermal_force = -1./density * dPdr
        vx = box['vx_corrected'].in_units('cm/s').v
        vy = box['vy_corrected'].in_units('cm/s').v
        vz = box['vz_corrected'].in_units('cm/s').v
        if (args.cgm_only):
            vx_edges = vx[disk_edges]
            vy_edges = vy[disk_edges]
            vz_edges = vz[disk_edges]
            vx_interp_func = LinearNDInterpolator(list(zip(x_edges,y_edges,z_edges)), vx_edges)
            vy_interp_func = LinearNDInterpolator(list(zip(x_edges,y_edges,z_edges)), vy_edges)
            vz_interp_func = LinearNDInterpolator(list(zip(x_edges,y_edges,z_edges)), vz_edges)
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
        sig_x = (vx_masked - smooth_vx)**2.
        sig_y = (vy_masked - smooth_vy)**2.
        sig_z = (vz_masked - smooth_vz)**2.
        vdisp = np.sqrt((sig_x + sig_y + sig_z)/3.)
        turb_pressure = density*vdisp**2.
        pres_grad = np.gradient(turb_pressure, dx_cm)
        dPdr = pres_grad[0]*x_hat + pres_grad[1]*y_hat + pres_grad[2]*z_hat
        turb_force = -1./density * dPdr
        vr = box['radial_velocity_corrected'].in_units('cm/s').v
        if (args.cgm_only):
            vr_edges = vr[disk_edges]
            vr_interp_func = LinearNDInterpolator(list(zip(x_edges,y_edges,z_edges)), vr_edges)
            vr_masked = np.copy(vr)
            vr_masked[disk_mask] = vr_interp_func(x[disk_mask], y[disk_mask], z[disk_mask])
        else:
            vr_masked = vr
        vr_masked = gaussian_filter(vr_masked, smooth_scale)
        dvr = np.gradient(vr_masked, dx_cm)
        delta_vr = dvr[0]*dx_cm*x_hat + dvr[1]*dx_cm*y_hat + dvr[2]*dx_cm*z_hat
        ram_pressure = density*(delta_vr)**2.
        pres_grad = np.gradient(ram_pressure, dx_cm)
        dPdr = pres_grad[0]*x_hat + pres_grad[1]*y_hat + pres_grad[2]*z_hat
        ram_force = -1./density * dPdr
        vtheta = box['theta_velocity_corrected'].in_units('cm/s').v
        vphi = box['phi_velocity_corrected'].in_units('cm/s').v
        if (args.cgm_only):
            vtheta_edges = vtheta[disk_edges]
            vphi_edges = vphi[disk_edges]
            vtheta_interp_func = LinearNDInterpolator(list(zip(x_edges,y_edges,z_edges)), vtheta_edges)
            vtheta_masked = np.copy(vtheta)
            vphi_interp_func = LinearNDInterpolator(list(zip(x_edges,y_edges,z_edges)), vphi_edges)
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
        if (args.cgm_only):
            grav_edges = grav_force[disk_edges]
            grav_interp_func = LinearNDInterpolator(list(zip(x_edges,y_edges,z_edges)), grav_edges)
            grav_masked = np.copy(grav_force)
            grav_masked[disk_mask] = grav_interp_func(x[disk_mask], y[disk_mask], z[disk_mask])
        else:
            grav_masked = grav_force
        grav_force = grav_masked
        tot_force = thermal_force + turb_force + rot_force + ram_force + grav_force
        forces = [thermal_force, turb_force, ram_force, rot_force, grav_force, tot_force]

        density = density.flatten()
        temperature = temperature.flatten()
        radius = radius.flatten()
        for i in range(len(forces)):
            forces[i] = forces[i].flatten()

        if (args.cgm_only): radius = radius[(density < cgm_density_max) & (temperature > cgm_temperature_min)]
        if (args.weight=='mass'):
            weights = box['cell_mass'].in_units('g').v.flatten()
        if (args.weight=='volume'):
            weights = box['cell_volume'].in_units('kpc**3').v.flatten()
        if (args.cgm_only): weights = weights[(density < cgm_density_max) & (temperature > cgm_temperature_min)]
        if (args.region_filter=='metallicity'):
            metallicity = box['metallicity'].in_units('Zsun').v.flatten()
            if (args.cgm_only): metallicity = metallicity[(density < cgm_density_max) & (temperature > cgm_temperature_min)]
        if (args.region_filter=='velocity'):
            rv = box['radial_velocity_corrected'].in_units('km/s').v.flatten()
            vff = box['vff'].in_units('km/s').v.flatten()
            vesc = box['vesc'].in_units('km/s').v.flatten()
            if (args.cgm_only):
                rv = rv[(density < cgm_density_max) & (temperature > cgm_temperature_min)]
                vff = vff[(density < cgm_density_max) & (temperature > cgm_temperature_min)]
                vesc = vesc[(density < cgm_density_max) & (temperature > cgm_temperature_min)]

        if (args.cgm_only):
            for i in range(len(forces)):
                forces[i] = forces[i][(density < cgm_density_max) & (temperature > cgm_temperature_min)]
            new_density = density[(density < cgm_density_max) & (temperature > cgm_temperature_min)]
            new_temp = temperature[(density < cgm_density_max) & (temperature > cgm_temperature_min)]
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
        # Delete output from temp directory if on pleiades
        if (args.system=='pleiades_cassi') and (foggie_dir!='/nobackupp18/mpeeples/') and (args.copy_to_tmp):
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
        ax.set_ylabel('Force on CGM gas [cm/s$^2$]', fontsize=18)
        ax.axis([0,250,-1e-5,1e-5])
        ax.set_yscale('symlog', linthreshy=1e-9)
        ax.text(15, -3e-6, '$z=%.2f$' % (zsnap), fontsize=18, ha='left', va='center')
        #ax.text(15,-3e-6,halo_dict[args.halo],ha='left',va='center',fontsize=18)
        ax.text(Rvir-3., -3e-6, '$R_{200}$', fontsize=18, ha='right', va='center')
    else:
        ax.set_ylabel('Force on CGM gas [$M_\odot$ cm/s$^2$]', fontsize=18)
        ax.axis([0,250,-10,100])
        ax.set_yscale('symlog', linthreshy=1e-2)
        ax.text(15, -3, '$z=%.2f$' % (zsnap), fontsize=18, ha='left', va='center')
        #ax.text(15,-3e-6,halo_dict[args.halo],ha='left',va='center',fontsize=18)
        ax.text(Rvir-3., -3, '$R_{200}$', fontsize=18, ha='right', va='center')
    ax.set_xlabel('Radius [kpc]', fontsize=18)
    ax.tick_params(axis='both', which='both', direction='in', length=8, width=2, pad=5, labelsize=18, \
      top=True, right=True)
    ax.plot([ax.get_xlim()[0], ax.get_xlim()[1]], [0,0], 'k-', lw=1)
    ax.plot([Rvir, Rvir], [ax.get_ylim()[0], ax.get_ylim()[1]], 'k--', lw=1)
    if (args.halo=='8508'): ax.legend(loc=1, frameon=False, fontsize=14)
    fig.subplots_adjust(top=0.94,bottom=0.11,right=0.95,left=0.15)
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
            label_regions = ['Low ' + args.region_filter, 'Mid ' + args.region_filter, 'High ' + args.region_filter]
            if (i==0):
                label_regions_bigplot = ['Low ' + args.region_filter, 'Mid ' + args.region_filter, 'High ' + args.region_filter]
            else:
                label_regions_bigplot = ['__nolegend__', '__nolegend__', '__nolegend__']
            for j in range(len(regions)):
                #if (label=='Total'):
                    #ax1.plot(radius_list, (stats[regions[j] + args.region_filter + '_total_force_sum']-stats[regions[j] + args.region_filter + '_ram_force_sum'])/ \
                      #stats[regions[j] + args.region_filter + '_' + file_labels[i] + '_weight_sum'], \
                      #ls=linestyles[i], color=plot_colors[i], lw=2, alpha=alphas[j], label=label_regions_bigplot[j])
                    #axs2[j].plot(radius_list, (stats[regions[j] + args.region_filter + '_total_force_sum']-stats[regions[j] + args.region_filter + '_ram_force_sum'])/ \
                      #stats[regions[j] + args.region_filter + '_' + file_labels[i] + '_weight_sum'], \
                      #ls=linestyles[i], color=plot_colors[i], lw=2, alpha=alphas[j], label=label)
                    #ax3.plot(radius_list, (stats[regions[j] + args.region_filter + '_total_force_sum']-stats[regions[j] + args.region_filter + '_ram_force_sum'])/ \
                      #stats[regions[j] + args.region_filter + '_' + file_labels[i] + '_weight_sum'], \
                      #ls=linestyles[i], color=plot_colors[i], lw=2, alpha=alphas[j], label=label_regions[j])
                #else:
                if (args.normalized):
                    ax1.plot(radius_list, stats[regions[j] + args.region_filter + '_' + file_labels[i] + '_sum']/ \
                      stats[regions[j] + args.region_filter + '_' + file_labels[i] + '_weight_sum'], \
                      ls=linestyles[i], color=plot_colors[i], lw=2, alpha=alphas[j], label=label_regions_bigplot[j])
                    axs2[j].plot(radius_list, stats[regions[j] + args.region_filter + '_' + file_labels[i] + '_sum']/ \
                      stats[regions[j] + args.region_filter + '_' + file_labels[i] + '_weight_sum'], \
                      ls=linestyles[i], color=plot_colors[i], lw=2, alpha=alphas[j], label=label)
                    ax3.plot(radius_list, stats[regions[j] + args.region_filter + '_' + file_labels[i] + '_sum']/ \
                      stats[regions[j] + args.region_filter + '_' + file_labels[i] + '_weight_sum'], \
                      ls=linestyles[i], color=plot_colors[i], lw=2, alpha=alphas[j], label=label_regions[j])
                else:
                    ax1.plot(radius_list, stats[regions[j] + args.region_filter + '_' + file_labels[i] + '_sum'], \
                      ls=linestyles[i], color=plot_colors[i], lw=2, alpha=alphas[j], label=label_regions_bigplot[j])
                    axs2[j].plot(radius_list, stats[regions[j] + args.region_filter + '_' + file_labels[i] + '_sum'], \
                      ls=linestyles[i], color=plot_colors[i], lw=2, alpha=alphas[j], label=label)
                    ax3.plot(radius_list, stats[regions[j] + args.region_filter + '_' + file_labels[i] + '_sum'], \
                      ls=linestyles[i], color=plot_colors[i], lw=2, alpha=alphas[j], label=label_regions[j])
            #if (label=='Total'):
                #ax1.plot(radius_list, (stats['high_' + args.region_filter + '_total_force_sum']-stats['high_' + args.region_filter + '_ram_force_sum'])/ \
                  #stats['high_' + args.region_filter + '_' + file_labels[i] + '_weight_sum'], \
                  #ls=linestyles[i], color=plot_colors[i], lw=2, label=label)
                #ax3.plot(radius_list, (stats['high_' + args.region_filter + '_total_force_sum']-stats['high_' + args.region_filter + '_ram_force_sum'])/ \
                  #stats['high_' + args.region_filter + '_' + file_labels[i] + '_weight_sum'], \
                  #ls=linestyles[i], color=plot_colors[i], lw=2, label=label)
            #else:
            if (args.normalized):
                ax1.plot(radius_list, stats['high_' + args.region_filter + '_' + file_labels[i] + '_sum']/ \
                  stats['high_' + args.region_filter + '_' + file_labels[i] + '_weight_sum'], \
                  ls=linestyles[i], color=plot_colors[i], lw=2, label=label)
                ax3.plot(radius_list, stats['high_' + args.region_filter + '_' + file_labels[i] + '_sum']/ \
                  stats['high_' + args.region_filter + '_' + file_labels[i] + '_weight_sum'], \
                  ls=linestyles[i], color=plot_colors[i], lw=2, label=label)
            else:
                ax1.plot(radius_list, stats['high_' + args.region_filter + '_' + file_labels[i] + '_sum'], \
                  ls=linestyles[i], color=plot_colors[i], lw=2, label=label)
                ax3.plot(radius_list, stats['high_' + args.region_filter + '_' + file_labels[i] + '_sum'], \
                  ls=linestyles[i], color=plot_colors[i], lw=2, label=label)

            if (args.normalized):
                ax3.set_ylabel('Net Force on Shell [cm/s$^2$]', fontsize=18)
                ax3.axis([0,250,-1e-5,1e-5])
                ax3.set_yscale('symlog', linthreshy=1e-8)
            else:
                ax3.set_ylabel('Net Force on Shell [$M_\odot$ cm/s$^2$]', fontsize=18)
                ax3.axis([0,250,-10,100])
                ax3.set_yscale('symlog', linthreshy=1e-2)
            ax3.set_xlabel('Radius [kpc]', fontsize=18)
            ax3.tick_params(axis='both', which='both', direction='in', length=8, width=2, pad=5, labelsize=18, \
              top=True, right=True)
            ax3.plot([Rvir, Rvir], [ax.get_ylim()[0], ax.get_ylim()[1]], 'k--', lw=1)
            ax3.legend(loc=1, frameon=False, fontsize=14)
            fig3.subplots_adjust(top=0.94,bottom=0.11,right=0.95,left=0.17)
            fig3.savefig(save_dir + snap + '_' + file_labels[i] + '_vs_r_regions-' + args.region_filter + save_suffix + '.png')
            plt.close(fig3)

        for r in range(len(regions)):
            if (args.normalized):
                axs2[r].set_ylabel('Net Force on Shell [cm/s$^2$]', fontsize=18)
                axs2[r].axis([0,250,-1e-5,1e-5])
                axs2[r].set_yscale('symlog', linthreshy=1e-8)
            else:
                axs2[r].set_ylabel('Net Force on Shell [$M_\odot$ cm/s$^2$]', fontsize=18)
                axs2[r].axis([0,250,-10,100])
                axs2[r].set_yscale('symlog', linthreshy=1e-2)
            axs2[r].set_xlabel('Radius [kpc]', fontsize=18)
            axs2[r].tick_params(axis='both', which='both', direction='in', length=8, width=2, pad=5, labelsize=18, \
              top=True, right=True)
            axs2[r].plot([Rvir, Rvir], [ax.get_ylim()[0], ax.get_ylim()[1]], 'k--', lw=1)
            axs2[r].legend(loc=1, frameon=False, fontsize=14)
            figs2[r].subplots_adjust(top=0.94,bottom=0.11,right=0.95,left=0.17)
            figs2[r].savefig(save_dir + snap + '_forces_vs_r_' + regions[r] + args.region_filter + save_suffix + '.png')
            plt.close(figs2[r])

        if (args.normalized):
            ax1.set_ylabel('Net Force on Shell [cm/s$^2$]', fontsize=18)
            ax1.axis([0,250,-1e-5,1e-5])
            ax1.set_yscale('symlog', linthreshy=1e-8)
            ax1.text(15, -1e-6, '$z=%.2f$' % (zsnap), fontsize=18, ha='left', va='center')
            #ax1.text(15,-3e-6,halo_dict[args.halo],ha='left',va='center',fontsize=18)
            ax1.text(Rvir-3., -3e-6, '$R_{200}$', fontsize=18, ha='right', va='center')
        else:
            ax1.set_ylabel('Net Force on Shell [$M_\odot$ cm/s$^2$]', fontsize=18)
            ax1.axis([0,250,-10,100])
            ax1.set_yscale('symlog', linthreshy=1e-2)
            ax1.text(15, -3, '$z=%.2f$' % (zsnap), fontsize=18, ha='left', va='center')
            ax1.text(Rvir-3., -3, '$R_{200}$', fontsize=18, ha='right', va='center')
        ax1.set_xlabel('Radius [kpc]', fontsize=18)
        ax1.tick_params(axis='both', which='both', direction='in', length=8, width=2, pad=5, labelsize=18, \
          top=True, right=True)
        ax1.plot([Rvir, Rvir], [ax.get_ylim()[0], ax.get_ylim()[1]], 'k--', lw=1)
        if (args.halo=='8508'): ax1.legend(loc=1, frameon=False, fontsize=14)
        fig1.subplots_adjust(top=0.94,bottom=0.11,right=0.95,left=0.17)
        fig1.savefig(save_dir + snap + '_all_forces_vs_r_regions-' + args.region_filter + save_suffix + '.png')
        plt.close(fig1)

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
    den_stats = Table.read(tablename_prefix + snap + '_stats_pressure_density_velocity_sphere_mass-weighted_cgm-filtered.hdf5', path='all_data')
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
        dPdr = np.diff(pressure)/np.diff(radius_list_cm)
        force = -1./den_profile(radius_list[:-1])*dPdr
        ax.plot(radius_list[:-1], force, ls=linestyles[i], color=plot_colors[i], \
                lw=2, label=label)

    ax.set_ylabel('Force on CGM gas [cm/s$^2$]', fontsize=18)
    ax.axis([0,250,-1e-5,1e-5])
    ax.set_yscale('symlog', linthreshy=1e-9)
    ax.text(15, -3e-6, '$z=%.2f$' % (zsnap), fontsize=18, ha='left', va='center')
    ax.text(Rvir-3., -3e-6, '$R_{200}$', fontsize=18, ha='right', va='center')
    ax.set_xlabel('Radius [kpc]', fontsize=18)
    ax.tick_params(axis='both', which='both', direction='in', length=8, width=2, pad=5, labelsize=18, \
      top=True, right=True)
    ax.plot([ax.get_xlim()[0], ax.get_xlim()[1]], [0,0], 'k-', lw=1)
    ax.plot([Rvir, Rvir], [ax.get_ylim()[0], ax.get_ylim()[1]], 'k--', lw=1)
    ax.legend(loc=1, frameon=False, fontsize=14)
    fig.subplots_adjust(top=0.94,bottom=0.11,right=0.95,left=0.15)
    plt.savefig(save_dir + snap + '_force_from_med_pressure_vs_r' + save_suffix + '.png')
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

    plot_colors = ['r', 'g', 'b', 'gold', 'k']
    plot_labels = ['Thermal', 'Turbulent', 'Rotation', 'Gravity', 'Total']
    file_labels = ['thermal_force', 'turbulent_force', 'rotation_force', 'gravity_force', 'total_force']
    linestyles = ['-', '--', '-.', '--', '-']
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
                if (file_labels[j]=='total_force'):
                    if (args.normalized):
                        forces_list[j].append(np.sum(stats[file_labels[j] + '_sum'][rad_in:rad_out]-stats['ram_force_sum'][rad_in:rad_out])/np.sum(stats[file_labels[j] + '_weight_sum'][rad_in:rad_out]))
                        #forces_list[j].append(np.sum(stats[file_labels[j] + '_sum'][rad_in:rad_out])/np.sum(stats[file_labels[j] + '_weight_sum'][rad_in:rad_out]))
                    else:
                        #forces_list[j].append(np.sum(stats[file_labels[j] + '_sum'][rad_in:rad_out]-stats['ram_force_sum'][rad_in:rad_out])/gtoMsun)
                        forces_list[j].append(np.sum(stats[file_labels[j] + '_sum'][rad_in:rad_out])/gtoMsun)
                else:
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
                    if (file_labels[j]=='total_force'):
                        if (args.normalized):
                            forces_regions[0][j].append(np.sum(stats['low_' + args.region_filter + '_' + file_labels[j] + '_sum'][rad_in:rad_out]-stats['low_' + args.region_filter + '_ram_force_sum'][rad_in:rad_out]) / \
                              np.sum(stats['low_' + args.region_filter + '_' + file_labels[j] + '_weight_sum'][rad_in:rad_out]))
                            forces_regions[1][j].append(np.sum(stats['mid_' + args.region_filter + '_' + file_labels[j] + '_sum'][rad_in:rad_out]-stats['mid_' + args.region_filter + '_ram_force_sum'][rad_in:rad_out]) / \
                              np.sum(stats['mid_' + args.region_filter + '_' + file_labels[j] + '_weight_sum'][rad_in:rad_out]))
                            forces_regions[2][j].append(np.sum(stats['high_' + args.region_filter + '_' + file_labels[j] + '_sum'][rad_in:rad_out]-stats['high_' + args.region_filter + '_ram_force_sum'][rad_in:rad_out]) / \
                              np.sum(stats['high_' + args.region_filter + '_' + file_labels[j] + '_weight_sum'][rad_in:rad_out]))
                            #forces_regions[0][j].append(np.sum(stats['low_' + args.region_filter + '_' + file_labels[j] + '_sum'][rad_in:rad_out]) / \
                              #np.sum(stats['low_' + args.region_filter + '_' + file_labels[j] + '_weight_sum'][rad_in:rad_out]))
                            #forces_regions[1][j].append(np.sum(stats['mid_' + args.region_filter + '_' + file_labels[j] + '_sum'][rad_in:rad_out]) / \
                              #np.sum(stats['mid_' + args.region_filter + '_' + file_labels[j] + '_weight_sum'][rad_in:rad_out]))
                            #forces_regions[2][j].append(np.sum(stats['high_' + args.region_filter + '_' + file_labels[j] + '_sum'][rad_in:rad_out]) / \
                              #np.sum(stats['high_' + args.region_filter + '_' + file_labels[j] + '_weight_sum'][rad_in:rad_out]))
                        else:
                            #forces_regions[0][j].append(np.sum(stats['low_' + args.region_filter + '_' + file_labels[j] + '_sum'][rad_in:rad_out]-stats['low_' + args.region_filter + '_ram_force_sum'][rad_in:rad_out])/gtoMsun)
                            #forces_regions[1][j].append(np.sum(stats['mid_' + args.region_filter + '_' + file_labels[j] + '_sum'][rad_in:rad_out]-stats['mid_' + args.region_filter + '_ram_force_sum'][rad_in:rad_out])/gtoMsun)
                            #forces_regions[2][j].append(np.sum(stats['high_' + args.region_filter + '_' + file_labels[j] + '_sum'][rad_in:rad_out]-stats['high_' + args.region_filter + '_ram_force_sum'][rad_in:rad_out])/gtoMsun)
                            forces_regions[0][j].append(np.sum(stats['low_' + args.region_filter + '_' + file_labels[j] + '_sum'][rad_in:rad_out])/gtoMsun)
                            forces_regions[1][j].append(np.sum(stats['mid_' + args.region_filter + '_' + file_labels[j] + '_sum'][rad_in:rad_out])/gtoMsun)
                            forces_regions[2][j].append(np.sum(stats['high_' + args.region_filter + '_' + file_labels[j] + '_sum'][rad_in:rad_out])/gtoMsun)
                    else:
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
        ax.set_yscale('symlog', linthreshy=1e-9)
        ax.text(13,-2e-8, halo_dict[args.halo], ha='right', va='center', fontsize=14)
        ax.set_ylabel('log Net Force [cm/s$^2$]', fontsize=14)
        if (args.radius_range!='none'):
            ax.text(13,-1e-7, '$r=%.2f-%.2f R_{200}$' % (radius_range[0], radius_range[1]), ha='right', va='center', fontsize=14)
        else:
            ax.text(13,-1e-7, '$r=%.2f R_{200}$' % (args.radius), ha='right', va='center', fontsize=14)
    else:
        ax.axis([np.min(timelist), np.max(timelist), -1e4,1e4])
        ax.set_yscale('symlog', linthreshy=1e1)
        ax.text(13,-1, halo_dict[args.halo], ha='right', va='center', fontsize=14)
        ax.set_ylabel('log Net Force [$M_\odot$ cm/s$^2$]', fontsize=14)
        if (args.radius_range!='none'):
            ax.text(13,-3, '$r=%.2f-%.2f R_{200}$' % (radius_range[0], radius_range[1]), ha='right', va='center', fontsize=14)
        else:
            ax.text(13,-3, '$r=%.2f R_{200}$' % (args.radius), ha='right', va='center', fontsize=14)
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
                ax.set_yscale('symlog', linthreshy=1e-9)
                ax.set_ylabel('log Net Force [cm/s$^2$]', fontsize=14)
                if (args.radius_range!='none'):
                    ax.text(13,-1e-7, '$r=%.2f-%.2f R_{200}$' % (radius_range[0], radius_range[1]), ha='right', va='center', fontsize=14)
                else:
                    ax.text(13,-1e-7, '$r=%.2f R_{200}$' % (args.radius), ha='right', va='center', fontsize=14)
                ax.text(13,-2e-8, region_label[i], fontsize=14, ha='right', va='center')
            else:
                ax.axis([np.min(timelist), np.max(timelist), -1e4,1e4])
                ax.set_yscale('symlog', linthreshy=1e1)
                ax.set_ylabel('log Net Force [$M_\odot$ cm/s$^2$]', fontsize=14)
                if (args.radius_range!='none'):
                    ax.text(13,-3, '$r=%.2f-%.2f R_{200}$' % (radius_range[0], radius_range[1]), ha='right', va='center', fontsize=14)
                else:
                    ax.text(13,-3, '$r=%.2f R_{200}$' % (args.radius), ha='right', va='center', fontsize=14)
                ax.text(13,-1, region_label[i], fontsize=14, ha='right', va='center')

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
                axs_regions[j].set_yscale('symlog', linthreshy=1e-9)
                axs_regions[j].set_ylabel('log Net Force [cm/s$^2$]', fontsize=14)
                if (args.radius_range!='none'):
                    axs_regions[j].text(13,-1e-7, '$r=%.2f-%.2f R_{200}$' % (radius_range[0], radius_range[1]), ha='right', va='center', fontsize=14)
                else:
                    axs_regions[j].text(13,-1e-7, '$r=%.2f R_{200}$' % (args.radius), ha='right', va='center', fontsize=14)
                axs_regions[j].text(13,-2e-8, plot_labels[j], fontsize=14, ha='right', va='center')
            else:
                axs_regions[j].axis([np.min(timelist), np.max(timelist), -1e4,1e4])
                axs_regions[j].set_yscale('symlog', linthreshy=1e1)
                axs_regions[j].set_ylabel('log Net Force [$M_\odot$ cm/s$^2$]', fontsize=14)
                if (args.radius_range!='none'):
                    axs_regions[j].text(13,-3, '$r=%.2f-%.2f R_{200}$' % (radius_range[0], radius_range[1]), ha='right', va='center', fontsize=14)
                else:
                    axs_regions[j].text(13,-3, '$r=%.2f R_{200}$' % (args.radius), ha='right', va='center', fontsize=14)
                axs_regions[j].text(13,-1, plot_labels[j], fontsize=14, ha='right', va='center')

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
        shutil.copytree(foggie_dir + run_dir + snap, snap_dir)
        snap_name = snap_dir + '/' + snap
    else:
        snap_name = foggie_dir + run_dir + snap + '/' + snap
    ds, refine_box = foggie_load(snap_name, trackname, do_filter_particles=False, halo_c_v_name=halo_c_v_name, gravity=True, masses_dir=masses_dir)
    zsnap = ds.get_parameter('CosmologyCurrentRedshift')

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
    disk_mask = (density > cgm_density_max) & (temperature < cgm_temperature_min)
    # disk_mask_expanded is a binary mask of both ISM regions AND their surrounding pixels
    struct = ndimage.generate_binary_structure(3,3)
    disk_mask_expanded = ndimage.binary_dilation(disk_mask, structure=struct, iterations=3)
    disk_mask_expanded = ndimage.binary_closing(disk_mask_expanded, structure=struct, iterations=3)
    # disk_edges is a binary mask of ONLY pixels surrounding ISM regions -- nothing inside ISM regions
    disk_edges = disk_mask_expanded & ~disk_mask
    x_edges = x[disk_edges].flatten()
    y_edges = y[disk_edges].flatten()
    z_edges = z[disk_edges].flatten()

    thermal_pressure = box['pressure'].in_units('erg/cm**3').v
    pres_edges = thermal_pressure[disk_edges]
    pres_interp_func = LinearNDInterpolator(list(zip(x_edges,y_edges,z_edges)), pres_edges)
    pres_masked = np.copy(thermal_pressure)
    pres_masked[disk_mask] = pres_interp_func(x[disk_mask], y[disk_mask], z[disk_mask])
    pres_grad = np.gradient(pres_masked, dx_cm)
    dPdr = pres_grad[0]*x_hat + pres_grad[1]*y_hat + pres_grad[2]*z_hat
    thermal_force = -1./density * dPdr
    vx = box['vx_corrected'].in_units('cm/s').v
    vy = box['vy_corrected'].in_units('cm/s').v
    vz = box['vz_corrected'].in_units('cm/s').v
    vx_edges = vx[disk_edges]
    vy_edges = vy[disk_edges]
    vz_edges = vz[disk_edges]
    vx_interp_func = LinearNDInterpolator(list(zip(x_edges,y_edges,z_edges)), vx_edges)
    vy_interp_func = LinearNDInterpolator(list(zip(x_edges,y_edges,z_edges)), vy_edges)
    vz_interp_func = LinearNDInterpolator(list(zip(x_edges,y_edges,z_edges)), vz_edges)
    vx_masked = np.copy(vx)
    vy_masked = np.copy(vy)
    vz_masked = np.copy(vz)
    vx_masked[disk_mask] = vx_interp_func(x[disk_mask], y[disk_mask], z[disk_mask])
    vy_masked[disk_mask] = vy_interp_func(x[disk_mask], y[disk_mask], z[disk_mask])
    vz_masked[disk_mask] = vz_interp_func(x[disk_mask], y[disk_mask], z[disk_mask])
    smooth_vx = gaussian_filter(vx_masked, smooth_scale)
    smooth_vy = gaussian_filter(vy_masked, smooth_scale)
    smooth_vz = gaussian_filter(vz_masked, smooth_scale)
    sig_x = (vx_masked - smooth_vx)**2.
    sig_y = (vy_masked - smooth_vy)**2.
    sig_z = (vz_masked - smooth_vz)**2.
    vdisp = np.sqrt((sig_x + sig_y + sig_z)/3.)
    turb_pressure = density*vdisp**2.
    pres_grad = np.gradient(turb_pressure, dx_cm)
    dPdr = pres_grad[0]*x_hat + pres_grad[1]*y_hat + pres_grad[2]*z_hat
    turb_force = -1./density * dPdr
    vr = box['radial_velocity_corrected'].in_units('cm/s').v
    vr_edges = vr[disk_edges]
    vr_interp_func = LinearNDInterpolator(list(zip(x_edges,y_edges,z_edges)), vr_edges)
    vr_masked = np.copy(vr)
    vr_masked[disk_mask] = vr_interp_func(x[disk_mask], y[disk_mask], z[disk_mask])
    vr_masked = gaussian_filter(vr_masked, smooth_scale)
    dvr = np.gradient(vr_masked, dx_cm)
    delta_vr = dvr[0]*dx_cm*x_hat + dvr[1]*dx_cm*y_hat + dvr[2]*dx_cm*z_hat
    ram_pressure = density*(delta_vr)**2.
    pres_grad = np.gradient(ram_pressure, dx_cm)
    dPdr = pres_grad[0]*x_hat + pres_grad[1]*y_hat + pres_grad[2]*z_hat
    ram_force = -1./density * dPdr
    vtheta = box['theta_velocity_corrected'].in_units('cm/s').v
    vphi = box['phi_velocity_corrected'].in_units('cm/s').v
    vtheta_edges = vtheta[disk_edges]
    vphi_edges = vphi[disk_edges]
    vtheta_interp_func = LinearNDInterpolator(list(zip(x_edges,y_edges,z_edges)), vtheta_edges)
    vtheta_masked = np.copy(vtheta)
    vphi_interp_func = LinearNDInterpolator(list(zip(x_edges,y_edges,z_edges)), vphi_edges)
    vphi_masked = np.copy(vphi)
    vtheta_masked[disk_mask] = vtheta_interp_func(x[disk_mask], y[disk_mask], z[disk_mask])
    vphi_masked[disk_mask] = vphi_interp_func(x[disk_mask], y[disk_mask], z[disk_mask])
    smooth_vtheta = gaussian_filter(vtheta_masked, smooth_scale)
    smooth_vphi = gaussian_filter(vphi_masked, smooth_scale)
    rot_force = (smooth_vtheta**2. + smooth_vphi**2.)/r
    grav_force = -G*Menc_profile(r/(1000*cmtopc))*gtoMsun/r**2.
    grav_edges = grav_force[disk_edges]
    grav_interp_func = LinearNDInterpolator(list(zip(x_edges,y_edges,z_edges)), grav_edges)
    grav_masked = np.copy(grav_force)
    grav_masked[disk_mask] = grav_interp_func(x[disk_mask], y[disk_mask], z[disk_mask])
    grav_force = grav_masked
    tot_force = thermal_force + turb_force + rot_force + ram_force + grav_force
    forces = [thermal_force, turb_force, rot_force, grav_force, tot_force]

    # Rotate array to line up with disk
    '''print(ds.z_unit_disk)
    yz_rotate_angle = np.arccos(np.dot([0,1], [ds.z_unit_disk[1], ds.z_unit_disk[2]]))*180./np.pi
    xy_rotate_angle = np.arccos(np.dot([0,1], [ds.z_unit_disk[0], ds.z_unit_disk[1]]))*180./np.pi
    print(yz_rotate_angle, xy_rotate_angle)
    rotated_forces = []
    for i in range(len(forces)):
        force = forces[i]
        # Rotate first in x direction (rotation within y-z plane)
        rotated_force = rotate(force, angle=yz_rotate_angle, axes=(1,2), reshape=False, order=0)
        # Then rotate in z direction (rotation within x-y plane)
        rotated_force = rotate(rotated_force, angle=xy_rotate_angle, axes=(0,1), reshape=False, order=0)
        rotated_forces.append(rotated_force)
    rotated_radius = rotate(radius, angle=yz_rotate_angle, axes=(1,2), reshape=False, order=0)
    rotated_density = rotate(density, angle=yz_rotate_angle, axes=(1,2), reshape=False, order=0)
    rotated_temperature = rotate(temperature, angle=yz_rotate_angle, axes=(1,2), reshape=False, order=0)
    rotated_radius = rotate(rotated_radius, angle=xy_rotate_angle, axes=(0,1), reshape=False, order=0)
    rotated_density = rotate(rotated_density, angle=xy_rotate_angle, axes=(0,1), reshape=False, order=0)
    rotated_temperature = rotate(rotated_temperature, angle=xy_rotate_angle, axes=(0,1), reshape=False, order=0)
    forces = rotated_forces
    radius = rotated_radius
    density = rotated_density
    temperature = rotated_temperature'''

    radius_func = RegularGridInterpolator(coords, radius)
    density_func = RegularGridInterpolator(coords, density)
    temperature_func = RegularGridInterpolator(coords, temperature)
    ray_start = [0, 0, 0]
    # List end points of rays
    # Want: 3 within 30 deg opening angle of minor axis on both sides, 3 within 30 deg opening angle
    # of major axis on both sides, 2 close to halfway between major and minor axis in all 4 quadrants
    # = 6 + 6 + 8 = 20 total rays
    #ray_ends = [[0, -100, -150], [0, 150, -150], [0, 100, 150], [0, -150, 150]]
    ray_ends = [[0, -75, -100], [0, 100, -40]]
    rays = ray_ends
    '''rays = []
    rotate = np.linalg.inv(ds.disk_rot_arr)
    for r in range(len(ray_ends)):
        # Rotate rays so they align with disk
        ray_end = ray_ends[r]
        ray_end_rot = []
        for i in range(len(ray_end)):
            ray_end_rot.append(rotate[i][0]*ray_end[0] + rotate[i][1]*ray_end[1] + rotate[i][2]*ray_end[2])
        rays.append(ray_end_rot)'''

    plot_colors = ['r', 'g', 'b', 'gold', 'k']
    linestyles = ['-', '--', ':', '-.']
    labels = ['Thermal', 'Turbulent', 'Rotation', 'Gravity', 'Total']
    ftypes = ['thermal', 'turbulent', 'rotation', 'gravity', 'total']

    fig2 = plt.figure(figsize=(12,10), dpi=500)
    ax2 = fig2.add_subplot(1,1,1)
    im = ax2.imshow(rotate(temperature[len(temperature)//2,:,:], 90), cmap=sns.blend_palette(('salmon', "#984ea3", "#4daf4a", "#ffe34d", 'darkorange'), as_cmap=True), norm=colors.LogNorm(vmin=10**4, vmax=10**7), \
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
        im = ax2.imshow(rotate(force[len(force)//2,:,:], 90), cmap='BrBG', norm=colors.SymLogNorm(vmin=-1e-6, vmax=1e-6, linthresh=1e-9, base=10), \
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
            force_func = RegularGridInterpolator(coords, force)
            force_ray = force_func(points)
            radius_ray = radius_ray[(density_ray < cgm_density_max) & (temperature_ray > cgm_temperature_min)]
            force_ray = force_ray[(density_ray < cgm_density_max) & (temperature_ray > cgm_temperature_min)]
            ax1.plot(radius_ray, force_ray, color=plot_colors[i], ls=linestyles[r], lw=2, label=labels[i])
            ax3.plot(radius_ray, force_ray, color=plot_colors[i], ls=linestyles[r], lw=2)
            ax2.plot([ray_start[1], ray_end[1]], [ray_start[2], ray_end[2]], color='k', ls=linestyles[r], lw=2)

            if (i==len(forces)-1):
                ax1.plot([0,150],[0,0], 'k-', lw=1)
                ax1.set_xlabel('Distance along ray from galaxy center [kpc]', fontsize=18)
                ax1.set_ylabel('Force on CGM gas [cm/s$^2$]', fontsize=18)
                ax1.axis([0,150,-1e-5,1e-5])
                ax1.set_yscale('symlog', linthreshy=1e-9)
                ax1.tick_params(axis='both', which='both', direction='in', length=8, width=2, pad=5, labelsize=18, \
                  top=True, right=True)
                ax1.legend(loc=1, frameon=False, fontsize=14)
                fig1.subplots_adjust(top=0.94,bottom=0.11,right=0.95,left=0.17)
                fig1.savefig(save_dir + snap + '_forces_along_ray-' + str(r) + save_suffix + '.png')
                plt.close(fig1)

        ax2.axis([-150,150,-150,150])
        ax2.set_xlabel('y [kpc]', fontsize=20)
        ax2.set_ylabel('z [kpc]', fontsize=20)
        ax2.tick_params(axis='both', which='both', direction='in', length=8, width=2, pad=5, labelsize=20, \
          top=True, right=True)
        cax = fig2.add_axes([0.85, 0.08, 0.03, 0.9])
        cax.tick_params(axis='both', which='both', direction='in', length=8, width=2, pad=5, labelsize=20, \
          top=True, right=True)
        fig2.colorbar(im, cax=cax, orientation='vertical')
        ax2.text(1.17, 0.5, 'log ' + labels[i] + ' Force [cm/s$^2$]', fontsize=20, rotation='vertical', ha='center', va='center', transform=ax2.transAxes)
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

def support_vs_radius(snap):
    '''Plots different types of pressure (thermal, turbulent, rotational, bulk inflow/outflow ram)
    support as functions of radius for the simulation output given by 'snap'.'''

    tablename_prefix = output_dir + 'stats_halo_00' + args.halo + '/' + args.run + '/Tables/'
    stats = Table.read(tablename_prefix + snap + '_' + args.filename + '.hdf5', path='all_data')
    Rvir = rvir_masses['radius'][rvir_masses['snapshot']==snap]

    plot_colors = ['r', 'g', 'b', 'm', 'c', 'k']
    plot_labels = ['Thermal', 'Turbulent', 'Rotation', 'Ram - in', 'Ram - out', 'Total']
    linestyles = ['-', '--', ':', '-.', '-.', '-']
    linewidths = [2, 2, 2, 1, 1, 3]

    radius_list = 0.5*(stats['inner_radius'] + stats['outer_radius'])
    zsnap = stats['redshift'][0]

    grav_pot = stats['net_grav_pot_med']/(radius_list*1000*cmtopc)*10**stats['net_log_density_med']

    fig = plt.figure(figsize=(8,6), dpi=500)
    ax = fig.add_subplot(1,1,1)

    total_pres = np.zeros(len(radius_list))
    for i in range(len(plot_colors)):
        label = plot_labels[i]
        if (label=='Thermal'):
            thermal_support = np.diff(10**stats['net_log_pressure_med'])/np.diff(radius_list*1000*cmtopc)
            ax.plot(radius_list[:-1], thermal_support/grav_pot[:-1], ls=linestyles[i], color=plot_colors[i], \
                    lw=linewidths[i], label=label)
            total_pres += 10.**stats['net_log_pressure_med']
        if (label=='Turbulent'):
            turb_pres = 10.**stats['net_log_density_med']*(0.5*(stats['net_theta_velocity_sig']*1e5) + 0.5*(stats['net_phi_velocity_sig']*1e5))**2.
            turb_support = np.diff(turb_pres)/np.diff(radius_list*1000*cmtopc)
            total_pres += turb_pres
            ax.plot(radius_list[:-1], turb_support/grav_pot[:-1], ls=linestyles[i], color=plot_colors[i], lw=linewidths[i], label=label)
        if (label=='Rotation'):
            rot_pres = 10.**stats['net_log_density_med']*((stats['net_theta_velocity_mu']*1e5)**2.+(stats['net_phi_velocity_mu']*1e5)**2.)
            rot_support = np.diff(rot_pres)/np.diff(radius_list*1000*cmtopc)
            total_pres += rot_pres
            ax.plot(radius_list[:-1], rot_support/grav_pot[:-1], ls=linestyles[i], color=plot_colors[i], lw=linewidths[i], label=label)
        if (label=='Total'):
            total_support = np.diff(total_pres)/np.diff(radius_list*1000*cmtopc)
            ax.plot(radius_list[:-1], total_support/grav_pot[:-1], ls=linestyles[i], color=plot_colors[i], lw=linewidths[i], label=label)

    ax.set_ylabel('Pressure Support', fontsize=18)
    ax.set_xlabel('Radius [kpc]', fontsize=18)
    ax.axis([0,250,-2,4])
    ax.text(15, -1., '$z=%.2f$' % (zsnap), fontsize=18, ha='left', va='center')
    ax.text(15,-1.5,halo_dict[args.halo],ha='left',va='center',fontsize=18)
    ax.tick_params(axis='both', which='both', direction='in', length=8, width=2, pad=5, labelsize=18, \
      top=True, right=True)
    ax.plot([Rvir, Rvir], [ax.get_ylim()[0], ax.get_ylim()[1]], 'k--', lw=1)
    ax.text(Rvir-3., -1.5, '$R_{200}$', fontsize=18, ha='right', va='center')
    ax.plot([ax.get_xlim()[0], ax.get_xlim()[1]], [1,1], 'k-', lw=1)
    if (args.halo=='8508'): ax.legend(loc=2, frameon=False, fontsize=18)
    plt.subplots_adjust(top=0.94,bottom=0.11,right=0.95,left=0.15)
    plt.savefig(save_dir + snap + '_support_vs_r' + save_suffix + '.pdf')
    plt.close()

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

def pressure_slice(snap):
    '''Plots a slice of pressure through the center of the halo. The option --pressure_type indicates
    what type of pressure to plot.'''

    masses_ind = np.where(masses['snapshot']==snap)[0]
    Menc_profile = IUS(np.concatenate(([0],masses['radius'][masses_ind])), np.concatenate(([0],masses['total_mass'][masses_ind])))
    Mvir = rvir_masses['total_mass'][rvir_masses['snapshot']==snap]
    Rvir = rvir_masses['radius'][rvir_masses['snapshot']==snap][0]

    if (args.system=='pleiades_cassi') and (foggie_dir!='/nobackupp18/mpeeples/') and (args.copy_to_tmp):
        print('Copying directory to /tmp')
        snap_dir = '/tmp/' + snap
        shutil.copytree(foggie_dir + run_dir + snap, snap_dir)
        snap_name = snap_dir + '/' + snap
    else:
        snap_name = foggie_dir + run_dir + snap + '/' + snap
    ds, refine_box = foggie_load(snap_name, trackname, do_filter_particles=False, halo_c_v_name=halo_c_v_name, gravity=True, masses_dir=masses_dir)

    if (args.pressure_type=='all'):
        ptypes = ['thermal', 'turbulent', 'ram']
    elif (',' in args.pressure_type):
        ptypes = args.pressure_type.split(',')
    else:
        ptypes = [args.pressure_type]

    pix_res = float(np.min(refine_box['gas','dx'].in_units('kpc')))  # at level 11
    lvl1_res = pix_res*2.**11.
    level = 9
    dx = lvl1_res/(2.**level)
    smooth_scale = int(25./dx)
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
    disk_mask = (density > cgm_density_max) & (temperature < cgm_temperature_min)
    # disk_mask_expanded is a binary mask of both ISM regions AND their surrounding pixels
    struct = ndimage.generate_binary_structure(3,3)
    disk_mask_expanded = ndimage.binary_dilation(disk_mask, structure=struct, iterations=3)
    disk_mask_expanded = ndimage.binary_closing(disk_mask_expanded, structure=struct, iterations=3)
    # disk_edges is a binary mask of ONLY pixels surrounding ISM regions -- nothing inside ISM regions
    disk_edges = disk_mask_expanded & ~disk_mask
    x_edges = x[disk_edges].flatten()
    y_edges = y[disk_edges].flatten()
    z_edges = z[disk_edges].flatten()

    for i in range(len(ptypes)):
        if (ptypes[i]=='thermal'):
            thermal_pressure = box['pressure'].in_units('erg/cm**3').v
            pressure = thermal_pressure
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
            vx_interp_func = LinearNDInterpolator(list(zip(x_edges,y_edges,z_edges)), vx_edges)
            vy_interp_func = LinearNDInterpolator(list(zip(x_edges,y_edges,z_edges)), vy_edges)
            vz_interp_func = LinearNDInterpolator(list(zip(x_edges,y_edges,z_edges)), vz_edges)
            vx_masked = np.copy(vx)
            vy_masked = np.copy(vy)
            vz_masked = np.copy(vz)
            # Replace removed ISM regions with interpolated values
            vx_masked[disk_mask] = vx_interp_func(x[disk_mask], y[disk_mask], z[disk_mask])
            vy_masked[disk_mask] = vy_interp_func(x[disk_mask], y[disk_mask], z[disk_mask])
            vz_masked[disk_mask] = vz_interp_func(x[disk_mask], y[disk_mask], z[disk_mask])
            # Smooth resulting velocity field -- without contamination from ISM regions
            smooth_vx = gaussian_filter(vx_masked, smooth_scale/6.)
            smooth_vy = gaussian_filter(vy_masked, smooth_scale/6.)
            smooth_vz = gaussian_filter(vz_masked, smooth_scale/6.)
            sig_x = (vx - smooth_vx)**2.
            sig_y = (vy - smooth_vy)**2.
            sig_z = (vz - smooth_vz)**2.
            vdisp = np.sqrt((sig_x + sig_y + sig_z)/3.)
            turb_pressure = density*vdisp**2.
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
            vr_interp_func = LinearNDInterpolator(list(zip(x_edges,y_edges,z_edges)), vr_edges)
            vr_masked = np.copy(vr)
            # Replace removed ISM regions with interpolated values
            vr_masked[disk_mask] = vr_interp_func(x[disk_mask], y[disk_mask], z[disk_mask])
            dvr = np.gradient(vr_masked, dx_cm)
            delta_vr = dvr[0]*dx_cm*x_hat + dvr[1]*dx_cm*y_hat + dvr[2]*dx_cm*z_hat
            smooth_delta_vr = gaussian_filter(delta_vr, smooth_scale/6.)
            ram_pressure = density*(smooth_delta_vr)**2.
            pressure = ram_pressure
            pressure_label = 'Ram'

        pressure = np.ma.masked_where((density > cgm_density_max) & (temperature < cgm_temperature_min), pressure)
        fig = plt.figure(figsize=(12,10),dpi=500)
        ax = fig.add_subplot(1,1,1)
        p_cmap = copy.copy(mpl.cm.get_cmap(pressure_color_map))
        p_cmap.set_over(color='w', alpha=1.)
        # Need to rotate to match up with how yt plots it
        im = ax.imshow(rotate(np.log10(pressure[len(pressure)//2,:,:]),90), cmap=p_cmap, norm=colors.Normalize(vmin=-18, vmax=-12), \
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
        ax.text(1.15, 0.5, 'log ' + pressure_label + ' Pressure [erg/cm$^3$]', fontsize=20, rotation='vertical', ha='center', va='center', transform=ax.transAxes)
        plt.subplots_adjust(bottom=0.08, top=0.98, left=0.08, right=0.88)
        plt.savefig(save_dir + snap + '_' + ptypes[i] + '_pressure_slice_x' + save_suffix + '.png')

    # Delete output from temp directory if on pleiades
    if (args.system=='pleiades_cassi') and (foggie_dir!='/nobackupp18/mpeeples/') and (args.copy_to_tmp):
        print('Deleting directory from /tmp')
        shutil.rmtree(snap_dir)

def force_slice(snap):
    '''Plots a slice of different force terms through the center of the halo. The option --force_type indicates
    what type of force to plot.'''

    masses_ind = np.where(masses['snapshot']==snap)[0]
    Menc_profile = IUS(np.concatenate(([0],masses['radius'][masses_ind])), np.concatenate(([0],masses['total_mass'][masses_ind])))
    Mvir = rvir_masses['total_mass'][rvir_masses['snapshot']==snap]
    Rvir = rvir_masses['radius'][rvir_masses['snapshot']==snap][0]

    if (args.system=='pleiades_cassi') and (args.copy_to_tmp):
        print('Copying directory to /tmp')
        snap_dir = '/tmp/' + snap
        shutil.copytree(foggie_dir + run_dir + snap, snap_dir)
        snap_name = snap_dir + '/' + snap
    else:
        snap_name = foggie_dir + run_dir + snap + '/' + snap
    ds, refine_box = foggie_load(snap_name, trackname, do_filter_particles=False, halo_c_v_name=halo_c_v_name, gravity=True, masses_dir=masses_dir)

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
    disk_mask = (density > cgm_density_max) & (temperature < cgm_temperature_min)
    # disk_mask_expanded is a binary mask of both ISM regions AND their surrounding pixels
    struct = ndimage.generate_binary_structure(3,3)
    disk_mask_expanded = ndimage.binary_dilation(disk_mask, structure=struct, iterations=3)
    disk_mask_expanded = ndimage.binary_closing(disk_mask_expanded, structure=struct, iterations=3)
    # disk_edges is a binary mask of ONLY pixels surrounding ISM regions -- nothing inside ISM regions
    disk_edges = disk_mask_expanded & ~disk_mask
    x_edges = x[disk_edges].flatten()
    y_edges = y[disk_edges].flatten()
    z_edges = z[disk_edges].flatten()

    for i in range(len(ftypes)):
        if (ftypes[i]=='thermal') or ((ftypes[i]=='total') and (args.force_type!='all')):
            thermal_pressure = box['pressure'].in_units('erg/cm**3').v
            # Cut to only those values closest to removed ISM regions
            pres_edges = thermal_pressure[disk_edges]
            # Interpolate across removed ISM regions
            pres_interp_func = LinearNDInterpolator(list(zip(x_edges,y_edges,z_edges)), pres_edges)
            pres_masked = np.copy(thermal_pressure)
            # Replace removed ISM regions with interpolated values
            pres_masked[disk_mask] = pres_interp_func(x[disk_mask], y[disk_mask], z[disk_mask])
            pres_grad = np.gradient(pres_masked, dx_cm)
            dPdr = pres_grad[0]*x_hat + pres_grad[1]*y_hat + pres_grad[2]*z_hat
            thermal_force = -1./density * dPdr
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
            vx_interp_func = LinearNDInterpolator(list(zip(x_edges,y_edges,z_edges)), vx_edges)
            vy_interp_func = LinearNDInterpolator(list(zip(x_edges,y_edges,z_edges)), vy_edges)
            vz_interp_func = LinearNDInterpolator(list(zip(x_edges,y_edges,z_edges)), vz_edges)
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
            turb_pressure = density*vdisp**2.
            pres_grad = np.gradient(turb_pressure, dx_cm)
            dPdr = pres_grad[0]*x_hat + pres_grad[1]*y_hat + pres_grad[2]*z_hat
            turb_force = -1./density * dPdr
            if (ftypes[i]=='turbulent'):
                force = turb_force
                force_label = 'Turbulent Pressure'
        if (ftypes[i]=='ram') or ((ftypes[i]=='total') and (args.force_type!='all')):
            vr = box['radial_velocity_corrected'].in_units('cm/s').v
            # Cut to only those values closest to removed ISM regions
            vr_edges = vr[disk_edges]
            # Interpolate across removed ISM regions
            vr_interp_func = LinearNDInterpolator(list(zip(x_edges,y_edges,z_edges)), vr_edges)
            vr_masked = np.copy(vr)
            # Replace removed ISM regions with interpolated values
            vr_masked[disk_mask] = vr_interp_func(x[disk_mask], y[disk_mask], z[disk_mask])
            dvr = np.gradient(vr_masked, dx_cm)
            delta_vr = dvr[0]*dx_cm*x_hat + dvr[1]*dx_cm*y_hat + dvr[2]*dx_cm*z_hat
            smooth_delta_vr = gaussian_filter(delta_vr, smooth_scale)
            ram_pressure = density*(smooth_delta_vr)**2.
            pres_grad = np.gradient(ram_pressure, dx_cm)
            dPdr = pres_grad[0]*x_hat + pres_grad[1]*y_hat + pres_grad[2]*z_hat
            ram_force = -1./density * dPdr
            if (ftypes[i]=='ram'):
                force = ram_force
                force_label = 'Ram Pressure'
        if (ftypes[i]=='rotation') or ((ftypes[i]=='total') and (args.force_type!='all')):
            vtheta = box['theta_velocity_corrected'].in_units('cm/s').v
            vphi = box['phi_velocity_corrected'].in_units('cm/s').v
            # Cut to only those values closest to removed ISM regions
            vtheta_edges = vtheta[disk_edges]
            vphi_edges = vphi[disk_edges]
            # Interpolate across removed ISM regions
            vtheta_interp_func = LinearNDInterpolator(list(zip(x_edges,y_edges,z_edges)), vtheta_edges)
            vtheta_masked = np.copy(vtheta)
            vphi_interp_func = LinearNDInterpolator(list(zip(x_edges,y_edges,z_edges)), vphi_edges)
            vphi_masked = np.copy(vphi)
            # Replace removed ISM regions with interpolated values
            vtheta_masked[disk_mask] = vtheta_interp_func(x[disk_mask], y[disk_mask], z[disk_mask])
            vphi_masked[disk_mask] = vphi_interp_func(x[disk_mask], y[disk_mask], z[disk_mask])
            smooth_vtheta = gaussian_filter(vtheta_masked, smooth_scale)
            smooth_vphi = gaussian_filter(vphi_masked, smooth_scale)
            rot_force = (smooth_vtheta**2. + smooth_vphi**2.)/r
            if (ftypes[i]=='rotation'):
                force = rot_force
                force_label = 'Rotation'
        if (ftypes[i]=='gravity') or ((ftypes[i]=='total') and (args.force_type!='all')):
            grav_force = -G*Menc_profile(r/(1000*cmtopc))*gtoMsun/r**2.
            # Cut to only those values closest to removed ISM regions
            grav_edges = grav_force[disk_edges]
            # Interpolate across removed ISM regions
            grav_interp_func = LinearNDInterpolator(list(zip(x_edges,y_edges,z_edges)), grav_edges)
            grav_masked = np.copy(grav_force)
            # Replace removed ISM regions with interpolated values
            grav_masked[disk_mask] = grav_interp_func(x[disk_mask], y[disk_mask], z[disk_mask])
            grav_force = grav_masked
            if (ftypes[i]=='gravity'):
                force = grav_force
                force_label = 'Gravity'
        if (ftypes[i]=='total'):
            tot_force = thermal_force + turb_force + rot_force + ram_force + grav_force
            force = tot_force
            force_label = 'Total'

        force = np.ma.masked_where((density > cgm_density_max) & (temperature < cgm_temperature_min), force)
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
        plt.savefig(save_dir + snap + '_' + ftypes[i] + '_force_slice_x' + save_suffix + '.png')

    # Delete output from temp directory if on pleiades
    if (args.system=='pleiades_cassi') and (args.copy_to_tmp):
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

    if (args.system=='pleiades_cassi') and (args.copy_to_tmp):
        print('Copying directory to /tmp')
        snap_dir = '/tmp/' + snap
        shutil.copytree(foggie_dir + run_dir + snap, snap_dir)
        snap_name = snap_dir + '/' + snap
    else:
        snap_name = foggie_dir + run_dir + snap + '/' + snap
    ds, refine_box = foggie_load(snap_name, trackname, do_filter_particles=False, halo_c_v_name=halo_c_v_name, gravity=True, masses_dir=masses_dir)

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
    disk_mask = (density > cgm_density_max) & (temperature < cgm_temperature_min)
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
            v_interp_func = LinearNDInterpolator(list(zip(x_edges,y_edges,z_edges)), v_edges)
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
            ax1.tick_params(axis='both', which='both', direction='in', length=8, width=2, pad=5, labelsize=20, \
              top=True, right=True)
            cb = fig.colorbar(im1, ax=ax1, orientation='vertical', pad=0)
            cb.ax.tick_params(labelsize=16)
            cb.ax.set_ylabel(vlabels[j][i] + ' Velocity [km/s]', fontsize=16)
            im2 = ax2.imshow(rotate(smooth_v[len(smooth_v)//2,:,:],90), cmap=v_cmap, norm=colors.Normalize(vmin=vmins[j][i], vmax=vmaxes[j][i]), \
                      extent=[-1.5*Rvir,1.5*Rvir,-1.5*Rvir,1.5*Rvir])
            ax2.set_xlabel('y [kpc]', fontsize=20)
            ax2.set_ylabel('z [kpc]', fontsize=20)
            ax2.tick_params(axis='both', which='both', direction='in', length=8, width=2, pad=5, labelsize=20, \
              top=True, right=True)
            cb = fig.colorbar(im2, ax=ax2, orientation='vertical', pad=0)
            cb.ax.tick_params(labelsize=16)
            cb.ax.set_ylabel('Smoothed ' + vlabels[j][i] + ' Velocity [km/s]', fontsize=16)
            im3 = ax3.imshow(rotate(sig_v[len(sig_v)//2,:,:],90), cmap=v_cmap, norm=colors.Normalize(vmin=-200, vmax=200), \
                      extent=[-1.5*Rvir,1.5*Rvir,-1.5*Rvir,1.5*Rvir])
            ax3.set_xlabel('y [kpc]', fontsize=20)
            ax3.set_ylabel('z [kpc]', fontsize=20)
            ax3.tick_params(axis='both', which='both', direction='in', length=8, width=2, pad=5, labelsize=20, \
              top=True, right=True)
            cb = fig.colorbar(im3, ax=ax3, orientation='vertical', pad=0)
            cb.ax.tick_params(labelsize=16)
            cb.ax.set_ylabel(vlabels[j][i] + ' Velocity - Smoothed Velocity [km/s]', fontsize=16)
        plt.subplots_adjust(bottom=0.05, top=0.97, left=0.04, right=0.97, wspace=0.15, hspace=0.15)
        plt.savefig(save_dir + snap + '_' + vfile[j] + '_velocities_slice_x' + save_suffix + '.png')
        plt.close(fig)

    # Delete output from temp directory if on pleiades
    if (args.system=='pleiades_cassi') and (args.copy_to_tmp):
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
    #foggie_dir = '/nobackupp18/mpeeples/'

    # Set directory for output location, making it if necessary
    save_dir = output_dir + 'pressures_halo_00' + args.halo + '/' + args.run + '/'
    if not (os.path.exists(save_dir)): os.system('mkdir -p ' + save_dir)

    print('foggie_dir: ', foggie_dir)
    halo_c_v_name = code_path + 'halo_infos/00' + args.halo + '/' + args.run + '/halo_c_v'
    masses_dir = code_path + 'halo_infos/00' + args.halo + '/' + args.run + '/'
    masses = Table.read(masses_dir + 'masses_z-less-2.hdf5', path='all_data')
    rvir_masses = Table.read(masses_dir + 'rvir_masses.hdf5', path='all_data')

    outs = make_output_list(args.output, output_step=args.output_step)

    if (not args.filename) and ((args.plot=='pressures_vs_radius') or (args.plot=='support_vs_radius')):
        sys.exit("You must specify a filename where the data you want to plot is saved.")

    if (args.save_suffix): save_suffix = '_' + args.save_suffix
    else: save_suffix = ''

    if (len(outs)>1) and ('time' not in args.plot):
        save_dir += 'Movie_frames/'

    if (args.plot=='pressure_vs_radius'):
        if (args.nproc==1):
            for i in range(len(outs)):
                pressures_vs_radius(outs[i])
        else:
            target = pressures_vs_radius
    elif (args.plot=='pressure_vs_time'):
        pressures_vs_time(outs)
    elif (args.plot=='force_vs_radius'):
        if (args.nproc==1):
            for i in range(len(outs)):
                forces_vs_radius(outs[i])
        else:
            target = forces_vs_radius
    elif (args.plot=='force_vs_radius_pres'):
        if (args.nproc==1):
            for i in range(len(outs)):
                forces_vs_radius_from_med_pressures(outs[i])
        else:
            target = forces_vs_radius_from_med_pressures
    elif (args.plot=='force_vs_time'):
        forces_vs_time(outs)
    elif (args.plot=='support_vs_radius'):
        if (args.nproc==1):
            for i in range(len(outs)):
                support_vs_radius(outs[i])
        else:
            target = support_vs_radius
    elif (args.plot=='velocity_PDF'):
        if (args.nproc==1):
            for i in range(len(outs)):
                velocity_PDF(outs[i])
        else:
            target = velocity_PDF
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
    elif (args.plot=='pressure_slice'):
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
        if (args.nproc==1):
            for i in range(len(outs)):
                force_slice(outs[i])
        else:
            target = force_slice
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
    else:
        sys.exit("That plot type hasn't been implemented!")

    '''if (args.nproc!=1):
        # Split into a number of groupings equal to the number of processors
        # and run one process per processor
        for i in range(len(outs)//args.nproc):
            threads = []
            for j in range(args.nproc):
                snap = outs[args.nproc*i+j]
                threads.append(multi.Process(target=target, args=[snap]))
            for t in threads:
                t.start()
            for t in threads:
                t.join()
        # For any leftover snapshots, run one per processor
        threads = []
        for j in range(len(outs)%args.nproc):
            snap = outs[-(j+1)]
            threads.append(multi.Process(target=target, args=[snap]))
        for t in threads:
            t.start()
        for t in threads:
            t.join()'''

    if (args.nproc!=1):
        '''# Split into a number of groupings equal to the number of processors
        # and run one process per processor
        for i in range(len(outs)//args.nproc):
            snaps = []
            for j in range(args.nproc):
                snaps.append(outs[args.nproc*i+j])
            pool = multi.Pool(processes=args.nproc)
            pool.map(target, snaps)
            pool.close()
        # For any leftover snapshots, run one per processor
        snaps = []
        for j in range(len(outs)%args.nproc):
            snaps.append(outs[-(j+1)])
        pool = multi.Pool(processes=args.nproc)
        pool.map(target, snaps)
        pool.close()'''
        pool = multi.Pool(processes=args.nproc)
        pool.map(target, outs)
        pool.close()

    print(str(datetime.datetime.now()))
    print("All snapshots finished!")
