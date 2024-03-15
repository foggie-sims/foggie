"""

Filename: fogghorn_analysis.py
Authors: Cassi, Ayan,
Created: 01-24-24
Last modified: by Ayan in March 2024

This script produces a set of basic analysis plots for all outputs in the directory passed to it.

Plots included so far:
- Gas density projection
- New stars density projection
- Kennicutt-Schmidt relation compared to KMT09 relation

Examples of how to run: run fogghorn_analysis.py --directory /Users/acharyya/models/simulation_output/foggie/halo_5205/natural_7n --upto_kpc 10 --docomoving --weight mass
                        run fogghorn_analysis.py --directory /Users/acharyya/models/simulation_output/foggie/halo_008508/nref11c_nref9f --system ayan_local --halo 8508 --output RD0030 --upto_kpc 10 --docomoving --weight mass
"""

from __future__ import print_function

import numpy as np
import argparse
import os
import copy
import time
import datetime

import matplotlib
#matplotlib.use('agg') # Ayan commented this out because it was leading to weird errors while running in ipython
import matplotlib.pyplot as plt

from numpy.polynomial import Polynomial
import multiprocessing as multi
from pathlib import Path
import pandas as pd
from uncertainties import ufloat, unumpy
import seaborn as sns

from astropy.table import Table
from astropy.io import ascii

import datashader as dsh
from datashader.utils import export_image
from datashader import transfer_functions as dstf
datashader_ver = float(dsh.__version__.split('.')[1])
if datashader_ver > 11: from datashader.mpl_ext import dsshow

import yt
from yt.units import *
from yt import YTArray

from foggie.utils.consistency import *
from foggie.utils.get_refine_box import get_refine_box
from foggie.utils.get_halo_center import get_halo_center
from foggie.utils.get_proper_box_size import get_proper_box_size
from foggie.utils.get_run_loc_etc import get_run_loc_etc
from foggie.utils.yt_fields import *
from foggie.utils.foggie_load import *
from foggie.utils.analysis_utils import *

import warnings
warnings.filterwarnings("ignore")
start_time = time.time()

# --------------------------------------------------------------------------------------------------------------------
def parse_args():
    '''Parse command line arguments. Returns args object.'''

    parser = argparse.ArgumentParser(description='Produces analysis plots for FOGGHORN runs.')

    # Optional arguments:
    parser.add_argument('--directory', metavar='directory', type=str, action='store', default='', help='What is the directory of the enzo outputs you want to make plots of?')
    parser.add_argument('--save_directory', metavar='save_directory', type=str, action='store', default=None, help='Where do you want to store the plots, if different from where the outputs are stored?')
    parser.add_argument('--output', metavar='output', type=str, action='store', default=None, help='If you want to make the plots for specific output/s then specify those here separated by comma (e.g., DD0030,DD0040). Otherwise (default) it will make plots for ALL outputs in that directory')
    parser.add_argument('--trackfile', metavar='trackfile', type=str, action='store', default=None, help='What is the directory of the track file for this halo?\n' + 'This is needed to find the center of the galaxy of interest.')
    parser.add_argument('--pwd', dest='pwd', action='store_true', default=False, help='Just use the working directory?, Default is no')
    parser.add_argument('--nproc', metavar='nproc', type=int, action='store', default=1, help='How many processes do you want? Default is 1 (no parallelization), if multiple processors are specified, code will run one output per processor')

    parser.add_argument('--clobber', dest='clobber', action='store_true', default=False, help='Over-write existing plots? Default is no.')
    parser.add_argument('--silent', dest='silent', action='store_true', default=False, help='Suppress some generic pritn statements? Default is no.')
    parser.add_argument('--upto_kpc', metavar='upto_kpc', type=float, action='store', default=None, help='Limit analysis out to a certain physical kpc. By default it does the entire refine box.')
    parser.add_argument('--docomoving', dest='docomoving', action='store_true', default=False, help='Consider the input upto_kpc as a comoving quantity? Default is No.')
    parser.add_argument('--weight', metavar='weight', type=str, action='store', default=None, help='Name of quantity to weight the metallicity by. Default is None i.e., no weighting.')
    parser.add_argument('--projection', metavar='projection', type=str, action='store', default='z', help='Which projection do you want to plot, i.e., which axis is your line of sight? Default is z')
    parser.add_argument('--disk_rel', dest='disk_rel', action='store_true', default=False, help='Consider projection plots w.r.t the disk rather than the box edges? Be aware that this will turn on disk_relative=True while reading in each snapshot whic might slow down the loading of data. Default is No.')
    parser.add_argument('--use_density_cut', dest='use_density_cut', action='store_true', default=False, help='Impose a density cut to get just the disk? Default is no.')
    parser.add_argument('--fontsize', metavar='fontsize', type=int, action='store', default=15, help='Fontsize of plot labels, etc. Default is 15')
    parser.add_argument('--nbins', metavar='nbins', type=int, action='store', default=100, help='Number of bins to use for the metallicity histogram plot. Default is 100')

    # The following three args are used for backward compatibility, to find the trackfile for production runs, if a trackfile has not been explicitly specified
    parser.add_argument('--system', metavar='system', type=str, action='store', default='ayan_local', help='Which system are you on? This is used only when trackfile is not specified. Default is ayan_local')
    parser.add_argument('--halo', metavar='halo', type=str, action='store', default='8508', help='Which halo? Default is Tempesxt. This is used only when trackfile is not specified.')
    parser.add_argument('--run', metavar='run', type=str, action='store', default='nref11c_nref9f', help='Which run? Default is nref11c_nref9f. This is used only when trackfile is not specified.')

    args = parser.parse_args()
    return args

# --------------------------------------------------------------------------------------------------------------------
def need_to_make_this_plot(output_filename, args):
    '''
    Determines whether a figure with this name already exists, and if so, should it be over-written
    :return boolean
    '''
    if os.path.exists(output_filename):
        if not args.silent: print(output_filename + 'already exists.')
        if args.clobber:
            if not args.silent: print('But we will re-make it...')
            return True
        else:
            if not args.silent: print('So we will skip it.')
            return False
    else:
        if not args.silent: print('About to make ' + output_filename + '...')
        return True

# --------------------------------------------------------------------
def get_density_cut(t):
    '''
    Function to get density cut based on Cassi's paper. The cut is a function of ime.
    if z > 0.5: rho_cut = 2e-26 g/cm**3
    elif z < 0.25: rho_cut = 2e-27 g/cm**3
    else: linearly from 2e-26 to 2e-27 from z = 0.5 to z = 0.25
    Takes time in Gyr as input
    '''
    t1, t2 = 8.628, 10.754 # Gyr; corresponds to z1 = 0.5 and z2 = 0.25
    rho1, rho2 = 2e-26, 2e-27 # g/cm**3
    t = np.float64(t)
    rho_cut = np.piecewise(t, [t < t1, (t >= t1) & (t <= t2), t > t2], [rho1, lambda t: rho1 + (t - t1) * (rho2 - rho1) / (t2 - t1), rho2])
    return rho_cut

# -------------------------------------------------------------------------------
def get_df_from_ds(box, args, outfilename=None):
    '''
    Function to make a pandas dataframe from the yt dataset, including only the metallicity profile (for now),
    then writes dataframe to file for faster access in future
    This function is somewhat based on foggie.utils.prep_dataframe.prep_dataframe()
    :return: pandas dataframe
    '''
    # ------------- Set up paths and dicts -------------------
    Path(args.save_directory + 'txtfiles/').mkdir(parents=True, exist_ok=True)  # creating the directory structure, if doesn't exist already
    if outfilename is None: outfilename = args.save_directory + 'txtfiles/' + args.snap + '_df_metallicity_vs_rad_%s%s.txt' % (args.upto_text, args.density_cut_text)

    field_dict = {'rad': ('gas', 'radius_corrected'), 'density': ('gas', 'density'), 'mass': ('gas', 'mass'), 'metal': ('gas', 'metallicity'), 'temp': ('gas', 'temperature'), \
                  'vrad': ('gas', 'radial_velocity_corrected'), 'vdisp': ('gas', 'velocity_dispersion_3d'), 'phi_L': ('gas', 'angular_momentum_phi'), 'theta_L': ('gas', 'angular_momentum_theta'), \
                  'volume': ('gas', 'volume'), 'phi_disk': ('gas', 'phi_pos_disk'), 'theta_disk': ('gas', 'theta_pos_disk')} # this is a superset of many quantities, only a few of these quantities will be extracted from the dataset to build the dataframe

    unit_dict = {'rad': 'kpc', 'rad_re': '', 'density': 'g/cm**3', 'metal': r'Zsun', 'temp': 'K', 'vrad': 'km/s',
                 'phi_L': 'deg', 'theta_L': 'deg', 'PDF': '', 'mass': 'Msun', 'stars_mass': 'Msun',
                 'ystars_mass': 'Msun', 'ystars_age': 'Gyr', 'gas_frac': '', 'gas_time': 'Gyr', 'volume': 'pc**3',
                 'phi_disk': 'deg', 'theta_disk': 'deg', 'vdisp': 'km/s'}

    # ------------- Write new pandas df file -------------------
    if not os.path.exists(outfilename) or args.clobber:
        print(outfilename + ' does not exist. Creating afresh..')

        if args.use_density_cut:
            rho_cut = get_density_cut(args.current_time)  # based on Cassi's CGM-ISM density cut-off
            box = box.cut_region(['obj["gas", "density"] > %.1E' % rho_cut])
            print('Imposing a density criteria to get ISM above density', rho_cut, 'g/cm^3')

        df = pd.DataFrame()
        fields = ['rad', 'metal'] # only the relevant properties
        if args.weight is not None: fields += [args.weight]

        for index, field in enumerate(fields):
            print('Loading property: ' + field + ', which is ' + str(index + 1) + ' of the ' + str(len(fields)) + ' fields..')
            df[field] = box[field_dict[field]].in_units(unit_dict[field]).ndarray_view()

        df.to_csv(outfilename, sep='\t', index=None)
    else:
        # ------------- Read from existing pandas df file -------------------
        print('Reading from existing file ' + outfilename)
        try:
            df = pd.read_table(outfilename, delim_whitespace=True, comment='#')
        except pd.errors.EmptyDataError:
            print('File existed, but it was empty, so making new file afresh..')
            dummy_args = copy.deepcopy(args)
            dummy_args.clobber = True
            df = get_df_from_ds(box, dummy_args, outfilename=outfilename)

    df['log_metal'] = np.log10(df['metal'])

    return df

# --------------------------------------------------------------------------------------------------------------------
def gas_metallicity_resolved_MZR(ds, region, args):
    '''
    Plots a spatially resolved gas metallicity vs gas mass relation.
    Returns nothing. Saves output as png file
    '''
    output_filename = args.save_directory + '/' + args.snap + '_resolved_gas_MZR' + args.upto_text + args.density_cut_text + '.png'

    if need_to_make_this_plot(output_filename, args):
        df = get_df_from_ds(region, args)

        # --------- Setting up the figure ---------
        fig, ax = plt.subplots(figsize=(8, 8))
        fig.subplots_adjust(hspace=0.05, wspace=0.05, right=0.95, top=0.95, bottom=0.1, left=0.15)

        # Ayan will add stuff here

        # ---------annotate and save the figure----------------------
        plt.text(0.97, 0.95, 'z = %.2F' % ds.current_redshift, ha='right', transform=ax.transAxes, fontsize=args.fontsize)
        plt.savefig(output_filename)
        print('Saved figure ' + output_filename)
        plt.close()

# ----------------------------------------------------------------------------
# Following function is adapted from https://stackoverflow.com/questions/21844024/weighted-percentile-using-numpy
def weighted_quantile(values, quantiles, weight=None):
    """ Very close to numpy.percentile, but supports weights.
    NOTE: quantiles should be in [0, 1]!
    :param values: numpy.array with data
    :param quantiles: array-like with many quantiles needed
    :param sample_weight: array-like of the same length as `array`
    :param values_sorted: bool, if True, then will avoid sorting of initial array
    :param old_style: if True, will correct output to be consistent with numpy.percentile.
    :return: numpy.array with computed quantiles.
    """
    if weight is None: weight = np.ones(len(values))
    values = np.array(values)
    quantiles = np.array(quantiles)
    weight = np.array(weight)
    assert np.all(quantiles >= 0) and np.all(quantiles <= 1), 'quantiles should be in [0, 1]'

    sorter = np.argsort(values)
    values = values[sorter]
    weight = weight[sorter]

    weighted_quantiles = np.cumsum(weight) - 0.5 * weight
    weighted_quantiles /= np.sum(weight)
    return np.interp(quantiles, weighted_quantiles, values)

# --------------------------------------------------------------------------------------------------------------------
def gas_metallicity_histogram(ds, region, args):
    '''
    Plots a histogram of the gas metallicity (No Gaussian fits, for now).
    Returns nothing. Saves output as png file
    '''
    output_filename = args.save_directory + '/' + args.snap + '_gas_metallicity_histogram' + args.upto_text + args.density_cut_text + '.png'

    if need_to_make_this_plot(output_filename, args):
        df = get_df_from_ds(region, args)

        # --------- Plotting the histogram ---------
        fig, ax = plt.subplots(figsize=(8, 8))
        fig.subplots_adjust(hspace=0.05, wspace=0.05, right=0.95, top=0.95, bottom=0.1, left=0.15)

        color = 'salmon'
        p = plt.hist(df['log_metal'], bins=args.nbins, histtype='step', lw=2, density=True, ec=color, weights=df[args.weight] if args.weight is not None else None)

        # ---------- Adding vertical lines for percentile -------------
        percentiles = weighted_quantile(df['log_metal'], [0.25, 0.50, 0.75], weight=df[args.weight] if args.weight is not None else None)
        for thispercentile in np.atleast_1d(percentiles): ax.axvline(thispercentile, lw=1, ls='solid', color='maroon')

        # ---------- Tidy up figure-------------
        plt.legend(loc='upper right', bbox_to_anchor=(1, 0.75), fontsize=args.fontsize)
        ax.set_xlim(-1.5, 2.0)
        ax.set_ylim(0, 2.5)

        ax.set_xticks(np.linspace(ax.get_xlim()[0], ax.get_xlim()[1], 5))
        ax.set_yticks(np.linspace(ax.get_ylim()[0], ax.get_ylim()[1], 6))

        ax.set_xlabel(r'Log Metallicity (Z$_{\odot}$)', fontsize=args.fontsize)
        ax.set_ylabel('Normalised distribution', fontsize=args.fontsize)
        ax.set_xticklabels(['%.1F' % item for item in ax.get_xticks()], fontsize=args.fontsize)
        ax.set_yticklabels(['%.2F' % item for item in ax.get_yticks()], fontsize=args.fontsize)

        # ---------annotate and save the figure----------------------
        plt.text(0.97, 0.95, 'z = %.2F' % ds.current_redshift, ha='right', transform=ax.transAxes, fontsize=args.fontsize)
        plt.savefig(output_filename)
        print('Saved figure ' + output_filename)
        plt.close()

# ---------------------------------------------------------------------------------
def bin_fit_radial_profile(df, xcol, ycol, x_bins, ax, args, color='darkorange'):
    '''
    Function to overplot binned data on existing plot of radial profile of gas metallicity
    '''
    df['binned_cat'] = pd.cut(df[xcol], x_bins)

    if args.weight is not None:
        agg_func = lambda x: np.sum(x * df.loc[x.index, args.weight]) / np.sum(df.loc[x.index, args.weight]) # function to get weighted mean
        agg_u_func = lambda x: np.sqrt(((np.sum(df.loc[x.index, args.weight] * x**2) / np.sum(df.loc[x.index, args.weight])) - (np.sum(x * df.loc[x.index, args.weight]) / np.sum(df.loc[x.index, args.weight]))**2) * (np.sum(df.loc[x.index, args.weight]**2)) / (np.sum(df.loc[x.index, args.weight])**2 - np.sum(df.loc[x.index, args.weight]**2))) # eq 6 of http://seismo.berkeley.edu/~kirchner/Toolkits/Toolkit_12.pdf
    else:
        agg_func, agg_u_func = np.mean, np.std

    y_binned = df.groupby('binned_cat', as_index=False).agg([(ycol, agg_func)])[ycol].values.flatten()
    y_u_binned = df.groupby('binned_cat', as_index=False).agg([(ycol, agg_u_func)])[ycol].values.flatten()
    x_bin_centers = x_bins[:-1] + np.diff(x_bins) / 2

    # --------- For correct propagation of errors, given that the actual fitting will be in log-space ----------
    quant = unumpy.log10(unumpy.uarray(y_binned, y_u_binned))
    y_binned, y_u_binned = unumpy.nominal_values(quant), unumpy.std_devs(quant)

    # getting rid of potential nan values
    indices = np.array(np.logical_not(np.logical_or(np.isnan(x_bin_centers), np.isnan(y_binned))))
    x_bin_centers = x_bin_centers[indices]
    y_binned = y_binned[indices]
    y_u_binned = y_u_binned[indices]

    # ---------- Plot mean binned y vs x profile--------------
    linefit, linecov = np.polyfit(x_bin_centers, y_binned, 1, cov=True)
    y_fitted = np.poly1d(linefit)(x_bin_centers)

    Zgrad = ufloat(linefit[0], np.sqrt(linecov[0][0]))
    Zcen = ufloat(linefit[1], np.sqrt(linecov[1][1]))

    ax.errorbar(x_bin_centers, y_binned, c=color, yerr=y_u_binned, lw=2, ls='none', zorder=1)
    ax.scatter(x_bin_centers, y_binned, c=color, s=150, lw=1, ec='black', zorder=10)
    ax.plot(x_bin_centers, y_fitted, color=color, lw=2.5, ls='dashed')
    ax.text(0.033, 0.2, 'Slope = %.2F ' % linefit[0] + 'dex/kpc', color=color, transform=ax.transAxes, fontsize=args.fontsize, va='center', bbox=dict(facecolor='k', alpha=0.6, edgecolor='k'))
    return ax, Zcen, Zgrad

# --------------------------------------------------------------------------------------------------------------------
def gas_metallicity_radial_profile(ds, region, args):
    '''
    Plots a radial profile of the gas metallicity, overplotted with the radially binned profile and the fit to the binned profile.
    Returns nothing. Saves output as png file
    '''
    output_filename = args.save_directory + '/' + args.snap + '_gas_metallicity_radial_profile' + args.upto_text + args.density_cut_text + '.png'
    args.ylim = [-2.2, 1.2]

    if need_to_make_this_plot(output_filename, args):
        df = get_df_from_ds(region, args)

        # --------- First, plot both cell-by-cell profile first, using datashader---------
        fig, ax = plt.subplots(figsize=(8, 8))
        fig.subplots_adjust(hspace=0.05, wspace=0.05, right=0.95, top=0.95, bottom=0.1, left=0.17)
        artist = dsshow(df, dsh.Point('rad', 'log_metal'), dsh.count(), norm='linear',x_range=(0, args.galrad), y_range=(args.ylim[0], args.ylim[1]), aspect='auto', ax=ax, cmap='Blues_r')

        # -------- Next, bin the metallicity profile and overplot the binned profile-----------
        bin_edges = np.linspace(0, args.galrad, 10)
        ax, Zcen, Zgrad = bin_fit_radial_profile(df, 'rad', 'metal', bin_edges, ax, args)
        linefit = [Zgrad.n, Zcen.n]

        # ---------- Then, plot the fitted metallicity profile---------------
        color = 'limegreen'
        fitted_y = np.poly1d(linefit)(bin_edges)
        ax.plot(bin_edges, fitted_y, color=color, lw=3, ls='solid')
        plt.text(0.033, 0.3, 'Slope = %.2F ' % linefit[0] + 'dex/kpc', color=color, transform=ax.transAxes, va='center', fontsize=args.fontsize,bbox=dict(facecolor='k', alpha=0.6, edgecolor='k'))

        # ---------- Tidy up figure-------------
        ax.set_xlim(0, args.upto_kpc)
        ax.set_ylim(args.ylim[0], args.ylim[1])
        ax.set_yscale('log')

        ax.set_xlabel('Radius (kpc)', fontsize=args.fontsize)
        ax.set_ylabel(r'Log Metallicity (Z$_{\odot}$)', fontsize=args.fontsize)

        ax.set_xticks(np.linspace(ax.get_xlim()[0], ax.get_xlim()[1], 6))
        ax.set_yticks(np.linspace(ax.get_ylim()[0], ax.get_ylim()[1], 5))

        ax.set_xticklabels(['%.1F' % item for item in ax.get_xticks()], fontsize=args.fontsize)
        ax.set_yticklabels(['%.2F' % item for item in ax.get_yticks()], fontsize=args.fontsize)

        # --------- Annotate and save the figure----------------------
        plt.text(0.033, 0.05, 'z = %.2F' % ds.current_redshift, transform=ax.transAxes, fontsize=args.fontsize)
        plt.savefig(output_filename)
        print('Saved figure ' + output_filename)
        plt.close()

# --------------------------------------------------------------------------------------------------------------------
def gas_metallicity_projection(ds, region, args):
    '''
    Plots a gas metallicity projection of the galaxy disk.
    If the --disk_rel argument was used, this function will automatically project w.r.t the disk, instead of the box edges.
    Returns nothing. Saves output as png file
    '''
    output_filename = args.save_directory + '/' + args.snap + '_gas_metallicity_projection_' + args.projection_text + args.upto_text + args.density_cut_text + '.png'

    if need_to_make_this_plot(output_filename, args):
        if args.disk_rel: p = yt.OffAxisProjectionPlot(ds, args.projection_axis_dict[args.projection], 'metallicity', data_source=region, width=(20, 'kpc'), center=ds.halo_center_code)
        else: p = yt.ProjectionPlot(ds, args.projection, 'metallicity', data_source=region, width=(20, 'kpc'), center=ds.halo_center_code)
        p.set_unit('metallicity','Zsun*cm') # the length dimension is because this is a projected quantity
        p.set_cmap('metallicity', old_metal_color_map)
        #p.set_zlim('metallicity', 2e-2, 4e0)
        p.save(output_filename)

# --------------------------------------------------------------------------------------------------------------------
def gas_density_projection(ds, region, args):
    '''
    Plots a gas density projection of the galaxy disk.
    If the --disk_rel argument was used, this function will automatically project w.r.t the disk, instead of the box edges.
    Returns nothing. Saves output as png file
    '''
    output_filename = args.save_directory + '/' + args.snap + '_gas_density_projection_disk-' + args.projection_text + args.upto_text + args.density_cut_text + '.png'

    if need_to_make_this_plot(output_filename, args):
        if args.disk_rel: p = yt.OffAxisProjectionPlot(ds, args.projection_axis_dict[args.projection], 'density', data_source=region, width=(20, 'kpc'), center=ds.halo_center_code)
        else: p = yt.ProjectionPlot(ds, args.projection, 'density', data_source=region, width=(20, 'kpc'), center=ds.halo_center_code)
        p.set_unit('density','Msun/pc**2')
        p.set_cmap('density', density_color_map)
        p.set_zlim('density', 1e-2, 3e2)
        p.save(output_filename)

# --------------------------------------------------------------------------------------------------------------------
def young_stars_density_projection(ds, region, args):
    '''
    Plots a young stars density projection of the galaxy disk.
    If the --disk_rel argument was used, this function will automatically project w.r.t the disk, instead of the box edges.
    Returns nothing. Saves output as png file
    '''

    output_filename = args.save_directory + '/' + args.snap + '_young_stars3_cic_projection_' + args.projection_text + args.upto_text + args.density_cut_text + '.png'

    if need_to_make_this_plot(output_filename, args):
        if args.disk_rel: p = yt.OffAxisProjectionPlot(ds, args.projection_axis_dict[args.projection], ('deposit', 'young_stars3_cic'), width=(20, 'kpc'), data_source=region, center=ds.halo_center_code)
        else: p = yt.ProjectionPlot(ds, args.projection, ('deposit', 'young_stars3_cic'), width=(20, 'kpc'), data_source=region, center=ds.halo_center_code)
        p.set_unit(('deposit','young_stars3_cic'),'Msun/kpc**2')
        p.set_zlim(('deposit','young_stars3_cic'), 1e3, 1e6)
        p.save(output_filename)

# --------------------------------------------------------------------------------------------------------------------
def edge_visualizations(ds, region, args):
    """Plot slices & thin projections of galaxy temperature viewed from the disk edge."""

    output_basename = args.save_directory + '/' + args.snap

    # ---------- Visualize along two perpendicular edge axes ---------------
    for label, axis in zip(['disk-x', 'disk-y'], [ds.x_unit_disk, ds.y_unit_disk]):

        p_filename = output_basename + f'_Projection_{label}_temperature_density.png'
        s_filename = output_basename + f'_Slice_{label}_temperature.png'

        if need_to_make_this_plot(p_filename, args):
            # --------------- "Thin" projections (30 kpc deep) -----------------------
            try:
                p = yt.ProjectionPlot(ds, axis, 'temperature', weight_field='density', center=ds.halo_center_code, data_source=region, width=(60, 'kpc'), depth=(30, 'kpc'), north_vector=ds.z_unit_disk)
            except TypeError: # in case 'depth' is an "unexpected keyword" for a different version of yt
                p = yt.OffAxisProjectionPlot(ds, axis, 'temperature', weight_field='density', center=ds.halo_center_code, data_source=region, width=(60, 'kpc'), depth=(30, 'kpc'), north_vector=ds.z_unit_disk)
            p.set_cmap('temperature', sns.blend_palette(('salmon', '#984ea3', '#4daf4a', '#ffe34d', 'darkorange'), as_cmap=True))
            p.set_zlim('temperature', 1e4,1e7)
            p.annotate_timestamp(corner='upper_left', redshift=True, time=True, draw_inset_box=True)
            p.save(p_filename)

        if need_to_make_this_plot(s_filename, args):
            # ---------- Slices ----------------------------
            s = yt.SlicePlot(ds, axis, "temperature", center=ds.halo_center_code, data_source=region, width=(60,"kpc"), north_vector=ds.z_unit_disk)
            s.set_cmap('temperature', sns.blend_palette(('salmon', "#984ea3", "#4daf4a", "#ffe34d", 'darkorange'), as_cmap=True))
            s.set_zlim('temperature', 1e4,1e7)
            s.annotate_timestamp(corner='upper_left', redshift=True, time=True, draw_inset_box=True)
            s.save(s_filename)

# --------------------------------------------------------------------------------------------------------------------
def KS_relation(ds, region, args):
    '''
    Plots the KS relation from the dataset as compared to a curve taken from KMT09.
    Returns nothing. Saves output as png file
    '''

    output_filename = args.save_directory + '/' + args.snap + '_KS-relation' + args.upto_text + args.density_cut_text + '.png'

    if need_to_make_this_plot(output_filename, args):
        # Make a projection and convert to FRB
        p = yt.ProjectionPlot(ds, args.projection, 'density', data_source=region, width=(20, 'kpc'), center=ds.halo_center_code)
        proj_frb = p.data_source.to_frb((20., "kpc"), 500)
        # Pull out the gas surface density and the star formation rate of the young stars
        projected_density = proj_frb['density'].in_units('Msun/pc**2')
        ks_nh1 = proj_frb['H_p0_number_density'].in_units('pc**-2') * yt.YTArray(1.67e-24/1.989e33, 'Msun')
        young_stars = proj_frb[('deposit', 'young_stars3_cic')].in_units('Msun/kpc**2')
        ks_sfr = young_stars / yt.YTArray(3e6, 'yr') + yt.YTArray(1e-6, 'Msun/kpc**2/yr')

        # These values are pulled from KMT09
        log_sigma_gas = [0.5278, 0.6571, 0.8165, 1.0151, 1.2034, 1.4506, 1.6286, 1.9399, 2.2663, 2.7905, 3.5817]
        log_sigma_sfr = [-5.1072, -4.4546, -3.5572, -2.7926, -2.3442, -2.0185, -1.8253, -1.5406, -1.0927, -0.3801, 0.6579]
        c = Polynomial.fit(log_sigma_gas, log_sigma_sfr, deg=5)

        # Make the plot
        plt.plot(np.log10(ks_nh1), np.log10(ks_sfr), '.')
        plt.plot(log_sigma_gas, log_sigma_sfr, marker='o', color='red')
        plt.xlabel('$\Sigma _{g} \,\, (M_{\odot} / pc^2)$', fontsize=16)
        plt.ylabel('$\dot{M} _{*} \,\, (M_{\odot} / yr / kpc^2)$', fontsize=16)
        plt.axis([-1,5,-6,3])
        plt.tick_params(axis='both', which='both', direction='in', length=8, width=2, pad=5, labelsize=14, top=True, right=True)
        plt.tight_layout()
        plt.savefig(output_filename, dpi=300)
        plt.close()

# --------------------------------------------------------------------------------------------------------------------
def outflow_rates(ds, region, args):
    '''Plots the mass and metals outflow rates, both as a function of radius centered on the galaxy
    and as a function of height through 20x20 kpc horizontal planes above and below the disk of young stars.
    Uses only gas with outflow velocities greater than 50 km/s.'''

    output_filename = args.save_directory + '/' + args.snap + '_outflows.png'

    if need_to_make_this_plot(output_filename, args):
        # ------------- Load needed fields into arrays --------------------
        radius = region['gas','radius_corrected'].in_units('kpc')
        x = region['gas', 'x_disk'].in_units('kpc').v
        y = region['gas', 'y_disk'].in_units('kpc').v
        z = region['gas', 'z_disk'].in_units('kpc').v
        vx = region['gas','vx_disk'].in_units('kpc/yr').v
        vy = region['gas','vy_disk'].in_units('kpc/yr').v
        vz = region['gas','vz_disk'].in_units('kpc/yr').v
        mass = region['gas', 'cell_mass'].in_units('Msun').v
        metals = region['gas','metal_mass'].in_units('Msun').v
        rv = region['gas','radial_velocity_corrected'].in_units('km/s').v
        hv = region['gas','vz_disk'].in_units('km/s').v

        # ----------- Define radius and height lists ---------------------------
        radii = np.linspace(0.5, 20., 40)
        heights = np.linspace(0.5, 20., 40)

        # ---------- Calculate new positions of gas cells 10 Myr later ---------------
        dt = 10.e6
        new_x = vx*dt + x
        new_y = vy*dt + y
        new_z = vz*dt + z
        new_radius = np.sqrt(new_x**2. + new_y**2. + new_z**2.)

        # ------------- Sum the mass and metals passing through the boundaries ---------------
        mass_sph = []
        metal_sph = []
        mass_horiz = []
        metal_horiz = []
        for i in range(len(radii)):
            r = radii[i]
            mass_sph.append(np.sum(mass[(radius < r) & (new_radius > r) & (rv > 50.)])/dt)
            metal_sph.append(np.sum(metals[(radius < r) & (new_radius > r) & (rv > 50.)])/dt)
        for i in range(len(heights)):
            h = heights[i]
            mass_horiz.append(np.sum(mass[(np.abs(z) < h) & (np.abs(new_z) > h) & (np.abs(hv) > 50.)])/dt)
            metal_horiz.append(np.sum(metals[(np.abs(z) < h) & (np.abs(new_z) > h) & (np.abs(hv) > 50.)])/dt)

        # --------------- Plot the outflow rates ------------------------------------
        fig = plt.figure(1, figsize=(10,4))
        ax1 = fig.add_subplot(1,2,1)
        ax2 = fig.add_subplot(1,2,2)
        ax1.plot(radii, mass_sph, 'k-', lw=2, label='Mass')
        ax1.plot(radii, metal_sph, 'k--', lw=2, label='Metals')
        ax1.set_ylabel(r'Mass outflow rate [$M_\odot$/yr]', fontsize=16)
        ax1.set_xlabel('Radius [kpc]', fontsize=16)
        ax1.set_yscale('log')
        ax1.tick_params(axis='both', which='both', direction='in', length=8, width=2, pad=5, labelsize=14)
        ax1.legend(loc=1, frameon=False, fontsize=16)
        ax2.plot(heights, mass_horiz, 'k-', lw=2, label='Mass')
        ax2.plot(heights, metal_horiz, 'k--', lw=2, label='Metals')
        ax2.set_ylabel(r'Mass outflow rate [$M_\odot$/yr]', fontsize=16)
        ax2.set_xlabel('Height from disk midplane [kpc]', fontsize=16)
        ax2.set_yscale('log')
        ax2.tick_params(axis='both', which='both', direction='in', length=8, width=2, pad=5, labelsize=14)
        plt.tight_layout()
        plt.savefig(output_filename, dpi=300)
        plt.close()

# --------------------------------------------------------------------------------------------------------------------
def make_plots(snap, args):
    '''
    Finds the halo center and other properties of the dataset and then calls the plotting scripts.
    Returns nothing. Saves outputs as multiple png files
    '''

    # ----------------------- Read the snapshot ----------------------
    filename = args.directory + '/' + snap + '/' + snap
    ds, region = foggie_load(filename, args.trackfile)#, disk_relative=True)

    # ----------------- Add some parameters to args that will be used throughout ----------------------------------
    args.snap = snap
    args.projection_axis_dict = {'x': ds.x_unit_disk, 'y': ds.y_unit_disk, 'z': ds.z_unit_disk}
    args.projection_text = '_disk-' + args.projection if args.disk_rel else '_' + args.projection
    args.density_cut_text = '_wdencut' if args.use_density_cut else ''

    # --------- If a upto_kpc is specified, then the analysis 'region' will be restricted up to that value ---------
    if args.upto_kpc is not None:
        if args.docomoving: args.galrad = args.upto_kpc / (1 + ds.current_redshift) / 0.695  # include stuff within a fixed comoving kpc h^-1, 0.695 is Hubble constant
        else: args.galrad = args.upto_kpc  # include stuff within a fixed physical kpc
        region = ds.sphere(ds.halo_center_kpc, ds.arr(args.galrad, 'kpc'))

        args.upto_text = '_upto%.1Fckpchinv' % args.upto_kpc if args.docomoving else '_upto%.1Fkpc' % args.upto_kpc
    else:
        args.galrad = ds.refine_width / 2.
        args.upto_text = ''

    # ----------------------- Make the plots ---------------------------------------------
    gas_density_projection(ds, region, args)
    edge_visualizations(ds, region, args)
    young_stars_density_projection(ds, region, args)
    KS_relation(ds, region, args)
    outflow_rates(ds, region, args)
    gas_metallicity_projection(ds, region, args)
    gas_metallicity_radial_profile(ds, region, args)
    gas_metallicity_histogram(ds, region, args)
    gas_metallicity_resolved_MZR(ds, region, args)
    print('Yayyy you have completed making all plots for this snap ' + snap)

# --------------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    cli_args = parse_args()

    # ------------------ Figure out directory and outputs -------------------------------------
    if cli_args.save_directory is None:
        cli_args.save_directory = cli_args.directory + '/plots'
        Path(cli_args.save_directory).mkdir(parents=True, exist_ok=True)

    if cli_args.trackfile is None: _, _, _, _, cli_args.trackfile, _, _, _ = get_run_loc_etc(cli_args) # for FOGGIE production runs it knows which trackfile to grab

    if cli_args.output is not None: # Running on specific output/s
        outputs = make_output_list(cli_args.output)
    else: # Running on all snapshots in the directory
        outputs = []
        for fname in os.listdir(cli_args.directory):
            folder_path = os.path.join(cli_args.directory, fname)
            if os.path.isdir(folder_path) and ((fname[0:2]=='DD') or (fname[0:2]=='RD')):
                outputs.append(fname)

    # --------- Loop over outputs, for either single-processor or parallel processor computing ---------------
    if (cli_args.nproc == 1):
        for snap in outputs:
            make_plots(snap, cli_args)
        print('Serially: time taken for ' + str(len(outputs)) + ' snapshots with ' + str(cli_args.nproc) + ' core was %s mins' % ((time.time() - start_time) / 60))
    else:
        # ------- Split into a number of groupings equal to the number of processors and run one process per processor ---------
        for i in range(len(outputs)//cli_args.nproc):
            threads = []
            for j in range(cli_args.nproc):
                snap = outputs[cli_args.nproc*i+j]
                threads.append(multi.Process(target=make_plots, args=[snap, cli_args]))
            for t in threads:
                t.start()
            for t in threads:
                t.join()
        # ----- For any leftover snapshots, run one per processor ------------------
        threads = []
        for j in range(len(outputs) % cli_args.nproc):
            snap = outputs[-(j+1)]
            threads.append(multi.Process(target=make_plots, args=[snap, cli_args]))
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        print('Parallely: time taken for ' + str(len(outputs)) + ' snapshots with ' + str(cli_args.nproc) + ' cores was %s mins' % ((time.time() - start_time) / 60))
