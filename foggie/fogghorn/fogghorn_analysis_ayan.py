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
                        run fogghorn_analysis.py --directory /Users/acharyya/models/simulation_output/foggie/halo_008508/nref11c_nref9f --system ayan_local --halo 8508 --output RD0030 --upto_kpc 10 --docomoving --clobber
"""

from __future__ import print_function

import numpy as np
import argparse
import os
import copy

import matplotlib
#matplotlib.use('agg') # Ayan commented this out because it was leading to weird errors while running in ipython
import matplotlib.pyplot as plt

from numpy.polynomial import Polynomial
import multiprocessing as multi
from pathlib import Path
import pandas as pd
from uncertainties import ufloat, unumpy

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
        print('Reading from existing file ' + outfilename, args)
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
        plt.savefig(output_filename)

# --------------------------------------------------------------------------------------------------------------------
def gas_metallicity_histogram(ds, region, args):
    '''
    Plots a histogram of the gas metallicity.
    Returns nothing. Saves output as png file
    '''
    output_filename = args.save_directory + '/' + args.snap + '_gas_metallicity_histogram' + args.upto_text + args.density_cut_text + '.png'

    if need_to_make_this_plot(output_filename, args):
        plt.savefig(output_filename)

# ---------------------------------------------------------------------------------
def fit_binned(df, xcol, ycol, x_bins, ax, color='darkorange', weightcol=None):
    '''
    Function to overplot binned data on existing plot of radial profile of gas metallicity
    '''
    df['binned_cat'] = pd.cut(df[xcol], x_bins)

    if weightcol is not None:
        agg_func = lambda x: np.sum(x * df.loc[x.index, weightcol]) / np.sum(df.loc[x.index, weightcol]) # function to get weighted mean
        agg_u_func = lambda x: np.sqrt(((np.sum(df.loc[x.index, weightcol] * x**2) / np.sum(df.loc[x.index, weightcol])) - (np.sum(x * df.loc[x.index, weightcol]) / np.sum(df.loc[x.index, weightcol]))**2) * (np.sum(df.loc[x.index, weightcol]**2)) / (np.sum(df.loc[x.index, weightcol])**2 - np.sum(df.loc[x.index, weightcol]**2))) # eq 6 of http://seismo.berkeley.edu/~kirchner/Toolkits/Toolkit_12.pdf
    else:
        agg_func, agg_u_func = np.mean, np.std

    y_binned = df.groupby('binned_cat', as_index=False).agg([(ycol, agg_func)])[ycol].values.flatten()
    y_u_binned = df.groupby('binned_cat', as_index=False).agg([(ycol, agg_u_func)])[ycol].values.flatten()
    x_bin_centers = x_bins[:-1] + np.diff(x_bins) / 2

    # --------- for correct propagation of errors ----------
    quant = unumpy.log10(unumpy.uarray(y_binned, y_u_binned))
    y_binned, y_u_binned = unumpy.nominal_values(quant), unumpy.std_devs(quant)

    # getting rid of potential nan values
    indices = np.array(np.logical_not(np.logical_or(np.isnan(x_bin_centers), np.isnan(y_binned))))
    x_bin_centers = x_bin_centers[indices]
    y_binned = y_binned[indices]
    y_u_binned = y_u_binned[indices]

    # ----------to plot mean binned y vs x profile--------------
    linefit, linecov = np.polyfit(x_bin_centers, y_binned, 1, cov=True)
    y_fitted = np.poly1d(linefit)(x_bin_centers)

    Zgrad = ufloat(linefit[0], np.sqrt(linecov[0][0]))
    Zcen = ufloat(linefit[1], np.sqrt(linecov[1][1]))

    y_binned, y_u_binned, y_fitted = 10**y_binned, 10**y_u_binned, 10**y_fitted # to convert things back to log space

    print('Upon radially binning: Inferred slope for output ' + args.snap + ' is', Zgrad, 'dex/kpc')

    ax.errorbar(x_bin_centers, y_binned, c=color, yerr=y_u_binned, lw=2, ls='none', zorder=1)
    ax.scatter(x_bin_centers, y_binned, c=color, s=150, lw=1, ec='black', zorder=10)
    ax.plot(x_bin_centers, y_fitted, color=color, lw=2.5, ls='dashed')
    ax.text(0.033, 0.2, 'Slope = %.2F ' % linefit[0] + 'dex/kpc', color=color, transform=ax.transAxes, fontsize=args.fontsize, va='center', bbox=dict(facecolor='k', alpha=0.6, edgecolor='k'))
    return ax, Zcen, Zgrad

# --------------------------------------------------------------------------------------------------------------------
def gas_metallicity_radial_profile(ds, region, args):
    '''
    Plots a radial profile of the gas metallicity.
    Returns nothing. Saves output as png file
    '''
    output_filename = args.save_directory + '/' + args.snap + '_gas_metallicity_radial_profile' + args.upto_text + args.density_cut_text + '.png'

    if need_to_make_this_plot(output_filename, args):
        df = get_df_from_ds(region, args)

        # ---------first, plot both cell-by-cell profile first, using datashader---------
        fig, ax = plt.subplots(figsize=(8, 8))
        fig.subplots_adjust(hspace=0.05, wspace=0.05, right=0.95, top=0.95, bottom=0.1, left=0.17)
        artist = dsshow(df, dsh.Point(args.xcol, 'log_metal'), dsh.count(), norm='linear',x_range=(0, args.galrad), y_range=(-2.2, 1.2), aspect='auto', ax=ax, cmap='Blues_r') # this is to make the background datashader plot

        # --------bin the metallicity profile and plot the binned profile-----------
        ax, Zcen, Zgrad = fit_binned(df, 'rad', 'metal', np.linspace(0, args.galrad, 10), ax, weightcol=args.weight)
        linefit = [Zgrad.n, Zcen.n]

        # ----------plot the fitted metallicity profile---------------
        color = 'limegreen'
        fitted_y = np.poly1d(linefit)(args.bin_edges)
        ax.plot(args.bin_edges, fitted_y, color=color, lw=3, ls='solid')
        plt.text(0.033, 0.3, 'Slope = %.2F ' % linefit[0] + 'dex/kpc', color=color, transform=ax.transAxes, va='center', fontsize=args.fontsize,bbox=dict(facecolor='k', alpha=0.6, edgecolor='k'))

        # ----------tidy up figure-------------
        ax.set_xlim(0, args.upto_re if 're' in args.xcol else args.upto_kpc)
        ax.set_ylim(args.ylim[0], args.ylim[1])
        if args.forproposal: ax.set_yscale('log')

        ax.set_xlabel('Radius (kpc)', fontsize=args.fontsize)
        ax.set_ylabel(r'Metallicity (Z$_{\odot}$)' if args.forproposal else r'Log Metallicity (Z$_{\odot}$)',
                      fontsize=args.fontsize)

        ax.set_xticks(np.linspace(ax.get_xlim()[0], ax.get_xlim()[1], 6))
        ax.set_yticks(np.linspace(ax.get_ylim()[0], ax.get_ylim()[1], 3 if args.forproposal else 5))

        ax.set_xticklabels(['%.1F' % item for item in ax.get_xticks()], fontsize=args.fontsize)
        ax.set_yticklabels(['%.2F' % item for item in ax.get_yticks()], fontsize=args.fontsize)

        # ---------annotate and save the figure----------------------
        plt.text(0.033, 0.05, 'z = %.2F' % ds.current_redshift, transform=ax.transAxes, fontsize=args.fontsize)
        plt.savefig(output_filename)
        print('Saved figure ' + output_filename)

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
        else: yt.ProjectionPlot(ds, args.projection, 'metallicity', data_source=region, width=(20, 'kpc'), center=ds.halo_center_code)
        p.set_unit('metallicity','Zsun')
        p.set_cmap('metallicity', old_metal_color_map)
        p.set_zlim('metallicity', 2e-2, 4e0)
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
def KS_relation(ds, region, args):
    '''
    Plots the KS relation from the dataset as compared to a curve taken from KMT09.
    Returns nothing. Saves output as png file
    '''

    output_filename = args.save_directory + '/' + args.snap + '_KS-relation' + args.upto_text + '.png'

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

# --------------------------------------------------------------------------------------------------------------------
def make_plots(snap, args):
    '''
    Finds the halo center and other properties of the dataset and then calls the plotting scripts.
    Returns nothing. Saves outputs as multiple png files
    '''

    # ----------------------- Read the snapshot ----------------------
    filename = args.directory + '/' + snap + '/' + snap
    ds, region = foggie_load(filename, args.trackfile, disk_relative=True)

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
    # gas_density_projection(ds, region, args)
    # young_stars_density_projection(ds, region, args)
    # KS_relation(ds, region, args)
    # gas_metallicity_projection(ds, region, args)
    # gas_metallicity_radial_profile(ds, region, args)
    # gas_metallicity_histogram(ds, region, args)
    # gas_resolved_MZR(ds, regio, args)

# --------------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    args = parse_args()

    # ------------------ Figure out directory and outputs -------------------------------------
    if args.save_directory is None:
        args.save_directory = args.directory + '/plots'
        Path(args.save_directory).mkdir(parents=True, exist_ok=True)

    if args.trackfile is None: _, _, _, _, args.trackfile, _, _, _ = get_run_loc_etc(args) # for FOGGIE production runs it knows which trackfile to grab

    if args.output is not None: # Running on specific output/s
        outputs =[item for item in args.output.split(',')]
    else: # Running on all snapshots in the directory
        outputs = []
        for fname in os.listdir(args.directory):
            folder_path = os.path.join(args.directory, fname)
            if os.path.isdir(folder_path) and ((fname[0:2]=='DD') or (fname[0:2]=='RD')):
                outputs.append(fname)

    # --------- Loop over outputs, for either single-processor or parallel processor computing ---------------
    if (args.nproc == 1):
        for snap in outputs:
            make_plots(snap, args)
    else:
        # Split into a number of groupings equal to the number of processors
        # and run one process per processor
        for i in range(len(outputs)//args.nproc):
            threads = []
            for j in range(args.nproc):
                snap = outputs[args.nproc*i+j]
                threads.append(multi.Process(target=make_plots, args=[snap, args]))
            for t in threads:
                t.start()
            for t in threads:
                t.join()
        # For any leftover snapshots, run one per processor
        threads = []
        for j in range(len(outputs)%args.nproc):
            snap = outputs[-(j+1)]
            threads.append(multi.Process(target=make_plots, args=[snap, args]))
        for t in threads:
            t.start()
        for t in threads:
            t.join()