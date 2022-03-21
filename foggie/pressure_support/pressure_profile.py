# Filename: pressure_profile.py
# Author: Cassi
# Created: 2-22-22
# This script makes plots of thermal pressure profiles.

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

    parser = argparse.ArgumentParser(description='Plots pressure profiles.')

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

    parser.add_argument('--save_suffix', metavar='save_suffix', type=str, action='store', \
                        help='Do you want to append a string onto the names of the saved files? Default is no.')
    parser.set_defaults(save_suffix="")

    parser.add_argument('--region_filter', metavar='region_filter', type=str, action='store', \
                        help='Do you want to show pressures in different regions? Options are:\n' + \
                        '"velocity", "temperature", or "metallicity". If plotting from a stats_in_shells file,\n' + \
                        'the files must have been created with the same specified region_filter.')
    parser.set_defaults(region_filter='none')

    parser.add_argument('--plot', dest='plot', action='store_true', \
                        help='Do you want to plot pressure profiles? Default is no.')
    parser.set_defaults(plot=True)

    parser.add_argument('--table', dest='table', action='store_true', \
                        help='Do you want to make an ascii table of the pressure profiles? Default is no.')
    parser.set_defaults(table=True)

    args = parser.parse_args()
    return args

def pressures_vs_radius(snap):
    '''Plots mean and median thermal pressure profiles, weighted by either mass or volume, and the
    spreads around the mean or median, as functions of radius for the simulation output given
    by 'snap'. A file where these have already been calculated must exist.'''

    tablename_prefix = output_dir + 'stats_halo_00' + args.halo + '/' + args.run + '/Tables/'
    if (args.region_filter=='temperature') or (args.region_filter=='none'):
        region_file = 'T-split'
    if (args.region_filter=='metallicity'):
        region_file = 'Z-split'
    if (args.region_filter=='velocity'):
        region_file = 'v-split'
    stats_mass = Table.read(tablename_prefix + snap + '_stats_pressure-types_' + region_file + '_cgm-only_mass-weighted.hdf5', path='all_data')
    stats_volume = Table.read(tablename_prefix + snap + '_stats_pressure-types_' + region_file + '_cgm-only_volume-weighted.hdf5', path='all_data')
    Rvir = rvir_masses['radius'][rvir_masses['snapshot']==snap][0]

    plot_colors = ['b', 'r', 'b', 'r']
    plot_labels = ['Mass-weighted mean', 'Volume-weighted mean', 'Mass-weighted median', 'Volume-weighted median']
    file_labels = ['_avg', '_avg', '_med', '_med']
    linestyles = ['-', '-', '--', '--']

    radius_list = 0.5*(stats_mass['inner_radius'] + stats_mass['outer_radius'])

    if (args.region_filter!='none'):
        fig = plt.figure(figsize=(20,5), dpi=500)
        ax1 = fig.add_subplot(1,4,1)
        ax2 = fig.add_subplot(1,4,2)
        ax3 = fig.add_subplot(1,4,3)
        ax4 = fig.add_subplot(1,4,4)
    else:
        fig = plt.figure(figsize=(8,6), dpi=500)
        ax1 = fig.add_subplot(1,1,1)

    for i in range(len(plot_colors)):
        label = plot_labels[i]
        if ('Mass' in plot_labels[i]):
            stats = stats_mass
        else:
            stats = stats_volume
        ax1.plot(radius_list, stats['thermal_pressure' + file_labels[i]], ls=linestyles[i], color=plot_colors[i], \
                lw=2, label=label)
        if (file_labels[i]=='_avg'):
            lower = stats['thermal_pressure' + file_labels[i]] - stats['thermal_pressure_std']
            upper = stats['thermal_pressure' + file_labels[i]] + stats['thermal_pressure_std']
        else:
            lower = stats['thermal_pressure' + file_labels[i]] - 0.5*stats['thermal_pressure_iqr']
            upper = stats['thermal_pressure' + file_labels[i]] + 0.5*stats['thermal_pressure_iqr']
        ax1.fill_between(radius_list, lower, upper, alpha=0.25, color=plot_colors[i])

    ax1.set_ylabel('log Thermal Pressure [erg/cm$^3$]', fontsize=14)
    ax1.set_xlabel('Radius [kpc]', fontsize=14)
    ax1.axis([0,250,-17,-12])
    ax1.text(15,-16.5,halo_dict[args.halo],ha='left',va='center',fontsize=14)
    ax1.tick_params(axis='both', which='both', direction='in', length=8, width=2, pad=5, labelsize=14, \
      top=True, right=True)
    ax1.plot([Rvir, Rvir], [ax1.get_ylim()[0], ax1.get_ylim()[1]], 'k--', lw=1)
    ax1.text(Rvir-3., -16.5, '$R_{200}$', fontsize=14, ha='right', va='center')
    ax1.legend(loc=1, frameon=False, fontsize=14)
    if (args.region_filter=='none'):
        plt.subplots_adjust(top=0.94,bottom=0.11,right=0.95,left=0.15)
        plt.savefig(save_dir + snap + '_pressures_vs_r' + save_suffix + '.png')
        plt.close(fig)

    if (args.region_filter!='none'):
        for i in range(len(plot_colors)):
            if ('Mass' in plot_labels[i]):
                stats = stats_mass
            else:
                stats = stats_volume
            ax2.plot(radius_list, stats['low_' + args.region_filter + '_thermal_pressure' + file_labels[i]], ls=linestyles[i], color=plot_colors[i], \
                    lw=2)
            ax3.plot(radius_list, stats['mid_' + args.region_filter + '_thermal_pressure' + file_labels[i]], ls=linestyles[i], color=plot_colors[i], \
                    lw=2)
            ax4.plot(radius_list, stats['high_' + args.region_filter + '_thermal_pressure' + file_labels[i]], ls=linestyles[i], color=plot_colors[i], \
                    lw=2)
            if (file_labels[i]=='_avg'):
                lower_l = stats['low_' + args.region_filter + '_thermal_pressure' + file_labels[i]] - stats['low_' + args.region_filter + '_thermal_pressure_std']
                upper_l = stats['low_' + args.region_filter + '_thermal_pressure' + file_labels[i]] + stats['low_' + args.region_filter + '_thermal_pressure_std']
                lower_m = stats['mid_' + args.region_filter + '_thermal_pressure' + file_labels[i]] - stats['mid_' + args.region_filter + '_thermal_pressure_std']
                upper_m = stats['mid_' + args.region_filter + '_thermal_pressure' + file_labels[i]] + stats['mid_' + args.region_filter + '_thermal_pressure_std']
                lower_h = stats['high_' + args.region_filter + '_thermal_pressure' + file_labels[i]] - stats['high_' + args.region_filter + '_thermal_pressure_std']
                upper_h = stats['high_' + args.region_filter + '_thermal_pressure' + file_labels[i]] + stats['high_' + args.region_filter + '_thermal_pressure_std']
            else:
                lower_l = stats['low_' + args.region_filter + '_thermal_pressure' + file_labels[i]] - 0.5*stats['low_' + args.region_filter + '_thermal_pressure_iqr']
                upper_l = stats['low_' + args.region_filter + '_thermal_pressure' + file_labels[i]] + 0.5*stats['low_' + args.region_filter + '_thermal_pressure_iqr']
                lower_m = stats['mid_' + args.region_filter + '_thermal_pressure' + file_labels[i]] - 0.5*stats['mid_' + args.region_filter + '_thermal_pressure_iqr']
                upper_m = stats['mid_' + args.region_filter + '_thermal_pressure' + file_labels[i]] + 0.5*stats['mid_' + args.region_filter + '_thermal_pressure_iqr']
                lower_h = stats['high_' + args.region_filter + '_thermal_pressure' + file_labels[i]] - 0.5*stats['high_' + args.region_filter + '_thermal_pressure_iqr']
                upper_h = stats['high_' + args.region_filter + '_thermal_pressure' + file_labels[i]] + 0.5*stats['high_' + args.region_filter + '_thermal_pressure_iqr']
            ax2.fill_between(radius_list, lower_l, upper_l, alpha=0.25, color=plot_colors[i])
            ax3.fill_between(radius_list, lower_m, upper_m, alpha=0.25, color=plot_colors[i])
            ax4.fill_between(radius_list, lower_h, upper_h, alpha=0.25, color=plot_colors[i])

        #ax2.set_ylabel('log Thermal Pressure [erg/cm$^3$]', fontsize=14)
        ax2.set_xlabel('Radius [kpc]', fontsize=14)
        ax2.axis([0,250,-17,-12])
        ax2.plot([Rvir, Rvir], [ax2.get_ylim()[0], ax2.get_ylim()[1]], 'k--', lw=1)
        ax2.text(Rvir-3., -16.5, '$R_{200}$', fontsize=14, ha='right', va='center')
        ax2.tick_params(axis='both', which='both', direction='in', length=8, width=2, pad=5, labelsize=14, \
          top=True, right=True)
        #ax3.set_ylabel('log Thermal Pressure [erg/cm$^3$]', fontsize=14)
        ax3.set_xlabel('Radius [kpc]', fontsize=14)
        ax3.axis([0,250,-17,-12])
        ax3.tick_params(axis='both', which='both', direction='in', length=8, width=2, pad=5, labelsize=14, \
          top=True, right=True)
        ax3.plot([Rvir, Rvir], [ax3.get_ylim()[0], ax3.get_ylim()[1]], 'k--', lw=1)
        ax3.text(Rvir-3., -16.5, '$R_{200}$', fontsize=14, ha='right', va='center')
        #ax4.set_ylabel('log Thermal Pressure [erg/cm$^3$]', fontsize=14)
        ax4.set_xlabel('Radius [kpc]', fontsize=14)
        ax4.axis([0,250,-17,-12])
        ax4.tick_params(axis='both', which='both', direction='in', length=8, width=2, pad=5, labelsize=14, \
          top=True, right=True)
        ax4.plot([Rvir, Rvir], [ax4.get_ylim()[0], ax4.get_ylim()[1]], 'k--', lw=1)
        ax4.text(Rvir-3., -16.5, '$R_{200}$', fontsize=14, ha='right', va='center')

        if (args.region_filter=='temperature'):
            ax2.text(25, -12.5, '$T < 10^{4.8}$', fontsize=14, ha='left', va='center')
            ax3.text(25, -12.5, '$10^{4.8} < T < 10^{6.3}$', fontsize=14, ha='left', va='center')
            ax4.text(25, -12.5, '$T > 10^{6.3}$', fontsize=14, ha='left', va='center')
        if (args.region_filter=='metallicity'):
            ax2.text(25, -12.5, '$Z < 0.01 Z_\odot$', fontsize=14, ha='left', va='center')
            ax3.text(25, -12.5, '$0.01 Z_\odot < Z < 1 Z_\odot$', fontsize=14, ha='left', va='center')
            ax4.text(25, -12.5, '$Z > 1 Z_\odot$', fontsize=14, ha='left', va='center')
        if (args.region_filter=='velocity'):
            ax2.text(25, -12.5, '$v_r < -75$ km/s', fontsize=14, ha='left', va='center')
            ax3.text(25, -12.5, '$-75 < v_r < 75$ km/s', fontsize=14, ha='left', va='center')
            ax4.text(25, -12.5, '$v_r > 75$ km/s', fontsize=14, ha='left', va='center')

        plt.subplots_adjust(top=0.94,bottom=0.11,right=0.98,left=0.05,wspace=0.12)
        plt.savefig(save_dir + snap + '_pressures_vs_r_regions-' + args.region_filter + save_suffix + '.png')
        plt.close()

def table_combine(snap):
    '''Reads the pressure profiles from individual files and combines them into one ascii table.'''

    tablename_prefix = output_dir + 'stats_halo_00' + args.halo + '/' + args.run + '/Tables/'
    masses_ind = np.where(masses['snapshot']==snap)[0]
    Menc_profile = IUS(np.concatenate(([0],masses['radius'][masses_ind])), np.concatenate(([0],masses['total_mass'][masses_ind])))

    f1 = open(halo_dict[args.halo] + '_pressure_profile.dat', 'w')
    f1.write('# Radius (kpc)   vcirc (km/s)   ')
    f1.write('Mass-weighted mean   Mass-weighted median   Volume-weighted mean   Volume-weighted median   ' + \
            'Mass-weighted std   Mass-weighted IQR   Volume-weighted std   Volume-weighted IQR\n')

    filters = ['temperature', 'metallicity', 'velocity']
    filter_file = ['T-split', 'Z-split', 'v-split']
    filter_table = ['low_', 'mid_', 'high_']

    stats_mass = Table.read(tablename_prefix + snap + '_stats_pressure-types_T-split_cgm-only_mass-weighted.hdf5', path='all_data')
    stats_vol = Table.read(tablename_prefix + snap + '_stats_pressure-types_T-split_cgm-only_volume-weighted.hdf5', path='all_data')
    radius_list = 0.5*(stats_mass['inner_radius'] + stats_mass['outer_radius'])
    for i in range(len(radius_list)):
        vcirc = np.sqrt((G*Menc_profile(radius_list[i])*gtoMsun)/(radius_list[i]*1000*cmtopc))/1e5
        f1.write('%.2f' % (radius_list[i]))
        f1.write('   %.2f' % (vcirc))
        f1.write('   %.2f' % (stats_mass['thermal_pressure_avg'][i]))
        f1.write('   %.2f' % (stats_mass['thermal_pressure_med'][i]))
        f1.write('   %.2f' % (stats_vol['thermal_pressure_avg'][i]))
        f1.write('   %.2f' %  (stats_vol['thermal_pressure_med'][i]))
        f1.write('   %.2f' % (stats_mass['thermal_pressure_std'][i]))
        f1.write('   %.2f' % (stats_mass['thermal_pressure_iqr'][i]))
        f1.write('   %.2f' % (stats_vol['thermal_pressure_std'][i]))
        f1.write('   %.2f' % (stats_vol['thermal_pressure_iqr'][i]))
        f1.write('\n')
    f1.close()
    for j in range(len(filters)):
        f2 = open(halo_dict[args.halo] + '_pressure_profile_' + filter_file[j] + '.dat', 'w')
        if (filters[j]=='temperature'):
            f2.write('#                T < 10^4.8 gas' + 162*' ' + '10^4.8 < T < 10^6.3 gas' + 153*' ' + 'T > 10^6.3 gas\n')
        if (filters[j]=='metallicity'):
            f2.write('#                Z < 0.01 Zsun gas' + 159*' ' + '0.01 < Z < 1 Zsun gas' + 155*' ' + 'Z > 1 Zsun gas\n')
        if (filters[j]=='velocity'):
            f2.write('#                vr < -75 km/s gas' + 159*' ' + '-75 < vr < 75 km/s gas' + 154*' ' + 'vr > 75 km/s gas\n')
        f2.write('# Radius (kpc)   ')
        f2.write(('Mass-weighted mean   Mass-weighted median   Volume-weighted mean   Volume-weighted median   ' + \
                'Mass-weighted std   Mass-weighted IQR   Volume-weighted std   Volume-weighted IQR   ')*3 + '\n')
        stats_mass = Table.read(tablename_prefix + snap + '_stats_pressure-types_' + filter_file[j] + \
          '_cgm-only_mass-weighted.hdf5', path='all_data')
        stats_vol = Table.read(tablename_prefix + snap + '_stats_pressure-types_' + filter_file[j] + \
          '_cgm-only_volume-weighted.hdf5', path='all_data')
        for i in range(len(radius_list)):
            f2.write('%.2f' % (radius_list[i]))
            for k in range(len(filter_table)):
                f2.write('   %.2f' % (stats_mass[filter_table[k] + filters[j] + '_thermal_pressure_avg'][i]))
                f2.write('   %.2f' % (stats_mass[filter_table[k] + filters[j] + '_thermal_pressure_med'][i]))
                f2.write('   %.2f' % (stats_vol[filter_table[k] + filters[j] + '_thermal_pressure_avg'][i]))
                f2.write('   %.2f' %  (stats_vol[filter_table[k] + filters[j] + '_thermal_pressure_med'][i]))
                f2.write('   %.2f' % (stats_mass[filter_table[k] + filters[j] + '_thermal_pressure_std'][i]))
                f2.write('   %.2f' % (stats_mass[filter_table[k] + filters[j] + '_thermal_pressure_iqr'][i]))
                f2.write('   %.2f' % (stats_vol[filter_table[k] + filters[j] + '_thermal_pressure_std'][i]))
                f2.write('   %.2f' % (stats_vol[filter_table[k] + filters[j] + '_thermal_pressure_iqr'][i]))
            f2.write('\n')
        f2.close()

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
    
    if (args.save_suffix): save_suffix = '_' + args.save_suffix
    else: save_suffix = ''

    for i in range(len(outs)):
        if (args.plot):
            pressures_vs_radius(outs[i])
        if (args.table):
            table_combine(outs[i])

    print(str(datetime.datetime.now()))
    print("All snapshots finished!")
