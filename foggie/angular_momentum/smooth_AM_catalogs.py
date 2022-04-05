'''
Filename: smooth_AM_catalogs.py
Author: Cassi
Created: 4-4-22

This script reads in the angular momentum catalogs (created by Raymond) for every DD output for a
given halo, removes outliers, and interpolates over the points to create a smooth AM catalog for
less jittery disk-oriented videos and analysis.
'''

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
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

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

    parser.add_argument('--system', metavar='system', type=str, action='store', \
                        help='Which system are you on? Default is cassiopeia')
    parser.set_defaults(system='cassiopeia')

    parser.add_argument('--pwd', dest='pwd', action='store_true',
                        help='Just use the working directory?, Default is no')
    parser.set_defaults(pwd=False)

    parser.add_argument('--plot', dest='plot', action='store_true',
                        help='Want to plot the AM direction along with the smoothed curve? Default is no.')
    parser.set_defaults(plot=False)

    args = parser.parse_args()
    return args

def plot_AM():
    '''Plots the time evolution of the x, y, and z components of
    the stellar angular momentum vector saved in the catalog.'''

    fig = plt.figure(figsize=(20,15),dpi=500)
    ax_x = fig.add_subplot(3,1,1)
    ax_y = fig.add_subplot(3,1,2)
    ax_z = fig.add_subplot(3,1,3)

    ax_x.plot(DD, Lx, 'ko-', lw=1, markersize=2)
    ax_y.plot(DD, Ly, 'ko-', lw=1, markersize=2)
    ax_z.plot(DD, Lz, 'ko-', lw=1, markersize=2)

    ax_x.plot(DD, Lx_avg, 'b-', lw=2)
    ax_y.plot(DD, Ly_avg, 'b-', lw=2)
    ax_z.plot(DD, Lz_avg, 'b-', lw=2)

    axs = [ax_x, ax_y, ax_z]
    labels = ['$L_x$', '$L_y$', '$L_z$']
    for i in range(len(axs)):
        ax = axs[i]
        ax.set_xlabel('DD number', fontsize=14)
        ax.set_ylabel(labels[i], fontsize=14)
        ax.axis([0,2500,-1,1])
        ax.tick_params(axis='both', which='both', direction='in', length=8, width=2, pad=5, \
          labelsize=12, top=True, right=True)
    plt.subplots_adjust(bottom=0.05, top=0.98, left=0.05, right=0.97, hspace=0.15)
    plt.savefig(args.halo + '_AM_direction_vs_DD.png')

def plot_center():
    '''Plots the time evolution of the x, y, and z components of
    the center of the halo saved in the catalog.'''

    fig = plt.figure(figsize=(20,15),dpi=500)
    ax_x = fig.add_subplot(3,1,1)
    ax_y = fig.add_subplot(3,1,2)
    ax_z = fig.add_subplot(3,1,3)

    ax_x.plot(DD_cen, x_diff, 'ko-', lw=1, markersize=2)
    ax_y.plot(DD_cen, y_diff, 'ko-', lw=1, markersize=2)
    ax_z.plot(DD_cen, z_diff, 'ko-', lw=1, markersize=2)

    axs = [ax_x, ax_y, ax_z]
    labels = ['$x_\mathrm{cen}-x_\mathrm{cen,smooth}$ [kpc]', '$y_\mathrm{cen}-y_\mathrm{cen,smooth}$ [kpc]', '$z_\mathrm{cen}-z_\mathrm{cen,smooth}$ [kpc]']
    for i in range(len(axs)):
        ax = axs[i]
        ax.set_xlabel('DD number', fontsize=14)
        ax.set_ylabel(labels[i], fontsize=14)
        ax.axis([0,2500,-50,50])
        ax.tick_params(axis='both', which='both', direction='in', length=8, width=2, pad=5, \
          labelsize=12, top=True, right=True)
    plt.subplots_adjust(bottom=0.05, top=0.98, left=0.05, right=0.97, hspace=0.15)
    plt.savefig(args.halo + '_halo_center_vs_DD.png')

if __name__ == "__main__":

    args = parse_args()
    print(args.halo)
    print(args.run)
    print(args.system)
    foggie_dir, output_dir, run_dir, code_path, trackname, haloname, spectra_dir, infofile = get_run_loc_etc(args)

    print('foggie_dir: ', foggie_dir)
    halo_c_v_name = code_path + 'halo_infos/00' + args.halo + '/' + args.run + '/halo_c_v'
    catalog_dir = code_path + 'halo_infos/00' + args.halo + '/' + args.run + '/'

    # Read in AM direction catalog
    prefix = output_dir + '/angular_momentum_catalogs/'
    DD, Lx, Ly, Lz = np.loadtxt(prefix + args.halo + '_stellar_AM_vec.cat', unpack=True, usecols=[0,1,2,3])

    # Read in halo center catalog
    halo_center = Table.read(halo_c_v_name, format='ascii')
    DD_col = halo_center['col3'][1:]
    x_col = halo_center['col4'][1:]
    y_col = halo_center['col5'][1:]
    z_col = halo_center['col6'][1:]
    DD_cen = []
    x_cen = []
    y_cen = []
    z_cen = []
    for i in range(len(DD_col)):
        if (DD_col[i][:2] == 'DD'):
            DD_cen.append(int(DD_col[i][2:]))
            x_cen.append(float(x_col[i]))
            y_cen.append(float(y_col[i]))
            z_cen.append(float(z_col[i]))
    DD_cen = np.array(DD_cen)
    x_cen = np.array(x_cen)
    y_cen = np.array(y_cen)
    z_cen = np.array(z_cen)

    # Smooth with an averaging window
    smoothed_x_cen = gaussian_filter(x_cen, 8)
    smoothed_y_cen = gaussian_filter(y_cen, 8)
    smoothed_z_cen = gaussian_filter(z_cen, 8)

    # Take difference with smoothed version so that it actually shows up in the plot
    x_diff = x_cen - smoothed_x_cen
    y_diff = y_cen - smoothed_y_cen
    z_diff = z_cen - smoothed_z_cen

    # Smooth with an averaging window
    Lx_avg = gaussian_filter(Lx, 8)
    Ly_avg = gaussian_filter(Ly, 8)
    Lz_avg = gaussian_filter(Lz, 8)

    if (args.plot):
        plot_AM()
        plot_center()

    # Save to file smoothed versions of halo center and AM direction
    f_cen = open(catalog_dir + 'halo_cen_smoothed', 'w')
    f_cen.write('# snapshot   x_cen   y_cen   z_cen\n')
    for i in range(len(DD_cen)):
        if (DD_cen[i]<100):
            DD_string = 'DD00' + str(DD_cen[i])
        elif (DD_cen[i]<1000):
            DD_string = 'DD0' + str(DD_cen[i])
        else:
            DD_string = 'DD' + str(DD_cen[i])
        f_cen.write('%s   %.3f   %.3f   %.3f\n' % (DD_string, smoothed_x_cen[i], smoothed_y_cen[i], smoothed_z_cen[i]))
    f_cen.close()

    f_AM = open(catalog_dir + 'AM_direction_smoothed', 'w')
    f_AM.write('# snapshot   Lx   Ly   Lz\n')
    for i in range(len(DD)):
        if (DD[i]<100):
            DD_string = 'DD00' + str(DD[i])
        elif (DD[i]<1000):
            DD_string = 'DD0' + str(DD[i])
        else:
            DD_string = 'DD' + str(DD[i])
        f_AM.write('%s   %.6f   %.6f   %.6f\n' % (DD_string, Lx_avg[i], Ly_avg[i], Lz_avg[i]))
    f_AM.close()
