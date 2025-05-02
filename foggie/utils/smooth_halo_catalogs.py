'''
Filename: smooth_halo_catalogs.py
Author: Cassi
Created: 4-4-22

This script reads in the halo center catalogs (created by get_halo_c_v_parallel.py) for every DD output for a
given halo, removes outliers, and fits polynomials to the points to create a smooth halo center catalog for
less jittery videos and analysis.
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
from scipy.interpolate import interp1d

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
                        help='Want to plot the halo center along with the smoothed curve? Default is no.')
    parser.set_defaults(plot=False)

    args = parser.parse_args()
    return args

def plot_center():
    '''Plots the time evolution of the x, y, and z components of
    the center of the halo saved in the catalog.'''

    fig = plt.figure(figsize=(20,15))
    ax_x = fig.add_subplot(3,1,1)
    ax_y = fig.add_subplot(3,1,2)
    ax_z = fig.add_subplot(3,1,3)

    ax_x.plot(times, x_cen_corr, 'ko-', lw=1, markersize=2)
    ax_y.plot(times, y_cen_corr, 'ko-', lw=1, markersize=2)
    ax_z.plot(times, z_cen_corr, 'ko-', lw=1, markersize=2)

    ax_x.plot(times, clipped_x_cen, 'b-', lw=2)
    ax_y.plot(times, clipped_y_cen, 'b-', lw=2)
    ax_z.plot(times, clipped_z_cen, 'b-', lw=2)

    axs = [ax_x, ax_y, ax_z]
    labels = ['$x_\mathrm{cen}$ [kpc]', '$y_\mathrm{cen}$ [kpc]', '$z_\mathrm{cen}$ [kpc]']
    for i in range(len(axs)):
        ax = axs[i]
        ax.set_xlabel('Time [Myr]', fontsize=14)
        ax.set_ylabel(labels[i], fontsize=14)
        ax.tick_params(axis='both', which='both', direction='in', length=8, width=2, pad=5, \
          labelsize=12, top=True, right=True)
    plt.subplots_adjust(bottom=0.05, top=0.98, left=0.05, right=0.97, hspace=0.15)
    plt.savefig(args.halo + '_halo_center_vs_time_clipped-and-smoothed.png')
    plt.show()

def plot_AM():
    '''Plots the time evolution of the x, y, and z components of
    the angular momentum vector saved in the catalog.'''

    fig = plt.figure(figsize=(20,15))
    ax_x = fig.add_subplot(3,1,1)
    ax_y = fig.add_subplot(3,1,2)
    ax_z = fig.add_subplot(3,1,3)

    ax_x.plot(times, Lx, 'ko-', lw=1, markersize=2)
    ax_y.plot(times, Ly, 'ko-', lw=1, markersize=2)
    ax_z.plot(times, Lz, 'ko-', lw=1, markersize=2)

    ax_x.plot(times, smooth_Lx, 'b-', lw=2)
    ax_y.plot(times, smooth_Ly, 'b-', lw=2)
    ax_z.plot(times, smooth_Lz, 'b-', lw=2)

    axs = [ax_x, ax_y, ax_z]
    labels = ['$L_x$', '$L_y$', '$L_z$']
    for i in range(len(axs)):
        ax = axs[i]
        ax.set_xlabel('Time [Myr]', fontsize=14)
        ax.set_ylabel(labels[i], fontsize=14)
        ax.tick_params(axis='both', which='both', direction='in', length=8, width=2, pad=5, \
          labelsize=12, top=True, right=True)
    plt.subplots_adjust(bottom=0.05, top=0.98, left=0.05, right=0.97, hspace=0.15)
    plt.savefig(args.halo + '_AM_vs_time_smoothed.png')
    plt.show()

def plot_vel():
    '''Plots the time evolution of the x, y, and z components of
    the halo bulk velocity saved in the catalog.'''

    fig = plt.figure(figsize=(20,15))
    ax_x = fig.add_subplot(3,1,1)
    ax_y = fig.add_subplot(3,1,2)
    ax_z = fig.add_subplot(3,1,3)

    ax_x.plot(times, vx, 'ko-', lw=1, markersize=2)
    ax_y.plot(times, vy, 'ko-', lw=1, markersize=2)
    ax_z.plot(times, vz, 'ko-', lw=1, markersize=2)

    ax_x.plot(times, smooth_vx, 'r-', lw=2)
    ax_y.plot(times, smooth_vy, 'r-', lw=2)
    ax_z.plot(times, smooth_vz, 'r-', lw=2)

    axs = [ax_x, ax_y, ax_z]
    labels = ['$v_x$', '$v_y$', '$v_z$']
    for i in range(len(axs)):
        ax = axs[i]
        ax.set_xlabel('Time [Myr]', fontsize=14)
        ax.set_ylabel(labels[i], fontsize=14)
        ax.tick_params(axis='both', which='both', direction='in', length=8, width=2, pad=5, \
          labelsize=12, top=True, right=True)
    plt.subplots_adjust(bottom=0.05, top=0.98, left=0.05, right=0.97, hspace=0.15)
    plt.savefig(args.halo + '_vel_vs_time_smoothed.png')
    plt.show()

if __name__ == "__main__":

    args = parse_args()
    print(args.halo)
    print(args.run)
    print(args.system)
    foggie_dir, output_dir, run_dir, code_path, trackname, haloname, spectra_dir, infofile = get_run_loc_etc(args)

    print('foggie_dir: ', foggie_dir)
    halo_c_v_name = code_path + 'halo_infos/00' + args.halo + '/' + args.run + '/halo_c_v'
    catalog_dir = code_path + 'halo_infos/00' + args.halo + '/' + args.run + '/'

    # Smooth halo center position

    # Read in halo center catalog
    halo_center = Table.read(halo_c_v_name, format='ascii')
    redshift_col = halo_center['col2'][1:]
    t_col = halo_center['col4'][1:]
    DD_col = halo_center['col3'][1:]
    x_col = halo_center['col5'][1:]
    y_col = halo_center['col6'][1:]
    z_col = halo_center['col7'][1:]
    times = []
    redshifts = []
    snaps = []
    x_cen = []
    y_cen = []
    z_cen = []
    for i in range(len(DD_col)):
        snaps.append(DD_col[i])
        redshifts.append(float(redshift_col[i]))
        times.append(float(t_col[i]))
        x_cen.append(float(x_col[i]))
        y_cen.append(float(y_col[i]))
        z_cen.append(float(z_col[i]))
    times = np.array(times)
    redshifts = np.array(redshifts)
    snaps = np.array(snaps)
    x_cen = np.array(x_cen)
    y_cen = np.array(y_cen)
    z_cen = np.array(z_cen)

    # Fit a polynomial to the large-scale halo path and subtract it off to make deviations more evident
    degree = 4
    coeff_x = np.polyfit(times, x_cen, degree)
    coeff_y = np.polyfit(times, y_cen, degree)
    coeff_z = np.polyfit(times, z_cen, degree)
    major_motion_x = np.zeros(len(times))+coeff_x[-1]
    major_motion_y = np.zeros(len(times))+coeff_y[-1]
    major_motion_z = np.zeros(len(times))+coeff_z[-1]
    for i in range(degree):
        major_motion_x += coeff_x[i]*times**(degree-i)
        major_motion_y += coeff_y[i]*times**(degree-i)
        major_motion_z += coeff_z[i]*times**(degree-i)

    x_cen_corr = x_cen - major_motion_x
    y_cen_corr = y_cen - major_motion_y
    z_cen_corr = z_cen - major_motion_z

    # Clip outliers in a moving window and then fit a polynomial to the path in the window
    clipped_x_cen = [[] for _ in range(len(x_cen_corr))]
    clipped_y_cen = [[] for _ in range(len(y_cen_corr))]
    clipped_z_cen = [[] for _ in range(len(z_cen_corr))]
    window = 50
    sig = 1
    small_degree = 2
    for i in range(0, len(x_cen_corr)-window+1):
        win_x = x_cen_corr[i:i+window]
        win_y = y_cen_corr[i:i+window]
        win_z = z_cen_corr[i:i+window]
        indices = np.array(range(window))
        outliers_x = np.where((win_x > np.median(win_x) + sig*np.std(win_x)) | (win_x < np.median(win_x) - sig*np.std(win_x)))[0]
        outliers_y = np.where((win_y > np.median(win_y) + sig*np.std(win_y)) | (win_y < np.median(win_y) - sig*np.std(win_y)))[0]
        outliers_z = np.where((win_z > np.median(win_z) + sig*np.std(win_z)) | (win_z < np.median(win_z) - sig*np.std(win_z)))[0]
        x_no_outliers = np.delete(win_x, outliers_x)
        indices_x = np.delete(indices, outliers_x)
        coeff_x = np.polyfit(indices_x, x_no_outliers, small_degree)
        fixed_x = np.zeros(len(indices))+coeff_x[-1]
        y_no_outliers = np.delete(win_y, outliers_y)
        indices_y = np.delete(indices, outliers_y)
        coeff_y = np.polyfit(indices_y, y_no_outliers, small_degree)
        fixed_y = np.zeros(len(indices))+coeff_y[-1]
        z_no_outliers = np.delete(win_z, outliers_z)
        indices_z = np.delete(indices, outliers_z)
        coeff_z = np.polyfit(indices_z, z_no_outliers, small_degree)
        fixed_z = np.zeros(len(indices))+coeff_z[-1]
        for j in range(small_degree):
            fixed_x += coeff_x[j]*indices**(small_degree-j)
            fixed_y += coeff_y[j]*indices**(small_degree-j)
            fixed_z += coeff_z[j]*indices**(small_degree-j)
        for j in range(window):
            clipped_x_cen[i+j].append(fixed_x[j])
            clipped_y_cen[i+j].append(fixed_y[j])
            clipped_z_cen[i+j].append(fixed_z[j])
    # For each point in the halo path, take the median of all fitted polynomials from all windows this point was in
    # If window=50, each point falls into 50 windows so this is a median of 50 fitted polynomials to this point and
    # the points around it
    for i in range(len(clipped_x_cen)):
        clipped_x_cen[i] = np.median(clipped_x_cen[i])
        clipped_y_cen[i] = np.median(clipped_y_cen[i])
        clipped_z_cen[i] = np.median(clipped_z_cen[i])

    # Finally, smooth the clipped-and-fitted halo paths
    smoothed_x_cen = gaussian_filter(clipped_x_cen, 5)
    smoothed_y_cen = gaussian_filter(clipped_y_cen, 5)
    smoothed_z_cen = gaussian_filter(clipped_z_cen, 5)

    if (args.plot):
        plot_center()

    # and add back on the major motion path
    smoothed_x_cen += major_motion_x
    smoothed_y_cen += major_motion_y
    smoothed_z_cen += major_motion_z

    # Save to file smoothed versions of halo center
    f_cen = Table(dtype=('S6', 'f8', 'f8', 'f8', 'f8', 'f8'),
            names=('snap', 'redshift', 'time', 'xc', 'yc', 'zc'))
    for i in range(len(snaps)):
        row = [snaps[i], redshifts[i], times[i], \
               smoothed_x_cen[i], smoothed_y_cen[i], smoothed_z_cen[i]]
        f_cen.add_row(row)
    ascii.write(f_cen, catalog_dir + 'halo_cen_smoothed', format='fixed_width', overwrite=True)



    # Smooth AM vector
    
    # Read in AM vector catalog
    am_table = Table.read(catalog_dir + 'angmom_table.hdf5', path='all_data')
    times = np.array(am_table['time'])
    Lx = np.array(am_table['Lx'])
    Ly = np.array(am_table['Ly'])
    Lz = np.array(am_table['Lz'])
    Lx = np.nan_to_num(Lx)
    Ly = np.nan_to_num(Ly)
    Lz = np.nan_to_num(Lz)

    # Clip outliers in a moving window and then fit a polynomial to the path in the window
    clipped_Lx = [[] for _ in range(len(Lx))]
    clipped_Ly = [[] for _ in range(len(Ly))]
    clipped_Lz = [[] for _ in range(len(Lz))]
    window = 50
    sig = 1
    small_degree = 2
    for i in range(0, len(Lx)-window+1):
        win_Lx = Lx[i:i+window]
        win_Ly = Ly[i:i+window]
        win_Lz = Lz[i:i+window]
        indices = np.array(range(window))
        outliers_Lx = np.where((win_Lx > np.median(win_Lx) + sig*np.std(win_Lx)) | (win_Lx < np.median(win_Lx) - sig*np.std(win_Lx)))[0]
        outliers_Ly = np.where((win_Ly > np.median(win_Ly) + sig*np.std(win_Ly)) | (win_Ly < np.median(win_Ly) - sig*np.std(win_Ly)))[0]
        outliers_Lz = np.where((win_Lz > np.median(win_Lz) + sig*np.std(win_Lz)) | (win_Lz < np.median(win_Lz) - sig*np.std(win_Lz)))[0]
        Lx_no_outliers = np.delete(win_Lx, outliers_Lx)
        indices_Lx = np.delete(indices, outliers_Lx)
        coeff_Lx = np.polyfit(indices_Lx, Lx_no_outliers, small_degree)
        fixed_Lx = np.zeros(len(indices))+coeff_Lx[-1]
        Ly_no_outliers = np.delete(win_Ly, outliers_Ly)
        indices_Ly = np.delete(indices, outliers_Ly)
        coeff_Ly = np.polyfit(indices_Ly, Ly_no_outliers, small_degree)
        fixed_Ly = np.zeros(len(indices))+coeff_Ly[-1]
        Lz_no_outliers = np.delete(win_Lz, outliers_Lz)
        indices_Lz = np.delete(indices, outliers_Lz)
        coeff_Lz = np.polyfit(indices_Lz, Lz_no_outliers, small_degree)
        fixed_Lz = np.zeros(len(indices))+coeff_Lz[-1]
        for j in range(small_degree):
            fixed_Lx += coeff_Lx[j]*indices**(small_degree-j)
            fixed_Ly += coeff_Ly[j]*indices**(small_degree-j)
            fixed_Lz += coeff_Lz[j]*indices**(small_degree-j)
        for j in range(window):
            clipped_Lx[i+j].append(fixed_Lx[j])
            clipped_Ly[i+j].append(fixed_Ly[j])
            clipped_Lz[i+j].append(fixed_Lz[j])
    # For each point in the halo path, take the median of all fitted polynomials from all windows this point was in
    # If window=50, each point falls into 50 windows so this is a median of 50 fitted polynomials to this point and
    # the points around it
    for i in range(len(clipped_Lx)):
        clipped_Lx[i] = np.median(clipped_Lx[i])
        clipped_Ly[i] = np.median(clipped_Ly[i])
        clipped_Lz[i] = np.median(clipped_Lz[i])

    # Finally, smooth the clipped-and-fitted halo paths
    smooth_Lx = gaussian_filter(clipped_Lx, 5)
    smooth_Ly = gaussian_filter(clipped_Ly, 5)
    smooth_Lz = gaussian_filter(clipped_Lz, 5)

    if (args.plot):
        plot_AM()

    # Save to file smoothed version of AM direction
    f_cen = Table(dtype=('S6', 'f8', 'f8', 'f8', 'f8', 'f8'),
            names=('snap', 'redshift', 'time', 'Lx', 'Ly', 'Lz'))
    for i in range(len(times)):
        row = [am_table['snap'][i], am_table['redshift'][i], times[i], \
               smooth_Lx[i], smooth_Ly[i], smooth_Lz[i]]
        f_cen.add_row(row)
    ascii.write(f_cen, catalog_dir + 'AM_direction_smoothed', format='fixed_width', overwrite=True)
    
    # Smooth bulk halo velocity

    '''# Read in velocity vector catalog
    vel_table = Table.read(output_dir + 'halo_centers/halo_00' + args.halo + '/' + args.run + '/bulk-v_table.dat', format='ascii')
    DD_col = vel_table['col2'][1:]
    time_col = vel_table['col4'][1:]
    redshift_col = vel_table['col3'][1:]
    vx_col = vel_table['col5'][1:]
    vy_col = vel_table['col6'][1:]
    vz_col = vel_table['col7'][1:]
    times = []
    redshifts = []
    vx = []
    vy = []
    vz = []
    for i in range(len(DD_col)):
        times.append(float(time_col[i]))
        redshifts.append(float(redshift_col[i]))
        vx.append(float(vx_col[i]))
        vy.append(float(vy_col[i]))
        vz.append(float(vz_col[i]))
    times = np.array(times)
    redshifts = np.array(redshifts)
    vx = np.array(vx)
    vy = np.array(vy)
    vz = np.array(vz)

    # Clip outliers in a moving window and then fit a polynomial to the path in the window
    clipped_vx = [[] for _ in range(len(vx))]
    clipped_vy = [[] for _ in range(len(vy))]
    clipped_vz = [[] for _ in range(len(vz))]
    window = 50
    sig = 1
    small_degree = 2
    for i in range(0, len(vx)-window+1):
        win_vx = vx[i:i+window]
        win_vy = vy[i:i+window]
        win_vz = vz[i:i+window]
        indices = np.array(range(window))
        outliers_vx = np.where((win_vx > np.median(win_vx) + sig*np.std(win_vx)) | (win_vx < np.median(win_vx) - sig*np.std(win_vx)))[0]
        outliers_vy = np.where((win_vy > np.median(win_vy) + sig*np.std(win_vy)) | (win_vy < np.median(win_vy) - sig*np.std(win_vy)))[0]
        outliers_vz = np.where((win_vz > np.median(win_vz) + sig*np.std(win_vz)) | (win_vz < np.median(win_vz) - sig*np.std(win_vz)))[0]
        vx_no_outliers = np.delete(win_vx, outliers_vx)
        indices_vx = np.delete(indices, outliers_vx)
        coeff_vx = np.polyfit(indices_vx, vx_no_outliers, small_degree)
        fixed_vx = np.zeros(len(indices))+coeff_vx[-1]
        vy_no_outliers = np.delete(win_vy, outliers_vy)
        indices_vy = np.delete(indices, outliers_vy)
        coeff_vy = np.polyfit(indices_vy, vy_no_outliers, small_degree)
        fixed_vy = np.zeros(len(indices))+coeff_vy[-1]
        vz_no_outliers = np.delete(win_vz, outliers_vz)
        indices_vz = np.delete(indices, outliers_vz)
        coeff_vz = np.polyfit(indices_vz, vz_no_outliers, small_degree)
        fixed_vz = np.zeros(len(indices))+coeff_vz[-1]
        for j in range(small_degree):
            fixed_vx += coeff_vx[j]*indices**(small_degree-j)
            fixed_vy += coeff_vy[j]*indices**(small_degree-j)
            fixed_vz += coeff_vz[j]*indices**(small_degree-j)
        for j in range(window):
            clipped_vx[i+j].append(fixed_vx[j])
            clipped_vy[i+j].append(fixed_vy[j])
            clipped_vz[i+j].append(fixed_vz[j])
    # For each point in the halo path, take the median of all fitted polynomials from all windows this point was in
    # If window=50, each point falls into 50 windows so this is a median of 50 fitted polynomials to this point and
    # the points around it
    for i in range(len(clipped_vx)):
        clipped_vx[i] = np.median(clipped_vx[i])
        clipped_vy[i] = np.median(clipped_vy[i])
        clipped_vz[i] = np.median(clipped_vz[i])

    # Finally, smooth the clipped-and-fitted halo paths
    smooth_vx = gaussian_filter(clipped_vx, 5)
    smooth_vy = gaussian_filter(clipped_vy, 5)
    smooth_vz = gaussian_filter(clipped_vz, 5)

    if (args.plot):
        plot_vel()

    # Save to file smoothed version of velocity
    f_cen = Table(dtype=('S6', 'f8', 'f8', 'f8', 'f8', 'f8'),
            names=('snap', 'redshift', 'time', 'vx', 'vy', 'vz'))
    for i in range(len(times)):
        row = [DD_col[i], redshifts[i], times[i], \
               smooth_vx[i], smooth_vy[i], smooth_vz[i]]
        f_cen.add_row(row)
    ascii.write(f_cen, output_dir + 'halo_centers/halo_00' + args.halo + '/' + args.run + '/bulk-v_smoothed.dat', format='fixed_width', overwrite=True)'''
