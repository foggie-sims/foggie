"""
Filename: turbulence.py
Author: Cassi
Date created: 7-27-21
Date last modified: 7-27-21
This file takes command line arguments and plots various turbulence statistics.

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
import copy
import matplotlib.colors as colors
import random

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

    parser.add_argument('--save_suffix', metavar='save_suffix', type=str, action='store', \
                        help='Do you want to append a string onto the names of the saved files? Default is no.')
    parser.set_defaults(save_suffix="")

    parser.add_argument('--plot', metavar='plot', type=str, action='store', \
                        help='What plot do you want? Options are:\n' + \
                        'velocity_slice         -  x-slices of the three spherical components of velocity, comparing the velocity,\n' + \
                        '                          the smoothed velocity, and the difference between the velocity and the smoothed velocity\n' + \
                        'vdisp_slice            -  x-slice of the velocity dispersion\n' + \
                        'vorticity_slice        -  x-slice of the velocity vorticity magnitude\n' + \
                        'vorticity_direction    -  2D histograms of vorticity direction split by temperature and radius\n' + \
                        'turbulent_spectrum     -  Turbulent energy power spectrum\n' + \
                        'vel_struc_func         -  Velocity structure function\n' + \
                        'vdisp_vs_radius        -  Velocity dispersion vs radius\n' + \
                        'vdisp_vs_mass_res      -  Datashader plot of cell-by-cell velocity dispersion vs. cell mass\n' + \
                        'vdisp_vs_spatial_res   -  Plot of average velocity dispersion vs. spatial resolution\n' + \
                        'vdisp_vs_time          -  Plot of velocity dispersion vs cosmic time\n' + \
                        'vdisp_SFR_xcorr        -  Time-delay cross-correlation between velocity dispersion and SFR')

    parser.add_argument('--region_filter', metavar='region_filter', type=str, action='store', \
                        help='Do you want to calculate turbulence statistics in different regions? Options are:\n' + \
                        '"velocity", "temperature", or "metallicity".')
    parser.set_defaults(region_filter='none')

    parser.add_argument('--nproc', metavar='nproc', type=int, action='store', \
                        help='How many processes do you want? Default is 1 ' + \
                        '(no parallelization), if multiple outputs and multiple processors are' + \
                        ' specified, code will run one output per processor.\n' + \
                        'This option only has meaning for running things on multiple snapshots.')
    parser.set_defaults(nproc=1)

    parser.add_argument('--load_vsf', metavar='load_vsf', type=str, action='store', \
                        help='If you have already calculated the VSF and saved to file and you just\n' + \
                        'want to plot, use --load_vsf to specify the save_suffix of the file to load from.')
    parser.set_defaults(load_vsf='none')

    parser.add_argument('--load_stats', dest='load_stats', action='store_true', \
                        help='If plotting vdisp_vs_radius,\n' + \
                        'do you want to load from file for plotting? This requires the files you need\n' + \
                        'to already exist. Run first without this command to make the files.')
    parser.set_defaults(load_stats=False)

    parser.add_argument('--weight', metavar='weight', type=str, action='store', \
                        help='If plotting vdisp_vs_radius\n' + \
                        'do you want to weight statistics by mass or volume? Default is mass.')
    parser.set_defaults(weight='mass')

    parser.add_argument('--filename', metavar='filename', type=str, action='store', \
                        help='What is the name of the file (after the snapshot name) to pull vdisp from?\n' + \
                        'There is no default for this, you must specify a filename, unless you are plotting\n' + \
                        'a datashader plot.')

    parser.add_argument('--time_radius', metavar='time_radius', type=float, action='store', \
                        help='If plotting vdisp_vs_time or vdisp_SFR_xcorr, at what radius do you\n' + \
                        'want to plot, in units of fractions of Rvir? Default is 0.3Rvir.')
    parser.set_defaults(time_radius=0.3)

    parser.add_argument('--copy_to_tmp', dest='copy_to_tmp', action='store_true', \
                        help='If running on pleiades, do you want to copy simulation outputs too the\n' + \
                        '/tmp directory on the run node before doing calculations? This may speed up\n' + \
                        'run time and reduce weight on IO file system. Default is no.')
    parser.set_defaults(copy_to_tmp=False)

    args = parser.parse_args()
    return args

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
        elif ('vdisp' in key):
            table[key].unit = 'km/s'
    return table

def make_table(stat_types):
    '''Makes the giant table that will be saved to file.'''

    names_list = ['redshift', 'inner_radius', 'outer_radius']
    types_list = ['f8', 'f8', 'f8']

    if (args.region_filter!='none'):
        regions_name = ['', 'low_' + args.region_filter + '_', 'mid_' + args.region_filter + '_', \
                        'high_' + args.region_filter + '_']
    else: regions_name = ['']
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

def velocity_slice(snap):
    '''Plots slices of radial, theta, and phi velocity fields through the center of the halo. The field,
    the smoothed field, and the difference between the field and the smoothed field are all plotted.'''

    masses_ind = np.where(masses['snapshot']==snap)[0]
    Menc_profile = IUS(np.concatenate(([0],masses['radius'][masses_ind])), np.concatenate(([0],masses['total_mass'][masses_ind])))
    Mvir = rvir_masses['total_mass'][rvir_masses['snapshot']==snap]
    Rvir = rvir_masses['radius'][rvir_masses['snapshot']==snap][0]

    if (args.system=='pleiades_cassi') and (foggie_dir!='/nobackupp18/mpeeples/'):
        print('Copying directory to /tmp')
        snap_dir = '/tmp/' + snap
        shutil.copytree(foggie_dir + run_dir + snap, snap_dir)
        snap_name = snap_dir + '/' + snap
    else:
        snap_name = foggie_dir + run_dir + snap + '/' + snap
    ds, refine_box = foggie_load(snap_name, trackname, do_filter_particles=False, halo_c_v_name=halo_c_v_name, gravity=True, masses_dir=masses_dir)

    vtypes = ['x', 'y', 'z']
    vlabels = ['$x$', '$y$', '$z$']
    vmins = [-500, -500, -500]
    vmaxes = [500, 500, 500]

    pix_res = float(np.min(refine_box['dx'].in_units('kpc')))  # at level 11
    lvl1_res = pix_res*2.**11.
    level = 9
    dx = lvl1_res/(2.**level)
    smooth_scale = int(25./dx)
    refine_res = int(3.*Rvir/dx)
    box = ds.covering_grid(level=level, left_edge=ds.halo_center_kpc-ds.arr([1.5*Rvir,1.5*Rvir,1.5*Rvir],'kpc'), dims=[refine_res, refine_res, refine_res])
    density = box['density'].in_units('g/cm**3').v
    temperature = box['temperature'].v

    for i in range(len(vtypes)):
        v = box['v' + vtypes[i] + '_corrected'].in_units('km/s').v
        smooth_v = uniform_filter(v, size=smooth_scale)
        sig_v = v - smooth_v
        v = np.ma.masked_where((density > cgm_density_max) & (temperature < cgm_temperature_min), v)
        smooth_v = np.ma.masked_where((density > cgm_density_max) & (temperature < cgm_temperature_min), smooth_v)
        sig_v = np.ma.masked_where((density > cgm_density_max) & (temperature < cgm_temperature_min), sig_v)
        fig = plt.figure(figsize=(23,6),dpi=500)
        ax1 = fig.add_subplot(1,3,1)
        ax2 = fig.add_subplot(1,3,2)
        ax3 = fig.add_subplot(1,3,3)
        cmap = copy.copy(mpl.cm.RdBu)
        # Need to rotate to match up with how yt plots it
        im1 = ax1.imshow(rotate(v[len(v)//2,:,:],90), cmap=cmap, norm=colors.Normalize(vmin=vmins[i], vmax=vmaxes[i]), \
                  extent=[-1.5*Rvir,1.5*Rvir,-1.5*Rvir,1.5*Rvir])
        ax1.set_xlabel('y [kpc]', fontsize=20)
        ax1.set_ylabel('z [kpc]', fontsize=20)
        ax1.tick_params(axis='both', which='both', direction='in', length=8, width=2, pad=5, labelsize=20, \
          top=True, right=True)
        cb = fig.colorbar(im1, ax=ax1, orientation='vertical', pad=0)
        cb.ax.tick_params(labelsize=16)
        cb.ax.set_ylabel(vlabels[i] + ' Velocity [km/s]', fontsize=16)
        im2 = ax2.imshow(rotate(smooth_v[len(smooth_v)//2,:,:],90), cmap=cmap, norm=colors.Normalize(vmin=vmins[i], vmax=vmaxes[i]), \
                  extent=[-1.5*Rvir,1.5*Rvir,-1.5*Rvir,1.5*Rvir])
        ax2.set_xlabel('y [kpc]', fontsize=20)
        ax2.set_ylabel('z [kpc]', fontsize=20)
        ax2.tick_params(axis='both', which='both', direction='in', length=8, width=2, pad=5, labelsize=20, \
          top=True, right=True)
        cb = fig.colorbar(im2, ax=ax2, orientation='vertical', pad=0)
        cb.ax.tick_params(labelsize=16)
        cb.ax.set_ylabel('Smoothed ' + vlabels[i] + ' Velocity [km/s]', fontsize=16)
        im3 = ax3.imshow(rotate(sig_v[len(sig_v)//2,:,:],90), cmap=cmap, norm=colors.Normalize(vmin=vmins[i], vmax=vmaxes[i]), \
                  extent=[-1.5*Rvir,1.5*Rvir,-1.5*Rvir,1.5*Rvir])
        ax3.set_xlabel('y [kpc]', fontsize=20)
        ax3.set_ylabel('z [kpc]', fontsize=20)
        ax3.tick_params(axis='both', which='both', direction='in', length=8, width=2, pad=5, labelsize=20, \
          top=True, right=True)
        cb = fig.colorbar(im3, ax=ax3, orientation='vertical', pad=0)
        cb.ax.tick_params(labelsize=16)
        cb.ax.set_ylabel(vlabels[i] + ' Velocity - Smoothed Velocity [km/s]', fontsize=16)
        plt.subplots_adjust(bottom=0.15, top=0.97, left=0.04, right=0.97, wspace=0.22)
        plt.savefig(save_dir + snap + '_' + vtypes[i] + '_velocity_slice_x' + save_suffix + '.png')

    # Delete output from temp directory if on pleiades
    if (args.system=='pleiades_cassi') and (foggie_dir!='/nobackupp18/mpeeples/'):
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

def vsf_cubeshift(snap):
    '''I DON'T THINK THIS WORKED PROPERLY. DO NOT USE.

    Calculates and plots the velocity structure function for the snapshot 'snap' by shifting the
    whole dataset by set amounts and calculating the velocity difference between shifted and un-shifted data.'''

    Rvir = rvir_masses['radius'][rvir_masses['snapshot']==snap][0]

    if (args.load_vsf=='none'):
        if (args.system=='pleiades_cassi') and (foggie_dir!='/nobackupp18/mpeeples/'):
            print('Copying directory to /tmp')
            snap_dir = '/tmp/' + snap
            shutil.copytree(foggie_dir + run_dir + snap, snap_dir)
            snap_name = snap_dir + '/' + snap
        else:
            snap_name = foggie_dir + run_dir + snap + '/' + snap
        ds, refine_box = foggie_load(snap_name, trackname, do_filter_particles=False, halo_c_v_name=halo_c_v_name, gravity=True, masses_dir=masses_dir)

        vtypes = ['x', 'y', 'z']
        vlabels = ['$x$', '$y$', '$z$']
        vmins = [-500, -500, -500]
        vmaxes = [500, 500, 500]

        pix_res = float(np.min(refine_box['dx'].in_units('kpc')))  # at level 11
        lvl1_res = pix_res*2.**11.
        level = 9
        refine_res = int(3.*Rvir/(lvl1_res/(2.**level)))
        dx = lvl1_res/(2.**level)
        box = ds.covering_grid(level=level, left_edge=ds.halo_center_kpc-ds.arr([1.5*Rvir,1.5*Rvir,1.5*Rvir],'kpc'), dims=[refine_res, refine_res, refine_res])
        vx = box['vx_corrected'].in_units('km/s').v
        vy = box['vy_corrected'].in_units('km/s').v
        vz = box['vz_corrected'].in_units('km/s').v
        density = box['density'].in_units('g/cm**3').v
        temperature = box['temperature'].v
        vx[(density > cgm_density_max) & (temperature < cgm_temperature_min)] = np.nan
        vy[(density > cgm_density_max) & (temperature < cgm_temperature_min)] = np.nan
        vz[(density > cgm_density_max) & (temperature < cgm_temperature_min)] = np.nan
        print('Fields loaded')

        seps = np.linspace(1, 101, 51)
        vsf = []
        f = open(save_dir + snap + '_VSF' + save_suffix + '.dat', 'w')
        f.write('# Separation [kpc]   VSF [km/s]\n')
        for s in range(len(seps)):
            print('Calculating separation %d/%d' % (s, len(seps)))
            sep = seps[s]
            shift_axes = [[sep, 0, 0], [-sep, 0, 0], [0, sep, 0], [0, -sep, 0], [0, 0, sep], [0, 0, -sep]]
            dv_arrays = []
            for sh in range(len(shift_axes)):
                vx_shift = shift(vx, shift_axes[sh], order=1, cval=np.nan)
                vy_shift = shift(vy, shift_axes[sh], order=1, cval=np.nan)
                vz_shift = shift(vz, shift_axes[sh], order=1, cval=np.nan)
                dv = np.sqrt((vx-vx_shift)**2. + (vy-vy_shift)**2. + (vz-vz_shift)**2.)
                dv_arrays.append(dv)
            dv_arrays = np.array(dv_arrays)
            vsf.append(np.nanmean(dv_arrays))
            f.write('%.5f             %.5f\n' % (sep*dx, vsf[-1]))
            f.flush()
        f.close()
        seps = seps*dx

    else:
        seps, vsf = np.loadtxt(save_dir + snap + '_VSF' + args.load_vsf + '.dat', usecols=[0,1], unpack=True)

    Kolmogorov_slope = []
    for i in range(len(seps)):
        Kolmogorov_slope.append(vsf[0]*(seps[i]/seps[0])**(2./3.))

    fig = plt.figure(figsize=(8,6),dpi=500)
    ax = fig.add_subplot(1,1,1)

    ax.plot(seps, vsf, 'k-', lw=2)
    ax.plot(seps, Kolmogorov_slope, 'k--', lw=2)

    ax.set_xlabel('Separation [kpc]', fontsize=14)
    ax.set_ylabel('$\\langle | \\delta v | \\rangle$ [km/s]', fontsize=14)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.axis([0.5,120,1,300])
    ax.tick_params(axis='both', which='both', direction='in', length=8, width=2, pad=5, labelsize=18, \
      top=True, right=True)
    plt.subplots_adjust(bottom=0.12, top=0.97, left=0.12, right=0.97)
    plt.savefig(save_dir + snap + '_VSF' + save_suffix + '.png')

    # Delete output from temp directory if on pleiades
    if (args.system=='pleiades_cassi') and (foggie_dir!='/nobackupp18/mpeeples/'):
        print('Deleting directory from /tmp')
        shutil.rmtree(snap_dir)

def vsf_randompoints(snap):
    '''Calculates and plots the velocity structure function for the snapshot 'snap' by drawing a large
    random number of pixel pairs and calculating velocity difference between them.'''

    global skipped_outs

    Rvir = rvir_masses['radius'][rvir_masses['snapshot']==snap][0]

    if (args.load_vsf=='none'):
        if (args.system=='pleiades_cassi') and (args.copy_to_tmp):
            print('Copying directory to /tmp')
            snap_dir = '/tmp/' + args.halo + '/' + args.run + '/' + target_dir + '/' + snap
            shutil.copytree(foggie_dir + run_dir + snap, snap_dir)
            snap_name = snap_dir + '/' + snap
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

        cgm = refine_box.cut_region("obj['density'] < %.3e" % (cgm_density_max * density_cut_factor))
        if (args.region_filter=='temperature'):
            filter = cgm['temperature'].v
            low = 10**4.8
            high = 10**6.3
        if (args.region_filter=='metallicity'):
            filter = cgm['metallicity'].in_units('Zsun').v
            low = 0.01
            high = 1.
        if (args.region_filter=='velocity'):
            filter = cgm['radial_velocity_corrected'].in_units('km/s').v
            low = -75.
            high = 75.

        x = cgm['x'].in_units('kpc').v
        y = cgm['y'].in_units('kpc').v
        z = cgm['z'].in_units('kpc').v
        vx = cgm['vx_corrected'].in_units('km/s').v
        vy = cgm['vy_corrected'].in_units('km/s').v
        vz = cgm['vz_corrected'].in_units('km/s').v
        print('Fields loaded')

        # Select random pairs of pixels
        npairs = int(len(x)/2)
        ind_A = random.sample(range(len(x)), npairs)
        ind_B = random.sample(range(len(x)), npairs)

        # Calculate separations and velocity differences
        sep = np.sqrt((x[ind_A] - x[ind_B])**2. + (y[ind_A] - y[ind_B])**2. + (z[ind_A] - z[ind_B])**2.)
        vdiff = np.sqrt((vx[ind_A] - vx[ind_B])**2. + (vy[ind_A] - vy[ind_B])**2. + (vz[ind_A] - vz[ind_B])**2.)

        if (args.region_filter!='none'):
            seps_fil = []
            vdiffs_fil = []
            for i in range(3):
                if (i==0): bool = (filter < low)
                if (i==1): bool = (filter > low) & (filter < high)
                if (i==2): bool = (filter > high)
                x_fil = x[bool]
                y_fil = y[bool]
                z_fil = z[bool]
                vx_fil = vx[bool]
                vy_fil = vy[bool]
                vz_fil = vz[bool]
                npairs_fil = int(len(x_fil)/2)
                ind_A_fil = random.sample(range(len(x_fil)), npairs_fil)
                ind_B_fil = random.sample(range(len(x_fil)), npairs_fil)
                sep_fil = np.sqrt((x_fil[ind_A_fil] - x_fil[ind_B_fil])**2. + (y_fil[ind_A_fil] - y_fil[ind_B_fil])**2. + (z_fil[ind_A_fil] - z_fil[ind_B_fil])**2.)
                vdiff_fil = np.sqrt((vx_fil[ind_A_fil] - vx_fil[ind_B_fil])**2. + (vy_fil[ind_A_fil] - vy_fil[ind_B_fil])**2. + (vz_fil[ind_A_fil] - vz_fil[ind_B_fil])**2.)
                seps_fil.append(sep_fil)
                vdiffs_fil.append(vdiff_fil)

        # Find average vdiff in bins of pixel separation and save to file
        f = open(save_dir + snap + '_VSF' + save_suffix + '.dat', 'w')
        f.write('# Separation [kpc]   VSF [km/s]')
        if (args.region_filter=='temperature'):
            f.write('   low-T VSF [km/s]   mid-T VSF[km/s]   high-T VSF [km/s]\n')
        elif (args.region_filter=='metallicity'):
            f.write('   low-Z VSF [km/s]   mid-Z VSF[km/s]   high-Z VSF [km/s]\n')
        elif (args.region_filter=='velocity'):
            f.write('   low-v VSF [km/s]   mid-v VSF[km/s]   high-v VSF [km/s]\n')
        else: f.write('\n')
        sep_bins = np.arange(0.,2.*Rvir+1,1)
        vsf = np.zeros(len(sep_bins)-1)
        if (args.region_filter!='none'):
            vsf_low = np.zeros(len(sep_bins)-1)
            vsf_mid = np.zeros(len(sep_bins)-1)
            vsf_high = np.zeros(len(sep_bins)-1)
            npairs_bins_low = np.zeros(len(sep_bins))
            npairs_bins_mid = np.zeros(len(sep_bins))
            npairs_bins_high = np.zeros(len(sep_bins))
        npairs_bins = np.zeros(len(sep_bins))
        for i in range(len(sep_bins)-1):
            npairs_bins[i] += len(sep[(sep > sep_bins[i]) & (sep < sep_bins[i+1])])
            vsf[i] += np.mean(vdiff[(sep > sep_bins[i]) & (sep < sep_bins[i+1])])
            f.write('%.5f              %.5f' % (sep_bins[i], vsf[i]))
            if (args.region_filter!='none'):
                npairs_bins_low[i] += len(seps_fil[0][(seps_fil[0] > sep_bins[i]) & (seps_fil[0] < sep_bins[i+1])])
                vsf_low[i] += np.mean(vdiffs_fil[0][(seps_fil[0] > sep_bins[i]) & (seps_fil[0] < sep_bins[i+1])])
                npairs_bins_mid[i] += len(seps_fil[1][(seps_fil[1] > sep_bins[i]) & (seps_fil[1] < sep_bins[i+1])])
                vsf_mid[i] += np.mean(vdiffs_fil[1][(seps_fil[1] > sep_bins[i]) & (seps_fil[1] < sep_bins[i+1])])
                npairs_bins_high[i] += len(seps_fil[2][(seps_fil[2] > sep_bins[i]) & (seps_fil[2] < sep_bins[i+1])])
                vsf_high[i] += np.mean(vdiffs_fil[2][(seps_fil[2] > sep_bins[i]) & (seps_fil[2] < sep_bins[i+1])])
                f.write('     %.5f           %.5f          %.5f\n' % (vsf_low[i], vsf_mid[i], vsf_high[i]))
            else:
                f.write('\n')
        f.close()
        bin_centers = sep_bins[:-1] + np.diff(sep_bins)
    else:
        if (args.region_filter!='none'):
            sep_bins, vsf, vsf_low, vsf_mid, vsf_high = np.loadtxt(save_dir + 'Tables/' + snap + '_VSF' + args.load_vsf + '.dat', unpack=True, usecols=[0,1,2,3,4])
        else:
            sep_bins, vsf = np.loadtxt(save_dir + 'Tables/' + snap + '_VSF' + args.load_vsf + '.dat', unpack=True, usecols=[0,1])
        sep_bins = np.append(sep_bins, sep_bins[-1]+np.diff(sep_bins)[-1])
        bin_centers = sep_bins[:-1] + np.diff(sep_bins)

    # Calculate expected VSF from subsonic Kolmogorov turbulence
    Kolmogorov_slope = []
    for i in range(len(bin_centers)):
        Kolmogorov_slope.append(vsf[10]*(bin_centers[i]/bin_centers[10])**(1./3.))

    # Plot
    fig = plt.figure(figsize=(8,6),dpi=500)
    ax = fig.add_subplot(1,1,1)

    ax.plot(bin_centers, vsf, 'k-', lw=2)
    ax.plot(bin_centers, Kolmogorov_slope, 'k--', lw=2)
    if (args.region_filter!='none'):
        ax.plot(bin_centers, vsf_low, 'b--', lw=2)
        ax.plot(bin_centers, vsf_mid, 'g--', lw=2)
        ax.plot(bin_centers, vsf_high, 'r--', lw=2)

    time_table = Table.read(output_dir + 'times_halo_00' + args.halo + '/' + args.run + '/time_table.hdf5', path='all_data')
    zsnap = time_table['redshift'][time_table['snap']==snap]
    ax.text(0.7,8e2,'$z=%.2f$' % (zsnap), fontsize=14, ha='left', va='top')
    ax.set_xlabel('Separation [kpc]', fontsize=14)
    ax.set_ylabel('$\\langle | \\delta v | \\rangle$ [km/s]', fontsize=14)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.axis([0.5,350,10,1000])
    ax.tick_params(axis='both', which='both', direction='in', length=8, width=2, pad=5, labelsize=18, \
      top=True, right=True)
    plt.subplots_adjust(bottom=0.12, top=0.97, left=0.12, right=0.97)
    plt.savefig(save_dir + 'Movie_frames/' + snap + '_VSF' + save_suffix + '.png')

    # Delete output from temp directory if on pleiades
    if (args.system=='pleiades_cassi') and (args.copy_to_tmp):
        print('Deleting directory from /tmp')
        shutil.rmtree(snap_dir)

    skipped_outs.remove(snap)

def vdisp_vs_radius(snap):
    '''Plots the turbulent velocity dispersion in hot, warm, and cool gas as functions of galactocentric
    radius.'''

    tablename_prefix = output_dir + 'turbulence_halo_00' + args.halo + '/' + args.run + '/Tables/'
    if (args.load_stats):
        if (args.filename!=''): filename = '_' + args.filename
        else: filename = ''
        stats = Table.read(tablename_prefix + snap + '_stats_vdisp' + filename + '.hdf5', path='all_data')
    Rvir = rvir_masses['radius'][rvir_masses['snapshot']==snap][0]

    if (not args.load_stats):
        if (args.system=='pleiades_cassi'):
            print('Copying directory to /tmp')
            snap_dir = '/tmp/' + target_dir + '/' + args.halo + '/' + args.run + '/' + snap
            if (args.copy_to_tmp):
                shutil.copytree(foggie_dir + run_dir + snap, snap_dir)
                snap_name = snap_dir + '/' + snap
            else:
                # Make a dummy directory with the snap name so the script later knows the process running
                # this snapshot failed if the directory is still there
                os.makedirs(snap_dir)
        else:
            snap_name = foggie_dir + run_dir + snap + '/' + snap
        ds, refine_box = foggie_load(snap_name, trackname, do_filter_particles=False, halo_c_v_name=halo_c_v_name, gravity=True, masses_dir=masses_dir)
        zsnap = ds.get_parameter('CosmologyCurrentRedshift')

        pix_res = float(np.min(refine_box['dx'].in_units('kpc')))  # at level 11
        lvl1_res = pix_res*2.**11.
        level = 9
        dx = lvl1_res/(2.**level)
        smooth_scale = int(25./dx)
        refine_res = int(3.*Rvir/dx)
        box = ds.covering_grid(level=level, left_edge=ds.halo_center_kpc-ds.arr([1.5*Rvir,1.5*Rvir,1.5*Rvir],'kpc'), dims=[refine_res, refine_res, refine_res])
        density = box['density'].in_units('g/cm**3').v.flatten()
        temperature = box['temperature'].v.flatten()
        radius = box['radius_corrected'].in_units('kpc').v.flatten()
        radius = radius[(density < cgm_density_max) & (temperature > cgm_temperature_min)]
        if (args.weight=='mass'):
            weights = box['cell_mass'].in_units('Msun').v.flatten()
        if (args.weight=='volume'):
            weights = box['cell_volume'].in_units('kpc**3').v.flatten()
        weights = weights[(density < cgm_density_max) & (temperature > cgm_temperature_min)]
        if (args.region_filter=='metallicity'):
            metallicity = box['metallicity'].in_units('Zsun').v.flatten()
            metallicity = metallicity[(density < cgm_density_max) & (temperature > cgm_temperature_min)]
        if (args.region_filter=='velocity'):
            rv = box['radial_velocity_corrected'].in_units('km/s').v.flatten()
            rv = rv[(density < cgm_density_max) & (temperature > cgm_temperature_min)]
            vff = box['vff'].in_units('km/s').v.flatten()
            vesc = box['vesc'].in_units('km/s').v.flatten()
            vff = vff[(density < cgm_density_max) & (temperature > cgm_temperature_min)]
            vesc = vesc[(density < cgm_density_max) & (temperature > cgm_temperature_min)]

        vx = box['vx_corrected'].in_units('km/s').v
        vy = box['vy_corrected'].in_units('km/s').v
        vz = box['vz_corrected'].in_units('km/s').v
        smooth_vx = uniform_filter(vx, size=smooth_scale)
        smooth_vy = uniform_filter(vy, size=smooth_scale)
        smooth_vz = uniform_filter(vz, size=smooth_scale)
        sig_x = (vx - smooth_vx)**2.
        sig_y = (vy - smooth_vy)**2.
        sig_z = (vz - smooth_vz)**2.
        vdisp = np.sqrt((sig_x + sig_y + sig_z)/3.).flatten()

        vdisp = vdisp[(density < cgm_density_max) & (temperature > cgm_temperature_min)]
        new_density = density[(density < cgm_density_max) & (temperature > cgm_temperature_min)]
        new_temp = temperature[(density < cgm_density_max) & (temperature > cgm_temperature_min)]
        density = new_density
        temperature = new_temp

        stats = ['vdisp']
        table = make_table(stats)
        table_pdf = make_pdf_table(stats)

        radius_list = np.linspace(0., 1.5*Rvir, 100)
        if (args.region_filter!='none'):
            vdisp_regions = []
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
            vdisp_regions.append(vdisp[temperature < 10**5])
            vdisp_regions.append(vdisp[(temperature > 10**5) & (temperature < 10**6)])
            vdisp_regions.append(vdisp[temperature > 10**6])
        elif (args.region_filter=='metallicity'):
            regions = ['_low-Z', '_mid-Z', '_high-Z']
            weights_regions.append(weights[metallicity < 0.01])
            weights_regions.append(weights[(metallicity > 0.01) & (metallicity < 1)])
            weights_regions.append(weights[metallicity > 1])
            radius_regions.append(radius[metallicity < 0.01])
            radius_regions.append(radius[(metallicity > 0.01) & (metallicity < 1)])
            radius_regions.append(radius[metallicity > 1])
            vdisp_regions.append(vdisp[metallicity < 0.01])
            vdisp_regions.append(vdisp[(metallicity > 0.01) & (metallicity < 1)])
            vdisp_regions.append(vdisp[metallicity > 1])
        elif (args.region_filter=='velocity'):
            regions = ['_low-v', '_mid-v', '_high-v']
            weights_regions.append(weights[rv < 0.5*vff])
            weights_regions.append(weights[(rv > 0.5*vff) & (rv < vesc)])
            weights_regions.append(weights[rv > vesc])
            radius_regions.append(radius[rv < 0.5*vff])
            radius_regions.append(radius[(rv > 0.5*vff) & (rv < vesc)])
            radius_regions.append(radius[rv > vesc])
            vdisp_regions.append(vdisp[rv < 0.5*vff])
            vdisp_regions.append(vdisp[(rv > 0.5*vff) & (rv < vesc)])
            vdisp_regions.append(vdisp[rv > vesc])
        else:
            regions = []

        for i in range(len(radius_list)-1):
            row = [zsnap, radius_list[i], radius_list[i+1]]
            pdf_array = []
            vdisp_shell = vdisp[(radius >= radius_list[i]) & (radius < radius_list[i+1])]
            weights_shell = weights[(radius >= radius_list[i]) & (radius < radius_list[i+1])]
            if (len(vdisp_shell)!=0.):
                quantiles = weighted_quantile(vdisp_shell, weights_shell, np.array([0.25,0.5,0.75]))
                row.append(quantiles[1])
                row.append(quantiles[2]-quantiles[0])
                avg, std = weighted_avg_and_std(vdisp_shell, weights_shell)
                row.append(avg)
                row.append(std)
                hist, bin_edges = np.histogram(vdisp_shell, weights=weights_shell, bins=(200), range=[-20, -12], density=True)
                pdf_array.append(bin_edges[:-1])
                pdf_array.append(bin_edges[1:])
                pdf_array.append(hist)
                for k in range(len(regions)):
                    vdisp_shell = vdisp_regions[k][(radius_regions[k] >= radius_list[i]) & (radius_regions[k] < radius_list[i+1])]
                    weights_shell = weights_regions[k][(radius_regions[k] >= radius_list[i]) & (radius_regions[k] < radius_list[i+1])]
                    if (len(vdisp_shell)!=0.):
                        quantiles = weighted_quantile(vdisp_shell, weights_shell, np.array([0.25,0.5,0.75]))
                        row.append(quantiles[1])
                        row.append(quantiles[2]-quantiles[0])
                        avg, std = weighted_avg_and_std(vdisp_shell, weights_shell)
                        row.append(avg)
                        row.append(std)
                        hist, bin_edges = np.histogram(vdisp_shell, weights=weights_shell, bins=(200), range=[-20, -12], density=True)
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
        table.write(tablename_prefix + snap + '_stats_vdisp' + save_suffix + '.hdf5', path='all_data', serialize_meta=True, overwrite=True)
        table_pdf.write(tablename_prefix + snap + '_stats_vdisp_pdf' + save_suffix + '.hdf5', path='all_data', serialize_meta=True, overwrite=True)

        stats = table
        print("Stats have been calculated and saved to file for snapshot " + snap + "!")
        # Delete output from temp directory if on pleiades
        if (args.system=='pleiades_cassi'):
            print('Deleting directory from /tmp')
            shutil.rmtree(snap_dir)


    radius_list = 0.5*(stats['inner_radius'] + stats['outer_radius'])
    zsnap = stats['redshift'][0]

    fig = plt.figure(figsize=(8,6), dpi=500)
    ax = fig.add_subplot(1,1,1)

    ax.plot(radius_list, stats['vdisp_avg'], 'k-', lw=2)

    ax.set_ylabel('Velocity Dispersion [km/s]', fontsize=18)
    ax.set_xlabel('Radius [kpc]', fontsize=18)
    ax.axis([0,250,0,200])
    ax.text(240, 165, '$z=%.2f$' % (zsnap), fontsize=18, ha='right', va='center')
    ax.text(240,180,halo_dict[args.halo],ha='right',va='center',fontsize=18)
    ax.tick_params(axis='both', which='both', direction='in', length=8, width=2, pad=5, labelsize=18, \
      top=True, right=True)
    ax.plot([Rvir, Rvir], [ax.get_ylim()[0], ax.get_ylim()[1]], 'k--', lw=1)
    ax.text(Rvir+3., 150, '$R_{200}$', fontsize=18, ha='left', va='center')
    plt.subplots_adjust(top=0.94,bottom=0.11,right=0.95,left=0.12)
    plt.savefig(save_dir + snap + '_vdisp_vs_r' + save_suffix + '.png')
    plt.close()

    if (args.region_filter!='none'):
        plot_colors = ["#984ea3", "#4daf4a", 'darkorange']
        table_labels = ['low_', 'mid_', 'high_']
        linestyles = ['--', '-', ':']
        labels = ['Low ' + args.region_filter, 'Mid ' + args.region_filter, 'High ' + args.region_filter]

        fig = plt.figure(figsize=(8,6), dpi=500)
        ax = fig.add_subplot(1,1,1)

        for i in range(len(plot_colors)):
            ax.plot(radius_list, stats[table_labels[i] + args.region_filter + '_vdisp_avg'], ls=linestyles[i], color=plot_colors[i], \
                    lw=2, label=labels[i])

        ax.set_ylabel('Velocity Dispersion [km/s]', fontsize=18)
        ax.set_xlabel('Radius [kpc]', fontsize=18)
        ax.axis([0,250,0,200])
        ax.text(240, 165, '$z=%.2f$' % (zsnap), fontsize=18, ha='right', va='center')
        ax.text(240,180,halo_dict[args.halo],ha='right',va='center',fontsize=18)
        ax.tick_params(axis='both', which='both', direction='in', length=8, width=2, pad=5, labelsize=18, \
          top=True, right=True)
        ax.plot([Rvir, Rvir], [ax.get_ylim()[0], ax.get_ylim()[1]], 'k--', lw=1)
        ax.text(Rvir+3., 150, '$R_{200}$', fontsize=18, ha='left', va='center')
        ax.legend(loc=2, fontsize=18, frameon=False, bbox_to_anchor=(0.1,1))
        plt.subplots_adjust(top=0.94,bottom=0.11,right=0.95,left=0.12)
        plt.savefig(save_dir + snap + '_vdisp_vs_r_regions-' + args.region_filter + save_suffix + '.png')
        plt.close()

def vdisp_vs_mass_res(snap):
    '''Plots the velocity dispersion as a function of cell mass, for all cells in a datashader plot,
    color-coded by temperature.'''

    masses_ind = np.where(masses['snapshot']==snap)[0]
    Menc_profile = IUS(np.concatenate(([0],masses['radius'][masses_ind])), np.concatenate(([0],masses['total_mass'][masses_ind])))
    Mvir = rvir_masses['total_mass'][rvir_masses['snapshot']==snap]
    Rvir = rvir_masses['radius'][rvir_masses['snapshot']==snap][0]

    # Copy output to temp directory if on pleiades
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
    else:
        snap_name = foggie_dir + run_dir + snap + '/' + snap
    ds, refine_box = foggie_load(snap_name, trackname, do_filter_particles=False, halo_c_v_name=halo_c_v_name, gravity=True, masses_dir=masses_dir)
    zsnap = ds.get_parameter('CosmologyCurrentRedshift')

    FIRE_res = np.log10(7100.)        # Pandya et al. (2021)
    Illustris_res = np.log10(8.5e4)   # IllustrisTNG website https://www.tng-project.org/about/

    colorparam = 'temperature'
    data_frame = pd.DataFrame({})
    pix_res = float(np.min(refine_box['dx'].in_units('kpc')))  # at level 11
    lvl1_res = pix_res*2.**11.
    level = 9
    dx = lvl1_res/(2.**level)
    smooth_scale = int(25./dx)
    refine_res = int(3.*Rvir/dx)
    box = ds.covering_grid(level=level, left_edge=ds.halo_center_kpc-ds.arr([1.5*Rvir,1.5*Rvir,1.5*Rvir],'kpc'), dims=[refine_res, refine_res, refine_res])
    mass = box['cell_mass'].in_units('Msun').v
    temperature = box['temperature'].v
    density = box['density'].in_units('g/cm**3').v
    vx = box['vx_corrected'].in_units('km/s').v
    vy = box['vy_corrected'].in_units('km/s').v
    vz = box['vz_corrected'].in_units('km/s').v
    smooth_vx = uniform_filter(vx, size=smooth_scale)
    smooth_vy = uniform_filter(vy, size=smooth_scale)
    smooth_vz = uniform_filter(vz, size=smooth_scale)
    sig_x = (vx - smooth_vx)**2.
    sig_y = (vy - smooth_vy)**2.
    sig_z = (vz - smooth_vz)**2.
    vdisp = np.sqrt((sig_x + sig_y + sig_z)/3.)
    data_frame['temperature'] = np.log10(temperature).flatten()
    data_frame['temp_cat'] = categorize_by_temp(data_frame['temperature'])
    data_frame.temp_cat = data_frame.temp_cat.astype('category')
    color_key = new_phase_color_key
    cat = 'temp_cat'
    vdisp_filtered = 1.0*vdisp
    vdisp_filtered[(density > cgm_density_max) & (temperature < cgm_temperature_min)] = 0.
    data_frame['vdisp'] = vdisp_filtered.flatten()
    data_frame['mass'] = np.log10(mass).flatten()
    x_range = [0., 6.]
    y_range = [0, 200]
    cvs = dshader.Canvas(plot_width=1000, plot_height=800, x_range=x_range, y_range=y_range)
    agg = cvs.points(data_frame, 'mass', 'vdisp', dshader.count_cat(cat))
    img = tf.spread(tf.shade(agg, color_key=color_key, how='eq_hist',min_alpha=40), shape='square', px=0)
    export_image(img, save_dir + snap + '_vdisp_vs_cell-mass_temperature-colored' + save_suffix + '_intermediate')
    fig = plt.figure(figsize=(10,8),dpi=500)
    ax = fig.add_subplot(1,1,1)
    image = plt.imread(save_dir + snap + '_vdisp_vs_cell-mass_temperature-colored' + save_suffix + '_intermediate.png')
    ax.imshow(image, extent=[x_range[0],x_range[1],y_range[0],y_range[1]])
    ax.set_aspect(8*abs(x_range[1]-x_range[0])/(10*abs(y_range[1]-y_range[0])))
    ax.set_xlabel('log Mass Resolution [$M_\odot$]', fontsize=20)
    ax.set_ylabel('Velocity Dispersion [km/s]', fontsize=20)
    ax.tick_params(axis='both', which='both', direction='in', length=8, width=2, pad=5, labelsize=20, \
      top=True, right=True)
    ax.text(5.75, 180, '$z=%.2f$' % (zsnap), fontsize=20, ha='right', va='center')
    #ax.text(5.75, 165, halo_dict[args.halo],ha='right',va='center',fontsize=20)
    ax.plot([FIRE_res, FIRE_res],[0,200], 'k-', lw=1)
    ax.text(FIRE_res+0.05, 25, 'FIRE', ha='left', va='center', fontsize=20)
    ax.plot([Illustris_res,Illustris_res],[0,200], 'k-', lw=1)
    ax.text(Illustris_res+0.05, 25, 'Illustris\nTNG50', ha='left', va='center', fontsize=20)
    ax2 = fig.add_axes([0.7, 0.93, 0.25, 0.06])
    cmap = create_foggie_cmap(temperature_min_datashader, temperature_max_datashader, categorize_by_temp, new_phase_color_key, log=True)
    ax2.imshow(np.flip(cmap.to_pil(), 1))
    ax2.set_xticks([50,300,550])
    ax2.set_xticklabels(['4','5','6'],fontsize=16)
    ax2.text(400, 150, 'log T [K]',fontsize=20, ha='center', va='center')
    ax2.spines["top"].set_color('white')
    ax2.spines["bottom"].set_color('white')
    ax2.spines["left"].set_color('white')
    ax2.spines["right"].set_color('white')
    ax2.set_ylim(60, 180)
    ax2.set_xlim(-10, 750)
    ax2.set_yticklabels([])
    ax2.set_yticks([])
    plt.savefig(save_dir + snap + '_vdisp_vs_cell-mass_temperature-colored' + save_suffix + '.png')
    os.system('rm ' + save_dir + snap + '_vdisp_vs_cell-mass_temperature-colored' + save_suffix + '_intermediate.png')
    plt.close()

    # Delete output from temp directory if on pleiades
    if (args.system=='pleiades_cassi'):
        print('Deleting directory from /tmp')
        shutil.rmtree(snap_dir)

def vdisp_vs_spatial_res(snap):
    '''Plots the velocity dispersion as a function of simulation spatial resolution in hot, warm,
    and cold gas regions.'''

    masses_ind = np.where(masses['snapshot']==snap)[0]
    Menc_profile = IUS(np.concatenate(([0],masses['radius'][masses_ind])), np.concatenate(([0],masses['total_mass'][masses_ind])))
    Mvir = rvir_masses['total_mass'][rvir_masses['snapshot']==snap]
    Rvir = rvir_masses['radius'][rvir_masses['snapshot']==snap][0]

    # Copy output to temp directory if on pleiades
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
    else:
        snap_name = foggie_dir + run_dir + snap + '/' + snap
    ds, refine_box = foggie_load(snap_name, trackname, do_filter_particles=False, halo_c_v_name=halo_c_v_name, gravity=True, masses_dir=masses_dir)
    zsnap = ds.get_parameter('CosmologyCurrentRedshift')

    Fielding_res = 1.4
    Li_res = 0.39
    Illustris_res = 8.

    pix_res = float(np.min(refine_box['dx'].in_units('kpc')))  # at level 11
    lvl1_res = pix_res*2.**11.
    levels = [5,6,7,8,9]
    dxs = []
    vdisp_all_list = []
    vdisp_high_list = []
    vdisp_med_list = []
    vdisp_low_list = []
    for i in range(len(levels)):
        level = levels[i]
        dx = lvl1_res/(2.**level)
        dxs.append(dx)
        smooth_scale = int(25./dx)
        refine_res = int(3.*Rvir/dx)
        box = ds.covering_grid(level=level, left_edge=ds.halo_center_kpc-ds.arr([1.5*Rvir,1.5*Rvir,1.5*Rvir],'kpc'), dims=[refine_res, refine_res, refine_res])
        temperature = box['temperature'].v
        density = box['density'].in_units('g/cm**3').v
        mass = box['cell_mass'].in_units('Msun').v
        vx = box['vx_corrected'].in_units('km/s').v
        vy = box['vy_corrected'].in_units('km/s').v
        vz = box['vz_corrected'].in_units('km/s').v
        smooth_vx = uniform_filter(vx, size=smooth_scale)
        smooth_vy = uniform_filter(vy, size=smooth_scale)
        smooth_vz = uniform_filter(vz, size=smooth_scale)
        sig_x = (vx - smooth_vx)**2.
        sig_y = (vy - smooth_vy)**2.
        sig_z = (vz - smooth_vz)**2.
        vdisp = np.sqrt((sig_x + sig_y + sig_z)/3.)
        vdisp_filtered = vdisp[(density < cgm_density_max) & (temperature > cgm_temperature_min)]
        mass_filtered = mass[(density < cgm_density_max) & (temperature > cgm_temperature_min)]
        temperature_filtered = temperature[(density < cgm_density_max) & (temperature > cgm_temperature_min)]
        vdisp_all = weighted_avg_and_std(vdisp_filtered, mass_filtered)[0]
        vdisp_high = weighted_avg_and_std(vdisp_filtered[(temperature_filtered > 10**6)], \
          mass_filtered[(temperature_filtered > 10**6)])[0]
        vdisp_med = weighted_avg_and_std(vdisp_filtered[(temperature_filtered > 10**5) & (temperature_filtered < 10**6)], \
          mass_filtered[(temperature_filtered > 10**5) & (temperature_filtered < 10**6)])[0]
        vdisp_low = weighted_avg_and_std(vdisp_filtered[(temperature_filtered < 10**5)], \
          mass_filtered[(temperature_filtered < 10**5)])[0]
        vdisp_all_list.append(vdisp_all)
        vdisp_high_list.append(vdisp_high)
        vdisp_med_list.append(vdisp_med)
        vdisp_low_list.append(vdisp_low)

    fig = plt.figure(figsize=(10,8),dpi=500)
    ax = fig.add_subplot(1,1,1)

    #ax.plot(dxs, vdisp_all_list, marker='o', ls='-', color='k', markersize=8, label='All gas')
    ax.plot(dxs, vdisp_high_list, marker='o', ls='-', color='darkorange', markersize=8, label='High temperature')
    ax.plot(dxs, vdisp_med_list, marker='o', ls='-', color="#4daf4a", markersize=8, label='Mid temperature')
    ax.plot(dxs, vdisp_low_list, marker='o', ls='-', color="#984ea3", markersize=8, label='Low temperature')

    ax.plot([Illustris_res,Illustris_res],[-5,70], 'k-', lw=1)
    ax.text(Illustris_res-0.5, 2, 'Illustris\nTNG50', ha='right', va='center', fontsize=20)
    ax.plot([Fielding_res,Fielding_res],[-5,70], 'k-', lw=1)
    ax.text(Fielding_res-0.1, 2, 'Fielding+\n2017', ha='right', va='center', fontsize=20)
    ax.plot([Li_res,Li_res],[-5,70], 'k-', lw=1)
    ax.text(Li_res+0.03, 18, 'Li+\n2020', ha='left', va='center', fontsize=20)

    ax.set_xlabel('Spatial Resolution [kpc]', fontsize=20)
    ax.set_ylabel('Velocity dispersion [km/s]', fontsize=20)
    ax.axis([0.3,20,-5,70])
    ax.set_xscale('log')
    ax.text(19, 65, '$z=%.2f$' % (zsnap), fontsize=18, ha='right', va='center')
    ax.tick_params(axis='both', which='both', direction='in', length=8, width=2, pad=5, labelsize=18, \
      top=True, right=True)
    ax.legend(loc=2, fontsize=18, bbox_to_anchor=(0.06,0.9))
    plt.savefig(save_dir + snap + '_vdisp_vs_spatial-res_temperature-colored' + save_suffix + '.png')
    plt.close()

    # Delete output from temp directory if on pleiades
    if (args.system=='pleiades_cassi'):
        print('Deleting directory from /tmp')
        shutil.rmtree(snap_dir)

def vdisp_vs_time(snaplist):
    '''Plots the velocity dispersion of hot, warm, and cool gas at 0.3 Rvir as a function of time
    for all outputs in the list 'snaplist'.'''

    tablename_prefix = output_dir + 'turbulence_halo_00' + args.halo + '/' + args.run + '/Tables/'
    time_table = Table.read(output_dir + 'times_halo_00' + args.halo + '/' + args.run + '/time_table.hdf5', path='all_data')

    plot_colors = ['darkorange', "#4daf4a", "#984ea3"]
    plot_labels = ['High temperature', 'Mid temperature', 'Low temperature']
    table_labels = ['high_temperature_', 'mid_temperature_', 'low_temperature_']
    linestyles = ['-', '-', '-']

    zlist = []
    timelist = []
    data_list = []
    for j in range(len(table_labels)):
        data_list.append([])

    for i in range(len(snaplist)):
        data = Table.read(tablename_prefix + snaplist[i] + '_' + args.filename + '.hdf5', path='all_data', format='hdf5')
        rvir = rvir_masses['radius'][rvir_masses['snapshot']==snaplist[i]]
        pos_ind = np.where(data['outer_radius']>=args.time_radius*rvir)[0][0]

        for k in range(len(table_labels)):
            data_list[k].append(data[table_labels[k] + 'vdisp_avg'][pos_ind])

        zlist.append(data['redshift'][0])
        timelist.append(time_table['time'][time_table['snap']==snaplist[i]][0]/1000.)

    fig = plt.figure(figsize=(8,6), dpi=500)
    ax = fig.add_subplot(1,1,1)
    for i in range(len(table_labels)):
        label = plot_labels[i]
        ax.plot(timelist, np.array(data_list[i]), \
                color=plot_colors[i], ls=linestyles[i], lw=2, label=label)

    ax.set_ylabel('Velocity Dispersion [km/s]', fontsize=20)

    z_sfr, sfr = np.loadtxt(code_path + 'halo_infos/00' + args.halo + '/' + args.run + '/sfr', unpack=True, usecols=[1,2], skiprows=1)
    start_ind = np.where(z_sfr<=zlist[0])[0][0]

    ax2 = ax.twiny()
    ax.tick_params(axis='both', which='both', direction='in', length=8, width=2, pad=5, labelsize=18, \
      top=False, right=False)
    ax2.tick_params(axis='x', which='both', direction='in', length=8, width=2, pad=5, labelsize=18, \
      top=True)
    ax3 = ax.twinx()
    zlist.reverse()
    timelist.reverse()
    time_func = IUS(zlist, timelist)
    timelist.reverse()
    ax.set_xlim(np.min(timelist), np.max(timelist))
    ax.set_ylim(0, 200)
    x0, x1 = ax.get_xlim()
    z_ticks = [2,1.5,1,.75,.5,.3,.2,.1,0]
    last_z = np.where(z_ticks >= zlist[0])[0][-1]
    z_ticks = z_ticks[:last_z+1]
    tick_pos = [z for z in time_func(z_ticks)]
    tick_labels = ['%.2f' % (z) for z in z_ticks]
    ax2.set_xlim(x0,x1)
    ax2.set_xticks(tick_pos)
    ax2.set_xticklabels(tick_labels)
    ax.set_xlabel('Cosmic Time [Gyr]', fontsize=18)
    ax2.set_xlabel('Redshift', fontsize=18)
    ax3.plot(time_func(np.array(z_sfr)), sfr, 'b-', lw=1)
    ax.plot([timelist[0],timelist[-1]], [-100,-100], 'b-', lw=1, label='SFR (right axis)')
    ax.text(4, 185, '$%.2f R_{200}$' % (args.time_radius), fontsize=20, ha='left', va='center')
    ax3.tick_params(axis='y', which='both', direction='in', length=8, width=2, pad=5, labelsize=18, right=True)
    ax3.set_ylim(-5,100)
    ax3.set_ylabel('SFR [$M_\odot$/yr]', fontsize=18)
    if (args.halo=='8508'):
        ax.legend(loc=1, frameon=False, fontsize=18)
    #ax.text(4,9,halo_dict[args.halo],ha='left',va='center',fontsize=18)
    plt.subplots_adjust(top=0.9, bottom=0.12, right=0.88, left=0.15)
    plt.savefig(save_dir + 'vdisp_vs_t' + save_suffix + '.pdf')
    plt.close()

    print('Plot made!')

def vdisp_SFR_xcorr(snaplist):
    '''Plots a cross-correlation between velocity dispersion at 0.3Rvir and SFR as a function of time delay between them.'''

    tablename_prefix = output_dir + 'turbulence_halo_00' + args.halo + '/' + args.run + '/Tables/'
    time_table = Table.read(output_dir + 'times_halo_00' + args.halo + '/' + args.run + '/time_table.hdf5', path='all_data')

    plot_colors = ['darkorange', "#4daf4a", "#984ea3"]
    plot_labels = ['High temperature', 'Mid temperature', 'Low temperature']
    table_labels = ['high_temperature_', 'mid_temperature_', 'low_temperature_']
    linestyles = ['-', '-', '-']

    zlist = []
    timelist = []
    data_list = []
    for j in range(len(table_labels)):
        data_list.append([])

    for i in range(len(snaplist)):
        data = Table.read(tablename_prefix + snaplist[i] + '_' + args.filename + '.hdf5', path='all_data')
        rvir = rvir_masses['radius'][rvir_masses['snapshot']==snaplist[i]]
        pos_ind = np.where(data['outer_radius']>=args.time_radius*rvir)[0][0]

        for k in range(len(table_labels)):
            data_list[k].append(data[table_labels[k] + 'vdisp_avg'][pos_ind])

        zlist.append(data['redshift'][0])
        timelist.append(time_table['time'][time_table['snap']==snaplist[i]][0]/1000.)

    z_sfr, sfr = np.loadtxt(code_path + 'halo_infos/00' + args.halo + '/' + args.run + '/sfr', unpack=True, usecols=[1,2], skiprows=1)
    snap_sfr = np.loadtxt(code_path + 'halo_infos/00' +args.halo + '/' + args.run + '/sfr', unpack=True, usecols=[0], skiprows=1, dtype=str)
    SFR_list = []
    for i in range(len(snaplist)):
        SFR_list.append(sfr[np.where(snap_sfr==snaplist[i])[0][0]])
    SFR_list = np.array(SFR_list)
    SFR_list2 = np.roll(np.array(SFR_list), 200)
    SFR_mean = np.mean(SFR_list)
    SFR_std = np.std(SFR_list)

    delay_list = np.array(range(int(len(timelist)/3)))        # Consider delay times of zero all the way
    dt = 5.38*args.output_step                                # up to a third of the full time evolution

    xcorr_list = []
    for i in range(len(table_labels)):
        xcorr_list.append([])
        mean = np.mean(data_list[i])
        std = np.std(data_list[i])
        data_list[i] = np.array(data_list[i])
        for j in range(len(delay_list)):
            xcorr_list[i].append(np.sum((SFR_list-SFR_mean)*(np.roll(data_list[i], -delay_list[j])-mean))/(SFR_std*std*len(data_list[i])))

    fig = plt.figure(figsize=(8,6), dpi=500)
    ax = fig.add_subplot(1,1,1)
    for i in range(len(table_labels)):
        label = plot_labels[i]
        ax.plot(delay_list*dt, np.array(xcorr_list[i]), \
                color=plot_colors[i], ls=linestyles[i], lw=2, label=label)

    ax.set_ylabel('Velocity dispersion x-corr with SFR', fontsize=18)
    ax.set_xlabel('Time delay [Myr]', fontsize=18)
    ax.set_xlim(0., 2000.)
    ax.set_ylim(-0.25, 1)
    xticks = np.arange(0,2100,100)
    ax.set_xticks(xticks)
    xticklabels = []
    for i in range(len(xticks)):
        if (xticks[i]%500!=0): xticklabels.append('')
        else: xticklabels.append(str(xticks[i]))
    ax.set_xticklabels(xticklabels)
    ax.text(1900, 0.9, '$%.2f R_{200}$' % (args.time_radius), fontsize=20, ha='right', va='center')
    if (args.halo=='8508'): ax.legend(loc=3, fontsize=18)
    #ax.text(200.,-.8,halo_dict[args.halo],ha='left',va='center',fontsize=18)
    ax.plot([0,2000],[0,0],'k-',lw=1)
    ax.tick_params(axis='both', which='both', direction='in', length=8, width=2, pad=5, labelsize=18)
    ax.grid(which='both',axis='both',alpha=0.25,color='k',lw=1,ls='-')
    plt.subplots_adjust(top=0.97, bottom=0.12, right=0.95, left=0.15)
    plt.savefig(save_dir + 'xcorr_vdisp-SFR_vs_delay-t' + save_suffix + '.pdf')
    plt.close()

    print('Plot made!')

def vdisp_slice(snap):
    '''Plots a slice of velocity dispersion.'''

    if (args.system=='pleiades_cassi'):
        print('Copying directory to /tmp')
        snap_dir = '/tmp/' + target_dir + '/' + args.halo + '/' + args.run + '/' + snap
        if (args.copy_to_tmp):
            shutil.copytree(foggie_dir + run_dir + snap, snap_dir)
            snap_name = snap_dir + '/' + snap
        else:
            # Make a dummy directory with the snap name so the script later knows the process running
            # this snapshot failed if the directory is still there
            os.makedirs(snap_dir)
    else:
        snap_name = foggie_dir + run_dir + snap + '/' + snap
    ds, refine_box = foggie_load(snap_name, trackname, do_filter_particles=False, halo_c_v_name=halo_c_v_name, gravity=True, masses_dir=masses_dir)
    zsnap = ds.get_parameter('CosmologyCurrentRedshift')

    masses_ind = np.where(masses['snapshot']==snap)[0]
    Menc_profile = IUS(np.concatenate(([0],masses['radius'][masses_ind])), np.concatenate(([0],masses['total_mass'][masses_ind])))
    Mvir = rvir_masses['total_mass'][rvir_masses['snapshot']==snap]
    Rvir = rvir_masses['radius'][rvir_masses['snapshot']==snap][0]

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
    mass = box['cell_mass'].in_units('Msun').v
    metallicity = box['metallicity'].in_units('Zsun').v
    x = box[('gas','x')].in_units('cm').v - ds.halo_center_kpc[0].to('cm').v
    y = box[('gas','y')].in_units('cm').v - ds.halo_center_kpc[1].to('cm').v
    z = box[('gas','z')].in_units('cm').v - ds.halo_center_kpc[2].to('cm').v
    r = box['radius_corrected'].in_units('cm').v
    radius = box['radius_corrected'].in_units('kpc').v
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
    disk_mask_expanded = disk_mask_expanded | disk_mask
    # disk_edges is a binary mask of ONLY pixels surrounding ISM regions -- nothing inside ISM regions
    disk_edges = disk_mask_expanded & ~disk_mask
    x_edges = x[disk_edges].flatten()
    y_edges = y[disk_edges].flatten()
    z_edges = z[disk_edges].flatten()

    vx = box['vx_corrected'].in_units('cm/s').v
    vy = box['vy_corrected'].in_units('cm/s').v
    vz = box['vz_corrected'].in_units('cm/s').v
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
    '''smooth_vx = gaussian_filter(vx_masked, smooth_scale)
    smooth_vy = gaussian_filter(vy_masked, smooth_scale)
    smooth_vz = gaussian_filter(vz_masked, smooth_scale)
    sig_x = (vx_masked - smooth_vx)**2.
    sig_y = (vy_masked - smooth_vy)**2.
    sig_z = (vz_masked - smooth_vz)**2.
    #vdisp = np.sqrt((sig_x + sig_y + sig_z)/3.)/1e5
    #vavg = np.sqrt((smooth_vx**2. + smooth_vy**2. + smooth_vz**2.))/1e5
    #vdisp = np.sqrt(sig_x)/1e5'''

    radius_list = np.linspace(0., 1.5*Rvir, 100)
    #vdisp = vx_masked
    vavg = np.copy(vx_masked)
    for i in range(len(radius_list)-1):
        vx_shell = vx_masked[(radius >= radius_list[i]) & (radius < radius_list[i+1])]
        vy_shell = vy_masked[(radius >= radius_list[i]) & (radius < radius_list[i+1])]
        vz_shell = vz_masked[(radius >= radius_list[i]) & (radius < radius_list[i+1])]
        mass_shell = mass[(radius >= radius_list[i]) & (radius < radius_list[i+1])]
        #vx_shell = smooth_vx[(radius >= radius_list[i]) & (radius < radius_list[i+1])]
        #vy_shell = smooth_vy[(radius >= radius_list[i]) & (radius < radius_list[i+1])]
        #vz_shell = smooth_vz[(radius >= radius_list[i]) & (radius < radius_list[i+1])]

        vx_avg = np.mean(vx_shell)
        vx_disp = np.std(vx_shell)
        vy_avg = np.mean(vy_shell)
        vy_disp = np.std(vy_shell)
        vz_avg = np.mean(vz_shell)
        vz_disp = np.std(vz_shell)

        #vdisp[(radius >= radius_list[i]) & (radius < radius_list[i+1])] = np.sqrt((vx_disp**2. + vy_disp**2. + vz_disp**2.)/3.)/1e5
        #vavg[(radius >= radius_list[i]) & (radius < radius_list[i+1])] = np.sqrt((vx_avg**2. + vy_avg**2. + vz_avg**2.))/1e5

        vavg[(radius >= radius_list[i]) & (radius < radius_list[i+1])] = vx_avg
    vdisp = np.sqrt((vx_masked - vavg)**2.)/1e5

    vdisp = np.ma.masked_where((density > cgm_density_max) & (temperature < cgm_temperature_min), vdisp)
    fig = plt.figure(figsize=(12,10),dpi=500)
    ax = fig.add_subplot(1,1,1)
    f_cmap = copy.copy(mpl.cm.get_cmap('YlGnBu'))
    #f_cmap.set_over(color='w', alpha=1.)
    # Need to rotate to match up with how yt plots it
    im = ax.imshow(rotate(vdisp[len(vdisp)//2,:,:],90), cmap=f_cmap, norm=colors.Normalize(vmin=0., vmax=250), \
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
    ax.text(1.2, 0.5, '$v_x$ dispersion [km/s]', fontsize=20, rotation='vertical', ha='center', va='center', transform=ax.transAxes)
    plt.subplots_adjust(bottom=0.08, top=0.98, left=0.12, right=0.82)
    plt.savefig(save_dir + snap + '_vx-disp_slice_x' + save_suffix + '.png')

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

    args = parse_args()
    print(args.halo)
    print(args.run)
    print(args.system)
    foggie_dir, output_dir, run_dir, code_path, trackname, haloname, spectra_dir, infofile = get_run_loc_etc(args)
    if ('feedback' in args.run):
        foggie_dir = '/nobackup/clochhaa/'
        run_dir = args.run + '/'

    # Set directory for output location, making it if necessary
    save_dir = output_dir + 'turbulence_halo_00' + args.halo + '/' + args.run + '/'
    if not (os.path.exists(save_dir)): os.system('mkdir -p ' + save_dir)

    print('foggie_dir: ', foggie_dir)
    halo_c_v_name = code_path + 'halo_infos/00' + args.halo + '/' + args.run + '/halo_c_v'
    masses_dir = code_path + 'halo_infos/00' + args.halo + '/' + args.run + '/'
    masses = Table.read(masses_dir + 'masses_z-less-2.hdf5', path='all_data')
    rvir_masses = Table.read(masses_dir + 'rvir_masses.hdf5', path='all_data')

    outs = make_output_list(args.output, output_step=args.output_step)

    if (args.save_suffix): save_suffix = '_' + args.save_suffix
    else: save_suffix = ''

    if (args.plot=='velocity_slice'):
        if (args.nproc==1):
            for i in range(len(outs)):
                velocity_slice(outs[i])
        else:
            target = velocity_slice
    elif (args.plot=='vdisp_slice'):
        if (args.nproc==1):
            for i in range(len(outs)):
                vdisp_slice(outs[i])
        else:
            target = vdisp_slice
            target_dir = 'vdisp_slice'
    elif (args.plot=='vorticity_slice'):
        if (args.nproc==1):
            for i in range(len(outs)):
                vorticity_slice(outs[i])
        else:
            target = vorticity_slice
    elif (args.plot=='vorticity_direction'):
        if (args.nproc==1):
            for i in range(len(outs)):
                vorticity_direction(outs[i])
        else:
            target = vorticity_direction
    elif (args.plot=='turbulent_spectrum'):
        if (args.nproc==1):
            for i in range(len(outs)):
                Pk_turbulence(outs[i])
        else:
            target = Pk_turbulence
    elif (args.plot=='vel_struc_func'):
        if (args.nproc==1):
            for i in range(len(outs)):
                vsf_randompoints(outs[i])
        else:
            target = vsf_randompoints
            target_dir = 'vsf_randompoints'
    elif (args.plot=='vdisp_vs_radius'):
        if (args.nproc==1):
            for i in range(len(outs)):
                vdisp_vs_radius(outs[i])
        else:
            target = vdisp_vs_radius
            target_dir = 'vdisp_vs_radius'
    elif (args.plot=='vdisp_vs_mass_res'):
        if (args.nproc==1):
            for i in range(len(outs)):
                vdisp_vs_mass_res(outs[i])
        else:
            target = vdisp_vs_mass_res
    elif (args.plot=='vdisp_vs_spatial_res'):
        if (args.nproc==1):
            for i in range(len(outs)):
                vdisp_vs_spatial_res(outs[i])
        else:
            target = vdisp_vs_spatial_res
    elif (args.plot=='vdisp_vs_time'):
        vdisp_vs_time(outs)
    elif (args.plot=='vdisp_SFR_xcorr'):
        vdisp_SFR_xcorr(outs)
    else:
        sys.exit("That plot type hasn't been implemented!")

    if (args.nproc!=1):
        skipped_outs = outs
        while (len(skipped_outs)>0):
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
                if (args.system=='pleiades_cassi') and (args.copy_to_tmp):
                    for s in range(len(snaps)):
                        if (os.path.exists('/tmp/' + args.halo + '/' + args.run + '/' + target_dir + '/' + snaps[s])):
                            print('Deleting failed %s from /tmp' % (snaps[s]))
                            shutil.rmtree('/tmp/' + args.halo + '/' + args.run + '/' + target_dir + '/' + snaps[s])
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
            if (args.system=='pleiades_cassi') and (args.copy_to_tmp):
                for s in range(len(snaps)):
                    if (os.path.exists('/tmp/' + args.halo + '/' + args.run + '/' + target_dir + '/' + snaps[s])):
                        print('Deleting failed %s from /tmp' % (snaps[s]))
                        shutil.rmtree('/tmp/' + args.halo + '/' + args.run + '/' + target_dir + '/' + snaps[s])
            outs = skipped_outs

    print(str(datetime.datetime.now()))
    print("All snapshots finished!")
