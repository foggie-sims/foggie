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
import trident
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter
from scipy.ndimage import rotate
from scipy.ndimage import shift
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

    parser.add_argument('--save_suffix', metavar='save_suffix', type=str, action='store', \
                        help='Do you want to append a string onto the names of the saved files? Default is no.')
    parser.set_defaults(save_suffix="")

    parser.add_argument('--plot', metavar='plot', type=str, action='store', \
                        help='What plot do you want? Options are:\n' + \
                        'velocity_slice         -  x-slices of the three spherical components of velocity, comparing the velocity,\n' + \
                        '                          the smoothed velocity, and the difference between the velocity and the smoothed velocity\n' + \
                        'vorticity_slice        -  x-slice of the velocity vorticity magnitude\n' + \
                        'vorticity_direction    -  2D histograms of vorticity direction split by temperature and radius\n' + \
                        'turbulent_spectrum     -  Turbulent energy power spectrum\n' + \
                        'vel_struc_func         -  Velocity structure function')

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

    args = parser.parse_args()
    return args

def velocity_slice(snap):
    '''Plots slices of radial, theta, and phi velocity fields through the center of the halo. The field,
    the smoothed field, and the difference between the field and the smoothed field are all plotted.'''

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

    vtypes = ['x', 'y', 'z']
    vlabels = ['$x$', '$y$', '$z$']
    vmins = [-500, -500, -500]
    vmaxes = [500, 500, 500]

    pix_res = float(np.min(refine_box['dx'].in_units('kpc')))  # at level 11
    lvl1_res = pix_res*2.**11.
    level = 9
    refine_res = int(3.*Rvir/(lvl1_res/(2.**level)))
    box = ds.covering_grid(level=level, left_edge=ds.halo_center_kpc-ds.arr([1.5*Rvir,1.5*Rvir,1.5*Rvir],'kpc'), dims=[refine_res, refine_res, refine_res])
    density = box['density'].in_units('g/cm**3').v
    temperature = box['temperature'].v

    for i in range(len(vtypes)):
        v = box['v' + vtypes[i] + '_corrected'].in_units('km/s').v
        smooth_v = uniform_filter(v, size=20)
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

def vsf(snap):
    '''Calculates and plots the velocity structure function for the snapshot 'snap'.'''

    Rvir = rvir_masses['radius'][rvir_masses['snapshot']==snap][0]

    if (args.load_vsf=='none'):
        if (args.system=='pleiades_cassi'):
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

    fig = plt.figure(figsize=(8,6),dpi=500)
    ax = fig.add_subplot(1,1,1)

    ax.plot(seps, vsf, 'k-', lw=2)

    ax.set_xlabel('Separation [kpc]', fontsize=14)
    ax.set_ylabel('$\\langle | \\delta v | \\rangle$ [km/s]', fontsize=14)
    ax.axis([0,120,0,300])
    ax.tick_params(axis='both', which='both', direction='in', length=8, width=2, pad=5, labelsize=18, \
      top=True, right=True)
    plt.subplots_adjust(bottom=0.12, top=0.97, left=0.12, right=0.97)
    plt.savefig(save_dir + snap + '_VSF' + save_suffix + '.png')

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
                vsf(outs[i])
        else:
            target = vsf
    else:
        sys.exit("That plot type hasn't been implemented!")

    if (args.nproc!=1):
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
            t.join()

    print(str(datetime.datetime.now()))
    print("All snapshots finished!")
