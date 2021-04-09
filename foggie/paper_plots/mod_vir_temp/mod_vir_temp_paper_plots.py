'''
Filename: mod_vir_temp_paper_plots.py
Author: Cassi
Created: 12/2/20
Last updated: 12/2/20

This file makes all the plots for the modified virial temperature paper (FOGGIE V). Specify which
plot for which halo and snapshots with command line arguments.
'''

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
import matplotlib.pyplot as plt
import matplotlib.cm as cmx
import matplotlib.colors as colors
import matplotlib.patheffects as pe
import random
from photutils.segmentation import detect_sources
from scipy.interpolate import InterpolatedUnivariateSpline as IUS
from scipy.optimize import curve_fit
import scipy.special as sse
import trident
import ast
import math
import shutil

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
    '''Parse command line arguments. Returns args object.'''

    parser = argparse.ArgumentParser(description='Makes plots for modified virial temperature paper.')

    parser.add_argument('--system', metavar='system', type=str, action='store', \
                        help='Which system are you on? Default is cassiopeia')
    parser.set_defaults(system='cassiopeia')

    parser.add_argument('--halo', metavar='halo', type=str, action='store', \
                        help='Which halo? Default is 8508 (Tempest)')
    parser.set_defaults(halo='8508')

    parser.add_argument('--run', metavar='run', type=str, action='store', \
                        help='Which run? Default is nref11c_nref9f')
    parser.set_defaults(run='nref11c_nref9f')

    parser.add_argument('--output', metavar='output', type=str, action='store', \
                        help='If you are plotting from a simulation output, which output?\n' + \
                        'Options: Specify a single output or specify a range of outputs\n' + \
                        'using commas to list individual outputs and dashes for ranges of outputs\n' + \
                        '(e.g. "RD0020-RD0025" or "DD1341,DD1353,DD1600-DD1700", no spaces!)\n' + \
                        'If the plot you are making is a plot over time or redshift, it will make one plot\n' + \
                        'over the range of outputs specified. If the plot you are making is not over time\n' + \
                        'or redshift, it will make one plot for each snapshot.')
    parser.set_defaults(output='none')

    parser.add_argument('--output_step', metavar='output_step', type=int, action='store', \
                        help='If you want to do every Nth output, this specifies N. Default: 1 (every output in specified range)')
    parser.set_defaults(output_step=1)

    parser.add_argument('--pwd', dest='pwd', action='store_true',
                        help='Just use the working directory?, Default is no')
    parser.set_defaults(pwd=False)

    parser.add_argument('--plot', metavar='plot', type=str, action='store', \
                        help='What plot do you want? Options are:\n' + \
                        'den_vel_proj           -  Projection plots of density in radial velocity bins\n' + \
                        'visualization          -  Slice plots of temperature and radial velocity, with and without filaments\n' + \
                        'mass_vs_time           -  DM, stellar, gas, and total masses within Rvir over time\n' + \
                        'velocity_PDF           -  1D PDFs of each velocity component in a radius bin near Rvir\n' + \
                        'energy_vs_time         -  energies (virial, kinetic, thermal, etc.) at Rvir over time\n' + \
                        'energy_vs_radius       -  energies (virial, kinetic, thermal, etc.) over radius\n' + \
                        'temperature_vs_time    -  temperature at Rvir over time compared to modified virial temp\n' + \
                        'temperature_vs_radius  -  temperature over radius compared to modified virial temp NOTE:' + \
                        ' requires full simulation output for the snapshot you want unless you specify --hist_from_file!\n' + \
                        'energy_SFR_xcorr       -  cross-correlation of energies at Rvir with SFR vs delay time\n' + \
                        'temp_SFR_xcorr         -  cross-correlation of mass in temp bins at Rvir with SFR vs delay time\n' + \
                        'energy_vs_radius_comp  -  compare exact vs. SIS or cumulative vs. shell in energies over radius\n' + \
                        'energy_vs_time_comp    -  compare exact vs. SIS or cumulative vs. shell in energies over time')

    parser.add_argument('--filename', metavar='filename', type=str, action='store', \
                        help='What is the filename (after the snapshot name) where appropriate data is stored?\n' + \
                        'For PDFs this must be a file containing PDFs, for energy it must be a file containing totals,\n' + \
                        'for temperature it must be a file containing PDFs.')

    parser.add_argument('--save_suffix', metavar='save_suffix', type=str, action='store', \
                        help='If you want to append a string to the end of the save file(s), what is it?\n' + \
                        'Default is nothing appended.')
    parser.set_defaults(save_suffix='')

    parser.add_argument('--Rvir_compare', dest='Rvir_compare', action='store_true', \
                        help='Do you want to compare R200 and Rvir in either a time or a radius plot?\n' + \
                        'Default is no.')
    parser.set_defaults(Rvir_compare=False)

    parser.add_argument('--hist_from_file', dest='hist_from_file', action='store_true', \
                        help='If plotting temperature_vs_radius, do you want to pull the gas temperature\n' + \
                        'histograms from a file rather than computing directly from the dataset? Do this if\n' + \
                        'you do not have the simulation outputs stored or if you want each\n' + \
                        "radial bin's histogram to be normalized. Default is not to do this.")
    parser.set_defaults(hist_from_file=False)

    parser.add_argument('--large_Rbin', dest='large_Rbin', action='store_true', \
                        help='Do you want to use a radius bin from 0.95Rvir to 1.05Rvir rather than the\n' + \
                        'default of 0.99Rvir to Rvir?')
    parser.set_defaults(large_Rbin=False)

    parser.add_argument('--nproc', metavar='nproc', type=int, action='store', \
                        help='How many processes do you want? Default is 1 ' + \
                        '(no parallelization), if multiple outputs and multiple processors are' + \
                        ' specified, code will run one output per processor.\n' + \
                        'This option only has meaning for running temperature_vs_radius on a large number of snaps.')
    parser.set_defaults(nproc=1)

    parser.add_argument('--sfr_on_mass', dest='sfr_on_mass', action='store_true', \
                        help = 'Do you want to plot the SFR on the mass vs time plot? Default is no.')
    parser.set_defaults(sfr_on_mass=False)

    parser.add_argument('--inner_r', metavar='inner_r', type=float, action='store', \
                        help='If comparing cumulative calculations to shell calculations, use this\n' + \
                        'to set an inner boundary for the cumulative calculation in units of fractional Rvir\n' + \
                        '(helps with removing galaxy disk). Default is 0 (no inner boundary).')
    parser.set_defaults(inner_r=0.)

    args = parser.parse_args()
    return args

def gauss_pos(x, mu, sig, A):
    func = A*np.exp(-(x-mu)**2/2/sig**2)
    func[x<=0.] = 0.
    return func

def double_gauss(x, mu1, sig1, A1, mu2, sig2, A2):
    func1 = A1*np.exp(-(x-mu1)**2/2/sig1**2)
    func2 = A2*np.exp(-(x-mu2)**2/2/sig2**2)
    func2[x<=0.] = 0.
    return func1 + func2

def gauss(x, mu, sig, A):
    return A*np.exp(-(x-mu)**2/2/sig**2)

def den_vel_proj(snap):
    '''Plots projections of gas density in bins of radial velocity for a given snapshot 'snap'.'''

    masses_ind = np.where(masses['snapshot']==snap)[0]
    Menc_profile = IUS(np.concatenate(([0],masses['radius'][masses_ind])), np.concatenate(([0],masses['total_mass'][masses_ind])))
    Mvir = rvir_masses['total_mass'][rvir_masses['snapshot']==snap]
    Rvir = rvir_masses['radius'][rvir_masses['snapshot']==snap][0]
    sat = Table.read(masses_dir + 'satellites.hdf5', path='all_data')

    snap_name = foggie_dir + run_dir + snap + '/' + snap
    ds, refine_box = foggie_load(snap_name, trackname, do_filter_particles=False, halo_c_v_name=halo_c_v_name, tff=True, masses_dir=masses_dir)
    cgm = ds.sphere(ds.halo_center_kpc, (2.*ds.refine_width, 'kpc')) - ds.sphere(ds.halo_center_kpc, (0.3*Rvir, 'kpc'))
    cgm = cgm.cut_region("(obj['density'] < %.2e) & (obj['temperature'] > %.2e)" % (cgm_density_max, cgm_temperature_min))

    velbins = [-1000,-200,-100,0,100,200,300,5000]
    for i in range(len(velbins)):
        if (i==len(velbins)-1):
            cgm_filtered = cgm
        else:
            cgm_filtered = cgm.cut_region("(obj['radial_velocity_corrected'] > %d) & (obj['radial_velocity_corrected'] < %d)" % (velbins[i], velbins[i+1]))

        proj = yt.ProjectionPlot(ds, 'x', 'density', center=ds.halo_center_kpc, data_source=cgm_filtered, \
          width=(2.*ds.refine_width, 'kpc'))
        proj.set_unit('density','Msun/pc**2')
        proj.set_cmap('density', density_color_map)
        proj.set_zlim('density', density_proj_min, density_proj_max)
        proj.annotate_sphere(ds.halo_center_kpc, radius=(Rvir, 'kpc'), circle_args={'color':'r'})
        proj.annotate_timestamp(corner='upper_left', redshift=True, time=True, draw_inset_box=True)
        if (i==0):
            proj.annotate_text((0.1, 0.1), 'v < %d' % (velbins[i+1]), coord_system='axis', text_args={'color':'r'})
            file_label = '_vbin' + str(i)
        elif (i==len(velbins)-2):
            proj.annotate_text((0.1, 0.1), 'v > %d' % (velbins[i]), coord_system='axis', text_args={'color':'r'})
            file_label = '_vbin' + str(i)
        elif (i==len(velbins)-1):
            file_label = ''
        else:
            proj.annotate_text((0.1, 0.1), '%d < v < %d' % (velbins[i], velbins[i+1]), coord_system='axis', text_args={'color':'r'})
            file_label = '_vbin' + str(i)
        proj.save(save_dir + snap + '_density_x_proj' + file_label + save_suffix + '.pdf')

def visualization(snap):
    '''Plots slices of temperature and radial velocity with and without the cuts that remove
    filaments for a given snapshot 'snap'.'''

    masses_ind = np.where(masses['snapshot']==snap)[0]
    Menc_profile = IUS(np.concatenate(([0],masses['radius'][masses_ind])), np.concatenate(([0],masses['total_mass'][masses_ind])))
    Mvir = rvir_masses['total_mass'][rvir_masses['snapshot']==snap]
    Rvir = rvir_masses['radius'][rvir_masses['snapshot']==snap][0]

    snap_name = foggie_dir + run_dir + snap + '/' + snap
    ds, refine_box = foggie_load(snap_name, trackname, do_filter_particles=False, halo_c_v_name=halo_c_v_name, gravity=True, masses_dir=masses_dir)
    cgm = ds.sphere(ds.halo_center_kpc, (1.25*ds.refine_width, 'kpc')) - ds.sphere(ds.halo_center_kpc, (0.3*Rvir, 'kpc'))
    #cgm = ds.sphere(ds.halo_center_kpc, (1.25*ds.refine_width, 'kpc'))
    cgm = cgm.cut_region("(obj['density'] < %.2e) & (obj['temperature'] > %.2e)" % (cgm_density_max, cgm_temperature_min))

    '''in_rvir = ds.sphere(ds.halo_center_kpc, (Rvir, 'kpc')) - ds.sphere(ds.halo_center_kpc, (0.3*Rvir, 'kpc'))
    cut_in_rvir = in_rvir.cut_region("(obj['density'] < %.2e) & (obj['temperature'] > %.2e)" % (cgm_density_max, cgm_temperature_min))
    in_rvir_nofils = cut_in_rvir.cut_region("obj['radial_velocity_corrected'] > 0.5*obj['vff']")
    print('Gas mass between 0.3Rvir and Rvir:', np.sum(in_rvir['cell_mass'].in_units('Msun')))
    print('Gas mass between 0.3Rvir and Rvir no sats:', np.sum(cut_in_rvir['cell_mass'].in_units('Msun')))
    print('Gas mass between 0.3Rvir and Rvir no fils or sats:', np.sum(in_rvir_nofils['cell_mass'].in_units('Msun')))

    print('Volume between 0.3Rvir and Rvir:', np.sum(in_rvir['cell_volume'].in_units('kpc**3')))
    print('Volume between 0.3Rvir and Rvir no sats:', np.sum(cut_in_rvir['cell_volume'].in_units('kpc**3')))
    print('Volume between 0.3Rvir and Rvir no fils or sats:', np.sum(in_rvir_nofils['cell_volume'].in_units('kpc**3')))'''

    slc = yt.SlicePlot(ds, 'x', 'temperature', center=ds.halo_center_kpc, data_source=cgm, \
      width=(1.25*ds.refine_width, 'kpc'), fontsize=30)
    slc.set_cmap('temperature', sns.blend_palette(('salmon', "#984ea3", "#4daf4a", "#ffe34d", 'darkorange'), as_cmap=True))
    slc.set_zlim('temperature', 1e4,1e7)
    slc.annotate_sphere(ds.halo_center_kpc, radius=(Rvir, 'kpc'), circle_args={'color':'w'})
    #slc.annotate_text((100, -160), '$R_{200}$', coord_system='plot')
    slc.annotate_timestamp(corner='upper_left', redshift=True, time=True, draw_inset_box=True)
    slc.save(save_dir + snap + '_temperature_x_slice' + save_suffix + '.pdf')

    slc = yt.SlicePlot(ds, 'x', 'radial_velocity_corrected', center=ds.halo_center_kpc, data_source=cgm, \
      width=(1.25*ds.refine_width, 'kpc'), fontsize=30)
    slc.set_cmap('radial_velocity_corrected', 'RdBu')
    slc.set_zlim('radial_velocity_corrected', -200, 200)
    slc.annotate_sphere(ds.halo_center_kpc, radius=(Rvir, 'kpc'), circle_args={'color':'k'})
    slc.annotate_timestamp(corner='upper_left', redshift=True, time=True, draw_inset_box=True)
    slc.save(save_dir + snap + '_radial_velocity_x_slice' + save_suffix + '.pdf')

    cgm_fils_removed = cgm.cut_region("obj['radial_velocity_corrected'] > 0.5*obj['vff']")

    slc = yt.SlicePlot(ds, 'x', 'temperature', center=ds.halo_center_kpc, data_source=cgm_fils_removed, \
      width=(1.25*ds.refine_width, 'kpc'), fontsize=30)
    slc.set_cmap('temperature', sns.blend_palette(('salmon', "#984ea3", "#4daf4a", "#ffe34d", 'darkorange'), as_cmap=True))
    slc.set_zlim('temperature', 1e4,1e7)
    slc.annotate_sphere(ds.halo_center_kpc, radius=(Rvir, 'kpc'), circle_args={'color':'k'})
    slc.annotate_timestamp(corner='upper_left', redshift=True, time=True, draw_inset_box=True)
    slc.save(save_dir + snap + '_temperature_x_slice_filaments-removed' + save_suffix + '.pdf')

    slc = yt.SlicePlot(ds, 'x', 'radial_velocity_corrected', center=ds.halo_center_kpc, data_source=cgm_fils_removed, \
      width=(1.25*ds.refine_width, 'kpc'), fontsize=30)
    slc.set_cmap('radial_velocity_corrected', 'RdBu')
    slc.set_zlim('radial_velocity_corrected', -200, 200)
    slc.annotate_sphere(ds.halo_center_kpc, radius=(Rvir, 'kpc'), circle_args={'color':'k'})
    slc.annotate_timestamp(corner='upper_left', redshift=True, time=True, draw_inset_box=True)
    slc.save(save_dir + snap + '_radial_velocity_x_slice_filaments-removed' + save_suffix + '.pdf')

    print('All plots made!')

def mass_vs_time(snaplist):
    '''Plots DM, stellar, gas, and total (Mvir) masses within Rvir as a function of time.'''

    time_table = Table.read(output_dir + 'times_halo_00' + args.halo + '/' + args.run + '/time_table.hdf5', path='all_data')

    gas_masses = []
    stars_masses = []
    DM_masses = []
    tot_masses = []
    zlist = []
    timelist = []
    for i in range(len(snaplist)):
        snap = snaplist[i]
        rvir = rvir_masses['radius'][rvir_masses['snapshot']==snap][0]
        rind = np.where(masses['radius'][masses['snapshot']==snap]<=rvir)[0][-1]
        gas_masses.append(masses['gas_mass'][masses['snapshot']==snap][rind])
        stars_masses.append(masses['stars_mass'][masses['snapshot']==snap][rind])
        DM_masses.append(masses['dm_mass'][masses['snapshot']==snap][rind])
        tot_masses.append(masses['total_mass'][masses['snapshot']==snap][rind])
        zlist.append(rvir_masses['redshift'][rvir_masses['snapshot']==snap][0])
        timelist.append(time_table['time'][time_table['snap']==snap][0]/1000.)
    print(tot_masses[-1],DM_masses[-1],stars_masses[-1],gas_masses[-1],rvir,zlist[-1])

    gas_masses = np.array(gas_masses)
    stars_masses = np.array(stars_masses)
    DM_masses = np.array(DM_masses)
    tot_masses = np.array(tot_masses)

    zlist.reverse()
    timelist.reverse()
    time_func = IUS(zlist, timelist)
    timelist.reverse()
    timelist = np.array(timelist).flatten()
    zlist = np.array(zlist)

    fig = plt.figure(figsize=(8,6),dpi=500)
    ax = fig.add_subplot(1,1,1)

    ax.plot(timelist, np.log10(tot_masses), 'k-', lw=2, label='Total mass')
    ax.plot(timelist, np.log10(DM_masses), 'r--', lw=2, label='Dark matter mass')
    ax.plot(timelist, np.log10(stars_masses), 'b:', lw=2, label='Stellar mass')
    ax.plot(timelist, np.log10(gas_masses), 'g-.', lw=2, label='Gas mass')

    if (args.sfr_on_mass):
        z_sfr, sfr = np.loadtxt(code_path + 'halo_infos/00' + args.halo + '/' + args.run + '/sfr', unpack=True, usecols=[1,2], skiprows=1)
        t_sfr = time_func(z_sfr)

    ax.set_xlim(np.min(timelist), np.max(timelist))
    if (args.halo=='2392'):
        ax.set_ylim(10,13)
    elif (args.halo=='5036'):
        ax.set_ylim(9.5,12.5)
    else:
        ax.set_ylim(9, 12)
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
    ax.set_ylabel('log Mass within $R_{200}$ [$M_\odot$]', fontsize=18)
    if (args.halo=='5036'):
        ax.text(4.,9.75,halo_dict[args.halo], ha='left', va='center', fontsize=18)
    else:
        ax.text(4.,9.25, halo_dict[args.halo], ha='left', va='center', fontsize=18)
    if (args.sfr_on_mass):
        ax3 = ax.twinx()
        ax3.plot(t_sfr, sfr, 'k-', lw=1)
        ax.plot([timelist[0],timelist[-1]], [0,0], 'k-', lw=1, label='SFR (right axis)')
        ax3.tick_params(axis='y', which='both', direction='in', length=8, width=2, pad=5, labelsize=18, right=True)
        ax3.set_ylim(-5,200)
        ax3.set_ylabel('SFR [$M_\odot$/yr]', fontsize=18)
        plt.subplots_adjust(left=0.13, bottom=0.12, right=0.87, top=0.89)
    else:
        plt.subplots_adjust(left=0.13, bottom=0.12, right=0.96, top=0.89)
    if (args.halo=='8508'): ax.legend(loc=4, fontsize=18, frameon=False)
    #ax.legend(loc=2, fontsize=14, frameon=False, ncol=2)
    plt.savefig(save_dir + 'mass_vs_t' + save_suffix + '.pdf')
    plt.close()

    print('Plot made!')

def velocity_PDF(snap):
    '''Plots PDFs of the three velocity components in a given radius bin for a given snapshot.'''

    stats_dir = output_dir + 'stats_halo_00' + args.halo + '/' + args.run + '/'
    stats_data = Table.read(stats_dir + '/Tables/' + snap + '_' + args.filename.replace('_pdf', '') + '.hdf5', path='all_data')
    pdf_data = Table.read(stats_dir + '/Tables/' + snap + '_' + args.filename + '.hdf5', path='all_data')
    rvir = rvir_masses['radius'][rvir_masses['snapshot']==snap]
    rvir_ind = np.where(stats_data['inner_radius']<=rvir)[0][-1]
    inds = np.where(pdf_data['inner_radius']==stats_data['inner_radius'][rvir_ind])[0]

    fig = plt.figure(figsize=(16,4),dpi=500)
    components_list = ['theta_velocity','phi_velocity','radial_velocity']
    for j in range(len(components_list)):
        comp = components_list[j]
        ax = fig.add_subplot(1,3,j+1)
        data = 0.5 * (pdf_data['lower_' + comp][inds] + pdf_data['upper_' + comp][inds])
        hist_data = pdf_data['net_' + comp + '_pdf'][inds]
        ax.plot(data, hist_data, 'k-', lw=2, label='PDF')
        ax.set_ylabel('PDF', fontsize=18)
        if (j<2): ax.set_xlim(-200,200)
        else: ax.set_xlim(-200,500)
        ax.tick_params(axis='both', which='both', direction='in', length=8, width=2, pad=5, labelsize=18, \
                       top=True, right=True)
        x_range = ax.get_xlim()
        y_range = ax.get_ylim()
        if (comp=='radial_velocity'):
            vr_mu = stats_data['net_radial_velocity_mu'][rvir_ind]
            vr_sig = stats_data['net_radial_velocity_sig'][rvir_ind]
            vr_A1 = stats_data['net_radial_velocity_A1'][rvir_ind]
            vr_A2 = stats_data['net_radial_velocity_A2'][rvir_ind]
            vtan_sig = stats_data['net_tangential_velocity_sig'][rvir_ind]
            fit = double_gauss(data, 0., vtan_sig, vr_A2, vr_mu, vr_sig, vr_A1)
            fit1 = gauss_pos(data, vr_mu, vr_sig, vr_A1)
            fit2 = gauss(data, 0., vtan_sig, vr_A2)
            ax.plot(data, fit, 'b--', lw=2, label='Best fit')
            ax.plot(data, fit1, 'b:', lw=1)
            ax.plot(data, fit2, 'b:', lw=1)
            ax.plot([vr_mu, vr_mu], [y_range[0], y_range[1]], 'b:', lw=1)
            ax.plot([vr_mu - vr_sig, vr_mu + vr_sig], [0.1*(y_range[1]-y_range[0])+y_range[0],0.1*(y_range[1]-y_range[0])+y_range[0]], 'b:', lw=1)
            xloc = 0.8*(x_range[1]-x_range[0])+x_range[0]
            yloc = 0.7*(y_range[1]-y_range[0])+y_range[0]
            ax.text(xloc, yloc, '$\mu_r=%.1f$\n$\sigma_r=%.1f$\n$\sigma_\mathrm{tan}=%.1f$' % (vr_mu, vr_sig, vtan_sig), va='top', ha='center', fontsize=18)
            xlabel = 'Radial Velocity [km/s]'
        else:
            v_mu = stats_data['net_' + comp + '_mu'][rvir_ind]
            v_sig = stats_data['net_' + comp + '_sig'][rvir_ind]
            v_A = stats_data['net_' + comp + '_A'][rvir_ind]
            fit = gauss(data, v_mu, v_sig, v_A)
            ax.plot(data, fit, 'b--', lw=2, label='Best fit')
            ax.plot([v_mu, v_mu], [y_range[0], y_range[1]], 'b:', lw=1)
            ax.plot([v_mu - v_sig, v_mu + v_sig], [0.1*(y_range[1]-y_range[0])+y_range[0],0.1*(y_range[1]-y_range[0])+y_range[0]], 'b:', lw=1)
            xloc = 0.8*(x_range[1]-x_range[0])+x_range[0]
            yloc = 0.7*(y_range[1]-y_range[0])+y_range[0]
            ax.text(xloc, yloc, '$\mu=%.1f$\n$\sigma=%.1f$' % (v_mu, v_sig), va='top', ha='center', fontsize=18)
            if (j==0):
                xlabel = '$\\theta$ Velocity [km/s]'
                xloc = 0.2*(x_range[1]-x_range[0])+x_range[0]
                yloc = 0.85*(y_range[1]-y_range[0])+y_range[0]
                ax.text(xloc, yloc, '$z=%.2f$' % (stats_data['redshift'][0]), fontsize=18, ha='center', va='center')
            if (j==1): xlabel = '$\\phi$ Velocity [km/s]'

        if (j==0): ax.legend(loc=1, frameon=False, fontsize=18)
        ax.set_xlabel(xlabel, fontsize=18)
        ax.set_ylim(y_range)

    plt.subplots_adjust(left=0.07, bottom=0.15, right=0.98, top=0.97, wspace=0.35)
    plt.savefig(save_dir + snap + '_velocity_PDFs' + save_suffix + '.pdf')
    plt.close()

    print('Plot made!')

def energy_vs_time(snaplist):
    '''Plots various types of energies (virial, kinetic, thermal) at Rvir as a function of time
    for all outputs in the list 'snaplist'.'''

    tablename_prefix = output_dir + 'totals_halo_00' + args.halo + '/' + args.run + '/Tables/'
    time_table = Table.read(output_dir + 'times_halo_00' + args.halo + '/' + args.run + '/time_table.hdf5', path='all_data')

    if (args.Rvir_compare):
        plot_colors = ['r', 'r']
        plot_labels = ['Virial', 'Thermal-only Virial']
        table_labels = ['virial_energy', 'virial_energy2']
        linewidths = [3, 1]
        linestyles = ['-', '-']
    else:
        plot_colors = ['r', 'r', 'g', 'b']
        plot_labels = ['Virial', 'Thermal-only Virial', 'KE$_\mathrm{th}$', 'KE$_\mathrm{nt}$']
        table_labels = ['virial_energy', 'virial_energy2', 'thermal_energy', 'kinetic_energy']
        linestyles = ['-', '-', '--', ':']
        linewidths = [3, 1, 2, 2]

    zlist = []
    timelist = []
    data_list = []
    data_list2 = []
    data_list_largeR = []
    for j in range(len(table_labels)):
        data_list.append([])
        if (args.large_Rbin):
            data_list_largeR.append([])
        if (args.Rvir_compare):
            data_list2.append([])

    for i in range(len(snaplist)):
        data = Table.read(tablename_prefix + snaplist[i] + '_' + args.filename + '.hdf5', path='all_data')
        rvir = rvir_masses['radius'][rvir_masses['snapshot']==snaplist[i]]
        mvir = rvir_masses['total_mass'][rvir_masses['snapshot']==snaplist[i]]
        pos_ind = np.where(data['outer_radius']>=rvir)[0]
        if (args.large_Rbin):
            pos_ind_low = np.where(data['inner_radius']>=0.95*rvir)[0][0]
            pos_ind_high = np.where(data['outer_radius']>=1.05*rvir)[0][0]
        if (len(pos_ind)==0):
            pos_ind = len(data['outer_radius'])-1
        else:
            pos_ind = pos_ind[0]
        mgas = np.cumsum(data['net_mass'])[pos_ind]
        if (args.Rvir_compare):
            rvir2 = rvir_masses2['radius'][rvir_masses2['snapshot']==snaplist[i]]
            pos_ind2 = np.where(data['outer_radius']>rvir2)[0]
            if (len(pos_ind2)==0):
                pos_ind2 = len(data['outer_radius'])-1
            else:
                pos_ind2 = pos_ind2[0]

        for k in range(len(table_labels)):
            if (table_labels[k]=='virial_energy2'):
                data_list[k].append((2.*(data['net_thermal_energy'][pos_ind]) + 3./2.*data['net_potential_energy'][pos_ind])/-data['net_potential_energy'][pos_ind])
                if (args.large_Rbin):
                    data_list_largeR[k].append((2.*(np.sum(data['net_thermal_energy'][pos_ind_low:pos_ind_high])) + 3./2.*np.sum(data['net_potential_energy'][pos_ind_low:pos_ind_high]))/-np.sum(data['net_potential_energy'][pos_ind_low:pos_ind_high]))
                if (args.Rvir_compare):
                    data_list2[k].append((2.*(data['net_thermal_energy'][pos_ind2]) + 3./2.*data['net_potential_energy'][pos_ind2])/-data['net_potential_energy'][pos_ind2])
            else:
                data_list[k].append(data['net_' + table_labels[k]][pos_ind]/-data['net_potential_energy'][pos_ind])
                if (args.large_Rbin):
                    data_list_largeR[k].append(np.sum(data['net_' + table_labels[k]][pos_ind_low:pos_ind_high])/-np.sum(data['net_potential_energy'][pos_ind_low:pos_ind_high]))
                if (args.Rvir_compare):
                    data_list2[k].append(data['net_' + table_labels[k]][pos_ind2]/-data['net_potential_energy'][pos_ind2])

        zlist.append(data['redshift'][0])
        timelist.append(time_table['time'][time_table['snap']==snaplist[i]][0]/1000.)

    fig = plt.figure(figsize=(8,6), dpi=500)
    ax = fig.add_subplot(1,1,1)
    for i in range(len(table_labels)):
        label = plot_labels[i]
        if (args.large_Rbin):
            ax.plot(timelist, np.array(data_list_largeR[i]), \
                    color=plot_colors[i], ls=linestyles[i], lw=linewidths[i], label=label)
        else:
            ax.plot(timelist, np.array(data_list[i]), \
                    color=plot_colors[i], ls=linestyles[i], lw=linewidths[i], label=label)
        if (args.Rvir_compare):
            ax.plot(timelist, np.array(data_list2[i]), \
                    color=plot_colors[i], ls='--', lw=linewidths[i], label='_nolabel_')
    if (args.Rvir_compare):
        ax.plot([-100],[-100], color=plot_colors[i], ls='-', lw=linewidths[i], label='$R_{200}$')
        ax.plot([-100],[-100], color=plot_colors[i], ls='--', lw=linewidths[i], label='$R_\mathrm{vir}$')
        ax.set_ylabel('Energy at $R$ / PE($R$)', fontsize=18)
    else:
        ax.set_ylabel('Energy at $R_{200}$ / PE($R_{200}$)', fontsize=18)

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
    ax.set_ylim(-3, 10)
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
    ax3.plot(time_func(np.array(z_sfr)), sfr, 'k-', lw=1)
    ax.plot([timelist[0],timelist[-1]], [0,0], 'k-', lw=1, label='SFR (right axis)')
    ax3.tick_params(axis='y', which='both', direction='in', length=8, width=2, pad=5, labelsize=18, right=True)
    ax3.set_ylim(-5,100)
    ax3.set_ylabel('SFR [$M_\odot$/yr]', fontsize=18)
    if (args.halo=='8508'):
        ax.legend(loc=1, frameon=False, fontsize=18)
    ax.text(4,9,halo_dict[args.halo],ha='left',va='center',fontsize=18)
    plt.subplots_adjust(top=0.9, bottom=0.12, right=0.88, left=0.1)
    plt.savefig(save_dir + 'all_energies_vs_t' + save_suffix + '.pdf')
    plt.close()

    print('Plot made!')

def energy_vs_radius(snap):
    '''Plots the various types of energy as a function of radius for given snapshot 'snap'.'''

    tablename_prefix = output_dir + 'totals_halo_00' + args.halo + '/' + args.run + '/Tables/'
    totals = Table.read(tablename_prefix + snap + '_' + args.filename + '.hdf5', path='all_data')
    Rvir = rvir_masses['radius'][rvir_masses['snapshot']==snap]
    if (args.Rvir_compare):
        Rvir2 = rvir_masses2['radius'][rvir_masses2['snapshot']==snap]

    plot_colors = ['r', 'r', 'g', 'b']
    plot_labels = ['Virial', 'Thermal-only Virial', 'KE$_\mathrm{th}$', 'KE$_\mathrm{nt}$']
    table_labels = ['virial_energy', 'virial_energy2', 'thermal_energy', 'kinetic_energy']
    linestyles = ['-', '-', '--', ':']
    linewidths = [3, 1, 2, 2]

    radius_list = 0.5*(totals['inner_radius'] + totals['outer_radius'])
    zsnap = totals['redshift'][0]

    fig = plt.figure(figsize=(8,6), dpi=500)
    ax = fig.add_subplot(1,1,1)

    for i in range(len(plot_colors)):
        label = plot_labels[i]
        if (table_labels[i]=='virial_energy2'):
            vir_en = 2.*totals['net_thermal_energy'] + 3./2.*totals['net_potential_energy']
            ax.plot(radius_list, vir_en/(-totals['net_potential_energy']), color=plot_colors[i], ls=linestyles[i], lw=linewidths[i], label=label)
        else:
            ax.plot(radius_list, totals['net_' + table_labels[i]]/(-totals['net_potential_energy']), \
              color=plot_colors[i], ls=linestyles[i], lw=linewidths[i], label=label)

    ax.set_ylabel('Energy / PE', fontsize=18)
    ax.set_xlabel('Radius [kpc]', fontsize=18)
    ax.axis([0,250,-1,1])
    ax.text(15, 0.85, '$z=%.2f$' % (zsnap), fontsize=18, ha='left', va='center')
    ax.text(15,0.7,halo_dict[args.halo],ha='left',va='center',fontsize=18)
    ax.tick_params(axis='both', which='both', direction='in', length=8, width=2, pad=5, labelsize=18, \
      top=True, right=True)
    ax.plot([ax.get_xlim()[0], ax.get_xlim()[1]], [0,0], 'k-', lw=1)
    ax.plot([Rvir, Rvir], [ax.get_ylim()[0], ax.get_ylim()[1]], 'k--', lw=2)
    if (args.Rvir_compare):
        ax.text(Rvir-3., -0.8, '$R_{200}$', fontsize=18, ha='right', va='center')
        ax.plot([Rvir2, Rvir2], [ax.get_ylim()[0], ax.get_ylim()[1]], 'k--', lw=2)
        ax.text(Rvir2-3., -0.8, '$R_\mathrm{vir}$', fontsize=18, ha='right', va='center')
    else:
        ax.text(Rvir-3., -0.8, '$R_{200}$', fontsize=18, ha='right', va='center')
    if (args.halo=='8508'): ax.legend(loc=3, frameon=False, fontsize=18)
    plt.subplots_adjust(top=0.94,bottom=0.11,right=0.95,left=0.15)
    plt.savefig(save_dir + snap + '_all_energies_vs_r' + save_suffix + '.pdf')
    plt.close()

def temperature_vs_time(snaplist):
    '''Plots a 2D histogram of temperature at Rvir vs. cosmic time, compared to the standard and
    modified virial temperatures.'''

    stats_dir = output_dir + 'stats_halo_00' + args.halo + '/' + args.run + '/'
    time_table = Table.read(output_dir + 'times_halo_00' + args.halo + '/' + args.run + '/time_table.hdf5', path='all_data')

    zlist = []
    timelist = []
    weights_data = []
    max_weights = []
    cassiT_list1 = []
    cassiT_list2 = []
    Tvir_list = []
    if (args.Rvir_compare):
        Tvir_list2 = []
        cassiT_list1_2 = []
        cassiT_list2_2 = []
        weights_data2 = []
        max_weights2 = []
    for i in range(len(snaplist)):
        timelist.append(time_table['time'][time_table['snap']==snaplist[i]][0]/1000.)
        stats_data = Table.read(stats_dir + '/Tables/' + snaplist[i] + '_' + args.filename.replace('_pdf', '') + '.hdf5', path='all_data')
        pdf_data = Table.read(stats_dir + '/Tables/' + snaplist[i] + '_' + args.filename + '.hdf5', path='all_data')

        zsnap = pdf_data['redshift'][0]
        zlist.append(zsnap)

        masses_ind = np.where(masses['snapshot']==snaplist[i])[0]
        Menc_profile = IUS(np.concatenate(([0],masses['radius'][masses_ind])), np.concatenate(([0],masses['total_mass'][masses_ind])))
        Mvir = rvir_masses['total_mass'][rvir_masses['snapshot']==snaplist[i]][0]
        if (args.Rvir_compare):
            Mvir2 = rvir_masses2['total_mass'][rvir_masses2['snapshot']==snaplist[i]][0]

        rvir = rvir_masses['radius'][rvir_masses['snapshot']==snaplist[i]]
        rvir_ind = np.where(stats_data['outer_radius']>=rvir)[0]
        if (len(rvir_ind)==0):
            rvir_ind = len(stats_data['outer_radius'])-1
        else:
            rvir_ind = rvir_ind[0]
        inds = np.where(pdf_data['inner_radius']==stats_data['inner_radius'][rvir_ind])[0]
        if (args.Rvir_compare):
            rvir2 = rvir_masses2['radius'][rvir_masses2['snapshot']==snaplist[i]]
            rvir_ind2 = np.where(stats_data['outer_radius']>rvir2)[0]
            if (len(rvir_ind2)==0):
                rvir_ind2 = len(stats_data['outer_radius'])-1
            else:
                rvir_ind2 = rvir_ind2[0]
            inds2 = np.where(pdf_data['inner_radius']==stats_data['inner_radius'][rvir_ind2])[0]

        Tvir = (mu*mp/kB)*(1./2.*G*Mvir*gtoMsun)/(rvir*1000*cmtopc)
        Tvir_list.append(Tvir)
        vesc = np.sqrt(2.*G*Mvir*gtoMsun/(rvir*1000.*cmtopc))/1e5
        if (args.Rvir_compare):
            Tvir2 = (mu*mp/kB)*(1./2.*G*Mvir2*gtoMsun)/(rvir2*1000*cmtopc)
            Tvir_list2.append(Tvir2)
            vesc2 = np.sqrt(2.*G*Mvir2*gtoMsun/(rvir2*1000.*cmtopc))/1e5

        cassiT_rinner = stats_data['inner_radius'][rvir_ind]
        cassiT_router = stats_data['outer_radius'][rvir_ind]
        cassiT_r = 0.5*(cassiT_rinner + cassiT_router)
        vtheta_sig = stats_data['net_theta_velocity_sig'][rvir_ind]
        vphi_sig = stats_data['net_phi_velocity_sig'][rvir_ind]
        vr_sig = stats_data['net_radial_velocity_sig'][rvir_ind]
        vr_mu = stats_data['net_radial_velocity_mu'][rvir_ind]
        vr_A1 = stats_data['net_radial_velocity_A1'][rvir_ind]
        vr_A2 = stats_data['net_radial_velocity_A2'][rvir_ind]
        vtan_sig = stats_data['net_tangential_velocity_sig'][rvir_ind]
        kinetic_energy1 = 1./2.*(3./2.*(vtheta_sig*1e5)**2. + 3./2.*(vphi_sig*1e5)**2.)
        rv = np.linspace(-200,vesc,400)
        fit = double_gauss(rv, 0., vtan_sig, vr_A2, vr_mu, vr_sig, vr_A1)
        kinetic_energy2 = 1./2.*np.sum((rv*1e5)**2*(fit/np.sum(fit))) + 1./2.*((vtheta_sig*1e5)**2. + (vphi_sig*1e5)**2.)
        cassiT_T1 = (mu*mp/kB)*(1./2.*G*Menc_profile(cassiT_r)*gtoMsun/(cassiT_r*1000*cmtopc) - 2./3.*kinetic_energy1)
        cassiT_T2 = (mu*mp/kB)*(1./2.*G*Menc_profile(cassiT_r)*gtoMsun/(cassiT_r*1000*cmtopc) - 2./3.*kinetic_energy2)
        cassiT_list1.append(cassiT_T1)
        cassiT_list2.append(cassiT_T2)

        if (args.Rvir_compare):
            cassiT_rinner = stats_data['inner_radius'][rvir_ind2]
            cassiT_router = stats_data['outer_radius'][rvir_ind2]
            cassiT_r = 0.5*(cassiT_rinner + cassiT_router)
            vtheta_sig = stats_data['net_theta_velocity_sig'][rvir_ind2]
            vphi_sig = stats_data['net_phi_velocity_sig'][rvir_ind2]
            vr_sig = stats_data['net_radial_velocity_sig'][rvir_ind2]
            vr_mu = stats_data['net_radial_velocity_mu'][rvir_ind2]
            vr_A1 = stats_data['net_radial_velocity_A1'][rvir_ind2]
            vr_A2 = stats_data['net_radial_velocity_A2'][rvir_ind2]
            vtan_sig = stats_data['net_tangential_velocity_sig'][rvir_ind2]
            kinetic_energy1 = 1./2.*(3./2.*(vtheta_sig*1e5)**2. + 3./2.*(vphi_sig*1e5)**2.)
            rv = np.linspace(-200,vesc2,400)
            fit = double_gauss(rv, 0., vtan_sig, vr_A2, vr_mu, vr_sig, vr_A1)
            kinetic_energy2 = 1./2.*np.sum((rv*1e5)**2*(fit/np.sum(fit))) + 1./2.*((vtheta_sig*1e5)**2. + (vphi_sig*1e5)**2.)
            cassiT_T1 = (mu*mp/kB)*(1./2.*G*Menc_profile(cassiT_r)*gtoMsun/(cassiT_r*1000*cmtopc) - 2./3.*kinetic_energy1)
            cassiT_T2 = (mu*mp/kB)*(1./2.*G*Menc_profile(cassiT_r)*gtoMsun/(cassiT_r*1000*cmtopc) - 2./3.*kinetic_energy2)
            cassiT_list1_2.append(cassiT_T1)
            cassiT_list2_2.append(cassiT_T2)

        if (i==0):
            bins_lower = pdf_data['lower_log_temperature'][inds]
            bins_upper = pdf_data['upper_log_temperature'][inds]
            bin_edges = np.array(bins_lower)
            bin_edges = np.append(bin_edges, bins_upper[-1])
            bin_centers = np.diff(bin_edges)/2. + np.array(bins_lower)
        weight_data = pdf_data['net_log_temperature_pdf'][inds]
        weights_data.append(weight_data)
        max_weights.append(np.max(weight_data))

        if (args.Rvir_compare):
            if (i==0):
                bins_lower = pdf_data['lower_log_temperature'][inds2]
                bins_upper = pdf_data['upper_log_temperature'][inds2]
                bin_edges = np.array(bins_lower)
                bin_edges = np.append(bin_edges, bins_upper[-1])
                bin_centers = np.diff(bin_edges)/2. + np.array(bins_lower)
            weight_data = pdf_data['net_log_temperature_pdf'][inds2]
            weights_data2.append(weight_data)
            max_weights2.append(np.max(weight_data))

    weights_data = np.array(weights_data)
    cassiT_list1 = np.array(cassiT_list1)
    cassiT_list2 = np.array(cassiT_list2)
    Tvir_list = np.array(Tvir_list)
    if (args.Rvir_compare):
        weights_data2 = np.array(weights_data2)
        cassiT_list1_2 = np.array(cassiT_list1_2)
        cassiT_list2_2 = np.array(cassiT_list2_2)
        Tvir_list2 = np.array(Tvir_list2)
    zlist.insert(0, zlist[0]+(zlist[0]-zlist[1]))
    timelist.insert(0, timelist[0] - (timelist[1]-timelist[0]))
    zlist.reverse()
    timelist.reverse()
    time_func = IUS(zlist, timelist)
    timelist.reverse()
    timelist = np.array(timelist).flatten()
    zlist = np.array(zlist)
    xdata = np.tile(timelist[:-1], (len(bin_edges)-1, 1)).flatten()
    xbins = timelist
    weights_data = np.transpose(weights_data).flatten()
    if (args.Rvir_compare): weights_data2 = np.transpose(weights_data2).flatten()
    ydata = np.transpose(np.tile(bin_edges[:-1], (len(zlist)-1, 1))).flatten()

    z_sfr, sfr = np.loadtxt(code_path + 'halo_infos/00' + args.halo + '/' + args.run + '/sfr', unpack=True, usecols=[1,2], skiprows=1)

    fig = plt.figure(figsize=(8,6),dpi=500)
    ax = fig.add_subplot(1,1,1)
    plt.subplots_adjust(left=0.13, bottom=0.11, right=0.88, top=0.85)
    if (args.Rvir_compare):
        hist = ax.hist2d(xdata, ydata, weights=weights_data2, bins=[xbins,bin_edges], vmin=0., vmax=np.mean(max_weights2), cmap=plt.cm.Greys)
        ax.plot(timelist[1:], np.log10(Tvir_list2), color='darkorange', ls='--', lw=4, label='$T_\mathrm{vir}$')
        ax.plot(timelist[1:], np.log10(cassiT_list1_2), 'r-', lw=3, label='$T_\mathrm{mod}^\mathrm{turb}$')
        ax.plot(timelist[1:], np.log10(cassiT_list2_2), 'r:', lw=3, label='$T_\mathrm{mod}^\mathrm{turb+out}$')
    else:
        hist = ax.hist2d(xdata, ydata, weights=weights_data, bins=[xbins,bin_edges], vmin=0., vmax=np.mean(max_weights), cmap=plt.cm.Greys)
        ax.plot(timelist[1:], np.log10(Tvir_list), color='darkorange', ls='--', lw=4, label='$T_\mathrm{vir}$')
        ax.plot(timelist[1:], np.log10(cassiT_list1), 'r-', lw=3, label='$T_\mathrm{mod}^\mathrm{turb}$')
        ax.plot(timelist[1:], np.log10(cassiT_list2), 'r:', lw=3, label='$T_\mathrm{mod}^\mathrm{turb+out}$')
    ax.set_xlim(np.min(timelist), np.max(timelist))
    ax.set_ylim(4.5, 6.5)
    ax2 = ax.twiny()
    ax.tick_params(axis='both', which='both', direction='in', length=8, width=2, pad=5, labelsize=18, \
      top=False, right=False)
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
    cbaxes = fig.add_axes([0.7, 0.95, 0.25, 0.03])
    cbar = plt.colorbar(hist[3], cax=cbaxes, orientation='horizontal', ticks=[])
    cbar.set_label('log Mass', fontsize=18)
    ax.set_ylabel('log Temperature at $R_{200}$ [K]', fontsize=18)
    ax3 = ax.twinx()
    ax3.plot(time_func(np.array(z_sfr)), sfr, 'k-', lw=1)
    ax.plot([timelist[0],timelist[-1]], [0,0], 'k-', lw=1, label='SFR (right axis)')
    ax3.tick_params(axis='y', which='both', direction='in', length=8, width=2, pad=5, labelsize=18, right=True)
    ax3.set_ylim(-5,100)
    ax3.set_ylabel('SFR [$M_\odot$/yr]', fontsize=18)
    if (args.halo=='8508'): ax.legend(loc=1, fontsize=18, frameon=False)
    ax.text(4,6.3,halo_dict[args.halo],ha='left',va='center',fontsize=18)
    plt.savefig(save_dir + 'temperature_vs_t_mass-colored' + save_suffix + '.pdf')
    plt.close()

    print('Plot made!')

def temperature_vs_radius(snap):
    '''Plots a 2D mass histogram of temperature vs radius at given snapshot 'snap' with standard and
    modified virial temperatures compared.'''

    stats_dir = output_dir + 'stats_halo_00' + args.halo + '/' + args.run + '/'
    if (args.system=='cassiopeia'):
        stats_data = Table.read(stats_dir + '/Tables/' + snap + '_' + args.filename.replace('_pdf', '') + '.hdf5', path='all_data')
        if (args.hist_from_file):
            pdf_data = Table.read(stats_dir + '/Tables/' + snap + '_' + args.filename + '.hdf5', path='all_data')
    else:
        stats_data = Table.read(stats_dir + '/' + snap + '_' + args.filename.replace('_pdf', '') + '.hdf5', path='all_data')
        if (args.hist_from_file):
            pdf_data = Table.read(stats_dir + '/' + snap + '_' + args.filename + '.hdf5', path='all_data')

    zsnap = stats_data['redshift'][0]

    masses_ind = np.where(masses['snapshot']==snap)[0]
    Menc_profile = IUS(np.concatenate(([0],masses['radius'][masses_ind])), np.concatenate(([0],masses['total_mass'][masses_ind])))
    Mvir = rvir_masses['total_mass'][rvir_masses['snapshot']==snap]
    Rvir = rvir_masses['radius'][rvir_masses['snapshot']==snap][0]
    Tvir = (mu*mp/kB)*(1./2.*G*Mvir*gtoMsun)/(Rvir*1000*cmtopc)
    if (args.Rvir_compare):
        Mvir2 = rvir_masses2['total_mass'][rvir_masses2['snapshot']==snap]
        Rvir2 = rvir_masses2['radius'][rvir_masses2['snapshot']==snap]
        Tvir2 = (mu*mp/kB)*(1./2.*G*Mvir2*gtoMsun)/(Rvir2*1000*cmtopc)

    stats_inner_r = stats_data['inner_radius']
    stats_outer_r = stats_data['outer_radius']
    cassiT_r = 0.5*(stats_inner_r + stats_outer_r)
    vr_sig = stats_data['net_radial_velocity_sig']
    vr_mu = stats_data['net_radial_velocity_mu']
    vr_A1 = stats_data['net_radial_velocity_A1']
    vr_A2 = stats_data['net_radial_velocity_A2']
    vtheta_sig = stats_data['net_theta_velocity_sig']
    vphi_sig = stats_data['net_phi_velocity_sig']
    vtan_sig = stats_data['net_tangential_velocity_sig']
    kinetic_energy1 = 1./2.*(3./2.*(vtheta_sig*1e5)**2. + 3./2.*(vphi_sig*1e5)**2.)
    kinetic_energy2 = []
    for k in range(len(vtan_sig)):
        vesc = np.sqrt(2.*G*Menc_profile(cassiT_r[k])*gtoMsun/(cassiT_r[k]*1000.*cmtopc))/1e5
        rv = np.linspace(-200,vesc,400)
        fit = double_gauss(rv, 0., vtan_sig[k], vr_A2[k], vr_mu[k], vr_sig[k], vr_A1[k])
        kinetic_energy2.append(1./2.*np.sum((rv*1e5)**2*(fit/np.sum(fit))) + 1./2.*((vtheta_sig[k]*1e5)**2. + (vphi_sig[k]*1e5)**2.))
    kinetic_energy2 = np.array(kinetic_energy2)
    cassiT_T1 = (mu*mp/kB)*(1./2.*G*Menc_profile(cassiT_r)*gtoMsun/(cassiT_r*1000*cmtopc) - 2./3.*kinetic_energy1)
    cassiT_T2 = (mu*mp/kB)*(1./2.*G*Menc_profile(cassiT_r)*gtoMsun/(cassiT_r*1000*cmtopc) - 2./3.*kinetic_energy2)

    if (args.hist_from_file):
        weights_data = []
        max_weights = []
        rbins = []
        n_bins = len(np.where(pdf_data['inner_radius']==pdf_data['inner_radius'][0])[0])
        for i in range(int(len(pdf_data['inner_radius'])/n_bins)):
            inds = np.where(pdf_data['inner_radius']==pdf_data['inner_radius'][n_bins*i])
            rbins.append(pdf_data['inner_radius'][n_bins*i])
            if (i==0):
                bins_lower = pdf_data['lower_log_temperature'][inds]
                bins_upper = pdf_data['upper_log_temperature'][inds]
                bin_edges = np.array(bins_lower)
                bin_edges = np.append(bin_edges, bins_upper[-1])
                bin_centers = np.diff(bin_edges)/2. + np.array(bins_lower)
            weight_data = pdf_data['net_log_temperature_pdf'][inds]
            weights_data.append(weight_data)
            max_weights.append(np.max(weight_data))
        rbins.append(pdf_data['outer_radius'][-1])
        xdata = np.tile(rbins[:-1], (len(bin_edges)-1, 1)).flatten()
        xbins = rbins
        weights_data = np.transpose(weights_data).flatten()
        ydata = np.transpose(np.tile(bin_edges[:-1], (len(rbins)-1, 1))).flatten()
        y_range = [np.min(bin_edges), np.max(bin_edges)]
        x_range = [np.min(rbins), np.max(rbins)]
    else:
        snap_name = foggie_dir + run_dir + snap + '/' + snap
        ds, refine_box = foggie_load(snap_name, trackname, do_filter_particles=False, halo_c_v_name=halo_c_v_name)
        cgm = ds.sphere(ds.halo_center_kpc, (2.*ds.refine_width, 'kpc')) - ds.sphere(ds.halo_center_kpc, (0.3*Rvir, 'kpc'))
        cgm = cgm.cut_region("(obj['density'] < %.2e) & (obj['temperature'] > %.2e)" % (cgm_density_max, cgm_temperature_min))
        radius = cgm['radius_corrected'].in_units('kpc').flatten().v
        temperature = np.log10(cgm['temperature'].in_units('K').flatten().v)
        weights = np.log10(cgm['cell_mass'].in_units('Msun').flatten().v)
        rho = Menc_profile(radius)*gtoMsun/((radius*1000*cmtopc)**3.) * 3./(4.*np.pi)
        vff = -(radius*1000*cmtopc)/np.sqrt(3.*np.pi/(32.*G*rho))/1e5
        rad_vel = cgm['radial_velocity_corrected'].in_units('km/s').flatten().v
        temperature = temperature[(rad_vel > 0.5*vff)]
        weights = weights[(rad_vel > 0.5*vff)]
        radius = radius[(rad_vel > 0.5*vff)]
        x_range = [np.min(radius), np.max(radius)]
        y_range = [np.min(temperature), np.max(temperature)]
        cmin = np.min(np.array(weights)[np.nonzero(weights)[0]])

    fig = plt.figure(figsize=(8,6),dpi=500)
    ax = fig.add_subplot(1,1,1)

    if (args.hist_from_file):
        hist = ax.hist2d(xdata, ydata, weights=weights_data, bins=[xbins,bin_edges], vmin=0., vmax=np.mean(max_weights), cmap=plt.cm.Greys)
    else:
        hist = ax.hist2d(radius, temperature, weights=weights, bins=(500, 500), cmin=cmin, range=[x_range,y_range], cmap=plt.cm.Greys)
    cbaxes = fig.add_axes([0.7, 0.95, 0.25, 0.03])
    cbar = plt.colorbar(hist[3], cax=cbaxes, orientation='horizontal', ticks=[])
    cbar.set_label('log Mass', fontsize=18)
    ax.plot([Rvir, Rvir], [y_range[0], y_range[1]], 'k--', lw=2)
    if (args.Rvir_compare):
        ax.plot(cassiT_r, np.log10(np.zeros(len(cassiT_r))+Tvir), color='darkorange', ls='--', lw=4, label='$T_{200}$')
        ax.plot(cassiT_r, np.log10(np.zeros(len(cassiT_r))+Tvir2), color='darkorange', ls=':', lw=4, label='$T_\mathrm{vir}$')
        ax.plot([Rvir2, Rvir2], [y_range[0], y_range[1]], 'k--', lw=2)
        ax.text(Rvir-3., 4.7, '$R_{200}$', fontsize=18, ha='right', va='center')
        ax.text(Rvir2-3., 4.7, '$R_\mathrm{vir}$', fontsize=18, ha='right', va='center')
    else:
        ax.plot(cassiT_r, np.log10(np.zeros(len(cassiT_r))+Tvir), color='darkorange', ls='--', lw=4, label='$T_\mathrm{vir}$')
        ax.text(Rvir-3., 4.7, '$R_{200}$', fontsize=18, ha='right', va='center')
    ax.plot(cassiT_r, np.log10(cassiT_T1), color='r', ls='-', lw=3, label='$T_\mathrm{mod}^\mathrm{turb}$')
    ax.plot(cassiT_r, np.log10(cassiT_T2), color='r', ls=':', lw=3, label='$T_\mathrm{mod}^\mathrm{turb+out}$')

    if (args.halo=='8508'): ax.legend(loc=3, frameon=False, fontsize=18)
    ax.set_xlabel('Radius [kpc]', fontsize=18)
    ax.set_ylabel('log Temperature [K]', fontsize=18)
    ax.axis([0,250,4.5,6.5])
    ax.text(8,6.3, '$z=%.2f$' % (zsnap), fontsize=18, ha='left', va='center')
    ax.text(8,6.15,halo_dict[args.halo],ha='left',va='center',fontsize=18)
    ax.tick_params(axis='both', which='both', direction='in', length=8, width=2, pad=5, labelsize=18, \
      top=True, right=True)
    plt.subplots_adjust(left=0.12, bottom=0.11, right=0.95)
    plt.savefig(save_dir + snap + '_temperature_vs_radius_mass-colored' + save_suffix + '.pdf')
    print('Plot made.')

def energy_SFR_xcorr(snaplist):
    '''Plots a cross-correlation between VE (at Rvir) and SFR as a function of time delay between them.'''

    tablename_prefix = output_dir + 'totals_halo_00' + args.halo + '/' + args.run + '/Tables/'
    time_table = Table.read(output_dir + 'times_halo_00' + args.halo + '/' + args.run + '/time_table.hdf5', path='all_data')

    if (args.Rvir_compare):
        plot_colors = ['r']
        plot_labels = ['Virial']
        table_labels = ['virial_energy']
        linewidths = [3]
        linestyles = ['-']
    else:
        plot_colors = ['r', 'g', 'b']
        plot_labels = ['Virial', 'KE$_\mathrm{th}$', 'KE$_\mathrm{nt}$']
        table_labels = ['virial_energy', 'thermal_energy', 'kinetic_energy']
        linestyles = ['-', '--', ':']
        linewidths = [3, 2, 2]

    zlist = []
    timelist = []
    data_list = []
    for j in range(len(table_labels)):
        data_list.append([])
    if (args.Rvir_compare):
        data_list2 = []
        for j in range(len(table_labels)):
            data_list2.append([])

    for i in range(len(snaplist)):
        data = Table.read(tablename_prefix + snaplist[i] + '_' + args.filename + '.hdf5', path='all_data')
        rvir = rvir_masses['radius'][rvir_masses['snapshot']==snaplist[i]]
        pos_ind = np.where(data['outer_radius']>=rvir/2.)[0]
        if (len(pos_ind)==0):
            pos_ind = len(data['outer_radius'])-1
        else:
            pos_ind = pos_ind[0]
        if (args.Rvir_compare):
            rvir2 = rvir_masses2['radius'][rvir_masses2['snapshot']==snaplist[i]]
            pos_ind2 = np.where(data['outer_radius']>rvir2)[0]
            if (len(pos_ind2)==0):
                pos_ind2 = len(data['outer_radius'])-1
            else:
                pos_ind2 = pos_ind2[0]
        for k in range(len(table_labels)):
            data_list[k].append(data['net_' + table_labels[k]][pos_ind]/-data['net_potential_energy'][pos_ind])
            if (args.Rvir_compare):
                data_list2[k].append(data['net_' + table_labels[k]][pos_ind2]/-data['net_potential_energy'][pos_ind2])
        zlist.append(data['redshift'][0])
        timelist.append(time_table['time'][time_table['snap']==snaplist[i]][0]/1000.)

    z_sfr, sfr = np.loadtxt(code_path + 'halo_infos/00' + args.halo + '/' + args.run + '/sfr', unpack=True, usecols=[1,2], skiprows=1)
    snap_sfr = np.loadtxt(code_path + 'halo_infos/00' +args.halo + '/' + args.run + '/sfr', unpack=True, usecols=[0], skiprows=1, dtype=str)
    SFR_list = []
    for i in range(len(snaplist)):
        # Cap SFR at 20 Msun/yr:
        #if (sfr[np.where(snap_sfr==snaplist[i])[0][0]]>20.):
            #SFR_list.append(20.)
        #else:
            #SFR_list.append(sfr[np.where(snap_sfr==snaplist[i])[0][0]])
        SFR_list.append(sfr[np.where(snap_sfr==snaplist[i])[0][0]])
    SFR_list = np.array(SFR_list)
    SFR_list2 = np.roll(np.array(SFR_list), 200)
    SFR_mean = np.mean(SFR_list)
    SFR_std = np.std(SFR_list)

    delay_list = np.array(range(int(len(timelist)/3)))        # Consider delay times of zero all the way
    dt = 5.38*args.output_step                                # up to a third of the full time evolution

    xcorr_list = []
    xcorr_list_test = []
    if (args.Rvir_compare):
        xcorr_list2 = []
    for i in range(len(table_labels)):
        xcorr_list.append([])
        mean = np.mean(data_list[i])
        std = np.std(data_list[i])
        data_list[i] = np.array(data_list[i])
        if (args.Rvir_compare):
            xcorr_list2.append([])
            mean2 = np.mean(data_list2[i])
            std2 = np.std(data_list2[i])
            data_list2[i] = np.array(data_list2[i])
        for j in range(len(delay_list)):
            xcorr_list[i].append(np.sum((SFR_list-SFR_mean)*(np.roll(data_list[i], -delay_list[j])-mean))/(SFR_std*std*len(data_list[i])))
            if (args.Rvir_compare):
                xcorr_list2[i].append(np.sum((SFR_list-SFR_mean)*(np.roll(data_list2[i], -delay_list[j])-mean2))/(SFR_std*std2*len(data_list2[i])))


    fig = plt.figure(figsize=(8,6), dpi=500)
    ax = fig.add_subplot(1,1,1)
    for i in range(len(table_labels)):
        label = plot_labels[i]
        ax.plot(delay_list*dt, np.array(xcorr_list[i]), \
                color=plot_colors[i], ls=linestyles[i], lw=linewidths[i], label=label)
        if (args.Rvir_compare):
            ax.plot(delay_list*dt, np.array(xcorr_list2[i]), \
                    color=plot_colors[i], ls='--', lw=linewidths[i], label='_nolabel_')
    if (args.Rvir_compare):
        ax.plot([-100],[-100], color=plot_colors[i], ls='-', lw=linewidths[i], label='$R_{200}')
        ax.plot([-100],[-100], color=plot_colors[i], ls='--', lw=linewidths[i], label='$R_\mathrm{vir}')

    ax.set_ylabel('Cross-correlation with SFR', fontsize=18)
    ax.set_xlabel('Time delay [Myr]', fontsize=18)
    ax.set_xlim(0., 2000.)
    ax.set_ylim(-1, 1)
    xticks = np.arange(0,2100,100)
    ax.set_xticks(xticks)
    xticklabels = []
    for i in range(len(xticks)):
        if (xticks[i]%500!=0): xticklabels.append('')
        else: xticklabels.append(str(xticks[i]))
    ax.set_xticklabels(xticklabels)
    if (args.halo=='8508'): ax.legend(loc=1, fontsize=18)
    ax.text(200.,-.8,halo_dict[args.halo],ha='left',va='center',fontsize=18)
    ax.plot([0,2000],[0,0],'k-',lw=1)
    ax.tick_params(axis='both', which='both', direction='in', length=8, width=2, pad=5, labelsize=18)
    ax.grid(which='both',axis='both',alpha=0.25,color='k',lw=1,ls='-')
    plt.subplots_adjust(top=0.97, bottom=0.12, right=0.95, left=0.15)
    plt.savefig(save_dir + 'xcorr_energy-SFR_vs_delay-t' + save_suffix + '.pdf')
    plt.close()

    print('Plot made!')

def temp_SFR_xcorr(snaplist):
    '''Plots the cross-correlation between SFR and amount of mass at Rvir in different temp bins as
    a function of delay time.'''

    tablename_prefix = output_dir + 'totals_halo_00' + args.halo + '/' + args.run + '/Tables/'
    time_table = Table.read(output_dir + 'times_halo_00' + args.halo + '/' + args.run + '/time_table.hdf5', path='all_data')

    zlist = []
    timelist = []
    temp_data = []
    Tvir_list = []
    for i in range(len(snaplist)):
        timelist.append(time_table['time'][time_table['snap']==snaplist[i]][0]/1000.)
        tot_data = Table.read(tablename_prefix + snaplist[i] + '_' + args.filename + '.hdf5', path='all_data')
        temp_data.append([])

        zsnap = tot_data['redshift'][0]
        zlist.append(zsnap)

        masses_ind = np.where(masses['snapshot']==snaplist[i])[0]
        Menc_profile = IUS(np.concatenate(([0],masses['radius'][masses_ind])), np.concatenate(([0],masses['total_mass'][masses_ind])))
        Mvir = rvir_masses['total_mass'][rvir_masses['snapshot']==snaplist[i]][0]

        rvir = rvir_masses['radius'][rvir_masses['snapshot']==snaplist[i]]
        rvir_ind = np.where(tot_data['outer_radius']>=rvir)[0]
        if (len(rvir_ind)==0):
            rvir_ind = len(tot_data['outer_radius'])-1
        else:
            rvir_ind = rvir_ind[0]

        Tvir = (mu*mp/kB)*(1./2.*G*Mvir*gtoMsun)/(rvir*1000*cmtopc)
        Tvir_list.append(Tvir)

        nTbins = 4
        for j in range(nTbins):
            if (j==0): bins = [0,1,2]
            if (j==1): bins = [3,4]
            if (j==2): bins = [5,6]
            if (j==3): bins = [7,8,9]
            mass = 0.
            for k in bins:
                mass += tot_data['net_Tbin' + str(k) + '_mass'][rvir_ind]
            temp_data[i].append(mass)

    temp_data = np.array(temp_data)
    temp_data = np.transpose(temp_data)     # Now first index is temperature bin (relative to Tvir)
                                            # and second index is time snapshot

    z_sfr, sfr = np.loadtxt(code_path + 'halo_infos/00' + args.halo + '/' + args.run + '/sfr', unpack=True, usecols=[1,2], skiprows=1)
    snap_sfr = np.loadtxt(code_path + 'halo_infos/00' +args.halo + '/' + args.run + '/sfr', unpack=True, usecols=[0], skiprows=1, dtype=str)
    SFR_list = []
    for i in range(len(snaplist)):
        # Cap SFR at 20 Msun/yr:
        #if (sfr[np.where(snap_sfr==snaplist[i])[0][0]]>20.):
            #SFR_list.append(20.)
        #else:
            #SFR_list.append(sfr[np.where(snap_sfr==snaplist[i])[0][0]])
        SFR_list.append(sfr[np.where(snap_sfr==snaplist[i])[0][0]])
    SFR_list = np.array(SFR_list)
    SFR_mean = np.mean(SFR_list)
    SFR_std = np.std(SFR_list)

    delay_list = np.array(range(int(len(timelist)/3)))        # Consider delay times of zero all the way
    dt = 5.38*args.output_step                                # up to a third of the full time evolution

    xcorr_list = []
    for i in range(len(temp_data)):
        xcorr_list.append([])
        mean = np.mean(temp_data[i])
        std = np.std(temp_data[i])
        temp_data[i] = np.array(temp_data[i])
        for j in range(len(delay_list)):
            xcorr_list[i].append(np.sum((SFR_list-SFR_mean)*(np.roll(temp_data[i], -delay_list[j])-mean))/(SFR_std*std*len(temp_data[i])))

    plot_labels = ['$T < T_\mathrm{vir}-0.5$ dex', '0 to 0.5 dex below $T_\mathrm{vir}$', \
      '0 to 0.5 dex above $T_\mathrm{vir}$', '$T>T_\mathrm{vir}+0.5$ dex']
    plot_ls = [':', '-.', '--', '-']
    plasma = plt.get_cmap('plasma')
    cNorm  = colors.Normalize(vmin=0, vmax=len(temp_data))
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=plasma)

    fig = plt.figure(figsize=(8,6), dpi=500)
    ax = fig.add_subplot(1,1,1)
    for i in range(len(temp_data)):
        label = plot_labels[i]
        ax.plot(delay_list*dt, np.array(xcorr_list[i]), \
                color=scalarMap.to_rgba(i), ls=plot_ls[i], lw=2, label=label)
    if (args.Rvir_compare):
        ax.plot([-100],[-100], color=plot_colors[i], ls='-', lw=linewidths[i], label='$R_{200}')
        ax.plot([-100],[-100], color=plot_colors[i], ls='--', lw=linewidths[i], label='$R_\mathrm{vir}')

    ax.set_ylabel('Cross-correlation with SFR\nof Mass Fraction in $T$ bin', fontsize=18)
    ax.set_xlabel('Time delay [Myr]', fontsize=18)
    ax.set_xlim(0., 2000.)
    ax.set_ylim(-1, 1)
    if (args.halo=='8508'): ax.legend(loc=1, fontsize=18)
    ax.text(200.,-.8,halo_dict[args.halo],ha='left',va='center',fontsize=18)
    ax.plot([0,2000],[0,0],'k-',lw=1)
    ax.tick_params(axis='both', which='both', direction='in', length=8, width=2, pad=5, labelsize=18)
    xticks = np.arange(0,2100,100)
    ax.set_xticks(xticks)
    xticklabels = []
    for i in range(len(xticks)):
        if (xticks[i]%500!=0): xticklabels.append('')
        else: xticklabels.append(str(xticks[i]))
    ax.set_xticklabels(xticklabels)
    ax.grid(which='both',axis='both',alpha=0.25,color='k',lw=1,ls='-')
    plt.subplots_adjust(top=0.97, bottom=0.12, right=0.95, left=0.2)
    plt.savefig(save_dir + 'xcorr_temp-SFR_vs_delay-t' + save_suffix + '.pdf')
    plt.close()

    print('Plot made!')

def energy_vs_radius_compare(snap):
    '''Compares the energies computed cumulatively within R as a function of R to energies computed
    in shells with the SIS assumption as a function of R.'''

    tablename_prefix = output_dir + 'totals_halo_00' + args.halo + '/' + args.run + '/Tables/'
    totals = Table.read(tablename_prefix + snap + '_' + args.filename + '.hdf5', path='all_data')
    Rvir = rvir_masses['radius'][rvir_masses['snapshot']==snap]
    stats_dir = output_dir + 'stats_halo_00' + args.halo + '/' + args.run
    # No filaments:
    vel_stats_name = 'stats_temperature_energy_velocity_sphere_mass-weighted_vcut-p5vff_dbl-gauss-fit_cgm-filtered'
    den_stats_name = 'stats_pressure_density_sphere_mass-weighted_vcut-p5vff_cgm-filtered'
    # Including filaments:
    #vel_stats_name = 'stats_pressure_density_velocity_sphere_mass-weighted_cgm-filtered'
    #den_stats_name = 'stats_pressure_density_velocity_sphere_mass-weighted_cgm-filtered'
    vel_stats_data = Table.read(stats_dir + '/Tables/' + snap + '_' + vel_stats_name + '.hdf5', path='all_data')
    den_stats_data = Table.read(stats_dir + '/Tables/' + snap + '_' + den_stats_name + '.hdf5', path='all_data')

    Menc_allgas = IUS(np.concatenate(([0],masses['radius'][masses['snapshot']==snap])), np.concatenate(([0],masses['gas_mass'][masses['snapshot']==snap])))

    # This whole next part is correcting for minor differences in the radius lists between the totals
    # file and the different stats files. They should have the same bin values (within rounding) but
    # may start or end in different places.
    radius_list = 0.5*(totals['inner_radius'] + totals['outer_radius'])
    den_radius_list = 0.5*(den_stats_data['inner_radius'] + den_stats_data['outer_radius'])
    vel_radius_list = 0.5*(vel_stats_data['inner_radius'] + vel_stats_data['outer_radius'])
    if (len(den_radius_list)!=len(vel_radius_list)):
        if (den_radius_list[0]>vel_radius_list[0]+0.5*np.diff(vel_radius_list)[0]):
            vel_start_ind = np.where(vel_radius_list>=den_radius_list[0]-0.5*np.diff(den_radius_list)[0])[0][0]
            den_start_ind = 0
        elif (vel_radius_list[0]>den_radius_list[0]+0.5*np.diff(den_radius_list)[0]):
            den_start_ind = np.where(den_radius_list>=vel_radius_list[0]-0.5*np.diff(vel_radius_list)[0])[0][0]
            vel_start_ind = 0
        else:
            den_start_ind = 0
            vel_start_ind = 0
        if (den_radius_list[-1]>vel_radius_list[-1]+0.5*np.diff(vel_radius_list)[-1]):
            vel_end_ind = -1
            den_end_ind = np.where(den_radius_list>=(vel_radius_list[-1]-0.5*np.diff(vel_radius_list)[-1]))[0][0]
        elif (vel_radius_list[-1]>den_radius_list[-1]+0.5*np.diff(den_radius_list)[-1]):
            den_end_ind = -1
            vel_end_ind = np.where(vel_radius_list>=(den_radius_list[-1]-0.5*np.diff(den_radius_list)[-1]))[0][0]
        else:
            den_end_ind = -1
            vel_end_ind = -1
    else:
        vel_start_ind = 0
        den_start_ind = 0
        vel_end_ind = -1
        den_end_ind = -1
    tot_start_ind = np.where(radius_list>=(den_radius_list[den_start_ind]-0.5*np.diff(den_radius_list)[den_start_ind]))[0][0]
    if (len(radius_list[tot_start_ind:]) > len(den_radius_list[den_start_ind:den_end_ind])):
        tot_end_ind = np.where(radius_list>=(den_radius_list[den_end_ind]-0.5*np.diff(den_radius_list)[den_end_ind]))[0][0]
    elif (len(radius_list[tot_start_ind:]) < len(den_radius_list[den_start_ind:den_end_ind])):
        tot_end_ind = -1
        den_end_ind -= 2
        vel_end_ind -= 2
    else:
        tot_end_ind = -1
        den_end_ind -= 1
        vel_end_ind -= 1

    inner_r_ind = np.where(radius_list>=args.inner_r*Rvir)[0][0]

    zsnap = totals['redshift'][0]

    # No filaments:
    Pb = 0.5*10**den_stats_data['net_log_density_med'][den_start_ind:den_end_ind]*(vel_stats_data['net_tangential_velocity_sig'][vel_start_ind:vel_end_ind]*1e5)**2. + 10**den_stats_data['net_log_pressure_med'][den_start_ind:den_end_ind]
    # Including filaments:
    #Pb = 0.5*10**den_stats_data['net_log_density_med'][den_start_ind:den_end_ind]*(vel_stats_data['net_tangential_velocity_std'][vel_start_ind:vel_end_ind]*1e5)**2. + 10**den_stats_data['net_log_pressure_med'][den_start_ind:den_end_ind] + 0.5*10**den_stats_data['log_density_med_in'][den_start_ind:den_end_ind]*(vel_stats_data['radial_velocity_med_in'][vel_start_ind:vel_end_ind]*1e5)**2.
    Mgas_enc = np.cumsum(totals['net_mass'])*gtoMsun - np.sum(totals['net_mass'][:inner_r_ind+1])*gtoMsun
    Mgas_enc[Mgas_enc<0.] = 0.
    Sigma = Pb*4.*np.pi/3.*(den_radius_list[den_start_ind:den_end_ind]*cmtopc*1000.)**3./(Mgas_enc[tot_start_ind:tot_end_ind])
    KE_enc = np.cumsum(totals['net_kinetic_energy']) - np.sum(totals['net_kinetic_energy'][:inner_r_ind+1])
    KE_enc[KE_enc<0.] = 0.
    TE_enc = np.cumsum(totals['net_thermal_energy']) - np.sum(totals['net_thermal_energy'][:inner_r_ind+1])
    TE_enc[TE_enc<0.] = 0.
    PE_enc = np.cumsum(totals['net_potential_energy']) - np.sum(totals['net_potential_energy'][:inner_r_ind+1])
    PE_enc[PE_enc>0.] = 0.
    plot_colors = ['r', 'k', 'g', 'b', 'm']
    plot_labels = ['VE', 'PE', 'KE$_\mathrm{th}$', 'KE$_\mathrm{nt}$', '$\Sigma$']
    table_labels = ['virial_energy', 'potential_energy', 'thermal_energy', 'kinetic_energy', 'boundary_pressure']
    linestyles = ['-', '-', '--', ':', '-.']

    fig = plt.figure(figsize=(8,6), dpi=500)
    ax = fig.add_subplot(1,1,1)

    for i in range(len(plot_colors)):
        label = plot_labels[i]
        if (table_labels[i]=='virial_energy'):
                vir_en_cum_exact = 2.*(KE_enc[tot_start_ind:tot_end_ind]+TE_enc[tot_start_ind:tot_end_ind])/Mgas_enc[tot_start_ind:tot_end_ind] + PE_enc[tot_start_ind:tot_end_ind]/Mgas_enc[tot_start_ind:tot_end_ind] - Sigma
                ax.plot(radius_list[tot_start_ind:tot_end_ind][radius_list[tot_start_ind:tot_end_ind]>(args.inner_r+0.1)*Rvir], vir_en_cum_exact[radius_list[tot_start_ind:tot_end_ind]>(args.inner_r+0.1)*Rvir]/1e15, color=plot_colors[i], ls=linestyles[i], lw=1, label=label)
                vir_en_shell_SIS = 2.*(totals['net_kinetic_energy'][tot_start_ind:tot_end_ind]+totals['net_thermal_energy'][tot_start_ind:tot_end_ind])/(totals['net_mass'][tot_start_ind:tot_end_ind]*gtoMsun) + 3./2.*totals['net_potential_energy'][tot_start_ind:tot_end_ind]/(totals['net_mass'][tot_start_ind:tot_end_ind]*gtoMsun)
                ax.plot(radius_list[tot_start_ind:tot_end_ind], vir_en_shell_SIS/1e15, color=plot_colors[i], ls=linestyles[i], lw=2)
        elif (table_labels[i]=='boundary_pressure'):
            ax.plot(radius_list[tot_start_ind:tot_end_ind][radius_list[tot_start_ind:tot_end_ind]>(args.inner_r+0.1)*Rvir], Sigma[radius_list[tot_start_ind:tot_end_ind]>(args.inner_r+0.1)*Rvir]/1e15, color=plot_colors[i], ls=linestyles[i], lw=1, label=label)
        else:
            if (table_labels[i]=='kinetic_energy'):
                ax.plot(radius_list, KE_enc/Mgas_enc/1e15, \
                  color=plot_colors[i], ls=linestyles[i], lw=1, label=label)
            if (table_labels[i]=='thermal_energy'):
                ax.plot(radius_list, TE_enc/Mgas_enc/1e15, \
                  color=plot_colors[i], ls=linestyles[i], lw=1, label=label)
            if (table_labels[i]=='potential_energy'):
                ax.plot(radius_list, PE_enc/Mgas_enc/1e15, \
                  color=plot_colors[i], ls=linestyles[i], lw=1, label=label)
            ax.plot(radius_list[tot_start_ind:tot_end_ind], totals['net_' + table_labels[i]][tot_start_ind:tot_end_ind]/(totals['net_mass'][tot_start_ind:tot_end_ind]*gtoMsun)/1e15, \
              color=plot_colors[i], ls=linestyles[i], lw=2)

    ax.plot([-100,-100],[-100,-100], 'k-', lw=1, label='Thin lines:\nexact, cumulative')
    ax.plot([-100,-100],[-100,-100], 'k-', lw=2, label='Thick lines:\nSIS, shells')

    ax.set_xlabel('Radius [kpc]', fontsize=18)
    ax.set_ylabel('Specific energy [$10^{15}$ erg/g]', fontsize=18)
    ax.axis([0,250,-0.5,1])
    ax.text(15, 0.85, '$z=%.2f$' % (zsnap), fontsize=18, ha='left', va='center')
    ax.text(15,0.7,halo_dict[args.halo],ha='left',va='center',fontsize=18)
    ax.text(15,-0.4,'$%.1f R_{200} < r < R_{200}$' % (args.inner_r), ha='left', va='center', fontsize=18)
    if (args.halo=='8508') and (args.inner_r==0.3): ax.legend(loc=1, frameon=False, fontsize=14, ncol=2)
    ax.tick_params(axis='both', which='both', direction='in', length=8, width=2, pad=5, labelsize=18, \
      top=True, right=True)
    ax.plot([Rvir, Rvir], [ax.get_ylim()[0], ax.get_ylim()[1]], 'k--', lw=1)
    ax.text(Rvir-3., -0.4, '$R_{200}$', fontsize=18, ha='right', va='center')
    plt.subplots_adjust(top=0.94,bottom=0.11,right=0.95,left=0.15)
    plt.savefig(save_dir + snap + '_all_energies_vs_r_exact-SIS-compare' + save_suffix + '.pdf')
    plt.close()

def energy_vs_time_compare(snaplist):
    '''Compares the energies computed cumulatively within R_200 to energies computed
    in a shell at R_200 with the SIS assumption, over time.'''

    tablename_prefix = output_dir + 'totals_halo_00' + args.halo + '/' + args.run + '/Tables/'
    time_table = Table.read(output_dir + 'times_halo_00' + args.halo + '/' + args.run + '/time_table.hdf5', path='all_data')
    stats_dir = output_dir + 'stats_halo_00' + args.halo + '/' + args.run
    # No filaments:
    vel_stats_name = 'stats_temperature_energy_velocity_sphere_mass-weighted_vcut-p5vff_dbl-gauss-fit_cgm-filtered'
    den_stats_name = 'stats_pressure_density_sphere_mass-weighted_vcut-p5vff_cgm-filtered'
    # Including filaments:
    #vel_stats_name = 'stats_pressure_density_velocity_sphere_mass-weighted_cgm-filtered'
    #den_stats_name = 'stats_pressure_density_velocity_sphere_mass-weighted_cgm-filtered'

    table_labels = ['virial_energy', 'potential_energy', 'thermal_energy', 'kinetic_energy']
    plot_labels = ['VE', 'PE', 'KE$_\mathrm{th}$', 'KE$_\mathrm{nt}$']
    linestyles = ['-', '-', '--', ':']
    plot_colors = ['r', 'k', 'g', 'b']

    zlist = []
    timelist = []
    data_list_enc = []
    data_list_shell = []
    Sigma_list = []
    for j in range(len(table_labels)):
        data_list_enc.append([])
        data_list_shell.append([])

    for i in range(len(snaplist)):
        snap = snaplist[i]
        totals = Table.read(tablename_prefix + snap + '_' + args.filename + '.hdf5', path='all_data')
        Rvir = rvir_masses['radius'][rvir_masses['snapshot']==snap]
        vel_stats_data = Table.read(stats_dir + '/Tables/' + snap + '_' + vel_stats_name + '.hdf5', path='all_data')
        den_stats_data = Table.read(stats_dir + '/Tables/' + snap + '_' + den_stats_name + '.hdf5', path='all_data')
        tot_ind = np.where(totals['outer_radius']>=Rvir)[0][0]
        vel_ind = np.where(vel_stats_data['outer_radius']>=Rvir)[0][0]
        den_ind = np.where(den_stats_data['outer_radius']>=Rvir)[0][0]
        inner_r_ind = np.where(totals['inner_radius']>=args.inner_r*Rvir)[0][0]

        Menc_allgas = IUS(np.concatenate(([0],masses['radius'][masses['snapshot']==snap])), np.concatenate(([0],masses['gas_mass'][masses['snapshot']==snap])))

        # No filaments:
        Pb = 0.5*10**den_stats_data['net_log_density_med'][den_ind]*(vel_stats_data['net_tangential_velocity_sig'][vel_ind]*1e5)**2. + 10**den_stats_data['net_log_pressure_med'][den_ind]
        # Including filaments:
        #Pb = 0.5*10**den_stats_data['net_log_density_med'][den_ind]*(vel_stats_data['net_tangential_velocity_std'][vel_ind]*1e5)**2. + 10**den_stats_data['net_log_pressure_med'][den_ind] + 0.5*10**den_stats_data['log_density_med_in'][den_ind]*(vel_stats_data['radial_velocity_med_in'][vel_ind]*1e5)**2.
        Mgas_enc = np.cumsum(totals['net_mass'])*gtoMsun - np.sum(totals['net_mass'][:inner_r_ind+1])*gtoMsun
        Mgas_enc[Mgas_enc<0.] = 0.
        Sigma = Pb*4.*np.pi/3.*(totals['outer_radius'][tot_ind]*cmtopc*1000.)**3./(Mgas_enc[tot_ind])
        Sigma_list.append(Sigma)
        KE_enc = np.cumsum(totals['net_kinetic_energy']) - np.sum(totals['net_kinetic_energy'][:inner_r_ind+1])
        KE_enc[KE_enc<0.] = 0.
        TE_enc = np.cumsum(totals['net_thermal_energy']) - np.sum(totals['net_thermal_energy'][:inner_r_ind+1])
        TE_enc[TE_enc<0.] = 0.
        PE_enc = np.cumsum(totals['net_potential_energy']) - np.sum(totals['net_potential_energy'][:inner_r_ind+1])
        PE_enc[PE_enc>0.] = 0.

        for j in range(len(table_labels)):
            if (table_labels[j]=='virial_energy'):
                data_list_enc[j].append(2.*(KE_enc[tot_ind]+TE_enc[tot_ind])/Mgas_enc[tot_ind] + PE_enc[tot_ind]/Mgas_enc[tot_ind] - Sigma)
                data_list_shell[j].append(2.*(totals['net_kinetic_energy'][tot_ind]+totals['net_thermal_energy'][tot_ind])/(totals['net_mass'][tot_ind]*gtoMsun) + 3./2.*totals['net_potential_energy'][tot_ind]/(totals['net_mass'][tot_ind]*gtoMsun))
            else:
                if (table_labels[j]=='potential_energy'):
                    data_list_enc[j].append(PE_enc[tot_ind]/Mgas_enc[tot_ind])
                if (table_labels[j]=='thermal_energy'):
                    data_list_enc[j].append(TE_enc[tot_ind]/Mgas_enc[tot_ind])
                if (table_labels[j]=='kinetic_energy'):
                    data_list_enc[j].append(KE_enc[tot_ind]/Mgas_enc[tot_ind])
                data_list_shell[j].append(totals['net_' + table_labels[j]][tot_ind]/(totals['net_mass'][tot_ind]*gtoMsun))

        zlist.append(totals['redshift'][0])
        timelist.append(time_table['time'][time_table['snap']==snap]/1000.)

    fig = plt.figure(figsize=(8,6), dpi=500)
    ax = fig.add_subplot(1,1,1)
    for i in range(len(table_labels)):
        label = plot_labels[i]
        ax.plot(timelist, np.array(data_list_enc[i])/1e15, lw=1, ls=linestyles[i], color=plot_colors[i], label=label)
        ax.plot(timelist, np.array(data_list_shell[i])/1e15, lw=2, ls=linestyles[i], color=plot_colors[i])
    ax.plot(timelist, np.array(Sigma_list)/1e15, 'm-.', lw=1, label='$\Sigma$')

    ax.plot([-100,-100],[-100,-100], 'k-', lw=1, label='Thin lines:\nexact, cumulative')
    ax.plot([-100,-100],[-100,-100], 'k-', lw=2, label='Thick lines:\nSIS, shells')
    ax.plot([np.min(timelist), np.max(timelist)], [0,0], 'k-', lw=1, alpha=0.5)

    ax2 = ax.twiny()
    ax.tick_params(axis='both', which='both', direction='in', length=8, width=2, pad=5, labelsize=18, \
      top=False, right=False)
    ax2.tick_params(axis='x', which='both', direction='in', length=8, width=2, pad=5, labelsize=18, \
      top=True)
    zlist.reverse()
    timelist.reverse()
    time_func = IUS(zlist, timelist)
    timelist.reverse()
    ax.set_xlim(np.min(timelist), np.max(timelist))
    ax.set_ylim(-0.5, 1)
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
    ax.set_ylabel('Specific energy [$10^{15}$ erg/g]', fontsize=18)
    if (args.halo=='8508') and (args.inner_r==0.3):
        ax.legend(loc=1, frameon=False, fontsize=14, ncol=2)
    ax.text(4,0.9,halo_dict[args.halo],ha='left',va='center',fontsize=18)
    ax.text(4,-0.4,'$%.1f R_{200} < r < R_{200}$' % (args.inner_r), ha='left', va='center', fontsize=18)
    plt.subplots_adjust(top=0.9, bottom=0.12, right=0.95, left=0.15)
    plt.savefig(save_dir + 'all_energies_vs_t_exact-SIS-compare' + save_suffix + '.pdf')
    plt.close()

if __name__ == "__main__":
    args = parse_args()
    print(args.system)
    print(args.halo)
    print(args.run)
    foggie_dir, output_dir, run_dir, code_path, trackname, haloname, spectra_dir, infofile = get_run_loc_etc(args)
    halo_c_v_name = code_path + 'halo_infos/00' + args.halo + '/' + args.run + '/halo_c_v'

    gtoMsun = 1.989e33
    cmtopc = 3.086e18
    G = 6.673e-8
    kB = 1.38e-16
    mu = 0.6
    mp = 1.67e-24

    masses_dir = code_path + 'halo_infos/00' + args.halo + '/' + args.run + '/'
    masses = Table.read(masses_dir + 'masses_z-less-2.hdf5', path='all_data')
    rvir_masses = Table.read(masses_dir + 'rvir_masses.hdf5', path='all_data')
    if (args.Rvir_compare):
        rvir_masses2 = Table.read(masses_dir + 'rvir_masses_evolving-delta-c.hdf5', \
          path='all_data')

    # Set directory for output location
    if (args.system=='cassiopeia'):
        save_dir = "/Users/clochhaas/Documents/Research/FOGGIE/Papers/Modified Virial Temperature/"
    elif (args.system=='pleiades_cassi'):
        save_dir = '/nobackup/clochhaa/Outputs/temperatures_halo_00' + args.halo + '/' + args.run + '/'
        if not (os.path.exists(save_dir)): os.system('mkdir -p ' + save_dir)
    else:
        save_dir = ''

    outs = make_output_list(args.output, output_step=args.output_step)

    if (not args.filename) and (args.plot!='visualization') and (args.plot!='den_vel_proj') and \
       (args.plot!='mass_vs_time'):
        sys.exit("You must specify a filename where the data you want to plot is saved.")

    if (args.save_suffix): save_suffix = '_' + args.save_suffix

    if (len(outs)>1) and ('time' not in args.plot) and ('xcorr' not in args.plot):
        if (args.system=='cassiopeia'):
            save_dir = "/Users/clochhaas/Documents/Research/FOGGIE/Outputs/"
        elif (args.system=='pleiades_cassi'):
            save_dir = '/nobackup/clochhaa/Outputs/'
        else:
            save_dir = ''
        if ('energy' in args.plot):
            save_dir += 'energies_halo_00' + args.halo + '/' + args.run + '/'
        elif ('temperature' in args.plot):
            save_dir += 'temperatures_halo_00' + args.halo + '/' + args.run + '/'
        elif ('velocity' in args.plot):
            save_dir += 'velocities_halo_00' + args.halo + '/' + args.run + '/'
        else:
            save_dir += 'projections_halo_00' + args.halo + '/' + args.run + '/'
        if not (os.path.exists(save_dir)): os.system('mkdir -p ' + save_dir)

    if (args.plot=='visualization'):
        for i in range(len(outs)):
            visualization(outs[i])
    elif (args.plot=='den_vel_proj'):
        for i in range(len(outs)):
            den_vel_proj(outs[i])
    elif (args.plot=='mass_vs_time'):
        mass_vs_time(outs)
    elif (args.plot=='velocity_PDF'):
        for i in range(len(outs)):
            velocity_PDF(outs[i])
    elif (args.plot=='energy_vs_time'):
        energy_vs_time(outs)
    elif (args.plot=='energy_vs_radius'):
        for i in range(len(outs)):
            energy_vs_radius(outs[i])
    elif (args.plot=='energy_vs_radius_comp'):
        for i in range(len(outs)):
            energy_vs_radius_compare(outs[i])
    elif (args.plot=='energy_vs_time_comp'):
        energy_vs_time_compare(outs)
    elif (args.plot=='temperature_vs_time'):
        temperature_vs_time(outs)
    elif (args.plot=='temperature_vs_radius'):
        if (args.nproc==1):
            for i in range(len(outs)):
                temperature_vs_radius(outs[i])
        else:
            # Split into a number of groupings equal to the number of processors
            # and run one process per processor
            for i in range(len(outs)//args.nproc):
                threads = []
                for j in range(args.nproc):
                    snap = outs[args.nproc*i+j]
                    threads.append(multi.Process(target=temperature_vs_radius, args=[snap]))
                for t in threads:
                    t.start()
                for t in threads:
                    t.join()
            # For any leftover snapshots, run one per processor
            threads = []
            for j in range(len(outs)%args.nproc):
                snap = outs[-(j+1)]
                threads.append(multi.Process(target=temperature_vs_radius, args=[snap]))
            for t in threads:
                t.start()
            for t in threads:
                t.join()
        print("All snapshots finished!")
    elif (args.plot=='energy_SFR_xcorr'):
        energy_SFR_xcorr(outs)
    elif (args.plot=='temp_SFR_xcorr'):
        temp_SFR_xcorr(outs)

    else:
        sys.exit('You must specify what to plot from these options:\n' + \
                 'visualization, den_vel_proj, mass_vs_time, velocity_PDF, energy_vs_time,\n' + \
                 'energy_vs_radius, temperature_vs_time, temperature_vs_radius, energy_SFR_xcorr,\n' + \
                 'energy_vs_radius_comp, energy_vs_time_comp')
