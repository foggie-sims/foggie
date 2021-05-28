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
import shutil
import ast
import trident
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter
from scipy.ndimage import rotate
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
                        'pressure_vs_r_shaded   -  pressure over radius or radial velocity for each cell as a datashader plot\n' + \
                        '       or                 For these options, specify what type of pressure to plot with the --pressure_type keyword\n' + \
                        'pressure_vs_rv_shaded     and what you want to color-code the points by with the --shader_color keyword'
                        'support_vs_time        -  pressure support (thermal, turb, ram) relative to gravity at a specified radius over time\n' + \
                        'support_vs_radius      -  pressure support (thermal, turb, ram) relative to gravity over radius\n' + \
                        'support_vs_r_shaded    -  pressure support relative to gravity over radius or radial velocity for each cell as a datashader plot\n' + \
                        '       or                 For these options, specify what type of support to plot with the --pressure_type keyword\n' + \
                        'support_vs_rv_shaded      and what you want to color-code the points by with the --shader_color keyword\n' + \
                        'pressure_slice         -  x-slices of different types of pressure (specify with --pressure_type keyword)\n' + \
                        'support_slice          -  x-slices of different types of pressure support (specify with --pressure_type keyword)\n' + \
                        'velocity_slice         -  x-slices of the three spherical components of velocity, comparing the velocity,\n' + \
                        '                          the smoothed velocity, and the difference between the velocity and the smoothed velocity')

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

    parser.add_argument('--shader_color', metavar='shader_color', type=str, action='store', \
                        help='If plotting support_vs_r_shaded, what field do you want to color-code the points by?\n' + \
                        'Options are "temperature" and "metallicity" and the default is "temperature".')
    parser.set_defaults(shader_color='temperature')

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

    df['cat'] = cfunc(df['rand'])
    for index in np.flip(np.arange(n_labels), 0):
        df['cat'][df['x'] > value - sightline_length*(1.*index+1)/n_labels] = \
          list(color_key)[index]
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
    '''Plots different types of pressure (thermal, turbulent, rotational, bulk inflow/outflow ram)
    as functions of radius for the simulation output given by 'snap'.'''

    tablename_prefix = output_dir + 'stats_halo_00' + args.halo + '/' + args.run + '/Tables/'
    stats = Table.read(tablename_prefix + snap + '_' + args.filename + '.hdf5', path='all_data')
    Rvir = rvir_masses['radius'][rvir_masses['snapshot']==snap]

    if (args.spheres!='none'):
        stats_sph = Table.read(tablename_prefix + snap + '_' + args.spheres + '.hdf5', path='all_data')
        radius_list_sph = stats_sph['center_radius']

    plot_colors = ['r', 'g', 'b', 'm', 'c', 'k']
    plot_labels = ['Thermal', 'Turbulent', 'Rotation', 'Ram - in', 'Ram - out', 'Total']
    linestyles = ['-', '--', ':', '-.', '-.', '-']
    linewidths = [2, 2, 2, 1, 1, 3]

    radius_list = 0.5*(stats['inner_radius'] + stats['outer_radius'])
    zsnap = stats['redshift'][0]

    fig = plt.figure(figsize=(8,6), dpi=500)
    ax = fig.add_subplot(1,1,1)

    total_pres = np.zeros(len(radius_list))
    if (args.spheres!='none'):
        total_pres_sph = np.zeros(len(radius_list_sph))
    for i in range(len(plot_colors)):
        label = plot_labels[i]
        if (label=='Thermal'):
            ax.plot(radius_list, stats['net_log_pressure_med'], ls=linestyles[i], color=plot_colors[i], \
                    lw=linewidths[i], label=label)
            total_pres += 10.**stats['net_log_pressure_med']
            if (args.spheres!='none'):
                ax.plot(radius_list_sph, stats_sph['log_pressure_med'], ls='', marker='.', color=plot_colors[i], \
                        markersize=5, alpha=0.25)
                total_pres_sph += 10**stats_sph['log_pressure_med']
        if (label=='Turbulent'):
            turb_pres = 10.**stats['net_log_density_med']*(0.5*(stats['net_theta_velocity_sig']*1e5) + 0.5*(stats['net_phi_velocity_sig']*1e5))**2.
            total_pres += turb_pres
            ax.plot(radius_list, np.log10(turb_pres), ls=linestyles[i], color=plot_colors[i], lw=linewidths[i], label=label)
            if (args.spheres!='none'):
                turb_pres_sph = 10.**stats_sph['log_density_med']*(0.5*(stats_sph['theta_velocity_std']*1e5) + 0.5*(stats_sph['phi_velocity_std']*1e5))**2.
                ax.plot(radius_list_sph, np.log10(turb_pres_sph), ls='', marker='.', color=plot_colors[i], \
                        markersize=5, alpha=0.25)
                total_pres_sph += turb_pres_sph
        if (label=='Rotation'):
            rot_pres = 10.**stats['net_log_density_med']*((stats['net_theta_velocity_mu']*1e5)**2.+(stats['net_phi_velocity_mu']*1e5)**2.)
            total_pres += rot_pres
            ax.plot(radius_list, np.log10(rot_pres), ls=linestyles[i], color=plot_colors[i], lw=linewidths[i], label=label)
            if (args.spheres!='none'):
                rot_pres_sph = 10.**stats_sph['log_density_med']*(0.5*(stats_sph['theta_velocity_med']*1e5) + 0.5*(stats_sph['phi_velocity_med']*1e5))**2.
                ax.plot(radius_list_sph, np.log10(rot_pres_sph), ls='', marker='.', color=plot_colors[i], \
                        markersize=5, alpha=0.25)
                total_pres_sph += rot_pres_sph
        '''if (label=='Ram - in'):
            ram_in_pres = 10.**stats['net_log_density_med']*(stats['net_radial_velocity_muIn']**2.)*(stats['net_radial_velocity_Ain']*np.sqrt(2.*np.pi)*(stats['net_radial_velocity_sigIn'])**2.)
            print(ram_in_pres[np.where(radius_list>=50.)[0][0]])
            total_pres -= ram_in_pres
            ax.plot(radius_list, np.log10(ram_in_pres), ls=linestyles[i], color=plot_colors[i], lw=linewidths[i], label=label)
        if (label=='Ram - out'):
            ram_out_pres = 10.**stats['net_log_density_med']*(stats['net_radial_velocity_muOut']**2.)*(stats['net_radial_velocity_AOut']*np.sqrt(2.*np.pi)*(stats['net_radial_velocity_sigOut'])**2.)
            print(ram_out_pres[np.where(radius_list>=50.)[0][0]])
            total_pres += ram_out_pres
            ax.plot(radius_list, np.log10(ram_out_pres), ls=linestyles[i], color=plot_colors[i], lw=linewidths[i], label=label)'''
        if (label=='Total'):
            ax.plot(radius_list, np.log10(total_pres), ls=linestyles[i], color=plot_colors[i], lw=linewidths[i], label=label)
            if (args.spheres!='none'):
                ax.plot(radius_list_sph, np.log10(total_pres_sph), ls='', marker='.', color=plot_colors[i], \
                        markersize=5, alpha=0.25)

    ax.set_ylabel('log Pressure [erg/cm$^3$]', fontsize=18)
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
    plt.savefig(save_dir + snap + '_pressures_vs_r' + save_suffix + '.pdf')
    plt.close()

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
    if (system=='pleiades_cassi'):
        print('Copying directory to /tmp')
        snap_dir = '/tmp/' + snap
        shutil.copytree(foggie_dir + run_dir + snap, snap_dir)
        snap_name = snap_dir + '/' + snap
    else:
        snap_name = foggie_dir + run_dir + snap + '/' + snap
    ds, refine_box = foggie_load(snap_name, trackname, do_filter_particles=False, halo_c_v_name=halo_c_v_name, gravity=True, masses_dir=masses_dir)

    if (args.pressure_type=='all'):
        ptypes = ['thermal', 'turbulent', 'outflow', 'inflow', 'rotation', 'total']
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
    refine_res = int(500./(lvl1_res/(2.**level)))
    box = ds.covering_grid(level=level, left_edge=ds.halo_center_kpc-ds.arr([250.,250.,250.],'kpc'), dims=[refine_res, refine_res, refine_res])
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
        if (ptypes[i]=='thermal') or ('total' in ptypes):
            thermal_pressure = box['pressure'].in_units('erg/cm**3').v
            if (ptypes[i]=='thermal'):
                pressure = thermal_pressure
                pressure_label = 'Thermal'
        if (ptypes[i]=='turbulent') or (ptypes[i]=='rotation') or (ptypes[i]=='inflow') or \
           (ptypes[i]=='outflow') or ('total' in ptypes):
            vtheta = box['theta_velocity_corrected'].in_units('cm/s').v
            vphi = box['phi_velocity_corrected'].in_units('cm/s').v
            vr = box['radial_velocity_corrected'].in_units('cm/s').v
            smooth_vtheta = uniform_filter(vtheta, size=20)
            smooth_vphi = uniform_filter(vphi, size=20)
            smooth_vr = uniform_filter(vr, size=20)
            if (ptypes[i]=='turbulent') or ('total' in ptypes):
                sig_theta = (vtheta - smooth_vtheta)**2.
                sig_phi = (vphi - smooth_vphi)**2.
                sig_r = (vr - smooth_vr)**2.
                vdisp = np.sqrt((sig_theta + sig_phi + sig_r)/3.)
                turb_pressure = density*vdisp**2.
                if (ptypes[i]=='turbulent'):
                    pressure = turb_pressure
                    pressure_label = 'Turbulent'
            if (ptypes[i]=='rotation') or ('total' in ptypes):
                rot_pressure = density*smooth_vtheta**2. + density*smooth_vphi**2.
                if (ptypes[i]=='rotation'):
                    pressure = rot_pressure
                    pressure_label = 'Rotational'
            if (ptypes[i]=='inflow') or ('total' in ptypes):
                vr_in = 1.0*smooth_vr
                vr_in[smooth_vr > 0.] = 0.
                in_pressure = density*vr_in**2.
                if (ptypes[i]=='inflow'):
                    pressure = in_pressure
                    pressure_label = 'Inflow Ram'
            if (ptypes[i]=='outflow') or ('total' in ptypes):
                vr_out = 1.0*smooth_vr
                vr_out[smooth_vr < 0.] = 0.
                out_pressure = density*vr_out**2.
                if (ptypes[i]=='outflow'):
                    pressure = out_pressure
                    pressure_label = 'Outflow Ram'
        if (ptypes[i]=='total'):
            tot_pressure = thermal_pressure + turb_pressure + rot_pressure + out_pressure
            pressure = tot_pressure
            pressure_label = 'Total'
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
            export_image(img, save_dir + snap + '_' + ptypes[i] + '_pressure_vs_' + file_xaxis + '_' + args.shader_color + '-colored' + regions[j] + save_suffix)
            fig = plt.figure(figsize=(10,8),dpi=500)
            ax = fig.add_subplot(1,1,1)
            image = plt.imread(save_dir + snap + '_' + ptypes[i] + '_pressure_vs_' + file_xaxis + '_' + args.shader_color + '-colored' + regions[j] + save_suffix + '.png')
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
            plt.savefig(save_dir + snap + '_' + ptypes[i] + '_pressure_vs_' + file_xaxis + '_' + args.shader_color + '-colored' + regions[j] + save_suffix + '.pdf')
            os.system('rm ' + save_dir + snap + '_' + ptypes[i] + '_pressure_vs_' + file_xaxis + '_' + args.shader_color + '-colored' + regions[j] + save_suffix + '.png')
            plt.close()

    # Delete output from temp directory if on pleiades
    if (system=='pleiades_cassi'):
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

    if (system=='pleiades_cassi'):
        print('Copying directory to /tmp')
        snap_dir = '/tmp/' + snap
        shutil.copytree(foggie_dir + run_dir + snap, snap_dir)
        snap_name = snap_dir + '/' + snap
    else:
        snap_name = foggie_dir + run_dir + snap + '/' + snap
    ds, refine_box = foggie_load(snap_name, trackname, do_filter_particles=False, halo_c_v_name=halo_c_v_name, gravity=True, masses_dir=masses_dir)

    if (args.pressure_type=='all'):
        ptypes = ['thermal', 'turbulent', 'outflow', 'inflow', 'rotation', 'total']
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
    refine_res = int(500./dx)
    box = ds.covering_grid(level=level, left_edge=ds.halo_center_kpc-ds.arr([250.,250.,250.],'kpc'), dims=[refine_res, refine_res, refine_res])
    density = box['density'].in_units('g/cm**3').v
    temperature = box['temperature'].v
    r = box['radius_corrected'].in_units('kpc').v
    grav_pot = box['grav_pot'].v
    x_hat = box['x'].in_units('kpc').v - ds.halo_center_kpc[0].v
    y_hat = box['y'].in_units('kpc').v - ds.halo_center_kpc[1].v
    z_hat = box['z'].in_units('kpc').v - ds.halo_center_kpc[2].v
    grav_pot_grad = np.gradient(grav_pot, dx)
    g = -density*grav_pot_grad
    gx = g[0]
    gy = g[1]
    gz = g[2]
    x_hat /= r
    y_hat /= r
    z_hat /= r
    if (args.shader_color=='temperature'):
        data_frame['temperature'] = np.log10(temperature).flatten()
    if (args.shader_color=='metallicity') or (args.region_filter=='metallicity'):
        metallicity = box['metallicity'].in_units('Zsun').v
        data_frame['metallicity'] = metallicity.flatten()
    if (args.plot=='support_vs_r_shaded'):
        data_frame['radius'] = r.flatten()
    if (args.plot=='support_vs_rv_shaded') or (args.region_filter=='velocity'):
        rv = box['radial_velocity_corrected'].in_units('km/s').v
        data_frame['rv'] = rv.flatten()
    for i in range(len(ptypes)):
        if (ptypes[i]=='thermal') or ('total' in ptypes):
            thermal_pressure = box['pressure'].in_units('erg/cm**3').v
            if (ptypes[i]=='thermal'):
                pressure = thermal_pressure
                pressure_label = 'Thermal'
        if (ptypes[i]=='turbulent') or (ptypes[i]=='rotation') or (ptypes[i]=='inflow') or \
           (ptypes[i]=='outflow') or ('total' in ptypes):
            vtheta = box['theta_velocity_corrected'].in_units('cm/s').v
            vphi = box['phi_velocity_corrected'].in_units('cm/s').v
            vr = box['radial_velocity_corrected'].in_units('cm/s').v
            smooth_vtheta = uniform_filter(vtheta, size=20)
            smooth_vphi = uniform_filter(vphi, size=20)
            smooth_vr = uniform_filter(vr, size=20)
            if (ptypes[i]=='turbulent') or ('total' in ptypes):
                sig_theta = (vtheta - smooth_vtheta)**2.
                sig_phi = (vphi - smooth_vphi)**2.
                sig_r = (vr - smooth_vr)**2.
                vdisp = np.sqrt((sig_theta + sig_phi + sig_r)/3.)
                turb_pressure = density*vdisp**2.
                if (ptypes[i]=='turbulent'):
                    pressure = turb_pressure
                    pressure_label = 'Turbulent'
            if (ptypes[i]=='rotation') or ('total' in ptypes):
                rot_pressure = density*smooth_vtheta**2. + density*smooth_vphi**2.
                if (ptypes[i]=='rotation'):
                    pressure = rot_pressure
                    pressure_label = 'Rotational'
            if (ptypes[i]=='inflow') or ('total' in ptypes):
                vr_in = 1.0*smooth_vr
                vr_in[smooth_vr > 0.] = 0.
                in_pressure = density*vr_in**2.
                if (ptypes[i]=='inflow'):
                    pressure = in_pressure
                    pressure_label = 'Inflow Ram'
            if (ptypes[i]=='outflow') or ('total' in ptypes):
                vr_out = 1.0*smooth_vr
                vr_out[smooth_vr < 0.] = 0.
                out_pressure = density*vr_out**2.
                if (ptypes[i]=='outflow'):
                    pressure = out_pressure
                    pressure_label = 'Outflow Ram'
        if (ptypes[i]=='total'):
            tot_pressure = thermal_pressure + turb_pressure + rot_pressure + out_pressure
            pressure = tot_pressure
            pressure_label = 'Total'
            in_pres_grad = np.gradient(in_pressure, dx)
            gx -= in_pres_grad[0]
            gy -= in_pres_grad[1]
            gz -= in_pres_grad[2]

        gr = gx*x_hat + gy*y_hat + gz*z_hat
        pres_grad = np.gradient(pressure, dx)
        pr = pres_grad[0]*x_hat + pres_grad[1]*y_hat + pres_grad[2]*z_hat
        support = np.sqrt(pr**2./gr**2.)

        support_filtered = 1.0*support
        support_filtered[(density > cgm_density_max) & (temperature < cgm_temperature_min)] = 1e-10
        for j in range(len(regions)):
            if (regions[j]=='_low-T'):
                support_masked = 1.0*support_filtered
                support_masked[(temperature > 10**5)] = 1e-10
            if (regions[j]=='_mid-T'):
                support_masked = 1.0*support_filtered
                support_masked[(temperature > 10**6) | (temperature < 10**5)] = 1e-10
            if (regions[j]=='_high-T'):
                support_masked = 1.0*support_filtered
                support_masked[(temperature < 10**6)] = 1e-10
            if (regions[j]=='_low-Z'):
                support_masked = 1.0*support_filtered
                support_masked[(metallicity > 0.01)] = 1e-10
            if (regions[j]=='_mid-Z'):
                support_masked = 1.0*support_filtered
                support_masked[(metallicity < 0.01) | (metallicity > 1)] = 1e-10
            if (regions[j]=='_high-Z'):
                support_masked = 1.0*support_filtered
                support_masked[(metallicity < 1)] = 1e-10
            if (regions[j]=='_low-v'):
                support_masked = 1.0*support_filtered
                support_masked[(rv > 0.5*vff)] = 1e-10
            if (regions[j]=='_mid-v'):
                support_masked = 1.0*support_filtered
                support_masked[(rv < 0.5*vff) | (rv > vesc)] = 1e-10
            if (regions[j]=='_high-v'):
                support_masked = 1.0*support_filtered
                support_masked[(rv < vesc)] = 1e-10
            if (regions[j]==''): support_masked = support_filtered
            support_masked[support_masked <= 0.] = 1e-10
            data_frame['support'] = np.log10(support_masked).flatten()
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
            y_range = [-4, 4]
            cvs = dshader.Canvas(plot_width=1000, plot_height=800, x_range=x_range, y_range=y_range)
            if (args.plot=='support_vs_r_shaded'):
                agg = cvs.points(data_frame, 'radius', 'support', dshader.count_cat(cat))
                file_xaxis = 'r'
            if (args.plot=='support_vs_rv_shaded'):
                agg = cvs.points(data_frame, 'rv', 'support', dshader.count_cat(cat))
                file_xaxis = 'rv'
            img = tf.spread(tf.shade(agg, color_key=color_key, how='eq_hist',min_alpha=40), shape='square', px=0)
            export_image(img, save_dir + snap + '_' + ptypes[i] + '_support_vs_' + file_xaxis + '_' + args.shader_color + '-colored' + regions[j] + save_suffix)
            fig = plt.figure(figsize=(10,8),dpi=500)
            ax = fig.add_subplot(1,1,1)
            image = plt.imread(save_dir + snap + '_' + ptypes[i] + '_support_vs_' + file_xaxis + '_' + args.shader_color + '-colored' + regions[j] + save_suffix + '.png')
            ax.imshow(image, extent=[x_range[0],x_range[1],y_range[0],y_range[1]])
            ax.set_aspect(8*abs(x_range[1]-x_range[0])/(10*abs(y_range[1]-y_range[0])))
            if (args.plot=='support_vs_r_shaded'): ax.set_xlabel('Radius [kpc]', fontsize=20)
            if (args.plot=='support_vs_rv_shaded'): ax.set_xlabel('Radial velocity [km/s]', fontsize=20)
            ax.set_ylabel('log ' + pressure_label + ' Pressure Support', fontsize=20)
            ax.plot(x_range, [0,0], 'k-', lw=2)
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
            plt.savefig(save_dir + snap + '_' + ptypes[i] + '_support_vs_' + file_xaxis + '_' + args.shader_color + '-colored' + regions[j] + save_suffix + '.pdf')
            os.system('rm ' + save_dir + snap + '_' + ptypes[i] + '_support_vs_' + file_xaxis + '_' + args.shader_color + '-colored' + regions[j] + save_suffix + '.png')
            plt.close()

    # Delete output from temp directory if on pleiades
    if (system=='pleiades_cassi'):
        print('Deleting directory from /tmp')
        shutil.rmtree(snap_dir)

def pressure_slice(snap):
    '''Plots a slice of pressure through the center of the halo. The option --pressure_type indicates
    what type of pressure to plot.'''

    masses_ind = np.where(masses['snapshot']==snap)[0]
    Menc_profile = IUS(np.concatenate(([0],masses['radius'][masses_ind])), np.concatenate(([0],masses['total_mass'][masses_ind])))
    Mvir = rvir_masses['total_mass'][rvir_masses['snapshot']==snap]
    Rvir = rvir_masses['radius'][rvir_masses['snapshot']==snap][0]

    if (system=='pleiades_cassi'):
        print('Copying directory to /tmp')
        snap_dir = '/tmp/' + snap
        shutil.copytree(foggie_dir + run_dir + snap, snap_dir)
        snap_name = snap_dir + '/' + snap
    else:
        snap_name = foggie_dir + run_dir + snap + '/' + snap
    ds, refine_box = foggie_load(snap_name, trackname, do_filter_particles=False, halo_c_v_name=halo_c_v_name, gravity=True, masses_dir=masses_dir)

    if (args.pressure_type=='all'):
        ptypes = ['thermal', 'turbulent', 'outflow', 'inflow', 'rotation', 'total']
    elif (',' in args.pressure_type):
        ptypes = args.pressure_type.split(',')
    else:
        ptypes = [args.pressure_type]

    pix_res = float(np.min(refine_box['dx'].in_units('kpc')))  # at level 11
    lvl1_res = pix_res*2.**11.
    level = 9
    refine_res = int(500./(lvl1_res/(2.**level)))
    box = ds.covering_grid(level=level, left_edge=ds.halo_center_kpc-ds.arr([250.,250.,250.],'kpc'), dims=[refine_res, refine_res, refine_res])
    density = box['density'].in_units('g/cm**3').v
    temperature = box['temperature'].v

    for i in range(len(ptypes)):
        if (ptypes[i]=='thermal') or ('total' in ptypes):
            thermal_pressure = box['pressure'].in_units('erg/cm**3').v
            if (ptypes[i]=='thermal'):
                pressure = thermal_pressure
                pressure_label = 'Thermal'
        if (ptypes[i]=='turbulent') or (ptypes[i]=='rotation') or (ptypes[i]=='inflow') or \
           (ptypes[i]=='outflow') or ('total' in ptypes):
            vtheta = box['theta_velocity_corrected'].in_units('cm/s').v
            vphi = box['phi_velocity_corrected'].in_units('cm/s').v
            vr = box['radial_velocity_corrected'].in_units('cm/s').v
            smooth_vtheta = uniform_filter(vtheta, size=20)
            smooth_vphi = uniform_filter(vphi, size=20)
            smooth_vr = uniform_filter(vr, size=20)
            if (ptypes[i]=='turbulent') or ('total' in ptypes):
                sig_theta = (vtheta - smooth_vtheta)**2.
                sig_phi = (vphi - smooth_vphi)**2.
                sig_r = (vr - smooth_vr)**2.
                vdisp = np.sqrt((sig_theta + sig_phi + sig_r)/3.)
                turb_pressure = density*vdisp**2.
                if (ptypes[i]=='turbulent'):
                    pressure = turb_pressure
                    print(np.min(pressure), np.max(pressure))
                    pressure_label = 'Turbulent'
            if (ptypes[i]=='rotation') or ('total' in ptypes):
                rot_pressure = density*smooth_vtheta**2. + density*smooth_vphi**2.
                if (ptypes[i]=='rotation'):
                    pressure = rot_pressure
                    pressure_label = 'Rotational'
            if (ptypes[i]=='inflow') or ('total' in ptypes):
                vr_in = 1.0*smooth_vr
                vr_in[smooth_vr > 0.] = 0.
                in_pressure = density*vr_in**2.
                if (ptypes[i]=='inflow'):
                    pressure = in_pressure
                    pressure_label = 'Inflow Ram'
            if (ptypes[i]=='outflow') or ('total' in ptypes):
                vr_out = 1.0*smooth_vr
                vr_out[smooth_vr < 0.] = 0.
                out_pressure = density*vr_out**2.
                if (ptypes[i]=='outflow'):
                    pressure = out_pressure
                    pressure_label = 'Outflow Ram'
        if (ptypes[i]=='total'):
            tot_pressure = thermal_pressure + turb_pressure + rot_pressure + out_pressure
            pressure = tot_pressure
            pressure_label = 'Total'

        pressure = np.ma.masked_where((density > cgm_density_max) & (temperature < cgm_temperature_min), pressure)
        fig = plt.figure(figsize=(12,10),dpi=500)
        ax = fig.add_subplot(1,1,1)
        cmap = copy.copy(mpl.cm.get_cmap(pressure_color_map))
        cmap.set_over(color='w')
        # Need to rotate to match up with how yt plots it
        im = ax.imshow(rotate(np.log10(pressure[len(pressure)//2,:,:]),90), cmap=cmap, norm=colors.Normalize(vmin=-18, vmax=-12), \
                  extent=[-250,250,-250,250])
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
        plt.savefig(save_dir + snap + '_' + ptypes[i] + '_pressure_slice_x' + save_suffix + '.pdf')

    # Delete output from temp directory if on pleiades
    if (system=='pleiades_cassi'):
        print('Deleting directory from /tmp')
        shutil.rmtree(snap_dir)

def support_slice(snap):
    '''Plots a slice of pressure support through the center of the halo. The option --pressure_type indicates
    what type of pressure support to plot.'''

    masses_ind = np.where(masses['snapshot']==snap)[0]
    Menc_profile = IUS(np.concatenate(([0],masses['radius'][masses_ind])), np.concatenate(([0],masses['total_mass'][masses_ind])))
    Mvir = rvir_masses['total_mass'][rvir_masses['snapshot']==snap]
    Rvir = rvir_masses['radius'][rvir_masses['snapshot']==snap][0]

    if (system=='pleiades_cassi'):
        print('Copying directory to /tmp')
        snap_dir = '/tmp/' + snap
        shutil.copytree(foggie_dir + run_dir + snap, snap_dir)
        snap_name = snap_dir + '/' + snap
    else:
        snap_name = foggie_dir + run_dir + snap + '/' + snap
    ds, refine_box = foggie_load(snap_name, trackname, do_filter_particles=False, halo_c_v_name=halo_c_v_name, gravity=True, masses_dir=masses_dir)
    ds.add_gradient_fields(('gas','pressure'))

    if (args.pressure_type=='all'):
        ptypes = ['thermal', 'turbulent', 'outflow', 'inflow', 'rotation', 'total']
    elif (',' in args.pressure_type):
        ptypes = args.pressure_type.split(',')
    else:
        ptypes = [args.pressure_type]

    pix_res = float(np.min(refine_box['dx'].in_units('kpc')))  # at level 11
    lvl1_res = pix_res*2.**11.
    level = 9
    dx = lvl1_res/(2.**level)
    refine_res = int(500./dx)
    box = ds.covering_grid(level=level, left_edge=ds.halo_center_kpc-ds.arr([250.,250.,250.],'kpc'), dims=[refine_res, refine_res, refine_res])
    density = box['density'].in_units('g/cm**3').v
    temperature = box['temperature'].v
    grav_pot = box['grav_pot'].v
    x_hat = box['x'].in_units('kpc').v - ds.halo_center_kpc[0].v
    y_hat = box['y'].in_units('kpc').v - ds.halo_center_kpc[1].v
    z_hat = box['z'].in_units('kpc').v - ds.halo_center_kpc[2].v
    r = box['radius_corrected'].in_units('kpc').v
    grav_pot_grad = np.gradient(grav_pot, dx)
    g = -density*grav_pot_grad
    gx = g[0]
    gy = g[1]
    gz = g[2]
    x_hat /= r
    y_hat /= r
    z_hat /= r

    for i in range(len(ptypes)):
        if (ptypes[i]=='thermal') or ('total' in ptypes):
            thermal_pressure = box['pressure'].in_units('erg/cm**3').v
            if (ptypes[i]=='thermal'):
                pressure = thermal_pressure
                pressure_label = 'Thermal'
        if (ptypes[i]=='turbulent') or (ptypes[i]=='rotation') or (ptypes[i]=='inflow') or \
           (ptypes[i]=='outflow') or ('total' in ptypes):
            vtheta = box['theta_velocity_corrected'].in_units('cm/s').v
            vphi = box['phi_velocity_corrected'].in_units('cm/s').v
            vr = box['radial_velocity_corrected'].in_units('cm/s').v
            smooth_vtheta = uniform_filter(vtheta, size=20)
            smooth_vphi = uniform_filter(vphi, size=20)
            smooth_vr = uniform_filter(vr, size=20)
            if (ptypes[i]=='turbulent') or ('total' in ptypes):
                sig_theta = (vtheta - smooth_vtheta)**2.
                sig_phi = (vphi - smooth_vphi)**2.
                sig_r = (vr - smooth_vr)**2.
                vdisp = np.sqrt((sig_theta + sig_phi + sig_r)/3.)
                turb_pressure = density*vdisp**2.
                if (ptypes[i]=='turbulent'):
                    pressure = turb_pressure
                    pressure_label = 'Turbulent'
            if (ptypes[i]=='rotation') or ('total' in ptypes):
                rot_pressure = density*smooth_vtheta**2. + density*smooth_vphi**2.
                if (ptypes[i]=='rotation'):
                    pressure = rot_pressure
                    pressure_label = 'Rotational'
            if (ptypes[i]=='inflow') or ('total' in ptypes):
                vr_in = 1.0*smooth_vr
                vr_in[smooth_vr > 0.] = 0.
                in_pressure = density*vr_in**2.
                if (ptypes[i]=='inflow'):
                    pressure = in_pressure
                    pressure_label = 'Inflow Ram'
            if (ptypes[i]=='outflow') or ('total' in ptypes):
                vr_out = 1.0*smooth_vr
                vr_out[smooth_vr < 0.] = 0.
                out_pressure = density*vr_out**2.
                if (ptypes[i]=='outflow'):
                    pressure = out_pressure
                    pressure_label = 'Outflow Ram'
        if (ptypes[i]=='total'):
            tot_pressure = thermal_pressure + turb_pressure + rot_pressure + out_pressure
            pressure = tot_pressure
            pressure_label = 'Total'
            in_pres_grad = np.gradient(in_pressure, dx)
            gx -= in_pres_grad[0]
            gy -= in_pres_grad[1]
            gz -= in_pres_grad[2]

        gr = gx*x_hat + gy*y_hat + gz*z_hat
        pres_grad = np.gradient(pressure, dx)
        pr = pres_grad[0]*x_hat + pres_grad[1]*y_hat + pres_grad[2]*z_hat
        support = np.sqrt(pr**2./gr**2.)

        support[(density > cgm_density_max) & (temperature < cgm_temperature_min)] = 1.
        fig = plt.figure(figsize=(12,10),dpi=500)
        ax = fig.add_subplot(1,1,1)
        cmap = copy.copy(mpl.cm.BrBG)
        # Need to rotate to match up with how yt plots it
        im = ax.imshow(rotate(np.log10(support[len(support)//2,:,:]),90), cmap=cmap, norm=colors.Normalize(vmin=-2, vmax=2), \
                  extent=[-250,250,-250,250])
        ax.set_xlabel('y [kpc]', fontsize=20)
        ax.set_ylabel('z [kpc]', fontsize=20)
        ax.tick_params(axis='both', which='both', direction='in', length=8, width=2, pad=5, labelsize=20, \
          top=True, right=True)
        cax = fig.add_axes([0.855, 0.08, 0.03, 0.9])
        cax.tick_params(axis='both', which='both', direction='in', length=8, width=2, pad=5, labelsize=20, \
          top=True, right=True)
        fig.colorbar(im, cax=cax, orientation='vertical')
        ax.text(1.15, 0.5, 'log ' + pressure_label + ' Pressure Support', fontsize=20, rotation='vertical', ha='center', va='center', transform=ax.transAxes)
        plt.subplots_adjust(bottom=0.08, top=0.98, left=0.08, right=0.88)
        plt.savefig(save_dir + snap + '_' + ptypes[i] + '_support_slice_x' + save_suffix + '.pdf')

    # Delete output from temp directory if on pleiades
    if (system=='pleiades_cassi'):
        print('Deleting directory from /tmp')
        shutil.rmtree(snap_dir)

def velocity_slice(snap):
    '''Plots slices of radial, theta, and phi velocity fields through the center of the halo. The field,
    the smoothed field, and the difference between the field and the smoothed field are all plotted.'''

    masses_ind = np.where(masses['snapshot']==snap)[0]
    Menc_profile = IUS(np.concatenate(([0],masses['radius'][masses_ind])), np.concatenate(([0],masses['total_mass'][masses_ind])))
    Mvir = rvir_masses['total_mass'][rvir_masses['snapshot']==snap]
    Rvir = rvir_masses['radius'][rvir_masses['snapshot']==snap][0]

    if (system=='pleiades_cassi'):
        print('Copying directory to /tmp')
        snap_dir = '/tmp/' + snap
        shutil.copytree(foggie_dir + run_dir + snap, snap_dir)
        snap_name = snap_dir + '/' + snap
    else:
        snap_name = foggie_dir + run_dir + snap + '/' + snap
    ds, refine_box = foggie_load(snap_name, trackname, do_filter_particles=False, halo_c_v_name=halo_c_v_name, gravity=True, masses_dir=masses_dir)

    vtypes = ['radial', 'theta', 'phi']
    vlabels = ['Radial', '$\\theta$', '$\phi$']
    vmins = [-500, -200, -200]
    vmaxes = [500, 200, 200]

    pix_res = float(np.min(refine_box['dx'].in_units('kpc')))  # at level 11
    lvl1_res = pix_res*2.**11.
    level = 9
    refine_res = int(500./(lvl1_res/(2.**level)))
    box = ds.covering_grid(level=level, left_edge=ds.halo_center_kpc-ds.arr([250.,250.,250.],'kpc'), dims=[refine_res, refine_res, refine_res])
    density = box['density'].in_units('g/cm**3').v
    temperature = box['temperature'].v
    vtheta = box['theta_velocity_corrected'].in_units('cm/s').v
    vphi = box['phi_velocity_corrected'].in_units('cm/s').v
    vr = box['radial_velocity_corrected'].in_units('cm/s').v
    smooth_vtheta = uniform_filter(vtheta, size=20)
    smooth_vphi = uniform_filter(vphi, size=20)
    smooth_vr = uniform_filter(vr, size=20)
    sig_theta = vtheta - smooth_vtheta
    sig_phi = vphi - smooth_vphi
    sig_r = vr - smooth_vr

    for i in range(len(vtypes)):
        v = box[vtypes[i] + '_velocity_corrected'].in_units('km/s').v
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
                  extent=[-250,250,-250,250])
        ax1.set_xlabel('y [kpc]', fontsize=20)
        ax1.set_ylabel('z [kpc]', fontsize=20)
        ax1.tick_params(axis='both', which='both', direction='in', length=8, width=2, pad=5, labelsize=20, \
          top=True, right=True)
        cb = fig.colorbar(im1, ax=ax1, orientation='vertical', pad=0)
        cb.ax.tick_params(labelsize=16)
        cb.ax.set_ylabel(vlabels[i] + ' Velocity [km/s]', fontsize=16)
        im2 = ax2.imshow(rotate(smooth_v[len(smooth_v)//2,:,:],90), cmap=cmap, norm=colors.Normalize(vmin=vmins[i], vmax=vmaxes[i]), \
                  extent=[-250,250,-250,250])
        ax2.set_xlabel('y [kpc]', fontsize=20)
        ax2.set_ylabel('z [kpc]', fontsize=20)
        ax2.tick_params(axis='both', which='both', direction='in', length=8, width=2, pad=5, labelsize=20, \
          top=True, right=True)
        cb = fig.colorbar(im2, ax=ax2, orientation='vertical', pad=0)
        cb.ax.tick_params(labelsize=16)
        cb.ax.set_ylabel('Smoothed ' + vlabels[i] + ' Velocity [km/s]', fontsize=16)
        im3 = ax3.imshow(rotate(sig_v[len(sig_v)//2,:,:],90), cmap=cmap, norm=colors.Normalize(vmin=-200, vmax=200), \
                  extent=[-250,250,-250,250])
        ax3.set_xlabel('y [kpc]', fontsize=20)
        ax3.set_ylabel('z [kpc]', fontsize=20)
        ax3.tick_params(axis='both', which='both', direction='in', length=8, width=2, pad=5, labelsize=20, \
          top=True, right=True)
        cb = fig.colorbar(im3, ax=ax3, orientation='vertical', pad=0)
        cb.ax.tick_params(labelsize=16)
        cb.ax.set_ylabel(vlabels[i] + ' Velocity - Smoothed Velocity [km/s]', fontsize=16)
        plt.subplots_adjust(bottom=0.15, top=0.97, left=0.04, right=0.97, wspace=0.22)
        plt.savefig(save_dir + snap + '_' + vtypes[i] + '_velocity_slice_x' + save_suffix + '.pdf')

    # Delete output from temp directory if on pleiades
    if (system=='pleiades_cassi'):
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

    if (args.plot=='pressure_vs_radius'):
        for i in range(len(outs)):
            pressures_vs_radius(outs[i])
    elif (args.plot=='support_vs_radius'):
        for i in range(len(outs)):
            support_vs_radius(outs[i])
    elif (args.plot=='velocity_PDF'):
        for i in range(len(outs)):
            velocity_PDF(outs[i])
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
            target = support_vr_r_rv_shaded
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
