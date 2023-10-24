##!/usr/bin/env python3

"""

    Title :      plot_vdisp_frb
    Notes :      make FRB and projection plots for velocity dispersion; this is a standalone script
    Output :     projection plots as png files
    Author :     Ayan Acharyya
    Started :    October 2023
    Example :    run plot_vdisp_frb.py --system ayan_local --halo 8508 --output RD0030 --upto_kpc 10 --proj x --res_arc 0.1 --get3d --plot_frb
"""
import numpy as np
from matplotlib import pyplot as plt
plt.style.use('seaborn-whitegrid')
plt.rcParams['axes.edgecolor'] = 'black'
plt.rcParams['axes.linewidth'] = 2
plt.style.use('seaborn-white')
from mpl_toolkits.axes_grid1 import make_axes_locatable
from pathlib import Path
from astropy import units as u
from astropy.cosmology import FlatLambdaCDM
from scipy.ndimage import gaussian_filter

from foggie.utils.get_run_loc_etc import *
from foggie.utils.yt_fields import *
from foggie.utils.foggie_load import *

import time, datetime, argparse, yt, re

start_time = time.time()

# ---------------------------------------------------------------------------
def get_kpc_from_arc_at_redshift(arcseconds, redshift):
    '''
    Function to convert arcseconds on sky to physical kpc, at a given redshift
    '''
    cosmo = FlatLambdaCDM(H0=67.8, Om0=0.308)
    d_A = cosmo.angular_diameter_distance(z=redshift)
    kpc = (d_A * arcseconds * u.arcsec).to(u.kpc, u.dimensionless_angles()).value # in kpc
    print('Converted resolution of %.2f arcseconds to %.2F kpc at target redshift of %.2f' %(arcseconds, kpc, redshift))
    return kpc

# -------------------------------------------------------------------------------------------------------------
def get_smoothing_scale(data, args):
    '''
    Function to derive a smoothing scale for computing velocity dispersion
    '''
    pix_res = float(np.min(data['dx'].in_units('kpc')))  # at level 11
    cooling_level = int(re.search('nref(.*)c', args.run).group(1))
    string_to_skip = '%dc' % cooling_level
    forced_level = int(re.search('nref(.*)f', args.run[args.run.find(string_to_skip) + len(string_to_skip):]).group(1))
    lvl1_res = pix_res * 2. ** cooling_level
    level = forced_level
    dx = lvl1_res / (2. ** level)
    smooth_scale = int(25. / dx) / 6.
    print('Smoothing velocity field at %.2f kpc to compute velocity dispersion..'%smooth_scale)

    return smooth_scale

# --------------------------------------------------------------------------------------------------------------
def get_velocity_dispersion_3d(field, data):
    '''
    Function to compute gas velocity dispersion, given a YT box object
    This function is based on Cassi's script vdisp_vs_mass_res() in foggie/turbulence/turbulence.py
    '''
    smooth_scale = 6.33 # kpc; this is hard-oded for a very specific scenario of nref11c_nref9f
    vx = data['vx_corrected'].in_units('km/s').v
    vy = data['vy_corrected'].in_units('km/s').v
    vz = data['vz_corrected'].in_units('km/s').v
    smooth_vx = gaussian_filter(vx, smooth_scale)
    smooth_vy = gaussian_filter(vy, smooth_scale)
    smooth_vz = gaussian_filter(vz, smooth_scale)
    sig_x = (vx - smooth_vx)**2.
    sig_y = (vy - smooth_vy)**2.
    sig_z = (vz - smooth_vz)**2.
    vdisp = np.sqrt((sig_x + sig_y + sig_z)/3.)
    vdisp = yt.YTArray(vdisp, 'km/s')
    return vdisp

# --------------------------------------------------------------------------------------------------------------
def get_velocity_dispersion_x(field, data):
    '''
    Function to compute gas velocity dispersion, given a YT box object
    This function is based on Cassi's script vdisp_vs_mass_res() in foggie/turbulence/turbulence.py
    '''
    smooth_scale = 6.33 # kpc; this is hard-oded for a very specific scenario of nref11c_nref9f
    v = data['vx' + '_corrected'].in_units('km/s').v
    smooth_v = gaussian_filter(v, smooth_scale)
    vdisp = np.abs(v - smooth_v)
    vdisp = yt.YTArray(vdisp, 'km/s')

    return vdisp

# --------------------------------------------------------------------------------------------------------------
def get_velocity_dispersion_y(field, data):
    '''
    Function to compute gas velocity dispersion, given a YT box object
    This function is based on Cassi's script vdisp_vs_mass_res() in foggie/turbulence/turbulence.py
    '''
    smooth_scale = 6.33 # kpc; this is hard-oded for a very specific scenario of nref11c_nref9f
    v = data['vy' + '_corrected'].in_units('km/s').v
    smooth_v = gaussian_filter(v, smooth_scale)
    vdisp = np.abs(v - smooth_v)
    vdisp = yt.YTArray(vdisp, 'km/s')

    return vdisp

# --------------------------------------------------------------------------------------------------------------
def get_velocity_dispersion_z(field, data):
    '''
    Function to compute gas velocity dispersion, given a YT box object
    This function is based on Cassi's script vdisp_vs_mass_res() in foggie/turbulence/turbulence.py
    '''
    smooth_scale = 6.33 # kpc; this is hard-oded for a very specific scenario of nref11c_nref9f
    v = data['vz' + '_corrected'].in_units('km/s').v
    smooth_v = gaussian_filter(v, smooth_scale)
    vdisp = np.abs(v - smooth_v)
    vdisp = yt.YTArray(vdisp, 'km/s')

    return vdisp

# ----------------------------------------------------------------
def get_vdisp_frb(box, box_center, box_width, args):
    '''
    Function to convert a given dataset to Fixed Resolution Buffer (FRB) for a given angle of projection and resolution (args.res)
    :return: FRB (2D numpy array)
    '''
    proj = box.ds.proj(('gas', 'velocity_dispersion_' + args.projection), args.projection, center=box_center, data_source=box)
    frb = proj.to_frb(box.ds.arr(box_width, 'kpc'), args.ncells, center=box_center)

    map_vdisp = frb['gas', 'velocity_dispersion_3d'] if args.get3d else frb['gas', 'velocity_dispersion_' + args.projection]# cm*km/s
    map_vdisp = (map_vdisp / (2 * yt.YTArray(args.galrad, 'kpc'))).in_units('km/s') # km/s

    return map_vdisp

# -----------------------------------------
def get_box(args):
    '''
    Function to load data and extract small box
    '''
    # ----------- load in dataset-------------------
    halos_df_name = args.code_path + 'halo_infos/00' + args.halo + '/' + args.run + '/halo_c_v'
    ds, refine_box = load_sim(args, region='refine_box', do_filter_particles=True, disk_relative=False, halo_c_v_name=halos_df_name)
    ds.add_field(('gas', 'velocity_dispersion_3d'), function=get_velocity_dispersion_3d, units='km/s', take_log=False, sampling_type='cell')
    ds.add_field(('gas', 'velocity_dispersion_x'), function=get_velocity_dispersion_x, units='km/s', take_log=False, sampling_type='cell')
    ds.add_field(('gas', 'velocity_dispersion_y'), function=get_velocity_dispersion_y, units='km/s', take_log=False, sampling_type='cell')
    ds.add_field(('gas', 'velocity_dispersion_z'), function=get_velocity_dispersion_z, units='km/s', take_log=False, sampling_type='cell')
    args.current_redshift = ds.current_redshift

    # ------------- extract small box -------------------------------
    if args.res_arc is not None:
        args.res = get_kpc_from_arc_at_redshift(args.res_arc, args.current_redshift)
        native_res_at_z = 0.27 / (1 + args.current_redshift)  # converting from comoving kpc to physical kpc
        if args.res < native_res_at_z:
            print('Computed resolution %.2f kpc is below native FOGGIE res at z=%.2f, so we set resolution to the native res = %.2f kpc.' % (args.res, args.current_redshift, native_res_at_z))
            args.res = native_res_at_z  # kpc
    else:
        args.res = args.res_kpc / (1 + args.current_redshift) / 0.695  # converting from comoving kcp h^-1 to physical kpc

    args.galrad = args.upto_kpc / (1 + args.current_redshift) / 0.695  # fit within a fixed comoving kpc h^-1, 0.695 is Hubble constant
    args.ncells = int(2 * args.galrad / args.res)

    # extract the required box
    box_center = ds.halo_center_kpc
    box_width = 2 * args.galrad * kpc
    box = ds.r[box_center[0] - box_width / 2.: box_center[0] + box_width / 2., box_center[1] - box_width / 2.: box_center[1] + box_width / 2., box_center[2] - box_width / 2.: box_center[2] + box_width / 2., ]

    return box, box_center, box_width

# -------------------------------------------------
def saveplot(fig, label, args):
    '''
    Function to save figure
    '''
    args.fig_dir = args.output_dir + 'figs/' + args.output + '/'
    Path(args.fig_dir).mkdir(parents=True, exist_ok=True)
    args.upto_text = '_upto%.1Fckpchinv' % args.upto_kpc
    args.res_text = '_res%.1fkpc' % float(args.res)
    args.weightby_text = '_weightby_%s' %args.weightby if args.weightby is not None else ''
    outfile_rootname = '%s_%s_%s_%s%s%s%s.png' % (args.halo, args.output, args.projection, label, args.upto_text, args.weightby_text, args.res_text)
    figname = args.fig_dir + outfile_rootname
    fig.savefig(figname)
    print('Saved plot as ' + figname)

    fig.show()
    return fig

# -----------------------------------------
def plot_vdisp_projection(box, box_center, box_width, args):
    '''
    Function to generate a projection plot directly from the data
    '''
    field = ('gas', 'velocity_dispersion_3d') if args.get3d else ('gas', 'velocity_dispersion_' + args.projection)
    unit = 'cm*km/s' if args.weightby is None else 'km/s'
    weight_field = ('gas', args.weightby) if args.weightby is not None else None

    prj = yt.ProjectionPlot(box.ds, args.projection, field, center=box_center, data_source=box, width=box_width, weight_field=weight_field, fontsize=args.fontsize)
    prj.set_unit(field, unit)
    if args.cmin is not None: prj.set_zlim(field, zmin=args.cmin, zmax=args.cmax)

    # ------plotting onto a matplotlib figure--------------
    fig, ax = plt.subplots(figsize=(7, 7))
    prj.plots[field].axes = ax
    divider = make_axes_locatable(ax)
    prj._setup_plots()

    cax = divider.append_axes('right', size='5%', pad=0.05)
    cbar = fig.colorbar(prj.plots[field].cb.mappable, orientation='vertical', cax=cax)
    cbar.ax.tick_params(labelsize=args.fontsize)
    cbar.set_label(prj.plots[field].cax.get_ylabel(), fontsize=args.fontsize)

    ax.xaxis.set_major_locator(plt.MaxNLocator(5))
    ax.yaxis.set_major_locator(plt.MaxNLocator(5))
    ax.set_xticklabels(['%.1F' % item for item in ax.get_xticks()], fontsize=args.fontsize)
    ax.set_yticklabels(['%.1F' % item for item in ax.get_yticks()], fontsize=args.fontsize)
    ax.set_xlabel(ax.get_xlabel(), fontsize=args.fontsize)
    ax.set_ylabel(ax.get_ylabel(), fontsize=args.fontsize)

    fig = saveplot(fig, '3Dprojection' if args.get3d else args.projection + 'projection', args)
    return fig

# -----------------------------------------
def plot_vdisp_frb(box, box_center, box_width, args):
    '''
    Function to generate a projection plot from FRB
    '''
    map_vdisp = get_vdisp_frb(box, box_center, box_width, args)

    fig, ax = plt.subplots(figsize=(7, 7))
    proj = ax.imshow(map_vdisp, cmap='viridis', extent=[-args.galrad, args.galrad, -args.galrad, args.galrad], vmin=args.cmin, vmax=args.cmax)

    # -----------making the axis labels etc--------------
    ax.set_xticks(np.linspace(ax.get_xlim()[0], ax.get_xlim()[1], 5))
    ax.set_yticks(np.linspace(ax.get_ylim()[0], ax.get_ylim()[1], 5))
    ax.set_xlabel('Offset (kpc)', fontsize=args.fontsize)
    ax.set_ylabel('Offset (kpc)', fontsize=args.fontsize)
    ax.set_xticklabels(['%.1F' % item for item in ax.get_xticks()], fontsize=args.fontsize)
    ax.set_yticklabels(['%.1F' % item for item in ax.get_yticks()], fontsize=args.fontsize)

    # ---------making the colorbar axis once, that will correspond to all projections--------------
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    plt.colorbar(proj, cax=cax, orientation='vertical')

    cax.set_yticklabels(['%d' % index for index in cax.get_yticks()], fontsize=args.fontsize)
    cax.set_ylabel(r'LoS $\sigma_v$ (km/s)', fontsize=args.fontsize)

    fig = saveplot(fig, '3Dfrb' if args.get3d else args.projection + 'frb', args)
    return fig

# -----main code-----------------
if __name__ == '__main__':
    # ------------- arg parse -------------------------------
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, description='''kill me please''')

    parser.add_argument('--system', metavar='system', type=str, action='store', default='ayan_pleiades', help='Which system are you on? Default is ayan_pleiades')
    parser.add_argument('--pwd', dest='pwd', action='store_true', default=False, help='Just use the current working directory?, default is no')
    parser.add_argument('--foggie_dir', metavar='foggie_dir', type=str, action='store', default=None, help='Specify which directory the dataset lies in, otherwise, by default it will use the args.system variable to determine the FOGGIE data location')
    parser.add_argument('--halo', metavar='halo', type=str, action='store', default='8508', help='which halo?')
    parser.add_argument('--run', metavar='run', type=str, action='store', default='nref11c_nref9f', help='which run?')
    parser.add_argument('--output', metavar='output', type=str, action='store', default='RD0042', help='which output?')
    parser.add_argument('--projection', metavar='projection', type=str, action='store', default='x', help='Which projection do you want to plot, i.e., which axis is your line of sight? Default is x')
    parser.add_argument('--upto_kpc', metavar='upto_kpc', type=float, action='store', default=10, help='fit metallicity gradient out to what absolute kpc? default is None')
    parser.add_argument('--res_kpc', metavar='res_kpc', type=str, action='store', default=0.7, help='spatial sampling resolution, in kpc, at redshift 0; default is 0.3 kpc')
    parser.add_argument('--res_arc', metavar='res_arc', type=float, action='store', default=None, help='spatial sampling resolution, in arcseconds, to compute the Z statistics; default is None')
    parser.add_argument('--weightby', metavar='weightby', type=str, action='store', default=None, help='gas quantity to weight by; default is None')
    parser.add_argument('--get3d', dest='get3d', action='store_true', default=False, help='plot the 3D velocity dispersion?, default is no, only the LoS vel disp')
    parser.add_argument('--fontsize', metavar='fontsize', type=int, action='store', default=15, help='fontsize of plot labels, etc.; default is 15')
    parser.add_argument('--cmin', metavar='cmin', type=float, action='store', default=None, help='minimum value for plotting imshow colorbar; default is None')
    parser.add_argument('--cmax', metavar='cmax', type=float, action='store', default=None, help='maximum value for plotting imshow colorbar; default is None')
    parser.add_argument('--keep', dest='keep', action='store_true', default=False, help='keep previously displayed plots on screen?, default is no')
    parser.add_argument('--plot_frb', dest='plot_frb', action='store_true', default=False, help='plot the FRB projection too?, default is no')

    args = parser.parse_args()
    args.output_arr = [item for item in args.output.split(',')]
    args.foggie_dir, args.output_dir, args.run_loc, args.code_path, args.trackname, args.haloname, args.spectra_dir, args.infofile = get_run_loc_etc(args)
    if not args.keep: plt.close('all')

    # ------------- paths, dict, etc. set up -------------------------------
    if args.system == 'ayan_hd' or args.system == 'ayan_local': args.root_dir = '/Users/acharyya/models/simulation_output/'
    elif args.system == 'ayan_pleiades': args.root_dir = '/nobackup/aachary2/'
    args.halo_name = 'halo_' + args.halo
    args.output_path = args.root_dir + args.foggie_dir + '/' + args.halo_name + '/' + args.run + '/'

    for index, thisoutput in enumerate(args.output_arr):
        args.output = thisoutput
        print('Starting snapshot', args.output, 'i.e.,', index+1, 'out of', len(args.output_arr), 'snapshots..')
        args.snap_name = args.output_path + args.output + '/' + args.output
        box, box_center, box_width = get_box(args)
        fig1 = plot_vdisp_projection(box, box_center, box_width, args)
        if args.plot_frb: fig2 = plot_vdisp_frb(box, box_center, box_width, args)

    print('All snapshots completed in %s' % (datetime.timedelta(minutes=(time.time() - start_time) / 60)))
