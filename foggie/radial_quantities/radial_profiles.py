'''
Filename: radial_profiles.py
Author: Cassi
Date started: 4/28/22

This script makes datashader radial profile plots of temperature, density, pressure, entropy,
and radial velocity, color-coded by metallicity, temperature, or radial velocity.
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
import random
from scipy.interpolate import InterpolatedUnivariateSpline as IUS
import multiprocessing as multi
import shutil
import ast

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

    parser = argparse.ArgumentParser(description='Makes projection plots and data shader plots from a saved FRB.')

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

    parser.add_argument('--plot_y', metavar='plot_y', type=str, action='store', \
                        help='What do you want to plot on the y-axis? Options are:\n' + \
                        "density, temperature, metallicity, pressure, entropy, radial velocity\n" + \
                        "and you can make multiple at once by giving a list separated by commas (no spaces), like\n" + \
                        "'density,temperature,pressure'\n" + \
                        "Default is temperature.")
    parser.set_defaults(plot_y='temperature')

    parser.add_argument('--plot_color', metavar='plot_color', type=str, action='store', \
                        help='What field do you want to color-code the plot by?\n' + \
                        'Options are temperature, metallicity, or radial velocity. Default is metallicity.')
    parser.set_defaults(plot_color='metallicity')

    parser.add_argument('--output', metavar='output', type=str, action='store', \
                        help='Which output(s)? Options: Specify a single output (this is default' \
                        + ' and the default output is DD2427) or specify a range of outputs ' + \
                        'using commas to list individual outputs and dashes for ranges of outputs ' + \
                        '(e.g. "RD0020-RD0025" or "DD1341,DD1353,DD1600-DD1700", no spaces!)')
    parser.set_defaults(output='DD2427')

    parser.add_argument('--output_step', metavar='output_step', type=int, action='store', \
                        help='If you want to do every Nth output, this specifies N. Default: 1 (every output in specified range)')
    parser.set_defaults(output_step=1)

    parser.add_argument('--save_suffix', metavar='save_suffix', type=str, action='store', \
                        help='If you want to append a string to the end of the save file(s), what is it?\n' + \
                        'Default is nothing appended.')
    parser.set_defaults(save_suffix='')

    parser.add_argument('--cgm_only', dest='cgm_only', action='store_true', \
                        help='Specify this if you want to filter on density to remove the disk and satellites.')
    parser.set_defaults(cgm_only=False)

    parser.add_argument('--nproc', metavar='nproc', type=int, action='store', \
                        help='How many processes do you want? Default is 1 ' + \
                        '(no parallelization), if multiple outputs and multiple processors are' + \
                        ' specified, code will run one output per processor')
    parser.set_defaults(nproc=1)

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

    if (len(values)==0):
        return np.zeros(len(quantiles))
    else:
        return np.interp(quantiles, weighted_quantiles, values)

def make_profile_plot(snap):
    '''Makes and saves to file datashader radial profile plots of the quantities given in args.plot_y,
    colored by the quantity given in args.plot_color.'''

    Rvir = rvir_masses['radius'][rvir_masses['snapshot']==snap][0]

    if (args.system=='pleiades_cassi'):
        print('Copying directory to /tmp')
        snap_dir = '/nobackup/clochhaa/tmp/' + args.halo + '/' + args.run + '/profiles/' + snap
        # Make a dummy directory with the snap name so the script later knows the process running
        # this snapshot failed if the directory is still there
        os.makedirs(snap_dir)
    snap_name = foggie_dir + run_dir + snap + '/' + snap
    ds, refine_box = foggie_load(snap_name, trackname, do_filter_particles=False, halo_c_v_name=halo_c_v_name)
    zsnap = ds.get_parameter('CosmologyCurrentRedshift')

    # Define the density cut between disk and CGM to vary smoothly between 1 and 0.1 between z = 0.5 and z = 0.25,
    # with it being 1 at higher redshifts and 0.1 at lower redshifts
    current_time = ds.current_time.in_units('Myr').v
    if (current_time<=8656.88):
        density_cut_factor = 1.
    elif (current_time<=10787.12):
        density_cut_factor = 1. - 0.9*(current_time-8656.88)/2130.24
    else:
        density_cut_factor = 0.1

    sphere = ds.sphere(center=ds.halo_center_kpc, radius=(4.*Rvir, 'kpc'))
    if (args.cgm_only):
        sph_cgm = sphere.cut_region("obj['density'] < %.3e" % (density_cut_factor * cgm_density_max))
    else:
        sph_cgm = sphere

    if (',' in args.plot_y):
        plots = args.plot_y.split(',')
    else:
        plots = [args.plot_y]

    unit_dict = {'density':'g/cm**3',
                 'temperature':'K',
                 'metallicity':'Zsun',
                 'pressure':'erg/cm**3',
                 'entropy':'keV*cm**2',
                 'radial_velocity':'km/s'}
    if (args.cgm_only):
        y_range_dict = {'density':[-32,-26],
                        'temperature':[3,8],
                        'metallicity':[-3,2],
                        'pressure':[-19,-12],
                        'entropy':[-1,5],
                        'radial_velocity':[-500,1000]}
    else:
        y_range_dict = {'density':[-32,-23],
                        'temperature':[1,8],
                        'metallicity':[-3,2],
                        'pressure':[-19,-12],
                        'entropy':[-5,5],
                        'radial_velocity':[-500,1000]}
    label_dict = {'density':'log Density [g/cm$^3$]',
                  'temperature':'log Temperature [K]',
                  'metallicity':'log Metallicity [$Z_\odot$]',
                  'pressure':'log Pressure [erg/cm$^3$]',
                  'entropy':'log Entropy [keV cm$^2$]',
                  'radial_velocity':'Radial Velocity [km/s]'}
    radius_range = [0., 4.*Rvir]
    radius_bins = np.linspace(0.,4.*Rvir,200)

    for p in range(len(plots)):
        plot_y = plots[p]
        data_frame = pd.DataFrame({})
        data_frame['radius'] = sph_cgm['gas', 'radius_corrected'].in_units('kpc').v
        if (plot_y=='radial_velocity'):
            data_frame['y'] = sph_cgm['gas', 'radial_velocity_corrected'].in_units(unit_dict[plot_y]).v
        else:
            data_frame['y'] = np.log10(sph_cgm['gas', plot_y].in_units(unit_dict[plot_y]).v)
        if (color_log):
            data_frame['coloring'] = np.log10(sph_cgm['gas', args.plot_color].in_units(unit_dict[args.plot_color]).v)
        else:
            data_frame['coloring'] = sph_cgm['gas', args.plot_color].in_units(unit_dict[args.plot_color]).v
        data_frame['mass'] = sph_cgm['gas','cell_mass'].in_units('Msun').v
        data_frame['color'] = color_func(data_frame['coloring'])
        data_frame.color = data_frame.color.astype('category')
        cvs = dshader.Canvas(plot_width=1200, plot_height=800, x_range=radius_range, y_range=y_range_dict[plot_y])
        agg = cvs.points(data_frame, 'radius', 'y', dshader.count_cat('color'))
        img = tf.spread(tf.shade(agg, color_key=color_key, how='eq_hist',min_alpha=100), shape='square', px=0)
        export_image(img, save_dir + snap + '_' + plot_y + '_vs_radius_' + args.plot_color + '-colored' + save_suffix)
        fig = plt.figure(figsize=(10,8))
        ax = fig.add_subplot(1,1,1)
        image = plt.imread(save_dir + snap + '_' + plot_y + '_vs_radius_' + args.plot_color + '-colored' + save_suffix + '.png')
        ax.imshow(image, extent=[radius_range[0],radius_range[1],y_range_dict[plot_y][0],y_range_dict[plot_y][1]])
        ax.set_aspect(8*abs(radius_range[1]-radius_range[0])/(12*abs(y_range_dict[plot_y][1]-y_range_dict[plot_y][0])))
        ax.set_xlabel('Radius [kpc]', fontsize=20)
        ax.set_ylabel(label_dict[plot_y], fontsize=20)
        ax.tick_params(axis='both', which='both', direction='in', length=8, width=2, pad=5, labelsize=18, \
          top=True, right=True)
        bin_indices = np.digitize(data_frame['radius'].to_numpy(), radius_bins)
        median_profile = []
        for i in range(1,len(radius_bins)):
            if (plot_y=='radial_velocity'):
                values = data_frame['y'][np.where(bin_indices==i)[0]].to_numpy()
            else:
                values = 10**data_frame['y'][np.where(bin_indices==i)[0]].to_numpy()
            weights = data_frame['mass'][np.where(bin_indices==i)[0]].to_numpy()
            median_profile.append(weighted_quantile(values, weights, [0.5])[0])
        radius_bin_centers = 0.5*np.diff(radius_bins)+radius_bins[1:]
        if (plot_y=='radial_velocity'):
            ax.plot(radius_bin_centers, np.array(median_profile), 'k-', lw=2)
        else:
            ax.plot(radius_bin_centers, np.log10(np.array(median_profile)), 'k-', lw=2)
        ax.plot([Rvir,Rvir],[y_range_dict[plot_y][0], y_range_dict[plot_y][1]], 'k--', lw=1)
        ax.text(Rvir+5, abs(y_range_dict[plot_y][1]-y_range_dict[plot_y][0])*0.05+y_range_dict[plot_y][0], '$R_{200}$', fontsize=20, ha='left', va='center')
        ax.axis([radius_range[0], radius_range[1], y_range_dict[plot_y][0], y_range_dict[plot_y][1]])
        ax2 = fig.add_axes([0.7, 0.93, 0.25, 0.06])
        cmap = create_foggie_cmap(cmin, cmax, color_func, color_key, color_log)
        ax2.imshow(np.flip(cmap.to_pil(), 1))
        ax2.set_xticks(color_ticks)
        ax2.set_xticklabels(color_ticklabels, fontsize=18)
        ax2.text(400, 150, field_label, fontsize=20, ha='center', va='center')
        ax2.spines["top"].set_color('white')
        ax2.spines["bottom"].set_color('white')
        ax2.spines["left"].set_color('white')
        ax2.spines["right"].set_color('white')
        ax2.set_ylim(60, 180)
        ax2.set_xlim(-10, 750)
        ax2.set_yticklabels([])
        ax2.set_yticks([])
        plt.subplots_adjust(left=0.12, bottom=0.03, top=0.95, right=0.98)
        plt.savefig(save_dir + snap + '_' + plot_y + '_vs_radius_' + args.plot_color + '-colored' + save_suffix + '.png')
        plt.close()
        print('Plot of %s vs. radius, colored by %s, made for snapshot %s.' % (plot_y, args.plot_color, snap))
        f = open(save_dir + snap + '_' + plot_y + '_vs_radius_mass-weighted-median-profile' + save_suffix + '.txt', 'w')
        f.write('# radius (kpc)  %s (%s)\n' % (plot_y, unit_dict[plot_y]))
        for i in range(len(median_profile)):
            f.write('%.2f            %.3e\n' % (radius_bin_centers[i], median_profile[i]))
        f.close()

    # Delete output from temp directory if on pleiades
    if (args.system=='pleiades_cassi'):
        print('Deleting directory from /tmp')
        shutil.rmtree(snap_dir)

if __name__ == "__main__":
    args = parse_args()
    print(args.halo)
    print(args.run)
    print(args.system)
    foggie_dir, output_dir, run_dir, code_path, trackname, haloname, spectra_dir, infofile = get_run_loc_etc(args)

    # Set directory for output location, making it if necessary
    save_dir = output_dir + 'profiles_halo_00' + args.halo + '/' + args.run + '/'
    if not (os.path.exists(save_dir)): os.system('mkdir -p ' + save_dir)

    print('foggie_dir: ', foggie_dir)
    halo_c_v_name = code_path + 'halo_infos/00' + args.halo + '/' + args.run + '/halo_c_v'
    masses_dir = code_path + 'halo_infos/00' + args.halo + '/' + args.run + '/'
    masses = Table.read(masses_dir + 'masses_z-less-2.hdf5', path='all_data')
    rvir_masses = Table.read(masses_dir + 'rvir_masses.hdf5', path='all_data')

    outs = make_output_list(args.output, output_step=args.output_step)

    if (args.save_suffix!=''):
        save_suffix = '_' + args.save_suffix
    else:
        save_suffix = ''

    if (args.plot_color=='temperature'):
        color_func = categorize_by_temp
        color_key = new_phase_color_key
        cmin = temperature_min_datashader
        cmax = temperature_max_datashader
        color_ticks = [50,300,550]
        color_ticklabels = ['4','5','6']
        field_label = 'log T [K]'
        color_log = True
    elif (args.plot_color=='density'):
        color_func = categorize_by_den
        color_key = density_color_key
        cmin = dens_phase_min
        cmax = dens_phase_max
        step = 750./np.size(list(color_key))
        color_ticks = [step,step*3.,step*5.,step*7.,step*9.]
        color_ticklabels = ['-30','-28','-26','-24','-22']
        field_label = 'log $\\rho$ [g/cm$^3$]'
        color_log = True
    elif (args.plot_color=='radial_velocity'):
        color_func = categorize_by_outflow_inflow
        color_key = outflow_inflow_color_key
        cmin = -200.
        cmax = 200.
        step = 750./np.size(list(color_key))
        color_ticks = [step,step*3.,step*5.,step*7.,step*9.]
        color_ticklabels = ['-200','-100','0','100','200']
        field_label = 'Radial velocity [km/s]'
        color_log = False
    elif (args.plot_color=='metallicity'):
        color_func = categorize_by_metals
        color_key = new_metals_color_key
        cmin = metal_min
        cmax = metal_max
        rng = (np.log10(metal_max)-np.log10(metal_min))/750.
        start = np.log10(metal_min)
        color_ticks = [(np.log10(0.01)-start)/rng,(np.log10(0.1)-start)/rng,(np.log10(0.5)-start)/rng,(np.log10(1.)-start)/rng,(np.log10(2.)-start)/rng]
        color_ticklabels = ['0.01','0.1','0.5','1','2']
        field_label = 'Metallicity [$Z_\odot$]'
        color_log = False

    # Loop over outputs, for either single-processor or parallel processor computing
    if (args.nproc==1):
        for i in range(len(outs)):
            snap = outs[i]
            make_profile_plot(snap)
    else:
        skipped_outs = outs
        while (len(skipped_outs)>0):
            skipped_outs = []
            # Split into a number of groupings equal to the number of processors
            # and run one process per processor
            for i in range(len(outs)//args.nproc):
                threads = []
                snaps = []
                for j in range(args.nproc):
                    snap = outs[args.nproc*i+j]
                    snaps.append(snap)
                    threads.append(multi.Process(target=make_profile_plot, args=[snap]))
                for t in threads:
                    t.start()
                for t in threads:
                    t.join()
                # Delete leftover outputs from failed processes from tmp directory if on pleiades
                if (args.system=='pleiades_cassi'):
                    for s in range(len(snaps)):
                        if (os.path.exists('/nobackup/clochhaa/tmp/' + args.halo + '/' + args.run + '/profiles/' + snaps[s])):
                            print('Deleting failed %s from /tmp' % (snaps[s]))
                            skipped_outs.append(snaps[s])
                            shutil.rmtree('/nobackup/clochhaa/tmp/' + args.halo + '/' + args.run + '/profiles/' + snaps[s])
            # For any leftover snapshots, run one per processor
            threads = []
            snaps = []
            for j in range(len(outs)%args.nproc):
                snap = outs[-(j+1)]
                snaps.append(snap)
                threads.append(multi.Process(target=make_profile_plot, args=[snap]))
            for t in threads:
                t.start()
            for t in threads:
                t.join()
            # Delete leftover outputs from failed processes from tmp directory if on pleiades
            if (args.system=='pleiades_cassi'):
                for s in range(len(snaps)):
                    if (os.path.exists('/nobackup/clochhaa/tmp/' + args.halo + '/' + args.run + '/profiles/' + snaps[s])):
                        print('Deleting failed %s from /tmp' % (snaps[s]))
                        skipped_outs.append(snaps[s])
                        shutil.rmtree('/nobackup/clochhaa/tmp/' + args.halo + '/' + args.run + '/profiles/' + snaps[s])
            outs = skipped_outs