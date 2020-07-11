'''
Filename: find_shape_for_region.py
This script uses FRBs or yt datasets to pick an elliptical cone that best captures a region,
using photutils.segmentation.
Author: Cassi
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
from photutils import detect_threshold, detect_sources, source_properties, EllipticalAperture
from scipy.interpolate import InterpolatedUnivariateSpline as IUS
from scipy.optimize import minimize
import trident
import ast
import emcee
import numpy.random as rn
from multiprocessing import Pool
import matplotlib.patches as patches

# These imports are FOGGIE-specific files
from foggie.utils.consistency import *
from foggie.utils.get_refine_box import get_refine_box
from foggie.utils.get_halo_center import get_halo_center
from foggie.utils.get_proper_box_size import get_proper_box_size
from foggie.utils.get_run_loc_etc import get_run_loc_etc
from foggie.utils.yt_fields import *
from foggie.utils.foggie_load import *

# These imports for datashader plots
import datashader as dshader
from datashader.utils import export_image
import datashader.transfer_functions as tf
import pandas as pd
import matplotlib as mpl

def parse_args():
    '''Parse command line arguments. Returns args object.'''

    parser = argparse.ArgumentParser(description='Finds ellipses that encapsulates filament or wind regions from a dataset or an FRB.')

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

    parser.add_argument('--output', metavar='output', type=str, action='store', \
                        help='If you are plotting from a simulation output, which output?\n' + \
                        'Options: Specify a single output or specify a range of outputs\n' + \
                        'using commas to list individual outputs and dashes for ranges of outputs\n' + \
                        '(e.g. "RD0020-RD0025" or "DD1341,DD1353,DD1600-DD1700", no spaces!)\n' + \
                        'If you specify multiple outputs, the plotted histogram will stack all data from each.')
    parser.set_defaults(output='none')

    parser.add_argument('--output_step', metavar='output_step', type=int, action='store', \
                        help='If you want to do every Nth output, this specifies N. Default: 1 (every output in specified range)')
    parser.set_defaults(output_step=1)

    parser.add_argument('--pwd', dest='pwd', action='store_true',
                        help='Just use the working directory?, Default is no')
    parser.set_defaults(pwd=False)

    parser.add_argument('--region', metavar='region', type=str, action='store',\
                        help='What region do you want to find ellipses to identify? Options are\n' + \
                        "'filament' or 'wind' or 'both', where 'filament' finds metal-pristine (inflowing) filaments\n" + \
                        "and 'wind' finds metal-enriched (outflowing) galactic winds. If 'both' is\n" + \
                        "specified, it will find and save elliptical regions for both filaments and winds separately.\n" + \
                        "Default is 'filament'.")
    parser.set_defaults(region='filament')

    parser.add_argument('--region_weight', metavar='region_weight', type=str, action='store', \
                        help='What field do you want to weight the region by? Options are cell_mass\n' + \
                        "or cell_volume. Default is cell_volume.")
    parser.set_defaults(region_weight='cell_volume')

    parser.add_argument('--FRB_name', metavar='FRB_name', type=str, action='store', \
                        help='If using an FRB, what is the file name of the FRB?')
    parser.set_defaults(FRB_name='none')

    parser.add_argument('--save_suffix', metavar='save_suffix', type=str, action='store', \
                        help='If you want to append a string to the end of the save file(s), what is it?\n' + \
                        'Default is nothing appended.')
    parser.set_defaults(save_suffix='')

    args = parser.parse_args()
    return args

def filter_ds(box):
    '''This function filters the yt data object passed in as 'box' into inflow and outflow regions,
    based on metallicity, and returns the box filtered into these regions.'''

    bool_inflow = box['metallicity'] < 0.01
    bool_outflow = box['metallicity'] > 1.
    bool_neither = (~bool_inflow) & (~bool_outflow)
    box_inflow = box.include_below('metallicity', 0.01, 'Zsun')
    box_outflow = box.include_above('metallicity', 1., 'Zsun')
    box_neither = box.include_above('metallicity', 0.01, 'Zsun')
    box_neither = box_neither.include_below('metallicity', 1., 'Zsun')

    return box_inflow, box_outflow, box_neither

def filter_FRB(FRB):
    '''This function filters the FRB passed in into inflow, outflow, and neither regions, based on metallicity,
    and returns the inflow FRB, outflow FRB, and neither FRB. The field 'metallicity'
    must exist within the FRB.'''

    bool_inflow = FRB['metallicity'] < 0.01
    bool_outflow = FRB['metallicity'] > 1.
    bool_neither = (~bool_inflow) & (~bool_outflow)

    FRB_inflow = Table()
    FRB_outflow = Table()
    FRB_neither = Table()
    for j in range(len(FRB.columns)):
        FRB_inflow.add_column(FRB.columns[j][bool_inflow], name=FRB.columns[j].name)
        FRB_inflow[FRB.columns[j].name].unit = FRB[FRB.columns[j].name].unit
        FRB_outflow.add_column(FRB.columns[j][bool_outflow], name=FRB.columns[j].name)
        FRB_outflow[FRB.columns[j].name].unit = FRB[FRB.columns[j].name].unit
        FRB_neither.add_column(FRB.columns[j][bool_neither], name=FRB.columns[j].name)
        FRB_neither[FRB.columns[j].name].unit = FRB[FRB.columns[j].name].unit

    return FRB_inflow, FRB_outflow, FRB_neither

def ellipse(center_x, center_y, a, b, rot_angle, x, y):
    '''This function returns True if the point (x, y) is within the ellipse defined by
    (center_x,center_y), the horizontal axis a, the vertical axis b, and the rotation from the horizontal axis rot_angle,
    and returns False otherwise.'''

    A = a**2. * np.sin(rot_angle)**2. + b**2. * np.cos(rot_angle)**2.
    B = 2. * (b**2. - a**2.) * np.sin(rot_angle) * np.cos(rot_angle)
    C = a**2. * np.cos(rot_angle)**2. + b**2. * np.sin(rot_angle)**2.
    D = -2.*A*center_x - B*center_y
    E = -B*center_x - 2.*C*center_y
    F = A*center_x**2. + B*center_x*center_y + C*center_y**2. - a**2.*b**2.

    return A*x**2. + B*x*y + C*y**2. + D*x + E*y + F < 0.

def ellipses_from_segmentation(x_range, y_range, weight_region, threshold, pix_size):
    '''This function uses the photutils segmentation package to identify ellipses that capture
    the different regions.'''

    #threshold = detect_threshold(weight_region, nsigma=1.)
    segm = detect_sources(weight_region, threshold, npixels=1000)
    prop = source_properties(weight_region, segm)
    r = 2.          # Ratio to upscale the ellipse
    best_ellipses = []
    aperatures = []
    for i in range(len(prop)):
        center_x = prop[i].xcentroid.value
        center_y = prop[i].ycentroid.value
        a = prop[i].semimajor_axis_sigma.value * r
        b = prop[i].semiminor_axis_sigma.value * r
        rot_ang = prop[i].orientation.value / 180. * np.pi
        aperture = EllipticalAperture((center_x, center_y), a, b, theta=rot_ang)
        center_x = center_x * pix_size + x_range[0]
        center_y = center_y * pix_size + y_range[0]
        a = a * pix_size
        b = b * pix_size
        best_ellipses.append([center_x, center_y, a, b, rot_ang])
    return best_ellipses

def find_regions(theta_region, phi_region, weight_region, save_dir, FRB_name, save_suffix, weight_label):
    '''This function takes in the theta and phi positions, as well as the field to weight by, for
    both the full dataset '_all' and only the region of interest '_region' and returns the parameters
    of conical ellipses that capture the most of the weight field of the region.'''

    # Find initial guesses for the number of regions and locations in theta,phi space by producing
    # a histogram of weight field in theta,phi, identifying contours in this space, and identifying
    # circles in theta,phi that identify the contours without overlapping
    x_range = [0., np.pi]
    y_range = [-np.pi, np.pi]
    pix_size = np.pi/500.
    hist2d, xbins, ybins = np.histogram2d(theta_region, phi_region, weights=weight_region, bins=(500, 1000), range=[x_range,y_range])
    hist2d = np.transpose(hist2d)
    #hist2d[hist2d==0.] = np.nan
    threshold = 0.1*np.nanmax(hist2d)
    hist2d[np.isnan(hist2d)] = 0.
    xbins = xbins[:-1]
    ybins = ybins[:-1]
    best_ellipses = ellipses_from_segmentation(x_range, y_range, hist2d, threshold, pix_size)
    # Combine any overlapping ellipses
    hist_ellipses_only = np.zeros(np.shape(hist2d))
    xdata_region = np.tile(xbins, (1000, 1))
    ydata_region = np.transpose(np.tile(ybins, (500, 1)))
    for i in range(len(best_ellipses)):
        in_ellipse = ellipse(best_ellipses[i][0], best_ellipses[i][1], best_ellipses[i][2], \
          best_ellipses[i][3], best_ellipses[i][4], xdata_region, ydata_region)
        hist_ellipses_only[in_ellipse] = 1.
    best_combined_ellipses = ellipses_from_segmentation(x_range, y_range, hist_ellipses_only, 0.5, pix_size)
    fig = plt.figure(figsize=(8,8),dpi=500)
    ax = fig.add_subplot(1,1,1)
    cmin = np.min(np.array(weight_region)[np.nonzero(weight_region)[0]])
    x_range = [0., np.pi]
    y_range = [-np.pi, np.pi]
    hist = ax.hist2d(theta_region, phi_region, weights=weight_region, bins=(500, 1000), cmin=cmin, range=[x_range,y_range])
    hist2d = hist[0]
    xbins = hist[1][:-1]
    ybins = hist[2][:-1]
    c = ax.contour(xbins, ybins, np.transpose(hist2d), [threshold], \
      colors='w')
    for i in range(len(best_combined_ellipses)):
        ell = patches.Ellipse((best_combined_ellipses[i][0], best_combined_ellipses[i][1]), \
          2.*best_combined_ellipses[i][2], 2.*best_combined_ellipses[i][3], best_combined_ellipses[i][4]/np.pi*180., \
          color='m', lw=2, fill=False, zorder=10)
        ax.add_artist(ell)
        ax.plot([best_combined_ellipses[i][0]], [best_combined_ellipses[i][1]], marker='x', color='m')
    cbaxes = fig.add_axes([0.7, 0.92, 0.25, 0.03])
    cbar = plt.colorbar(hist[3], cax=cbaxes, orientation='horizontal', ticks=[])
    cbar.set_label(weight_label, fontsize=14)
    #ax.set_aspect('equal')
    ax.set_xlabel('$\\theta$ [rad]', fontsize=14)
    ax.set_ylabel('$\\phi$ [rad]', fontsize=14)
    ax.axis([x_range[0], x_range[1], y_range[0], y_range[1]])
    ax.tick_params(axis='both', which='both', direction='in', length=8, width=2, pad=5, labelsize=14, \
      top=True, right=True)
    plt.subplots_adjust(left=0.12, bottom=0.08, right=0.95)
    plt.savefig(save_dir + FRB_name + '_phi_vs_theta_hist_best_ellipses' + save_suffix + '.png')
    plt.close()
    f = open(save_dir + FRB_name + save_suffix + '.txt', 'w')
    f.write('# center_theta    center_phi    theta_axis    phi_axis    rotation\n')
    for i in range(len(best_combined_ellipses)):
        if (best_combined_ellipses[i][1]<0.): phic_extra = ''
        else: phic_extra = ' '
        f.write('  %.6f        %.6f     ' % (best_combined_ellipses[i][0], best_combined_ellipses[i][1]) + \
          phic_extra + '%.6f      %.6f    %.6f\n' % (best_combined_ellipses[i][2], \
          best_combined_ellipses[i][3], best_combined_ellipses[i][4]))
    f.close()


if __name__ == "__main__":
    args = parse_args()
    print(args.halo)
    print(args.run)
    print(args.system)
    foggie_dir, output_dir, run_dir, code_path, trackname, haloname, spectra_dir, infofile = get_run_loc_etc(args)

    halo_c_v_name = code_path + 'halo_infos/00' + args.halo + '/' + args.run + '/halo_c_v'

    # Set directory for output location, making it if necessary
    save_dir = output_dir + 'ellipse_regions_halo_00' + args.halo + '/' + args.run + '/'
    if not (os.path.exists(save_dir)): os.system('mkdir -p ' + save_dir)

    if (args.FRB_name=='none') and (args.output=='none'):
        sys.exit('You must specify either simulation output(s) or an FRB!')

    if (args.save_suffix != ''): save_suffix = '_' + args.save_suffix
    else: save_suffix = ''

    if (args.region_weight=='cell_mass'): weight_label = 'Mass'
    if (args.region_weight=='cell_volume'): weight_label = 'Volume'

    if (args.FRB_name!='none'):
        FRB = Table.read(output_dir + 'FRBs_halo_00' + args.halo + '/' + args.run + '/' + \
          args.FRB_name + '.hdf5', path='all_data')
        if ('theta_pos' not in FRB.columns):
            sys.exit("The FRB doesn't contain the field theta_pos!")
        if ('phi_pos' not in FRB.columns):
            sys.exit("The FRB doesn't contain the field phi_pos!")
        if (args.region_weight not in FRB.columns):
            sys.exit("The FRB doesn't contain the region weight field %s!" % (args.region_weight))
        print('FRB loaded, filtering...')
        FRB_inflow, FRB_outflow, FRB_neither = filter_FRB(FRB)
        print('FRB filtered. Finding best ellipses')
        if (args.region=='filament') or (args.region=='both'):
            theta_inflow = FRB_inflow['theta_pos']
            phi_inflow = FRB_inflow['phi_pos']
            weight_data_inflow = FRB_inflow[args.region_weight]
            if (args.region_weight=='cell_mass'): weight_label = 'Mass'
            if (args.region_weight=='cell_volume'): weight_label = 'Volume'
            theta_inflow[np.isnan(theta_inflow)] = 0.
            phi_inflow[np.isnan(phi_inflow)] = 0.
            weight_data_inflow[np.isnan(weight_data_inflow)] = 0.
            region_params_inflow = find_regions(theta_inflow, phi_inflow, weight_data_inflow, save_dir, args.FRB_name, save_suffix + '_filament', weight_label)
        if (args.region=='wind') or (args.region=='both'):
            theta_outflow = FRB_outflow['theta_pos']
            phi_outflow = FRB_outflow['phi_pos']
            weight_data_outflow = FRB_outflow[args.region_weight]
            theta_outflow[np.isnan(theta_outflow)] = 0.
            phi_outflow[np.isnan(phi_outflow)] = 0.
            weight_data_outflow[np.isnan(weight_data_outflow)] = 0.
            region_params_outflow = find_regions(theta_outflow, phi_outflow, weight_data_outflow, save_dir, args.FRB_name, save_suffix + '_wind', weight_label)
        print('Ellipse files saved to', save_dir)

    elif (args.output!='none'):
        if (',' in args.output):
            outs = args.output.split(',')
            for i in range(len(outs)):
                if ('-' in outs[i]):
                    ind = outs[i].find('-')
                    first = outs[i][2:ind]
                    last = outs[i][ind+3:]
                    output_type = outs[i][:2]
                    outs_sub = []
                    for j in range(int(first), int(last)+1, args.output_step):
                        if (j < 10):
                            pad = '000'
                        elif (j >= 10) and (j < 100):
                            pad = '00'
                        elif (j >= 100) and (j < 1000):
                            pad = '0'
                        elif (j >= 1000):
                            pad = ''
                        outs_sub.append(output_type + pad + str(j))
                    outs[i] = outs_sub
            flat_outs = []
            for i in outs:
                if (type(i)==list):
                    for j in i:
                        flat_outs.append(j)
                else:
                    flat_outs.append(i)
            outs = flat_outs
            outs_save = outs[0] + '-' + outs[-1]
        elif ('-' in args.output):
            ind = args.output.find('-')
            first = args.output[2:ind]
            last = args.output[ind+3:]
            output_type = args.output[:2]
            outs = []
            for i in range(int(first), int(last)+1, args.output_step):
                if (i < 10):
                    pad = '000'
                elif (i >= 10) and (i < 100):
                    pad = '00'
                elif (i >= 100) and (i < 1000):
                    pad = '0'
                elif (i >= 1000):
                    pad = ''
                outs.append(output_type + pad + str(i))
            outs_save = outs[0] + '-' + outs[-1]
        else:
            outs = [args.output]
            outs_save = args.output

        # Loop through outputs and stack necessary fields for combined histogram
        stacked_theta_inflow = []
        stacked_phi_inflow = []
        stacked_hist_inflow = []
        stacked_theta_outflow = []
        stacked_phi_outflow = []
        stacked_hist_outflow = []
        for i in range(len(outs)):
            snap = outs[i]
            snap_name = foggie_dir + run_dir + snap + '/' + snap
            if (args.system=='pleiades_cassi'):
                print('Copying directory to /tmp')
                snap_dir = '/tmp/' + snap
                shutil.copytree(foggie_dir + run_dir + snap, snap_dir)
                snap_name = snap_dir + '/' + snap
            ds, refine_box = foggie_load(snap_name, trackname, do_filter_particles=False, halo_c_v_name=halo_c_v_name)
            print('Filtering dataset')
            box_inflow, box_outflow, box_neither = filter_ds(refine_box)
            if (args.region=='filament') or (args.region=='both'):
                theta_inflow = box_inflow['theta_pos'].flatten().v
                phi_inflow = box_inflow['phi_pos'].flatten().v
                hist_inflow = box_inflow[args.region_weight].flatten().v
                stacked_theta_inflow += list(theta_inflow)
                stacked_phi_inflow += list(phi_inflow)
                stacked_hist_inflow += list(hist_inflow)
            if (args.region=='wind') or (args.region=='both'):
                theta_outflow = box_outflow['theta_pos'].flatten().v
                phi_outflow = box_outflow['phi_pos'].flatten().v
                hist_outflow = box_outflow[args.region_weight].flatten().v
                stacked_theta_outflow += list(theta_outflow)
                stacked_phi_outflow += list(phi_outflow)
                stacked_hist_outflow += list(hist_outflow)

            if (args.system=='pleiades_cassi'):
                print('Deleting directory from /tmp')
                shutil.rmtree(snap_dir)
        print('Dataset(s) stacked and filtered. Finding best ellipses')
        if (args.region=='filament') or (args.region=='both'):
            region_params_inflow = find_regions(stacked_theta_inflow, stacked_phi_inflow, stacked_hist_inflow, \
              save_dir, outs_save, save_suffix + '_filament', weight_label)
        if (args.region=='wind') or (args.region=='both'):
            region_params_outflow = find_regions(stacked_theta_outflow, stacked_phi_outflow, stacked_hist_outflow, \
              save_dir, outs_save, save_suffix + '_wind', weight_label)
        print('Ellipses saved to', save_dir)
