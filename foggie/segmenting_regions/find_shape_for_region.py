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
from foggie.utils.analysis_utils import *

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

    parser.add_argument('--filter', metavar='filter', type=str, action='store',\
                        help='How do you want to define regions? Options are "metallicity" or "radial_velocity" and\n' + \
                            'default is "radial_velocity".')
    parser.set_defaults(filter='radial_velocity')

    parser.add_argument('--region_weight', metavar='region_weight', type=str, action='store', \
                        help='What field do you want to weight the region by? Options are cell_mass\n' + \
                        "or cell_volume. Default is cell_volume.")
    parser.set_defaults(region_weight='cell_volume')

    parser.add_argument('--save_suffix', metavar='save_suffix', type=str, action='store', \
                        help='If you want to append a string to the end of the save file(s), what is it?\n' + \
                        'Default is nothing appended.')
    parser.set_defaults(save_suffix='')

    parser.add_argument('--radbins', metavar='radbins', type=str, action='store', \
                        help='If you want to compute ellipses in radius bins rather than the full box,\n' + \
                        'enter in the list of radius bins like:\n' + \
                        '"[10,20,30,40,50,60,70,80,90,100,110,120,130,140,150,160]" (don\'t forget the outer quotes!)\n' + \
                        'If you want this list above, just give "default" instead of specifying your own list.')
    parser.set_defaults(radbins='none')

    args = parser.parse_args()
    return args

def filter_ds(box):
    '''This function filters the yt data object passed in as 'box' into inflow and outflow regions,
    based on metallicity, and returns the box filtered into these regions.'''

    if (args.filter=='metallicity'):
        box_inflow = box.include_below(('gas','metallicity'), 0.01, 'Zsun')
        box_outflow = box.include_above(('gas','metallicity'), 1., 'Zsun')
        box_neither = box.include_above(('gas','metallicity'), 0.01, 'Zsun')
        box_neither = box_neither.include_below(('gas','metallicity'), 1., 'Zsun')
    elif (args.filter=='radial_velocity'):
        box_inflow = box.include_below(('gas','radial_velocity_corrected'), -100., 'km/s')
        box_outflow = box.include_above(('gas','radial_velocity_corrected'), 200., 'km/s')
        box_neither = box.include_above(('gas','radial_velocity_corrected'), -100., 'km/s')
        box_neither = box_neither.include_below(('gas','radial_velocity_corrected'), 200., 'km/s')

    return box_inflow, box_outflow, box_neither

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

    segm = detect_sources(weight_region, threshold, npixels=100)
    if (segm==None):
        best_ellipses = [[0,0,0,0,0]]
    else:
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

def find_regions(theta_region, phi_region, radius_region, weight_region, radbins, save_dir, FRB_name, save_suffix, weight_label):
    '''This function takes in the theta, phi, and radius positions, as well as the field to weight by, for
    the region of interest and returns the parameters of conical ellipses that capture the most of
    the weight field of the region within each radial bin given by 'radbins'.'''

    x_range = [0., np.pi]
    y_range = [-np.pi, np.pi]
    pix_size = np.pi/200.
    if (radbins=='none'):
        radbins = [np.min(radius_region), np.max(radius_region)]
        radbins_filename = ''
    else:
        radbins_filename = 'radbins'
    ellipses = [[]]*(len(radbins)-1)
    for i in range(len(radbins)-1):
        inner_r = radbins[i]
        outer_r = radbins[i+1]
        in_bin = (radius_region > inner_r) & (radius_region < outer_r)
        theta_bin = theta_region[in_bin]
        phi_bin = phi_region[in_bin]
        weight_bin = weight_region[in_bin]

        hist2d, xbins, ybins = np.histogram2d(theta_bin, phi_bin, weights=weight_bin, bins=(200, 400), range=[x_range,y_range])
        hist2d = np.transpose(hist2d)
        if (len(hist2d[hist2d!=0])==0):
            best_combined_ellipses = [[0,0,0,0,0]]
            ellipses[i] = best_combined_ellipses
        else:
            threshold = 0.1*(np.nanmedian(hist2d[hist2d!=0]) - np.nanmin(hist2d[hist2d!=0])) + np.nanmin(hist2d[hist2d!=0])
            #print(np.nanmax(hist2d), threshold, np.nanmean(hist2d[hist2d!=0]), np.nanmedian(hist2d[hist2d!=0]), np.nanstd(hist2d[hist2d!=0]), np.nanmin(hist2d[hist2d!=0]))
            hist2d[np.isnan(hist2d)] = 0.
            xbins = xbins[:-1]
            ybins = ybins[:-1]
            best_ellipses = ellipses_from_segmentation(x_range, y_range, hist2d, threshold, pix_size)
            # Combine any overlapping ellipses
            hist_ellipses_only = np.zeros(np.shape(hist2d))
            xdata_region = np.tile(xbins, (400, 1))
            ydata_region = np.transpose(np.tile(ybins, (200, 1)))
            for j in range(len(best_ellipses)):
                in_ellipse = ellipse(best_ellipses[j][0], best_ellipses[j][1], best_ellipses[j][2], \
                  best_ellipses[j][3], best_ellipses[j][4], xdata_region, ydata_region)
                hist_ellipses_only[in_ellipse] = 1.
            best_combined_ellipses = ellipses_from_segmentation(x_range, y_range, hist_ellipses_only, 0.5, pix_size)
            ellipses[i] = best_combined_ellipses
            fig = plt.figure(figsize=(8,8),dpi=500)
            ax = fig.add_subplot(1,1,1)
            cmin = np.min(np.array(weight_region)[np.nonzero(weight_region)[0]])
            x_range = [0., np.pi]
            y_range = [-np.pi, np.pi]
            hist = ax.hist2d(theta_bin, phi_bin, weights=weight_bin, bins=(200, 400), cmin=cmin, range=[x_range,y_range])
            hist2d = hist[0]
            xbins = hist[1][:-1]
            ybins = hist[2][:-1]
            c = ax.contour(xbins, ybins, np.transpose(hist2d), [threshold], \
              colors='w')
            for j in range(len(best_combined_ellipses)):
                ell = patches.Ellipse((best_combined_ellipses[j][0], best_combined_ellipses[j][1]), \
                  2.*best_combined_ellipses[j][2], 2.*best_combined_ellipses[j][3], best_combined_ellipses[j][4]/np.pi*180., \
                  color='m', lw=2, fill=False, zorder=10)
                ax.add_artist(ell)
                ax.plot([best_combined_ellipses[j][0]], [best_combined_ellipses[j][1]], marker='x', color='m')
            cbaxes = fig.add_axes([0.7, 0.92, 0.25, 0.03])
            cbar = plt.colorbar(hist[3], cax=cbaxes, orientation='horizontal', ticks=[])
            cbar.set_label(weight_label, fontsize=14)
            ax.set_xlabel('$\\theta$ [rad]', fontsize=14)
            ax.set_ylabel('$\\phi$ [rad]', fontsize=14)
            ax.axis([x_range[0], x_range[1], y_range[0], y_range[1]])
            ax.tick_params(axis='both', which='both', direction='in', length=8, width=2, pad=5, labelsize=14, \
              top=True, right=True)
            plt.subplots_adjust(left=0.12, bottom=0.08, right=0.95)
            if (radbins_filename==''):
                plt.savefig(save_dir + FRB_name + '_phi_vs_theta_hist_best_ellipses' + save_suffix + '.png')
            else:
                plt.savefig(save_dir + FRB_name + '_phi_vs_theta_hist_r' + str(inner_r) + '-' + str(outer_r) + '_best_ellipses' + save_suffix + '.png')
            plt.close()
    f = open(save_dir + FRB_name + save_suffix + '.txt', 'w')
    f.write('# inner_r      outer_r     center_theta    center_phi    theta_axis    phi_axis    rotation\n')
    for i in range(len(ellipses)):
        for j in range(len(ellipses[i])):
            f.write('  %.2f        %.2f       %.6f        %.6f     ' % (radbins[i], radbins[i+1], \
              ellipses[i][j][0], ellipses[i][j][1]) + \
              '%.6f      %.6f    %.6f\n' % (ellipses[i][j][2], \
              ellipses[i][j][3], ellipses[i][j][4]))
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

    if (args.radbins!='none'):
        if (args.radbins=='default'):
            radbins = [10,20,30,40,50,60,70,80,90,100,110,120,130,140,150,160]
        else:
            try:
                radbins = ast.literal_eval(args.radbins)
            except ValueError:
                sys.exit("Something's wrong with your radbins. Make sure to include the outer " + \
                "quotes, like so:\n" + \
                '"[10,20,30,40,50]"')
    else:
        radbins = 'none'

    outs = make_output_list(args.output, output_step=args.output_step)
    outs_save = args.output

    # Loop through outputs and stack necessary fields for combined histogram
    stacked_theta_inflow = []
    stacked_phi_inflow = []
    stacked_radius_inflow = []
    stacked_hist_inflow = []
    stacked_theta_outflow = []
    stacked_phi_outflow = []
    stacked_radius_outflow = []
    stacked_hist_outflow = []
    for i in range(len(outs)):
        snap = outs[i]
        snap_name = foggie_dir + run_dir + snap + '/' + snap
        ds, refine_box = foggie_load(snap_name, trackname, do_filter_particles=False, halo_c_v_name=halo_c_v_name)
        sphere = ds.sphere(ds.halo_center_kpc, (2.*ds.refine_width, 'kpc'))
        print('Filtering dataset')
        box_inflow, box_outflow, box_neither = filter_ds(sphere)
        if (args.region=='filament') or (args.region=='both'):
            theta_inflow = box_inflow['theta_pos'].flatten().v
            phi_inflow = box_inflow['phi_pos'].flatten().v
            radius_inflow = box_inflow['radius_corrected'].in_units('kpc').flatten().v
            hist_inflow = box_inflow[args.region_weight].flatten().v
            stacked_theta_inflow += list(theta_inflow)
            stacked_phi_inflow += list(phi_inflow)
            stacked_radius_inflow += list(radius_inflow)
            stacked_hist_inflow += list(hist_inflow)
        if (args.region=='wind') or (args.region=='both'):
            theta_outflow = box_outflow['theta_pos'].flatten().v
            phi_outflow = box_outflow['phi_pos'].flatten().v
            radius_outflow = box_outflow['radius_corrected'].in_units('kpc').flatten().v
            hist_outflow = box_outflow[args.region_weight].flatten().v
            stacked_theta_outflow += list(theta_outflow)
            stacked_phi_outflow += list(phi_outflow)
            stacked_radius_outflow += list(radius_outflow)
            stacked_hist_outflow += list(hist_outflow)

    print('Dataset(s) stacked and filtered. Finding best ellipses')
    if (args.region=='filament') or (args.region=='both'):
        region_params_inflow = find_regions(np.array(stacked_theta_inflow), np.array(stacked_phi_inflow), np.array(stacked_radius_inflow), np.array(stacked_hist_inflow), \
            radbins, save_dir, outs_save, save_suffix + '_filament', weight_label)
    if (args.region=='wind') or (args.region=='both'):
        region_params_outflow = find_regions(np.array(stacked_theta_outflow), np.array(stacked_phi_outflow), np.array(stacked_radius_outflow), np.array(stacked_hist_outflow), \
            radbins, save_dir, outs_save, save_suffix + '_wind', weight_label)
    print('Ellipses saved to', save_dir)
