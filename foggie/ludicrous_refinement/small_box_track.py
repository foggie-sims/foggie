'''
Filename: small_box_track.py
Author: Cassi
Last modified: 8-30-21

This file produces a track file for a small box relative to the old refine box center.
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
#from photutils.segmentation import detect_sources
from scipy.interpolate import InterpolatedUnivariateSpline as IUS
#import trident
from scipy.spatial.distance import cdist
from scipy.ndimage import uniform_filter
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
    '''Parse command line arguments. Returns args object.
    NOTE: Need to move command-line argument parsing to separate file.'''

    parser = argparse.ArgumentParser(description='Calculates and saves to file a bunch of fluxes.')

    # Optional arguments:
    parser.add_argument('--halo', metavar='halo', type=str, action='store', \
                        help='Which halo? Default is 8508 (Tempest)')
    parser.set_defaults(halo='8508')

    parser.add_argument('--run', metavar='run', type=str, action='store', \
                        help='Which run of the big box? Default is nref11c_nref9f')
    parser.set_defaults(run='nref11c_nref9f')

    parser.add_argument('--smallbox_run', metavar='smallbox_run', type=str, action='store', \
                        help='Which run of the small box? Default is small_box_test')
    parser.set_defaults(smallbox_run='small_box_test')

    parser.add_argument('--output', metavar='output', type=str, action='store', \
                        help='Which output do you want to start the small box run from?\n' + \
                        'Default is the z=0.5 output for Tempest, DD1477.')
    parser.set_defaults(output='DD1477')

    parser.add_argument('--system', metavar='system', type=str, action='store', \
                        help='Which system are you on? Default is cassiopeia')
    parser.set_defaults(system='cassiopeia')

    parser.add_argument('--pwd', dest='pwd', action='store_true',
                        help='Just use the working directory?, Default is no')
    parser.set_defaults(pwd=False)

    parser.add_argument('--filename', metavar='filename', type=str, action='store', \
                        help='What do you want to call the new track file? Default is small_box_track.')
    parser.set_defaults(filename='small_box_track')

    parser.add_argument('--new_center', metavar='new_center', type=str, action='store', \
                        help="Give the center of the new small box as '[x, y, z]' in kpc, relative\n" + \
                        "to the halo center (don't forget the surrounding quotes!)")

    parser.add_argument('--box_size', metavar='box_size', type=float, action='store', \
                        help='Give the size of the new small box in kpc. Default is 5 kpc.')
    parser.set_defaults(box_size=5.)

    parser.add_argument('--box_refine_level', metavar='box_refine_level', type=int, action='store', \
                        help='What level do you want to refine the new small box to? Default is 11.')
    parser.set_defaults(box_refine_level=11)

    parser.add_argument('--make_track', dest='make_track', action='store_true', \
                        help='Do you want to make a new track for a small box from --new_center and --box_size?\n' + \
                        'Default is no.')
    parser.set_defaults(make_track=False)

    parser.add_argument('--plot_box', dest='plot_box', action='store_true', \
                        help='Do you want to plot projections of the box using the track and the datasets?\n' + \
                        "Don't forget to specify --filename for the name of the small box track file. Default is no.")
    parser.set_defaults(plot_box=False)

    parser.add_argument('--save_suffix', metavar='save_suffix', type=str, action='store', \
                        help='If you want to append a string to the file name of the saved images, give it here.')
    parser.set_defaults(save_suffix='')

    args = parser.parse_args()
    return args

def sliceplot(ds, refine_box, new_center, new_left_edge, new_right_edge):
    '''Plots slice plots of the refine box with the new small box annotated, and zoom-in slices
    of the new small box alone.'''

    small_box = ds.box(new_left_edge, new_right_edge)

    big_box_centers = [[new_center[0], ds.halo_center_kpc[1], ds.halo_center_kpc[2]],
                       [ds.halo_center_kpc[0], new_center[1], ds.halo_center_kpc[2]],
                       [ds.halo_center_kpc[0], ds.halo_center_kpc[1], new_center[2]]]

    dirs = ['x', 'y', 'z']

    for a in range(len(dirs)):
        for f in ['density', 'temperature']:
            slc = yt.SlicePlot(ds, dirs[a], f, data_source=refine_box, center=big_box_centers[a], width=(ds.refine_width, 'kpc'), fontsize=34)
            if (f=='density'):
                slc.set_cmap('density', cmap=density_color_map)
                slc.set_zlim('density', 1e-30, 1e-22)
                slc.hide_axes()
            if (f=='temperature'):
                slc.set_cmap('temperature', cmap=temperature_color_map)
                slc.set_zlim('temperature', 3e3, 1e6)
                slc.hide_axes()
            slc.annotate_timestamp(redshift=True)
            slc.annotate_line([new_left_edge[0], new_left_edge[1], new_left_edge[2]],
                        [new_right_edge[0], new_left_edge[1], new_left_edge[2]], coord_system='data')
            slc.annotate_line([new_left_edge[0], new_left_edge[1], new_left_edge[2]],
                        [new_left_edge[0], new_right_edge[1], new_left_edge[2]], coord_system='data')
            slc.annotate_line([new_left_edge[0], new_left_edge[1], new_left_edge[2]],
                        [new_left_edge[0], new_left_edge[1], new_right_edge[2]], coord_system='data')

            slc.annotate_line([new_right_edge[0], new_right_edge[1], new_right_edge[2]],
                        [new_left_edge[0], new_right_edge[1], new_right_edge[2]], coord_system='data')
            slc.annotate_line([new_right_edge[0], new_right_edge[1], new_right_edge[2]],
                        [new_right_edge[0], new_left_edge[1], new_right_edge[2]], coord_system='data')
            slc.annotate_line([new_right_edge[0], new_right_edge[1], new_right_edge[2]],
                        [new_right_edge[0], new_right_edge[1], new_left_edge[2]], coord_system='data')

            slc.save(output_dir + args.output + '_Slice_refine_box_' + dirs[a] + '_' + f + save_suffix + '.png')

            slc = yt.SlicePlot(ds, dirs[a], f, data_source=small_box, center=new_center_code, width=box_size_code, fontsize=34)
            slc.set_axes_unit('kpc')
            if (f=='density'):
                slc.set_cmap('density', cmap=density_color_map)
                slc.set_zlim('density', 1e-30, 1e-22)
                slc.hide_axes()
            if (f=='temperature'):
                slc.set_cmap('temperature', cmap=temperature_color_map)
                slc.set_zlim('temperature', 3e3, 1e6)
                slc.hide_axes()
            '''if (a=='x'): slc.annotate_quiver(("gas", "vy_corrected"), ("gas", "vz_corrected"), factor=48,
                  plot_args={"color": "white"})
            if (a=='y'): slc.annotate_quiver(("gas", "vz_corrected"), ("gas", "vx_corrected"), factor=48,
                  plot_args={"color": "white"})
            if (a=='z'): slc.annotate_quiver(("gas", "vx_corrected"), ("gas", "vy_corrected"), factor=48,
                  plot_args={"color": "white"})'''
            slc.annotate_scale()

            slc.save(output_dir + args.output + '_Slice_small_box_' + dirs[a] + '_' + f + save_suffix + '.png')

def projection(ds, refine_box, new_left_edge, new_right_edge):
    '''Plots projection plots of the refine box with the new small box annotated, and zoom-in projections
    of the new small box alone.'''

    small_box = ds.box(new_left_edge, new_right_edge)
    print(np.mean(small_box['vel_mag_corrected']), (np.mean(small_box['vel_mag_corrected'])*ds.quan(5.36e6, 'yr')).to('kpc'))

    for a in ['x', 'y', 'z']:
        for f in ['density', 'temperature']:
            proj = yt.ProjectionPlot(ds, a, f, center=ds.halo_center_kpc, width=(ds.refine_width, 'kpc'), weight_field='density', fontsize=34)
            if (f=='density'):
                proj.set_cmap('density', cmap=density_color_map)
                proj.set_zlim('density', 1e-30, 1e-22)
                proj.hide_axes()
            if (f=='temperature'):
                proj.set_cmap('temperature', cmap=temperature_color_map)
                proj.set_zlim('temperature', 3e3, 1e6)
                proj.hide_axes()
            proj.annotate_timestamp(redshift=True)
            proj.annotate_line([new_left_edge[0], new_left_edge[1], new_left_edge[2]],
                        [new_right_edge[0], new_left_edge[1], new_left_edge[2]], coord_system='data')
            proj.annotate_line([new_left_edge[0], new_left_edge[1], new_left_edge[2]],
                        [new_left_edge[0], new_right_edge[1], new_left_edge[2]], coord_system='data')
            proj.annotate_line([new_left_edge[0], new_left_edge[1], new_left_edge[2]],
                        [new_left_edge[0], new_left_edge[1], new_right_edge[2]], coord_system='data')

            proj.annotate_line([new_right_edge[0], new_right_edge[1], new_right_edge[2]],
                        [new_left_edge[0], new_right_edge[1], new_right_edge[2]], coord_system='data')
            proj.annotate_line([new_right_edge[0], new_right_edge[1], new_right_edge[2]],
                        [new_right_edge[0], new_left_edge[1], new_right_edge[2]], coord_system='data')
            proj.annotate_line([new_right_edge[0], new_right_edge[1], new_right_edge[2]],
                        [new_right_edge[0], new_right_edge[1], new_left_edge[2]], coord_system='data')


            proj.annotate_scale()

            proj.save(output_dir + args.output + '_Projection_refine_box_' + a + '_' + f + save_suffix + '.png')

            proj = yt.ProjectionPlot(ds, a, f, data_source=small_box, center=new_center_code, width=box_size_code, weight_field='density', fontsize=34)
            proj.set_axes_unit('kpc')
            if (f=='density'):
                proj.set_cmap('density', cmap=density_color_map)
                proj.set_zlim('density', 1e-30, 1e-22)
                proj.hide_axes()
            if (f=='temperature'):
                proj.set_cmap('temperature', cmap=temperature_color_map)
                proj.set_zlim('temperature', 3e3, 1e6)
                proj.hide_axes()
            '''if (a=='x'): proj.annotate_quiver(("gas", "vy_corrected"), ("gas", "vz_corrected"), factor=48,
                  plot_args={"color": "white"})
            if (a=='y'): proj.annotate_quiver(("gas", "vz_corrected"), ("gas", "vx_corrected"), factor=48,
                  plot_args={"color": "white"})
            if (a=='z'): proj.annotate_quiver(("gas", "vx_corrected"), ("gas", "vy_corrected"), factor=48,
                  plot_args={"color": "white"})'''

            proj.save(output_dir + args.output + '_Projection_small_box_' + a + '_' + f + save_suffix + '.png')

        pix_res = float(np.min(refine_box['gas','dx'].in_units('kpc')))  # at level 11
        lvl1_res = pix_res*2.**11.
        min_dx = pix_res
        max_dx = lvl1_res/(2.**5.)

        slc = yt.SlicePlot(ds, a, ('gas','d' + a), center=ds.halo_center_kpc, width=(ds.refine_width, 'kpc'), fontsize=34)
        slc.set_axes_unit('kpc')
        slc.set_unit(('gas', 'd' + a), 'kpc')
        slc.set_zlim(('gas', 'd' + a), min_dx, max_dx)
        #slc.set_log(('gas', 'd' + a), False)
        slc.set_cmap(('gas', 'd' + a), cmap=discrete_cmap)
        slc.set_colorbar_label(('gas', 'd' + a), 'Cell Size (kpc)')
        slc.save(output_dir + args.output + '_Slice_refine_box_' + a + '_resolution' + save_suffix + '.png')
        slc = yt.SlicePlot(ds, a, ('gas','d' + a), center=new_center_code, width=box_size_code, fontsize=34)
        slc.set_axes_unit('kpc')
        slc.set_unit(('gas', 'd' + a), 'kpc')
        slc.set_zlim(('gas', 'd' + a), min_dx, max_dx)
        #slc.set_log(('gas', 'd' + a), False)
        slc.set_cmap(('gas', 'd' + a), cmap=discrete_cmap)
        slc.set_colorbar_label(('gas', 'd' + a), 'Cell Size (kpc)')
        slc.save(output_dir + args.output + '_Slice_small_box_' + a + '_resolution' + save_suffix + '.png')

def resolution_comparison(ds, new_center, new_left_edge, new_right_edge):
    '''Makes a datashader plot of cell mass vs. temperature compared to other simulations for the cells
    in the small box.'''

    small_box = ds.box(new_left_edge, new_right_edge)

    FIRE_res = np.log10(7100.)        # Pandya et al. (2021)
    Illustris_res = np.log10(8.5e4)   # IllustrisTNG website https://www.tng-project.org/about/

    colorparam = 'temperature'
    data_frame = pd.DataFrame({})
    mass = small_box['cell_mass'].in_units('Msun').v
    temperature = small_box['temperature'].v
    data_frame['temperature'] = np.log10(temperature).flatten()
    data_frame['temp_cat'] = categorize_by_temp(data_frame['temperature'])
    data_frame.temp_cat = data_frame.temp_cat.astype('category')
    color_key = new_phase_color_key
    cat = 'temp_cat'
    data_frame['mass'] = np.log10(mass).flatten()
    x_range = [4., 6.]
    y_range = [-5, 5]
    cvs = dshader.Canvas(plot_width=1000, plot_height=800, x_range=x_range, y_range=y_range)
    agg = cvs.points(data_frame, 'temperature', 'mass', dshader.count_cat(cat))
    img = tf.spread(tf.shade(agg, color_key=color_key, how='eq_hist',min_alpha=40), shape='square', px=1)
    export_image(img, output_dir + args.output + '_cell-mass_vs_temperature_temperature-colored' + save_suffix + '_intermediate')
    fig = plt.figure(figsize=(10,8),dpi=500)
    ax = fig.add_subplot(1,1,1)
    image = plt.imread(output_dir + args.output + '_cell-mass_vs_temperature_temperature-colored' + save_suffix + '_intermediate.png')
    ax.imshow(image, extent=[x_range[0],x_range[1],y_range[0],y_range[1]])
    ax.set_aspect(8*abs(x_range[1]-x_range[0])/(10*abs(y_range[1]-y_range[0])))
    ax.set_ylabel('log Resolution Element Mass [$M_\odot$]', fontsize=24)
    ax.set_xlabel('log Temperature [K]', fontsize=24)
    ax.tick_params(axis='both', which='both', direction='in', length=8, width=2, pad=5, labelsize=24, \
      top=True, right=True)
    #ax.plot([FIRE_res, FIRE_res],[y_range[0],y_range[1]], 'k-', lw=1)
    #ax.text(FIRE_res+0.05, -1, 'FIRE', ha='left', va='center', fontsize=20)
    #ax.plot([Illustris_res,Illustris_res],[y_range[0],y_range[1]], 'k-', lw=1)
    #ax.text(Illustris_res+0.05, -1, 'Illustris\nTNG50', ha='left', va='center', fontsize=20)
    ax2 = fig.add_axes([0.7, 0.93, 0.25, 0.06])
    cmap = create_foggie_cmap(temperature_min_datashader, temperature_max_datashader, categorize_by_temp, new_phase_color_key, log=True)
    ax2.imshow(np.flip(cmap.to_pil(), 1))
    ax2.set_xticks([50,300,550])
    ax2.set_xticklabels(['4','5','6'],fontsize=20)
    ax2.text(400, 150, 'log T [K]',fontsize=24, ha='center', va='center')
    ax2.spines["top"].set_color('white')
    ax2.spines["bottom"].set_color('white')
    ax2.spines["left"].set_color('white')
    ax2.spines["right"].set_color('white')
    ax2.set_ylim(60, 180)
    ax2.set_xlim(-10, 750)
    ax2.set_yticklabels([])
    ax2.set_yticks([])
    plt.savefig(output_dir + args.output + '_cell-mass_vs_temperature_temperature-colored' + save_suffix + '.png')
    os.system('rm ' + output_dir + args.output + '_cell-mass_vs_temperature_temperature-colored' + save_suffix + '_intermediate.png')
    plt.close()

if __name__ == "__main__":

    args = parse_args()
    print(args.halo)
    print(args.run)
    print(args.system)
    foggie_dir, output_dir, run_dir, code_path, trackname, haloname, spectra_dir, infofile = get_run_loc_etc(args)
    halo_c_v_name = code_path + 'halo_infos/00' + args.halo + '/' + args.run + '/halo_c_v'
    output_dir = code_path + '/ludicrous_refinement/'
    #output_dir = '/Users/clochhaas/Documents/Applications/Postdoc Applications 2021/'
    run_dir = 'halo_00' + args.halo + '/' + args.smallbox_run + '/'

    if (args.save_suffix!=''):
        save_suffix = '_' + args.save_suffix
    else:
        save_suffix = ''

    if (args.make_track):

        snap_name = foggie_dir + run_dir + args.output + '/' + args.output
        ds, refine_box = foggie_load(snap_name, trackname, do_filter_particles=False, halo_c_v_name=halo_c_v_name)
        zsnap = ds.get_parameter('CosmologyCurrentRedshift')

        new_center = ast.literal_eval(args.new_center)
        new_center = ds.arr(new_center, 'kpc')

        box_size = ds.quan(args.box_size, 'kpc')
        box_size_code = box_size.to('code_length')

        new_center_code = (new_center + ds.halo_center_kpc).in_units('code_length')

        new_left_edge = ds.arr(new_center_code - box_size_code*0.5).v
        new_right_edge = ds.arr(new_center_code + box_size_code*0.5).v

        if (args.plot_box):
            projection(ds, refine_box, new_left_edge, new_right_edge)
            #sliceplot(ds, refine_box, new_center, new_left_edge, new_right_edge)

        track = Table.read(trackname, format='ascii')
        track_start_ind = np.where(track['col1']==zsnap)[0][0]
        left_edge_start = [track['col2'][track_start_ind], track['col3'][track_start_ind], track['col4'][track_start_ind]]
        right_edge_start = [track['col5'][track_start_ind], track['col6'][track_start_ind], track['col7'][track_start_ind]]

        left_edge_offset = new_left_edge - left_edge_start
        right_edge_offset = right_edge_start - new_right_edge

        track['col2'] += left_edge_offset[0]
        track['col3'] += left_edge_offset[1]
        track['col4'] += left_edge_offset[2]
        track['col5'] -= right_edge_offset[0]
        track['col6'] -= right_edge_offset[1]
        track['col7'] -= right_edge_offset[2]
        track['col8'] = (np.zeros(len(track['col8'])) + args.box_refine_level).astype(int)

        track.write(output_dir + args.filename, format='ascii', overwrite=True)

    elif (args.plot_box):

        snap_name = foggie_dir + run_dir + args.output + '/' + args.output
        ds, refine_box = foggie_load(snap_name, trackname, do_filter_particles=False, halo_c_v_name=halo_c_v_name)
        zsnap = ds.get_parameter('CosmologyCurrentRedshift')
        small_box_track = np.transpose(np.loadtxt(output_dir + args.filename, usecols=[0,1,2,3,4,5,6], skiprows=1))
        rounded_z = []
        for i in range(len(small_box_track[0])):
            rounded_z.append(round(small_box_track[0][i], 3))
        rounded_z = np.array(rounded_z)
        ind = np.where(rounded_z==round(zsnap,3))[0][0]
        small_box_left_edge = np.array([small_box_track[1][ind], small_box_track[2][ind], small_box_track[3][ind]])
        small_box_right_edge = np.array([small_box_track[4][ind], small_box_track[5][ind], small_box_track[6][ind]])

        box_size_code = ds.arr(small_box_right_edge - small_box_left_edge, 'code_length')[0]
        new_center_code = ds.arr(small_box_left_edge, 'code_length') + 0.5*box_size_code
        new_center = new_center_code.to('kpc')

        projection(ds, refine_box, small_box_left_edge, small_box_right_edge)
        sliceplot(ds, refine_box, new_center, small_box_left_edge, small_box_right_edge)
        #resolution_comparison(ds, new_center, small_box_left_edge, small_box_right_edge)
