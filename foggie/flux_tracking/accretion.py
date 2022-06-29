"""
Filename: accretion.py
Author: Cassi
First made: 6/21/22
Date last modified: 6/21/22

This file investigates and plots various properties of accretion (breakdown by phase, direction, and
time) at both the galaxy disk and at various places in the halo. It uses edge finding through
binary dilation and a new way of calculating fluxes into and out of arbitrary shapes on a grid.

Dependencies:
utils/consistency.py
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
import scipy.ndimage as ndimage
import datetime
from scipy.interpolate import InterpolatedUnivariateSpline as IUS
import shutil
import ast
import trident
import matplotlib.pyplot as plt

# These imports are FOGGIE-specific files
from foggie.utils.consistency import *
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

    parser.add_argument('--surface', metavar='surface', type=str, action='store', \
                        help='What closed surface do you want to compute flux across? Options are sphere,' + \
                        ' cylinder, and disk.\n' + \
                        'To specify the shape, size, and orientation of the surface you want, ' + \
                        'input a list as follows (don\'t forget the outer quotes, and put the shape in a different quote type!):\n' + \
                        'If you want a sphere, give:\n' + \
                        '"[\'sphere\', radius]"\n' + \
                        'where radius is by default in units of kpc but can be in units of Rvir if --Rvir keyword is used.\n' + \
                        'If you want a cylinder, give:\n' + \
                        '"[\'cylinder\', radius, height, axis]"\n' + \
                        'where axis specifies what axis to align the length of the cylinder with and can be one of the following:\n' + \
                        "'x'\n'y'\n'z'\n'minor' (aligns with disk minor axis)\n(x,y,z) (a tuple giving a 3D vector for an arbitrary axis).\n" + \
                        'radius and height give the dimensions of the cylinder,\n' + \
                        'by default in units of kpc but can be in units of Rvir if --Rvir keyword is used.\n' + \
                        'If you want to use the non-standard-shaped galaxy disk as identified by a density cut, give:\n' + \
                        '"[\'disk\']"\n' + \
                        'This will calculate fluxes into and out of whatever shape the disk takes on the grid.\n' + \
                        'The default option is to calculate fluxes into and out of the disk.')
    parser.set_defaults(surface="['disk']")

    parser.add_argument('--Rvir', dest='Rvir', action='store_true',
                        help='Do you want to specify your surface dimensions in units of Rvir rather than the default of kpc?\n' + \
                        'Default is no.')
    parser.set_defaults(Rvir=False)

    parser.add_argument('--flux_type', metavar='flux_type', type=str, action='store', \
                        help='What fluxes do you want to compute? Currently, the options are "mass" (includes metal masses)' + \
                        ' and "energy".\nYou can compute all of them by inputting ' + \
                        '"mass,energy" (no spaces!) ' + \
                        'and the default is to do all.')
    parser.set_defaults(flux_type="mass,energy")

    parser.add_argument('--cgm_only', dest='cgm_only', action='store_true',
                        help='Do you want to remove gas above a certain density threshold?\n' + \
                        'Default is not to do this.')
    parser.set_defaults(cgm_only=False)

    parser.add_argument('--region_filter', metavar='region_filter', type=str, action='store', \
                        help='Do you want to calculate fluxes separately for the different CGM segments?\n' + \
                        'Options are "temperature", "metallicity", and "radial velocity".\n' + \
                        'Default is not to do this.')
    parser.set_defaults(region_filter='none')

    parser.add_argument('--direction', dest='direction', action='store_true',
                        help='Do you want to calculate fluxes separately based on the direction the gas is coming from?\n' + \
                        'This will break up the fluxes into a few bins of theta and phi directions relative to\n' + \
                        'the minor axis of the galaxy disk. Default is not to do this.')
    parser.set_defaults(direction=False)

    parser.add_argument('--level', metavar='level', type=int, action='store', \
                        help='What refinement level do you want for the grid on which fluxes are calculated?\n' + \
                        'If using whole refine box or larger, going above level 9 will consume significant memory.\n' + \
                        'Default is level 9 (forced refinement level).')
    parser.set_defaults(level=9)

    parser.add_argument('--nproc', metavar='nproc', type=int, action='store', \
                        help='How many processes do you want? Default is 1 ' + \
                        '(no parallelization), if multiple outputs and multiple processors are' + \
                        ' specified, code will run one output per processor')
    parser.set_defaults(nproc=1)

    parser.add_argument('--save_suffix', metavar='save_suffix', type=str, action='store', \
                        help='Do you want to append a string onto the names of the saved files? Default is no.')
    parser.set_defaults(save_suffix="")

    parser.add_argument('--copy_to_tmp', dest='copy_to_tmp', action='store_true', \
                        help="If running on pleiades, do you want to copy the snapshot to the node's /tmp/\n" + \
                        "directory? This may speed up analysis somewhat, but also requires a large-memory node.\n" + \
                        "Default is not to do this.")
    parser.set_defaults(copy_to_tmp=False)

    parser.add_argument('--plot', metavar='plot', type=str, action='store', \
                        help='Do you want to plot fluxes, in addition to or instead of merely calculating them?\n' + \
                        'Plot options are:\n' + \
                        'accretion_viz          - Projection plots of the shape use for calculation and cells that will accrete to it\n' + \
                        'accretion_direction    - 2D plots in theta and phi bins showing location of inflow cells, colored by mass, temperature, and metallicity\n' + \
                        'accretion_vs_time      - line plot of inward mass flux vs time and redshift\n' + \
                        'Default is not to do any plotting. Specify multiple plots by listing separated with commas, no spaces.')
    parser.set_defaults(plot='none')

    parser.add_argument('--load_from_file', metavar='load_from_file', type=str, action='store', \
                        help='If plotting something, do you want to read in from file rather than re-calculating\n' + \
                        "everything? Note this doesn't work for any datashader plots; those need the full simulation\n" + \
                        'output to make. Pass the name of the file after the snapshot name with this option. Default is not to do this.')
    parser.set_defaults(load_from_file='none')

    parser.add_argument('--constant_box', metavar='constant_box', type=float, action='store', \
                        help='Do you want to use a constant box size for all calculations? If so, use\n' + \
                        'this to specify the length of one side of the box, using the same units you used for --surface.\n' + \
                        'This is useful for making videos but may not allow for higher resolution if the box is unnecessarily large.\n' + \
                        'Default is not to do this and dynamically select box size from the surface.')
    parser.set_defaults(constant_box=0.)


    args = parser.parse_args()
    return args

def make_table(flux_types):
    '''Makes the giant table that will be saved to file.'''

    if (args.direction):
        names_list = ['theta_bin', 'phi_bin']
        types_list = ['f8', 'f8']
    else:
        names_list = []
        types_list = []

    dir_name = ['net_', '_in', '_out']
    if (args.region_filter != 'none'): region_name = ['', 'lowest_', 'low-mid_', 'high-mid_', 'highest_']
    else: region_name = ['']
    for i in range(len(flux_types)):
        for k in range(len(region_name)):
            for j in range(len(dir_name)):
                if (j==0): name = dir_name[j]
                else: name = ''
                name += region_name[k]
                name += flux_types[i]
                if (j>0): name += dir_name[j]
                names_list += [name]
                types_list += ['f8']

    table = Table(names=names_list, dtype=types_list)

    return table

def set_table_units(table):
    '''Sets the units for the table. Note this needs to be updated whenever something is added to
    the table. Returns the table.'''

    for key in table.keys():
        if ('mass' in key) or ('metal' in key):
            table[key].unit = 'Msun/yr'
        elif ('energy' in key):
            table[key].unit = 'erg/yr'
        else:
            table[key].unit = 'none'
    return table

def calculate_flux(ds, grid, shape, edge_width, snap, snap_props):
    '''Calculates the flux into and out of the specified shape at the snapshot 'snap' and saves to file.'''

    tablename = prefix + 'Tables/' + snap + '_fluxes'
    Menc_profile, Mvir, Rvir = snap_props
    tsnap = ds.current_time.in_units('Gyr').v
    zsnap = ds.get_parameter('CosmologyCurrentRedshift')

    # Set up table of everything we want
    fluxes = []
    flux_filename = ''
    if ('mass' in flux_types):
        fluxes.append('mass_flux')
        fluxes.append('metal_flux')
        flux_filename += '_mass'
    if ('energy' in flux_types):
        fluxes.append('thermal_energy_flux')
        fluxes.append('kinetic_energy_flux')
        fluxes.append('potential_energy_flux')
        fluxes.append('bernoulli_energy_flux')
        fluxes.append('cooling_energy_flux')
        flux_filename += '_energy'
    table = make_table(fluxes)

    if (args.cgm_only):
        # Define the density cut between disk and CGM to vary smoothly between 1 and 0.1 between z = 0.5 and z = 0.25,
        # with it being 1 at higher redshifts and 0.1 at lower redshifts
        current_time = ds.current_time.in_units('Myr').v
        if (current_time<=8656.88):
            density_cut_factor = 1.
        elif (current_time<=10787.12):
            density_cut_factor = 1. - 0.9*(current_time-8656.88)/2130.24
        else:
            density_cut_factor = 0.1
        cgm_bool = (grid['gas','density'].in_units('g/cm**3').v < density_cut_factor * cgm_density_max)
    else:
        cgm_bool = (grid['gas','density'].in_units('g/cm**3').v > 0.)
    shape = shape & cgm_bool

    # Load grid properties
    x = grid['gas', 'x'].in_units('kpc').v - ds.halo_center_kpc[0].v
    y = grid['gas', 'y'].in_units('kpc').v - ds.halo_center_kpc[1].v
    z = grid['gas', 'z'].in_units('kpc').v - ds.halo_center_kpc[2].v
    xbins = x[:,0,0][:-1] - 0.5*np.diff(x[:,0,0])
    ybins = y[0,:,0][:-1] - 0.5*np.diff(y[0,:,0])
    zbins = z[0,0,:][:-1] - 0.5*np.diff(z[0,0,:])
    vx = grid['gas','vx_corrected'].in_units('kpc/yr').v
    vy = grid['gas','vy_corrected'].in_units('kpc/yr').v
    vz = grid['gas','vz_corrected'].in_units('kpc/yr').v
    radius = grid['gas','radius_corrected'].in_units('kpc').v
    properties = []
    if ('mass' in flux_types):
        mass = grid['gas', 'cell_mass'].in_units('Msun').v
        metals = grid['gas', 'metal_mass'].in_units('Msun').v
        properties.append(mass)
        properties.append(metals)
    if ('energy' in flux_types):
        kinetic_energy = grid['gas','kinetic_energy_corrected'].in_units('erg').v
        thermal_energy = (grid['gas','cell_mass']*grid['gas','thermal_energy']).in_units('erg').v
        potential_energy = -G * Menc_profile(radius)*gtoMsun / (radius*1000.*cmtopc)*grid['gas','cell_mass'].in_units('g').v
        bernoulli_energy = kinetic_energy + 5./3.*thermal_energy + potential_energy
        cooling_energy = thermal_energy/grid['gas','cooling_time'].in_units('yr').v
        properties.append(thermal_energy)
        properties.append(kinetic_energy)
        properties.append(potential_energy)
        properties.append(bernoulli_energy)
        properties.append(cooling_energy)

    # Define which cells are entering and leaving shape
    new_x = vx*dt + x
    new_y = vy*dt + y
    new_z = vz*dt + z
    inds_x = np.digitize(new_x, xbins)-1      # indices of new x positions
    inds_y = np.digitize(new_y, ybins)-1      # indices of new y positions
    inds_z = np.digitize(new_z, zbins)-1      # indices of new z positions
    new_inds = np.array([inds_x, inds_y, inds_z])
    new_in_shape = shape[tuple(new_inds)]
    from_shape = shape & ~new_in_shape
    to_shape = ~shape & new_in_shape

    if ('accretion_viz' in plots):
        # Set all values outside of the shapes of interest to zero
        temperature = grid['gas','temperature'].in_units('K').v
        density = grid['gas','density'].in_units('g/cm**3').v
        temp_shape = np.copy(temperature)
        temp_shape[~shape] = 0.
        temp_acc = np.copy(temperature)
        temp_acc[~to_shape] = 0.
        # Load these back into yt so we can make projections
        data = dict(temperature = (temperature, "K"), temperature_shape = (temp_shape, 'K'), \
                    temperature_accreting = (temp_acc, 'K'), density = (density, 'g/cm**3'))
        bbox = np.array([[np.min(x), np.max(x)], [np.min(y), np.max(y)], [np.min(z), np.max(z)]])
        ds_viz = yt.load_uniform_grid(data, temperature.shape, length_unit="kpc", bbox=bbox)
        ad = ds_viz.all_data()
        # Make cut regions to remove the "null values" from before
        shape_region = ad.cut_region("obj['temperature_shape'] > 0")
        accreting_region = ad.cut_region("obj['temperature_accreting'] > 0")
        # Make projection plots
        proj = yt.ProjectionPlot(ds_viz, 'x', 'temperature_shape', data_source=shape_region, weight_field='density', fontsize=28)
        proj.set_log('temperature_shape', True)
        proj.set_cmap('temperature_shape', sns.blend_palette(('salmon', "#984ea3", "#4daf4a", "#ffe34d", 'darkorange'), as_cmap=True))
        proj.set_zlim('temperature_shape', 1e4,1e7)
        proj.set_colorbar_label('temperature_shape', 'Temperature [K]')
        proj.annotate_text((0.03, 0.885), '%.2f Gyr\n$z=%.2f$' % (tsnap, zsnap), coord_system="axis", text_args={'color':'black'}, \
          inset_box_args={"boxstyle":"round,pad=0.3","facecolor":"white","linewidth":2,"edgecolor":"black"})
        proj.save(prefix + 'Plots/' + snap + '_temperature-shape_x' + save_suffix + '.png')
        proj = yt.ProjectionPlot(ds_viz, 'x', 'temperature_accreting', data_source=accreting_region, weight_field='density', fontsize=28)
        proj.set_log('temperature_accreting', True)
        proj.set_cmap('temperature_accreting', sns.blend_palette(('salmon', "#984ea3", "#4daf4a", "#ffe34d", 'darkorange'), as_cmap=True))
        proj.set_zlim('temperature_accreting', 1e4,1e7)
        proj.set_colorbar_label('temperature_accreting', 'Temperature [K]')
        proj.annotate_text((0.03, 0.885), '%.2f Gyr\n$z=%.2f$' % (tsnap, zsnap), coord_system="axis", text_args={'color':'black'}, \
          inset_box_args={"boxstyle":"round,pad=0.3","facecolor":"white","linewidth":2,"edgecolor":"black"})
        proj.save(prefix + 'Plots/' + snap + '_temperature-accreting_x' + save_suffix + '.png')

    # Set up filtering
    if (args.region_filter!='none'):
        if (args.region_filter=='temperature'):
            regions = [0., 10**4., 10**5., 10**6., np.inf]
            filter = grid['gas','temperature'].in_units('K').v
        elif (args.region_filter=='metallicity'):
            regions = [0., 0.1, 0.5, 1., np.inf]
            filter = grid['gas','metallicity'].in_units('Zsun').v
        elif (args.region_filter=='velocity'):
            regions = [-np.inf, -100., 0., 100., np.inf]
            filter = grid['gas','radial_velocity_corrected'].in_units('km/s').v

    if (args.direction):
        theta_bins = np.arange(-1., 1.25, 0.25)*np.pi
        phi_bins = np.arange(0., 1.25, 0.25)*np.pi
        theta = grid['gas','theta_pos_disk'].v
        phi = grid['gas','phi_pos_disk'].v
        theta_to = theta[to_shape]
        phi_to = phi[to_shape]
        theta_from = theta[from_shape]
        phi_from = phi[from_shape]
    else:
        theta_bins = np.array([-1.,1.])*np.pi
        phi_bins = np.array([0.,1.])*np.pi

    for t in range(len(theta_bins)-1):
        for p in range(len(phi_bins)-1):
            angle_bin_to = (theta_to >= theta_bins[t]) & (theta_to < theta_bins[t+1]) & (phi_to >= phi_bins[p]) & (phi_to < phi_bins[p+1])
            angle_bin_from = (theta_from >= theta_bins[t]) & (theta_from < theta_bins[t+1]) & (phi_from >= phi_bins[p]) & (phi_from < phi_bins[p+1])
            if (args.direction): results = [theta_bins[t], phi_bins[p]]
            else: results = []
            for i in range(len(fluxes)):
                prop_to = properties[i][to_shape][angle_bin_to]
                prop_from = properties[i][from_shape][angle_bin_from]
                flux_in = np.sum(prop_to)/dt
                flux_out = np.sum(prop_from)/dt
                flux_net = flux_in - flux_out
                results.append(flux_net)
                results.append(flux_in)
                results.append(flux_out)
                if (args.region_filter!='none'):
                    region_to = filter[to_shape][angle_bin_to]
                    region_from = filter[from_shape][angle_bin_from]
                    for j in range(len(regions)-1):
                        prop_to_region = prop_to[(region_to > regions[j]) & (region_to < regions[j+1])]
                        prop_from_region = prop_from[(region_from > regions[j]) & (region_from < regions[j+1])]
                        flux_in = np.sum(prop_to_region)/dt
                        flux_out = np.sum(prop_from_region)/dt
                        flux_net = flux_in - flux_out
                        results.append(flux_net)
                        results.append(flux_in)
                        results.append(flux_out)
            table.add_row(results)

    table = set_table_units(table)
    table.write(tablename + flux_filename + save_suffix + '.hdf5', path='all_data', serialize_meta=True, overwrite=True)

    if ('accretion_direction' in plots):
        tsnap = ds.current_time.in_units('Gyr').v
        zsnap = ds.get_parameter('CosmologyCurrentRedshift')
        for c in ['temperature','metallicity']:
            if (c=='temperature'):
                color_field = 'temperature'
                color_val = np.log10(grid['gas','temperature'].in_units('K').v)[to_shape]
                color_func = categorize_by_temp
                color_key = new_phase_color_key
                cmin = temperature_min_datashader
                cmax = temperature_max_datashader
                color_ticks = [50,300,550]
                color_ticklabels = ['4','5','6']
                field_label = 'log T [K]'
                color_log = True
            elif (c=='metallicity'):
                color_field = 'metallicity'
                color_val = (grid['gas','metallicity'].in_units('Zsun').v)[to_shape]
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
            data_frame = pd.DataFrame({})
            data_frame['theta'] = theta_to
            data_frame['phi'] = phi_to
            data_frame[color_field] = color_val
            data_frame['color'] = color_func(data_frame[color_field])
            data_frame.color = data_frame.color.astype('category')
            x_range = [-np.pi, np.pi]
            y_range = [0., np.pi]
            cvs = dshader.Canvas(plot_width=1200, plot_height=600, x_range=x_range, y_range=y_range)
            agg = cvs.points(data_frame, 'theta', 'phi', dshader.count_cat('color'))
            img = tf.spread(tf.shade(agg, color_key=color_key, how='eq_hist',min_alpha=100), shape='circle', px=2)
            export_image(img, prefix + 'Plots/' + snap + '_accretion-direction_' + color_field + '-colored' + save_suffix)
            fig = plt.figure(figsize=(11,7),dpi=300)
            ax = fig.add_subplot(1,1,1)
            image = plt.imread(prefix + 'Plots/' + snap + '_accretion-direction_' + color_field + '-colored' + save_suffix + '.png')
            ax.imshow(image, extent=[x_range[0],x_range[1],y_range[0],y_range[1]])
            ax.plot([-np.pi, np.pi], [np.pi/2., np.pi/2.], 'k-', lw=1)
            ax.text(0., 1.15, '%.2f Gyr\n$z=%.2f$' % (tsnap, zsnap), fontsize=20, ha='left', va='center', transform=ax.transAxes, bbox={'fc':'white','ec':'black','boxstyle':'round','lw':2})
            ax.set_xlabel('Angle around disk ($\\theta$)', fontsize=24)
            ax.set_ylabel('Angle from minor axis ($\\phi$)', fontsize=24)
            ax.tick_params(axis='both', which='both', direction='in', length=8, width=2, pad=5, labelsize=18, \
              top=True, right=True)
            ax2 = fig.add_axes([0.52, 0.89, 0.4, 0.1])
            cmap = create_foggie_cmap(cmin, cmax, color_func, color_key, color_log)
            ax2.imshow(np.flip(cmap.to_pil(), 1))
            ax2.set_xticks(color_ticks)
            ax2.set_xticklabels(color_ticklabels, fontsize=18)
            ax2.text(400, 150, field_label, fontsize=24, ha='center', va='center')
            ax2.spines["top"].set_color('white')
            ax2.spines["bottom"].set_color('white')
            ax2.spines["left"].set_color('white')
            ax2.spines["right"].set_color('white')
            ax2.set_ylim(60, 180)
            ax2.set_xlim(-10, 750)
            ax2.set_yticklabels([])
            ax2.set_yticks([])
            plt.subplots_adjust(left=0.09, bottom=0.05, top=0.87, right=0.98)
            plt.savefig(prefix + 'Plots/' + snap + '_accretion-direction_' + color_field + '-colored' + save_suffix + '.png')
            plt.close()

        data_frame = pd.DataFrame({})
        data_frame['theta'] = theta_to
        data_frame['phi'] = phi_to
        data_frame['mass'] = mass[to_shape]/dt
        x_range = [-np.pi, np.pi]
        y_range = [0., np.pi]
        cvs = dshader.Canvas(plot_width=1200, plot_height=600, x_range=x_range, y_range=y_range)
        agg = cvs.points(data_frame, 'theta', 'phi', dshader.sum('mass'))
        img = tf.spread(tf.shade(agg, cmap=mpl.cm.get_cmap('PuBuGn')), shape='circle', px=2)
        export_image(img, prefix + 'Plots/' + snap + '_accretion-direction_mass-colored' + save_suffix)
        fig = plt.figure(figsize=(11,7),dpi=300)
        ax = fig.add_subplot(1,1,1)
        image = plt.imread(prefix + 'Plots/' + snap + '_accretion-direction_mass-colored' + save_suffix + '.png')
        im = ax.imshow(image, extent=[x_range[0],x_range[1],y_range[0],y_range[1]])
        ax.plot([-np.pi, np.pi], [np.pi/2., np.pi/2.], 'k-', lw=1)
        ax.text(0., 1.15, '%.2f Gyr\n$z=%.2f$' % (tsnap, zsnap), fontsize=20, ha='left', va='center', transform=ax.transAxes, bbox={'fc':'white','ec':'black','boxstyle':'round','lw':2})
        ax.set_xlabel('Angle around disk ($\\theta$)', fontsize=24)
        ax.set_ylabel('Angle from minor axis ($\\phi$)', fontsize=24)
        ax.tick_params(axis='both', which='both', direction='in', length=8, width=2, pad=5, labelsize=18, \
          top=True, right=True)
        ax2 = fig.add_axes([0.52, 0.87, 0.4, 0.05])
        fig.colorbar(plt.cm.ScalarMappable(cmap=mpl.cm.get_cmap('PuBuGn')), cax=ax2, orientation='horizontal', ticks=[])
        ax2.text(0.5, 1.1, 'Accreting mass flux', fontsize=20, ha='center', va='bottom', transform=ax2.transAxes)
        ax2.text(0., -0.1, 'Less mass', fontsize=18, ha='left', va='top', transform=ax2.transAxes)
        ax2.text(1., -0.1, 'More mass', fontsize=18, ha='right', va='top', transform=ax2.transAxes)
        plt.subplots_adjust(left=0.09, bottom=0.05, top=0.87, right=0.98)
        plt.savefig(prefix + 'Plots/' + snap + '_accretion-direction_mass-colored' + save_suffix + '.png')
        plt.close()

        data_frame = pd.DataFrame({})
        data_frame['theta'] = theta_to
        data_frame['phi'] = phi_to
        data_frame['mass'] = metals[to_shape]/dt
        x_range = [-np.pi, np.pi]
        y_range = [0., np.pi]
        cvs = dshader.Canvas(plot_width=1200, plot_height=600, x_range=x_range, y_range=y_range)
        agg = cvs.points(data_frame, 'theta', 'phi', dshader.sum('mass'))
        img = tf.spread(tf.shade(agg, cmap=mpl.cm.get_cmap('PuBuGn')), shape='circle', px=2)
        export_image(img, prefix + 'Plots/' + snap + '_accretion-direction_metal-mass-colored' + save_suffix)
        fig = plt.figure(figsize=(11,7),dpi=300)
        ax = fig.add_subplot(1,1,1)
        image = plt.imread(prefix + 'Plots/' + snap + '_accretion-direction_metal-mass-colored' + save_suffix + '.png')
        im = ax.imshow(image, extent=[x_range[0],x_range[1],y_range[0],y_range[1]])
        ax.plot([-np.pi, np.pi], [np.pi/2., np.pi/2.], 'k-', lw=1)
        ax.text(0., 1.15, '%.2f Gyr\n$z=%.2f$' % (tsnap, zsnap), fontsize=20, ha='left', va='center', transform=ax.transAxes, bbox={'fc':'white','ec':'black','boxstyle':'round','lw':2})
        ax.set_xlabel('Angle around disk ($\\theta$)', fontsize=24)
        ax.set_ylabel('Angle from minor axis ($\\phi$)', fontsize=24)
        ax.tick_params(axis='both', which='both', direction='in', length=8, width=2, pad=5, labelsize=18, \
          top=True, right=True)
        ax2 = fig.add_axes([0.52, 0.87, 0.4, 0.05])
        fig.colorbar(plt.cm.ScalarMappable(cmap=mpl.cm.get_cmap('PuBuGn')), cax=ax2, orientation='horizontal', ticks=[])
        ax2.text(0.5, 1.1, 'Accreting metal mass flux', fontsize=20, ha='center', va='bottom', transform=ax2.transAxes)
        ax2.text(0., -0.1, 'Less mass', fontsize=18, ha='left', va='top', transform=ax2.transAxes)
        ax2.text(1., -0.1, 'More mass', fontsize=18, ha='right', va='top', transform=ax2.transAxes)
        plt.subplots_adjust(left=0.09, bottom=0.05, top=0.87, right=0.98)
        plt.savefig(prefix + 'Plots/' + snap + '_accretion-direction_metal-mass-colored' + save_suffix + '.png')
        plt.close()

def find_shape(ds, surface, snap_props):
    '''Defines the grid within the data set, identifies the specified shape,
    and returns both the full grid and the boolean array for the shape.'''

    Menc_profile, Mvir, Rvir = snap_props
    edge_kpc = 20.      # Edge region around shape in kpc

    if (args.constant_box!=0.):
        if (args.Rvir):
            max_extent = args.constant_box/2.*Rvir
        else:
            max_extent = args.constant_box/2.
    else:
        if (surface[0]=='disk'):
            max_extent = edge_kpc * 3.
        elif (surface[0]=='sphere'):
            if (args.Rvir):
                max_extent = surface[1]*Rvir + 2.*edge_kpc
                R = surface[1] * Rvir
            else:
                max_extent = surface[1] + 2.*edge_kpc
                R = surface[1]
        elif (surface[0]=='cylinder'):
            if (args.Rvir):
                radius = surface[1] * Rvir
                height = surface[2] * Rvir
            else:
                radius = surface[1]
                height = surface[2]
            max_extent = np.max([radius, height/2.])*np.sqrt(2.) + edge_kpc*2.

    data = ds.sphere(ds.halo_center_kpc, (max_extent, 'kpc'))
    pix_res = float(np.min(data[('gas','dx')].in_units('kpc')))  # at level 11
    lvl1_res = pix_res*2.**11.
    dx = lvl1_res/(2.**args.level)
    edge_width = int(edge_kpc/dx)

    if (args.constant_box!=0.):
        left_edge = ds.halo_center_kpc - ds.arr([max_extent, max_extent, max_extent], 'kpc')
        box_width = ds.arr([int(2.*max_extent/dx), int(2.*max_extent/dx), int(2.*max_extent/dx)])
        box = ds.covering_grid(level=args.level, left_edge=left_edge, dims=box_width)
    else:
        if (surface[0]=='disk'):
            # Define the density cut between disk and CGM to vary smoothly between 1 and 0.1 between z = 0.5 and z = 0.25,
            # with it being 1 at higher redshifts and 0.1 at lower redshifts
            current_time = ds.current_time.in_units('Myr').v
            if (current_time<=8656.88):
                density_cut_factor = 1.
            elif (current_time<=10787.12):
                density_cut_factor = 1. - 0.9*(current_time-8656.88)/2130.24
            else:
                density_cut_factor = 0.1
            density = data['gas','density'].in_units('g/cm**3')
            disk = data.include_above(('gas','density'), density_cut_factor * cgm_density_max)
            x = disk['gas','x'].in_units('kpc').v - ds.halo_center_kpc[0].v
            y = disk['gas','y'].in_units('kpc').v - ds.halo_center_kpc[1].v
            z = disk['gas','z'].in_units('kpc').v - ds.halo_center_kpc[2].v
            x_extent = max([np.max(x)+2.*edge_kpc,np.abs(np.min(x)-2.*edge_kpc)])
            y_extent = max([np.max(y)+2.*edge_kpc,np.abs(np.min(y)-2.*edge_kpc)])
            z_extent = max([np.max(z)+2.*edge_kpc,np.abs(np.min(z)-2.*edge_kpc)])
            left_edge = ds.halo_center_kpc - ds.arr([x_extent, y_extent, z_extent], 'kpc')
            box_width = np.array([int(2.*x_extent/dx), int(2.*y_extent/dx), int(2.*z_extent/dx)])
            box = ds.covering_grid(level=args.level, left_edge=left_edge, dims=box_width)
        elif (surface[0]=='sphere'):
            left_edge = ds.halo_center_kpc - ds.arr([max_extent, max_extent, max_extent], 'kpc')
            box_width = ds.arr([int(max_extent*2./dx), int(max_extent*2./dx), int(max_extent*2./dx)], 'kpc')
            box = ds.covering_grid(level=args.level, left_edge=left_edge, dims=box_width)
        elif (surface[0]=='cylinder'):
            left_edge = ds.halo_center_kpc - ds.arr([max_extent, max_extent, max_extent], 'kpc')
            box_width = ds.arr([int(max_extent*2./dx), int(max_extent*2./dx), int(max_extent*2./dx)])
            box = ds.covering_grid(level=args.level, left_edge=left_edge, dims=box_width)

    if (surface[0]=='disk'):
        # Define the density cut between disk and CGM to vary smoothly between 1 and 0.1 between z = 0.5 and z = 0.25,
        # with it being 1 at higher redshifts and 0.1 at lower redshifts
        current_time = ds.current_time.in_units('Myr').v
        if (current_time<=8656.88):
            density_cut_factor = 1.
        elif (current_time<=10787.12):
            density_cut_factor = 1. - 0.9*(current_time-8656.88)/2130.24
        else:
            density_cut_factor = 0.1
        density = box['gas','density'].in_units('g/cm**3').v
        shape = (density > density_cut_factor * cgm_density_max)
    elif (surface[0]=='sphere'):
        radius = box['gas','radius_corrected'].in_units('kpc').v
        shape = (radius < R)
    elif (surface[0]=='cylinder'):
        if (surface[3]=='minor'):
            x = box['gas','x_disk'].in_units('kpc').v - ds.halo_center_kpc[0].v
            y = box['gas','y_disk'].in_units('kpc').v - ds.halo_center_kpc[1].v
            z = box['gas','z_disk'].in_units('kpc').v - ds.halo_center_kpc[2].v
        else:
            x = box['gas','x'].in_units('kpc').v - ds.halo_center_kpc[0].v
            y = box['gas','y'].in_units('kpc').v - ds.halo_center_kpc[1].v
            z = box['gas','z'].in_units('kpc').v - ds.halo_center_kpc[2].v
        if (surface[3]=='z') or (surface[3]=='minor'):
            norm_coord = z
            rad_coord = np.sqrt(x**2. + y**2.)
        if (surface[3]=='x'):
            norm_coord = x
            rad_coord = np.sqrt(y**2. + z**2.)
        if (surface[3]=='y'):
            norm_coord = y
            rad_coord = np.sqrt(x**2. + z**2.)
        if (type(surface[3])==tuple) or (type(surface[3])==list):
            axis = np.array(surface[3])
            norm_axis = axis / np.sqrt((axis**2.).sum())
            # Define other unit vectors orthagonal to the angular momentum vector
            np.random.seed(99)
            x_axis = np.random.randn(3)            # take a random vector
            x_axis -= x_axis.dot(norm_axis) * norm_axis       # make it orthogonal to L
            x_axis /= np.linalg.norm(x_axis)            # normalize it
            y_axis = np.cross(norm_axis, x_axis)           # cross product with L
            x_vec = np.array(x_axis)
            y_vec = np.array(y_axis)
            z_vec = np.array(norm_axis)
            # Calculate the rotation matrix for converting from original coordinate system
            # into this new basis
            xhat = np.array([1,0,0])
            yhat = np.array([0,1,0])
            zhat = np.array([0,0,1])
            transArr0 = np.array([[xhat.dot(x_vec), xhat.dot(y_vec), xhat.dot(z_vec)],
                                 [yhat.dot(x_vec), yhat.dot(y_vec), yhat.dot(z_vec)],
                                 [zhat.dot(x_vec), zhat.dot(y_vec), zhat.dot(z_vec)]])
            rotationArr = np.linalg.inv(transArr0)
            x_rot = rotationArr[0][0]*x + rotationArr[0][1]*y + rotationArr[0][2]*z
            y_rot = rotationArr[1][0]*x + rotationArr[1][1]*y + rotationArr[1][2]*z
            z_rot = rotationArr[2][0]*x + rotationArr[2][1]*y + rotationArr[2][2]*z
            norm_coord = z_rot
            rad_coord = np.sqrt(x_rot**2. + y_rot**2.)

        shape = (norm_coord >= -height/2.) & (norm_coord <= height/2.) & (rad_coord <= radius)

    return box, shape, edge_width

def load_and_calculate(snap, surface):
    '''Loads the simulation output given by 'snap' and calls the functions to define the surface
    and calculate flux through that surface.'''

    # Load simulation output
    if (args.system=='pleiades_cassi'):
        print('Copying directory to /tmp')
        if (args.copy_to_tmp):
            snap_dir = '/tmp/' + args.halo + '/' + args.run + '/' + target_dir + '/' + snap
            shutil.copytree(foggie_dir + run_dir + snap, snap_dir)
            snap_name = snap_dir + '/' + snap
        else:
            # Make a dummy directory with the snap name so the script later knows the process running
            # this snapshot failed if the directory is still there
            snap_dir = '/nobackup/clochhaa/tmp/' + args.halo + '/' + args.run + '/' + target_dir + '/' + snap
            os.makedirs(snap_dir)
            snap_name = foggie_dir + run_dir + snap + '/' + snap
    else:
        snap_name = foggie_dir + run_dir + snap + '/' + snap
    if ((surface[0]=='cylinder') and (surface[3]=='minor')) or (args.direction):
        ds, refine_box = foggie_load(snap_name, trackname, do_filter_particles=False, halo_c_v_name=halo_c_v_name, gravity=True, masses_dir=masses_dir, disk_relative=True, particle_type_for_angmom='gas')
    else:
        ds, refine_box = foggie_load(snap_name, trackname, do_filter_particles=False, halo_c_v_name=halo_c_v_name, gravity=True, masses_dir=masses_dir)
    zsnap = ds.get_parameter('CosmologyCurrentRedshift')

    # Load the mass enclosed profile
    if (zsnap > 2.):
        masses = Table.read(masses_dir + 'masses_z-gtr-2.hdf5', path='all_data')
    else:
        masses = Table.read(masses_dir + 'masses_z-less-2.hdf5', path='all_data')
    rvir_masses = Table.read(masses_dir + 'rvir_masses.hdf5', path='all_data')
    masses_ind = np.where(masses['snapshot']==snap)[0]
    Menc_profile = IUS(np.concatenate(([0],masses['radius'][masses_ind])), np.concatenate(([0],masses['total_mass'][masses_ind])))
    Mvir = rvir_masses['total_mass'][rvir_masses['snapshot']==snap][0]
    Rvir = rvir_masses['radius'][rvir_masses['snapshot']==snap][0]
    snap_props = [Menc_profile, Mvir, Rvir]

    # Find the covering grid and the shape specified by 'surface'
    grid, shape, edge_width = find_shape(ds, surface, snap_props)
    # Calculate fluxes and save to file
    calculate_flux(ds, grid, shape, edge_width, snap, snap_props)

    print('Fluxes calculated for snapshot', snap)

if __name__ == "__main__":

    gtoMsun = 1.989e33
    cmtopc = 3.086e18
    stoyr = 3.155e7
    G = 6.673e-8
    kB = 1.38e-16
    mu = 0.6
    mp = 1.67e-24
    dt = 5.38e6

    args = parse_args()
    print(args.halo)
    print(args.run)
    print(args.system)
    foggie_dir, output_dir, run_dir, code_path, trackname, haloname, spectra_dir, infofile = get_run_loc_etc(args)

    # Set directory for output location, making it if necessary
    prefix = output_dir + 'fluxes_halo_00' + args.halo + '/' + args.run + '/'
    if not (os.path.exists(prefix)): os.system('mkdir -p ' + prefix)

    print('foggie_dir: ', foggie_dir)
    halo_c_v_name = code_path + 'halo_infos/00' + args.halo + '/' + args.run + '/halo_c_v'
    masses_dir = code_path + 'halo_infos/00' + args.halo + '/' + args.run + '/'

    if (args.save_suffix!=''):
        save_suffix = '_' + args.save_suffix
    else:
        save_suffix = ''

    # Build flux type list
    if (',' in args.flux_type):
        flux_types = args.flux_type.split(',')
    else:
        flux_types = [args.flux_type]

    # Build plots list
    if (',' in args.plot):
        plots = args.plot.split(',')
    else:
        plots = [args.plot]

    surface = ast.literal_eval(args.surface)
    outs = make_output_list(args.output, output_step=args.output_step)

    if (args.nproc==1):
        for snap in outs:
            load_and_calculate(snap, surface)
    else:
        if (save_suffix != ''):
            target_dir = save_suffix
        else:
            target_dir = 'fluxes'
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
                    threads.append(multi.Process(target=load_and_calculate, args=[snap, surface]))
                for t in threads:
                    t.start()
                for t in threads:
                    t.join()
                # Delete leftover outputs from failed processes from tmp directory if on pleiades
                if (args.system=='pleiades_cassi'):
                    if (args.copy_to_tmp):
                        snap_dir = '/tmp/' + args.halo + '/' + args.run + '/' + target_dir + '/'
                    else:
                        snap_dir = '/nobackup/clochhaa/tmp/' + args.halo + '/' + args.run + '/' + target_dir + '/'
                    for s in range(len(snaps)):
                        if (os.path.exists(snap_dir + snaps[s])):
                            print('Deleting failed %s from /tmp' % (snaps[s]))
                            skipped_outs.append(snaps[s])
                            shutil.rmtree(snap_dir + snaps[s])
            # For any leftover snapshots, run one per processor
            threads = []
            snaps = []
            for j in range(len(outs)%args.nproc):
                snap = outs[-(j+1)]
                snaps.append(snap)
                threads.append(multi.Process(target=load_and_calculate, args=[snap, surface]))
            for t in threads:
                t.start()
            for t in threads:
                t.join()
            # Delete leftover outputs from failed processes from tmp directory if on pleiades
            if (args.system=='pleiades_cassi'):
                if (args.copy_to_tmp):
                    snap_dir = '/tmp/' + args.halo + '/' + args.run + '/' + target_dir + '/'
                else:
                    snap_dir = '/nobackup/clochhaa/tmp/' + args.halo + '/' + args.run + '/' + target_dir + '/'
                for s in range(len(snaps)):
                    if (os.path.exists(snap_dir + snaps[s])):
                        print('Deleting failed %s from /tmp' % (snaps[s]))
                        skipped_outs.append(snaps[s])
                        shutil.rmtree(snap_dir + snaps[s])
            outs = skipped_outs

    print(str(datetime.datetime.now()))
    print("All snapshots finished!")
