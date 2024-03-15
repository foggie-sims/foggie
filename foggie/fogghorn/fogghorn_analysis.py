"""

Filename: fogghorn_analysis.py
Authors: Cassi, Ayan, Anna, Claire
Created: 01-24-24
Last modified: by Ayan in March 2024

This script produces a set of basic analysis plots for all outputs in the directory passed to it.

Plots included so far:
- Gas density projection
- New stars density projection
- Kennicutt-Schmidt relation compared to KMT09 relation

Examples of how to run: run fogghorn_analysis.py --directory /Users/acharyya/models/simulation_output/foggie/halo_5205/natural_7n --upto_kpc 10 --docomoving --weight mass
                        run fogghorn_analysis.py --directory /Users/acharyya/models/simulation_output/foggie/halo_008508/nref11c_nref9f --system ayan_local --halo 8508 --output RD0030 --upto_kpc 10 --docomoving --clobber
"""

from __future__ import print_function

import numpy as np
import argparse
import os
import matplotlib
#matplotlib.use('agg')
import matplotlib.pyplot as plt
from numpy.polynomial import Polynomial
import multiprocessing as multi
from pathlib import Path

from astropy.table import Table
from astropy.io import ascii

import yt
from yt.units import *
from yt import YTArray

from foggie.utils.consistency import *
from foggie.utils.get_refine_box import get_refine_box
from foggie.utils.get_halo_center import get_halo_center
from foggie.utils.get_proper_box_size import get_proper_box_size
from foggie.utils.get_run_loc_etc import get_run_loc_etc
from foggie.utils.yt_fields import *
from foggie.utils.foggie_load import *
from foggie.utils.analysis_utils import *

# --------------------------------------------------------------------------------------------------------------------
def parse_args():
    '''Parse command line arguments. Returns args object.'''

    parser = argparse.ArgumentParser(description='Produces analysis plots for FOGGHORN runs.')

    # Optional arguments:
    parser.add_argument('--directory', metavar='directory', type=str, action='store', default='', help='What is the directory of the enzo outputs you want to make plots of?')
    parser.add_argument('--save_directory', metavar='save_directory', type=str, action='store', default=None, help='Where do you want to store the plots, if different from where the outputs are stored?')
    parser.add_argument('--output', metavar='output', type=str, action='store', default=None, help='If you want to make the plots for specific output/s then specify those here separated by comma (e.g., DD0030,DD0040) or with dashes to include all outputs between (e.g., DD0030-DD040). Otherwise (default) it will make plots for ALL outputs in that directory')
    parser.add_argument('--trackfile', metavar='trackfile', type=str, action='store', default=None, help='What is the directory of the track file for this halo?\n' + 'This is needed to find the center of the galaxy of interest.')
    parser.add_argument('--pwd', dest='pwd', action='store_true', default=False, help='Just use the working directory?, Default is no')
    parser.add_argument('--nproc', metavar='nproc', type=int, action='store', default=1, help='How many processes do you want? Default is 1 (no parallelization), if multiple processors are specified, code will run one output per processor')

    parser.add_argument('--clobber', dest='clobber', action='store_true', default=False, help='Over-write existing plots? Default is no.')
    parser.add_argument('--silent', dest='silent', action='store_true', default=False, help='Suppress some generic pritn statements? Default is no.')
    parser.add_argument('--upto_kpc', metavar='upto_kpc', type=float, action='store', default=None, help='Limit analysis out to a certain physical kpc. By default it does the entire refine box.')
    parser.add_argument('--docomoving', dest='docomoving', action='store_true', default=False, help='Consider the input upto_kpc as a comoving quantity? Default is No.')
    parser.add_argument('--weight', metavar='weight', type=str, action='store', default=None, help='Name of quantity to weight the metallicity by. Default is None i.e., no weighting.')
    parser.add_argument('--projection', metavar='projection', type=str, action='store', default='z', help='Which projection do you want to plot, i.e., which axis is your line of sight? Default is z')

    # The following three args are used for backward compatibility, to find the trackfile for production runs, if a trackfile has not been explicitly specified
    parser.add_argument('--system', metavar='system', type=str, action='store', default='ayan_local', help='Which system are you on? This is used only when trackfile is not specified. Default is ayan_local')
    parser.add_argument('--halo', metavar='halo', type=str, action='store', default='8508', help='Which halo? Default is Tempesxt. This is used only when trackfile is not specified.')
    parser.add_argument('--run', metavar='run', type=str, action='store', default='nref11c_nref9f', help='Which run? Default is nref11c_nref9f. This is used only when trackfile is not specified.')

    args = parser.parse_args()
    return args

# --------------------------------------------------------------------------------------------------------------------
def need_to_make_this_plot(output_filename, args):
    '''
    Determines whether a figure with this name already exists, and if so, should it be over-written
    :return boolean
    '''
    if os.path.exists(output_filename):
        if not args.silent: print(output_filename + ' already exists.', )
        if args.clobber:
            if not args.silent: print(' But we will re-make it...')
            return True
        else:
            if not args.silent: print(' So we will skip it.')
            return False
    else:
        if not args.silent: print('About to make ' + output_filename + '...')
        return True

# --------------------------------------------------------------------------------------------------------------------
def gas_density_projection(ds, region, args):
    '''Plots a gas density projection of the galaxy disk.'''
    output_filename = args.save_directory + '/' + args.snap + '_Projection_' + args.projection + '_density.png'

    if need_to_make_this_plot(output_filename, args):
        p = yt.ProjectionPlot(ds, args.projection, 'density', data_source=region, width=(20, 'kpc'), center=ds.halo_center_code)
        p.set_unit('density','Msun/pc**2')
        p.set_cmap('density', density_color_map)
        p.set_zlim('density',0.01,300)
        p.save(output_filename)

# --------------------------------------------------------------------------------------------------------------------
def young_stars_density_projection(ds, region, args):
    '''Plots a young stars density projection of the galaxy disk.'''

    output_filename = args.save_directory + '/' + args.snap + '_Projection_disk-z_young_stars3_cic.png'

    if need_to_make_this_plot(output_filename, args):
        p = yt.ProjectionPlot(ds, ds.z_unit_disk, ('deposit', 'young_stars3_cic'), width=(20, 'kpc'), data_source=region, center=ds.halo_center_code)
        p.set_unit(('deposit','young_stars3_cic'),'Msun/kpc**2')
        p.set_zlim(('deposit','young_stars3_cic'),1000,1000000)
        p.save(output_filename)

# --------------------------------------------------------------------------------------------------------------------
def edge_visualizations(ds, region, args):
    """Plot slices & thin projections of galaxy temperature viewed from the disk edge."""

    output_basename = args.save_directory + '/' + args.snap

    # Visualize along two perpendicular edge axes
    for label, axis in zip(["disk-x","disk-y"],
                           [ds.x_unit_disk, ds.y_unit_disk]):

        p_filename = output_basename + f"_Projection_{label}_temperature_density.png"
        s_filename = output_basename + f"_Slice_{label}_temperature.png"

        if need_to_make_this_plot(p_filename, args):
            # "Thin" projections (30 kpc deep).
            p = yt.ProjectionPlot(ds, axis, "temperature", weight_field="density",
                                center=ds.halo_center_code, data_source=region,
                                width=(60,"kpc"), depth=(30,"kpc"),
                                north_vector=ds.z_unit_disk)
            p.save(p_filename)

        if need_to_make_this_plot(s_filename, args):
            # Slices
            s = yt.SlicePlot(ds, axis, "temperature",
                            center=ds.halo_center_code, data_source=region,
                            width=(60,"kpc"), north_vector=ds.z_unit_disk)
            s.save(s_filename)

# --------------------------------------------------------------------------------------------------------------------
def KS_relation(ds, region, args):
    '''Plots the KS relation from the dataset as compared to a curve taken from Krumholz, McKee, & Tumlinson (2009), ApJ 699, 850.'''

    output_filename = args.save_directory + '/' + args.snap + '_KS-relation.png'

    if need_to_make_this_plot(output_filename, args):
        # Make a projection and convert to FRB
        p = yt.ProjectionPlot(ds, 'z', 'density', data_source=region, width=(20, 'kpc'), center=ds.halo_center_code)
        proj_frb = p.data_source.to_frb((20., "kpc"), 500)
        # Pull out the gas surface density and the star formation rate of the young stars
        projected_density = proj_frb['density'].in_units('Msun/pc**2')
        ks_nh1 = proj_frb['H_p0_number_density'].in_units('pc**-2') * yt.YTArray(1.67e-24/1.989e33, 'Msun')
        young_stars = proj_frb[('deposit', 'young_stars3_cic')].in_units('Msun/kpc**2')
        ks_sfr = young_stars / yt.YTArray(3e6, 'yr') + yt.YTArray(1e-6, 'Msun/kpc**2/yr')

        # These values are pulled from KMT09 Figure 2, the log cZ' = 0.2 curve
        log_sigma_gas = [0.5278, 0.6571, 0.8165, 1.0151, 1.2034, 1.4506, 1.6286, 1.9399, 2.2663, 2.7905, 3.5817]
        log_sigma_sfr = [-5.1072, -4.4546, -3.5572, -2.7926, -2.3442, -2.0185, -1.8253, -1.5406, -1.0927, -0.3801, 0.6579]
        c = Polynomial.fit(log_sigma_gas, log_sigma_sfr, deg=5)

        # Make the plot
        plt.plot(np.log10(ks_nh1), np.log10(ks_sfr), '.')
        plt.plot(log_sigma_gas, log_sigma_sfr, marker='o', color='red')
        plt.xlabel('$\Sigma _{g} \,\, (M_{\odot} / pc^2)$', fontsize=16)
        plt.ylabel('$\dot{M} _{*} \,\, (M_{\odot} / yr / kpc^2)$', fontsize=16)
        plt.axis([-1,5,-6,3])
        plt.tick_params(axis='both', which='both', direction='in', length=8, width=2, pad=5, labelsize=14, top=True, right=True)
        plt.tight_layout()
        plt.savefig(output_filename, dpi=300)

# --------------------------------------------------------------------------------------------------------------------
def outflow_rates(ds, region, args):
    '''Plots the mass and metals outflow rates, both as a function of radius centered on the galaxy
    and as a function of height through 20x20 kpc horizontal planes above and below the disk of young stars.
    Uses only gas with outflow velocities greater than 50 km/s.'''

    output_filename = args.save_directory + '/' + args.snap + '_outflows.png'

    if need_to_make_this_plot(output_filename, args):
        # Load needed fields into arrays
        radius = region['gas','radius_corrected'].in_units('kpc')
        x = region['gas', 'x_disk'].in_units('kpc').v
        y = region['gas', 'y_disk'].in_units('kpc').v
        z = region['gas', 'z_disk'].in_units('kpc').v
        vx = region['gas','vx_disk'].in_units('kpc/yr').v
        vy = region['gas','vy_disk'].in_units('kpc/yr').v
        vz = region['gas','vz_disk'].in_units('kpc/yr').v
        mass = region['gas', 'cell_mass'].in_units('Msun').v
        metals = region['gas','metal_mass'].in_units('Msun').v
        rv = region['gas','radial_velocity_corrected'].in_units('km/s').v
        hv = region['gas','vz_disk'].in_units('km/s').v

        # Define radius and height lists
        radii = np.linspace(0.5, 20., 40)
        heights = np.linspace(0.5, 20., 40)

        # Calculate new positions of gas cells 10 Myr later
        dt = 10.e6
        new_x = vx*dt + x
        new_y = vy*dt + y
        new_z = vz*dt + z
        new_radius = np.sqrt(new_x**2. + new_y**2. + new_z**2.)

        # Sum the mass and metals passing through the boundaries
        mass_sph = []
        metal_sph = []
        mass_horiz = []
        metal_horiz = []
        for i in range(len(radii)):
            r = radii[i]
            mass_sph.append(np.sum(mass[(radius < r) & (new_radius > r) & (rv > 50.)])/dt)
            metal_sph.append(np.sum(metals[(radius < r) & (new_radius > r) & (rv > 50.)])/dt)
        for i in range(len(heights)):
            h = heights[i]
            mass_horiz.append(np.sum(mass[(np.abs(z) < h) & (np.abs(new_z) > h) & (np.abs(hv) > 50.)])/dt)
            metal_horiz.append(np.sum(metals[(np.abs(z) < h) & (np.abs(new_z) > h) & (np.abs(hv) > 50.)])/dt)

        # Plot the outflow rates
        fig = plt.figure(1, figsize=(10,4))
        ax1 = fig.add_subplot(1,2,1)
        ax2 = fig.add_subplot(1,2,2)
        ax1.plot(radii, mass_sph, 'k-', lw=2, label='Mass')
        ax1.plot(radii, metal_sph, 'k--', lw=2, label='Metals')
        ax1.set_ylabel(r'Mass outflow rate [$M_\odot$/yr]', fontsize=16)
        ax1.set_xlabel('Radius [kpc]', fontsize=16)
        ax1.set_yscale('log')
        ax1.tick_params(axis='both', which='both', direction='in', length=8, width=2, pad=5, labelsize=14)
        ax1.legend(loc=1, frameon=False, fontsize=16)
        ax2.plot(heights, mass_horiz, 'k-', lw=2, label='Mass')
        ax2.plot(heights, metal_horiz, 'k--', lw=2, label='Metals')
        ax2.set_ylabel(r'Mass outflow rate [$M_\odot$/yr]', fontsize=16)
        ax2.set_xlabel('Height from disk midplane [kpc]', fontsize=16)
        ax2.set_yscale('log')
        ax2.tick_params(axis='both', which='both', direction='in', length=8, width=2, pad=5, labelsize=14)
        plt.tight_layout()
        plt.savefig(output_filename, dpi=300)
        plt.close()

# --------------------------------------------------------------------------------------------------------------------
def make_plots(snap, args):
    '''Finds the halo center and other properties of the dataset and then calls
    the plotting scripts.'''

    # ----------------------- Read the snapshot ---------------------------------------------
    filename = args.directory + '/' + snap + '/' + snap
    ds, region = foggie_load(filename, args.trackfile, disk_relative=True)
    args.snap = snap

    # --------- If a upto_kpc is specified, then the analysis 'region' will be restricted up to that value ---------
    if args.upto_kpc is not None:
        if args.docomoving: args.galrad = args.upto_kpc / (1 + ds.current_redshift) / 0.695  # include stuff within a fixed comoving kpc h^-1, 0.695 is Hubble constant
        else: args.galrad = args.upto_kpc  # include stuff within a fixed physical kpc
        region = ds.sphere(ds.halo_center_kpc, ds.arr(args.galrad, 'kpc'))

    # ----------------------- Make the plots ---------------------------------------------
    #gas_density_projection(ds, region, args)
    #young_stars_density_projection(ds, region, args)
    #edge_visualizations(ds, region, args)
    #KS_relation(ds, region, args)
    outflow_rates(ds, region, args)


# --------------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    args = parse_args()

    # ------------------ Figure out directory and outputs -------------------------------------
    if args.save_directory is None:
        args.save_directory = args.directory + '/plots'
        Path(args.save_directory).mkdir(parents=True, exist_ok=True)

    if args.trackfile is None: _, _, _, _, args.trackfile, _, _, _ = get_run_loc_etc(args) # for FOGGIE production runs it knows which trackfile to grab

    if args.output is not None: # Running on specific output/s
        outputs = make_output_list(args.output)
    else: # Running on all snapshots in the directory
        outputs = []
        for fname in os.listdir(args.directory):
            folder_path = os.path.join(args.directory, fname)
            if os.path.isdir(folder_path) and ((fname[0:2]=='DD') or (fname[0:2]=='RD')):
                outputs.append(fname)

    # --------- Loop over outputs, for either single-processor or parallel processor computing ---------------
    if (args.nproc == 1):
        for snap in outputs:
            make_plots(snap, args)
    else:
        # Split into a number of groupings equal to the number of processors
        # and run one process per processor
        for i in range(len(outputs)//args.nproc):
            threads = []
            for j in range(args.nproc):
                snap = outputs[args.nproc*i+j]
                threads.append(multi.Process(target=make_plots, args=[snap, args]))
            for t in threads:
                t.start()
            for t in threads:
                t.join()
        # For any leftover snapshots, run one per processor
        threads = []
        for j in range(len(outputs)%args.nproc):
            snap = outputs[-(j+1)]
            threads.append(multi.Process(target=make_plots, args=[snap, args]))
        for t in threads:
            t.start()
        for t in threads:
            t.join()