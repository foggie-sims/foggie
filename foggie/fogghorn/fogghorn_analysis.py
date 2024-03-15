"""

Filename: fogghorn_analysis.py
Authors: Cassi,
Created: 01-24-24

This script produces a set of basic analysis plots for all outputs in the directory
passed to it.

Plots included so far:
- Gas density projection
- New stars density projection
- Kennicutt-Schmidt relation compared to KMT09 relation

"""

from __future__ import print_function

import numpy as np
import yt
from yt.units import *
from yt import YTArray
import argparse
import os
from astropy.table import Table
from astropy.io import ascii
import multiprocessing as multi
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from numpy.polynomial import Polynomial

from foggie.utils.consistency import *
from foggie.utils.get_refine_box import get_refine_box
from foggie.utils.get_halo_center import get_halo_center
from foggie.utils.get_proper_box_size import get_proper_box_size
from foggie.utils.get_run_loc_etc import get_run_loc_etc
from foggie.utils.yt_fields import *
from foggie.utils.foggie_load import *
from foggie.utils.analysis_utils import *

def parse_args():
    '''Parse command line arguments. Returns args object.'''

    parser = argparse.ArgumentParser(description='Produces analysis plots for FOGGHORN runs.')

    # Optional arguments:
    parser.add_argument('--directory', metavar='directory', type=str, action='store', \
                        help='What is the directory of the enzo outputs you want to make plots of?')
    parser.set_defaults(directory='')

    parser.add_argument('--save_directory', metavar='save_directory', type=str, action='store', \
                        help='Where do you want to store the plots, if different from where the outputs are stored?')
    parser.set_defaults(save_directory='')

    parser.add_argument('--trackfile', metavar='trackfile', type=str, action='store', \
                        help='What is the directory of the track file for this halo?\n' + \
                            'This is needed to find the center of the galaxy of interest.')
    parser.set_defaults(trackfile='')

    parser.add_argument('--pwd', dest='pwd', action='store_true',
                        help='Just use the working directory?, Default is no')
    parser.set_defaults(pwd=False)

    parser.add_argument('--nproc', metavar='nproc', type=int, action='store', \
                        help='How many processes do you want? Default is 1 ' + \
                        '(no parallelization), if multiple processors are' + \
                        ' specified, code will run one output per processor')
    parser.set_defaults(nproc=1)

    args = parser.parse_args()
    return args

def gas_density_projection(ds, region, snap, save_directory):
    '''Plots a gas density projection of the galaxy disk.'''

    p = yt.ProjectionPlot(ds, ds.z_unit_disk, 'density', data_source=region,
                          width=(20, 'kpc'), center=ds.halo_center_code)
    p.set_unit('density','Msun/pc**2')
    p.set_cmap('density', density_color_map)
    p.set_zlim('density',0.01,300)
    p.save(save_directory + '/' + snap + '_Projection_disk-z_density.png')

def young_stars_density_projection(ds, region, snap, save_directory):
    '''Plots a young stars density projection of the galaxy disk.'''

    p = yt.ProjectionPlot(ds, ds.z_unit_disk, ('deposit', 'young_stars3_cic'), 
                          width=(20, 'kpc'), data_source=region, center=ds.halo_center_code)
    p.set_unit(('deposit','young_stars3_cic'),'Msun/kpc**2')
    p.set_zlim(('deposit','young_stars3_cic'),1000,1000000)
    p.save(save_directory + '/' + snap + '_Projection_disk-z_young_stars3_cic.png')

def edge_visualizations(ds, region, snap, save_directory):
    """Plot slices & thin projections of galaxy temperature viewed from the disk edge."""

    # Visualize along two perpendicular edge axes
    for label, axis in zip(["disk-x","disk-y"],
                           [ds.x_unit_disk, ds.y_unit_disk]):

        # "Thin" projections (30 kpc deep).        
        p = yt.ProjectionPlot(ds, axis, "temperature", weight_field="density",
                              center=ds.halo_center_code, data_source=region,
                              width=(60,"kpc"), depth=(30,"kpc"),
                              north_vector=ds.z_unit_disk)
        p.save(save_directory + '/' + snap + "_Projection_" + label + "temperature_density.png")

        # Slices
        s = yt.SlicePlot(ds, axis, "temperature",
                         center=ds.halo_center_code, data_source=region,
                         width=(60,"kpc"), north_vector=ds.z_unit_disk)
        s.save(save_directory + '/' + snap + "_Slice_" + label + "temperature.png")

def KS_relation(ds, region, snap, save_directory):
    '''Plots the KS relation from the dataset as compared to a curve taken from KMT09.'''

    # Make a projection and convert to FRB
    p = yt.ProjectionPlot(ds, ds.z_unit_disk, 'density', 
                          data_source=region, width=(20, 'kpc'), center=ds.halo_center_code)
    proj_frb = p.data_source.to_frb((20., "kpc"), 500)
    # Pull out the gas surface density and the star formation rate of the young stars
    projected_density = proj_frb['density'].in_units('Msun/pc**2')
    ks_nh1 = proj_frb['H_p0_number_density'].in_units('pc**-2') * yt.YTArray(1.67e-24/1.989e33, 'Msun')
    young_stars = proj_frb[('deposit', 'young_stars3_cic')].in_units('Msun/kpc**2')
    ks_sfr = young_stars / yt.YTArray(3e6, 'yr') + yt.YTArray(1e-6, 'Msun/kpc**2/yr')

    # These values are pulled from KMT09
    log_sigma_gas = [0.5278, 0.6571, 0.8165, 1.0151, 1.2034, 1.4506, 1.6286, 1.9399, 2.2663, 2.7905, 3.5817]
    log_sigma_sfr = [-5.1072, -4.4546, -3.5572, -2.7926, -2.3442, -2.0185, -1.8253, -1.5406, -1.0927, -0.3801, 0.6579]
    c = Polynomial.fit(log_sigma_gas, log_sigma_sfr, deg=5)

    # Make the plot
    plt.plot(np.log10(ks_nh1), np.log10(ks_sfr), '.')
    plt.plot(log_sigma_gas, log_sigma_sfr, marker='o', color='red')
    plt.xlabel('$\Sigma _{g} \,\, (M_{\odot} / pc^2)$', fontsize=16)
    plt.ylabel('$\dot{M} _{*} \,\, (M_{\odot} / yr / kpc^2)$', fontsize=16)
    plt.axis([-1,5,-6,3])
    plt.tick_params(axis='both', which='both', direction='in', length=8, width=2, pad=5, labelsize=14, \
        top=True, right=True)
    plt.tight_layout()
    plt.savefig(save_directory + '/' + snap + '_KS-relation.png', dpi=300)

def make_plots(snap, directory, save_directory, trackfile):
    '''Finds the halo center and other properties of the dataset and then calls
    the plotting scripts.'''

    filename = directory + '/' + snap + '/' + snap
    ds, region = foggie_load(filename, trackfile, disk_relative=True)

    # Make the plots!
    # Eventually want to make this check to see if the plots already exist first before re-making them
    gas_density_projection(ds, region, snap, save_directory)
    young_stars_density_projection(ds, region, snap, save_directory)
    edge_visualizations(ds, region, snap, save_directory)
    KS_relation(ds, region, snap, save_directory)

if __name__ == "__main__":
    args = parse_args()

    if (args.save_directory==''): args.save_directory = args.directory

    outputs = []
    for fname in os.listdir(args.directory):
        folder_path = os.path.join(args.directory, fname)
        if os.path.isdir(folder_path) and ((fname[0:2]=='DD') or (fname[0:2]=='RD')):
            outputs.append(fname)
    # Loop over outputs, for either single-processor or parallel processor computing
    if (args.nproc==1):
        for snap in outputs:
            make_plots(snap)
    else:
        # Split into a number of groupings equal to the number of processors
        # and run one process per processor
        for i in range(len(outputs)//args.nproc):
            threads = []
            for j in range(args.nproc):
                snap = outputs[args.nproc*i+j]
                threads.append(multi.Process(target=make_plots, \
    			       args=[snap, args.directory, args.save_directory, args.trackfile]))
            for t in threads:
                t.start()
            for t in threads:
                t.join()
        # For any leftover snapshots, run one per processor
        threads = []
        for j in range(len(outputs)%args.nproc):
            snap = outputs[-(j+1)]
            threads.append(multi.Process(target=make_plots, \
                   args=[snap, args.directory, args.save_directory, args.trackfile]))
        for t in threads:
            t.start()
        for t in threads:
            t.join()