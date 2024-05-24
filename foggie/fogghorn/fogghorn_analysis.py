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

from astropy.cosmology import Planck15 as cosmo

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
    parser.add_argument('--projection', metavar='projection', type=str, action='store', default=None, help='Which projection do you want to plot, i.e., which axes are your line of sight? Default is to do x and z. Can specify multiple axes split by commas, and can do disk-relative as e.g. "x-disk".')

    parser.add_argument('--plot', metavar='plot', type=str, action='store', default='halo_info,density_projection,young_stars_projection,temperature_projection,KS_relation,outflow_rates', help='Which plots do you want to make? Give a comma-separated list. Default is all plots.')

    # The following three args are used for backward compatibility, to find the trackfile for production runs, if a trackfile has not been explicitly specified
    parser.add_argument('--system', metavar='system', type=str, action='store', default=None, help='Which system are you on? This is used only when trackfile is not specified.')
    parser.add_argument('--halo', metavar='halo', type=str, action='store', default='8508', help='Which halo? Default is Tempest. This is used only when trackfile is not specified.')
    parser.add_argument('--run', metavar='run', type=str, action='store', default='nref11c_nref9f', help='Which run? Default is nref11c_nref9f. This is used only when trackfile is not specified.')

    args = parser.parse_args()
    return args

# --------------------------------------------------------------------------------------------------------------------
def need_to_make_this_plot(output_filename, args):
    '''
    Determines whether a figure with this name already exists, and if so, should it be over-written
    :return boolean

    NOTE: NEEDS TO BE REWRITTEN SO CHECK HAPPENS *BEFORE* THE SNAPSHOT IS LOADED
    It takes too much time to load a snapshot if you're not making any plots with it
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

    for p in projections:
        output_filename = args.save_directory + '/' + args.snap + '_Projection_' + p + '_density.png'

        if need_to_make_this_plot(output_filename, args):
            if '-disk' in p:
                if 'x' in p:
                    p_dir = ds.x_unit_disk
                    north_vector = ds.z_unit_disk
                if 'y' in p:
                    p_dir = ds.y_unit_disk
                    north_vector = ds.z_unit_disk
                if 'z' in p:
                    p_dir = ds.z_unit_disk
                    north_vector = ds.x_unit_disk
                p = yt.ProjectionPlot(ds, p_dir, 'density', data_source=region, width=(20, 'kpc'), center=ds.halo_center_code, north_vector=north_vector)
            else: p = yt.ProjectionPlot(ds, p, 'density', data_source=region, width=(20, 'kpc'), center=ds.halo_center_code)
            p.set_unit('density','Msun/pc**2')
            p.set_cmap('density', density_color_map)
            p.set_zlim('density',0.01,300)
            p.set_font_size(16)
            p.annotate_timestamp(corner='upper_left', redshift=True, time=True, draw_inset_box=True)
            p.save(output_filename)

# --------------------------------------------------------------------------------------------------------------------
def young_stars_density_projection(ds, region, args):
    '''Plots a young stars density projection of the galaxy disk.'''

    for p in projections:
        output_filename = args.save_directory + '/' + args.snap + '_Projection_' + p + '_young_stars3_cic.png'

        if need_to_make_this_plot(output_filename, args):
            if '-disk' in p:
                if 'x' in p:
                    p_dir = ds.x_unit_disk
                    north_vector = ds.z_unit_disk
                if 'y' in p:
                    p_dir = ds.y_unit_disk
                    north_vector = ds.z_unit_disk
                if 'z' in p:
                    p_dir = ds.z_unit_disk
                    north_vector = ds.x_unit_disk
                p = yt.ProjectionPlot(ds, p_dir, ('deposit', 'young_stars3_cic'), width=(20, 'kpc'), data_source=region, center=ds.halo_center_code, north_vector=north_vector)
            else: p = yt.ProjectionPlot(ds, p, ('deposit', 'young_stars3_cic'), width=(20, 'kpc'), data_source=region, center=ds.halo_center_code)
            p.set_unit(('deposit','young_stars3_cic'),'Msun/kpc**2')
            p.set_zlim(('deposit','young_stars3_cic'),1000,1000000)
            p.set_cmap(('deposit','young_stars3_cic'), density_color_map)
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
            # "Thin" projections (20 kpc deep).
            p = yt.ProjectionPlot(ds, axis, "temperature", weight_field="density",
                                center=ds.halo_center_code, data_source=region,
                                width=(60,"kpc"), depth=(20,"kpc"),
                                north_vector=ds.z_unit_disk)
            p.set_cmap('temperature', sns.blend_palette(('salmon', "#984ea3", "#4daf4a", "#ffe34d", 'darkorange'), as_cmap=True))
            p.set_zlim('temperature', 1e4,1e7)
            p.annotate_timestamp(corner='upper_left', redshift=True, time=True, draw_inset_box=True)
            p.save(p_filename)

        if need_to_make_this_plot(s_filename, args):
            # Slices
            s = yt.SlicePlot(ds, axis, "temperature",
                            center=ds.halo_center_code, data_source=region,
                            width=(60,"kpc"), north_vector=ds.z_unit_disk)
            s.set_cmap('temperature', sns.blend_palette(('salmon', "#984ea3", "#4daf4a", "#ffe34d", 'darkorange'), as_cmap=True))
            s.set_zlim('temperature', 1e4,1e7)
            s.annotate_timestamp(corner='upper_left', redshift=True, time=True, draw_inset_box=True)
            s.save(s_filename)

# --------------------------------------------------------------------------------------------------------------------
def get_halo_info(ds, region, args, queue):
    '''Calculates basic information about the halo: snapshot name, time, redshift, halo x,y,z location, halo vx,vy,vz bulk velocity, virial mass, virial radius, stellar mass, star formation rate.
    NOTE: The virial mass and radius as currently written will only work for the central galaxies! Rockstar is not being run to find satellite halos.'''
    global data

    # Determine if this snapshot has already had its information calculated
    if (args.snap in args.data['snapshot']):
        if not args.silent: print('Halo info for snapshot ' + args.snap + ' already calculated.', )
        if args.clobber:
            if not args.silent: print(' But we will re-calculate it...')
            calc = True
        else:
            if not args.silent: print(' So we will skip it.')
            calc = False
    else: calc = True

    if (calc):
        if not args.silent: print('About to calculate halo info for ' + args.snap + '...')

        row = [args.snap, ds.current_time.in_units('Myr').v, ds.get_parameter('CosmologyCurrentRedshift'), \
               ds.halo_center_kpc[0], ds.halo_center_kpc[1], ds.halo_center_kpc[2], \
               ds.halo_velocity_kms[0], ds.halo_velocity_kms[1], ds.halo_velocity_kms[2]]
        
        sph = ds.sphere(center = ds.halo_center_kpc, radius = (400., 'kpc'))
        filter_particles(sph)
        prof_dm = yt.create_profile(sph, ('index', 'radius'), fields = [('deposit', 'dm_mass')], \
                                    n_bins = 500, weight_field = None, accumulation = True)
        prof_stars = yt.create_profile(sph, ('index', 'radius'), fields = [('deposit', 'stars_mass')], \
                                    n_bins = 500, weight_field = None, accumulation = True)
        prof_young_stars = yt.create_profile(sph, ('index', 'radius'), fields = [('deposit', 'young_stars_mass')], \
                                    n_bins = 500, weight_field = None, accumulation = True)
        prof_gas = yt.create_profile(sph, ('index', 'radius'), fields = [('gas', 'cell_mass')],\
                                    n_bins = 500, weight_field = None, accumulation = True)

        internal_density =  (prof_dm[('deposit', 'dm_mass')].to('g') + prof_stars[('deposit', 'stars_mass')].to('g') + \
                             prof_gas[('gas', 'cell_mass')].to('g'))/(4*np.pi*prof_dm.x.to('cm')**3./3.)

        rho_crit = cosmo.critical_density(ds.current_redshift)
        rvir = prof_dm.x[np.argmin(abs(internal_density.value - 200*rho_crit.value))]
        Mdm_rvir    = prof_dm[('deposit', 'dm_mass')][np.argmin(abs(internal_density.value - 200*rho_crit.value))]
        Mstars_rvir = prof_stars[('deposit', 'stars_mass')][np.argmin(abs(internal_density.value - 200*rho_crit.value))]
        Mgas_rvir   = prof_gas[('gas', 'cell_mass')][np.argmin(abs(internal_density.value - 200*rho_crit.value))]
        Mvir = Mdm_rvir + Mstars_rvir + Mgas_rvir
        Myoung_stars = prof_young_stars[('deposit','young_stars_mass')][np.where(prof_young_stars.x.to('kpc') >= 20.)[0][0]]
        SFR = Myoung_stars.to('Msun').v/1e7

        row.append(Mvir.to('Msun').v)
        row.append(rvir.to('kpc').v)
        row.append(Mstars_rvir.to('Msun').v)
        row.append(SFR)

        if (args.nproc != 1):
            queue.put(row)
        else:
            queue.add_row(row)

# --------------------------------------------------------------------------------------------------------------------
def KS_relation(ds, region, args):
    '''Plots the KS relation from the dataset as compared to a curve taken from Krumholz, McKee, & Tumlinson (2009), ApJ 699, 850.'''

    output_filename = args.save_directory + '/' + args.snap + '_KS-relation.png'

    if need_to_make_this_plot(output_filename, args):
        # Make a projection and convert to FRB
        p = yt.ProjectionPlot(ds, ds.z_unit_disk, 'density', data_source=region, width=(20, 'kpc'), center=ds.halo_center_code, north_vector=ds.x_unit_disk, buff_size=[500,500])
        proj_frb = p.frb
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
        plt.close()

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
def make_plots(snap, args, queue):
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
    print(args.data['snapshot'])
    if ('halo_info' in args.plot): get_halo_info(ds, region, args, queue)
    if ('density_projection' in args.plot): gas_density_projection(ds, region, args)
    if ('temperature_projection' in args.plot): edge_visualizations(ds, region, args)
    if ('young_stars_projection' in args.plot): young_stars_density_projection(ds, region, args)
    if ('KS_relation' in args.plot): KS_relation(ds, region, args)
    if ('outflow_rates' in args.plot): outflow_rates(ds, region, args)

# --------------------------------------------------------------------------------------------------------------------
def make_table():
    data_names = ['snapshot','time','redshift','halo_x','halo_y','halo_z','halo_vx','halo_vy','halo_vz','halo_mass','halo_radius','stellar_mass','SFR']
    data_types = ['S6','f8','f8','f8','f8','f8','f8','f8','f8','f8','f8','f8','f8']
    data_units = ['none','Myr','none','kpc','kpc','kpc','km/s','km/s','km/s','Msun','kpc','Msun','Msun/yr']
    data = Table(names=data_names, dtype=data_types)
    for i in range(len(data.keys())):
        key = data.keys()[i]
        data[key].unit = data_units[i]
    return data

# --------------------------------------------------------------------------------------------------------------------
def plot_SFMS(args):
    '''Plots the star-forming main sequence of the simulated galaxy -- one point per output on the plot -- and compares
    to best fit curves from observations at different redshifts.'''

    output_filename = args.save_directory + '/SFMS.png'

    # Read in the previously-saved halo information
    data = Table.read(args.save_directory + '/halo_data.txt', format='ascii.fixed_width')

    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(1,1,1)

    colormap = plt.cm.rainbow
    normalize = matplotlib.colors.Normalize(vmin=0., vmax=13.759)

    # Plot the observed best-fit values for a handful of redshifts
    Mstar_list = np.arange(8.,12.5,0.5)
    tlist = [13.759, 10.788, 8.657, 5.925, 3.330, 1.566]
    zlist = [0., 0.25, 0.5, 1., 2., 4.]
    SFR_list = []
    for i in range(len(tlist)):
        SFR_list.append([])
        for j in range(len(Mstar_list)):
            SFR_list[i].append((0.84-0.026*tlist[i])*Mstar_list[j] - (6.51-0.11*tlist[i]))  # This equation comes from Speagle et al. (2014), first row of Table 9
        ax.plot(Mstar_list, SFR_list[i], '-', lw=1, color=colormap(normalize(tlist[i])), label='z=%.2f' % zlist[i])
        ax.fill_between(Mstar_list, np.array(SFR_list[i])-0.2, y2=np.array(SFR_list[i])+0.2, color=colormap(normalize(tlist[i])), alpha=0.2)

    # Plot the simulated galaxy's location in stellar mass-SFR space as a scatterplot color-coded by redshift
    ax.scatter(np.log10(data['stellar_mass']), np.log10(data['SFR']), c=data['time'], cmap=colormap, norm=normalize, s=20)

    ax.set_xlabel('$\log M_\star$ [$M_\odot$]', fontsize=16)
    ax.set_ylabel('$\log$ SFR [$M_\odot$/yr]', fontsize=16)
    ax.tick_params(axis='both', which='both', direction='in', length=8, width=2, pad=5, labelsize=14, top=True, right=True)
    ax.legend(loc=2, frameon=False, fontsize=14)
    plt.tight_layout()
    plt.savefig(output_filename, dpi=300)
    plt.close()

# --------------------------------------------------------------------------------------------------------------------
def plot_SMHM(args):
    '''Plots the stellar mass-halo mass relation of the simulated galaxy -- one point per output on the plot -- and compares
    to best fit curves from observations at different redshifts.'''

    output_filename = args.save_directory + '/SMHM.png'

    # Read in the previously-saved halo information
    data = Table.read(args.save_directory + '/halo_data.txt', format='ascii.fixed_width')

    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(1,1,1)

    colormap = plt.cm.rainbow
    normalize = matplotlib.colors.LogNorm(vmin=0.5, vmax=13.759)

    # Plot the observed best-fit values for a handful of redshifts
    tlist = [12.447, 5.925, 3.330, 2.182, 1.566, 1.193, 0.947]
    zlist = [0.1, 1., 2., 3., 4., 5., 6.]
    # These values come from digitizing Figure 7 of Behroozi et al. (2013)
    Mhalo_list = [[10.010, 10.379, 10.746, 11.131, 11.500, 11.874, 12.239, 12.627, 13.002, 13.380, 13.751, 14.122, 14.510, 14.876],
                  [10.876, 11.257, 11.675, 12.000, 12.374, 12.750, 13.129, 13.504, 13.875, 14.251, 14.633],
                  [11.266, 11.637, 11.998, 12.375, 12.762, 13.124, 13.505, 13.875],
                  [11.510, 11.875, 12.247, 12.624, 13.002, 13.386],
                  [11.376, 11.753, 12.119, 12.503, 12.875, 13.251],
                  [11.256, 11.634, 12.006, 12.374, 12.757],
                  [11.136, 11.504, 11.881, 12.251]]
    Mstar_list = [[7.246, 7.750, 8.291, 8.901, 9.697, 10.275, 10.633, 10.821, 10.937, 11.078, 11.152, 11.224, 11.290, 11.372],
                  [8.306, 8.918, 9.882, 10.420, 10.774, 10.955, 11.045, 11.114, 11.151, 11.205, 11.229],
                  [8.780, 9.569, 10.364, 10.789, 10.960, 11.055, 11.102, 11.136],
                  [9.349, 10.091, 10.626, 10.862, 10.970, 11.040],
                  [9.240, 9.981, 10.478, 10.704, 10.822, 10.886],
                  [9.213, 9.904, 10.323, 10.515, 10.644],
                  [9.190, 9.799, 10.161, 10.317]]
    for i in range(len(tlist)):
        ax.plot(Mhalo_list[i], Mstar_list[i], '-', lw=1, color=colormap(normalize(tlist[i])), label='z=%.2f' % zlist[i])
        ax.fill_between(Mhalo_list[i], np.array(Mstar_list[i])-0.15, y2=np.array(Mstar_list[i])+0.15, color=colormap(normalize(tlist[i])), alpha=0.2)

    # Plot the simulated galaxy's location in stellar mass-SFR space as a scatterplot color-coded by redshift
    ax.scatter(np.log10(data['halo_mass']), np.log10(data['stellar_mass']), c=data['time'], cmap=colormap, norm=normalize, s=20)

    ax.set_xlabel('$\log M_\mathrm{halo}$ [$M_\odot$]', fontsize=16)
    ax.set_ylabel('$\log$ M_\star [$M_\odot$]', fontsize=16)
    ax.tick_params(axis='both', which='both', direction='in', length=8, width=2, pad=5, labelsize=14, top=True, right=True)
    ax.legend(loc=2, frameon=False, fontsize=14)
    plt.tight_layout()
    plt.savefig(output_filename, dpi=300)
    plt.close()


# --------------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    cli_args = parse_args()

    # ------------------ Figure out directory and outputs -------------------------------------
    if cli_args.save_directory is None:
        cli_args.save_directory = cli_args.directory + '/plots'
        Path(cli_args.save_directory).mkdir(parents=True, exist_ok=True)
    else:
        # In case users save to their home directory using "~"
        cli_args.save_directory = os.path.expanduser(cli_args.save_directory)

    if cli_args.trackfile is None:
        if cli_args.system is None:
            sys.exit('You must provide either the path to the track file or the name of the system you are on!')
        _, _, _, _, cli_args.trackfile, _, _, _ = get_run_loc_etc(cli_args) # for FOGGIE production runs it knows which trackfile to grab

    if cli_args.projection is not None:
        if ',' in cli_args.projection:
            projections = cli_args.projection.split(',')
        else:
            projections = [cli_args.projection]
    else:
        projections = ['x','z']

    if cli_args.output is not None: # Running on specific output/s
        outputs = make_output_list(cli_args.output)
    else: # Running on all snapshots in the directory
        outputs = []
        for fname in os.listdir(cli_args.directory):
            folder_path = os.path.join(cli_args.directory, fname)
            if os.path.isdir(folder_path) and ((fname[0:2]=='DD') or (fname[0:2]=='RD')):
                outputs.append(fname)

    if ('halo_info' in cli_args.plot):
        if (os.path.exists(cli_args.save_directory + '/halo_data.txt')):
            data = Table.read(cli_args.save_directory + '/halo_data.txt', format='ascii.fixed_width')
        else:
            data = make_table()
        cli_args.data = data

    # --------- Loop over outputs, for either single-processor or parallel processor computing ---------------
    if (cli_args.nproc == 1):
        for snap in outputs:
            queue = []
            make_plots(snap, cli_args, queue)
            if ('halo_info' in cli_args.plot):
                data.sort('time')
                data.write(cli_args.save_directory + '/halo_data.txt', format='ascii.fixed_width', overwrite=True)
    else:
        # Split into a number of groupings equal to the number of processors
        # and run one process per processor
        for i in range(len(outputs)//cli_args.nproc):
            queue = multi.Queue()
            rows = []
            threads = []
            for j in range(cli_args.nproc):
                snap = outputs[cli_args.nproc*i+j]
                threads.append(multi.Process(target=make_plots, args=[snap, cli_args, queue]))
            for t in threads:
                t.start()
            for t in threads:
                row = queue.get()
                rows.append(row)
            for t in threads:
                t.join()
            # Append halo data to file
            if ('halo_info' in cli_args.plot):
                for row in rows:
                    data.add_row(row)
                data.sort('time')
                data.write(cli_args.save_directory + '/halo_data.txt', format='ascii.fixed_width', overwrite=True)
        # For any leftover snapshots, run one per processor
        queue = multi.Queue()
        rows = []
        threads = []
        for j in range(len(outputs)%cli_args.nproc):
            snap = outputs[-(j+1)]
            threads.append(multi.Process(target=make_plots, args=[snap, cli_args, queue]))
        for t in threads:
            t.start()
        for t in threads:
            row = queue.get()
            rows.append(row)
        for t in threads:
            t.join()
        # Append halo data to file
        if ('halo_info' in cli_args.plot):
            for row in rows:
                data.add_row(row)
            data.sort('time')
            data.write(cli_args.save_directory + '/halo_data.txt', format='ascii.fixed_width', overwrite=True)

    if ('halo_info' in cli_args.plot):
        plot_SFMS(cli_args)
        plot_SMHM(cli_args)