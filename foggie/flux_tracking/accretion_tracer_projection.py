'''
Filename: accretion_tracer_projection.py
Author: Cassi
Created: 4-18-25

This script saves to file images of the projected tracer fields.'''

import numpy as np
import yt
import unyt
from yt import YTArray
import argparse
import os
from astropy.table import Table
from astropy.io import ascii
import multiprocessing as multi
import time
from datetime import timedelta
from scipy import interpolate
import shutil
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import cmasher as cmr
import matplotlib.colors as mcolors
from mpl_toolkits.axes_grid1 import make_axes_locatable
import h5py

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
    '''Parse command line arguments. Returns args object.'''

    parser = argparse.ArgumentParser(description='Makes projection plots of tracer fields.')

    # Optional arguments:
    parser.add_argument('--sim_dir', metavar='sim_dir', type=str, action='store', \
                        help='Where are the outputs stored?')
    parser.set_defaults(sim_dir='/nobackup/clochhaa/accretion_tracer/')

    parser.add_argument('--halo', metavar='halo', type=str, action='store', \
                        help='Which halo? Default is 8508 (Tempest)')
    parser.set_defaults(halo='8508')

    parser.add_argument('--run', metavar='run', type=str, action='store', \
                        help='Which run? Default is nref11c_nref9f')
    parser.set_defaults(run='nref11c_nref9f')

    parser.add_argument('--output', metavar='output', type=str, action='store', \
                        help='Which output(s)? Options: Specify a single output (this is default' \
                        + ' and the default output is DD2427) or specify a range of outputs ' + \
                        '(e.g. "RD0020,RD0025" or "DD1340-DD2029").')
    parser.set_defaults(output='DD2427')

    parser.add_argument('--output_step', metavar='output_step', type=int, action='store', \
                        help='If you want to do every Nth output, this specifies N. Default: 1 (every output in specified range)')
    parser.set_defaults(output_step=1)

    parser.add_argument('--system', metavar='system', type=str, action='store', \
                        help='Which system are you on? Default is cassiopeia')
    parser.set_defaults(system='cassiopeia')

    parser.add_argument('--pwd', dest='pwd', action='store_true',
                        help='Just use the working directory?, Default is no')
    parser.set_defaults(pwd=False)

    parser.add_argument('--nproc', metavar='nproc', type=int, action='store', \
                        help='How many processes do you want? Default is 1 ' + \
                        '(no parallelization), if multiple outputs and multiple processors are' + \
                        ' specified, code will run one output per processor')
    parser.set_defaults(nproc=1)

    parser.add_argument('--proj', metavar='proj', type=str, action='store', \
                        help='What axis do you want for the projection direction?\n' + \
                            'Options are: x, y, z, x-disk, y-disk, z-disk. Default is x.\n' + \
                            'Specify multiple with commas (no space) between them, like: x,y,z')
    parser.set_defaults(proj='x')
    
    parser.add_argument('--save_suffix', metavar='save_suffix', type=str, action='store', \
                        help='Do you want to append a string onto the names of the saved files? Default is no.')
    parser.set_defaults(save_suffix="")

    args = parser.parse_args()
    return args

def tracer_density1(field, data):
    return data.ds.arr(data['enzo','TracerFluid01'], 'code_density')

def tracer_density2(field, data):
    return data.ds.arr(data['enzo','TracerFluid02'], 'code_density')

def tracer_density3(field, data):
    return data.ds.arr(data['enzo','TracerFluid03'], 'code_density')

def tracer_density4(field, data):
    return data.ds.arr(data['enzo','TracerFluid04'], 'code_density')

def tracer_density5(field, data):
    return data.ds.arr(data['enzo','TracerFluid05'], 'code_density')

def tracer_density6(field, data):
    return data.ds.arr(data['enzo','TracerFluid06'], 'code_density')

def project_tracer(ds, snap, tracer_number, proj_direction):
    '''Makes the projected images in proj_direction of the tracer field given by tracer_number for the snapshot snap.'''

    tracer_name = 'TracerFluid0' + str(tracer_number)
    tracden_name = 'tracer_density0' + str(tracer_number)

    if ('disk' in proj_direction):
        proj_dict = {'x-disk':ds.x_unit_disk, 'y-disk':ds.y_unit_disk, 'z-disk':ds.z_unit_disk}
        north_dict = {'x-disk':ds.z_unit_disk, 'y-disk':ds.z_unit_disk, 'z-disk':ds.y_unit_disk}
    else:
        proj_dict = {'x':'x', 'y':'y', 'z':'z'}
    xlabel_dict = {'x':'y', 'y':'x', 'z':'x', 'x-disk':'y', 'y-disk':'x', 'z-disk':'x'}
    ylabel_dict = {'x':'z', 'y':'z', 'z':'y', 'x-disk':'z', 'y-disk':'z', 'z-disk':'y'}

    zsnap = ds.get_parameter('CosmologyCurrentRedshift')
    fig = plt.figure(figsize=(14,5.5), dpi=250)
    ax1 = fig.add_subplot(1,2,1)
    ax2 = fig.add_subplot(1,2,2)
    #ax3 = fig.add_subplot(1,3,3)
    fig.subplots_adjust(left=0.07, bottom=0.1, top=0.98, right=0.92, wspace=0.43)

    # Plot gas density
    if ('disk' in proj_direction):
        den_proj = yt.ProjectionPlot(ds, proj_dict[proj_direction], ('gas','density'), weight_field=('gas','density'), center=ds.halo_center_kpc, width=(400, 'kpc'), north_vector=north_dict[proj_direction])
    else:
        den_proj = yt.ProjectionPlot(ds, proj_dict[proj_direction], ('gas','density'), weight_field=('gas','density'), center=ds.halo_center_kpc, width=(400, 'kpc'))
    den_frb = den_proj.frb[('gas','density')]
    den_im = ax1.imshow(den_frb, extent=[-200,200,-200,200], cmap='viridis', norm=mcolors.LogNorm(vmin=1e-30, vmax=1e-24), origin='lower')
    ax1.set_xlabel(xlabel_dict[proj_direction] + ' [kpc]', fontsize=16)
    ax1.set_ylabel(ylabel_dict[proj_direction] + ' [kpc]', fontsize=16)
    ax1.tick_params(axis='both', which='both', direction='in', length=8, width=2, pad=5, labelsize=14, \
            top=True, right=True)
    ax1.text(190, 190, '$z = %.2f$\n%.2f Gyr' % (zsnap, ds.current_time.in_units('Gyr')), fontsize=16, ha='right', va='top', color='white')
    pos = ax1.get_position()
    den_cax = fig.add_axes([pos.x1, pos.y0, 0.015, pos.height])  # [left, bottom, width, height]
    fig.colorbar(den_im, cax=den_cax, orientation='vertical')
    den_cax.tick_params(axis='both', which='both', top=False, right=True, labelsize=14, direction='in', length=8, width=2, pad=5)
    pos_cax = den_cax.get_position()
    den_cax.text(pos_cax.x1 + 4.5, pos_cax.y0 + pos_cax.height/2.-0.02, 'Projected Density [g/cm$^3$]', fontsize=16, ha='center', va='center', rotation=90, transform=den_cax.transAxes)

    # Plot tracer field
    if ('disk' in proj_direction):
        tracer_proj = yt.ProjectionPlot(ds, proj_dict[proj_direction], ('gas', tracden_name), weight_field=('gas', tracden_name), center=ds.halo_center_kpc, width=(400, 'kpc'), north_vector=north_dict[proj_direction])
    else:
        tracer_proj = yt.ProjectionPlot(ds, proj_dict[proj_direction], ('gas', tracden_name), weight_field=('gas', tracden_name), center=ds.halo_center_kpc, width=(400, 'kpc'))
    tracer_frb = tracer_proj.frb[('gas', tracden_name)]
    tracer_im = ax2.imshow(tracer_frb, extent=[-200,200,-200,200], cmap='magma', norm=mcolors.LogNorm(vmin=1e-30, vmax=1e-24), origin='lower')
    ax2.set_xlabel(xlabel_dict[proj_direction] + ' [kpc]', fontsize=16)
    ax2.set_ylabel(ylabel_dict[proj_direction] + ' [kpc]', fontsize=16)
    ax2.tick_params(axis='both', which='both', direction='in', length=8, width=2, pad=5, labelsize=14, \
            top=True, right=True)
    pos = ax2.get_position()
    tracer_cax = fig.add_axes([pos.x1, pos.y0, 0.015, pos.height])  # [left, bottom, width, height]
    fig.colorbar(tracer_im, cax=tracer_cax, orientation='vertical')
    tracer_cax.tick_params(axis='both', which='both', top=False, right=True, labelsize=14, direction='in', length=8, width=2, pad=5)
    pos_cax = tracer_cax.get_position()
    tracer_cax.text(pos_cax.x1 + 4, pos_cax.y0 + pos_cax.height/2.-0.02, 'Projected Tracer [g/cm$^3$]', fontsize=16, ha='center', va='center', rotation=90, transform=tracer_cax.transAxes)

    plt.savefig(output_dir + '/' + snap + '_tracer0' + str(tracer_number) + '_projection_' + proj_direction + save_suffix + '.png')
    plt.close()

    # Plot tracer field overlaid on gas density
    fig2 = plt.figure(figsize=(8.5,6), dpi=250)
    ax3 = fig2.add_subplot(1,1,1)
    fig2.subplots_adjust(left=0.08, bottom=0.02, top=0.98, right=0.92)

    den_im = ax3.imshow(den_frb, extent=[-200,200,-200,200], cmap='viridis', norm=mcolors.LogNorm(vmin=1e-30, vmax=1e-24), origin='lower')
    alpha_tracer = np.log10(tracer_frb)
    alpha_tracer[alpha_tracer<-30.] = -30.
    alpha_tracer[alpha_tracer>-27.] = -27.
    alpha_tracer = (alpha_tracer+30.)/3.        # log'd and then adding 30 makes values go from 0 to 3, dividing by 3 makes it 0 to 1, then multiplying by 0.8 makes it 0 to 0.8
    tracer_im = ax3.imshow(tracer_frb, extent=[-200,200,-200,200], cmap='magma', norm=mcolors.LogNorm(vmin=1e-30, vmax=1e-27), origin='lower', alpha=alpha_tracer)
    ax3.tick_params(axis='both', which='both', direction='in', length=8, width=2, pad=5, labelsize=14, \
            top=True, right=True, labelleft=False, labelbottom=False)
    pos = ax3.get_position()
    den_cax = fig2.add_axes([pos.x1, pos.y0, 0.03, pos.height])  # [left, bottom, width, height]
    fig2.colorbar(den_im, cax=den_cax, orientation='vertical')
    den_cax.tick_params(axis='both', which='both', top=False, right=True, labelsize=14, direction='in', length=8, width=2, pad=5)
    pos_cax = den_cax.get_position()
    den_cax.text(pos_cax.x1 + 3.5, pos_cax.y0 + pos_cax.height/2.-0.02, 'Projected Density [g/cm$^3$]', fontsize=16, ha='center', va='center', rotation=90, transform=den_cax.transAxes)
    tracer_cax = fig2.add_axes([pos.x0-0.03, pos.y0, 0.03, pos.height])  # [left, bottom, width, height]
    fig2.colorbar(tracer_im, cax=tracer_cax, orientation='vertical')
    tracer_cax.tick_params(axis='both', which='both', top=False, right=False, left=True, labelleft=True, labelright=False, labelsize=14, direction='in', length=8, width=2, pad=5)
    pos_cax = tracer_cax.get_position()
    tracer_cax.text(pos_cax.x1 - 3.5, pos_cax.y0 + pos_cax.height/2.-0.02, 'Projected Tracer [g/cm$^3$]', fontsize=16, ha='center', va='center', rotation=90, transform=tracer_cax.transAxes)
    ax3.text(190, 190, '$z = %.2f$\n%.2f Gyr' % (zsnap, ds.current_time.in_units('Gyr')), fontsize=16, ha='right', va='top', color='white')
    ax3.plot([-150,-100],[-180,-180], color='white', ls='-', lw=2)
    ax3.text(-125, -177, '50 kpc', fontsize=16, ha='center', va='bottom', color='white')

    plt.savefig(output_dir + '/' + snap + '_tracer0' + str(tracer_number) + '-overlay_projection_' + proj_direction + save_suffix + '.png')
    plt.close()


def load_and_calculate(snap):
    '''Loads the simulation snapshot and makes the requested plots.'''

    # Load simulation output
    if (args.system=='pleiades_cassi'):
        print('Copying directory to /tmp')
        # Make a dummy directory with the snap name so the script later knows the process running
        # this snapshot failed if the directory is still there
        snap_dir = '/nobackup/clochhaa/tmp/' + args.halo + '/' + args.run + '/' + target_dir + '/' + snap
        os.makedirs(snap_dir)
        snap_name = run_dir + snap + '/' + snap
    else:
        snap_name = run_dir + snap + '/' + snap
    
    ds, refine_box = foggie_load(snap_name, trackfile_name=trackname, do_filter_particles=True, disk_relative=disk_needed)
    ds.add_field(('gas','tracer_density01'), function=tracer_density1, units='g/cm**3', take_log=True, \
                 sampling_type='cell')
    ds.add_field(('gas','tracer_density02'), function=tracer_density2, units='g/cm**3', take_log=True, \
                 sampling_type='cell')
    ds.add_field(('gas','tracer_density03'), function=tracer_density3, units='g/cm**3', take_log=True, \
                 sampling_type='cell')
    ds.add_field(('gas','tracer_density04'), function=tracer_density4, units='g/cm**3', take_log=True, \
                 sampling_type='cell')
    ds.add_field(('gas','tracer_density05'), function=tracer_density5, units='g/cm**3', take_log=True, \
                 sampling_type='cell')
    #ds.add_field(('gas','tracer_density06'), function=tracer_density6, units='g/cm**3', take_log=True, \
                 #sampling_type='cell')
    
    for p in range(len(projections)):
        for i in range(5):
            project_tracer(ds, snap, i+1, projections[p])

    # Delete output from temp directory if on pleiades
    if (args.system=='pleiades_cassi'):
        print('Deleting directory from /tmp')
        shutil.rmtree(snap_dir)

if __name__ == "__main__":

    start = time.perf_counter()

    args = parse_args()
    print(args.halo)
    print(args.run)
    print(args.system)
    foggie_dir, output_dir, run_dir, code_path, trackname, haloname, spectra_dir, infofile = get_run_loc_etc(args)

    foggie_dir = args.sim_dir
    run_dir = foggie_dir + 'halo_00' + args.halo + '/' + args.run + '/'

    # Set directory for output location, making it if necessary
    output_dir = foggie_dir + 'halo_00' + args.halo + '/' + args.run + '/Projections'
    if not (os.path.exists(output_dir)): os.system('mkdir -p ' + output_dir)

    print('foggie_dir: ', foggie_dir)

    if (args.save_suffix!=''):
        save_suffix = '_' + args.save_suffix
    else:
        save_suffix = ''

    if (',' in args.proj):
        projections = args.proj.split(',')
    else:
        projections = [args.proj]
    
    disk_needed = False
    for i in range(len(projections)):
        if ('disk' in projections[i]): disk_needed = True

    # Build outputs list
    outs = make_output_list(args.output, output_step=args.output_step)
    target_dir = 'projs'
    if (args.nproc==1):
        for snap in outs:
            load_and_calculate(snap)
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
                    threads.append(multi.Process(target=load_and_calculate, args=[snap]))
                for t in threads:
                    t.start()
                for t in threads:
                    t.join()
                # Delete leftover outputs from failed processes from tmp directory if on pleiades
                if (args.system=='pleiades_cassi'):
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
                threads.append(multi.Process(target=load_and_calculate, args=[snap]))
            for t in threads:
                t.start()
            for t in threads:
                t.join()
            # Delete leftover outputs from failed processes from tmp directory if on pleiades
            if (args.system=='pleiades_cassi'):
                snap_dir = '/nobackup/clochhaa/tmp/' + args.halo + '/' + args.run + '/' + target_dir + '/'
                for s in range(len(snaps)):
                    if (os.path.exists(snap_dir + snaps[s])):
                        print('Deleting failed %s from /tmp' % (snaps[s]))
                        skipped_outs.append(snaps[s])
                        shutil.rmtree(snap_dir + snaps[s])
            outs = skipped_outs

    end = time.perf_counter()
    elapsed = end - start
    duration = timedelta(seconds=elapsed)
    print("All snapshots finished!")
    print(f"Elapsed time: {duration}")