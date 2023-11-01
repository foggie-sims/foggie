'''
Filename: OVI_emission_SFR.py
Author: Cassi
Date created: 9-28-23

This file contains everything needed for investigations into O VI emission dependence on star formation rate.
All CLOUDY and emission code copy-pasted from Lauren's foggie/emission/emission_functions.py and simplied for
just O VI.'''

import numpy as np
import yt
import unyt
from yt import YTArray
import argparse
import os
from astropy.table import Table
from astropy.io import ascii
import multiprocessing as multi
import datetime
from scipy import interpolate
import shutil
import matplotlib.pyplot as plt
import cmasher as cmr

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

    parser = argparse.ArgumentParser(description='Makes plots of various O VI related analysis.')

    # Optional arguments:
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

    parser.add_argument('--plot', metavar='plot', type=str, action='store', \
                       help='What do you want to plot? Options are emission_map, sb_profile,\n' + \
                       'or sb_time_hist and you can select multiple options separated by commas (no spaces!)')
    parser.set_defaults(plot='emission_map,sb_profile,sb_time_hist')
    
    parser.add_argument('--save_suffix', metavar='save_suffix', type=str, action='store', \
                        help='Do you want to append a string onto the names of the saved files? Default is no.')
    parser.set_defaults(save_suffix="")

    args = parser.parse_args()
    return args

# The Cloudy calculations assumed a single metallicity (typically solar).
# This function scales the emission by the metallicity of the gas itself to
# account for this discrepancy.
def scale_by_metallicity(values,assumed_Z,wanted_Z):
        wanted_ratio = (10.**(wanted_Z))/(10.**(assumed_Z))
        return values*wanted_ratio

# This function takes all of the Cloudy files and compiles them into one table
# for use in the emission functions
# table_index is the column in the Cloudy output files that is being read.
# each table_index value corresponds to a different emission line
def make_Cloudy_table(table_index):
        # this is the the range and number of bins for which Cloudy was run
        # i.e. the temperature and hydrogen number densities gridded in the
        # Cloudy run. They must match or the table will be incorrect.
        hden_n_bins, hden_min, hden_max = 15, -5, 2 #17, -6, 2 #23, -9, 2
        T_n_bins, T_min, T_max = 51, 3, 8 #71, 2, 8

        hden=np.linspace(hden_min,hden_max,hden_n_bins)
        T=np.linspace(T_min,T_max, T_n_bins)
        table = np.zeros((hden_n_bins,T_n_bins))
        for i in range(hden_n_bins):
                table[i,:]=[float(l.split()[table_index]) for l in open(cloudy_path%(i+1)) if l[0] != "#"]
        return hden,T,table

def _Emission_OVI(field,data):
        H_N=np.log10(np.array(data["H_nuclei_density"]))
        Temperature=np.log10(np.array(data["Temperature"]))
        dia1 = bl_OVI_1(H_N,Temperature)
        dia2 = bl_OVI_2(H_N,Temperature)
        idx = np.isnan(dia1)
        dia1[idx] = -200.
        dia2[idx] = -200.
        emission_line=((10.0**dia1)+(10**dia2))*((10.0**H_N)**2.0)
        emission_line = emission_line/(4.*np.pi*1.92e-11)
        emission_line = scale_by_metallicity(emission_line,0.0,np.log10(np.array(data['metallicity'])))
        return emission_line * ytEmUALT

def make_pdf_table():
    '''Makes the giant table of O VI surface brightness histograms that will be saved to file.'''

    names_list = ['inner_radius', 'outer_radius', 'lower_SB', 'upper_SB', 'all', 'inflow', 'outflow', 'major', 'minor']
    types_list = ['f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8']

    table = Table(names=names_list, dtype=types_list)

    return table

def make_sb_profile_table():
    '''Makes the giant table of O VI surface brightness profiles that will be saved to file.'''

    names_list = ['inner_radius', 'outer_radius', 'all', 'inflow', 'outflow', 'major', 'minor']
    types_list = ['f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8']

    table = Table(names=names_list, dtype=types_list)

    return table

def set_table_units(table, pdf=False):
    '''Sets the units for the table. Note this needs to be updated whenever something is added to
    the table. Returns the table.'''

    for key in table.keys():
        if ('radius' in key):
            table[key].unit = 'kpc'
        elif ('SB' in key):
            table[key].unit = 'erg/s/cm^2/arcsec^2'
        else:
            if (pdf):
                table[key].unit = 'none'
            else:
                table[key].unit = 'erg/s/cm^2/arcsec^2'
    return table

def surface_brightness_profile(ds, refine_box, snap):
    '''Makes radial surface brightness profiles of O VI emission for full image, along major and minor axes, and for inflows and outflows separately.'''

    profile_table = make_sb_profile_table()
    pdf_table = make_pdf_table()

    sph = ds.sphere(center=ds.halo_center_kpc, radius=(75., 'kpc'))
    sph_inflow = sph.include_below(('gas','radial_velocity_corrected'), -100.)
    sph_outflow = sph.include_above(('gas','radial_velocity_corrected'), 200.)

    proj = yt.ProjectionPlot(ds, ds.x_unit_disk, ('gas','Emission_OVI'), center=ds.halo_center_kpc, data_source=sph, width=(100., 'kpc'), north_vector=ds.z_unit_disk)
    FRB = proj.frb[('gas','Emission_OVI')].v
    proj_in = yt.ProjectionPlot(ds, ds.x_unit_disk, ('gas','Emission_OVI'), center=ds.halo_center_kpc, data_source=sph_inflow, width=(100., 'kpc'), north_vector=ds.z_unit_disk)
    FRB_in = proj_in.frb[('gas','Emission_OVI')].v
    proj_out = yt.ProjectionPlot(ds, ds.x_unit_disk, ('gas','Emission_OVI'), center=ds.halo_center_kpc, data_source=sph_outflow, width=(100., 'kpc'), north_vector=ds.z_unit_disk)
    FRB_out = proj_out.frb[('gas','Emission_OVI')].v

    dx = 100./800.              # physical width of FRB (kpc) divided by resolution
    FRB_x = np.indices((800,800))[0]
    FRB_y = np.indices((800,800))[1]
    FRB_x = FRB_x*dx - 50.
    FRB_y = FRB_y*dx - 50.
    radius = np.sqrt(FRB_x**2. + FRB_y**2.)

    minor = ((FRB_x > -10.) & (FRB_x < 10.))
    major = ((FRB_y < 10.) & (FRB_y > -10.))

    profile_row = [0., 50., np.log10(np.mean(FRB[(radius < 50.)])), np.log10(np.mean(FRB_in[(radius<50.)])), np.log10(np.mean(FRB_out[(radius<50.)])), np.log10(np.mean(FRB[major])), np.log10(np.mean(FRB[minor]))]
    profile_table.add_row(profile_row)

    SB_bins = np.linspace(-1,5,61)
    SB_hist, bins = np.histogram(np.log10(FRB[radius<50.]), bins=SB_bins)
    SB_hist_in, bins = np.histogram(np.log10(FRB_in[radius<50.]), bins=SB_bins)
    SB_hist_out, bins = np.histogram(np.log10(FRB_out[radius<50.]), bins=SB_bins)
    SB_hist_major, bins = np.histogram(np.log10(FRB[(radius<50.) & (major)]), bins=SB_bins)
    SB_hist_minor, bins = np.histogram(np.log10(FRB[(radius<50.) & (minor)]), bins=SB_bins)

    for i in range(len(bins)-1):
        pdf_row = [0., 50., bins[i], bins[i+1], SB_hist[i], SB_hist_in[i], SB_hist_out[i], SB_hist_major[i], SB_hist_minor[i]]
        pdf_table.add_row(pdf_row)

    rbins = np.linspace(0., 50., 26)
    rbin_centers = rbins[:-1] + np.diff(rbins)
    full_profile = []
    major_profile = []
    minor_profile = []
    inflow_profile = []
    outflow_profile = []
    for r in range(len(rbins)-1):
        r_low = rbins[r]
        r_upp = rbins[r+1]
        shell = (radius > r_low) & (radius < r_upp)
        full_profile.append(np.mean(FRB[shell]))
        major_profile.append(np.mean(FRB[shell & major]))
        minor_profile.append(np.mean(FRB[shell & minor]))
        inflow_profile.append(np.mean(FRB_in[shell]))
        outflow_profile.append(np.mean(FRB_out[shell]))
        profile_row = [r_low, r_upp, np.log10(full_profile[-1]), np.log10(inflow_profile[-1]), np.log10(outflow_profile[-1]), np.log10(major_profile[-1]), np.log10(minor_profile[-1])]
        profile_table.add_row(profile_row)

        SB_hist, bins = np.histogram(np.log10(FRB[shell]), bins=SB_bins)
        SB_hist_in, bins = np.histogram(np.log10(FRB_in[shell]), bins=SB_bins)
        SB_hist_out, bins = np.histogram(np.log10(FRB_out[shell]), bins=SB_bins)
        SB_hist_major, bins = np.histogram(np.log10(FRB[shell & major]), bins=SB_bins)
        SB_hist_minor, bins = np.histogram(np.log10(FRB[shell & minor]), bins=SB_bins)
        for i in range(len(bins)-1):
            pdf_row = [r_low, r_upp, bins[i], bins[i+1], SB_hist[i], SB_hist_in[i], SB_hist_out[i], SB_hist_major[i], SB_hist_minor[i]]
            pdf_table.add_row(pdf_row)

    profile_table = set_table_units(profile_table)
    pdf_table = set_table_units(pdf_table, pdf=True)
    profile_table.write(prefix + 'Tables/' + snap + '_SB_profiles' + save_suffix + '.hdf5', path='all_data', serialize_meta=True, overwrite=True)
    pdf_table.write(prefix + 'Tables/' + snap + '_SB_pdf' + save_suffix + '.hdf5', path='all_data', serialize_meta=True, overwrite=True)

    full_profile = np.log10(np.array(full_profile))
    major_profile = np.log10(np.array(major_profile))
    minor_profile = np.log10(np.array(minor_profile))
    inflow_profile = np.log10(np.array(inflow_profile))
    outflow_profile = np.log10(np.array(outflow_profile))

    fig = plt.figure(figsize=(10,4), dpi=300)
    ax1 = fig.add_subplot(1,2,1)
    ax2 = fig.add_subplot(1,2,2)

    ax1.plot(rbin_centers, full_profile, 'k-', lw=2, label='Full profile')
    ax1.plot(rbin_centers, major_profile, 'b--', lw=2, label='Major axis')
    ax1.plot(rbin_centers, minor_profile, 'r:', lw=2, label='Minor axis')
    ax1.axis([0,50,-1,5])
    ax1.set_xlabel('Radius [kpc]', fontsize=12)
    ax1.set_ylabel('log O VI SB [erg s$^{-1}$ cm$^{-2}$ arcsec$^{-2}$]', fontsize=12)
    ax1.tick_params(axis='both', which='both', direction='in', length=8, width=2, pad=5, labelsize=10, \
                top=True, right=True)
    ax1.legend(loc=1, fontsize=12, frameon=False)
    ax2.plot(rbin_centers, full_profile, 'k-', lw=2, label='Full profile')
    ax2.plot(rbin_centers, inflow_profile, color="#984ea3", ls='--', lw=2, label='Inflowing gas')
    ax2.plot(rbin_centers, outflow_profile, color='darkorange', ls=':', lw=2, label='Outflowing gas')
    ax2.axis([0,50,-1,5])
    ax2.set_xlabel('Radius [kpc]', fontsize=12)
    ax2.set_ylabel('log O VI SB [erg s$^{-1}$ cm$^{-2}$ arcsec$^{-2}$]', fontsize=12)
    ax2.tick_params(axis='both', which='both', direction='in', length=8, width=2, pad=5, labelsize=10, \
                top=True, right=True)
    ax2.legend(loc=1, fontsize=12, frameon=False)
    plt.subplots_adjust(left=0.06, bottom=0.12, top=0.96, right=0.98)
    plt.savefig(prefix + 'Profiles/' + snap + '_OVI_surface_brightness_profile_edge-on' + save_suffix + '.png')

def surface_brightness_time_histogram(outs):
    '''Makes a plot of surface brightness histograms vs time for the outputs in 'outs'. Requires
    surface brightness tables to have already been created for the outputs plotted using the
    surface_brightness_profile function.'''

    halo_c_v = Table.read(halo_c_v_name, format='ascii')
    timelist = halo_c_v['col4']
    snaplist = halo_c_v['col3']
    zlist = halo_c_v['col2']

    sb_hists = []
    time_hist = []
    max_weights = []
    sb_hists_sections = []
    max_weights_sections = []
    sections = ['inflow','outflow','major','minor']
    for j in range(len(sections)):
        sb_hists_sections.append([])
        max_weights_sections.append([])
    for i in range(len(outs)):
        snap = outs[i]
        time_hist.append(float(timelist[snaplist==snap])/1000.)
        sb_data = Table.read(table_loc + snap + '_SB_pdf' + save_suffix + '.hdf5', path='all_data')
        radial_range = (sb_data['inner_radius']==0.) & (sb_data['outer_radius']==50.)
        sb_hists.append(sb_data['all'][radial_range])
        max_weights.append(np.max(sb_data['all'][radial_range]))
        for j in range(len(sections)):
            sb_hists_sections[j].append(sb_data[sections[j]][radial_range])
            max_weights_sections[j].append(np.max(sb_data[sections[j]][radial_range]))

    sb_hists = np.array(sb_hists)
    sb_hists = np.transpose(sb_hists).flatten()
    for j in range(len(sections)):
        sb_hists_sections[j] = np.transpose(np.array(sb_hists_sections[j])).flatten()
    sb_bins_l = sb_data['lower_SB'][radial_range]
    sb_bins_u = sb_data['upper_SB'][radial_range]
    bin_edges = np.array(sb_bins_l)
    bin_edges = np.append(bin_edges, sb_bins_u[-1])
    time_bins = np.array(time_hist)
    time_bins = np.append(time_bins, time_hist[-1] + np.diff(time_hist)[-1])
    xdata = np.tile(time_bins[:-1], (len(bin_edges)-1, 1)).flatten()
    ydata = np.transpose(np.tile(bin_edges[:-1], (len(time_bins)-1, 1))).flatten()

    fig = plt.figure(figsize=(6,4), dpi=300)
    ax = fig.add_subplot(1,1,1)
    ax.hist2d(xdata, ydata, weights=sb_hists, bins=[time_bins[:-1],bin_edges], vmin=0., vmax=np.mean(max_weights), cmap=cmr.get_sub_cmap('cmr.flamingo', 0.2, 1.))
    ax.set_xlabel('Time [Gyr]', fontsize=12)
    ax.set_ylabel('log O VI SB [erg s$^{-1}$ cm$^{-2}$ arcsec$^{-2}$]', fontsize=12)
    plt.subplots_adjust(left=0.1, bottom=0.12, top=0.96, right=0.98)
    plt.savefig(prefix + 'OVI_SB_histogram_vs_time' + save_suffix + '.png')
    plt.close()

    section_labels = ['Inflow','Outflow','Major axis','Minor axis']
    for j in range(len(sections)):
        fig = plt.figure(figsize=(6,4), dpi=300)
        ax = fig.add_subplot(1,1,1)
        ax.hist2d(xdata, ydata, weights=sb_hists_sections[j], bins=[time_bins[:-1],bin_edges], vmin=0., vmax=np.mean(max_weights_sections[j]), cmap=cmr.get_sub_cmap('cmr.flamingo', 0.2, 1.))
        ax.set_xlabel('Time [Gyr]', fontsize=12)
        ax.set_ylabel('log O VI SB [erg s$^{-1}$ cm$^{-2}$ arcsec$^{-2}$]', fontsize=12)
        ax.set_title(section_labels[j], fontsize=12)
        plt.subplots_adjust(left=0.1, bottom=0.12, top=0.93, right=0.98)
        plt.savefig(prefix + 'OVI_SB_histogram_vs_time_' + sections[j] + save_suffix + '.png')
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
        snap_name = foggie_dir + run_dir + snap + '/' + snap
    else:
        snap_name = foggie_dir + run_dir + snap + '/' + snap
    
    ds, refine_box = foggie_load(snap_name, trackname, do_filter_particles=True, halo_c_v_name=halo_c_v_name, disk_relative=True, correct_bulk_velocity=True, smooth_AM_name=smooth_AM_name)
    zsnap = ds.get_parameter('CosmologyCurrentRedshift')

    if ('emission_map' in args.plot):
        proj = yt.ProjectionPlot(ds, ds.x_unit_disk, ('gas','Emission_OVI'), center=ds.halo_center_kpc, data_source=refine_box, width=(100., 'kpc'), north_vector=ds.z_unit_disk)
        proj.set_cmap('Emission_OVI', cmr.get_sub_cmap('cmr.flamingo', 0.2, 1.))
        proj.set_zlim('Emission_OVI', 1e-1, 1e5)
        proj.set_colorbar_label('Emission_OVI', 'O VI Emission [erg s$^{-1}$ cm$^{-2}$ arcsec$^{-2}$]')
        proj.set_font_size(20)
        proj.annotate_timestamp(corner='upper_left', redshift=True, time=True, draw_inset_box=True)
        proj.save(prefix + 'Projections/' + snap + '_OVI_emission_map_edge-on' + save_suffix + '.png')

    if ('sb_profile' in args.plot):
        surface_brightness_profile(ds, refine_box, snap)

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
    halo_c_v_name = code_path + 'halo_infos/00' + args.halo + '/' + args.run + '/halo_c_v'
    
    # Set directory for output location, making it if necessary
    prefix = output_dir + 'ions_halo_00' + args.halo + '/' + args.run + '/'
    if not (os.path.exists(prefix)): os.system('mkdir -p ' + prefix)
    table_loc = prefix + 'Tables/'

    print('foggie_dir: ', foggie_dir)
    catalog_dir = code_path + 'halo_infos/00' + args.halo + '/' + args.run + '/'
    halo_c_v_name = catalog_dir + 'halo_c_v'
    smooth_AM_name = catalog_dir + 'AM_direction_smoothed'

    cloudy_path = code_path + "emission/cloudy_z0_selfshield/sh_z0_HM12_run%i.dat"
    # These are the typical units that Lauren uses
    # NOTE: This is a volumetric unit since it's for the emissivity of each cell
    # Emission / surface brightness comes from the projections
    emission_units = 's**-1 * cm**-3 * steradian**-1'
    ytEmU = unyt.second**-1 * unyt.cm**-3 * unyt.steradian**-1

    # These are a second set of units that a lot of observers prefer
    # NOTE: This is a volumetric unit since it's for the emissivity of each cell
    # Emission / surface brightness comes from the projections
    emission_units_ALT = 'erg * s**-1 * cm**-3 * arcsec**-2'
    ytEmUALT = unyt.erg * unyt.second**-1 * unyt.cm**-3 * unyt.arcsec**-2

    ####################################
    ## BEGIN CREATING EMISSION FIELDS ##
    ####################################

    # To make the emissivity fields, you need to follow a number of steps
    # 1. Read in the Cloudy values for a given emission line
    # 2. Create the n_H and T grids that represent the desired range of values
    # 3. Set up interpolation function for the emissivity values across the grids
    #    so the code can use the n_H and T values of a simulation grid cell to
    #    interpolate the correct emissivity value
    # 4. Define the emission field for the line
    # 5. Add the line as a value in yt

    ############################
    ## OVI (2 lines combined) ##
    ############################
    # 1. Read cloudy file
    hden_pts,T_pts,table_HA = make_Cloudy_table(2)
    # 2. Create grids
    hden_pts,T_pts = np.meshgrid(hden_pts,T_pts)
    pts = np.array((hden_pts.ravel(),T_pts.ravel())).T
    # 3. Set up interpolation fundtion
    hden1, T1, table_OVI_1 = make_Cloudy_table(5)
    hden1, T1, table_OVI_2 = make_Cloudy_table(6)
    sr_OVI_1 = table_OVI_1.T.ravel()
    sr_OVI_2 = table_OVI_2.T.ravel()
    bl_OVI_1 = interpolate.LinearNDInterpolator(pts,sr_OVI_1)
    bl_OVI_2 = interpolate.LinearNDInterpolator(pts,sr_OVI_2)
    # 4. and 5. Define emission field and add it to yt
    yt.add_field(("gas","Emission_OVI"),units=emission_units_ALT,function=_Emission_OVI,take_log=True,force_override=True,sampling_type='cell')

    if (args.save_suffix!=''):
        save_suffix = '_' + args.save_suffix
    else:
        save_suffix = ''

    # Build plots list
    if (',' in args.plot):
        plots = args.plot.split(',')
    else:
        plots = [args.plot]

    # Build outputs list
    outs = make_output_list(args.output, output_step=args.output_step)

    if ('sb_time_hist' in args.plot):
        surface_brightness_time_histogram(outs)
    else:
        target_dir = 'ions'
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

    print(str(datetime.datetime.now()))
    print("All snapshots finished!")