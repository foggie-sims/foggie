'''
Filename: emission_maps.py
Author: Cassi
Date created: 8-26-24
Date last modified: 10-16-24

This file contains everything needed to make emission maps and FRBs from CLOUDY tables.
All CLOUDY and emission code copy-pasted from Lauren's foggie/emission/emission_functions.py.'''

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
import matplotlib.colors as mcolors
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

    parser = argparse.ArgumentParser(description='Makes emission maps and/or FRBs of various emission lines.')

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
                       help='What do you want to plot? Options are:\n' + \
                        'emission_map       -  Plots an image of projected emission lines edge-on and face-on\n' + \
                        'emission_map_vbins -  Plots many images of projected emission lines edge-on and face-on in line-of-sight velocity bins\n' + \
                        'emission_FRB       -  Makes FRBs of projected emission lines edge-on and face-on')
    parser.set_defaults(plot='emission_map')

    parser.add_argument('--ions', metavar='ions', type=str, action='store', \
                      help='What ions do you want emission maps or FRBs for? Options are:\n' + \
                        'Lyalpha, Halpha, CIII, CIV, MgII, OVI, SiII, SiIII, SiIV\n' + \
                        "If you want multiple, use a comma-separated list with no spaces like:\n" + \
                        "--ions 'CIII,OVI,SiIII'")

    parser.add_argument('--Dragonfly_limit', dest='Dragonfly_limit', action='store_true', \
                        help='Do you want to calculate and plot only above the Dragonfly limit? Default is no. This only matters for Halpha.')
    parser.set_defaults(Dragonfly_limit=False)

    parser.add_argument('--Aspera_limit', dest='Aspera_limit', action='store_true', \
                        help='Do you want to calculate and plot only above the Aspera limit? Default is no. This only matters for O VI.')
    parser.set_defaults(Aspera_limit=False)
    
    parser.add_argument('--save_suffix', metavar='save_suffix', type=str, action='store', \
                        help='Do you want to append a string onto the names of the saved files? Default is no.')
    parser.set_defaults(save_suffix="")

    parser.add_argument('--file_suffix', metavar='file_suffix', type=str, action='store', \
                        help='If plotting from saved surface brightness files, use this to pass the file name suffix.')
    parser.set_defaults(file_suffix="")

    args = parser.parse_args()
    return args

def scale_by_metallicity(values,assumed_Z,wanted_Z):
    # The Cloudy calculations assumed a single metallicity (typically solar).
    # This function scales the emission by the metallicity of the gas itself to
    # account for this discrepancy.
    wanted_ratio = (10.**(wanted_Z))/(10.**(assumed_Z))
    return values*wanted_ratio

def make_Cloudy_table(table_index):
    # This function takes all of the Cloudy files and compiles them into one table
    # for use in the emission functions
    # table_index is the column in the Cloudy output files that is being read.
    # each table_index value corresponds to a different emission line

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

def make_Cloudy_table_thin(table_index):
    hden_n_bins, hden_min, hden_max = 17, -5, 2
    T_n_bins, T_min, T_max = 51, 3, 8 #71, 2, 8

    hden=np.linspace(hden_min,hden_max,hden_n_bins)
    T=np.linspace(T_min,T_max, T_n_bins)
    table = np.zeros((hden_n_bins,T_n_bins))
    for i in range(hden_n_bins):
            table[i,:]=[float(l.split()[table_index]) for l in open(cloudy_path_thin%(i+1)) if l[0] != "#"]
    return hden,T,table

def _Emission_LyAlpha(field,data):
    H_N=np.log10(np.array(data["H_nuclei_density"]))
    Temperature=np.log10(np.array(data["Temperature"]))
    dia1 = bl_LA(H_N,Temperature)
    idx = np.isnan(dia1)
    dia1[idx] = -200.
    emission_line = (10**dia1)*((10.0**H_N)**2.0)
    emission_line = emission_line/(4.*np.pi*1.63e-11)
    return emission_line*ytEmU

def _Emission_HAlpha_ALTunits(field,data):
    H_N = np.log10(np.array(data['H_nuclei_density']))
    Temperature = np.log10(np.array(data['Temperature']))
    dia1 = bl_HA(H_N,Temperature)
    idx = np.isnan(dia1)
    dia1[idx] = -200.
    emission_line=(10.**dia1)*((10.**H_N)**2.0)
    emission_line = emission_line/(4.*np.pi)
    emission_line = emission_line/4.25e10 # convert steradian to arcsec**2
    return emission_line*ytEmUALT

def _Emission_HAlpha(field,data):
    H_N = np.log10(np.array(data['H_nuclei_density']))
    Temperature = np.log10(np.array(data['Temperature']))
    dia1 = bl_HA(H_N,Temperature)
    idx = np.isnan(dia1)
    dia1[idx] = -200.
    emission_line=(10.**dia1)*((10.**H_N)**2.0)
    emission_line = emission_line/(4.*np.pi*3.03e-12)
    return emission_line*ytEmU

def _Emission_CIII_977(field,data):
    H_N=np.log10(np.array(data["H_nuclei_density"]))
    Temperature=np.log10(np.array(data["Temperature"]))
    dia1 = bl_CIII_977(H_N,Temperature)
    idx = np.isnan(dia1)
    dia1[idx] = -200.
    emission_line=(10.0**dia1)*((10.0**H_N)**2.0)
    emission_line = emission_line/(4.*np.pi*2.03e-11)
    emission_line = scale_by_metallicity(emission_line,0.0,np.log10(np.array(data['metallicity'])))
    return emission_line*ytEmU

def _Emission_CIV_1548(field,data):
        H_N=np.log10(np.array(data["H_nuclei_density"]))
        Temperature=np.log10(np.array(data["Temperature"]))
        dia1 = bl_CIV_1(H_N,Temperature)
        idx = np.isnan(dia1)
        dia1[idx] = -200.
        emission_line=(10.0**dia1)*((10.0**H_N)**2.0)
        emission_line = emission_line/(4.*np.pi*1.28e-11)
        emission_line = scale_by_metallicity(emission_line,0.0,np.log10(np.array(data['metallicity'])))
        return emission_line*ytEmU

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
    return emission_line * ytEmU

def _Emission_SiIII_1207(field,data):
        H_N=np.log10(np.array(data["H_nuclei_density"]))
        Temperature=np.log10(np.array(data["Temperature"]))
        dia1 = bl_SiIII_1207(H_N,Temperature)
        idx = np.isnan(dia1)
        dia1[idx] = -200.
        emission_line=(10.0**dia1)*((10.0**H_N)**2.0)
        emission_line = emission_line/(4.*np.pi*1.65e-11)
        emission_line = scale_by_metallicity(emission_line,0.0,np.log10(np.array(data['metallicity'])))
        return emission_line*ytEmU

def make_FRB(ds, refine_box, snap, ions):
    '''This function takes the dataset 'ds' and the refine box region 'refine_box' and
    makes a fixed resolution buffer of surface brightness from edge-on, face-on,
    and arbitrary orientation projections of all ions in the list 'ions'.'''

    halo_name = halo_dict[str(args.halo)]

    pix_res = float(np.min(refine_box['dx'].in_units('kpc')))
    res = int(ds.refine_width/pix_res)
    print('z=%.1f' % ds.get_parameter('CosmologyCurrentRedshift', 1))

    f = h5py.File(prefix + 'FRBs/' + halo_name + '_emission_maps' + save_suffix + '.hdf5', 'a')
    grp = f.create_group('z=%.1f' % ds.get_parameter('CosmologyCurrentRedshift', 1))
    grp.attrs.create("image_extent_kpc", ds.refine_width)
    grp.attrs.create("redshift", ds.get_parameter('CosmologyCurrentRedshift'))
    grp.attrs.create("halo_name", halo_name)
    grp.attrs.create("emission_units", 'photons/sec/cm^2/sr')

    for i in range(len(ions)):
        ion = ions[i]
        proj_edge = yt.ProjectionPlot(ds, ds.x_unit_disk, ('gas','Emission_' + ions_dict[ion]), center=ds.halo_center_kpc, data_source=refine_box, width=(ds.refine_width, 'kpc'), north_vector=ds.z_unit_disk, buff_size=[res,res])
        frb_edge = proj_edge.frb[('gas','Emission_' + ions_dict[ion])]
        dset1 = grp.create_dataset(ion + "_emission_edge", data=frb_edge)
        mymap = cmr.get_sub_cmap('cmr.flamingo', 0.2, 0.8)
        mymap.set_bad("#421D0F")
        proj_edge.set_cmap('Emission_' + ions_dict[ion], mymap)
        proj_edge.set_zlim('Emission_' + ions_dict[ion], zlim_dict[ion][0], zlim_dict[ion][1])
        proj_edge.set_colorbar_label('Emission_' + ions_dict[ion], label_dict[ion] + ' Emission [photons s$^{-1}$ cm$^{-2}$ sr$^{-1}$]')
        proj_edge.set_font_size(20)
        proj_edge.annotate_timestamp(corner='upper_left', redshift=True, time=True, draw_inset_box=True)
        proj_edge.save(prefix + 'FRBs/' + snap + '_' + ion + '_emission_map_edge-on' + save_suffix + '.png')

        proj_face = yt.ProjectionPlot(ds, ds.z_unit_disk, ('gas','Emission_' + ions_dict[ion]), center=ds.halo_center_kpc, data_source=refine_box, width=(ds.refine_width, 'kpc'), north_vector=ds.x_unit_disk, buff_size=[res,res])
        frb_face = proj_face.frb[('gas','Emission_' + ions_dict[ion])]
        dset2 = grp.create_dataset(ion + "_emission_face", data=frb_face)
        mymap = cmr.get_sub_cmap('cmr.flamingo', 0.2, 0.8)
        mymap.set_bad("#421D0F")
        proj_face.set_cmap('Emission_' + ions_dict[ion], mymap)
        proj_face.set_zlim('Emission_' + ions_dict[ion], zlim_dict[ion][0], zlim_dict[ion][1])
        proj_face.set_colorbar_label('Emission_' + ions_dict[ion], label_dict[ion] + ' Emission [photons s$^{-1}$ cm$^{-2}$ sr$^{-1}$]')
        proj_face.set_font_size(20)
        proj_face.annotate_timestamp(corner='upper_left', redshift=True, time=True, draw_inset_box=True)
        proj_face.save(prefix + 'FRBs/' + snap + '_' + ion + '_emission_map_face-on' + save_suffix + '.png')

    f.close()

def emission_map_vbins(ds, snap, ions):
    '''Makes many emission maps for each ion in 'ions', oriented both edge-on and face-on, for each line-of-sight velocity bin.'''

    vbins = np.arange(-500.,550.,50.)
    
    ad = ds.all_data()
    for i in range(len(ions)):
        ion = ions[i]
        if (ion=='Halpha') and (args.Dragonfly_limit):
            cmap1 = cmr.take_cmap_colors('cmr.flamingo', 9, cmap_range=(0.4, 0.8), return_fmt='rgba')
            cmap2 = cmr.take_cmap_colors('cmr.neutral_r', 3, cmap_range=(0.2, 0.6), return_fmt='rgba')
            cmap = np.hstack([cmap2, cmap1])
            mymap = mcolors.LinearSegmentedColormap.from_list('cmap', cmap)
        elif (ion=='OVI') and (args.Aspera_limit):
            cmap1 = cmr.take_cmap_colors('cmr.flamingo', 4, cmap_range=(0.4, 0.8), return_fmt='rgba')
            cmap2 = cmr.take_cmap_colors('cmr.neutral_r', 6, cmap_range=(0.2, 0.6), return_fmt='rgba')
            cmap = np.hstack([cmap2, cmap1])
            mymap = mcolors.LinearSegmentedColormap.from_list('cmap', cmap)
        else:
            mymap = cmr.get_sub_cmap('cmr.flamingo', 0.2, 0.8)
        mymap.set_bad(mymap.colors[0])
        for v in range(len(vbins)-1):
            vbox = ds.cut_region(ad, ["obj[('gas', 'vx_disk')] > %.1f" % vbins[v]])
            vbox = ds.cut_region(vbox, ["obj[('gas', 'vx_disk')] < %.1f" % vbins[v+1]])
            proj_edge = yt.ProjectionPlot(ds, ds.x_unit_disk, ('gas','Emission_' + ions_dict[ion]), center=ds.halo_center_kpc, width=(ds.refine_width, 'kpc'), north_vector=ds.z_unit_disk, data_source=vbox)
            proj_edge.set_cmap('Emission_' + ions_dict[ion], mymap)
            proj_edge.set_zlim('Emission_' + ions_dict[ion], zlim_dict[ion][0], zlim_dict[ion][1])
            proj_edge.set_colorbar_label('Emission_' + ions_dict[ion], label_dict[ion] + ' Emission [photons s$^{-1}$ cm$^{-2}$ sr$^{-2}$]')
            proj_edge.set_font_size(20)
            proj_edge.annotate_title('$%d < v_{\\rm los} < %d$' % (vbins[v], vbins[v+1]))
            proj_edge.annotate_timestamp(corner='upper_left', redshift=True, time=True, draw_inset_box=True)
            proj_edge.save(prefix + 'Projections/' + snap + '_' + ion + '_emission_map_edge-on_vbin' + str(v) + save_suffix + '.png')

            proj_face = yt.ProjectionPlot(ds, ds.z_unit_disk, ('gas','Emission_' + ions_dict[ion]), center=ds.halo_center_kpc, width=(ds.refine_width, 'kpc'), north_vector=ds.x_unit_disk, data_source=vbox)
            proj_face.set_cmap('Emission_' + ions_dict[ion], mymap)
            proj_face.set_zlim('Emission_' + ions_dict[ion], zlim_dict[ion][0], zlim_dict[ion][1])
            proj_face.set_colorbar_label('Emission_' + ions_dict[ion], label_dict[ion] + ' Emission [photons s$^{-1}$ cm$^{-2}$ sr$^{-2}$]')
            proj_face.set_font_size(20)
            proj_face.annotate_title('$%d < v_{\\rm los} < %d$' % (vbins[v], vbins[v+1]))
            proj_face.annotate_timestamp(corner='upper_left', redshift=True, time=True, draw_inset_box=True)
            proj_face.save(prefix + 'Projections/' + snap + '_' + ion + '_emission_map_face-on_vbin' + str(v) + save_suffix + '.png')

def emission_map(ds, snap, ions):
    '''Makes emission maps for each ion in 'ions', oriented both edge-on and face-on.'''

    for i in range(len(ions)):
        ion = ions[i]
        if (ion=='Halpha') and (args.Dragonfly_limit):
            cmap1 = cmr.take_cmap_colors('cmr.flamingo', 9, cmap_range=(0.4, 0.8), return_fmt='rgba')
            cmap2 = cmr.take_cmap_colors('cmr.neutral_r', 3, cmap_range=(0.2, 0.6), return_fmt='rgba')
            cmap = np.hstack([cmap2, cmap1])
            mymap = mcolors.LinearSegmentedColormap.from_list('cmap', cmap)
        elif (ion=='OVI') and (args.Aspera_limit):
            cmap1 = cmr.take_cmap_colors('cmr.flamingo', 4, cmap_range=(0.4, 0.8), return_fmt='rgba')
            cmap2 = cmr.take_cmap_colors('cmr.neutral_r', 6, cmap_range=(0.2, 0.6), return_fmt='rgba')
            cmap = np.hstack([cmap2, cmap1])
            mymap = mcolors.LinearSegmentedColormap.from_list('cmap', cmap)
        else:
            mymap = cmr.get_sub_cmap('cmr.flamingo', 0.2, 0.8)

        proj_edge = yt.ProjectionPlot(ds, ds.x_unit_disk, ('gas','Emission_' + ions_dict[ion]), center=ds.halo_center_kpc, width=(ds.refine_width, 'kpc'), north_vector=ds.z_unit_disk)
        proj_edge.set_cmap('Emission_' + ions_dict[ion], mymap)
        proj_edge.set_zlim('Emission_' + ions_dict[ion], zlim_dict[ion][0], zlim_dict[ion][1])
        proj_edge.set_colorbar_label('Emission_' + ions_dict[ion], label_dict[ion] + ' Emission [photons s$^{-1}$ cm$^{-2}$ sr$^{-2}$]')
        proj_edge.set_font_size(20)
        proj_edge.annotate_timestamp(corner='upper_left', redshift=True, time=True, draw_inset_box=True)
        proj_edge.save(prefix + 'Projections/' + snap + '_' + ion + '_emission_map_edge-on' + save_suffix + '.png')

        proj_face = yt.ProjectionPlot(ds, ds.z_unit_disk, ('gas','Emission_' + ions_dict[ion]), center=ds.halo_center_kpc, width=(ds.refine_width, 'kpc'), north_vector=ds.x_unit_disk)
        proj_face.set_cmap('Emission_' + ions_dict[ion], mymap)
        proj_face.set_zlim('Emission_' + ions_dict[ion], zlim_dict[ion][0], zlim_dict[ion][1])
        proj_face.set_colorbar_label('Emission_' + ions_dict[ion], label_dict[ion] + ' Emission [photons s$^{-1}$ cm$^{-2}$ sr$^{-2}$]')
        proj_face.set_font_size(20)
        proj_face.annotate_timestamp(corner='upper_left', redshift=True, time=True, draw_inset_box=True)
        proj_face.save(prefix + 'Projections/' + snap + '_' + ion + '_emission_map_face-on' + save_suffix + '.png')

def load_and_calculate(snap, ions):
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
        if ('vbins' not in args.plot):
            emission_map(ds, snap, ions)
        else:
            emission_map_vbins(ds, snap, ions)
    if ('emission_FRB' in args.plot):
        make_FRB(ds, refine_box, snap, ions)

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

    if ('feedback' in args.run) and ('track' in args.run):
        foggie_dir = '/nobackup/jtumlins/halo_008508/feedback-track/'
        run_dir = args.run + '/'
    
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
    ## H-alpha ##
    ############################
    # 1. Read cloudy file
    hden_pts,T_pts,table_HA = make_Cloudy_table(2)
    # 2. Create grids (only need to do this for first emission line)
    hden_pts,T_pts = np.meshgrid(hden_pts,T_pts)
    pts = np.array((hden_pts.ravel(),T_pts.ravel())).T

    #set up interpolation fundtion
    sr_HA = table_HA.T.ravel()
    bl_HA = interpolate.LinearNDInterpolator(pts,sr_HA)
    # 5. Add field
    yt.add_field(('gas','Emission_HAlpha'),units=emission_units,function=_Emission_HAlpha,take_log=True,force_override=True,sampling_type='cell')

    ############################
    ## Ly-alpha ##
    ############################
    hden_pts,T_pts,table_LA = make_Cloudy_table(1)
    sr_LA = table_LA.T.ravel()
    bl_LA = interpolate.LinearNDInterpolator(pts,sr_LA)
    yt.add_field(('gas','Emission_LyAlpha'),units=emission_units,function=_Emission_LyAlpha,take_log=True,force_override=True,sampling_type='cell')

    ############################
    ## CIII 977 ##
    ############################
    hden1,T1,table_CIII_977 = make_Cloudy_table(7)
    sr_CIII_977 = table_CIII_977.T.ravel()
    bl_CIII_977 = interpolate.LinearNDInterpolator(pts,sr_CIII_977)
    yt.add_field(("gas","Emission_CIII_977"),units=emission_units,function=_Emission_CIII_977,take_log=True,force_override=True,sampling_type='cell')

    ############################
    ## CIV 1548 ##
    ############################
    hden1, T1, table_CIV_1 = make_Cloudy_table(3)
    sr_CIV_1 = table_CIV_1.T.ravel()
    bl_CIV_1 = interpolate.LinearNDInterpolator(pts,sr_CIV_1)
    yt.add_field(("gas","Emission_CIV_1548"),units=emission_units,function=_Emission_CIV_1548,take_log=True,force_override=True,sampling_type='cell')

    ############################
    ## O VI both 1032 and 1037 ##
    ############################
    hden1, T1, table_OVI_1 = make_Cloudy_table(5)
    hden1, T1, table_OVI_2 = make_Cloudy_table(6)
    sr_OVI_1 = table_OVI_1.T.ravel()
    sr_OVI_2 = table_OVI_2.T.ravel()
    bl_OVI_1 = interpolate.LinearNDInterpolator(pts,sr_OVI_1)
    bl_OVI_2 = interpolate.LinearNDInterpolator(pts,sr_OVI_2)
    yt.add_field(("gas","Emission_OVI"),units=emission_units,function=_Emission_OVI,take_log=True,force_override=True,sampling_type='cell')

    ############################
    ## Si III 1207 ##
    ############################
    # This one in optically thin Cloudy table, so need to load new table
    cloudy_path_thin = code_path + "emission/cloudy_z0_HM05/bertone_run%i.dat"
    hden_pts,T_pts,table_SiIII_1207 = make_Cloudy_table_thin(11)
    hden_pts,T_pts = np.meshgrid(hden_pts,T_pts)
    pts = np.array((hden_pts.ravel(),T_pts.ravel())).T
    sr_SiIII_1207 = table_SiIII_1207.T.ravel()
    bl_SiIII_1207 = interpolate.LinearNDInterpolator(pts,sr_SiIII_1207)
    yt.add_field(('gas','Emission_SiIII_1207'),units=emission_units,function=_Emission_SiIII_1207,take_log=True,force_override=True,sampling_type='cell')

    ions_dict = {'Lyalpha':'LyAlpha', 'Halpha':'HAlpha', 'CIII':'CIII_977',
                 'CIV':'CIV_1548','OVI':'OVI', 'SiIII':'SiIII_1207'}
    label_dict = {'Lyalpha':r'Ly-$\alpha$', 'Halpha':r'H$\alpha$', 'CIII':'C III',
                'CIV':'C IV','OVI':'O VI', 'SiIII':'Si III'}
    zlim_dict = {'Lyalpha':[1e-1,1e7], 'Halpha':[1e-1,1e6], 'CIII':[1e-4,1e0],
                 'CIV':[1e-2,1e4], 'OVI':[1e-2,1e5], 'SiIII':[1e-1,1e4]}


    if (args.save_suffix!=''):
        save_suffix = '_' + args.save_suffix
    else:
        save_suffix = ''

    if (args.file_suffix!=''):
        file_suffix = '_' + args.file_suffix
    else:
        file_suffix = ''

    # Build plots list
    if (',' in args.plot):
        plots = args.plot.split(',')
    else:
        plots = [args.plot]

    # Build ions list
    if (',' in args.ions):
        ions = args.ions.split(',')
    else:
        ions = [args.ions]

    # Build outputs list
    outs = make_output_list(args.output, output_step=args.output_step)

    target_dir = 'ions'
    if (args.nproc==1):
        for snap in outs:
            load_and_calculate(snap, ions)
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
                    threads.append(multi.Process(target=load_and_calculate, args=[snap, ions]))
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
                threads.append(multi.Process(target=load_and_calculate, args=[snap, ions]))
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