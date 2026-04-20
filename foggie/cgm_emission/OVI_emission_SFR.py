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
from datetime import timedelta
import time
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
from foggie.cgm_emission.make_rates_table import combine_rates

# These imports for datashader plot
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
                       help='What do you want to plot? Options are:\n' + \
                        'emission_map       -  Plots an image of projected O VI emission edge-on\n' + \
                        'all_halos_map      - Makes an image of projected density and O VI emission edge-on for all halos at once\n' + \
                        'emission_FRB       -  Saves to file FRBs of O VI surface brightness\n' + \
                        'ionization_equilibrium -  Plots an image of projected O VI emission edge-on along with projected O VI ionization equilibrium timescale\n' + \
                        'sb_profile         -  Plots the surface brightness profile\n' + \
                        'radial_profiles    -  Calculates and plots radial profiles of gas properties weighted by emissivity\n' + \
                        'phase_plot         -  Plots a density-temperature phase plot of O VI emissivity\n' + \
                        'sb_profile_time_avg - Plots the time- and halo-averaged surface brightness profile\n' + \
                        'sb_time_hist       -  Plots a histogram of surface brightness over time\n' + \
                        'sb_time_hist_all_halos - Plots histograms of surface brightness over time for all halos\n' + \
                        'sb_time_radius     -  Plots surface brightness profiles over time and radius on 2D color plot\n' + \
                        'sb_vs_sfr          -  Plots a scatterplot of surface brightness vs. SFR\n' + \
                        'sb_vs_mh           -  Plots a scatterplot of surface brightness vs. halo mass\n' + \
                        'sb_vs_den          -  Plots a scatterplot of surface brightness vs. avg CGM density\n' + \
                        'den_vs_time        -  Plots a scatterplot of avg CGM density vs. time and redshift\n' + \
                        'sb_vs_Z            -  Plots a scatterplot of surface brightness vs. avg CGM metallicity\n' + \
                        'sb_vs_temp         -  Plots a scatterplot of surface brightness vs. avg CGM temperature\n' + \
                        'histograms         -  Plots histograms of all gas and emission-weighted gas in density, temperature, metallicity, radial velocity, and cooling time\n' + \
                        'histograms_radbins -  Plots histograms of all gas and emission-weighted gas in density, temperature, metallicity, radial velocity, and cooling time in radial bins\n' + \
                        'histograms_radbins_all_halos -  Plots histograms of all gas and emission-weighted gas in density, temperature, metallicity, radial velocity, and cooling time in radial bins, stacking all halos\n' + \
                        'emiss_area_vs_sfr  -  Plots a scatterplot of the fraction of the area with a surface brightness above the Aspera limit vs. SFR\n' + \
                        'emiss_area_vs_mh   -  Plots a scatterplot of the fraction of the area with a surface brightness above the Aspera limit vs. halo mass\n' + \
                        'emiss_area_vs_den  -  Plots a scatterplot of the fraction of the area with a surface brightness above the Aspera limit vs. average CGM density\n' + \
                        'emiss_area_vs_Z    -  Plots a scatterplot of the fraction of the area with a surface brightness above the Aspera limit vs. average CGM metallicity\n' + \
                        'Emission maps and SB profiles are calculated from the simulation snapshot.\n' + \
                        'SB over time or vs. SFR or Mh are plotted from the SB pdfs that are generated when sb_profile is run.\n' + \
                        'You can plot multiple things by separating keywords with a comma (no spaces!), and the default is "emission_map,sb_profile".')
    parser.set_defaults(plot='emission_map,sb_profile')

    parser.add_argument('--Aspera_limit', dest='Aspera_limit', action='store_true', \
                        help='Do you want to calculate and plot only above the Aspera limit? Default is no.')
    parser.set_defaults(Aspera_limit=False)

    parser.add_argument('--DISCO_limit', dest='DISCO_limit', action='store_true', \
                        help='Do you want to calculate and plot only above the DISCO limit? Default is no.')
    parser.set_defaults(DISCO_limit=False)

    parser.add_argument('--constant_Z', dest='constant_Z', action='store_true', \
                        help='Do you want to calculate emissivity assuming solar metallicity? Default is no.')
    parser.set_defaults(constant_Z=False)
    
    parser.add_argument('--save_suffix', metavar='save_suffix', type=str, action='store', \
                        help='Do you want to append a string onto the names of the saved files? Default is no.')
    parser.set_defaults(save_suffix="")

    parser.add_argument('--file_suffix', metavar='file_suffix', type=str, action='store', \
                        help='If plotting from saved surface brightness files, use this to pass the file name suffix.')
    parser.set_defaults(file_suffix="")

    parser.add_argument('--weight', metavar='weight', type=str, action='store', \
                        help='If calculating or plotting radial profiles or histograms, do you want to weight by mass or volume? Default is mass.')
    parser.set_defaults(weight="mass")

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
        hden_n_bins, hden_min, hden_max = 17, -6, 2
        T_n_bins, T_min, T_max = 51, 3, 8

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
        if (not args.constant_Z):
            emission_line = scale_by_metallicity(emission_line,0.0,np.log10(np.array(data['metallicity'])))
        emission_line[emission_line==0.0] = 5e-324
        return emission_line * ytEmU

def _Emission_OVI_ALTunits(field,data):
        H_N=np.log10(np.array(data["H_nuclei_density"]))
        Temperature=np.log10(np.array(data["Temperature"]))
        dia1 = bl_OVI_1(H_N,Temperature)
        dia2 = bl_OVI_2(H_N,Temperature)
        idx = np.isnan(dia1)
        dia1[idx] = -200.
        dia2[idx] = -200.
        emission_line=((10.0**dia1))*((10.0**H_N)**2.0)
        emission_line= emission_line + ((10**dia2))*((10.0**H_N)**2.0)
        emission_line = emission_line/(4.*np.pi)
        emission_line = emission_line/4.25e10 # convert steradian to arcsec**2
        if (not args.constant_Z):
            emission_line = scale_by_metallicity(emission_line,0.0,np.log10(np.array(data['metallicity'])))
        emission_line[emission_line==0.0] = 5e-324
        return emission_line*ytEmUALT

def _ionization_OVI(field, data):
    H_N=np.log10(np.array(data["H_nuclei_density"]))
    Temperature=np.log10(np.array(data["Temperature"]))
    ion_rate = ionization_interp(H_N,Temperature)
    idx = np.isnan(ion_rate)
    ion_rate[idx] = 0.
    return ion_rate * unyt.second**-1

def _recombination_OVI(field, data):
    H_N=np.log10(np.array(data["H_nuclei_density"]))
    Temperature=np.log10(np.array(data["Temperature"]))
    rec_rate = recombination_interp(H_N,Temperature)
    idx = np.isnan(rec_rate)
    rec_rate[idx] = 0.
    return rec_rate * unyt.second**-1

def _equilibration_time_OVI(field, data):
    sum_rates = data["Recombination_Rate_OVI"] + data["Ionization_Rate_OVI"]
    sum_rates[sum_rates==0.] = 1e20
    eq_time = 1./(sum_rates)
    return eq_time

def _masked_density(field, data):
    masked_den = data["density"]
    masked_den[data["Equilibration_Time_OVI"]==1e-20] = 1e-50
    return masked_den

def _teq_over_tcool_OVI(field, data):
    return data["Equilibration_Time_OVI"]/data[("gas","cooling_time")]

def weighted_quantile(values, weights, quantiles):
    """ Very close to numpy.percentile, but supports weights.
    NOTE: quantiles should be in [0, 1]!
    :param values: numpy.array with data
    :param weights: array-like of the same length as `array`
    :param quantiles: array-like with many quantiles needed
    :return: numpy.array with computed quantiles.
    """

    sorter = np.argsort(values)
    values = values[sorter]
    weights = weights[sorter]
    weighted_quantiles = np.cumsum(weights) - 0.5 * weights
    weighted_quantiles /= np.sum(weights)

    return np.interp(quantiles, weighted_quantiles, values)

def weighted_avg_and_std(values, weights):
    """
    Return the weighted average and standard deviation.
    values, weights -- Numpy ndarrays with the same shape.
    """
    average = np.average(values, weights=weights)
    variance = np.average((values-average)**2, weights=weights)
    return average, np.sqrt(variance)

def make_pdf_table():
    '''Makes the giant table of O VI surface brightness histograms that will be saved to file.'''

    names_list = ['inner_radius', 'outer_radius', 'lower_SB', 'upper_SB', 'all', 'inflow', 'outflow', 'neither', 'major', 'minor', 'cold', 'hot']
    types_list = ['f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8']

    table = Table(names=names_list, dtype=types_list)

    return table

def make_rad_pdf_table():
    '''Makes the giant table of radial properties histograms that will be saved to file.'''

    names_list = ['inner_radius', 'outer_radius', 'bin']
    types_list = ['f8', 'f8', 'f8']

    for p in ['den', 'temp', 'met', 'rv', 'tcool']:
        for i in ['all_', 'cgm_', 'OVI_']:
            for j in ['', '_inflow', '_outflow']:
                names_list.append(i + p + j)
                types_list.append('f8')

    table = Table(names=names_list, dtype=types_list)

    return table

def make_sb_profile_table():
    '''Makes the giant table of O VI surface brightness profiles that will be saved to file.'''

    names_list = ['inner_radius', 'outer_radius', 'all_mean', 'all_med', 'inflow_mean', 'inflow_med', \
                  'outflow_mean', 'outflow_med', 'neither_mean', 'neither_med', 'major_mean', 'major_med', 'minor_mean', 'minor_med', \
                  'cold_mean', 'cold_med', 'hot_mean', 'hot_med']
    types_list = ['f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8']

    table = Table(names=names_list, dtype=types_list)

    return table

def make_rad_profile_table():
    '''Makes the giant table of radial profiles that will be saved to file.'''

    names_list = ['inner_radius', 'outer_radius']
    types_list = ['f8', 'f8']

    for p in ['den', 'temp', 'met', 'rv', 'tcool']:
        for i in ['all_', 'cgm_', 'OVI_']:
            for j in ['', '_inflow', '_outflow']:
                names_list.append(i + p + '_mean' + j)
                types_list.append('f8')
                names_list.append(i + p + '_med' + j)
                types_list.append('f8')

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
        elif ('den' in key) and (not pdf):
            table[key].unit = 'g/cm^2'
        elif ('temp' in key) and (not pdf):
            table[key].unit = 'K'
        elif ('met' in key) and (not pdf):
            table[key].unit = 'Zsun'
        elif ('rv' in key) and (not pdf):
            table[key].unit = 'km/s'
        else:
            table[key].unit = 'none'
    return table

def make_FRB(ds, refine_box, snap):
    '''This function takes the dataset 'ds' and the refine box region 'refine_box' and
    makes a fixed resolution buffer of O VI emission surface brightness from edge-on, face-on,
    and arbitrary orientation projections.'''

    halo_name = halo_dict[str(args.halo)]

    pix_res = float(np.min(refine_box['dx'].in_units('kpc')))
    res = int(ds.refine_width/pix_res)
    print('z=%.1f' % ds.get_parameter('CosmologyCurrentRedshift', 1))

    f = h5py.File(prefix + 'FRBs/' + halo_name + '_OVI_emission_maps' + save_suffix + '.hdf5', 'a')
    grp = f.create_group('z=%.1f' % ds.get_parameter('CosmologyCurrentRedshift', 1))
    grp.attrs.create("image_extent_kpc", ds.refine_width)
    grp.attrs.create("redshift", ds.get_parameter('CosmologyCurrentRedshift'))
    grp.attrs.create("halo_name", halo_name)

    proj_edge = yt.ProjectionPlot(ds, ds.x_unit_disk, ('gas','Emission_OVI'), center=ds.halo_center_kpc, data_source=refine_box, width=(ds.refine_width, 'kpc'), north_vector=ds.z_unit_disk, buff_size=[res,res])
    frb_edge = proj_edge.frb[('gas','Emission_OVI')]
    dset1 = grp.create_dataset("OVI_emission_edge", data=frb_edge)
    mymap = cmr.get_sub_cmap('cmr.flamingo', 0.2, 0.8)
    mymap.set_bad("#421D0F")
    proj_edge.set_cmap('Emission_OVI', mymap)
    proj_edge.set_zlim('Emission_OVI', 1e-2, 1e5)
    proj_edge.set_colorbar_label('Emission_OVI', 'O VI Emission [photons s$^{-1}$ cm$^{-2}$ sr$^{-1}$]')
    proj_edge.set_font_size(20)
    proj_edge.annotate_timestamp(corner='upper_left', redshift=True, time=True, draw_inset_box=True)
    proj_edge.save(prefix + 'FRBs/' + snap + '_OVI_emission_map_edge-on' + save_suffix + '.png')

    proj_face = yt.ProjectionPlot(ds, ds.z_unit_disk, ('gas','Emission_OVI'), center=ds.halo_center_kpc, data_source=refine_box, width=(ds.refine_width, 'kpc'), north_vector=ds.x_unit_disk, buff_size=[res,res])
    frb_face = proj_face.frb[('gas','Emission_OVI')]
    dset2 = grp.create_dataset("OVI_emission_face", data=frb_face)
    mymap = cmr.get_sub_cmap('cmr.flamingo', 0.2, 0.8)
    mymap.set_bad("#421D0F")
    proj_face.set_cmap('Emission_OVI', mymap)
    proj_face.set_zlim('Emission_OVI', 1e-2, 1e5)
    proj_face.set_colorbar_label('Emission_OVI', 'O VI Emission [photons s$^{-1}$ cm$^{-2}$ sr$^{-1}$]')
    proj_face.set_font_size(20)
    proj_face.annotate_timestamp(corner='upper_left', redshift=True, time=True, draw_inset_box=True)
    proj_face.save(prefix + 'FRBs/' + snap + '_OVI_emission_map_face-on' + save_suffix + '.png')

    f.close()

def surface_brightness_profile(ds, refine_box, snap):
    '''Makes radial surface brightness profiles of O VI emission for full image, along major and minor axes, and for inflows and outflows and hot and cold gas separately.'''

    profile_table = make_sb_profile_table()
    pdf_table = make_pdf_table()

    sph = ds.sphere(center=ds.halo_center_kpc, radius=(75., 'kpc'))
    sph_inflow = sph.include_below(('gas','radial_velocity_corrected'), -100.)
    sph_outflow = sph.include_above(('gas','radial_velocity_corrected'), 100.)
    sph_neither = sph - sph_inflow - sph_outflow
    sph_cold = sph.include_below(('gas','temperature'), 1e5)
    sph_hot = sph.include_above(('gas','temperature'), 1e5)

    proj = yt.ProjectionPlot(ds, ds.x_unit_disk, ('gas','Emission_OVI'), center=ds.halo_center_kpc, data_source=sph, width=(100., 'kpc'), depth=(100., 'kpc'), north_vector=ds.z_unit_disk)
    FRB = np.log10(proj.frb[('gas','Emission_OVI')].v)
    proj_in = yt.ProjectionPlot(ds, ds.x_unit_disk, ('gas','Emission_OVI'), center=ds.halo_center_kpc, data_source=sph_inflow, width=(100., 'kpc'), depth=(100., 'kpc'), north_vector=ds.z_unit_disk)
    FRB_in = np.log10(proj_in.frb[('gas','Emission_OVI')].v)
    proj_out = yt.ProjectionPlot(ds, ds.x_unit_disk, ('gas','Emission_OVI'), center=ds.halo_center_kpc, data_source=sph_outflow, width=(100., 'kpc'), depth=(100., 'kpc'), north_vector=ds.z_unit_disk)
    FRB_out = np.log10(proj_out.frb[('gas','Emission_OVI')].v)
    proj_neither = yt.ProjectionPlot(ds, ds.x_unit_disk, ('gas','Emission_OVI'), center=ds.halo_center_kpc, data_source=sph_neither, width=(100., 'kpc'), depth=(100., 'kpc'), north_vector=ds.z_unit_disk)
    FRB_neither = np.log10(proj_neither.frb[('gas','Emission_OVI')].v)
    proj_cold = yt.ProjectionPlot(ds, ds.x_unit_disk, ('gas','Emission_OVI'), center=ds.halo_center_kpc, data_source=sph_cold, width=(100., 'kpc'), depth=(100., 'kpc'), north_vector=ds.z_unit_disk)
    FRB_cold = np.log10(proj_cold.frb[('gas','Emission_OVI')].v)
    proj_hot = yt.ProjectionPlot(ds, ds.x_unit_disk, ('gas','Emission_OVI'), center=ds.halo_center_kpc, data_source=sph_hot, width=(100., 'kpc'), depth=(100., 'kpc'), north_vector=ds.z_unit_disk)
    FRB_hot = np.log10(proj_hot.frb[('gas','Emission_OVI')].v)
    if (args.Aspera_limit):
        FRB[FRB<-19] = np.nan
        FRB_in[FRB_in<-19] = np.nan
        FRB_out[FRB_out<-19] = np.nan
        FRB_cold[FRB_cold<-19] = np.nan
        FRB_hot[FRB_hot<-19] = np.nan
    else:
        FRB[FRB<-34] = np.nan
        FRB_in[FRB_in<-34] = np.nan
        FRB_out[FRB_out<-34] = np.nan
        FRB_neither[FRB_neither<-34] = np.nan
        FRB_cold[FRB_cold<-34] = np.nan
        FRB_hot[FRB_hot<-34] = np.nan

    dx = 100./800.              # physical width of FRB (kpc) divided by resolution
    FRB_x = np.indices((800,800))[0]
    FRB_y = np.indices((800,800))[1]
    FRB_x = FRB_x*dx - 50.
    FRB_y = FRB_y*dx - 50.
    radius = np.sqrt(FRB_x**2. + FRB_y**2.)
    theta = np.arctan2(FRB_x, FRB_y)*180./np.pi

    # These are backwards! Note that in individual snapshot profiles, they're backwards.
    # I fixed it for the time-averaged SB profile plot that's in the paper, but not the individual snapshot profiles.
    major = ((np.abs(theta) > 45.) & (np.abs(theta) <135.))
    minor = ~major

    profile_row = [0., 50., np.nanmean(FRB[(radius < 50.)]), np.nanmedian(FRB[(radius < 50.)]), \
                np.nanmean(FRB_in[(radius<50.)]), np.nanmedian(FRB_in[(radius<50.)]), \
                np.nanmean(FRB_out[(radius<50.)]), np.nanmedian(FRB_out[(radius<50.)]), \
                np.nanmean(FRB_neither[(radius<50.)]), np.nanmedian(FRB_neither[(radius<50.)]), \
                np.nanmean(FRB[major & (radius<50.)]), np.nanmedian(FRB[major & (radius<50.)]), \
                np.nanmean(FRB[minor & (radius<50.)]), np.nanmedian(FRB[minor & (radius<50.)]), \
                np.nanmean(FRB_cold[(radius<50.)]), np.nanmedian(FRB_cold[(radius<50.)]), \
                np.nanmean(FRB_hot[(radius<50.)]), np.nanmedian(FRB_hot[(radius<50.)])]
    profile_table.add_row(profile_row)

    profile_row = [0., 20., np.nanmean(FRB[(radius < 20.)]), np.nanmedian(FRB[(radius < 20.)]), \
                np.nanmean(FRB_in[(radius<20.)]), np.nanmedian(FRB_in[(radius<20.)]), \
                np.nanmean(FRB_out[(radius<20.)]), np.nanmedian(FRB_out[(radius<20.)]), \
                np.nanmean(FRB_neither[(radius<20.)]), np.nanmedian(FRB_neither[(radius<20.)]), \
                np.nanmean(FRB[major & (radius<20.)]), np.nanmedian(FRB[major & (radius<20.)]), \
                np.nanmean(FRB[minor & (radius<20.)]), np.nanmedian(FRB[minor & (radius<20.)]), \
                np.nanmean(FRB_cold[(radius<20.)]), np.nanmedian(FRB_cold[(radius<20.)]), \
                np.nanmean(FRB_hot[(radius<20.)]), np.nanmedian(FRB_hot[(radius<20.)])]
    profile_table.add_row(profile_row)

    if (args.Aspera_limit): SB_bins = np.linspace(-19,-16,31)
    else: SB_bins = np.linspace(-25,-16,61)
    SB_hist, bins = np.histogram(FRB[radius<50.], bins=SB_bins)
    SB_hist_in, bins = np.histogram(FRB_in[radius<50.], bins=SB_bins)
    SB_hist_out, bins = np.histogram(FRB_out[radius<50.], bins=SB_bins)
    SB_hist_neither, bins = np.histogram(FRB_neither[radius<50.], bins=SB_bins)
    SB_hist_major, bins = np.histogram(FRB[(radius<50.) & (major)], bins=SB_bins)
    SB_hist_minor, bins = np.histogram(FRB[(radius<50.) & (minor)], bins=SB_bins)
    SB_hist_cold, bins = np.histogram(FRB_cold[radius<50.], bins=SB_bins)
    SB_hist_hot, bins = np.histogram(FRB_hot[radius<50.], bins=SB_bins)
    for i in range(len(bins)-1):
        pdf_row = [0., 50., bins[i], bins[i+1], SB_hist[i], SB_hist_in[i], SB_hist_out[i], SB_hist_neither[i], SB_hist_major[i], SB_hist_minor[i], SB_hist_cold[i], SB_hist_hot[i]]
        pdf_table.add_row(pdf_row)

    SB_hist, bins = np.histogram(FRB[radius<20.], bins=SB_bins)
    SB_hist_in, bins = np.histogram(FRB_in[radius<20.], bins=SB_bins)
    SB_hist_out, bins = np.histogram(FRB_out[radius<20.], bins=SB_bins)
    SB_hist_neither, bins = np.histogram(FRB_neither[radius<20.], bins=SB_bins)
    SB_hist_major, bins = np.histogram(FRB[(radius<20.) & (major)], bins=SB_bins)
    SB_hist_minor, bins = np.histogram(FRB[(radius<20.) & (minor)], bins=SB_bins)
    SB_hist_cold, bins = np.histogram(FRB_cold[radius<20.], bins=SB_bins)
    SB_hist_hot, bins = np.histogram(FRB_hot[radius<20.], bins=SB_bins)
    for i in range(len(bins)-1):
        pdf_row = [0., 20., bins[i], bins[i+1], SB_hist[i], SB_hist_in[i], SB_hist_out[i], SB_hist_neither[i], SB_hist_major[i], SB_hist_minor[i], SB_hist_cold[i], SB_hist_hot[i]]
        pdf_table.add_row(pdf_row)

    rbins = np.linspace(0., 50., 26)
    rbin_centers = rbins[:-1] + np.diff(rbins)
    full_profile = []
    major_profile = []
    minor_profile = []
    inflow_profile = []
    outflow_profile = []
    neither_profile = []
    cold_profile = []
    hot_profile = []
    for r in range(len(rbins)-1):
        r_low = rbins[r]
        r_upp = rbins[r+1]
        shell = (radius > r_low) & (radius < r_upp)
        full_profile.append(np.nanmedian(FRB[shell]))
        major_profile.append(np.nanmedian(FRB[shell & major]))
        minor_profile.append(np.nanmedian(FRB[shell & minor]))
        inflow_profile.append(np.nanmedian(FRB_in[shell]))
        outflow_profile.append(np.nanmedian(FRB_out[shell]))
        neither_profile.append(np.nanmedian(FRB_neither[shell]))
        cold_profile.append(np.nanmedian(FRB_cold[shell]))
        hot_profile.append(np.nanmedian(FRB_hot[shell]))
        profile_row = [r_low, r_upp, np.nanmean(FRB[shell]), np.nanmedian(FRB[shell]), \
                    np.nanmean(FRB_in[shell]), np.nanmedian(FRB_in[shell]), \
                    np.nanmean(FRB_out[shell]), np.nanmedian(FRB_out[shell]), \
                    np.nanmean(FRB_neither[shell]), np.nanmedian(FRB_neither[shell]), \
                    np.nanmean(FRB[shell & major]), np.nanmedian(FRB[shell & major]), \
                    np.nanmean(FRB[shell & minor]), np.nanmedian(FRB[shell & minor]), \
                    np.nanmean(FRB_cold[shell]), np.nanmedian(FRB_cold[shell]), \
                    np.nanmean(FRB_hot[shell]), np.nanmedian(FRB_hot[shell])]
        profile_table.add_row(profile_row)

        SB_hist, bins = np.histogram(FRB[shell], bins=SB_bins)
        SB_hist_in, bins = np.histogram(FRB_in[shell], bins=SB_bins)
        SB_hist_out, bins = np.histogram(FRB_out[shell], bins=SB_bins)
        SB_hist_neither, bins = np.histogram(FRB_neither[shell], bins=SB_bins)
        SB_hist_major, bins = np.histogram(FRB[shell & major], bins=SB_bins)
        SB_hist_minor, bins = np.histogram(FRB[shell & minor], bins=SB_bins)
        SB_hist_cold, bins = np.histogram(FRB_cold[shell], bins=SB_bins)
        SB_hist_hot, bins = np.histogram(FRB_hot[shell], bins=SB_bins)
        for i in range(len(bins)-1):
            pdf_row = [r_low, r_upp, bins[i], bins[i+1], SB_hist[i], SB_hist_in[i], SB_hist_out[i], SB_hist_neither[i], SB_hist_major[i], SB_hist_minor[i], SB_hist_cold[i], SB_hist_hot[i]]
            pdf_table.add_row(pdf_row)

    profile_table = set_table_units(profile_table)
    pdf_table = set_table_units(pdf_table, pdf=True)
    profile_table.write(prefix + 'Tables/' + snap + '_SB_profiles' + save_suffix + '.hdf5', path='all_data', serialize_meta=True, overwrite=True)
    pdf_table.write(prefix + 'Tables/' + snap + '_SB_pdf' + save_suffix + '.hdf5', path='all_data', serialize_meta=True, overwrite=True)

    full_profile = np.array(full_profile)
    major_profile = np.array(major_profile)
    minor_profile = np.array(minor_profile)
    inflow_profile = np.array(inflow_profile)
    outflow_profile = np.array(outflow_profile)
    neither_profile = np.array(neither_profile)
    cold_profile = np.array(cold_profile)
    hot_profile = np.array(hot_profile)

    if (not args.Aspera_limit):
        fig = plt.figure(figsize=(15,5), dpi=300)
        ax1 = fig.add_subplot(1,3,1)
        ax2 = fig.add_subplot(1,3,2)
        ax3 = fig.add_subplot(1,3,3)

        ax1.plot(rbin_centers, full_profile, 'k-', lw=2, label='Full profile')
        ax1.plot(rbin_centers, major_profile, color='#159615', ls='--', lw=2, label='Major axis')
        ax1.plot(rbin_centers, minor_profile, color='#c4379f', ls=':', lw=2, label='Minor axis')
        ax1.plot([0, 50], [np.log10(3.7e-19), np.log10(3.7e-19)], 'k:', lw=1)
        ax1.axis([0,50,-23,-18])
        ax1.set_xlabel('Radius [kpc]', fontsize=12)
        ax1.set_ylabel('log O VI SB [erg s$^{-1}$ cm$^{-2}$ arcsec$^{-2}$]', fontsize=12)
        ax1.tick_params(axis='both', which='both', direction='in', length=8, width=2, pad=5, labelsize=10, \
                    top=True, right=True)
        ax1.legend(loc=1, fontsize=12, frameon=False)
        ax2.plot(rbin_centers, full_profile, 'k-', lw=2, label='Full profile')
        ax2.plot(rbin_centers, inflow_profile, 'b--', lw=2, label='Inflowing gas')
        ax2.plot(rbin_centers, outflow_profile, 'r:', lw=2, label='Outflowing gas')
        ax2.plot(rbin_centers, neither_profile, 'k-.', alpha=0.5, lw=2, label='Slow-flow gas')
        ax2.plot([0, 50], [np.log10(3.7e-19), np.log10(3.7e-19)], 'k:', lw=1)
        ax2.axis([0,50,-23,-18])
        ax2.set_xlabel('Radius [kpc]', fontsize=12)
        ax2.set_ylabel('log O VI SB [erg s$^{-1}$ cm$^{-2}$ arcsec$^{-2}$]', fontsize=12)
        ax2.tick_params(axis='both', which='both', direction='in', length=8, width=2, pad=5, labelsize=10, \
                    top=True, right=True)
        ax2.legend(loc=1, fontsize=12, frameon=False)
        ax3.plot(rbin_centers, full_profile, 'k-', lw=2, label='Full profile')
        ax3.plot(rbin_centers, cold_profile, color="#984ea3", ls='--', lw=2, label='$T<10^5$ K')
        ax3.plot(rbin_centers, hot_profile, color='darkorange', ls=':', lw=2, label='$T>10^5$ K')
        ax3.plot([0, 50], [np.log10(3.7e-19), np.log10(3.7e-19)], 'k:', lw=1)
        ax3.axis([0,50,-23,-18])
        ax3.set_xlabel('Radius [kpc]', fontsize=12)
        ax3.set_ylabel('log O VI SB [erg s$^{-1}$ cm$^{-2}$ arcsec$^{-2}$]', fontsize=12)
        ax3.tick_params(axis='both', which='both', direction='in', length=8, width=2, pad=5, labelsize=10, \
                    top=True, right=True)
        ax3.legend(loc=1, fontsize=12, frameon=False)
        plt.subplots_adjust(left=0.07, bottom=0.12, top=0.96, right=0.98)
        plt.savefig(prefix + 'Profiles/' + snap + '_OVI_surface_brightness_profile_edge-on' + save_suffix + '.png')

def weighted_radial_profiles(ds, refine_box, snap):
    '''Makes radial profiles of density, temperature, metallicity, radial velocity, and cooling time weighted by either volume (for all gas) or O VI emissivity.'''

    profile_table = make_rad_profile_table()
    pdf_table = make_rad_pdf_table()

    sph = ds.sphere(center=ds.halo_center_kpc, radius=(50., 'kpc'))
    sph_inflow = sph.include_below(('gas','radial_velocity_corrected'), -100.)
    sph_outflow = sph.include_above(('gas','radial_velocity_corrected'), 100.)

    radius = sph['gas','radius_corrected'].in_units('kpc').v
    radius_in = sph_inflow['gas','radius_corrected'].in_units('kpc').v
    radius_out = sph_outflow['gas','radius_corrected'].in_units('kpc').v
    density = sph['gas','density'].in_units('g/cm**3').v
    density_in = sph_inflow['gas','density'].in_units('g/cm**3').v
    density_out = sph_outflow['gas','density'].in_units('g/cm**3').v
    temperature = sph['gas','temperature'].in_units('K').v
    temperature_in = sph_inflow['gas','temperature'].in_units('K').v
    temperature_out = sph_outflow['gas','temperature'].in_units('K').v
    metallicity = sph['gas','metallicity'].in_units('Zsun').v
    metallicity_in = sph_inflow['gas','metallicity'].in_units('Zsun').v
    metallicity_out = sph_outflow['gas','metallicity'].in_units('Zsun').v
    rv = sph['gas','radial_velocity_corrected'].in_units('km/s').v
    rv_in = sph_inflow['gas','radial_velocity_corrected'].in_units('km/s').v
    rv_out = sph_outflow['gas','radial_velocity_corrected'].in_units('km/s').v
    tcool = sph['gas','cooling_time'].in_units('Myr').v
    tcool_in = sph_inflow['gas','cooling_time'].in_units('Myr').v
    tcool_out = sph_outflow['gas','cooling_time'].in_units('Myr').v
    emissivity = np.log10(sph['gas','Emission_OVI'].in_units(emission_units_ALT).v)
    emissivity_inflow = np.log10(sph_inflow['gas','Emission_OVI'].in_units(emission_units_ALT).v)
    emissivity_outflow = np.log10(sph_outflow['gas','Emission_OVI'].in_units(emission_units_ALT).v)
    if (args.weight=='volume'):
        weight = np.log10(sph['gas','cell_volume'].in_units('kpc**3').v)
        weight_inflow = np.log10(sph_inflow['gas','cell_volume'].in_units('kpc**3').v)
        weight_outflow = np.log10(sph_outflow['gas','cell_volume'].in_units('kpc**3').v)
    else:
        weight = np.log10(sph['gas','cell_mass'].in_units('Msun').v)
        weight_inflow = np.log10(sph_inflow['gas','cell_mass'].in_units('Msun').v)
        weight_outflow = np.log10(sph_outflow['gas','cell_mass'].in_units('Msun').v)

    
    # Define the density cut between disk and CGM to vary smoothly between 1 and 0.1 between z = 0.5 and z = 0.25,
    # with it being 1 at higher redshifts and 0.1 at lower redshifts
    current_time = ds.current_time.in_units('Myr').v
    if (current_time<=7091.48):
        density_cut_factor = 20. - 19.*current_time/7091.48
    elif (current_time<=8656.88):
        density_cut_factor = 1.
    elif (current_time<=10787.12):
        density_cut_factor = 1. - 0.9*(current_time-8656.88)/2130.24
    else:
        density_cut_factor = 0.1
    cgm = (density < density_cut_factor * cgm_density_max)
    cgm_in = (density_in < density_cut_factor * cgm_density_max)
    cgm_out = (density_out < density_cut_factor * cgm_density_max)

    props = [np.log10(density), np.log10(temperature), np.log10(metallicity), rv, np.log10(tcool)]
    props_in = [np.log10(density_in), np.log10(temperature_in), np.log10(metallicity_in), rv_in, np.log10(tcool_in)]
    props_out = [np.log10(density_out), np.log10(temperature_out), np.log10(metallicity_out), rv_out, np.log10(tcool_out)]
    props_hist = [np.log10(density), np.log10(temperature), np.log10(metallicity), rv/100., np.log10(tcool)]
    props_hist_in = [np.log10(density_in), np.log10(temperature_in), np.log10(metallicity_in), rv_in/100., np.log10(tcool_in)]
    props_hist_out = [np.log10(density_out), np.log10(temperature_out), np.log10(metallicity_out), rv_out/100., np.log10(tcool_out)]

    ranges = [(-31,-23), (3,9), (-2.5,1.5), (-5,10), (-2,7)]

    rbins = np.linspace(0., 50., 26)
    for r in range(len(rbins) + 1):
        if (r==0):
            inn_r = 0.
            out_r = 20.
        elif (r==1):
            inn_r = 0.
            out_r = 50.
        else:
            inn_r = rbins[r-2]
            out_r = rbins[r-1]

        rad_bin = ((radius >= inn_r) & (radius < out_r))
        rad_bin_in = ((radius_in >= inn_r) & (radius_in < out_r))
        rad_bin_out = ((radius_out >= inn_r) & (radius_out < out_r))

        profile_row = [inn_r, out_r]
        for p in range(len(props)):
            if (len(weight[rad_bin])>0):
                all_mean, all_std = weighted_avg_and_std(props[p][rad_bin], weight[rad_bin])
                all_med = weighted_quantile(props[p][rad_bin], weight[rad_bin], [0.5])
            else:
                all_mean = np.nan
                all_med = np.nan
            if (len(weight_inflow[rad_bin_in])>0):
                all_mean_in, all_std_in = weighted_avg_and_std(props_in[p][rad_bin_in], weight_inflow[rad_bin_in])
                all_med_in = weighted_quantile(props_in[p][rad_bin_in], weight_inflow[rad_bin_in], [0.5])
            else:
                all_mean_in = np.nan
                all_med_in = np.nan
            if (len(weight_outflow[rad_bin_out])>0):
                all_mean_out, all_std_out = weighted_avg_and_std(props_out[p][rad_bin_out], weight_outflow[rad_bin_out])
                all_med_out = weighted_quantile(props_out[p][rad_bin_out], weight_outflow[rad_bin_out], [0.5])
            else:
                all_mean_out = np.nan
                all_med_out = np.nan
            profile_row += [all_mean, all_med, all_mean_in, all_med_in, all_mean_out, all_med_out]

            if (len(weight[(rad_bin) & (cgm)])>0):
                cgm_mean, cgm_std = weighted_avg_and_std(props[p][(rad_bin) & (cgm)], weight[(rad_bin) & (cgm)])
                cgm_med = weighted_quantile(props[p][(rad_bin) & (cgm)], weight[(rad_bin) & (cgm)], [0.5])
            else:
                cgm_mean = np.nan
                cgm_med = np.nan
            if (len(weight_inflow[(rad_bin_in) & (cgm_in)])>0):
                cgm_mean_in, cgm_std_in = weighted_avg_and_std(props_in[p][(rad_bin_in) & (cgm_in)], weight_inflow[(rad_bin_in) & (cgm_in)])
                cgm_med_in = weighted_quantile(props_in[p][(rad_bin_in) & (cgm_in)], weight_inflow[(rad_bin_in) & (cgm_in)], [0.5])
            else:
                cgm_mean_in = np.nan
                cgm_med_in = np.nan
            if (len(weight_outflow[(rad_bin_out) & (cgm_out)])>0):
                cgm_mean_out, cgm_std_out = weighted_avg_and_std(props_out[p][(rad_bin_out) & (cgm_out)], weight_outflow[(rad_bin_out) & (cgm_out)])
                cgm_med_out = weighted_quantile(props_out[p][(rad_bin_out) & (cgm_out)], weight_outflow[(rad_bin_out) & (cgm_out)], [0.5])
            else:
                cgm_mean_out = np.nan
                cgm_med_out = np.nan
            profile_row += [cgm_mean, cgm_med, cgm_mean_in, cgm_med_in, cgm_mean_out, cgm_med_out]

            if (len(emissivity[rad_bin])>0):
                OVI_mean, OVI_std = weighted_avg_and_std(props[p][rad_bin], emissivity[rad_bin])
                OVI_med = weighted_quantile(props[p][rad_bin], emissivity[rad_bin], [0.5])
            else:
                OVI_mean = np.nan
                OVI_med = np.nan
            if (len(emissivity_inflow[rad_bin_in])>0):
                OVI_mean_in, OVI_std_in = weighted_avg_and_std(props_in[p][rad_bin_in], emissivity_inflow[rad_bin_in])
                OVI_med_in = weighted_quantile(props_in[p][rad_bin_in], emissivity_inflow[rad_bin_in], [0.5])
            else:
                OVI_mean_in = np.nan
                OVI_med_in = np.nan
            if (len(emissivity_outflow[rad_bin_out])>0):
                OVI_mean_out, OVI_std_out = weighted_avg_and_std(props_out[p][rad_bin_out], emissivity_outflow[rad_bin_out])
                OVI_med_out = weighted_quantile(props_out[p][rad_bin_out], emissivity_outflow[rad_bin_out], [0.5])
            else:
                OVI_mean_out = np.nan
                OVI_med_out = np.nan
            profile_row += [OVI_mean, OVI_med, OVI_mean_in, OVI_med_in, OVI_mean_out, OVI_med_out]

            all_hist = np.histogram(props_hist[p][rad_bin], weights=weight[rad_bin], bins=50, range=ranges[p], density=True)
            all_hist = all_hist[0]
            all_hist_in = np.histogram(props_hist_in[p][rad_bin_in], weights=weight_inflow[rad_bin_in], bins=50, range=ranges[p], density=True)
            all_hist_in = all_hist_in[0]
            all_hist_out = np.histogram(props_hist_out[p][rad_bin_out], weights=weight_outflow[rad_bin_out], bins=50, range=ranges[p], density=True)
            all_hist_out = all_hist_out[0]
            cgm_hist = np.histogram(props_hist[p][(rad_bin) & (cgm)], weights=weight[(rad_bin) & (cgm)], bins=50, range=ranges[p], density=True)
            cgm_hist = cgm_hist[0]
            cgm_hist_in = np.histogram(props_hist_in[p][(rad_bin_in) & (cgm_in)], weights=weight_inflow[(rad_bin_in) & (cgm_in)], bins=50, range=ranges[p], density=True)
            cgm_hist_in = cgm_hist_in[0]
            cgm_hist_out = np.histogram(props_hist_out[p][(rad_bin_out) & (cgm_out)], weights=weight_outflow[(rad_bin_out) & (cgm_out)], bins=50, range=ranges[p], density=True)
            cgm_hist_out = cgm_hist_out[0]
            OVI_hist = np.histogram(props_hist[p][rad_bin], weights=emissivity[rad_bin], bins=50, range=ranges[p], density=True)
            OVI_hist = OVI_hist[0]
            OVI_hist_in = np.histogram(props_hist_in[p][rad_bin_in], weights=emissivity_inflow[rad_bin_in], bins=50, range=ranges[p], density=True)
            OVI_hist_in = OVI_hist_in[0]
            OVI_hist_out = np.histogram(props_hist_out[p][rad_bin_out], weights=emissivity_outflow[rad_bin_out], bins=50, range=ranges[p], density=True)
            OVI_hist_out = OVI_hist_out[0]

            if (p==0):
                hists = np.vstack([all_hist, all_hist_in, all_hist_out, cgm_hist, cgm_hist_in, cgm_hist_out, OVI_hist, OVI_hist_in, OVI_hist_out])
            else:
                hists = np.vstack([hists, all_hist, all_hist_in, all_hist_out, cgm_hist, cgm_hist_in, cgm_hist_out, OVI_hist, OVI_hist_in, OVI_hist_out])

        profile_table.add_row(profile_row)
        hists = np.transpose(hists)
        for h in range(len(hists)):
            pdf_table.add_row(np.hstack([inn_r, out_r, h, hists[h]]))

    profile_table = set_table_units(profile_table)
    pdf_table = set_table_units(pdf_table, pdf=True)
    profile_table.write(prefix + 'Tables/' + snap + '_radial_profiles' + save_suffix + '.hdf5', path='all_data', serialize_meta=True, overwrite=True)
    pdf_table.write(prefix + 'Tables/' + snap + '_radial_pdf' + save_suffix + '.hdf5', path='all_data', serialize_meta=True, overwrite=True)

    rbin_centers = np.linspace(1., 49., 25)
    ranges = [(-31,-23), (3,9), (-2.5,1.5), (-5,10), (-2,7)]
    profiles = Table.read(prefix + 'Tables/' + snap + '_radial_profiles' + save_suffix + '.hdf5', path='all_data')
    for i in ['', '_inflow', '_outflow']:
        fig = plt.figure(figsize=(15,8), dpi=250)
        ax1 = fig.add_subplot(2,3,1)
        ax2 = fig.add_subplot(2,3,2)
        ax3 = fig.add_subplot(2,3,3)
        ax4 = fig.add_subplot(2,3,4)
        ax5 = fig.add_subplot(2,3,5)

        ax1.plot(rbin_centers, profiles['cgm_den_med' + i][2:], 'k-', lw=2, label='CGM gas median')
        ax1.plot(rbin_centers, profiles['OVI_den_med' + i][2:], color='#db1d8f', ls='-', lw=2, label='O VI gas median')
        ax1.plot(rbin_centers, profiles['all_den_med' + i][2:], 'k-', alpha=0.5, lw=2, label='All gas median')
        ax1.plot(rbin_centers, profiles['cgm_den_mean' + i][2:], 'k:', lw=2, label='CGM gas mean')
        ax1.plot(rbin_centers, profiles['OVI_den_mean' + i][2:], color='#db1d8f', ls=':', lw=2, label='O VI gas mean')
        ax1.plot(rbin_centers, profiles['all_den_mean' + i][2:], 'k:', lw=2, alpha=0.5, label='All gas mean')
        ax1.axis([0,50,-31,-23])
        ax1.set_ylabel('log Gas Density [g/cm$^3$]', fontsize=14)
        ax1.tick_params(axis='both', which='both', direction='in', length=8, width=2, pad=5, labelsize=12, \
                    top=True, right=True, labelbottom=False)
        yticks = ax1.get_yticks()
        ax1.set_yticks(yticks[1:])
        ax1.legend(loc=4, fontsize=12, frameon=False, ncol=2, columnspacing=1.2, handlelength=1.5)

        ax2.plot(rbin_centers, profiles['cgm_temp_med' + i][2:], 'k-', lw=2, label='CGM gas median')
        ax2.plot(rbin_centers, profiles['OVI_temp_med' + i][2:], color='#db1d8f', ls='-', lw=2, label='O VI gas median')
        ax2.plot(rbin_centers, profiles['all_temp_med' + i][2:], 'k-', alpha=0.5, lw=2, label='All gas median')
        ax2.plot(rbin_centers, profiles['cgm_temp_mean' + i][2:], 'k:', lw=2, label='CGM gas mean')
        ax2.plot(rbin_centers, profiles['OVI_temp_mean' + i][2:], color='#db1d8f', ls=':', lw=2, label='O VI gas mean')
        ax2.plot(rbin_centers, profiles['all_temp_mean' + i][2:], 'k:', lw=2, alpha=0.5, label='All gas mean')
        ax2.axis([0,50,3,9])
        ax2.set_ylabel('log Gas Temperature [K]', fontsize=14)
        ax2.tick_params(axis='both', which='both', direction='in', length=8, width=2, pad=5, labelsize=12, \
                    top=True, right=True, labelbottom=False)
        yticks = ax2.get_yticks()
        ax2.set_yticks(yticks[1:])
        ax2.text(45, 8.5, halo_dict[args.halo], fontsize=14, va='top', ha='right')
        
        ax3.plot(rbin_centers, profiles['cgm_met_med' + i][2:], 'k-', lw=2, label='CGM gas median')
        ax3.plot(rbin_centers, profiles['OVI_met_med' + i][2:], color='#db1d8f', ls='-', lw=2, label='O VI gas median')
        ax3.plot(rbin_centers, profiles['all_met_med' + i][2:], 'k-', alpha=0.5, lw=2, label='All gas median')
        ax3.plot(rbin_centers, profiles['cgm_met_mean' + i][2:], 'k:', lw=2, label='CGM gas mean')
        ax3.plot(rbin_centers, profiles['OVI_met_mean' + i][2:], color='#db1d8f', ls=':', lw=2, label='O VI gas mean')
        ax3.plot(rbin_centers, profiles['all_met_mean' + i][2:], 'k:', lw=2, alpha=0.5, label='All gas mean')
        ax3.axis([0,50,-2.5,1.5])
        ax3.set_xlabel('Radius [kpc]', fontsize=14)
        ax3.set_ylabel(r'log Gas Metallicity [$Z_\odot$]', fontsize=14)
        ax3.tick_params(axis='both', which='both', direction='in', length=8, width=2, pad=5, labelsize=12, \
                    top=True, right=True)
        
        ax4.plot(rbin_centers, profiles['cgm_rv_med' + i][2:], 'k-', lw=2, label='CGM gas median')
        ax4.plot(rbin_centers, profiles['OVI_rv_med' + i][2:], color='#db1d8f', ls='-', lw=2, label='O VI gas median')
        ax4.plot(rbin_centers, profiles['all_rv_med' + i][2:], 'k-', alpha=0.5, lw=2, label='All gas median')
        ax4.plot(rbin_centers, profiles['cgm_rv_mean' + i][2:], 'k:', lw=2, label='CGM gas mean')
        ax4.plot(rbin_centers, profiles['OVI_rv_mean' + i][2:], color='#db1d8f', ls=':', lw=2, label='O VI gas mean')
        ax4.plot(rbin_centers, profiles['all_rv_mean' + i][2:], 'k:', lw=2, alpha=0.5, label='All gas mean')
        ax4.axis([0,50,-500,1000])
        ax4.set_xlabel('Radius [kpc]', fontsize=14)
        ax4.set_ylabel(r'Gas Radial Velocity [km/s]', fontsize=14)
        ax4.tick_params(axis='both', which='both', direction='in', length=8, width=2, pad=5, labelsize=12, \
                    top=True, right=True)
        ax4.text(45, 900, '$z=%.2f$\n%.2f Gyr' % (ds.get_parameter('CosmologyCurrentRedshift'), ds.current_time.in_units('Gyr')), fontsize=14, va='top', ha='right')
        
        ax5.plot(rbin_centers, profiles['cgm_tcool_med' + i][2:], 'k-', lw=2, label='CGM gas median')
        ax5.plot(rbin_centers, profiles['OVI_tcool_med' + i][2:], color='#db1d8f', ls='-', lw=2, label='O VI gas median')
        ax5.plot(rbin_centers, profiles['all_tcool_med' + i][2:], 'k-', alpha=0.5, lw=2, label='All gas median')
        ax5.plot(rbin_centers, profiles['cgm_tcool_mean' + i][2:], 'k:', lw=2, label='CGM gas mean')
        ax5.plot(rbin_centers, profiles['OVI_tcool_mean' + i][2:], color='#db1d8f', ls=':', lw=2, label='O VI gas mean')
        ax5.plot(rbin_centers, profiles['all_tcool_mean' + i][2:], 'k:', lw=2, alpha=0.5, label='All gas mean')
        ax5.axis([0,50,-2,7])
        ax5.set_xlabel('Radius [kpc]', fontsize=14)
        ax5.set_ylabel(r'Cooling Time [Myr]', fontsize=14)
        ax5.tick_params(axis='both', which='both', direction='in', length=8, width=2, pad=5, labelsize=12, \
                    top=True, right=True)
        
        plt.subplots_adjust(left=0.06, right=0.99, bottom=0.08, top=0.97, wspace=0.22, hspace=0.)
        plt.savefig(prefix + 'Profiles/' + snap + '_OVI_weighted_radial_profile' + i + save_suffix + '.png')
        plt.close()

def phase_plot(ds, refine_box, snap):
    '''Makes 2D histograms of O VI emissivity in density-temperature space.'''

    sph = ds.sphere(center=ds.halo_center_kpc, radius=(50., 'kpc'))

    phaseplot = yt.PhasePlot(sph, ('gas', 'number_density'), ('gas', 'temperature'), [('gas', 'Emission_OVI')], weight_field=('gas','cell_mass'))
    phaseplot.set_xlim(1e-6,2e1)
    phaseplot.set_ylim(5,2e8)
    phase = phaseplot.profile[('gas','Emission_OVI')].v
    phase[phase==0.0] = np.nan

    cmap = plt.get_cmap('cmr.lilac')
    cmap.set_under('k')
    cmap.set_bad('w')

    fig = plt.figure(figsize=(8.5,7), dpi=250)
    fig.subplots_adjust(left=0.12, bottom=0.12, top=0.97, right=0.82)
    ax = fig.add_subplot(1,1,1)
    im = ax.imshow(phase.T, cmap=cmap, origin='lower', extent=[np.log10(1e-6), np.log10(2e1), np.log10(5), np.log10(2e8)], norm=mcolors.LogNorm(vmin=1e-49, vmax=1e-39))
    ax.set_xlabel('Number density [cm$^{-3}$]', fontsize=22)
    ax.set_ylabel('Temperature [K]', fontsize=22)
    ax.tick_params(axis='both', which='both', direction='in', length=8, width=2, pad=5, labelsize=20, \
            top=True, right=True)
    xticks = [-6, -5, -4, -3, -2, -1, 0, 1]
    ax.set_xticks(xticks)
    ax.set_xticklabels([r"$10^{{{}}}$".format(k) for k in xticks])
    yticks = [1, 2, 3, 4, 5, 6, 7, 8]
    ax.set_yticks(yticks)
    ax.set_yticklabels([r"$10^{{{}}}$".format(k) for k in yticks])
    pos = ax.get_position()
    cax = fig.add_axes([pos.x1, pos.y0, 0.03, pos.height])  # [left, bottom, width, height]
    fig.colorbar(im, cax=cax, orientation='vertical')
    cax.tick_params(axis='both', which='both', direction='in', length=6, width=2, pad=5, labelsize=20, \
                    top=True, right=True)
    cax.text(5.5, 0.5, 'O VI Emissivity [erg s$^{-1}$ cm$^{-3}$ arcsec$^{-2}$]', fontsize=22, ha='center', va='center', rotation='vertical', transform=cax.transAxes)
    fig.savefig(prefix + 'Phase_Plots/' + snap + '_OVI_emission_phase' + save_suffix + '.png')

    #phaseplot.set_unit(('gas', 'number_density'), '1/cm**3')
    #phaseplot.set_unit(('gas', 'temperature'), 'K')
    #phaseplot.set_unit(('gas', 'Emission_OVI'), emission_units_ALT)
    #phaseplot.set_log(('gas','Emission_OVI'), False)
    #phaseplot.set_unit(('gas', 'cell_mass'), 'Msun')
    #phaseplot.set_colorbar_label(('gas', 'Emission_OVI'), 'O VI Emissivity [erg s$^{-1}$ cm$^{-3}$ arcsec$^{-2}$]')
    #phaseplot.set_zlim(('gas','Emission_OVI'), 1e-47, 1e-37)
    #phaseplot.set_colorbar_label(('gas', 'cell_mass'), 'Cell Mass [$M_\odot$]')
    #phaseplot.set_font_size(20)
    #phaseplot.annotate_timestamp(corner='upper_left', redshift=True, time=True, draw_inset_box=True)
    #phaseplot.save(prefix + 'Phase_Plots/' + snap + '_OVI_emission_phase' + save_suffix + '.png')

    #phaseplot = yt.PhasePlot(sph, ('gas', 'number_density'), ('gas', 'metallicity'), [('gas', 'Emission_OVI')], weight_field=None)
    #phaseplot.set_unit(('gas', 'number_density'), '1/cm**3')
    #phaseplot.set_unit(('gas', 'metallicity'), 'Zsun')
    #phaseplot.set_unit(('gas', 'Emission_OVI'), emission_units_ALT)
    #phaseplot.set_log(('gas','Emission_OVI'), False)
    #phaseplot.set_unit(('gas', 'cell_mass'), 'Msun')
    #phaseplot.set_colorbar_label(('gas', 'Emission_OVI'), 'O VI Emissivity [erg s$^{-1}$ cm$^{-3}$ arcsec$^{-2}$]')
    #phaseplot.set_zlim(('gas','Emission_OVI'), 1e-47, 1e-37)
    #phaseplot.set_colorbar_label(('gas', 'cell_mass'), 'Cell Mass [$M_\odot$]')
    #phaseplot.set_font_size(20)
    #phaseplot.annotate_timestamp(corner='upper_left', redshift=True, time=True, draw_inset_box=True)
    #phaseplot.save(prefix + 'Phase_Plots/' + snap + '_OVI_emission_phase-Z' + save_suffix + '.png')

def sb_profile_time_avg(halos, outs):
    '''Plots the median surface brightness profile with shading indicating IQR variation
    for all the halos and outputs given in 'halos' and 'outs'.'''

    SB_tablenames = ['all', 'inflow', 'outflow', 'neither', 'major', 'minor', 'hot', 'cold']
    SB_profiles = [[],[],[],[],[],[],[],[]]
    SB_meds = []
    SB_lows = []
    SB_upps = []
    for h in range(len(halos)):
        sb_table_loc = output_dir + 'ions_halo_00' + halos[h] + '/' + args.run + '/Tables/'
        for i in range(len(outs[h])):
            # Load the PDF of OVI emission
            snap = outs[h][i]
            sb_data = Table.read(sb_table_loc + snap + '_SB_profiles' + file_suffix + '.hdf5', path='all_data')
            for j in range(len(SB_tablenames)):
                SB_profiles[j].append(list(sb_data[SB_tablenames[j] + '_med'][2:]))
    for j in range(len(SB_profiles)):
        prof = np.array(SB_profiles[j])
        med = np.median(prof, axis=(0))
        SB_meds.append(med)
        low = np.percentile(prof, 25, axis=(0))
        upp = np.percentile(prof, 75, axis=(0))
        SB_lows.append(low)
        SB_upps.append(upp)

    fig = plt.figure(figsize=(16,4),dpi=250)
    ax1 = fig.add_subplot(1,3,1)
    ax2 = fig.add_subplot(1,3,2)
    ax3 = fig.add_subplot(1,3,3)
    rbins = np.linspace(0., 50., 26)
    rbin_centers = rbins[:-1] + 0.5*np.diff(rbins)

    ax1.plot(rbin_centers, SB_meds[0], 'k-', lw=2, alpha=0.5, label='All gas')
    ax1.fill_between(rbin_centers, y1=SB_lows[0], y2=SB_upps[0], color='k', alpha=0.2)
    # I stupidly flipped major and minor axes in the SB tables calculation, so this plotting flips them back:
    ax1.plot(rbin_centers, SB_meds[5], color='#79d41e', ls='--', lw=2, label='Major axis')
    ax1.fill_between(rbin_centers, y1=SB_lows[5], y2=SB_upps[5], color='#79d41e', alpha=0.4)
    ax1.plot(rbin_centers, SB_meds[4], color='#c4379f', ls=':', lw=2, label='Minor axis')
    ax1.fill_between(rbin_centers, y1=SB_lows[4], y2=SB_upps[4], color='#c4379f', alpha=0.4)
    #ax1.plot([0, 50], [np.log10(3.7e-19), np.log10(3.7e-19)], 'k:', lw=1)
    ax1.axis([0,50,-22,-18])
    yticks = [-22,-21,-20,-19,-18]
    ax1.set_yticks(yticks)
    ax1.set_yticklabels(r"$10^{{{}}}$".format(y) for y in yticks)
    ax1.set_xlabel('Radius [kpc]', fontsize=16)
    ax1.set_ylabel('O VI SB [erg s$^{-1}$ cm$^{-2}$ arcsec$^{-2}$]', fontsize=16)
    ax1.tick_params(axis='both', which='both', direction='in', length=8, width=2, pad=5, labelsize=14, \
                top=True, right=True)
    ax1.legend(loc=1, fontsize=16, frameon=False)
    ax2.plot(rbin_centers, SB_meds[0], 'k-', lw=2, alpha=0.5, label='All gas')
    ax2.fill_between(rbin_centers, y1=SB_lows[0], y2=SB_upps[0], color='k', alpha=0.2)
    ax2.plot(rbin_centers, SB_meds[1], color="b", ls='--', lw=2, label='Inflowing gas')
    ax2.fill_between(rbin_centers, y1=SB_lows[1], y2=SB_upps[1], color="b", alpha=0.4)
    ax2.plot(rbin_centers, SB_meds[2], color='r', ls=':', lw=2, label='Outflowing gas')
    ax2.fill_between(rbin_centers, y1=SB_lows[2], y2=SB_upps[2], color='r', alpha=0.4)
    ax2.plot(rbin_centers, SB_meds[3], color='#027020', ls='-.', lw=2, label='Slow-flow gas')
    ax2.fill_between(rbin_centers, y1=SB_lows[3], y2=SB_upps[3], color='#027020', alpha=0.4)
    #ax2.plot([0, 50], [np.log10(3.7e-19), np.log10(3.7e-19)], 'k:', lw=1)
    ax2.axis([0,50,-24,-18])
    yticks = [-24,-23,-22,-21,-20,-19,-18]
    ax2.set_yticks(yticks)
    ax2.set_yticklabels(r"$10^{{{}}}$".format(y) for y in yticks)
    ax2.set_xlabel('Radius [kpc]', fontsize=16)
    ax2.set_ylabel('O VI SB [erg s$^{-1}$ cm$^{-2}$ arcsec$^{-2}$]', fontsize=16)
    ax2.tick_params(axis='both', which='both', direction='in', length=8, width=2, pad=5, labelsize=14, \
                top=True, right=True)
    ax2.legend(loc=1, fontsize=16, frameon=False)
    ax3.plot(rbin_centers, SB_meds[0], 'k-', lw=2, alpha=0.5, label='All gas')
    ax3.fill_between(rbin_centers, y1=SB_lows[0], y2=SB_upps[0], color='k', alpha=0.2)
    ax3.plot(rbin_centers, SB_meds[7], color="#984ea3", ls='--', lw=2, label='Cold gas')
    ax3.fill_between(rbin_centers, y1=SB_lows[7], y2=SB_upps[7], color="#984ea3", alpha=0.4)
    ax3.plot(rbin_centers, SB_meds[6], color='darkorange', ls=':', lw=2, label='Hot gas')
    ax3.fill_between(rbin_centers, y1=SB_lows[6], y2=SB_upps[6], color='darkorange', alpha=0.4)
    ax3.legend(loc=1, fontsize=16, frameon=False)
    ax3.axis([0,50,-24,-18])
    yticks = [-24,-23,-22,-21,-20,-19,-18]
    ax3.set_yticks(yticks)
    ax3.set_yticklabels(r"$10^{{{}}}$".format(y) for y in yticks)
    ax3.set_xlabel('Radius [kpc]', fontsize=16)
    ax3.set_ylabel('O VI SB [erg s$^{-1}$ cm$^{-2}$ arcsec$^{-2}$]', fontsize=16)
    ax3.tick_params(axis='both', which='both', direction='in', length=8, width=2, pad=5, labelsize=14, \
                top=True, right=True)
    plt.subplots_adjust(left=0.06, bottom=0.14, top=0.96, right=0.99, wspace=0.25)
    plt.savefig(prefix + '/OVI_surface_brightness_profile_edge-on_time-avg' + save_suffix + '.png')

def sb_profile_nofdbk_compare(snap):
    '''Plots the median surface brightness profiles for fiducial Tempest and the
    feedback-10-track run of Tempest, and for the constant-metallicity versions
    of these runs.'''

    fig = plt.figure(figsize=(6,5),dpi=250)
    ax = fig.add_subplot(1,1,1)
    rbins = np.linspace(0., 50., 26)
    rbin_centers = rbins[:-1] + 0.5*np.diff(rbins)
    runs = ['nref11c_nref9f', 'feedback-10-track']
    colors = ['k','#8e9091']
    colors_inflow = ['b', "#04C8CF"]
    labels = ['Fiducial', 'No feedback']
    labels_inflow = ['Fiducial inflows', 'No feedback inflows']
    for h in range(len(runs)):
        sb_table_loc = output_dir + 'ions_halo_008508/' + runs[h] + '/Tables/'
        if (runs[h]=='nref11c_nref9f'): snap_file = snap
        if (runs[h]=='feedback-10-track'): snap_file = 'DD' + str(int(snap[2:]) + 93)
        sb_data = Table.read(sb_table_loc + snap_file + '_SB_profiles' + file_suffix + '.hdf5', path='all_data')
        ax.plot(rbin_centers, sb_data['all_med'][2:], color=colors[h], lw=2, ls='-', label=labels[h])
        ax.plot(rbin_centers, sb_data['inflow_med'][2:], color=colors_inflow[h], lw=2, ls='--', label=labels_inflow[h])
    ax.set_xlabel('Radius [kpc]', fontsize=16)
    ax.set_ylabel('log O VI SB [erg s$^{-1}$ cm$^{-2}$ arcsec$^{-2}$]', fontsize=16)
    ax.tick_params(axis='both', which='both', direction='in', length=8, width=2, pad=5, labelsize=14, \
                top=True, right=True)
    ax.legend(loc=1, fontsize=16, frameon=False)
    ax.axis([0,50,-24,-18])
    ax.set_title('Simulated Metallicity', fontsize=16)

    fig.tight_layout()
    plt.savefig(prefix + '/' + snap + '_OVI_surface_brightness_profile_edge-on_fid-nofdbk' + save_suffix + '.png')

def sb_profile_nofdbk_compare_time_avg(outs):
    '''Plots the median surface brightness profiles for fiducial Tempest and the
    feedback-10-track run of Tempest, averaged over the snapshots in outs, and
    also for the constant-metallicity versions of these runs.'''

    fig = plt.figure(figsize=(6,5),dpi=250)
    ax = fig.add_subplot(1,1,1)
    rbins = np.linspace(0., 50., 26)
    rbin_centers = rbins[:-1] + 0.5*np.diff(rbins)
    runs = ['nref11c_nref9f', 'feedback-10-track']
    colors = ['k','#8e9091']
    colors_inflow = ['b', "#04C8CF"]
    labels = ['Fiducial', 'No feedback']
    labels_inflow = ['Fiducial inflows', 'No feedback inflows']


    for h in range(len(runs)):
        SB_meds = []
        SB_lows = []
        SB_upps = []
        sb_table_loc = output_dir + 'ions_halo_008508/' + runs[h] + '/Tables/'
        SB_tablenames = ['all', 'inflow']
        SB_profiles = [[],[]]
        for i in range(len(outs)):
            # Load the PDF of OVI emission
            snap = outs[i]
            if (runs[h]=='nref11c_nref9f'): snap_file = snap
            if (runs[h]=='feedback-10-track'): snap_file = 'DD' + str(int(snap[2:]) + 93)
            sb_data = Table.read(sb_table_loc + snap_file + '_SB_profiles' + file_suffix + '.hdf5', path='all_data')
            for j in range(len(SB_tablenames)):
                SB_profiles[j].append(list(sb_data[SB_tablenames[j] + '_med'][2:]))
        for j in range(len(SB_profiles)):
            prof = np.array(SB_profiles[j])
            med = np.median(prof, axis=(0))
            SB_meds.append(med)
            low = np.percentile(prof, 25, axis=(0))
            upp = np.percentile(prof, 75, axis=(0))
            SB_lows.append(low)
            SB_upps.append(upp)

        ax.plot(rbin_centers, SB_meds[0], color=colors[h], ls='-', lw=2, label=labels[h])
        ax.fill_between(rbin_centers, y1=SB_lows[0], y2=SB_upps[0], color=colors[h], alpha=0.25)
        ax.plot(rbin_centers, SB_meds[1], color=colors_inflow[h], ls='--', lw=2, label=labels_inflow[h])
        ax.fill_between(rbin_centers, y1=SB_lows[1], y2=SB_upps[1], color=colors_inflow[h], alpha=0.25)

    ax.set_xlabel('Radius [kpc]', fontsize=16)
    ax.set_ylabel('O VI SB [erg s$^{-1}$ cm$^{-2}$ arcsec$^{-2}$]', fontsize=16)
    ax.tick_params(axis='both', which='both', direction='in', length=8, width=2, pad=5, labelsize=14, \
                top=True, right=True)
    ax.axis([0,50,-24,-18])
    yticks = [-24,-23,-22,-21,-20,-19,-18]
    ax.set_yticks(yticks)
    ax.set_yticklabels(r"$10^{{{}}}$".format(y) for y in yticks)
    if (args.constant_Z):
        ax.set_title('Constant Metallicity', fontsize=16)
    else:
        ax.set_title('Simulated Metallicity', fontsize=16)
        ax.legend(loc=1, fontsize=16, frameon=False)

    fig.tight_layout()
    plt.savefig(prefix + '/OVI_surface_brightness_profile_edge-on_fid-nofdbk_time-avg' + save_suffix + '.png')

def surface_brightness_time_histogram(outs):
    '''Makes a plot of surface brightness histograms vs time for the outputs in 'outs'. Requires
    surface brightness tables to have already been created for the outputs plotted using the
    surface_brightness_profile function.'''

    halo_c_v = Table.read(halo_c_v_name, format='ascii')
    timelist = halo_c_v['col4']
    snaplist = halo_c_v['col3']
    zlist = halo_c_v['col2']
    sfr_data = Table.read(sfr_name, format='ascii')
    sfr_snap = sfr_data['col1']
    sfr = sfr_data['col3']
    sfr_time = []
    for i in range(len(sfr_snap)):
        sfr_time.append(float(timelist[snaplist==sfr_snap[i]])/1000.)

    if (args.run=='feedback-10-track'):
        outs_Temp = make_output_list('DD0967-DD2427', output_step=args.output_step)

    z_list = []
    sb_hists = []
    time_list = []
    sb_hists_sections = []
    meds = []
    means = []
    meds_sections = []
    means_sections = []
    med_above_limit = []
    med_fdbk = []
    med_above_limit_sections = []
    sections = ['inflow','outflow','major','minor']
    for j in range(len(sections)):
        sb_hists_sections.append([])
        meds_sections.append([])
        means_sections.append([])
        med_above_limit_sections.append([])
    for i in range(len(outs)):
        snap = outs[i]
        time_list.append(float(timelist[snaplist==snap])/1000.)
        z_list.append(float(zlist[snaplist==snap]))
        sb_data = Table.read(table_loc + snap + '_SB_pdf' + file_suffix + '.hdf5', path='all_data')
        sb_stats = Table.read(table_loc + snap + '_SB_profiles' + file_suffix + '.hdf5', path='all_data')
        if (args.run=='feedback-10-track'):
            snap_temp = outs_Temp[i]
            sb_stats_Temp = Table.read(output_dir + '/ions_halo_008508/nref11c_nref9f/Tables/' + snap_temp + '_SB_profiles' + file_suffix + '.hdf5', path='all_data')
        sb_limit = Table.read(table_loc + snap + '_SB_profiles_sym-vel_and_temp_Aspera-limit.hdf5', path='all_data')
        radial_range = (sb_data['inner_radius']==0.) & (sb_data['outer_radius']==20.)
        #sb_hists.append(sb_data['all'][radial_range]/6.4e5)     # Normalize by number of pixels in FRB (800x800)
        sb_hists.append(sb_data['all'][radial_range]/1.02e5)     # Normalize by number of pixels in FRB between -20 and 20 kpc in both directions (320x320)
        sb_hists[-1][sb_hists[-1]==np.nan] = 1e-5
        meds.append(sb_stats['all_med'][(sb_stats['inner_radius']==0.) & (sb_stats['outer_radius']==20.)])
        means.append(sb_stats['all_mean'][(sb_stats['inner_radius']==0.) & (sb_stats['outer_radius']==20.)])
        med_above_limit.append(sb_limit['all_med'][(sb_limit['inner_radius']==0.) & (sb_limit['outer_radius']==20.)])
        if (args.run=='feedback-10-track'): med_fdbk.append(sb_stats_Temp['all_med'][(sb_stats_Temp['inner_radius']==0.) & (sb_stats_Temp['outer_radius']==20.)])
        for j in range(len(sections)):
            #sb_hists_sections[j].append(sb_data[sections[j]][radial_range]/6.4e5)
            sb_hists_sections[j].append(sb_data[sections[j]][radial_range]/1.02e5)
            sb_hists_sections[j][-1][sb_hists[-1]==np.nan] = 1e-5
            meds_sections[j].append(sb_stats[sections[j] + '_med'][(sb_stats['inner_radius']==0.) & (sb_stats['outer_radius']==20.)])
            means_sections[j].append(sb_stats[sections[j] + '_mean'][(sb_stats['inner_radius']==0.) & (sb_stats['outer_radius']==20.)])
            med_above_limit_sections[j].append(sb_limit[sections[j] + '_med'][(sb_limit['inner_radius']==0.) & (sb_limit['outer_radius']==20.)])

    sb_hists = np.transpose(np.array(sb_hists))
    for j in range(len(sections)):
        sb_hists_sections[j] = np.transpose(np.array(sb_hists_sections[j]))
    sb_bins = (sb_data['lower_SB'][radial_range] + sb_data['upper_SB'][radial_range])/2.

    cmap = cmr.get_sub_cmap('cmr.flamingo', 0.2, 1.)
    cmap.set_bad(cmap(0.))

    fig = plt.figure(figsize=(7,6), dpi=300)
    gs = GridSpec(nrows=2, ncols=1, height_ratios=[3, 1])
    plt.subplots_adjust(left=0.18, bottom=0.1, top=0.9, right=0.84, hspace=0.)
    ax = fig.add_subplot(gs[0])
    im = ax.imshow(np.log10(sb_hists), origin='lower', aspect='auto', extent=[time_list[0], time_list[-1], sb_bins[0], sb_bins[-1]], cmap=cmap, vmin=-5, vmax=-0.5)
    ax.plot(time_list, meds, 'k-', lw=2, label='Median')
    ax.plot(time_list, means, 'k:', lw=2, label='Mean')
    ax.plot(time_list, med_above_limit, color='#00F5FF', ls='--', lw=2, label='Median above $10^{-19}$')
    if (args.run=='feedback-10-track'): ax.plot(time_list, med_fdbk, color='b', ls='--', lw=2, label='Median with feedback')
    ax.plot(time_list, np.zeros(len(time_list))-19, 'k-', lw=1)
    #ax.text(6.25, -16.25, halo_dict[args.halo], fontsize=16, ha='left', va='top', color='w')
    ax.axis([time_list[0], time_list[-1], -23, sb_bins[-1]])
    ax.set_ylabel('O VI SB\n[erg s$^{-1}$ cm$^{-2}$ arcsec$^{-2}$]', fontsize=16)
    ax.tick_params(axis='both', which='both', direction='in', length=8, width=2, pad=5, labelsize=14, \
                top=False, right=True)
    ax.set_xticklabels([])
    yticks = [-22,-21,-20,-19,-18,-17]
    ax.set_yticks(yticks)
    ax.set_yticklabels(r"$10^{{{}}}$".format(y) for y in yticks)
    ax.legend(loc=1, fontsize=14, ncols=2, columnspacing=1.7)

    pos = ax.get_position()
    cax = fig.add_axes([pos.x1, pos.y0, 0.03, pos.height])  # [left, bottom, width, height]
    fig.colorbar(im, cax=cax)
    cax.tick_params(axis='both', which='both', top=False, right=True, labelsize=14, direction='in', length=8, width=2, pad=5)
    pos_cax = cax.get_position()
    yticks = [-5,-4,-3,-2,-1]
    cax.set_yticks(yticks)
    cax.set_yticklabels(r"$10^{{{}}}$".format(y) for y in yticks)
    cax.text(pos_cax.x1 + 3.5, pos_cax.y0 + pos_cax.height/2. - 0.1, 'Pixel Fraction', fontsize=16, ha='center', va='center', rotation=90, transform=cax.transAxes)

    z_list.reverse()
    time_list.reverse()
    time_func = IUS(z_list, time_list)
    time_list.reverse()

    ax2 = ax.twiny()
    ax2.tick_params(axis='x', which='both', direction='in', length=8, width=2, pad=5, labelsize=14, \
      top=True)
    x0, x1 = ax.get_xlim()
    z_ticks = np.array([1,.7,.5,.3,.2,.1,0])
    last_z = np.where(z_ticks >= z_list[0])[0][-1]
    first_z = np.where(z_ticks <= z_list[-1])[0][0]
    z_ticks = z_ticks[first_z:last_z+1]
    tick_pos = [z for z in time_func(z_ticks)]
    tick_labels = ['%.2f' % (z) for z in z_ticks]
    ax2.set_xlim(x0,x1)
    ax2.set_xticks(tick_pos)
    ax2.set_xticklabels(tick_labels)
    ax2.set_xlabel('Redshift', fontsize=16)

    ax_sfr = fig.add_subplot(gs[1])
    ax_sfr.plot(sfr_time, sfr, 'k-', lw=1)
    ax_sfr.axis([np.min(time_list), np.max(time_list), 0.1, 100])
    ax_sfr.set_yscale('log')
    ax_sfr.set_xlabel('Time [Gyr]', fontsize=16)
    ax_sfr.set_ylabel('SFR\n' + r'[$M_\odot$/yr]', fontsize=16)
    ax_sfr.tick_params(axis='both', which='both', direction='in', length=8, width=2, pad=5, labelsize=14, \
                top=True, right=True)
    #ax_sfr.set_yticklabels(ax_sfr.get_yticklabels()[:-1])

    plt.savefig(prefix + 'OVI_SB_histogram_vs_time' + save_suffix + '.png')
    plt.close()

    section_labels = ['Inflow','Outflow','Major axis','Minor axis']
    for j in range(len(sections)):
        fig = plt.figure(figsize=(9,8), dpi=300)
        gs = GridSpec(nrows=2, ncols=1, height_ratios=[3, 1])
        plt.subplots_adjust(left=0.1, bottom=0.07, top=0.92, right=0.87, hspace=0.)
        ax = fig.add_subplot(gs[0])
        im = ax.imshow(np.log10(sb_hists_sections[j]), origin='lower', aspect='auto', extent=[time_list[0], time_list[-1], sb_bins[0], sb_bins[-1]], cmap=cmap, vmin=-5, vmax=-0.5)
        ax.plot(time_list, meds_sections[j], 'k-', lw=2, label='Median')
        ax.plot(time_list, means_sections[j], 'k:', lw=2, label='Mean')
        #ax.plot(time_list, med_above_limit_sections[j], color='#00F5FF', ls='--', lw=2, label='Median above $10^{-19}$')
        ax.plot(time_list, np.zeros(len(time_list))-19, 'k-', lw=1)
        ax.text(6.25, -16.25, halo_dict[args.halo]+'\n'+section_labels[j], fontsize=14, ha='left', va='top', color='w')
        ax.set_ylabel('log O VI SB [erg s$^{-1}$ cm$^{-2}$ arcsec$^{-2}$]', fontsize=16)
        ax.axis([time_list[0], time_list[-1], sb_bins[0], sb_bins[-1]])
        ax.tick_params(axis='both', which='both', direction='in', length=8, width=2, pad=5, labelsize=12, \
                top=False, right=True)
        ax.set_xticklabels([])
        ax.set_yticks([-22,-21,-20,-19,-18,-17])
        ax.legend(loc=1, fontsize=14, ncols=2)

        pos = ax.get_position()
        cax = fig.add_axes([pos.x1, pos.y0, 0.03, pos.height])  # [left, bottom, width, height]
        fig.colorbar(im, cax=cax)
        cax.tick_params(axis='both', which='both', top=False, right=True, labelsize=12, direction='in', length=8, width=2, pad=5)
        pos_cax = cax.get_position()
        cax.text(pos_cax.x1 + 2.7, pos_cax.y0 + pos_cax.height/2. - 0.1, 'log Pixel Fraction', fontsize=16, ha='center', va='center', rotation=90, transform=cax.transAxes)

        ax2 = ax.twiny()
        ax2.tick_params(axis='x', which='both', direction='in', length=8, width=2, pad=5, labelsize=12, \
        top=True)
        x0, x1 = ax.get_xlim()
        z_ticks = np.array([1,.7,.5,.3,.2,.1,0])
        last_z = np.where(z_ticks >= z_list[0])[0][-1]
        first_z = np.where(z_ticks <= z_list[-1])[0][0]
        z_ticks = z_ticks[first_z:last_z+1]
        tick_pos = [z for z in time_func(z_ticks)]
        tick_labels = ['%.2f' % (z) for z in z_ticks]
        ax2.set_xlim(x0,x1)
        ax2.set_xticks(tick_pos)
        ax2.set_xticklabels(tick_labels)
        ax2.set_xlabel('Redshift', fontsize=16)

        ax_sfr = fig.add_subplot(gs[1])
        ax_sfr.plot(sfr_time, sfr, 'k-', lw=1)
        ax_sfr.axis([np.min(time_list), np.max(time_list), 0.1, 100])
        ax_sfr.set_yscale('log')
        ax_sfr.set_xlabel('Time [Gyr]', fontsize=16)
        ax_sfr.set_ylabel(r'SFR [$M_\odot$/yr]', fontsize=16)
        ax_sfr.tick_params(axis='both', which='both', direction='in', length=8, width=2, pad=5, labelsize=12, \
                    top=True, right=True)
        #ax_sfr.set_yticklabels(ax_sfr.get_yticklabels()[:-1])

        plt.savefig(prefix + 'OVI_SB_histogram_vs_time_' + sections[j] + save_suffix + '.png')
        plt.close()

def sb_time_histogram_allhalos(halos, outs):
    '''Makes a plot of surface brightness histograms vs time for the outputs in 'outs' for
    halos in 'halos'. Requires
    surface brightness tables to have already been created for the outputs plotted using the
    surface_brightness_profile function.'''

    save_dir = '/Users/clochhaas/Documents/Research/FOGGIE/Papers/Aspera Predictions/Figures/'

    fig = plt.figure(figsize=(12,15), dpi=250)
    outer = mpl.gridspec.GridSpec(3, 2, hspace=0.03, wspace=0.03, left=0.13, right=0.87, top=0.95, bottom=0.08)

    for h in range(len(halos)):
        axes = mpl.gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=outer[h//2, h%2], hspace=0., height_ratios=[3,1])
        ax1 = fig.add_subplot(axes[0, 0])
        ax_sfr = fig.add_subplot(axes[1, 0])

        halo_c_v_name = code_path + '/halo_infos/00' + halos[h] + '/nref11c_nref9f/halo_c_v'
        sfr_name = code_path + '/halo_infos/00' + halos[h] + '/nref11c_nref9f/sfr'
        table_loc = output_dir + 'ions_halo_00' + halos[h] + '/nref11c_nref9f/Tables/'

        halo_c_v = Table.read(halo_c_v_name, format='ascii')
        timelist = halo_c_v['col4']
        snaplist = halo_c_v['col3']
        zlist = halo_c_v['col2']
        sfr_data = Table.read(sfr_name, format='ascii')
        sfr_snap = sfr_data['col1']
        sfr = sfr_data['col3']

        z_list = []
        sfr_list = []
        sb_hists = []
        time_list = []
        Z_list = []
        den_list = []
        temp_list = []
        meds = []
        means = []
        med_above_limit = []
        for i in range(len(outs[h])):
            snap = outs[h][i]
            time_list.append(float(timelist[snaplist==snap][0])/1000.)
            z_list.append(float(zlist[snaplist==snap][0]))
            sfr_list.append(float(sfr[sfr_snap==snap][0]))
            sb_data = Table.read(table_loc + snap + '_SB_pdf' + file_suffix + '.hdf5', path='all_data')
            sb_stats = Table.read(table_loc + snap + '_SB_profiles' + file_suffix + '.hdf5', path='all_data')
            if (halos[h]=='2878'): sb_limit = Table.read(table_loc + snap + '_SB_profiles_sym-vel_and_temp_Aspera-limit.hdf5', path='all_data')
            else: sb_limit = Table.read(table_loc + snap + '_SB_profiles_new-table_Aspera-limit.hdf5', path='all_data')
            radial_range = (sb_data['inner_radius']==0.) & (sb_data['outer_radius']==20.)
            sb_hists.append(sb_data['all'][radial_range]/1.02e5)     # Normalize by number of pixels in FRB between -20 and 20 kpc in both directions (320x320)
            sb_hists[-1][sb_hists[-1]==np.nan] = 1e-5
            meds.append(sb_stats['all_med'][(sb_stats['inner_radius']==0.) & (sb_stats['outer_radius']==20.)])
            means.append(sb_stats['all_mean'][(sb_stats['inner_radius']==0.) & (sb_stats['outer_radius']==20.)])
            med_above_limit.append(sb_limit['all_med'][(sb_limit['inner_radius']==0.) & (sb_limit['outer_radius']==20.)])
            cgm_data = Table.read(table_loc + snap + '_radial_profiles_vol-weighted.hdf5', path='all_data')
            radial_range_cgm = (cgm_data['inner_radius']==0.) & (cgm_data['outer_radius']==20.)
            Z_list.append(cgm_data['cgm_met_med'][radial_range_cgm][0])
            den_list.append(cgm_data['cgm_den_med'][radial_range_cgm][0])
            temp_list.append(cgm_data['cgm_temp_med'][radial_range_cgm][0])


        sb_hists = np.transpose(np.array(sb_hists))
        sb_bins = (sb_data['lower_SB'][radial_range] + sb_data['upper_SB'][radial_range])/2.

        cmap = cmr.get_sub_cmap('cmr.flamingo', 0.2, 1.)
        cmap.set_bad(cmap(0.))

        im = ax1.imshow(np.log10(sb_hists), origin='lower', aspect='auto', extent=[time_list[0], time_list[-1], sb_bins[0], sb_bins[-1]], cmap=cmap, vmin=-5, vmax=-0.5)
        ax1.plot(time_list, meds, 'k-', lw=2, label='Median')
        ax1.plot(time_list, means, 'k:', lw=2, label='Mean')
        ax1.plot(time_list, med_above_limit, color='#00F5FF', ls='--', lw=2, label='Median above $10^{-19}$')
        ax1.plot(time_list, np.zeros(len(time_list))-19, 'k-', lw=1)
        ax1.text(6.25, -16.25, halo_dict[halos[h]], fontsize=18, ha='left', va='top', color='w')
        ax1.axis([time_list[0], time_list[-1], -23, sb_bins[-1]])
        ax1.set_xticklabels([])

        if (h%2==0):
            ax1.set_ylabel('O VI SB\n[erg s$^{-1}$ cm$^{-2}$ arcsec$^{-2}$]', fontsize=18)
            yticks = [-22,-21,-20,-19,-18,-17]
            ax1.set_yticks(yticks)
            ax1.set_yticklabels(r"$10^{{{}}}$".format(y) for y in yticks)
            ax1.tick_params(axis='both', which='both', direction='in', length=8, width=2, pad=5, labelsize=16, \
                    top=False, right=True)
        else:
            ax1.tick_params(axis='both', which='both', direction='in', length=8, width=2, pad=5, labelsize=16, \
                    top=False, right=True, labelleft=False)

        if (h==4): ax1.legend(loc=2, fontsize=18, ncols=3, bbox_to_anchor=(0.1,-0.5))

        if (h%2==1):
            pos = ax1.get_position()
            cax = fig.add_axes([pos.x1, pos.y0, 0.03, pos.height])  # [left, bottom, width, height]
            fig.colorbar(im, cax=cax)
            cax.tick_params(axis='both', which='both', top=False, right=True, labelsize=16, direction='in', length=8, width=2, pad=5)
            pos_cax = cax.get_position()
            ticks = [-5,-4,-3,-2,-1]
            cax.set_yticks(ticks)
            cax.set_yticklabels(r"$10^{{{}}}$".format(y) for y in ticks)
            if (h==1): cax.text(pos_cax.x1 + 2.7, pos_cax.y0 + pos_cax.height/2.-0.35, 'Pixel Fraction', fontsize=18, ha='center', va='center', rotation=90, transform=cax.transAxes)
            if (h==3): cax.text(pos_cax.x1 + 2.7, pos_cax.y0 + pos_cax.height/2.-0.1, 'Pixel Fraction', fontsize=18, ha='center', va='center', rotation=90, transform=cax.transAxes)
            if (h==5): cax.text(pos_cax.x1 + 2.7, pos_cax.y0 + pos_cax.height/2.+0.2, 'Pixel Fraction', fontsize=18, ha='center', va='center', rotation=90, transform=cax.transAxes)

        z_list.reverse()
        time_list.reverse()
        time_func = IUS(z_list, time_list)
        time_list.reverse()

        ax2 = ax1.twiny()
        x0, x1 = ax1.get_xlim()
        z_ticks = np.array([1,.7,.5,.3,.2,.1,0])
        last_z = np.where(z_ticks >= z_list[0])[0][-1]
        first_z = np.where(z_ticks <= z_list[-1])[0][0]
        z_ticks = z_ticks[first_z:last_z+1]
        tick_pos = [z for z in time_func(z_ticks)]
        tick_labels = ['%.1f' % (z) for z in z_ticks]
        ax2.set_xlim(x0,x1)
        ax2.set_xticks(tick_pos)
        if (h//2==0):
            if (h==0): 
                ticks_without_last = tick_labels
                ticks_without_last[-1] = ''
                ax2.set_xticklabels(ticks_without_last)
            else: ax2.set_xticklabels(tick_labels)
            ax2.set_xlabel('Redshift', fontsize=18)
            ax2.tick_params(axis='x', which='both', direction='in', length=8, width=2, pad=5, labelsize=16, \
                            top=True)
        else:
            ax2.tick_params(axis='x', which='both', direction='in', length=8, width=2, pad=5, labelsize=16, \
                            top=True, labeltop=False)

        ax_sfr.plot(time_list, sfr_list, 'k-', lw=1)
        ax_sfr.axis([np.min(time_list), np.max(time_list), 0.1, 100])
        ax_sfr.set_yscale('log')
        if (h//2==2):
            ax_sfr.set_xlabel('Time [Gyr]', fontsize=18)
            if (h%2==0):
                ax_sfr.set_ylabel('SFR\n' + r'[$M_\odot$/yr]', fontsize=18)
                ax_sfr.tick_params(axis='both', which='both', direction='in', length=8, width=2, pad=5, labelsize=16, \
                    top=True, right=True)
            else:
                ax_sfr.tick_params(axis='both', which='both', direction='in', length=8, width=2, pad=5, labelsize=16, \
                    top=True, right=True, labelleft=False)
        else:
            if (h%2==0):
                ax_sfr.set_ylabel('SFR\n' + r'[$M_\odot$/yr]', fontsize=18)
                ax_sfr.tick_params(axis='both', which='both', direction='in', length=8, width=2, pad=5, labelsize=16, \
                    top=True, right=True, labelbottom=False)
            else:
                ax_sfr.tick_params(axis='both', which='both', direction='in', length=8, width=2, pad=5, labelsize=16, \
                    top=True, right=True, labelleft=False, labelbottom=False)
                
        ax_cgm = ax_sfr.twinx()
        # CGM metallicity vs time
        #ax_cgm.plot(time_list, Z_list, 'b-', lw=1)
        #ax_cgm.set_ylim([-1.75,0.5])
        #ax_cgm.set_yticks([-1.5,-1,-0.5,0.,0.5])
        #ax_cgm.set_yticklabels(['', '-1', '', '0', ''])
        #if (h%2==0):
            #ax_cgm.tick_params(axis='both', which='both', direction='in', length=8, width=2, pad=5, labelsize=16, \
                    #top=False, bottom=False, labelbottom=False, colors='blue', right=True, labelleft=False, labelright=False, left=False)
        #else:
            #ax_cgm.tick_params(axis='both', which='both', direction='in', length=8, width=2, pad=5, labelsize=16, \
                    #top=False, bottom=False, labelbottom=False, colors='blue', right=True, labelleft=False, labelright=True, left=False)
            #ax_cgm.set_ylabel('log Z\n' + r'[$Z_\odot$]', fontsize=18, color='blue')

        # CGM density vs time
        #ax_cgm.plot(time_list, den_list, 'b-', lw=1)
        #ax_cgm.set_ylim([-28.4,-26.8])
        #ax_cgm.set_yticks([-28,-27.5,-27])
        #ax_cgm.set_yticklabels(['-28', '', '-27'])
        #if (h%2==0):
            #ax_cgm.tick_params(axis='both', which='both', direction='in', length=8, width=2, pad=5, labelsize=16, \
                    #top=False, bottom=False, labelbottom=False, colors='blue', right=True, labelleft=False, labelright=False, left=False)
        #else:
            #ax_cgm.tick_params(axis='both', which='both', direction='in', length=8, width=2, pad=5, labelsize=16, \
                    #top=False, bottom=False, labelbottom=False, colors='blue', right=True, labelleft=False, labelright=True, left=False)
            #ax_cgm.set_ylabel(r'log $\rho$' + '\n' + r'[g/cm$^3$]', fontsize=18, color='blue')

        # CGM temperature vs time
        ax_cgm.plot(time_list, temp_list, 'b-', lw=1)
        ax_cgm.set_ylim([5,6.75])
        ax_cgm.set_yticks([5,5.5,6,6.5])
        ax_cgm.set_yticklabels(['5', '', '6', ''])
        if (h%2==0):
            ax_cgm.tick_params(axis='both', which='both', direction='in', length=8, width=2, pad=5, labelsize=16, \
                    top=False, bottom=False, labelbottom=False, colors='blue', right=True, labelleft=False, labelright=False, left=False)
        else:
            ax_cgm.tick_params(axis='both', which='both', direction='in', length=8, width=2, pad=5, labelsize=16, \
                    top=False, bottom=False, labelbottom=False, colors='blue', right=True, labelleft=False, labelright=True, left=False)
            ax_cgm.set_ylabel(r'log $\rho$' + '\n' + r'[g/cm$^3$]', fontsize=18, color='blue')

    plt.savefig(prefix + 'OVI_SB_histogram_vs_time' + save_suffix + '.png')
    plt.close()

def sb_time_radius(outs):
    '''Makes a plot of surface brightness profiles vs time and radius for the outputs in 'outs'. Requires
    surface brightness tables to have already been created for the outputs plotted using the
    surface_brightness_profile function.'''

    halo_c_v = Table.read(halo_c_v_name, format='ascii')
    timelist = halo_c_v['col4']
    snaplist = halo_c_v['col3']
    zlist = halo_c_v['col2']
    sfr_data = Table.read(sfr_name, format='ascii')
    sfr_snap = sfr_data['col1']
    sfr = sfr_data['col3']
    sfr_time = []
    for i in range(len(sfr_snap)):
        sfr_time.append(float(timelist[snaplist==sfr_snap[i]])/1000.)

    time_list = []
    sb_profiles = []
    sb_profiles_sections = []
    z_list = []
    sections = ['inflow','outflow','major','minor']
    for j in range(len(sections)):
        sb_profiles_sections.append([])
    for i in range(len(outs)):
        snap = outs[i]
        time_list.append(float(timelist[snaplist==snap])/1000.)
        z_list.append(float(zlist[snaplist==snap]))
        sb_stats = Table.read(table_loc + snap + '_SB_profiles' + file_suffix + '.hdf5', path='all_data')
        sb_profiles.append(sb_stats['all_mean'][2:])
        if (i==0): radius_list = (sb_stats['inner_radius'][2:] + sb_stats['outer_radius'][2:])/2.
        for j in range(len(sections)):
            sb_profiles_sections[j].append(sb_stats[sections[j] + '_mean'][2:])

    sb_profiles = np.transpose(np.array(sb_profiles))
    for j in range(len(sections)):
        sb_profiles_sections[j] = np.transpose(np.array(sb_profiles_sections[j]))

    cmap = cmr.get_sub_cmap('cmr.torch', 0., 0.9)
    cmap.set_bad(cmap(0.))

    fig = plt.figure(figsize=(9,8), dpi=300)
    gs = GridSpec(nrows=2, ncols=1, height_ratios=[3, 1])
    plt.subplots_adjust(left=0.08, bottom=0.07, top=0.92, right=0.85, hspace=0.)
    ax = fig.add_subplot(gs[0])
    im = ax.imshow(sb_profiles, origin='lower', aspect='auto', extent=[time_list[0], time_list[-1], radius_list[0], radius_list[-1]], cmap=cmap, vmin=-20, vmax=-16)
    ax.text(6.25, 47, halo_dict[args.halo], fontsize=16, ha='left', va='top', color='w')
    ax.set_ylabel('Galactocentric Radius [kpc]', fontsize=16)
    ax.tick_params(axis='both', which='both', direction='in', length=8, width=2, pad=5, labelsize=12, \
                top=False, right=True)
    ax.set_xticklabels([])

    pos = ax.get_position()
    cax = fig.add_axes([pos.x1, pos.y0, 0.03, pos.height])  # [left, bottom, width, height]
    fig.colorbar(im, cax=cax)
    cax.tick_params(axis='both', which='both', top=False, right=True, labelsize=12, direction='in', length=8, width=2, pad=5)
    pos_cax = cax.get_position()
    cax.text(pos_cax.x1 + 3.3, pos_cax.y0 + pos_cax.height/2. - 0.1, 'log O VI SB [ergs s$^{-1}$ cm$^{-2}$ arcsec$^{-2}$]', fontsize=16, ha='center', va='center', rotation=90, transform=cax.transAxes)

    z_list.reverse()
    time_list.reverse()
    time_func = IUS(z_list, time_list)
    time_list.reverse()

    ax2 = ax.twiny()
    ax2.tick_params(axis='x', which='both', direction='in', length=8, width=2, pad=5, labelsize=12, \
      top=True)
    x0, x1 = ax.get_xlim()
    z_ticks = np.array([1,.7,.5,.3,.2,.1,0])
    last_z = np.where(z_ticks >= z_list[0])[0][-1]
    first_z = np.where(z_ticks <= z_list[-1])[0][0]
    z_ticks = z_ticks[first_z:last_z+1]
    tick_pos = [z for z in time_func(z_ticks)]
    tick_labels = ['%.2f' % (z) for z in z_ticks]
    ax2.set_xlim(x0,x1)
    ax2.set_xticks(tick_pos)
    ax2.set_xticklabels(tick_labels)
    ax2.set_xlabel('Redshift', fontsize=16)

    ax_sfr = fig.add_subplot(gs[1])
    ax_sfr.plot(sfr_time, sfr, 'k-', lw=1)
    ax_sfr.axis([np.min(time_list), np.max(time_list), 0, 75])
    ax_sfr.set_xlabel('Time [Gyr]', fontsize=16)
    ax_sfr.set_ylabel(r'SFR [$M_\odot$/yr]', fontsize=16)
    ax_sfr.tick_params(axis='both', which='both', direction='in', length=8, width=2, pad=5, labelsize=12, \
                top=True, right=True)
    ax_sfr.set_yticklabels(ax_sfr.get_yticklabels()[:-1])

    plt.savefig(prefix + 'OVI_SB_profile_vs_time-radius' + save_suffix + '.png')
    plt.close()

    section_labels = ['Inflow','Outflow','Major axis','Minor axis']
    for j in range(len(sections)):
        fig = plt.figure(figsize=(9,8), dpi=300)
        gs = GridSpec(nrows=2, ncols=1, height_ratios=[3, 1])
        plt.subplots_adjust(left=0.08, bottom=0.07, top=0.92, right=0.85, hspace=0.)
        ax = fig.add_subplot(gs[0])
        im = ax.imshow(sb_profiles_sections[j], origin='lower', aspect='auto', extent=[time_list[0], time_list[-1], radius_list[0], radius_list[-1]], cmap=cmap, vmin=-20, vmax=-16)
        ax.text(6.25, 47, halo_dict[args.halo]+'\n'+section_labels[j], fontsize=14, ha='left', va='top', color='w')
        ax.set_ylabel('Galactocentric Radius [kpc]', fontsize=16)
        ax.tick_params(axis='both', which='both', direction='in', length=8, width=2, pad=5, labelsize=12, \
                top=False, right=True)
        ax.set_xticklabels([])

        pos = ax.get_position()
        cax = fig.add_axes([pos.x1, pos.y0, 0.03, pos.height])  # [left, bottom, width, height]
        fig.colorbar(im, cax=cax)
        cax.tick_params(axis='both', which='both', top=False, right=True, labelsize=12, direction='in', length=8, width=2, pad=5)
        pos_cax = cax.get_position()
        cax.text(pos_cax.x1 + 3.3, pos_cax.y0 + pos_cax.height/2. - 0.1, 'log O VI SB [ergs s$^{-1}$ cm$^{-2}$ arcsec$^{-2}$]', fontsize=16, ha='center', va='center', rotation=90, transform=cax.transAxes)

        ax2 = ax.twiny()
        ax2.tick_params(axis='x', which='both', direction='in', length=8, width=2, pad=5, labelsize=12, \
        top=True)
        x0, x1 = ax.get_xlim()
        z_ticks = np.array([1,.7,.5,.3,.2,.1,0])
        last_z = np.where(z_ticks >= z_list[0])[0][-1]
        first_z = np.where(z_ticks <= z_list[-1])[0][0]
        z_ticks = z_ticks[first_z:last_z+1]
        tick_pos = [z for z in time_func(z_ticks)]
        tick_labels = ['%.2f' % (z) for z in z_ticks]
        ax2.set_xlim(x0,x1)
        ax2.set_xticks(tick_pos)
        ax2.set_xticklabels(tick_labels)
        ax2.set_xlabel('Redshift', fontsize=16)

        ax_sfr = fig.add_subplot(gs[1])
        ax_sfr.plot(sfr_time, sfr, 'k-', lw=1)
        ax_sfr.axis([np.min(time_list), np.max(time_list), 0, 60])
        ax_sfr.set_xlabel('Time [Gyr]', fontsize=16)
        ax_sfr.set_ylabel(r'SFR [$M_\odot$/yr]', fontsize=16)
        ax_sfr.tick_params(axis='both', which='both', direction='in', length=8, width=2, pad=5, labelsize=12, \
                    top=True, right=True)
        ax_sfr.set_yticklabels(ax_sfr.get_yticklabels()[:-1])

        plt.savefig(prefix + 'OVI_SB_profile_vs_time-radius_' + sections[j] + save_suffix + '.png')
        plt.close()

def sb_vs_sfr(halos, outs):
    '''Plots the median surface brightness vs. SFR for all the halos and outputs listed in halos and outs.'''

    fig = plt.figure(figsize=(7,5),dpi=250)
    ax = fig.add_subplot(1,1,1)
    #halo_names = ['Tempest', 'Tempest (no feedback)', 'Squall', 'Maelstrom', 'Blizzard', 'Hurricane']
    halo_names = ['Tempest', 'Squall', 'Maelstrom', 'Blizzard', 'Hurricane', 'Cyclone']
    halo_colors = ['#8FDC97','#6A0136','#188FA7','#CC3F0C', '#D5A021', '#73599e']
    for h in range(len(halo_names)):
        if (halo_names[h]=='Tempest (no feedback)'):
            sb_table_loc = output_dir + 'ions_halo_008508/feedback-10-track/Tables/'
            sfr_table = Table.read(code_path + 'halo_infos/008508/feedback-10-track/sfr', format='ascii')
        else:
            sb_table_loc = output_dir + 'ions_halo_00' + halos[h] + '/' + args.run + '/Tables/'
            sfr_table = Table.read(code_path + 'halo_infos/00' + halos[h] + '/' + args.run + '/sfr', format='ascii')
        SFR_list = []
        SB_med_list = []
        for i in range(len(outs[h])):
            # Load the PDF of OVI emission
            snap = outs[h][i]
            sfr = sfr_table['col3'][sfr_table['col1']==snap][0]
            if (args.Aspera_limit):
                if (halo_names[h]=='Cyclone'):
                    sb_data = Table.read(sb_table_loc + snap + '_SB_profiles_sym-vel_and_temp' + file_suffix + '.hdf5', path='all_data')
                else:
                    sb_data = Table.read(sb_table_loc + snap + '_SB_profiles_new-table' + file_suffix + '.hdf5', path='all_data')
            else:
                sb_data = Table.read(sb_table_loc + snap + '_SB_profiles' + file_suffix + '.hdf5', path='all_data')
            radial_range = (sb_data['inner_radius']==0.) & (sb_data['outer_radius']==20.)
            SB_med_list.append(sb_data['all_med'][radial_range][0])
            SFR_list.append(sfr)

        SB_med_list = np.array(SB_med_list)
        SFR_list = np.array(SFR_list)
        color_list = [halo_colors[h]] * len(SFR_list)
        alphas = np.linspace(0.1, 1.0, len(SFR_list))
        color_alpha_list = [mcolors.to_rgba(c, alpha=a) for c, a in zip(color_list, alphas)]
        ax.scatter(SFR_list, SB_med_list, marker='.', s=80, fc=color_alpha_list, ec='none')
        
    ax.grid(True, color='#c2c2c2', linestyle='-', linewidth=1)
    ax.set_axisbelow(True)
    ax.set_xlabel(r'Star formation rate [$M_\odot$/yr]', fontsize=16)
    ax.set_ylabel('Median O VI SB [erg s$^{-1}$ cm$^{-2}$ arcsec$^{-2}$]', fontsize=16)
    ax.set_xscale('log')
    ax.tick_params(axis='both', which='both', direction='in', length=8, width=2, pad=5, labelsize=14, \
                top=True, right=True)
    if (args.Aspera_limit):
        ax.axis([0.1,100,-18.85,-18])
        yticks = [-18.8,-18.6,-18.4,-18.2,-18]
    else:
        ax.axis([0.1,100,-21.,-18.5])
        yticks = [-21,-20.5,-20,-19.5,-19,-18.5]
    ax.set_yticks(yticks)
    ax.set_yticklabels(r"$10^{{{:.1f}}}$".format(y) for y in yticks)
    ax.text(0.05, 0.95, halo_names[0], fontsize=16, weight='semibold', c=halo_colors[0], ha='left', va='top', transform=ax.transAxes)
    ax.text(0.05, 0.89, halo_names[1], fontsize=16, weight='semibold', c=halo_colors[1], ha='left', va='top', transform=ax.transAxes)
    ax.text(0.05, 0.83, halo_names[2], fontsize=16, weight='semibold', c=halo_colors[2], ha='left', va='top', transform=ax.transAxes)
    ax.text(0.32, 0.95, halo_names[3], fontsize=16, weight='semibold', c=halo_colors[3], ha='left', va='top', transform=ax.transAxes)
    ax.text(0.32, 0.89, halo_names[4], fontsize=16, weight='semibold', c=halo_colors[4], ha='left', va='top', transform=ax.transAxes)
    ax.text(0.32, 0.83, halo_names[5], fontsize=16, weight='semibold', c=halo_colors[5], ha='left', va='top', transform=ax.transAxes)
    fig.subplots_adjust(left=0.18, bottom=0.13, top=0.95, right=0.97)
    fig.savefig(prefix + 'OVI_SB_vs_SFR' + save_suffix + '.png')

def sb_vs_Mh(halos, outs):
    '''Plots the median surface brightness vs. halo mass for all the halos and outputs listed in halos and outs.'''

    fig = plt.figure(figsize=(7,5),dpi=250)
    ax = fig.add_subplot(1,1,1)
    halo_names = ['Tempest', 'Squall', 'Maelstrom', 'Blizzard', 'Hurricane', 'Cyclone']
    halo_colors = ['#8FDC97','#6A0136','#188FA7','#CC3F0C', '#D5A021', '#73599e']
    for h in range(len(halo_names)):
        if (halo_names[h]=='Tempest (no feedback)'):
            sb_table_loc = output_dir + 'ions_halo_008508/feedback-10-track/Tables/'
            mvir_table = Table.read(code_path + 'halo_infos/008508/feedback-10-track/rvir_masses.hdf5', path='all_data')
        else:
            sb_table_loc = output_dir + 'ions_halo_00' + halos[h] + '/' + args.run + '/Tables/'
            mvir_table = Table.read(code_path + 'halo_infos/00' + halos[h] + '/' + args.run + '/rvir_masses.hdf5', path='all_data')
        mvir_list = []
        SB_med_list = []
        for i in range(len(outs[h])):
            # Load the PDF of OVI emission
            snap = outs[h][i]
            sb_data = Table.read(sb_table_loc + snap + '_SB_profiles' + file_suffix + '.hdf5', path='all_data')
            radial_range = (sb_data['inner_radius']==0.) & (sb_data['outer_radius']==20.)
            SB_med_list.append(sb_data['all_med'][radial_range][0])
            mvir_list.append(mvir_table['total_mass'][mvir_table['snapshot']==snap][0])

        SB_med_list = np.array(SB_med_list)
        mvir_list = np.array(mvir_list)
        color_list = [halo_colors[h]] * len(mvir_list)
        alphas = np.linspace(0.1, 1.0, len(mvir_list))
        color_alpha_list = [mcolors.to_rgba(c, alpha=a) for c, a in zip(color_list, alphas)]
        ax.scatter(np.log10(mvir_list), SB_med_list, marker='.', s=80, fc=color_alpha_list, ec='none')
        
    ax.grid(True, color='#c2c2c2', linestyle='-', linewidth=1)
    ax.set_axisbelow(True)
    ax.set_xlabel(r'Halo mass [$M_\odot$]', fontsize=16)
    ax.set_ylabel('Median O VI SB [erg s$^{-1}$ cm$^{-2}$ arcsec$^{-2}$]', fontsize=16)
    ax.tick_params(axis='both', which='both', direction='in', length=8, width=2, pad=5, labelsize=14, \
                top=True, right=True)
    if (args.Aspera_limit): ax.axis([11.5,12.25,-19,-18])
    else: ax.axis([11.5,12.3,-21.,-18.5])
    xticks = [11.6,11.8,12,12.2]
    yticks = [-21,-20.5,-20,-19.5,-19,-18.5]
    ax.set_xticks(xticks)
    ax.set_xticklabels(r"$10^{{{:.1f}}}$".format(x) for x in xticks)
    ax.set_yticks(yticks)
    ax.set_yticklabels(r"$10^{{{:.1f}}}$".format(y) for y in yticks)
    ax.text(0.05, 0.95, halo_names[0], fontsize=16, weight='semibold', c=halo_colors[0], ha='left', va='top', transform=ax.transAxes)
    ax.text(0.05, 0.89, halo_names[1], fontsize=16, weight='semibold', c=halo_colors[1], ha='left', va='top', transform=ax.transAxes)
    ax.text(0.05, 0.83, halo_names[2], fontsize=16, weight='semibold', c=halo_colors[2], ha='left', va='top', transform=ax.transAxes)
    ax.text(0.32, 0.95, halo_names[3], fontsize=16, weight='semibold', c=halo_colors[3], ha='left', va='top', transform=ax.transAxes)
    ax.text(0.32, 0.89, halo_names[4], fontsize=16, weight='semibold', c=halo_colors[4], ha='left', va='top', transform=ax.transAxes)
    ax.text(0.32, 0.83, halo_names[5], fontsize=16, weight='semibold', c=halo_colors[5], ha='left', va='top', transform=ax.transAxes)
    fig.subplots_adjust(left=0.18, bottom=0.13, top=0.95, right=0.97)
    plt.savefig(prefix + 'OVI_SB_vs_Mh' + save_suffix + '.png')

def sb_vs_den(halos, outs):
    '''Plots the median surface brightness vs. average CGM density for all the halos and outputs listed in halos and outs.'''

    fig = plt.figure(figsize=(7,5),dpi=250)
    ax = fig.add_subplot(1,1,1)
    halo_names = ['Tempest', 'Squall', 'Maelstrom', 'Blizzard', 'Hurricane', 'Cyclone']
    halo_colors = ['#8FDC97','#6A0136','#188FA7','#CC3F0C', '#D5A021', '#73599e']
    for h in range(len(halo_names)):
        if (halo_names[h]=='Tempest (no feedback)'):
            sb_table_loc = output_dir + 'ions_halo_008508/feedback-10-track/Tables/'
        else:
            sb_table_loc = output_dir + 'ions_halo_00' + halos[h] + '/' + args.run + '/Tables/'
        den_list = []
        SB_med_list = []
        for i in range(len(outs[h])):
            # Load the PDF of OVI emission
            snap = outs[h][i]
            sb_data = Table.read(sb_table_loc + snap + '_SB_profiles' + file_suffix + '.hdf5', path='all_data')
            den_data = Table.read(sb_table_loc + snap + '_radial_profiles_vol-weighted.hdf5', path='all_data')
            radial_range = (sb_data['inner_radius']==0.) & (sb_data['outer_radius']==20.)
            #radial_range_den = (den_data['col1']<=50.)
            radial_range_den = (den_data['inner_radius']==0.) & (den_data['outer_radius']==20.)
            SB_med_list.append(sb_data['all_med'][radial_range][0])
            den_list.append(den_data['cgm_den_med'][radial_range_den][0])

        SB_med_list = np.array(SB_med_list)
        den_list = np.array(den_list)
        color_list = [halo_colors[h]] * len(den_list)
        alphas = np.linspace(0.1, 1.0, len(den_list))
        color_alpha_list = [mcolors.to_rgba(c, alpha=a) for c, a in zip(color_list, alphas)]
        ax.scatter(den_list, SB_med_list, marker='.', s=80, fc=color_alpha_list, ec='none')
        
    ax.grid(True, color='#c2c2c2', linestyle='-', linewidth=1)
    ax.set_axisbelow(True)
    ax.set_xlabel('Median CGM Density [g cm$^{-3}$]', fontsize=16)
    ax.set_ylabel('Median O VI SB [erg s$^{-1}$ cm$^{-2}$ arcsec$^{-2}$]', fontsize=16)
    ax.tick_params(axis='both', which='both', direction='in', length=8, width=2, pad=5, labelsize=14, \
                top=True, right=True)
    if (args.Aspera_limit): ax.axis([-28.5,-26.75,-19,-18])
    else: ax.axis([-28.5,-26.75,-21.,-18.5])
    xticks = [-28.4,-28,-27.6,-27.2,-26.8]
    yticks = [-21,-20.5,-20,-19.5,-19,-18.5]
    ax.set_xticks(xticks)
    ax.set_xticklabels(r"$10^{{{:.1f}}}$".format(x) for x in xticks)
    ax.set_yticks(yticks)
    ax.set_yticklabels(r"$10^{{{:.1f}}}$".format(y) for y in yticks)
    #ax.text(0.05, 0.95, halo_names[0], fontsize=14, weight='semibold', c=halo_colors[0], ha='left', va='top', transform=ax.transAxes)
    #ax.text(0.05, 0.89, halo_names[1], fontsize=14, weight='semibold', c=halo_colors[1], ha='left', va='top', transform=ax.transAxes)
    #ax.text(0.05, 0.83, halo_names[2], fontsize=14, weight='semibold', c=halo_colors[2], ha='left', va='top', transform=ax.transAxes)
    #ax.text(0.28, 0.95, halo_names[3], fontsize=14, weight='semibold', c=halo_colors[3], ha='left', va='top', transform=ax.transAxes)
    #ax.text(0.28, 0.89, halo_names[4], fontsize=14, weight='semibold', c=halo_colors[4], ha='left', va='top', transform=ax.transAxes)
    #ax.text(0.28, 0.83, halo_names[5], fontsize=14, weight='semibold', c=halo_colors[5], ha='left', va='top', transform=ax.transAxes)
    fig.subplots_adjust(left=0.18, bottom=0.13, top=0.95, right=0.97)
    plt.savefig(prefix + 'OVI_SB_vs_den' + save_suffix + '.png')

def den_vs_time(halos, outs):
    '''Plots the average CGM density vs. time and redshift for all the halos and outputs listed in halos and outs.'''

    #fig = plt.figure(figsize=(9,8), dpi=300)
    #gs = GridSpec(nrows=2, ncols=1, height_ratios=[3, 1])
    #plt.subplots_adjust(left=0.13, bottom=0.07, top=0.92, right=0.96, hspace=0.)
    #ax = fig.add_subplot(gs[0])
    #ax_sfr = fig.add_subplot(gs[1])
    fig = plt.figure(figsize=(7,5), dpi=250)
    ax = fig.add_subplot(1,1,1)
    #halo_names = ['Tempest', 'Squall', 'Maelstrom', 'Blizzard', 'Hurricane', 'Cyclone', 'Tempest\n(no feedback)']
    #halo_colors = ['#8FDC97','#6A0136','#188FA7','#CC3F0C', '#D5A021', '#b04f86', 'gray']
    halo_names = ['Tempest', 'Squall', 'Maelstrom', 'Blizzard', 'Hurricane', 'Cyclone']
    halo_colors = ['#8FDC97','#6A0136','#188FA7','#CC3F0C', '#D5A021', '#73599e']
    for h in range(len(halo_names)):
        if (halo_names[h]=='Tempest\n(no feedback)'):
            sb_table_loc = output_dir + 'ions_halo_008508/feedback-10-track/Tables/'
            halo_c_v_name = code_path + 'halo_infos/008508/feedback-10-track/halo_c_v'
            sfr_name = code_path + 'halo_infos/008508/feedback-10-track/sfr'
        else:
            sb_table_loc = output_dir + 'ions_halo_00' + halos[h] + '/' + args.run + '/Tables/'
            halo_c_v_name = code_path + 'halo_infos/00' + halos[h] + '/' + args.run + '/halo_c_v'
            sfr_name = code_path + 'halo_infos/00' + halos[h] + '/' + args.run + '/sfr'
        halo_c_v = Table.read(halo_c_v_name, format='ascii')
        timelist = halo_c_v['col4']
        snaplist = halo_c_v['col3']
        zlist = halo_c_v['col2']
        sfr_data = Table.read(sfr_name, format='ascii')
        sfr_snap = sfr_data['col1']
        sfr = sfr_data['col3']
        sfr_time = []
        for i in range(len(sfr_snap)):
            sfr_time.append(float(timelist[snaplist==sfr_snap[i]][0])/1000.)
        den_list = []
        time_list = []
        z_list = []
        for i in range(len(outs[h])):
            # Load the PDF of OVI emission
            snap = outs[h][i]
            den_data = Table.read(sb_table_loc + snap + '_radial_profiles_vol-weighted.hdf5', path='all_data')
            radial_range_den = (den_data['inner_radius']==0.) & (den_data['outer_radius']==20.)
            den_list.append(den_data['cgm_den_med'][radial_range_den][0])
            time_list.append(float(timelist[snaplist==snap][0])/1000.)
            z_list.append(float(zlist[snaplist==snap][0]))

        ax.scatter(time_list, den_list, marker='.', s=80, ec='none', fc=halo_colors[h], ls='-', lw=0.5)
        #ax_sfr.plot(sfr_time, sfr, ls='-', lw=1, color=halo_colors[h])
        if (h==0): ax.axis([np.min(time_list), np.max(time_list), -28.5, -26.75])

    #ax_sfr.axis([np.min(time_list), np.max(time_list), 0, 75])
    #ax_sfr.set_xlabel('Time [Gyr]', fontsize=16)
    #ax_sfr.set_ylabel(r'SFR [$M_\odot$/yr]', fontsize=16)
    #ax_sfr.tick_params(axis='both', which='both', direction='in', length=8, width=2, pad=5, labelsize=12, \
                #top=True, right=True)
    #ax_sfr.set_yticklabels(ax_sfr.get_yticklabels()[:-1])
        
    ax.grid(True, color='#c2c2c2', linestyle='-', linewidth=1)
    ax.set_axisbelow(True)
    ax.set_ylabel('Median CGM Density [g cm$^{-3}$]', fontsize=16)
    ax.tick_params(axis='both', which='both', direction='in', length=8, width=2, pad=5, labelsize=14, \
                top=False, right=True)
    ax.set_xticklabels([])
    yticks = [-28.4,-28,-27.6,-27.2,-26.8]
    ax.set_yticks(yticks)
    ax.set_yticklabels(r"$10^{{{:.1f}}}$".format(y) for y in yticks)
    ax.text(0.72, 0.95, halo_names[0], fontsize=16, weight='semibold', c=halo_colors[0], ha='right', va='top', transform=ax.transAxes)
    ax.text(0.72, 0.89, halo_names[1], fontsize=16, weight='semibold', c=halo_colors[1], ha='right', va='top', transform=ax.transAxes)
    ax.text(0.72, 0.83, halo_names[2], fontsize=16, weight='semibold', c=halo_colors[2], ha='right', va='top', transform=ax.transAxes)
    ax.text(0.95, 0.95, halo_names[3], fontsize=16, weight='semibold', c=halo_colors[3], ha='right', va='top', transform=ax.transAxes)
    ax.text(0.95, 0.89, halo_names[4], fontsize=16, weight='semibold', c=halo_colors[4], ha='right', va='top', transform=ax.transAxes)
    ax.text(0.95, 0.83, halo_names[5], fontsize=16, weight='semibold', c=halo_colors[5], ha='right', va='top', transform=ax.transAxes)
    #ax.text(0.95, 0.77, halo_names[6], fontsize=16, weight='semibold', c=halo_colors[6], ha='right', va='top', transform=ax.transAxes)

    z_list.reverse()
    time_list.reverse()
    time_func = IUS(z_list, time_list)
    time_list.reverse()

    ax2 = ax.twiny()
    ax2.tick_params(axis='x', which='both', direction='in', length=8, width=2, pad=5, labelsize=12, \
    top=True)
    x0, x1 = ax.get_xlim()
    z_ticks = np.array([1,.7,.5,.3,.2,.1,0])
    last_z = np.where(z_ticks >= z_list[0])[0][-1]
    first_z = np.where(z_ticks <= z_list[-1])[0][0]
    z_ticks = z_ticks[first_z:last_z+1]
    tick_pos = [z for z in time_func(z_ticks)]
    tick_labels = ['%.2f' % (z) for z in z_ticks]
    ax2.set_xlim(x0,x1)
    ax2.set_xticks(tick_pos)
    ax2.set_xticklabels(tick_labels)
    ax2.set_xlabel('Redshift', fontsize=16)

    fig.subplots_adjust(left=0.18,bottom=0.1,top=0.9,right=0.97)
    plt.savefig(prefix + 'den_vs_time' + save_suffix + '.png')

def sb_vs_Z(halos, outs):
    '''Plots the median surface brightness vs. average CGM metallicity for all the halos and outputs listed in halos and outs.'''

    fig = plt.figure(figsize=(7,5),dpi=250)
    ax = fig.add_subplot(1,1,1)
    halo_names = ['Tempest', 'Squall', 'Maelstrom', 'Blizzard', 'Hurricane', 'Cyclone']
    halo_colors = ['#8FDC97','#6A0136','#188FA7','#CC3F0C', '#D5A021', '#73599e']
    for h in range(len(halo_names)):
        if (halo_names[h]=='Tempest (no feedback)'):
            sb_table_loc = output_dir + 'ions_halo_008508/feedback-10-track/Tables/'
        else:
            sb_table_loc = output_dir + 'ions_halo_00' + halos[h] + '/' + args.run + '/Tables/'
        Z_list = []
        SB_med_list = []
        for i in range(len(outs[h])):
            # Load the PDF of OVI emission
            snap = outs[h][i]
            sb_data = Table.read(sb_table_loc + snap + '_SB_profiles' + file_suffix + '.hdf5', path='all_data')
            Z_data = Table.read(sb_table_loc + snap + '_radial_profiles_vol-weighted.hdf5', path='all_data')
            radial_range = (sb_data['inner_radius']==0.) & (sb_data['outer_radius']==20.)
            #radial_range_Z = (Z_data['col1']<=50.)
            radial_range_Z = (Z_data['inner_radius']==0.) & (Z_data['outer_radius']==20.)
            SB_med_list.append(sb_data['all_med'][radial_range][0])
            Z_list.append(Z_data['cgm_met_med'][radial_range_Z][0])

        SB_med_list = np.array(SB_med_list)
        Z_list = np.array(Z_list)
        color_list = [halo_colors[h]] * len(Z_list)
        alphas = np.linspace(0.1, 1.0, len(Z_list))
        color_alpha_list = [mcolors.to_rgba(c, alpha=a) for c, a in zip(color_list, alphas)]
        ax.scatter(Z_list, SB_med_list, marker='.', s=80, fc=color_alpha_list, ec='none')
        
    ax.grid(True, color='#c2c2c2', linestyle='-', linewidth=1)
    ax.set_axisbelow(True)
    ax.set_xlabel(r'Median CGM Metallicity [$Z_\odot$]', fontsize=16)
    ax.set_ylabel('Median O VI SB [erg s$^{-1}$ cm$^{-2}$ arcsec$^{-2}$]', fontsize=16)
    ax.tick_params(axis='both', which='both', direction='in', length=8, width=2, pad=5, labelsize=14, \
                top=True, right=True)
    if (args.Aspera_limit): ax.axis([-1.75,0.5,-19,-18])
    else: ax.axis([-1.75,0.5,-21.,-18.5])
    xticks = [-1.5,-1,-0.5,0,0.5]
    yticks = [-21,-20.5,-20,-19.5,-19,-18.5]
    ax.set_xticks(xticks)
    ax.set_xticklabels(r"$10^{{{:.1f}}}$".format(x) for x in xticks)
    ax.set_yticks(yticks)
    ax.set_yticklabels(r"$10^{{{:.1f}}}$".format(y) for y in yticks)
    #ax.text(0.05, 0.95, halo_names[0], fontsize=14, weight='semibold', c=halo_colors[0], ha='left', va='top', transform=ax.transAxes)
    #ax.text(0.05, 0.89, halo_names[1], fontsize=14, weight='semibold', c=halo_colors[1], ha='left', va='top', transform=ax.transAxes)
    #ax.text(0.05, 0.83, halo_names[2], fontsize=14, weight='semibold', c=halo_colors[2], ha='left', va='top', transform=ax.transAxes)
    #ax.text(0.28, 0.95, halo_names[3], fontsize=14, weight='semibold', c=halo_colors[3], ha='left', va='top', transform=ax.transAxes)
    #ax.text(0.28, 0.89, halo_names[4], fontsize=14, weight='semibold', c=halo_colors[4], ha='left', va='top', transform=ax.transAxes)
    fig.subplots_adjust(left=0.18, bottom=0.13, top=0.95, right=0.97)
    plt.savefig(prefix + 'OVI_SB_vs_Z' + save_suffix + '.png')

def Z_vs_time(halos, outs):
    '''Plots the average CGM metallicity vs. time and redshift for all the halos and outputs listed in halos and outs.'''

    #fig = plt.figure(figsize=(9,8), dpi=300)
    #gs = GridSpec(nrows=2, ncols=1, height_ratios=[3, 1])
    #plt.subplots_adjust(left=0.13, bottom=0.07, top=0.92, right=0.96, hspace=0.)
    #ax = fig.add_subplot(gs[0])
    #ax_sfr = fig.add_subplot(gs[1])
    fig = plt.figure(figsize=(7,5), dpi=250)
    ax = fig.add_subplot(1,1,1)
    #halo_names = ['Tempest', 'Squall', 'Maelstrom', 'Blizzard', 'Hurricane', 'Cyclone', 'Tempest\n(no feedback)']
    #halo_colors = ['#8FDC97','#6A0136','#188FA7','#CC3F0C', '#D5A021', '#b04f86', 'grey']
    halo_names = ['Tempest', 'Squall', 'Maelstrom', 'Blizzard', 'Hurricane', 'Cyclone']
    halo_colors = ['#8FDC97','#6A0136','#188FA7','#CC3F0C', '#D5A021', '#73599e']
    for h in range(len(halo_names)):
        if (halo_names[h]=='Tempest\n(no feedback)'):
            Z_table_loc = output_dir + 'ions_halo_008508/feedback-10-track/Tables/'
            halo_c_v_name = code_path + 'halo_infos/008508/feedback-10-track/halo_c_v'
            sfr_name = code_path + 'halo_infos/008508/feedback-10-track/sfr'
        else:
            Z_table_loc = output_dir + 'ions_halo_00' + halos[h] + '/' + args.run + '/Tables/'
            halo_c_v_name = code_path + 'halo_infos/00' + halos[h] + '/' + args.run + '/halo_c_v'
            sfr_name = code_path + 'halo_infos/00' + halos[h] + '/' + args.run + '/sfr'
        halo_c_v = Table.read(halo_c_v_name, format='ascii')
        timelist = halo_c_v['col4']
        snaplist = halo_c_v['col3']
        zlist = halo_c_v['col2']
        sfr_data = Table.read(sfr_name, format='ascii')
        sfr_snap = sfr_data['col1']
        sfr = sfr_data['col3']
        sfr_time = []
        for i in range(len(sfr_snap)):
            sfr_time.append(float(timelist[snaplist==sfr_snap[i]][0])/1000.)
        Z_list = []
        time_list = []
        z_list = []
        for i in range(len(outs[h])):
            # Load the PDF of OVI emission
            snap = outs[h][i]
            Z_data = Table.read(Z_table_loc + snap + '_radial_profiles_vol-weighted.hdf5', path='all_data')
            radial_range_Z = (Z_data['inner_radius']==0.) & (Z_data['outer_radius']==20.)
            Z_list.append(Z_data['cgm_met_med'][radial_range_Z][0])
            time_list.append(float(timelist[snaplist==snap][0])/1000.)
            z_list.append(float(zlist[snaplist==snap][0]))

        ax.scatter(time_list, Z_list, marker='.', s=60, ec='none', fc=halo_colors[h])
        #ax_sfr.plot(sfr_time, sfr, ls='-', lw=1, color=halo_colors[h])
        if (h==0): ax.axis([np.min(time_list), np.max(time_list), -1.75, 0.5])

    #ax_sfr.axis([np.min(time_list), np.max(time_list), 0, 75])
    #ax_sfr.set_xlabel('Time [Gyr]', fontsize=16)
    #ax_sfr.set_ylabel(r'SFR [$M_\odot$/yr]', fontsize=16)
    #ax_sfr.tick_params(axis='both', which='both', direction='in', length=8, width=2, pad=5, labelsize=12, \
                #top=True, right=True)
    #ax_sfr.set_yticklabels(ax_sfr.get_yticklabels()[:-1])
        
    ax.set_ylabel(r'log Median CGM Metallicity [$Z_\odot$]', fontsize=14)
    #ax.set_xlabel('Time [Gyr]', fontsize=20)
    #ax.legend(loc=4, ncols=2, fontsize=16)
    ax.tick_params(axis='both', which='both', direction='in', length=8, width=2, pad=5, labelsize=12, \
                top=False, right=True)
    ax.set_xticklabels([])

    z_list.reverse()
    time_list.reverse()
    time_func = IUS(z_list, time_list)
    time_list.reverse()

    ax2 = ax.twiny()
    ax2.tick_params(axis='x', which='both', direction='in', length=8, width=2, pad=5, labelsize=12, \
    top=True, labeltop=False)
    x0, x1 = ax.get_xlim()
    z_ticks = np.array([1,.7,.5,.3,.2,.1,0])
    last_z = np.where(z_ticks >= z_list[0])[0][-1]
    first_z = np.where(z_ticks <= z_list[-1])[0][0]
    z_ticks = z_ticks[first_z:last_z+1]
    tick_pos = [z for z in time_func(z_ticks)]
    tick_labels = ['%.2f' % (z) for z in z_ticks]
    ax2.set_xlim(x0,x1)
    ax2.set_xticks(tick_pos)
    ax2.set_xticklabels([])
    #ax2.set_xlabel('Redshift', fontsize=20)

    fig.subplots_adjust(left=0.13,bottom=0.1,top=0.9,right=0.97)
    plt.savefig(prefix + 'Z_vs_time' + save_suffix + '.png')

def sb_vs_temp(halos, outs):
    '''Plots the median surface brightness vs. median CGM temperature for all the halos and outputs listed in halos and outs.'''

    fig = plt.figure(figsize=(7,5),dpi=250)
    ax = fig.add_subplot(1,1,1)
    halo_names = ['Tempest', 'Squall', 'Maelstrom', 'Blizzard', 'Hurricane', 'Cyclone']
    halo_colors = ['#8FDC97','#6A0136','#188FA7','#CC3F0C', '#D5A021', '#73599e']
    for h in range(len(halo_names)):
        if (halo_names[h]=='Tempest (no feedback)'):
            sb_table_loc = output_dir + 'ions_halo_008508/feedback-10-track/Tables/'
        else:
            sb_table_loc = output_dir + 'ions_halo_00' + halos[h] + '/' + args.run + '/Tables/'
        temp_list = []
        SB_med_list = []
        for i in range(len(outs[h])):
            # Load the PDF of OVI emission
            snap = outs[h][i]
            sb_data = Table.read(sb_table_loc + snap + '_SB_profiles' + file_suffix + '.hdf5', path='all_data')
            temp_data = Table.read(sb_table_loc + snap + '_radial_profiles_vol-weighted.hdf5', path='all_data')
            radial_range = (sb_data['inner_radius']==0.) & (sb_data['outer_radius']==20.)
            #radial_range_temp = (temp_data['col1']<=50.)
            radial_range_temp = (temp_data['inner_radius']==0.) & (temp_data['outer_radius']==20.)
            SB_med_list.append(sb_data['all_med'][radial_range][0])
            temp_list.append(temp_data['cgm_temp_med'][radial_range_temp][0])

        SB_med_list = np.array(SB_med_list)
        temp_list = np.array(temp_list)
        color_list = [halo_colors[h]] * len(temp_list)
        alphas = np.linspace(0.1, 1.0, len(temp_list))
        color_alpha_list = [mcolors.to_rgba(c, alpha=a) for c, a in zip(color_list, alphas)]
        ax.scatter(temp_list, SB_med_list, marker='.', s=80, fc=color_alpha_list, ec='none')
        
    ax.grid(True, color='#c2c2c2', linestyle='-', linewidth=1)
    ax.set_axisbelow(True)
    ax.set_xlabel('Median CGM Temperature [K]', fontsize=16)
    ax.set_ylabel('Median O VI SB [erg s$^{-1}$ cm$^{-2}$ arcsec$^{-2}$]', fontsize=16)
    ax.tick_params(axis='both', which='both', direction='in', length=8, width=2, pad=5, labelsize=14, \
                top=True, right=True)
    if (args.Aspera_limit): ax.axis([4,6.75,-19,-18])
    else: ax.axis([4,6.75,-21.,-18.5])
    xticks = [4,4.5,5,5.5,6,6.5]
    yticks = [-21,-20.5,-20,-19.5,-19,-18.5]
    ax.set_xticks(xticks)
    ax.set_xticklabels(r"$10^{{{:.1f}}}$".format(x) for x in xticks)
    ax.set_yticks(yticks)
    ax.set_yticklabels(r"$10^{{{:.1f}}}$".format(y) for y in yticks)
    #ax.text(0.05, 0.95, halo_names[0], fontsize=14, weight='semibold', c=halo_colors[0], ha='left', va='top', transform=ax.transAxes)
    #ax.text(0.05, 0.89, halo_names[1], fontsize=14, weight='semibold', c=halo_colors[1], ha='left', va='top', transform=ax.transAxes)
    #ax.text(0.05, 0.83, halo_names[2], fontsize=14, weight='semibold', c=halo_colors[2], ha='left', va='top', transform=ax.transAxes)
    #ax.text(0.28, 0.95, halo_names[3], fontsize=14, weight='semibold', c=halo_colors[3], ha='left', va='top', transform=ax.transAxes)
    #ax.text(0.28, 0.89, halo_names[4], fontsize=14, weight='semibold', c=halo_colors[4], ha='left', va='top', transform=ax.transAxes)
    #ax.text(0.28, 0.83, halo_names[5], fontsize=14, weight='semibold', c=halo_colors[5], ha='left', va='top', transform=ax.transAxes)
    fig.subplots_adjust(left=0.18, bottom=0.13, top=0.95, right=0.97)
    plt.savefig(prefix + 'OVI_SB_vs_temp' + save_suffix + '.png')

def temp_vs_time(halos, outs):
    '''Plots the average CGM temperature vs. time and redshift for all the halos and outputs listed in halos and outs.'''

    #fig = plt.figure(figsize=(9,8), dpi=300)
    #gs = GridSpec(nrows=2, ncols=1, height_ratios=[3, 1])
    #plt.subplots_adjust(left=0.13, bottom=0.07, top=0.92, right=0.96, hspace=0.)
    #ax = fig.add_subplot(gs[0])
    #ax_sfr = fig.add_subplot(gs[1])
    fig = plt.figure(figsize=(7,5), dpi=250)
    ax = fig.add_subplot(1,1,1)
    #halo_names = ['Tempest', 'Squall', 'Maelstrom', 'Blizzard', 'Hurricane', 'Cyclone', 'Tempest\n(no feedback)']
    #halo_colors = ['#8FDC97','#6A0136','#188FA7','#CC3F0C', '#D5A021', '#73599e', 'grey']
    halo_names = ['Tempest', 'Squall', 'Maelstrom', 'Blizzard', 'Hurricane', 'Cyclone']
    halo_colors = ['#8FDC97','#6A0136','#188FA7','#CC3F0C', '#D5A021', '#73599e']
    for h in range(len(halo_names)):
        if (halo_names[h]=='Tempest\n(no feedback)'):
            temp_table_loc = output_dir + 'ions_halo_008508/feedback-10-track/Tables/'
            halo_c_v_name = code_path + 'halo_infos/008508/feedback-10-track/halo_c_v'
            sfr_name = code_path + 'halo_infos/008508/feedback-10-track/sfr'
        else:
            temp_table_loc = output_dir + 'ions_halo_00' + halos[h] + '/' + args.run + '/Tables/'
            halo_c_v_name = code_path + 'halo_infos/00' + halos[h] + '/' + args.run + '/halo_c_v'
            sfr_name = code_path + 'halo_infos/00' + halos[h] + '/' + args.run + '/sfr'
        halo_c_v = Table.read(halo_c_v_name, format='ascii')
        timelist = halo_c_v['col4']
        snaplist = halo_c_v['col3']
        zlist = halo_c_v['col2']
        sfr_data = Table.read(sfr_name, format='ascii')
        sfr_snap = sfr_data['col1']
        sfr = sfr_data['col3']
        sfr_time = []
        for i in range(len(sfr_snap)):
            sfr_time.append(float(timelist[snaplist==sfr_snap[i]][0])/1000.)
        temp_list = []
        time_list = []
        z_list = []
        for i in range(len(outs[h])):
            # Load the PDF of OVI emission
            snap = outs[h][i]
            temp_data = Table.read(temp_table_loc + snap + '_radial_profiles_vol-weighted.hdf5', path='all_data')
            radial_range_temp = (temp_data['inner_radius']==0.) & (temp_data['outer_radius']==20.)
            temp_list.append(temp_data['cgm_temp_med'][radial_range_temp][0])
            time_list.append(float(timelist[snaplist==snap][0])/1000.)
            z_list.append(float(zlist[snaplist==snap][0]))

        ax.scatter(time_list, temp_list, marker='.', s=60, ec='none', fc=halo_colors[h])
        #ax_sfr.plot(sfr_time, sfr, ls='-', lw=1, color=halo_colors[h])
        if (h==0): ax.axis([np.min(time_list), np.max(time_list), 5.25, 6.75])

    #ax_sfr.axis([np.min(time_list), np.max(time_list), 0, 75])
    #ax_sfr.set_xlabel('Time [Gyr]', fontsize=16)
    #ax_sfr.set_ylabel(r'SFR [$M_\odot$/yr]', fontsize=16)
    #ax_sfr.tick_params(axis='both', which='both', direction='in', length=8, width=2, pad=5, labelsize=12, \
                #top=True, right=True)
    #ax_sfr.set_yticklabels(ax_sfr.get_yticklabels()[:-1])
        
    ax.set_ylabel('log Median CGM Temperature [K]', fontsize=14)
    ax.set_xlabel('Time [Gyr]', fontsize=14)
    #ax.legend(loc=8, ncols=2, fontsize=16)
    ax.tick_params(axis='both', which='both', direction='in', length=8, width=2, pad=5, labelsize=12, \
                top=False, right=True)

    z_list.reverse()
    time_list.reverse()
    time_func = IUS(z_list, time_list)
    time_list.reverse()

    ax2 = ax.twiny()
    ax2.tick_params(axis='x', which='both', direction='in', length=8, width=2, pad=5, labelsize=12, \
    top=True)
    x0, x1 = ax.get_xlim()
    z_ticks = np.array([1,.7,.5,.3,.2,.1,0])
    last_z = np.where(z_ticks >= z_list[0])[0][-1]
    first_z = np.where(z_ticks <= z_list[-1])[0][0]
    z_ticks = z_ticks[first_z:last_z+1]
    tick_pos = [z for z in time_func(z_ticks)]
    tick_labels = ['%.2f' % (z) for z in z_ticks]
    ax2.set_xlim(x0,x1)
    ax2.set_xticks(tick_pos)
    ax2.set_xticklabels([])
    #ax2.set_xlabel('Redshift', fontsize=20)

    fig.subplots_adjust(left=0.13,bottom=0.1,top=0.9,right=0.97)
    plt.savefig(prefix + 'temp_vs_time' + save_suffix + '.png')

def den_Z_temp_vs_time(halos, outs):
    '''Plots the median CGM density, metallicity, and temperature vs time.'''

    halo_names = ['Tempest', 'Squall', 'Maelstrom', 'Blizzard', 'Hurricane', 'Cyclone']
    halo_colors = ['#8FDC97','#6A0136','#188FA7','#CC3F0C', '#D5A021', '#73599e']

    fig = plt.figure(figsize=(7,12), dpi=250)
    ax1 = fig.add_subplot(3,1,1)
    ax2 = fig.add_subplot(3,1,2)
    ax3 = fig.add_subplot(3,1,3)

    for h in range(len(halo_names)):
        if (halo_names[h]=='Tempest\n(no feedback)'):
            sb_table_loc = output_dir + 'ions_halo_008508/feedback-10-track/Tables/'
            halo_c_v_name = code_path + 'halo_infos/008508/feedback-10-track/halo_c_v'
        else:
            sb_table_loc = output_dir + 'ions_halo_00' + halos[h] + '/' + args.run + '/Tables/'
            halo_c_v_name = code_path + 'halo_infos/00' + halos[h] + '/' + args.run + '/halo_c_v'
        halo_c_v = Table.read(halo_c_v_name, format='ascii')
        timelist = halo_c_v['col4']
        snaplist = halo_c_v['col3']
        zlist = halo_c_v['col2']
        den_list = []
        Z_list = []
        temp_list = []
        time_list = []
        z_list = []
        for i in range(len(outs[h])):
            # Load the PDF of OVI emission
            snap = outs[h][i]
            data = Table.read(sb_table_loc + snap + '_radial_profiles_vol-weighted.hdf5', path='all_data')
            radial_range = (data['inner_radius']==0.) & (data['outer_radius']==20.)
            den_list.append(data['cgm_den_med'][radial_range][0])
            Z_list.append(data['cgm_met_med'][radial_range][0])
            temp_list.append(data['cgm_temp_med'][radial_range][0])
            time_list.append(float(timelist[snaplist==snap][0])/1000.)
            z_list.append(float(zlist[snaplist==snap][0]))

        ax1.plot(time_list, den_list, marker='.', markersize=10, mec='none', mfc=halo_colors[h], ls='-', lw=0.5, color=halo_colors[h])
        if (h==0): ax1.axis([np.min(time_list), np.max(time_list), -28.5, -26.75])

        ax2.plot(time_list, Z_list, marker='.', markersize=10, mec='none', mfc=halo_colors[h], ls='-', lw=0.5, color=halo_colors[h])
        if (h==0): ax2.axis([np.min(time_list), np.max(time_list), -1.75, 0.5])
        
        ax3.plot(time_list, temp_list, marker='.', markersize=10, mec='none', mfc=halo_colors[h], ls='-', lw=0.5, color=halo_colors[h])
        if (h==0): ax3.axis([np.min(time_list), np.max(time_list), 5.25, 6.75])
        
    ax1.grid(True, color='#c2c2c2', linestyle='-', linewidth=1)
    ax1.set_axisbelow(True)
    ax1.set_ylabel('Median CGM Density [g cm$^{-3}$]', fontsize=16)
    ax1.tick_params(axis='both', which='both', direction='in', length=8, width=2, pad=5, labelsize=14, \
                top=False, right=True)
    ax1.set_xticklabels([])
    yticks = [-28.4,-28,-27.6,-27.2,-26.8]
    ax1.set_yticks(yticks)
    ax1.set_yticklabels(r"$10^{{{:.1f}}}$".format(y) for y in yticks)
    ax1.text(0.7, 0.95, halo_names[0], fontsize=16, weight='semibold', c=halo_colors[0], ha='right', va='top', transform=ax1.transAxes)
    ax1.text(0.7, 0.89, halo_names[1], fontsize=16, weight='semibold', c=halo_colors[1], ha='right', va='top', transform=ax1.transAxes)
    ax1.text(0.7, 0.83, halo_names[2], fontsize=16, weight='semibold', c=halo_colors[2], ha='right', va='top', transform=ax1.transAxes)
    ax1.text(0.95, 0.95, halo_names[3], fontsize=16, weight='semibold', c=halo_colors[3], ha='right', va='top', transform=ax1.transAxes)
    ax1.text(0.95, 0.89, halo_names[4], fontsize=16, weight='semibold', c=halo_colors[4], ha='right', va='top', transform=ax1.transAxes)
    ax1.text(0.95, 0.83, halo_names[5], fontsize=16, weight='semibold', c=halo_colors[5], ha='right', va='top', transform=ax1.transAxes)
    #ax1.text(0.95, 0.77, halo_names[6], fontsize=16, weight='semibold', c=halo_colors[6], ha='right', va='top', transform=ax1.transAxes)

    z_list.reverse()
    time_list.reverse()
    time_func = IUS(z_list, time_list)
    time_list.reverse()
    z_ticks = np.array([1,.7,.5,.3,.2,.1,0])
    last_z = np.where(z_ticks >= z_list[0])[0][-1]
    first_z = np.where(z_ticks <= z_list[-1])[0][0]
    z_ticks = z_ticks[first_z:last_z+1]
    tick_pos = [z for z in time_func(z_ticks)]
    tick_labels = ['%.1f' % (z) for z in z_ticks]

    ax_z = ax1.twiny()
    ax_z.tick_params(axis='x', which='both', direction='in', length=8, width=2, pad=5, labelsize=14, \
    top=True)
    x0, x1 = ax1.get_xlim()
    ax_z.set_xlim(x0,x1)
    ax_z.set_xticks(tick_pos)
    ax_z.set_xticklabels(tick_labels)
    ax_z.set_xlabel('Redshift', fontsize=16)

    ax2.grid(True, color='#c2c2c2', linestyle='-', linewidth=1)
    ax2.set_axisbelow(True)
    ax2.set_ylabel(r'Median CGM Metallicity [$Z_\odot$]', fontsize=16)
    ax2.tick_params(axis='both', which='both', direction='in', length=8, width=2, pad=5, labelsize=14, \
                top=False, right=True)
    ax2.set_xticklabels([])
    yticks = [-1.5,-1,-0.5,0,0.5]
    ax2.set_yticks(yticks)
    ax2.set_yticklabels(r"$10^{{{:.1f}}}$".format(y) for y in yticks)

    ax_z = ax2.twiny()
    ax_z.tick_params(axis='x', which='both', direction='in', length=8, width=2, pad=5, labelsize=14, \
    top=True)
    x0, x1 = ax2.get_xlim()
    ax_z.set_xlim(x0,x1)
    ax_z.set_xticks(tick_pos)
    ax_z.set_xticklabels([])

    ax3.grid(True, color='#c2c2c2', linestyle='-', linewidth=1)
    ax3.set_axisbelow(True)
    ax3.set_ylabel('Median CGM Temperature [K]', fontsize=16)
    ax3.tick_params(axis='both', which='both', direction='in', length=8, width=2, pad=5, labelsize=14, \
                top=False, right=True)
    yticks = [4,4.5,5,5.5,6,6.5]
    ax3.set_yticks(yticks)
    ax3.set_yticklabels(r"$10^{{{:.1f}}}$".format(y) for y in yticks)
    ax3.set_xlabel('Time [Gyr]', fontsize=16)

    ax_z = ax3.twiny()
    ax_z.tick_params(axis='x', which='both', direction='in', length=8, width=2, pad=5, labelsize=14, \
    top=True)
    x0, x1 = ax3.get_xlim()
    ax_z.set_xlim(x0,x1)
    ax_z.set_xticks(tick_pos)
    ax_z.set_xticklabels([])

    fig.subplots_adjust(left=0.18,bottom=0.05,top=0.95,right=0.97, hspace=0.)
    plt.savefig(prefix + 'den-Z-temp_vs_time' + save_suffix + '.png')

def sb_vs_den_temp_Z(halos, outs):
    '''Plots the median O VI surface brightness vs. the average CGM density, temperature, and metallicity.'''

    fig = plt.figure(figsize=(15,5),dpi=250)
    ax_den = fig.add_subplot(1,3,1)
    ax_temp = fig.add_subplot(1,3,2)
    ax_Z = fig.add_subplot(1,3,3)
    halo_names = ['Tempest', 'Squall', 'Maelstrom', 'Blizzard', 'Hurricane', 'Cyclone']
    halo_colors = ['#8FDC97','#6A0136','#188FA7','#CC3F0C', '#D5A021', '#73599e']
    for h in range(len(halo_names)):
        if (halo_names[h]=='Tempest (no feedback)'):
            sb_table_loc = output_dir + 'ions_halo_008508/feedback-10-track/Tables/'
            prop_table_loc = output_dir + 'profiles_halo_008508/feedback-10-track/Tables/'
        else:
            sb_table_loc = output_dir + 'ions_halo_00' + halos[h] + '/' + args.run + '/Tables/'
            prop_table_loc = output_dir + 'profiles_halo_00' + halos[h] + '/' + args.run + '/Tables/'
        den_list = []
        temp_list = []
        Z_list = []
        SB_med_list = []
        for i in range(len(outs[h])):
            # Load the PDF of OVI emission
            snap = outs[h][i]
            sb_data = Table.read(sb_table_loc + snap + '_SB_profiles' + file_suffix + '.hdf5', path='all_data')
            den_data = Table.read(prop_table_loc + snap + '_density_vs_radius_volume-weighted_profiles_cgm-only.txt', format='ascii')
            temp_data = Table.read(prop_table_loc + snap + '_temperature_vs_radius_volume-weighted_profiles_cgm-only.txt', format='ascii')
            Z_data = Table.read(prop_table_loc + snap + '_metallicity_vs_radius_volume-weighted_profiles_cgm-only.txt', format='ascii')
            radial_range = (sb_data['inner_radius']==0.) & (sb_data['outer_radius']==20.)
            radial_range_den = (den_data['col1']<=20.)
            radial_range_temp = (temp_data['col1']<=20.)
            radial_range_Z = (Z_data['col1']<=20.)
            SB_med_list.append(np.mean(sb_data['all_med'][radial_range]))
            den_list.append(np.mean(den_data['col3'][radial_range_den]))
            temp_list.append(np.mean(temp_data['col3'][radial_range_temp]))
            Z_list.append(np.mean(Z_data['col3'][radial_range_Z]))

        SB_med_list = np.array(SB_med_list)
        den_list = np.array(den_list)
        temp_list = np.array(temp_list)
        Z_list = np.array(Z_list)
        ax_den.scatter(np.log10(den_list), SB_med_list, marker='.', color=halo_colors[h], label=halo_names[h])
        ax_temp.scatter(np.log10(temp_list), SB_med_list, marker='.', color=halo_colors[h])
        ax_Z.scatter(np.log10(Z_list), SB_med_list, marker='.', color=halo_colors[h])
        
    ax_den.set_xlabel('log Mean CGM Density [g cm$^{-3}$]', fontsize=14)
    ax_temp.set_xlabel('log Mean CGM Temperature [K]', fontsize=14)
    ax_Z.set_xlabel(r'log Mean CGM Metallicity [$Z_\odot$]', fontsize=14)

    ax_den.set_xticks([-28,-27.5,-27,-26.5])
    ax_temp.set_xticks([5.5, 6, 6.25,6.5,6.75,7])
    ax_Z.set_xticks([-1.5,-1,-0.5,0,0.5,1])

    if (args.Aspera_limit):
        ax_den.axis([-28.2,-26,-19,-18])
        ax_temp.axis([5.5,7.25,-19,-18])
        ax_Z.axis([-1.5,1,-19,-18])
    else:
        ax_den.axis([-28.2,-26,-21.,-18])
        ax_temp.axis([5.5,7.25,-21.,-18])
        ax_Z.axis([-1.5,1,-21.,-18])

    ax_den.tick_params(axis='both', which='both', direction='in', length=8, width=2, pad=5, labelsize=12, \
            top=True, right=True)
    ax_temp.tick_params(axis='both', which='both', direction='in', length=8, width=2, pad=5, labelsize=12, \
            top=True, right=True, labelleft=False)
    ax_Z.tick_params(axis='both', which='both', direction='in', length=8, width=2, pad=5, labelsize=12, \
            top=True, right=True, labelleft=False)

    ax_den.set_ylabel('log Median O VI SB [erg s$^{-1}$ cm$^{-2}$ arcsec$^{-2}$]', fontsize=14)
    ax_den.legend(loc=2, ncols=2, fontsize=14)
    fig.subplots_adjust(left=0.07, bottom=0.13, top=0.95, right=0.97, wspace=0)
    plt.savefig(prefix + 'OVI_SB_vs_den-temp-Z' + save_suffix + '.png')

def emiss_area_vs_sfr(halos, outs):
    '''Plots the fractional area of pixels above the Aspera limit vs SFR.'''

    dx = 100./800.              # physical width of FRB (kpc) divided by resolution
    FRB_x = np.indices((800,800))[0]
    FRB_y = np.indices((800,800))[1]
    FRB_x = FRB_x*dx - 50.
    FRB_y = FRB_y*dx - 50.
    radius = np.sqrt(FRB_x**2. + FRB_y**2.)

    fig = plt.figure(figsize=(7,5),dpi=250)
    ax = fig.add_subplot(1,1,1)
    halo_names = ['Tempest', 'Squall', 'Maelstrom', 'Blizzard', 'Hurricane', 'Cyclone']
    halo_colors = ['#8FDC97','#6A0136','#188FA7','#CC3F0C', '#D5A021', '#73599e']
    alphas = np.linspace(1., 0.1, 10)
    for h in range(len(halo_names)):
        if (halo_names[h]=='Tempest (no feedback)'):
            sb_table_loc = output_dir + 'ions_halo_008508/feedback-10-track/Tables/'
            sfr_table = Table.read(code_path + 'halo_infos/008508/feedback-10-track/sfr', format='ascii')
        else:
            sb_table_loc = output_dir + 'ions_halo_00' + halos[h] + '/' + args.run + '/Tables/'
            sfr_table = Table.read(code_path + 'halo_infos/00' + halos[h] + '/' + args.run + '/sfr', format='ascii')
        SFR_list = []
        SB_frac_list = []
        for i in range(len(outs[h])):
            # Load the PDF of OVI emission
            snap = outs[h][i]
            sfr = sfr_table['col3'][sfr_table['col1']==snap][0]
            if (halo_names[h]=='Cyclone'): sb_pdf = Table.read(sb_table_loc + snap + '_SB_pdf_sym-vel_and_temp' + file_suffix + '.hdf5', path='all_data')
            else: sb_pdf = Table.read(sb_table_loc + snap + '_SB_pdf_new-table' + file_suffix + '.hdf5', path='all_data')
            radial_range = (sb_pdf['inner_radius']==0.) & (sb_pdf['outer_radius']==20.)
            #radial_range = (sb_pdf['outer_radius']<=20.)
            npix = len(radius[radius<=20.])
            frac_area = np.sum(sb_pdf['all'][radial_range])/npix # Use the Aspera-limit pdf file, then the number of pixels above the limit is just the number of pixels in the radial bin, which is the sum across the histogram
            SB_frac_list.append(frac_area)
            SFR_list.append(sfr)

        SB_frac_list = np.array(SB_frac_list)
        SFR_list = np.array(SFR_list)
        color_list = [halo_colors[h]] * len(SFR_list)
        alphas = np.linspace(0.1, 1.0, len(SFR_list))
        color_alpha_list = [mcolors.to_rgba(c, alpha=a) for c, a in zip(color_list, alphas)]
        ax.scatter(SFR_list, SB_frac_list, marker='.', s=80, fc=color_alpha_list, ec='none')
        
    ax.grid(True, color='#c2c2c2', linestyle='-', linewidth=1)
    ax.set_axisbelow(True)
    ax.set_xlabel(r'Star formation rate [$M_\odot$/yr]', fontsize=16)
    ax.set_ylabel('Fraction of area above O VI SB limit', fontsize=16)
    ax.tick_params(axis='both', which='both', direction='in', length=8, width=2, pad=5, labelsize=14, \
                top=True, right=True)
    ax.axis([0.1,100,0.,0.6])
    ax.set_xscale('log')
    #ax.text(0.05, 0.95, halo_names[0], fontsize=14, weight='semibold', c=halo_colors[0], ha='left', va='top', transform=ax.transAxes)
    #ax.text(0.05, 0.89, halo_names[1], fontsize=14, weight='semibold', c=halo_colors[1], ha='left', va='top', transform=ax.transAxes)
    #ax.text(0.05, 0.83, halo_names[2], fontsize=14, weight='semibold', c=halo_colors[2], ha='left', va='top', transform=ax.transAxes)
    #ax.text(0.28, 0.95, halo_names[3], fontsize=14, weight='semibold', c=halo_colors[3], ha='left', va='top', transform=ax.transAxes)
    #ax.text(0.28, 0.89, halo_names[4], fontsize=14, weight='semibold', c=halo_colors[4], ha='left', va='top', transform=ax.transAxes)
    #ax.text(0.28, 0.83, halo_names[5], fontsize=14, weight='semibold', c=halo_colors[5], ha='left', va='top', transform=ax.transAxes)
    fig.tight_layout()
    fig.savefig(prefix + 'fractional-area-above-limit_vs_SFR' + save_suffix + '.png')

def emiss_area_vs_Mh(halos, outs):
    '''Plots the fractional area of pixels above the Aspera limit vs halo mass.'''

    dx = 100./800.              # physical width of FRB (kpc) divided by resolution
    FRB_x = np.indices((800,800))[0]
    FRB_y = np.indices((800,800))[1]
    FRB_x = FRB_x*dx - 50.
    FRB_y = FRB_y*dx - 50.
    radius = np.sqrt(FRB_x**2. + FRB_y**2.)

    fig = plt.figure(figsize=(10,6),dpi=250)
    ax = fig.add_subplot(1,1,1)
    fig2 = plt.figure(figsize=(10,6),dpi=250)
    ax2 = fig2.add_subplot(1,1,1)
    halo_colors = ['r','orange','b','g','c','m']
    halo_names = ['Tempest', 'Tempest (no feedback)', 'Squall', 'Maelstrom', 'Blizzard', 'Hurricane']
    alphas = np.linspace(1., 0.1, 10)
    #halo_names = ['Tempest', 'Squall', 'Maelstrom', 'Blizzard']
    #halo_colors = ['r','b','g','c']
    for h in range(len(halo_names)):
        if (halo_names[h]=='Tempest (no feedback)'):
            sb_table_loc = output_dir + 'ions_halo_008508/feedback-10-track/Tables/'
            mvir_table = Table.read(code_path + 'halo_infos/008508/feedback-10-track/rvir_masses.hdf5', path='all_data')
        else:
            sb_table_loc = output_dir + 'ions_halo_00' + halos[h] + '/' + args.run + '/Tables/'
            mvir_table = Table.read(code_path + 'halo_infos/00' + halos[h] + '/' + args.run + '/rvir_masses.hdf5', path='all_data')
        mvir_list = []
        SB_frac_list = []
        for i in range(len(outs[h])):
            # Load the PDF of OVI emission
            snap = outs[h][i]
            mvir_list.append(np.log10(mvir_table['total_mass'][mvir_table['snapshot']==snap][0]))
            sb_pdf = Table.read(sb_table_loc + snap + '_SB_pdf' + file_suffix + '.hdf5', path='all_data')
            radial_range = (sb_pdf['inner_radius']==0.) & (sb_pdf['outer_radius']==50.)
            #radial_range = (sb_pdf['outer_radius']<=20.)
            npix = len(radius[radius<=50.])
            frac_area = np.sum(sb_pdf['all'][(radial_range) & (sb_pdf['lower_SB']>=-18.5)])/npix
            SB_frac_list.append(frac_area)

        SB_frac_list = np.array(SB_frac_list)
        mvir_list = np.array(mvir_list)
        ax.scatter(mvir_list, SB_frac_list, marker='.', color=halo_colors[h], label=halo_names[h])
        
    ax.set_xlabel(r'log Halo mass [$M_\odot$]', fontsize=16)
    ax.set_ylabel('Fraction of area above O VI SB limit', fontsize=16)
    ax.legend(loc=2, frameon=False, fontsize=16)
    ax.tick_params(axis='both', which='both', direction='in', length=8, width=2, pad=5, labelsize=12, \
                top=True, right=True)
    ax.axis([11.5,12.25,0.,0.2])
    fig.subplots_adjust(left=0.1, bottom=0.12, top=0.93, right=0.98)
    fig.savefig(prefix + 'fractional-area-above-limit_vs_Mh' + save_suffix + '.png')

def emiss_area_vs_den(halos, outs):
    '''Plots the fractional area of pixels above the Aspera limit vs mean CGM density.'''

    dx = 100./800.              # physical width of FRB (kpc) divided by resolution
    FRB_x = np.indices((800,800))[0]
    FRB_y = np.indices((800,800))[1]
    FRB_x = FRB_x*dx - 50.
    FRB_y = FRB_y*dx - 50.
    radius = np.sqrt(FRB_x**2. + FRB_y**2.)

    fig = plt.figure(figsize=(10,6),dpi=250)
    ax = fig.add_subplot(1,1,1)
    fig2 = plt.figure(figsize=(10,6),dpi=250)
    ax2 = fig2.add_subplot(1,1,1)
    halo_colors = ['r','orange','b','g','c','m']
    halo_names = ['Tempest', 'Tempest (no feedback)', 'Squall', 'Maelstrom', 'Blizzard', 'Hurricane']
    alphas = np.linspace(1., 0.1, 10)
    for h in range(len(halo_names)):
        if (halo_names[h]=='Tempest (no feedback)'):
            sb_table_loc = output_dir + 'ions_halo_008508/feedback-10-track/Tables/'
            den_table_loc = output_dir + 'profiles_halo_008508/feedback-10-track/Tables/'
        else:
            sb_table_loc = output_dir + 'ions_halo_00' + halos[h] + '/' + args.run + '/Tables/'
            den_table_loc = output_dir + 'profiles_halo_00' + halos[h] + '/' + args.run + '/Tables/'
        den_list = []
        SB_frac_list = []
        for i in range(len(outs[h])):
            # Load the PDF of OVI emission
            snap = outs[h][i]
            den_data = Table.read(den_table_loc + snap + '_density_vs_radius_volume-weighted_profiles_cgm-only.txt', format='ascii')
            sb_pdf = Table.read(sb_table_loc + snap + '_SB_pdf' + file_suffix + '.hdf5', path='all_data')
            radial_range = (sb_pdf['inner_radius']==0.) & (sb_pdf['outer_radius']==50.)
            #radial_range = (sb_pdf['outer_radius']<=20.)
            radial_range_den = (den_data['col1']<=50.)
            #radial_range_den = (den_data['col1']<=20.)
            den_list.append(np.mean(den_data['col3'][radial_range_den]))
            npix = len(radius[radius<=50.])
            frac_area = np.sum(sb_pdf['all'][(radial_range) & (sb_pdf['lower_SB']>=-18.5)])/npix
            SB_frac_list.append(frac_area)

        SB_frac_list = np.array(SB_frac_list)
        den_list = np.array(den_list)
        ax.scatter(np.log10(den_list), SB_frac_list, marker='.', color=halo_colors[h], label=halo_names[h])
        
    ax.set_xlabel('log Mean CGM Density [g cm$^{-3}$]', fontsize=16)
    ax.set_ylabel('Fraction of area above O VI SB limit', fontsize=16)
    ax.legend(loc=2, frameon=False, fontsize=16)
    ax.tick_params(axis='both', which='both', direction='in', length=8, width=2, pad=5, labelsize=12, \
                top=True, right=True)
    ax.axis([-28.2,-26,0.,0.2])
    fig.subplots_adjust(left=0.1, bottom=0.12, top=0.93, right=0.98)
    fig.savefig(prefix + 'fractional-area-above-limit_vs_den' + save_suffix + '.png')

def emiss_area_vs_Z(halos, outs):
    '''Plots the fractional area of pixels above the Aspera limit vs mean CGM metallicity.'''

    dx = 100./800.              # physical width of FRB (kpc) divided by resolution
    FRB_x = np.indices((800,800))[0]
    FRB_y = np.indices((800,800))[1]
    FRB_x = FRB_x*dx - 50.
    FRB_y = FRB_y*dx - 50.
    radius = np.sqrt(FRB_x**2. + FRB_y**2.)

    fig = plt.figure(figsize=(10,6),dpi=250)
    ax = fig.add_subplot(1,1,1)
    fig2 = plt.figure(figsize=(10,6),dpi=250)
    ax2 = fig2.add_subplot(1,1,1)
    halo_colors = ['r','orange','b','g','c','m']
    halo_names = ['Tempest', 'Tempest (no feedback)', 'Squall', 'Maelstrom', 'Blizzard', 'Hurricane']
    alphas = np.linspace(1., 0.1, 10)
    for h in range(len(halo_names)):
        if (halo_names[h]=='Tempest (no feedback)'):
            sb_table_loc = output_dir + 'ions_halo_008508/feedback-10-track/Tables/'
            Z_table_loc = output_dir + 'profiles_halo_008508/feedback-10-track/Tables/'
        else:
            sb_table_loc = output_dir + 'ions_halo_00' + halos[h] + '/' + args.run + '/Tables/'
            Z_table_loc = output_dir + 'profiles_halo_00' + halos[h] + '/' + args.run + '/Tables/'
        Z_list = []
        SB_frac_list = []
        for i in range(len(outs[h])):
            # Load the PDF of OVI emission
            snap = outs[h][i]
            Z_data = Table.read(Z_table_loc + snap + '_metallicity_vs_radius_volume-weighted_profiles_cgm-only.txt', format='ascii')
            sb_pdf = Table.read(sb_table_loc + snap + '_SB_pdf' + file_suffix + '.hdf5', path='all_data')
            radial_range = (sb_pdf['inner_radius']==0.) & (sb_pdf['outer_radius']==50.)
            #radial_range = (sb_pdf['outer_radius']<=20.)
            radial_range_Z = (Z_data['col1']<=50.)
            #radial_range_Z = (Z_data['col1']<=20.)
            Z_list.append(np.mean(Z_data['col3'][radial_range_Z]))
            npix = len(radius[radius<=50.])
            frac_area = np.sum(sb_pdf['all'][(radial_range) & (sb_pdf['lower_SB']>=-18.5)])/npix
            SB_frac_list.append(frac_area)

        SB_frac_list = np.array(SB_frac_list)
        Z_list = np.array(Z_list)
        ax.scatter(np.log10(Z_list), SB_frac_list, marker='.', color=halo_colors[h], label=halo_names[h])
        
    ax.set_xlabel(r'log Mean CGM Metallicity [$Z_\odot$]', fontsize=16)
    ax.set_ylabel('Fraction of area above O VI SB limit', fontsize=16)
    ax.legend(loc=2, frameon=False, fontsize=16)
    ax.tick_params(axis='both', which='both', direction='in', length=8, width=2, pad=5, labelsize=12, \
                top=True, right=True)
    ax.axis([-2.5,1,0.,0.2])
    fig.subplots_adjust(left=0.1, bottom=0.12, top=0.93, right=0.98)
    fig.savefig(prefix + 'fractional-area-above-limit_vs_Z' + save_suffix + '.png')

def hists_den_temp_Z_rv_tcool(ds, refine_box, snap):
    '''Plots histograms of pixel densities, temperatures, metallicities, radial velocities, and cooling times along with
    emissivity-weighted histograms.'''

    sph = ds.sphere(center=ds.halo_center_kpc, radius=(20., 'kpc'))
    sph_inflow = sph.include_below(('gas','radial_velocity_corrected'), -100.)
    sph_outflow = sph.include_above(('gas','radial_velocity_corrected'), 100.)

    densities = sph['gas','density'].in_units('g/cm**3').v
    densities_inflow = sph_inflow['gas','density'].in_units('g/cm**3').v
    densities_outflow = sph_outflow['gas','density'].in_units('g/cm**3').v
    temperatures = sph['gas','temperature'].in_units('K').v
    temperatures_inflow = sph_inflow['gas','temperature'].in_units('K').v
    temperatures_outflow = sph_outflow['gas','temperature'].in_units('K').v
    metallicities = sph['gas','metallicity'].in_units('Z_sun').v
    metallicities_inflow = sph_inflow['gas','metallicity'].in_units('Zsun').v
    metallicities_outflow = sph_outflow['gas','metallicity'].in_units('Zsun').v
    rvs = sph['gas','radial_velocity_corrected'].in_units('km/s').v
    rvs_inflow = sph_inflow['gas','radial_velocity_corrected'].in_units('km/s').v
    rvs_outflow = sph_outflow['gas','radial_velocity_corrected'].in_units('km/s').v
    tcools = sph['gas','cooling_time'].in_units('Myr').v
    tcools_inflow = sph_inflow['gas','cooling_time'].in_units('Myr').v
    tcools_outflow = sph_outflow['gas','cooling_time'].in_units('Myr').v
    emissivities = sph['gas','Emission_OVI'].in_units(emission_units_ALT).v
    emissivities_inflow = sph_inflow['gas','Emission_OVI'].in_units(emission_units_ALT).v
    emissivities_outflow = sph_outflow['gas','Emission_OVI'].in_units(emission_units_ALT).v
    if (args.weight=='volume'):
        weights = np.log10(sph['gas','cell_volume'].in_units('kpc**3').v)
        weights_inflow = np.log10(sph_inflow['gas','cell_volume'].in_units('kpc**3').v)
        weights_outflow = np.log10(sph_outflow['gas','cell_volume'].in_units('kpc**3').v)
    else:
        weights = np.log10(sph['gas','cell_mass'].in_units('Msun').v)
        weights_inflow = np.log10(sph_inflow['gas','cell_mass'].in_units('Msun').v)
        weights_outflow = np.log10(sph_outflow['gas','cell_mass'].in_units('Msun').v)

    for i in range(4):
        if (i==0):
            density = densities
            temperature = temperatures
            metallicity = metallicities
            rv = rvs
            tcool = tcools
            emissivity = emissivities
            weight = weights
            flow_file = ''
            gas_label = 'All gas'
            OVI_label = 'O VI emitting gas'
        if (i==1):
            density = densities_inflow
            temperature = temperatures_inflow
            metallicity = metallicities_inflow
            rv = rvs_inflow
            tcool = tcools_inflow
            emissivity = emissivities_inflow
            weight = weights_inflow
            flow_file = '_inflow'
            gas_label = 'Inflowing gas'
            OVI_label = 'Inflowing O VI'
        if (i==2):
            density = densities_outflow
            temperature = temperatures_outflow
            metallicity = metallicities_outflow
            rv = rvs_outflow
            tcool = tcools_outflow
            emissivity = emissivities_outflow
            weight = weights_outflow
            flow_file = '_outflow'
            gas_label = 'Outflowing gas'
            OVI_label = 'Outflowing O VI'
        if (i==3):
            flow_file = '_inflow-outflow'

        fig = plt.figure(figsize=(15,10), dpi=250)
        ax1 = fig.add_subplot(2,3,1)
        ax2 = fig.add_subplot(2,3,2)
        ax3 = fig.add_subplot(2,3,3)
        ax4 = fig.add_subplot(2,3,4)
        ax5 = fig.add_subplot(2,3,5)

        if (i<3):
            ax1.hist(np.log10(density), weights=weight, bins=50, range=(-31,-23), density=True, histtype='stepfilled', lw=2, ls='-', ec=(0,0,0,1), fc=(0,0,0,0.2), label=gas_label)
            ax1.hist(np.log10(density), weights=emissivity, bins=50, range=(-31,-23), density=True, histtype='stepfilled', lw=2, ls='-', fc=(219/250., 29/250., 143/250., 0.2), ec=(219/250., 29/250., 143/250., 1), label=OVI_label)
            ax1.legend(loc=2, frameon=False, fontsize=12)
        else:
            ax1.hist(np.log10(densities_inflow), weights=weights_inflow, bins=50, range=(-31,-23), density=True, histtype='step', lw=2, ls='--', color=(84/250.,104/250.,184/250.,1), label='Inflowing gas')
            ax1.hist(np.log10(densities_outflow), weights=weights_outflow, bins=50, range=(-31,-23), density=True, histtype='step', lw=2, ls=':', color=(219/250., 92/250., 29/250.,1), label='Outflowing gas')
            ax1.hist(np.log10(densities_inflow), weights=emissivities_inflow, bins=50, range=(-31,-23), density=True, histtype='stepfilled', lw=2, ls='--', fc=(162/250., 29/250., 219/250., 0.2), ec=(162/250., 29/250., 219/250., 1), label='Inflowing O VI')
            ax1.hist(np.log10(densities_outflow), weights=emissivities_outflow, bins=50, range=(-31,-23), density=True, histtype='stepfilled', lw=2, ls=':', fc=(247/250.,211/250.,111/250., 0.2), ec=(247/250.,211/250.,111/250.,1), label='Outflowing O VI')
            ax1.legend(loc=2, fontsize=12, ncol=2)
        ax1.set_yscale('log')
        ax1.set_ylim(1e-3,10)
        ax1.set_xlabel('log Density [g/cm$^3$]', fontsize=14)
        ax1.set_ylabel('PDF', fontsize=14)
        ax1.tick_params(axis='both', which='both', direction='in', length=8, width=2, pad=5, labelsize=12, \
                top=True, right=True)
        
        if (i<3):
            ax2.hist(np.log10(temperature), weights=weight, bins=50, range=(3,9), density=True, histtype='stepfilled', lw=2, ls='--', ec=(0,0,0,1), fc=(0,0,0,0.3))
            ax2.hist(np.log10(temperature), weights=emissivity, bins=50, range=(3,9), density=True, histtype='stepfilled', lw=2, ls='-', fc=(219/250., 29/250., 143/250., 0.3), ec=(219/250., 29/250., 143/250., 1))
        else:
            ax2.hist(np.log10(temperatures_inflow), weights=weights_inflow, bins=50, range=(3,9), density=True, histtype='step', lw=2, ls='--', color=(84/250.,104/250.,184/250.,1), label='Inflowing gas')
            ax2.hist(np.log10(temperatures_outflow), weights=weights_outflow, bins=50, range=(3,9), density=True, histtype='step', lw=2, ls=':', color=(219/250., 92/250., 29/250.,1), label='Outflowing gas')
            ax2.hist(np.log10(temperatures_inflow), weights=emissivities_inflow, bins=50, range=(3,9), density=True, histtype='stepfilled', lw=2, ls='--', fc=(162/250., 29/250., 219/250., 0.2), ec=(162/250., 29/250., 219/250., 1), label='Inflowing O VI')
            ax2.hist(np.log10(temperatures_outflow), weights=emissivities_outflow, bins=50, range=(3,9), density=True, histtype='stepfilled', lw=2, ls=':', fc=(247/250.,211/250.,111/250., 0.2), ec=(247/250.,211/250.,111/250.,1), label='Outflowing O VI')
        ax2.set_xlabel('log Temperature [K]', fontsize=14)
        ax2.set_yscale('log')
        ax2.set_ylim(1e-3,10)
        ax2.tick_params(axis='both', which='both', direction='in', length=8, width=2, pad=5, labelsize=12, \
                top=True, right=True, labelleft=False)
        ax2.text(8.8, 6, halo_dict[args.halo], fontsize=14, va='top', ha='right')
        
        if (i<3):
            ax3.hist(np.log10(metallicity), weights=weight, bins=50, range=(-2.5,1.5), density=True, histtype='stepfilled', lw=2, ls='--', ec=(0,0,0,1), fc=(0,0,0,0.3))
            ax3.hist(np.log10(metallicity), weights=emissivity, bins=50, range=(-2.5,1.5), density=True, histtype='stepfilled', lw=2, ls='-', fc=(219/250., 29/250., 143/250., 0.3), ec=(219/250., 29/250., 143/250., 1))
        else:
            ax3.hist(np.log10(metallicities_inflow), weights=weights_inflow, bins=50, range=(-2.5,1.5), density=True, histtype='step', lw=2, ls='--', color=(84/250.,104/250.,184/250.,1), label='Inflowing gas')
            ax3.hist(np.log10(metallicities_outflow), weights=weights_outflow, bins=50, range=(-2.5,1.5), density=True, histtype='step', lw=2, ls=':', color=(219/250., 92/250., 29/250.,1), label='Outflowing gas')
            ax3.hist(np.log10(metallicities_inflow), weights=emissivities_inflow, bins=50, range=(-2.5,1.5), density=True, histtype='stepfilled', lw=2, ls='--', fc=(162/250., 29/250., 219/250., 0.2), ec=(162/250., 29/250., 219/250., 1), label='Inflowing O VI')
            ax3.hist(np.log10(metallicities_outflow), weights=emissivities_outflow, bins=50, range=(-2.5,1.5), density=True, histtype='stepfilled', lw=2, ls=':', fc=(247/250.,211/250.,111/250., 0.2), ec=(247/250.,211/250.,111/250.,1), label='Outflowing O VI')
        ax3.set_xlabel(r'log Metallicity [$Z_\odot$]', fontsize=14)
        ax3.set_yscale('log')
        ax3.set_ylim(1e-3,10)
        ax3.tick_params(axis='both', which='both', direction='in', length=8, width=2, pad=5, labelsize=12, \
                top=True, right=True, labelleft=False)
        ax3.text(-2.4, 6, '$z=%.2f$\n%.2f Gyr' % (ds.get_parameter('CosmologyCurrentRedshift'), ds.current_time.in_units('Gyr')), fontsize=14, va='top', ha='left')

        if (i<3):
            ax4.hist(rv/100., bins=50, weights=weight, range=(-5,10), density=True, histtype='stepfilled', lw=2, ls='--', ec=(0,0,0,1), fc=(0,0,0,0.3))
            ax4.hist(rv/100., weights=emissivity, bins=50, range=(-5,10), density=True, histtype='stepfilled', lw=2, ls='-', fc=(219/250., 29/250., 143/250., 0.3), ec=(219/250., 29/250., 143/250., 1))
        else:
            ax4.hist(rvs_inflow/100., bins=50, weights=weights_inflow, range=(-5,10), density=True, histtype='step', lw=2, ls='--', color=(84/250.,104/250.,184/250.,1), label='Inflowing gas')
            ax4.hist(rvs_outflow/100., bins=50, weights=weights_outflow, range=(-5,10), density=True, histtype='step', lw=2, ls=':', color=(219/250., 92/250., 29/250.,1), label='Outflowing gas')
            ax4.hist(rvs_inflow/100., weights=emissivities_inflow, bins=50, range=(-5,10), density=True, histtype='stepfilled', lw=2, ls='--', fc=(162/250., 29/250., 219/250., 0.2), ec=(162/250., 29/250., 219/250., 1), label='Inflowing O VI')
            ax4.hist(rvs_outflow/100., weights=emissivities_outflow, bins=50, range=(-5,10), density=True, histtype='stepfilled', lw=2, ls=':', fc=(247/250.,211/250.,111/250., 0.2), ec=(247/250.,211/250.,111/250.,1), label='Outflowing O VI')
        ax4.set_xlabel(r'Radial velocity [km/s]', fontsize=14)
        ax4.set_yscale('log')
        ax4.set_ylim(1e-3,10)
        ax4.set_xticks([-4, -2, 0, 2, 4, 6, 8, 10])
        ax4.set_xticklabels(['$-400$', '$-200$', '$0$', '$200$', '$400$', '$600$', '$800$', '$1000$'])
        ax4.set_ylabel('PDF', fontsize=14)
        ax4.tick_params(axis='both', which='both', direction='in', length=8, width=2, pad=5, labelsize=12, \
                top=True, right=True)
        
        if (i<3):
            ax5.hist(np.log10(tcool), bins=50, weights=weight, range=(-2,7), density=True, histtype='stepfilled', lw=2, ls='--', ec=(0,0,0,1), fc=(0,0,0,0.3))
            ax5.hist(np.log10(tcool), weights=emissivity, bins=50, range=(-2,7), density=True, histtype='stepfilled', lw=2, ls='-', fc=(219/250., 29/250., 143/250., 0.3), ec=(219/250., 29/250., 143/250., 1))
        else:
            ax5.hist(np.log10(tcools_inflow), bins=50, weights=weights_inflow, range=(-2,7), density=True, histtype='step', lw=2, ls='--', color=(84/250.,104/250.,184/250.,1), label='Inflowing gas')
            ax5.hist(np.log10(tcools_outflow), bins=50, weights=weights_outflow, range=(-2,7), density=True, histtype='step', lw=2, ls=':', color=(219/250., 92/250., 29/250.,1), label='Outflowing gas')
            ax5.hist(np.log10(tcools_inflow), weights=emissivities_inflow, bins=50, range=(-2,7), density=True, histtype='stepfilled', lw=2, ls='--', fc=(162/250., 29/250., 219/250., 0.2), ec=(162/250., 29/250., 219/250., 1), label='Inflowing O VI')
            ax5.hist(np.log10(tcools_outflow), weights=emissivities_outflow, bins=50, range=(-2,7), density=True, histtype='stepfilled', lw=2, ls=':', fc=(247/250.,211/250.,111/250., 0.2), ec=(247/250.,211/250.,111/250.,1), label='Outflowing O VI')
        ax5.set_xlabel(r'log Cooling time [Myr]', fontsize=14)
        ax5.set_yscale('log')
        ax5.set_ylim(1e-3,10)
        ax5.tick_params(axis='both', which='both', direction='in', length=8, width=2, pad=5, labelsize=12, \
                top=True, right=True, labelleft=False)
        
        plt.tight_layout()
        plt.subplots_adjust(wspace=0.)
        plt.savefig(prefix + 'Histograms/' + snap + '_gas-hist_OVI' + flow_file + save_suffix + '.png')

def hists_rad_bins(ds, refine_box, snap):
    '''Plots histograms of pixel densities, temperatures, metallicities, radial velocities, and cooling times along with
    emissivity-weighted histograms, in a few radial bins to see the radial profile.'''

    sph = ds.sphere(center=ds.halo_center_kpc, radius=(50., 'kpc'))
    sph_inflow = sph.include_below(('gas','radial_velocity_corrected'), -100.)
    sph_outflow = sph.include_above(('gas','radial_velocity_corrected'), 100.)
    

    radii = sph['gas','radius_corrected'].in_units('kpc').v
    radii_inflow = sph_inflow['gas','radius_corrected'].in_units('kpc').v
    radii_outflow = sph_outflow['gas','radius_corrected'].in_units('kpc').v

    densities = sph['gas','density'].in_units('g/cm**3').v
    densities_inflow = sph_inflow['gas','density'].in_units('g/cm**3').v
    densities_outflow = sph_outflow['gas','density'].in_units('g/cm**3').v
    temperatures = sph['gas','temperature'].in_units('K').v
    temperatures_inflow = sph_inflow['gas','temperature'].in_units('K').v
    temperatures_outflow = sph_outflow['gas','temperature'].in_units('K').v
    metallicities = sph['gas','metallicity'].in_units('Z_sun').v
    metallicities_inflow = sph_inflow['gas','metallicity'].in_units('Zsun').v
    metallicities_outflow = sph_outflow['gas','metallicity'].in_units('Zsun').v
    rvs = sph['gas','radial_velocity_corrected'].in_units('km/s').v
    rvs_inflow = sph_inflow['gas','radial_velocity_corrected'].in_units('km/s').v
    rvs_outflow = sph_outflow['gas','radial_velocity_corrected'].in_units('km/s').v
    tcools = sph['gas','cooling_time'].in_units('Myr').v
    tcools_inflow = sph_inflow['gas','cooling_time'].in_units('Myr').v
    tcools_outflow = sph_outflow['gas','cooling_time'].in_units('Myr').v
    emissivities = sph['gas','Emission_OVI'].in_units(emission_units_ALT).v
    emissivities_inflow = sph_inflow['gas','Emission_OVI'].in_units(emission_units_ALT).v
    emissivities_outflow = sph_outflow['gas','Emission_OVI'].in_units(emission_units_ALT).v
    if (args.weight=='volume'):
        weights = np.log10(sph['gas','cell_volume'].in_units('kpc**3').v)
        weights_inflow = np.log10(sph_inflow['gas','cell_volume'].in_units('kpc**3').v)
        weights_outflow = np.log10(sph_outflow['gas','cell_volume'].in_units('kpc**3').v)
    else:
        weights = np.log10(sph['gas','cell_mass'].in_units('Msun').v)
        weights_inflow = np.log10(sph_inflow['gas','cell_mass'].in_units('Msun').v)
        weights_outflow = np.log10(sph_outflow['gas','cell_mass'].in_units('Msun').v)

    #print(np.min(emissivities), np.max(emissivities))
    #emissivities[emissivities < 1e-300] = np.nan
    #print(np.nanmin(emissivities), np.nanmax(emissivities), np.nanmedian(emissivities), np.nanmean(emissivities))
    #plt.hist(np.log10(emissivities), bins=300)
    #plt.show()

    # Define the density cut between disk and CGM to vary smoothly between 1 and 0.1 between z = 0.5 and z = 0.25,
    # with it being 1 at higher redshifts and 0.1 at lower redshifts
    current_time = ds.current_time.in_units('Myr').v
    if (current_time<=7091.48):
        density_cut_factor = 20. - 19.*current_time/7091.48
    elif (current_time<=8656.88):
        density_cut_factor = 1.
    elif (current_time<=10787.12):
        density_cut_factor = 1. - 0.9*(current_time-8656.88)/2130.24
    else:
        density_cut_factor = 0.1
    #cgm = (densities < density_cut_factor * cgm_density_max)
    #cgm_in = (densities_inflow < density_cut_factor * cgm_density_max)
    #cgm_out = (densities_outflow < density_cut_factor * cgm_density_max)
    cgm = (densities > 0.)
    cgm_in = (densities_inflow > 0.)
    cgm_out = (densities_outflow > 0.)

    for i in range(4):
        if (i==0):
            radius = radii[cgm]
            density = densities[cgm]
            temperature = temperatures[cgm]
            metallicity = metallicities[cgm]
            rv = rvs[cgm]
            tcool = tcools[cgm]
            emissivity = emissivities[cgm]
            weight = weights[cgm]
            flow_file = ''
            gas_label = 'All gas'
            OVI_label = 'O VI emitting gas'
        if (i==1):
            radius = radii_inflow[cgm_in]
            density = densities_inflow[cgm_in]
            temperature = temperatures_inflow[cgm_in]
            metallicity = metallicities_inflow[cgm_in]
            rv = rvs_inflow[cgm_in]
            tcool = tcools_inflow[cgm_in]
            emissivity = emissivities_inflow[cgm_in]
            weight = weights_inflow[cgm_in]
            flow_file = '_inflow'
            gas_label = 'Inflowing gas'
            OVI_label = 'Inflowing O VI'
        if (i==2):
            radius = radii_outflow[cgm_out]
            density = densities_outflow[cgm_out]
            temperature = temperatures_outflow[cgm_out]
            metallicity = metallicities_outflow[cgm_out]
            rv = rvs_outflow[cgm_out]
            tcool = tcools_outflow[cgm_out]
            emissivity = emissivities_outflow[cgm_out]
            weight = weights_outflow[cgm_out]
            flow_file = '_outflow'
            gas_label = 'Outflowing gas'
            OVI_label = 'Outflowing O VI'
        if (i==3):
            flow_file = '_inflow-outflow'

        fig = plt.figure(figsize=(18,8), dpi=250)
        outer = mpl.gridspec.GridSpec(2, 3, hspace=0.24, wspace=0.23, left=0.07, right=0.99, top=0.98, bottom=0.08)
        ax_den = outer[0, 0]
        ax_temp = outer[0, 1]
        ax_met = outer[0, 2]
        ax_rv = outer[1, 0]
        ax_tcool = outer[1, 1]
        inner_den = mpl.gridspec.GridSpecFromSubplotSpec(1, 5, subplot_spec=ax_den, wspace=0.)
        inner_temp = mpl.gridspec.GridSpecFromSubplotSpec(1, 5, subplot_spec=ax_temp, wspace=0.)
        inner_met = mpl.gridspec.GridSpecFromSubplotSpec(1, 5, subplot_spec=ax_met, wspace=0.)
        inner_rv = mpl.gridspec.GridSpecFromSubplotSpec(1, 5, subplot_spec=ax_rv, wspace=0.)
        inner_tcool = mpl.gridspec.GridSpecFromSubplotSpec(1, 5, subplot_spec=ax_tcool, wspace=0.)
        
        rad_bins = np.array([0.,10.,20.,30.,40.,50.])

        for r in range(len(rad_bins)-1):
            inn_r = rad_bins[r]
            out_r = rad_bins[r+1]

            density_bin = density[(radius >= inn_r) & (radius < out_r)]
            temperature_bin = temperature[(radius >= inn_r) & (radius < out_r)]
            metallicity_bin = metallicity[(radius >= inn_r) & (radius < out_r)]
            rv_bin = rv[(radius >= inn_r) & (radius < out_r)]
            tcool_bin = tcool[(radius >= inn_r) & (radius < out_r)]
            emissivity_bin = emissivity[(radius >= inn_r) & (radius < out_r)]
            weight_bin = weight[(radius >= inn_r) & (radius < out_r)]

            density_inflow_bin = densities_inflow[(radii_inflow >= inn_r) & (radii_inflow < out_r) & cgm_in]
            temperature_inflow_bin = temperatures_inflow[(radii_inflow >= inn_r) & (radii_inflow < out_r) & cgm_in]
            metallicity_inflow_bin = metallicities_inflow[(radii_inflow >= inn_r) & (radii_inflow < out_r) & cgm_in]
            rv_inflow_bin = rvs_inflow[(radii_inflow >= inn_r) & (radii_inflow < out_r) & cgm_in]
            tcool_inflow_bin = tcools_inflow[(radii_inflow >= inn_r) & (radii_inflow < out_r) & cgm_in]
            emissivity_inflow_bin = emissivities_inflow[(radii_inflow >= inn_r) & (radii_inflow < out_r) & cgm_in]
            weight_inflow_bin = weights_inflow[(radii_inflow >= inn_r) & (radii_inflow < out_r) & cgm_in]

            density_outflow_bin = densities_outflow[(radii_outflow >= inn_r) & (radii_outflow < out_r) & cgm_out]
            temperature_outflow_bin = temperatures_outflow[(radii_outflow >= inn_r) & (radii_outflow < out_r) & cgm_out]
            metallicity_outflow_bin = metallicities_outflow[(radii_outflow >= inn_r) & (radii_outflow < out_r) & cgm_out]
            rv_outflow_bin = rvs_outflow[(radii_outflow >= inn_r) & (radii_outflow < out_r) & cgm_out]
            tcool_outflow_bin = tcools_outflow[(radii_outflow >= inn_r) & (radii_outflow < out_r) & cgm_out]
            emissivity_outflow_bin = emissivities_outflow[(radii_outflow >= inn_r) & (radii_outflow < out_r) & cgm_out]
            weight_outflow_bin = weights_outflow[(radii_outflow >= inn_r) & (radii_outflow < out_r) & cgm_out]

            ax1 = fig.add_subplot(inner_den[0, r])
            if (i<3):
                ax1.hist(np.log10(density_bin), weights=weight_bin, bins=50, range=(-31,-23), density=True, histtype='stepfilled', orientation='horizontal', lw=2, ls='--', ec=(0,0,0,1), fc=(0,0,0,0.2), label=gas_label)
                ax1.hist(np.log10(density_bin), weights=emissivity_bin, bins=50, range=(-31,-23), density=True, histtype='stepfilled', orientation='horizontal', lw=2, ls='-', fc=(219/250., 29/250., 143/250., 0.2), ec=(219/250., 29/250., 143/250., 1), label=OVI_label)
                #ax1.legend(loc=2, frameon=False, fontsize=12)
            else:
                ax1.hist(np.log10(density_inflow_bin), weights=weight_inflow_bin, bins=50, range=(-31,-23), density=True, histtype='step', orientation='horizontal', lw=2, ls='--', color=(84/250.,104/250.,184/250.,1), label='Inflowing gas')
                ax1.hist(np.log10(density_outflow_bin), weights=weight_outflow_bin, bins=50, range=(-31,-23), density=True, histtype='step', orientation='horizontal', lw=2, ls=':', color=(219/250., 92/250., 29/250.,1), label='Outflowing gas')
                ax1.hist(np.log10(density_inflow_bin), weights=emissivity_inflow_bin, bins=50, range=(-31,-23), density=True, histtype='stepfilled', orientation='horizontal', lw=2, ls='--', fc=(162/250., 29/250., 219/250., 0.2), ec=(162/250., 29/250., 219/250., 1), label='Inflowing O VI')
                ax1.hist(np.log10(density_outflow_bin), weights=emissivity_outflow_bin, bins=50, range=(-31,-23), density=True, histtype='stepfilled', orientation='horizontal', lw=2, ls=':', fc=(247/250.,211/250.,111/250., 0.2), ec=(247/250.,211/250.,111/250.,1), label='Outflowing O VI')
                #ax1.legend(loc=2, fontsize=12, ncol=2)
            ax1.set_xscale('log')
            ax1.axis([1e-3,10,-31,-23])
            if (r==0): 
                ax1.tick_params(axis='both', which='both', direction='in', length=8, width=2, pad=5, labelsize=18, \
                    top=True, right=False, labelleft=True)
                ax1.set_ylabel('Density [g/cm$^3$]', fontsize=20)
                yticks = [-30,-28,-26,-24]
                ax1.set_yticks(yticks)
                ax1.set_yticklabels(r"$10^{{{}}}$".format(y) for y in yticks)
                ax1.set_xticks([1e-3,10])
                ax1.set_xticklabels([0,10])
            else:
                if (r==len(rad_bins)-2):
                    ax1.tick_params(axis='both', which='both', direction='in', length=8, width=2, pad=5, labelsize=18, \
                        top=True, right=True, left=False, labelleft=False)
                else:
                    ax1.tick_params(axis='both', which='both', direction='in', length=8, width=2, pad=5, labelsize=18, \
                        top=True, left=False, right=False, labelleft=False)
                ax1.set_xticks([10])
                ax1.set_xticklabels([int(out_r)])
            if (r==2): ax1.set_xlabel('Galactocentric Radius [kpc]', fontsize=18)

            ax2 = fig.add_subplot(inner_temp[0, r])
            if (i<3):
                ax2.hist(np.log10(temperature_bin), weights=weight_bin, bins=50, range=(3,9), density=True, histtype='stepfilled', orientation='horizontal', lw=2, ls='--', ec=(0,0,0,1), fc=(0,0,0,0.2))
                ax2.hist(np.log10(temperature_bin), weights=emissivity_bin, bins=50, range=(3,9), density=True, histtype='stepfilled', orientation='horizontal', lw=2, ls='-', fc=(219/250., 29/250., 143/250., 0.3), ec=(219/250., 29/250., 143/250., 1))
            else:
                ax2.hist(np.log10(temperature_inflow_bin), weights=weight_inflow_bin, bins=50, range=(3,9), density=True, histtype='step', orientation='horizontal', lw=2, ls='--', color=(84/250.,104/250.,184/250.,1), label='Inflowing gas')
                ax2.hist(np.log10(temperature_outflow_bin), weights=weight_outflow_bin, bins=50, range=(3,9), density=True, histtype='step', orientation='horizontal', lw=2, ls=':', color=(219/250., 92/250., 29/250.,1), label='Outflowing gas')
                ax2.hist(np.log10(temperature_inflow_bin), weights=emissivity_inflow_bin, bins=50, range=(3,9), density=True, histtype='stepfilled', orientation='horizontal', lw=2, ls='--', fc=(162/250., 29/250., 219/250., 0.2), ec=(162/250., 29/250., 219/250., 1), label='Inflowing O VI')
                ax2.hist(np.log10(temperature_outflow_bin), weights=emissivity_outflow_bin, bins=50, range=(3,9), density=True, histtype='stepfilled', orientation='horizontal', lw=2, ls=':', fc=(247/250.,211/250.,111/250., 0.2), ec=(247/250.,211/250.,111/250.,1), label='Outflowing O VI')
            ax2.set_xscale('log')
            ax2.axis([1e-3,10,3,9])
            if (r==0):
                ax2.tick_params(axis='both', which='both', direction='in', length=8, width=2, pad=5, labelsize=18, \
                    top=True, right=False, labelleft=True)
                ax2.set_ylabel('Temperature [K]', fontsize=20)
                yticks = [3,4,5,6,7,8,9]
                ax2.set_yticks(yticks)
                ax2.set_yticklabels(r"$10^{{{}}}$".format(y) for y in yticks)
                ax2.set_xticks([1e-3,10])
                ax2.set_xticklabels([0,10])
            else:
                if (r==len(rad_bins)-2):
                    ax2.tick_params(axis='both', which='both', direction='in', length=8, width=2, pad=5, labelsize=18, \
                        top=True, right=True, left=False, labelleft=False)
                else:
                    ax2.tick_params(axis='both', which='both', direction='in', length=8, width=2, pad=5, labelsize=18, \
                        top=True, left=False, right=False, labelleft=False)
                ax2.set_xticks([10])
                ax2.set_xticklabels([int(out_r)])
            if (r==2): ax2.set_xlabel('Galactocentric Radius [kpc]', fontsize=20)
            
            ax3 = fig.add_subplot(inner_met[0, r])
            if (i<3):
                ax3.hist(np.log10(metallicity_bin), weights=weight_bin, bins=50, range=(-2.5,1.5), density=True, histtype='stepfilled', orientation='horizontal', lw=2, ls='--', ec=(0,0,0,1), fc=(0,0,0,0.2))
                ax3.hist(np.log10(metallicity_bin), weights=emissivity_bin, bins=50, range=(-2.5,1.5), density=True, histtype='stepfilled', orientation='horizontal', lw=2, ls='-', fc=(219/250., 29/250., 143/250., 0.3), ec=(219/250., 29/250., 143/250., 1))
            else:
                ax3.hist(np.log10(metallicity_inflow_bin), weights=weight_inflow_bin, bins=50, range=(-2.5,1.5), density=True, histtype='step', orientation='horizontal', lw=2, ls='--', color=(84/250.,104/250.,184/250.,1), label='Inflowing gas')
                ax3.hist(np.log10(metallicity_outflow_bin), weights=weight_outflow_bin, bins=50, range=(-2.5,1.5), density=True, histtype='step', orientation='horizontal', lw=2, ls=':', color=(219/250., 92/250., 29/250.,1), label='Outflowing gas')
                ax3.hist(np.log10(metallicity_inflow_bin), weights=emissivity_inflow_bin, bins=50, range=(-2.5,1.5), density=True, histtype='stepfilled', orientation='horizontal', lw=2, ls='--', fc=(162/250., 29/250., 219/250., 0.2), ec=(162/250., 29/250., 219/250., 1), label='Inflowing O VI')
                ax3.hist(np.log10(metallicity_outflow_bin), weights=emissivity_outflow_bin, bins=50, range=(-2.5,1.5), density=True, histtype='stepfilled', orientation='horizontal', lw=2, ls=':', fc=(247/250.,211/250.,111/250., 0.2), ec=(247/250.,211/250.,111/250.,1), label='Outflowing O VI')
            ax3.set_xscale('log')
            ax3.axis([1e-3,10,-2.5,1.5])
            if (r==0):
                ax3.tick_params(axis='both', which='both', direction='in', length=8, width=2, pad=5, labelsize=18, \
                    top=True, right=False, labelleft=True)
                ax3.set_ylabel(r'Metallicity [$Z_\odot$]', fontsize=20)
                yticks = [-2,-1,0,1]
                ax3.set_yticks(yticks)
                ax3.set_yticklabels(r"$10^{{{}}}$".format(y) for y in yticks)
                ax3.set_xticks([1e-3,10])
                ax3.set_xticklabels([0,10])
            else:
                if (r==len(rad_bins)-2):
                    ax3.tick_params(axis='both', which='both', direction='in', length=8, width=2, pad=5, labelsize=18, \
                        top=True, right=True, left=False, labelleft=False)
                else:
                    ax3.tick_params(axis='both', which='both', direction='in', length=8, width=2, pad=5, labelsize=18, \
                        top=True, left=False, right=False, labelleft=False)
                ax3.set_xticks([10])
                ax3.set_xticklabels([int(out_r)])
            if (r==2): ax3.set_xlabel('Galactocentric Radius [kpc]', fontsize=20)
            
            ax4 = fig.add_subplot(inner_rv[0, r])
            if (i<3):
                ax4.hist(rv_bin/100., bins=50, weights=weight_bin, range=(-5,10), density=True, histtype='stepfilled', orientation='horizontal', lw=2, ls='--', ec=(0,0,0,1), fc=(0,0,0,0.2))
                ax4.hist(rv_bin/100., weights=emissivity_bin, bins=50, range=(-5,10), density=True, histtype='stepfilled', orientation='horizontal', lw=2, ls='-', fc=(219/250., 29/250., 143/250., 0.3), ec=(219/250., 29/250., 143/250., 1))
            else:
                ax4.hist(rv_inflow_bin/100., bins=50, weights=weight_inflow_bin, range=(-5,10), density=True, histtype='step', orientation='horizontal', lw=2, ls='--', color=(84/250.,104/250.,184/250.,1), label='Inflowing gas')
                ax4.hist(rv_outflow_bin/100., bins=50, weights=weight_outflow_bin, range=(-5,10), density=True, histtype='step', orientation='horizontal', lw=2, ls=':', color=(219/250., 92/250., 29/250.,1), label='Outflowing gas')
                ax4.hist(rv_inflow_bin/100., weights=emissivity_inflow_bin, bins=50, range=(-5,10), density=True, histtype='stepfilled', orientation='horizontal', lw=2, ls='--', fc=(162/250., 29/250., 219/250., 0.2), ec=(162/250., 29/250., 219/250., 1), label='Inflowing O VI')
                ax4.hist(rv_outflow_bin/100., weights=emissivity_outflow_bin, bins=50, range=(-5,10), density=True, histtype='stepfilled', orientation='horizontal', lw=2, ls=':', fc=(247/250.,211/250.,111/250., 0.2), ec=(247/250.,211/250.,111/250.,1), label='Outflowing O VI')
            ax4.set_xscale('log')
            ax4.axis([1e-3,10,-5,10])
            if (r==0):
                ax4.tick_params(axis='both', which='both', direction='in', length=8, width=2, pad=5, labelsize=18, \
                    top=True, right=False, labelleft=True)
                ax4.set_ylabel('Radial Velocity [km/s]', fontsize=20)
                ax4.set_xticks([1e-3,10])
                ax4.set_xticklabels([0,10])
                ax4.set_yticks([-4,-2,0,2,4,6,8,10])
                ax4.set_yticklabels(['$-400$','$-200$','$0$','$200$','$400$','$600$','$800$','$1000$'])
            else:
                if (r==len(rad_bins)-2):
                    ax4.tick_params(axis='both', which='both', direction='in', length=8, width=2, pad=5, labelsize=18, \
                        top=True, right=True, left=False, labelleft=False)
                else:
                    ax4.tick_params(axis='both', which='both', direction='in', length=8, width=2, pad=5, labelsize=18, \
                        top=True, left=False, right=False, labelleft=False)
                ax4.set_xticks([10])
                ax4.set_xticklabels([int(out_r)])
            if (r==2): ax4.set_xlabel('Galactocentric Radius [kpc]', fontsize=20)
                
            ax5 = fig.add_subplot(inner_tcool[0, r])
            if (i<3):
                ax5.hist(np.log10(tcool_bin), bins=50, weights=weight_bin, range=(-2,7), density=True, histtype='stepfilled', orientation='horizontal', lw=2, ls='--', ec=(0,0,0,1), fc=(0,0,0,0.2), label='All gas')
                ax5.hist(np.log10(tcool_bin), weights=emissivity_bin, bins=50, range=(-2,7), density=True, histtype='stepfilled', orientation='horizontal', lw=2, ls='-', fc=(219/250., 29/250., 143/250., 0.3), ec=(219/250., 29/250., 143/250., 1), label='O VI emitting gas')
            else:
                ax5.hist(np.log10(tcool_inflow_bin), bins=50, weights=weight_inflow_bin, range=(-2,7), density=True, histtype='step', orientation='horizontal', lw=2, ls='--', color=(84/250.,104/250.,184/250.,1), label='Inflowing gas')
                ax5.hist(np.log10(tcool_outflow_bin), bins=50, weights=weight_outflow_bin, range=(-2,7), density=True, histtype='step', orientation='horizontal', lw=2, ls=':', color=(219/250., 92/250., 29/250.,1), label='Outflowing gas')
                ax5.hist(np.log10(tcool_inflow_bin), weights=emissivity_inflow_bin, bins=50, range=(-2,7), density=True, histtype='stepfilled', orientation='horizontal', lw=2, ls='--', fc=(162/250., 29/250., 219/250., 0.2), ec=(162/250., 29/250., 219/250., 1), label='Inflowing O VI')
                ax5.hist(np.log10(tcool_outflow_bin), weights=emissivity_outflow_bin, bins=50, range=(-2,7), density=True, histtype='stepfilled', orientation='horizontal', lw=2, ls=':', fc=(247/250.,211/250.,111/250., 0.2), ec=(247/250.,211/250.,111/250.,1), label='Outflowing O VI')
            ax5.set_xscale('log')
            ax5.axis([1e-3,10,-2,7])
            if (r==0): 
                ax5.tick_params(axis='both', which='both', direction='in', length=8, width=2, pad=5, labelsize=18, \
                    top=True, right=False, labelleft=True)
                yticks = [-2,0,2,4,6]
                ax5.set_yticks(yticks)
                ax5.set_yticklabels(r"$10^{{{}}}$".format(y) for y in yticks)
                ax5.set_ylabel('Cooling Time [Myr]', fontsize=20)
                ax5.set_xticks([1e-3,10])
                ax5.set_xticklabels([0,10])
            else:
                if (r==len(rad_bins)-2):
                    ax5.tick_params(axis='both', which='both', direction='in', length=8, width=2, pad=5, labelsize=18, \
                        top=True, right=True, left=False, labelleft=False)
                    ax5.text(1.3, 0.8, '$z=0$', fontsize=20, va='top', ha='left', transform=ax5.transAxes)
                    ax5.text(1.3, 0.9, halo_dict[args.halo], fontsize=20, va='top', ha='left', transform=ax5.transAxes)
                else:
                    ax5.tick_params(axis='both', which='both', direction='in', length=8, width=2, pad=5, labelsize=18, \
                        top=True, left=False, right=False, labelleft=False)
                ax5.set_xticks([10])
                ax5.set_xticklabels([int(out_r)])
            if (r==2): ax5.set_xlabel('Galactocentric Radius [kpc]', fontsize=20)
            if (r==4): ax5.legend(loc=2, fontsize=20, frameon=False, bbox_to_anchor=(1.,0.7))

        plt.savefig(prefix + 'Histograms/' + snap + '_gas-hist_OVI_radbins' + flow_file + save_suffix + '.png')

def hists_rad_bins_all_halos(halos, outs):
    '''Plots histograms of pixel densities, temperatures, metallicities, radial velocities, and cooling times along with
    emissivity-weighted histograms, in a few radial bins to see the radial profile. Stacks outputs from all 6 halos.'''

    radius_halos = np.array([])
    density_halos = np.array([])
    temperature_halos = np.array([])
    metallicity_halos = np.array([])
    rv_halos = np.array([])
    tcool_halos = np.array([])
    emissivity_halos = np.array([])
    weight_halos = np.array([])

    for h in range(len(halos)):
        snap = outs[h]
        print(halos[h], snap)
        snap_name = foggie_dir + 'halo_00' + halos[h] + '/nref11c_nref9f/' + snap + '/' + snap
        halo_c_v_name = code_path + 'halo_infos/00' + halos[h] + '/nref11c_nref9f/halo_c_v'
        trackname = code_path + 'halo_tracks/00' + halos[h] + '/nref11n_selfshield_15/halo_track_200kpc_nref9'
        ds, refine_box = foggie_load(snap_name, trackfile_name=trackname, do_filter_particles=False, halo_c_v_name=halo_c_v_name)

        sph = ds.sphere(center=ds.halo_center_kpc, radius=(50., 'kpc'))
        
        radii = sph['gas','radius_corrected'].in_units('kpc').v
        densities = sph['gas','density'].in_units('g/cm**3').v
        temperatures = sph['gas','temperature'].in_units('K').v
        metallicities = sph['gas','metallicity'].in_units('Z_sun').v
        rvs = sph['gas','radial_velocity_corrected'].in_units('km/s').v
        tcools = sph['gas','cooling_time'].in_units('Myr').v
        emissivities = sph['gas','Emission_OVI'].in_units(emission_units_ALT).v
        if (args.weight=='volume'):
            weights = np.log10(sph['gas','cell_volume'].in_units('kpc**3').v)
        else:
            weights = np.log10(sph['gas','cell_mass'].in_units('Msun').v)

        # Define the density cut between disk and CGM to vary smoothly between 1 and 0.1 between z = 0.5 and z = 0.25,
        # with it being 1 at higher redshifts and 0.1 at lower redshifts
        current_time = ds.current_time.in_units('Myr').v
        if (current_time<=7091.48):
            density_cut_factor = 20. - 19.*current_time/7091.48
        elif (current_time<=8656.88):
            density_cut_factor = 1.
        elif (current_time<=10787.12):
            density_cut_factor = 1. - 0.9*(current_time-8656.88)/2130.24
        else:
            density_cut_factor = 0.1
        cgm = (densities > 0.)

        radius = radii[cgm]
        density = densities[cgm]
        temperature = temperatures[cgm]
        metallicity = metallicities[cgm]
        rv = rvs[cgm]
        tcool = tcools[cgm]
        emissivity = emissivities[cgm]
        weight = weights[cgm]

        radius_halos = np.hstack([radius_halos, radius])
        density_halos = np.hstack([density_halos, density])
        temperature_halos = np.hstack([temperature_halos, temperature])
        metallicity_halos = np.hstack([metallicity_halos, metallicity])
        rv_halos = np.hstack([rv_halos, rv])
        tcool_halos = np.hstack([tcool_halos, tcool])
        emissivity_halos = np.hstack([emissivity_halos, emissivity])
        weight_halos = np.hstack([weight_halos, weight])

    gas_label = 'All gas'
    OVI_label = 'O VI emitting gas'

    fig = plt.figure(figsize=(18,8), dpi=250)
    outer = mpl.gridspec.GridSpec(2, 3, hspace=0.24, wspace=0.23, left=0.07, right=0.99, top=0.98, bottom=0.08)
    ax_den = outer[0, 0]
    ax_temp = outer[0, 1]
    ax_met = outer[0, 2]
    ax_rv = outer[1, 0]
    ax_tcool = outer[1, 1]
    inner_den = mpl.gridspec.GridSpecFromSubplotSpec(1, 5, subplot_spec=ax_den, wspace=0.)
    inner_temp = mpl.gridspec.GridSpecFromSubplotSpec(1, 5, subplot_spec=ax_temp, wspace=0.)
    inner_met = mpl.gridspec.GridSpecFromSubplotSpec(1, 5, subplot_spec=ax_met, wspace=0.)
    inner_rv = mpl.gridspec.GridSpecFromSubplotSpec(1, 5, subplot_spec=ax_rv, wspace=0.)
    inner_tcool = mpl.gridspec.GridSpecFromSubplotSpec(1, 5, subplot_spec=ax_tcool, wspace=0.)
        
    rad_bins = np.array([0.,10.,20.,30.,40.,50.])

    for r in range(len(rad_bins)-1):
        inn_r = rad_bins[r]
        out_r = rad_bins[r+1]

        density_bin = density_halos[(radius_halos >= inn_r) & (radius_halos < out_r)]
        temperature_bin = temperature_halos[(radius_halos >= inn_r) & (radius_halos < out_r)]
        metallicity_bin = metallicity_halos[(radius_halos >= inn_r) & (radius_halos < out_r)]
        rv_bin = rv_halos[(radius_halos >= inn_r) & (radius_halos < out_r)]
        tcool_bin = tcool_halos[(radius_halos >= inn_r) & (radius_halos < out_r)]
        emissivity_bin = emissivity_halos[(radius_halos >= inn_r) & (radius_halos < out_r)]
        weight_bin = weight_halos[(radius_halos >= inn_r) & (radius_halos < out_r)]

        ax1 = fig.add_subplot(inner_den[0, r])
        ax1.hist(np.log10(density_bin), weights=weight_bin, bins=50, range=(-31,-23), density=True, histtype='stepfilled', orientation='horizontal', lw=2, ls='--', ec=(0,0,0,1), fc=(0,0,0,0.2), label=gas_label)
        ax1.hist(np.log10(density_bin), weights=emissivity_bin, bins=50, range=(-31,-23), density=True, histtype='stepfilled', orientation='horizontal', lw=2, ls='-', fc=(219/250., 29/250., 143/250., 0.2), ec=(219/250., 29/250., 143/250., 1), label=OVI_label)
        #ax1.legend(loc=2, frameon=False, fontsize=12)
        ax1.set_xscale('log')
        ax1.axis([1e-3,10,-31,-23])
        if (r==0): 
            ax1.tick_params(axis='both', which='both', direction='in', length=8, width=2, pad=5, labelsize=18, \
                top=True, right=False, labelleft=True)
            ax1.set_ylabel('Density [g/cm$^3$]', fontsize=20)
            yticks = [-30,-28,-26,-24]
            ax1.set_yticks(yticks)
            ax1.set_yticklabels(r"$10^{{{}}}$".format(y) for y in yticks)
            ax1.set_xticks([1e-3,10])
            ax1.set_xticklabels([0,10])
        else:
            if (r==len(rad_bins)-2):
                ax1.tick_params(axis='both', which='both', direction='in', length=8, width=2, pad=5, labelsize=18, \
                    top=True, right=True, left=False, labelleft=False)
            else:
                ax1.tick_params(axis='both', which='both', direction='in', length=8, width=2, pad=5, labelsize=18, \
                    top=True, left=False, right=False, labelleft=False)
            ax1.set_xticks([10])
            ax1.set_xticklabels([int(out_r)])
        if (r==2): ax1.set_xlabel('Galactocentric Radius [kpc]', fontsize=18)

        ax2 = fig.add_subplot(inner_temp[0, r])
        ax2.hist(np.log10(temperature_bin), weights=weight_bin, bins=50, range=(3,9), density=True, histtype='stepfilled', orientation='horizontal', lw=2, ls='--', ec=(0,0,0,1), fc=(0,0,0,0.2))
        ax2.hist(np.log10(temperature_bin), weights=emissivity_bin, bins=50, range=(3,9), density=True, histtype='stepfilled', orientation='horizontal', lw=2, ls='-', fc=(219/250., 29/250., 143/250., 0.3), ec=(219/250., 29/250., 143/250., 1))
        ax2.set_xscale('log')
        ax2.axis([1e-3,10,3,9])
        if (r==0):
            ax2.tick_params(axis='both', which='both', direction='in', length=8, width=2, pad=5, labelsize=18, \
                top=True, right=False, labelleft=True)
            ax2.set_ylabel('Temperature [K]', fontsize=20)
            yticks = [3,4,5,6,7,8,9]
            ax2.set_yticks(yticks)
            ax2.set_yticklabels(r"$10^{{{}}}$".format(y) for y in yticks)
            ax2.set_xticks([1e-3,10])
            ax2.set_xticklabels([0,10])
        else:
            if (r==len(rad_bins)-2):
                ax2.tick_params(axis='both', which='both', direction='in', length=8, width=2, pad=5, labelsize=18, \
                    top=True, right=True, left=False, labelleft=False)
            else:
                ax2.tick_params(axis='both', which='both', direction='in', length=8, width=2, pad=5, labelsize=18, \
                    top=True, left=False, right=False, labelleft=False)
            ax2.set_xticks([10])
            ax2.set_xticklabels([int(out_r)])
        if (r==2): ax2.set_xlabel('Galactocentric Radius [kpc]', fontsize=20)
        
        ax3 = fig.add_subplot(inner_met[0, r])
        ax3.hist(np.log10(metallicity_bin), weights=weight_bin, bins=50, range=(-2.5,1.5), density=True, histtype='stepfilled', orientation='horizontal', lw=2, ls='--', ec=(0,0,0,1), fc=(0,0,0,0.2))
        ax3.hist(np.log10(metallicity_bin), weights=emissivity_bin, bins=50, range=(-2.5,1.5), density=True, histtype='stepfilled', orientation='horizontal', lw=2, ls='-', fc=(219/250., 29/250., 143/250., 0.3), ec=(219/250., 29/250., 143/250., 1))
        ax3.set_xscale('log')
        ax3.axis([1e-3,10,-2.5,1.5])
        if (r==0):
            ax3.tick_params(axis='both', which='both', direction='in', length=8, width=2, pad=5, labelsize=18, \
                top=True, right=False, labelleft=True)
            ax3.set_ylabel(r'Metallicity [$Z_\odot$]', fontsize=20)
            yticks = [-2,-1,0,1]
            ax3.set_yticks(yticks)
            ax3.set_yticklabels(r"$10^{{{}}}$".format(y) for y in yticks)
            ax3.set_xticks([1e-3,10])
            ax3.set_xticklabels([0,10])
        else:
            if (r==len(rad_bins)-2):
                ax3.tick_params(axis='both', which='both', direction='in', length=8, width=2, pad=5, labelsize=18, \
                    top=True, right=True, left=False, labelleft=False)
            else:
                ax3.tick_params(axis='both', which='both', direction='in', length=8, width=2, pad=5, labelsize=18, \
                    top=True, left=False, right=False, labelleft=False)
            ax3.set_xticks([10])
            ax3.set_xticklabels([int(out_r)])
        if (r==2): ax3.set_xlabel('Galactocentric Radius [kpc]', fontsize=20)
        
        ax4 = fig.add_subplot(inner_rv[0, r])
        ax4.hist(rv_bin/100., bins=50, weights=weight_bin, range=(-5,10), density=True, histtype='stepfilled', orientation='horizontal', lw=2, ls='--', ec=(0,0,0,1), fc=(0,0,0,0.2))
        ax4.hist(rv_bin/100., weights=emissivity_bin, bins=50, range=(-5,10), density=True, histtype='stepfilled', orientation='horizontal', lw=2, ls='-', fc=(219/250., 29/250., 143/250., 0.3), ec=(219/250., 29/250., 143/250., 1))
        ax4.set_xscale('log')
        ax4.axis([1e-3,10,-5,10])
        if (r==0):
            ax4.tick_params(axis='both', which='both', direction='in', length=8, width=2, pad=5, labelsize=18, \
                top=True, right=False, labelleft=True)
            ax4.set_ylabel('Radial Velocity [km/s]', fontsize=20)
            ax4.set_xticks([1e-3,10])
            ax4.set_xticklabels([0,10])
            ax4.set_yticks([-4,-2,0,2,4,6,8,10])
            ax4.set_yticklabels(['$-400$','$-200$','$0$','$200$','$400$','$600$','$800$','$1000$'])
        else:
            if (r==len(rad_bins)-2):
                ax4.tick_params(axis='both', which='both', direction='in', length=8, width=2, pad=5, labelsize=18, \
                    top=True, right=True, left=False, labelleft=False)
            else:
                ax4.tick_params(axis='both', which='both', direction='in', length=8, width=2, pad=5, labelsize=18, \
                    top=True, left=False, right=False, labelleft=False)
            ax4.set_xticks([10])
            ax4.set_xticklabels([int(out_r)])
        if (r==2): ax4.set_xlabel('Galactocentric Radius [kpc]', fontsize=20)
            
        ax5 = fig.add_subplot(inner_tcool[0, r])
        ax5.hist(np.log10(tcool_bin), bins=50, weights=weight_bin, range=(-2,7), density=True, histtype='stepfilled', orientation='horizontal', lw=2, ls='--', ec=(0,0,0,1), fc=(0,0,0,0.2), label='All gas')
        ax5.hist(np.log10(tcool_bin), weights=emissivity_bin, bins=50, range=(-2,7), density=True, histtype='stepfilled', orientation='horizontal', lw=2, ls='-', fc=(219/250., 29/250., 143/250., 0.3), ec=(219/250., 29/250., 143/250., 1), label='O VI emitting gas')
        ax5.set_xscale('log')
        ax5.axis([1e-3,10,-2,7])
        if (r==0): 
            ax5.tick_params(axis='both', which='both', direction='in', length=8, width=2, pad=5, labelsize=18, \
                top=True, right=False, labelleft=True)
            yticks = [-2,0,2,4,6]
            ax5.set_yticks(yticks)
            ax5.set_yticklabels(r"$10^{{{}}}$".format(y) for y in yticks)
            ax5.set_ylabel('Cooling Time [Myr]', fontsize=20)
            ax5.set_xticks([1e-3,10])
            ax5.set_xticklabels([0,10])
        else:
            if (r==len(rad_bins)-2):
                ax5.tick_params(axis='both', which='both', direction='in', length=8, width=2, pad=5, labelsize=18, \
                    top=True, right=True, left=False, labelleft=False)
                ax5.text(1.3, 0.8, '$z=0$', fontsize=20, va='top', ha='left', transform=ax5.transAxes)
                ax5.text(1.3, 0.9, 'All halos', fontsize=20, va='top', ha='left', transform=ax5.transAxes)
            else:
                ax5.tick_params(axis='both', which='both', direction='in', length=8, width=2, pad=5, labelsize=18, \
                    top=True, left=False, right=False, labelleft=False)
            ax5.set_xticks([10])
            ax5.set_xticklabels([int(out_r)])
        if (r==2): ax5.set_xlabel('Galactocentric Radius [kpc]', fontsize=20)
        if (r==4): ax5.legend(loc=2, fontsize=20, frameon=False, bbox_to_anchor=(1.,0.7))

        plt.savefig(prefix + 'Histograms/' + snap + '_gas-hist_OVI_radbins_all-halos' + save_suffix + '.png')

def ionization_equilibrium(ds, refine_box, snap):
    '''Make projections of O VI emission side-by-side with projections of the ratio 
    between O VI ionization equilibration time and cooling time.'''

    zsnap = ds.get_parameter('CosmologyCurrentRedshift')
    print('Making plots')
    proj = yt.ProjectionPlot(ds, ds.x_unit_disk, ('gas','Emission_OVI'), center=ds.halo_center_kpc, data_source=refine_box, width=(100., 'kpc'), depth=(100.,'kpc'), north_vector=ds.z_unit_disk, buff_size=(1024,1024))
    proj_ion = yt.ProjectionPlot(ds, ds.x_unit_disk, ('gas','Equilibration_Time_OVI'), weight_field=('gas','masked_density'), center=ds.halo_center_kpc, data_source=refine_box, width=(100., 'kpc'), depth=(100.,'kpc'), north_vector=ds.z_unit_disk, buff_size=(1024,1024))
    proj_times = yt.ProjectionPlot(ds, ds.x_unit_disk, ('gas','teq_tcool_OVI'), weight_field=('gas','masked_density'), center=ds.halo_center_kpc, data_source=refine_box, width=(100., 'kpc'), depth=(100.,'kpc'), north_vector=ds.z_unit_disk, buff_size=(1024,1024))
    mymap1 = ['#000000']
    mymap2 = cmr.take_cmap_colors('cmr.flamingo', 7, cmap_range=(0.2, 0.8), return_fmt='rgba')
    cmap = np.hstack([mymap1, mymap2])
    mymap = mcolors.LinearSegmentedColormap.from_list('cmap', cmap)
    cmap_ion = cmr.eclipse
    cmap_times = cmr.emerald
    #cmap_ion.set_bad('k')
    #cmap_ion.set_under('k')
    proj.render()
    proj_ion.render()
    proj_times.render()
    frb = proj.frb[('gas','Emission_OVI')]
    frb_ion = proj_ion.frb[('gas','Equilibration_Time_OVI')]
    frb_times = proj_times.frb[('gas','teq_tcool_OVI')]
    fig = plt.figure(figsize=(19,7.5), dpi=250)
    fig.subplots_adjust(left=0.07, bottom=0.01, top=0.94, right=0.97, wspace=0., hspace=0.)
    ax_OVI = fig.add_subplot(1,3,1)
    ax_ion = fig.add_subplot(1,3,2)
    ax_times = fig.add_subplot(1,3,3)
    im_ion = ax_ion.imshow(frb_ion, extent=[-50,50,-50,50], cmap=cmap_ion, origin='lower', norm=mcolors.LogNorm(vmin=1e-3,vmax=5e1))
    im_OVI = ax_OVI.imshow(frb, extent=[-50,50,-50,50], cmap=mymap, norm=mcolors.LogNorm(vmin=1e-22, vmax=1e-16), origin='lower')
    im_times = ax_times.imshow(frb_times, extent=[-50,50,-50,50], cmap=cmap_times, norm=mcolors.LogNorm(vmin=5e-2,vmax=3), origin='lower')
    ax_ion.set_xlabel('x [kpc]', fontsize=18)
    ax_times.set_xlabel('x [kpc]', fontsize=18)
    ax_OVI.set_xlabel('x [kpc]', fontsize=18)
    ax_OVI.set_ylabel('y [kpc]', fontsize=18)
    ax_ion.tick_params(axis='both', which='both', direction='in', length=8, width=2, pad=5, labelsize=16, \
            top=True, right=True, labelleft=False)
    ax_times.tick_params(axis='both', which='both', direction='in', length=8, width=2, pad=5, labelsize=16, \
            top=True, right=True, labelleft=False)
    ax_OVI.tick_params(axis='both', which='both', direction='in', length=8, width=2, pad=5, labelsize=16, \
            top=True, right=True)
    ax_OVI.text(-45, 45, halo_dict[args.halo], fontsize=18, ha='left', va='top', color='white')
    ax_ion.text(45, 45, '$z = %.2f$\n%.2f Gyr' % (zsnap, ds.current_time.in_units('Gyr')), fontsize=18, ha='right', va='top', color='white')
    pos_ion = ax_ion.get_position()
    cax_ion = fig.add_axes([pos_ion.x0, pos_ion.y1, pos_ion.width, 0.03])  # [left, bottom, width, height]
    fig.colorbar(im_ion, cax=cax_ion, orientation='horizontal')
    cax_ion.tick_params(axis='both', which='both', direction='in', length=6, width=2, pad=5, labelsize=16, \
                    top=True, right=True, labelbottom=False, labeltop=True, bottom=False)
    cax_ion.text(0.5, 3.5, 'Projected O VI Equilibration Time [Myr]', fontsize=18, ha='center', va='center', transform=cax_ion.transAxes)
    cax_ion.set_xticks([1e-2,1e-1,1e0,1e1])
    pos_times = ax_times.get_position()
    cax_times = fig.add_axes([pos_times.x0, pos_times.y1, pos_times.width, 0.03])  # [left, bottom, width, height]
    fig.colorbar(im_times, cax=cax_times, orientation='horizontal')
    cax_times.tick_params(axis='both', which='both', direction='in', length=6, width=2, pad=5, labelsize=16, \
                    top=True, right=True, labelbottom=False, labeltop=True, bottom=False)
    cax_times.text(0.5, 3.5, r'Projected $t_\mathrm{ion,eq}/t_\mathrm{cool}$', fontsize=18, ha='center', va='center', transform=cax_times.transAxes)
    #cax_times.set_xticks([1e-2,1e-1,1e0,1e1])
    pos_OVI = ax_OVI.get_position()
    cax_OVI = fig.add_axes([pos_OVI.x0, pos_OVI.y1, pos_OVI.width, 0.03])  # [left, bottom, width, height]
    fig.colorbar(im_OVI, cax=cax_OVI, orientation='horizontal')
    cax_OVI.tick_params(axis='both', which='both', direction='in', length=6, width=2, pad=5, labelsize=16, \
                    top=True, right=True, labelbottom=False, labeltop=True, bottom=False)
    cax_OVI.text(0.5, 3.5, 'O VI Emission [erg s$^{-1}$ cm$^{-2}$ arcsec$^{-2}$]', fontsize=18, ha='center', va='center', transform=cax_OVI.transAxes)
    fig.savefig(prefix + 'Projections/' + snap + '_OVI_emission_teq_tcool_map_edge-on' + save_suffix + '.png')

    sph = ds.sphere(center=ds.halo_center_kpc, radius=(50., 'kpc'))
    phaseplot = yt.PhasePlot(sph, ('gas', 'number_density'), ('gas', 'temperature'), [('gas', 'teq_tcool_OVI')], weight_field=('gas','cell_mass'))
    phaseplot.set_xlim(1e-6,2e1)
    phaseplot.set_ylim(5,2e8)
    phase = phaseplot.profile[('gas','teq_tcool_OVI')].v
    phase[phase==0.0] = np.nan

    cmap_times.set_under('k')
    cmap_times.set_bad('w')

    fig = plt.figure(figsize=(13,7.5), dpi=250)
    ax_proj = fig.add_subplot(1,2,1)
    ax_phase = fig.add_subplot(1,2,2)
    fig.subplots_adjust(left=0.07, bottom=0.08, top=0.88, right=0.98, wspace=0.19, hspace=0.)
    im_proj = ax_proj.imshow(frb_times, extent=[-50,50,-50,50], cmap=cmap_times, norm=mcolors.LogNorm(vmin=5e-2,vmax=3), origin='lower')
    ax_proj.set_xlabel('x [kpc]', fontsize=18)
    ax_proj.set_ylabel('y [kpc]', fontsize=18)
    ax_proj.tick_params(axis='both', which='both', direction='in', length=8, width=2, pad=5, labelsize=16, \
            top=True, right=True)
    ax_proj.text(-45, 45, halo_dict[args.halo], fontsize=18, ha='left', va='top', color='white')
    ax_proj.text(45, 45, '$z = %.2f$' % (zsnap), fontsize=18, ha='right', va='top', color='white')
    pos_proj = ax_proj.get_position()
    cax_proj = fig.add_axes([pos_proj.x0, pos_proj.y1, pos_proj.width, 0.03])  # [left, bottom, width, height]
    fig.colorbar(im_proj, cax=cax_proj, orientation='horizontal')
    cax_proj.tick_params(axis='both', which='both', direction='in', length=6, width=2, pad=5, labelsize=16, \
                    top=True, right=True, labelbottom=False, labeltop=True, bottom=False)
    cax_proj.text(0.5, 3.5, r'Projected $t_\mathrm{ion,eq}/t_\mathrm{cool}$', fontsize=18, ha='center', va='center', transform=cax_proj.transAxes)

    im_phase = ax_phase.imshow(phase.T, cmap=cmap_times, origin='lower', extent=[np.log10(1e-6), np.log10(2e1), np.log10(5), np.log10(2e8)], norm=mcolors.LogNorm(vmin=5e-2,vmax=1e1))
    ax_phase.set_xlabel('Number density [cm$^{-3}$]', fontsize=18)
    ax_phase.set_ylabel('Temperature [K]', fontsize=18)
    ax_phase.tick_params(axis='both', which='both', direction='in', length=8, width=2, pad=5, labelsize=16, \
            top=True, right=True)
    xticks = [-6, -5, -4, -3, -2, -1, 0, 1]
    ax_phase.set_xticks(xticks)
    ax_phase.set_xticklabels([r"$10^{{{}}}$".format(k) for k in xticks])
    yticks = [1, 2, 3, 4, 5, 6, 7, 8]
    ax_phase.set_yticks(yticks)
    ax_phase.set_yticklabels([r"$10^{{{}}}$".format(k) for k in yticks])
    pos_phase = ax_phase.get_position()
    cax_phase = fig.add_axes([pos_phase.x0, pos_phase.y1, pos_phase.width, 0.03])  # [left, bottom, width, height]
    fig.colorbar(im_phase, cax=cax_phase, orientation='horizontal')
    cax_phase.tick_params(axis='both', which='both', direction='in', length=6, width=2, pad=5, labelsize=16, \
                    top=True, right=True, labelbottom=False, labeltop=True, bottom=False)
    cax_phase.text(0.5, 3.5, r'$t_\mathrm{ion,eq}/t_\mathrm{cool}$', fontsize=18, ha='center', va='center', transform=cax_phase.transAxes)
    
    fig.savefig(prefix + 'Projections/' + snap + '_OVI_emission_teq_tcool_map_edge-on_phase' + save_suffix + '.png')

def all_halos_emission_map(halos, outs):
    '''This function plots maps of projected density and O VI emission for all halos together.
    It saves two images, one of Tempest, Squall, and Maelstrom, and the other of Blizzard,
    Hurricane, and Cyclone.'''

    save_dir = '/Users/clochhaas/Documents/Research/FOGGIE/Papers/Aspera Predictions/Figures/'

    if (args.Aspera_limit):
        cmap = cmr.take_cmap_colors('cmr.flamingo', 16, cmap_range=(0.2, 0.8), return_fmt='rgba')
        cmap[0] = 'white'
        cmap[1] = 'white'
        cmap[2] = 'white'
        cmap[3] = 'white'
        cmap[4] = 'white'
        cmap[5] = 'white'
        cmap[6] = 'white'
        cmap[7] = '#000000'
        mymap = mcolors.LinearSegmentedColormap.from_list('cmap', cmap)
    else:
        mymap1 = ['#000000']
        mymap2 = cmr.take_cmap_colors('cmr.flamingo', 7, cmap_range=(0.2, 0.8), return_fmt='rgba')
        cmap = np.hstack([mymap1, mymap2])
        mymap = mcolors.LinearSegmentedColormap.from_list('cmap', cmap)
        cmap_den = cmr.get_sub_cmap('cmr.rainforest', 0., 0.85)

    if (args.Aspera_limit):
        fig = plt.figure(figsize=(12,19), dpi=250)
        fig.subplots_adjust(left=0.03, bottom=0.01, right=0.965, top=0.898, wspace=0., hspace=0.)
        for h in range(len(halos)):
            snap = outs[h]
            snap_name = foggie_dir + 'halo_00' + halos[h] + '/nref11c_nref9f/' + snap + '/' + snap
            trackname = code_path + 'halo_tracks/00' + halos[h] + '/nref11n_selfshield_15/halo_track_200kpc_nref9'
            halo_c_v_name = code_path + 'halo_infos/00' + halos[h] + '/nref11c_nref9f/halo_c_v'
            smooth_AM_name = code_path + 'halo_infos/00' + halos[h] + '/nref11c_nref9f/AM_direction_smoothed'
            ds, refine_box = foggie_load(snap_name, trackfile_name=trackname, do_filter_particles=True, halo_c_v_name=halo_c_v_name, disk_relative=True, correct_bulk_velocity=True, smooth_AM_name=smooth_AM_name)
            ax_OVI = fig.add_subplot(3,2,h+1)

            proj = yt.ProjectionPlot(ds, ds.x_unit_disk, ('gas','Emission_OVI'), center=ds.halo_center_kpc, data_source=refine_box, width=(100., 'kpc'), depth=(100.,'kpc'), north_vector=ds.z_unit_disk, buff_size=(370,370))
            proj.render()
            frb = proj.frb[('gas','Emission_OVI')]
            im_OVI = ax_OVI.imshow(frb, extent=[-50,50,-50,50], cmap=mymap, norm=mcolors.LogNorm(vmin=1e-22, vmax=1e-16), origin='lower')

            ax_OVI.tick_params(length=0, labelleft=False, labelbottom=False)
            ax_OVI.text(-45, 45, halo_dict[halos[h]], fontsize=24, ha='left', va='top')
            if (h%2==0):
                ax_OVI.plot([-40,-20],[-45,-45], color='black', ls='-', lw=2)
                ax_OVI.text(-30, -43, '20 kpc', fontsize=24, ha='center', va='bottom')
            else:
                ax_OVI.plot([20,40],[-45,-45], color='black', ls='-', lw=2)
                ax_OVI.text(30, -43, '20 kpc', fontsize=24, ha='center', va='bottom')
            
            if (h==0):
                pos_OVI = ax_OVI.get_position()
                cax_OVI = fig.add_axes([pos_OVI.x0, pos_OVI.y1, pos_OVI.width*2., 0.03])  # [left, bottom, width, height]
                fig.colorbar(im_OVI, cax=cax_OVI, orientation='horizontal')
                cax_OVI.tick_params(axis='both', which='both', direction='in', length=10, width=3, pad=5, labelsize=20, \
                                top=True, right=True, labelbottom=False, labeltop=True, bottom=False)
                cax_OVI.set_xticks([1e-22,1e-21,1e-20,1e-19,1e-18,1e-17,1e-16])
                cax_OVI.text(0.5, 2.25, 'O VI Emission [erg s$^{-1}$ cm$^{-2}$ arcsec$^{-2}$]', fontsize=22, ha='center', va='center', transform=cax_OVI.transAxes)
            
        fig.savefig(save_dir + 'OVI_emission_maps_edge-on_Aspera-limit' + save_suffix + '.png')

    else:
        for i in range(2):
            fig = plt.figure(figsize=(12,19), dpi=250)
            fig.subplots_adjust(left=0.02, bottom=0.01, right=0.965, top=0.905, wspace=0., hspace=0.)
            for h in range(3*i,3*i+3):
                snap = outs[h]
                print(h, halos[h], outs[h])
                if (halos[h]=='8508-feedback'):
                    snap_name = foggie_dir + 'halo_008508/feedback-10-track/' + snap + '/' + snap
                    trackname = code_path + 'halo_tracks/008508/nref11n_selfshield_15/halo_track_200kpc_nref9'
                    halo_c_v_name = code_path + 'halo_infos/008508/feedback-10-track/halo_c_v'
                    smooth_AM_name = code_path + 'halo_infos/008508/feedback-10-track/AM_direction_smoothed'
                else:
                    snap_name = foggie_dir + 'halo_00' + halos[h] + '/nref11c_nref9f/' + snap + '/' + snap
                    trackname = code_path + 'halo_tracks/00' + halos[h] + '/nref11n_selfshield_15/halo_track_200kpc_nref9'
                    halo_c_v_name = code_path + 'halo_infos/00' + halos[h] + '/nref11c_nref9f/halo_c_v'
                    smooth_AM_name = code_path + 'halo_infos/00' + halos[h] + '/nref11c_nref9f/AM_direction_smoothed'
                ds, refine_box = foggie_load(snap_name, trackfile_name=trackname, do_filter_particles=True, halo_c_v_name=halo_c_v_name, disk_relative=True, correct_bulk_velocity=True, smooth_AM_name=smooth_AM_name)
                if (i==0):
                    ax_den = fig.add_subplot(3,2,h*2+1)
                    ax_OVI = fig.add_subplot(3,2,h*2+2)
                if (i==1):
                    ax_den = fig.add_subplot(3,2,(h-3)*2+1)
                    ax_OVI = fig.add_subplot(3,2,(h-3)*2+2)

                proj = yt.ProjectionPlot(ds, ds.x_unit_disk, ('gas','Emission_OVI'), center=ds.halo_center_kpc, data_source=refine_box, width=(100., 'kpc'), depth=(100.,'kpc'), north_vector=ds.z_unit_disk, buff_size=(370,370))
                proj_den = yt.ProjectionPlot(ds, ds.x_unit_disk, ('gas','density'), center=ds.halo_center_kpc, data_source=refine_box, width=(100., 'kpc'), depth=(100.,'kpc'), north_vector=ds.z_unit_disk, buff_size=(370,370))
                proj.render()
                proj_den.render()
                frb = proj.frb[('gas','Emission_OVI')]
                frb_den = proj_den.frb[('gas','density')]
                im_den = ax_den.imshow(frb_den, extent=[-50,50,-50,50], cmap=cmap_den, origin='lower', norm=mcolors.LogNorm(vmin=5e-6, vmax=2e-1))
                im_OVI = ax_OVI.imshow(frb, extent=[-50,50,-50,50], cmap=mymap, norm=mcolors.LogNorm(vmin=1e-22, vmax=1e-16), origin='lower')

                ax_den.tick_params(length=0, labelleft=False, labelbottom=False)
                ax_OVI.tick_params(length=0, labelleft=False, labelbottom=False)
                ax_den.text(-45, 45, halo_dict[halos[h]], fontsize=24, ha='left', va='top', color='white')
                #ax_OVI.text(-45, 45, 'Tempest\nNo feedback', fontsize=24, ha='left', va='top', color='white')
                ax_OVI.text(45, 45, '$z = 0$', fontsize=24, ha='right', va='top', color='white')
                ax_den.plot([-40,-20],[-45,-45], color='white', ls='-', lw=2)
                ax_den.text(-30, -43, '20 kpc', fontsize=24, ha='center', va='bottom', color='white')
                ax_OVI.plot([20,40],[-45,-45], color='white', ls='-', lw=2)
                ax_OVI.text(30, -43, '20 kpc', fontsize=24, ha='center', va='bottom', color='white')

                if (h==0) or (h==3):
                    pos_den = ax_den.get_position()
                    cax_den = fig.add_axes([pos_den.x0, pos_den.y1, pos_den.width, 0.03])  # [left, bottom, width, height]
                    fig.colorbar(im_den, cax=cax_den, orientation='horizontal')
                    cax_den.tick_params(axis='both', which='both', direction='in', length=10, width=3, pad=5, labelsize=20, color='white', \
                                    top=True, right=True, labelbottom=False, labeltop=True, bottom=False)
                    cax_den.text(0.5, 2.25, 'Projected Gas Density [g cm$^{-2}$]', fontsize=22, ha='center', va='center', transform=cax_den.transAxes)
                    pos_OVI = ax_OVI.get_position()
                    cax_OVI = fig.add_axes([pos_OVI.x0, pos_OVI.y1, pos_OVI.width, 0.03])  # [left, bottom, width, height]
                    fig.colorbar(im_OVI, cax=cax_OVI, orientation='horizontal')
                    cax_OVI.tick_params(axis='both', which='both', direction='in', length=10, width=3, pad=5, labelsize=20, color='white', \
                                    top=True, right=True, labelbottom=False, labeltop=True, bottom=False)
                    cax_OVI.set_xticks([1e-21,1e-20,1e-19,1e-18,1e-17,1e-16])
                    cax_OVI.text(0.5, 2.25, 'O VI Emission [erg s$^{-1}$ cm$^{-2}$ arcsec$^{-2}$]', fontsize=22, ha='center', va='center', transform=cax_OVI.transAxes)
                
            fig.savefig(save_dir + 'OVI_emission_maps_edge-on_' + str(i+1) + save_suffix + '.png')

def all_halos_emission_map_oneplot(halos, outs):
    '''This function plots maps of projected density and O VI emission for all halos together.
    It saves all six halos in one image, as requested by the referee.'''

    save_dir = '/Users/clochhaas/Documents/Research/FOGGIE/Papers/Aspera Predictions/Figures/'

    if (args.Aspera_limit):
        cmap = cmr.take_cmap_colors('cmr.flamingo', 16, cmap_range=(0.2, 0.8), return_fmt='rgba')
        cmap[0] = 'white'
        cmap[1] = 'white'
        cmap[2] = 'white'
        cmap[3] = 'white'
        cmap[4] = 'white'
        cmap[5] = 'white'
        cmap[6] = 'white'
        cmap[7] = '#000000'
        mymap = mcolors.LinearSegmentedColormap.from_list('cmap', cmap)
    else:
        mymap1 = ['#000000']
        mymap2 = cmr.take_cmap_colors('cmr.flamingo', 7, cmap_range=(0.2, 0.8), return_fmt='rgba')
        cmap = np.hstack([mymap1, mymap2])
        mymap = mcolors.LinearSegmentedColormap.from_list('cmap', cmap)
        cmap_den = cmr.get_sub_cmap('cmr.rainforest', 0., 0.85)

    if (args.Aspera_limit):
        fig = plt.figure(figsize=(12,19), dpi=250)
        fig.subplots_adjust(left=0.03, bottom=0.01, right=0.965, top=0.898, wspace=0., hspace=0.)
        for h in range(len(halos)):
            snap = outs[h]
            snap_name = foggie_dir + 'halo_00' + halos[h] + '/nref11c_nref9f/' + snap + '/' + snap
            trackname = code_path + 'halo_tracks/00' + halos[h] + '/nref11n_selfshield_15/halo_track_200kpc_nref9'
            halo_c_v_name = code_path + 'halo_infos/00' + halos[h] + '/nref11c_nref9f/halo_c_v'
            smooth_AM_name = code_path + 'halo_infos/00' + halos[h] + '/nref11c_nref9f/AM_direction_smoothed'
            ds, refine_box = foggie_load(snap_name, trackfile_name=trackname, do_filter_particles=True, halo_c_v_name=halo_c_v_name, disk_relative=True, correct_bulk_velocity=True, smooth_AM_name=smooth_AM_name)
            ax_OVI = fig.add_subplot(3,2,h+1)

            proj = yt.ProjectionPlot(ds, ds.x_unit_disk, ('gas','Emission_OVI'), center=ds.halo_center_kpc, data_source=refine_box, width=(100., 'kpc'), depth=(100.,'kpc'), north_vector=ds.z_unit_disk, buff_size=(370,370))
            proj.render()
            frb = proj.frb[('gas','Emission_OVI')]
            im_OVI = ax_OVI.imshow(frb, extent=[-50,50,-50,50], cmap=mymap, norm=mcolors.LogNorm(vmin=1e-22, vmax=1e-16), origin='lower')

            ax_OVI.tick_params(length=0, labelleft=False, labelbottom=False)
            ax_OVI.text(-45, 45, halo_dict[halos[h]], fontsize=24, ha='left', va='top')
            if (h%2==0):
                ax_OVI.plot([-40,-20],[-45,-45], color='black', ls='-', lw=2)
                ax_OVI.text(-30, -43, '20 kpc', fontsize=24, ha='center', va='bottom')
            else:
                ax_OVI.plot([20,40],[-45,-45], color='black', ls='-', lw=2)
                ax_OVI.text(30, -43, '20 kpc', fontsize=24, ha='center', va='bottom')
            
            if (h==0):
                pos_OVI = ax_OVI.get_position()
                cax_OVI = fig.add_axes([pos_OVI.x0, pos_OVI.y1, pos_OVI.width*2., 0.03])  # [left, bottom, width, height]
                fig.colorbar(im_OVI, cax=cax_OVI, orientation='horizontal')
                cax_OVI.tick_params(axis='both', which='both', direction='in', length=10, width=3, pad=5, labelsize=20, \
                                top=True, right=True, labelbottom=False, labeltop=True, bottom=False)
                cax_OVI.set_xticks([1e-22,1e-21,1e-20,1e-19,1e-18,1e-17,1e-16])
                cax_OVI.text(0.5, 2.25, 'O VI Emission [erg s$^{-1}$ cm$^{-2}$ arcsec$^{-2}$]', fontsize=22, ha='center', va='center', transform=cax_OVI.transAxes)
            
        fig.savefig(save_dir + 'OVI_emission_maps_edge-on_Aspera-limit' + save_suffix + '.png')

    else:
        fig = plt.figure(figsize=(16.75,19), dpi=250)
        outer = mpl.gridspec.GridSpec(2, 3, hspace=0.01, wspace=0.01, left=0.01, top=0.99, bottom=0.01, right=0.85)
        for h in range(len(halos)):
            axes = mpl.gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=outer[h], hspace=0., height_ratios=[1,1])
            ax_den = fig.add_subplot(axes[0, 0])
            ax_OVI = fig.add_subplot(axes[1, 0])
            snap = outs[h]
            print(h, halos[h], outs[h])
            snap_name = foggie_dir + 'halo_00' + halos[h] + '/nref11c_nref9f/' + snap + '/' + snap
            trackname = code_path + 'halo_tracks/00' + halos[h] + '/nref11n_selfshield_15/halo_track_200kpc_nref9'
            halo_c_v_name = code_path + 'halo_infos/00' + halos[h] + '/nref11c_nref9f/halo_c_v'
            smooth_AM_name = code_path + 'halo_infos/00' + halos[h] + '/nref11c_nref9f/AM_direction_smoothed'
            ds, refine_box = foggie_load(snap_name, trackfile_name=trackname, do_filter_particles=True, halo_c_v_name=halo_c_v_name, disk_relative=True, correct_bulk_velocity=True, smooth_AM_name=smooth_AM_name)

            proj = yt.ProjectionPlot(ds, ds.x_unit_disk, ('gas','Emission_OVI'), center=ds.halo_center_kpc, data_source=refine_box, width=(100., 'kpc'), depth=(100.,'kpc'), north_vector=ds.z_unit_disk, buff_size=(370,370))
            proj_den = yt.ProjectionPlot(ds, ds.x_unit_disk, ('gas','density'), center=ds.halo_center_kpc, data_source=refine_box, width=(100., 'kpc'), depth=(100.,'kpc'), north_vector=ds.z_unit_disk, buff_size=(370,370))
            proj.render()
            proj_den.render()
            frb = proj.frb[('gas','Emission_OVI')]
            frb_den = proj_den.frb[('gas','density')]
            im_den = ax_den.imshow(frb_den, extent=[-50,50,-50,50], cmap=cmap_den, origin='lower', norm=mcolors.LogNorm(vmin=5e-6, vmax=2e-1))
            im_OVI = ax_OVI.imshow(frb, extent=[-50,50,-50,50], cmap=mymap, norm=mcolors.LogNorm(vmin=1e-22, vmax=1e-16), origin='lower')

            ax_den.tick_params(length=0, labelleft=False, labelbottom=False)
            ax_OVI.tick_params(length=0, labelleft=False, labelbottom=False)
            ax_den.text(-45, 45, halo_dict[halos[h]], fontsize=24, ha='left', va='top', color='white')
            ax_den.text(45, 45, '$z = 0$', fontsize=24, ha='right', va='top', color='white')
            ax_den.plot([-40,-20],[-45,-45], color='white', ls='-', lw=2)
            ax_den.text(-30, -43, '20 kpc', fontsize=24, ha='center', va='bottom', color='white')
            ax_OVI.plot([-40,-20],[-45,-45], color='white', ls='-', lw=2)
            ax_OVI.text(-30, -43, '20 kpc', fontsize=24, ha='center', va='bottom', color='white')

            if (h==2) or (h==5):
                pos_den = ax_den.get_position()
                cax_den = fig.add_axes([pos_den.x1, pos_den.y0, 0.03, pos_den.height])  # [left, bottom, width, height]
                fig.colorbar(im_den, cax=cax_den, orientation='vertical')
                cax_den.tick_params(axis='both', which='both', direction='in', length=10, width=3, pad=5, labelsize=20, color='white', \
                                top=False, right=True, labelright=True, labelbottom=False, labeltop=False, bottom=False)
                cax_den.text(3.5, 0.5, 'Projected Gas Density\n' + r'[g cm$^{-2}$]', fontsize=22, ha='center', va='center', rotation=90, transform=cax_den.transAxes)
                pos_OVI = ax_OVI.get_position()
                cax_OVI = fig.add_axes([pos_OVI.x1, pos_OVI.y0, 0.03, pos_OVI.height])  # [left, bottom, width, height]
                fig.colorbar(im_OVI, cax=cax_OVI, orientation='vertical')
                cax_OVI.tick_params(axis='both', which='both', direction='in', length=10, width=3, pad=5, labelsize=20, color='white', \
                                top=False, right=True, labelright=True, labelbottom=False, labeltop=False, bottom=False)
                cax_OVI.set_xticks([1e-22,1e-21,1e-20,1e-19,1e-18,1e-17,1e-16])
                cax_OVI.text(3.75, 0.5, 'O VI Emission\n' + r'[erg s$^{-1}$ cm$^{-2}$ arcsec$^{-2}$]', fontsize=22, ha='center', va='center', rotation=90, transform=cax_OVI.transAxes)
            
        fig.savefig(save_dir + 'OVI_emission_maps_edge-on_all-halos' + save_suffix + '.png')


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
    
    ds, refine_box = foggie_load(snap_name, trackfile_name=trackname, do_filter_particles=True, halo_c_v_name=halo_c_v_name, disk_relative=True, correct_bulk_velocity=True, smooth_AM_name=smooth_AM_name)
    zsnap = ds.get_parameter('CosmologyCurrentRedshift')

    if ('emission_map' in args.plot):
        proj = yt.ProjectionPlot(ds, ds.x_unit_disk, ('gas','Emission_OVI'), center=ds.halo_center_kpc, data_source=refine_box, width=(100., 'kpc'), depth=(100.,'kpc'), north_vector=ds.z_unit_disk)#, buff_size=(370,370))
        proj_den = yt.ProjectionPlot(ds, ds.x_unit_disk, ('gas','density'), center=ds.halo_center_kpc, data_source=refine_box, width=(100., 'kpc'), depth=(100.,'kpc'), north_vector=ds.z_unit_disk, buff_size=(370,370))
        if (args.Aspera_limit):
            cmap = cmr.take_cmap_colors('cmr.flamingo', 16, cmap_range=(0.2, 0.8), return_fmt='rgba')
            #cmap2 = cmr.take_cmap_colors('cmr.neutral_r', 6, cmap_range=(0.2, 0.6), return_fmt='rgba')
            cmap[0] = '#fafafa'
            cmap[1] = '#fafafa'
            cmap[2] = '#fafafa'
            cmap[3] = '#fafafa'
            cmap[4] = '#fafafa'
            cmap[5] = '#fafafa'
            cmap[6] = '#fafafa'
            cmap[7] = '#000000'
            mymap = mcolors.LinearSegmentedColormap.from_list('cmap', cmap)
        elif (args.DISCO_limit):
            cmap1 = cmr.take_cmap_colors('cmr.flamingo', 6, cmap_range=(0.4, 0.8), return_fmt='rgba')
            cmap2 = cmr.take_cmap_colors('cmr.neutral_r', 4, cmap_range=(0.2, 0.6), return_fmt='rgba')
            cmap = np.hstack([cmap2, cmap1])
            mymap = mcolors.LinearSegmentedColormap.from_list('cmap', cmap)
        else:
            mymap1 = ['#000000']
            mymap2 = cmr.take_cmap_colors('cmr.flamingo', 7, cmap_range=(0.2, 0.8), return_fmt='rgba')
            cmap = np.hstack([mymap1, mymap2])
            mymap = mcolors.LinearSegmentedColormap.from_list('cmap', cmap)
        cmap_den = cmr.get_sub_cmap('cmr.rainforest', 0., 0.85)
        proj.render()
        proj_den.render()
        frb = proj.frb[('gas','Emission_OVI')]
        frb_den = proj_den.frb[('gas','density')]
        fig = plt.figure(figsize=(13,7.5), dpi=500)
        fig.subplots_adjust(left=0.07, bottom=0.01, top=0.94, right=0.97, wspace=0., hspace=0.)
        ax_den = fig.add_subplot(1,2,1)
        ax_OVI = fig.add_subplot(1,2,2)
        im_den = ax_den.imshow(frb_den, extent=[-50,50,-50,50], cmap=cmap_den, origin='lower', norm=mcolors.LogNorm(vmin=5e-6, vmax=2e-1))
        im_OVI = ax_OVI.imshow(frb, extent=[-50,50,-50,50], cmap=mymap, norm=mcolors.LogNorm(vmin=1e-22, vmax=1e-16), origin='lower')
        ax_den.set_xlabel('x [kpc]', fontsize=18)
        ax_OVI.set_xlabel('x [kpc]', fontsize=18)
        ax_den.set_ylabel('y [kpc]', fontsize=18)
        ax_den.tick_params(axis='both', which='both', direction='in', length=8, width=2, pad=5, labelsize=16, \
                top=True, right=True)
        ax_OVI.tick_params(axis='both', which='both', direction='in', length=8, width=2, pad=5, labelsize=16, \
                top=True, right=True, labelleft=False)
        if (args.Aspera_limit): ax_OVI.text(-45, 45, halo_dict[args.halo], fontsize=18, ha='left', va='top', color='black')
        else: 
            ax_den.text(-45, 45, halo_dict[args.halo], fontsize=18, ha='left', va='top', color='white')
            #ax_OVI.text(45, 45, '$z = %.2f$\n%.2f Gyr' % (zsnap, ds.current_time.in_units('Gyr')), fontsize=18, ha='right', va='top', color='white')
        pos_den = ax_den.get_position()
        cax_den = fig.add_axes([pos_den.x0, pos_den.y1, pos_den.width, 0.03])  # [left, bottom, width, height]
        fig.colorbar(im_den, cax=cax_den, orientation='horizontal')
        cax_den.tick_params(axis='both', which='both', direction='in', length=6, width=2, pad=5, labelsize=16, \
                        top=True, right=True, labelbottom=False, labeltop=True, bottom=False)
        cax_den.text(0.5, 3.5, 'Projected Gas Density [g/cm$^{-2}$]', fontsize=18, ha='center', va='center', transform=cax_den.transAxes)
        pos_OVI = ax_OVI.get_position()
        cax_OVI = fig.add_axes([pos_OVI.x0, pos_OVI.y1, pos_OVI.width, 0.03])  # [left, bottom, width, height]
        fig.colorbar(im_OVI, cax=cax_OVI, orientation='horizontal')
        cax_OVI.tick_params(axis='both', which='both', direction='in', length=6, width=2, pad=5, labelsize=16, \
                        top=True, right=True, labelbottom=False, labeltop=True, bottom=False)
        cax_OVI.set_xticks([1e-21,1e-20,1e-19,1e-18,1e-17,1e-16])
        cax_OVI.text(0.5, 3.5, 'O VI Emission [erg s$^{-1}$ cm$^{-2}$ arcsec$^{-2}$]', fontsize=18, ha='center', va='center', transform=cax_OVI.transAxes)
        fig.savefig(prefix + 'Projections/' + snap + '_OVI_emission_map_edge-on' + save_suffix + '.png')

    if ('sb_profile' in args.plot):
        surface_brightness_profile(ds, refine_box, snap)

    if ('emission_FRB' in args.plot):
        make_FRB(ds, refine_box, snap)

    if ('phase_plot' in args.plot):
        phase_plot(ds, refine_box, snap)

    if ('histograms' in args.plot):
        if ('radbins' in args.plot):
            hists_rad_bins(ds, refine_box, snap)
        else:
            hists_den_temp_Z_rv_tcool(ds, refine_box, snap)

    if ('radial_profiles' in args.plot):
        weighted_radial_profiles(ds, refine_box, snap)

    if ('ionization_equilibrium' in args.plot):
        ionization_equilibrium(ds, refine_box, snap)

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
    halo_c_v_name = code_path + 'halo_infos/00' + args.halo + '/' + args.run + '/halo_c_v'

    #if ('feedback' in args.run) and ('track' in args.run):
        #foggie_dir = '/nobackupnfs1/jtumlins/halo_008508/feedback-track/'
        #run_dir = args.run + '/'
    
    # Set directory for output location, making it if necessary
    prefix = output_dir + 'ions_halo_00' + args.halo + '/' + args.run + '/'
    if not (os.path.exists(prefix)): os.system('mkdir -p ' + prefix)
    table_loc = prefix + 'Tables/'

    print('foggie_dir: ', foggie_dir)
    catalog_dir = code_path + 'halo_infos/00' + args.halo + '/' + args.run + '/'
    halo_c_v_name = catalog_dir + 'halo_c_v'
    smooth_AM_name = catalog_dir + 'AM_direction_smoothed'
    sfr_name = catalog_dir + 'sfr'

    cloudy_path = code_path + "cgm_emission/cloudy_extended_z0_selfshield/TEST_z0_HM12_sh_run%i.dat"
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
    hden_list,T_list,table_HA = make_Cloudy_table(2)
    # 2. Create grids
    hden_pts,T_pts = np.meshgrid(hden_list,T_list)
    pts = np.array((hden_pts.ravel(),T_pts.ravel())).T
    # 3. Set up interpolation function
    hden1, T1, table_OVI_1 = make_Cloudy_table(5)
    hden1, T1, table_OVI_2 = make_Cloudy_table(6)
    sr_OVI_1 = table_OVI_1.T.ravel()
    sr_OVI_2 = table_OVI_2.T.ravel()
    bl_OVI_1 = interpolate.LinearNDInterpolator(pts,sr_OVI_1)
    bl_OVI_2 = interpolate.LinearNDInterpolator(pts,sr_OVI_2)
    # 4. and 5. Define emission field and add it to yt
    if ('emission_FRB' in args.plot):
        yt.add_field(("gas","Emission_OVI"),units=emission_units,function=_Emission_OVI,take_log=True,force_override=True,sampling_type='cell')
    else:
        yt.add_field(("gas","Emission_OVI"),units=emission_units_ALT,function=_Emission_OVI_ALTunits,take_log=True,force_override=True,sampling_type='cell')

    # Set up interpolation functions for ionization and recombination rates for O VI
    ion_rates, rec_rates = combine_rates(hden_list, T_list, 6, code_path)
    ionization_interp = interpolate.LinearNDInterpolator(pts, ion_rates.T.flatten())
    recombination_interp = interpolate.LinearNDInterpolator(pts, rec_rates.T.flatten())
    # Add ionization and recombination rates as fields, and add 
    # equilibration time t_eq = 1 / (ion_rate + rec_rate) as field
    yt.add_field(("gas","Ionization_Rate_OVI"), units='1/s',function=_ionization_OVI,take_log=True,force_override=True,sampling_type='cell')
    yt.add_field(("gas","Recombination_Rate_OVI"), units='1/s',function=_recombination_OVI,take_log=True,force_override=True,sampling_type='cell')
    yt.add_field(("gas","Equilibration_Time_OVI"), units='Myr',function=_equilibration_time_OVI,take_log=True,force_override=True,sampling_type='cell')
    yt.add_field(("gas","teq_tcool_OVI"), units=None,function=_teq_over_tcool_OVI,take_log=True,force_override=True,sampling_type='cell')
    yt.add_field(("gas","masked_density"), units='g*cm**-3',function=_masked_density,take_log=True,force_override=True,sampling_type='cell')

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

    # Build outputs list
    outs = make_output_list(args.output, output_step=args.output_step)

    if (args.plot=='sb_time_hist'):
        surface_brightness_time_histogram(outs)
    if ('sb_time_radius' in args.plot):
        sb_time_radius(outs)
    if ('sb_profile_fdbk' in args.plot) and (not 'time_avg' in args.plot):
        for i in range(len(outs)):
            snap = outs[i]
            sb_profile_nofdbk_compare(snap)
    if ('vs_sfr' in args.plot) or ('vs_mh' in args.plot) or ('vs_den' in args.plot) or ('vs_Z' in args.plot) or ('vs_temp' in args.plot) or ('time_avg' in args.plot) or ('vs_time' in args.plot) or ('all_halos' in args.plot):
        if ('time_avg' in args.plot) or ('all_halos' in args.plot):
            halos = ['8508', '5016', '5036', '4123', '2392', '2878']
            #halos = ['8508']
        else: 
            halos = ['8508', '5016', '5036', '4123', '2392', '2878']
            #halos = ['8508', '5016', '5036', '4123', '2392']
        if ('all_halos_map' in args.plot) or ('histograms_radbins_all_halos' in args.plot):
            outs = ['DD2427', 'DD2520', 'DD2520', 'DD2520', 'DD2520', 'RD0042']
            #outs = ['DD2520', 'DD2520', 'DD2520', 'DD2520', 'DD2520', 'RD0042']
            if ('all_halos_map' in args.plot): all_halos_emission_map_oneplot(halos, outs)
            if ('histograms_radbins_all_halos' in args.plot): hists_rad_bins_all_halos(halos, outs)
        outs = []
        for h in range(len(halos)):
            if (halos[h]=='8508') and ('feedback' not in args.run):
                outs.append(make_output_list('DD0967-DD2427', output_step=args.output_step))
                #outs.append(make_output_list('DD0967-DD1212', output_step=args.output_step)) # z = 1 to 0.726 (~1335 Myr)
                #outs.append(make_output_list('DD2182-DD2427', output_step=args.output_step)) # z = 0.1 to 0 (~1326 Myr)
                #outs.append(make_output_list('DD2427'))
            elif (halos[h]=='2878'):
                outs.append(make_output_list('DD1060-DD2515', output_step=args.output_step))
                #outs.append(make_output_list('DD1060-DD1305', output_step=args.output_step))  # z = 1 to 0.726 (~1335 Myr)
                #outs.append(make_output_list('DD2275-DD2515', output_step=args.output_step)) # z = 0.1 to 0.02 (~996 Myr) This is as far as Cyclone has gone at time of writing
                #outs.append(make_output_list('RD0042'))
                outs[h].append('RD0042')
            else:
                outs.append(make_output_list('DD1060-DD2520', output_step=args.output_step))
                #outs.append(make_output_list('DD1060-DD1305', output_step=args.output_step))  # z = 1 to 0.726 (~1335 Myr)
                #outs.append(make_output_list('DD2275-DD2520', output_step=args.output_step)) # z = 0.1 to 0 (~1326 Myr)
                #outs.append(make_output_list('DD2520'))
        if ('sb_time_hist_all_halos' in args.plot): sb_time_histogram_allhalos(halos, outs)
        if ('sb_vs_sfr' in args.plot): sb_vs_sfr(halos, outs)
        if ('sb_vs_mh' in args.plot): sb_vs_Mh(halos, outs)
        if ('sb_vs_den' in args.plot): sb_vs_den(halos, outs)
        if ('den_vs_time' in args.plot): den_vs_time(halos, outs)
        if ('temp_vs_time' in args.plot): temp_vs_time(halos, outs)
        if ('Z_vs_time' in args.plot): Z_vs_time(halos, outs)
        if ('den_Z_temp_vs_time' in args.plot): den_Z_temp_vs_time(halos, outs)
        if ('sb_vs_Z' in args.plot): sb_vs_Z(halos, outs)
        if ('sb_vs_temp' in args.plot): sb_vs_temp(halos, outs)
        if ('sb_vs_den_temp_Z' in args.plot): sb_vs_den_temp_Z(halos, outs)
        if ('emiss_area_vs_sfr' in args.plot): emiss_area_vs_sfr(halos, outs)
        if ('emiss_area_vs_mh' in args.plot): emiss_area_vs_Mh(halos, outs)
        if ('emiss_area_vs_den' in args.plot): emiss_area_vs_den(halos, outs)
        if ('emiss_area_vs_Z' in args.plot): emiss_area_vs_Z(halos, outs)
        if ('sb_profile_time_avg' in args.plot): sb_profile_time_avg(halos, outs)
        if ('sb_profile_fdbk_time_avg' in args.plot):
            outs = make_output_list('DD2182-DD2427', output_step=args.output_step) # z = 0.1 to 0 (~1326 Myr)
            sb_profile_nofdbk_compare_time_avg(outs)
    if (('emission_map' in args.plot) or ('sb_profile' in args.plot) or ('emission_FRB' in args.plot) or ('phase_plot' in args.plot) or ('histograms' in args.plot) or ('radial_profiles' in args.plot) or ('ionization_equilibrium' in args.plot)) and ('time_avg' not in args.plot) and ('fdbk' not in args.plot) and ('all_halos' not in args.plot):
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

    end = time.perf_counter()
    elapsed = end - start
    duration = timedelta(seconds=elapsed)
    print("All snapshots finished!")
    print(f"Elapsed time: {duration}")