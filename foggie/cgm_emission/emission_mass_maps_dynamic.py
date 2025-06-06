'''
Filename: emission_maps_dynamic.py
Author: Vida
Date created: 1-15-25
Date last modified: 3-14-25

This file contains everything that is needed to make emission maps and FRBs from CLOUDY tables.
All CLOUDY and emission code copy-pasted from Lauren's foggie/cgm_emission/emission_functions.py and Cassi's foggie/cgm_emission/emission_maps.py 

This code is modified Cassi's foggie/cgm_emission/emission_maps.py to dynamically get:
1. yt cut regions: tempreture, density, inflow, outflow, disk, cgm 
Note: if you want to use disk and cgm filters then you need to run clump_finder.py first and then use the output file in the code
2. pixel size for frb maps
3. units for emission: photons: photons s$^{-1}$ cm$^{-2}$ sr$^{-1}$ and erg: erg s$^{-1}$ cm$^{-2}$ arcsec$^{-2}$
4. Field of view size
5. intruments name: Aspera,Juniper, Magpie, HWO, MUSE

It saves:
1. emission maps for edge-on and face-on projections for each ion in the list 'ions' 
2. Saves the emission values in a hdf5 file  
'''

import numpy as np
import yt
import unyt
from yt import YTArray
from yt.data_objects.level_sets.api import * 
import argparse
import os
from astropy.table import Table
from astropy.io import ascii
import multiprocessing as multi
import time


import datetime
from scipy import interpolate
import shutil
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable 
import cmasher as cmr
import matplotlib.colors as mcolors
import h5py
import trident
from yt.units import define_unit, Msun, erg, arcsec, cm, s, sr  

import datashader as dshader
from datashader.utils import export_image
import datashader.transfer_functions as tf
import pandas as pd
import matplotlib as mpl
import numpy as np
from yt.units.yt_array import YTQuantity
from scipy.ndimage import gaussian_filter
from astropy.cosmology import Planck18 as cosmology
from yt.visualization.volume_rendering.off_axis_projection import off_axis_projection


# These imports are FOGGIE-specific files
from foggie.utils.consistency import *
from foggie.utils.get_run_loc_etc import get_run_loc_etc
from foggie.utils.yt_fields import *
from foggie.utils.foggie_load import *
from foggie.utils.analysis_utils import *

# These imports are for disk and clump finders
from foggie.clumps.clump_finder.utils_clump_finder import *
from foggie.clumps.clump_finder.clump_finder_argparser import *
from foggie.clumps.clump_finder.fill_topology import *
from foggie.clumps.clump_finder.clump_load import *
from foggie.clumps.clump_finder.clump_finder import *

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
                        + ' and the default output is RD0042) or specify a range of outputs ' + \
                        '(e.g. "20,25" or "20-25").')
    parser.set_defaults(output='42')

    

    parser.add_argument('--output_step', metavar='output_step', type=int, action='store', \
                        help='If you want to do every Nth output, this specifies N. Default: 1 (every output in specified range)')
    parser.set_defaults(output_step=1)

    parser.add_argument('--system', metavar='system', type=str, action='store', \
                        help='Which system are you on? Default is vida_local')
    parser.set_defaults(system='vida_local')

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
    parser.set_defaults(plot='emission_FRB')

    parser.add_argument('--ions', metavar='ions', type=str, action='store', \
                      help='What ions do you want emission maps or FRBs for? Options are:\n' + \
                        'Lyalpha, Halpha, CII, CIII, CIV, MgII, OVI, SiII, SiIII, SiIV\n' + \
                        "If you want multiple, use a comma-separated list with no spaces like:\n" + \
                        "--ions 'CIII,OVI,SiIII'")

    parser.add_argument('--instrument', metavar='instrument', type=str, action='store', \
                        help='Which instrument criteria do you want to use?use all caps for name of instruments. Default: None. \n' + \
                              'Options: DRAGONFLY, ASPERA, JUNIPER, MAGPIE, HWO, MUSE.')
    parser.set_defaults(instrument=None)
    
    parser.add_argument('--save_suffix', metavar='save_suffix', type=str, action='store', \
                        help='Do you want to append a string onto the names of the saved files? Default is no.')
    parser.set_defaults(save_suffix="")

    parser.add_argument('--file_suffix', metavar='file_suffix', type=str, action='store', \
                        help='If plotting from saved surface brightness files, use this to pass the file name suffix.')
    parser.set_defaults(file_suffix="")

    parser.add_argument('--unit_system', metavar='unit_system', type=str, action='store', \
                        help='What unit system? Default is photons. Options are:\n' + \
                            'default - photons (photons * s**-1 * cm**-3 * sr**-1)\n' + \
                            'erg - erg (ergs * s**-1 * cm**-3 * arcsec**-2)')
    parser.set_defaults(unit_system='photons')

    parser.add_argument('--res_arcsec', metavar='res_arcsec', type=str, action='store', \
                        help='What is the instrument resolution in arcsec? Default is None. If it is None then simulation resolution will be used.')
    parser.set_defaults(res_arcsec=None)

    parser.add_argument('--res_kpc', metavar='res_kpc', type=str, action='store', \
                        help='What spatial reolution you want to make the emission maps for? Default is None. If it is None then simulation resolution will be used.')
    parser.set_defaults(res_kpc=None)

    parser.add_argument('--target_z', metavar='target_z', type=str, action='store', \
                        help='What is the target redshift for your instrument? Default is 0.1')
    parser.set_defaults(target_z=None)

    parser.add_argument('--fov_kpc', metavar='fov_kpc', type=str, action='store', \
                        help='what is the field of view width in kpc (e.g. 100 )? Default is None. If it is None then it will take the refine box width')
    parser.set_defaults(fov_kpc=None)

    parser.add_argument('--fov_arcmin', metavar='fov_arcmin', type=str, action='store', \
                        help='what is the field of view width in arcmin (e.g. 4)? Default is None. If it is None then it will take the refine box width')
    parser.set_defaults(fov_arcmin=None)

    parser.add_argument('--filter_type', metavar='filter_type', type=str, action='store', \
                        help='What filter type? Default is None (Options are: inflow_outflow or disk_cgm )')
    parser.set_defaults(filter_type=None)

    parser.add_argument('--shell_count', metavar='shell_count', type=str, action='store', \
                        help='How many shell you have around disk when you are using disk_cgm filter? defualt is 0')
    parser.set_defaults(shell_count=0)

    parser.add_argument('--shell_cut', metavar='shell_cut', type=str, action='store', \
                        help='What filter type? Default is None (Option is = shell )')
    parser.set_defaults(shell_cut=None)
    
    parser.add_argument('--scale_factor', metavar='scale_factor', type=str, action='store', \
                        help='Do you want to scale the emissivity to observation? How much? The default is 1 i.e. no scaling.')
    parser.set_defaults(scale_factor=1)

    args = parser.parse_args()
    return args
##################################################################################################### 
def add_ion_fields(ds, ions):
    # Ensure ions is a list
    if isinstance(ions, str):
        ions = ions.split(',')  # Split the string into a list if necessary
    
    # Preprocess ions to ensure they are in the format expected by Trident
    formatted_ions = [trident_dict.get(ion, ion) for ion in ions]  
    
    trident.add_ion_fields(ds, formatted_ions)
    return ds

def scale_by_metallicity(values,assumed_Z,wanted_Z):
    # The Cloudy calculations assumed a single metallicity (typically solar).
    # This function scales the emission by the metallicity of the gas itself to
    # account for this discrepancy.
    wanted_ratio = (10.**(wanted_Z))/(10.**(assumed_Z))
    return values*wanted_ratio

def make_Cloudy_table(table_index,cloudy_path):
    # This function takes all of the Cloudy files and compiles them into one table
    # for use in the emission functions
    # table_index is the column in the Cloudy output files that is being read.
    # each table_index value corresponds to a different emission line

    # this is the the range and number of bins for which Cloudy was run
    # i.e. the temperature and hydrogen number densities gridded in the
    # Cloudy run. They must match or the table will be incorrect.

    #make sure these match the values in the Cloudy run
    hden_n_bins, hden_min, hden_max = 17, -6, 2 
    T_n_bins, T_min, T_max = 51, 3, 8 

    hden=np.linspace(hden_min,hden_max,hden_n_bins)
    T=np.linspace(T_min,T_max, T_n_bins)
    table = np.zeros((hden_n_bins,T_n_bins))
    for i in range(hden_n_bins):
            table[i,:]=[float(l.split()[table_index]) for l in open(cloudy_path%(i+1)) if l[0] != "#"]
    return hden,T,table

def make_Cloudy_table_thin(table_index,cloudy_path_thin):

    #make sure these match the values in the Cloudy run
    hden_n_bins, hden_min, hden_max = 17, -5, 2
    T_n_bins, T_min, T_max = 51, 3, 8 

    hden=np.linspace(hden_min,hden_max,hden_n_bins)
    T=np.linspace(T_min,T_max, T_n_bins)
    table = np.zeros((hden_n_bins,T_n_bins))
    for i in range(hden_n_bins):
            table[i,:]=[float(l.split()[table_index]) for l in open(cloudy_path_thin%(i+1)) if l[0] != "#"]
    return hden,T,table

def Emission_LyAlpha(field, data,scale_factor, unit_system='photons'):
    H_N = np.log10(np.array(data["H_nuclei_density"]))
    Temperature = np.log10(np.array(data["Temperature"]))
    dia1 = bl_LA(H_N, Temperature)
    idx = np.isnan(dia1)
    dia1[idx] = -200.
    emission_line = (10**dia1) * ((10.0**H_N)**2.0)
    
    if unit_system == 'photons':
        emission_line = emission_line / (4. * np.pi * 1.63e-11)
        return emission_line * ytEmU
    elif unit_system == 'erg':
        emission_line = emission_line / (4. * np.pi)
        emission_line = emission_line / 4.25e10
        return emission_line * ytEmUALT
    
def Emission_HAlpha(field, data,scale_factor, unit_system='photons'):
    
    H_N = np.log10(np.array(data['H_nuclei_density']))
    Temperature = np.log10(np.array(data['Temperature']))
    dia1 = bl_HA(H_N, Temperature)
    idx = np.isnan(dia1)
    dia1[idx] = -200.
    emission_line = (10.**dia1) * ((10.**H_N)**2.0)
    
    if unit_system == 'photons':
        emission_line = emission_line / (4. * np.pi * 3.03e-12)
        return emission_line * ytEmU
    elif unit_system == 'erg':
        emission_line = emission_line / (4. * np.pi)
        emission_line = emission_line / 4.25e10
        return emission_line * ytEmUALT
    
def Emission_CII_1335(field, data,scale_factor, unit_system='photons'):
    
    H_N = np.log10(np.array(data["H_nuclei_density"]))
    Temperature = np.log10(np.array(data["Temperature"]))
    dia1 = bl_CII_1335(H_N, Temperature)
    idx = np.isnan(dia1)
    dia1[idx] = -200.
    if scale_factor > 1:
        emission_line = scale_factor * ((10.0**dia1) * ((10.0**H_N)**2.0))
    else:
        emission_line = (10.0**dia1) * ((10.0**H_N)**2.0)
    emission_line = scale_by_metallicity(emission_line, 0.0, np.log10(np.array(data['metallicity'])))
    
    if unit_system == 'photons':
        emission_line = emission_line / (4. * np.pi * 2.03e-11)
        return emission_line * ytEmU
    elif unit_system == 'erg':
        emission_line = emission_line / (4. * np.pi)
        emission_line = emission_line / 4.25e10
        return emission_line * ytEmUALT
    
# def Emission_CIII_977(field, data,scale_factor, unit_system='photons'):
    
#     H_N = np.log10(np.array(data["H_nuclei_density"]))
#     Temperature = np.log10(np.array(data["Temperature"]))
#     dia1 = bl_CIII_977(H_N, Temperature)
#     idx = np.isnan(dia1)
#     dia1[idx] = -200.
#     if scale_factor > 1:
#         emission_line = scale_factor * ((10.0**dia1) * ((10.0**H_N)**2.0))
#     else:
#         emission_line = ((10.0**dia1) * ((10.0**H_N)**2.0))
#     emission_line = scale_by_metallicity(emission_line, 0.0, np.log10(np.array(data['metallicity'])))
    
#     if unit_system == 'photons':
#         emission_line = emission_line / (4. * np.pi * 2.03e-11) # the constant value 2.03e-11 is energy per photon for CIII 977
#         return emission_line * ytEmU
#     elif unit_system == 'erg':
#         emission_line = emission_line / (4. * np.pi)
#         emission_line = emission_line / 4.25e10
#         return emission_line * ytEmUALT
      
def Emission_CIII_1910(field, data,scale_factor, unit_system='photons'):
    
    H_N = np.log10(np.array(data["H_nuclei_density"]))
    Temperature = np.log10(np.array(data["Temperature"]))
    dia1 = bl_CIII_1910(H_N, Temperature)
    idx = np.isnan(dia1)
    dia1[idx] = -200.
    if scale_factor > 1:
        emission_line = scale_factor * ((10.0**dia1) * ((10.0**H_N)**2.0))
    else:
        emission_line = ((10.0**dia1) * ((10.0**H_N)**2.0))
    emission_line = scale_by_metallicity(emission_line, 0.0, np.log10(np.array(data['metallicity'])))
    
    if unit_system == 'photons':
        emission_line = emission_line / (4. * np.pi * 2.03e-11)
        return emission_line * ytEmU
    elif unit_system == 'erg':
        emission_line = emission_line / (4. * np.pi)
        emission_line = emission_line / 4.25e10
        return emission_line * ytEmUALT
    
def Emission_CIV_1548(field, data,scale_factor, unit_system='photons'):
    
    H_N = np.log10(np.array(data["H_nuclei_density"]))
    Temperature = np.log10(np.array(data["Temperature"]))
    dia1 = bl_CIV_1548(H_N, Temperature)
    idx = np.isnan(dia1)
    dia1[idx] = -200.
    if scale_factor > 1:
        emission_line = scale_factor * ((10.0**dia1) * ((10.0**H_N)**2.0))
    else:
        emission_line = ((10.0**dia1) * ((10.0**H_N)**2.0))
    emission_line = scale_by_metallicity(emission_line, 0.0, np.log10(np.array(data['metallicity'])))
    
    if unit_system == 'photons':
        emission_line = emission_line / (4. * np.pi * 1.28e-11)
        return emission_line * ytEmU
    elif unit_system == 'erg':
        emission_line = emission_line / (4. * np.pi)
        emission_line = emission_line / 4.25e10
        return emission_line * ytEmUALT

def Emission_OVI(field, data,scale_factor, unit_system='photons'):
    
    H_N = np.log10(np.array(data["H_nuclei_density"]))
    
    Temperature = np.log10(np.array(data["Temperature"]))
    dia1 = bl_OVI_1(H_N, Temperature)
    dia2 = bl_OVI_2(H_N, Temperature)
    idx = np.isnan(dia1)
    dia1[idx] = -200.
    dia2[idx] = -200.
    if scale_factor > 1:
        emission_line = scale_factor * ((10.0**dia1) + (10**dia2)) * ((10.0**H_N)**2.0)
    else:
        emission_line = ((10.0**dia1) + (10**dia2)) * ((10.0**H_N)**2.0)
    emission_line = scale_by_metallicity(emission_line, 0.0, np.log10(np.array(data['metallicity'])))
    
    if unit_system == 'photons':
        emission_line = emission_line / (4. * np.pi * 1.92e-11)
        return emission_line * ytEmU
    elif unit_system == 'erg':
        emission_line = emission_line / (4. * np.pi)
        emission_line = emission_line / 4.25e10 # convert sr to arcsec^2
        return emission_line * ytEmUALT

def Emission_SiIII_1207(field, data,scale_factor, unit_system='photons'):
    
    H_N = np.log10(np.array(data["H_nuclei_density"]))
    Temperature = np.log10(np.array(data["Temperature"]))
    dia1 = bl_SiIII_1207(H_N, Temperature)
    idx = np.isnan(dia1)
    dia1[idx] = -200.
    if scale_factor > 1:
        emission_line = scale_factor * (10.0**dia1) * ((10.0**H_N)**2.0)
    else:
        emission_line = (10.0**dia1) * ((10.0**H_N)**2.0)
    
    emission_line = scale_by_metallicity(emission_line, 0.0, np.log10(np.array(data['metallicity'])))
    
    if unit_system == 'photons':
        emission_line = emission_line / (4. * np.pi * 1.65e-11)
        return emission_line * ytEmU
    elif unit_system == 'erg':
        emission_line = emission_line / (4. * np.pi)
        emission_line = emission_line / 4.25e10
        return emission_line * ytEmUALT

# def Emission_SiII_1814(field, data,scale_factor, unit_system='photons'):
    
#     H_N = np.log10(np.array(data["H_nuclei_density"]))
#     Temperature = np.log10(np.array(data["Temperature"]))
#     dia1 = bl_SiII_1814(H_N, Temperature)
#     idx = np.isnan(dia1)
#     dia1[idx] = -200.
#     if scale_factor > 1:
#         emission_line = scale_factor * (10.0**dia1) * ((10.0**H_N)**2.0)
#     else:
#         emission_line = (10.0**dia1) * ((10.0**H_N)**2.0)
    
#     emission_line = scale_by_metallicity(emission_line, 0.0, np.log10(np.array(data['metallicity'])))
    
#     if unit_system == 'photons':
#         emission_line = emission_line / (4.*np.pi*1.10e-11)
#         return emission_line * ytEmU
#     elif unit_system == 'erg':
#         emission_line = emission_line / (4. * np.pi)
#         emission_line = emission_line / 4.25e10
#         return emission_line * ytEmUALT
#     else:
#         raise ValueError("Invalid unit_system specified. Use 'default' or 'ALT'.")
def Emission_SiII_1260(field, data,scale_factor, unit_system='photons'):
    
    H_N = np.log10(np.array(data["H_nuclei_density"]))
    Temperature = np.log10(np.array(data["Temperature"]))
    dia1 = bl_SiII_1260(H_N, Temperature)
    idx = np.isnan(dia1)
    dia1[idx] = -200.
    if scale_factor > 1:
        emission_line = scale_factor * (10.0**dia1) * ((10.0**H_N)**2.0)
    else:
        emission_line = (10.0**dia1) * ((10.0**H_N)**2.0)
    
    emission_line = scale_by_metallicity(emission_line, 0.0, np.log10(np.array(data['metallicity'])))
    
    if unit_system == 'photons':
        emission_line = emission_line / (4.*np.pi*1.10e-11)
        return emission_line * ytEmU
    elif unit_system == 'erg':
        emission_line = emission_line / (4. * np.pi)
        emission_line = emission_line / 4.25e10
        return emission_line * ytEmUALT
    else:
        raise ValueError("Invalid unit_system specified. Use 'default' or 'ALT'.")

def Emission_SiIV_1394(field, data,scale_factor, unit_system='photons'):
    
    H_N = np.log10(np.array(data["H_nuclei_density"]))
    Temperature = np.log10(np.array(data["Temperature"]))
    dia1 = bl_SiIV_1394(H_N, Temperature)
    idx = np.isnan(dia1)
    dia1[idx] = -200.
    if scale_factor > 1:
        emission_line = scale_factor * (10.0**dia1) * ((10.0**H_N)**2.0)
    else:
        emission_line = (10.0**dia1) * ((10.0**H_N)**2.0)
    
    emission_line = scale_by_metallicity(emission_line, 0.0, np.log10(np.array(data['metallicity'])))
    
    if unit_system == 'photons':
        emission_line = emission_line / (4.*np.pi*1.43e-11)
        return emission_line * ytEmU
    elif unit_system == 'erg':
        emission_line = emission_line / (4. * np.pi)
        emission_line = emission_line / 4.25e10
        return emission_line * ytEmUALT
    else:
        raise ValueError("Invalid unit_system specified. Use 'default' or 'ALT'.")

def Emission_MgII_2796(field, data,scale_factor, unit_system='photons'):
    
    H_N = np.log10(np.array(data["H_nuclei_density"]))
    Temperature = np.log10(np.array(data["Temperature"]))
    dia1 = bl_MgII_2796(H_N, Temperature)
    idx = np.isnan(dia1)
    dia1[idx] = -200.
    if scale_factor > 1:
        emission_line = scale_factor * (10.0**dia1) * ((10.0**H_N)**2.0)
    else:
        emission_line = (10.0**dia1) * ((10.0**H_N)**2.0)
    
    emission_line = scale_by_metallicity(emission_line, 0.0, np.log10(np.array(data['metallicity'])))
    
    if unit_system == 'photons':
        emission_line = emission_line / (4.*np.pi*7.11e-12) 
        return emission_line * ytEmU
    elif unit_system == 'erg':
        emission_line = emission_line / (4. * np.pi)
        emission_line = emission_line / 4.25e10
        return emission_line * ytEmUALT

def filter_ds(box,segmentation_filter='radial_velocity'):
    '''This function filters the yt data object passed in as 'box' into inflow and outflow regions,
    based on metallicity, and returns the box filtered into these regions.'''

    if (segmentation_filter=='metallicity'):
        box_inflow = box.include_below(('gas','metallicity'), 0.01, 'Zsun')
        box_outflow = box.include_above(('gas','metallicity'), 1., 'Zsun')
        box_neither = box.include_above(('gas','metallicity'), 0.01, 'Zsun')
        box_neither = box_neither.include_below(('gas','metallicity'), 1., 'Zsun')
    elif (segmentation_filter=='radial_velocity'):
        box_inflow = box.include_below(('gas','radial_velocity_corrected'), -100., 'km/s')
        box_outflow = box.include_above(('gas','radial_velocity_corrected'), 200., 'km/s')
        box_neither = box.include_above(('gas','radial_velocity_corrected'), -100., 'km/s')
        box_neither = box_neither.include_below(('gas','radial_velocity_corrected'), 200., 'km/s')

    return box_inflow, box_outflow, box_neither
#####################################################################################################  
# Prepare data for generating emission maps
# This section was previously repeated in every emission map function. Now, it is modularized into a series of 
# functions that handle key preprocessing steps. These functions collectively determine the halo name, resolution, 
# pixel size, field of view, filtering criteria, unit label, and HDF5 file structure. 

def get_halo_name_and_resolution(halo_dict, args, refine_box):
    """Retrieve halo name and pixel resolution from refine_box."""
    halo_name = halo_dict[str(args.halo)]
    pix_res = np.min(refine_box['dx'].in_units('kpc'))
    return halo_name, pix_res

def get_arcmin_kpc_scale(args):
    """Compute the arcmin to kpc conversion scale based on redshift."""
    if args.target_z is not None:
        target_redshift = args.target_z
        arcmin_kpc_scale = cosmology.kpc_proper_per_arcmin(target_redshift)
        return YTQuantity(arcmin_kpc_scale, 'kpc/arcmin') 
    return 1  # Default if no instrument resolution

def determine_pixel_size(args, arcmin_kpc_scale, pix_res):
    """Determine the pixel size for the FRB in kpc."""
    if args.res_arcsec is not None:
        res_arcsec = YTQuantity(float(args.res_arcsec), 'arcsec')
        bin_size_kpc = (res_arcsec.in_units('arcmin')) * arcmin_kpc_scale
        bin_size_kpc = bin_size_kpc.in_units('kpc')
    elif args.res_kpc is not None:
        bin_size_kpc = YTQuantity(float(args.res_kpc), 'kpc')
    else:
        bin_size_kpc = pix_res  # Default to simulation resolution
    round_bin_size_kpc = round(bin_size_kpc.to_value(), 2)
    bin_size_cm = bin_size_kpc.in_units('cm')
    return round_bin_size_kpc,bin_size_cm

def determine_fov(args, arcmin_kpc_scale, ds):
    """Determine the field of view (FOV) in kpc."""
    if args.fov_kpc is not None:
        return YTQuantity(float(args.fov_kpc), 'kpc')
    elif args.fov_arcmin is not None:
        fov_kpc = float(args.fov_arcmin) * arcmin_kpc_scale
        return YTQuantity(fov_kpc, 'kpc')
    else:
        return YTQuantity(ds.refine_width, 'kpc') if not hasattr(ds.refine_width, 'in_units') else ds.refine_width.in_units('kpc')

def filter_data(refine_box, filter_type, filter_value, ds, disk_file, shell_count, shell_path,radii=[20,30,50,100]):
    """Apply various data filtering methods."""
    if filter_type == 'inflow_outflow':
        box_inflow, box_outflow, box_neither = filter_ds(refine_box, segmentation_filter='radial_velocity')
        return {'inflow': box_inflow, 'outflow': box_outflow, 'neither': box_neither}

    elif filter_type == 'disk_cgm':
        disk_cut_region = load_clump(ds, disk_file, source_cut=refine_box)
        box_cgm = refine_box - disk_cut_region

        if shell_count == 0:
            return {'cgm': box_cgm}
        
        elif shell_count > 0:
            for i in range(shell_count):
                shell_clump_file = shell_path + f'test_DiskDilationShell_n{i}.h5'
                shell_cut_region = load_clump(ds, shell_clump_file)
                box_cgm = box_cgm - shell_cut_region
            return {'cgm': box_cgm}
        
    elif shell_cut == 'shell':
        # Create a thick shell around the disk
        center=ds.halo_center_kpc
        spheres = []
        for r_index, radius in enumerate(radii):
            print(f"Creating sphere with radius {radius} kpc")
            spheres.append(ds.sphere(center, (radius, 'kpc')))

        sphere1 = spheres[0]
        sphere2 = spheres[1] - spheres[0]
        sphere3 = spheres[2] - spheres[1]
        sphere4 = spheres[3] - spheres[2]

        return {'thick_shell1': sphere1, 'thick_shell2': sphere2, 'thick_shell3': sphere3, 'thick_shell4': sphere4}

    elif filter_type == 'temperature' and filter_value is not None:
        return {'all': refine_box.cut_region([f"(obj['gas', 'temperature'] < {filter_value})"])}
    
    elif filter_type == 'density' and filter_value is not None:
        return {'all': refine_box.cut_region([f"(obj['gas', 'density'] > {filter_value})"])}
    
    # # YOUR FILTER AND CUT REGIONS GO HERE
    # #elif filter_type == 'your_filter':
    # #    data_sources = {'your_region': your_cut_region}

    return {'all': refine_box}  # Default case with no filter

def determine_unit_label(unit_system):
    """Determine the unit label for output."""
    if unit_system == 'photons':
        return '[photons s$^{-1}$ cm$^{-2}$ sr$^{-1}$]'
    elif unit_system == 'erg':
        return '[erg s$^{-1}$ cm$^{-2}$ arcsec$^{-2}$]'
    else:
        raise ValueError("Invalid unit_system specified. Use 'photons' or 'erg'.")

def create_or_open_hdf5_group(prefix, halo_name, save_suffix, ds, width, unit_label, round_bin_size_kpc, ions, args):
    """Create or open an HDF5 group for saving emission maps."""
    save_path = prefix + 'FRBs/'
    os.makedirs(save_path, exist_ok=True)

    # Open HDF5 file in append mode
    f = h5py.File(save_path + halo_name + '_emission_maps' + save_suffix + '.hdf5', 'a')

    # Define the redshift group name
    z_group_name = 'z=%.1f' % ds.get_parameter('CosmologyCurrentRedshift', 1)

    # Check if the group already exists
    if z_group_name in f:
        grp = f[z_group_name]
    else:
        grp = f.create_group(z_group_name)
        grp.attrs.create("image_extent_kpc", width)
        grp.attrs.create("redshift", ds.get_parameter('CosmologyCurrentRedshift'))
        grp.attrs.create("halo_name", halo_name)
        grp.attrs.create("emission_units", unit_label)
        grp.attrs.create("FRB_pixel_size_kpc", round_bin_size_kpc)
        grp.attrs.create("ion_list", ions)

    if args.instrument is not None:
        grp.attrs["instrument_name"] = args.instrument
        grp.attrs["instrument_special_res"] = round_bin_size_kpc

    return f, grp  # Return file and group handle

def process_emission_maps(args, ds, refine_box, halo_dict, filter_type, filter_value, disk_file, shell_count, shell_path, unit_system, prefix, save_suffix, ions):
    """Main function that integrates all steps for computing emission maps."""
    halo_name, pix_res = get_halo_name_and_resolution(halo_dict, args, refine_box)
    arcmin_kpc_scale = get_arcmin_kpc_scale(args)
    round_bin_size_kpc,bin_size_cm = determine_pixel_size(args, arcmin_kpc_scale, pix_res)
    width = determine_fov(args, arcmin_kpc_scale, ds)

    width_value = width.v  # Extract width value
    res = int(width_value / round_bin_size_kpc)  # Calculate FRB resolution
    

    print(f"z={ds.get_parameter('CosmologyCurrentRedshift', 1):.1f}")
    print(f"Simulation resolution (pix_res): {pix_res:.2f} kpc")
    print(f"Field of view (FOV): {width_value:.3f} kpc")
    print(f"FRB pixel size (bin_size_kpc): {round_bin_size_kpc:.2f} kpc")
    print(f"FRB number of bins (res): {res}")

    # Filter data
    data_sources = filter_data(refine_box, filter_type, filter_value, ds, disk_file, shell_count, shell_path,radii=[20,30,50,100])

    # Determine unit label
    unit_label = determine_unit_label(unit_system)

    # Create or open HDF5 file and group
    f, grp = create_or_open_hdf5_group(prefix, halo_name, save_suffix, ds, width, unit_label, round_bin_size_kpc, ions, args)

    return f, grp, data_sources, width_value, res, round_bin_size_kpc, bin_size_cm, unit_label  # Return file, group, and data sources for further processing

#####################################################################################################  

def make_FRB(ds, refine_box, snap, ions, unit_system='photons', filter_type=None, filter_value=None,res_arcsec=None):
    '''This function takes the dataset 'ds' and the refine box region 'refine_box' and
    makes a fixed resolution buffer of surface brightness from edge-on and face-on orientation 
    projections of all ions in the list 'ions'.'''

    
    save_path = prefix + f'FRBs/' 
    os.makedirs(save_path, exist_ok=True)  # Ensure the directory exists
    print('save path:',save_path)
    f, grp, data_sources, width_value, res, round_bin_size_kpc, bin_size_cm, unit_label = process_emission_maps(args, ds, refine_box, halo_dict, filter_type, filter_value, disk_file, shell_count, shell_path, unit_system, prefix, save_suffix, ions)

    # Loop through ions and create projections for each region
    for region, data_source in data_sources.items():
        for ion in ions:
            print(ion)

            # Costum colormap settings with the instrument detection limit for each ion
            if args.instrument is not None and ion in ions:

                # Get the limit range for the total colormap
                zmin, zmax = zlim_dict[ion]
        
                # Get the flux threshold and normalize the threshold position to change cmaps at the detection limit
                threshold = flux_threshold_dict[ion]
                fraction_below = (np.log10(threshold) - np.log10(zmin)) / (np.log10(zmax) - np.log10(zmin))
                fraction_above = 1 - fraction_below

                # Basically change the fractions to percentage, I found that works best for these colormaps
                below_n = int((fraction_below * 100))
                above_n = 100 - below_n 
                
                # Generate colormap sections
                cmap1 = cmr.take_cmap_colors('cmr.flamingo', above_n, cmap_range=(0.5, 0.8), return_fmt='rgba')
                cmap2 = cmr.take_cmap_colors('cmr.neutral_r', below_n, cmap_range=(0.0, 0.6), return_fmt='rgba')
                cmap = np.hstack([cmap2, cmap1])
                mymap = mcolors.LinearSegmentedColormap.from_list('cmap', cmap)
            else:
                # Default colormap
                mymap = cmr.get_sub_cmap('cmr.flamingo', 0.2, 0.9)

            mymap.set_bad(mymap(0))


            #Edge-on projection
            proj_edge = yt.ProjectionPlot(ds, ds.x_unit_disk, ('gas', 'Emission_' + ions_dict[ion]),
                                          center=ds.halo_center_kpc, data_source=data_source,width=(float(width_value), 'kpc'),
                                          north_vector=ds.z_unit_disk, buff_size=[res, res], method = 'integrate', weight_field=None) #(ds.refine_width, 'kpc')
            frb_edge = proj_edge.frb[('gas', 'Emission_' + ions_dict[ion])]
            grp.create_dataset(f"{ion}_emission_edge_{region}", data=frb_edge)
            #Set colormap and save projection plot
            proj_edge.set_cmap('Emission_' + ions_dict[ion], mymap)
            proj_edge.set_zlim('Emission_' + ions_dict[ion], zlim_dict[ion][0], zlim_dict[ion][1])
            proj_edge.set_colorbar_label('Emission_' + ions_dict[ion], label_dict[ion] + ' Emission ' + unit_label)
            proj_edge.set_font_size(24)
            proj_edge.set_xlabel('x (kpc)')
            proj_edge.set_ylabel('y (kpc)')
            proj_edge.annotate_timestamp(corner='upper_left', redshift=True, time=True, draw_inset_box=True)
            proj_edge.annotate_title(f"bin size = {round_bin_size_kpc} kpc")
            proj_edge.save(save_path + f'{snap}_{ion}_emission_map_edge-on_{region}' + save_suffix + '.png')
            

            #Face-on projection
            proj_face = yt.ProjectionPlot(ds, ds.z_unit_disk, ('gas', 'Emission_' + ions_dict[ion]),
                                          center=ds.halo_center_kpc, data_source=data_source,width=(float(width_value), 'kpc'),
                                          north_vector=ds.x_unit_disk, buff_size=[res, res],weight_field=None) #(ds.refine_width, 'kpc')

            frb_face = proj_face.frb[('gas', 'Emission_' + ions_dict[ion])]
            grp.create_dataset(f"{ion}_emission_face_{region}", data=frb_face)

            # Set colormap and save projection plot
            proj_face.set_cmap('Emission_' + ions_dict[ion], mymap)
            proj_face.set_zlim('Emission_' + ions_dict[ion], zlim_dict[ion][0], zlim_dict[ion][1])
            proj_face.set_colorbar_label('Emission_' + ions_dict[ion], label_dict[ion] + ' Emission ' + unit_label)
            proj_face.set_font_size(24)
            proj_face.set_xlabel('x (kpc)')
            proj_face.set_ylabel('y (kpc)')
            proj_face.annotate_timestamp(corner='upper_left', redshift=True, time=True, draw_inset_box=True)
            proj_face.annotate_title(f"bin size = {round_bin_size_kpc} kpc")
            proj_face.save(save_path + f'{snap}_{ion}_emission_map_face-on_{region}' + save_suffix + '.png')

            

    # Close the HDF5 file after saving the datasets
    print('Emission frb saved')
    f.close()
  
def compute_mass_in_emission_pixels(ds, refine_box, snap, ions, unit_system='photons', filter_type=None, filter_value=None,res_arcsec=None):
    """Compute the total ion mass that contributes to each pixel of an emission FRB for edge-on and face-on views."""

    f, grp, data_sources,width_value, res, min_res, bin_size_cm, unit_label = process_emission_maps(args, ds, refine_box, halo_dict, filter_type, filter_value, disk_file, shell_count, shell_path, unit_system, prefix, save_suffix, ions)

    for region, data_source in data_sources.items():
        for ion in ions:
            print(f"Calculating Mass in each pixel for {ion} in {region} region")
            # Extract necessary fields from refine_box
            half_width = (width_value / 2) 
            data_source = data_source.cut_region([
                f"(abs(obj['gas', 'x_disk']) < {half_width})",
                f"(abs(obj['gas', 'y_disk']) < {half_width})",
                f"(abs(obj['gas', 'z_disk']) < {half_width})"
            ])

            x_disk = data_source[('gas', 'x_disk')].in_units('kpc')  # X positions
            y_disk = data_source[('gas', 'y_disk')].in_units('kpc')  # Y positions
            z_disk = data_source[('gas', 'z_disk')].in_units('kpc')  # Z positions
            ion_mass = data_source[('gas', ions_mass_dict[ion])].in_units('Msun')  # Ion mass
            #ion_emission = data_source[('gas', 'Emission_' + ions_dict[ion])]

            
            # Create empty mass FRBs
            mass_frb_edge = np.zeros((res, res), dtype=np.float64) 
            mass_frb_face = np.zeros((res, res), dtype=np.float64) 

            
            # if args.unit_system == 'erg':
            #     emission_frb_edge = np.zeros((res, res)) * (erg / (arcsec**2 * cm**3 * s))
            #     emission_frb_face = np.zeros((res, res)) * (erg / (arcsec**2 * cm**3 * s))
            # elif args.unit_system == 'photons': 
            #     #define_unit("photon", 1.0)  # Define 'photon' as a dimensionless unit
            #     emission_frb_edge = np.zeros((res, res)) * (1.0 * sr**-1 * cm**-3 * s**-1)
            #     emission_frb_face = np.zeros((res, res)) * (1.0 * sr**-1 * cm**-3 * s**-1)


            # Ensure width is evenly divisible by min_dx
            width_min = np.floor((-width_value / 2) / min_res) * min_res  # Snap left edge to nearest cell boundary
            width_max = np.ceil((width_value / 2) / min_res) * min_res  # Snap right edge to nearest cell boundary

            # Compute pixel indices for edge-on and face-on views
            x_bins = np.linspace(width_min, width_max, res+1)  # FRB pixel edges
            y_bins = np.linspace(width_min, width_max, res+1)  # Same binning for both views


            x_indices = np.digitize(x_disk, x_bins) - 1  # Assign x_proj positions to pixels (edge-on)
            y_indices = np.digitize(y_disk, y_bins) - 1  # Assign y_proj positions to pixels (edge-on)
            z_indices = np.digitize(z_disk, y_bins) - 1  # Assign z_proj positions to pixels (face-on)

            # Flip the X-axis indices to correct the left-right mirroring
            face_y_indices = (res - 1) - y_indices


            # Ensure indices are valid
            valid_edge = (z_indices >= 0) & (z_indices < res) & (y_indices >= 0) & (y_indices < res)
            valid_face = (face_y_indices >= 0) & (face_y_indices < res) & (x_indices >= 0) & (x_indices < res)
            print("start calculating mass")

            ion_mass = ion_mass.in_units('Msun')
            
            # EDGE-ON map
            mass_vals_edge = ion_mass[valid_edge].in_units('Msun').v  # strip units
            yi_edge = y_indices[valid_edge]
            zi_edge = z_indices[valid_edge]
            np.add.at(mass_frb_edge, (zi_edge, yi_edge), mass_vals_edge)
            mass_frb_edge = mass_frb_edge * yt.units.Msun


            # FACE-ON map
            mass_vals_face = ion_mass[valid_face].in_units('Msun').v  # strip units
            xi_face = x_indices[valid_face]
            yi_face = face_y_indices[valid_face]
            np.add.at(mass_frb_face, (xi_face, yi_face), mass_vals_face)
            mass_frb_face = mass_frb_face * yt.units.Msun
            #for yi, zi, mass,emission in zip(y_indices[valid_edge], z_indices[valid_edge], ion_mass[valid_edge].in_units('Msun'), ion_emission[valid_edge]):
            # for yi, zi, mass in zip(y_indices[valid_edge], z_indices[valid_edge], ion_mass[valid_edge].in_units('Msun')):
            #     mass_frb_edge[zi, yi] += mass  # Edge-on uses Y (horizontal) & Z (vertical)
            #     #emission_frb_edge[zi, yi] += emission  # Edge-on uses Y (horizontal) & Z (vertical)

            # #for xi, yi, mass,emission in zip(x_indices[valid_face], face_y_indices[valid_face], ion_mass[valid_face].in_units('Msun'), ion_emission[valid_face]):
            # for xi, yi, mass in zip(x_indices[valid_face], face_y_indices[valid_face], ion_mass[valid_face].in_units('Msun')):
            #     mass_frb_face[xi, yi] += mass  # Face-on uses Y (horizontal) & X (vertical)
                #emission_frb_face[xi, yi] += emission  # Face-on uses Y (horizontal) & X (vertical)

            # emission_frb_edge = (emission_frb_edge*bin_size_cm).to(1.0 * sr**-1 * cm**-2 * s**-1)
            # emission_frb_face = (emission_frb_face*bin_size_cm).to(1.0 * sr**-1 * cm**-2 * s**-1)

            
            # Compute total ion mass only within the FRB region
            total_mass_box = ion_mass.sum().in_units('Msun')

            # Compute total mass from FRB maps
            total_mass_frb_edge = mass_frb_edge.sum().in_units('Msun')
            total_mass_frb_face = mass_frb_face.sum().in_units('Msun')

            # Print results
            print(f"Total {ion} mass in FRB-sized region: {total_mass_box:.2e} Msun")
            print(f"Total {ion} mass in mass FRB (edge-on): {total_mass_frb_edge:.2e} Msun")
            print(f"Total {ion} mass in mass FRB (face-on): {total_mass_frb_face:.2e} Msun")

            # Check relative difference
            error_edge = abs(total_mass_frb_edge - total_mass_box) / total_mass_box
            error_face = abs(total_mass_frb_face - total_mass_box) / total_mass_box

            print(f"Relative error (edge-on): {error_edge:.2%}")
            print(f"Relative error (face-on): {error_face:.2%}")

            # Save mass FRBs in the same HDF5 file under the correct group
            dset_edge = grp.create_dataset(f"{ion}_mass_edge_{region}", data=mass_frb_edge.in_units('Msun'))
            dset_face = grp.create_dataset(f"{ion}_mass_face_{region}", data=mass_frb_face.in_units('Msun'))

            #emission_dset_edge = grp.create_dataset(f"{ion}_emission_manualpixel_edge_{region}", data=emission_frb_edge)
            #emission_dset_face = grp.create_dataset(f"{ion}_emission_manualpixel_face_{region}", data=emission_frb_face)

            print(f"Saved mass FRBs for {ion} in HDF5")

           # make a plot of the mass and emission maps
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))

            extent = [-width_value/2, width_value/2, -width_value/2, width_value/2]  # Define extent in kpc

            # Edge-on Mass Map
            # edge_on_msun = mass_frb_edge.in_units('Msun')
            # im1 = axes[0].imshow(edge_on_msun, origin='lower', cmap='inferno', norm=LogNorm(vmin=1e-5, vmax=np.max(edge_on_msun)))
            # axes[0].set_title(f"{ion} Mass Distribution (Edge-on)")
            # axes[0].set_xlabel("Y (kpc)")
            # axes[0].set_ylabel("Z (kpc)")
            # fig.colorbar(im1, ax=axes[0], label="log Mass per pixel (Msun)")

            # # Face-on Mass Map
            # face_on_msun = mass_frb_face.in_units('Msun')
            # im2 = axes[1].imshow(face_on_msun, origin='lower', cmap='inferno', norm=LogNorm(vmin=1e-5, vmax=np.max(face_on_msun)))
            # axes[1].set_title(f"{ion} Mass Distribution (Face-on)")
            # axes[1].set_xlabel("X (kpc)")
            # axes[1].set_ylabel("Z (kpc)")
            # fig.colorbar(im2, ax=axes[1], label="log Mass per pixel (Msun)")

            # plt.tight_layout()
            # plt.show()

            ###Plot emission Distribution Maps** ###
            # fig, axes = plt.subplots(1, 2, figsize=(12, 5))

            # extent = [-width/2, width/2, -width/2, width/2]  # Define extent in kpc

            # # Edge-on Mass Map
            
            # im1 = axes[0].imshow(emission_frb_edge, origin='lower', cmap='inferno', norm=LogNorm(vmin=1e-1, vmax=np.max(emission_frb_edge)))
            # axes[0].set_title(f"{ion} emission Distribution (Edge-on)")
            # axes[0].set_xlabel("Y (kpc)")
            # axes[0].set_ylabel("Z (kpc)")
            # fig.colorbar(im1, ax=axes[0], label="log emission per pixel")

            # # Face-on Mass Map
            
            # im2 = axes[1].imshow(emission_frb_face, origin='lower', cmap='inferno', norm=LogNorm(vmin=1e-1, vmax=np.max(emission_frb_face)))
            # axes[1].set_title(f"{ion} Emission Distribution (Face-on)")
            # axes[1].set_xlabel("X (kpc)")
            # axes[1].set_ylabel("Z (kpc)")
            # fig.colorbar(im2, ax=axes[1], label="log emission per pixel ")

            # plt.tight_layout()
            # plt.show()

    print("Mass FRB calculations and saved.")

def emission_map(ds, refine_box, snap, ions, unit_system='photons', filter_type=None, filter_value=None,res_arcsec=None):
    '''Makes emission maps for each ion in 'ions', oriented both edge-on and face-on.'''
    save_path = prefix + f'FRBs/' 
    os.makedirs(save_path, exist_ok=True)  # Ensure the directory exists
    f, grp, data_sources, width_value, res, round_bin_size_kpc, bin_size_cm, unit_label = process_emission_maps(args, ds, refine_box, halo_dict, filter_type, filter_value, disk_file, shell_count, shell_path, unit_system, prefix, save_suffix, ions)


    # Loop through ions and create projections for each region
    for region, data_source in data_sources.items():
        for i in range(len(ions)):
            ion = ions[i]
            print(ion)

            # Costum colormap settings with the instrument detection limit for each ion
            if args.instrument is not None and ion in ions:

                # Get the limit range for the total colormap
                zmin, zmax = zlim_dict[ion]
        
                # Get the flux threshold and normalize the threshold position to change cmaps at the detection limit
                threshold = flux_threshold_dict[ion]
                fraction_below = (np.log10(threshold) - np.log10(zmin)) / (np.log10(zmax) - np.log10(zmin))
                fraction_above = 1 - fraction_below

                # Basically change the fractions to percentage, I found that works best for these colormaps
                below_n = int((fraction_below * 100))
                above_n = 100 - below_n 
                
                # Generate colormap sections
                cmap1 = cmr.take_cmap_colors('cmr.flamingo', above_n, cmap_range=(0.5, 0.8), return_fmt='rgba')
                cmap2 = cmr.take_cmap_colors('cmr.neutral_r', below_n, cmap_range=(0.0, 0.6), return_fmt='rgba')
                cmap = np.hstack([cmap2, cmap1])
                mymap = mcolors.LinearSegmentedColormap.from_list('cmap', cmap)
            else:
                # Default colormap
                #mymap = cmr.get_sub_cmap('magma', 0.0, 1.0)
                mymap = cmr.get_sub_cmap('cmr.flamingo', 0.2, 1.0)


            mymap.set_bad(mymap(0))
                
            proj_edge = yt.ProjectionPlot(ds, ds.x_unit_disk, ('gas','Emission_' + ions_dict[ion]), center=ds.halo_center_kpc, data_source=data_source, width=(float(width_value), 'kpc'),
                                          buff_size=[res, res], method = 'integrate', weight_field=None, north_vector=ds.z_unit_disk)
            
            proj_edge.set_cmap('Emission_' + ions_dict[ion], mymap)
            proj_edge.set_zlim('Emission_' + ions_dict[ion], zlim_dict[ion][0], zlim_dict[ion][1])
            if args.unit_system == 'photons':
                proj_edge.set_colorbar_label('Emission_' + ions_dict[ion], label_dict[ion] + ' Emission [photons s$^{-1}$ cm$^{-2}$ sr]')
            elif args.unit_system == 'erg':
                proj_edge.set_colorbar_label('Emission_' + ions_dict[ion], label_dict[ion] + ' Emission [erg s$^{-1}$ cm$^{-2}$ arcsec$^{-2}$]')
            proj_edge.set_font_size(20)
            #proj_edge.annotate_timestamp(corner='upper_left', redshift=True, time=True, draw_inset_box=True)
            proj_edge.save(prefix + 'Projections/' + ion + '_emission_map_edge_on' + '_' + snap + save_suffix + '.png')

            proj_face = yt.ProjectionPlot(ds, ds.z_unit_disk, ('gas','Emission_' + ions_dict[ion]), center=ds.halo_center_kpc, data_source=data_source, width=(float(width_value), 'kpc'),
                                          buff_size=[res, res], method = 'integrate', weight_field=None, north_vector=ds.x_unit_disk)
            
            proj_face.set_cmap('Emission_' + ions_dict[ion], mymap)
            proj_face.set_zlim('Emission_' + ions_dict[ion], zlim_dict[ion][0], zlim_dict[ion][1])
            if args.unit_system == 'photons':
                proj_face.set_colorbar_label('Emission_' + ions_dict[ion], label_dict[ion] + ' Emission [photons s$^{-1}$ cm$^{-2}$ sr]')
            elif args.unit_system == 'erg':
                proj_face.set_colorbar_label('Emission_' + ions_dict[ion], label_dict[ion] + ' Emission [erg s$^{-1}$ cm$^{-2}$ arcsec$^{-2}$]')
            proj_face.set_font_size(20)
            #proj_face.annotate_timestamp(corner='upper_left', redshift=True, time=True, draw_inset_box=True)
            proj_face.save(prefix + 'Projections/' + ion + '_emission_map_face_on'+ '_' + snap + save_suffix + '.png')

            
            print('Emission maps saved')

def combined_emission_map(ds, refine_box, snap, ions, unit_system='photons', filter_type=None, filter_value=None, res_arcsec=None):
    '''Makes emission maps for each ion in 'ions', oriented both edge-on and face-on,
    and arranges them side by side with a shared y-axis, no space between maps, and colorbars on top.'''

    f, grp, data_sources, width_value, res, round_bin_size_kpc, bin_size_cm, unit_label = process_emission_maps(args, ds, refine_box, halo_dict, filter_type, filter_value, disk_file, shell_count, shell_path, unit_system, prefix, save_suffix, ions)

    width = [(73.7, 'kpc'), (140.88, 'kpc')]  # Define width as a tuple for rectangular projection 63.7
    res_w = int(width[0][0] / round_bin_size_kpc) 
    res_h = int(width[1][0] / round_bin_size_kpc) 
    print('res w',res_w, 'res h', res_h)

    for region, data_source in data_sources.items():
        fig, axes = plt.subplots(1, len(ions), figsize=(len(ions) * 6, 15), sharey=True, gridspec_kw={'wspace': 0, 'hspace': 0})  # No space between images

        if len(ions) == 1:
            axes = [axes]  # Ensure axes is iterable

        for i, ion in enumerate(ions):
            print(f"Processing {ion}...")

            # Get the colormap limits from zlim_dict
            zmin, zmax = zlim_dict[ion]

            # Custom colormap settings based on detection limit
            if args.instrument is not None and ion in ions:
                threshold = flux_threshold_dict[ion]
                fraction_below = (np.log10(threshold) - np.log10(zmin)) / (np.log10(zmax) - np.log10(zmin))
                fraction_above = 1 - fraction_below

                below_n = int(fraction_below * 100)
                above_n = 100 - below_n 

                cmap1 = cmr.take_cmap_colors('cmr.flamingo', above_n, cmap_range=(0.5, 0.8), return_fmt='rgba')
                cmap2 = cmr.take_cmap_colors('cmr.neutral_r', below_n, cmap_range=(0.0, 0.6), return_fmt='rgba')
                cmap = np.hstack([cmap2, cmap1])
                mymap = mcolors.LinearSegmentedColormap.from_list('cmap', cmap)
            else:
                mymap = cmr.get_sub_cmap('cmr.neutral_r', 0.0, 1.0)#cmr.get_sub_cmap('magma', 0.0, 1.0)

            mymap.set_bad(mymap(0))

            # Create the projection with correct width
            proj = yt.ProjectionPlot(ds, ds.x_unit_disk, ('gas', 'Emission_' + ions_dict[ion]), 
                                     center=ds.halo_center_kpc, data_source=data_source, 
                                     width=width, buff_size=[res_w, res_h], method='integrate', weight_field=None, 
                                     north_vector=ds.z_unit_disk)
            
            proj_face = yt.ProjectionPlot(ds, ds.z_unit_disk, ('gas', 'Emission_' + ions_dict[ion]), 
                                     center=ds.halo_center_kpc, data_source=data_source, 
                                     width=width, buff_size=[res_w, res_h], method='integrate', weight_field=None, 
                                     north_vector=ds.x_unit_disk)

            proj.set_cmap('Emission_' + ions_dict[ion], mymap)
            proj.set_zlim('Emission_' + ions_dict[ion], zmin, zmax)
            proj.set_font_size(20)

            # Extract the image array
            img = proj.frb['gas', 'Emission_' + ions_dict[ion]].to_ndarray()
            extent = [-30, 30, -75, 75]  # X: [-30,30] kpc, Y: [-75,75] kpc

            # Plot in the combined figure using a logarithmic color scale
            ax = axes[i]
            norm = mcolors.LogNorm(vmin=zmin, vmax=zmax)  # Logarithmic normalization
            im = ax.imshow(img, origin='lower', extent=extent, cmap=mymap, norm=norm, aspect = 'auto')

            if i == 0:
                ax.set_ylabel("Y (kpc)", fontsize=16)
                ax.yaxis.set_ticks_position('left')
                ax.tick_params(axis='y', direction='out', labelsize=16, width=2, length=8)
            
            

            ax.set_xlabel("X (kpc)", fontsize=16)
            ax.xaxis.set_major_locator(MaxNLocator(nbins=5, prune='both'))
            ax.tick_params(axis='x', which='both', direction='out', labelsize=16, width=2, length=8)
            
            

            # Add colorbar ABOVE the subplot
            cbar = fig.colorbar(im, ax=ax, orientation='horizontal', pad=0.005, aspect=30, location="top",shrink=0.9)

            # **Set the colorbar label in two rows**
            cbar.set_label(
                label_dict[ion] + ' Emission\n' + unit_label, 
                fontsize=16, labelpad=15
            )

            # **Increase tick size and tick label size**
            cbar.ax.tick_params(labelsize=14, width=1.5, length=8, direction='out')
            cbar.ax.tick_params(labelsize=14, width=1, length=6, direction='out')
  

        plt.subplots_adjust(wspace=0, hspace=0)  # Remove space between subplots
        plt.tight_layout(rect=[0, 0.02, 1, 1])  # Slight adjustment for better layout
        save_path = prefix + 'Projections/'
        os.makedirs(save_path, exist_ok=True)
        #plt.savefig( save_path + snap + '_greycombined_emission_map_' + region + save_suffix + '.png', dpi=300, bbox_inches='tight')
        plt.savefig( save_path + snap + f'_{ion}_emission_map_' + region + save_suffix + '.png', dpi=300, bbox_inches='tight')
        plt.show()
        plt.close()

def make_column_density_FRB(ds, refine_box, snap, ions,scaling = True, scale_factor=100, filter_type=None, filter_value=None, resolution=2):
    '''This function calculates and saves projected column density FRBs '''

    # # Create HDF5 file for saving mass maps
    save_path = prefix + f'FRBs/' 
    os.makedirs(save_path, exist_ok=True)  # Ensure the directory exists
   

    f, grp, data_sources, width_value, res, round_bin_size_kpc, bin_size_cm, unit_label = process_emission_maps(args, ds, refine_box, halo_dict, filter_type, filter_value, disk_file, shell_count, shell_path, unit_system, prefix, save_suffix, ions)
    # Loop through ions and create projections for each region
    for region, data_source in data_sources.items():
        for ion in ions:
            print(f"Processing ion: {ion}")

            #scaled density
            if (scaling == True):
            
                def scaled_numdensity(field, data):
                    return scale_factor * data[('gas', ions_number_density_dict[ion])]

                ds.add_field(
                    name=("gas", f"{ion}_scaled_numdensity"),
                    function=scaled_numdensity,
                    units="cm**-3",  # Same units as the original field
                    sampling_type="cell",
                )

                numdensity_field = ('gas', f"{ion}_scaled_numdensity") 
            else:
                numdensity_field = ('gas', ions_number_density_dict[ion]) 


            # Edge-on projection (surface density)
            proj_edge = yt.ProjectionPlot(ds, ds.x_unit_disk, numdensity_field,
                                          center=ds.halo_center_kpc, data_source=data_source,
                                          width=(100, 'kpc'), north_vector=ds.z_unit_disk,
                                          buff_size=[res, res], weight_field=None)#width=(ds.refine_width, 'kpc')
            frb_edge = proj_edge.frb[numdensity_field]  # Surface density in g/cm^2
            # Save mass FRB, total mass, and positions in HDF5
            edge_name = f"{ion}_numdensity_edge_{region}"
            if edge_name in grp:
                print(f"Skipping {edge_name} (already exists)")
            else:
                dset1 = grp.create_dataset(edge_name, data=frb_edge)

            
            # # Calculate positions
            # # Calculate positions relative to the halo center
            # halo_center_x = ds.halo_center_kpc[0].in_units('kpc')
            # halo_center_y = ds.halo_center_kpc[1].in_units('kpc')

            # # Edge-on projection
            # x_min = -width_value/2
            # x_max = width_value/2
            # y_min = -width_value/2
            # y_max = width_value/2

            # x_edges = np.linspace(x_min, x_max, res + 1)
            # y_edges = np.linspace(y_min, y_max, res + 1)

            # x_positions = 0.5 * (x_edges[:-1] + x_edges[1:]) 
            # y_positions = 0.5 * (y_edges[:-1] + y_edges[1:]) 

            # # Debugging
            # print(f"x_min: {x_min}, x_max: {x_max}, y_min: {y_min}, y_max: {y_max}")
            # print(f"x_positions (kpc): {x_positions}")
            # print(f"y_positions (kpc): {y_positions}")

            # # Save positions
            # dset_x = grp.create_dataset(f"{ion}_x_edge_{region}", data=x_positions)
            # dset_y = grp.create_dataset(f"{ion}_y_edge_{region}", data=y_positions)

            
            

            # Face-on projection (surface density)
            proj_face = yt.ProjectionPlot(ds, ds.z_unit_disk, numdensity_field,
                                          center=ds.halo_center_kpc, data_source=data_source,
                                          width=(100, 'kpc'), north_vector=ds.x_unit_disk,
                                          buff_size=[res, res], weight_field=None)#(width=(ds.refine_width, 'kpc')
            frb_face = proj_face.frb[numdensity_field]  # Surface density in g/cm^2
        
            face_name = f"{ion}_numdensity_face_{region}"
            if face_name in grp:
                print(f"Skipping {face_name} (already exists)")
            else:
                dset2 = grp.create_dataset(face_name, data=frb_face)


            # Save relative positions
            #dset_x = grp.create_dataset(f"{ion}_x_face_{region}", data=x_positions)
            #dset_y = grp.create_dataset(f"{ion}_y_face_{region}", data=y_positions)

            proj_face.set_cmap(numdensity_field, h1_color_map)
            proj_face.set_zlim(numdensity_field , h1_proj_min, h1_proj_max)
            proj_face.set_colorbar_label(numdensity_field, label_dict[ion] + ' Column Density ')
            proj_face.set_font_size(24)
            proj_face.set_xlabel('x (kpc)')
            proj_face.set_ylabel('y (kpc)')
            print('save path', save_path)
            proj_face.save(save_path + f'{snap}_{ion}_column_density_faceon_{region}' + save_suffix + '.png')


    # Close the HDF5 file after saving the datasets
    print('Number density FRBs finished')
    f.close()

def make_emissivity_weighted_velocity_FRB(ds, refine_box, snap, ions, unit_system='photons',
                                          filter_type=None, filter_value=None, res_arcsec=None):
    print("Starting emissivity-weighted velocity FRBs")
    
    save_path = prefix + f'FRBs/' 
    os.makedirs(save_path, exist_ok=True)
    
    f, grp, data_sources,width_value, res, min_res, bin_size_cm, unit_label = process_emission_maps(args, ds, refine_box, halo_dict, filter_type, filter_value, disk_file, shell_count, shell_path, unit_system, prefix, save_suffix, ions)
    
    for region, data_source in data_sources.items():
        for ion in ions:
            print(f"Making emissivity-weighted velocity FRBs for {ion} in {region} region")
            mymap = cmr.get_sub_cmap('cmr.viola', 0.2, 0.8)
            mymap.set_bad(mymap(0))

            emission_field = ('gas', f'Emission_{ions_dict[ion]}')
            def _vx_disk_cgs(field, data):
                return data[('gas', 'vx_disk')].to('cm/s')

            ds.add_field(('gas', 'vx_disk_cgs'), function=_vx_disk_cgs, units='cm/s',sampling_type='cell')
            ### EDGE-ON (LOS = x, proj plane = y-z) ###
            proj_edge = yt.ProjectionPlot(ds, ds.x_unit_disk, ('gas', 'vx_disk_cgs'),
                                          center=ds.halo_center_kpc, data_source=data_source,
                                          width=(float(width_value), 'kpc'),
                                          north_vector=ds.z_unit_disk, buff_size=[res, res],
                                          method='integrate', weight_field=emission_field)
            frb_edge = proj_edge.frb[('gas', 'vx_disk_cgs')].to('km/s')
            print('projected vx unit:',frb_edge.units)
            print(frb_edge)
            print(f"{ion} vx_disk_cgs stats (edge-on): min = {frb_edge.v.min():.2f}, max = {frb_edge.v.max():.2f}, mean = {frb_edge.v.mean():.2f} [km/s]")

            grp.create_dataset(f"{ion}_vlos_emissweighted_edge_{region}", data=frb_edge)
            #Set colormap and save projection plot
            # proj_edge.set_cmap('vx_disk_cgs', mymap)
            # #proj_edge.set_zlim('vx_disk_cgs', -1000, 1000)
            # proj_edge.set_colorbar_label('vx_disk_cgs', label_dict[ion] + ' V los ' + 'km/s')
            # proj_edge.set_font_size(24)
            # proj_edge.set_xlabel('x (kpc)')
            # proj_edge.set_ylabel('y (kpc)')
            # proj_edge.annotate_timestamp(corner='upper_left', redshift=True, time=True, draw_inset_box=True)
            # print('save path', save_path)
            # proj_edge.save(save_path + f'{snap}_{ion}_vlos_map_edge-on_{region}' + save_suffix + '.png')
            

            fig, ax = plt.subplots(figsize=(8, 6))
            im = ax.imshow(frb_edge.v, origin='lower', extent=[-width_value/2, width_value/2]*2,
                        vmin=-200, vmax=200, cmap=mymap)

            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label('Emissivity-weighted $v_{los}$ [km/s]')

            ax.set_xlabel('Y (kpc)')
            ax.set_ylabel('Z (kpc)')
            ax.set_title(f'{ion} $v_{{los}}$ (edge-on)')

            plt.tight_layout()
            plt.savefig(save_path + f'{snap}_{ion}_vlos_map_edge-on_{region}' + save_suffix + '.png', dpi=300)
            plt.close()


            ### FACE-ON (LOS = z, proj plane = x-y) ###
            def _vz_disk_cgs(field, data):
                return data[('gas', 'vz_disk')].to('cm/s')

            ds.add_field(('gas', 'vz_disk_cgs'), function=_vz_disk_cgs, units='cm/s',sampling_type='cell')
            
            proj_face = yt.ProjectionPlot(ds, ds.z_unit_disk, ('gas', 'vz_disk_cgs'),
                                          center=ds.halo_center_kpc, data_source=data_source,
                                          width=(float(width_value), 'kpc'),
                                          north_vector=ds.x_unit_disk, buff_size=[res, res],
                                          method='integrate', weight_field=emission_field)
            frb_face = proj_face.frb[('gas', 'vz_disk_cgs')].to('km/s')
            grp.create_dataset(f"{ion}_vlos_emissweighted_face_{region}", data=frb_face)
            print('projected vx unit:',frb_face.units)
            print(frb_face)
            print(f"{ion} vz_disk_cgs stats (edge-on): min = {frb_face.v.min():.2f}, max = {frb_face.v.max():.2f}, mean = {frb_face.v.mean():.2f} [km/s]")
            #Set colormap and save projection plot
            # proj_face.set_cmap('vz_disk_cgs', mymap)
            # #proj_face.set_zlim('vz_disk_cgs', -1000, 1000)
            # proj_face.set_colorbar_label('vz_disk_cgs', label_dict[ion] + ' V los ' + 'km/s')
            # proj_face.set_font_size(24)
            # proj_face.set_xlabel('x (kpc)')
            # proj_face.set_ylabel('y (kpc)')
            # proj_face.annotate_timestamp(corner='upper_left', redshift=True, time=True, draw_inset_box=True)
            # proj_face.save(save_path + f'{snap}_{ion}_vlos_map_face-on_{region}' + save_suffix + '.png')

            fig, ax = plt.subplots(figsize=(8, 6))
            im = ax.imshow(frb_face.v, origin='lower', extent=[-width_value/2, width_value/2]*2,
                        vmin=-200, vmax=200, cmap=mymap)

            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label('Emissivity-weighted $v_{los}$ [km/s]')

            ax.set_xlabel('X (kpc)')
            ax.set_ylabel('Y (kpc)')
            ax.set_title(f'{ion} $v_{{los}}$ (face-on)')

            plt.tight_layout()
            plt.savefig(save_path + f'{snap}_{ion}_vlos_map_face-on_{region}' + save_suffix + '.png', dpi=300)
            plt.close()


            # === Coordinate positions ===
            x_min = -width_value / 2
            x_max = width_value / 2
            y_min = -width_value / 2
            y_max = width_value / 2
            z_min = -width_value / 2
            z_max = width_value / 2

            x_edges = np.linspace(x_min, x_max, res + 1)
            y_edges = np.linspace(y_min, y_max, res + 1)
            z_edges = np.linspace(z_min, z_max, res + 1)
            x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
            y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])
            z_centers = 0.5 * (z_edges[:-1] + z_edges[1:])

            # Save pixel center coordinates
            grp.create_dataset(f"{ion}_x_edge_{region}", data=y_centers)  # Horizontal axis = Y in edge-on
            grp.create_dataset(f"{ion}_y_edge_{region}", data=z_centers)  # Vertical axis = Z in edge-on
            grp.create_dataset(f"{ion}_x_face_{region}", data=x_centers)  # Horizontal axis = X in face-on
            grp.create_dataset(f"{ion}_y_face_{region}", data=y_centers)  # Vertical axis = Y in face-on

            print(f"Saved vlos FRBs + pixel coordinates for {ion}")
    
    f.close()
    print("Finished emissivity-weighted velocity FRBs.")

def make_hi_emission_grid(ds, refine_box, snap, ions,unit_system='photons', filter_type=None, filter_value=None,res_arcsec=None):
    
    region = 'all'
    # Ion list (excluding HI)
    emission_ions = ['OVI', 'CIV', 'SiIV', 'CIII', 'SiIII', 'MgII', 'SiII']
    f, grp, data_sources, width_value, res, round_bin_size_kpc, bin_size_cm, unit_label = process_emission_maps(args, ds, refine_box, halo_dict, filter_type, filter_value, disk_file, shell_count, shell_path, unit_system, prefix, save_suffix, ions)

    # Loop through ions and create projections for each region
    for region, data_source in data_sources.items():

        # Use global vmin and vmax for emission maps
        global_vmin = 1e-2
        global_vmax = 1e5

        # Open the HDF5 file
        h5_filename = prefix + f'FRBs/{halo_dict[str(args.halo)]}_emission_maps{save_suffix}.hdf5'
        with h5py.File(h5_filename, 'r') as f:
            group = f[f'z={ds.get_parameter("CosmologyCurrentRedshift"):.1f}']

            for orientation in ['face', 'edge']:
                fig, axes = plt.subplots(4, 2, figsize=(10, 18))
                plt.subplots_adjust(wspace=0.01, hspace=0.01)

                # Plot HI at (0, 0)
                hi_data = group[f"HI_numdensity_{orientation}_{region}"][:]
                ax_hi = axes[0, 0]
                im_hi = ax_hi.imshow(hi_data, origin='lower', cmap=h1_color_map, norm=LogNorm(vmin=h1_proj_min, vmax=h1_proj_max))
                ax_hi.text(0.05, 0.90, 'HI', transform=ax_hi.transAxes, fontsize=16, color='white', weight='bold', va='top', ha='left')
                ax_hi.set_xticks([])
                ax_hi.set_yticks([])

                # Fill in the rest with emission ions
                idx = 0
                last_im = None
                for i in range(4):
                    for j in range(2):
                        if i == 0 and j == 0:
                            continue
                        if idx >= len(emission_ions):
                            continue

                        ion = emission_ions[idx]
                        data = group[f"{ion}_emission_{orientation}_{region}"][:]

                        ax = axes[i, j]
                        im = ax.imshow(data, origin='lower', cmap=cmr.flamingo, norm=LogNorm(vmin=global_vmin, vmax=global_vmax))
                        ax.text(0.05, 0.90, ion, transform=ax.transAxes, fontsize=16, color='white', weight='bold', va='top', ha='left')
                        ax.set_xticks([])
                        ax.set_yticks([])

                        last_im = im  # Save the last im for colorbar use
                        idx += 1

                # Add HI colorbar above [0, 0]
                cbar_ax_hi = fig.add_axes([0.17, 0.89, 0.3, 0.015])
                cbar_hi = fig.colorbar(im_hi, cax=cbar_ax_hi, orientation='horizontal',shrink=0.9)
                cbar_hi.ax.tick_params(labelsize=12, direction='out')
                cbar_hi.ax.xaxis.set_label_position('top')
                cbar_hi.ax.xaxis.set_ticks_position('top')
                cbar_hi.set_label('HI Column Density [cm$^{-2}$]', fontsize=12)

                # Add global emission colorbar above [0, 1]
                cbar_ax_em = fig.add_axes([0.56, 0.89, 0.3, 0.015])
                cbar_em = fig.colorbar(last_im, cax=cbar_ax_em, orientation='horizontal',shrink=0.9)
                cbar_em.ax.tick_params(labelsize=12, direction='out')
                cbar_em.ax.xaxis.set_label_position('top')
                cbar_em.ax.xaxis.set_ticks_position('top')
                cbar_em.set_label('Emission [photons s$^{-1}$ cm$^{-2}$ sr$^{-1}$]', fontsize=12)

                output_path = prefix + 'Projections/'
                os.makedirs(output_path, exist_ok=True)
                fig.savefig(os.path.join(output_path, f"{snap}_HI_plus_emission_grid_{orientation}_{region}{save_suffix}.png"), dpi=300, bbox_inches='tight')
                plt.show()
                plt.close()

    print("Saved HI+Emission grid plots for both face-on and edge-on.")

def cgm_make_hi_emission_grid(ds, refine_box, snap, ions,unit_system='photons', filter_type=None, filter_value=None,res_arcsec=None):

    
    emission_ions = ['OVI', 'CIV', 'SiIV', 'CIII', 'SiIII', 'MgII', 'SiII']

    f, grp, data_sources, width_value, res, round_bin_size_kpc, bin_size_cm, unit_label = process_emission_maps(
        args, ds, refine_box, halo_dict, filter_type, filter_value, disk_file, shell_count,
        shell_path, unit_system, prefix, save_suffix, ions
    )

    for region, data_source in data_sources.items():

        global_vmin = 1e-2
        global_vmax = 1e5

        h5_filename = prefix + f'FRBs/{halo_dict[str(args.halo)]}_emission_maps{save_suffix}.hdf5'
        with h5py.File(h5_filename, 'r') as f:
            group = f[f'z={ds.get_parameter("CosmologyCurrentRedshift"):.1f}']

            for orientation in ['face', 'edge']:
                fig, axes = plt.subplots(4, 2, figsize=(10, 18))
                plt.subplots_adjust(wspace=0.01, hspace=0.01)

                hi_data = group[f"HI_numdensity_{orientation}_{region}"][:]
                ax_hi = axes[0, 0]
                im_hi = ax_hi.imshow(hi_data, origin='lower', cmap=h1_color_map,
                                     norm=LogNorm(vmin=h1_proj_min, vmax=h1_proj_max))
                ax_hi.text(0.05, 0.90, 'HI', transform=ax_hi.transAxes, fontsize=16, color='white', weight='bold', va='top', ha='left')
                ax_hi.set_xticks([])
                ax_hi.set_yticks([])

                if region == 'cgm':
                    try:
                        disk_cut = load_clump(ds, disk_file, source_cut=refine_box)
                        proj_disk = yt.ProjectionPlot(ds, ds.x_unit_disk if orientation == 'edge' else ds.z_unit_disk,
                                                      ('gas', 'H_p0_number_density'),
                                                      center=ds.halo_center_kpc, data_source=disk_cut,
                                                      width=(100, 'kpc'),
                                                      north_vector=ds.z_unit_disk if orientation == 'edge' else ds.x_unit_disk,
                                                      buff_size=[res, res], weight_field=None)
                        disk_frb = proj_disk.frb[('gas', 'H_p0_number_density')].v
                        ax_hi.contour(disk_frb, levels=4, colors='white', linewidths=0.8)
                    except Exception as e:
                        print(f"Could not overlay disk contours on HI: {e}")

                idx = 0
                last_im = None
                for i in range(4):
                    for j in range(2):
                        if i == 0 and j == 0:
                            continue
                        if idx >= len(emission_ions):
                            continue

                        ion = emission_ions[idx]
                        data = group[f"{ion}_emission_{orientation}_{region}"][:]
                        ax = axes[i, j]
                        im = ax.imshow(data, origin='lower', cmap=cmr.flamingo,
                                       norm=LogNorm(vmin=global_vmin, vmax=global_vmax))
                        ax.text(0.05, 0.90, ion, transform=ax.transAxes, fontsize=16, color='white', weight='bold', va='top', ha='left')
                        ax.set_xticks([])
                        ax.set_yticks([])

                        if region == 'cgm':
                            try:
                                proj_disk = yt.ProjectionPlot(ds, ds.x_unit_disk if orientation == 'edge' else ds.z_unit_disk,
                                                              ('gas', 'H_p0_number_density'),
                                                              center=ds.halo_center_kpc, data_source=disk_cut,
                                                              width=(100, 'kpc'),
                                                              north_vector=ds.z_unit_disk if orientation == 'edge' else ds.x_unit_disk,
                                                              buff_size=[res, res], weight_field=None)
                                disk_frb = proj_disk.frb[('gas', 'H_p0_number_density')].v
                                ax.contour(disk_frb, levels=5, colors='white', linewidths=0.8)
                            except Exception as e:
                                print(f"Could not overlay disk contours on {ion}: {e}")

                        last_im = im
                        idx += 1

                cbar_ax_hi = fig.add_axes([0.17, 0.89, 0.3, 0.015])
                cbar_hi = fig.colorbar(im_hi, cax=cbar_ax_hi, orientation='horizontal', shrink=0.9)
                cbar_hi.ax.tick_params(labelsize=12, direction='out')
                cbar_hi.ax.xaxis.set_label_position('top')
                cbar_hi.ax.xaxis.set_ticks_position('top')
                cbar_hi.set_label('HI Column Density [cm$^{-2}$]', fontsize=12)

                cbar_ax_em = fig.add_axes([0.56, 0.89, 0.3, 0.015])
                cbar_em = fig.colorbar(last_im, cax=cbar_ax_em, orientation='horizontal', shrink=0.9)
                cbar_em.ax.tick_params(labelsize=12, direction='out')
                cbar_em.ax.xaxis.set_label_position('top')
                cbar_em.ax.xaxis.set_ticks_position('top')
                cbar_em.set_label('Emission [photons s$^{-1}$ cm$^{-2}$ sr$^{-1}$]', fontsize=12)

                output_path = prefix + 'Projections/'
                os.makedirs(output_path, exist_ok=True)
                fig.savefig(os.path.join(output_path, f"{snap}_HI_plus_emission_grid_{orientation}_{region}{save_suffix}.png"), dpi=300, bbox_inches='tight')
                plt.show()
                plt.close()

    print("Saved HI+Emission grid plots for both face-on and edge-on.")



######################################################################################################
def load_and_calculate(snap, ions,scale_factor=None, unit_system='photons', filter_type=None, filter_value=None, res_arcsec=None):
    start_time = time.time()
    '''Loads the simulation snapshot and makes the requested plots, with optional filtering.'''

    # Load simulation output
    snap_name = foggie_dir + 'halo_00' + args.halo + '/' + args.run + '/' + snap + '/' + snap
    ds, refine_box = foggie_load(snap_name, trackname, do_filter_particles=True, halo_c_v_name=halo_c_v_name, disk_relative=True, correct_bulk_velocity=True)#, smooth_AM_name=smooth_AM_name)
    zsnap = ds.get_parameter('CosmologyCurrentRedshift')
    #finding disk
    
    if filter_type == 'disk_cgm':
        if not os.path.exists(disk_file):
            print('Disk file not found. Running clump finder to generate disk')
            disk_args = get_default_args()
            disk_args.refinement_level = 11
            disk_args.identify_disk = True
            disk_args.snapshot = 'RD00' + args.output
            disk_args.output = output_dir + '/FOGGIE' + '/'+ 'RD00' + args.output + '/'+ '/Disk/H1'
            disk_args.n_dilation_iterations = 10
            disk_args.n_cells_per_dilation = 1
            disk_args.clumping_field = 'H_p0_number_density'
            disk_args.system = args.system

            clump_finder(disk_args, ds, refine_box)
        else:
            print(f'Disk file already exists at {disk_file}, skipping clump finder.')

    add_ion_fields(ds,ions)
    
    # Generate emission maps based on the plot type
    if ('emission_map' in args.plot):
        if ('vbins' not in args.plot):
            emission_map(ds, refine_box, snap, ions, unit_system=unit_system, filter_type=filter_type, filter_value=filter_value,res_arcsec=res_arcsec)
            #combined_emission_map(ds, refine_box, snap, ions, unit_system=unit_system, filter_type=filter_type, filter_value=filter_value, res_arcsec=res_arcsec)
            

    if ('emission_FRB' in args.plot):
        make_FRB(ds, refine_box, snap, ions, unit_system=unit_system, filter_type=filter_type, filter_value=filter_value, res_arcsec=res_arcsec)
        compute_mass_in_emission_pixels(ds, refine_box, snap, ions, unit_system=unit_system, filter_type=filter_type, filter_value=filter_value, res_arcsec=res_arcsec)
        #make_column_density_FRB(ds, refine_box, snap, ions, scale_factor=scale_factor, filter_type=filter_type, filter_value=filter_value)  
        make_emissivity_weighted_velocity_FRB(ds, refine_box, snap, ions, unit_system=unit_system,filter_type=filter_type, filter_value=filter_value, res_arcsec=res_arcsec)
        #make_hi_emission_grid(ds, refine_box, snap, ions,unit_system=unit_system, filter_type=filter_type, filter_value=filter_value,res_arcsec=res_arcsec)
        #cgm_make_hi_emission_grid(ds, refine_box, snap, ions,unit_system=unit_system, filter_type=filter_type, filter_value=filter_value,res_arcsec=res_arcsec)

    end_time = time.time()
    elapsed = end_time - start_time
    print(f" Total script runtime: {elapsed//60:.0f} min {elapsed%60:.2f} sec")      
                
if __name__ == "__main__":

    args = parse_args()
    print('Halo:',args.halo)

    foggie_dir, output_dir, run_dir, code_path, trackname, haloname, spectra_dir, infofile = get_run_loc_etc(args)
    halo_c_v_name = code_path + 'halo_infos/00' + args.halo + '/' + args.run + '/halo_c_v'


    if ('feedback' in args.run) and ('track' in args.run):
        foggie_dir = '/nobackup/jtumlins/halo_008508/feedback-track/'
        run_dir = args.run + '/'

    #set the clump/disk file directory that you saved the Disk files that you produced by running clump_finder.py
    disk_file = output_dir + '/FOGGIE' + '/'+ 'RD00' + args.output + '/'+ '/Disk/H1_Disk.h5'
    shell_path = output_dir + '/FOGGIE' + '/'+ 'RD00' + args.output + '/'+ '/Disk/'
    
    # Set directory for output location, making it if necessary
    if args.fov_kpc is not None:
        box_name = args.fov_kpc
    elif args.fov_arcmin is not None:
        box_name = args.fov_arcmin
    else:
        box_name = 'refine_box'

    if args.res_arcsec is not None:
        resolution = args.res_arcsec
    elif args.res_kpc is not None:
        resolution = args.res_kpc
    else:
        resolution = '0.27'

    if args.instrument is not None:
        prefix = output_dir + '/' + args.instrument + '/' + 'box_' + box_name + '/' + resolution + '/' 
        if args.filter_type is not None:
            prefix = prefix + '/' + args.filter_type + '/'
    else:
        prefix = output_dir + '/FOGGIE' + '/'+ 'RD00' + args.output + '/'+ 'box_' + box_name + '/' + 'with_disk' + '/' + resolution + '/'
        if args.filter_type is not None:
            if args.filter_type == 'disk_cgm':
                prefix = output_dir + '/FOGGIE' + '/'+ 'RD00' + args.output + '/'+ 'box_' + box_name + '/' + 'without_disk' + '/' + resolution + '/' + args.filter_type + '/'
            else:
                prefix = output_dir + '/FOGGIE' + '/'+ 'RD00' + args.output + '/'+ 'box_' + box_name + '/' + args.filter_type + '/' + resolution + '/' 

    if not (os.path.exists(prefix)): os.system('mkdir -p ' + prefix)
    table_loc = prefix + 'Tables/'

    print('foggie_dir: ', foggie_dir)
    catalog_dir = code_path + 'halo_infos/00' + args.halo + '/' + args.run + '/'
    halo_c_v_name = catalog_dir + 'halo_c_v'
    #smooth_AM_name = catalog_dir + 'AM_direction_smoothed'

    # right now using the test tables that vida made 
    #cloudy_path = "/Users/vidasaeedzadeh/Documents/02-Projects/02-FOGGIE/Cloudy-runs/outputs/test-z0/TEST_z0_HM12_sh_run%i.dat"
    cloudy_path = code_path + "cgm_emission/cloudy_extended_z0_selfshield/TEST_z0_HM12_sh_run%i.dat"
    cloudy_path_thin = code_path + "cgm_emission/cloudy_z0_HM05/bertone_run%i.dat"

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
    # Function to register emission fields with unit options
    def register_emission_field_with_unit(field_name, function, emission_units, unit_system,scale_factor):
        yt.add_field(
            ('gas', field_name),
            units=emission_units if unit_system == 'photons' else emission_units_ALT,
            function=lambda field, data: function(field, data,scale_factor=scale_factor, unit_system=unit_system),
            take_log=True,
            force_override=True,
            sampling_type='cell',
        )

    ####################################
    unit_system = args.unit_system
    scale_factor = float(args.scale_factor)
    instrument_name = args.instrument

    ####################################

    # H-Alpha
    hden_pts, T_pts, table_HA = make_Cloudy_table(2,cloudy_path)
    hden_pts, T_pts = np.meshgrid(hden_pts, T_pts)
    pts = np.array((hden_pts.ravel(), T_pts.ravel())).T
    
    sr_HA = table_HA.T.ravel()
    bl_HA = interpolate.LinearNDInterpolator(pts, sr_HA)
    register_emission_field_with_unit('Emission_HAlpha', Emission_HAlpha, emission_units, unit_system,scale_factor)
    
    ############################
    # Ly-Alpha
    hden_pts, T_pts, table_LA = make_Cloudy_table(1,cloudy_path)
    sr_LA = table_LA.T.ravel()
    bl_LA = interpolate.LinearNDInterpolator(pts, sr_LA)
    register_emission_field_with_unit('Emission_LyAlpha', Emission_LyAlpha, emission_units, unit_system,scale_factor)
    ############################
    # CII 1335
    hden_pts, T_pts, table_CII_1335 = make_Cloudy_table(10,cloudy_path)
    sr_CII_1335 = table_CII_1335.T.ravel()
    bl_CII_1335 = interpolate.LinearNDInterpolator(pts, sr_CII_1335)
    register_emission_field_with_unit('Emission_CII_1335', Emission_CII_1335, emission_units, unit_system,scale_factor)
    
    ############################
    # CIII 977
    hden_pts, T_pts, table_CIII_977 = make_Cloudy_table(7,cloudy_path)
    sr_CIII_977 = table_CIII_977.T.ravel()
    bl_CIII_977 = interpolate.LinearNDInterpolator(pts, sr_CIII_977)
    register_emission_field_with_unit('Emission_CIII_977', Emission_CIII_977, emission_units, unit_system,scale_factor)

    ############################
    # CIII 1910
    hden_pts, T_pts, table_CIII_1910 = make_Cloudy_table(9,cloudy_path)
    sr_CIII_1910 = table_CIII_1910.T.ravel()
    bl_CIII_1910 = interpolate.LinearNDInterpolator(pts, sr_CIII_1910)
    register_emission_field_with_unit('Emission_CIII_1910', Emission_CIII_1910, emission_units, unit_system,scale_factor)

    ############################
    # CIV 1548
    hden_pts, T_pts, table_CIV_1548 = make_Cloudy_table(3,cloudy_path)
    sr_CIV_1548  = table_CIV_1548.T.ravel()
    bl_CIV_1548  = interpolate.LinearNDInterpolator(pts, sr_CIV_1548 )
    register_emission_field_with_unit('Emission_CIV_1548', Emission_CIV_1548, emission_units, unit_system,scale_factor)
    
    ############################
    # O VI (1032 and 1038 combined)
    hden_pts, T_pts, table_OVI_1 = make_Cloudy_table(5,cloudy_path)
    hden_pts, T_pts, table_OVI_2 = make_Cloudy_table(6,cloudy_path)
    sr_OVI_1 = table_OVI_1.T.ravel()
    sr_OVI_2 = table_OVI_2.T.ravel()
    bl_OVI_1 = interpolate.LinearNDInterpolator(pts, sr_OVI_1)
    bl_OVI_2 = interpolate.LinearNDInterpolator(pts, sr_OVI_2)
    register_emission_field_with_unit('Emission_OVI', Emission_OVI, emission_units, unit_system,scale_factor)
    ############################
    # # SiII 1814
    
    # hden_pts, T_pts, table_SiII_1814 = make_Cloudy_table(11,cloudy_path)
    # sr_SiII_1814 = table_SiII_1814.T.ravel()
    # bl_SiII_1814 = interpolate.LinearNDInterpolator(pts, sr_SiII_1814)
    # register_emission_field_with_unit('Emission_SiII_1814', Emission_SiII_1814, emission_units, unit_system,scale_factor)
    # ############################
    ############################
    # SiII 1260
    
    hden_pts, T_pts, table_SiII_1260 = make_Cloudy_table(12,cloudy_path)
    sr_SiII_1260 = table_SiII_1260.T.ravel()
    bl_SiII_1260 = interpolate.LinearNDInterpolator(pts, sr_SiII_1260)
    register_emission_field_with_unit('Emission_SiII_1260', Emission_SiII_1260, emission_units, unit_system,scale_factor)
    ############################
    # SiIII 1207
    
    hden_pts, T_pts, table_SiIII_1207 = make_Cloudy_table(13,cloudy_path)
    sr_SiIII_1207 = table_SiIII_1207.T.ravel()
    bl_SiIII_1207 = interpolate.LinearNDInterpolator(pts, sr_SiIII_1207)
    register_emission_field_with_unit('Emission_SiIII_1207', Emission_SiIII_1207, emission_units, unit_system,scale_factor)
    ############################
    # SiIV 1394
    
    hden_pts, T_pts, table_SiIV_1394 = make_Cloudy_table(15,cloudy_path)
    sr_SiIV_1394 = table_SiIV_1394.T.ravel()
    bl_SiIV_1394 = interpolate.LinearNDInterpolator(pts, sr_SiIV_1394)
    register_emission_field_with_unit('Emission_SiIV_1394', Emission_SiIV_1394, emission_units, unit_system,scale_factor)
    ############################
    # MgII 2796
    
    hden_pts, T_pts, table_MgII_2796 = make_Cloudy_table(17,cloudy_path)
    sr_MgII_2796 = table_MgII_2796.T.ravel()
    bl_MgII_2796 = interpolate.LinearNDInterpolator(pts, sr_MgII_2796)
    register_emission_field_with_unit('Emission_MgII_2796', Emission_MgII_2796, emission_units, unit_system,scale_factor)
    ############################

    ions_dict = {'Lyalpha':'LyAlpha', 'HI':'HAlpha', 'CII': 'CII_1335','CIII':'CIII_1910', 
                 'CIV':'CIV_1548','OVI':'OVI','SiII':'SiII_1260','SiIII':'SiIII_1207','SiIV':'SiIV_1394','MgII':'MgII_2796'}
    
    label_dict = {'Lyalpha':r'Ly-$\alpha$', 'HI':r'H$\alpha$', 'CII':'C II','CIII':'C III',
                'CIV':'C IV','OVI':'O VI','SiII':'Si II','SiIII':'Si III','SiIV':'Si IV','MgII':'Mg II'}
    
    trident_dict = { 'HI':'H I', 'CII':'C II','CIII':'C III',
                'CIV':'C IV','OVI':'O VI','SiII':'Si II','SiIII':'Si III','SiIV':'Si IV','MgII':'Mg II'}
    
    ions_number_density_dict = {'Lyalpha':'LyAlpha', 'HI':'H_p0_number_density', 'CII':'C_p1_number_density', 'CIII':'C_p2_number_density',
                                'CIV':'C_p3_number_density','OVI':'O_p5_number_density','SiII':'Si_p1_number_density','SiIII':'Si_p2_number_density',
                                 'SiIV':'Si_p3_number_density','MgII':'Mg_p1_number_density'}
    ions_mass_dict = {'Lyalpha':'LyAlpha', 'HI':'H_p0_mass', 'CII':'C_p1_mass', 'CIII':'C_p2_mass',
                      'CIV':'C_p3_mass','OVI':'O_p5_mass','SiII':'Si_p1_mass','SiIII':'Si_p2_mass','SiIV':'Si_p3_mass','MgII':'Mg_p1_mass'}

    
    #Set the colormap range to look best for each ion and each boxsize
    if args.fov_kpc is not None or args.fov_arcmin is not None:
        if unit_system  == 'photons':
            if (args.fov_kpc == None) and (args.halo == '2392'):
                zlim_dict = {'Lyalpha':[1e-1,1e7], 'HI':[1e-1,1e6], 'CII':[1e-7,1e4], 'CIII':[1e-1,1e5],
                        'CIV':[1e-2,1e4], 'OVI':[1e-2,1e4],'SiII':[1e-6,1e5],'SiIII':[1e-6,1e5],'SiIV':[1e-6,1e5],'MgII':[1e-6,1e5]}
            if int(args.scale_factor) > 1:
                zlim_dict = {'Lyalpha':[1e-1,1e7], 'HI':[1e-1,1e6], 'CII':[1e-3,1e4], 'CIII':[1e0,1e5],
                        'CIV':[1e0,1e6], 'OVI':[1e0,1e5],'SiII':[1e-1,1e5],'SiIII':[1e-2,1e5],'SiIV':[1e-2,1e5],'MgII':[1e-1,1e6]}

            else:
                zlim_dict = {'Lyalpha':[1e-1,1e7], 'HI':[1e-1,1e6], 'CII':[1e-3,1e4], 'CIII':[1e-1,1e4],
                        'CIV':[1e-1,1e5], 'OVI':[1e0,1e5],'SiII':[1e-1,1e5],'SiIII':[1e-2,1e5],'SiIV':[1e-2,1e5],'MgII':[1e-1,1e6]}
        elif unit_system == 'erg':
            zlim_dict = {'Lyalpha':[1e-22,1e-16], 'HI':[1e-22,1e-16], 'CII':[1e-23,1e-16], 'CIII':[1e-23,1e-16],
                        'CIV':[1e-23,1e-16], 'OVI':[1e-22,1e-17],'SiII':[1e-23,1e-16],'SiIII':[1e-23,1e-16],'SiIV':[1e-23,1e-16],'MgII':[1e-23,1e-16]}
    else:
        if unit_system  == 'photons':
            if (args.fov_kpc == None) and (args.halo == '2392'):
                zlim_dict = {'Lyalpha':[1e-1,1e7], 'HI':[1e-1,1e6], 'CII':[1e-7,1e4], 'CIII':[1e-1,1e5],
                        'CIV':[1e-2,1e4], 'OVI':[1e-2,1e4],'SiII':[1e-6,1e5],'SiIII':[1e-6,1e5],'SiIV':[1e-6,1e5],'MgII':[1e-6,1e5]}
            else:
                zlim_dict = {'Lyalpha':[1e-1,1e7], 'HI':[1e-1,1e6], 'CII':[1e-7,1e4], 'CIII':[1e-4,1e3],
                        'CIV':[1e-2,1e4], 'OVI':[1e-2,1e4],'SiII':[1e-5,1e5],'SiIII':[1e-5,1e5],'SiIV':[1e-5,1e5],'MgII':[1e-5,1e5]} 
        elif unit_system == 'erg':
            zlim_dict = {'Lyalpha':[1e-22,1e-16], 'HI':[1e-22,1e-16], 'CII':[1e-23,1e-16], 'CIII':[1e-23,1e-16],
                        'CIV':[1e-23,1e-16], 'OVI':[1e-22,1e-17],'SiII':[1e-23,1e-16],'SiIII':[1e-23,1e-16],'SiIV':[1e-23,1e-16],'MgII':[1e-23,1e-16]}
        
    # Set the detection limits for each ion for each intrument
    if instrument_name == 'JUNIPER':
        if args.unit_system == 'photons':
            flux_threshold_dict = {'CII':1588.24, 'CIII':9000.00,'CIV':3857.14, 'OVI':1800.00} #photons/s/cm^2/sr
        elif args.unit_system == 'erg':
            flux_threshold_dict = {'OVI': 4e-19} 
        
    elif instrument_name == 'ASPERA':
        flux_threshold_dict = {'OVI': 3e-19} #ergs/s/cm^2/arcsec^2

    elif instrument_name == 'MAGPIE':
        if args.unit_system == 'photons':
            flux_threshold_dict = {'CIII': 675,'CIV': 650,'OVI': 270, 'MgII':675} #photons/s/cm^2/sr
        elif args.unit_system == 'erg':
            flux_threshold_dict = {'OVI': 3e-19} #ergs/s/cm^2/arcsec^2

    elif instrument_name == 'DISCO':
        if args.unit_system == 'erg':
            flux_threshold_dict = {'CII': 2e-19,'CIII': 2e-19,'CIV': 2e-19,'OVI': 2e-19, 'SiII':2e-19, 'SiIII':2e-19, 'SiIV':2e-19} #ergs/s/cm^2/arcsec^2
        elif args.unit_system == 'photons':
            flux_threshold_dict = {'CII': 300,'CIII': 300,'CIV': 300,'OVI': 300, 'SiII':300, 'SiIII':300, 'SiIV':300}
        
    elif instrument_name == 'MUSE':
        if args.unit_system == 'erg':
            flux_threshold_dict = {'OVI': 3e-19} #ergs/s/cm^2/arcsec^2

    elif instrument_name == 'HWO':
        if args.unit_system == 'photons':
            flux_threshold_dict = {'OVI': 200} #photons/s/cm^2/sr
        elif args.unit_system == 'erg':
            flux_threshold_dict = {'OVI': 1.5e-20} #ergs/s/cm^2/arcsec^2  
    elif instrument_name == 'flux100':
        if args.unit_system == 'photons':
            flux_value = 100
            flux_threshold_dict = {'HI':flux_value,'CII':flux_value,'CIII': flux_value,'CIV': flux_value,'OVI': flux_value,'MgII':flux_value,'SiII':flux_value,'SiIII':flux_value,'SiIV':flux_value,} #photons/s/cm^2/sr
        elif args.unit_system == 'erg':
            flux_threshold_dict = {'OVI': 3e-19} #ergs/s/cm^2/arcsec^2
    elif instrument_name == 'flux500':
        if args.unit_system == 'photons':
            flux_value = 500
            flux_threshold_dict = flux_threshold_dict = {'HI':flux_value,'CII':flux_value,'CIII': flux_value,'CIV': flux_value,'OVI': flux_value,'MgII':flux_value,'SiII':flux_value,'SiIII':flux_value,'SiIV':flux_value,} #photons/s/cm^2/sr
        elif args.unit_system == 'erg':
            flux_threshold_dict = {'OVI': 3e-19} #ergs/s/cm^2/arcsec^2
    elif instrument_name == 'flux1000':
        if args.unit_system == 'photons':
            flux_value = 1000
            flux_threshold_dict = flux_threshold_dict = {'HI':flux_value,'CII':flux_value,'CIII': flux_value,'CIV': flux_value,'OVI': flux_value,'MgII':flux_value,'SiII':flux_value,'SiIII':flux_value,'SiIV':flux_value,} #photons/s/cm^2/sr
        elif args.unit_system == 'erg':
            flux_threshold_dict = {'OVI': 3e-19} #ergs/s/cm^2/arcsec^2
    elif instrument_name == 'flux2000':
        if args.unit_system == 'photons':
            flux_value = 2000
            flux_threshold_dict = flux_threshold_dict = {'HI':flux_value,'CII':flux_value,'CIII': flux_value,'CIV': flux_value,'OVI': flux_value,'MgII':flux_value,'SiII':flux_value,'SiIII':flux_value,'SiIV':flux_value,} #photons/s/cm^2/sr
        elif args.unit_system == 'erg':
            flux_threshold_dict = {'OVI': 3e-19} #ergs/s/cm^2/arcsec^2
    else:
        flux_threshold_dict = {}

    #YOUR INSTRUMENT HERE
    #elif instrument_name == 'YOUR INSTRUMENT NAME':
    #    if args.unit_system == 'photons':
    #        flux_threshold_dict = {'CII': YOUR_VALUE, 'CIII': YOUR_VALUE,'CIV': YOUR_VALUE, 'OVI': YOUR_VALUE} #photons/s/cm^2/sr
    #    elif args.unit_system == 'erg':        
    #        flux_threshold_dict = {'CII': YOUR_VALUE, 'CIII': YOUR_VALUE,'CIV': YOUR_VALUE, 'OVI': YOUR_VALUE} #ergs/s/cm^2/arcsec^2
      
        
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

    #Build unit_system
    unit_system = args.unit_system

    #Build filter_type
    filter_type = args.filter_type

    shell_cut = args.shell_cut

    res_arcsec = args.res_arcsec
    
    shell_count = int(args.shell_count)

    # Build outputs list
    outs = make_output_list('RD00'+args.output, output_step=args.output_step)

    # Code for running in parallel
    target_dir = 'ions'
    if (args.nproc==1):
        for snap in outs:
            load_and_calculate(snap, ions,scale_factor=scale_factor, unit_system=unit_system, filter_type=filter_type, filter_value=None, res_arcsec=res_arcsec)
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
    print(f"All snapshots finished and saved in {prefix}!")







