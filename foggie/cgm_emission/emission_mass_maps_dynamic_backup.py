'''
Filename: emission_maps_dynamic.py
Author: Vida
Date created: 1-15-25
Date last modified: 3-11-25

This file contains everything that is needed to make emission maps and FRBs from CLOUDY tables.
All CLOUDY and emission code copy-pasted from Lauren's foggie/cgm_emission/emission_functions.py and Cassi's foggie/cgm_emission/emission_maps.py 

This code is modified Cassi's foggie/cgm_emission/emission_maps.py to dynamically get:
1. filters: tempreture, density, inflow, outflow, disk, cgm
2. resolution for frb maps
3. units for emission: default: photons s$^{-1}$ cm$^{-2}$ sr$^{-1}$ and ALT: erg s$^{-1}$ cm$^{-2}$ arcsec$^{-2}$
4. Field of view size
5. intruments name 

It saves:
1. emission maps for edge-on and face-on projections for each ion in the list 'ions' (if the instrument name argument is given than it makes mass using limit of that intrument)
2. Saves the emission values in a hdf5 file
3. It also save mass of each ion generating from the surface density in the hdf5 file  for later analysis (in emission_analysis.py file) of how much mass is in each pixel with a spcific emission 
4. density projections  
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
from matplotlib.colors import LogNorm


import datetime
from scipy import interpolate
import shutil
import matplotlib.pyplot as plt
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
from foggie.clumps.clump_finder.clump_finder import clump_finder

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
                        '(e.g. "RD0020,RD0025" or "DD1340-DD2029").')
    parser.set_defaults(output='RD0042')

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

    parser.add_argument('--Dragonfly_limit', dest='Dragonfly_limit', action='store_true', \
                        help='Do you want to calculate and plot only above the Dragonfly limit? Default is no. This only matters for Halpha.')
    parser.set_defaults(Dragonfly_limit=False)

    parser.add_argument('--Aspera_limit', dest='Aspera_limit', action='store_true', \
                        help='Do you want to calculate and plot only above the Aspera limit? Default is no. This only matters for O VI.')
    parser.set_defaults(Aspera_limit=False)

    parser.add_argument('--Juniper_limit', dest='Juniper_limit', action='store_true', \
                        help='Do you want to calculate and plot only above the Juniper_limit limit? Default is no. This only matters for CII, CIII, CIV, and O VI')
    parser.set_defaults(Juniper_limit=False)

    parser.add_argument('--Magpie_limit', dest='Magpie_limit', action='store_true', \
                        help='Do you want to calculate and plot only above the Magpie_limit limit? Default is no. This only matters for MgII, CIII, CIV, and O VI')
    parser.set_defaults(Magpie_limit=False)

    parser.add_argument('--Muse_limit', dest='Muse_limit', action='store_true', \
                        help='Do you want to calculate and plot only above the Muse_limit limit? Default is no. This only matters for O VI ??')
    parser.set_defaults(Muse_limit=False)

    parser.add_argument('--HWO_limit', dest='HWO_limit', action='store_true', \
                        help='Do you want to calculate and plot only above the HWO_limit limit? Default is no. This only matters for O VI ??')
    parser.set_defaults(HWO_limit=False)
    
    
    parser.add_argument('--save_suffix', metavar='save_suffix', type=str, action='store', \
                        help='Do you want to append a string onto the names of the saved files? Default is no.')
    parser.set_defaults(save_suffix="")

    parser.add_argument('--file_suffix', metavar='file_suffix', type=str, action='store', \
                        help='If plotting from saved surface brightness files, use this to pass the file name suffix.')
    parser.set_defaults(file_suffix="")

    parser.add_argument('--unit_system', metavar='unit_system', type=str, action='store', \
                        help='What unit system? Default is default (s**-1 * cm**-3 * steradian**-1)')
    parser.set_defaults(unit_system='default')

    parser.add_argument('--resolution', metavar='resolution', type=str, action='store', \
                        help='How many times larger than simulation resolution? Default is 2')
    parser.set_defaults(resolution=1)

    parser.add_argument('--filter_type', metavar='filter_type', type=str, action='store', \
                        help='What filter type? Default is None (Options are: inflow_outflow or disk_cgm )')
    parser.set_defaults(filter_type=None)

    parser.add_argument('--scaling', dest='scaling', action='store_true', \
                        help='Do you want to scale the emissivity to observation? Default is no.')
    parser.set_defaults(scaling=False)

    parser.add_argument('--scale_factor', metavar='scale_factor', type=str, action='store', \
                        help='Do you want to scale the emissivity to observation? How much? The default is 1 because the default for scaling is no.')
    parser.set_defaults(scale_factor=1)

    parser.add_argument('--shell_count', metavar='shell_count', type=str, action='store', \
                        help='How many shell you have around disk when running disk finder? defualt is 0')
    parser.set_defaults(shell_count=0)

    parser.add_argument('--fov', metavar='fov', type=str, action='store', \
                        help='what is the field of view? Default is None. If it is None then it will take the refine box width')
    parser.set_defaults(fov=None)

    

    args = parser.parse_args()
    return args
##################################################################################################### 
def add_ion_fields(ds, ions):
    # Ensure ions is a list
    if isinstance(ions, str):
        ions = ions.split(',')  # Split the string into a list if necessary
    
    # Preprocess ions to ensure they are in the format expected by Trident
    formatted_ions = [trident_dict.get(ion, ion) for ion in ions]  # Replace using the dictionary if available
    
    # Add the formatted ions to the dataset
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
    hden_n_bins, hden_min, hden_max = 15, -5, 2 #17, -6, 2 #23, -9, 2
    T_n_bins, T_min, T_max = 51, 3, 8 #71, 2, 8

    hden=np.linspace(hden_min,hden_max,hden_n_bins)
    T=np.linspace(T_min,T_max, T_n_bins)
    table = np.zeros((hden_n_bins,T_n_bins))
    for i in range(hden_n_bins):
            table[i,:]=[float(l.split()[table_index]) for l in open(cloudy_path%(i+1)) if l[0] != "#"]
    return hden,T,table

def make_Cloudy_table_thin(table_index,cloudy_path):
    hden_n_bins, hden_min, hden_max = 17, -5, 2
    T_n_bins, T_min, T_max = 51, 3, 8 #71, 2, 8

    hden=np.linspace(hden_min,hden_max,hden_n_bins)
    T=np.linspace(T_min,T_max, T_n_bins)
    table = np.zeros((hden_n_bins,T_n_bins))
    for i in range(hden_n_bins):
            table[i,:]=[float(l.split()[table_index]) for l in open(cloudy_path_thin%(i+1)) if l[0] != "#"]
    return hden,T,table

def Emission_LyAlpha(field, data,scale_factor, unit_system='default', scaling  = False):
    H_N = np.log10(np.array(data["H_nuclei_density"]))
    Temperature = np.log10(np.array(data["Temperature"]))
    dia1 = bl_LA(H_N, Temperature)
    idx = np.isnan(dia1)
    dia1[idx] = -200.
    emission_line = (10**dia1) * ((10.0**H_N)**2.0)
    
    if unit_system == 'default':
        emission_line = emission_line / (4. * np.pi * 1.63e-11)
        return emission_line * ytEmU
    elif unit_system == 'ALT':
        emission_line = emission_line / (4. * np.pi)
        emission_line = emission_line / 4.25e10
        return emission_line * ytEmUALT
    else:
        raise ValueError("Invalid unit_system specified. Use 'default' or 'ALT'.")

def Emission_HAlpha(field, data,scale_factor, unit_system='default', scaling  = False):
    
    H_N = np.log10(np.array(data['H_nuclei_density']))
    Temperature = np.log10(np.array(data['Temperature']))
    dia1 = bl_HA(H_N, Temperature)
    idx = np.isnan(dia1)
    dia1[idx] = -200.
    emission_line = (10.**dia1) * ((10.**H_N)**2.0)
    
    if unit_system == 'default':
        emission_line = emission_line / (4. * np.pi * 3.03e-12)
        return emission_line * ytEmU
    elif unit_system == 'ALT':
        emission_line = emission_line / (4. * np.pi)
        emission_line = emission_line / 4.25e10
        return emission_line * ytEmUALT
    else:
        raise ValueError("Invalid unit_system specified. Use 'default' or 'ALT'.")
    
def Emission_CII_1335(field, data,scale_factor, unit_system='default', scaling  = False):
    
    H_N = np.log10(np.array(data["H_nuclei_density"]))
    Temperature = np.log10(np.array(data["Temperature"]))
    dia1 = bl_CII_1335(H_N, Temperature)
    idx = np.isnan(dia1)
    dia1[idx] = -200.
    if scaling == True:
        emission_line = scale_factor * ((10.0**dia1) * ((10.0**H_N)**2.0))
    else:
        emission_line = (10.0**dia1) * ((10.0**H_N)**2.0)
    emission_line = scale_by_metallicity(emission_line, 0.0, np.log10(np.array(data['metallicity'])))
    
    if unit_system == 'default':
        emission_line = emission_line / (4. * np.pi * 2.03e-11)
        return emission_line * ytEmU
    elif unit_system == 'ALT':
        emission_line = emission_line / (4. * np.pi)
        emission_line = emission_line / 4.25e10
        return emission_line * ytEmUALT
    else:
        raise ValueError("Invalid unit_system specified. Use 'default' or 'ALT'.")

def Emission_CIII_977(field, data,scale_factor, unit_system='default', scaling  = False):
    
    H_N = np.log10(np.array(data["H_nuclei_density"]))
    Temperature = np.log10(np.array(data["Temperature"]))
    dia1 = bl_CIII_977(H_N, Temperature)
    idx = np.isnan(dia1)
    dia1[idx] = -200.
    if scaling == True:
        emission_line = scale_factor * ((10.0**dia1) * ((10.0**H_N)**2.0))
    else:
        emission_line = ((10.0**dia1) * ((10.0**H_N)**2.0))
    emission_line = scale_by_metallicity(emission_line, 0.0, np.log10(np.array(data['metallicity'])))
    
    if unit_system == 'default':
        emission_line = emission_line / (4. * np.pi * 2.03e-11) # the constant value 2.03e-11 is energy per photon for CIII 977
        return emission_line * ytEmU
    elif unit_system == 'ALT':
        emission_line = emission_line / (4. * np.pi)
        emission_line = emission_line / 4.25e10
        return emission_line * ytEmUALT
    else:
        raise ValueError("Invalid unit_system specified. Use 'default' or 'ALT'.")
    
def Emission_CIII_1910(field, data,scale_factor, unit_system='default', scaling  = False):
    
    H_N = np.log10(np.array(data["H_nuclei_density"]))
    Temperature = np.log10(np.array(data["Temperature"]))
    dia1 = bl_CIII_1910(H_N, Temperature)
    idx = np.isnan(dia1)
    dia1[idx] = -200.
    if scaling == True:
        emission_line = scale_factor * ((10.0**dia1) * ((10.0**H_N)**2.0))
    else:
        emission_line = ((10.0**dia1) * ((10.0**H_N)**2.0))
    emission_line = scale_by_metallicity(emission_line, 0.0, np.log10(np.array(data['metallicity'])))
    
    if unit_system == 'default':
        emission_line = emission_line / (4. * np.pi * 2.03e-11)
        return emission_line * ytEmU
    elif unit_system == 'ALT':
        emission_line = emission_line / (4. * np.pi)
        emission_line = emission_line / 4.25e10
        return emission_line * ytEmUALT
    else:
        raise ValueError("Invalid unit_system specified. Use 'default' or 'ALT'.")

def Emission_CIV_1548(field, data,scale_factor, unit_system='default', scaling  = False):
    
    H_N = np.log10(np.array(data["H_nuclei_density"]))
    Temperature = np.log10(np.array(data["Temperature"]))
    dia1 = bl_CIV_1(H_N, Temperature)
    idx = np.isnan(dia1)
    dia1[idx] = -200.
    if scaling == True:
        emission_line = scale_factor * ((10.0**dia1) * ((10.0**H_N)**2.0))
    else:
        emission_line = ((10.0**dia1) * ((10.0**H_N)**2.0))
    emission_line = scale_by_metallicity(emission_line, 0.0, np.log10(np.array(data['metallicity'])))
    
    if unit_system == 'default':
        emission_line = emission_line / (4. * np.pi * 1.28e-11)
        return emission_line * ytEmU
    elif unit_system == 'ALT':
        emission_line = emission_line / (4. * np.pi)
        emission_line = emission_line / 4.25e10
        return emission_line * ytEmUALT
    else:
        raise ValueError("Invalid unit_system specified. Use 'default' or 'ALT'.")

def Emission_OVI(field, data,scale_factor, unit_system='default', scaling  = True):
    
    H_N = np.log10(np.array(data["H_nuclei_density"]))
    
    Temperature = np.log10(np.array(data["Temperature"]))
    dia1 = bl_OVI_1(H_N, Temperature)
    dia2 = bl_OVI_2(H_N, Temperature)
    idx = np.isnan(dia1)
    dia1[idx] = -200.
    dia2[idx] = -200.
    if scaling == True:
        emission_line = scale_factor * ((10.0**dia1) + (10**dia2)) * ((10.0**H_N)**2.0)
    else:
        emission_line = ((10.0**dia1) + (10**dia2)) * ((10.0**H_N)**2.0)
    emission_line = scale_by_metallicity(emission_line, 0.0, np.log10(np.array(data['metallicity'])))
    
    if unit_system == 'default':
        emission_line = emission_line / (4. * np.pi * 1.92e-11)
        return emission_line * ytEmU
    elif unit_system == 'ALT':
        emission_line = emission_line / (4. * np.pi)
        emission_line = emission_line / 4.25e10 # convert sr to arcsec^2
        return emission_line * ytEmUALT
    else:
        raise ValueError("Invalid unit_system specified. Use 'default' or 'ALT'.")

def Emission_SiIII_1207(field, data,scale_factor, unit_system='default', scaling  = False):
    
    H_N = np.log10(np.array(data["H_nuclei_density"]))
    Temperature = np.log10(np.array(data["Temperature"]))
    dia1 = bl_SiIII_1207(H_N, Temperature)
    idx = np.isnan(dia1)
    dia1[idx] = -200.
    emission_line = (10.0**dia1) * ((10.0**H_N)**2.0)
    emission_line = scale_by_metallicity(emission_line, 0.0, np.log10(np.array(data['metallicity'])))
    
    if unit_system == 'default':
        emission_line = emission_line / (4. * np.pi * 1.65e-11)
        return emission_line * ytEmU
    elif unit_system == 'ALT':
        emission_line = emission_line / (4. * np.pi)
        emission_line = emission_line / 4.25e10
        return emission_line * ytEmUALT
    else:
        raise ValueError("Invalid unit_system specified. Use 'default' or 'ALT'.")

def Emission_SiII_1814(field, data,scale_factor, unit_system='default', scaling  = False):
    
    H_N = np.log10(np.array(data["H_nuclei_density"]))
    Temperature = np.log10(np.array(data["Temperature"]))
    dia1 = bl_SiIII_1207(H_N, Temperature)
    idx = np.isnan(dia1)
    dia1[idx] = -200.
    emission_line = (10.0**dia1) * ((10.0**H_N)**2.0)
    emission_line = scale_by_metallicity(emission_line, 0.0, np.log10(np.array(data['metallicity'])))
    
    if unit_system == 'default':
        emission_line = emission_line / (4.*np.pi*1.10e-11)
        return emission_line * ytEmU
    elif unit_system == 'ALT':
        emission_line = emission_line / (4. * np.pi)
        emission_line = emission_line / 4.25e10
        return emission_line * ytEmUALT
    else:
        raise ValueError("Invalid unit_system specified. Use 'default' or 'ALT'.")

def Emission_SiIV_1394(field, data,scale_factor, unit_system='default', scaling  = False):
    
    H_N = np.log10(np.array(data["H_nuclei_density"]))
    Temperature = np.log10(np.array(data["Temperature"]))
    dia1 = bl_SiIII_1207(H_N, Temperature)
    idx = np.isnan(dia1)
    dia1[idx] = -200.
    emission_line = (10.0**dia1) * ((10.0**H_N)**2.0)
    emission_line = scale_by_metallicity(emission_line, 0.0, np.log10(np.array(data['metallicity'])))
    
    if unit_system == 'default':
        emission_line = emission_line / (4.*np.pi*1.43e-11)
        return emission_line * ytEmU
    elif unit_system == 'ALT':
        emission_line = emission_line / (4. * np.pi)
        emission_line = emission_line / 4.25e10
        return emission_line * ytEmUALT
    else:
        raise ValueError("Invalid unit_system specified. Use 'default' or 'ALT'.")

def Emission_MgII_2796(field, data,scale_factor, unit_system='default', scaling  = False):
    
    H_N = np.log10(np.array(data["H_nuclei_density"]))
    Temperature = np.log10(np.array(data["Temperature"]))
    dia1 = bl_SiIII_1207(H_N, Temperature)
    idx = np.isnan(dia1)
    dia1[idx] = -200.
    emission_line = (10.0**dia1) * ((10.0**H_N)**2.0)
    emission_line = scale_by_metallicity(emission_line, 0.0, np.log10(np.array(data['metallicity'])))
    
    if unit_system == 'default':
        emission_line = emission_line / (4.*np.pi*7.11e-12) 
        return emission_line * ytEmU
    elif unit_system == 'ALT':
        emission_line = emission_line / (4. * np.pi)
        emission_line = emission_line / 4.25e10
        return emission_line * ytEmUALT
    else:
        raise ValueError("Invalid unit_system specified. Use 'default' or 'ALT'.")

def filter_ds(box):
    '''This function filters the yt data object passed in as 'box' into inflow and outflow regions,
    based on metallicity, and returns the box filtered into these regions.'''

    if (segmentation_filter=='metallicity'):
        box_#inflow = box.include_below(('gas','metallicity'), 0.01, 'Zsun')
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
def original_make_FRB(ds, refine_box, snap, ions, unit_system='default', filter_type=None, filter_value=None,resolution=2):
    '''This function takes the dataset 'ds' and the refine box region 'refine_box' and
    makes a fixed resolution buffer of surface brightness from edge-on, face-on,
    and arbitrary orientation projections of all ions in the list 'ions'.'''

    halo_name = halo_dict[str(args.halo)]

    # Determine resolution and bin size
    pix_res = float(np.min(refine_box['dx'].in_units('kpc')))
    bin_size_kpc = resolution*pix_res
    round_bin_size_kpc = round(bin_size_kpc,2)
    # Ensure fov_refine_box is in kpc
    if not hasattr(ds.refine_width, 'in_units'):
        fov_refine_box = YTQuantity(ds.refine_width, 'kpc')  # Wrap in YTQuantity with units
    else:
        fov_refine_box = ds.refine_width.in_units('kpc')  # Convert to kpc if it has units

    # Set width consistently as a YTQuantity
    if args.fov is not None:
        width = YTQuantity(float(args.fov), 'kpc')  # Ensure YTQuantity type for consistency
    else:
        width = fov_refine_box  # Already in YTQuantity

    print('width:', width)
    width_value = width.v  # Extract value
    print('width_value:', width_value)
    res = int(width_value / bin_size_kpc)

    # Print for debugging
    print(f"Native resolution (pix_res): {pix_res:.2f} kpc")
    print(f"Field of view (FOV): {width_value:.3f} kpc")
    print(f"Adjusted bin size (bin_size_kpc): {bin_size_kpc:.2f} kpc")
    print(f"Adjusted number of bins (res): {res}")
    
    print('z=%.1f' % ds.get_parameter('CosmologyCurrentRedshift', 1))

    # Apply inflow/outflow or disk/CGM filtering using the filter_ds function if specified
    if filter_type == 'inflow_outflow':
        # Apply the inflow/outflow filtering
        box_inflow, box_outflow, box_neither = filter_ds(ds.all_data())
        data_sources = {'inflow': box_inflow, 'outflow': box_outflow, 'neither': box_neither}

    elif filter_type == 'disk_cgm':
        # Apply the disk/CGM filtering

        disk_cut_region = load_clump(ds, disk_file,source_cut=refine_box)
        box_cgm = refine_box - disk_cut_region

        if shell_count == 0:
            data_sources = {'cgm': box_cgm}
        else: 
            for i in range(0,shell_count):
                print('shell number',i)
                shell_clump_file = shell_path + f'test_DiskDilationShell_n{i}.h5' 
                shell_cut_region = load_clump(ds, shell_clump_file)
                box_cgm = box_cgm - shell_cut_region
            
            data_sources = {'cgm': box_cgm}
    else:
        # Standard filtering or no filter
        data_sources = {'all': refine_box}
        if filter_type and filter_value:
            if filter_type == 'temperature':
                data_sources['all'] = data_sources['all'].cut_region([f"(obj['gas', 'temperature'] < {filter_value})"])
            elif filter_type == 'density':
                data_sources['all'] = data_sources['all'].cut_region([f"(obj['gas', 'density'] > {filter_value})"])
            else:
                raise ValueError("Unsupported filter type. Supported types: 'temperature', 'density'.")

    # Define the unit string based on unit_system
    if unit_system == 'default':
        unit_label = '[photons s$^{-1}$ cm$^{-2}$ sr$^{-1}$]'
    elif unit_system == 'ALT':
        unit_label = '[erg s$^{-1}$ cm$^{-2}$ arcsec$^{-2}$]'
    else:
        raise ValueError("Invalid unit_system specified. Use 'default' or 'ALT'.")
    

    # Create HDF5 file for saving emission maps
    save_path = prefix + f'FRBs/' 
    os.makedirs(save_path, exist_ok=True)  # Ensure the directory exists
    f = h5py.File(save_path + halo_name + '_emission_maps' + save_suffix + '.hdf5', 'a')
    grp = f.create_group('z=%.1f' % ds.get_parameter('CosmologyCurrentRedshift', 1))
    grp.attrs.create("image_extent_kpc", ds.refine_width)
    grp.attrs.create("redshift", ds.get_parameter('CosmologyCurrentRedshift'))
    grp.attrs.create("halo_name", halo_name)
    grp.attrs.create("emission_units", unit_label)
    grp.attrs.create("gas_density_units", 'g/cm^2')
    grp.attrs.create("stars_density_units", 'Msun/kpc^2')
    grp.attrs.create("bin_size_kpc", round_bin_size_kpc)
    grp.attrs.create("number_of_bins", res)





    # Loop through ions and create projections for each region
    for region, data_source in data_sources.items():
        for ion in ions:
            print(ion)

             # Choose colormap based on ion and emission limits
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
            elif args.Juniper_limit and ion in  ['OVI','CII']:
                cmap1 = cmr.take_cmap_colors('magma', 4, cmap_range=(0.5, 1), return_fmt='rgba')
                cmap2 = cmr.take_cmap_colors('cmr.neutral_r', 5, cmap_range=(0.0, 0.6), return_fmt='rgba')
                cmap = np.hstack([cmap2, cmap1])
                mymap = mcolors.LinearSegmentedColormap.from_list('cmap', cmap)
            elif args.Juniper_limit and ion == 'CIV':
                cmap1 = cmr.take_cmap_colors('magma', 5, cmap_range=(0.5, 1), return_fmt='rgba')
                cmap2 = cmr.take_cmap_colors('cmr.neutral_r', 7, cmap_range=(0.0, 0.6), return_fmt='rgba')
                cmap = np.hstack([cmap2, cmap1])
                mymap = mcolors.LinearSegmentedColormap.from_list('cmap', cmap)
            elif args.Juniper_limit and ion == 'CIII':
                cmap1 = cmr.take_cmap_colors('magma', 3, cmap_range=(0.5, 1), return_fmt='rgba')
                cmap2 = cmr.take_cmap_colors('cmr.neutral_r', 7, cmap_range=(0.0, 0.6), return_fmt='rgba')
                cmap = np.hstack([cmap2, cmap1])
                mymap = mcolors.LinearSegmentedColormap.from_list('cmap', cmap)
            else:
                mymap = cmr.get_sub_cmap('cmr.flamingo', 0.2, 1)
                #mymap = cmr.get_sub_cmap('magma', 0, 1)
            mymap.set_bad(mymap(0))

            #Edge-on projection
            print('projection width',width)
            proj_edge = yt.ProjectionPlot(ds, ds.x_unit_disk, ('gas', 'Emission_' + ions_dict[ion]),
                                          center=ds.halo_center_kpc, data_source=data_source,width=(float(width_value), 'kpc'),
                                          north_vector=ds.z_unit_disk, buff_size=[res, res], method = 'integrate', weight_field=None) #(ds.refine_width, 'kpc')
            frb_edge = proj_edge.frb[('gas', 'Emission_' + ions_dict[ion])]
            dset1 = grp.create_dataset(f"{ion}_emission_edge_{region}", data=frb_edge)

            # Set colormap and save projection plot
        
            proj_edge.set_cmap('Emission_' + ions_dict[ion], mymap)
            proj_edge.set_zlim('Emission_' + ions_dict[ion], zlim_dict[ion][0], zlim_dict[ion][1])
            proj_edge.set_colorbar_label('Emission_' + ions_dict[ion], label_dict[ion] + ' Emission ' + unit_label)
            proj_edge.set_font_size(24)
            proj_edge.set_xlabel('x (kpc)')
            proj_edge.set_ylabel('y (kpc)')
            proj_edge.annotate_timestamp(corner='upper_left', redshift=True, time=True, draw_inset_box=True)
            proj_edge.save(save_path + f'{snap}_{ion}_emission_map_edge-on_{region}' + save_suffix + '.png')

            # Face-on projection
            proj_face = yt.ProjectionPlot(ds, ds.z_unit_disk, ('gas', 'Emission_' + ions_dict[ion]),
                                          center=ds.halo_center_kpc, data_source=data_source,width=(float(width_value), 'kpc'),
                                          north_vector=ds.x_unit_disk, buff_size=[res, res],weight_field=None) #(ds.refine_width, 'kpc')
            frb_face = proj_face.frb[('gas', 'Emission_' + ions_dict[ion])]
            dset2 = grp.create_dataset(f"{ion}_emission_face_{region}", data=frb_face)

            # Set colormap and save projection plot
            proj_face.set_cmap('Emission_' + ions_dict[ion], mymap)
            proj_face.set_zlim('Emission_' + ions_dict[ion], zlim_dict[ion][0], zlim_dict[ion][1])
            proj_face.set_colorbar_label('Emission_' + ions_dict[ion], label_dict[ion] + ' Emission ' + unit_label)
            proj_face.set_font_size(24)
            proj_face.set_xlabel('x (kpc)')
            proj_face.set_ylabel('y (kpc)')
            proj_face.annotate_timestamp(corner='upper_left', redshift=True, time=True, draw_inset_box=True)
            proj_face.save(save_path + f'{snap}_{ion}_emission_map_face-on_{region}' + save_suffix + '.png')

    # Close the HDF5 file after saving the datasets
    print('finished')
    f.close()
  
def make_FRB(ds, refine_box, snap, ions, unit_system='default', filter_type=None, filter_value=None,resolution=2):
    '''This function takes the dataset 'ds' and the refine box region 'refine_box' and
    makes a fixed resolution buffer of surface brightness from edge-on, face-on,
    and arbitrary orientation projections of all ions in the list 'ions'.'''

    halo_name = halo_dict[str(args.halo)]
    print('box size:',ds.refine_width)

    # Determine resolution and bin size
    pix_res = float(np.min(refine_box['dx'].in_units('kpc')))
    bin_size_kpc = resolution*pix_res
    round_bin_size_kpc = round(bin_size_kpc,2)
    # Ensure fov_refine_box is in kpc
    if not hasattr(ds.refine_width, 'in_units'):
        fov_refine_box = YTQuantity(ds.refine_width, 'kpc')  # Wrap in YTQuantity with units
    else:
        fov_refine_box = ds.refine_width.in_units('kpc')  # Convert to kpc if it has units

    # Set width consistently as a YTQuantity
    if args.fov is not None:
        width = YTQuantity(float(args.fov), 'kpc')  # Ensure YTQuantity type for consistency
    else:
        width = fov_refine_box  # Already in YTQuantity

    print('width:', width)
    width_value = width.v  # Extract value
    print('width_value:', width_value)
    res = int(width_value / bin_size_kpc)

    # Print for debugging
    print(f"Native resolution (pix_res): {pix_res:.2f} kpc")
    print(f"Field of view (FOV): {width_value:.3f} kpc")
    print(f"Adjusted bin size (bin_size_kpc): {bin_size_kpc:.2f} kpc")
    print(f"Adjusted number of bins (res): {res}")
    
    print('z=%.1f' % ds.get_parameter('CosmologyCurrentRedshift', 1))

    # Apply inflow/outflow or disk/CGM filtering using the filter_ds function if specified
    if filter_type == 'inflow_outflow':
        # Apply the inflow/outflow filtering
        box_inflow, box_outflow, box_neither = filter_ds(ds.all_data())
        data_sources = {'inflow': box_inflow, 'outflow': box_outflow, 'neither': box_neither}

    elif filter_type == 'disk_cgm':
        # Apply the disk/CGM filtering

        disk_cut_region = load_clump(ds, disk_file,source_cut=refine_box)
        box_cgm = refine_box - disk_cut_region

        if shell_count == 0:
            data_sources = {'cgm': box_cgm}
        else: 
            for i in range(0,shell_count):
                print('shell number',i)
                shell_clump_file = shell_path + f'test_DiskDilationShell_n{i}.h5' 
                shell_cut_region = load_clump(ds, shell_clump_file)
                box_cgm = box_cgm - shell_cut_region
            
            data_sources = {'cgm': box_cgm}
    else:
        # Standard filtering or no filter
        data_sources = {'all': refine_box}
        if filter_type and filter_value:
            if filter_type == 'temperature':
                data_sources['all'] = data_sources['all'].cut_region([f"(obj['gas', 'temperature'] < {filter_value})"])
            elif filter_type == 'density':
                data_sources['all'] = data_sources['all'].cut_region([f"(obj['gas', 'density'] > {filter_value})"])
            else:
                raise ValueError("Unsupported filter type. Supported types: 'temperature', 'density'.")

    # Define the unit string based on unit_system
    if unit_system == 'default':
        unit_label = '[photons s$^{-1}$ cm$^{-2}$ sr$^{-1}$]'
    elif unit_system == 'ALT':
        unit_label = '[erg s$^{-1}$ cm$^{-2}$ arcsec$^{-2}$]'
    else:
        raise ValueError("Invalid unit_system specified. Use 'default' or 'ALT'.")
    

    # Create HDF5 file for saving emission maps
    save_path = prefix + f'FRBs/' 
    os.makedirs(save_path, exist_ok=True)  # Ensure the directory exists
    f = h5py.File(save_path + halo_name + '_emission_maps' + save_suffix + '.hdf5', 'a')
    grp = f.create_group('z=%.1f' % ds.get_parameter('CosmologyCurrentRedshift', 1))
    grp.attrs.create("image_extent_kpc", ds.refine_width)
    grp.attrs.create("redshift", ds.get_parameter('CosmologyCurrentRedshift'))
    grp.attrs.create("halo_name", halo_name)
    grp.attrs.create("emission_units", unit_label)
    grp.attrs.create("gas_density_units", 'g/cm^2')
    grp.attrs.create("stars_density_units", 'Msun/kpc^2')
    grp.attrs.create("bin_size_kpc", round_bin_size_kpc)
    grp.attrs.create("number_of_bins", res)





    # Loop through ions and create projections for each region
    for region, data_source in data_sources.items():

        # Define ion-specific colormap settings
        ion_colormaps = {
            'Halpha': {'limit_flag': 'Dragonfly_limit', 'cmap_above': 'cmr.flamingo', 'cmap_below': 'cmr.neutral_r'},
            'OVI_Aspera': {'limit_flag': 'Aspera_limit', 'cmap_above': 'cmr.flamingo', 'cmap_below': 'cmr.neutral_r'},
            'OVI_Juniper': {'limit_flag': 'Juniper_limit', 'cmap_above': 'cmr.flamingo', 'cmap_below': 'cmr.neutral_r'},
            'OVI_Magpie': {'limit_flag': 'Juniper_limit', 'cmap_above': 'cmr.flamingo', 'cmap_below': 'cmr.neutral_r'},
            'OVI_HWO': {'limit_flag': 'Juniper_limit', 'cmap_above': 'cmr.flamingo', 'cmap_below': 'cmr.neutral_r'},
            'OVI_Muse': {'limit_flag': 'Juniper_limit', 'cmap_above': 'cmr.flamingo', 'cmap_below': 'cmr.neutral_r'},
            'CII': {'limit_flag': 'Juniper_limit', 'cmap_above': 'cmr.flamingo', 'cmap_below': 'cmr.neutral_r'},
            'CIV': {'limit_flag': 'Juniper_limit', 'cmap_above': 'cmr.flamingo', 'cmap_below': 'cmr.neutral_r'},
            'CIII': {'limit_flag': 'Juniper_limit', 'cmap_above': 'cmr.flamingo', 'cmap_below': 'cmr.neutral_r'}
        }

        for ion in ions:
            print(ion)

            # Default colormap
            mymap = cmr.get_sub_cmap('cmr.flamingo', 0.2, 1)

            # Determine which colormap settings to use based on active limits
            if ion == 'OVI':
                if args.Aspera_limit:
                    settings = ion_colormaps['OVI_Aspera']
                elif args.Juniper_limit:
                    settings = ion_colormaps['OVI_Juniper']
                elif args.Magpie_limit:
                    settings = ion_colormaps['OVI_Magpie']
                elif args.Muse_limit:
                    settings = ion_colormaps['OVI_Muse']
                elif args.HWO_limit:
                    settings = ion_colormaps['OVI_HWO']
            elif ion in ion_colormaps:
                settings = ion_colormaps[ion]
            
            # Specify intrument and ions
            if (
                (args.Juniper_limit and ion in ['OVI', 'CII', 'CIV', 'CIII']) or
                (args.Dragonfly_limit and ion == 'Halpha') or
                (args.Aspera_limit and ion == 'OVI') or
                (args.Magpie_limit and ion in ['OVI', 'MgII', 'CIV', 'CIII']) or
                (args.Muse_limit and ion == 'OVI') or
                (args.HWO_limit and ion == 'OVI')
            ):
                # Get the zlim range
                if ion in zlim_dict:
                    zmin, zmax = zlim_dict[ion]
                

                # Get the flux threshold, defaulting to midpoint if not defined
                threshold = flux_threshold_dict[ion]
                # Normalize the threshold position
                fraction_below = (np.log10(threshold) - np.log10(zmin)) / (np.log10(zmax) - np.log10(zmin))
                fraction_above = 1 - fraction_below

                print('fraction_below',fraction_below)
                print('fraction_above',fraction_above)

                # Convert fractions to discrete colormap sizes
                total_colors = 100  # Total number of colors in the map
                below_n = max(1, int((fraction_below * total_colors)))
                above_n = total_colors - below_n #max(1, int((fraction_above * total_colors)))
                
                print('below_n',below_n)
                print('above_n',above_n)

                # Ensure sum doesn't exceed total
                # if below_n + above_n > total_colors:
                #     above_n = total_colors - below_n

                # Generate colormap sections
                cmap1 = cmr.take_cmap_colors(settings['cmap_above'], above_n, cmap_range=(0.5, 1), return_fmt='rgba')
                cmap2 = cmr.take_cmap_colors(settings['cmap_below'], below_n, cmap_range=(0.0, 0.6), return_fmt='rgba')
                cmap = np.hstack([cmap2, cmap1])
                mymap = mcolors.LinearSegmentedColormap.from_list('cmap', cmap)

            mymap.set_bad(mymap(0))


            #Edge-on projection
            print('projection width',width)
            proj_edge = yt.ProjectionPlot(ds, ds.x_unit_disk, ('gas', 'Emission_' + ions_dict[ion]),
                                          center=ds.halo_center_kpc, data_source=data_source,width=(float(width_value), 'kpc'),
                                          north_vector=ds.z_unit_disk, buff_size=[res, res], method = 'integrate', weight_field=None) #(ds.refine_width, 'kpc')
            frb_edge = proj_edge.frb[('gas', 'Emission_' + ions_dict[ion])]
            dset1 = grp.create_dataset(f"{ion}_emission_edge_{region}", data=frb_edge)

            # Set colormap and save projection plot
        
            proj_edge.set_cmap('Emission_' + ions_dict[ion], mymap)
            proj_edge.set_zlim('Emission_' + ions_dict[ion], zlim_dict[ion][0], zlim_dict[ion][1])
            proj_edge.set_colorbar_label('Emission_' + ions_dict[ion], label_dict[ion] + ' Emission ' + unit_label)
            proj_edge.set_font_size(24)
            proj_edge.set_xlabel('x (kpc)')
            proj_edge.set_ylabel('y (kpc)')
            #proj_edge.annotate_timestamp(corner='upper_left', redshift=True, time=True, draw_inset_box=True)
            proj_edge.save(save_path + f'{snap}_{ion}_emission_map_edge-on_{region}' + save_suffix + '.png')

            # Face-on projection
            proj_face = yt.ProjectionPlot(ds, ds.z_unit_disk, ('gas', 'Emission_' + ions_dict[ion]),
                                          center=ds.halo_center_kpc, data_source=data_source,width=(float(width_value), 'kpc'),
                                          north_vector=ds.x_unit_disk, buff_size=[res, res],weight_field=None) #(ds.refine_width, 'kpc')
            frb_face = proj_face.frb[('gas', 'Emission_' + ions_dict[ion])]
            dset2 = grp.create_dataset(f"{ion}_emission_face_{region}", data=frb_face)

            # Set colormap and save projection plot
            proj_face.set_cmap('Emission_' + ions_dict[ion], mymap)
            proj_face.set_zlim('Emission_' + ions_dict[ion], zlim_dict[ion][0], zlim_dict[ion][1])
            proj_face.set_colorbar_label('Emission_' + ions_dict[ion], label_dict[ion] + ' Emission ' + unit_label)
            proj_face.set_font_size(24)
            proj_face.set_xlabel('x (kpc)')
            proj_face.set_ylabel('y (kpc)')
            #proj_face.annotate_timestamp(corner='upper_left', redshift=True, time=True, draw_inset_box=True)
            proj_face.save(save_path + f'{snap}_{ion}_emission_map_face-on_{region}' + save_suffix + '.png')

    # Close the HDF5 file after saving the datasets
    print('Emission frb saved')
    f.close()
  
def emission_map_vbins(ds, snap, ions,unit_system='default', filter_type=None, filter_value=None):
    '''Makes many emission maps for each ion in 'ions', oriented both edge-on and face-on, for each line-of-sight velocity bin.'''

    vbins = np.arange(-500., 550., 50.)  # Velocity bins
    ad = ds.all_data()

    for i in range(len(ions)):
        ion = ions[i]

        # Choose colormap based on ion and emission limits
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
        elif args.Juniper_limit:
            cmap1 = cmr.take_cmap_colors('cmr.flamingo', 1, cmap_range=(0.4, 0.8), return_fmt='rgba')
            cmap2 = cmr.take_cmap_colors('cmr.neutral_r', 6, cmap_range=(0.2, 0.6), return_fmt='rgba')
            cmap = np.hstack([cmap2, cmap1])
            mymap = mcolors.LinearSegmentedColormap.from_list('cmap', cmap)
        else:
            mymap = cmr.get_sub_cmap('cmr.flamingo', 0.2, 0.8)
        mymap.set_bad(mymap.colors[0])

        # Loop through each velocity bin
        for v in range(len(vbins) - 1):
            # Filter the data by the current velocity bin
            vbox = ds.cut_region(ad, [f"obj[('gas', 'vx_disk')] > {vbins[v]:.1f}"])
            vbox = ds.cut_region(vbox, [f"obj[('gas', 'vx_disk')] < {vbins[v+1]:.1f}"])

            # Apply filtering if specified (e.g., temperature or density cut)
            if filter_type and filter_value:
                if filter_type == 'temperature':
                    vbox = vbox.cut_region([f"(obj['gas', 'temperature'] < {filter_value})"])
                elif filter_type == 'density':
                    vbox = vbox.cut_region([f"(obj['gas', 'density'] > {filter_value})"])
                else:
                    raise ValueError("Unsupported filter type. Supported types: 'temperature', 'density'.")

            # Edge-on projection
            proj_edge = yt.ProjectionPlot(ds, ds.x_unit_disk, ('gas', 'Emission_' + ions_dict[ion]), 
                                        center=ds.halo_center_kpc, width=(ds.refine_width, 'kpc'),
                                        north_vector=ds.z_unit_disk, data_source=vbox)
            proj_edge.set_cmap('Emission_' + ions_dict[ion], mymap)
            proj_edge.set_zlim('Emission_' + ions_dict[ion], zlim_dict[ion][0], zlim_dict[ion][1])
            proj_edge.set_colorbar_label('Emission_' + ions_dict[ion], label_dict[ion] + 'Emission' + unit_label)
            proj_edge.set_font_size(20)
            proj_edge.annotate_title(f'$%d < v_{{\\rm los}} < %d$' % (vbins[v], vbins[v+1]))
            proj_edge.annotate_timestamp(corner='upper_left', redshift=True, time=True, draw_inset_box=True)
            proj_edge.save(prefix + 'EmissionMap/' + snap + '_' + ion + '_emission_map_edge-on_vbin' + str(v) + save_suffix + '.png')

            # Face-on projection
            proj_face = yt.ProjectionPlot(ds, ds.z_unit_disk, ('gas', 'Emission_' + ions_dict[ion]), 
                                        center=ds.halo_center_kpc, width=(ds.refine_width, 'kpc'),
                                        north_vector=ds.x_unit_disk, data_source=vbox)
            proj_face.set_cmap('Emission_' + ions_dict[ion], mymap)
            proj_face.set_zlim('Emission_' + ions_dict[ion], zlim_dict[ion][0], zlim_dict[ion][1])
            proj_face.set_colorbar_label('Emission_' + ions_dict[ion], label_dict[ion] + 'Emission' + unit_label)
            proj_face.set_font_size(20)
            proj_face.annotate_title(f'$%d < v_{{\\rm los}} < %d$' % (vbins[v], vbins[v+1]))
            proj_face.annotate_timestamp(corner='upper_left', redshift=True, time=True, draw_inset_box=True)
            proj_face.save(prefix + 'EmissionMap/' + snap + '_' + ion + '_emission_map_face-on_vbin' + str(v) + save_suffix + '.png')

def emission_map(ds, refine_box, snap, ions, unit_system='default', filter_type=None, filter_value=None,resolution=2):
    '''Makes emission maps for each ion in 'ions', oriented both edge-on and face-on.'''

    halo_name = halo_dict[str(args.halo)]

    # Determine resolution and bin size
    pix_res = float(np.min(refine_box['dx'].in_units('kpc')))
    bin_size_kpc = resolution*pix_res
    round_bin_size_kpc = round(bin_size_kpc,2)
    # Ensure fov_refine_box is in kpc
    if not hasattr(ds.refine_width, 'in_units'):
        fov_refine_box = YTQuantity(ds.refine_width, 'kpc')  # Wrap in YTQuantity with units
    else:
        fov_refine_box = ds.refine_width.in_units('kpc')  # Convert to kpc if it has units

    # Set width consistently as a YTQuantity
    if args.fov is not None:
        width = YTQuantity(float(args.fov), 'kpc')  # Ensure YTQuantity type for consistency
    else:
        width = fov_refine_box  # Already in YTQuantity

    print('width:', width)
    width_value = width.v  # Extract value
    print('width_value:', width_value)
    res = int(width_value / bin_size_kpc)

    # Apply inflow/outflow or disk/CGM filtering using the filter_ds function if specified
    if filter_type == 'inflow_outflow':
        # Apply the inflow/outflow filtering
        box_inflow, box_outflow, box_neither = filter_ds(ds.all_data())
        data_sources = {'inflow': box_inflow, 'outflow': box_outflow, 'neither': box_neither}

    elif filter_type == 'disk_cgm':
        # Apply the disk/CGM filtering

        disk_cut_region = load_clump(ds, disk_file,source_cut=refine_box)
        box_cgm = refine_box - disk_cut_region

        if shell_count == 0:
            data_sources = {'cgm': box_cgm}
        else: 
            for i in range(0,shell_count):
                print('shell number',i)
                shell_clump_file = shell_path + f'test_DiskDilationShell_n{i}.h5' 
                shell_cut_region = load_clump(ds, shell_clump_file)
                box_cgm = box_cgm - shell_cut_region
            
            data_sources = {'cgm': box_cgm}
    else:
        # Standard filtering or no filter
        data_sources = {'all': refine_box}
        if filter_type and filter_value:
            if filter_type == 'temperature':
                data_sources['all'] = data_sources['all'].cut_region([f"(obj['gas', 'temperature'] < {filter_value})"])
            elif filter_type == 'density':
                data_sources['all'] = data_sources['all'].cut_region([f"(obj['gas', 'density'] > {filter_value})"])
            else:
                raise ValueError("Unsupported filter type. Supported types: 'temperature', 'density'.")

    # Define the unit string based on unit_system
    if unit_system == 'default':
        unit_label = '[photons s$^{-1}$ cm$^{-2}$ sr$^{-1}$]'
    elif unit_system == 'ALT':
        unit_label = '[erg s$^{-1}$ cm$^{-2}$ arcsec$^{-2}$]'
    else:
        raise ValueError("Invalid unit_system specified. Use 'default' or 'ALT'.")
    

    # Loop through ions and create projections for each region
    for region, data_source in data_sources.items():
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
                mymap = cmr.get_sub_cmap('cmr.flamingo', 0.2, 1)
                

            proj_edge = yt.ProjectionPlot(ds, ds.x_unit_disk, ('gas','Emission_' + ions_dict[ion]), center=ds.halo_center_kpc, data_source=data_source, width=(float(width_value), 'kpc'),
                                          buff_size=[res, res], method = 'integrate', weight_field=None, north_vector=ds.z_unit_disk)
            proj_edge.set_cmap('Emission_' + ions_dict[ion], mymap)
            proj_edge.set_zlim('Emission_' + ions_dict[ion], zlim_dict[ion][0], zlim_dict[ion][1])
            proj_edge.set_colorbar_label('Emission_' + ions_dict[ion], label_dict[ion] + ' Emission [photons s$^{-1}$ cm$^{-2}$ sr$^{-2}$]')
            proj_edge.set_font_size(20)
            proj_edge.annotate_timestamp(corner='upper_left', redshift=True, time=True, draw_inset_box=True)
            proj_edge.save(prefix + 'Projections/' + snap + '_' + ion + '_emission_map_edge-on' + save_suffix + '.png')

            proj_face = yt.ProjectionPlot(ds, ds.z_unit_disk, ('gas','Emission_' + ions_dict[ion]), center=ds.halo_center_kpc, data_source=data_source, width=(float(width_value), 'kpc'),
                                          buff_size=[res, res], method = 'integrate', weight_field=None, north_vector=ds.x_unit_disk)
            proj_face.set_cmap('Emission_' + ions_dict[ion], mymap)
            proj_face.set_zlim('Emission_' + ions_dict[ion], zlim_dict[ion][0], zlim_dict[ion][1])
            proj_face.set_colorbar_label('Emission_' + ions_dict[ion], label_dict[ion] + ' Emission [photons s$^{-1}$ cm$^{-2}$ sr$^{-2}$]')
            proj_face.set_font_size(20)
            proj_face.annotate_timestamp(corner='upper_left', redshift=True, time=True, draw_inset_box=True)
            proj_face.save(prefix + 'Projections/' + snap + '_' + ion + '_emission_map_face-on' + save_suffix + '.png')

def make_column_density_FRB(ds, refine_box, snap, ions,scaling = True, scale_factor=100, filter_type=None, filter_value=None, resolution=2):
    '''This function calculates and saves projected column density FRBs '''

    halo_name = halo_dict[str(args.halo)]

    # Determine resolution and bin size
    pix_res = float(np.min(refine_box['dx'].in_units('kpc')))
    bin_size_kpc = resolution*pix_res
    round_bin_size_kpc = round(bin_size_kpc,2)

    # Ensure fov_refine_box is in kpc
    if not hasattr(ds.refine_width, 'in_units'):
        fov_refine_box = YTQuantity(ds.refine_width, 'kpc')  # Wrap in YTQuantity with units
    else:
        fov_refine_box = ds.refine_width.in_units('kpc')  # Convert to kpc if it has units

    # Set width consistently as a YTQuantity
    if args.fov is not None:
        width = YTQuantity(float(args.fov), 'kpc')  # Ensure YTQuantity type for consistency
    else:
        width = fov_refine_box  # Already in YTQuantity

    print('width:', width)
    width_value = width.v  # Extract value
    print('width_value:', width_value)
    res = int(width_value / bin_size_kpc)

    # Apply filtering (if specified)
    if filter_type == 'inflow_outflow':
        box_inflow, box_outflow, box_neither = filter_ds(ds.all_data())
        data_sources = {'inflow': box_inflow, 'outflow': box_outflow, 'neither': box_neither}
    elif filter_type == 'disk_cgm':
        # Apply the disk/CGM filtering

        disk_cut_region = load_clump(ds, disk_file,source_cut=refine_box)
        box_cgm = refine_box - disk_cut_region

        if shell_count == 0:
            data_sources = {'cgm': box_cgm}
        else: 
            for i in range(0,shell_count):
                print('shell number',i)
                shell_clump_file = shell_path + f'test_DiskDilationShell_n{i}.h5' 
                shell_cut_region = load_clump(ds, shell_clump_file)
                box_cgm = box_cgm - shell_cut_region
            
            data_sources = {'cgm': box_cgm}
    else:
        data_sources = {'all': refine_box}
        if filter_type and filter_value:
            if filter_type == 'temperature':
                data_sources['all'] = data_sources['all'].cut_region([f"(obj['gas', 'temperature'] < {filter_value})"])
            elif filter_type == 'density':
                data_sources['all'] = data_sources['all'].cut_region([f"(obj['gas', 'density'] > {filter_value})"])
            else:
                raise ValueError("Unsupported filter type. Supported types: 'temperature', 'density'.")

    # Create HDF5 file for saving mass maps
    save_path = prefix + f'FRBs/' 
    os.makedirs(save_path, exist_ok=True)  # Ensure the directory exists
    print('savepath',save_path)
    f = h5py.File(save_path + halo_name + '_emission_maps' + save_suffix + '.hdf5', 'a')
    # Check if the group already exists
    redshift_group_name = 'z=%.1f' % ds.get_parameter('CosmologyCurrentRedshift', 1)
    if redshift_group_name not in f:
        grp = f.create_group(redshift_group_name)
        grp.attrs.create("image_extent_kpc", ds.refine_width)
        grp.attrs.create("redshift", ds.get_parameter('CosmologyCurrentRedshift'))
        grp.attrs.create("halo_name", halo_name)
        grp.attrs.create("bin_size_kpc", round_bin_size_kpc)
        grp.attrs.create("number_of_bins", res)
    else:
        grp = f[redshift_group_name]  # Open the existing group


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
                                          width=(ds.refine_width, 'kpc'), north_vector=ds.z_unit_disk,
                                          buff_size=[res, res], weight_field=None)
            frb_edge = proj_edge.frb[numdensity_field]  # Surface density in g/cm^2
            # Save mass FRB, total mass, and positions in HDF5
            dset1 = grp.create_dataset(f"{ion}_numdensity_edge_{region}", data=frb_edge)
            
            # Calculate positions
            # Calculate positions relative to the halo center
            halo_center_x = ds.halo_center_kpc[0].in_units('kpc')
            halo_center_y = ds.halo_center_kpc[1].in_units('kpc')

            # Edge-on projection
            x_min = -width_value/2
            x_max = width_value/2
            y_min = -width_value/2
            y_max = width_value/2

            x_edges = np.linspace(x_min, x_max, res + 1)
            y_edges = np.linspace(y_min, y_max, res + 1)

            x_positions = 0.5 * (x_edges[:-1] + x_edges[1:]) #- halo_center_x
            y_positions = 0.5 * (y_edges[:-1] + y_edges[1:]) #- halo_center_y

            # Debugging
            print(f"x_min: {x_min}, x_max: {x_max}, y_min: {y_min}, y_max: {y_max}")
            print(f"x_positions (kpc): {x_positions}")
            print(f"y_positions (kpc): {y_positions}")

            # Save positions
            dset_x = grp.create_dataset(f"{ion}_x_edge_{region}", data=x_positions)
            dset_y = grp.create_dataset(f"{ion}_y_edge_{region}", data=y_positions)

            
            

            # Face-on projection (surface density)
            proj_face = yt.ProjectionPlot(ds, ds.z_unit_disk, numdensity_field,
                                          center=ds.halo_center_kpc, data_source=data_source,
                                          width=(ds.refine_width, 'kpc'), north_vector=ds.x_unit_disk,
                                          buff_size=[res, res], weight_field=None)
            frb_face = proj_face.frb[numdensity_field]  # Surface density in g/cm^2
        
            dset2 = grp.create_dataset(f"{ion}_numdensity_face_{region}", data=frb_face)

            # Save relative positions
            dset_x = grp.create_dataset(f"{ion}_x_face_{region}", data=x_positions)
            dset_y = grp.create_dataset(f"{ion}_y_face_{region}", data=y_positions)


    # Close the HDF5 file after saving the datasets
    print('Number density FRBs finished')
    f.close()

def compute_mass_in_emission_pixels(ds, refine_box, snap, ions, resolution=2):
    """Compute the total ion mass that contributes to each pixel of an emission FRB for edge-on and face-on views."""

    halo_name = halo_dict[str(args.halo)]

    # Determine resolution and bin size
    pix_res = float(np.min(refine_box['dx'].in_units('kpc')))
    bin_size_kpc = resolution*pix_res
    round_bin_size_kpc = round(bin_size_kpc,2)
    # pixel sizes in cm for mass and emission maps
    min_res = resolution * (np.min(refine_box['dx'].in_units('kpc')))
    bin_size_cm = min_res.in_units('cm')
    print('bin_size_cm',bin_size_cm)

    # Ensure fov_refine_box is in kpc
    if not hasattr(ds.refine_width, 'in_units'):
        fov_refine_box = YTQuantity(ds.refine_width, 'kpc')  # Wrap in YTQuantity with units
    else:
        fov_refine_box = ds.refine_width.in_units('kpc')  # Convert to kpc if it has units

    # Set width consistently as a YTQuantity
    if args.fov is not None:
        width = YTQuantity(float(args.fov), 'kpc')  # Ensure YTQuantity type for consistency
    else:
        width = fov_refine_box  # Already in YTQuantity

    print('width:', width)
    width_value = width.v  # Extract value
    print('width_value:', width_value)
    res = int(width_value / bin_size_kpc)

    # Print for double checking the resolution and pixel sizes
    print(f"Native resolution (pix_res): {pix_res:.2f} kpc")
    print(f"Field of view (FOV): {width_value:.3f} kpc")
    print(f"Adjusted bin size (bin_size_kpc): {bin_size_kpc:.2f} kpc")
    print(f"Adjusted number of bins (res): {res}")

    # Apply filtering (if specified)
    if filter_type == 'inflow_outflow':
        box_inflow, box_outflow, box_neither = filter_ds(ds.all_data())
        data_sources = {'inflow': box_inflow, 'outflow': box_outflow, 'neither': box_neither}
    elif filter_type == 'disk_cgm':
        # Apply the disk/CGM filtering

        disk_cut_region = load_clump(ds, disk_file,source_cut=refine_box)
        box_cgm = refine_box - disk_cut_region

        if shell_count == 0:
            data_sources = {'cgm': box_cgm}
        else: 
            for i in range(0,shell_count):
                print('shell number',i)
                shell_clump_file = shell_path + f'test_DiskDilationShell_n{i}.h5' 
                shell_cut_region = load_clump(ds, shell_clump_file)
                box_cgm = box_cgm - shell_cut_region
            
            data_sources = {'cgm': box_cgm}
    else:
        data_sources = {'all': refine_box}
        if filter_type and filter_value:
            if filter_type == 'temperature':
                data_sources['all'] = data_sources['all'].cut_region([f"(obj['gas', 'temperature'] < {filter_value})"])
            elif filter_type == 'density':
                data_sources['all'] = data_sources['all'].cut_region([f"(obj['gas', 'density'] > {filter_value})"])
            else:
                raise ValueError("Unsupported filter type. Supported types: 'temperature', 'density'.")

    # Create HDF5 file for saving mass maps
    save_path = prefix + f'FRBs/' 
    os.makedirs(save_path, exist_ok=True)  # Ensure the directory exists
    f = h5py.File(save_path + halo_name + '_emission_maps' + save_suffix + '.hdf5', 'a')
    # Check if the group already exists
    redshift_group_name = 'z=%.1f' % ds.get_parameter('CosmologyCurrentRedshift', 1)
    if redshift_group_name not in f:
        grp = f.create_group(redshift_group_name)
        grp.attrs.create("image_extent_kpc", ds.refine_width)
        grp.attrs.create("redshift", ds.get_parameter('CosmologyCurrentRedshift'))
        grp.attrs.create("halo_name", halo_name)
        grp.attrs.create("bin_size_kpc", round_bin_size_kpc)
        grp.attrs.create("number_of_bins", res)
    else:
        grp = f[redshift_group_name]  # Open the existing group



    for region, data_source in data_sources.items():
        for ion in ions:
            print(f"Calculating Mass in each pixel for {ion} in {region} region")
            # Extract necessary fields from refine_box
            half_width = (width / 2).v  # Extract value from YTQuantity
            data_source = data_source.cut_region([
                f"(abs(obj['gas', 'x_disk']) < {half_width})",
                f"(abs(obj['gas', 'y_disk']) < {half_width})",
                f"(abs(obj['gas', 'z_disk']) < {half_width})"
            ])

            x_disk = data_source[('gas', 'x_disk')].in_units('kpc')  # X positions
            y_disk = data_source[('gas', 'y_disk')].in_units('kpc')  # Y positions
            z_disk = data_source[('gas', 'z_disk')].in_units('kpc')  # Z positions
            ion_mass = data_source[('gas', ions_mass_dict[ion])].in_units('Msun')  # Ion mass
            ion_emission = data_source[('gas', 'Emission_' + ions_dict[ion])]

            
            # Create empty mass FRBs
            mass_frb_edge = np.zeros((res, res)) * Msun
            mass_frb_face = np.zeros((res, res)) * Msun

            
            if args.unit_system == 'ALT':
                emission_frb_edge = np.zeros((res, res)) * (erg / (arcsec**2 * cm**3 * s))
                emission_frb_face = np.zeros((res, res)) * (erg / (arcsec**2 * cm**3 * s))
            elif args.unit_system == 'default': 
                #define_unit("photon", 1.0)  # Define 'photon' as a dimensionless unit
                emission_frb_edge = np.zeros((res, res)) * (1.0 * sr**-1 * cm**-3 * s**-1)
                emission_frb_face = np.zeros((res, res)) * (1.0 * sr**-1 * cm**-3 * s**-1)


            # Ensure width is evenly divisible by min_dx
            width_min = np.floor((-width / 2) / min_res) * min_res  # Snap left edge to nearest cell boundary
            width_max = np.ceil((width / 2) / min_res) * min_res  # Snap right edge to nearest cell boundary

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

            for yi, zi, mass,emission in zip(y_indices[valid_edge], z_indices[valid_edge], ion_mass[valid_edge].in_units('Msun'), ion_emission[valid_edge]):
                mass_frb_edge[zi, yi] += mass  # Edge-on uses Y (horizontal) & Z (vertical)
                emission_frb_edge[zi, yi] += emission  # Edge-on uses Y (horizontal) & Z (vertical)

            for xi, yi, mass,emission in zip(x_indices[valid_face], face_y_indices[valid_face], ion_mass[valid_face].in_units('Msun'), ion_emission[valid_face]):
                mass_frb_face[xi, yi] += mass  # Face-on uses Y (horizontal) & X (vertical)
                emission_frb_face[xi, yi] += emission  # Face-on uses Y (horizontal) & X (vertical)

            emission_frb_edge = (emission_frb_edge*bin_size_cm).to(1.0 * sr**-1 * cm**-2 * s**-1)
            emission_frb_face = (emission_frb_face*bin_size_cm).to(1.0 * sr**-1 * cm**-2 * s**-1)

            
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

            emission_dset_edge = grp.create_dataset(f"{ion}_emission_manualpixel_edge_{region}", data=emission_frb_edge)
            emission_dset_face = grp.create_dataset(f"{ion}_emission_manualpixel_face_{region}", data=emission_frb_face)

            print(f"Saved mass FRBs for {ion} in HDF5")

           # make a plot of the mass and emission maps
            # fig, axes = plt.subplots(1, 2, figsize=(12, 5))

            # extent = [-width/2, width/2, -width/2, width/2]  # Define extent in kpc

            # # Edge-on Mass Map
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

            # ###Plot emission Distribution Maps** ###
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
######################################################################################################
def load_and_calculate(snap, ions,scale_factor=None, unit_system='default', filter_type=None, filter_value=None, resolution=2):

    '''Loads the simulation snapshot and makes the requested plots, with optional filtering.'''

    # Load simulation output
    snap_name = foggie_dir + 'halo_00' + args.halo + '/' + args.run + '/' + snap + '/' + snap
    
    ds, refine_box = foggie_load(snap_name, trackname, do_filter_particles=True, halo_c_v_name=halo_c_v_name, disk_relative=True, correct_bulk_velocity=True)#, smooth_AM_name=smooth_AM_name)
    zsnap = ds.get_parameter('CosmologyCurrentRedshift')
    add_ion_fields(ds,ions)

    
    # Generate emission maps based on the plot type
    if ('emission_map' in args.plot):
        if ('vbins' not in args.plot):
            emission_map(ds, refine_box, snap, ions, unit_system=unit_system, filter_type=filter_type, filter_value=filter_value,resolution=resolution)
        else:
            emission_map_vbins(ds, snap, ions, unit_system=unit_system, filter_type=filter_type, filter_value=filter_value)
    if ('emission_FRB' in args.plot):
        
        #make_FRB(ds, refine_box, snap, ions, unit_system=unit_system, filter_type=filter_type, filter_value=filter_value, resolution=resolution)
        #compute_mass_in_emission_pixels(ds, refine_box, snap, ions, resolution=resolution)
        make_column_density_FRB(ds, refine_box, snap, ions,scaling = scaling, scale_factor=scale_factor, filter_type=filter_type, filter_value=filter_value, resolution=resolution)
        
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

    #set the clump file directory
    disk_file = output_dir + '/Disk/test_Disk.h5'
    shell_path = output_dir + '/Disk/'
    
    # Set directory for output location, making it if necessary
    box_name = args.fov if args.fov is not None else 'refine_box'
    prefix = output_dir + '/res_' + args.resolution + '/' + 'box_' + box_name + '/'
    if not (os.path.exists(prefix)): os.system('mkdir -p ' + prefix)
    table_loc = prefix + 'Tables/'

    print('foggie_dir: ', foggie_dir)
    catalog_dir = code_path + 'halo_infos/00' + args.halo + '/' + args.run + '/'
    halo_c_v_name = catalog_dir + 'halo_c_v'
    #smooth_AM_name = catalog_dir + 'AM_direction_smoothed'

    # right now using the test tables that vida made 
    cloudy_path = "/Users/vidasaeedzadeh/Documents/02-Projects/02-FOGGIE/Cloudy-runs/outputs/test-z0/TEST_z0_HM12_sh_run%i.dat"
    #code_path + "emission/cloudy_z0_selfshield/sh_z0_HM12_run%i.dat"

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
    # Function to register emission fields with unit options
    def register_emission_field_with_unit(field_name, function, emission_units, unit_system,scale_factor,scaling):
        yt.add_field(
            ('gas', field_name),
            units=emission_units if unit_system == 'default' else emission_units_ALT,
            function=lambda field, data: function(field, data,scale_factor=scale_factor, unit_system=unit_system, scaling = scaling),
            take_log=True,
            force_override=True,
            sampling_type='cell',
        )
    
    
    unit_system = args.unit_system
    scale_factor = float(args.scale_factor)
    scaling = args.scaling
    ############################
    # H-Alpha
    hden_pts, T_pts, table_HA = make_Cloudy_table(2,cloudy_path)
    hden_pts, T_pts = np.meshgrid(hden_pts, T_pts)
    pts = np.array((hden_pts.ravel(), T_pts.ravel())).T
    
    sr_HA = table_HA.T.ravel()
    bl_HA = interpolate.LinearNDInterpolator(pts, sr_HA)
    register_emission_field_with_unit('Emission_HAlpha', Emission_HAlpha, emission_units, unit_system,scale_factor,scaling)
    
    ############################
    # Ly-Alpha
    hden_pts, T_pts, table_LA = make_Cloudy_table(1,cloudy_path)
    sr_LA = table_LA.T.ravel()
    bl_LA = interpolate.LinearNDInterpolator(pts, sr_LA)
    register_emission_field_with_unit('Emission_LyAlpha', Emission_LyAlpha, emission_units, unit_system,scale_factor,scaling)
    ############################
    # CII 1335
    hden_pts, T_pts, table_CII_1335 = make_Cloudy_table(10,cloudy_path)
    sr_CII_1335 = table_CII_1335.T.ravel()
    bl_CII_1335 = interpolate.LinearNDInterpolator(pts, sr_CII_1335)
    register_emission_field_with_unit('Emission_CII_1335', Emission_CII_1335, emission_units, unit_system,scale_factor,scaling)
    
    ############################
    # CIII 977
    hden_pts, T_pts, table_CIII_977 = make_Cloudy_table(7,cloudy_path)
    sr_CIII_977 = table_CIII_977.T.ravel()
    bl_CIII_977 = interpolate.LinearNDInterpolator(pts, sr_CIII_977)
    register_emission_field_with_unit('Emission_CIII_977', Emission_CIII_977, emission_units, unit_system,scale_factor,scaling)

    ############################
    # CIII 1910
    hden_pts, T_pts, table_CIII_1910 = make_Cloudy_table(9,cloudy_path)
    sr_CIII_1910 = table_CIII_1910.T.ravel()
    bl_CIII_1910 = interpolate.LinearNDInterpolator(pts, sr_CIII_1910)
    register_emission_field_with_unit('Emission_CIII_1910', Emission_CIII_1910, emission_units, unit_system,scale_factor,scaling)

    ############################
    # CIV 1548
    hden_pts, T_pts, table_CIV_1 = make_Cloudy_table(3,cloudy_path)
    sr_CIV_1 = table_CIV_1.T.ravel()
    bl_CIV_1 = interpolate.LinearNDInterpolator(pts, sr_CIV_1)
    register_emission_field_with_unit('Emission_CIV_1548', Emission_CIV_1548, emission_units, unit_system,scale_factor,scaling)
    
    ############################
    # O VI (1032 and 1037 combined)
    hden_pts, T_pts, table_OVI_1 = make_Cloudy_table(5,cloudy_path)
    hden_pts, T_pts, table_OVI_2 = make_Cloudy_table(6,cloudy_path)
    sr_OVI_1 = table_OVI_1.T.ravel()
    sr_OVI_2 = table_OVI_2.T.ravel()
    bl_OVI_1 = interpolate.LinearNDInterpolator(pts, sr_OVI_1)
    bl_OVI_2 = interpolate.LinearNDInterpolator(pts, sr_OVI_2)
    register_emission_field_with_unit('Emission_OVI', Emission_OVI, emission_units, unit_system,scale_factor,scaling)
    ############################
    # SiII 1814
    cloudy_path_thin = code_path + "emission/cloudy_z0_HM05/bertone_run%i.dat"
    hden_pts, T_pts, table_SiII_1814 = make_Cloudy_table_thin(11,cloudy_path)
    hden_pts, T_pts = np.meshgrid(hden_pts, T_pts)
    pts = np.array((hden_pts.ravel(), T_pts.ravel())).T
    sr_SiII_1814 = table_SiII_1814.T.ravel()
    bl_SiII_1814 = interpolate.LinearNDInterpolator(pts, sr_SiII_1814)
    register_emission_field_with_unit('Emission_SiII_1814', Emission_SiII_1814, emission_units, unit_system,scale_factor,scaling)
    ############################
    # SiIII 1207
    cloudy_path_thin = code_path + "emission/cloudy_z0_HM05/bertone_run%i.dat"
    hden_pts, T_pts, table_SiIII_1207 = make_Cloudy_table_thin(12,cloudy_path)
    hden_pts, T_pts = np.meshgrid(hden_pts, T_pts)
    pts = np.array((hden_pts.ravel(), T_pts.ravel())).T
    sr_SiIII_1207 = table_SiIII_1207.T.ravel()
    bl_SiIII_1207 = interpolate.LinearNDInterpolator(pts, sr_SiIII_1207)
    register_emission_field_with_unit('Emission_SiIII_1207', Emission_SiIII_1207, emission_units, unit_system,scale_factor,scaling)
    ############################
    # SiIV 1394
    cloudy_path_thin = code_path + "emission/cloudy_z0_HM05/bertone_run%i.dat"
    hden_pts, T_pts, table_SiIV_1394 = make_Cloudy_table_thin(14,cloudy_path)
    hden_pts, T_pts = np.meshgrid(hden_pts, T_pts)
    pts = np.array((hden_pts.ravel(), T_pts.ravel())).T
    sr_SiIV_1394 = table_SiIV_1394.T.ravel()
    bl_SiIV_1394 = interpolate.LinearNDInterpolator(pts, sr_SiIV_1394)
    register_emission_field_with_unit('Emission_SiIV_1394', Emission_SiIV_1394, emission_units, unit_system,scale_factor,scaling)
    ############################
    # MgII 2796
    cloudy_path_thin = code_path + "emission/cloudy_z0_HM05/bertone_run%i.dat"
    hden_pts, T_pts, table_MgII_2796 = make_Cloudy_table_thin(16,cloudy_path)
    hden_pts, T_pts = np.meshgrid(hden_pts, T_pts)
    pts = np.array((hden_pts.ravel(), T_pts.ravel())).T
    sr_MgII_2796 = table_MgII_2796.T.ravel()
    bl_MgII_2796 = interpolate.LinearNDInterpolator(pts, sr_MgII_2796)
    register_emission_field_with_unit('Emission_MgII_2796', Emission_MgII_2796, emission_units, unit_system,scale_factor,scaling)
    ############################

    ions_dict = {'Lyalpha':'LyAlpha', 'HI':'HAlpha', 'CII': 'CII_1335','CIII':'CIII_1910', 
                 'CIV':'CIV_1548','OVI':'OVI','SiII':'SiII_1814','SiIII':'SiIII_1207','SiIV':'SiIV_1394','MgII':'MgII_2796'}
    ions_density_dict = {'Lyalpha':'LyAlpha', 'HI':'H_p0_density', 'CII':'C_p1_density', 'CIII':'C_p2_density',
                        'CIV':'C_p3_density','OVI':'O_p5_density','SiII':'Si_p1_density','SiIII':'Si_p2_density','SiIV':'Si_p3_density','MgII':'Mg_p1_density'}
    ions_number_density_dict = {'Lyalpha':'LyAlpha', 'HI':'H_p0_number_density', 'CII':'C_p1_number_density', 'CIII':'C_p2_number_density',
                                'CIV':'C_p3_number_density','OVI':'O_p5_number_density','SiII':'Si_p1_number_density','SiIII':'Si_p2_number_density',
                                 'SiIV':'Si_p3_number_density','MgII':'Mg_p1_number_density'}
    ions_mass_dict = {'Lyalpha':'LyAlpha', 'HI':'H_p0_mass', 'CII':'C_p1_mass', 'CIII':'C_p2_mass',
                      'CIV':'C_p3_mass','OVI':'O_p5_mass','SiII':'Si_p1_mass','SiIII':'Si_p2_mass','SiIV':'Si_p3_mass','MgII':'Mg_p1_mass'}
    
    label_dict = {'Lyalpha':r'Ly-$\alpha$', 'HI':r'H$\alpha$', 'CII':'C II','CIII':'C III',
                'CIV':'C IV','OVI':'O VI','SiII':'Si II','SiIII':'Si III','SiIV':'Si IV','MgII':'Mg II'}
    
    trident_dict = { 'HI':'H I', 'CII':'C II','CIII':'C III',
                'CIV':'C IV','OVI':'O VI','SiII':'Si II','SiIII':'Si III','SiIV':'Si IV','MgII':'Mg II'}

    if args.fov is not None:
        if unit_system  == 'default':
            if (args.fov == None) and (args.halo == '2392'):
                zlim_dict = {'Lyalpha':[1e-1,1e7], 'HI':[1e-1,1e6], 'CII':[1e-7,1e4], 'CIII':[1e-1,1e5],
                        'CIV':[1e-2,1e4], 'OVI':[1e-2,1e4],'SiII':[1e-6,1e5],'SiIII':[1e-6,1e5],'SiIV':[1e-6,1e5],'MgII':[1e-6,1e5]}
            else:
                zlim_dict = {'Lyalpha':[1e-1,1e7], 'HI':[1e-1,1e6], 'CII':[1e-3,1e4], 'CIII':[1e-1,1e4],
                        'CIV':[1e-1,1e5], 'OVI':[1e0,1e4],'SiII':[1e-1,1e5],'SiIII':[1e-2,1e5],'SiIV':[1e-2,1e5],'MgII':[1e-1,1e5]}
        elif unit_system == 'ALT':
            zlim_dict = {'Lyalpha':[1e-22,1e-16], 'HI':[1e-22,1e-16], 'CII':[1e-23,1e-16], 'CIII':[1e-23,1e-16],
                        'CIV':[1e-23,1e-16], 'OVI':[1e-22,1e-17],'SiII':[1e-23,1e-16],'SiIII':[1e-23,1e-16],'SiIV':[1e-23,1e-16],'MgII':[1e-23,1e-16]}
    else:
        if unit_system  == 'default':
            if (args.fov == None) and (args.halo == '2392'):
                zlim_dict = {'Lyalpha':[1e-1,1e7], 'HI':[1e-1,1e6], 'CII':[1e-7,1e4], 'CIII':[1e-1,1e5],
                        'CIV':[1e-2,1e4], 'OVI':[1e-2,1e4],'SiII':[1e-6,1e5],'SiIII':[1e-6,1e5],'SiIV':[1e-6,1e5],'MgII':[1e-6,1e5]}
            else:
                zlim_dict = {'Lyalpha':[1e-1,1e7], 'HI':[1e-1,1e6], 'CII':[1e-7,1e4], 'CIII':[1e-4,1e3],
                        'CIV':[1e-2,1e4], 'OVI':[1e-2,1e4],'SiII':[1e-5,1e5],'SiIII':[1e-5,1e5],'SiIV':[1e-5,1e5],'MgII':[1e-5,1e5]} 
        elif unit_system == 'ALT':
            zlim_dict = {'Lyalpha':[1e-22,1e-16], 'HI':[1e-22,1e-16], 'CII':[1e-23,1e-16], 'CIII':[1e-23,1e-16],
                        'CIV':[1e-23,1e-16], 'OVI':[1e-22,1e-17],'SiII':[1e-23,1e-16],'SiIII':[1e-23,1e-16],'SiIV':[1e-23,1e-16],'MgII':[1e-23,1e-16]}
        
    if args.Juniper_limit:
        if args.unit_system == 'default':
            flux_threshold_dict = {'CII':1588.24, 'CIII':9000.00,'CIV':3857.14, 'OVI':1800.00} #photons/s/cm^2/sr
        elif args.unit_system == 'ALT':
            flux_threshold_dict = {'OVI': 4e-19} 
        
    elif args.Aspera_limit:
        flux_threshold_dict = {'OVI': 3e-19} #ergs/s/cm^2/arcsec^2
    elif args.Magpie_limit:
        if args.unit_system == 'default':
            flux_threshold_dict = {'CIII': 675,'CIV': 650,'OVI': 270, 'MgII':675} #photons/s/cm^2/sr
        elif args.unit_system == 'ALT':
            flux_threshold_dict = {'OVI': 3e-19, } #ergs/s/cm^2/arcsec^2
        
    elif args.Muse_limit:
        flux_threshold_dict = {'OVI': 3e-19} #ergs/s/cm^2/arcsec^2
    elif args.HWO_limit:
        flux_threshold_dict = {'OVI': 1.5e-20} #ergs/s/cm^2/arcsec^2
        #{'OVI': 200} #photons/s/cm^2/sr
    else:
        flux_threshold_dict = {}
        



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

    resolution = args.resolution
    resolution = int(resolution)

    shell_count = int(args.shell_count)


    # Build outputs list
    outs = make_output_list(args.output, output_step=args.output_step)

    target_dir = 'ions'
    if (args.nproc==1):
        for snap in outs:
            load_and_calculate(snap, ions,scale_factor=scale_factor, unit_system=unit_system, filter_type=filter_type, filter_value=None, resolution=resolution)
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

    




