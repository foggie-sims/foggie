'''
Filename: emission-analysis.py
Author: Vida
Date created: 1-15-25
Date last modified: 2-20-25

This file contains analysis that uses FRBs that are made using emission_maps_dynamic.py

Currently the code results in following plots for both face on and edge on FRBs:
- scatter plot of number density vs emissivity for each ion (to check their relation)
- median number density vs emissivity 
- emission vs tempreture plot for each ion
- Histogram of surface brighness for each ion
- Histogram of mass for each ion
- Mass vs Surface brighness plot
- Cumulative mass vs surface brightness for each ion
- number density projections for each ion
- rotation curve of ions all in one plot
- number density vs radius (to reproduce and compare to Fig 18 of Lehner et al.2020)


'''

from __future__ import print_function

import numpy as np
import yt
import unyt
from yt.units import *
from yt import YTArray
import argparse
import os
import glob
import sys
from astropy.table import Table
from astropy.io import ascii
import matplotlib.pyplot as plt
import random
from scipy import interpolate
#from photutils import detect_threshold, detect_sources, source_properties, EllipticalAperture
from scipy.interpolate import InterpolatedUnivariateSpline as IUS
from scipy.optimize import minimize
import trident
import ast
#import emcee
import numpy.random as rn
from multiprocessing import Pool
import matplotlib.patches as patches
from matplotlib.colors import ListedColormap
from yt.units import kpc, cm

# These imports are FOGGIE-specific files
from foggie.utils.consistency import *
from foggie.utils.get_refine_box import get_refine_box
from foggie.utils.get_halo_center import get_halo_center
from foggie.utils.get_proper_box_size import get_proper_box_size
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
import h5py
import matplotlib.cm as mtcm
from matplotlib.colors import LogNorm

from foggie.clumps.clump_finder.utils_diskproject import load_disk 
from collections import defaultdict
import pandas as pd
import math
import matplotlib.colors as colors
from mpl_toolkits.axes_grid1 import make_axes_locatable


from foggie.clumps.clump_finder.utils_clump_finder import *
from foggie.clumps.clump_finder.clump_finder_argparser import *
from foggie.clumps.clump_finder.fill_topology import *
from foggie.clumps.clump_finder.clump_load import *
from foggie.clumps.clump_finder.clump_finder import clump_finder

from foggie.emission.emission_maps_dynamic import * 


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
    parser.set_defaults(output='RD0042')

    parser.add_argument('--output_step', metavar='output_step', type=int, action='store', \
                        help='If you want to do every Nth output, this specifies N. Default: 1 (every output in specified range)')
    parser.set_defaults(output_step=1)

    parser.add_argument('--system', metavar='system', type=str, action='store', \
                        help='Which system are you on? Default is cassiopeia')
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
    
    parser.add_argument('--save_suffix', metavar='save_suffix', type=str, action='store', \
                        help='Do you want to append a string onto the names of the saved files? Default is no.')
    parser.set_defaults(save_suffix="")

    parser.add_argument('--file_suffix', metavar='file_suffix', type=str, action='store', \
                        help='If plotting from saved surface brightness files, use this to pass the file name suffix.')
    parser.set_defaults(file_suffix="")

    parser.add_argument('--resolution', metavar='resolution', type=str, action='store', \
                        help='How many times larger than simulation resolution? Default is 2')
    parser.set_defaults(resolution=1)

    parser.add_argument('--bin_sizes', metavar='bin_sizes', type=str, action='store', \
                        help='What is the bin size of the FRB? Default is nreff resolution = 0.27 kpc.')
    parser.set_defaults(bin_sizes=[0.27])

    parser.add_argument('--filter_type', metavar='filter_type', type=str, action='store', \
                        help='What filter type? Default is None (Options are: inflow_outflow or disk_cgm )')
    parser.set_defaults(filter_type=None)

    parser.add_argument('--unit_system', metavar='unit_system', type=str, action='store', \
                        help='What unit system? Default is default (s**-1 * cm**-3 * steradian**-1)')
    parser.set_defaults(unit_system='default')
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

#######################################################
# Add Trident ion fields
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

#####################################################################################################   
def numdensity_emissivity_scatter(ds,ions_number_density_dict,ions_dict):
    ad = ds.all_data()
    save_path = prefix + f'emission_numdensity_plots/'
    os.makedirs(save_path, exist_ok=True)

    for ion, numdensity in ions_number_density_dict.items():
        if ion == 'OVI':
            emission_field = ad[('gas', 'Emission_' + ions_dict[ion])]
            emission_values = emission_field.v
            valid_indices = emission_values > 1e-40
            emission_values = np.log10(emission_values[valid_indices])
            numdensity_field = ad[('gas', numdensity)]
            numdensity_values = numdensity_field.v
            numdensity_values = np.log10(numdensity_values[valid_indices])
            
            plt.scatter(emission_values, numdensity_values, marker = 'o', s = 1, color = 'skyblue')
            plt.xlabel('OVI Emissivity [$photon/s/cm^3/sr$]')
            plt.ylabel('OVI Number Density [$1/cm^2$]')
            plt.savefig(save_path + f'emission_numdensity_{ion}.png')

def numdensity_emissivity(ds, ions_number_density_dict, ions_dict):
    ad = ds.all_data()
    save_path = prefix + f'emission_numdensity_plots/'
    os.makedirs(save_path, exist_ok=True)

    for ion, numdensity in ions_number_density_dict.items():
        if ion == 'OVI':
            emission_field = ad[('gas', 'Emission_' + ions_dict[ion])]
            emission_values = emission_field.v
            numdensity_field = ad[('gas', numdensity)]
            numdensity_values = numdensity_field.v
            valid_indices = (emission_values > 1e-40) & (numdensity_values > 1e-40)
            emission_values = np.log10(emission_values[valid_indices])
            
            numdensity_values = np.log10(numdensity_values[valid_indices])
            
            # Bin data
            bin_edges = np.linspace(emission_values.min(), emission_values.max(), 40)  # 30 bins
            bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
            median_values = []
            lower_percentile = []
            upper_percentile = []

            for i in range(len(bin_edges) - 1):
                in_bin = (emission_values >= bin_edges[i]) & (emission_values < bin_edges[i+1])
                if np.any(in_bin):
                    median_values.append(np.median(numdensity_values[in_bin]))
                    lower_percentile.append(np.percentile(numdensity_values[in_bin], 16))
                    upper_percentile.append(np.percentile(numdensity_values[in_bin], 84))
                else:
                    median_values.append(np.nan)
                    lower_percentile.append(np.nan)
                    upper_percentile.append(np.nan)
            
            # Convert to arrays for plotting
            median_values = np.array(median_values)
            lower_percentile = np.array(lower_percentile)
            upper_percentile = np.array(upper_percentile)

            # Plot median line and shaded region
            plt.figure(figsize=(8, 6))
            #plt.scatter(emission_values, numdensity_values, marker = 'o', s = 0.5, alpha=0.3,color = 'skyblue')
            plt.plot(bin_centers, median_values, color='blue', label='Median')
            plt.fill_between(bin_centers, lower_percentile, upper_percentile, color='skyblue', alpha=0.5, label='16th-84th Percentile')
            plt.xlabel('Log(OVI Emissivity) [$photon/s/cm^3/sr$]')
            plt.ylabel('Log(OVI Number Density) [$1/cm^2$]')
            plt.title('Tempest - z = 0')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.savefig(save_path + f'median_emission_numdensity_{ion}.png')
            plt.close()

def emission_temp(ds,ions):
    # Extract the fields: 'Emission_{ion}' and 'temperature'
    ad = ds.all_data()

    save_path = prefix + f'emission_temp_plots/'
    os.makedirs(save_path, exist_ok=True)  # Ensure the directory exists
    cmap = mtcm.get_cmap('Set1', 10)  # Colormap for plots
    ion_label_list = ['H I', 'C II (1335)','C III (1910)', 'C IV (1548)', 'O VI (1032 & 1038)']
    # Create a single figure for all histograms
    plt.figure(figsize=(6.4, 4.8))

    for i, (ion, ion_label) in enumerate(zip(ions, ion_label_list)):
        print(ion)

        color = cmap(i)

        # Extract temperature and emission fields
        temperature_field = ad[('gas', 'temperature')]
        emission_field = ad[('gas', 'Emission_' + ions_dict[ion])]

        # Convert fields to numpy arrays for plotting
        temperature_values = temperature_field.v
        emission_values = emission_field.v

        # Filter out invalid data
        valid_indices = (emission_values > 1e-40) & (temperature_values > 1e-40)
        emission_values = emission_values[valid_indices]
        temperature_values = np.log10(temperature_values[valid_indices])

        # Create bins for temperature
        #bins = np.logspace(np.log10(temperature_values.min()), np.log10(temperature_values.max()), 64)
        bins = np.linspace(temperature_values.min(), temperature_values.max(), 64)
        bin_centers = 0.5 * (bins[:-1] + bins[1:])

        # Compute the total emissivity for each temperature bin
        bin_indices = np.digitize(temperature_values, bins)
        emissivity_per_bin = [emission_values[bin_indices == j].sum() for j in range(1, len(bins))]

        # Plot 1D histogram using step plot
        plt.step(bin_centers, emissivity_per_bin, where='mid', color=color, label=f'{ion_label} ')

    # Set up plot scales and labels
    #plt.xscale('log')
    plt.yscale('log')
    plt.ylim(1e-22, 1e-9)
    plt.xlim(3, 8)
    plt.xlabel('log(Temperature [K])',fontsize=16)
    plt.ylabel('Emissivity [$photon/s/cm^{3}/sr$]',fontsize=16)
    # Adjust tick size
    plt.tick_params(axis='both', which='major', labelsize=14)  # Major ticks
    plt.tick_params(axis='both', which='minor', labelsize=12)  # Minor ticks
    plt.legend()

    # Save and show the single plot
    plt.tight_layout()
    plt.grid(color='gray', linestyle='--', linewidth=0.1)
    plt.savefig(save_path + f'emission_temp_combined_OVI_linear.png')

    plt.close()

def sb_mass_hist(ds,ions,orientations,bin_sizes,flux_threshold_dict):
    
    cmap = mtcm.get_cmap('Set2', 3) # Colormap for plots

    # Extract all dataset names dynamically from one of the HDF5 files
    bin_size = bin_sizes[0]
    save_path = prefix + f'FRBs/res_{bin_size}'
    halo_name = halo_dict[str(args.halo)]
    file_path =  save_path + '/' + halo_name + '_emission_maps' + '.hdf5'
    
    # Open the HDF5 file to get the dataset names
    with h5py.File(file_path, 'r') as example_file:
        set_of_datasets = list(next(iter(example_file.values())).keys())  # Extract dataset names

    # Iterate over datasets
    for dataset_name in set_of_datasets:
        plt.figure(figsize=(8, 6))  # Initialize the figure for each dataset

        for i, bin_size_kpc in enumerate(bin_sizes):
            color = cmap(i)  # Choose color from colormap based on index

            # Open the HDF5 file and iterate through redshifts and datasets
            with h5py.File(file_path, 'r') as f:
                for redshift_group in f.keys():  # Iterate through redshift groups
                    redshift_data = f[redshift_group]

                    # Extract FOV, resolution, and bin size from the attributes
                    fov_kpc = redshift_data.attrs.get("image_extent_kpc", "FOV Unknown")
                    resolution = redshift_data.attrs.get("number_of_bins", "Unknown")

                    if fov_kpc != "FOV Unknown":
                        fov_kpc = round(fov_kpc, 1)  # Round FOV to one decimal place

                    # Get the dataset for the current bin size
                    if dataset_name in redshift_data.keys():
                        # in dataset_name include mass in the name then:
                        if 'mass' in dataset_name.lower(): #method ensures the checks are case-insensitive.
                            
                            mass_data = np.array(redshift_data[dataset_name])

                            # Flatten the emission data to calculate flux per pixel
                            mass_per_pixel = mass_data.flatten()

                            # Define log-spaced bins
                            mass_min = mass_per_pixel[mass_per_pixel > 1e-40].min()  # Avoid bins starting at 0
                            mass_max = mass_per_pixel.max()
                            bins = np.logspace(np.log10(mass_min), np.log10(mass_max), 50)  # 50 log-spaced bins

                            # Add histogram for this bin size
                            plt.hist(mass_per_pixel, bins=bins, color=color, alpha=0.6,
                                    label=f'Bin Size: {bin_size_kpc} kpc')
                            
                            # Finalize the plot for this dataset
                            plt.yscale('log')  # Use logarithmic scale for better visibility
                            plt.xscale('log')  # Log scale for x-axis
                            plt.xlabel('Mass [$M_{\odot}$]')
                            plt.ylabel('Number of Pixels')
                            plt.title(f'{dataset_name} - Histogram for Different Bin Sizes\nFOV: {fov_kpc} kpc')
                            plt.legend()
                            plt.grid(True, alpha=0.3)
                            plt.tight_layout()

                        #if it include emission then
                        elif 'emission' in dataset_name.lower():

                            emission_data = np.array(redshift_data[dataset_name])

                            # Flatten the emission data to calculate flux per pixel
                            flux_per_pixel = emission_data.flatten()

                            # Define log-spaced bins
                            flux_min = flux_per_pixel[flux_per_pixel > 1e-40].min()  # Avoid bins starting at 0
                            flux_max = flux_per_pixel.max()
                            bins = np.logspace(np.log10(flux_min), np.log10(flux_max), 50)  # 50 log-spaced bins

                            # Add histogram for this bin size
                            plt.hist(flux_per_pixel, bins=bins, color=color, alpha=0.6,
                                    label=f'Bin Size: {bin_size_kpc} kpc')
                            
                            # Finalize the plot for this dataset
                            plt.yscale('log')  # Use logarithmic scale for better visibility
                            plt.xscale('log')  # Log scale for x-axis
                            plt.xlabel('Surface Brightness [$photons/s/cm^2/sr$]') 
                            plt.ylabel('Number of Pixels')
                            plt.title(f'{dataset_name} - Histogram for Different Bin Sizes\nFOV: {fov_kpc} kpc')
                            plt.legend()
                            plt.grid(True, alpha=0.3)
                            plt.tight_layout()
        
        

        # Save the combined plot in a directory for all resolutions
        combined_dir = os.path.join(prefix, "histograms", redshift_group)
        os.makedirs(combined_dir, exist_ok=True)
        plt.savefig(os.path.join(combined_dir, f"{dataset_name}_histogram.png"))
        
        plt.close()

def area_frac(ds,ions,orientations,cmap,flux_threshold_dict,regions):
    results = []  # List to store results
    for ion in ions:

        flux_lim = flux_threshold_dict[ion]
        
        print('ion',ion)
        for orientation in orientations:
            print('orientation',orientation)
            plt.figure(figsize=(10, 7))
            
            # Dataset names for mass and emission
            emission_dataset_name = f"z=0.0/{ion}_emission_{orientation}_all"

            for i, bin_size_kpc in enumerate(bin_sizes):
                color = cmap(i)  # Select color from colormap
                save_path = prefix + f'FRBs/'
                halo_name = halo_dict[str(args.halo)]
                file_path = save_path + halo_name + '_emission_maps' + '.hdf5'

                cell_area = float(bin_size_kpc) ** 2  # Pixel area in kpc^2

                # Open the HDF5 file and extract mass and emission data
                with h5py.File(file_path, 'r') as f:
                    emission_data = np.array(f[emission_dataset_name]).flatten()
                    
                    
                    # Filter valid values
                    valid_indices = (emission_data > 1e-40) 
                    emission_data = emission_data[valid_indices]
                    all_cell_num = len(emission_data)
                    all_area = all_cell_num*cell_area

                    above_flux_idx = emission_data > flux_lim
                    cell_num_above_limit = len(emission_data[above_flux_idx])
                    above_lim_area = cell_num_above_limit* cell_area

                    area_frac = len(emission_data[above_flux_idx])/len(emission_data) #above_lim_area/all_area
                    area_frac_pers = round(area_frac *100,2)

                    print('full area',len(emission_data))
                    print('above lim area',len(emission_data[above_flux_idx]))
                    print('frac',area_frac_pers)

                    # Store results in list
                    results.append([ion, orientation, bin_size_kpc, flux_lim, area_frac_pers])

    # Convert results into a Pandas DataFrame
    df = pd.DataFrame(results, columns=['Ion', 'Orientation', 'Bin Size (kpc)', 'Flux Limit', 'Area Fraction'])

    # Save the table as a PNG
    fig, ax = plt.subplots(figsize=(10, len(df) * 0.4))  # Adjust figure size dynamically
    ax.axis("tight")
    ax.axis("off")
    table = ax.table(cellText=df.values, colLabels=df.columns, cellLoc='center', loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.auto_set_column_width([i for i in range(len(df.columns))])  # Auto adjust column width

    # Save the table image
    table_path = save_path + "ion_area_fractions.png"
    plt.savefig(table_path, bbox_inches="tight", dpi=300)
    print(f"Table saved as: {table_path}")

    return df  # Return the DataFrame if needed

def mass_sb_backup(ds,ions,orientations,cmap,flux_threshold_dict,regions):
    
    num_ions = len(ions)
    
    cols =  __builtins__.min(2, num_ions)  # Limit to max 3 columns per row
    rows = math.ceil(num_ions / cols)  # Compute required rows
    

    for orientation in orientations:
        print('orientation',orientation)
        fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows), squeeze=False)
    
        for index, ion in enumerate(ions):
            row, col = divmod(index, cols)
            ax = axes[row, col]  # Select corresponding subplot
            flux_lim = flux_threshold_dict[ion]
            
            for i, bin_size_kpc in enumerate(bin_sizes):
                color = cmap(i)  # Select color from colormap
                save_path = prefix + f'FRBs/'
                halo_name = halo_dict[str(args.halo)]
                file_path = save_path + halo_name + '_emission_maps' + '.hdf5'

                mass_dataset_name = f"z=0.0/{ion}_mass_{orientation}_all"
                emission_dataset_name = f"z=0.0/{ion}_emission_{orientation}_all"

                with h5py.File(file_path, 'r') as f:
                    mass_data = np.array(f[mass_dataset_name]).flatten()
                    emission_data = np.array(f[emission_dataset_name]).flatten()
                    
                    valid_indices = (emission_data > 1e-40) & (mass_data > 0)
                    emission_data = emission_data[valid_indices]
                    mass_data = mass_data[valid_indices]
                    total_mass = mass_data.sum()

                    above_flux_lim_indices = emission_data > flux_lim
                    mass_above_flux_lim = mass_data[above_flux_lim_indices].sum()
                    mass_fraction = mass_above_flux_lim / total_mass if total_mass > 0 else 0
                    
                    flux_min = np.log10(emission_data).min()
                    flux_max = np.log10(emission_data).max()
                    bins = np.linspace(flux_min, flux_max, 64)
                    bin_indices = np.digitize(np.log10(emission_data), bins)
                    mass_per_bin = [mass_data[bin_indices == j].sum() for j in range(1, len(bins))]
                    bin_centers = 0.5 * (bins[:-1] + bins[1:])
                    
                    ax.fill_between(bin_centers, mass_per_bin, step='mid', color=color, alpha=0.7)
                    ax.axvline(x=np.log10(flux_lim), color='k', linestyle='--', linewidth=2)
            
            annotation_text = f"$M_f$: {mass_fraction:.1%}"
            #(f"{ion} Total Mass: {total_mass:.2e} $M_{{\odot}}$\n"
            #                f"{ion} Mass Above Flux Lim: {mass_above_flux_lim:.2e} $M_{{\odot}}$\n"
            #                f"Mass Fraction: {mass_fraction:.2%}")
            ax.annotate(annotation_text, xy=(0.02, 0.98), xycoords='axes fraction', ha='left', va='top', fontsize=10, bbox=dict(facecolor='white', alpha=0.8))
            
            ax.set_xlabel('Log Surface Brightness [$photons/s/cm^2/sr$]', fontsize=12)
            ax.set_ylabel(rf'{ion} Mass [$M_\odot$]', fontsize=12)
            #ax.set_title(f'{ion} Mass vs Surface Brightness ({orientation}-on)')
            #ax.grid(True, alpha=0.3)

            plt.tight_layout()
            plot_dir = os.path.join(prefix, "mass_vs_emission_plots", orientation)
            os.makedirs(plot_dir, exist_ok=True)
            plt.savefig(os.path.join(plot_dir, f"{orientation}_mass_vs_emission.png"))
        plt.close()

def mass_sb_histbackup(ds,ions,orientations,cmap,flux_threshold_dict,regions):
    
    num_ions = len(ions)
    cols = __builtins__.min(2, num_ions)  # Limit to max 2 columns per row
    rows = math.ceil(num_ions / cols)  # Compute required rows
    
    for orientation in orientations:
        print('orientation', orientation)
        fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows), squeeze=False)
    
        for index, ion in enumerate(ions):
            row, col = divmod(index, cols)
            ax = axes[row, col]  # Select corresponding subplot
            flux_lim = flux_threshold_dict[ion]
            
            for i, bin_size_kpc in enumerate(bin_sizes):
                color = cmap(i)  # Select color from colormap
                save_path = prefix + f'FRBs/'
                halo_name = halo_dict[str(args.halo)]
                file_path = save_path + halo_name + '_emission_maps' + '.hdf5'

                mass_dataset_name = f"z=0.0/{ion}_mass_{orientation}_all"
                emission_dataset_name = f"z=0.0/{ion}_emission_{orientation}_all"

                with h5py.File(file_path, 'r') as f:
                    mass_data = np.array(f[mass_dataset_name]).flatten()
                    emission_data = np.array(f[emission_dataset_name]).flatten()
                    
                    valid_indices = (emission_data > 1e-40) & (mass_data > 0)
                    emission_data = emission_data[valid_indices]
                    mass_data = mass_data[valid_indices]
                    total_mass = mass_data.sum()

                    above_flux_lim_indices = emission_data > flux_lim
                    mass_above_flux_lim = mass_data[above_flux_lim_indices].sum()
                    mass_fraction = mass_above_flux_lim / total_mass if total_mass > 0 else 0
                    
                    flux_min = np.log10(emission_data).min()
                    flux_max = np.log10(emission_data).max()
                    bins = np.linspace(flux_min, flux_max, 64)
                    bin_indices = np.digitize(np.log10(emission_data), bins)
                    mass_per_bin = np.array([mass_data[bin_indices == j].sum() for j in range(1, len(bins))])
                    bin_centers = 0.5 * (bins[:-1] + bins[1:])
                    
                    below_flux_lim = bin_centers <= np.log10(flux_lim+1)
                    at_flux_lim = bin_centers == np.log10(flux_lim)
                    above_flux_lim = bin_centers >= np.log10(flux_lim-1)
                    
                    ax.fill_between(bin_centers[below_flux_lim], mass_per_bin[below_flux_lim], step='mid', color=color, alpha=0.2)
                    ax.fill_between(bin_centers[at_flux_lim], mass_per_bin[at_flux_lim], step='mid', color=color, alpha=0.9)
                    ax.fill_between(bin_centers[above_flux_lim], mass_per_bin[above_flux_lim], step='mid', color=color, alpha=0.9)
                    ax.axvline(x=np.log10(flux_lim), color='k', linestyle='--', linewidth=2)
            
            annotation_text = f"$M_f$: {mass_fraction:.1%}"
            ax.annotate(annotation_text, xy=(0.02, 0.98), xycoords='axes fraction', ha='left', va='top', fontsize=10, bbox=dict(facecolor='white', alpha=0.8))
            
            ax.set_xlabel('Log Surface Brightness [$photons/s/cm^2/sr$]', fontsize=12)
            ax.set_ylabel(rf'{ion} Mass [$M_\odot$]', fontsize=12)
    
        plt.tight_layout()
        plot_dir = os.path.join(prefix, "mass_vs_emission_plots", orientation)
        os.makedirs(plot_dir, exist_ok=True)
        plt.savefig(os.path.join(plot_dir, f"{orientation}_mass_vs_emission.png"))
        plt.close()

def mass_sb(ds,refine_box, ions, orientations, cmap, flux_threshold_dict, regions):
    detectable_ions = ions  
    num_ions = len(detectable_ions)
    cols = __builtins__.min(3, num_ions)  # Limit to max 3 columns per row 
    rows = math.ceil(num_ions / cols)  # Compute required rows
    
    for orientation in orientations:
        print('orientation', orientation)
        
        for region in regions:
            print('region', region)
            fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows), squeeze=False)
            
            halo_name = halo_dict[str(args.halo)]
            
            
            # Set a single title for the whole figure
            plt.suptitle(f'Halo: {halo_name} - {orientation}-on - Box: {box_name}', fontsize=20, fontweight='bold')

            for index, ion in enumerate(detectable_ions):
                row, col = divmod(index, cols)
                ax = axes[row, col]  # Select corresponding subplot
                
                color = cmap(0)  
                save_path = prefix + 'FRBs/'
                file_path = os.path.join(save_path, f"{halo_name}_emission_maps.hdf5")

                mass_dataset_name = f"z=0.0/{ion}_mass_{orientation}_{region}"
                emission_dataset_name = f"z=0.0/{ion}_emission_{orientation}_{region}"

                with h5py.File(file_path, 'r') as f:
                    mass_data = np.array(f[mass_dataset_name]).flatten()
                    emission_data = np.array(f[emission_dataset_name]).flatten()
                    
                    valid_indices = (emission_data > 1e-40) & (mass_data > 0)
                    emission_data = emission_data[valid_indices]
                    mass_data = mass_data[valid_indices]
                    total_mass = mass_data.sum()
                    

                    # Compute bins and mass per bin
                    flux_min = np.log10(emission_data).min()
                    flux_max = np.log10(emission_data).max()
                    bins = np.linspace(flux_min, flux_max, 64)
                    bin_centers = 0.5 * (bins[:-1] + bins[1:])
                    mass_per_bin = np.histogram(np.log10(emission_data), bins=bins, weights=mass_data)[0]

                    ax.plot(bin_centers, mass_per_bin, color=color, linewidth=4)
                    
                    if args.Juniper_limit and ion in ['OVI', 'CII', 'CIII', 'CIV']:
                        flux_lim = flux_threshold_dict[ion]
                        above_flux_lim_indices = emission_data > flux_lim
                        mass_above_flux_lim = mass_data[above_flux_lim_indices].sum()
                        mass_fraction = mass_above_flux_lim / total_mass if total_mass > 0 else 0
                        ax.fill_between(bin_centers, mass_per_bin, where=(bin_centers >= np.log10(flux_lim)), color=color, alpha=0.5)
                        ax.axvline(x=np.log10(flux_lim), color='k', linestyle='--', linewidth=2)

                    

                if args.Juniper_limit and ion in ['OVI', 'CII', 'CIII', 'CIV']:
                    annotation_text = "$M_{frac}$:" f"{mass_fraction:.1%}"
                    ax.annotate(annotation_text, xy=(0.06, 0.94), xycoords='axes fraction', ha='left', va='top', fontsize=18, bbox=dict(facecolor='white', alpha=0.5))
                
                ax.set_xlabel('Log Surface Brightness \n[$photons/s/cm^2/sr$]', fontsize=16)
                ax.set_ylabel(r'Mass [$M_\odot$]', fontsize=16)
                ax.set_xlim(-6,8)
                

                # Make borders (spines) bold
                for spine in ax.spines.values():
                    spine.set_linewidth(2)
                    spine.set_color("k")

                # Set ticks outward
                ax.tick_params(axis='both', which='major', labelsize=14, length=10, width=2, color="k", labelcolor="k", direction="out", top=False, right=False)
                ax.tick_params(axis='both', which='minor', labelsize=14, length=5, width=1, color="k", labelcolor="k", direction="out", top=False, right=False)
                
                #ax.legend(fontsize=14)
                ax.yaxis.offsetText.set_fontsize(14)

                # Set subplot title with ion name
                ax.set_title(f'{ion}', fontsize=16, fontweight='bold')

            # Remove any empty subplots if they exist
            for j in range(index + 1, len(axes.flatten())):
                fig.delaxes(axes.flatten()[j])

            # Adjust layout and spacing
            plt.tight_layout(rect=[0, 0, 1, 0.96])  # Leaves space for suptitle

            # Save the figure
            plot_dir = os.path.join(prefix, "mass_vs_emission_plots")
            os.makedirs(plot_dir, exist_ok=True)
            plt.savefig(os.path.join(plot_dir, f"{orientation}_mass_vs_emission.png"))
            plt.close()

            # for index, ion in enumerate(detectable_ions):
            #     with h5py.File(file_path, 'r') as f:
                    
            #         mass_data = np.array(f[f"z=0.0/{ion}_mass_{orientation}_{region}"])
            #         emission_data = np.array(f[f"z=0.0/{ion}_emission_manualpixel_{orientation}_{region}"])
            #         emissionfrb_data = np.array(f[f"z=0.0/{ion}_emission_{orientation}_{region}"])


            #     print("Total emission from YT projection:", np.sum(emissionfrb_data))
            #     print("Total emission from manual calculation (original):", np.sum(emission_data))
                


            #     # Determine the shape of the grid
            #     grid_size = int(np.sqrt(mass_data.size))  # Assuming a square grid
            #     mass_map = mass_data.reshape((grid_size, grid_size))
            #     emission_map = emission_data.reshape((grid_size, grid_size))
            #     emissionfrb_map = emissionfrb_data.reshape((grid_size, grid_size))
            
            #     # Create plots
            #     fig, axs = plt.subplots(1, 3, figsize=(18, 6))

            #     # Plot Mass Map
            #     im1 = axs[0].imshow(mass_map, origin='lower', cmap='viridis', extent=[-grid_size//2, grid_size//2, -grid_size//2, grid_size//2], norm=LogNorm(vmin=1e-5, vmax=np.max(mass_map)))
            #     axs[0].set_title(f"{ion} Mass Map ({orientation}-on)")
            #     fig.colorbar(im1, ax=axs[0], fraction=0.046, pad=0.04)

            #     # Plot Emission Map
            #     im2 = axs[1].imshow(emission_map, origin='lower', cmap='inferno', extent=[-grid_size//2, grid_size//2, -grid_size//2, grid_size//2], norm=LogNorm(vmin=1e-5, vmax=np.max(emission_map)))
            #     axs[1].set_title(f"{ion} Emission Map ({orientation}-on)")
            #     fig.colorbar(im2, ax=axs[1], fraction=0.046, pad=0.04)

            #     # Plot Emission FRB Map
            #     im3 = axs[2].imshow(emissionfrb_map, origin='lower', cmap='inferno', extent=[-grid_size//2, grid_size//2, -grid_size//2, grid_size//2], norm=LogNorm(vmin=1e-5, vmax=np.max(emissionfrb_map)))
            #     axs[2].set_title(f"{ion} Emission FRB Map ({orientation}-on)")
            #     fig.colorbar(im3, ax=axs[2], fraction=0.046, pad=0.04)

            #     # Display the plots
            #     plt.tight_layout()
            #     plt.show()

def mass_from_SD_sb(ds, ions, orientations, cmap, flux_threshold_dict, regions):
    detectable_ions = ions  
    num_ions = len(detectable_ions)
    cols = __builtins__.min(3, num_ions)  # Limit to max 3 columns per row 
    rows = math.ceil(num_ions / cols)  # Compute required rows
    
    for orientation in orientations:
        print('orientation', orientation)
        
        for region in regions:
            print('region', region)
            fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows), squeeze=False)
            
            halo_name = halo_dict[str(args.halo)]
            
            
            # Set a single title for the whole figure
            plt.suptitle(f'Halo: {halo_name} - {orientation}-on - Box: {box_name}', fontsize=20, fontweight='bold')

            for index, ion in enumerate(detectable_ions):
                row, col = divmod(index, cols)
                ax = axes[row, col]  # Select corresponding subplot
                
                for i, bin_size_kpc in enumerate(bin_sizes):
                    color = cmap(1)  
                    save_path = prefix + 'FRBs/'
                    file_path = os.path.join(save_path, f"{halo_name}_emission_maps.hdf5")

                    mass_dataset_name = f"z=0.0/{ion}_mass_{orientation}_{region}"
                    emission_dataset_name = f"z=0.0/{ion}_emission_{orientation}_{region}"

                    with h5py.File(file_path, 'r') as f:
                        mass_data = np.array(f[mass_dataset_name]).flatten()
                        emission_data = np.array(f[emission_dataset_name]).flatten()
                        
                        valid_indices = (emission_data > 1e-40) & (mass_data > 0)
                        emission_data = emission_data[valid_indices]
                        mass_data = mass_data[valid_indices]
                        total_mass = mass_data.sum()

                        # Compute bins and mass per bin
                        flux_min = np.log10(emission_data).min()
                        flux_max = np.log10(emission_data).max()
                        bins = np.linspace(flux_min, flux_max, 64)
                        bin_centers = 0.5 * (bins[:-1] + bins[1:])
                        mass_per_bin = np.histogram(np.log10(emission_data), bins=bins, weights=mass_data)[0]

                        ax.plot(bin_centers, mass_per_bin, color=color, linewidth=4)
                        
                        if args.Juniper_limit and ion in ['OVI', 'CII', 'CIII', 'CIV']:
                            flux_lim = flux_threshold_dict[ion]
                            above_flux_lim_indices = emission_data > flux_lim
                            mass_above_flux_lim = mass_data[above_flux_lim_indices].sum()
                            mass_fraction = mass_above_flux_lim / total_mass if total_mass > 0 else 0
                            ax.fill_between(bin_centers, mass_per_bin, where=(bin_centers >= np.log10(flux_lim)), color=color, alpha=0.5)
                            ax.axvline(x=np.log10(flux_lim), color='k', linestyle='--', linewidth=2)

                if args.Juniper_limit and ion in ['OVI', 'CII', 'CIII', 'CIV']:
                    annotation_text = "$M_{frac}$:" f"{mass_fraction:.1%}"
                    ax.annotate(annotation_text, xy=(0.06, 0.94), xycoords='axes fraction', ha='left', va='top', fontsize=18, bbox=dict(facecolor='white', alpha=0.5))
                
                ax.set_xlabel('Log Surface Brightness \n[$photons/s/cm^2/sr$]', fontsize=16)
                ax.set_ylabel(r'Mass [$M_\odot$]', fontsize=16)

                # Make borders (spines) bold
                for spine in ax.spines.values():
                    spine.set_linewidth(2)
                    spine.set_color("k")

                # Set ticks outward
                ax.tick_params(axis='both', which='major', labelsize=14, length=10, width=2, color="k", labelcolor="k", direction="out", top=False, right=False)
                ax.tick_params(axis='both', which='minor', labelsize=14, length=5, width=1, color="k", labelcolor="k", direction="out", top=False, right=False)
                
                #ax.legend(fontsize=14)
                ax.yaxis.offsetText.set_fontsize(14)

                # Set subplot title with ion name
                ax.set_title(f'{ion}', fontsize=16, fontweight='bold')

            # Remove any empty subplots if they exist
            for j in range(index + 1, len(axes.flatten())):
                fig.delaxes(axes.flatten()[j])

            # Adjust layout and spacing
            plt.tight_layout(rect=[0, 0, 1, 0.96])  # Leaves space for suptitle

            # Save the figure
            plot_dir = os.path.join(prefix, "mass_vs_emission_plots")
            os.makedirs(plot_dir, exist_ok=True)
            plt.savefig(os.path.join(plot_dir, f"{orientation}_mass_from_SD_vs_emission.png"))
            plt.close()

def cumulative(ions, orientations, cmap):
    detectable_ions = ions
    num_ions = len(detectable_ions)
    cols = __builtins__.min(3, num_ions)  # Limit to max 3 columns per row 
    rows = math.ceil(num_ions / cols)  # Compute required rows

    for orientation in orientations:
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 6, rows * 5))
        axes = np.array(axes).reshape(-1)  # Flatten in case of 1 row
        
        halo_name = halo_dict[str(args.halo)]
        plt.suptitle(f'Halo: {halo_name} - {orientation}-on - Box: {box_name}', fontsize=20, fontweight='bold')

        for i, ion in enumerate(detectable_ions):
            ax = axes[i]  # Select subplot
            

            for region in regions:
                mass_dataset_name = f"z=0.0/{ion}_mass_{orientation}_{region}"
                emission_dataset_name = f"z=0.0/{ion}_emission_{orientation}_{region}"

                for j, bin_size_kpc in enumerate(bin_sizes):
                    color = cmap(0)  # Select color from colormap
                    save_path = prefix + f'FRBs/'
                    file_path = os.path.join(save_path, f"{halo_name}_emission_maps.hdf5")

                    with h5py.File(file_path, 'r') as f:
                        mass_data = np.array(f[mass_dataset_name]).flatten()
                        emission_data = np.array(f[emission_dataset_name]).flatten()

                        # Convert bin size and compute area
                        bin_size = float(bin_size_kpc) * kpc
                        bin_area = bin_size.to(cm) ** 2  

                        # Filter valid values
                        valid_indices = (emission_data > 1e-40) & (mass_data > 1e-40)
                        emission_data = emission_data[valid_indices]
                        mass_data = mass_data[valid_indices]

                        # Define log-spaced bins for emission
                        bins = np.logspace(np.log10(emission_data.min()), np.log10(emission_data.max()), 64)

                        # Calculate cumulative mass for each bin
                        bin_indices = np.digitize(emission_data, bins)
                        cumulative_mass = np.array([mass_data[bin_indices <= k].sum() for k in range(1, len(bins))])

                        # Plot the cumulative results
                        bin_centers = 0.5 * (bins[:-1] + bins[1:])
                        ax.plot(bin_centers, cumulative_mass, color=color, lw=4, label=f'Bin Size: {bin_size_kpc} kpc')

                        if args.Juniper_limit and ion in ['OVI', 'CII', 'CIII', 'CIV']:
                            flux_lim = flux_threshold_dict[ion]
                            ax.axvline(x=flux_lim, color='k', linestyle='--', linewidth=2)

            # Formatting for each subplot
            ax.set_xscale('log')
            ax.set_xlabel('Surface Brightness [$photons/s/cm^2/sr$]', fontsize=16)
            ax.set_ylabel('Cumulative Mass [$M_{sun}$]', fontsize=16)
            ax.tick_params(axis='both', which='major', labelsize=14, size=10, width=2)
            ax.set_title(f'{ion}', fontsize=16, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.yaxis.offsetText.set_fontsize(14)

            # Make borders (spines) bold
            for spine in ax.spines.values():
                spine.set_linewidth(2)
                spine.set_color("k")

        # Remove empty subplots if any
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])

        plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust for suptitle

        # Save the plot
        plot_dir = os.path.join(prefix, "mass_vs_emission_plots")
        os.makedirs(plot_dir, exist_ok=True)
        plt.savefig(os.path.join(plot_dir, f"cumulative_mass_vs_emission_{orientation}.png"))

        plt.close()

def mass_b(ds,ions,orientations,cmap,flux_threshold_dict,Aeff_dict):
    # make mass vs surfave brightness 
    for (ion,flux_lim),region in zip(flux_threshold_dict.items(),regions):
        print('flux_lim',flux_lim)
        print('ion',ion)
        Aeff = Aeff_dict.get(ion, 0)
        for orientation in orientations:
            print('orientation',orientation)
            plt.figure(figsize=(10, 7))
            
            # Dataset names for mass and emission
            mass_dataset_name = f"z=0.0/{ion}_mass_{orientation}_{region}"
            emission_dataset_name = f"z=0.0/{ion}_emission_{orientation}_{region}"

            for i, bin_size_kpc in enumerate(bin_sizes):
                color = cmap(i)  # Select color from colormap
                save_path = prefix + f'FRBs/'
                halo_name = halo_dict[str(args.halo)]
                file_path = save_path + halo_name + '_emission_maps' + '.hdf5'

                # Open the HDF5 file and extract mass and emission data
                with h5py.File(file_path, 'r') as f:
                    mass_data = np.array(f[mass_dataset_name]).flatten()
                    emission_data = np.array(f[emission_dataset_name]).flatten()
                    

                    # Filter valid values
                    valid_indices = (emission_data > 1e-40) & (mass_data > 1e-40)
                    emission_data = emission_data[valid_indices]*((float(bin_size_kpc)*3.086e21)**2)
                    mass_data = mass_data[valid_indices]
                    total_mass = mass_data.sum()

                    # Calculate total mass of gas with emission above flux_lim
                    flux_lim = flux_lim *((float(bin_size_kpc)*3.086e21)**2)
                    above_flux_lim_indices = emission_data > flux_lim
                    mass_above_flux_lim = mass_data[above_flux_lim_indices].sum()

                    # Calculate mass fraction
                    mass_fraction = mass_above_flux_lim / total_mass if total_mass > 0 else 0

                    # Define log-spaced bins for emission
                    flux_min = emission_data.min()
                    flux_max = emission_data.max()
                    bins = np.logspace(np.log10(flux_min), np.log10(flux_max), 64)

                    # Calculate total mass per bin
                    bin_indices = np.digitize(emission_data, bins)
                    mass_per_bin = [mass_data[bin_indices == j].sum() for j in range(1, len(bins))]

                    # Plot the results
                    bin_centers = 0.5 * (bins[:-1] + bins[1:])
                    plt.fill_between(bin_centers, mass_per_bin, step='mid', color=color, alpha=0.7,
                                        label=f'Bin Size: {bin_size_kpc} kpc')

                    plt.axvline(x = flux_lim, color = 'k', linestyle='--', linewidth=2)

            # Add the annotations
            annotation_text = (f"Total Mass: {total_mass:.2e} $M_{{\odot}}$\n"
                               f"Total Mass Above Flux Lim: {mass_above_flux_lim:.2e} $M_{{\odot}}$\n"
                               f"Mass Fraction: {mass_fraction:.2%}")
            plt.annotate(annotation_text, xy=(0.02, 0.98), xycoords='axes fraction',
                         ha='left', va='top', fontsize=14, bbox=dict(facecolor='white', alpha=0.8))
                    

            # Finalize the plot for this ion and orientation
            plt.yscale('log')  # Logarithmic scale for y-axis
            plt.xscale('log')  # Logarithmic scale for x-axis
            plt.xlabel(' Brightness [$photons/s/sr$]')
            plt.ylabel('Mass [$M_{sun}$]')
            #plt.xlim(1e-15,1e10)
            #plt.ylim(1e-5,1e10)
            plt.title(f'{ion} Mass vs Brightness ({orientation}-on)')
            #plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()

            # Save the plot
            plot_dir = os.path.join(prefix, "mass_vs_brightness_plots", orientation)
            os.makedirs(plot_dir, exist_ok=True)
            plt.savefig(os.path.join(plot_dir, f"{ion.replace(' ', '_')}_{orientation}_mass_vs_brightness.png"))
        
            plt.close()

def projection_num_density(ds,refine_box, ions, ions_number_density_dict, label_dict):
    """
    Create projection plots for the number density of specified ions.

    Parameters:
    -----------
    ds : yt.Dataset
        Loaded yt dataset.
    ions : list of str
        List of ion names for projection.
    save_path : str
        Directory where the plots will be saved.
    ions_density_dict : dict
        Dictionary mapping ions to their number density fields.
    label_dict : dict
        Dictionary mapping ions to their plot labels.

    Returns:
    --------
    None
    """

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
                
        
        for region, data_source in data_sources.items():
            for ion in ions:
                if ion == 'HI':
                    print(f"Generating projection plot for {ion}...")
                    
                    # Get the corresponding number density field for the ion
                    num_density_field = ('gas', ions_number_density_dict[ion])

                    ##Edge-on projection
                    #Create the projection plot along the z-axis
                    orientation = 'edge'
                    proj_edge = yt.ProjectionPlot(ds, ds.x_unit_disk, ('gas', ions_number_density_dict[ion]), center=ds.halo_center_kpc, data_source=data_source, width=(float(width_value), 'kpc'),
                                        north_vector=ds.z_unit_disk, method = 'integrate', weight_field=None) 
                    #p.set_unit(density_field, 'cm**-3')
                    proj_edge.set_cmap(num_density_field, h1_color_map)
                    proj_edge.set_zlim(ions_number_density_dict[ion], 1e15,1e24)
                    proj_edge.set_font_size(36)
                    proj_edge.set_xlabel('x (kpc)')
                    proj_edge.set_ylabel('y (kpc)')

                    # Make borders (spines) bold and white
                    fig = proj_edge.plots[('gas', ions_number_density_dict[ion])].figure  # Get Matplotlib figure
                    ax = fig.axes[0]  # Access the first axis

                    for spine in ax.spines.values():  # Now modify spines
                        spine.set_linewidth(2)  # Example: Set spine thickness
                        spine.set_color("k")  # Example: Set spine color to black


                    # Set ticks outward and white
                    ax.tick_params(axis='both', which='major', labelsize=36, length=15, width=3,
                                color="k", labelcolor="k", direction="in",top=False, right=False)
                    ax.tick_params(axis='both', which='minor', labelsize=36, length=10, width=2,
                                color="k", labelcolor="k", direction="in",top=False, right=False)
                
                
                    #p.annotate_title(f"{label_dict.get(ion, ion)} Number Density")
                    
                    # Save the plot
                    plot_dir = os.path.join(prefix, "number_density_projections", orientation)
                    os.makedirs(plot_dir, exist_ok=True)
                    proj_edge.save(os.path.join(plot_dir, f"{ion.replace(' ', '_')}_{orientation}_number_density.png"))
                    

                    # Face-on projection
                    # Create the projection plot along the z-axis
                    orientation = 'face'
                    proj_face = yt.ProjectionPlot(ds, ds.z_unit_disk, ('gas', ions_number_density_dict[ion]), center=ds.halo_center_kpc, data_source=data_source, width=(float(width_value), 'kpc'),
                                        north_vector=ds.x_unit_disk, method = 'integrate', weight_field=None) 
                    #p.set_unit(density_field, 'cm**-3')
                    proj_face.set_cmap(num_density_field, h1_color_map)
                    proj_face.set_zlim(ions_number_density_dict[ion], 1e15,1e24)
                    #p.annotate_title(f"{label_dict.get(ion, ion)} Number Density")
                    proj_face.set_font_size(32)
                    proj_face.set_xlabel('x (kpc)')
                    proj_face.set_ylabel('y (kpc)')


                    # Make borders (spines) bold and white
                    fig = proj_face.plots[('gas', ions_number_density_dict[ion])].figure  # Get Matplotlib figure
                    ax = fig.axes[0]  # Access the first axis

                    for spine in ax.spines.values():  # Now modify spines
                        spine.set_linewidth(2)  # Example: Set spine thickness
                        spine.set_color("k")  # Example: Set spine color to black


                    # Set ticks outward and white
                    ax.tick_params(axis='both', which='major', labelsize=36, length=15, width=3,
                                color="k", labelcolor="k", direction="in", top=False, right=False)
                    ax.tick_params(axis='both', which='minor', labelsize=36, length=10, width=2,
                                color="k", labelcolor="k", direction="in",top=False, right=False)
                    
                    
                    
                    # Save the plot
                    plot_dir = os.path.join(prefix, "number_density_projections", orientation)
                    os.makedirs(plot_dir, exist_ok=True)
                    proj_face.save(os.path.join(plot_dir, f"{ion.replace(' ', '_')}_{orientation}_number_density_.png"))

def rotation_curve(ds,refine_box, bin_width_kpc=2):
    """
    This function creates a plot of the mass-weighted radial velocity (rotation curve)
    vs. distance for ions in the CGM, considering only the data inside the ds.refine_width box size.
    """

    
    save_path = prefix + f'rotation_curve_plots/'
    os.makedirs(save_path, exist_ok=True)  # Ensure the directory exists

    # Define the box boundaries from refine_width
    refine_width = ds.refine_width
    halo_center_kpc = ds.halo_center_kpc

    if filter_type == 'inflow_outflow':
        # Apply the inflow/outflow filtering
        box_inflow, box_outflow, box_neither = filter_ds(ds.all_data())
        data_sources = {'inflow': box_inflow, 'outflow': box_outflow, 'neither': box_neither}

    elif filter_type == 'disk_cgm':
        # Apply the disk/CGM filtering
        box_cgm = load_disk(ds,clump_file,source_cut=None)
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
            

    for region, data in data_sources.items():

        print('Rotation curve for',region)

        # Compute coordinates relative to the halo center
        x_coord = data['gas', 'x'].to('kpc')
        y_coord = data['gas', 'y'].to('kpc')
        z_coord = data['gas', 'z'].to('kpc')

        x_pos = x_coord - halo_center_kpc[0]
        y_pos = y_coord - halo_center_kpc[1]
        z_pos = z_coord - halo_center_kpc[2]

        # Filter data to only include points within the refine_width box
        mask = (
            (x_pos >= -refine_width / 2) & (x_pos <= refine_width / 2) &
            (y_pos >= -refine_width / 2) & (y_pos <= refine_width / 2) &
            (z_pos >= -refine_width / 2) & (z_pos <= refine_width / 2)
        )

        # Apply the mask to filter data
        x_pos = x_pos[mask]
        y_pos = y_pos[mask]
        z_pos = z_pos[mask]
        radius = np.sqrt((x_pos**2) + (y_pos**2) + (z_pos**2))

        # Process each ion and its mass field
        for ion, mass_field in ion_mass_fields.items():
            Vr = data['gas', 'radial_velocity_corrected'][mask]
            Mass = data["gas", f"{mass_field}"].to("Msun")[mask]

            # Only consider data with non-zero mass
            idx = np.where(Mass > 0)
            radius_valid = radius[idx]
            Vr_valid = Vr[idx]
            Mass_valid = Mass[idx]

            # Calculate bin edges dynamically
            r_min = radius_valid.min().to_value()
            r_max = radius_valid.max().to_value()
            bin_edges = np.arange(r_min, r_max + bin_width_kpc, bin_width_kpc)
            # Compute mass-weighted radial velocity for each bin
            hist_sum_mass, _ = np.histogram(radius_valid, bins=bin_edges, weights=Mass_valid)
            hist_sum_vr_mass, _ = np.histogram(radius_valid, bins=bin_edges, weights=Vr_valid * Mass_valid)
            Vr_mass_weighted = hist_sum_vr_mass / hist_sum_mass

            # Plot the mass-weighted rotation curve
            bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
            plt.plot(bin_centers, Vr_mass_weighted, linestyle="-", linewidth=2, label=f"Ion: {ion}")

        # Finalize the plot
        plt.xlabel("Radius (kpc)")
        plt.ylabel("Radial Velocity (km/s)")
        plt.xlim(0,200)
        plt.title("Rotation Curve (Mass-Weighted Radial Velocity)")
        plt.legend()
        plt.savefig(save_path + f'rotation_curve.png')

def number_density_vs_radius(ds, ions, orientations, cmap, flux_threshold_dict, regions):
    """
    This function generates plots of total number density vs. radius for each ion and orientation.

    Parameters:
        ds: yt dataset
        ions: List of ions
        orientations: List of orientations (e.g., edge-on, face-on)
        cmap: Colormap
        flux_threshold_dict: Dictionary of flux thresholds for each ion
        regions: List of regions
    """

    for ion,region in zip(ions,regions):
        for orientation in orientations:
            plt.figure(figsize=(10, 7))
            print(f"Processing ion: {ion}, orientation: {orientation}")

            # HDF5 file path
            halo_name = halo_dict[str(args.halo)]
            file_path = os.path.join(prefix, f"FRBs/res_0.27/{halo_name}_emission_maps.hdf5")

            # Dataset names for number density and positions
            numdensity_dataset = f"z=0.0/{ion}_numdensity_{orientation}_{region}"
            x_dataset = f"z=0.0/{ion}_x_{orientation}_{region}"
            y_dataset = f"z=0.0/{ion}_y_{orientation}_{region}"

            # Open the HDF5 file and extract data
            with h5py.File(file_path, "r") as f:
                numdensity_data = np.array(f[numdensity_dataset]).flatten()
                x_positions = np.array(f[x_dataset]).flatten()
                y_positions = np.array(f[y_dataset]).flatten()
                

            # Create a 2D grid of positions
            x_grid, y_grid = np.meshgrid(x_positions, y_positions, indexing='ij')

            # Compute radius for each pixel and flatten
            radius = np.sqrt(x_grid**2 + y_grid**2).flatten()
            print(f"radius shape: {radius.shape}")
            print('num density shape',numdensity_data.shape)
            plt.scatter(radius,np.log10(numdensity_data),marker='o',s=1,color='green')

            # Annotate plot
            #plt.yscale('log')
            plt.xlabel("Radius [kpc]")
            plt.ylabel("Log Column Density [$cm^{-2}$]")
            plt.xlim(0,150)
            plt.ylim(12,15)
            plt.title(f"Column Density vs Radius ({ion}, {orientation})")
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.tight_layout()

            # Save the plot
            plot_dir = os.path.join(prefix, "number_density_vs_radius_plots", orientation)
            os.makedirs(plot_dir, exist_ok=True)
            plt.savefig(os.path.join(plot_dir, f"{ion.replace(' ', '_')}_{orientation}_numdensity_vs_radius.png"))

            plt.close()


#####################################################################################################
def load_and_calculate(snap, ions,filter_value=None,resolution=1):

    '''Loads the simulation snapshot and makes the requested plots, with optional filtering.'''

    # Load simulation output
    snap_name = foggie_dir + 'halo_00' + args.halo + '/' + args.run + '/' + snap + '/' + snap
    
    ds, refine_box = foggie_load(snap_name, trackname, do_filter_particles=True, halo_c_v_name=halo_c_v_name, disk_relative=True, correct_bulk_velocity=True)#, smooth_AM_name=smooth_AM_name)
    zsnap = ds.get_parameter('CosmologyCurrentRedshift')

    add_ion_fields(ds,ions)

    #emission_temp(ds,ions)
    #sb_mass_hist(ds,ions,orientations,bin_sizes,flux_threshold_dict)
    mass_sb(ds,refine_box,ions,orientations,cmap,flux_threshold_dict,regions)
    cumulative(ions,orientations,cmap)
    #mass_from_SD_sb(ds, ions, orientations, cmap, flux_threshold_dict, regions)
    #mass_b(ds,ions,orientations,cmap,flux_threshold_dict,Aeff_dict)
    projection_num_density(ds,refine_box, ions, ions_number_density_dict, label_dict)
    #rotation_curve(ds, bin_width_kpc=2)
    #numdensity_emissivity_scatter(ds,ions_number_density_dict,ions_dict)
    #numdensity_emissivity(ds,ions_number_density_dict,ions_dict)
    number_density_vs_radius(ds, ions, orientations, cmap, flux_threshold_dict, regions)
    #vr_vlos(ds, bin_width_kpc=2,bin_count=50)
    #area_frac(ds,ions,orientations,cmap,flux_threshold_dict,regions)

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
    clump_file = output_dir + '/Disk/test_Disk.h5'
    
    # Set directory for output location, making it if necessary
    box_name = args.fov if args.fov is not None else 'refine_box'
    prefix = output_dir + '/res_' + args.resolution + '/' + 'box_' + box_name + '/'
    if not (os.path.exists(prefix)): os.system('mkdir -p ' + prefix)
    table_loc = prefix + 'Tables/'

    print('foggie_dir: ', foggie_dir)
    catalog_dir = code_path + 'halo_infos/00' + args.halo + '/' + args.run + '/'
    halo_c_v_name = catalog_dir + 'halo_c_v'
    #smooth_AM_name = catalog_dir + 'AM_direction_smoothed'

    cloudy_path = "/Users/vidasaeedzadeh/Documents/02-Projects/02-FOGGIE/Cloudy-runs/outputs/test-z0/TEST_z0_HM12_sh_run%i.dat"
    #code_path + "emission/cloudy_z0_selfshield/sh_z0_HM12_run%i.dat"
    #set the clump file directory
    disk_file = output_dir + '/Disk/test_Disk.h5'
    shell_path = output_dir + '/Disk/'
    #"/Users/vidasaeedzadeh/Documents/02-Projects/02-FOGGIE/Cloudy-runs/outputs/test-z0/TEST_z0_HM12_sh_run%i.dat"
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
    hden_pts, T_pts, table_SiII_1814 = make_Cloudy_table(11,cloudy_path)
    hden_pts, T_pts = np.meshgrid(hden_pts, T_pts)
    pts = np.array((hden_pts.ravel(), T_pts.ravel())).T
    sr_SiII_1814 = table_SiII_1814.T.ravel()
    bl_SiII_1814 = interpolate.LinearNDInterpolator(pts, sr_SiII_1814)
    register_emission_field_with_unit('Emission_SiII_1814', Emission_SiII_1814, emission_units, unit_system,scale_factor,scaling)
    ############################
    # SiIII 1207
    cloudy_path_thin = code_path + "emission/cloudy_z0_HM05/bertone_run%i.dat"
    hden_pts, T_pts, table_SiIII_1207 = make_Cloudy_table(12,cloudy_path)
    hden_pts, T_pts = np.meshgrid(hden_pts, T_pts)
    pts = np.array((hden_pts.ravel(), T_pts.ravel())).T
    sr_SiIII_1207 = table_SiIII_1207.T.ravel()
    bl_SiIII_1207 = interpolate.LinearNDInterpolator(pts, sr_SiIII_1207)
    register_emission_field_with_unit('Emission_SiIII_1207', Emission_SiIII_1207, emission_units, unit_system,scale_factor,scaling)
    ############################
    # SiIV 1394
    cloudy_path_thin = code_path + "emission/cloudy_z0_HM05/bertone_run%i.dat"
    hden_pts, T_pts, table_SiIV_1394 = make_Cloudy_table(14,cloudy_path)
    hden_pts, T_pts = np.meshgrid(hden_pts, T_pts)
    pts = np.array((hden_pts.ravel(), T_pts.ravel())).T
    sr_SiIV_1394 = table_SiIV_1394.T.ravel()
    bl_SiIV_1394 = interpolate.LinearNDInterpolator(pts, sr_SiIV_1394)
    register_emission_field_with_unit('Emission_SiIV_1394', Emission_SiIV_1394, emission_units, unit_system,scale_factor,scaling)
    ############################
    # MgII 2796
    cloudy_path_thin = code_path + "emission/cloudy_z0_HM05/bertone_run%i.dat"
    hden_pts, T_pts, table_MgII_2796 = make_Cloudy_table(16,cloudy_path)
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
    
    label_dict = {'Lyalpha':r'Ly-$\alpha$', 'HI':r'H$\alpha$', 'CII':'C II','CIII':'C III',
                'CIV':'C IV','OVI':'O VI','SiII':'Si II','SiIII':'Si III','SiIV':'Si IV','MgII':'Mg II'}
    
    trident_dict = { 'HI':'H I', 'CII':'C II','CIII':'C III',
                'CIV':'C IV','OVI':'O VI','SiII':'Si II','SiIII':'Si III','SiIV':'Si IV','MgII':'Mg II'}

    if unit_system  == 'default':
        zlim_dict = {'Lyalpha':[1e-1,1e7], 'HI':[1e-1,1e6], 'CII':[1e-7,1e2], 'CIII':[1e-4,1e2],
                     'CIV':[1e-2,1e4], 'OVI':[1e-2,1e4],'SiII':[1e-6,1e2],'SiIII':[1e-6,1e2],'SiIV':[1e-6,1e2],'MgII':[1e-6,1e2]}
        
    elif unit_system == 'ALT':
        zlim_dict = {'Lyalpha':[1e-22,1e-16], 'HI':[1e-22,1e-16], 'CII':[1e-23,1e-16], 'CIII':[1e-23,1e-16],
                     'CIV':[1e-23,1e-16], 'OVI':[1e-23,1e-16],'SiII':[1e-23,1e-16],'SiIII':[1e-23,1e-16],'SiIV':[1e-23,1e-16],'MgII':[1e-23,1e-16]}
        
        
    numden_zlim_dict = {'HI':[1e12,1e23], 'CIII':[1e9,1e16],'CII':[1e7,1e19],
                 'CIV':[1e10,2e15], 'OVI':[1e10,3e14],'SiII':[1e8,1e18],'SiIII':[1e8,1e18],'SiIV':[1e8,1e18],'MgII':[1e8,1e18]}
        

    orientations = ["edge", "face"]

    if args.Juniper_limit:
        flux_threshold_dict = {'CII':1588.24, 'CIII':9000.00,'CIV':3857.14, 'OVI':1800.00} #update them with the correct values
    else:
        flux_threshold_dict = {}

    cmap = mtcm.get_cmap('Dark2', 8) 


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

    #Build bin_sizes
    bin_sizes = [args.bin_sizes]

    filter_type = args.filter_type

    # make mass vs surfave brightness 
    if filter_type == 'inflow_outflow':

        regions = {'inflow', 'outflow', 'neither'}

    elif filter_type == 'disk_cgm':

        regions = {'cgm'}   

    else:   
         
         regions = {'all'}
    
    shell_count = int(args.shell_count)

    resolution = args.resolution
    resolution = int(resolution)




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



