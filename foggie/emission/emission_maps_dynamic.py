'''
Filename: emission_maps_dynamic.py
Author: Vida
Date created: 1-15-25
Date last modified: 2-4-25

This file contains everything needed to make emission maps and FRBs from CLOUDY tables.
All CLOUDY and emission code copy-pasted from Lauren's foggie/cgm_emission/emission_functions.py and Cassi's foggie/cgm_emission/emission_maps.py 

This code is modified Cassi's foggie/cgm_emission/emission_maps.py to to dynamically get:
1. filters: tempreture, density, inflow, outflow, disk, cgm
2. resolution for frb maps
3. units for emission: default: photons s$^{-1}$ cm$^{-2}$ sr$^{-1}$ and ALT: erg s$^{-1}$ cm$^{-2}$ arcsec$^{-2}$

usual command to run for z = 0: python emission_maps_dynamic.py --ions 'HI,CII,CIII,CIV,OVI' --resolution 1 --halo 8508

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


import datetime
from scipy import interpolate
import shutil
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable 
import cmasher as cmr
import matplotlib.colors as mcolors
import h5py
import trident

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
import numpy as np
from yt.units.yt_array import YTQuantity
from scipy.ndimage import gaussian_filter

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

    

    args = parser.parse_args()
    return args


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
        emission_line = emission_line / (4. * np.pi * 2.03e-11)
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
        emission_line = emission_line / (4.*np.pi*1.65e-11)
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
        emission_line = emission_line / (4.*np.pi*1.65e-11)
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
        emission_line = emission_line / (4.*np.pi*1.65e-11) # what should be instead of 1.65e-11 for MgII? or anyother new element I use?
        return emission_line * ytEmU
    elif unit_system == 'ALT':
        emission_line = emission_line / (4. * np.pi)
        emission_line = emission_line / 4.25e10
        return emission_line * ytEmUALT
    else:
        raise ValueError("Invalid unit_system specified. Use 'default' or 'ALT'.")

#FILTERS : Inflow and Outflow
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

def projection(ds, refine_box, snap, ions, filter_type=None, filter_value=None):

    # Create HDF5 file for saving emission maps
    save_path = prefix + f'Projections/density/' 
    os.makedirs(save_path, exist_ok=True)  # Ensure the directory exists

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
        data_sources = {'all': ds.all_data()}
        if filter_type and filter_value:
            if filter_type == 'temperature':
                data_sources['all'] = data_sources['all'].cut_region([f"(obj['gas', 'temperature'] < {filter_value})"])
            elif filter_type == 'density':
                data_sources['all'] = data_sources['all'].cut_region([f"(obj['gas', 'density'] > {filter_value})"])
            else:
                raise ValueError("Unsupported filter type. Supported types: 'temperature', 'density'.")
            
    # Loop through ions and create projections for each region
    for region, data_source in data_sources.items():
        print('region',region)
        
        #Edge-on projection
        proj_edge = yt.ProjectionPlot(ds, ds.x_unit_disk, ('gas', 'density'),
                                        center=ds.halo_center_kpc, data_source=data_source,width=(100, 'kpc'),
                                        north_vector=ds.z_unit_disk, method = 'integrate', weight_field=None)

        # Set colormap and save projection plot
        mymap = cmr.get_sub_cmap('plasma', 0, 1)
        mymap.set_bad("#421D0F")
        
        proj_edge.save(save_path + f'{snap}_density_map_edge-on_{region}' + save_suffix + '.png')

        # Face-on projection
        projface = yt.ProjectionPlot(ds, ds.z_unit_disk, ('gas', 'density'),
                                        center=ds.halo_center_kpc, data_source=data_source,width=(100, 'kpc'),
                                        north_vector=ds.x_unit_disk, method = 'integrate', weight_field=None)

        # Set colormap and save projection plot
        mymap = cmr.get_sub_cmap('plasma', 0, 1)
        mymap.set_bad("#421D0F")
        
        projface.save(save_path + f'{snap}_density_map_face-on_{region}' + save_suffix + '.png')

    print('finished density projections')


def original_make_FRB(ds, refine_box, snap, ions, unit_system='default', filter_type=None, filter_value=None,resolution=2):
    '''This function takes the dataset 'ds' and the refine box region 'refine_box' and
    makes a fixed resolution buffer of surface brightness from edge-on, face-on,
    and arbitrary orientation projections of all ions in the list 'ions'.'''

    halo_name = halo_dict[str(args.halo)]

    # Determine resolution and bin size
    pix_res = float(np.min(refine_box['dx'].in_units('kpc')))
    bin_size_kpc = resolution*pix_res
    round_bin_size_kpc = round(bin_size_kpc,2)
    # Ensure fov_kpc is in kpc
    if not hasattr(ds.refine_width, 'in_units'):
        fov_kpc = YTQuantity(ds.refine_width, 'kpc')  # Wrap in YTQuantity with units
    else:
        fov_kpc = ds.refine_width.in_units('kpc')  # Convert to kpc if it has units
    
    # Convert to numeric value (without units) for calculations
    
    width = (80, 'kpc') #fov_kpc
    width_value = width[0]
    res= int(width_value/bin_size_kpc)

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
        data_sources = {'all': ds.all_data()}
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
    save_path = prefix + f'FRBs/res_{bin_size_kpc:.2f}/' 
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
            elif args.Juniper_limit and ion == 'OVI':
                cmap1 = cmr.take_cmap_colors('magma', 7, cmap_range=(0.5, 1), return_fmt='rgba')
                cmap2 = cmr.take_cmap_colors('cmr.neutral_r', 5, cmap_range=(0.0, 0.6), return_fmt='rgba')
                cmap = np.hstack([cmap2, cmap1])
                mymap = mcolors.LinearSegmentedColormap.from_list('cmap', cmap)
            elif args.Juniper_limit and ion == 'CIV':
                cmap1 = cmr.take_cmap_colors('magma', 6, cmap_range=(0.5, 1), return_fmt='rgba')
                cmap2 = cmr.take_cmap_colors('cmr.neutral_r', 7, cmap_range=(0.0, 0.6), return_fmt='rgba')
                cmap = np.hstack([cmap2, cmap1])
                mymap = mcolors.LinearSegmentedColormap.from_list('cmap', cmap)
            else:
                #mymap = cmr.get_sub_cmap('cmr.flamingo', 0.2, 0.8)
                mymap = cmr.get_sub_cmap('magma', 0, 1)
            mymap.set_bad(mymap(0))

            #Edge-on projection
            proj_edge = yt.ProjectionPlot(ds, ds.x_unit_disk, ('gas', 'Emission_' + ions_dict[ion]),
                                          center=ds.halo_center_kpc, data_source=data_source,width=width,
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
                                          center=ds.halo_center_kpc, data_source=data_source,width=width,
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
    print('finished')
    f.close()

def make_FRB(ds, refine_box, snap, ions, unit_system='default', filter_type=None, filter_value=None, resolution=2):
    '''This function takes the dataset 'ds' and the refine box region 'refine_box' and
    makes a fixed resolution buffer of surface brightness from edge-on, face-on,
    and arbitrary orientation projections of all ions in the list 'ions'.'''

    halo_name = halo_dict[str(args.halo)]

    # Determine resolution and bin size
    pix_res = float(np.min(refine_box['dx'].in_units('kpc')))
    bin_size_kpc = resolution * pix_res
    round_bin_size_kpc = round(bin_size_kpc, 2)
    
    # Ensure fov_kpc is in kpc
    if not hasattr(ds.refine_width, 'in_units'):
        fov_kpc = YTQuantity(ds.refine_width, 'kpc')
    else:
        fov_kpc = ds.refine_width.in_units('kpc')
    
    width = (220, 'kpc')
    width_value = width[0]
    res = int(width_value / bin_size_kpc)
    
    # Apply filtering
    if filter_type == 'disk_cgm':
        disk_cut_region = load_clump(ds, disk_file, source_cut=refine_box)
        box_cgm = refine_box - disk_cut_region
        data_sources = {'cgm': box_cgm}
    else:
        data_sources = {'all': ds.all_data()}
    
    save_path = prefix + f'FRBs/res_{bin_size_kpc:.2f}/'
    os.makedirs(save_path, exist_ok=True)

    
    for region, data_source in data_sources.items():
        for ion in ions:
            
            print(f"Processing {ion}...")

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
            elif args.Juniper_limit and ion == 'OVI':
                cmap1 = cmr.take_cmap_colors('magma', 7, cmap_range=(0.5, 1), return_fmt='rgba')
                cmap2 = cmr.take_cmap_colors('cmr.neutral_r', 5, cmap_range=(0.0, 0.6), return_fmt='rgba')
                cmap = np.hstack([cmap2, cmap1])
                mymap = mcolors.LinearSegmentedColormap.from_list('cmap', cmap)
            elif args.Juniper_limit and ion == 'CIV':
                cmap1 = cmr.take_cmap_colors('magma', 6, cmap_range=(0.5, 1), return_fmt='rgba')
                cmap2 = cmr.take_cmap_colors('cmr.neutral_r', 7, cmap_range=(0.0, 0.6), return_fmt='rgba')
                cmap = np.hstack([cmap2, cmap1])
                mymap = mcolors.LinearSegmentedColormap.from_list('cmap', cmap)
            else:
                #mymap = cmr.get_sub_cmap('cmr.flamingo', 0.2, 0.8)
                mymap = cmr.get_sub_cmap('magma', 0, 1)
            mymap.set_bad(mymap(0))


            # Edge-on Projection
            proj_edge = yt.ProjectionPlot(ds, ds.x_unit_disk, ('gas', f'Emission_{ions_dict[ion]}'),
                                          center=ds.halo_center_kpc, data_source=data_source,
                                          width=width, north_vector=ds.z_unit_disk,
                                          buff_size=[res, res], method='integrate', weight_field=None)
            proj_edge.set_cmap(f'Emission_{ions_dict[ion]}', mymap)
            proj_edge.set_zlim(f'Emission_{ions_dict[ion]}', zlim_dict[ion][0], zlim_dict[ion][1])
            proj_edge.set_font_size(24)

            # Hide colormap, labels, and ticks
            fig = proj_edge.plots['Emission_' + ions_dict[ion]].figure
            ax = proj_edge.plots['Emission_' + ions_dict[ion]].axes
            
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_xlabel("")
            ax.set_ylabel("")
            
            cbar = proj_edge.plots['Emission_' + ions_dict[ion]].cb
            cbar.remove()
            
            fig.savefig(
                save_path + f'{snap}_{ion}_emission_map_edge-on_{region}' + save_suffix + '.png',
                bbox_inches="tight", dpi=300
            )

            # face-on Projection
            proj_face = yt.ProjectionPlot(ds, ds.z_unit_disk, ('gas', f'Emission_{ions_dict[ion]}'),
                                          center=ds.halo_center_kpc, data_source=data_source,
                                          width=width, north_vector=ds.x_unit_disk,
                                          buff_size=[res, res], method='integrate', weight_field=None)
            proj_face.set_cmap(f'Emission_{ions_dict[ion]}', mymap)
            proj_face.set_zlim(f'Emission_{ions_dict[ion]}', zlim_dict[ion][0], zlim_dict[ion][1])
            proj_face.set_font_size(24)

            # Hide colormap, labels, and ticks
            fig = proj_face.plots['Emission_' + ions_dict[ion]].figure
            ax = proj_face.plots['Emission_' + ions_dict[ion]].axes
            
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_xlabel("")
            ax.set_ylabel("")
            
            cbar = proj_face.plots['Emission_' + ions_dict[ion]].cb
            cbar.remove()
            
            fig.savefig(
                save_path + f'{snap}_{ion}_emission_map_face-on_{region}' + save_suffix + '.png',
                bbox_inches="tight", dpi=300
            )
    

def make_black_FRB(ds, refine_box, snap, ions, unit_system='default', filter_type=None, filter_value=None, resolution=2):
    """
    Creates a fixed resolution buffer of surface brightness from edge-on and face-on projections of specified ions.
    The plot is generated with a black background, outward ticks, and a colorbar on the right.
    """
    
    halo_name = halo_dict[str(args.halo)]

    # Determine resolution and bin size
    pix_res = float(np.min(refine_box['dx'].in_units('kpc')))
    bin_size_kpc = resolution * pix_res
    round_bin_size_kpc = round(bin_size_kpc, 2)
    fov_kpc = ds.refine_width.in_units('kpc') if hasattr(ds.refine_width, 'in_units') else YTQuantity(ds.refine_width, 'kpc')
    
    width = (80, 'kpc') 
    width_value = width[0]
    res = int(width_value / bin_size_kpc)

    print(f"Field of view (FOV): {width_value:.3f} kpc")
    print(f"Adjusted bin size: {bin_size_kpc:.2f} kpc")
    print(f"Resolution: {res} bins")

    # Apply filtering
    if filter_type == 'disk_cgm':
        disk_cut_region = load_clump(ds, disk_file, source_cut=refine_box)
        box_cgm = refine_box - disk_cut_region
        for i in range(shell_count):
            shell_clump_file = shell_path + f'test_DiskDilationShell_n{i}.h5'
            box_cgm -= load_clump(ds, shell_clump_file)
        data_sources = {'cgm': box_cgm}
    else:
        data_sources = {'all': ds.all_data()}

    # Set unit label
    unit_label = '[photons s$^{-1}$ cm$^{-2}$ sr$^{-1}$]' if unit_system == 'default' else '[erg s$^{-1}$ cm$^{-2}$ arcsec$^{-2}$]'

    # Create HDF5 file
    save_path = prefix + f'FRBs/res_{bin_size_kpc:.2f}/' 
    os.makedirs(save_path, exist_ok=True)
    f = h5py.File(save_path + halo_name + '_emission_maps' + save_suffix + '.hdf5', 'a')
    grp = f.create_group(f'z={ds.get_parameter("CosmologyCurrentRedshift", 1):.1f}')
    grp.attrs.update({
        "image_extent_kpc": ds.refine_width,
        "redshift": ds.get_parameter("CosmologyCurrentRedshift"),
        "halo_name": halo_name,
        "emission_units": unit_label,
        "gas_density_units": 'g/cm^2',
        "stars_density_units": 'Msun/kpc^2',
        "bin_size_kpc": round_bin_size_kpc,
        "number_of_bins": res
    })

    # Loop through ions
    for region, data_source in data_sources.items():
        for ion in ions:
            print(f"Processing {ion}...")

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
            elif args.Juniper_limit and ion == 'OVI':
                cmap1 = cmr.take_cmap_colors('magma', 7, cmap_range=(0.5, 1), return_fmt='rgba')
                cmap2 = cmr.take_cmap_colors('cmr.neutral_r', 5, cmap_range=(0.0, 0.6), return_fmt='rgba')
                cmap = np.hstack([cmap2, cmap1])
                mymap = mcolors.LinearSegmentedColormap.from_list('cmap', cmap)
            elif args.Juniper_limit and ion == 'CIV':
                cmap1 = cmr.take_cmap_colors('magma', 6, cmap_range=(0.5, 1), return_fmt='rgba')
                cmap2 = cmr.take_cmap_colors('cmr.neutral_r', 7, cmap_range=(0.0, 0.6), return_fmt='rgba')
                cmap = np.hstack([cmap2, cmap1])
                mymap = mcolors.LinearSegmentedColormap.from_list('cmap', cmap)
            else:
                #mymap = cmr.get_sub_cmap('cmr.flamingo', 0.2, 0.8)
                mymap = cmr.get_sub_cmap('magma', 0, 1)
            mymap.set_bad(mymap(0))


            # Edge-on Projection
            proj_edge = yt.ProjectionPlot(ds, ds.x_unit_disk, ('gas', f'Emission_{ions_dict[ion]}'),
                                          center=ds.halo_center_kpc, data_source=data_source,
                                          width=width, north_vector=ds.z_unit_disk,
                                          buff_size=[res, res], method='integrate', weight_field=None)
            proj_edge.set_cmap(f'Emission_{ions_dict[ion]}', mymap)
            proj_edge.set_zlim(f'Emission_{ions_dict[ion]}', zlim_dict[ion][0], zlim_dict[ion][1])
            proj_edge.set_font_size(24)

            if filter_type == 'disk_cgm':
                # Generate projection of the disk-only region
                disk_res= res
                proj_disk = yt.ProjectionPlot(ds, ds.x_unit_disk, ('gas', 'Emission_' + ions_dict[ion]),
                                            center=ds.halo_center_kpc, data_source=disk_cut_region,  # Only use the disk
                                            width=width, north_vector=ds.z_unit_disk, buff_size=[disk_res, disk_res], 
                                            method='integrate', weight_field=None)

                # Get the FRB (disk-only emission)
                frb_disk = proj_disk.frb[('gas', 'Emission_' + ions_dict[ion])].to_ndarray()

                # Compute HI column density in the disk
                proj_hi = yt.ProjectionPlot(ds, ds.x_unit_disk, ('gas', 'H_p0_number_density'),
                                            center=ds.halo_center_kpc, data_source=disk_cut_region,
                                            width=width, north_vector=ds.z_unit_disk, buff_size=[disk_res, disk_res],
                                            method='integrate')

                # Convert to NumPy array
                frb_hi = proj_hi.frb[('gas', 'H_p0_number_density')].to_ndarray()

            # Convert to Matplotlib figure
            fig = proj_edge.plots[f'Emission_{ions_dict[ion]}'].figure
            ax = proj_edge.plots[f'Emission_{ions_dict[ion]}'].axes
            plt.rc('font', family='Aptos')
            fig.patch.set_facecolor("black")  # Set figure background black
            ax.set_facecolor("black")  # Set axis background black

            # Set white labels
            ax.set_xlabel("x (kpc)", fontsize=36, color="white", fontweight='bold')
            ax.set_ylabel("y (kpc)", fontsize=36, color="white", fontweight='bold')

            if filter_type == 'disk_cgm':
            # Define contour levels (adjust levels as needed)
                print('frb_disk max',np.max(frb_hi))
                print('frb_disk min',np.min(frb_hi))
                

                low_percentile = 16    # Avoid extreme low values
                high_percentile = 84   # Capture most of the disk emission
                num_levels = 6         # Number of contour levels

                # Ensure frb_disk is a NumPy array and strip units
                frb_disk_array = frb_hi  # Convert yt unyt_array to a NumPy array

                # Find the minimum nonzero value safely
                min_nonzero_frb = np.min(frb_disk_array[frb_disk_array > 0])  # Ignores zeros

                print("Minimum nonzero frb value:", min_nonzero_frb)


                low_value = np.percentile(min_nonzero_frb, low_percentile)
                high_value = np.percentile(np.max(frb_hi), high_percentile)

                # Ensure values are strictly positive to avoid log(0) errors
                low_value = max(low_value, 1e-3)  # Prevent log(0)
                high_value = max(high_value, low_value * 1.1)  # Ensure valid range

                # Generate logarithmically spaced contour levels
                levels = np.logspace(np.log10(low_value), np.log10(high_value), num_levels)

                print("Contour levels:", levels)

                # Overlay contours on the CGM emission map
                ax.contour(frb_hi, levels=levels, 
                        extent=[-width_value/2, width_value/2, -width_value/2, width_value/2], 
                        colors='cyan', linewidths=2)  # Light green contours
                

            # Make borders (spines) bold and white
            for spine in ax.spines.values():
                spine.set_linewidth(2.5)
                spine.set_color("white")

            # Set ticks outward and white
            ax.tick_params(axis='both', which='major', labelsize=36, length=15, width=3,
                           color="white", labelcolor="white", direction="out",top=False, right=False)
            ax.tick_params(axis='both', which='minor', labelsize=36, length=10, width=2,
                           color="white", labelcolor="white", direction="out",top=False, right=False)

            # Remove existing colorbar
            cbar = proj_edge.plots[f'Emission_{ions_dict[ion]}'].cb
            cbar.remove()

            # Create a new colorbar axis on the right
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.1)

            # Get the mappable object
            mappable = proj_edge.plots[f'Emission_{ions_dict[ion]}'].image
            new_cbar = fig.colorbar(mappable, cax=cax, orientation="vertical")

            # Make colorbar text and border white
            new_cbar.outline.set_linewidth(2.5)
            new_cbar.outline.set_edgecolor("white")
            new_cbar.set_label(f'{label_dict[ion]} Emission {unit_label}', fontsize=26, color="white", labelpad=15, fontweight='bold')

            new_cbar.ax.tick_params(labelsize=24, which='major', length=15, width=3,
                                    color="white", labelcolor="white", direction="out")
            new_cbar.ax.tick_params(labelsize=24, which='minor', length=10, width=2,
                                    color="white", labelcolor="white", direction="out")
            
            # Define target positions in plot coordinates
            target_positions = [
                (5.0, 0.0),    # x, y in kpc
                (-12.0, -8.0)  # x, y in kpc
            ]
            los_colors = ['dodgerblue', 'lime']

            # Add markers using ax.scatter
            for (x, y), color in zip(target_positions, los_colors):
                ax.scatter(x, y, s=400, facecolors='none', edgecolors=color, linewidths=5, marker='s')

            # Save figure
            fig.savefig(save_path + f'{snap}_{ion}_emission_map_edge-on_{region}' + save_suffix + '.png',
                        bbox_inches="tight", dpi=300, facecolor="black")

            # Store data in HDF5
            frb_edge = proj_edge.frb[('gas', f'Emission_{ions_dict[ion]}')]
            grp.create_dataset(f"{ion}_emission_edge_{region}", data=frb_edge)

            # Face-on Projection
            proj_face = yt.ProjectionPlot(ds, ds.z_unit_disk, ('gas', f'Emission_{ions_dict[ion]}'),
                                          center=ds.halo_center_kpc, data_source=data_source,
                                          width=width, north_vector=ds.x_unit_disk,
                                          buff_size=[res, res], method='integrate', weight_field=None)

            proj_face.set_cmap(f'Emission_{ions_dict[ion]}', mymap)
            proj_face.set_zlim(f'Emission_{ions_dict[ion]}', zlim_dict[ion][0], zlim_dict[ion][1])
            #proj_face.set_font_size(24)

            # Store data in HDF5
            frb_face = proj_face.frb[('gas', f'Emission_{ions_dict[ion]}')]
            grp.create_dataset(f"{ion}_emission_face_{region}", data=frb_face)

            # Define target positions in plot coordinates
            target_positions = [
                (5.0, 0.0),    # x, y in kpc
                (-12.0, -8.0)  # x, y in kpc
            ]
            los_colors = ['dodgerblue', 'lime']

            # Add markers using ax.scatter
            for (x, y), color in zip(target_positions, los_colors):
                ax.scatter(x, y, s=400, facecolors='none', edgecolors=color, linewidths=4, marker='s')

            # Save face-on plot
            fig.savefig(save_path + f'{snap}_{ion}_emission_map_face-on_{region}' + save_suffix + '.png',
                        bbox_inches="tight", dpi=300, facecolor="black")

    f.close()
    print("Finished generating FRB projections.")

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
            #mymap = cmr.get_sub_cmap('cmr.flamingo', 0.2, 0.8)
            mymap = cmr.get_sub_cmap('hsv', 0, 1)

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


def make_column_density_FRB(ds, refine_box, snap, ions,scaling = True, scale_factor=100, filter_type=None, filter_value=None, resolution=2):
    '''This function calculates and saves projected column density FRBs and total mass for each ion from surface density.'''

    halo_name = halo_dict[str(args.halo)]

    # Determine resolution and bin size
    pix_res = float(np.min(refine_box['dx'].in_units('kpc')))
    bin_size_kpc = resolution*pix_res
    round_bin_size_kpc = round(bin_size_kpc,2)

    # Ensure fov_kpc is in kpc
    if not hasattr(ds.refine_width, 'in_units'):
        fov_kpc = YTQuantity(ds.refine_width, 'kpc')  # Wrap in YTQuantity with units
    else:
        fov_kpc = ds.refine_width.in_units('kpc')  # Convert to kpc if it has units
    
    # Convert to numeric value (without units) for calculations

    width = fov_kpc#(50, 'kpc') 
    width_value = fov_kpc.v#width[0]
    res= int(width_value/bin_size_kpc)

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
        data_sources = {'all': ds.all_data()}
        if filter_type and filter_value:
            if filter_type == 'temperature':
                data_sources['all'] = data_sources['all'].cut_region([f"(obj['gas', 'temperature'] < {filter_value})"])
            elif filter_type == 'density':
                data_sources['all'] = data_sources['all'].cut_region([f"(obj['gas', 'density'] > {filter_value})"])
            else:
                raise ValueError("Unsupported filter type. Supported types: 'temperature', 'density'.")

    # Create HDF5 file for saving mass maps
    save_path = prefix + f'FRBs/res_{bin_size_kpc:.2f}/' 
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

def make_mass_FRB(ds, refine_box, snap, ions, filter_type=None, filter_value=None, resolution=2):
    '''This function calculates and saves mass FRBs and total mass for each ion from surface density.'''

    halo_name = halo_dict[str(args.halo)]

    # Determine resolution and bin size
    pix_res = float(np.min(refine_box['dx'].in_units('kpc')))
    bin_size_kpc = resolution*pix_res
    round_bin_size_kpc = round(bin_size_kpc,2)

    # Ensure fov_kpc is in kpc
    if not hasattr(ds.refine_width, 'in_units'):
        fov_kpc = YTQuantity(ds.refine_width, 'kpc')  # Wrap in YTQuantity with units
    else:
        fov_kpc = ds.refine_width.in_units('kpc')  # Convert to kpc if it has units
    
    # Convert to numeric value (without units) for calculations

    width = (50, 'kpc') #fov_kpc
    width_value = width[0]
    res= int(width_value/bin_size_kpc)

    # Print for debugging
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
        data_sources = {'all': ds.all_data()}
        if filter_type and filter_value:
            if filter_type == 'temperature':
                data_sources['all'] = data_sources['all'].cut_region([f"(obj['gas', 'temperature'] < {filter_value})"])
            elif filter_type == 'density':
                data_sources['all'] = data_sources['all'].cut_region([f"(obj['gas', 'density'] > {filter_value})"])
            else:
                raise ValueError("Unsupported filter type. Supported types: 'temperature', 'density'.")

    # Create HDF5 file for saving mass maps
    save_path = prefix + f'FRBs/res_{bin_size_kpc:.2f}/' 
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

    # Compute pixel area in cm^2
    pixel_area_kpc2 = (fov_kpc / res) ** 2  # Pixel area in kpc^2
    pixel_area_cm2 = pixel_area_kpc2.in_units('cm**2')  # Convert to cm^2

    # Loop through ions and create projections for each region
    for region, data_source in data_sources.items():
        for ion in ions:
            print(f"Processing ion: {ion}")

            # Replace mass field with ion density
            density_field = ('gas', ions_density_dict[ion]) 

            # Edge-on projection (surface density)
            proj_edge = yt.ProjectionPlot(ds, ds.x_unit_disk, density_field,
                                          center=ds.halo_center_kpc, data_source=data_source,
                                          width=width, north_vector=ds.z_unit_disk,
                                          buff_size=[res, res], weight_field=None)
            frb_edge = proj_edge.frb[density_field]  # Surface density in g/cm^2
            frb_edge_mass = (frb_edge * pixel_area_cm2).in_units('Msun') 
            # Compute total mass for edge-on projection
            total_mass_edge = (frb_edge * pixel_area_cm2).sum().in_units('Msun')  # Convert to solar masses

            # Save mass frb and total mass in HDF5
            dset1 = grp.create_dataset(f"{ion}_mass_edge_{region}", data=frb_edge_mass)
            dset1.attrs.create("total_mass_Msun", total_mass_edge)

            # Face-on projection (surface density)
            proj_face = yt.ProjectionPlot(ds, ds.z_unit_disk, density_field,
                                          center=ds.halo_center_kpc, data_source=data_source,
                                          width=width, north_vector=ds.x_unit_disk,
                                          buff_size=[res, res], weight_field=None)
            frb_face = proj_face.frb[density_field]  # Surface density in g/cm^2
            frb_face_mass = (frb_face* pixel_area_cm2).in_units('Msun') 
            # Compute total mass for face-on projection
            total_mass_face = (frb_face * pixel_area_cm2).sum().in_units('Msun')  # Convert to solar masses

            # Save surface density and total mass in HDF5
            dset2 = grp.create_dataset(f"{ion}_mass_face_{region}", data=frb_face_mass)
            dset2.attrs.create("total_mass_Msun", total_mass_face)

            print(f"Edge total mass for {ion}: {total_mass_edge}")
            print(f"Face total mass for {ion}: {total_mass_face}")

    # Close the HDF5 file after saving the datasets
    print('Mass FRBs finished')
    f.close()


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
            emission_map(ds, snap, ions)
        else:
            emission_map_vbins(ds, snap, ions, unit_system=unit_system, filter_type=filter_type, filter_value=filter_value)
    if ('emission_FRB' in args.plot):
        #projection(ds, refine_box, snap, ions, filter_type=filter_type, filter_value=filter_value)
        make_FRB(ds, refine_box, snap, ions, unit_system=unit_system, filter_type=filter_type, filter_value=filter_value, resolution=resolution)
        #make_mass_FRB(ds, refine_box, snap, ions, filter_type=filter_type, filter_value=filter_value, resolution=resolution)
        #make_column_density_FRB(ds, refine_box, snap, ions,scaling = scaling, scale_factor=scale_factor, filter_type=filter_type, filter_value=filter_value, resolution=resolution)
        


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
    prefix = output_dir 
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
    # SiIII 1207
    cloudy_path_thin = code_path + "emission/cloudy_z0_HM05/bertone_run%i.dat"
    hden_pts, T_pts, table_SiIII_1207 = make_Cloudy_table_thin(11,cloudy_path)
    hden_pts, T_pts = np.meshgrid(hden_pts, T_pts)
    pts = np.array((hden_pts.ravel(), T_pts.ravel())).T
    sr_SiIII_1207 = table_SiIII_1207.T.ravel()
    bl_SiIII_1207 = interpolate.LinearNDInterpolator(pts, sr_SiIII_1207)
    register_emission_field_with_unit('Emission_SiIII_1207', Emission_SiIII_1207, emission_units, unit_system,scale_factor,scaling)


    ############################
    ions_dict = {'Lyalpha':'LyAlpha', 'HI':'HAlpha', 'CII': 'CII_1335','CIII':'CIII_1910', 
                 'CIV':'CIV_1548','OVI':'OVI'}
    ions_density_dict = {'Lyalpha':'LyAlpha', 'HI':'H_p0_density', 'CII':'C_p1_density', 'CIII':'C_p2_density',
                 'CIV':'C_p3_density','OVI':'O_p5_density'}
    ions_number_density_dict = {'Lyalpha':'LyAlpha', 'HI':'H_p0_number_density', 'CII':'C_p1_number_density', 'CIII':'C_p2_number_density',
                 'CIV':'C_p3_number_density','OVI':'O_p5_number_density'}
    
    label_dict = {'Lyalpha':r'Ly-$\alpha$', 'HI':r'H$\alpha$', 'CII':'C II','CIII':'C III',
                'CIV':'C IV','OVI':'O VI'}
    
    trident_dict = { 'HI':'H I', 'CII':'C II','CIII':'C III',
                'CIV':'C IV','OVI':'O VI'}

    if unit_system  == 'default':
        #zlim_dict = {'Lyalpha':[1e-1,1e7], 'HI':[1e-1,1e6], 'CII':[1e-7,1e2], 'CIII':[1e-4,1e2],
        #         'CIV':[1e-2,1e4], 'OVI':[1e-2,1e4]}
        zlim_dict = {'Lyalpha':[1e0,1e7], 'HI':[1e0,1e6], 'CII':[1e0,1e5], 'CIII':[1e1,1e5],
                 'CIV':[1e2,1e5], 'OVI':[1e1,1e5]}
    elif unit_system == 'ALT':
        zlim_dict = {'Lyalpha':[1e-22,1e-16], 'HI':[1e-22,1e-16], 'CII':[1e-23,1e-16], 'CIII':[1e-23,1e-16],
                 'CIV':[1e-23,1e-16], 'OVI':[1e-23,1e-16]}
        
        




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

    




