"""
This code is producing output that will later will be used in FOGGIE. XIV kinematic section
"""


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
from itertools import combinations


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
from scipy.ndimage import gaussian_filter1d

from astropy.cosmology import Planck18 as cosmology
from yt.visualization.volume_rendering.off_axis_projection import off_axis_projection

import os
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d


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

def make_Cloudy_table_thin(table_index,cloudy_path):

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
    
def Emission_CIII_977(field, data,scale_factor, unit_system='photons'):
    
    H_N = np.log10(np.array(data["H_nuclei_density"]))
    Temperature = np.log10(np.array(data["Temperature"]))
    dia1 = bl_CIII_977(H_N, Temperature)
    idx = np.isnan(dia1)
    dia1[idx] = -200.
    if scale_factor > 1:
        emission_line = scale_factor * ((10.0**dia1) * ((10.0**H_N)**2.0))
    else:
        emission_line = ((10.0**dia1) * ((10.0**H_N)**2.0))
    emission_line = scale_by_metallicity(emission_line, 0.0, np.log10(np.array(data['metallicity'])))
    
    if unit_system == 'photons':
        emission_line = emission_line / (4. * np.pi * 2.03e-11) # the constant value 2.03e-11 is energy per photon for CIII 977
        return emission_line * ytEmU
    elif unit_system == 'erg':
        emission_line = emission_line / (4. * np.pi)
        emission_line = emission_line / 4.25e10
        return emission_line * ytEmUALT
      
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
    dia1 = bl_CIV_1(H_N, Temperature)
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

def Emission_SiII_1814(field, data,scale_factor, unit_system='photons'):
    
    H_N = np.log10(np.array(data["H_nuclei_density"]))
    Temperature = np.log10(np.array(data["Temperature"]))
    dia1 = bl_SiII_1814(H_N, Temperature)
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
        print("Filtering data for inflow and outflow regions based on radial velocity.")
        disk_cut_region = load_clump(ds, disk_file, source_cut=refine_box)
        box_cgm = refine_box - disk_cut_region
        box_inflow, box_outflow, box_neither = filter_ds(refine_box, segmentation_filter='radial_velocity')
        return {'all': box_cgm, 'inflow': box_inflow, 'outflow': box_outflow, 'neither': box_neither}

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

    #return {'all': refine_box}  # Default case with no filter

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
    #Now data_sources includes all three regions plus the unfiltered data.
    data_sources["all"] = refine_box  # Add full box to get total gas

    # Determine unit label
    unit_label = determine_unit_label(unit_system)

    # Create or open HDF5 file and group
    f, grp = create_or_open_hdf5_group(prefix, halo_name, save_suffix, ds, width, unit_label, round_bin_size_kpc, ions, args)

    return f, grp, data_sources, width_value, res, round_bin_size_kpc, bin_size_cm, unit_label  # Return file, group, and data sources for further processing


#####################################################################################################   

def emission_velocity_with_pixel_size(ds, refine_box, ions, ions_dict, prefix,
                                      res_kpc, filter_value=None, filter_type=None,
                                      shell_count=0, bin_width_kms_list=[25],
                                      apply_smoothing=False, sigma=1.5):
    """
    Extracts emissivity-weighted LOS velocity histograms for fixed LOS positions,
    using square regions of side length = res_kpc to simulate different pixel sizes.

    Parameters:
        res_kpc (float): Physical size of the pixel in kpc (used to define the LOS extraction box).
    """
    import os
    from scipy.ndimage import gaussian_filter1d

    save_path = prefix + 'velocity_plots/'
    os.makedirs(save_path, exist_ok=True)

    f, grp, data_sources, width_value, res, min_res, bin_size_cm, unit_label = process_emission_maps(
        args, ds, refine_box, halo_dict, filter_type, filter_value,
        disk_file, shell_count, shell_path, unit_system, prefix, save_suffix, ions)

    velocity_hist_data_all_bins = {}
    raw_data = {}

    # Use fixed target positions (same for all resolutions)
    # target_positions = [
    #     (yt.YTQuantity(0, "kpc"), yt.YTQuantity(5, "kpc")),
    #     (yt.YTQuantity(0, "kpc"), yt.YTQuantity(10, "kpc")),
    #     (yt.YTQuantity(0, "kpc"), yt.YTQuantity(15, "kpc")),
    #     (yt.YTQuantity(0, "kpc"), yt.YTQuantity(20, "kpc")),
    #     (yt.YTQuantity(5, "kpc"), yt.YTQuantity(0, "kpc")),
    #     (yt.YTQuantity(5, "kpc"), yt.YTQuantity(5, "kpc")),
    #     (yt.YTQuantity(5, "kpc"), yt.YTQuantity(10, "kpc")),
    #     (yt.YTQuantity(5, "kpc"), yt.YTQuantity(15, "kpc")),
    #     (yt.YTQuantity(5, "kpc"), yt.YTQuantity(20, "kpc")),
    #     (yt.YTQuantity(10, "kpc"), yt.YTQuantity(0, "kpc")),
    #     (yt.YTQuantity(10, "kpc"), yt.YTQuantity(5, "kpc")),
    #     (yt.YTQuantity(10, "kpc"), yt.YTQuantity(10, "kpc")),
    #     (yt.YTQuantity(10, "kpc"), yt.YTQuantity(15, "kpc")),
    #     (yt.YTQuantity(10, "kpc"), yt.YTQuantity(20, "kpc")),
    #     (yt.YTQuantity(15, "kpc"), yt.YTQuantity(0, "kpc")),
    #     (yt.YTQuantity(15, "kpc"), yt.YTQuantity(5, "kpc")),
    #     (yt.YTQuantity(15, "kpc"), yt.YTQuantity(10, "kpc")),
    #     (yt.YTQuantity(15, "kpc"), yt.YTQuantity(15, "kpc")),
    #     (yt.YTQuantity(15, "kpc"), yt.YTQuantity(20, "kpc")),
    #     (yt.YTQuantity(20, "kpc"), yt.YTQuantity(0, "kpc")),
    #     (yt.YTQuantity(20, "kpc"), yt.YTQuantity(5, "kpc")),
    #     (yt.YTQuantity(20, "kpc"), yt.YTQuantity(10, "kpc")),
    #     (yt.YTQuantity(20, "kpc"), yt.YTQuantity(15, "kpc")),
    #     (yt.YTQuantity(20, "kpc"), yt.YTQuantity(20, "kpc"))]
    
    # target_positions = [
    #     (yt.YTQuantity(0, "kpc"), yt.YTQuantity(-5, "kpc")),
    #     (yt.YTQuantity(0, "kpc"), yt.YTQuantity(-10, "kpc")),
    #     (yt.YTQuantity(0, "kpc"), yt.YTQuantity(-15, "kpc")),
    #     (yt.YTQuantity(0, "kpc"), yt.YTQuantity(-20, "kpc")),
    #     (yt.YTQuantity(5, "kpc"), yt.YTQuantity(-0, "kpc")),
    #     (yt.YTQuantity(5, "kpc"), yt.YTQuantity(-5, "kpc")),
    #     (yt.YTQuantity(5, "kpc"), yt.YTQuantity(-10, "kpc")),
    #     (yt.YTQuantity(5, "kpc"), yt.YTQuantity(-15, "kpc")),
    #     (yt.YTQuantity(5, "kpc"), yt.YTQuantity(-20, "kpc")),
    #     (yt.YTQuantity(10, "kpc"), yt.YTQuantity(-0, "kpc")),
    #     (yt.YTQuantity(10, "kpc"), yt.YTQuantity(-5, "kpc")),
    #     (yt.YTQuantity(10, "kpc"), yt.YTQuantity(-10, "kpc")),
    #     (yt.YTQuantity(10, "kpc"), yt.YTQuantity(-15, "kpc")),
    #     (yt.YTQuantity(10, "kpc"), yt.YTQuantity(-20, "kpc")),
    #     (yt.YTQuantity(15, "kpc"), yt.YTQuantity(-0, "kpc")),
    #     (yt.YTQuantity(15, "kpc"), yt.YTQuantity(-5, "kpc")),
    #     (yt.YTQuantity(15, "kpc"), yt.YTQuantity(-10, "kpc")),
    #     (yt.YTQuantity(15, "kpc"), yt.YTQuantity(-15, "kpc")),
    #     (yt.YTQuantity(15, "kpc"), yt.YTQuantity(-20, "kpc")),
    #     (yt.YTQuantity(20, "kpc"), yt.YTQuantity(-0, "kpc")),
    #     (yt.YTQuantity(20, "kpc"), yt.YTQuantity(-5, "kpc")),
    #     (yt.YTQuantity(20, "kpc"), yt.YTQuantity(-10, "kpc")),
    #     (yt.YTQuantity(20, "kpc"), yt.YTQuantity(-15, "kpc")),
    #     (yt.YTQuantity(20, "kpc"), yt.YTQuantity(-20, "kpc"))]

    # target_positions = [
    #     (yt.YTQuantity(0, "kpc"), yt.YTQuantity(5, "kpc")),
    #     (yt.YTQuantity(0, "kpc"), yt.YTQuantity(10, "kpc")),
    #     (yt.YTQuantity(0, "kpc"), yt.YTQuantity(15, "kpc")),
    #     (yt.YTQuantity(0, "kpc"), yt.YTQuantity(20, "kpc")),
    #     (yt.YTQuantity(-5, "kpc"), yt.YTQuantity(0, "kpc")),
    #     (yt.YTQuantity(-5, "kpc"), yt.YTQuantity(5, "kpc")),
    #     (yt.YTQuantity(-5, "kpc"), yt.YTQuantity(10, "kpc")),
    #     (yt.YTQuantity(-5, "kpc"), yt.YTQuantity(15, "kpc")),
    #     (yt.YTQuantity(-5, "kpc"), yt.YTQuantity(20, "kpc")),
    #     (yt.YTQuantity(-10, "kpc"), yt.YTQuantity(0, "kpc")),
    #     (yt.YTQuantity(-10, "kpc"), yt.YTQuantity(5, "kpc")),
    #     (yt.YTQuantity(-10, "kpc"), yt.YTQuantity(10, "kpc")),
    #     (yt.YTQuantity(-10, "kpc"), yt.YTQuantity(15, "kpc")),
    #     (yt.YTQuantity(-10, "kpc"), yt.YTQuantity(20, "kpc")),
    #     (yt.YTQuantity(-15, "kpc"), yt.YTQuantity(0, "kpc")),
    #     (yt.YTQuantity(-15, "kpc"), yt.YTQuantity(5, "kpc")),
    #     (yt.YTQuantity(-15, "kpc"), yt.YTQuantity(10, "kpc")),
    #     (yt.YTQuantity(-15, "kpc"), yt.YTQuantity(15, "kpc")),
    #     (yt.YTQuantity(-15, "kpc"), yt.YTQuantity(20, "kpc")),
    #     (yt.YTQuantity(-20, "kpc"), yt.YTQuantity(0, "kpc")),
    #     (yt.YTQuantity(-20, "kpc"), yt.YTQuantity(5, "kpc")),
    #     (yt.YTQuantity(-20, "kpc"), yt.YTQuantity(10, "kpc")),
    #     (yt.YTQuantity(-20, "kpc"), yt.YTQuantity(15, "kpc")),
    #     (yt.YTQuantity(-20, "kpc"), yt.YTQuantity(20, "kpc"))]

    # target_positions = [
    #     (yt.YTQuantity(0, "kpc"), yt.YTQuantity(-5, "kpc")),
    #     (yt.YTQuantity(0, "kpc"), yt.YTQuantity(-10, "kpc")),
    #     (yt.YTQuantity(0, "kpc"), yt.YTQuantity(-15, "kpc")),
    #     (yt.YTQuantity(0, "kpc"), yt.YTQuantity(-20, "kpc")),
    #     (yt.YTQuantity(-5, "kpc"), yt.YTQuantity(0, "kpc")),
    #     (yt.YTQuantity(-5, "kpc"), yt.YTQuantity(-5, "kpc")),
    #     (yt.YTQuantity(-5, "kpc"), yt.YTQuantity(-10, "kpc")),
    #     (yt.YTQuantity(-5, "kpc"), yt.YTQuantity(-15, "kpc")),
    #     (yt.YTQuantity(-5, "kpc"), yt.YTQuantity(-20, "kpc")),
    #     (yt.YTQuantity(-10, "kpc"), yt.YTQuantity(0, "kpc")),
    #     (yt.YTQuantity(-10, "kpc"), yt.YTQuantity(-5, "kpc")),
    #     (yt.YTQuantity(-10, "kpc"), yt.YTQuantity(-10, "kpc")),
    #     (yt.YTQuantity(-10, "kpc"), yt.YTQuantity(-15, "kpc")),
    #     (yt.YTQuantity(-10, "kpc"), yt.YTQuantity(-20, "kpc")),
    #     (yt.YTQuantity(-15, "kpc"), yt.YTQuantity(0, "kpc")),
    #     (yt.YTQuantity(-15, "kpc"), yt.YTQuantity(-5, "kpc")),
    #     (yt.YTQuantity(-15, "kpc"), yt.YTQuantity(-10, "kpc")),
    #     (yt.YTQuantity(-15, "kpc"), yt.YTQuantity(-15, "kpc")),
    #     (yt.YTQuantity(-15, "kpc"), yt.YTQuantity(-20, "kpc")),
    #     (yt.YTQuantity(-20, "kpc"), yt.YTQuantity(0, "kpc")),
    #     (yt.YTQuantity(-20, "kpc"), yt.YTQuantity(-5, "kpc")),
    #     (yt.YTQuantity(-20, "kpc"), yt.YTQuantity(-10, "kpc")),
    #     (yt.YTQuantity(-20, "kpc"), yt.YTQuantity(-15, "kpc")),
    #     (yt.YTQuantity(-20, "kpc"), yt.YTQuantity(-20, "kpc"))]

    # target_positions = [
    #     (yt.YTQuantity(-10, "kpc"), yt.YTQuantity(5, "kpc")),
    #     (yt.YTQuantity(-20, "kpc"), yt.YTQuantity(-20, "kpc")),
    #     (yt.YTQuantity(-5, "kpc"), yt.YTQuantity(20, "kpc")),
    #     (yt.YTQuantity(-15, "kpc"), yt.YTQuantity(0, "kpc")),
    #     (yt.YTQuantity(-15, "kpc"), yt.YTQuantity(10, "kpc")),
    #     (yt.YTQuantity(-20, "kpc"), yt.YTQuantity(15, "kpc")),
    #     (yt.YTQuantity(20, "kpc"), yt.YTQuantity(-20, "kpc")),
    #     (yt.YTQuantity(20, "kpc"), yt.YTQuantity(-10, "kpc")),
    #     (yt.YTQuantity(20, "kpc"), yt.YTQuantity(15, "kpc")),
    #     (yt.YTQuantity(25, "kpc"), yt.YTQuantity(25, "kpc")),
    #     (yt.YTQuantity(30, "kpc"), yt.YTQuantity(30, "kpc")),
    #     (yt.YTQuantity(35, "kpc"), yt.YTQuantity(35, "kpc")),
    #     (yt.YTQuantity(40, "kpc"), yt.YTQuantity(40, "kpc")),
    #     (yt.YTQuantity(-25, "kpc"), yt.YTQuantity(-25, "kpc")),
    #     (yt.YTQuantity(-30, "kpc"), yt.YTQuantity(-30, "kpc")),
    #     (yt.YTQuantity(-35, "kpc"), yt.YTQuantity(-35, "kpc")),
    #     (yt.YTQuantity(-40, "kpc"), yt.YTQuantity(-40, "kpc")),
    #     (yt.YTQuantity(25, "kpc"), yt.YTQuantity(-25, "kpc")),
    #     (yt.YTQuantity(30, "kpc"), yt.YTQuantity(-30, "kpc")),
    #     (yt.YTQuantity(35, "kpc"), yt.YTQuantity(-35, "kpc")),
    #     (yt.YTQuantity(40, "kpc"), yt.YTQuantity(-40, "kpc")),
    #     (yt.YTQuantity(-25, "kpc"), yt.YTQuantity(25, "kpc")),
    #     (yt.YTQuantity(-30, "kpc"), yt.YTQuantity(30, "kpc")),
    #     (yt.YTQuantity(-35, "kpc"), yt.YTQuantity(35, "kpc"))]


    target_positions = [
        (yt.YTQuantity(-10, "kpc"), yt.YTQuantity(5, "kpc")),
        (yt.YTQuantity(-20, "kpc"), yt.YTQuantity(-20, "kpc")),
        (yt.YTQuantity(-35, "kpc"), yt.YTQuantity(35, "kpc"))]

    
    
    half_pixel = yt.YTQuantity(float(res_kpc) / 2.0, "kpc")
    print("Regions in data_sources:", data_sources.keys())
    velocity_hist_data = {}
    for ion in ions:
        velocity_hist_data[ion] = {}

    for region, data_source in data_sources.items():
        raw_data[region] = {}

        for bin_width_kms in bin_width_kms_list:
            for ion in ions:
                ion_line = ions_dict[ion]
                emissivity = data_source["gas", f"Emission_{ion_line}"]
                vx = data_source['gas', 'vx_disk'].to("km/s")
                y = data_source['gas', 'y_disk'].to('kpc')
                z = data_source['gas', 'z_disk'].to('kpc')

                #velocity_hist_data[ion] = {}
                raw_data[region][ion] = {}

                for (y_center, z_center) in target_positions:
                    matching_indices = np.where(
                        (y >= y_center - half_pixel) & (y <= y_center + half_pixel) &
                        (z >= z_center - half_pixel) & (z <= z_center + half_pixel)
                    )

                    if len(matching_indices[0]) == 0:
                        continue

                    vx_selected = vx[matching_indices]
                    emissivity_selected = emissivity[matching_indices]

                    velocity_min = -600
                    velocity_max = 500
                    bin_edges = np.arange(velocity_min, velocity_max + bin_width_kms, bin_width_kms)
                    hist, _ = np.histogram(vx_selected, bins=bin_edges, weights=emissivity_selected)
                    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

                    if apply_smoothing:
                        hist = gaussian_filter1d(hist, sigma=sigma)

                    key = (float(y_center.to_value()), float(z_center.to_value()))
                    raw_data[region][ion][key] = {
                        "velocity": vx_selected,
                        "emissivity": emissivity_selected
                    }
                    if key not in velocity_hist_data[ion]:
                        velocity_hist_data[ion][key] = {}
                    velocity_hist_data[ion][key][region] = (bin_centers, hist)

            #print('velocity_hist_data:', velocity_hist_data)
            velocity_hist_data_all_bins[bin_width_kms] = velocity_hist_data
    
    print(f"{ion}: added LOS @ y={y_center:.1f}, z={z_center:.1f}, N={len(matching_indices[0])}")
            


    return target_positions, data_source, region, velocity_hist_data_all_bins, raw_data


def plot_velocity_profiles_grid(hist_data_for_binwidth, ions,
                                bin_width_kms=25, normalize=True,
                                smoothing=False, sigma=1.5,
                                vmax=None, prefix='plots'):
    import math
    import numpy as np
    from scipy.ndimage import gaussian_filter1d  # needed if smoothing=True

    for ion in ions:
        if ion not in hist_data_for_binwidth:
            print(f"Skipping {ion}, no data.")
            continue

        keys = list(hist_data_for_binwidth[ion].keys())
        n_los = len(keys)
        ncols = 1
        nrows = math.ceil(n_los / ncols)

        fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 6, nrows * 2), sharex=True, sharey=True)
        axes = axes.flatten()

        for i, ((y_kpc, z_kpc), region_data) in enumerate(hist_data_for_binwidth[ion].items()):
            ax = axes[i]
            order  = ['all','inflow','outflow']
            styles = {'all':('green', '--', 0.9), 'inflow':('red','-',1), 'outflow':('blue','-',1)}

            show_labels = (i == 0)
            first_handles, first_labels = [], []

            # track emissivity-weighted means ONLY for inflow/outflow
            mu_by_region = {}

            for region in order:
                if region not in region_data:
                    continue

                bin_centers, hist_raw = region_data[region]
                hist = hist_raw.copy()
                hist = np.asarray(hist_raw, dtype=float)

                # --- emissivity-weighted mean BEFORE normalization/smoothing ---
                denom = np.sum(hist_raw)
                if denom > 0:
                    mu = np.sum(hist_raw * bin_centers) / denom
                else:
                    mu = np.nan

                # --- prep curve for plotting ---
                if normalize:
                    norm = np.sum(hist * bin_width_kms)
                    if norm > 0:
                        hist = hist / norm
                if smoothing:
                    hist = gaussian_filter1d(hist, sigma=sigma)

                color, ls, alpha = styles[region]

                # curves
                if region == 'all':
                    ax.fill_between(bin_centers, hist, 0, color=color, alpha=0.15, step=None, zorder=1)
                    line, = ax.plot(bin_centers, hist, color=color, lw=2.5, ls='-', label=('all' if show_labels else None), zorder=4)
                    if show_labels:
                        first_handles.append(line); first_labels.append('all')
                else:
                    line, = ax.plot(bin_centers, hist, color=color, lw=2, ls=ls, alpha=alpha,
                                    label=(region if show_labels else None), zorder=3)
                    if show_labels:
                        first_handles.append(line); first_labels.append(region)

                    # ---- draw mean line ONLY for inflow/outflow ----
                    if np.isfinite(mu):
                        ax.axvline(mu, color=color, ls='--', lw=1.6, alpha=0.9, zorder=5)
                        mu_by_region[region] = float(mu)
                        # legend proxy for mean line (first panel only)
                        if show_labels:
                            hmean, = ax.plot([], [], color=color, ls='--', lw=1.6, label=f'{region} ⟨v⟩')
                            first_handles.append(hmean); first_labels.append(f'{region} ⟨v⟩')

            # --- annotate Δv between outflow and inflow means ---
            # mu_in  = mu_by_region.get('inflow', np.nan)
            # mu_out = mu_by_region.get('outflow', np.nan)
            # if np.isfinite(mu_in) and np.isfinite(mu_out):
            #     dv  = mu_out - mu_in          # signed difference (outflow − inflow)
            #     adv = abs(dv)                 # absolute difference
            #     ax.text(0.98, 0.85, f'|Δv| = {abs(mu_out - mu_in):.0f} km/s',
            #             transform=ax.transAxes, ha='right', va='top', fontsize=12,
            #             bbox=dict(boxstyle='round,pad=0.25', fc='white', ec='none', alpha=0.7))
            # else:
            #     ax.text(0.98, 0.05, 'Δv = n/a', transform=ax.transAxes, ha='right', va='bottom',
            #             fontsize=12, bbox=dict(boxstyle='round,pad=0.25', fc='white', ec='none', alpha=0.7))

            #ax.text(0.5, 0.95, f'y={y_kpc:.1f}, z={z_kpc:.1f}',
                    #transform=ax.transAxes, fontsize=14, ha='center', va='top')
            ax.set_xlim(-400, 400)
            if vmax: ax.set_ylim(0, vmax)
            ax.tick_params(labelsize=16)
            ax.grid(True, linestyle='--', alpha=0.3)

            # stash legend entries from first panel
            if show_labels and len(first_handles) > 0:
                handles, labels = first_handles, first_labels

        # Turn off unused subplots
        for j in range(i + 1, len(axes)):
            axes[j].axis('off')

        #fig.suptitle(f'{ion} LOS Velocity Profiles (bin = {bin_width_kms} km/s)', fontsize=24)
        fig.text(0.5, 0.03, 'Emissivity weighted LOS Velocity (km/s)', ha='center', fontsize=16)
        fig.text(0.03, 0.5, 'Normalized Emissivity' if normalize else 'Emissivity',
                 va='center', rotation='vertical', fontsize=16)

        if 'handles' in locals() and len(handles) > 0:
                # Put legend inside the first subplot (top-left)
                axes[0].legend(handles, labels,
                            loc='upper left',       # relative to axes
                            bbox_to_anchor=(0.03, 0.98),  # fine-tune inside
                            fontsize=10,
                            frameon=True,
                            ncol=1)                  # single column for clarity


        plt.tight_layout(rect=[0.05, 0.05, 1, 0.92])
        save_dir = os.path.join('/Users/vidasaeedzadeh/Projects/foggie_outputs/', 'All_halos', 'velocity_profiles')
        os.makedirs(save_dir, exist_ok=True)
        save_name = f"{ion}_emissivity_velocity_profiles.png"
        plt.savefig(os.path.join(save_dir, save_name), dpi=400)
        plt.close()

def plot_velocity_profiles_grid_gaussian(hist_data_for_binwidth, ions,
                                bin_width_kms=25, normalize=True,
                                smoothing=False,
                                fwhm_kms=None,         # <-- NEW: set this (e.g., 50, 100, 200)
                                sigma_kms=None,        # <-- OR set this directly
                                truncate=4.0,          # kernel half-width (σ*truncate on each side)
                                vmax=None, prefix='plots'):
    

    def _sigma_bins_from_inputs(bin_width_kms, fwhm_kms, sigma_kms):
        """
        Convert either FWHM (km/s) or sigma (km/s) to sigma in *bins* for gaussian_filter1d.
        If neither is provided, default to sigma_bins=1.5/ (bin_width_kms) to roughly match your old behavior.
        """
        if fwhm_kms is not None:
            sigma_kms_local = fwhm_kms / 2.355
        elif sigma_kms is not None:
            sigma_kms_local = sigma_kms
        else:
            sigma_kms_local = 1.5 * bin_width_kms  # roughly what σ=1.5 bins used to do
        sigma_bins = max(sigma_kms_local / bin_width_kms, 1e-6)
        return sigma_bins

    for ion in ions:
        if ion not in hist_data_for_binwidth:
            print(f"Skipping {ion}, no data.")
            continue

        keys = list(hist_data_for_binwidth[ion].keys())
        n_los = len(keys)
        ncols = 1
        nrows = math.ceil(n_los / ncols)

        fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 6, nrows * 2), sharex=True, sharey=True)
        axes = np.atleast_1d(axes).flatten()

        for i, ((y_kpc, z_kpc), region_data) in enumerate(hist_data_for_binwidth[ion].items()):
            ax = axes[i]
            order  = ['all','inflow','outflow']
            styles = {'all':('green', '--', 0.9), 'inflow':('red','-',1), 'outflow':('blue','-',1)}

            show_labels = (i == 0)
            first_handles, first_labels = [], []
            mu_by_region = {}

            for region in order:
                if region not in region_data:
                    continue

                bin_centers, hist_raw = region_data[region]
                hist = hist_raw.astype(float).copy()

                # emissivity-weighted mean BEFORE normalization/smoothing
                denom = np.sum(hist_raw)
                mu = np.sum(hist_raw * bin_centers) / denom if denom > 0 else np.nan

                # --- smoothing ---
                if smoothing:
                    sigma_bins = _sigma_bins_from_inputs(bin_width_kms, fwhm_kms, sigma_kms)
                    # reflect avoids losing flux at edges; truncate controls kernel support
                    hist = gaussian_filter1d(hist, sigma=sigma_bins, mode='reflect', truncate=truncate)

                # --- normalization AFTER smoothing to preserve unit area visually ---
                if normalize:
                    norm = np.sum(hist * bin_width_kms)
                    if norm > 0:
                        hist = hist / norm

                # numeric cleanliness for fill_between
                hist = np.maximum(hist, 0.0)

                color, ls, alpha = styles[region]

                if region == 'all':
                    ax.fill_between(bin_centers, hist, 0, color=color, alpha=0.15, step=None, zorder=1)
                    line, = ax.plot(bin_centers, hist, color=color, lw=2.5, ls='-', label=('all' if show_labels else None), zorder=4)
                    if show_labels:
                        first_handles.append(line); first_labels.append('all')
                else:
                    line, = ax.plot(bin_centers, hist, color=color, lw=2, ls=ls, alpha=alpha,
                                    label=(region if show_labels else None), zorder=3)
                    if show_labels:
                        first_handles.append(line); first_labels.append(region)

                    # mean markers (computed from *raw* distribution)
                    if np.isfinite(mu):
                        ax.axvline(mu, color=color, ls='--', lw=1.6, alpha=0.9, zorder=5)
                        mu_by_region[region] = float(mu)
                        if show_labels:
                            hmean, = ax.plot([], [], color=color, ls='--', lw=1.6, label=f'Mean')
                            first_handles.append(hmean); first_labels.append(f'Mean')

            ax.set_xlim(-400, 400)
            if vmax is not None:
                ax.set_ylim(0, vmax)
            ax.tick_params(labelsize=16)
            ax.tick_params(axis='x', labelsize=16, width=1.5, length=6)
            #ax.grid(True, linestyle='--', alpha=0.3)
            
            # hide y-axis values and ticks
            ax.tick_params(left=False, labelleft=False)

            if show_labels and len(first_handles) > 0:
                handles, labels = first_handles, first_labels

        # Turn off any unused subplots
        for j in range(i + 1, len(axes)):
            axes[j].axis('off')

        fig.text(0.5, 0.03, 'Emissivity-weighted LOS Velocity (km/s)', ha='center', fontsize=16)
        fig.text(0.03, 0.5, 'Normalized Emissivity' if normalize else 'Emissivity',
                 va='center', rotation='vertical', fontsize=16)

        if 'handles' in locals() and len(handles) > 0:
            axes[0].legend(handles, labels,
                           loc='upper left',
                           bbox_to_anchor=(0.03, 0.98),
                           fontsize=10,
                           frameon=True,
                           ncol=1)

        plt.tight_layout(rect=[0.05, 0.05, 1, 0.92])
        save_dir = os.path.join('/Users/vidasaeedzadeh/Projects/foggie_outputs/', 'All_halos', 'velocity_profiles')
        os.makedirs(save_dir, exist_ok=True)
        save_name = f"{ion}_emissivity_velocity_profiles_gaussian.png"
        plt.savefig(os.path.join(save_dir, save_name), dpi=400)
        plt.close()



def plot_emission_maps_with_sightlines(ds, refine_box, snap, ions, target_positions, prefix,filter_value=None, save_suffix=''):
    """
    Generate and save emission projection maps for multiple ions, regions, and data sources with sightlines annotated.

    Parameters:
        ds (yt.Dataset): The dataset object
        ions (list): List of ions to plot (e.g., ['OVI', 'CIII'])
        target_positions (list): List of (y, z) tuples for edge-on sightline annotations
        data_sources (dict): Dictionary of {region_name: yt data container}
        prefix (str): Output path prefix
        save_suffix (str): Optional suffix for saved file names
    """
    print("Generating emission projection maps with sightlines...")
    save_path = os.path.join(prefix, 'velocity_plots')
    os.makedirs(save_path, exist_ok=True)

    f, grp, data_sources, width_value, res, min_res, bin_size_cm, unit_label = process_emission_maps(
        args, ds, refine_box, halo_dict, filter_type, filter_value,
        disk_file, shell_count, shell_path, unit_system, prefix, save_suffix, ions)

    mymap = cmr.get_sub_cmap('cmr.flamingo', 0.2, 1)
    width_kpc = 80.0
    width = (width_kpc, 'kpc')
    res = int(width_kpc / 0.27)

    for region, data_source in data_sources.items():
        region_path = os.path.join(save_path, region)
        os.makedirs(region_path, exist_ok=True)

        for ion in ions:
            print(f"Creating emission map for {ion} in region '{region}'")

            field = ('gas', f'Emission_{ions_dict[ion]}')

            proj = yt.ProjectionPlot(ds, ds.x_unit_disk, field,
                                     center=ds.halo_center_kpc,
                                     width=width, data_source=data_source,
                                     buff_size=(res, res))
            
            # Dynamic zlim depending on ion (or use a dict for better control)
            proj.set_zlim(field, 1e2, 1e5)
            proj.set_cmap(field, cmap=mymap)
            proj.set_colorbar_label(field, f"{ion} Emission [photons s$^{{-1}}$ cm$^{{-2}}$ sr$^{{-1}}$]")

            # Annotate sightlines
            for i, (y_kpc, z_kpc) in enumerate(target_positions):
                color = f"C{i % 10}"  # Use matplotlib's built-in cycle
                proj.annotate_marker(
                    (y_kpc, z_kpc), coord_system="plot",
                    marker="s", plot_args={"s": 50, "facecolors": "none",
                                           "edgecolors": color, "linewidths": 2}
                )

            # Save
            save_dir = os.path.join('/Users/vidasaeedzadeh/Projects/foggie_outputs/', 'All_halos', 'velocity_profiles')
            os.makedirs(save_dir, exist_ok=True)
            filename = f"{ion}_emission_projection_edge-on_{region}{save_suffix}.png"
            proj.save(os.path.join(save_dir, filename))
            plt.close()

    print("Finished emission projection maps.")


#########################################################################################################################

def load_and_calculate(snap, ions,scale_factor=None, unit_system='photons', filter_type=None, filter_value=None, res_arcsec=None):
    start_time = time.time()
    '''Loads the simulation snapshot and makes the requested plots, with optional filtering.'''

    # Load simulation output
    snap_name = foggie_dir + 'halo_00' + args.halo + '/' + args.run + '/' + snap + '/' + snap
    ds, refine_box = foggie_load(snap_name, trackfile_name = trackname, do_filter_particles=True, halo_c_v_name=halo_c_v_name, disk_relative=True, correct_bulk_velocity=True)#, smooth_AM_name=smooth_AM_name)
    zsnap = ds.get_parameter('CosmologyCurrentRedshift')

    # Add ion fields
    add_ion_fields(ds,ions)
    
    ######################without covering grid###################################################
    
    velocity_bin_widths = [10]

    target_positions, data_source, region, velocity_hist_data_all_bins, raw_data = emission_velocity_with_pixel_size(ds, refine_box, ions, ions_dict, prefix,
                                      resolution, filter_value=filter_value, filter_type=filter_type,
                                      shell_count=0, bin_width_kms_list=velocity_bin_widths,
                                      apply_smoothing=False, sigma=0)
    

    for bin_width in velocity_bin_widths:
        # this is the function that gives profile with sharp shapes
        # plot_velocity_profiles_grid(velocity_hist_data_all_bins[bin_width], ions,
        #                             bin_width_kms=bin_width, normalize=True,
        #                             smoothing=False, sigma=1.5,
        #                             vmax=None, prefix='plots')
        
        plot_velocity_profiles_grid_gaussian(velocity_hist_data_all_bins[bin_width], ions,
                                bin_width_kms=bin_width, normalize=True,
                                smoothing=True,
                                fwhm_kms=60,         # set this (e.g., 50, 100, 200)
                                sigma_kms=None,        # OR set this directly
                                truncate=3.0,          # kernel half-width (σ*truncate on each side)
                                vmax=None, prefix='plots')
        
    # making emission maps
    #plot_emission_maps_with_sightlines(ds, refine_box, snap, ions, target_positions, prefix,filter_value=filter_value, save_suffix='')


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
                prefix = output_dir + '/FOGGIE' + '/'+ 'RD00' + args.output + '/'+ 'box_' + box_name + '/' + args.filter_type + '/' + resolution + '/' + args.filter_type + '/'

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
    hden_pts, T_pts, table_CIV_1 = make_Cloudy_table(3,cloudy_path)
    sr_CIV_1 = table_CIV_1.T.ravel()
    bl_CIV_1 = interpolate.LinearNDInterpolator(pts, sr_CIV_1)
    register_emission_field_with_unit('Emission_CIV_1548', Emission_CIV_1548, emission_units, unit_system,scale_factor)
    
    ############################
    # O VI (1032 and 1037 combined)
    hden_pts, T_pts, table_OVI_1 = make_Cloudy_table(5,cloudy_path)
    hden_pts, T_pts, table_OVI_2 = make_Cloudy_table(6,cloudy_path)
    sr_OVI_1 = table_OVI_1.T.ravel()
    sr_OVI_2 = table_OVI_2.T.ravel()
    bl_OVI_1 = interpolate.LinearNDInterpolator(pts, sr_OVI_1)
    bl_OVI_2 = interpolate.LinearNDInterpolator(pts, sr_OVI_2)
    register_emission_field_with_unit('Emission_OVI', Emission_OVI, emission_units, unit_system,scale_factor)
    ############################
    # SiII 1814
    
    hden_pts, T_pts, table_SiII_1814 = make_Cloudy_table_thin(11,cloudy_path)
    hden_pts, T_pts = np.meshgrid(hden_pts, T_pts)
    pts = np.array((hden_pts.ravel(), T_pts.ravel())).T
    sr_SiII_1814 = table_SiII_1814.T.ravel()
    bl_SiII_1814 = interpolate.LinearNDInterpolator(pts, sr_SiII_1814)
    register_emission_field_with_unit('Emission_SiII_1814', Emission_SiII_1814, emission_units, unit_system,scale_factor)
    ############################
    # SiIII 1207
    
    hden_pts, T_pts, table_SiIII_1207 = make_Cloudy_table_thin(12,cloudy_path)
    hden_pts, T_pts = np.meshgrid(hden_pts, T_pts)
    pts = np.array((hden_pts.ravel(), T_pts.ravel())).T
    sr_SiIII_1207 = table_SiIII_1207.T.ravel()
    bl_SiIII_1207 = interpolate.LinearNDInterpolator(pts, sr_SiIII_1207)
    register_emission_field_with_unit('Emission_SiIII_1207', Emission_SiIII_1207, emission_units, unit_system,scale_factor)
    ############################
    # SiIV 1394
    
    hden_pts, T_pts, table_SiIV_1394 = make_Cloudy_table_thin(14,cloudy_path)
    hden_pts, T_pts = np.meshgrid(hden_pts, T_pts)
    pts = np.array((hden_pts.ravel(), T_pts.ravel())).T
    sr_SiIV_1394 = table_SiIV_1394.T.ravel()
    bl_SiIV_1394 = interpolate.LinearNDInterpolator(pts, sr_SiIV_1394)
    register_emission_field_with_unit('Emission_SiIV_1394', Emission_SiIV_1394, emission_units, unit_system,scale_factor)
    ############################
    # MgII 2796
    
    hden_pts, T_pts, table_MgII_2796 = make_Cloudy_table_thin(16,cloudy_path)
    hden_pts, T_pts = np.meshgrid(hden_pts, T_pts)
    pts = np.array((hden_pts.ravel(), T_pts.ravel())).T
    sr_MgII_2796 = table_MgII_2796.T.ravel()
    bl_MgII_2796 = interpolate.LinearNDInterpolator(pts, sr_MgII_2796)
    register_emission_field_with_unit('Emission_MgII_2796', Emission_MgII_2796, emission_units, unit_system,scale_factor)
    ############################

    ions_dict = {'Lyalpha':'LyAlpha', 'HI':'HAlpha', 'CII': 'CII_1335','CIII':'CIII_1910', 
                 'CIV':'CIV_1548','OVI':'OVI','SiII':'SiII_1814','SiIII':'SiIII_1207','SiIV':'SiIV_1394','MgII':'MgII_2796'}
    
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
    print("All snapshots finished!")