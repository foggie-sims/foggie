'''
Filename: kinematic_analysis.py
Author: Vida
Date created: 1-16-25
Date last modified: 2-10-25

This file contains analysis regarding kinematics and velocities using yt covering grids.

Currently the code results in following plots:

- Mass vs LOS velocity
- Emissivity vs LOS velocity
- Density projection with each LOS annnotated on
- Emission projection with each LOS annnotated on

'''

import numpy as np
import yt
import unyt
from yt import YTArray
from yt.data_objects.level_sets.api import Clump, find_clumps
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
import matplotlib.cm as mtcm
from matplotlib.colors import LogNorm
from scipy.ndimage import gaussian_filter1d

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
    
    parser.add_argument('--save_suffix', metavar='save_suffix', type=str, action='store', \
                        help='Do you want to append a string onto the names of the saved files? Default is no.')
    parser.set_defaults(save_suffix="")

    parser.add_argument('--file_suffix', metavar='file_suffix', type=str, action='store', \
                        help='If plotting from saved surface brightness files, use this to pass the file name suffix.')
    parser.set_defaults(file_suffix="")

    parser.add_argument('--flux_threshold', metavar='flux_threshold', type=str, action='store', \
                        help='What is the detection limit? Default is 1e0')
    parser.set_defaults(flux_threshold=1e0)

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
    
def Emission_CIII_1910(field, data,scale_factor, unit_system='default', scaling  = False):
    H_N = np.log10(np.array(data["H_nuclei_density"]))
    Temperature = np.log10(np.array(data["Temperature"]))
    dia1 = bl_CIII_1910(H_N, Temperature)
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

def Emission_CIV_1548(field, data,scale_factor, unit_system='default', scaling  = False):
    H_N = np.log10(np.array(data["H_nuclei_density"]))
    Temperature = np.log10(np.array(data["Temperature"]))
    dia1 = bl_CIV_1(H_N, Temperature)
    idx = np.isnan(dia1)
    dia1[idx] = -200.
    if scaling == True:
        emission_line = scale_factor * ((10.0**dia1) * ((10.0**H_N)**2.0))
    else:
        emission_line = (10.0**dia1) * ((10.0**H_N)**2.0)
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

def Emission_OVI(field, data,scale_factor, unit_system='default', scaling  = False):

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
        emission_line = emission_line / 4.25e10
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
##################################################################################################### 

def make_velocity_function(mass_field, comp):
    def _velocity_function(field, data):
        # Calculate thermal broadening directly
        k_B = unyt.unyt_quantity(1.38e-16, "erg/K")  # Boltzmann constant
        temperature = data["gas", "temperature"].to("K")
        ion_mass = data["gas", mass_field].to("g")
        b_thermal = np.sqrt((2 * k_B * temperature) / ion_mass).to("cm/s")

        # Add thermal broadening to the bulk velocity
        bulk_velocity = data["gas", comp].to("cm/s")
        return bulk_velocity + b_thermal
    return _velocity_function

def create_covering_grid(ds, halo_center, refine_width, level=0):
    """
    Create a covering grid for the specified refinement region and level,
    ensuring alignment with the underlying simulation cells.

    Parameters:
    -----------
    ds : yt.Dataset
        The yt dataset.
    halo_center : array-like
        The center of the region of interest (in code_length).
    refine_width : float
        The width of the refinement region (in code_length).
    level : int, optional
        The level of the covering grid (default: 0).

    Returns:
    --------
    covering_grid : yt.data_objects.covering_grid.YTCoveringGrid
        The covering grid object.
    """
    # Define the left and right edges of the refinement region
    left_edge = halo_center - 0.5 * refine_width
    right_edge = halo_center + 0.5 * refine_width

    # Ensure the edges are within the dataset's domain bounds
    left_edge = np.maximum(left_edge, 0.0)  # Lower bound of the domain
    right_edge = np.minimum(right_edge, 1.0)  # Upper bound of the domain

    # Snap edges to align with simulation grid boundaries
    cell_size = 1.0 / (ds.domain_dimensions * (2**level))  # Size of a single cell at this level
    left_edge = (left_edge / cell_size).astype(int) * cell_size
    right_edge = (right_edge / cell_size).astype(int) * cell_size

    # Calculate grid dimensions to ensure perfect alignment
    dims = ((right_edge - left_edge) / cell_size).astype(int)

    # Step 1: Define a box from the calculated edges
    refinement_box = ds.box(
        left_edge * ds.domain_width + ds.domain_left_edge,
        right_edge * ds.domain_width + ds.domain_left_edge
    )

    # Step 2: Use the box to create the covering grid
    covering_grid = ds.covering_grid(
        level=level,
        left_edge=refinement_box.left_edge,
        dims=dims
    )


    # Step 3: Convert edges and halo center to kpc for verification
    halo_center_kpc = halo_center * ds.domain_width.in_units("kpc")
    refinement_left_edge_kpc = (left_edge * ds.domain_width + ds.domain_left_edge).in_units("kpc")
    refinement_right_edge_kpc = (right_edge * ds.domain_width + ds.domain_left_edge).in_units("kpc")

    print("Halo Center (kpc):", halo_center_kpc)
    print("Refinement Left Edge (kpc):", refinement_left_edge_kpc)
    print("Refinement Right Edge (kpc):", refinement_right_edge_kpc)

    # Print details for verification
    print("Refinement Region Left Edge (code_length):", left_edge)
    print("Refinement Region Right Edge (code_length):", right_edge)
    print("Grid Dimensions:", dims)
    print("Covering Grid Shape:", covering_grid["density"].shape)
    Grid_size = (refinement_right_edge_kpc - refinement_left_edge_kpc) / covering_grid["density"].shape
    print('Grid size (kpc):', Grid_size)

    return covering_grid, halo_center_kpc, refinement_left_edge_kpc, refinement_right_edge_kpc, Grid_size

def plot_emissivity_histograms(covering_grid, halo_center_kpc, x_id_list, y_id_list,z_id_list, ions_dict, ion_field_names, save_path, bin_width_kms=30):
    """
    Create histograms of velocity where each bin shows emissivity contribution.

    Parameters:
    -----------
    covering_grid : yt.data_objects.covering_grid.YTCoveringGrid
        The covering grid containing simulation data.
    halo_center_kpc : array-like
        The halo center coordinates in kpc.
    x_id_list, y_id_list : list of int
        Lists of grid cell indices in x and y.
    ions_dict : dict
        Dictionary mapping ion field names to emission lines.
    ion_field_names : list of str
        List of ion field names (e.g., ["C_IV", "O_VI"]).
    save_path : str
        Directory where the histograms will be saved.
    bin_width_kms : float
        Desired width of each histogram bin in km/s.

    Returns:
    --------
    None
    """

    def get_emissivity_and_velocity(y_id,z_id, ion_line, ion_field_name):
        """
        Extract emissivity and total velocity for a given ion at specified grid coordinates.

        Parameters:
        -----------
        x_id, y_id : int
            Grid cell indices in x and y.
        ion_field_name : str
            Ion field name (e.g., "C_IV").

        Returns:
        --------
        emissivity : array-like
            Emissivity values for the ion (photon/s/cm^3/sr).
        velocity : array-like
            Total velocities along the z-axis (km/s).
        """
        emissivity = covering_grid["gas", f"Emission_{ion_line}"][y_id][z_id]
        velocity = covering_grid["gas", f"{ion_field_name}_vx_corrected_velocity_with_thermal"][y_id][z_id].to("km/s")
        return emissivity, velocity

    colors = plt.cm.Set1(range(len(y_id_list)+1))

    for ion_field_name in ion_field_names:
        ion_line = ions_dict[ion_field_name]
        print(f'Processing ion: {ion_field_name} ({ion_line})')

        fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(12, 8))
        axes = axes.flatten()

        for i, (z_id, y_id) in enumerate(zip(z_id_list, y_id_list)):
            emissivity, velocity = get_emissivity_and_velocity(y_id,z_id, ion_line, ion_field_name)

            # Filter out very low emissivity values
            idx = np.where(emissivity > 1e-40)
            velocity = velocity[idx]
            emissivity = emissivity[idx]

            # x_coord = covering_grid['gas','x'][y_id][z_id].to('kpc') 
            # x_pos = x_coord - halo_center_kpc[0]
            
            y_coord = covering_grid['gas','y'][y_id][z_id].to('kpc') 
            y_pos = y_coord - halo_center_kpc[1]

            z_coord = covering_grid['gas','z'][y_id][z_id].to('kpc') 
            z_pos = z_coord - halo_center_kpc[2]
            
            #xpos_value = x_pos[0].to_value()
            ypos_value = y_pos[0].to_value()
            zpos_value = z_pos[0].to_value()

            # Calculate bin edges dynamically
            velocity_min = -220#velocity.min()
            velocity_max = 220#velocity.max()
            bin_edges = np.arange(velocity_min, velocity_max + bin_width_kms, bin_width_kms)

            # Compute histogram
            hist, _ = np.histogram(velocity, bins=bin_edges, weights=emissivity)
            bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

            # Plot histogram
            ax = axes[i]
            #ax.bar(bin_centers, hist, width=bin_width_kms, alpha=0.7,
              #     label=f"x={round(xpos_value)}, y={round(ypos_value)}", color=colors[i+1])
            ax.plot(bin_centers, hist, linestyle="-", linewidth=2, label=f"y={round(ypos_value)}, z={round(zpos_value)}", color=colors[i+1])
            ax.set_xlabel("Velocity (Vx) (km/s)", fontsize=12)
            ax.set_ylabel(f"{ion_field_name} Emissivity [photon/s/cm^3/sr]", fontsize=12)
            #ax.set_yscale("log")
            #ax.set_ylim(1e-28,1e-21)
            ax.set_xlim(-250,220)
            ax.grid(True, linestyle="--", alpha=0.7)
            ax.legend(fontsize=10)

        # Turn off unused subplots
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])

        # Adjust layout and save the histogram
        plt.tight_layout()
        plt.savefig(save_path + f'emissivity_histogram_{ion_field_name}.png')
        
def plot_velocity_histograms(covering_grid, halo_center_kpc, x_id_list, y_id_list, ions_dict, ion_field_names, save_path,ion_mass_fields, bin_width_kms=5):
    """
    Create histograms of velocity where each bin shows emissivity contribution.

    Parameters:
    -----------
    covering_grid : yt.data_objects.covering_grid.YTCoveringGrid
        The covering grid containing simulation data.
    halo_center_kpc : array-like
        The halo center coordinates in kpc.
    x_id_list, y_id_list : list of int
        Lists of grid cell indices in x and y.
    ions_dict : dict
        Dictionary mapping ion field names to emission lines.
    ion_field_names : list of str
        List of ion field names (e.g., ["C_IV", "O_VI"]).
    save_path : str
        Directory where the histograms will be saved.
    bin_width_kms : float
        Desired width of each histogram bin in km/s.

    Returns:
    --------
    None
    """

    def get_velocity(x_id, y_id, ion_line, ion_field_name,mass_field):
        """
        Extract emissivity and total velocity for a given ion at specified grid coordinates.

        Parameters:
        -----------
        x_id, y_id : int
            Grid cell indices in x and y.
        ion_field_name : str
            Ion field name (e.g., "C_IV").

        Returns:
        --------
    
        velocity : array-like
            Total velocities along the z-axis (km/s).
        """
        
        velocity = covering_grid["gas", f"{ion_field_name}_vz_corrected_velocity_with_thermal"][x_id][y_id].to("km/s")
        Mass = covering_grid["gas", f"{mass_field}"][x_id][y_id].to("Msun") #[:, y_id, x_id] #O_p5_mass
        return velocity,Mass

    colors = plt.cm.Set1(range(len(x_id_list)+1))

    for ion_field_name,mass_field in ion_mass_fields.items():#zip(ion_field_names,ion_mass_fields.items()):
        ion_line = ions_dict[ion_field_name]
        print(f'Processing ion: {ion_field_name} ({ion_line})')

        fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(12, 8))
        axes = axes.flatten()

        for i, (x_id, y_id) in enumerate(zip(x_id_list, y_id_list)):
            velocity,Mass = get_velocity(x_id, y_id, ion_line, ion_field_name,mass_field)


            x_coord = covering_grid['gas','x'][x_id][y_id].to('kpc') 
            x_pos = x_coord - halo_center_kpc[0]
            
            y_coord = covering_grid['gas','y'][x_id][y_id].to('kpc') 
            y_pos = y_coord - halo_center_kpc[1]

            z_coord = covering_grid['gas','z'][x_id][y_id].to('kpc') 
            z_pos = z_coord - halo_center_kpc[2]
            
            xpos_value = x_pos[0].to_value()
            ypos_value = y_pos[0].to_value()

            # Calculate bin edges dynamically
            velocity_min = -300#velocity.min()
            velocity_max = 300#velocity.max()
            bin_edges = np.arange(velocity_min, velocity_max + bin_width_kms, bin_width_kms)

            # Compute histogram
            hist, _ = np.histogram(velocity, bins=bin_edges, weights=Mass)
            bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

            # Plot histogram
            ax = axes[i]
            # ax.bar(bin_centers, hist, width=bin_width_kms, alpha=0.7,
            #        label=f"x={round(xpos_value)}, y={round(ypos_value)}", color=colors[i+1])
            # Plot histogram with smooth lines
            ax.plot(bin_centers, hist, linestyle="-", linewidth=2, label=f"x={round(xpos_value)}, y={round(ypos_value)}", color=colors[i+1])

            
            ax.set_xlabel("Velocity (Vz) (km/s)", fontsize=12)
            ax.set_ylabel(f"{ion_field_name} Mass ($M_\odot$)", fontsize=12)
            #ax.set_yscale("log")
            #ax.set_ylim(1e-28,1e-21)
            ax.set_xlim(-250,220)
            ax.grid(True, linestyle="--", alpha=0.7)
            ax.legend(fontsize=10)

        # Turn off unused subplots
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])

        # Adjust layout and save the histogram
        plt.tight_layout()
        plt.savefig(save_path + f'velocity_histogram_{ion_field_name}.png')
        
def plot_mass(covering_grid, halo_center_kpc, x_id_list, y_id_list, ions_dict, ion_field_names, save_path,ion_mass_fields):
    """
    Create histograms of mass

    Parameters:
    -----------
    covering_grid : yt.data_objects.covering_grid.YTCoveringGrid
        The covering grid containing simulation data.
    halo_center_kpc : array-like
        The halo center coordinates in kpc.
    x_id_list, y_id_list : list of int
        Lists of grid cell indices in x and y.
    ions_dict : dict
        Dictionary mapping ion field names to emission lines.
    ion_field_names : list of str
        List of ion field names (e.g., ["C_IV", "O_VI"]).
    save_path : str
        Directory where the histograms will be saved.
    bin_width_kms : float
        Desired width of each histogram bin in km/s.

    Returns:
    --------
    None
    """

    def get_mass(x_id, y_id,mass_field, ion_mass_fields):
        """
        Extract mass for a given ion at specified grid coordinates.

        Parameters:
        -----------
        x_id, y_id : int
            Grid cell indices in x and y.
        ion_mass_fields : str
            Ion field name (e.g., "O_5").

        Returns:
        --------
        mass : array-like
            Mass values for the ion (Msun).
        """
    
        Mass = covering_grid["gas", f"{mass_field}"][x_id][y_id].to("Msun") #[:, y_id, x_id] #O_p5_mass

        
        
        return Mass

    colors = plt.cm.Set1(range(len(x_id_list)+1))


    for ion, mass_field in ion_mass_fields.items():
        
        print(f'Processing ion: {ion} ')

        fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(12, 8))
        axes = axes.flatten()

        for i, (x_id, y_id) in enumerate(zip(x_id_list, y_id_list)):
            print('mass field',mass_field)
            mass = get_mass(x_id, y_id,mass_field, ion_mass_fields)

            x_coord = covering_grid['gas','x'][x_id][y_id].to('kpc') 
            x_pos = x_coord - halo_center_kpc[0]
            xpos_value = x_pos[0].to_value()
            
            y_coord = covering_grid['gas','y'][x_id][y_id].to('kpc') 
            y_pos = y_coord - halo_center_kpc[1]
            ypos_value = y_pos[0].to_value()

            z_coord = covering_grid['gas','z'][x_id][y_id].to('kpc') 
            z_pos = z_coord - halo_center_kpc[2]
            
            # Plot histogram
            ax = axes[i]
            ax.plot(z_pos, mass, lw=2, label=f"x={round(xpos_value)}, y={round(ypos_value)}", color=colors[i+1])
     
            ax.set_xlabel("Z (kpc)", fontsize=12)
            ax.set_ylabel(f"{ion} Mass [$M_\odot$]", fontsize=12)
            #ax.set_yscale("log")
            #ax.set_ylim(1e-28,1e-21)
            #ax.set_xlim(-250,220)
            ax.grid(True, linestyle="--", alpha=0.7)
            ax.legend(fontsize=10)

        # Turn off unused subplots
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])

        # Adjust layout and save the histogram
        plt.tight_layout()
        plt.savefig(save_path + f'mass_{ion}.png')
        

#####################################################################Projection Plot#####################################################################
def projection_density(covering_grid,halo_center_kpc,refinement_right_edge_kpc,refinement_left_edge_kpc,x_id_list, y_id_list,save_path,Grid_size):
    # Extract the density field from the covering grid
    cg_density = covering_grid["density"]
    density_unit = str(cg_density.units)

    # Sum (integrate) the density along the chosen axis
    projected_density = np.sum(cg_density, axis=1)

    # Calculate bounds
    refinement_center_kpc = halo_center_kpc
    right = refinement_right_edge_kpc - refinement_center_kpc
    left = refinement_left_edge_kpc - refinement_center_kpc
    x_bounds = [left[1], right[0]]
    y_bounds = [left[1], right[1]]
    

    colors = plt.cm.Set1(range(len(x_id_list)+1))

    # Create the projection plot
    plt.figure(figsize=(10, 8))
    plt.imshow(
        projected_density.T,
        extent=[x_bounds[0], x_bounds[1], y_bounds[0], y_bounds[1]],
        origin="lower",
        aspect="auto",
        cmap="viridis",
        norm=LogNorm(vmin=projected_density[projected_density > 0].min(), vmax=projected_density.max()),
    )
    plt.colorbar(label=f"Projected Density (g/cm^2)")
    plt.xlabel("X (kpc)")
    plt.ylabel("Y (kpc)")

    # Annotate positions on the projection plot
    for i, (x_id, y_id) in enumerate(zip(x_id_list, y_id_list)):
        x_coord = covering_grid['gas','x'][x_id][y_id].to('kpc') 
        x_pos = x_coord - halo_center_kpc[0]
        
        y_coord = covering_grid['gas','y'][x_id][y_id].to('kpc') 
        y_pos = y_coord - halo_center_kpc[1]

        z_coord = covering_grid['gas','z'][x_id][y_id].to('kpc') 
        z_pos = z_coord - halo_center_kpc[2]
        xpos_value = x_pos[0].to_value()
        ypos_value = y_pos[0].to_value()

        # Add a square at the position, matching the line color
        plt.scatter(
            xpos_value, ypos_value, 
            s=50, marker='s', facecolors='none', edgecolors=colors[i+1], linewidths=2, 
            label=f"x={round(xpos_value)}, y={round(ypos_value)}"
        )

    # Annotate grid size on the top-left corner

    plt.text(
        x_bounds[0] + 0.02 * (x_bounds[1] - x_bounds[0]),  # Shift right by 5% of the plot width
        y_bounds[1] - 0.02 * (y_bounds[1] - y_bounds[0]),
        f"Grid Size: {round(float(Grid_size[0]), 1)}", 
        fontsize=12, color="white", backgroundcolor="black", 
        ha="left", va="top", bbox=dict(facecolor="black", edgecolor="none", alpha=0.7)
    )

    # Add a legend for the annotated points
    plt.legend(fontsize=10, loc='upper right')
    plt.tight_layout()
    plt.savefig(save_path + f'grid_projection.png')
    

def projection_emission(covering_grid, halo_center_kpc, refinement_right_edge_kpc, refinement_left_edge_kpc,
                        x_id_list, y_id_list, save_path, Grid_size, ions_dict, zlim_dict, label_dict, unit_system='default'):
    # Define the unit string based on unit_system
    if unit_system == 'default':
        unit_label = '[photons s$^{-1}$ cm$^{-2}$ sr$^{-1}$]'
    elif unit_system == 'ALT':
        unit_label = '[erg s$^{-1}$ cm$^{-2}$ arcsec$^{-2}$]'
    else:
        raise ValueError("Invalid unit_system specified. Use 'default' or 'ALT'.")

    for ion in ions_dict.keys():
        print('Processing ion:', ion)

        # Extract the emission field from the covering grid
        emission_field = covering_grid["gas", f"Emission_{ions_dict[ion]}"]

        # Sum (integrate) the emission along the z-axis
        dz = Grid_size[2]  # Assuming Grid_size[2] corresponds to the z-dimension
        projected_emission = np.trapz(emission_field, dx=dz, axis=2)
        #projected_emission = np.sum(emission_field, axis=2)


        # Calculate bounds
        refinement_center_kpc = halo_center_kpc
        right = refinement_right_edge_kpc - refinement_center_kpc
        left = refinement_left_edge_kpc - refinement_center_kpc
        x_bounds = [left[1], right[0]]
        y_bounds = [left[1], right[1]]

        colors = plt.cm.Set1(range(len(x_id_list)+1))

        # Create the projection plot
        plt.figure(figsize=(10, 8))
        plt.imshow(
            projected_emission.T,
            extent=[x_bounds[0], x_bounds[1], y_bounds[0], y_bounds[1]],
            origin="lower",
            aspect="auto",
            cmap=cmr.get_sub_cmap('cmr.flamingo', 0.2, 0.8),
            norm=LogNorm(vmin=zlim_dict[ion][0], vmax=zlim_dict[ion][1]),
        )
        plt.colorbar(label=f"{label_dict[ion]} Emission ")
        plt.xlabel("X (kpc)")
        plt.ylabel("Y (kpc)")

        # Annotate positions on the projection plot
        for i, (x_id, y_id) in enumerate(zip(x_id_list, y_id_list)):
            x_coord = covering_grid['gas', 'x'][x_id][y_id].to('kpc')
            x_pos = x_coord - halo_center_kpc[0]

            y_coord = covering_grid['gas', 'y'][x_id][y_id].to('kpc')
            y_pos = y_coord - halo_center_kpc[1]

            z_coord = covering_grid['gas', 'z'][x_id][y_id].to('kpc')
            z_pos = z_coord - halo_center_kpc[2]
            xpos_value = x_pos[0].to_value()
            ypos_value = y_pos[0].to_value()

            # Add a square at the position, matching the line color
            plt.scatter(
                xpos_value, ypos_value,
                s=50, marker='s', facecolors='none', edgecolors=colors[i+1], linewidths=2,
                label=f"x={round(xpos_value)}, y={round(ypos_value)}"
            )

        # Annotate grid size on the top-left corner
        plt.text(
            x_bounds[0] + 0.02 * (x_bounds[1] - x_bounds[0]),  # Shift right by 5% of the plot width
            y_bounds[1] - 0.02 * (y_bounds[1] - y_bounds[0]),
            f"Grid Size: {round(float(Grid_size[0]), 2)}",
            fontsize=12, color="white", backgroundcolor="black",
            ha="left", va="top", bbox=dict(facecolor="black", edgecolor="none", alpha=0.7)
        )

        # Add a legend for the annotated points
        plt.legend(fontsize=10, loc='upper right')
        plt.tight_layout()

        # Save the projection plot
        plt.savefig(save_path + f'{ion}_emission_projection.png')
        

def register_ion_mass_fields(ds, ions_dict):
    """Register ion mass fields needed for velocity calculations."""
    for ion, mass_field in ion_mass_fields.items():
        # Check if the mass field exists in the dataset
        if ("gas", mass_field) in ds.derived_field_list:
            print(f"{mass_field} is already available for {ion}.")
        else:
            # Optionally register the missing field (if needed)
            print(f"Field {mass_field} is missing for {ion}. Registering...")
            def _ion_mass(field, data, ion_field=ion_mass_fields[ion]):
                return data[("gas", ion_field.replace("_mass", "_number_density"))] * (ion_mass_dict[ion] * unyt.atomic_mass)

            yt.add_field(
                ("gas", mass_field),
                function=_ion_mass,
                units="g/cm**3",
                sampling_type="cell",
                force_override=True,
            )
            print(f"Registered missing field: {mass_field} for {ion}.")



def register_velocity_fields(ds, ions_dict):
    """Register velocity fields with thermal broadening for each ion."""
    for ion in ions_dict.keys():
        for comp in ['vx_corrected', 'vy_corrected', 'vz_corrected']:
            field_name = f"{ion}_{comp}_velocity_with_thermal"
            ds.add_field(
                ("gas", field_name),
                function=make_velocity_function(ion_mass_fields[ion], comp),
                units="cm/s",
                sampling_type="cell",
                take_log=False,
                force_override=True,
            )
            print(f"Registered velocity field: {field_name}")

#######################################################VELOCITY PLOTS WITHOUT COVERING GRID##################################################################
def emisssion_velocity(ds,refine_box,ions_dict,filter_type=None,shell_count=0):



    save_path = prefix + 'velocity_plots/'
    os.makedirs(save_path, exist_ok=True)  # Ensure the directory exists

    

    # Load velocity and mass data
    if filter_type == 'disk_cgm':
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
    
    velocity_hist_data = {}  # Store histogram data
    raw_data = {}  # Store raw data for debugging

    for region, data_source in data_sources.items():
        velocity_hist_data[region] = {} 
        raw_data[region] = {}
        for ion in ions:
            ion_line = ions_dict[ion]
            emissivity = data_source["gas", f"Emission_{ion_line}"]
            vx = data_source['gas', 'vx_disk'].to("km/s")
            mass = data_source['gas','mass'].to('Msun')
            

            # Compute positions relative to the halo center
            halo_center_kpc = ds.halo_center_kpc
            x = data_source['gas', 'x_disk'].to('kpc')# - halo_center_kpc[0]
            y = data_source['gas', 'y_disk'].to('kpc')# - halo_center_kpc[1]
            z = data_source['gas', 'z_disk'].to('kpc')# - halo_center_kpc[2]
            print('min x:',x.min())
            print('max x:',x.max())
            print('min y:',y.min())
            print('max y:',y.max())

            # Define six (y, z) target positions
            target_positions = [
                (yt.YTQuantity(5.0, "kpc"), yt.YTQuantity(0.0, "kpc")),
                (yt.YTQuantity(-12.0, "kpc"), yt.YTQuantity(-8.0, "kpc"))]

            #     (yt.YTQuantity(-15.0, "kpc"), yt.YTQuantity(-8.0, "kpc")),
            #     (yt.YTQuantity(6.0, "kpc"), yt.YTQuantity(-5.0, "kpc")),
            #     (yt.YTQuantity(-3.0, "kpc"), yt.YTQuantity(-14.0, "kpc")),
            #     (yt.YTQuantity(-17.0, "kpc"), yt.YTQuantity(-8.0, "kpc")),
            #     (yt.YTQuantity(2.0, "kpc"), yt.YTQuantity(2.0, "kpc")),
            #     (yt.YTQuantity(-10.0, "kpc"), yt.YTQuantity(0.0, "kpc")),
            #     (yt.YTQuantity(-10.0, "kpc"), yt.YTQuantity(4.0, "kpc")),
            # ]

            # Set up subplot (2 rows × 3 columns)
            fig, axes = plt.subplots(3, 3, figsize=(6.8, 4.8), sharex=True)#, sharey=True)
            axes = axes.flatten()

            # Use a larger tolerance range instead of an exact match
            tolerance = yt.YTQuantity(0.5, "kpc")  # Increase range to ±0.5 kpc

            # Define a list of unique colors for each plot
            colors = ['dodgerblue','lime']#['blue', 'red', 'green', 'yellow', 'orange', 'brown','lime','cyan','skyblue']
            velocity_hist_data[region][ion] = {}  

            # Loop over all target (y, z) positions and create plots
            updates_target_positions = []
            for i, ((y_target_kpc, z_target_kpc), color) in enumerate(zip(target_positions, colors)):
                # Find indices where y and z are within the range
                matching_indices = np.where(
                    (y_target_kpc - tolerance <= y) & (y <= y_target_kpc + tolerance) &
                    (z_target_kpc - tolerance <= z) & (z <= z_target_kpc + tolerance)
                )
                single_target_y = y[matching_indices][0]
                single_target_z = z[matching_indices][0]
                # Extract corresponding x positions, mass, and velocities
                updates_target_positions.append((single_target_y, single_target_z))

                #matching_indices = np.where((y == single_target_y) & (z == single_target_z))
                x_selected = x[matching_indices]
                y_selected = y[matching_indices] 
                z_selected = z[matching_indices]
                print('y:',y_selected)
                print('lenghth of x:',len(x_selected))
                mass_selected = mass[matching_indices]
                emissivity_selected = emissivity[matching_indices]
                vx_selected = vx[matching_indices]

                # Compute histogram bin edges
                velocity_min = -600  
                velocity_max = 500  
                bin_width_kms = 25  # Slightly larger bin width
                bin_edges = np.arange(velocity_min, velocity_max + bin_width_kms, bin_width_kms)

                # Compute histogram
                hist, _ = np.histogram(vx_selected, bins=bin_edges, weights=emissivity_selected)
                bin_centers = 0.1 * (bin_edges[:-1] + bin_edges[1:])

                # Apply smoothing (choose one method)
                smoothed_hist = gaussian_filter1d(hist, sigma=1.5)  # Gaussian smoothing
                # smoothed_hist = moving_average(hist, window_size=5)  # Moving average

                # Plot smoothed histogram
                

                # Convert unyt_quantity to float before using as dictionary key

                key = (float(y_target_kpc.to_value()), float(z_target_kpc.to_value()))
                if ion not in raw_data[region]:  
                    raw_data[region][ion] = {} 

                raw_data[region][ion][key] = { 
                    "velocity": vx_selected,
                    "emissivity": emissivity_selected
                }


                velocity_hist_data[region][ion][key] = (bin_centers, smoothed_hist)


                # Plot in subplot with a unique color
                axes[i].plot(bin_centers, smoothed_hist, linestyle="-", linewidth=2, color=color)
                axes[i].set_title(f"y={y_target_kpc:.1f} kpc, z={z_target_kpc:.1f} kpc", fontsize=10)
                axes[i].set_xlabel("Velocity (km/s)")
                axes[i].set_ylabel(f"{ion} Emissivity [photons/s/cm^3/sr]")
                

            # Adjust layout and show plot
            plt.tight_layout()
            plt.savefig(save_path + '/' + region + '/' + f'emissivity_velocity_{ion}.png', dpi=150)
            
        return updates_target_positions,data_source,region,velocity_hist_data, raw_data



def plot_velocity_comparison(hist_with_disk, ions, region):
    save_path = prefix + 'velocity_plots/comparison/'
    os.makedirs(save_path, exist_ok=True)

    colors = ['dodgerblue','lime']#['blue', 'red', 'green', 'yellow', 'orange', 'brown','lime','cyan','dodgerblue']

    for ion in ions:
        print('ion:', ion)
        fig, ax = plt.subplots(figsize=(12.4, 4.8))  # Single plot for each ion

        items = list(hist_with_disk[region][ion].items())  # Convert dictionary to list for sequential access

        for i in range(len(items) - 1):  # Stop at second last element to always have a "next" element
            (y_current, z_current), (bin_centers_current, hist_current) = items[i]
            (y_next, z_next), (bin_centers_next, hist_next) = items[i + 1]

            # Plot current line
            ax.plot(bin_centers_current, hist_current, linestyle="-", linewidth=4, 
                    color=colors[0], label=f"y={y_current:.1f}, z={z_current:.1f}")

            # Plot next line for comparison
            ax.plot(bin_centers_next, hist_next, linestyle="-", linewidth=5, 
                    color=colors[1], alpha=1)#, label=f"y={y_next:.1f}, z={z_next:.1f}")

            # Optional: Show only the last legend (so it's not repeated in every loop iteration)
            #if i == len(items) - 2:
            #    ax.legend(fontsize=8, loc="upper right")

        ax.set_xlabel("LOS Velocity (km/s)", fontsize=20)
        ax.set_ylabel(f"{ion} Emissivity\n"
              "[$photons/s/cm^3/sr$]", fontsize=20)
        #ax.set_ylim(0,2.5e-19)
        ax.set_xlim(-600,350)
        # Increase tick label size
        ax.tick_params(axis='both', which='major', labelsize=20, length=10, width=2)  # Thicker tick marks
        ax.tick_params(axis='both', which='minor', labelsize=20, length=5, width=2)  # Optional
        ax.yaxis.get_offset_text().set_fontsize(20)  # Increase scientific notation size

        # Make axis spines thicker (bolder axes)
        line_thickness = 3
        ax.spines['bottom'].set_linewidth(line_thickness)  # X-axis
        ax.spines['left'].set_linewidth(line_thickness)    # Y-axis
        ax.spines['top'].set_linewidth(line_thickness)     # Top border
        ax.spines['right'].set_linewidth(line_thickness)   # Right border (optional)



        plt.tight_layout()
        plt.savefig(save_path + f'velocity_comparison_{ion}.png', dpi=300)
        plt.close(fig)  # Close to prevent overlapping in subsequent iterations


def projection(ds,target_positions,data_source,region):
    save_path = prefix + 'velocity_plots/'
    os.makedirs(save_path, exist_ok=True)  # Ensure the directory exists


    # Create the projection plot
    proj = yt.ProjectionPlot(ds, ds.x_unit_disk, ('gas', 'H_p0_number_density'), center=ds.halo_center_kpc, width=(80., 'kpc'),data_source=data_source)
    #proj.set_zlim('H_p0_number_density', 1e15, 1e23)
    proj.set_cmap(field=('gas', 'H_p0_number_density'), cmap=h1_color_map)

    # Same color list as used in the velocity plots
    colors = ['blue', 'red', 'green', 'yellow', 'orange', 'brown','lime','cyan','skyblue']

    # Loop through each target position and add a marker with the corresponding color
    for (y_target_kpc, z_target_kpc), color in zip(target_positions, colors):
        # Add a square marker at (y_target_kpc, z_target_kpc)
        proj.annotate_marker(
            (y_target_kpc, z_target_kpc), coord_system="plot",
            marker="s", plot_args={"s": 50, "facecolors": "none", "edgecolors": color, "linewidths": 2}
        )


    # Show the projection plot with the colored markers
    proj.save(save_path + '/' + region + '/' + f'projection_HI.png')
   

def emission_projection(ds,target_positions,data_source,region):
    save_path = prefix + 'velocity_plots/'
    os.makedirs(save_path, exist_ok=True)  # Ensure the directory exists


    # Create the projection plot
    mymap = cmr.get_sub_cmap('magma', 0, 1)
    width = (80, 'kpc') #fov_kpc
    width_value = width[0]
    res= int(width_value/0.27)
    proj = yt.ProjectionPlot(ds, ds.x_unit_disk, ('gas', 'Emission_OVI'), center=ds.halo_center_kpc, width=(80., 'kpc'),data_source=data_source, buff_size=[res, res])
    proj.set_zlim('Emission_OVI', 1e2, 1e5)
    proj.set_cmap(field=('gas', 'Emission_OVI'), cmap=mymap)
    #proj.set_colorbar_label(('gas', 'Emission_OVI'), 'Emission_OVI photons/s/cm^2/sr')

    # Same color list as used in the velocity plots
    los_colors = ['dodgerblue','lime']# ['blue', 'red', 'green', 'yellow', 'orange', 'brown','lime','cyan','skyblue']

    # Loop through each target position and add a marker with the corresponding color
    for (y_target_kpc, z_target_kpc), color in zip(target_positions, los_colors):
        # Add a square marker at (y_target_kpc, z_target_kpc)
        proj.annotate_marker(
            (y_target_kpc, z_target_kpc), coord_system="plot",
            marker="s", plot_args={"s": 50, "facecolors": "none", "edgecolors": color, "linewidths": 2}
        )


    # Show the projection plot with the colored markers
    proj.save(save_path + '/' + region + '/' + f'projection_OVI_{region}.png')
    

def save_velocity_hist_to_hdf5(filename, velocity_hist_data, raw_data, filter_type):
    """Save velocity histogram data + raw emissivity & velocity to an HDF5 file."""

    with h5py.File(filename, 'a') as hdf:  # Open in append mode
        filter_group = hdf.require_group(f"{filter_type}")  # 'all' or 'cgm'

        for region, ions_data in velocity_hist_data.items():
            region_group = filter_group.require_group(f"velocity_hist/{region}")  # Create region group
            
            for ion, position_data in ions_data.items():
                ion_group = region_group.require_group(ion)  # Create ion group
                
                for (y_target, z_target), (bin_centers, smoothed_hist) in position_data.items():
                    pos_group = ion_group.require_group(f"y_{y_target:.1f}_z_{z_target:.1f}")

                    # Save bin_centers and smoothed_hist without deleting
                    pos_group.create_dataset("bin_centers", data=bin_centers, shape=bin_centers.shape, dtype=bin_centers.dtype)
                    pos_group.create_dataset("smoothed_hist", data=smoothed_hist, shape=smoothed_hist.shape, dtype=smoothed_hist.dtype)

                    # Save raw_data
                    raw_group = filter_group.require_group(f"raw_data/{region}/{ion}/y_{y_target:.1f}_z_{z_target:.1f}")

                    raw_group.create_dataset("velocity", data=raw_data[region][ion][(y_target, z_target)]["velocity"], shape=raw_data[region][ion][(y_target, z_target)]["velocity"].shape, dtype=raw_data[region][ion][(y_target, z_target)]["velocity"].dtype)
                    
                    raw_group.create_dataset("emissivity", data=raw_data[region][ion][(y_target, z_target)]["emissivity"], shape=raw_data[region][ion][(y_target, z_target)]["emissivity"].shape, dtype=raw_data[region][ion][(y_target, z_target)]["emissivity"].dtype)

    print(f"Saved velocity histograms and raw emissivity/velocity data to {filename}")




#########################################################################################################################

def load_and_calculate(snap, ions):

    snap_name = foggie_dir + 'halo_00' + args.halo + '/' + args.run + '/' + snap + '/' + snap
    
    ds, refine_box = foggie_load(snap_name, trackname, do_filter_particles=True, halo_c_v_name=halo_c_v_name, disk_relative=True, correct_bulk_velocity=True)
    zsnap = ds.get_parameter('CosmologyCurrentRedshift')

    # Add ion fields
    add_ion_fields(ds,ions)
    

    ######################without covering grid###################################################
    

    target_positions,data_source,region,velocity_hist_data_all, raw_data_all = emisssion_velocity(ds,refine_box,ions_dict,filter_type=None,shell_count=0)
    cgm_target_positions,cgm_data_source,cgm_region,velocity_hist_data_cgm, raw_data_cgm = emisssion_velocity(ds,refine_box,ions_dict,filter_type='disk_cgm',shell_count=5)
    projection(ds,target_positions,data_source,region)
    emission_projection(ds,target_positions,data_source,region)

    plot_velocity_comparison(velocity_hist_data_all, ions,region)

    
    hdf5_filename = prefix + 'velocity_plots/velocity_histograms_' + snap + '.h5'
    save_velocity_hist_to_hdf5(hdf5_filename, velocity_hist_data_all, raw_data_all, filter_type="all")
    save_velocity_hist_to_hdf5(hdf5_filename, velocity_hist_data_cgm, raw_data_cgm, filter_type="cgm")
    
   
    
        ##########################################################WITH COVERING GRID#############################################
    # # Register ion mass fields
    # register_ion_mass_fields(ds, ions_dict)

    # # Register velocity fields
    # register_velocity_fields(ds, ions_dict)


    # # Debug: Check registered fields
    # print("Available fields after registration:")
    # print([f for f in ds.derived_field_list if 'velocity_with_thermal' in f[1]])

    # ad = ds.all_data()
    # try:
    #     vel = ad[('gas', 'CIV_vx_corrected_velocity_with_thermal')]
    #     print('Velocity field accessed successfully:', vel)
    # except Exception as e:
    #     print(f"Error accessing velocity field: {e}")
    #     raise

    # #FINAL VERSION OF CONVERTING HALO CENTER AND REFINMENT BOX WIDTH
    # # Step 1: Ensure halo_center is in the same unit as ds.length_unit
    # halo_center = ds.halo_center_kpc.in_units(ds.length_unit.units)

    # # Step 2: Convert refine_width to code_length using ds.length_unit
    # refine_width_kpc = YTQuantity(ds.refine_width, "kpc")
    # refine_width_code_length = refine_width_kpc / ds.length_unit

    # # Attach the length unit explicitly to refine_width_code_length
    # refine_width_code_length = refine_width_code_length * ds.length_unit

    # # convert from Mpccm/h to code_lenghth
    # length_unit = ds.length_unit
    # halo_center = halo_center/length_unit
    # refine_width = refine_width_code_length/length_unit

    #print('Creating covering grid')
    #covering_grid,halo_center_kpc,refinement_left_edge_kpc,refinement_right_edge_kpc,Grid_size = create_covering_grid(ds, halo_center, refine_width, level = 10)
    #plot_emissivity_histograms(covering_grid, halo_center_kpc, x_id_list, y_id_list,z_id_list, ions_dict, ion_field_names, save_path, bin_width_kms=30)
    #plot_velocity_histograms(covering_grid, halo_center_kpc, x_id_list, y_id_list, ions_dict, ion_field_names, save_path,ion_mass_fields, bin_width_kms=30)
    #projection_density(covering_grid,halo_center_kpc,refinement_right_edge_kpc,refinement_left_edge_kpc,z_id_list, y_id_list,save_path,Grid_size)
    #projection_emission(covering_grid, halo_center_kpc, refinement_right_edge_kpc, refinement_left_edge_kpc,x_id_list, y_id_list, save_path, Grid_size, ions_dict, zlim_dict, label_dict, unit_system='default')


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
    clump_file = output_dir + 'ions_halo_00' + args.halo + '/' + args.run + '/' '/Disk/test_Disk.h5'
    
    # Set directory for output location, making it if necessary
    prefix = output_dir 
    if not (os.path.exists(prefix)): os.system('mkdir -p ' + prefix)
    save_path = prefix + f'velocity_plots/'
    os.makedirs(save_path, exist_ok=True)  # Ensure the directory exists
    table_loc = prefix + 'Tables/'

    print('foggie_dir: ', foggie_dir)
    catalog_dir = code_path + 'halo_infos/00' + args.halo + '/' + args.run + '/'
    halo_c_v_name = catalog_dir + 'halo_c_v'
    #smooth_AM_name = catalog_dir + 'AM_direction_smoothed'

    cloudy_path = "/Users/vidasaeedzadeh/Documents/02-Projects/02-FOGGIE/Cloudy-runs/outputs/test-z0/TEST_z0_HM12_sh_run%i.dat"
    #code_path + "emission/cloudy_z0_selfshield/sh_z0_HM12_run%i.dat"
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
    hden_pts, T_pts, table_HA = make_Cloudy_table(2)
    hden_pts, T_pts = np.meshgrid(hden_pts, T_pts)
    pts = np.array((hden_pts.ravel(), T_pts.ravel())).T
    
    sr_HA = table_HA.T.ravel()
    bl_HA = interpolate.LinearNDInterpolator(pts, sr_HA)
    register_emission_field_with_unit('Emission_HAlpha', Emission_HAlpha, emission_units, unit_system,scale_factor,scaling)
    
    ############################
    # Ly-Alpha
    hden_pts, T_pts, table_LA = make_Cloudy_table(1)
    sr_LA = table_LA.T.ravel()
    bl_LA = interpolate.LinearNDInterpolator(pts, sr_LA)
    register_emission_field_with_unit('Emission_LyAlpha', Emission_LyAlpha, emission_units, unit_system,scale_factor,scaling)
    ############################
    # CII 1335
    hden_pts, T_pts, table_CII_1335 = make_Cloudy_table(10)
    sr_CII_1335 = table_CII_1335.T.ravel()
    bl_CII_1335 = interpolate.LinearNDInterpolator(pts, sr_CII_1335)
    register_emission_field_with_unit('Emission_CII_1335', Emission_CII_1335, emission_units, unit_system,scale_factor,scaling)
    
    ############################
    # CIII 977
    hden_pts, T_pts, table_CIII_977 = make_Cloudy_table(7)
    sr_CIII_977 = table_CIII_977.T.ravel()
    bl_CIII_977 = interpolate.LinearNDInterpolator(pts, sr_CIII_977)
    register_emission_field_with_unit('Emission_CIII_977', Emission_CIII_977, emission_units, unit_system,scale_factor,scaling)

    ############################
    # CIII 1910
    hden_pts, T_pts, table_CIII_1910 = make_Cloudy_table(9)
    sr_CIII_1910 = table_CIII_1910.T.ravel()
    bl_CIII_1910 = interpolate.LinearNDInterpolator(pts, sr_CIII_1910)
    register_emission_field_with_unit('Emission_CIII_1910', Emission_CIII_1910, emission_units, unit_system,scale_factor,scaling)

    ############################
    # CIV 1548
    hden_pts, T_pts, table_CIV_1 = make_Cloudy_table(3)
    sr_CIV_1 = table_CIV_1.T.ravel()
    bl_CIV_1 = interpolate.LinearNDInterpolator(pts, sr_CIV_1)
    register_emission_field_with_unit('Emission_CIV_1548', Emission_CIV_1548, emission_units, unit_system,scale_factor,scaling)
    
    ############################
    # O VI (1032 and 1037 combined)
    hden_pts, T_pts, table_OVI_1 = make_Cloudy_table(5)
    hden_pts, T_pts, table_OVI_2 = make_Cloudy_table(6)
    sr_OVI_1 = table_OVI_1.T.ravel()
    sr_OVI_2 = table_OVI_2.T.ravel()
    bl_OVI_1 = interpolate.LinearNDInterpolator(pts, sr_OVI_1)
    bl_OVI_2 = interpolate.LinearNDInterpolator(pts, sr_OVI_2)
    register_emission_field_with_unit('Emission_OVI', Emission_OVI, emission_units, unit_system,scale_factor,scaling)
    
    ############################
    # SiIII 1207
    cloudy_path_thin = code_path + "emission/cloudy_z0_HM05/bertone_run%i.dat"
    hden_pts, T_pts, table_SiIII_1207 = make_Cloudy_table_thin(11)
    hden_pts, T_pts = np.meshgrid(hden_pts, T_pts)
    pts = np.array((hden_pts.ravel(), T_pts.ravel())).T
    sr_SiIII_1207 = table_SiIII_1207.T.ravel()
    bl_SiIII_1207 = interpolate.LinearNDInterpolator(pts, sr_SiIII_1207)
    register_emission_field_with_unit('Emission_SiIII_1207', Emission_SiIII_1207, emission_units, unit_system,scale_factor,scaling)

    ############################
   
    ions_dict = {'HI':'HAlpha', 'CII': 'CII_1335','CIII':'CIII_1910', 
                 'CIV':'CIV_1548','OVI':'OVI'}
    ions_density_dict = {'Lyalpha':'LyAlpha', 'HI':'H_p0_density', 'CII':'C_p1_density', 'CIII':'C_p2_density',
                 'CIV':'C_p3_density','OVI':'O_p5_density'}
    
    label_dict = {'HI':r'H$\alpha$', 'CII':'C II','CIII':'C III',
                'CIV':'C IV','OVI':'O VI'}
    
    trident_dict = {'HI':'H I', 'CII':'C II','CIII':'C III',
                'CIV':'C IV','OVI':'O VI'}
    
    ion_mass_fields = {'HI':'H_p0_mass', 'CII':'C_p1_mass','CIII':'C_p2_mass',
                'CIV':'C_p3_mass','OVI':'O_p5_mass'}

    if unit_system  == 'default':
        zlim_dict = {'Lyalpha':[1e-22,1e-16], 'HI':[1e-22,1e-16], 'CII':[1e-26,1e-16], 'CIII':[1e-26,1e-16],
                 'CIV':[1e-23,1e-16], 'OVI':[1e-23,1e-16]}
    elif unit_system == 'ALT':
        zlim_dict = {'Lyalpha':[1e-22,1e-16], 'HI':[1e-22,1e-16], 'CII':[1e-26,1e-16], 'CIII':[1e-26,1e-16],
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

    ion_field_names = ions

    filter_type = args.filter_type

    # make mass vs surfave brightness 
    if filter_type == 'inflow_outflow':

        regions = {'inflow', 'outflow', 'neither'}

    elif filter_type == 'disk_cgm':

        regions = {'cgm'}   

    else:   
         
         regions = {'all'}
    
    shell_count = int(args.shell_count)



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






