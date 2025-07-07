#!/usr/bin/env python3

"""

    Title :      header
    Notes :      Header file for importing packages/modules and declaring global variables required for working with FOGGIE code.
    Author :     Ayan Acharyya
    Started :    January 2021

"""

import numpy as np
import multiprocessing as mproc
import seaborn as sns
import os
import sys
import argparse
import re
import subprocess
import time
import datetime
import math
import shutil
import copy
import glob
import random
import collections, itertools

from matplotlib import pyplot as plt
plt.style.use('seaborn-whitegrid')
plt.rcParams['axes.edgecolor'] = 'black'
plt.rcParams['axes.linewidth'] = 2
from matplotlib import colors as mplcolors
from matplotlib import patheffects as fx
from matplotlib.colors import LogNorm
from matplotlib import image as mpimg
from matplotlib.path import Path as mpl_Path
from matplotlib import cm as mpl_cm
import mplcyberpunk


from pathlib import Path
from importlib import reload

from mpi4py import MPI

from numpy import exp
from scipy import optimize as op
from scipy import stats
from scipy.interpolate import interp1d
from scipy.interpolate import RegularGridInterpolator as RGI
from scipy.interpolate import LinearNDInterpolator as LND
from scipy.special import erf
from scipy.optimize import curve_fit, fminbound
from scipy.ndimage import gaussian_filter

from astropy.io import ascii, fits
from astropy.table import Table
from astropy.stats import gaussian_fwhm_to_sigma as gf2s
from astropy import convolution as con
from astropy import units as u
from astropy.cosmology import FlatLambdaCDM, Planck13, z_at_value

from operator import itemgetter
from collections import defaultdict
import cmasher as cmr
import datetime
from datetime import timedelta
from uncertainties import ufloat, unumpy

import datashader as dsh
from datashader.utils import export_image
from datashader import transfer_functions as dstf

datashader_ver = float(dsh.__version__.split('.')[1])
if datashader_ver > 11: from datashader.mpl_ext import dsshow

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
pd.set_option('display.max_rows', 50)
pd.set_option('display.max_columns', 50)
pd.set_option('display.width', 1000)

import yt
#yt.toggle_interactivity()
from yt.units import *
#import yt.visualization.eps_writer as eps

from foggie.utils.get_run_loc_etc import *
from foggie.utils.consistency import *
from foggie.utils.yt_fields import *
from foggie.utils.foggie_load import *
from foggie.utils.get_proper_box_size import get_proper_box_size

# ------------declaring constants to be used globally-----------
c = 3e5  # km/s
H0 = 70.  # km/s/Mpc Hubble's constant
planck = 6.626e-27  # ergs.sec Planck's constant
nu = 5e14  # Hz H-alpha frequency to compute photon energy approximately
Mpc_to_m = 3.08e22
Mpc_to_cm = Mpc_to_m * 100
kpc_to_cm = Mpc_to_cm / 1000

#alpha_B = 3.46e-19  # m^3/s OR 3.46e-13 cc/s, Krumholz & Matzner (2009) for 7e3 K
alpha_B = 2.59e-19  # m^3/s OR 2.59e-13 cc/s, for Te = 1e4 K, referee quoted this values
k_B = 1.38e-23  # m^2kg/s^2/K
G = 6.67e-11  # Nm^2/kg^2
eps = 2.176e-18  # Joules or 13.6 eV
m_H = 1.67e-27  # kg; mass of proton

# ------------declaring overall paths (can be modified on a machine/user basis)-----------
HOME = os.getenv('HOME')
try:
    if not os.path.exists(HOME+'/Work/astro/ayan_codes'): # if the code directory does not exist in current home, then it must exist in /pleiades home
        HOME = '/pleiades/u/' + os.getenv('USER')
except:
    pass

#mappings_lab_dir = HOME + '/Mappings/lab/'  # if you are producing the MAPPINGS grid,
mappings_lab_dir = 'MAPPINGS/'  # if you are producing the MAPPINGS grid,
# this is where your MAPPINGS executable .map51 is installed,
# otherwise, this is where your MAPPINGS grid and your emission line list is
#mappings_input_dir = HOME + '/Mappings/HIIGrid306/Q/inputs/'  # if you are producing the MAPPINGS grid,
mappings_input_dir = 'MAPPINGS/'  # if you are producing the MAPPINGS grid,
# this is where your MAPPINGS input/ directory is
# otherwise, ignore this variable
#sb99_dir = HOME + '/SB99-v8-02/output/'  # this is where your Starburst99 model outputs reside
sb99_dir = 'starburst11/'  # this is where your Starburst99 model outputs reside
# this path is used only when you are using compute_hiir_radii.py or lookup_flux.py
sb99_model = 'starburst11'  # for fixed stellar mass input spectra = 1e6 Msun, run up to 10 Myr
sb99_mass = 1e6  # Msun, mass of star cluster in given SB99 model

# ------------declaring list of ALL simulations present locally (on HD)-----------
all_sims_dict = {'8508': [('8508', 'RD0042'), ('8508', 'RD0039'), ('8508', 'RD0031'), ('8508', 'RD0030'), ('8508', 'DD2288'), ('8508', 'DD2289')], \
                 '5036': [('5036', 'RD0039'), ('5036', 'RD0031'), ('5036', 'RD0030'), ('5036', 'RD0020')], \
                 '5016': [('5016', 'RD0042'), ('5016', 'RD0039'), ('5016', 'RD0031'), ('5016', 'RD0030'), ('5016', 'RD0020')], \
                 '4123': [('4123', 'RD0031'), ('4123', 'RD0030')], \
                 '2878': [('2878', 'RD0020'), ('2878', 'RD0018')], \
                 '2392': [('2392', 'RD0030')], \
    } # all snapshots in the HD

projection_dict = {'x': ('y', 'z', 'x'), 'y':('z', 'x', 'y'), 'z':('x', 'y', 'z')} # which axes are projected for which line of sight args.projection

# -----------declaring/modifying colormaps to be ued for certain properties throughout my code------------
# individually comment out following lines to keep the original color_map as defined in foggie.utils.consistency

#density_color_map = 'viridis'

velocity_discrete_cmap = 'coolwarm'

temperature_color_list = ("darkred", "#d73027", "darkorange", "#ffe34d")
temperature_color_map = sns.blend_palette(temperature_color_list, as_cmap=True)

metal_color_list = ("#4575b4", "#984ea3", "#984ea3", "#d73027", "darkorange", "#ffe34d")
#metal_color_list = ("black", "#4575b4", "#984ea3", "#d73027", "darkorange", "#ffe34d")
metal_color_map = sns.blend_palette(metal_color_list, as_cmap=True)
#metal_color_map = 'viridis'

metal_colors_mw = sns.blend_palette(metal_color_list, n_colors=6)
metal_discrete_cmap_mw = mplcolors.ListedColormap(metal_colors_mw)
metal_color_key_mw = collections.OrderedDict()
for i in np.arange(np.size(metal_color_labels_mw)): metal_color_key_mw[metal_color_labels_mw[i]] = to_hex(metal_colors_mw[i])