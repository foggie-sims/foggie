#!/usr/bin/env python3

"""

    Title :      header
    Notes :      Header file for importing packages/modules and declaring global variables required for working with FOGGIE code.
    Author :     Ayan Acharyya
    Started :    January 2021

"""

import numpy as np
import multiprocessing as mproc
import os, sys, argparse, re, subprocess, time, math, shutil

from matplotlib import pyplot as plt
from matplotlib import patheffects as fx
plt.style.use('seaborn')

from pathlib import Path
from importlib import reload

from scipy import optimize as op, exp
from scipy.interpolate import interp1d
from scipy.interpolate import RegularGridInterpolator as RGI
from scipy.interpolate import LinearNDInterpolator as LND
from scipy.special import erf
from scipy.optimize import curve_fit, fminbound

from astropy.io import ascii, fits
from astropy.table import Table
from astropy.stats import gaussian_fwhm_to_sigma as gf2s
from astropy import convolution as con

from operator import itemgetter

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
pd.set_option('display.max_rows', 50)
pd.set_option('display.max_columns', 50)
pd.set_option('display.width', 1000)

import yt
from yt.units import *
import yt.visualization.eps_writer as eps

from foggie.utils.get_run_loc_etc import *
from foggie.utils.consistency import *
from foggie.utils.yt_fields import *
from foggie.utils.foggie_load import *

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
mappings_lab_dir = HOME + '/Mappings/lab/' # if you are producing the MAPPINGS grid,
                                          # this is where your MAPPINGS executable .map51 is installed,
                                          # otherwise, this is where your MAPPINGS grid and your emission line list is
mappings_input_dir = HOME + '/Mappings/HIIGrid306/Q/inputs/' # if you are producing the MAPPINGS grid,
                                                             # this is where your MAPPINGS input/ directory is
                                                             # otherwise, ignore this variable
sb99_dir = HOME + '/SB99-v8-02/output/' # this is where your Starburst99 model outputs reside
                                        # this path is used only when you are using compute_hiir_radii.py or lookup_flux.py
sb99_model = 'starburst11'  # for fixed stellar mass input spectra = 1e6 Msun, run up to 10 Myr
sb99_mass = 1e6 # Msun, mass of star cluster in given SB99 model

projection_dict = {'x': ('y', 'z'), 'y':('z', 'x'), 'z':('x', 'y')} # which axes are projected for which line of sight args.projection

# ------------declaring list of ALL simulations-----------
#all_sims = [('8508', 'RD0042'), ('5036', 'RD0039'), ('5016', 'RD0042'), ('4123', 'RD0031'), ('2878', 'RD0020'), ('2392', 'RD0030')] # only the latest (lowest z) available snapshot for each halo

#all_sims = [('8508', 'RD0030'), ('5036', 'RD0030'), ('5016', 'RD0030'), ('4123', 'RD0030'), ('2392', 'RD0030')] # all same z (=0.7) snapshots for each halo

all_sims = [('8508', 'RD0042'), ('8508', 'RD0039'), ('8508', 'RD0031'), ('8508', 'RD0030'), ('8508', 'DD2288'), ('8508', 'DD2289'), \
            ('5036', 'RD0039'), ('5036', 'RD0031'), ('5036', 'RD0030'), ('5036', 'RD0020'), \
            ('5016', 'RD0042'), ('5016', 'RD0039'), ('5016', 'RD0031'), ('5016', 'RD0030'), ('5016', 'RD0020'), \
            ('4123', 'RD0031'), ('4123', 'RD0030'), \
            ('2878', 'RD0020'), ('2878', 'RD0018'), \
            ('2392', 'RD0030'), \
            ] # all snapshots in the HD
