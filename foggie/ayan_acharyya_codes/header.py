#!/usr/bin/env python3

""""

    Title :      header
    Notes :      Header file for importing packages/modulesand parsing args required for working with FOGGIE code.
    Author :     Ayan Acharyya
    Started :    January 2021

"""

import numpy as np
from matplotlib import pyplot as plt
from pathlib import Path
from scipy import optimize as op
from scipy.interpolate import interp1d

from astropy.io import ascii
from operator import itemgetter
from scipy.interpolate import RegularGridInterpolator as RGI
from scipy.interpolate import LinearNDInterpolator as LND
import multiprocessing as mproc

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
pd.set_option('display.max_rows', 50)
pd.set_option('display.max_columns', 50)
pd.set_option('display.width', 1000)

import yt
from yt.units import *
import yt.visualization.eps_writer as eps

import os, sys, argparse, re, subprocess, time, math

from foggie.utils.get_run_loc_etc import *
from foggie.utils.consistency import *
from foggie.utils.yt_fields import *
from foggie.utils.foggie_load import *

# ---------to parse keyword arguments----------
def parse_args(haloname, RDname):
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, description='''identify satellites in FOGGIE simulations''')
    # ---- common args used widely over the full codebase ------------
    parser.add_argument('--system', metavar='system', type=str, action='store', help='Which system are you on? Default is Jase')
    parser.set_defaults(system='ayan_local')

    parser.add_argument('--do', metavar='do', type=str, action='store', help='Which particles do you want to plot? Default is gas')
    parser.set_defaults(do='gas')

    parser.add_argument('--run', metavar='run', type=str, action='store', help='which run? default is natural')
    parser.set_defaults(run="nref11c_nref9f")

    parser.add_argument('--halo', metavar='halo', type=str, action='store', help='which halo? default is 8508 (Tempest)')
    parser.set_defaults(halo=haloname)

    parser.add_argument('--proj', metavar='proj', type=str, action='store', help='Which projection do you want to plot? Default is x')
    parser.set_defaults(proj='x')

    parser.add_argument('--output', metavar='output', type=str, action='store', help='which output? default is RD0020')
    parser.set_defaults(output=RDname)

    parser.add_argument('--pwd', dest='pwd', action='store_true', help='Just use the current working directory?, default is no')
    parser.set_defaults(pwd=False)

    # ------- args added for filter_star_properties.py ------------------------------
    parser.add_argument('--plotmap', dest='plotmap', action='store_true', help='plot projection map? default is no')
    parser.set_defaults(plotmap=False)

    parser.add_argument('--clobber', dest='clobber', action='store_true', help='overwrite existing outputs with same name?, default is no')
    parser.set_defaults(clobber=False)

    # ------- args added for compute_hii_radii.py ------------------------------
    parser.add_argument('--galrad', metavar='galrad', type=float, action='store', help='radius of the galaxy, in kpc, i.e. the radial extent to which computations will be done; default is 50')
    parser.set_defaults(galrad=50.)

    parser.add_argument('--galthick', metavar='galthick', type=float, action='store', help='thickness of stellar disk, in kpc; default is 0.3 kpc')
    parser.set_defaults(galthick=0.3)

    parser.add_argument('--mergeHII', metavar='mergeHII', type=str, action='store', help='separation btwn HII regions below which to merge them, in kpc; default is None i.e., do not merge')
    parser.set_defaults(mergeHII=None)

    parser.add_argument('--galcenter', metavar='galcenter', type=str, action='store', help='1x3 array to store the center of the simulation box, in kpc; default is that of Tempest')
    parser.set_defaults(galcenter=[70484.17266187, 67815.25179856, 73315.10791367]) # from halo_008508/nref11c_nref9f/RD0042

    # ------- args added for lookup_flux.py ------------------------------
    parser.add_argument('--diag_arr', metavar='diag_arr', type=str, action='store', help='list of metallicity diagnostics to use')
    parser.set_defaults(diag_arr='D16')

    parser.add_argument('--Om_arr', metavar='Om_arr', type=float, action='store', help='list of Omega values to use')
    parser.set_defaults(Om_arr=0.5)

    parser.add_argument('--nooutliers', dest='nooutliers', action='store_true', help='discard outlier HII regions (according to D16 diagnostic)?, default is no')
    parser.set_defaults(nooutliers=False)

    parser.add_argument('--xratio', metavar='xratio', type=str, action='store', help='ratio of lines to plot on X-axis; default is None')
    parser.set_defaults(xratio=None)

    parser.add_argument('--yratio', metavar='yratio', type=str, action='store', help='ratio of lines to plot on Y-axis; default is None')
    parser.set_defaults(yratio=None)

    parser.add_argument('--fontsize', metavar='fontsize', type=int, action='store', help='fontsize of plot labels, etc.; default is 15')
    parser.set_defaults(fontsize=15)

    parser.add_argument('--plot_metgrad', dest='plot_metgrad', action='store_true', help='make metallicity gradient plot?, default is no')
    parser.set_defaults(plot_metgrad=False)

    parser.add_argument('--plot_phase_space', dest='plot_phase_space', action='store_true', help='make P-r phase space plot?, default is no')
    parser.set_defaults(plot_phase_space=False)

    parser.add_argument('--plot_obsv_phase_space', dest='plot_obsv_phase_space', action='store_true', help='overlay observed P-r phase space on plot?, default is no')
    parser.set_defaults(plot_obsv_phase_space=False)

    parser.add_argument('--plot_fluxgrid', dest='plot_fluxgrid', action='store_true', help='make flux ratio grid plot?, default is no')
    parser.set_defaults(plot_fluxgrid=False)

    parser.add_argument('--annotate', dest='annotate', action='store_true', help='annotate grid plot?, default is no')
    parser.set_defaults(annotate=False)

    parser.add_argument('--pause', dest='pause', action='store_true', help='pause after annotating each grid?, default is no')
    parser.set_defaults(pause=False)

    parser.add_argument('--plot_Zin_Zout', dest='plot_Zin_Zout', action='store_true', help='make input vs output metallicity plot?, default is no')
    parser.set_defaults(plot_Zin_Zout=False)

    parser.add_argument('--saveplot', dest='saveplot', action='store_true', help='save the plot?, default is no')
    parser.set_defaults(saveplot=False)

    parser.add_argument('--keep', dest='keep', action='store_true', help='keep previously displayed plots on screen?, default is no')
    parser.set_defaults(keep=False)

    parser.add_argument('--use_RGI', dest='use_RGI', action='store_true', help='kuse RGI interpolation vs LND?, default is no')
    parser.set_defaults(use_RGI=False)

    args = parser.parse_args()

    args.diag_arr = [item for item in args.diag_arr.split(',')]
    args.Om_arr = [float(item) for item in str(args.Om_arr).split(',')]
    args.mergeHII_text = '_mergeHII=' + str(args.mergeHII) + 'kpc' if args.mergeHII is not None else '' # to be used as filename suffix to denote whether HII regions have been merged
    args.without_outlier = '_no_outlier' if args.nooutliers else '' # to be used as filename suffix to denote whether outlier HII regions (as per D16 density criteria) have been discarded

    return args

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
