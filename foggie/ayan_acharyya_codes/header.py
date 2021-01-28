#!/usr/bin/env python3

""""

    Title :      header
    Notes :      Header file for importing packages/modulesand parsing args required for working with FOGGIE code.
    Author:      Ayan Acharyya
    Started  :   January 2021

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
HOME = os.getenv('HOME') + '/'

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
    parser.add_argument('--galrad', metavar='galrad', type=str, action='store', help='radius of stellar disk, in kpc; default is 30 kpc')
    parser.set_defaults(galrad=30.)

    parser.add_argument('--galthick', metavar='galthick', type=str, action='store', help='thickness of stellar disk, in kpc; default is 0.3 kpc')
    parser.set_defaults(galthick=0.3)

    parser.add_argument('--mergeHII', metavar='mergeHII', type=str, action='store', help='separation btwn HII regions below which to merge them, in kpc; default is None i.e., do not merge')
    parser.set_defaults(mergeHII=None)

    parser.add_argument('--galcenter', metavar='galcenter', type=str, action='store', help='1x3 array to store the center of the simulation box, in kpc; default is that of Tempest')
    parser.set_defaults(galcenter=[70484.17266187, 67815.25179856, 73315.10791367]) # from halo_008508/nref11c_nref9f/RD0042

    args = parser.parse_args()
    return args
