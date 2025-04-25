#!/usr/bin/env python3

"""

    Filename :   header.py
    Notes :      Header file for importing packages/modules and declaring global variables required for working with FOGGIE code.
    Authors :    Ayan, Cassi
    Created: 06-12-24
    Last modified: 07-22-24 by Cassi

"""

from __future__ import print_function

import numpy as np
import argparse
import os
import copy
import time

import matplotlib
#matplotlib.use('agg') # Ayan commented this out because it was leading to weird errors while running in ipython
import matplotlib.pyplot as plt

from numpy.polynomial import Polynomial
import multiprocessing as multi
from pathlib import Path
import pandas as pd
import seaborn as sns
#from mpi4py import MPI    # This was causing issues for Cassi's pleiades setup
import scipy
import subprocess

from astropy.table import Table
from astropy.io import ascii

import datashader as dsh
from datashader.utils import export_image
from datashader import transfer_functions as dstf
datashader_ver = float(dsh.__version__.split('.')[1])
if datashader_ver > 11: from datashader.mpl_ext import dsshow

import yt
from yt.units import *
from yt import YTArray
from yt.data_objects.particle_filters import add_particle_filter
from yt.extensions.astro_analysis.halo_analysis import HaloCatalog, add_quantity

from foggie.utils.consistency import *
from foggie.utils.get_refine_box import get_refine_box
from foggie.utils.get_halo_center import get_halo_center
from foggie.utils.get_proper_box_size import get_proper_box_size
from foggie.utils.get_run_loc_etc import get_run_loc_etc
from foggie.utils.yt_fields import *
from foggie.utils.foggie_load import *
from foggie.utils.analysis_utils import *
from foggie.utils.halo_quantity_callbacks import * 

import warnings
warnings.filterwarnings("ignore")

from datetime import datetime, timedelta

from feedback_plots import *
from time_evol_plots import *
from star_formation_plots import *
from visualization_plots import *
from resolved_metallicity_plots import *
from central_info_table import *
from phase_plots import *
from highz_halos_plots import *

