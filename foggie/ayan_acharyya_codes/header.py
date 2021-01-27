#!/usr/bin/env python3

""""

    Title :      header
    Notes :      Header file importing packages/modules required for working with FOGGIE code.
    Author:      Ayan Acharyya
    Started  :   January 2021

"""

import numpy as np
from matplotlib import pyplot as plt
from pathlib import Path
import time

import pandas as pd
pd.set_option('display.max_rows', 50)
pd.set_option('display.max_columns', 50)
pd.set_option('display.width', 1000)

import yt
from yt.units import *
import yt.visualization.eps_writer as eps

import os, sys, argparse, re, subprocess
HOME = os.getenv('HOME') + '/'

from foggie.utils.get_run_loc_etc import *
from foggie.utils.consistency import *
from foggie.utils.yt_fields import *
from foggie.utils.foggie_load import *
