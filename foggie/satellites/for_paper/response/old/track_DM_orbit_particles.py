import astropy
from astropy.io import fits
import numpy as np
from numpy import *
import math
from joblib import Parallel, delayed
import os, sys, argparse
import yt
import matplotlib.pyplot as plt
import trident
import numpy
import foggie
from foggie.utils.get_run_loc_etc import get_run_loc_etc
from foggie.utils.get_refine_box import get_refine_box
from foggie.utils.get_halo_center import get_halo_center
from foggie.utils.get_proper_box_size import get_proper_box_size
import os
import argparse
import numpy as np
from astropy.table import Table
import matplotlib.pyplot as plt
from foggie.utils.consistency import *
from foggie.utils import yt_fields
from scipy.signal import find_peaks  
import yt
from numpy import *
from photutils.segmentation import detect_sources
from yt.units import kpc
from foggie.utils.foggie_load import *
from astropy.io import ascii

def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                     description='''identify satellites in FOGGIE simulations''')
    parser.add_argument('-system', '--system', metavar='system', type=str, action='store', \
                        help='Which system are you on? Default is Jase')
    parser.set_defaults(system="pleiades_raymond")

    parser.add_argument('-simname', '--simname', default=None, help='Simulation to be analyzed.')

    parser.add_argument('-simdir', '--simdir', default='/nobackupp2/mpeeples', help='simulation output directory')

    parser.add_argument('-haloname', '--haloname', default='halo_008508', help='halo_name')

    parser.add_argument('--run', metavar='run', type=str, action='store',
                        help='which run? default is natural')
    parser.set_defaults(run="nref11c_nref9f")

    parser.add_argument('--halo', metavar='halo', type=str, action='store',
                        help='which halo? default is 8508 (Tempest)')
    parser.set_defaults(halo="8508")

    parser.add_argument('--pwd', dest='pwd', action='store_true',
                        help='just use the pwd?, default is no')
    parser.set_defaults(pwd=False)

    parser.add_argument('--output', metavar='output', type=str, action='store',
                        help='which output? default is RD0020')
    parser.set_defaults(output="DD0392")


    parser.add_argument('--save_dir', metavar='save_dir', type=str, action='store',
                        help='directory to save products')
    parser.set_defaults(save_dir="~")

    parser.add_argument('--use_halo_c_v', dest='use_halo_c_v', action='store_true',
                        help='just use the pwd?, default is no')
    parser.set_defaults(use_halo_c_v=False)

    args = parser.parse_args()
    return args




if __name__ == '__main__':
    args = parse_args()
    ds, refine_box = load_sim(args)
    dm_particles = np.load('/nobackupp2/rcsimons/git/foggie/foggie/satellites/for_paper/response/DM_ids.npy', allow_pickle = True)[()]
    save_name = ''
    full_sphere =  ds.sphere(ds.halo_center_kpc, ds.arr(100.,'kpc'))

    dm_ids = full_sphere['dm', 'particle_index']
    xcorr  = full_sphere['dm', 'particle_position_relative_x']
    ycorr  = full_sphere['dm', 'particle_position_relative_y']
    zcorr  = full_sphere['dm', 'particle_position_relative_z']
    vxcorr = full_sphere['dm', 'particle_velocity_relative_x']
    vycorr = full_sphere['dm', 'particle_velocity_relative_y']
    vzcorr = full_sphere['dm', 'particle_velocity_relative_z']

    results = {}
    for i in np.arange(100):
        dm_id_i = a[i]['id']
        gd = np.where(dm_id_i == dm_ids)[0]

        if len(gd) > 0:
            results[i]['xcorr']  = xcorr[gd]
            results[i]['ycorr']  = ycorr[gd]
            results[i]['zcorr']  = zcorr[gd]
            results[i]['vxcorr'] = vxcorr[gd]
            results[i]['vycorr'] = vycorr[gd]
            results[i]['vzcorr'] = vzcorr[gd]
        else:
            results[i]['xcorr']  = np.nan
            results[i]['ycorr']  = np.nan
            results[i]['zcorr']  = np.nan
            results[i]['vxcorr'] = np.nan
            results[i]['vycorr'] = np.nan
            results[i]['vzcorr'] = np.nan



    np.save(save_name, results)



























