import yt
from yt import derived_field
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LogNorm
import numpy as np 

from astropy.table import Table
import astropy.units as u
from scipy import stats


import seaborn as sns
sns.set_style("whitegrid", {'axes.grid' : False})
import matplotlib as mpl
mpl.rcParams['font.family'] = 'stixgeneral'
import matplotlib.pyplot as plt
import matplotlib.lines as lines

import cmasher as cmr
import os
import argparse


from foggie.utils.consistency import *
from foggie.utils.get_refine_box import get_refine_box
from foggie.utils.get_halo_center import get_halo_center
from foggie.utils.get_proper_box_size import get_proper_box_size
from foggie.utils.get_run_loc_etc import get_run_loc_etc
from foggie.utils.yt_fields import *
from foggie.utils.foggie_load import *
from foggie.utils.analysis_utils import *

import h5py

from foggie.clumps.clump_finder.utils_clump_finder import halo_id_to_name
from foggie.clumps.clump_finder.utils_clump_finder import read_virial_mass_file
from foggie.clumps.clump_finder import *


import time
start_time =time.time()

def parse_args():
    '''Parse command line arguments. Returns args object.'''

    parser = argparse.ArgumentParser(description='Makes basic kinematic plots for the disk (and CGM)')

    # Optional arguments:
    parser.add_argument('--halo', metavar='halo', type=str, action='store', \
                        help='Which halo? Default is 8508 (Tempest)')
    parser.set_defaults(halo='008508')

    parser.add_argument('--run', metavar='run', type=str, action='store', \
                        help='Which run? Default is nref11c_nref9f')
    parser.set_defaults(run='nref11c_nref9f')

    parser.add_argument('--snapshot', metavar='snapshot', type=str, action='store', \
                        help='Which snapshot? Default is RD0042')
    parser.set_defaults(snapshot='RD0042')
    
    parser.add_argument('--snapshot_array_index', metavar='snapshot_array_index', type=str, action='store', \
                        help='Which snapshot number as fed in by the PBS Job array? Default is None, will override snapshot.')
    parser.set_defaults(snapshot_array_index=None)

    parser.add_argument('--clumping_field', metavar='clumping_field', type=str, action='store', \
                        help='Which gas field do you want to clump on? Defaults to clump_finder (density)')
    parser.set_defaults(clumping_field=None)


    parser.add_argument('--clump_min', metavar='clump_min', type=float, action='store', \
                        help='What is the starting cutoff you want to use for the clump finder? Defaults to the minimum in the refine box.')
    parser.set_defaults(clump_min=None)

    parser.add_argument('--clump_dir', metavar='clump_dir', type=str, action='store', \
                        help='Where is the clump file to define the disk')
    parser.set_defaults(clump_dir='./')

    parser.add_argument('--system', metavar='system', type=str, action='store', \
                        help='Where is the clump file to define the disk')
    parser.set_defaults(system='cameron_local')

    parser.add_argument('--pwd', metavar='pwd', type=bool, action='store', \
                        help='Use working directory in get_run_loc_etc. Default is False.')
    parser.set_defaults(pwd=False)

    parser.add_argument('--forcepath', metavar='forcepath', type=bool, action='store', \
                        help='Use forcepath in get_run_loc_etc. Default is False.')
    parser.set_defaults(forcepath=False)

    parser.add_argument('--is_rd', metavar='is_rd', type=bool, action='store', \
                        help='Are you analyzing RD snapshots instead of DD? Default is False.')
    parser.set_defaults(is_rd=False)

    parser.add_argument('--data_dir', metavar='data_dir', type=str, action='store', \
                        help='Overwrite data directory from get_run_loc_etc. Default is None.')
    parser.set_defaults(data_dir=None)


    args = parser.parse_args()
    return args


args = parse_args()

if args.snapshot_array_index is not None:
    if args.is_rd:
        args.snapshot = "RD"+args.snapshot_array_index.zfill(4)
    else:
        args.snapshot = "DD"+args.snapshot_array_index.zfill(4)

data_dir, output_path, run_loc, code_dir,trackname,halo_name,spectra_dir,infofile = get_run_loc_etc(args)

if args.data_dir is not None:
    data_dir = args.data_dir #overwrite

halo_id = args.halo #008508
snapshot = args.snapshot #RD0042
run = args.run #nref11c_nref9f

gal_name = halo_id_to_name(halo_id)
gal_name+="_"+snapshot+"_"+run

snap_name = data_dir + "halo_"+halo_id+"/"+run+"/"+snapshot+"/"+snapshot

trackname = code_dir+"/halo_tracks/"+halo_id+"/nref11n_selfshield_15/halo_track_200kpc_nref9"

halo_c_v_name = code_dir+"/halo_infos/"+halo_id+"/"+run+"/halo_c_v"

#particle_type_for_angmom = 'young_stars' ##Currently the default
particle_type_for_angmom = 'gas' #Should be defined by gas with Temps below 1e4 K

catalog_dir = code_dir + '/halo_infos/' + halo_id + '/'+run+'/'
#smooth_AM_name = catalog_dir + 'AM_direction_smoothed'
smooth_AM_name = None

ds, refine_box = foggie_load(snap_name, trackfile_name=trackname, halo_c_v_name=halo_c_v_name, do_filter_particles=True,disk_relative=True,particle_type_for_angmom=particle_type_for_angmom,smooth_AM_name = smooth_AM_name)

cf_args = get_default_args()
clump_file = args.clump_dir + gal_name

cf_args.output = clump_file
#cf_args.clump_min = 1.3e-30

if args.clump_min is not None:
    cf_args.clump_min = args.clump_min
else:
    #cf_args.clump_min = np.min(refine_box['gas','density']).v
    zsnap = ds.get_parameter('CosmologyCurrentRedshift')
    if args.clumping_field=="density" or args.clumping_field is None: cf_args.clump_min = (1.3e-30) * np.power(1+zsnap,3) / 8. #in g/cm^3, comoving

print("Clump min set to:",cf_args.clump_min)
if args.clumping_field is not None:
    cf_args.clumping_field = args.clumping_field
    cf_args.output = cf_args.output + args.clumping_field
master_clump = clump_finder(cf_args,ds,refine_box)