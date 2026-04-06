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


def parse_args():
    '''Parse command line arguments. Returns args object.'''

    parser = argparse.ArgumentParser(description='Makes basic kinematic plots for the disk (and CGM)')

    # Optional arguments:
    parser.add_argument('--run', metavar='run', type=str, action='store', \
                        help='Which run? Default is nref11c_nref9f')
    parser.set_defaults(run='nref11c_nref9f')

    parser.add_argument('--minsnap', metavar='minsnap', type=str, action='store', \
                        help='Which snapshot to start at? Default is RD0042')
    parser.set_defaults(minsnap='RD0042')

    parser.add_argument('--maxsnap', metavar='maxsnap', type=str, action='store', \
                        help='Which snapshot to end at? Default is RD0042')
    parser.set_defaults(maxsnap='RD0042')

    parser.add_argument('--system', metavar='system', type=str, action='store', \
                        help='Where is the clump file to define the disk')
    parser.set_defaults(system='cameron_local')

    parser.add_argument('--pwd', metavar='pwd', type=bool, action='store', \
                        help='Use working directory in get_run_loc_etc. Default is False.')
    parser.set_defaults(pwd=False)

    parser.add_argument('--forcepath', metavar='forcepath', type=bool, action='store', \
                        help='Use forcepath in get_run_loc_etc. Default is False.')
    parser.set_defaults(forcepath=False)

    parser.add_argument('--output_dir', metavar='output_dir', type=str, action='store', \
                        help='Where to save the values. Default is ./')
    parser.set_defaults(output_dir='./')


    parser.add_argument('--clumping_field', metavar='clumping_field', type=str, action='store', \
                        help='Which gas field do you want to clump on? Default is "density"')
    parser.set_defaults(clumping_field='density')

    args = parser.parse_args()
    return args

import time

def _inverse_temperature(field, data):
    return 1.0 / data['gas', 'temperature']



def get_clumping_range(data_dir,code_dir,clumping_field,minsnap,maxsnap,output_dir=None,halo_ids=['008508','004123','002392','002878','005016','005036'],run="nref11c_nref9f",search_full_box=False,scale_with_redshift=True):

    current_min = np.inf
    current_max = -np.inf


    for snapshot in [minsnap,maxsnap]:
        snapshot_min = np.inf
        snapshot_max = -np.inf
        for halo_id in halo_ids:
            try:
                snap_name = data_dir + "halo_"+halo_id+"/"+run+"/"+snapshot+"/"+snapshot

                trackname = code_dir+"/halo_tracks/"+halo_id+"/nref11n_selfshield_15/halo_track_200kpc_nref9"

                halo_c_v_name = code_dir+"/halo_infos/"+halo_id+"/"+run+"/halo_c_v"

                catalog_dir = code_dir + '/halo_infos/' + halo_id + '/'+run+'/'
                smooth_AM_name = catalog_dir + 'AM_direction_smoothed'

                ds, refine_box = foggie_load(snap_name, trackfile_name=trackname, halo_c_v_name=halo_c_v_name, do_filter_particles=True,disk_relative=True,smooth_AM_name = smooth_AM_name)

                if clumping_field ==  "inverse_temperature":
                     ds.add_field( ('gas','inverse_temperature'), function=_inverse_temperature, units="1/K", sampling_type="cell" )
    

                trident_dict = { 'HI':'H I', 'CII':'C II','CIII':'C III',
                            'CIV':'C IV','OI':'O I','OII':'O II','OIII':'O III','OIV':'O IV','OV':'O V','OVI':'O VI',
                            'SiII':'Si II','SiIII':'Si III','SiIV':'Si IV','MgII':'Mg II'}

                ions_number_density_dict = {'Lyalpha':'LyAlpha', 'HI':'H_p0_number_density', 'CII':'C_p1_number_density', 'CIII':'C_p2_number_density',
                                          'CIV':'C_p3_number_density','OI':'O_p0_number_density','OII':'O_p1_number_density','OIII':'O_p2_number_density','OIV':'O_p3_number_density','OV':'O_p4_number_density',
                                             'OVI':'O_p5_number_density','SiII':'Si_p1_number_density','SiIII':'Si_p2_number_density',
                                             'SiIV':'Si_p3_number_density','MgII':'Mg_p1_number_density',
                                             'HI':'H_p0_density', 'CII':'C_p1_density', 'CIII':'C_p2_density',
                                             'CIV':'C_p3_density','OI':'O_p0_density','OII':'O_p1_density','OIII':'O_p2_density','OIV':'O_p3_density','OV':'O_p4_density',
                                             'OVI':'O_p5_density','SiII':'Si_p1density','SiIII':'Si_p2_density',
                                             'SiIV':'Si_p3_density','MgII':'Mg_p1_density'}

                field_dict = {v: k for k, v in ions_number_density_dict.items()}

                if clumping_field in trident_dict:
                    import trident
                    trident.add_ion_fields(ds, ions=[trident_dict[clumping_field]])
                    args.clumping_field =ions_number_density_dict[clumping_field]
                elif args.clumping_field in field_dict:
                    import trident
                    trident.add_ion_fields(ds, ions=[trident_dict[field_dict[clumping_field]]])

                zsnap = ds.get_parameter('CosmologyCurrentRedshift')
                ascale = 1./(1.+zsnap)


                if search_full_box:
                    source_cut = ds.all_data()
                else:
                    source_cut = refine_box

                clumping_values = source_cut['gas',clumping_field]
                clumping_min = np.min(clumping_values[clumping_values>0])
                clumping_max = np.max(clumping_values[clumping_values>0])

                if scale_with_redshift:
                    clumping_min *= ascale**3
                    clumping_max *= ascale**3

                if clumping_min < current_min: current_min = clumping_min
                if clumping_max > current_max: current_max = clumping_max

                if clumping_min < snapshot_min: snapshot_min = clumping_min
                if clumping_max > snapshot_max: snapshot_max = clumping_max

            except Exception as e:
                print(f"Error loading snapshot {snapshot} for halo {halo_id}: {e}")
                continue
        print(f"Measured for {snapshot}, "
              f"the clumping field {clumping_field} ranges from "
              f"{snapshot_min/(ascale**3)} to {snapshot_max/(ascale**3)}")
    box_type = "full box" if search_full_box else "refine box"
    print(f"Measured in snapshots {minsnap} and {maxsnap}, across halos {halo_ids}, "
          f"and in the {box_type}, the clumping field {clumping_field} ranges from "
          f"{current_min} to {current_max}")
    
    if output_dir is not None:
        hf = h5py.File(output_dir + clumping_field + "_clumping_range.h5", 'w')
        units = clumping_min.units
        print("units are",units)
        hf.create_dataset("clumping_min", data=current_min.in_units(units).v)
        hf.create_dataset("clumping_max", data=current_max.in_units(units).v)
        hf.create_dataset("units", data=str(units))
        hf.create_dataset("clumping_field", data=clumping_field)
        hf.close()


    return current_min,current_max

args = parse_args()

args.halo = "008508"
data_dir, output_path, run_loc, code_dir,trackname,halo_name,spectra_dir,infofile = get_run_loc_etc(args)



clumping_field = args.clumping_field

min_clumping_field, max_clumping_field = get_clumping_range(data_dir,code_dir,clumping_field,args.minsnap,args.maxsnap,output_dir=args.output_dir,run=args.run)