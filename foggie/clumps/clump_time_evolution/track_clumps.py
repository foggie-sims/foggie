import yt
from yt import derived_field
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LogNorm
import numpy as np 

from astropy.table import Table
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

from scipy import stats

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
from foggie.clumps.clump_finder.clump_finder import TqdmProgressBar


import time
start_time =time.time()
import unyt as u

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

    parser.add_argument('--minsnap', metavar='minsnap', type=int, action='store', \
                        help='Which snapshot to start at? Default is RD0042')
    parser.set_defaults(minsnap='967')

    parser.add_argument('--maxsnap', metavar='maxsnap', type=int, action='store', \
                        help='Which snapshot to end at? Default is RD0042')
    parser.set_defaults(maxsnap='2427')

    parser.add_argument('--snapstep', metavar='snapstep', type=int, action='store', \
                        help='Step between snapshots? Default is 1')
    parser.set_defaults(snapstep=10)

    parser.add_argument('--clump_stat_base', metavar='clump_stat_base', type=str, action='store', \
                        help='Where is the clump stats file')
    parser.set_defaults(clump_stat_base='/Users/ctrapp/Documents/foggie_analysis/clump_project/histograms/')

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

    parser.add_argument('--output_dir', metavar='output_dir', type=str, action='store', \
                        help='Where to save the histograms. Default is ./')
    parser.set_defaults(output_dir='./')

    args = parser.parse_args()
    return args

def LoadRedshiftsAndTimes(halo_c_v_name,snapshots):
    halo_c_v = Table.read(halo_c_v_name, format='ascii')
    
    redshifts = {}
    times = {}
    xc = {}
    yc = {}
    zc = {}
    vxc = {}
    vyc = {}
    vzc = {}
    itr=0
    for row in halo_c_v:
        if itr>0:
            redshifts[row['col3']] = float(row['col2'])
            times[row['col3']] = ((float(row['col4'])) * u.Myr).in_units('s').v
            xc[row['col3']] = float(row['col5']) * u.kpc
            yc[row['col3']] = float(row['col6']) * u.kpc
            zc[row['col3']] = float(row['col7']) * u.kpc
            vxc[row['col3']] = float(row['col8']) * u.km/u.s
            vyc[row['col3']] = float(row['col9']) * u.km/u.s
            vzc[row['col3']] = float(row['col10']) * u.km/u.s
        itr+=1

    return redshifts,times,xc,yc,zc,vxc,vyc,vzc




args = parse_args()
halo_id = args.halo #008508
run = args.run #nref11c_nref9f
GalName = halo_id_to_name(halo_id)

snapshots = []
for snap in range(args.minsnap,args.maxsnap,args.snapstep):
    if args.is_rd:
        snapshots.append( "RD"+str(snap).zfill(4) )
    else:
        snapshots.append( "DD"+str(snap).zfill(4) )

data_dir, output_path, run_loc, code_dir,trackname,halo_name,spectra_dir,infofile = get_run_loc_etc(args)
halo_c_v_name = code_dir+"/halo_infos/"+halo_id+"/"+run+"/halo_c_v"

redshifts,times,xc,yc,zc,vxc,vyc,vzc = LoadRedshiftsAndTimes(halo_c_v_name,snapshots)
clump_mapping = {}
label0 = {}
itr=-1
duplicate_count=0
prev_snapshot=-1
for snapshot in snapshots:
    itr+=1
    hf = h5py.File(args.clump_stat_base + GalName+"_"+snapshot+"_"+args.run+"_clump_stats.h5",'r')
    leaf_x_1 = hf['leaf_x'][...] * u.kpc - xc[snapshot] #Center on disk
    leaf_y_1 = hf['leaf_y'][...] * u.kpc - yc[snapshot]
    leaf_z_1 = hf['leaf_z'][...] * u.kpc - zc[snapshot]
   
    leaf_ids_1 = hf['leaf_clump_ids'][...]
    if itr==0: original_leaf_ids = np.copy(leaf_ids_1)
    print("On snapshot",snapshot,"with",len(leaf_x_1),"clumps. prev_snapshot:",prev_snapshot)
    t1 = times[snapshot]
    if itr>0: #start tracking if on second snap
        dt = (t1-t0) * u.s#.in_units('s')
        print(dt.in_units("Myr"))
        best_match = -1*np.ones(len(leaf_x_1)).astype(int)
        current_min = 1e10*np.ones(len(leaf_x_1))
        new_labels = np.copy(best_match)
        for i in range(len(leaf_x_1)):
            leaf_x_0_projected = leaf_x_0 + leaf_vx_0*dt
            leaf_y_0_projected = leaf_y_0 + leaf_vy_0*dt
            leaf_z_0_projected = leaf_z_0 + leaf_vz_0*dt
            distance_offset = np.sqrt(np.power(leaf_vx_0*dt,2) + np.power(leaf_vy_0*dt,2) + np.power(leaf_vz_0*dt,2))
            distance_offset = distance_offset.in_units('kpc').v
            dist = np.sqrt( (leaf_x_0_projected-leaf_x_1[i])**2 + (leaf_y_0_projected-leaf_y_1[i])**2 + (leaf_z_0_projected-leaf_z_1[i])**2 )
            dist = dist.in_units('kpc').v
            if np.min(dist) < 10:#2*distance_offset[np.argmin(dist)]: #kpc
                current_min[i] = np.min(dist)
                best_match[i] = np.argmin(dist)
                new_labels[i] = label0[prev_snapshot][best_match[i]]
               # if distance_offset[np.argmin(dist)]>10:
               #     print("distance_offset=",distance_offset[np.argmin(dist)])
               #     print("Leaf_vx_0=",leaf_vx_0[np.argmin(dist)])
               #     print("Leaf_vy_0=",leaf_vy_0[np.argmin(dist)])
               #     print("Leaf_vz_0=",leaf_vz_0[np.argmin(dist)])


        #Check for duplicates? Assign the furthest clumps to new ids?
        for i in range(len(leaf_x_1)):
            matches = np.where(best_match==i)[0]
            if np.size(matches)>1:
                #Find furthest clump, assign new id
                dists = current_min[matches]
                furthest = matches[dists > np.min(dists)]
                best_match[furthest] = duplicate_count + np.max(original_leaf_ids) + 1
                new_labels[furthest] = duplicate_count + np.max(original_leaf_ids) + 1
                duplicate_count+=1

        #Store results
        clump_mapping[snapshot] = np.copy(best_match)
        label0[snapshot] = np.copy(new_labels)
    else:
        clump_mapping[snapshot] = np.copy(leaf_ids_1)
        label0[snapshot] = np.copy(leaf_ids_1)

    #Clump 1 is now clump 0
    leaf_x_0 = np.copy(leaf_x_1.in_units('kpc')) * u.kpc
    leaf_y_0 = np.copy(leaf_y_1.in_units('kpc')) * u.kpc
    leaf_z_0 = np.copy(leaf_z_1.in_units('kpc')) * u.kpc

    leaf_vx_0 = hf['leaf_vx'][...] * u.km/u.s - vxc[snapshot] #center on disk
    leaf_vy_0 = hf['leaf_vy'][...] * u.km/u.s - vyc[snapshot]
    leaf_vz_0 = hf['leaf_vz'][...] * u.km/u.s - vzc[snapshot]


    leaf_ids_0 = np.copy(leaf_ids_1)

    t0 = np.copy(t1)
    prev_snapshot = snapshot
    hf.close()

hf_o = h5py.File(args.output_dir + GalName+"_"+args.run+"_clump_tracking.h5",'w')
for snapshot in clump_mapping.keys():
    hf_o.create_dataset('direct_clump_mapping_'+snapshot,data=clump_mapping[snapshot])
    hf_o.create_dataset('clump_labels_'+snapshot,data=label0[snapshot])
#hf_o.create_dataset('clump_mapping',data=np.array(list(clump_mapping.values())).astype(int))
hf_o.create_dataset('minsnap',data=np.array(args.minsnap))
hf_o.create_dataset('maxsnap',data=np.array(args.maxsnap))
hf_o.create_dataset('snapstep',data=np.array(args.snapstep))
hf_o.close()