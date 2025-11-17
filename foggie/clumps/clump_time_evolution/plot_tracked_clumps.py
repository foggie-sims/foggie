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

    parser.add_argument('--stats_dir', metavar='stats_dir', type=str, action='store', \
                        help='Where to save the histograms. Default is ./')
    parser.set_defaults(stats_dir='/Users/ctrapp/Documents/foggie_analysis/clump_project/histograms/')

    parser.add_argument('--clump_tracking_dir', metavar='clump_tracking_dir', type=str, action='store', \
                        help='Where to save the histograms. Default is ./')
    parser.set_defaults(clump_tracking_dir='./')

    args = parser.parse_args()
    return args

def LoadRedshiftsAndTimes(halo_c_v_name,snapshots):
    halo_c_v = Table.read(halo_c_v_name, format='ascii')
    
    redshifts = {}
    times = {}
    itr=0
    for row in halo_c_v:
        if itr>0:
            redshifts[row['col3']] = float(row['col2'])
            times[row['col3']] = ((float(row['col4'])) * u.Myr).in_units('s').v
        itr+=1

    return redshifts,times



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




hf_c = h5py.File(args.clump_tracking_dir + GalName+"_"+args.run+"_clump_tracking.h5",'r')
print(hf_c.keys())
max_clump_label = -1
for snapshot in snapshots:
    max_clump_label = max(max_clump_label, np.max( hf_c['clump_labels_'+snapshot][...] ) )
clump_trajectories = np.zeros( (max_clump_label+1, len(snapshots), 3) ) * u.kpc

t=0
for snapshot in snapshots:
    hf_stats = h5py.File(args.stats_dir + GalName+"_"+snapshot+"_"+args.run+"_clump_stats.h5",'r')
    leaf_x_disk = hf_stats['leaf_x_disk'][...]
    leaf_y_disk = hf_stats['leaf_y_disk'][...]
    leaf_z_disk = hf_stats['leaf_z_disk'][...]

    leaf_clump_ids = hf_stats['leaf_clump_ids'][...]
    hf_stats.close()

    key = 'clump_labels_'+snapshot
    print("key is",key)
    clump_mapping = hf_c[key][...]
    for i in range(len(clump_mapping)):
        clump_trajectories[clump_mapping[i],t,:] = [leaf_x_disk[i], leaf_y_disk[i], leaf_z_disk[i]]

    t+=1
    hf_stats.close()
hf_c.close()

#Plot all clumps at a snapshot color coded by their clump id at the first snapshot
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

nc,nt,ndim = np.shape(clump_trajectories)
nplot=0
dc=0

count=0
for cc in np.arange(0,nc-1,1):
    if cc>=(nc):
        break
    x = clump_trajectories[cc+dc,:,0]
    y = clump_trajectories[cc+dc,:,1]
    z = clump_trajectories[cc+dc,:,2]
    mask = ((np.abs(x)>0) & (np.abs(y)>0) & (np.abs(z)>0))
    if (np.size(np.where(mask))>5):
        count+=1

for cc in np.arange(0,nc-1,10):
    if cc>=(nc):
        break
    x = clump_trajectories[cc+dc,:,0]
    y = clump_trajectories[cc+dc,:,1]
    z = clump_trajectories[cc+dc,:,2]
    mask = ((np.abs(x)>0) & (np.abs(y)>0) & (np.abs(z)>0))
    if (np.size(np.where(mask))>5):

        plt.plot(x[mask],y[mask],z[mask],lw=1.5, alpha=0.5, label=f'Trajectory {cc}')
        ax.scatter(x[mask][-1],y[mask][-1],z[mask][-1],s=30,alpha=0.5) #Mark the end point
        nplot+=1
        if nplot>100: break
# Optional styling
print("nplot=",nplot)
ax.set_xlabel('X', fontsize=14)
ax.set_ylabel('Y', fontsize=14)
ax.set_zlabel('Z', fontsize=14)
ax.grid(True)

print("Cloud count=",count)

plt.savefig("clump_trajectories_3d.png",dpi=300)
plt.show()