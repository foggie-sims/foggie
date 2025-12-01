import yt
from yt import derived_field
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LogNorm
import numpy as np 

from astropy.table import Table
#import astropy.units as u
import unyt as unyt
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

def parse_args():
    '''Parse command line arguments. Returns args object.'''

    parser = argparse.ArgumentParser(description='Makes basic kinematic plots for the disk (and CGM)')

    # Optional arguments:
    parser.add_argument('--halo', metavar='halo', type=str, action='store', \
                        help='Which halo? Default is 8508 (Tempest)')
    parser.set_defaults(halo='008508')

    parser.add_argument('--run', metavar='run', type=str, action='store', \
                        help='Which run? Default is nref11c_nref9f')
    parser.set_defaults(run='nref11c_nref9f_HvcClumps')

    parser.add_argument('--minsnap', metavar='minsnap', type=int, action='store', \
                        help='Which starting snapshot? Default is 967')
    parser.set_defaults(minsnap=967)

    parser.add_argument('--maxsnap', metavar='maxsnap', type=int, action='store', \
                        help='Which ending snapshot? Default is 1067')
    parser.set_defaults(maxsnap=1067)

    parser.add_argument('--snapstep', metavar='snapstep', type=int, action='store', \
                        help='Step between snapshots? Default is 1.')
    parser.set_defaults(snapstep=1)

    parser.add_argument('--clump_dir', metavar='clump_dir', type=str, action='store', \
                        help='Where is the clump stats file')
    parser.set_defaults(clump_dir='/Users/ctrapp/Documents/foggie_analysis/clump_project/histograms/tf_hvc/')

    parser.add_argument('--clump_tracking_dir', metavar='clump_tracking_dir', type=str, action='store', \
                        help='Where is the clump hiearchy file')
    parser.set_defaults(clump_tracking_dir='./')

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


def LoadRedshiftsAndTimes(halo_c_v_name,snapshots=None):
    halo_c_v = Table.read(halo_c_v_name, format='ascii')
    redshifts = {}
    times = {}
    itr=0
    for row in halo_c_v:
        if itr>0:
            redshifts[row['col3']] = float(row['col2'])
            times[row['col3']] = ((float(row['col4'])) * unyt.Myr).in_units('Gyr').v
        itr+=1


    if snapshots is not None:
        redshifts_out = []
        times_out = []
        for snap in snapshots:
            redshifts_out.append( redshifts[snap] )
            times_out.append( times[snap] )
        return redshifts_out,times_out
    
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
halo_c_v_name = code_dir+"/halo_infos/"+halo_id+"/nref11c_nref9f/halo_c_v"
redshifts,times = LoadRedshiftsAndTimes(halo_c_v_name,snapshots)


hf_c = h5py.File(args.clump_tracking_dir + GalName+"_"+args.run+"_clump_tracking.h5",'r')
max_clump_label = -1
for snapshot in snapshots:
    max_clump_label = max(max_clump_label, np.max( hf_c['clump_labels_'+snapshot][...] ) )
clump_tf_fraction = np.zeros( (max_clump_label+1, len(snapshots)) )

t=0
tf_conversion=None
for snapshot in snapshots:
    hf_stats = h5py.File(args.clump_dir + GalName+"_"+snapshot+"_"+args.run+"_clump_stats.h5",'r')

    clump_masses = hf_stats['leaf_masses'][...]

    tf5 = hf_stats['leaf_tf5_mass'][...] #HVCs, IVCs, and LVCs. Should correspond to all clumps!
    tf6 = hf_stats['leaf_tf6_mass'][...]
    tf7 = hf_stats['leaf_tf7_mass'][...]

    if tf_conversion is None:
        denom = np.copy(clump_masses)
        tf_conversion = np.mean( np.divide(denom[0], tf5[0]+tf6[0]+tf7[0]) )

    tf5=tf5 * tf_conversion
    tf6=tf6 * tf_conversion
    tf7=tf7 * tf_conversion

    leaf_clump_ids = hf_stats['leaf_clump_ids'][...]
    hf_stats.close()

    key = 'clump_labels_'+snapshot
    clump_mapping = hf_c[key][...]
    for i in range(len(clump_mapping)):
        denom = clump_masses[i]
        if denom==0: denom=1e-10
        clump_tf_fraction[clump_mapping[i],t] = (tf5[i] + tf6[i] + tf7[i]) / denom
    print(np.min(clump_tf_fraction[:,t]), np.max(clump_tf_fraction[:,t]))

    t+=1
    hf_stats.close()
hf_c.close()

#Plot 1: Histogram of TF fraction at all times
clump_tf_fraction = clump_tf_fraction / np.max(clump_tf_fraction[:,0]) #Normalize to max of 1
for t in range(len(snapshots)):
    plt.figure(figsize=(8,6))
    print("clump_tf_fraction[:,t]=",clump_tf_fraction[:,t])
    plt.hist(clump_tf_fraction[:,t], bins=100, range=(.01,1), histtype='stepfilled', color='blue', alpha=0.7)
    plt.xlabel('Tracer Fluid Fraction', fontsize=16)
    plt.ylabel('Number of Clumps', fontsize=16)
    #plt.title(f'Clump Tracer Fluid Fraction at {snapshots[t]}', fontsize=18)
    plt.xlim(0,1)
    #plt.yscale('log')
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(args.output_dir + "histograms/"+f'{GalName}_{snapshots[t]}_{args.run}_clump_tf_fraction_histogram.png')
    plt.close()

#Plot 2: TF fraction evolution for each clump
plt.figure(figsize=(10,8))
print(times)
print(clump_tf_fraction[0,:])
for clump_id in range(1, max_clump_label+1):
    if clump_tf_fraction[clump_id,0]>0.5:
        plt.plot(times, clump_tf_fraction[clump_id,:], linestyle='-', alpha=0.25)
plt.xlabel('Time (Gyr)', fontsize=16)
plt.ylabel('Tracer Fluid Fraction', fontsize=16)
plt.title(f'Clump Tracer Fluid Fraction Evolution', fontsize=18)
#plt.xlim(np.min(times), np.max(times))
plt.yscale('log')
plt.ylim([0.001,1])
plt.xlim([5.9,5.95])
plt.grid(True, which='both', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig(args.output_dir + f'{GalName}_{args.run}_clump_tf_fraction_evolution.png')
plt.close()
