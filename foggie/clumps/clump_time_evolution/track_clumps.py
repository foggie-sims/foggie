import numpy as np 

from astropy.table import Table

import argparse

from foggie.utils.consistency import *
from foggie.utils.get_run_loc_etc import get_run_loc_etc
from foggie.utils.yt_fields import *
from foggie.utils.foggie_load import *
from foggie.utils.analysis_utils import *

import h5py

from foggie.clumps.clump_finder.utils_clump_finder import halo_id_to_name
from foggie.clumps.clump_finder import *



import time
start_time =time.time()
import unyt as unyt

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
    parser.set_defaults(maxsnap='1067')

    parser.add_argument('--snapstep', metavar='snapstep', type=int, action='store', \
                        help='Step between snapshots? Default is 1')
    parser.set_defaults(snapstep=1)

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
            times[row['col3']] = ((float(row['col4'])) * unyt.Myr).in_units('s').v
            xc[row['col3']] = float(row['col5']) * unyt.kpc
            yc[row['col3']] = float(row['col6']) * unyt.kpc
            zc[row['col3']] = float(row['col7']) * unyt.kpc
            vxc[row['col3']] = float(row['col8']) * unyt.km/unyt.s
            vyc[row['col3']] = float(row['col9']) * unyt.km/unyt.s
            vzc[row['col3']] = float(row['col10']) * unyt.km/unyt.s
        itr+=1

    return redshifts,times,xc,yc,zc,vxc,vyc,vzc


from scipy.optimize import linear_sum_assignment
def link_snapshots(x, v, x1, v1, dt, previous_global_matches=None,max_global_clump_id=0, r_max=(.1*unyt.kpc).in_units('km').v, v_max=(100*unyt.km/unyt.s).in_units('km/s').v):
    x_pred = x + v * dt

    large_number = 1e10

    print('shape=',np.shape(x),np.shape(x1))

    ndim,nc0 = np.shape(x)
    ndim,nc1 = np.shape(x1)

    cost = np.full((nc0, nc1),large_number)

    finite_mask = ((np.isfinite(x[0,:])) & (np.isfinite(x[1,:])) & (np.isfinite(x[2,:])) & (np.isfinite(v[0,:])) & (np.isfinite(v[1,:])) & (np.isfinite(v[2,:])))
    sigma_x = np.std(x[:,finite_mask],axis=1)
    sigma_v = np.std(v[:,finite_mask],axis=1)

# Pairwise differences via broadcasting
    dx_vec = (x_pred[:, :, None] - x1[:, None, :]) / sigma_x[:, None, None]
    dv_vec = (v[:, :, None]      - v1[:, None, :]) / sigma_v[:, None, None]

# Norms over spatial/velocity dimensions
    dx = np.linalg.norm(dx_vec, axis=0)   # shape (nc0, nc1)
    dv = np.linalg.norm(dv_vec, axis=0)   # shape (nc0, nc1)

    # Gating
    mask = (dx < r_max) & (dv < v_max)

    # Initialize cost matrix

# Assign costs where valid
    cost[mask] = dx[mask]**2 + dv[mask]**2  
    print("cost=",np.mean(cost),np.std(cost))
    #for i in range(nc0):
    #    for j in range(nc1):
    #        dx = np.linalg.norm(np.divide(x_pred[:,i] - x1[:,j] , sigma_x)) 
    #        dv = np.linalg.norm(np.divide(v[:,i] - v1[:,j], sigma_v))
    #        if dx < r_max and dv < v_max:
    #            cost[i,j] = dx**2 + dv**2
    #            print("cost[",i,",",j,"] =",cost[i,j]," dx=",dx," dv=",dv)

    #print(cost)

    row, col = linear_sum_assignment(cost)

    matches = [(i,j) for i,j in zip(row,col) if (cost[i,j]<large_number)]
    matches = np.array(matches)
    disappeared = set(range(nc0)) - set(i for i,_ in matches)
    new = set(range(nc1)) - set(j for _,j in matches)
    print(np.shape(matches))
    print(np.max(np.array(matches)[:,0]))
    print(np.max(np.array(matches)[:,1]))

    #matches has shape (N_matched, 2) where each row is (index_in_first_snapshot, index_in_second_snapshot)
    if previous_global_matches is not None:
        Nmatches = len(matches)
        global_matches = np.zeros((Nmatches,2),dtype=int)
        global_matches[:,1] = np.array(matches)[:,1]
        n_old_clumps = 0
        n_new_clumps = 0
        for i in range(Nmatches):
            try:
               previous_match_idx = np.where(previous_global_matches[:,1]==matches[i,0])[0][0]
               global_matches[i,0] = previous_global_matches[previous_match_idx,0]
               n_old_clumps +=1

            except:
                n_new_clumps +=1
                global_matches[i,0] = max_global_clump_id + 1 + i

        print("n_old_clumps=",n_old_clumps," n_new_clumps=",n_new_clumps)
    else:
        global_matches = np.copy(matches)

    #print("matches=",matches)
    #print("disappeared=",disappeared)
    #print("new=",new)

    return matches, disappeared, new, global_matches




args = parse_args()
halo_id = args.halo #008508
run = args.run #nref11c_nref9f
GalName = halo_id_to_name(halo_id)

snapshots = []
snapshots.append( "DD"+str(args.minsnap).zfill(4) )
for snap in range(args.minsnap,args.maxsnap,args.snapstep):
    if args.is_rd:
        snapshots.append( "RD"+str(snap).zfill(4) )
    else:
        snapshots.append( "DD"+str(snap).zfill(4) )

data_dir, output_path, run_loc, code_dir,trackname,halo_name,spectra_dir,infofile = get_run_loc_etc(args)
halo_c_v_name = code_dir+"/halo_infos/"+halo_id+"/nref11c_nref9f/halo_c_v"

redshifts,times,xc,yc,zc,vxc,vyc,vzc = LoadRedshiftsAndTimes(halo_c_v_name,snapshots)
clump_mapping = {}
label0 = {}
itr=-1
duplicate_count=0
prev_snapshot=-1
clump_mapping = {}
disappeared_clumps = {}
new_clumps = {}
clump_global_mapping = {}
previous_global_matches = None
max_global_clump_id = 0

def load_stats_from_hierarchy_file(hierarchy_file):
    hf = h5py.File(hierarchy_file,'r')
    leaf_ids = hierarchy_file['leaf_clump_ids'][...]
    leaf_x=[]
    leaf_y=[]
    leaf_z=[]
    leaf_vx=[]
    leaf_vy=[]
    leaf_vz=[]


    for leaf_id in leaf_ids:
        leaf_x.append(hierarchy_file[str(leaf_id)]['leaf_x'][...] * unyt.kpc - xc[snapshot]) #Center on disk
        leaf_y.append(hierarchy_file[str(leaf_id)]['leaf_y'][...] * unyt.kpc - yc[snapshot])
        leaf_z.append(hierarchy_file[str(leaf_id)]['leaf_z'][...] * unyt.kpc - zc[snapshot])

        leaf_vx.append(hierarchy_file[str(leaf_id)]['leaf_vx'][...] * unyt.km/unyt.s - vxc[snapshot])
        leaf_vy.append(hierarchy_file[str(leaf_id)]['leaf_vy'][...] * unyt.km/unyt.s - vyc[snapshot])
        leaf_vz.append(hierarchy_file[str(leaf_id)]['leaf_vz'][...] * unyt.km/unyt.s - vzc[snapshot])

    hf.close()
    return np.array(leaf_x),np.array(leaf_y),np.array(leaf_z),np.array(leaf_vx),np.array(leaf_vy),np.array(leaf_vz),leaf_ids


for snapshot in snapshots:
    itr+=1
    hf = h5py.File(args.clump_stat_base + GalName+"_"+snapshot+"_"+args.run+"_clump_stats.h5",'r')
    leaf_x_1 = hf['leaf_x'][...] * unyt.kpc - xc[snapshot] #Center on disk
    leaf_y_1 = hf['leaf_y'][...] * unyt.kpc - yc[snapshot]
    leaf_z_1 = hf['leaf_z'][...] * unyt.kpc - zc[snapshot]

    leaf_vx_1 = hf['leaf_vx'][...] * unyt.km/unyt.s - vxc[snapshot] #center on disk
    leaf_vy_1 = hf['leaf_vy'][...] * unyt.km/unyt.s - vyc[snapshot]
    leaf_vz_1 = hf['leaf_vz'][...] * unyt.km/unyt.s - vzc[snapshot]
   
    leaf_ids_1 = hf['leaf_clump_ids'][...]
    #hierarchy_file = args.hierarchy_file_base + snapshot+args.run+"_ClumpTree.h5"
    #leaf_x_1,leaf_y_1,leaf_z_1,leaf_vx_1,leaf_vy_1,leaf_vz_1,leaf_ids_1 = load_stats_from_hierarchy_file(hierarchy_file)

    if itr==0: original_leaf_ids = np.copy(leaf_ids_1)

    print("On snapshot",snapshot,"with",len(leaf_x_1),"clumps. prev_snapshot:",prev_snapshot)
    t1 = times[snapshot]
    if itr>0: #start tracking if on second snap
        clump_position_1 = np.array([leaf_x_1.in_units('km').v,leaf_y_1.in_units('km').v,leaf_z_1.in_units('km').v])
        clump_position_0 = np.array([leaf_x_0.in_units('km').v,leaf_y_0.in_units('km').v,leaf_z_0.in_units('km').v])
        clump_velocity_1 = np.array([leaf_vx_1.in_units('km/s').v,leaf_vy_1.in_units('km/s').v,leaf_vz_1.in_units('km/s').v])
        clump_velocity_0 = np.array([leaf_vx_0.in_units('km/s').v,leaf_vy_0.in_units('km/s').v,leaf_vz_0.in_units('km/s').v])

        matches,disappeared,new,global_matches = link_snapshots(clump_position_0,clump_velocity_0,clump_position_1,clump_velocity_1,t1-t0,previous_global_matches,max_global_clump_id)
        max_global_clump_id = max(max_global_clump_id, np.max(global_matches[:,0])) #in case any clumps disappear

        clump_mapping[snapshot] = matches
        disappeared_clumps[snapshot] = disappeared
        new_clumps[snapshot] = new
        clump_global_mapping[snapshot] = global_matches
        previous_global_matches = np.copy(global_matches)
        #Define a global clump id based on these...


    #Clump 1 is now clump 0
    leaf_x_0 = np.copy(leaf_x_1.in_units('kpc').v) * unyt.kpc
    leaf_y_0 = np.copy(leaf_y_1.in_units('kpc').v) * unyt.kpc
    leaf_z_0 = np.copy(leaf_z_1.in_units('kpc').v) * unyt.kpc

    leaf_vx_0 = np.copy(leaf_vx_1.in_units('km/s').v) * unyt.km/unyt.s
    leaf_vy_0 = np.copy(leaf_vy_1.in_units('km/s').v) * unyt.km/unyt.s
    leaf_vz_0 = np.copy(leaf_vz_1.in_units('km/s').v) * unyt.km/unyt.s


    leaf_ids_0 = np.copy(leaf_ids_1)

    t0 = np.copy(t1)
    prev_snapshot = snapshot
    #hf.close()

hf_o = h5py.File(args.output_dir + GalName+"_"+args.run+"_clump_tracking.h5",'w')
for snapshot in snapshots:
    try:
        hf_o.create_dataset('direct_clump_mapping_'+snapshot,data=clump_mapping[snapshot])
        hf_o.create_dataset('global_clump_mapping_'+snapshot,data=clump_global_mapping[snapshot])
        hf_o.create_dataset('disappeared_clumps_'+snapshot,data=np.array(list(disappeared_clumps[snapshot])))
        hf_o.create_dataset('new_clumps_'+snapshot,data=np.array(list(new_clumps[snapshot])))
    except:
        print("Could not save data for snapshot",snapshot," skipping...")
hf_o.create_dataset('minsnap',data=np.array(args.minsnap))
hf_o.create_dataset('maxsnap',data=np.array(args.maxsnap))
hf_o.create_dataset('snapstep',data=np.array(args.snapstep))
hf_o.close()