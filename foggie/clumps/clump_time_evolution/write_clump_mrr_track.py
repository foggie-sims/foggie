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
    parser.set_defaults(maxsnap='2427')

    parser.add_argument('--snapstep', metavar='snapstep', type=int, action='store', \
                        help='Step between snapshots? Default is 1')
    parser.set_defaults(snapstep=1)

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

def LoadRedshiftsAndTimes(halo_c_v_name):
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
            times[row['col3']] = ((float(row['col4'])))# * unyt.Myr).in_units('s').v
            xc[row['col3']] = float(row['col5']) * unyt.kpc
            yc[row['col3']] = float(row['col6']) * unyt.kpc
            zc[row['col3']] = float(row['col7']) * unyt.kpc
            vxc[row['col3']] = float(row['col8']) * unyt.km/unyt.s
            vyc[row['col3']] = float(row['col9']) * unyt.km/unyt.s
            vzc[row['col3']] = float(row['col10']) * unyt.km/unyt.s
        itr+=1

    return redshifts,times,xc,yc,zc,vxc,vyc,vzc


def ReadTrackFile(track_filename):
    redshift = []
    blc = []
    trc = []

    with open(track_filename, "r") as f:
        for line in f:
            vals = line.split()
            if len(vals) < 7:
                continue
            redshift.append(float(vals[0]))
            blc.append([float(vals[1]), float(vals[2]), float(vals[3])])
            trc.append([float(vals[4]), float(vals[5]), float(vals[6])])

    return redshift, blc, trc


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
redshifts,times,xc,yc,zc,vxc,vyc,zyc = LoadRedshiftsAndTimes(halo_c_v_name)

print("Redshift at snapshot 967=",redshifts[snapshots[0]])
print("Time at snapshot 967=",times[snapshots[0]])
gal_name = halo_id_to_name(halo_id)
gal_name+="_"+snapshots[0]+"_"+run

snap_name = data_dir + "halo_"+halo_id+"/"+run+"/"+snapshots[0]+"/"+snapshots[0]
trackname = code_dir+"/halo_tracks/"+halo_id+"/nref11n_selfshield_15/halo_track_200kpc_nref9"

#particle_type_for_angmom = 'young_stars' ##Currently the default
particle_type_for_angmom = 'gas' #Should be defined by gas with Temps below 1e4 K

catalog_dir = code_dir + '/halo_infos/' + halo_id + '/'+run+'/'
#smooth_AM_name = catalog_dir + 'AM_direction_smoothed'
smooth_AM_name = None

track_redshifts,track_blc,track_trc = ReadTrackFile(trackname)

ds, refine_box = foggie_load(snap_name, trackfile_name=trackname, halo_c_v_name=halo_c_v_name, do_filter_particles=True,disk_relative=True,particle_type_for_angmom=particle_type_for_angmom,smooth_AM_name = smooth_AM_name)


# Define your variables
n_clumps = 1
n_snaps = int(np.round((args.maxsnap - args.minsnap + 1)/args.snapstep))
print("Writing out",n_clumps,"clumps across",n_snaps,"snapshots")
n_rows = n_snaps #per snapshot

global_clump_ids = [63166] #[128270] #clump id at starting snapshot
nref_forced = [12]
nref_cooling = [12]
main_nref_forced = 9 #Refinement parameters for the main halo track
main_nref_cooling = 11
star_particle_mass = 100000 #Place holder for now
box_size = [10,10,10] * ds.units.kpc #size of the box you want to write out around the clump center (in kpc))
box_size = box_size.in_units('code_length').v
catalog_dir = '/Users/ctrapp/Documents/foggie_analysis/clump_project/clump_catalog/'
output_file = '/Users/ctrapp/Documents/foggie_analysis/mrr_tests/clump_mrr_track.txt'
hierarchy_file_OII = catalog_dir+GalName+"_"+snapshots[0]+"_"+run+"O_p1_density_ClumpTree.h5"





hfc = h5py.File(hierarchy_file_OII,'r')
estimate_clump_center = True
use_clump_tracks = False

with open(output_file, "w") as f:
    f.write(f"{n_clumps+1}") #+1 for main halo track
    f.write(f"\n{n_rows}")

if estimate_clump_center:
    for cc in range(len(global_clump_ids)):
        clump_id = global_clump_ids[cc]
        clump_cell_ids = hfc[str(clump_id)]['cell_ids'][...]
        clump = load_clump(ds,clump_cell_ids=hfc[str(clump_id)]['cell_ids'][...])
        clump_mass = np.sum(clump[('gas','cell_mass')].in_units('Msun').v)
        clump_x = np.sum( np.multiply(clump[('gas','cell_mass')].in_units('Msun').v, clump[('gas','x')].in_units('code_length').v) ) / clump_mass
        clump_y = np.sum( np.multiply(clump[('gas','cell_mass')].in_units('Msun').v, clump[('gas','y')].in_units('code_length').v) ) / clump_mass
        clump_z = np.sum( np.multiply(clump[('gas','cell_mass')].in_units('Msun').v, clump[('gas','z')].in_units('code_length').v) ) / clump_mass

        #calculate mass weighted velocity in simulation frame
        clump_dxdt = np.sum( np.multiply(clump[('gas','cell_mass')].in_units('Msun').v, (clump[('gas','velocity_x')]).in_units('code_length/Myr').v) ) / clump_mass
        clump_dydt = np.sum( np.multiply(clump[('gas','cell_mass')].in_units('Msun').v, (clump[('gas','velocity_y')]).in_units('code_length/Myr').v) ) / clump_mass
        clump_dzdt = np.sum( np.multiply(clump[('gas','cell_mass')].in_units('Msun').v, (clump[('gas','velocity_z')]).in_units('code_length/Myr').v) ) / clump_mass
        itr=0
        for snapshot in snapshots:
            timestep = times[snapshot] - times[snapshots[0]]
            with open(output_file, "a") as f:
                blc = [clump_x+clump_dxdt*timestep-box_size[0]/2., clump_y+clump_dydt*timestep-box_size[1]/2., clump_z+clump_dzdt*timestep-box_size[2]/2.]
                trc = [clump_x+clump_dxdt*timestep+box_size[0]/2., clump_y+clump_dydt*timestep+box_size[1]/2., clump_z+clump_dzdt*timestep+box_size[2]/2.]
                f.write(f"\n{cc:<6} {redshifts[snapshot]:<24} {blc[0]:<24} {blc[1]:<24} {blc[2]:<24} {trc[0]:<24} {trc[1]:<24} {trc[2]:<24} {nref_forced[cc]:<6} {nref_cooling[cc]:<6} {star_particle_mass}")
elif use_clump_tracks:
    for cc in range(len(global_clump_ids)):
        clump_id = global_clump_ids[cc]
        hf_t = h5py.File(args.clump_tracking_dir + GalName+"_"+args.run+"_clump_tracking.h5",'r')
        itr=0
        for snapshot in snapshots:
            key = 'global_clump_mapping_'+snapshot
            clump_mapping = hf_t[key][...]
            clump_id = clump_mapping[clump_mapping[:,0]==global_clump_ids[cc],1] #find the clump id in this snapshot that corresponds to the global clump id
            clump_x = (hfc[str(clump_id)]['leaf_x'][...]*ds.units.kpc).in_units('code_length').v
            clump_y = (hfc[str(clump_id)]['leaf_y'][...]*ds.units.kpc).in_units('code_length').v
            clump_z = (hfc[str(clump_id)]['leaf_z'][...]*ds.units.kpc).in_units('code_length').v
            with open(output_file, "a") as f:
                blc = [clump_x-box_size[0]/2., clump_y-box_size[1]/2., clump_z-box_size[2]/2.]
                trc = [clump_x+box_size[0]/2., clump_y+box_size[1]/2., clump_z+box_size[2]/2.]
                f.write(f"\n{cc} {redshifts[snapshot]} {blc[0]} {blc[1]} {blc[2]} {trc[0]} {trc[1]} {trc[2]} {nref_forced[cc]} {nref_cooling[cc]} {star_particle_mass}")
hfc.close()

#Write the central halo track
track_redshifts=np.array(track_redshifts)
for snapshot in snapshots:
    z = redshifts[snapshot]
    zidx = np.argmin(np.abs(track_redshifts - z))
    
    blc = track_blc[zidx]
    trc = track_trc[zidx]
    with open(output_file, "a") as f:
        f.write(f"\n{cc+1:<6} {track_redshifts[zidx]:<24} {blc[0]:<24} {blc[1]:<24} {blc[2]:<24} {trc[0]:<24} {trc[1]:<24} {trc[2]:<24} {main_nref_forced:<6} {main_nref_cooling:<6} {star_particle_mass}")
