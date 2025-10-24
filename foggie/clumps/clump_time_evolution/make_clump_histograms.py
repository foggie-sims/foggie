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
    parser.set_defaults(run='nref11c_nref9f')

    parser.add_argument('--snapshot', metavar='snapshot', type=str, action='store', \
                        help='Which snapshot? Default is RD0042')
    parser.set_defaults(snapshot='RD0042')
    
    parser.add_argument('--snapshot_array_index', metavar='snapshot_array_index', type=str, action='store', \
                        help='Which snapshot number as fed in by the PBS Job array? Default is None, will override snapshot.')
    parser.set_defaults(snapshot_array_index=None)

    parser.add_argument('--clump_dir', metavar='clump_dir', type=str, action='store', \
                        help='Where is the clump hiearchy file')
    parser.set_defaults(clump_dir=None)

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




args = parse_args()

if args.snapshot_array_index is not None:
    if args.is_rd:
        args.snapshot = "RD"+args.snapshot_array_index.zfill(4)
    else:
        args.snapshot = "DD"+args.snapshot_array_index.zfill(4)

data_dir, output_path, run_loc, code_dir,trackname,halo_name,spectra_dir,infofile = get_run_loc_etc(args)

if args.system=='cameron_local':
    args.clump_dir = '/Users/ctrapp/Documents/foggie_analysis/clump_project/clump_catalog/'
elif args.system=='cameron_pleiades':
    args.clump_dir = '/nobackup/cwtrapp/clump_catalogs/halo_'+args.halo+'/'

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

GalName="UnknownHalo"
if halo_id == "008508":
    GalName="Tempest"
elif halo_id == "005036":
    GalName="Maelstrom"
elif halo_id == "005016":
    GalName="Squall"
elif halo_id == "004123":
    GalName="Blizzard"
elif halo_id == "002392":
    GalName="Hurricane"
elif halo_id == "002878":
    GalName="Cyclone"


#load the leaf clumps
leaf_masses = []
leaf_vx = []
leaf_vy = []
leaf_vz = []
leaf_vx_disk = []
leaf_vy_disk = []
leaf_vz_disk = []
leaf_hi_num_dense = []
leaf_mgii_num_dense = []
leaf_ovi_num_dense = []
leaf_volumes = []
leaf_x = []
leaf_y = []
leaf_z = []
leaf_x_disk = []
leaf_y_disk = []
leaf_z_disk = []

hiearchy_file = args.clump_dir + GalName+"_"+args.snapshot+"_"+args.run+"_ClumpTree.h5"
hf = h5py.File(hiearchy_file,'r')
leaf_clump_ids = hf['leaf_clump_ids'][...]

skip_adding_cell_ids = False
print("Adding cell ids...")
from foggie.clumps.clump_finder.utils_clump_finder import add_cell_id_field
add_cell_id_field(ds)

import trident
trident.add_ion_fields(ds, ions=['O VI','Mg II'])

gas_masses = refine_box['gas','mass'].in_units('Msun')
vx_disk = refine_box['gas','vx_disk'].in_units('km/s')
vy_disk = refine_box['gas','vy_disk'].in_units('km/s')
vz_disk = refine_box['gas','vz_disk'].in_units('km/s')
vx = refine_box['gas','vx'].in_units('km/s')
vy = refine_box['gas','vy'].in_units('km/s')
vz = refine_box['gas','vz'].in_units('km/s')
hi_num_dense = refine_box['gas','H_p0_number_density'].in_units('cm**-3')
mgii_num_dense = refine_box['gas','Mg_p1_number_density'].in_units('cm**-3')
ovi_num_dense = refine_box['gas','O_p5_number_density'].in_units('cm**-3')
volumes = refine_box['gas','cell_volume'].in_units('kpc**3')
x_disk = refine_box['gas','x_disk'].in_units('kpc')
y_disk = refine_box['gas','y_disk'].in_units('kpc')
z_disk = refine_box['gas','z_disk'].in_units('kpc')

x = refine_box['gas','x'].in_units('kpc')
y = refine_box['gas','y'].in_units('kpc')
z = refine_box['gas','z'].in_units('kpc')

cell_ids = refine_box['index','cell_id_2']


pbar = TqdmProgressBar("Loading Leaves...",len(leaf_clump_ids),position=0)
itr=0


for leaf_id in leaf_clump_ids:
    leaf_cell_ids = hf[str(leaf_id)]['cell_ids'][...]

    leaf_gas_mass = gas_masses[np.isin(cell_ids, leaf_cell_ids)].in_units('Msun')
    norm = np.sum(leaf_gas_mass)
    leaf_masses.append(norm.in_units('Msun').v)

    #mass weighted
    leaf_vx_disk.append( (np.sum( np.multiply(vx_disk[np.isin(cell_ids, leaf_cell_ids)],leaf_gas_mass)) / norm ).in_units('km/s').v)
    leaf_vy_disk.append( (np.sum( np.multiply(vy_disk[np.isin(cell_ids, leaf_cell_ids)],leaf_gas_mass)) / norm ).in_units('km/s').v)
    leaf_vz_disk.append( (np.sum( np.multiply(vz_disk[np.isin(cell_ids, leaf_cell_ids)],leaf_gas_mass)) / norm ).in_units('km/s').v)

    leaf_vx.append( (np.sum( np.multiply(vx[np.isin(cell_ids, leaf_cell_ids)],leaf_gas_mass)) / norm ).in_units('km/s').v)
    leaf_vy.append( (np.sum( np.multiply(vy[np.isin(cell_ids, leaf_cell_ids)],leaf_gas_mass)) / norm ).in_units('km/s').v)
    leaf_vz.append( (np.sum( np.multiply(vz[np.isin(cell_ids, leaf_cell_ids)],leaf_gas_mass)) / norm ).in_units('km/s').v)

    leaf_x_disk.append(  (np.sum( np.multiply(x_disk[np.isin(cell_ids, leaf_cell_ids)], leaf_gas_mass)) / norm ).in_units('kpc').v)
    leaf_y_disk.append(  (np.sum( np.multiply(y_disk[np.isin(cell_ids, leaf_cell_ids)], leaf_gas_mass)) / norm ).in_units('kpc').v)
    leaf_z_disk.append(  (np.sum( np.multiply(z_disk[np.isin(cell_ids, leaf_cell_ids)], leaf_gas_mass)) / norm ).in_units('kpc').v)

    leaf_x.append(  (np.sum( np.multiply(x[np.isin(cell_ids, leaf_cell_ids)], leaf_gas_mass)) / norm ).in_units('kpc').v)
    leaf_y.append(  (np.sum( np.multiply(y[np.isin(cell_ids, leaf_cell_ids)], leaf_gas_mass)) / norm ).in_units('kpc').v)
    leaf_z.append(  (np.sum( np.multiply(z[np.isin(cell_ids, leaf_cell_ids)], leaf_gas_mass)) / norm ).in_units('kpc').v)

    leaf_hi_num_dense.append(np.mean(hi_num_dense[np.isin(cell_ids, leaf_cell_ids)]).in_units('cm**-3').v)
    leaf_mgii_num_dense.append(np.mean(mgii_num_dense[np.isin(cell_ids, leaf_cell_ids)]).in_units('cm**-3').v)
    leaf_ovi_num_dense.append(np.mean(ovi_num_dense[np.isin(cell_ids, leaf_cell_ids)]).in_units('cm**-3').v)


    leaf_volumes.append(np.sum(volumes[np.isin(cell_ids, leaf_cell_ids)]).in_units('kpc**3').v)
 

    #leaf = load_clump(ds,clump_cell_ids=leaf_cell_ids, skip_adding_cell_ids=skip_adding_cell_ids)

   # leaf_masses.append(np.sum(leaf['gas','mass']).in_units('Msun').v)

    pbar.update(itr)
    itr+=1
    skip_adding_cell_ids = True


plt.figure()
min_mass = 1e2
max_mass = 1e10
min_mass = 2
max_mass = 10
nbins = 200
labelsize=18
ticksize=18
N, binedges, binnum = stats.binned_statistic(np.log10(np.array(leaf_masses)), np.ones_like(np.array(leaf_masses)), statistic='sum', bins=nbins, range=[min_mass,max_mass])

dM= np.copy(N)*0
for i in range(len(binedges)-1):
    #dM[i] = binedges[i+1]-binedges[i] #
    dM[i] = 10**binedges[i+1] - 10**binedges[i]

hf = h5py.File(args.output_dir + GalName+"_"+args.snapshot+"_"+args.run+"_clump_stats.h5",'w')
hf.create_dataset('leaf_vx_disk', data=np.array(leaf_vx_disk))
hf.create_dataset('leaf_vy_disk', data=np.array(leaf_vy_disk))
hf.create_dataset('leaf_vz_disk', data=np.array(leaf_vz_disk))
hf.create_dataset('leaf_vx', data=np.array(leaf_vx))
hf.create_dataset('leaf_vy', data=np.array(leaf_vy))
hf.create_dataset('leaf_vz', data=np.array(leaf_vz))
hf.create_dataset('leaf_x_disk', data=np.array(leaf_x_disk))
hf.create_dataset('leaf_y_disk', data=np.array(leaf_y_disk))
hf.create_dataset('leaf_z_disk', data=np.array(leaf_z_disk))
hf.create_dataset('leaf_x', data=np.array(leaf_x))
hf.create_dataset('leaf_y', data=np.array(leaf_y))
hf.create_dataset('leaf_z', data=np.array(leaf_z))
hf.create_dataset('leaf_hi_num_dense', data=np.array(leaf_hi_num_dense))
hf.create_dataset('leaf_mgii_num_dense', data=np.array(leaf_mgii_num_dense))
hf.create_dataset('leaf_ovi_num_dense', data=np.array(leaf_ovi_num_dense))
hf.create_dataset('leaf_volumes', data=np.array(leaf_volumes))
hf.create_dataset('leaf_masses', data=np.array(leaf_masses))
hf.create_dataset('leaf_clump_ids', data=np.array(leaf_clump_ids))
hf.close()