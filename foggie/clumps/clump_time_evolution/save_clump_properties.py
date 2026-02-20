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

    parser.add_argument('--clump_file', metavar='clump_file', type=str, action='store', \
                        help='Where is the clump hiearchy file')
    parser.set_defaults(clump_file=None)

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

    parser.add_argument('--data_dir', metavar='data_dir', type=str, action='store', \
                        help='Override data directory location in get_run_loc_etc. Default is None.')
    parser.set_defaults(data_dir=None)

    parser.add_argument('--do_tracer_fluids', metavar='do_tracer_fluids', type=bool, action='store', \
                        help='Calculate tracer fluid stats? Default is False.')
    parser.set_defaults(do_tracer_fluids=False)

    parser.add_argument('--modify_existing_clump_hierarchy', metavar='modify_existing_clump_hierarchy', type=bool, action=argparse.BooleanOptionalAction, \
                        help='Add fields to the current clump tree. Will write separate clump stats file if False. Default is True.')
    parser.set_defaults(modify_existing_clump_hierarchy=False)

    parser.add_argument('--write_separate_stats_file', metavar='write_separate_stats_file', type=bool, action=argparse.BooleanOptionalAction, \
                        help='Write a separate stats file (outside of the clump hierarchy). Default is False. If modify_existing_clump_hierarchy is False, this will be set to True regardless.')
    parser.set_defaults(write_separate_stats_file=False)

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
    data_dir = args.data_dir

if args.clump_file is None:
    if args.system=='cameron_local':
        args.clump_file = '/Users/ctrapp/Documents/foggie_analysis/clump_project/clump_catalog/'
    elif args.system=='cameron_pleiades':
        args.clump_file = '/nobackup/cwtrapp/clump_catalogs/halo_'+args.halo+'/'

halo_id = args.halo #008508
snapshot = args.snapshot #RD0042
run = args.run #nref11c_nref9f

gal_name = halo_id_to_name(halo_id)
gal_name+="_"+snapshot+"_"+run

snap_name = data_dir + "halo_"+halo_id+"/"+run+"/"+snapshot+"/"+snapshot

trackname = code_dir+"/halo_tracks/"+halo_id+"/nref11n_selfshield_15/halo_track_200kpc_nref9"

halo_c_v_name = code_dir+"/halo_infos/"+halo_id+"/nref11c_nref9f/halo_c_v"

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
#leaf_vx_disk = []
#leaf_vy_disk = []
#leaf_vz_disk = []
leaf_hi_num_dense = []
leaf_mgii_num_dense = []
leaf_oi_num_dense = []
leaf_oii_num_dense = []
leaf_oiii_num_dense = []
leaf_oiv_num_dense = []
leaf_ov_num_dense = []
leaf_ovi_num_dense = []
leaf_volumes = []
leaf_x = []
leaf_y = []
leaf_z = []
#leaf_x_disk = []
#leaf_y_disk = []
#leaf_z_disk = []
leaf_metallicity = []
leaf_pressure = []
leaf_temperature = []

leaf_tf1_mass = []
leaf_tf2_mass = []
leaf_tf3_mass = []
leaf_tf4_mass = []
leaf_tf5_mass = []
leaf_tf6_mass = []
leaf_tf7_mass = []
leaf_tf8_mass = []

shell_masses = []
shell_volumes = []
shell_vx = []
shell_vy = []
shell_vz = []
shell_hi_num_dense = []
shell_mgii_num_dense = []
shell_oi_num_dense = []
shell_oii_num_dense = []
shell_oiii_num_dense = []
shell_oiv_num_dense = []
shell_ov_num_dense = []
shell_ovi_num_dense = []
shell_metallicity = []
shell_pressure = []
shell_temperature = []


hiearchy_file = args.clump_file#args.clump_dir + GalName+"_"+args.snapshot+"_"+args.run+"_ClumpTree.h5"
if args.modify_existing_clump_hierarchy:
    hf = h5py.File(hiearchy_file,'r+')
else:
    hf = h5py.File(hiearchy_file,'r')

leaf_clump_ids = hf['leaf_clump_ids'][...]

skip_adding_cell_ids = False
print("Adding cell ids...")
from foggie.clumps.clump_finder.utils_clump_finder import add_cell_id_field
add_cell_id_field(ds)

import trident
trident.add_ion_fields(ds, ions=['O II','O III','O IV','O V','O VI','Mg II'])

gas_masses = refine_box['gas','mass'].in_units('Msun')
vx = refine_box['gas','velocity_x'].in_units('km/s')
vy = refine_box['gas','velocity_y'].in_units('km/s')
vz = refine_box['gas','velocity_z'].in_units('km/s')
hi_num_dense = refine_box['gas','H_p0_number_density'].in_units('cm**-3')
mgii_num_dense = refine_box['gas','Mg_p1_number_density'].in_units('cm**-3')
oii_num_dense = refine_box['gas','O_p1_number_density'].in_units('cm**-3')
oiii_num_dense = refine_box['gas','O_p2_number_density'].in_units('cm**-3')
oiv_num_dense = refine_box['gas','O_p3_number_density'].in_units('cm**-3')
ov_num_dense = refine_box['gas','O_p4_number_density'].in_units('cm**-3')
ovi_num_dense = refine_box['gas','O_p5_number_density'].in_units('cm**-3')
volumes = refine_box['gas','cell_volume'].in_units('kpc**3')


x = refine_box['gas','x'].in_units('kpc')
y = refine_box['gas','y'].in_units('kpc')
z = refine_box['gas','z'].in_units('kpc')

metallicity = refine_box['gas','metallicity']
pressure = refine_box['gas','pressure']
temperature = refine_box['gas','temperature']

cell_ids = refine_box['index','cell_id_2']


code_density = ds.units.code_mass / ds.units.code_length**3
code_density = ds.units.code_density
if args.do_tracer_fluids:
    #try:
        tf1 = refine_box['enzo','TracerFluid01'] * code_density
        tf2 = refine_box['enzo','TracerFluid02'] * code_density
        tf3 = refine_box['enzo','TracerFluid03'] * code_density
        tf4 = refine_box['enzo','TracerFluid04'] * code_density
        tf5 = refine_box['enzo','TracerFluid05'] * code_density
        tf6 = refine_box['enzo','TracerFluid06'] * code_density
        tf7 = refine_box['enzo','TracerFluid07'] * code_density
        tf8 = refine_box['enzo','TracerFluid08'] * code_density
    #except:
    #    args.do_tracer_fluids=False




pbar = TqdmProgressBar("Calculating Leaf stats...",len(leaf_clump_ids),position=0)
itr=0



for leaf_id in leaf_clump_ids:
    leaf_cell_ids = hf[str(leaf_id)]['cell_ids'][...]
    if itr==0: print(hf[str(leaf_id)].keys())
    try:
        shell_cell_ids = hf[str(leaf_id)]['shell_cell_ids'][...]
    except:
        shell_cell_ids = None

    mask = np.isin(cell_ids, leaf_cell_ids)
    leaf_gas_mass = gas_masses[mask].in_units('Msun')
    norm = np.sum(leaf_gas_mass)
    leaf_masses.append(norm.in_units('Msun').v)
    
    leaf_gas_volume = volumes[mask].in_units('kpc**3')
    vol_norm = np.sum(leaf_gas_volume)
    leaf_volumes.append(vol_norm.in_units('kpc**3').v)

    leaf_vx.append( (np.sum( np.multiply(vx[mask],leaf_gas_mass)) / norm ).in_units('km/s').v)
    leaf_vy.append( (np.sum( np.multiply(vy[mask],leaf_gas_mass)) / norm ).in_units('km/s').v)
    leaf_vz.append( (np.sum( np.multiply(vz[mask],leaf_gas_mass)) / norm ).in_units('km/s').v)

    leaf_x.append(  (np.sum( np.multiply(x[mask], leaf_gas_mass)) / norm ).in_units('kpc').v)
    leaf_y.append(  (np.sum( np.multiply(y[mask], leaf_gas_mass)) / norm ).in_units('kpc').v)
    leaf_z.append(  (np.sum( np.multiply(z[mask], leaf_gas_mass)) / norm ).in_units('kpc').v)


    leaf_hi_num_dense.append(  (np.sum( np.multiply(hi_num_dense[mask], leaf_gas_volume)) / vol_norm ).in_units('cm**-3').v)
    leaf_mgii_num_dense.append(  (np.sum( np.multiply(mgii_num_dense[mask], leaf_gas_volume)) / vol_norm ).in_units('cm**-3').v)
    leaf_oii_num_dense.append(  (np.sum( np.multiply(oii_num_dense[mask], leaf_gas_volume)) / vol_norm ).in_units('cm**-3').v)
    leaf_oiii_num_dense.append(  (np.sum( np.multiply(oiii_num_dense[mask], leaf_gas_volume)) / vol_norm ).in_units('cm**-3').v)
    leaf_oiv_num_dense.append(  (np.sum( np.multiply(oiv_num_dense[mask], leaf_gas_volume)) / vol_norm ).in_units('cm**-3').v)
    leaf_ov_num_dense.append(  (np.sum( np.multiply(ov_num_dense[mask], leaf_gas_volume)) / vol_norm ).in_units('cm**-3').v)
    leaf_ovi_num_dense.append(  (np.sum( np.multiply(ovi_num_dense[mask], leaf_gas_volume)) / vol_norm ).in_units('cm**-3').v)

    leaf_metallicity.append(  (np.sum( np.multiply(metallicity[mask], leaf_gas_mass)) / norm ))
    leaf_pressure.append(  (np.sum( np.multiply(pressure[mask], leaf_gas_mass)) / norm ).in_units('Ba').v)
    leaf_temperature.append(  (np.sum( np.multiply(temperature[mask], leaf_gas_mass)) / norm ).in_units('K').v)



    if args.do_tracer_fluids:
        leaf_tf1_mass.append( np.sum( np.multiply(tf1[mask] , volumes[mask] )) ) #Give tracer fluid mass in leaf clump
        leaf_tf2_mass.append( np.sum( np.multiply(tf2[mask] , volumes[mask] )) )
        leaf_tf3_mass.append( np.sum( np.multiply(tf3[mask] , volumes[mask] )) )
        leaf_tf4_mass.append( np.sum( np.multiply(tf4[mask] , volumes[mask] )) )
        leaf_tf5_mass.append( np.sum( np.multiply(tf5[mask] , volumes[mask] )) )
        leaf_tf6_mass.append( np.sum( np.multiply(tf6[mask] , volumes[mask] )) )
        leaf_tf7_mass.append( np.sum( np.multiply(tf7[mask] , volumes[mask] )) )
        leaf_tf8_mass.append( np.sum( np.multiply(tf8[mask] , volumes[mask] )) )

    if shell_cell_ids is not None:
        shell_mask = np.isin(cell_ids, shell_cell_ids)
        shell_gas_mass = gas_masses[shell_mask].in_units('Msun')
        shell_norm = np.sum(shell_gas_mass)

        shell_gas_volume = volumes[shell_mask].in_units('kpc**3')
        shell_vol_norm = np.sum(shell_gas_volume)

        shell_masses.append(shell_norm.in_units('Msun').v)
        shell_volumes.append(shell_vol_norm.in_units('kpc**3').v)

        shell_vx.append( (np.sum( np.multiply(vx[shell_mask],shell_gas_mass)) / shell_norm ).in_units('km/s').v)
        shell_vy.append( (np.sum( np.multiply(vy[shell_mask],shell_gas_mass)) / shell_norm ).in_units('km/s').v)
        shell_vz.append( (np.sum( np.multiply(vz[shell_mask],shell_gas_mass)) / shell_norm ).in_units('km/s').v)


        shell_hi_num_dense.append(  (np.sum( np.multiply(hi_num_dense[shell_mask], shell_gas_volume)) / shell_vol_norm ).in_units('cm**-3').v)
        shell_mgii_num_dense.append(  (np.sum( np.multiply(mgii_num_dense[shell_mask], shell_gas_volume)) / shell_vol_norm ).in_units('cm**-3').v)
        shell_oii_num_dense.append(  (np.sum( np.multiply(oii_num_dense[shell_mask], shell_gas_volume)) / shell_vol_norm ).in_units('cm**-3').v)
        shell_oiii_num_dense.append(  (np.sum( np.multiply(oiii_num_dense[shell_mask], shell_gas_volume)) / shell_vol_norm ).in_units('cm**-3').v)
        shell_oiv_num_dense.append(  (np.sum( np.multiply(oiv_num_dense[shell_mask], shell_gas_volume)) / shell_vol_norm ).in_units('cm**-3').v)
        shell_ov_num_dense.append(  (np.sum( np.multiply(ov_num_dense[shell_mask], shell_gas_volume)) / shell_vol_norm ).in_units('cm**-3').v)
        shell_ovi_num_dense.append(  (np.sum( np.multiply(ovi_num_dense[shell_mask], shell_gas_volume)) / shell_vol_norm ).in_units('cm**-3').v)

        shell_metallicity.append(  (np.sum( np.multiply(metallicity[shell_mask], shell_gas_mass)) / shell_norm ))
        shell_pressure.append(  (np.sum( np.multiply(pressure[shell_mask], shell_gas_mass)) / shell_norm ).in_units('Ba').v)
        shell_temperature.append(  (np.sum( np.multiply(temperature[shell_mask], shell_gas_mass)) / shell_norm ).in_units('K').v)


    pbar.update(itr)
    itr+=1
    skip_adding_cell_ids = True



    if args.modify_existing_clump_hierarchy:
        hf[str(leaf_id)].create_dataset('leaf_vx', data=np.array(leaf_vx[-1]))
        hf[str(leaf_id)].create_dataset('leaf_vy', data=np.array(leaf_vy[-1]))
        hf[str(leaf_id)].create_dataset('leaf_vz', data=np.array(leaf_vz[-1]))
        hf[str(leaf_id)].create_dataset('leaf_x', data=np.array(leaf_x[-1]))
        hf[str(leaf_id)].create_dataset('leaf_y', data=np.array(leaf_y[-1]))
        hf[str(leaf_id)].create_dataset('leaf_z', data=np.array(leaf_z[-1]))
        hf[str(leaf_id)].create_dataset('leaf_hi_num_dense', data=np.array(leaf_hi_num_dense[-1]))
        hf[str(leaf_id)].create_dataset('leaf_mgii_num_dense', data=np.array(leaf_mgii_num_dense[-1]))
        hf[str(leaf_id)].create_dataset('leaf_oii_num_dense', data=np.array(leaf_oii_num_dense[-1]))
        hf[str(leaf_id)].create_dataset('leaf_oiii_num_dense', data=np.array(leaf_oiii_num_dense[-1]))
        hf[str(leaf_id)].create_dataset('leaf_oiv_num_dense', data=np.array(leaf_oiv_num_dense[-1]))
        hf[str(leaf_id)].create_dataset('leaf_ov_num_dense', data=np.array(leaf_ov_num_dense[-1]))
        hf[str(leaf_id)].create_dataset('leaf_ovi_num_dense', data=np.array(leaf_ovi_num_dense[-1]))
        hf[str(leaf_id)].create_dataset('leaf_volume', data=np.array(leaf_volumes[-1]))
        hf[str(leaf_id)].create_dataset('leaf_metallicity', data=np.array(leaf_metallicity[-1]))
        hf[str(leaf_id)].create_dataset('leaf_pressure', data=np.array(leaf_pressure[-1]))
        hf[str(leaf_id)].create_dataset('leaf_temperature', data=np.array(leaf_temperature[-1]))
        hf[str(leaf_id)].create_dataset('leaf_mass', data=np.array(leaf_masses[-1]))
        if args.do_tracer_fluids:
            hf[str(leaf_id)].create_dataset('leaf_tf1_mass', data=np.array(leaf_tf1_mass[-1]))
            hf[str(leaf_id)].create_dataset('leaf_tf2_mass', data=np.array(leaf_tf2_mass[-1]))
            hf[str(leaf_id)].create_dataset('leaf_tf3_mass', data=np.array(leaf_tf3_mass[-1]))
            hf[str(leaf_id)].create_dataset('leaf_tf4_mass', data=np.array(leaf_tf4_mass[-1]))
            hf[str(leaf_id)].create_dataset('leaf_tf5_mass', data=np.array(leaf_tf5_mass[-1]))
            hf[str(leaf_id)].create_dataset('leaf_tf6_mass', data=np.array(leaf_tf6_mass[-1]))
            hf[str(leaf_id)].create_dataset('leaf_tf7_mass', data=np.array(leaf_tf7_mass[-1]))
            hf[str(leaf_id)].create_dataset('leaf_tf8_mass', data=np.array(leaf_tf8_mass[-1]))
        if shell_cell_ids is not None:
            hf[str(leaf_id)].create_dataset('shell_mass', data=np.array(shell_masses[-1]))
            hf[str(leaf_id)].create_dataset('shell_volume', data=np.array(shell_volumes[-1]))
            hf[str(leaf_id)].create_dataset('shell_vx', data=np.array(shell_vx[-1]))
            hf[str(leaf_id)].create_dataset('shell_vy', data=np.array(shell_vy[-1]))
            hf[str(leaf_id)].create_dataset('shell_vz', data=np.array(shell_vz[-1]))
            hf[str(leaf_id)].create_dataset('shell_hi_num_dense', data=np.array(shell_hi_num_dense[-1]))
            hf[str(leaf_id)].create_dataset('shell_mgii_num_dense', data=np.array(shell_mgii_num_dense[-1]))
            hf[str(leaf_id)].create_dataset('shell_oii_num_dense', data=np.array(shell_oii_num_dense[-1]))
            hf[str(leaf_id)].create_dataset('shell_oiii_num_dense', data=np.array(shell_oiii_num_dense[-1]))
            hf[str(leaf_id)].create_dataset('shell_oiv_num_dense', data=np.array(shell_oiv_num_dense[-1]))
            hf[str(leaf_id)].create_dataset('shell_ov_num_dense', data=np.array(shell_ov_num_dense[-1]))
            hf[str(leaf_id)].create_dataset('shell_ovi_num_dense', data=np.array(shell_ovi_num_dense[-1]))
            hf[str(leaf_id)].create_dataset('shell_metallicity', data=np.array(shell_metallicity[-1]))
            hf[str(leaf_id)].create_dataset('shell_pressure', data=np.array(shell_pressure[-1]))
            hf[str(leaf_id)].create_dataset('shell_temperature', data=np.array(shell_temperature[-1]))

hf.close()


if shell_cell_ids is not None:
    print("\nSuccessfully found clump and shell stats in clump hierarchy file:",hiearchy_file)
else:
    print("\nWarning: could not read shell data in clump hierarchy file. Only leaf clump stats found in:",hiearchy_file)

#Write a smaller, separate file for all clump stats
if args.write_separate_stats_file or not args.modify_existing_clump_hierarchy:  
    hf = h5py.File(args.output_dir + GalName+"_"+args.snapshot+"_"+args.run+"_clump_stats.h5",'w')
    hf.create_dataset('leaf_vx', data=np.array(leaf_vx))
    hf.create_dataset('leaf_vy', data=np.array(leaf_vy))
    hf.create_dataset('leaf_vz', data=np.array(leaf_vz))
    hf.create_dataset('leaf_x', data=np.array(leaf_x))
    hf.create_dataset('leaf_y', data=np.array(leaf_y))
    hf.create_dataset('leaf_z', data=np.array(leaf_z))
    hf.create_dataset('leaf_hi_num_dense', data=np.array(leaf_hi_num_dense))
    hf.create_dataset('leaf_mgii_num_dense', data=np.array(leaf_mgii_num_dense))
    hf.create_dataset('leaf_oii_num_dense', data=np.array(leaf_oii_num_dense))
    hf.create_dataset('leaf_oiii_num_dense', data=np.array(leaf_oiii_num_dense))
    hf.create_dataset('leaf_oiv_num_dense', data=np.array(leaf_oiv_num_dense))
    hf.create_dataset('leaf_ov_num_dense', data=np.array(leaf_ov_num_dense))
    hf.create_dataset('leaf_ovi_num_dense', data=np.array(leaf_ovi_num_dense))
    hf.create_dataset('leaf_volume', data=np.array(leaf_volumes))
    hf.create_dataset('leaf_metallicity', data=np.array(leaf_metallicity))
    hf.create_dataset('leaf_pressure', data=np.array(leaf_pressure))
    hf.create_dataset('leaf_temperature', data=np.array(leaf_temperature))
    hf.create_dataset('leaf_mass', data=np.array(leaf_masses))
    hf.create_dataset('leaf_clump_ids', data=np.array(leaf_clump_ids))
    if args.do_tracer_fluids:
        hf.create_dataset('leaf_tf1_mass', data=np.array(leaf_tf1_mass))
        hf.create_dataset('leaf_tf2_mass', data=np.array(leaf_tf2_mass))
        hf.create_dataset('leaf_tf3_mass', data=np.array(leaf_tf3_mass))
        hf.create_dataset('leaf_tf4_mass', data=np.array(leaf_tf4_mass))
        hf.create_dataset('leaf_tf5_mass', data=np.array(leaf_tf5_mass))
        hf.create_dataset('leaf_tf6_mass', data=np.array(leaf_tf6_mass))
        hf.create_dataset('leaf_tf7_mass', data=np.array(leaf_tf7_mass))
        hf.create_dataset('leaf_tf8_mass', data=np.array(leaf_tf8_mass))
    if shell_cell_ids is not None:
        hf.create_dataset('shell_mass', data=np.array(shell_masses))
        hf.create_dataset('shell_volume', data=np.array(shell_volumes))
        hf.create_dataset('shell_vx', data=np.array(shell_vx))
        hf.create_dataset('shell_vy', data=np.array(shell_vy))
        hf.create_dataset('shell_vz', data=np.array(shell_vz))
        hf.create_dataset('shell_hi_num_dense', data=np.array(shell_hi_num_dense))
        hf.create_dataset('shell_mgii_num_dense', data=np.array(shell_mgii_num_dense))
        hf.create_dataset('shell_oii_num_dense', data=np.array(shell_oii_num_dense))
        hf.create_dataset('shell_oiii_num_dense', data=np.array(shell_oiii_num_dense))
        hf.create_dataset('shell_oiv_num_dense', data=np.array(shell_oiv_num_dense))
        hf.create_dataset('shell_ov_num_dense', data=np.array(shell_ov_num_dense))
        hf.create_dataset('shell_ovi_num_dense', data=np.array(shell_ovi_num_dense))
        hf.create_dataset('shell_metallicity', data=np.array(shell_metallicity))
        hf.create_dataset('shell_pressure', data=np.array(shell_pressure))
        hf.create_dataset('shell_temperature', data=np.array(shell_temperature))
    hf.close()