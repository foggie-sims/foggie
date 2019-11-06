#####
# special on pleiades
#### Reading in the dataset ###
import os
import mocky_way_modules  # read this in before reading in foggie and yt
from mocky_way_modules import data_dir_sys_dir
data_dir, sys_dir = data_dir_sys_dir()
os.sys.path.insert(0, sys_dir)

import yt
# import foggie

import numpy as np
from ortho_find_yz import ortho_find_yz
from calc_ray_ion_column_density import calc_ray_ion_column_density

sim_name = 'nref11n_nref10f'
dd_name = 'DD2175'
nlos = 10000

import sys
ion = sys.argv[1]
# ion = 'HI'  # ['HI', 'SiII', 'SiIII', 'SiIV', 'CII', 'CIV', 'OVI', 'NV',
#             # 'OVII', 'OVIII', 'NeVII', 'NeVIII']

#### Reading in the dataset ###
ds_file = '%s/%s/%s/%s'%(data_dir, sim_name, dd_name, dd_name)
ds = yt.load(ds_file)
zsnap = ds.get_parameter('CosmologyCurrentRedshift')

## post processing the data
import trident
td_ion_list = ['Si II', 'Si III', 'Si IV', 'C II', 'C IV', 'O VI', 'N V',
               'O VII', 'O VIII', 'Ne VII', 'Ne VIII']
print("Adding ion fields: ", td_ion_list)
trident.add_ion_fields(ds, ftype="gas", ions=td_ion_list, force_override=True)

### now let's find halo center ###
from foggie.get_halo_center import get_halo_center
from foggie.get_refine_box import get_refine_box
from astropy.table import Table
halo_track = '%s/%s/halo_track'%(data_dir, sim_name)
track = Table.read(halo_track, format='ascii')
track.sort('col1')
box_paras = get_refine_box(ds, zsnap, track)
refine_box = box_paras[0]
refine_box_center = box_paras[1]
refine_width_code = box_paras[2]
halo_center, halo_velocity = get_halo_center(ds, refine_box_center)
halo_center = ds.arr(halo_center, 'code_length')
rvir_codelength = ds.quan(160, 'kpc').in_units('code_length')

tot_r = np.zeros(nlos)
tot_N = np.zeros(nlos)
for i in range(nlos):
    # first find the impact parameter
    random_seed = np.random.randint(0, high=nlos*100)
    rand_length_kpc = np.random.randint(0, high=160)+np.random.normal()
    rand_length_codelength = ds.quan(rand_length_kpc, 'kpc').in_units('code_length')
    tot_r[i] = rand_length_kpc

    # then find a random line of sight vector with the impact parameter
    rand_vec = np.random.sample(3)
    rand_vec /= np.linalg.norm(rand_vec) # normalize it
    mid_point = halo_center + rand_vec*rand_length_codelength

    los_vec = ortho_find_yz(rand_vec, random_seed=random_seed)[1]
    los_ray_start = mid_point - los_vec*rvir_codelength
    los_ray_end = mid_point + los_vec*rvir_codelength

    ray_Nion = calc_ray_ion_column_density(ds, ion, los_ray_start, los_ray_end)[0]
    tot_N[i] = ray_Nion

# save the data to fits file
##### now saving the data ####
import astropy.io.fits as fits
c1 = fits.Column(name='impact_para', array=tot_r, format='D')
c2 = fits.Column(name='Nion', array=tot_N, format='D')

all_cols = [c1, c2]
t = fits.BinTableHDU.from_columns(all_cols)
fig_dir = '%s/mocky_way/figs/Nr_exview/fits'%(sys_dir)
tb_name = '%s_%s_N%s_exview.fits'%(sim_name, dd_name, ion)

save_to_file = '%s/%s'%(fig_dir, tb_name)
print("I am saving it to ", save_to_file)
t.writeto(save_to_file, overwrite=True)
