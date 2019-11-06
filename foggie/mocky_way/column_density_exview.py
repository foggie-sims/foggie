### calculate the column denisty of a halo from external angle
# with random sightlines.
# python column_density_exview.py
#
# 10/26/2019, Yong Zheng, UCB. 

import numpy as np
from foggie.mocky_way.core_funcs import ortho_find_yz
from foggie.mocky_way.core_funcs import calc_ray_ion_column_density

from foggie.mocky_way.core_funcs import prepdata
ds, ds_paras = prepdata('DD2175')

halo_center = ds_paras['halo_center']
rvir_codelength = ds.quan(160, 'kpc').in_units('code_length')

ion_list = ['HI', 'SiII', 'SiIII', 'SiIV', 'CII', 'CIV', 'OVI', 'NV',
            'OVII', 'OVIII', 'NeVII', 'NeVIII']
ion = 'HI'

nlos = 3  # the total number of random sightlines
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
fig_dir = 'figs/Nr_exview/fits'
tb_name = 'nref11n_nref10f_DD2175_N%s_exview.fits'%(ion)

save_to_file = '%s/%s'%(fig_dir, tb_name)
print("I am saving it to ", save_to_file)
t.writeto(save_to_file, overwrite=True)
