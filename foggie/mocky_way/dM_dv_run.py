### This function calculates the mass flux per velocity bins, and also
# it split the gas into different temperature bins.
#
# rewrite the structure of the code for DD2175, and merge into foggie.mocky_way
# 10/14/2019, Yong Zheng, UCB.
#
# # 08/14/2019, Yong add in the option to shift the observer on the solar circle.
#
# Started, Aug 11, 2019, Yong Zheng, UCB

import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['font.family'] = 'stixgeneral'
from foggie.mocky_way.core_funcs import prepdata
from foggie.mocky_way.core_funcs import obj_source_all_disk_cgm


sim_name = 'nref11n_nref10f'
dd_name = 'DD2175'

ds, ds_paras = prepdata(dd_name, sim_name=sim_name)
obj_tag = 'cgm-20kpc' # cgm-15kpc, cgm-20kpc, cgm-rvir
#obs_point = 'offcenter_location'  # halo_center, offcenter_location
#obs_bulkvel = 'offcenter_bulkvel' # disk_bulkvel, offcenter_bulkvel
obs_point = 'halo_center'  # halo_center, offcenter_location
obs_bulkvel = 'disk_bulkvel' # disk_bulkvel, offcenter_bulkvel

#### No need to change starting this line ####
print("I am doing the calculation from %s for %s..."%(obs_point, obj_tag))
obj_source = obj_source_all_disk_cgm(ds, ds_paras, obj_tag)
obj_source.set_field_parameter("observer_location", ds_paras[obs_point])
obj_source.set_field_parameter("observer_bulkvel", ds_paras[obs_bulkvel])
obj_vlsr = obj_source["gas", "los_velocity_mw"]
obj_cell_mass = obj_source["gas", "cell_mass"]
obj_temperature = obj_source["gas", "temperature"]

# setup (dM, dv) step intervals
dv = 20  # the velocity bin every 20 km/s
vrange = [-1000, 1000]
dv_bins = np.mgrid[vrange[0]:vrange[1]+dv:dv]

### first let's get the halo mass distribution as a func of vel, regardless of T
print("Calculating dM-dv for all gas....")
dM_all = np.zeros(dv_bins.size)
for iv in range(dv_bins.size):
    va = dv_bins[iv]-dv/2.
    vb = dv_bins[iv]+dv/2.
    indv = np.all([obj_vlsr>=va, obj_vlsr<vb], axis=0)
    if len(obj_cell_mass[indv])!=0:
        dM_all[iv] = obj_cell_mass[indv].sum().in_units("Msun")/dv

### then let's break up the mass into different temperature categories
print("Now let us break up the gas mass into different temperature ranges...")
dM_cold = np.zeros(dv_bins.size)
dM_cool = np.zeros(dv_bins.size)
dM_warm = np.zeros(dv_bins.size)
dM_hot = np.zeros(dv_bins.size)

from foggie.mocky_way.core_funcs import temperature_category
temp_dict = temperature_category()

for iv in range(dv_bins.size):
    va = dv_bins[iv]-dv/2.
    vb = dv_bins[iv]+dv/2.
    indv = np.all([obj_vlsr>=va, obj_vlsr<vb], axis=0)

    if len(obj_cell_mass[indv])!=0:
        iv_mass = obj_cell_mass[indv]
        iv_T = obj_temperature[indv]

        ### cold gas ####
        # ind_cold = iv_T <= 1e4
        cold_T = temp_dict['cold']
        ind_cold = np.all([iv_T>cold_T[0], iv_T<=cold_T[1]], axis=0)
        if len(iv_mass[ind_cold])!= 0:
            dM_cold[iv] = iv_mass[ind_cold].sum().in_units("Msun")/dv

        ### cool gas ####
        # ind_cool = np.all([iv_T>1e4, iv_T<=1e5], axis=0)
        cool_T = temp_dict['cool']
        ind_cool = np.all([iv_T>cool_T[0], iv_T<=cool_T[1]], axis=0)
        if len(iv_mass[ind_cool]) != 0:
            dM_cool[iv] = iv_mass[ind_cool].sum().in_units("Msun")/dv

        ### warm gas ####
        # ind_warm = np.all([iv_T>1e5, iv_T<=1e6], axis=0)
        warm_T = temp_dict['warm']
        ind_warm = np.all([iv_T>warm_T[0], iv_T<=warm_T[1]], axis=0)
        if len(iv_mass[ind_warm]) != 0:
            dM_warm[iv] = iv_mass[ind_warm].sum().in_units("Msun")/dv

        ## now this is hot
        # ind_hot = iv_T > 1e6
        hot_T = temp_dict['hot']
        ind_hot = np.all([iv_T>hot_T[0], iv_T<hot_T[1]], axis=0)
        if len(iv_mass[ind_hot]) != 0:
            dM_hot[iv] = iv_mass[ind_hot].sum().in_units("Msun")/dv

##### now saving the data ####
c1 = fits.Column(name='v (km/s)', array=dv_bins, format='D')
c2 = fits.Column(name='dM (Msun/km/s)', array=dM_all, format='D')
c3 = fits.Column(name='dM_cold (Msun/km/s)', array=dM_cold, format='D')
c4 = fits.Column(name='dM_cool (Msun/km/s)', array=dM_cool, format='D')
c5 = fits.Column(name='dM_warm (Msun/km/s)', array=dM_warm, format='D')
c6 = fits.Column(name='dM_hot (Msun/km/s)', array=dM_hot, format='D')

all_cols = [c1, c2, c3, c4, c5, c6]
t = fits.BinTableHDU.from_columns(all_cols)
fig_dir = 'figs/dM_dv/fits'
tb_name = '%s_%s_dMdv_%s_%s.fits'%(sim_name, dd_name, obj_tag, obs_point)
save_to_file = '%s/%s'%(fig_dir, tb_name)
print("I am saving it to ", save_to_file)
t.writeto(save_to_file, overwrite=True)
