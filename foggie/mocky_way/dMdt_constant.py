# calculat the inflow and outflow mass flux rate using the observer way,
# which is dM/dt = sum(Mi*vi/Di), where Di is the distance between gas and
# observer, assuming the gas will make it to the disk
# This is to compare with dM_dt_instant.py, which calculate the absolute flux
# at the position of the cloud, in this setup, the cloud may not survive
# to the disk.
#
# History:
# 10/29/2019, created, Yong Zheng, in discussion with Cassi Lochhass. UCB.

import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['font.family'] = 'stixgeneral'
from foggie.mocky_way.core_funcs import prepdata
from foggie.mocky_way.core_funcs import obj_source_all_disk_cgm

import sys
sim_name = sys.argv[1] # 'nref11n_nref10f'
dd_name =  sys.argv[2] # 'DD2175'

#only use this argument if changing observer location inside the galaxy
shift_obs_location = True
shift_n45 = 7
# ds, ds_paras = prepdata(dd_name, sim_name=sim_name)
ds, ds_paras = prepdata(dd_name, sim_name=sim_name,
                        shift_obs_location=shift_obs_location,
                        shift_n45=shift_n45)

obj_tag = 'cgm-rvir' # cgm-15kpc, cgm-20kpc, cgm-rvir
#obs_point = 'halo_center'  # halo_center, offcenter_location
#obs_bulkvel = 'disk_bulkvel' # disk_bulkvel, offcenter_bulkvel
obs_point = 'offcenter_location'  # halo_center, offcenter_location
obs_bulkvel = 'offcenter_bulkvel' # disk_bulkvel, offcenter_bulkvel

#### No need to change starting this line ####
print("I am doing the calculation from %s for %s..."%(obs_point, obj_tag))
obj_source = obj_source_all_disk_cgm(ds, ds_paras, obj_tag)
obj_source.set_field_parameter("observer_location", ds_paras[obs_point])
obj_source.set_field_parameter("observer_bulkvel", ds_paras[obs_bulkvel])

obj_vlsr = obj_source["gas", "los_velocity_mw"]
obj_cell_mass = obj_source["gas", "cell_mass"].in_units('Msun')
obj_temperature = obj_source["gas", "temperature"]

from foggie.mocky_way.core_funcs import los_r
obj_r = los_r(ds, obj_source, ds_paras[obs_point])
# print(obj_r.min())

# calculate the flux in an observer way
cell_constant_flux = (obj_cell_mass*obj_vlsr/obj_r).in_units("Msun/yr")

# setup (dM, dv) step intervals
dv = 20  # the velocity bin every 20 km/s
vrange = [-1000, 1000]
dv_bins = np.mgrid[vrange[0]:vrange[1]+dv:dv]

### first let's get the halo mass distribution as a func of vel, regardless of T
print("Calculating dM-dv for all gas....")
dMdt_all = np.zeros(dv_bins.size)
for iv in range(dv_bins.size):
    va = dv_bins[iv]-dv/2.
    vb = dv_bins[iv]+dv/2.
    indv = np.all([obj_vlsr>=va, obj_vlsr<vb], axis=0)
    if len(cell_constant_flux[indv])!=0:
        dMdt_all[iv] = cell_constant_flux[indv].sum()

### then let's break up the mass into different temperature categories
print("Now let us break up the gas mass into different temperature ranges...")
dMdt_cold = np.zeros(dv_bins.size)
dMdt_cool = np.zeros(dv_bins.size)
dMdt_warm = np.zeros(dv_bins.size)
dMdt_hot = np.zeros(dv_bins.size)

from foggie.mocky_way.core_funcs import temperature_category
temp_dict = temperature_category()

for iv in range(dv_bins.size):
    va = dv_bins[iv]-dv/2.
    vb = dv_bins[iv]+dv/2.
    indv = np.all([obj_vlsr>=va, obj_vlsr<vb], axis=0)

    if len(cell_constant_flux[indv])!=0:
        iv_flux = cell_constant_flux[indv]
        iv_T = obj_temperature[indv]

        ### cold gas ####
        # ind_cold = iv_T <= 1e4
        cold_T = temp_dict['cold']
        ind_cold = np.all([iv_T>cold_T[0], iv_T<=cold_T[1]], axis=0)
        if len(iv_flux[ind_cold])!= 0:
            dMdt_cold[iv] = iv_flux[ind_cold].sum()

        ### cool gas ####
        # ind_cool = np.all([iv_T>1e4, iv_T<=1e5], axis=0)
        cool_T = temp_dict['cool']
        ind_cool = np.all([iv_T>cool_T[0], iv_T<=cool_T[1]], axis=0)
        if len(iv_flux[ind_cool]) != 0:
            dMdt_cool[iv] = iv_flux[ind_cool].sum()

        ### warm gas ####
        # ind_warm = np.all([iv_T>1e5, iv_T<=1e6], axis=0)
        warm_T = temp_dict['warm']
        ind_warm = np.all([iv_T>warm_T[0], iv_T<=warm_T[1]], axis=0)
        if len(iv_flux[ind_warm]) != 0:
            dMdt_warm[iv] = iv_flux[ind_warm].sum()

        ## now this is hot
        # ind_hot = iv_T > 1e6
        hot_T = temp_dict['hot']
        ind_hot = np.all([iv_T>hot_T[0], iv_T<hot_T[1]], axis=0)
        if len(iv_flux[ind_hot]) != 0:
            dMdt_hot[iv] = iv_flux[ind_hot].sum()

##### now saving the data ####
c1 = fits.Column(name='v (km/s)', array=dv_bins, format='D')
c2 = fits.Column(name='dMdt (Msun/yr)', array=dMdt_all, format='D')
c3 = fits.Column(name='dMdt_cold (Msun/yr)', array=dMdt_cold, format='D')
c4 = fits.Column(name='dMdt_cool (Msun/yr)', array=dMdt_cool, format='D')
c5 = fits.Column(name='dMdt_warm (Msun/yr)', array=dMdt_warm, format='D')
c6 = fits.Column(name='dMdt_hot (Msun/yr)', array=dMdt_hot, format='D')

all_cols = [c1, c2, c3, c4, c5, c6]
t = fits.BinTableHDU.from_columns(all_cols)
fig_dir = 'figs/dM_dt/fits'

if shift_obs_location == False:
    tb_name = '%s_%s_dMdt_%s_%s.fits'%(sim_name, dd_name, obj_tag, obs_point)
else:
    tb_name = '%s_%s_dMdt_%s_%s_%d.fits'%(sim_name, dd_name, obj_tag, obs_point,
                                          shift_n45*45)

save_to_file = '%s/%s'%(fig_dir, tb_name)
print("I am saving it to ", save_to_file)
t.writeto(save_to_file, overwrite=True)
