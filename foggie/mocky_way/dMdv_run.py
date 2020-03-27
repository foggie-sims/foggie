### This function calculates the mass flux per velocity bins, and also
# it split the gas into different temperature bins.
# 03/27/2020, add high and low galactic latitude selection, in response to
#             referee's request #1, and change figure 9
# 12/18/2019, add arg to measure kinematics from another seven location
# 10/14/2019, rewrite the structure of the code for DD2175,
#             and merge into foggie.mocky_way, Yong Zheng, UCB.
# 08/14/2019, Yong add in the option to shift the observer on the solar circle.
# 08/11/2019, created, Yong Zheng, UCB

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
shift_obs_location = False
shift_n45 = 1

# decide at the galactic latitude we are going to break up the halo
b_lim = 20 # degrees

ds, ds_paras = prepdata(dd_name, sim_name=sim_name,
                        shift_obs_location=shift_obs_location,
                        shift_n45=shift_n45)
obj_tag = 'cgm-rvir' # cgm-15kpc, cgm-20kpc, cgm-rvir
obs_point = 'halo_center'  # halo_center, offcenter_location
obs_bulkvel = 'disk_bulkvel' # disk_bulkvel, offcenter_bulkvel
#obs_point = 'halo_center'  # halo_center, offcenter_location
#obs_bulkvel = 'disk_bulkvel' # disk_bulkvel, offcenter_bulkvel

#### No need to change starting this line ####
print("I am doing the calculation from %s for %s..."%(obs_point, obj_tag))
obj_source = obj_source_all_disk_cgm(ds, ds_paras, obj_tag)
obj_source.set_field_parameter("observer_location", ds_paras[obs_point])
obj_source.set_field_parameter("observer_bulkvel", ds_paras[obs_bulkvel])

# this is added to derive galactic latitude and longitude
obj_source.set_field_parameter("L_vec", ds_paras["L_vec"])
obj_source.set_field_parameter("sun_vec", ds_paras["sun_vec"])

# now get the relevant field values for each cell
obj_l = obj_source["gas", "l"]
obj_b = obj_source["gas", "b"]
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
dM_all_lowb = np.zeros(dv_bins.size)
dM_all_highb = np.zeros(dv_bins.size)

for iv in range(dv_bins.size):
    va = dv_bins[iv]-dv/2.
    vb = dv_bins[iv]+dv/2.

    # dM/dv per bin for all b
    indv = np.all([obj_vlsr>=va, obj_vlsr<vb], axis=0)
    if len(obj_cell_mass[indv])!=0:
        dM_all[iv] = obj_cell_mass[indv].sum().in_units("Msun")/dv

    # dM/dv per bin for low latitude
    indv_lowb = indv & (np.abs(obj_b) < b_lim)
    if len(obj_cell_mass[indv_lowb])!=0:
        dM_all_lowb[iv] = obj_cell_mass[indv_lowb].sum().in_units("Msun")/dv

    # dM/dv per bin for high latitude
    indv_highb = indv & (np.abs(obj_b) >= b_lim)
    if len(obj_cell_mass[indv_highb])!=0:
        dM_all_highb[iv] = obj_cell_mass[indv_highb].sum().in_units("Msun")/dv

### then let's break up the mass into different temperature categories
print("Now let us break up the gas mass into different temperature ranges...")

from foggie.mocky_way.core_funcs import temperature_category
temp_dict = temperature_category()

dMdv_dict = {}

for temp_tag in ['cold', 'cool', 'warm', 'hot']:
    dM_thisT = np.zeros(dv_bins.size)
    dM_thisT_lowb = np.zeros(dv_bins.size)
    dM_thisT_highb = np.zeros(dv_bins.size)

    for iv in range(dv_bins.size):
        va = dv_bins[iv]-dv/2.
        vb = dv_bins[iv]+dv/2.
        indv = np.all([obj_vlsr>=va, obj_vlsr<vb], axis=0)

        if len(obj_cell_mass[indv])!=0:
            iv_mass = obj_cell_mass[indv]
            iv_T = obj_temperature[indv]
            iv_b = obj_b[indv]

            ### low and high b gas index
            ind_lowb = np.abs(iv_b) < b_lim
            ind_highb = np.abs(iv_b) >= b_lim

            # now filter specific temperature
            range_T = temp_dict[temp_tag]
            ind_T = np.all([iv_T>range_T[0], iv_T<=range_T[1]], axis=0)
            if len(iv_mass[ind_T])!= 0:
                dM_thisT[iv] = iv_mass[ind_T].sum().in_units("Msun")/dv

            ind_T_lowb = ind_T & ind_lowb
            if len(iv_mass[ind_T_lowb])!= 0:
                dM_thisT_lowb[iv] = iv_mass[ind_T_lowb].sum().in_units("Msun")/dv

            ind_T_highb = ind_T & ind_highb
            if len(iv_mass[ind_T_highb])!= 0:
                dM_thisT_highb[iv] = iv_mass[ind_T_highb].sum().in_units("Msun")/dv

        dMdv_dict['dM_%s'%(temp_tag)] = dM_thisT
        dMdv_dict['dM_%s_lowb'%(temp_tag)] = dM_thisT_lowb
        dMdv_dict['dM_%s_highb'%(temp_tag)] = dM_thisT_highb

print(dMdv_dict.keys())

##### now saving the data ####
c1 = fits.Column(name='v (km/s)', array=dv_bins, format='D')

c2 = fits.Column(name='dM_allb (Msun/km/s)', array=dM_all, format='D')
c2b = fits.Column(name='dM_lowb (Msun/km/s)', array=dM_all_lowb, format='D')
c2c = fits.Column(name='dM_highb (Msun/km/s)', array=dM_all_highb, format='D')

c3 = fits.Column(name='dM_cold_allb (Msun/km/s)', array=dMdv_dict['dM_cold'], format='D')
c3b = fits.Column(name='dM_cold_lowb (Msun/km/s)', array=dMdv_dict['dM_cold_lowb'], format='D')
c3c = fits.Column(name='dM_cold_highb (Msun/km/s)', array=dMdv_dict['dM_cold_highb'], format='D')

c4 = fits.Column(name='dM_cool_allb (Msun/km/s)', array=dMdv_dict['dM_cool'], format='D')
c4b = fits.Column(name='dM_cool_lowb (Msun/km/s)', array=dMdv_dict['dM_cool_lowb'], format='D')
c4c = fits.Column(name='dM_cool_highb (Msun/km/s)', array=dMdv_dict['dM_cool_highb'], format='D')

c5 = fits.Column(name='dM_warm_allb (Msun/km/s)', array=dMdv_dict['dM_warm'], format='D')
c5b = fits.Column(name='dM_warm_lowb (Msun/km/s)', array=dMdv_dict['dM_warm_lowb'], format='D')
c5c = fits.Column(name='dM_warm_highb (Msun/km/s)', array=dMdv_dict['dM_warm_highb'], format='D')

c6 = fits.Column(name='dM_hot_allb (Msun/km/s)', array=dMdv_dict['dM_hot'], format='D')
c6b = fits.Column(name='dM_hot_lowb (Msun/km/s)', array=dMdv_dict['dM_hot_lowb'], format='D')
c6c = fits.Column(name='dM_hot_highb (Msun/km/s)', array=dMdv_dict['dM_hot_highb'], format='D')

all_cols = [c1,
            c2, c2b, c2c,
            c3, c3b, c3c,
            c4, c4b, c4c,
            c5, c5b, c5c,
            c6, c6b, c6c]

t = fits.BinTableHDU.from_columns(all_cols)
fig_dir = 'figs/dM_dv/fits'

if shift_obs_location == False:
    tb_name = '%s_%s_dMdv_%s_%s_b%d.fits'%(sim_name, dd_name, obj_tag, obs_point, b_lim)
else:
    tb_name = '%s_%s_dMdv_%s_%s_b%d_%d.fits'%(sim_name, dd_name, obj_tag, obs_point, b_lim,
                                          shift_n45*45)

save_to_file = '%s/%s'%(fig_dir, tb_name)
print("I am saving it to ", save_to_file)
t.writeto(save_to_file, overwrite=True)
