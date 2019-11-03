#####
# special on pleiades, check column_density_inview_ray.py for a local version
#### Reading in the dataset ###

import os
import sys
import mocky_way_modules  # read this in before reading in foggie and yt
from mocky_way_modules import data_dir_sys_dir
data_dir, sys_dir = data_dir_sys_dir()
os.sys.path.insert(0, sys_dir)

import yt
import numpy as np
from calc_ray_end import calc_ray_end
from calc_ray_ion_column_density import calc_ray_ion_column_density

ion = sys.argv[1]
ion_list = ['HI', 'SiII', 'SiIII', 'SiIV', 'CII', 'CIV', 'OVI', 'NV',
            'OVII', 'OVIII', 'NeVII', 'NeVIII']
if ion not in ion_list:
    print("Sorry, %s is not on my ion list, please add it first. "%(ion))
    sys.exit()

rin = float(sys.argv[2])# 3 # kpc
rout = float(sys.argv[3])# 160 # kpc

sim_name = 'nref11n_nref10f'
dd_name = 'DD2175'
nlos = 100000

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

#### !!! DO NOT CHANGE THE FOLLOWING THREE LINES ###
# To be consistent with the new code in
# mocky_way.core_funcs.get_sphere_ang_mom_vecs
random_seed = 99      # DO NOT CHANGE.
use_gas = True        # DO NOT CHANGE
use_particles = False # DO NOT CHANGE
disk_rs = 3.4 # DO NOT CHANGE. See core_funcs.dict_disk_rs_zs
disk_zs = 0.5 # DO NOT CHANGE, See core_funcs.dict_disk_rs_zs
r_for_L = 5 # DO NOT CHANGE. See core_funcs.dict_sphere_for_gal_ang_mom
rvir = 161  # DO NOT CHANGE. in unit of kpc, pre-run already by foggie.mocky_way.find_r200
pathlength = ds.quan(120, 'kpc') # DO NOT CHANGE. within refinement box size.

## post processing the data
import trident
# td_ion_list = ['Si II', 'Si III', 'Si IV', 'C II', 'C IV', 'O VI', 'N V']
td_ion_list = ['Si II', 'Si III', 'Si IV', 'C II', 'C IV', 'O VI', 'N V',
               'O VII', 'O VIII', 'Ne VII', 'Ne VIII']
print("Adding ion fields: ", td_ion_list)
trident.add_ion_fields(ds, ftype="gas", ions=td_ion_list, force_override=True)
ion_list = [ss.replace(' ', '') for ss in td_ion_list]
ion_list.append('HI')

### now let's find halo center and offcenter location  ###
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

#### Find the angular momentum
sp = ds.h.sphere(halo_center, (r_for_L, 'kpc'))
# IMPROTANT!! need to set up the bulk velocity before geting L
sp_bulkvel = sp.quantities.bulk_velocity(use_gas=use_gas,
                                         use_particles=use_particles)
sp.set_field_parameter('bulk_velocity', sp_bulkvel)
spec_L = sp.quantities.angular_momentum_vector(use_gas=use_gas,
                                               use_particles=use_particles)
norm_L = spec_L / np.sqrt((spec_L**2).sum())

### find the sun_vec and phi_vec
np.random.seed(random_seed)
z = norm_L
x = np.random.randn(3)  # take a random vector
x -= x.dot(z) * z       # make it orthogonal to z
x /= np.linalg.norm(x)  # normalize it
y = np.cross(z, x)      # cross product with z
sun_vec = yt.YTArray(x)
phi_vec = yt.YTArray(y)
L_vec = yt.YTArray(z)

#### locate the observer to 2Rs
obs_vec = sun_vec
obs_dist = ds.quan(2*disk_rs, "kpc").in_units("code_length")
offcenter_location = halo_center + obs_vec*obs_dist # observer location
ds_paras = {'offcenter_location': offcenter_location,
            'sun_vec': sun_vec,
            'phi_vec': phi_vec,
            'L_vec': L_vec}
############################################
all_N = []
all_l = []
all_b = []
all_r = []

# now let's do a total number of nlos sightlines, and save it/overwrite it per
# step, so that data would get lost on pleiades
los_ray_start = ds_paras['offcenter_location']

for i in range(nlos):
    los_l_deg = np.random.uniform(low=0., high=360.)
    los_b_deg = np.random.uniform(low=-90., high=90.)
    los_length_kpc = np.random.uniform(low=rin, high=rout)

    los_ray_end, los_unit_vector = calc_ray_end(ds, ds_paras, los_l_deg, los_b_deg,
                                                los_ray_start, los_length_kpc)

    rayfilename = 'ray_%s_%d_%d.h5'%(ion, rin, rout)
    Nion, other_info = calc_ray_ion_column_density(ds, ion,
                                                   los_ray_start,
                                                   los_ray_end,
                                                   rayfilename=rayfilename)

    ## let's save it step by step
    all_N.append(Nion)
    all_l.append(los_l_deg)
    all_b.append(los_b_deg)
    all_r.append(los_length_kpc)

    # save the data to fits file
    ##### now saving the data ####
    import astropy.io.fits as fits
    c1 = fits.Column(name='N', array=np.asarray(all_N), format='D')
    c2 = fits.Column(name='l', array=np.asarray(all_l), format='D')
    c3 = fits.Column(name='b', array=np.asarray(all_b), format='D')
    c4 = fits.Column(name='r', array=np.asarray(all_r), format='D')

    all_cols = [c1, c2, c3, c4]
    t = fits.BinTableHDU.from_columns(all_cols)
    fig_dir = '%s/mocky_way/figs/Nr_inview/fits'%(sys_dir)
    tb_name = 'nref11n_nref10f_DD2175_N%s_inview_%d-%d.fits'%(ion, rin, rout)

    save_to_file = '%s/%s'%(fig_dir, tb_name)
    # print("%s: I am saving it to %s"%(i, save_to_file))
    t.writeto(save_to_file, overwrite=True)
