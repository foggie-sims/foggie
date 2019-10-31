# 10/13/2019, Yong Zheng, UCB.
# Changed the code to plot gc and offcenter allsky projections for diff ions.
# added trident to post process the simulation for different ions.
#
# 10/08/2019, Yong Zheng. UCB.
# Now adapted the code to check the all sky project from different off center
# location of the disk. The randdom_seed = 99 is good to generate UVW vectors
#
# 10/07/2019, Yong Zheng, UCB.
# We recently found problems with nref11c_nref9f sim outputs, so now
# switch to nref11n_nref10f/RD0039. Re-run everything from the beginnign
# meanwhile change mocky_way code structures to better fit the new foggie
# strucutres.
# !!!! This code still use the code foggie repo that I previously downloaded
# to pleiades/yzheng7. It also calls a special version of yt to do the allskyproj,
# so it is not suitable for anything else that can be run by other mathcinary.
#
# 08/27/2018, Yong Zheng, UCB.
# This function is used to make allsky prjection with an observer
# at 1.0, 1.5, and 2.0 Rs from the galactic center, to test whether
# that makes a difference in the result we are looking at.
# ions of [HI, SiII, SiIII, SiIV, CIV, OVI] can be used.

import os
import sys
import astropy.units as u
import numpy as np
import healpy as hp
import matplotlib.pyplot as plt

import mocky_way_modules  # read this in before reading in foggie and yt
from mocky_way_modules import data_dir_sys_dir
data_dir, sys_dir = data_dir_sys_dir()
os.sys.path.insert(0, sys_dir)

import yt
# import foggie
from yt.visualization.volume_rendering.healpix_projection import healpix_projection
from mocky_way_modules import save_allsky_healpix_img, plt_allsky_healpix_img
import foggie.consistency as consistency # for plotting

# sim_name = 'nref11c_nref9f_selfshield_z6'
# dd_name = 'RD0037'

obj_tag = sys.argv[1]
# obj_tag = 'all' # means cgm+disk, or can do 'cgm' only, or r0-10, r10-120,
                  # r0-20, r0-30, r0-40, r0-50, r0-60, r0-70, r0-80, r0-90
                  # r0-100, r0-110, r0-120,

sim_name = 'nref11n_nref10f'
# d_name = 'RD0039'
dd_name = 'DD2175'

#### Reading in the dataset ###
fig_dir = sys_dir+'/mocky_way/figs/allsky_diff_ions'
os.sys.path.insert(0, sys_dir)
ds_file = '%s/%s/%s/%s'%(data_dir, sim_name, dd_name, dd_name)
ds = yt.load(ds_file)
zsnap = ds.get_parameter('CosmologyCurrentRedshift')

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
td_ion_list = ['Si II', 'Si III', 'Si IV', 'C II', 'C IV', 'O VI', 'N V']
print("Adding ion fields: ", td_ion_list)
trident.add_ion_fields(ds, ftype="gas", ions=td_ion_list, force_override=True)
ion_list = [ss.replace(' ', '') for ss in td_ion_list]
ion_list.append('HI')

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
x -= x.dot(z) * z       # make it orthogonal to k
x /= np.linalg.norm(x)  # normalize it
y = np.cross(z, x)      # cross product with k
sun_vec = yt.YTArray(x)
phi_vec = yt.YTArray(y)
L_vec = yt.YTArray(z)

#### locate the observer to 2Rs
obs_vec = sun_vec
obs_dist = ds.quan(2*disk_rs, "kpc").in_units("code_length")
offcenter_location = halo_center + obs_vec*obs_dist # observer location

#### Setup plotting basics ####
nside = 2**8 # tested, 2**8 is the best, 2**10 is too much, not necessary
xsize = 800
gc = plt.cm.Greys(0.8) # gc = gridcolor

#### decide if only project cgm, or proj the whole cgm+disk ###
# obj_tag = 'all' # means cgm+disk, or can do 'cgm' only
if obj_tag == 'all':
    # sp = ds.sphere(halo_center, (120, 'kpc'))
    sp = ds.sphere(halo_center, (120, 'kpc'))
    obj = sp
elif obj_tag == 'cgm':
    # sp = ds.sphere(ds_paras['halo_center'], ds_paras['rvir'])
    sp = ds.sphere(halo_center, (120, 'kpc'))
    disk_size_r = 4*disk_rs # 4 is decided by eyeballing the size in find_flat_disk_offaxproj
    disk_size_z = 4*disk_zs # one side,
    disk = ds.disk(halo_center, L_vec,
                   (disk_size_r, 'kpc'),
                   (disk_size_z, 'kpc'))
    cgm = sp-disk
    obj = cgm
elif obj_tag == 'r0-10':
    sp = ds.sphere(halo_center, (10, 'kpc'))
    obj = sp
elif obj_tag == 'r10-120':
    sp_in = ds.sphere(halo_center, (10, 'kpc'))
    sp_out = ds.sphere(halo_center, (120, 'kpc'))
    shell = sp_out - sp_in
    obj = shell
elif obj_tag == 'r0-20':
     sp = ds.sphere(halo_center, (20, 'kpc'))
     obj = sp
elif obj_tag == 'r0-30':
    sp = ds.sphere(halo_center, (30, 'kpc'))
    obj = sp
elif obj_tag == 'r0-40':
    sp = ds.sphere(halo_center, (40, 'kpc'))
    obj = sp
elif obj_tag == 'r0-50':
    sp = ds.sphere(halo_center, (50, 'kpc'))
    obj = sp
elif obj_tag == 'r0-60':
    sp = ds.sphere(halo_center, (60, 'kpc'))
    obj = sp
elif obj_tag == 'r0-70':
    sp = ds.sphere(halo_center, (70, 'kpc'))
    obj = sp
elif obj_tag == 'r0-80':
    sp = ds.sphere(halo_center, (80, 'kpc'))
    obj = sp
elif obj_tag == 'r0-90':
    sp = ds.sphere(halo_center, (90, 'kpc'))
    obj = sp
elif obj_tag == 'r0-100':
    sp = ds.sphere(halo_center, (100, 'kpc'))
    obj = sp
elif obj_tag == 'r0-110':
    sp = ds.sphere(halo_center, (110, 'kpc'))
    obj = sp
elif obj_tag == 'r0-120':
    sp = ds.sphere(halo_center, (120, 'kpc'))
    obj = sp
else:
    print("Cannot recognize the obj_tag you put in, please check.")
    import sys
    sys.exit()

#### then, plot allsky projection from offcenter
for obs_xyz, obs_tag in zip([halo_center, offcenter_location],
                            ['halo_center', 'offcenter_location']):
    for ion in ion_list:
        field_to_proj = consistency.species_dict[ion]
        item_to_proj = ('gas', field_to_proj)
        img = healpix_projection(obj, obs_xyz, pathlength,
                                 nside, item_to_proj,
                                 normal_vector=-obs_vec,
                                 north_vector=L_vec)
                                 # normal vector points to the
                                 # center of the image

        # save the healpix projection result
        filename = '%s_%s_%s_%s_%s'%(sim_name, dd_name, obj_tag, obs_tag, ion)
        save_to_fits = '%s/%s.fits'%(fig_dir, filename)
        save_allsky_healpix_img(img, nside, save_to_fits)

        save_to_pdf = '%s/%s.pdf'%(fig_dir, filename)
        plt_allsky_healpix_img(img, ion, xsize, dd_name,
                               zsnap, save_to_pdf)
