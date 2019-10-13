# 10/08/2019, Yong Zheng. UCB.
# Now adapted the code to check the all sky project from different off center
# location of the disk. The randdom_seed = 99 is good to generate UVW vectors
#
# 10/07/2019, Yong Zheng, UCB.
# We recently found problems with nref11c_nref9f, so now
# switch to nref11n_nref10f/RD0039. Re-run everything from the beginnign
# meanwhile change mocky_way code structures to better fit the new foggie
# strucutres.
# !!!! This code still use the code foggie repo that I previously downloaded
# to pleiades/Yong Zhengheng7. It also calls a special version of yt to do the allskyproj,
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
from yt.utilities.math_utils import ortho_find
# import foggie
from yt.visualization.volume_rendering.healpix_projection import healpix_projection
from mocky_way_modules import save_allsky_healpix_img, plt_allsky_healpix_img

# sim_name = 'nref11c_nref9f_selfshield_z6'
# dd_name = 'RD0037'

sim_name = 'nref11n_nref10f'
# d_name = 'RD0039'
dd_name = 'DD2175'
ion_to_proj = 'HI'

#### Reading in the dataset ###
fig_dir = sys_dir+'/mocky_way/figs/locate_offcenters'
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
r_for_L = 5 # DO NOT CHANGE. See core_funcs.dict_sphere_for_gal_ang_mom
rvir = 161  # DO NOT CHANGE. in unit of kpc, pre-run already by foggie.mocky_way.find_r200
pathlength = ds.quan(120, 'kpc') # DO NOT CHANGE. within refinement box size.

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
np.random.seed(random_seed) ## to make sure we get the same thing everytime
L_vec, sun_vec, phi_vec = ortho_find(norm_L)  # UVW vector

#### Setup plotting basics ####
nside = 2**8 # tested, 2**8 is the best, 2**10 is too much, not necessary
xsize = 800
gc = plt.cm.Greys(0.8) # gc = gridcolor

import foggie.consistency as consistency # for plotting
field_to_proj = consistency.species_dict[ion_to_proj]
item = ('gas', field_to_proj)  # NHI across the sky
img_cmap = consistency.colormap_dict[field_to_proj]
img_min = np.log10(consistency.proj_min_dict[field_to_proj])
img_max = np.log10(consistency.proj_max_dict[field_to_proj])
img_label = consistency.axes_label_dict[field_to_proj]

##### loops over different location to find the best angular momentum ####
obs_loc_vectors = [sun_vec, sun_vec+phi_vec,
                   phi_vec, -sun_vec+phi_vec,
                   -sun_vec, -sun_vec-phi_vec,
                   -phi_vec, -phi_vec+sun_vec]
for ii, obs_phi in enumerate([0, 45, 90, 135, 180, 225, 270, 315]):
    #### Decide the UVW vector
    obs_vec = obs_loc_vectors[ii]
    obs_vec = obs_vec/np.sqrt(np.sum(obs_vec**2))
    new_sun_vec = obs_vec
    new_phi_vec = np.cross(obs_vec, L_vec)
    new_phi_vec = new_phi_vec/np.sqrt(np.sum(new_phi_vec**2))

    #### Now locate the observer to 2Rs
    obs_dist = ds.quan(2*disk_rs, "kpc").in_units("code_length")
    offcenter_location = halo_center + obs_vec*obs_dist # observer location

    # set the bulk velocity of the observer, taken to be gas within 1 kpc
    obs_sp = ds.sphere(offcenter_location, (1, "kpc"))
    obs_bv = obs_sp.quantities.bulk_velocity(use_gas=True, use_particles=True)
    obs_bv = obs_bv.in_units("km/s")

    #### make the all sky projection
    obs_xyz = offcenter_location
    # pathlength = ds.quan(rvir, 'kpc')
    img = healpix_projection(ds, obs_xyz,
                             pathlength, nside, item,
                             normal_vector=-obs_vec,
                             north_vector=L_vec)
                             # normal vector points to the center of the image

    # save the healpix projection result
    filename = '%s_%s_rand%d_phi%d_allsky'%(sim_name, dd_name, random_seed, obs_phi)
    save_to_fits = '%s/%s.fits'%(fig_dir, filename)
    save_allsky_healpix_img(img, nside, save_to_fits)

    save_to_pdf = '%s/%s.pdf'%(fig_dir, filename)
    plt_allsky_healpix_img(img, ion_to_proj, xsize, dd_name,
                           zsnap, save_to_pdf)
