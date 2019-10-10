# 10/07/2019, Yong Zheng, UCB.
# We recently found problems with nref11c_nref9f, so now
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
from yt.utilities.math_utils import ortho_find
# import foggie
from yt.visualization.volume_rendering.healpix_projection import healpix_projection
from mocky_way_modules import save_allsky_healpix_img, plt_allsky_healpix_img

# sim_name = 'nref11c_nref9f_selfshield_z6'
# dd_name = 'RD0037'

sim_name = 'nref11n_nref10f'
dd_name = 'RD0039'
ion_to_proj = 'HI'
rvir = 161  # in unit of kpc, pre-run already by foggie.mocky_way.find_r200

#### !!! DO NOT CHANGE THE FOLLOWING THREE LINES ###
# To be consistent with the new code in
# mocky_way.core_funcs.get_sphere_ang_mom_vecs
random_seed = 99      # DO NOT CHANGE.
use_gas = True        # DO NOT CHANGE
use_particles = False # DO NOT CHANGE

#### Reading in the dataset ###
fig_dir = sys_dir+'/mocky_way/figs/find_flat_disk'
os.sys.path.insert(0, sys_dir)
ds_file = '%s/%s/%s/%s'%(data_dir, sim_name, dd_name, dd_name)
ds = yt.load(ds_file)
zsnap = ds.get_parameter('CosmologyCurrentRedshift')

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

#### Setup plotting basics ####
nside = 2**5 # 2**10 is probably too big....pix number = 12*nside^2
xsize = 800
gc = plt.cm.Greys(0.8) # gc = gridcolor

import foggie.consistency as consistency # for plotting
field_to_proj = consistency.species_dict[ion_to_proj]
item = ('gas', field_to_proj)  # NHI across the sky
img_cmap = consistency.colormap_dict[field_to_proj]
img_min = np.log10(consistency.proj_min_dict[field_to_proj])
img_max = np.log10(consistency.proj_max_dict[field_to_proj])
img_label = consistency.axes_label_dict[field_to_proj]

##### loops over different size to find the best angular momentum ####
for r_for_L in [5, 10, 15, 20]:
    #### the angular momentum of the sphere
    sp = ds.h.sphere(halo_center, (r_for_L, 'kpc'))
    # IMPROTANT!! need to set up the bulk velocity before geting L
    sp_bulkvel = sp.quantities.bulk_velocity(use_gas=use_gas,
                                             use_particles=use_particles)
    sp.set_field_parameter('bulk_velocity', sp_bulkvel)
    #### angular momentum
    spec_L = sp.quantities.angular_momentum_vector(use_gas=use_gas,
                                                   use_particles=use_particles)
    norm_L = spec_L / np.sqrt((spec_L**2).sum())
    np.random.seed(random_seed) ## to make sure we get the same thing everytime
    L_vec, sun_vec, phi_vec = ortho_find(norm_L)  # UVW vector

    #### make the all sky projection
    im = healpix_projection(ds, halo_center,
                            ds.quan(rvir, 'kpc'),
                            nside, item,
                            normal_vector=sun_vec,
                            north_vector=L_vec)
    fig = plt.figure(figsize=(8, 4))
    hp.mollview(np.log10(im), cbar=None, cmap=img_cmap,
                xsize=xsize, min=img_min, max=img_max,
                title='%s (z=%.2f)'%(dd_name, zsnap))
    hp.graticule(color=gc)
    fig = plt.gcf()
    ax = plt.gca()
    image = ax.get_images()[0]
    cbar=fig.colorbar(image, ax=ax, pad=0.02, orientation='horizontal',
                      shrink=0.6, label=img_label)

    fig_name = '%s_%s_AngMon%skpc_allsky.pdf'%(sim_name, dd_name, r_for_L)
    plt.savefig('%s/%s'%(fig_dir, fig_name))
    plt.close()
    # break
