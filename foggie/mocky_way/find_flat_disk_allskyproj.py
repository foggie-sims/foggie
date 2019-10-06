# History:
# 10/04/2019, YZ.
# Change how to import the funcs and merge into foggie/mocky_way
#
# 08/07/2019, YZ.
# actually, now I set up bulk velocity for the designated sphere before
# calculating angular momentum. In allsky projection case, both ways result
# in similar projeciton (but in offaxis projeciton ,it is important to make sure to
# always set bulk velocity before calculating angular momentum)
#
# 03/27/2019, YZ.
# Check **/mocky_way/figs/allskyproj/nref11c_nref9f_selfshield_z6/***
# image name of [dd_name]_LNNkpc_n32_x800_R100.0.pdf, where NN=10, 20, 30, 50
# ang mom from 10 kpc sphere from halo center,
# with use_gas=True, use_particle=False
# no pre setup of bulk velocity before calculating angmom, see code line 72, 73.

### point to the right data path
import os
import sys
import numpy as np
import healpy as hp
import astropy.units as u
import matplotlib.pyplot as plt

from foggie.mocky_way.core_funcs import data_dir_sys_dir
from foggie.mocky_way.core_funcs import find_halo_center_yz
from foggie.utils import consistency
from foggie.mocky_way.core_funcs import dict_rvir_proper

import yt
from yt.visualization.volume_rendering.healpix_projection import healpix_projection

sim_name = sys.argv[1]    # nref11n_nref10f
dd_name = sys.argv[2]     # RD0039
ion_to_proj = sys.argv[3] # mainly for HI

data_dir, sys_dir = data_dir_sys_dir()
fig_dir = sys_dir+'%s/foggie/foggie/mocky_way/figs/find_flat_disk'
os.sys.path.insert(0, sys_dir)

ds_file = '%s/%s/%s/%s'%(data_dir, sim_name, dd_name, dd_name)
ds = yt.load(ds_file)
zsnap = ds.get_parameter('CosmologyCurrentRedshift')
halo_center = find_halo_center_yz(ds, zsnap, sim_name, data_dir)
rvir = dict_rvir_proper(dd_name, sim_name=sim_name)

# get the angular momentum
nside = 2**5 # 2**10 is probably too big....pix number = 12*nside^2
xsize = 800
gc = plt.cm.Greys(0.8) # gc = gridcolor

for r_for_L in [5, 10, 15, 20]:
    dict_vecs = get_sphere_ang_mom_vecs(ds, halo_center, r_for_L,
                                        random_seed=99)
    L_vec = dict_vecs['L_vec']
    phi_vec = dict_vecs['phi_vec']
    sun_vec = dict_vecs['sun_vec']

    # make the all sky projection
    field_to_proj = consistency.species_dict[ion_to_proj]
    item = ('gas', field_to_proj)  # NHI across the sky
    im = healpix_projection(ds, halo_center, (rvir, 'kpc'),
                            nside, item,
                            normal_vector=sun_vec,
                            north_vector=L_vec)

    fig = plt.figure(figsize=(8, 4))
    img_cmap = consistency.colormap_dict[field_to_proj]
    img_min = np.log10(consistency.proj_min_dict[field_to_proj])
    img_max = np.log10(consistency.proj_max_dict[field_to_proj])
    hp.mollview(np.log10(im), cbar=None, cmap=img_cmap,
                xsize=xsize, min=img_min, max=img_max,
                title='%s (z=%.2f)'%(dd_name, zsnap))
    hp.graticule(color=gc)

    fig = plt.gcf()
    ax = plt.gca()
    image = ax.get_images()[0]
    img_label = consistency.axes_label_dict[field_to_proj]
    cbar=fig.colorbar(image, ax=ax, pad=0.02, orientation='horizontal',
                      shrink=0.6, label=img_label)

    fig_name = '%s_%s_AngMon%skpc_allsky.pdf'%(sim_name, dd_name, r_for_L)
    plt.savefig('%s/%s'%(fig_dir, fig_name))
    plt.close()
    # break
